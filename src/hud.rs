//! Phase 7/18/21/23 in-window 2D HUD, embedded bitmap font, chat, modal menus (pause, load
//! list, score screen with graphs), clickable minimap, and the Phase 23 FA command interface
//! (blueprint-driven command card, orders panel, factory queue / upgrade and enhancement
//! panels, tooltips, cursor-mode label, map pings).

use std::path::PathBuf;
use std::time::Instant;

use crate::camera::OrbitCamera;
use crate::command_card::{CardModel, CardSource, EnhRow, EnhState, OrderKind, QueueEntry};
use crate::play::{PlaySession, army_rgb};
use crate::savegame::SaveEntry;
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{GameResult, IntelVisibility, MapContext, Role};
use crate::stats::Series;
use crate::viewer::HudVertex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudAction {
    BuildFactory,
    BuildExtractor,
    BuildGenerator,
    BuildPointDefense,
    BuildRadar,
    ProduceEngineer,
    ProduceTank,
    ProduceBot,
    ProduceArtillery,
    AttackMove,
    Stop,
    TogglePause,
    ToggleFog,
    ToggleAi,
    IdleEngineer,
    UnloadTransport,
    ToggleRepeat,
    EnhanceNext,
    ToggleShield,
    // ----- Phase 23 -----
    /// Order / system button.
    Order(OrderKind),
    /// Command card entry (index on the current page).
    CardItem(u16),
    CardTab(u8),
    /// Category chip (index into the model's groups, `255` = all).
    CardGroup(u8),
    CardPage,
    /// Factory queue entry.
    QueueItem(u16),
    Upgrade,
    /// Enhancement row (index into `PlaySession::enhancement_panel`).
    EnhanceRow(u16),
}

/// Phase 21 modal match overlay.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum MatchMenu {
    #[default]
    None,
    Pause,
    LoadList {
        entries: Vec<SaveEntry>,
        selected: usize,
    },
    Score {
        series: Series,
    },
}

impl MatchMenu {
    pub fn is_open(&self) -> bool {
        !matches!(self, MatchMenu::None)
    }

    pub fn blocks_sim(&self) -> bool {
        matches!(self, MatchMenu::Pause | MatchMenu::LoadList { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchMenuAction {
    Resume,
    QuickSave,
    OpenLoadList,
    OpenScore,
    SelectLoad(usize),
    LoadSelected(PathBuf),
    SwitchScoreSeries(Series),
    WatchReplay,
    ContinueWatching,
    QuitToMenu,
    QuitGame,
}

#[derive(Debug, Clone)]
pub struct MatchMenuButton {
    pub rect: [f32; 4],
    pub label: String,
    pub action: MatchMenuAction,
    pub active: bool,
    pub disabled: bool,
}

#[derive(Debug, Clone)]
pub struct MatchMenuLayout {
    pub panel: [f32; 4],
    pub buttons: Vec<MatchMenuButton>,
    pub load_rows: Vec<([f32; 4], usize)>,
    pub graph: Option<[f32; 4]>,
}

impl MatchMenuLayout {
    pub fn compute(session: &PlaySession, width: u32, height: u32) -> Option<Self> {
        if !session.menu.is_open() {
            return None;
        }
        let w = width.max(480) as f32;
        let h = height.max(360) as f32;
        match &session.menu {
            MatchMenu::None => None,
            MatchMenu::Pause => {
                let pw = 340.0;
                let ph = 326.0;
                let px = (w - pw) * 0.5;
                let py = (h - ph) * 0.5;
                let bw = pw - 40.0;
                let bh = 36.0;
                let gap = 8.0;
                let bx = px + 20.0;
                let mut by = py + 52.0;
                let is_net = session.net.is_some();
                let is_replay = session.is_replay();
                let items = [
                    ("RESUME MATCH", MatchMenuAction::Resume, false, false),
                    ("SAVE QUICKSAVE (F6)", MatchMenuAction::QuickSave, false, is_replay || session.loading()),
                    ("LOAD SAVE GAME", MatchMenuAction::OpenLoadList, false, is_net || is_replay || session.loading()),
                    ("MATCH SCORE & GRAPHS", MatchMenuAction::OpenScore, false, false),
                    ("QUIT TO MAIN MENU", MatchMenuAction::QuitToMenu, false, false),
                    ("QUIT TO DESKTOP", MatchMenuAction::QuitGame, false, false),
                ];
                let mut buttons = Vec::with_capacity(items.len());
                for (label, action, active, disabled) in items {
                    buttons.push(MatchMenuButton {
                        rect: [bx, by, bw, bh],
                        label: label.to_string(),
                        action,
                        active,
                        disabled,
                    });
                    by += bh + gap;
                }
                Some(Self { panel: [px, py, pw, ph], buttons, load_rows: Vec::new(), graph: None })
            }
            MatchMenu::LoadList { entries, selected } => {
                let pw = (w - 80.0).min(600.0);
                let ph = (h - 80.0).min(430.0);
                let px = (w - pw) * 0.5;
                let py = (h - ph) * 0.5;
                let row_y0 = py + 52.0;
                let row_h = 28.0;
                let max_rows = ((ph - 122.0) / row_h).floor().max(1.0) as usize;
                let mut load_rows = Vec::new();
                for idx in 0..entries.len().min(max_rows) {
                    load_rows.push(([px + 18.0, row_y0 + idx as f32 * row_h, pw - 36.0, row_h - 3.0], idx));
                }
                let by = py + ph - 52.0;
                let mut buttons = vec![MatchMenuButton {
                    rect: [px + 18.0, by, 140.0, 34.0],
                    label: "BACK".to_string(),
                    action: MatchMenuAction::Resume,
                    active: false,
                    disabled: false,
                }];
                if let Some(sel) = entries.get(*selected) {
                    buttons.push(MatchMenuButton {
                        rect: [px + pw - 198.0, by, 180.0, 34.0],
                        label: "LOAD SELECTED".to_string(),
                        action: MatchMenuAction::LoadSelected(sel.path.clone()),
                        active: true,
                        disabled: false,
                    });
                } else {
                    buttons.push(MatchMenuButton {
                        rect: [px + pw - 198.0, by, 180.0, 34.0],
                        label: "LOAD SELECTED".to_string(),
                        action: MatchMenuAction::SelectLoad(0),
                        active: false,
                        disabled: true,
                    });
                }
                Some(Self { panel: [px, py, pw, ph], buttons, load_rows, graph: None })
            }
            MatchMenu::Score { series } => {
                let pw = (w - 60.0).min(860.0);
                let ph = (h - 60.0).min(560.0);
                let px = (w - pw) * 0.5;
                let py = (h - ph) * 0.5;
                let armies = session.world.armies().len().min(8);
                let table_h = 26.0 + armies as f32 * 22.0;
                let tabs_y = py + 54.0 + table_h + 10.0;
                let mut buttons = Vec::new();
                let tab_w = ((pw - 36.0 - 3.0 * 8.0) / 4.0).max(90.0);
                for (k, s) in Series::ALL.into_iter().enumerate() {
                    let lbl = match s {
                        Series::MassIncome => "MASS /S",
                        Series::EnergyIncome => "ENERGY /S",
                        Series::ArmyValue => "ARMY VALUE",
                        Series::UnitCount => "UNIT COUNT",
                    };
                    buttons.push(MatchMenuButton {
                        rect: [px + 18.0 + k as f32 * (tab_w + 8.0), tabs_y, tab_w, 26.0],
                        label: lbl.to_string(),
                        action: MatchMenuAction::SwitchScoreSeries(s),
                        active: *series == s,
                        disabled: false,
                    });
                }
                let gy = tabs_y + 34.0;
                let gh = (py + ph - 62.0 - gy).max(90.0);
                let graph = Some([px + 18.0, gy, pw - 36.0, gh]);

                let by = py + ph - 48.0;
                let can_replay = !session.is_replay() && session.recorder.is_some() && session.world.tick() > 0;
                buttons.push(MatchMenuButton {
                    rect: [px + 18.0, by, 164.0, 32.0],
                    label: "CONTINUE WATCHING".to_string(),
                    action: MatchMenuAction::ContinueWatching,
                    active: false,
                    disabled: false,
                });
                buttons.push(MatchMenuButton {
                    rect: [px + 190.0, by, 146.0, 32.0],
                    label: "WATCH REPLAY".to_string(),
                    action: MatchMenuAction::WatchReplay,
                    active: false,
                    disabled: !can_replay,
                });
                buttons.push(MatchMenuButton {
                    rect: [px + pw - 314.0, by, 144.0, 32.0],
                    label: "QUIT TO MENU".to_string(),
                    action: MatchMenuAction::QuitToMenu,
                    active: true,
                    disabled: false,
                });
                buttons.push(MatchMenuButton {
                    rect: [px + pw - 162.0, by, 144.0, 32.0],
                    label: "QUIT GAME".to_string(),
                    action: MatchMenuAction::QuitGame,
                    active: false,
                    disabled: false,
                });
                Some(Self { panel: [px, py, pw, ph], buttons, load_rows: Vec::new(), graph })
            }
        }
    }

    pub fn hit(&self, px: f32, py: f32) -> Option<MatchMenuAction> {
        for &(r, idx) in &self.load_rows {
            if rect_hit(r, px, py) {
                return Some(MatchMenuAction::SelectLoad(idx));
            }
        }
        for b in &self.buttons {
            if !b.disabled && rect_hit(b.rect, px, py) {
                return Some(b.action.clone());
            }
        }
        None
    }
}

/// Replay timeline hit result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimelineHit {
    Seek(f32),
    TogglePause,
    Slower,
    Faster,
    Perspective,
}

/// Phase 15 replay timeline bar layout (screen pixels, `[x, y, w, h]`).
#[derive(Debug, Clone)]
pub struct TimelineLayout {
    pub panel: [f32; 4],
    pub bar: [f32; 4],
    pub pause: [f32; 4],
    pub slower: [f32; 4],
    pub faster: [f32; 4],
    pub view: [f32; 4],
}

fn rect_hit(r: [f32; 4], px: f32, py: f32) -> bool {
    px >= r[0] && px <= r[0] + r[2] && py >= r[1] && py <= r[1] + r[3]
}

impl TimelineLayout {
    fn compute(w: f32, h: f32, minimap: [f32; 4], palette_right: f32, palette_top: f32) -> Self {
        let mut x0 = palette_right + 10.0;
        let mut x1 = minimap[0] - 14.0;
        let mut y = h - 74.0;
        if x1 - x0 < 300.0 {
            x0 = 12.0;
            x1 = (w - 12.0).min(minimap[0] - 14.0).max(x0 + 200.0);
            y = palette_top - 4.0 - 70.0;
        }
        let pw = x1 - x0;
        let by = y + 6.0;
        Self {
            panel: [x0, y, pw, 64.0],
            bar: [x0 + 8.0, y + 38.0, pw - 16.0, 18.0],
            pause: [x0 + 8.0, by, 64.0, 24.0],
            slower: [x0 + 78.0, by, 26.0, 24.0],
            faster: [x0 + 108.0, by, 26.0, 24.0],
            view: [x0 + pw - 8.0 - 104.0, by, 104.0, 24.0],
        }
    }

    /// Fraction along the seek bar for screen x (clamped to `[0, 1]`).
    pub fn frac_at(&self, px: f32) -> f32 {
        ((px - self.bar[0]) / self.bar[2].max(1.0)).clamp(0.0, 1.0)
    }

    pub fn hit(&self, px: f32, py: f32) -> Option<TimelineHit> {
        if rect_hit(self.pause, px, py) {
            Some(TimelineHit::TogglePause)
        } else if rect_hit(self.slower, px, py) {
            Some(TimelineHit::Slower)
        } else if rect_hit(self.faster, px, py) {
            Some(TimelineHit::Faster)
        } else if rect_hit(self.view, px, py) {
            Some(TimelineHit::Perspective)
        } else if rect_hit([self.bar[0] - 4.0, self.bar[1] - 6.0, self.bar[2] + 8.0, self.bar[3] + 12.0], px, py) {
            Some(TimelineHit::Seek(self.frac_at(px)))
        } else {
            None
        }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        rect_hit(self.panel, px, py)
    }
}

#[derive(Debug, Clone)]
pub struct HudButton {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub label: &'static str,
    pub action: HudAction,
    pub active: bool,
}

// Phase 23 bottom-left command cluster geometry.
const MARGIN: f32 = 6.0;
const SLOT: f32 = 54.0;
const SLOT_GAP: f32 = 4.0;
const CARD_W: f32 = 8.0 + 5.0 * SLOT + 4.0 * SLOT_GAP + 8.0;
const CARD_H: f32 = 52.0 + 3.0 * SLOT + 2.0 * SLOT_GAP + 10.0;
const ORD_BW: f32 = 66.0;
const ORD_BH: f32 = 26.0;
const ORD_COLS: usize = 4;
const ORD_W: f32 = 6.0 + ORD_COLS as f32 * (ORD_BW + 4.0) + 2.0;
const QUEUE_H: f32 = 78.0;
const ENH_ROW_H: f32 = 22.0;

/// Command card geometry (tabs, category chips, page button, grid slots).
#[derive(Debug, Clone)]
pub struct CardLayout {
    pub panel: [f32; 4],
    pub model: CardModel,
    pub tabs: Vec<([f32; 4], u8)>,
    /// Category chips; index `255` = "all".
    pub groups: Vec<([f32; 4], u8)>,
    pub page: Option<[f32; 4]>,
    /// One rect per entry of `model.page_items()`.
    pub slots: Vec<[f32; 4]>,
}

impl CardLayout {
    fn compute(model: CardModel, x0: f32, y0: f32) -> Self {
        let panel = [x0, y0, CARD_W, CARD_H];
        let mut tabs = Vec::new();
        let mut groups = Vec::new();
        let mut page = None;
        let mut slots = Vec::new();
        if model.source.is_some() {
            for t in 1..=4u8 {
                tabs.push(([x0 + 8.0 + (t - 1) as f32 * 50.0, y0 + 6.0, 46.0, 20.0], t));
            }
            if model.pages > 1 {
                page = Some([x0 + CARD_W - 8.0 - 66.0, y0 + 6.0, 66.0, 20.0]);
            }
            let n = model.groups.len() + 1;
            let inner = CARD_W - 16.0;
            let chip_w = ((inner - (n as f32 - 1.0) * 3.0) / n as f32).min(56.0);
            for k in 0..n {
                let idx = if k == 0 { 255 } else { (k - 1) as u8 };
                groups.push(([x0 + 8.0 + k as f32 * (chip_w + 3.0), y0 + 30.0, chip_w, 18.0], idx));
            }
            for i in 0..model.page_items().len() {
                let (c, r) = ((i % 5) as f32, (i / 5) as f32);
                slots.push([x0 + 8.0 + c * (SLOT + SLOT_GAP), y0 + 52.0 + r * (SLOT + SLOT_GAP), SLOT, SLOT]);
            }
        }
        Self { panel, model, tabs, groups, page, slots }
    }
}

/// Factory queue / upgrade panel geometry.
#[derive(Debug, Clone)]
pub struct QueueLayout {
    pub panel: [f32; 4],
    pub entries: Vec<([f32; 4], QueueEntry)>,
    /// Repeat toggle (factories only) and its state.
    pub repeat: Option<([f32; 4], bool)>,
    /// Upgrade button and its target type.
    pub upgrade: Option<([f32; 4], String)>,
    /// `(target, per-mille)` of an upgrade in progress.
    pub upgrading: Option<(String, i64)>,
}

impl QueueLayout {
    fn compute(session: &PlaySession, x0: f32, top: f32, full_w: f32) -> Option<Self> {
        let fac = session.queue_panel_unit();
        let upgrade = session.upgrade_target();
        let upgrading = session.upgrade_progress();
        if fac.is_none() && upgrade.is_none() && upgrading.is_none() {
            return None;
        }
        let panel = [x0, top - MARGIN - QUEUE_H, full_w, QUEUE_H];
        let right = x0 + full_w - 8.0;
        let btn_x = right - 128.0;
        let max_entries = ((btn_x - 8.0 - (x0 + 8.0)) / 48.0).floor().max(1.0) as usize;
        let entries = session
            .queue_panel_entries()
            .into_iter()
            .take(max_entries)
            .enumerate()
            .map(|(i, e)| ([x0 + 8.0 + i as f32 * 48.0, panel[1] + 24.0, 44.0, 44.0], e))
            .collect();
        let repeat = fac.and_then(|id| session.world.unit(id)).map(|u| ([btn_x, panel[1] + 22.0, 128.0, 22.0], u.repeat_queue));
        let upgrade = upgrade
            .filter(|_| upgrading.is_none())
            .map(|(t, _)| ([btn_x, panel[1] + 48.0, 128.0, 22.0], t));
        Some(Self { panel, entries, repeat, upgrade, upgrading })
    }
}

/// ACU / SCU enhancement panel geometry (one column per slot).
#[derive(Debug, Clone)]
pub struct EnhLayout {
    pub panel: [f32; 4],
    pub unit: crate::skirmish::UnitId,
    pub headers: Vec<([f32; 4], String)>,
    /// `(rect, row, index into PlaySession::enhancement_panel rows)`.
    pub rows: Vec<([f32; 4], EnhRow, usize)>,
}

impl EnhLayout {
    fn compute(session: &PlaySession, x0: f32, top: f32, full_w: f32) -> Option<Self> {
        let (unit, rows) = session.enhancement_panel()?;
        let mut slots: Vec<String> = Vec::new();
        for r in &rows {
            if !slots.contains(&r.slot) {
                slots.push(r.slot.clone());
            }
        }
        let ncols = slots.len().clamp(1, 4);
        slots.truncate(ncols);
        let max_rows = slots
            .iter()
            .map(|s| rows.iter().filter(|r| r.slot == *s).count())
            .max()
            .unwrap_or(0)
            .min(8);
        let ph = 24.0 + 16.0 + max_rows as f32 * (ENH_ROW_H + 2.0) + 6.0;
        let panel = [x0, top - MARGIN - ph, full_w, ph];
        let col_w = (full_w - 16.0 - (ncols as f32 - 1.0) * 6.0) / ncols as f32;
        let mut headers = Vec::new();
        let mut out = Vec::new();
        for (c, s) in slots.iter().enumerate() {
            let cx = x0 + 8.0 + c as f32 * (col_w + 6.0);
            headers.push(([cx, panel[1] + 24.0, col_w, 14.0], s.clone()));
            for (k, (idx, r)) in rows.iter().enumerate().filter(|(_, r)| r.slot == *s).take(max_rows).enumerate() {
                out.push(([cx, panel[1] + 40.0 + k as f32 * (ENH_ROW_H + 2.0), col_w, ENH_ROW_H], r.clone(), idx));
            }
        }
        Some(Self { panel, unit, headers, rows: out })
    }
}

#[derive(Debug, Clone)]
pub struct HudLayout {
    pub width: f32,
    pub height: f32,
    /// Order / system buttons.
    pub buttons: Vec<HudButton>,
    pub minimap: [f32; 4], // [x, y, size, size] in screen pixels
    pub timeline: Option<TimelineLayout>,
    pub menu_open: bool,
    pub card: CardLayout,
    pub orders_panel: [f32; 4],
    pub queue: Option<QueueLayout>,
    pub enh: Option<EnhLayout>,
    /// Bounding box of the whole bottom-left cluster.
    pub cluster: [f32; 4],
}

impl HudLayout {
    pub fn compute(session: &PlaySession, width: u32, height: u32) -> Self {
        let w = width.max(480) as f32;
        let h = height.max(360) as f32;
        let mm_size = (h * 0.24).clamp(136.0, 220.0);
        let minimap = [w - mm_size - 12.0, h - mm_size - 12.0, mm_size, mm_size];

        let x0 = MARGIN;
        let card_y = h - MARGIN - CARD_H;
        let card = CardLayout::compute(session.card_model(), x0, card_y);
        let orders_panel = [x0 + CARD_W + MARGIN, card_y, ORD_W, CARD_H];
        let mut buttons = Vec::new();
        for (i, (kind, active)) in session.order_buttons().into_iter().enumerate() {
            let (col, row) = (i % ORD_COLS, i / ORD_COLS);
            if row >= 7 {
                break;
            }
            buttons.push(HudButton {
                x: orders_panel[0] + 6.0 + col as f32 * (ORD_BW + 4.0),
                y: card_y + 6.0 + row as f32 * (ORD_BH + 6.0),
                w: ORD_BW,
                h: ORD_BH,
                label: kind.label(),
                action: HudAction::Order(kind),
                active,
            });
        }
        let full_w = CARD_W + MARGIN + ORD_W;
        let mut top = card_y;
        let queue = QueueLayout::compute(session, x0, top, full_w);
        if let Some(q) = &queue {
            top = q.panel[1];
        }
        let enh = EnhLayout::compute(session, x0, top, full_w);
        if let Some(e) = &enh {
            top = e.panel[1];
        }
        let cluster = [x0, top, full_w, h - MARGIN - top];
        let timeline = session
            .is_replay()
            .then(|| TimelineLayout::compute(w, h, minimap, x0 + full_w, card_y));

        Self {
            width: w,
            height: h,
            buttons,
            minimap,
            timeline,
            menu_open: session.menu.is_open(),
            card,
            orders_panel,
            queue,
            enh,
            cluster,
        }
    }

    fn panels(&self) -> impl Iterator<Item = [f32; 4]> + '_ {
        [Some(self.card.panel), Some(self.orders_panel)]
            .into_iter()
            .chain([self.queue.as_ref().map(|q| q.panel), self.enh.as_ref().map(|e| e.panel)])
            .flatten()
    }

    /// True if `(px, py)` is over any HUD element that should swallow world hover/picks.
    pub fn over_hud(&self, px: f32, py: f32) -> bool {
        self.menu_open
            || self.panels().any(|r| rect_hit(r, px, py))
            || self.hit_test_button(px, py).is_some()
            || self.hit_test_minimap(px, py).is_some()
            || self.timeline.as_ref().is_some_and(|t| t.contains(px, py))
    }

    pub fn hit_test_button(&self, px: f32, py: f32) -> Option<HudAction> {
        if let Some(b) = self
            .buttons
            .iter()
            .find(|b| px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h)
        {
            return Some(b.action);
        }
        let c = &self.card;
        if let Some(i) = c.slots.iter().position(|r| rect_hit(*r, px, py)) {
            return Some(HudAction::CardItem(i as u16));
        }
        if let Some((_, t)) = c.tabs.iter().find(|(r, t)| rect_hit(*r, px, py) && c.model.tab_counts[(*t - 1) as usize] > 0) {
            return Some(HudAction::CardTab(*t));
        }
        if let Some((_, g)) = c.groups.iter().find(|(r, _)| rect_hit(*r, px, py)) {
            return Some(HudAction::CardGroup(*g));
        }
        if c.page.is_some_and(|r| rect_hit(r, px, py)) {
            return Some(HudAction::CardPage);
        }
        if let Some(q) = &self.queue {
            if let Some(i) = q.entries.iter().position(|(r, _)| rect_hit(*r, px, py)) {
                return Some(HudAction::QueueItem(i as u16));
            }
            if q.repeat.as_ref().is_some_and(|(r, _)| rect_hit(*r, px, py)) {
                return Some(HudAction::Order(OrderKind::ToggleRepeat));
            }
            if q.upgrade.as_ref().is_some_and(|(r, _)| rect_hit(*r, px, py)) {
                return Some(HudAction::Upgrade);
            }
        }
        if let Some(e) = &self.enh
            && let Some((_, _, idx)) = e.rows.iter().find(|(r, _, _)| rect_hit(*r, px, py))
        {
            return Some(HudAction::EnhanceRow(*idx as u16));
        }
        None
    }

    /// If `(px, py)` lies inside the minimap, returns normalized map `[u, v]` in `[0, 1]^2`.
    pub fn hit_test_minimap(&self, px: f32, py: f32) -> Option<(f32, f32)> {
        let [mx, my, mw, mh] = self.minimap;
        if px >= mx && px <= mx + mw && py >= my && py <= my + mh {
            Some(((px - mx) / mw, (py - my) / mh))
        } else {
            None
        }
    }
}

/// Builds `(minimap_v, minimap_i, bg_v, bg_i, icons_v, icons_i, text_v, text_i)` for the 2D HUD pass.
pub fn build_hud_geometry(
    session: &PlaySession,
    map: &MapContext,
    camera: &OrbitCamera,
    width: u32,
    height: u32,
) -> (
    Vec<HudVertex>,
    Vec<u32>,
    Vec<HudVertex>,
    Vec<u32>,
    Vec<HudVertex>,
    Vec<u32>,
    Vec<HudVertex>,
    Vec<u32>,
) {
    let layout = HudLayout::compute(session, width, height);
    let w = layout.width;
    let h = layout.height;

    let mut mm_v = Vec::new();
    let mut mm_i = Vec::new();
    let mut v = Vec::with_capacity(2048);
    let mut i = Vec::with_capacity(3072);
    let mut ic_v = Vec::with_capacity(64);
    let mut ic_i = Vec::with_capacity(96);
    let mut fg_v = Vec::with_capacity(4096);
    let mut fg_i = Vec::with_capacity(6144);

    // 1. Minimap textured backdrop quad
    let [mx, my, mw, mh] = layout.minimap;
    push_quad_uv(&mut mm_v, &mut mm_i, mx, my, mw, mh, [1.0, 1.0, 1.0, 0.96], w, h);

    // 2. Top Economy & Status Bar
    push_rect(&mut v, &mut i, 0.0, 0.0, w, 46.0, [0.06, 0.08, 0.12, 0.84], w, h);
    push_rect(&mut v, &mut i, 0.0, 46.0, w, 2.0, [0.22, 0.45, 0.72, 0.85], w, h);

    if let Some(a) = session.world.army(session.human_army) {
        let m_fill = (a.mass as f32 / a.mass_cap.max(1) as f32).clamp(0.0, 1.0);
        let e_fill = (a.energy as f32 / a.energy_cap.max(1) as f32).clamp(0.0, 1.0);

        // Mass gauge
        push_rect(&mut v, &mut i, 14.0, 10.0, 180.0, 11.0, [0.12, 0.15, 0.18, 0.95], w, h);
        push_rect(&mut v, &mut i, 14.0, 10.0, 180.0 * m_fill, 11.0, [0.24, 0.86, 0.44, 0.95], w, h);
        let m_txt = format!(
            "M {}/{} (+{:.1})",
            a.mass / SCALE,
            a.mass_cap / SCALE,
            a.income_mass as f64 * TICK_HZ as f64 / SCALE as f64
        );
        draw_text(&mut fg_v, &mut fg_i, 14.0, 26.0, 1.6, &m_txt, [0.55, 0.98, 0.65, 1.0], w, h);

        // Energy gauge
        push_rect(&mut v, &mut i, 214.0, 10.0, 180.0, 11.0, [0.12, 0.15, 0.18, 0.95], w, h);
        push_rect(&mut v, &mut i, 214.0, 10.0, 180.0 * e_fill, 11.0, [0.96, 0.80, 0.22, 0.95], w, h);
        let e_txt = format!(
            "E {}/{} (+{:.0})",
            a.energy / SCALE,
            a.energy_cap / SCALE,
            a.income_energy as f64 * TICK_HZ as f64 / SCALE as f64
        );
        draw_text(&mut fg_v, &mut fg_i, 214.0, 26.0, 1.6, &e_txt, [1.0, 0.90, 0.42, 1.0], w, h);
    }

    let secs = session.world.tick() / TICK_HZ;
    let speed_str = if session.paused {
        "PAUSED".to_string()
    } else {
        format!("{}X", session.speed)
    };
    let n_units = session
        .world
        .units()
        .filter(|u| u.army == session.human_army && !u.role.is_structure())
        .count();
    let mid_txt = format!(
        "{:02}:{:02} [{}] UNITS:{} SEL:{}",
        secs / 60,
        secs % 60,
        speed_str,
        n_units,
        session.selected.len()
    );
    draw_text(&mut fg_v, &mut fg_i, 420.0, 16.0, 1.9, &mid_txt, [0.92, 0.95, 1.0, 1.0], w, h);

    if let Some(res) = session.world.result() {
        let banner = match res {
            GameResult::Victory(a) if a == session.human_army => "VICTORY!",
            GameResult::Victory(_) => "DEFEAT",
            GameResult::Draw => "DRAW",
        };
        push_rect(&mut v, &mut i, w * 0.5 - 120.0, 56.0, 240.0, 34.0, [0.06, 0.08, 0.12, 0.90], w, h);
        draw_text(&mut fg_v, &mut fg_i, w * 0.5 - 64.0, 65.0, 2.5, banner, [1.0, 0.85, 0.20, 1.0], w, h);
    } else if !session.status_banner.is_empty() {
        let trunc: String = session.status_banner.chars().take(48).collect();
        draw_text(&mut fg_v, &mut fg_i, 420.0, 32.0, 1.4, &trunc, [0.72, 0.82, 0.95, 0.95], w, h);
    }

    // Phase 15 lockstep status (ping / turn buffer) and desync / network-failure banner.
    if let Some(net) = &session.net {
        draw_text(&mut fg_v, &mut fg_i, 420.0, 44.0, 1.4, &net.hud_line(), [0.60, 0.95, 0.80, 0.95], w, h);
        if let Some(fail) = net.failure_banner() {
            let fail: String = fail.chars().take(60).collect();
            push_rect(&mut v, &mut i, w * 0.5 - 260.0, 96.0, 520.0, 30.0, [0.55, 0.04, 0.04, 0.92], w, h);
            draw_text(&mut fg_v, &mut fg_i, w * 0.5 - 250.0, 104.0, 2.0, &fail, [1.0, 0.92, 0.85, 1.0], w, h);
        }
    }

    // 3. Phase 23 bottom-left command cluster (card, orders, queue, enhancements)
    {
        let mut hb = HudBufs { v: &mut v, i: &mut i, ic_v: &mut ic_v, ic_i: &mut ic_i, fg_v: &mut fg_v, fg_i: &mut fg_i, w, h };
        draw_command_interface(session, &layout, &mut hb);
    }

    // 3b. Hover / selection stats card (above the minimap)
    if let Some((cid, count)) = session.card_unit() {
        let lines = crate::controls::unit_card_lines(&session.world, cid, count);
        if !lines.is_empty() {
            let ts = 1.5;
            let lh = 5.0 * ts + 5.0;
            let max_chars = lines.iter().map(|l| l.len()).max().unwrap_or(0) as f32;
            let cw = (max_chars * 4.0 * ts + 22.0).clamp(160.0, 340.0);
            let ch = lines.len() as f32 * lh + 12.0;
            let cx = w - cw - 12.0;
            let cy = (my - ch - 12.0).max(52.0);
            push_rect(&mut v, &mut i, cx, cy, cw, ch, [0.05, 0.07, 0.11, 0.88], w, h);
            if let Some(u) = session.world.unit(cid) {
                let rgb = army_rgb(&session.world, u.army);
                push_rect(&mut fg_v, &mut fg_i, cx, cy, 4.0, ch, [rgb[0], rgb[1], rgb[2], 1.0], w, h);
                let frac = (u.hp as f32 / u.max_hp.max(1) as f32).clamp(0.0, 1.0);
                push_rect(&mut fg_v, &mut fg_i, cx + 4.0, cy + ch - 3.0, (cw - 4.0) * frac, 3.0, [0.22, 0.92, 0.35, 0.95], w, h);
            }
            push_rect_outline(&mut fg_v, &mut fg_i, cx, cy, cw, ch, 1.2, [0.30, 0.44, 0.64, 0.95], w, h);
            for (li, line) in lines.iter().enumerate() {
                let col = if li == 0 { [1.0, 0.92, 0.55, 1.0] } else { [0.88, 0.93, 1.0, 1.0] };
                draw_text(&mut fg_v, &mut fg_i, cx + 12.0, cy + 7.0 + li as f32 * lh, ts, line, col, w, h);
            }
        }
    }

    // 3c. Replay timeline (progress, seek, pause, speed, perspective)
    if let (Some(tl), Some((cur, fin))) = (&layout.timeline, session.replay_progress()) {
        let [px, py, pw, ph] = tl.panel;
        push_rect(&mut v, &mut i, px, py, pw, ph, [0.05, 0.07, 0.11, 0.86], w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, px, py, pw, ph, 1.2, [0.30, 0.44, 0.64, 0.95], w, h);
        let btn = |v: &mut Vec<HudVertex>, i: &mut Vec<u32>, fv: &mut Vec<HudVertex>, fi: &mut Vec<u32>, r: [f32; 4], label: &str, active: bool| {
            let bg = if active { [0.18, 0.48, 0.78, 0.95] } else { [0.11, 0.15, 0.22, 0.92] };
            push_rect(v, i, r[0], r[1], r[2], r[3], bg, w, h);
            push_rect_outline(fv, fi, r[0], r[1], r[2], r[3], 1.2, [0.26, 0.34, 0.46, 0.95], w, h);
            draw_text(fv, fi, r[0] + 6.0, r[1] + 8.0, 1.45, label, [0.94, 0.97, 1.0, 1.0], w, h);
        };
        btn(&mut v, &mut i, &mut fg_v, &mut fg_i, tl.pause, if session.paused { "PLAY" } else { "PAUSE" }, session.paused);
        btn(&mut v, &mut i, &mut fg_v, &mut fg_i, tl.slower, "-", false);
        btn(&mut v, &mut i, &mut fg_v, &mut fg_i, tl.faster, "+", false);
        let persp = session.ui.replay.as_ref().and_then(|r| r.perspective);
        let view_label = match persp {
            Some(a) => format!("VIEW:ARMY{a}"),
            None => "VIEW:OBS".to_string(),
        };
        btn(&mut v, &mut i, &mut fg_v, &mut fg_i, tl.view, &view_label, persp.is_some());
        let info = format!(
            "{}X  {} / {}{}",
            session.speed,
            crate::controls::fmt_ticks(cur),
            crate::controls::fmt_ticks(fin),
            if session.seeking() { "  SEEKING.." } else { "" }
        );
        draw_text(&mut fg_v, &mut fg_i, tl.faster[0] + 36.0, tl.pause[1] + 7.0, 1.6, &info, [0.92, 0.95, 1.0, 1.0], w, h);

        let [bx, by, bw, bh] = tl.bar;
        push_rect(&mut v, &mut i, bx, by, bw, bh, [0.12, 0.15, 0.18, 0.95], w, h);
        let frac = (cur as f32 / fin.max(1) as f32).clamp(0.0, 1.0);
        push_rect(&mut v, &mut i, bx, by, bw * frac, bh, [0.24, 0.62, 0.95, 0.95], w, h);
        // minute ticks
        let minute = 60 * TICK_HZ;
        let mut t = minute;
        while t < fin && fin / minute < 120 {
            let x = bx + bw * (t as f32 / fin as f32);
            push_rect(&mut fg_v, &mut fg_i, x, by + bh - 5.0, 1.0, 5.0, [0.70, 0.78, 0.90, 0.7], w, h);
            t += minute;
        }
        let marker = session
            .ui
            .replay
            .as_ref()
            .and_then(|r| r.drag_tick.or(r.seek_target));
        if let Some(mt) = marker {
            let x = bx + bw * (mt as f32 / fin.max(1) as f32).clamp(0.0, 1.0);
            push_rect(&mut fg_v, &mut fg_i, x - 1.5, by - 4.0, 3.0, bh + 8.0, [1.0, 0.85, 0.25, 1.0], w, h);
        }
        let hx = bx + bw * frac;
        push_rect(&mut fg_v, &mut fg_i, hx - 2.0, by - 3.0, 4.0, bh + 6.0, [1.0, 1.0, 1.0, 1.0], w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, bx, by, bw, bh, 1.0, [0.30, 0.44, 0.64, 0.95], w, h);
    }

    // 4. Minimap border + live unit dots + camera viewport target
    push_rect_outline(&mut fg_v, &mut fg_i, mx - 2.0, my - 2.0, mw + 4.0, mh + 4.0, 2.0, [0.30, 0.44, 0.64, 0.95], w, h);
    let map_w = (map.size.0 as f32 / SCALE as f32).max(1.0);
    let map_h = (map.size.1 as f32 / SCALE as f32).max(1.0);

    for &(mpx, mpz) in session.world.mass_points() {
        let ux = mx + ((mpx as f32 / SCALE as f32) / map_w) * mw;
        let uy = my + ((mpz as f32 / SCALE as f32) / map_h) * mh;
        push_rect(&mut fg_v, &mut fg_i, ux - 1.5, uy - 1.5, 3.0, 3.0, [0.22, 0.90, 0.42, 0.75], w, h);
    }
    for u in session.world.units() {
        if session.fog_of_war && u.army != session.human_army {
            match session.world.intel_visibility(session.human_army, u) {
                IntelVisibility::Hidden => continue,
                IntelVisibility::RadarBlip => {
                    let ux = mx + ((u.pos.x as f32 / SCALE as f32) / map_w) * mw;
                    let uy = my + ((u.pos.y as f32 / SCALE as f32) / map_h) * mh;
                    push_rect(&mut fg_v, &mut fg_i, ux - 2.0, uy - 2.0, 4.0, 4.0, [0.96, 0.30, 0.25, 0.85], w, h);
                    continue;
                }
                IntelVisibility::Visible => {}
            }
        }
        let ux = mx + ((u.pos.x as f32 / SCALE as f32) / map_w) * mw;
        let uy = my + ((u.pos.y as f32 / SCALE as f32) / map_h) * mh;
        let dot = if u.role == Role::Commander { 5.0 } else if u.role.is_structure() { 3.5 } else { 2.5 };
        let rgb = army_rgb(&session.world, u.army);
        push_rect(
            &mut fg_v,
            &mut fg_i,
            ux - dot * 0.5,
            uy - dot * 0.5,
            dot,
            dot,
            [rgb[0], rgb[1], rgb[2], 1.0],
            w,
            h,
        );
    }
    // Phase 23 map pings (allies) on the minimap
    for (p, age) in session.chat.active_pings(Instant::now()) {
        let ux = mx + ((p.x as f32 / SCALE as f32) / map_w).clamp(0.0, 1.0) * mw;
        let uy = my + ((p.z as f32 / SCALE as f32) / map_h).clamp(0.0, 1.0) * mh;
        let rgb = p.kind.rgb();
        let a = (1.0 - age).clamp(0.2, 1.0);
        let r = 3.0 + 9.0 * ((age * 8.0) % 1.0);
        push_rect_outline(&mut fg_v, &mut fg_i, ux - r, uy - r, 2.0 * r, 2.0 * r, 1.5, [rgb[0], rgb[1], rgb[2], a * 0.8], w, h);
        push_rect(&mut fg_v, &mut fg_i, ux - 2.5, uy - 2.5, 5.0, 5.0, [rgb[0], rgb[1], rgb[2], a], w, h);
    }
    // Camera focus reticle on minimap
    let cx = mx + (camera.target[0] / map_w).clamp(0.0, 1.0) * mw;
    let cy = my + (camera.target[2] / map_h).clamp(0.0, 1.0) * mh;
    let rw = (camera.distance / map_w * mw * 0.45).clamp(10.0, mw * 0.6);
    let rh = rw / camera.aspect.max(0.5);
    push_rect_outline(
        &mut fg_v,
        &mut fg_i,
        (cx - rw * 0.5).clamp(mx, mx + mw - 4.0),
        (cy - rh * 0.5).clamp(my, my + mh - 4.0),
        rw.min(mw),
        rh.min(mh),
        1.2,
        [1.0, 1.0, 1.0, 0.9],
        w,
        h,
    );

    // 5. Phase 21 in-game chat log + input bar (above the bottom-left palette)
    let chat_base_y = layout.cluster[1] - 12.0;
    let now = Instant::now();
    let vis_chat = session.chat.visible(now);
    let mut cy_chat = if session.chat.is_typing() { chat_base_y - 26.0 } else { chat_base_y };
    if !vis_chat.is_empty() {
        let box_h = vis_chat.len() as f32 * 16.0 + 8.0;
        let top_y = cy_chat - box_h;
        push_rect(&mut fg_v, &mut fg_i, 12.0, top_y, 490.0, box_h, [0.03, 0.05, 0.08, 0.72], w, h);
        for (idx_line, (line, alpha)) in vis_chat.iter().enumerate() {
            let y = top_y + 5.0 + idx_line as f32 * 16.0;
            let (col, formatted) = match line.from {
                None => ([0.95, 0.85, 0.38, *alpha], line.text.clone()),
                Some(army_id) => {
                    let rgb = army_rgb(&session.world, army_id);
                    let tag = if line.allies { "[ALLIES] " } else { "" };
                    (
                        [rgb[0] * 0.5 + 0.5, rgb[1] * 0.5 + 0.5, rgb[2] * 0.5 + 0.5, *alpha],
                        format!("{tag}{}: {}", line.name, line.text),
                    )
                }
            };
            let trimmed: String = formatted.chars().take(74).collect();
            draw_text(&mut fg_v, &mut fg_i, 18.0, y, 1.45, &trimmed, col, w, h);
        }
        cy_chat = top_y - 4.0;
    }
    let _ = cy_chat;
    if let Some(ref inp) = session.chat.input {
        let iy = chat_base_y - 24.0;
        push_rect(&mut fg_v, &mut fg_i, 12.0, iy, 490.0, 22.0, [0.04, 0.08, 0.14, 0.92], w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, 12.0, iy, 490.0, 22.0, 1.2, [0.40, 0.82, 0.96, 0.95], w, h);
        let prefix = if inp.allies { "ALLIES> " } else { "ALL> " };
        let shown: String = format!("{prefix}{}_", inp.text).chars().take(74).collect();
        let col = if inp.allies { [0.45, 0.95, 0.65, 1.0] } else { [0.95, 0.97, 1.0, 1.0] };
        draw_text(&mut fg_v, &mut fg_i, 18.0, iy + 6.0, 1.5, &shown, col, w, h);
    }

    // 6. Phase 21 save-load progress bar (blocking sim while re-simulating)
    if let Some((cur_t, target_t)) = session.load_progress() {
        let bw = 420.0;
        let bh = 74.0;
        let bx = (w - bw) * 0.5;
        let by = (h - bh) * 0.5;
        push_rect(&mut fg_v, &mut fg_i, bx, by, bw, bh, [0.04, 0.06, 0.09, 0.94], w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, bx, by, bw, bh, 1.5, [0.42, 0.84, 0.96, 1.0], w, h);
        let frac = (cur_t as f32 / target_t.max(1) as f32).clamp(0.0, 1.0);
        let pct = (frac * 100.0).round() as u32;
        let txt = format!("LOADING SAVE  {cur_t} / {target_t} ({pct}%)");
        draw_text(&mut fg_v, &mut fg_i, bx + 18.0, by + 16.0, 1.65, &txt, [0.95, 0.90, 0.45, 1.0], w, h);
        push_rect(&mut fg_v, &mut fg_i, bx + 18.0, by + 42.0, bw - 36.0, 16.0, [0.12, 0.16, 0.20, 0.95], w, h);
        push_rect(&mut fg_v, &mut fg_i, bx + 18.0, by + 42.0, (bw - 36.0) * frac, 16.0, [0.28, 0.78, 0.95, 0.95], w, h);
    }

    // 6b. Phase 23 tooltip and targeting-mode label at the cursor
    if !session.menu.is_open()
        && let Some(cur) = session.ui.cursor_px
    {
        let mut hb = HudBufs { v: &mut v, i: &mut i, ic_v: &mut ic_v, ic_i: &mut ic_i, fg_v: &mut fg_v, fg_i: &mut fg_i, w, h };
        if let Some(hit) = layout.hit_test_button(cur.0, cur.1) {
            let lines = hud_tooltip(session, &layout, hit);
            // Cluster elements show their tooltip above the whole cluster.
            let anchor = if rect_hit(layout.cluster, cur.0, cur.1) { (cur.0, layout.cluster[1]) } else { cur };
            draw_tooltip(&mut hb, anchor, &lines);
        } else if !layout.over_hud(cur.0, cur.1) {
            let label = match (session.current_order_mode(), session.build_mode.as_deref()) {
                (Some(k), _) => Some((k.label().to_string(), k.rgb())),
                (None, Some(t)) => Some((format!("BUILD {t}"), [0.45, 1.0, 0.55])),
                _ => None,
            };
            if let Some((txt, rgb)) = label {
                let tw = text_w(&txt, 1.5) + 8.0;
                hb.fg([cur.0 + 16.0, cur.1 + 14.0, tw, 13.0], [0.0, 0.0, 0.0, 0.6]);
                hb.text(cur.0 + 20.0, cur.1 + 17.0, 1.5, &txt, [rgb[0], rgb[1], rgb[2], 1.0]);
            }
        }
    }

    // 7. Phase 21 Modal Menus (Pause, Load Save List, End-of-game Score Screen & Graphs)
    if let Some(ml) = MatchMenuLayout::compute(session, width, height) {
        push_rect(&mut fg_v, &mut fg_i, 0.0, 0.0, w, h, [0.0, 0.0, 0.0, 0.52], w, h);
        let [px, py, pw, ph] = ml.panel;
        push_rect(&mut fg_v, &mut fg_i, px, py, pw, ph, [0.05, 0.07, 0.11, 0.96], w, h);
        push_rect(&mut fg_v, &mut fg_i, px + 1.0, py + 1.0, pw - 2.0, 34.0, [0.10, 0.18, 0.25, 0.96], w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, px, py, pw, ph, 1.6, [0.40, 0.74, 0.94, 0.95], w, h);
        let title_str = match &session.menu {
            MatchMenu::Pause => {
                if session.net.is_some() { "MATCH PAUSED (NETWORK)" } else { "MATCH PAUSED" }
            }
            MatchMenu::LoadList { .. } => "LOAD SAVE GAME",
            MatchMenu::Score { .. } => match session.world.result() {
                Some(GameResult::Victory(a)) if a == session.human_army => "VICTORY - SCORE & STATS",
                Some(GameResult::Victory(_)) => "DEFEAT - SCORE & STATS",
                Some(GameResult::Draw) => "DRAW - SCORE & STATS",
                None => "MATCH STATISTICS",
            },
            MatchMenu::None => "",
        };
        draw_text(&mut fg_v, &mut fg_i, px + 16.0, py + 12.0, 2.0, title_str, [0.96, 0.84, 0.45, 1.0], w, h);

        if let MatchMenu::LoadList { entries, selected } = &session.menu {
            if entries.is_empty() {
                draw_text(&mut fg_v, &mut fg_i, px + 22.0, py + 72.0, 1.6, "NO SAVE FILES FOUND (PRESS F6 TO QUICKSAVE)", [0.68, 0.74, 0.80, 1.0], w, h);
            } else {
                for (r, idx) in &ml.load_rows {
                    let e = &entries[*idx];
                    let sel = *idx == *selected;
                    let bg = if sel { [0.14, 0.34, 0.48, 0.95] } else { [0.09, 0.12, 0.17, 0.92] };
                    let brd = if sel { [0.48, 0.88, 1.0, 1.0] } else { [0.22, 0.30, 0.38, 0.9] };
                    push_rect(&mut fg_v, &mut fg_i, r[0], r[1], r[2], r[3], bg, w, h);
                    push_rect_outline(&mut fg_v, &mut fg_i, r[0], r[1], r[2], r[3], 1.0, brd, w, h);
                    let row_txt = format!("{}  [{}]", e.label, e.map_id);
                    let trim: String = row_txt.chars().take(62).collect();
                    draw_text(&mut fg_v, &mut fg_i, r[0] + 8.0, r[1] + 8.0, 1.45, &trim, [0.92, 0.95, 1.0, 1.0], w, h);
                }
            }
        }

        if let MatchMenu::Score { series } = &session.menu {
            let rows = session.stats.rows(&session.world);
            let ty = py + 48.0;
            let hdr = format!("{:<10} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>4}", "ARMY", "BUILT", "LOST", "KILLS", "DAMAGE", "MASS+", "ENGY+", "APM");
            draw_text(&mut fg_v, &mut fg_i, px + 18.0, ty, 1.45, &hdr, [0.58, 0.72, 0.84, 1.0], w, h);
            for (ri, r) in rows.iter().take(8).enumerate() {
                let ry = ty + 20.0 + ri as f32 * 22.0;
                let rgb = army_rgb(&session.world, r.army);
                push_rect(&mut fg_v, &mut fg_i, px + 18.0, ry + 2.0, 8.0, 12.0, [rgb[0], rgb[1], rgb[2], 1.0], w, h);
                let nm: String = r.name.chars().take(9).collect();
                let line = format!(
                    "{:<10} {:>5} {:>5} {:>5} {:>7} {:>7} {:>7} {:>4.0}",
                    nm, r.built, r.lost, r.kills, r.damage, r.mass_collected, r.energy_collected, r.apm
                );
                let col = if r.defeated { [0.62, 0.66, 0.72, 1.0] } else { [0.92, 0.96, 1.0, 1.0] };
                draw_text(&mut fg_v, &mut fg_i, px + 30.0, ry + 2.0, 1.45, &line, col, w, h);
            }

            if let Some([gx, gy, gw, gh]) = ml.graph {
                push_rect(&mut fg_v, &mut fg_i, gx, gy, gw, gh, [0.025, 0.035, 0.055, 0.96], w, h);
                push_rect_outline(&mut fg_v, &mut fg_i, gx, gy, gw, gh, 1.0, [0.26, 0.36, 0.46, 0.95], w, h);
                let inner_x = gx + 12.0;
                let inner_y = gy + 24.0;
                let inner_w = (gw - 24.0).max(20.0);
                let inner_h = (gh - 36.0).max(20.0);
                let max_val = session.stats.max_of(*series).max(1.0);
                let max_tick = session.stats.last_sample_tick().max(1);
                draw_text(&mut fg_v, &mut fg_i, gx + 10.0, gy + 7.0, 1.4, &format!("{} (PEAK {:.0})", series.title(), max_val), [0.85, 0.90, 0.96, 1.0], w, h);
                for g in 1..4 {
                    let y = inner_y + inner_h * (g as f32 / 4.0);
                    push_rect(&mut fg_v, &mut fg_i, inner_x, y, inner_w, 1.0, [0.16, 0.22, 0.28, 0.8], w, h);
                }
                for (ai, ser) in session.stats.series.iter().enumerate() {
                    if ser.len() < 2 {
                        continue;
                    }
                    let rgb = army_rgb(&session.world, ai as u8);
                    let col = [rgb[0], rgb[1], rgb[2], 0.98];
                    for pair in ser.windows(2) {
                        let x0 = inner_x + (pair[0].tick as f32 / max_tick as f32) * inner_w;
                        let y0 = inner_y + inner_h * (1.0 - (series.value(&pair[0]) / max_val).clamp(0.0, 1.0));
                        let x1 = inner_x + (pair[1].tick as f32 / max_tick as f32) * inner_w;
                        let y1 = inner_y + inner_h * (1.0 - (series.value(&pair[1]) / max_val).clamp(0.0, 1.0));
                        push_line(&mut fg_v, &mut fg_i, x0, y0, x1, y1, 2.2, col, w, h);
                    }
                }
            }
        }

        for b in &ml.buttons {
            let bg = if b.disabled {
                [0.08, 0.10, 0.12, 0.85]
            } else if b.active {
                [0.14, 0.42, 0.64, 0.96]
            } else {
                [0.11, 0.15, 0.22, 0.95]
            };
            let brd = if b.disabled {
                [0.20, 0.24, 0.28, 0.9]
            } else if b.active {
                [0.48, 0.90, 1.0, 1.0]
            } else {
                [0.30, 0.40, 0.52, 0.95]
            };
            push_rect(&mut fg_v, &mut fg_i, b.rect[0], b.rect[1], b.rect[2], b.rect[3], bg, w, h);
            push_rect_outline(&mut fg_v, &mut fg_i, b.rect[0], b.rect[1], b.rect[2], b.rect[3], 1.2, brd, w, h);
            let col = if b.disabled { [0.46, 0.50, 0.56, 1.0] } else { [0.94, 0.97, 1.0, 1.0] };
            let tx = b.rect[0] + 10.0;
            let ty = b.rect[1] + (b.rect[3] - 7.5) * 0.5;
            draw_text(&mut fg_v, &mut fg_i, tx, ty, 1.45, &b.label, col, w, h);
        }
    }

    (mm_v, mm_i, v, i, ic_v, ic_i, fg_v, fg_i)
}

/// Geometry buffers of the 2D HUD pass (`bg` under icons, `ic` = unit icon atlas, `fg` on top).
struct HudBufs<'a> {
    v: &'a mut Vec<HudVertex>,
    i: &'a mut Vec<u32>,
    ic_v: &'a mut Vec<HudVertex>,
    ic_i: &'a mut Vec<u32>,
    fg_v: &'a mut Vec<HudVertex>,
    fg_i: &'a mut Vec<u32>,
    w: f32,
    h: f32,
}

impl HudBufs<'_> {
    fn bg(&mut self, r: [f32; 4], c: [f32; 4]) {
        push_rect(self.v, self.i, r[0], r[1], r[2], r[3], c, self.w, self.h);
    }
    fn fg(&mut self, r: [f32; 4], c: [f32; 4]) {
        push_rect(self.fg_v, self.fg_i, r[0], r[1], r[2], r[3], c, self.w, self.h);
    }
    fn outline(&mut self, r: [f32; 4], t: f32, c: [f32; 4]) {
        push_rect_outline(self.fg_v, self.fg_i, r[0], r[1], r[2], r[3], t, c, self.w, self.h);
    }
    fn text(&mut self, x: f32, y: f32, scale: f32, s: &str, c: [f32; 4]) {
        draw_text(self.fg_v, self.fg_i, x, y, scale, s, c, self.w, self.h);
    }
    /// Draws `unit_type`'s atlas icon into `r`; false if the atlas has no such icon.
    fn icon(&mut self, unit_type: &str, r: [f32; 4], alpha: f32) -> bool {
        let Some([u0, v0, u1, v1]) = crate::command_card::icon_uv(unit_type) else {
            return false;
        };
        let (w, h) = (self.w, self.h);
        let c = [1.0, 1.0, 1.0, alpha];
        let base = self.ic_v.len() as u32;
        self.ic_v.extend_from_slice(&[
            HudVertex { pos: to_ndc(r[0], r[1], w, h), uv: [u0, v0], color: c },
            HudVertex { pos: to_ndc(r[0] + r[2], r[1], w, h), uv: [u1, v0], color: c },
            HudVertex { pos: to_ndc(r[0] + r[2], r[1] + r[3], w, h), uv: [u1, v1], color: c },
            HudVertex { pos: to_ndc(r[0], r[1] + r[3], w, h), uv: [u0, v1], color: c },
        ]);
        self.ic_i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        true
    }
    fn button(&mut self, r: [f32; 4], label: &str, active: bool, enabled: bool) {
        let bg = if !enabled {
            [0.07, 0.09, 0.11, 0.85]
        } else if active {
            [0.18, 0.48, 0.78, 0.95]
        } else {
            [0.11, 0.15, 0.22, 0.92]
        };
        let brd = if active { [0.45, 0.92, 1.0, 1.0] } else { [0.26, 0.34, 0.46, 0.95] };
        self.bg(r, bg);
        self.outline(r, 1.2, brd);
        let col = if enabled { [0.94, 0.97, 1.0, 1.0] } else { [0.45, 0.50, 0.56, 1.0] };
        let tw = text_w(label, 1.35);
        self.text(r[0] + ((r[2] - tw) * 0.5).max(3.0), r[1] + (r[3] - 6.75) * 0.5, 1.35, label, col);
    }
}

fn text_w(s: &str, scale: f32) -> f32 {
    s.len() as f32 * 4.0 * scale
}

fn trunc(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn fmt_cost(v: i64) -> String {
    let whole = v / SCALE;
    if whole >= 10_000 { format!("{}K", whole / 1000) } else { whole.to_string() }
}

/// Phase 23 bottom-left cluster: command card, orders, queue / upgrade, enhancements.
fn draw_command_interface(session: &PlaySession, layout: &HudLayout, b: &mut HudBufs) {
    let keymap = crate::keymap::active();
    let card_keys = crate::settings::card_hotkeys_enabled();
    let panel_bg = [0.04, 0.06, 0.10, 0.86];
    let panel_brd = [0.30, 0.44, 0.64, 0.95];

    // --- Command card -------------------------------------------------------
    let c = &layout.card;
    b.bg(c.panel, panel_bg);
    b.outline(c.panel, 1.2, panel_brd);
    match c.model.source {
        None => {
            let msg = if session.is_replay() { "REPLAY" } else { "SELECT A BUILDER" };
            b.text(c.panel[0] + 14.0, c.panel[1] + 14.0, 1.6, msg, [0.55, 0.62, 0.70, 1.0]);
            if !session.is_replay() {
                b.text(c.panel[0] + 14.0, c.panel[1] + 32.0, 1.6, "OR A FACTORY", [0.55, 0.62, 0.70, 1.0]);
                b.text(c.panel[0] + 14.0, c.panel[1] + 60.0, 1.3, "F: COMMANDER  H: FACTORIES", [0.42, 0.48, 0.56, 1.0]);
                b.text(c.panel[0] + 14.0, c.panel[1] + 74.0, 1.3, ".: IDLE ENGINEER", [0.42, 0.48, 0.56, 1.0]);
            }
        }
        Some(src) => {
            for (r, t) in &c.tabs {
                let n = c.model.tab_counts[(*t - 1) as usize];
                let label = format!("{}:{n}", crate::command_card::tab_label(*t));
                b.button(*r, &label, *t == c.model.tab, n > 0);
            }
            if let Some(r) = c.page {
                b.button(r, &format!("PG {}/{}", c.model.page + 1, c.model.pages), false, true);
            }
            for (r, g) in &c.groups {
                let (label, active) = if *g == 255 {
                    ("ALL", c.model.group.is_none())
                } else {
                    let grp = c.model.groups[*g as usize];
                    (grp.label(), c.model.group == Some(grp))
                };
                b.button(*r, label, active, true);
            }
            for (k, (r, item)) in c.slots.iter().zip(c.model.page_items()).enumerate() {
                let placing = session.build_mode.as_deref() == Some(item.unit_type.as_str());
                let hover = session.ui.hud_hover == Some(HudAction::CardItem(k as u16));
                let bg = if placing {
                    [0.18, 0.48, 0.78, 0.95]
                } else if hover {
                    [0.16, 0.24, 0.34, 0.95]
                } else {
                    [0.09, 0.12, 0.18, 0.95]
                };
                b.bg(*r, bg);
                let brd = if placing { [0.45, 0.92, 1.0, 1.0] } else { [0.24, 0.32, 0.44, 0.95] };
                b.outline(*r, 1.2, brd);
                if !b.icon(&item.unit_type, [r[0] + 3.0, r[1] + 3.0, r[2] - 6.0, r[3] - 6.0], 0.98) {
                    b.text(r[0] + 4.0, r[1] + 16.0, 1.3, &trunc(&item.unit_type, 8), [0.92, 0.95, 1.0, 1.0]);
                    let name = session.world.catalog().get(&item.unit_type).map(|t| t.name.trim().to_string()).unwrap_or_default();
                    b.text(r[0] + 4.0, r[1] + 28.0, 1.1, &trunc(&name, 9), [0.62, 0.70, 0.80, 1.0]);
                }
                if card_keys && let Some(k) = keymap.label(crate::keymap::Action::CardSlot(k as u8)) {
                    let kw = text_w(&k, 1.3) + 4.0;
                    b.fg([r[0] + 1.0, r[1] + 1.0, kw, 10.0], [0.0, 0.0, 0.0, 0.65]);
                    b.text(r[0] + 3.0, r[1] + 2.5, 1.3, &k, [1.0, 0.88, 0.35, 1.0]);
                }
                if src == CardSource::Production {
                    let n = session.queued_count(&item.unit_type);
                    if n > 0 {
                        let s = format!("{n}");
                        let tw = text_w(&s, 1.6) + 4.0;
                        b.fg([r[0] + r[2] - tw - 1.0, r[1] + r[3] - 12.0, tw, 11.0], [0.0, 0.0, 0.0, 0.7]);
                        b.text(r[0] + r[2] - tw + 1.0, r[1] + r[3] - 10.5, 1.6, &s, [0.55, 1.0, 0.60, 1.0]);
                    }
                }
            }
        }
    }

    // --- Orders panel -------------------------------------------------------
    b.bg(layout.orders_panel, panel_bg);
    b.outline(layout.orders_panel, 1.2, panel_brd);
    for btn in &layout.buttons {
        let r = [btn.x, btn.y, btn.w, btn.h];
        let HudAction::Order(kind) = btn.action else {
            continue;
        };
        let hover = session.ui.hud_hover == Some(btn.action);
        let bg = if btn.active {
            [0.18, 0.48, 0.78, 0.95]
        } else if hover {
            [0.16, 0.24, 0.34, 0.95]
        } else {
            [0.11, 0.15, 0.22, 0.92]
        };
        b.bg(r, bg);
        let rgb = kind.rgb();
        b.fg([r[0], r[1], 3.0, r[3]], [rgb[0], rgb[1], rgb[2], 0.95]);
        b.outline(r, 1.2, if btn.active { [0.45, 0.92, 1.0, 1.0] } else { [0.26, 0.34, 0.46, 0.95] });
        let label = if kind == OrderKind::IdleEngineer {
            format!("IDLE:{}", session.idle_engineers().len())
        } else if kind == OrderKind::LaunchMissile {
            let n: u32 = session
                .selected
                .iter()
                .filter_map(|id| session.world.unit(*id))
                .filter(|u| u.role == Role::MissileSilo)
                .map(|u| u.silo_missiles)
                .sum();
            format!("MISSILE:{n}")
        } else {
            btn.label.to_string()
        };
        b.text(r[0] + 6.0, r[1] + 4.0, 1.35, &label, [0.94, 0.97, 1.0, 1.0]);
        if let Some(k) = kind.key_action().and_then(|a| keymap.label(a)) {
            b.text(r[0] + 6.0, r[1] + 15.5, 1.1, &k, [0.62, 0.72, 0.84, 1.0]);
        }
    }

    // --- Factory queue / upgrade -------------------------------------------
    if let Some(q) = &layout.queue {
        b.bg(q.panel, panel_bg);
        b.outline(q.panel, 1.2, panel_brd);
        let title = match session.queue_panel_unit().and_then(|id| session.world.unit(id)) {
            Some(u) => format!("QUEUE  {}", trunc(&session.world.catalog().get(&u.type_id).map(|t| t.name.trim().to_string()).unwrap_or_else(|| u.type_id.clone()), 28)),
            None => "UPGRADE".to_string(),
        };
        b.text(q.panel[0] + 8.0, q.panel[1] + 7.0, 1.5, &title, [1.0, 0.92, 0.55, 1.0]);
        if q.entries.is_empty() && session.queue_panel_unit().is_some() {
            b.text(q.panel[0] + 10.0, q.panel[1] + 40.0, 1.35, "EMPTY - CLICK A UNIT ON THE CARD", [0.50, 0.58, 0.66, 1.0]);
        }
        for (k, (r, e)) in q.entries.iter().enumerate() {
            let hover = session.ui.hud_hover == Some(HudAction::QueueItem(k as u16));
            b.bg(*r, if hover { [0.30, 0.12, 0.12, 0.95] } else { [0.09, 0.12, 0.18, 0.95] });
            b.outline(*r, 1.0, if e.progress.is_some() { [0.45, 0.92, 1.0, 1.0] } else { [0.24, 0.32, 0.44, 0.95] });
            if !b.icon(&e.unit_type, [r[0] + 2.0, r[1] + 2.0, r[2] - 4.0, r[3] - 4.0], 0.98) {
                b.text(r[0] + 3.0, r[1] + 16.0, 1.15, &trunc(&e.unit_type, 7), [0.92, 0.95, 1.0, 1.0]);
            }
            let s = format!("{}", e.count);
            let tw = text_w(&s, 1.5) + 4.0;
            b.fg([r[0] + r[2] - tw - 1.0, r[1] + r[3] - 11.0, tw, 10.0], [0.0, 0.0, 0.0, 0.7]);
            b.text(r[0] + r[2] - tw + 1.0, r[1] + r[3] - 9.5, 1.5, &s, [1.0, 1.0, 1.0, 1.0]);
            if let Some(p) = e.progress {
                b.fg([r[0], r[1] + r[3] + 2.0, r[2], 4.0], [0.12, 0.15, 0.18, 0.95]);
                b.fg([r[0], r[1] + r[3] + 2.0, r[2] * p as f32 / 1000.0, 4.0], [0.30, 0.90, 0.45, 0.95]);
            }
        }
        if let Some((r, on)) = &q.repeat {
            b.button(*r, if *on { "REPEAT: ON" } else { "REPEAT: OFF" }, *on, true);
        }
        if let Some((r, target)) = &q.upgrade {
            let tier = session.world.catalog().get(target).map(|t| t.tier).unwrap_or(0);
            b.button(*r, &format!("UPGRADE > T{tier}"), session.ui.hud_hover == Some(HudAction::Upgrade), true);
        }
        if let Some((_, p)) = &q.upgrading {
            let r = [q.panel[0] + q.panel[2] - 136.0, q.panel[1] + 48.0, 128.0, 22.0];
            b.bg(r, [0.09, 0.12, 0.18, 0.95]);
            b.bg([r[0], r[1], r[2] * *p as f32 / 1000.0, r[3]], [0.20, 0.55, 0.30, 0.95]);
            b.outline(r, 1.0, [0.45, 0.92, 1.0, 1.0]);
            b.text(r[0] + 5.0, r[1] + 8.0, 1.3, &format!("UPGRADING {}%", p / 10), [1.0, 1.0, 1.0, 1.0]);
        }
    }

    // --- Enhancements -------------------------------------------------------
    if let Some(e) = &layout.enh {
        b.bg(e.panel, panel_bg);
        b.outline(e.panel, 1.2, panel_brd);
        let name = session
            .world
            .unit(e.unit)
            .and_then(|u| session.world.catalog().get(&u.type_id))
            .map(|t| t.name.trim().to_string())
            .unwrap_or_default();
        b.text(e.panel[0] + 8.0, e.panel[1] + 7.0, 1.5, &format!("ENHANCEMENTS  {}", trunc(&name, 30)), [1.0, 0.92, 0.55, 1.0]);
        for (r, slot) in &e.headers {
            b.text(r[0] + 2.0, r[1] + 2.0, 1.3, crate::command_card::slot_label(slot), [0.55, 0.72, 0.90, 1.0]);
        }
        for (r, row, idx) in &e.rows {
            let hover = session.ui.hud_hover == Some(HudAction::EnhanceRow(*idx as u16));
            let (bg, txt) = match row.state {
                EnhState::Installed => ([0.10, 0.30, 0.16, 0.95], [0.70, 1.0, 0.72, 1.0]),
                EnhState::Available => (
                    if hover { [0.16, 0.30, 0.46, 0.95] } else { [0.11, 0.18, 0.28, 0.95] },
                    [0.94, 0.97, 1.0, 1.0],
                ),
                EnhState::Locked => ([0.07, 0.08, 0.10, 0.90], [0.45, 0.50, 0.56, 1.0]),
                EnhState::InProgress(_) => ([0.09, 0.12, 0.18, 0.95], [1.0, 0.95, 0.70, 1.0]),
            };
            b.bg(*r, bg);
            if let EnhState::InProgress(p) = row.state {
                b.bg([r[0], r[1], r[2] * p as f32 / 1000.0, r[3]], [0.22, 0.50, 0.28, 0.95]);
            }
            b.outline(*r, 1.0, if hover { [0.45, 0.92, 1.0, 1.0] } else { [0.22, 0.30, 0.40, 0.95] });
            let cost = match row.state {
                EnhState::Installed => "OK".to_string(),
                EnhState::InProgress(p) => format!("{}%", p / 10),
                _ => format!("M{}", fmt_cost(row.cost_mass)),
            };
            let cw = text_w(&cost, 1.3);
            let max_chars = (((r[2] - cw - 14.0) / (4.0 * 1.3)).floor().max(3.0)) as usize;
            b.text(r[0] + 5.0, r[1] + 8.0, 1.3, &trunc(&row.display.to_ascii_uppercase(), max_chars), txt);
            b.text(r[0] + r[2] - cw - 5.0, r[1] + 8.0, 1.3, &cost, txt);
        }
    }
}

/// Tooltip lines for the HUD element `hit`.
fn hud_tooltip(session: &PlaySession, layout: &HudLayout, hit: HudAction) -> Vec<String> {
    let cat = session.world.catalog();
    let keymap = crate::keymap::active();
    match hit {
        HudAction::CardItem(i) => {
            let Some(item) = layout.card.model.page_items().get(i as usize) else {
                return Vec::new();
            };
            let mut v = crate::command_card::item_tooltip(cat, &item.unit_type, session.card_build_rate());
            v.push(match layout.card.model.source {
                Some(CardSource::Production) => "LMB QUEUE  SHIFT X5  RMB REMOVE".to_string(),
                _ => "LMB PLACE  DRAG = LINE  SHIFT QUEUE".to_string(),
            });
            v
        }
        HudAction::Order(k) => {
            let mut v = vec![k.label().to_string()];
            if let Some(key) = k.key_action().map(|a| keymap.get(a)).filter(|c| !c.is_empty()) {
                v.push(format!("KEY {}", crate::keymap::chords_to_string(key).to_ascii_uppercase()));
            }
            if k.is_targeted() {
                v.push("THEN CLICK THE MAP (SHIFT QUEUE)".to_string());
            }
            v
        }
        HudAction::QueueItem(i) => match layout.queue.as_ref().and_then(|q| q.entries.get(i as usize)) {
            Some((_, e)) => {
                let name = cat.get(&e.unit_type).map(|t| t.name.trim().to_string()).unwrap_or_else(|| e.unit_type.clone());
                vec![format!("{name} X{}", e.count), "CLICK: CANCEL 1  SHIFT: CANCEL 5".to_string()]
            }
            None => Vec::new(),
        },
        HudAction::Upgrade => match layout.queue.as_ref().and_then(|q| q.upgrade.as_ref()) {
            Some((_, t)) => {
                let mut v = crate::command_card::item_tooltip(cat, t, 0);
                v.insert(0, "UPGRADE SELECTED TO".to_string());
                v
            }
            None => Vec::new(),
        },
        HudAction::EnhanceRow(i) => match session.enhancement_panel().and_then(|(_, rows)| rows.get(i as usize).cloned()) {
            Some(r) => {
                let rate = session.card_build_rate().max(1);
                let mut v = vec![
                    format!("{} ({})", r.display.to_ascii_uppercase(), crate::command_card::slot_label(&r.slot)),
                    format!(
                        "MASS {}  ENERGY {}  TIME {:.0}S",
                        r.cost_mass / SCALE,
                        r.cost_energy / SCALE,
                        r.build_work as f64 / rate as f64 / TICK_HZ as f64
                    ),
                ];
                v.push(match (r.state, &r.prerequisite) {
                    (EnhState::Installed, _) => "INSTALLED".to_string(),
                    (EnhState::InProgress(p), _) => format!("INSTALLING {}%", p / 10),
                    (EnhState::Locked, Some(p)) => format!("REQUIRES {}", p.to_ascii_uppercase()),
                    (EnhState::Locked, None) => "SLOT OCCUPIED".to_string(),
                    (EnhState::Available, _) => "CLICK TO INSTALL".to_string(),
                });
                v
            }
            None => Vec::new(),
        },
        HudAction::CardTab(t) => vec![
            format!("TECH {} ({} ENTRIES)", crate::command_card::tab_label(t), layout.card.model.tab_counts[(t.clamp(1, 4) - 1) as usize]),
            format!("NEXT TAB: {}", keymap.label(crate::keymap::Action::CardNextTab).unwrap_or_default()),
        ],
        _ => Vec::new(),
    }
}

fn draw_tooltip(b: &mut HudBufs, cursor: (f32, f32), lines: &[String]) {
    if lines.is_empty() {
        return;
    }
    let ts = 1.4;
    let lh = 5.0 * ts + 5.0;
    let tw = lines.iter().map(|l| text_w(l, ts)).fold(0.0, f32::max) + 16.0;
    let th = lines.len() as f32 * lh + 10.0;
    let x = (cursor.0 + 16.0).min(b.w - tw - 4.0).max(4.0);
    let y = (cursor.1 - th - 10.0).max(52.0);
    b.fg([x, y, tw, th], [0.02, 0.03, 0.05, 0.94]);
    b.outline([x, y, tw, th], 1.2, [0.45, 0.80, 0.96, 0.95]);
    for (k, l) in lines.iter().enumerate() {
        let col = if k == 0 { [1.0, 0.92, 0.55, 1.0] } else { [0.88, 0.93, 1.0, 1.0] };
        b.text(x + 8.0, y + 6.0 + k as f32 * lh, ts, l, col);
    }
}

#[inline]
pub(crate) fn to_ndc(px: f32, py: f32, w: f32, h: f32) -> [f32; 2] {
    [(2.0 * px / w) - 1.0, 1.0 - (2.0 * py / h)]
}

fn push_quad_uv(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    qw: f32,
    qh: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    let base = verts.len() as u32;
    verts.extend_from_slice(&[
        HudVertex { pos: to_ndc(x, y, w, h), uv: [0.0, 0.0], color },
        HudVertex { pos: to_ndc(x + qw, y, w, h), uv: [1.0, 0.0], color },
        HudVertex { pos: to_ndc(x + qw, y + qh, w, h), uv: [1.0, 1.0], color },
        HudVertex { pos: to_ndc(x, y + qh, w, h), uv: [0.0, 1.0], color },
    ]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

pub(crate) fn push_rect(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    rw: f32,
    rh: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    push_quad_uv(verts, indices, x, y, rw, rh, color, w, h);
}

pub(crate) fn push_line(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = dx.hypot(dy);
    if len <= 1e-4 {
        return;
    }
    let hw = thickness * 0.5;
    let nx = -dy / len * hw;
    let ny = dx / len * hw;
    let base = verts.len() as u32;
    verts.extend_from_slice(&[
        HudVertex { pos: to_ndc(x0 + nx, y0 + ny, w, h), uv: [0.0, 0.0], color },
        HudVertex { pos: to_ndc(x1 + nx, y1 + ny, w, h), uv: [1.0, 0.0], color },
        HudVertex { pos: to_ndc(x1 - nx, y1 - ny, w, h), uv: [1.0, 1.0], color },
        HudVertex { pos: to_ndc(x0 - nx, y0 - ny, w, h), uv: [0.0, 1.0], color },
    ]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

pub(crate) fn push_rect_outline(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    rw: f32,
    rh: f32,
    t: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    push_rect(verts, indices, x, y, rw, t, color, w, h);
    push_rect(verts, indices, x, y + rh - t, rw, t, color, w, h);
    push_rect(verts, indices, x, y, t, rh, color, w, h);
    push_rect(verts, indices, x + rw - t, y, t, rh, color, w, h);
}

pub(crate) fn draw_text(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    scale: f32,
    text: &str,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    let mut cx = x;
    for ch in text.bytes() {
        let g = glyph_3x5(ch.to_ascii_uppercase());
        for row in 0..5usize {
            let bits = g[row];
            for col in 0..3usize {
                if (bits >> (2 - col)) & 1 != 0 {
                    push_rect(
                        verts,
                        indices,
                        cx + col as f32 * scale,
                        y + row as f32 * scale,
                        scale,
                        scale,
                        color,
                        w,
                        h,
                    );
                }
            }
        }
        cx += 4.0 * scale;
    }
}

/// Compact 3x5 bitmap glyphs (3 bits per row x 5 rows) for legible HUD text.
fn glyph_3x5(c: u8) -> [u8; 5] {
    match c {
        b'0' => [7, 5, 5, 5, 7],
        b'1' => [2, 6, 2, 2, 7],
        b'2' => [7, 1, 7, 4, 7],
        b'3' => [7, 1, 7, 1, 7],
        b'4' => [5, 5, 7, 1, 1],
        b'5' => [7, 4, 7, 1, 7],
        b'6' => [7, 4, 7, 5, 7],
        b'7' => [7, 1, 1, 2, 2],
        b'8' => [7, 5, 7, 5, 7],
        b'9' => [7, 5, 7, 1, 7],
        b'A' => [2, 5, 7, 5, 5],
        b'B' => [6, 5, 6, 5, 6],
        b'C' => [7, 4, 4, 4, 7],
        b'D' => [6, 5, 5, 5, 6],
        b'E' => [7, 4, 6, 4, 7],
        b'F' => [7, 4, 6, 4, 4],
        b'G' => [7, 4, 5, 5, 7],
        b'H' => [5, 5, 7, 5, 5],
        b'I' => [7, 2, 2, 2, 7],
        b'J' => [1, 1, 1, 5, 7],
        b'K' => [5, 5, 6, 5, 5],
        b'L' => [4, 4, 4, 4, 7],
        b'M' => [5, 7, 7, 5, 5],
        b'N' => [6, 5, 5, 5, 5],
        b'O' => [7, 5, 5, 5, 7],
        b'P' => [7, 5, 7, 4, 4],
        b'Q' => [7, 5, 5, 7, 3],
        b'R' => [7, 5, 6, 5, 5],
        b'S' => [7, 4, 7, 1, 7],
        b'T' => [7, 2, 2, 2, 2],
        b'U' => [5, 5, 5, 5, 7],
        b'V' => [5, 5, 5, 5, 2],
        b'W' => [5, 5, 7, 7, 5],
        b'X' => [5, 5, 2, 5, 5],
        b'Y' => [5, 5, 2, 2, 2],
        b'Z' => [7, 1, 2, 4, 7],
        b':' => [0, 2, 0, 2, 0],
        b'.' => [0, 0, 0, 0, 2],
        b'+' => [0, 2, 7, 2, 0],
        b'-' => [0, 0, 7, 0, 0],
        b'/' => [1, 1, 2, 4, 4],
        b'(' => [2, 4, 4, 4, 2],
        b')' => [2, 1, 1, 1, 2],
        b'[' => [6, 4, 4, 4, 6],
        b']' => [3, 1, 1, 1, 3],
        b'!' => [2, 2, 2, 0, 2],
        b'%' => [5, 1, 2, 4, 5],
        b'#' => [5, 7, 5, 7, 5],
        b',' => [0, 0, 0, 2, 4],
        b'>' => [4, 2, 1, 2, 4],
        b'<' => [1, 2, 4, 2, 1],
        b'=' => [0, 7, 0, 7, 0],
        b'|' => [2, 2, 2, 2, 2],
        b'_' => [0, 0, 0, 0, 7],
        b'*' => [5, 2, 7, 2, 5],
        b'\'' => [2, 2, 0, 0, 0],
        b'?' => [7, 1, 2, 0, 2],
        b'&' => [2, 5, 2, 5, 3],
        b'^' => [2, 5, 0, 0, 0],
        b'`' => [4, 2, 0, 0, 0],
        _ => [0, 0, 0, 0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathfinding::PathGrid;
    use crate::scmap::{ScMap, WaterInfo};
    use crate::skirmish::tests::test_world;

    fn dummy_map() -> MapContext {
        MapContext {
            map_id: "TEST".into(),
            map_name: "Test".into(),
            scmap: ScMap {
                version_major: 2,
                version_minor: 60,
                map_width: 256.0,
                map_height: 256.0,
                preview_dds: vec![],
                heightmap_width: 8,
                heightmap_height: 8,
                height_scale: 0.01,
                heightmap: vec![1000; 81],
                terrain_shader: String::new(),
                background_path: String::new(),
                sky_cubemap_path: String::new(),
                cubemaps: vec![],
                lighting: crate::scmap::ScMapLighting::default(),
                water: WaterInfo {
                    has_water: true,
                    elevation: 10.0,
                    elevation_deep: 8.0,
                    elevation_abyss: 2.0,
                },
                terrain_textures: None,
            },
            grid: PathGrid::new(64, 64, 4 * SCALE),
            naval_grid: PathGrid::new(64, 64, 4 * SCALE),
            amphib_grid: PathGrid::new(64, 64, 4 * SCALE),
            size: (256 * SCALE, 256 * SCALE),
            starts: vec![],
            mass_points: vec![],
            hydro_points: vec![],
            has_water: false,
        }
    }

    #[test]
    fn hud_layout_buttons_and_minimap_hit_test() {
        let w = test_world(50);
        let sess = PlaySession::new(w, 0, false, 1);
        let map = dummy_map();
        let cam = OrbitCamera::for_map(256.0, 256.0, 10.0, 1.6);
        let layout = HudLayout::compute(&sess, 1280, 800);

        // ACU selected: order buttons start with MOVE; the card lists its T1 structures.
        let b0 = &layout.buttons[0];
        assert_eq!(
            layout.hit_test_button(b0.x + 10.0, b0.y + 10.0),
            Some(HudAction::Order(OrderKind::Move))
        );
        assert_eq!(layout.card.model.source, Some(CardSource::Construction));
        assert_eq!(layout.card.slots.len(), layout.card.model.page_items().len());
        let s0 = layout.card.slots[0];
        assert_eq!(layout.hit_test_button(s0[0] + 5.0, s0[1] + 5.0), Some(HudAction::CardItem(0)));
        let (t1, _) = layout.card.tabs[0];
        assert_eq!(layout.hit_test_button(t1[0] + 2.0, t1[1] + 2.0), Some(HudAction::CardTab(1)));
        let (t2, _) = layout.card.tabs[1];
        assert_eq!(layout.hit_test_button(t2[0] + 2.0, t2[1] + 2.0), None, "empty tab is inert");
        assert!(layout.over_hud(layout.cluster[0] + 2.0, layout.cluster[1] + 2.0));
        assert!(!layout.over_hud(640.0, 300.0));
        assert!(layout.queue.is_none() && layout.enh.is_none());
        // Everything fits on screen.
        for b in &layout.buttons {
            assert!(b.x + b.w <= 1280.0 && b.y + b.h <= 800.0);
        }
        assert!(layout.cluster[0] + layout.cluster[2] < layout.minimap[0]);
        let [mx, my, mw, mh] = layout.minimap;
        let mm = layout
            .hit_test_minimap(mx + mw * 0.25, my + mh * 0.75)
            .expect("minimap hit");
        assert!((mm.0 - 0.25).abs() < 1e-4 && (mm.1 - 0.75).abs() < 1e-4);

        let (mv, mi, bg_v, bg_i, ic_v, ic_i, fg_v, fg_i) =
            build_hud_geometry(&sess, &map, &cam, 1280, 800);
        assert_eq!(mv.len(), 4);
        assert_eq!(mi.len(), 6);
        assert!(!bg_v.is_empty() && !bg_i.is_empty());
        // No icon atlas bound in tests: card entries fall back to text labels.
        assert_eq!(ic_v.len() * 6, ic_i.len() * 4);
        assert!(!fg_v.is_empty() && !fg_i.is_empty());
        assert!(fg_v.iter().all(|vx| vx.pos.iter().all(|c| c.is_finite())));
    }

    #[test]
    fn match_menu_pause_load_and_score_hit_test() {
        let w = test_world(51);
        let mut sess = PlaySession::new(w, 0, true, 1);
        for _ in 0..120 {
            sess.step_sim();
        }
        sess.menu = MatchMenu::Pause;
        let pause_lay = MatchMenuLayout::compute(&sess, 1280, 800).unwrap();
        let b0 = &pause_lay.buttons[0];
        assert_eq!(pause_lay.hit(b0.rect[0] + 5.0, b0.rect[1] + 5.0), Some(MatchMenuAction::Resume));

        sess.menu = MatchMenu::Score { series: Series::MassIncome };
        let score_lay = MatchMenuLayout::compute(&sess, 1280, 800).unwrap();
        assert!(score_lay.graph.is_some());
        let map = dummy_map();
        let cam = OrbitCamera::for_map(256.0, 256.0, 10.0, 1.6);
        let (_mv, _mi, _bg_v, _bg_i, _ic_v, _ic_i, fg_v, fg_i) =
            build_hud_geometry(&sess, &map, &cam, 1280, 800);
        assert!(!fg_v.is_empty() && !fg_i.is_empty());
        assert!(fg_v.iter().all(|vx| vx.pos.iter().all(|c| c.is_finite())));
    }
}
