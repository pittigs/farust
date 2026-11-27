//! Phase 7/18/21/23 in-window 2D HUD, embedded bitmap font, chat, modal menus (pause, load
//! list, score screen with graphs), clickable minimap, and the Phase 23 FA command interface
//! (blueprint-driven command card, orders panel, factory queue / upgrade and enhancement
//! panels, tooltips, cursor-mode label, map pings).

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::camera::OrbitCamera;
use crate::command_card::{CardModel, CardSource, EnhRow, EnhState, OrderKind, QueueEntry};
use crate::play::{PlaySession, army_rgb};
use crate::savegame::SaveEntry;
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{GameResult, IntelVisibility, MapContext, Role};
use crate::stats::Series;
use crate::ui_kit::{Fonts, body as ttf_body, bold as ttf_bold};
use crate::viewer::HudVertex;

static HUD_FONTS: RwLock<Option<Arc<Fonts>>> = RwLock::new(None);

/// Publishes the TrueType font atlas for crisp in-match HUD tooltips and stats cards.
pub fn set_hud_fonts(fonts: Option<Arc<Fonts>>) {
    if let Ok(mut g) = HUD_FONTS.write() {
        *g = fonts;
    }
}

fn active_fonts() -> Option<Arc<Fonts>> {
    HUD_FONTS.read().ok().and_then(|g| g.clone()).filter(|f| f.has_faces())
}

/// Complete 2D HUD geometry for one frame (including the optional TrueType overlay layer).
pub struct HudGeometry {
    pub mm_v: Vec<HudVertex>,
    pub mm_i: Vec<u32>,
    pub bg_v: Vec<HudVertex>,
    pub bg_i: Vec<u32>,
    pub ic_v: Vec<HudVertex>,
    pub ic_i: Vec<u32>,
    pub fg_v: Vec<HudVertex>,
    pub fg_i: Vec<u32>,
    pub ttf_v: Vec<HudVertex>,
    pub ttf_i: Vec<u32>,
}

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
    PrevBookmark,
    NextBookmark,
    Bookmark(usize),
}

/// Phase 15/47 replay timeline bar layout (screen pixels, `[x, y, w, h]`).
#[derive(Debug, Clone)]
pub struct TimelineLayout {
    pub panel: [f32; 4],
    pub bar: [f32; 4],
    pub pause: [f32; 4],
    pub slower: [f32; 4],
    pub faster: [f32; 4],
    pub view: [f32; 4],
    pub prev_bm: Option<[f32; 4]>,
    pub next_bm: Option<[f32; 4]>,
    pub bookmark_pins: Vec<([f32; 4], usize)>,
}

fn rect_hit(r: [f32; 4], px: f32, py: f32) -> bool {
    px >= r[0] && px <= r[0] + r[2] && py >= r[1] && py <= r[1] + r[3]
}

impl TimelineLayout {
    fn from_bounds(
        x0: f32,
        y: f32,
        pw: f32,
        bookmarks: &[crate::vault::Bookmark],
        final_tick: u64,
    ) -> Self {
        let by = y + 6.0;
        let view = [x0 + pw - 8.0 - 104.0, by, 104.0, 24.0];
        let (prev_bm, next_bm) = if !bookmarks.is_empty() && pw >= 380.0 {
            let next = [view[0] - 42.0, by, 38.0, 24.0];
            let prev = [next[0] - 42.0, by, 38.0, 24.0];
            (Some(prev), Some(next))
        } else {
            (None, None)
        };
        let bar = [x0 + 8.0, y + 38.0, pw - 16.0, 18.0];
        let mut bookmark_pins = Vec::with_capacity(bookmarks.len());
        let fin = final_tick.max(1) as f32;
        for (idx, bm) in bookmarks.iter().enumerate() {
            let frac = (bm.tick as f32 / fin).clamp(0.0, 1.0);
            let cx = bar[0] + bar[2] * frac;
            bookmark_pins.push(([cx - 5.0, bar[1] - 5.0, 10.0, bar[3] + 8.0], idx));
        }
        Self {
            panel: [x0, y, pw, 64.0],
            bar,
            pause: [x0 + 8.0, by, 64.0, 24.0],
            slower: [x0 + 78.0, by, 26.0, 24.0],
            faster: [x0 + 108.0, by, 26.0, 24.0],
            view,
            prev_bm,
            next_bm,
            bookmark_pins,
        }
    }

    fn compute(
        w: f32,
        h: f32,
        minimap: [f32; 4],
        palette_right: f32,
        palette_top: f32,
        bookmarks: &[crate::vault::Bookmark],
        final_tick: u64,
    ) -> Self {
        let mut x0 = palette_right + 10.0;
        let mut x1 = minimap[0] - 14.0;
        let mut y = h - 74.0;
        if x1 - x0 < 300.0 {
            x0 = 12.0;
            x1 = (w - 12.0).min(minimap[0] - 14.0).max(x0 + 200.0);
            y = palette_top - 4.0 - 70.0;
        }
        Self::from_bounds(x0, y, x1 - x0, bookmarks, final_tick)
    }

    fn compute_classic(
        w: f32,
        h: f32,
        dock_right: f32,
        dock_x: f32,
        dock_top: f32,
        bookmarks: &[crate::vault::Bookmark],
        final_tick: u64,
    ) -> Self {
        let mut x0 = dock_right + 10.0;
        let mut x1 = w - 12.0;
        let mut y = h - 70.0;
        if x1 - x0 < 300.0 {
            x0 = dock_x;
            x1 = (w - 12.0).max(x0 + 240.0);
            y = (dock_top - 70.0).max(60.0);
        }
        Self::from_bounds(x0, y, (x1 - x0).max(200.0), bookmarks, final_tick)
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
        } else if self.prev_bm.is_some_and(|r| rect_hit(r, px, py)) {
            Some(TimelineHit::PrevBookmark)
        } else if self.next_bm.is_some_and(|r| rect_hit(r, px, py)) {
            Some(TimelineHit::NextBookmark)
        } else if let Some(&(_, idx)) = self.bookmark_pins.iter().rev().find(|(r, _)| rect_hit(*r, px, py)) {
            Some(TimelineHit::Bookmark(idx))
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
    #[allow(dead_code)]
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
    pub classic: bool,
    /// Additional chrome panels (used in `UiTheme::Classic` top-left econ & top-right clock pods).
    pub top_panels: Vec<[f32; 4]>,
    /// Order / system buttons.
    pub buttons: Vec<HudButton>,
    pub minimap: [f32; 4], // [x, y, size, size] in screen pixels
    pub timeline: Option<TimelineLayout>,
    pub menu_open: bool,
    pub card: CardLayout,
    pub orders_panel: [f32; 4],
    pub queue: Option<QueueLayout>,
    pub enh: Option<EnhLayout>,
    /// Bounding box of the whole bottom command cluster.
    pub cluster: [f32; 4],
}

fn faction_palette(session: &PlaySession) -> crate::ui_kit::ClassicPalette {
    let key = session
        .world
        .faction(session.human_army)
        .map(|f| f.key.as_str())
        .unwrap_or("uef");
    crate::ui_kit::ClassicPalette::for_faction_key(key)
}

impl HudLayout {
    pub fn compute(session: &PlaySession, width: u32, height: u32) -> Self {
        Self::compute_with_theme(session, width, height, crate::settings::ui_theme())
    }

    pub fn compute_with_theme(
        session: &PlaySession,
        width: u32,
        height: u32,
        theme: crate::settings::UiTheme,
    ) -> Self {
        let w = width.max(480) as f32;
        let h = height.max(360) as f32;
        let classic = theme == crate::settings::UiTheme::Classic;

        let (minimap, dock_x, orders_x, card_x, top_panels) = if classic {
            let mm_size = (h * 0.24).clamp(136.0, 200.0);
            let full_w = ORD_W + MARGIN + CARD_W;
            let (mm, dx) = if w >= 6.0 + mm_size + 10.0 + full_w + 6.0 {
                ([6.0, h - mm_size - 6.0, mm_size, mm_size], 6.0 + mm_size + 10.0)
            } else {
                ([6.0, 62.0, mm_size.min(136.0), mm_size.min(136.0)], MARGIN)
            };
            let econ = [6.0, 6.0, 412.0f32.min((w - 140.0).max(240.0)), 50.0];
            let cw = 264.0f32.min((w - econ[0] - econ[2] - 18.0).max(120.0));
            let clock = [(w - cw - 6.0).max(econ[0] + econ[2] + 8.0), 6.0, cw, 38.0];
            // Classic FA puts Orders immediately beside the minimap, followed by Build Card.
            (mm, dx, dx, dx + ORD_W + MARGIN, vec![econ, clock])
        } else {
            let mm_size = (h * 0.24).clamp(136.0, 220.0);
            let mm = [w - mm_size - 12.0, h - mm_size - 12.0, mm_size, mm_size];
            (mm, MARGIN, MARGIN + CARD_W + MARGIN, MARGIN, Vec::new())
        };

        let card_y = h - MARGIN - CARD_H;
        let card = CardLayout::compute(session.card_model(), card_x, card_y);
        let orders_panel = [orders_x, card_y, ORD_W, CARD_H];
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
        let queue = QueueLayout::compute(session, dock_x, top, full_w);
        if let Some(q) = &queue {
            top = q.panel[1];
        }
        let enh = EnhLayout::compute(session, dock_x, top, full_w);
        if let Some(e) = &enh {
            top = e.panel[1];
        }
        let cluster = [dock_x, top, full_w, h - MARGIN - top];
        let timeline = session.replay_progress().map(|(_, fin)| {
            let bms = session.replay_bookmarks();
            if classic {
                TimelineLayout::compute_classic(w, h, dock_x + full_w, dock_x, top, bms, fin)
            } else {
                TimelineLayout::compute(w, h, minimap, dock_x + full_w, card_y, bms, fin)
            }
        });

        Self {
            width: w,
            height: h,
            classic,
            top_panels,
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
            .chain(self.top_panels.iter().copied())
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
#[allow(dead_code, clippy::type_complexity)]
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
    build_hud_geometry_with_theme(session, map, camera, width, height, crate::settings::ui_theme())
}

#[allow(dead_code, clippy::type_complexity)]
pub fn build_hud_geometry_with_theme(
    session: &PlaySession,
    map: &MapContext,
    camera: &OrbitCamera,
    width: u32,
    height: u32,
    theme: crate::settings::UiTheme,
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
    let g = build_hud_geometry_ex(session, map, camera, width, height, theme);
    (g.mm_v, g.mm_i, g.bg_v, g.bg_i, g.ic_v, g.ic_i, g.fg_v, g.fg_i)
}

pub fn build_hud_geometry_ex(
    session: &PlaySession,
    map: &MapContext,
    camera: &OrbitCamera,
    width: u32,
    height: u32,
    theme: crate::settings::UiTheme,
) -> HudGeometry {
    let layout = HudLayout::compute_with_theme(session, width, height, theme);
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
    let mut ttf_v: Vec<HudVertex> = Vec::with_capacity(1024);
    let mut ttf_i: Vec<u32> = Vec::with_capacity(1536);
    let ttf_fonts = active_fonts();

    let pal = layout.classic.then(|| faction_palette(session));
    let (brd_default, hdr_bg_default, title_gold, panel_bg_default) = match pal {
        Some(p) => (p.border_outer, p.header_bg, p.title_gold, p.panel_bot),
        None => ([0.30, 0.44, 0.64, 0.95], [0.10, 0.18, 0.25, 0.96], [0.96, 0.84, 0.45, 1.0], [0.05, 0.07, 0.11, 0.88]),
    };

    // 1. Minimap textured backdrop quad
    let [mx, my, mw, mh] = layout.minimap;
    push_quad_uv(&mut mm_v, &mut mm_i, mx, my, mw, mh, [1.0, 1.0, 1.0, 0.96], w, h);

    // 2. Top Economy & Status Bar
    if let Some(p) = pal {
        let econ = layout.top_panels.first().copied().unwrap_or([6.0, 6.0, 412.0, 50.0]);
        let clock = layout.top_panels.get(1).copied().unwrap_or([w - 270.0, 6.0, 264.0, 38.0]);
        push_classic_chrome(&mut v, &mut i, &mut fg_v, &mut fg_i, econ, &p, 4.0, w, h);
        push_classic_chrome(&mut v, &mut i, &mut fg_v, &mut fg_i, clock, &p, 4.0, w, h);
        let div_x = econ[0] + econ[2] * 0.5;
        push_rect(&mut fg_v, &mut fg_i, div_x - 1.0, econ[1] + 7.0, 2.0, econ[3] - 14.0, p.border_inner, w, h);

        if let Some(a) = session.world.army(session.human_army) {
            let m_fill = (a.mass as f32 / a.mass_cap.max(1) as f32).clamp(0.0, 1.0);
            let e_fill = (a.energy as f32 / a.energy_cap.max(1) as f32).clamp(0.0, 1.0);
            let bw = (econ[2] * 0.5 - 20.0).max(60.0);

            let mx0 = econ[0] + 10.0;
            push_rect(&mut v, &mut i, mx0, econ[1] + 10.0, bw, 11.0, [0.08, 0.12, 0.10, 0.96], w, h);
            push_rect(&mut v, &mut i, mx0, econ[1] + 10.0, bw * m_fill, 11.0, [0.24, 0.88, 0.44, 0.96], w, h);
            push_rect_outline(&mut fg_v, &mut fg_i, mx0, econ[1] + 10.0, bw, 11.0, 1.0, p.border_inner, w, h);
            let m_txt = format!(
                "M {}/{} (+{:.1})",
                a.mass / SCALE,
                a.mass_cap / SCALE,
                a.income_mass as f64 * TICK_HZ as f64 / SCALE as f64
            );
            draw_text(&mut fg_v, &mut fg_i, mx0, econ[1] + 27.0, 1.5, &m_txt, [0.55, 0.98, 0.65, 1.0], w, h);

            let ex0 = div_x + 10.0;
            push_rect(&mut v, &mut i, ex0, econ[1] + 10.0, bw, 11.0, [0.14, 0.12, 0.06, 0.96], w, h);
            push_rect(&mut v, &mut i, ex0, econ[1] + 10.0, bw * e_fill, 11.0, [0.96, 0.80, 0.22, 0.96], w, h);
            push_rect_outline(&mut fg_v, &mut fg_i, ex0, econ[1] + 10.0, bw, 11.0, 1.0, p.border_inner, w, h);
            let e_txt = format!(
                "E {}/{} (+{:.0})",
                a.energy / SCALE,
                a.energy_cap / SCALE,
                a.income_energy as f64 * TICK_HZ as f64 / SCALE as f64
            );
            draw_text(&mut fg_v, &mut fg_i, ex0, econ[1] + 27.0, 1.5, &e_txt, [1.0, 0.90, 0.42, 1.0], w, h);
        }

        let secs = session.world.tick() / TICK_HZ;
        let speed_str = if session.paused { "PAUSED".to_string() } else { format!("{}X", session.speed) };
        let n_units = session.world.units().filter(|u| u.army == session.human_army && !u.role.is_structure()).count();
        let top_txt = format!("{} {:02}:{:02} [{}]", p.name, secs / 60, secs % 60, speed_str);
        draw_text(&mut fg_v, &mut fg_i, clock[0] + 10.0, clock[1] + 10.0, 1.55, &top_txt, p.title_gold, w, h);
        let sub_txt = format!("UNITS:{n_units} SEL:{}", session.selected.len());
        draw_text(&mut fg_v, &mut fg_i, clock[0] + 10.0, clock[1] + 24.0, 1.4, &sub_txt, [0.90, 0.95, 1.0, 1.0], w, h);

        if let Some(res) = session.world.result() {
            let banner = match res {
                GameResult::Victory(a) if a == session.human_army => "VICTORY!",
                GameResult::Victory(_) => "DEFEAT",
                GameResult::Draw => "DRAW",
            };
            push_classic_chrome(&mut v, &mut i, &mut fg_v, &mut fg_i, [w * 0.5 - 120.0, 16.0, 240.0, 36.0], &p, 4.0, w, h);
            draw_text(&mut fg_v, &mut fg_i, w * 0.5 - 64.0, 27.0, 2.5, banner, p.title_gold, w, h);
        } else if !session.status_banner.is_empty() {
            let trunc: String = session.status_banner.chars().take(40).collect();
            let bw = (text_w(&trunc, 1.35) + 14.0).min(clock[2]);
            let bx = clock[0] + clock[2] - bw;
            push_rect(&mut v, &mut i, bx, clock[1] + clock[3] + 4.0, bw, 18.0, p.panel_bot, w, h);
            push_rect_outline(&mut fg_v, &mut fg_i, bx, clock[1] + clock[3] + 4.0, bw, 18.0, 1.0, p.border_outer, w, h);
            draw_text(&mut fg_v, &mut fg_i, bx + 7.0, clock[1] + clock[3] + 9.0, 1.35, &trunc, p.accent, w, h);
        }
        if let Some(net) = &session.net {
            let ny = clock[1] + clock[3] + if session.status_banner.is_empty() { 6.0 } else { 26.0 };
            let nline = net.hud_line();
            let nw = text_w(&nline, 1.35) + 12.0;
            draw_text(&mut fg_v, &mut fg_i, clock[0] + clock[2] - nw, ny, 1.35, &nline, [0.60, 0.95, 0.80, 0.95], w, h);
            if let Some(fail) = net.failure_banner() {
                let fail: String = fail.chars().take(60).collect();
                push_rect(&mut v, &mut i, w * 0.5 - 260.0, 64.0, 520.0, 30.0, [0.55, 0.04, 0.04, 0.92], w, h);
                draw_text(&mut fg_v, &mut fg_i, w * 0.5 - 250.0, 72.0, 2.0, &fail, [1.0, 0.92, 0.85, 1.0], w, h);
            }
        }
    } else {
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
    }

    // 3. Phase 23 command cluster (card, orders, queue, enhancements)
    {
        let mut hb = HudBufs { v: &mut v, i: &mut i, ic_v: &mut ic_v, ic_i: &mut ic_i, fg_v: &mut fg_v, fg_i: &mut fg_i, w, h, pal };
        draw_command_interface(session, &layout, &mut hb);
    }

    // 3b. Hover / selection stats card (above minimap in Modern, bottom-right Unit Avatar in Classic)
    if let Some((cid, count)) = session.card_unit() {
        let lines = crate::controls::unit_card_lines(&session.world, cid, count);
        if !lines.is_empty() {
            let portrait = if pal.is_some() { 54.0 } else { 0.0 };
            let (cw, ch, lh) = if let Some(ref fonts) = ttf_fonts {
                let lh = 17.0f32;
                let max_w = lines
                    .iter()
                    .enumerate()
                    .map(|(li, l)| fonts.measure(if li == 0 { ttf_bold(14.0) } else { ttf_body(12.5) }, l))
                    .fold(0.0f32, f32::max);
                let cw = (max_w + 26.0 + portrait).clamp(180.0 + portrait, 390.0);
                let ch = (lines.len() as f32 * lh + 16.0).max(if pal.is_some() { 70.0 } else { 0.0 });
                (cw, ch, lh)
            } else {
                let ts = 1.5;
                let lh = 5.0 * ts + 5.0;
                let max_chars = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as f32;
                let cw = (max_chars * 4.0 * ts + 22.0 + portrait).clamp(160.0 + portrait, 360.0);
                let ch = (lines.len() as f32 * lh + 16.0).max(if pal.is_some() { 68.0 } else { 0.0 });
                (cw, ch, lh)
            };
            let cx = w - cw - (if pal.is_some() { 6.0 } else { 12.0 });
            let cy = if pal.is_some() {
                let floor_y = layout.timeline.as_ref().map(|tl| tl.panel[1] - 8.0).unwrap_or(h - 6.0);
                (floor_y - ch).max(52.0)
            } else {
                (my - ch - 12.0).max(52.0)
            };
            if let Some(p) = pal {
                push_classic_chrome(&mut v, &mut i, &mut fg_v, &mut fg_i, [cx, cy, cw, ch], &p, 4.0, w, h);
                if let Some(u) = session.world.unit(cid) {
                    let pr = [cx + 8.0, cy + 8.0, 44.0, 44.0];
                    push_rect(&mut v, &mut i, pr[0], pr[1], pr[2], pr[3], [0.02, 0.03, 0.05, 0.96], w, h);
                    push_rect_outline(&mut fg_v, &mut fg_i, pr[0], pr[1], pr[2], pr[3], 1.2, p.border_outer, w, h);
                    let mut hb = HudBufs { v: &mut v, i: &mut i, ic_v: &mut ic_v, ic_i: &mut ic_i, fg_v: &mut fg_v, fg_i: &mut fg_i, w, h, pal };
                    if !hb.icon(&u.type_id, [pr[0] + 2.0, pr[1] + 2.0, pr[2] - 4.0, pr[3] - 4.0], 0.98) {
                        let rgb = army_rgb(&session.world, u.army);
                        push_rect(&mut fg_v, &mut fg_i, pr[0] + 6.0, pr[1] + 6.0, pr[2] - 12.0, pr[3] - 12.0, [rgb[0], rgb[1], rgb[2], 0.9], w, h);
                    }
                    let frac = (u.hp as f32 / u.max_hp.max(1) as f32).clamp(0.0, 1.0);
                    push_rect(&mut fg_v, &mut fg_i, pr[0], pr[1] + pr[3] + 3.0, pr[2], 4.0, [0.12, 0.15, 0.18, 0.95], w, h);
                    push_rect(&mut fg_v, &mut fg_i, pr[0], pr[1] + pr[3] + 3.0, pr[2] * frac, 4.0, [0.22, 0.92, 0.35, 0.95], w, h);
                }
                for (li, line) in lines.iter().enumerate() {
                    let col = if li == 0 { p.title_gold } else { [0.90, 0.94, 1.0, 1.0] };
                    if let Some(ref fonts) = ttf_fonts {
                        let st = if li == 0 { ttf_bold(14.0) } else { ttf_body(12.5) };
                        let fit = fonts.fit(st, line, cw - 68.0);
                        fonts.push_hud_text(&mut ttf_v, &mut ttf_i, cx + 60.0, cy + 7.0 + li as f32 * lh, st, &fit, col, w, h);
                    } else {
                        draw_text(&mut fg_v, &mut fg_i, cx + 60.0, cy + 8.0 + li as f32 * lh, 1.5, line, col, w, h);
                    }
                }
            } else {
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
                    if let Some(ref fonts) = ttf_fonts {
                        let st = if li == 0 { ttf_bold(14.0) } else { ttf_body(12.5) };
                        let fit = fonts.fit(st, line, cw - 20.0);
                        fonts.push_hud_text(&mut ttf_v, &mut ttf_i, cx + 12.0, cy + 6.0 + li as f32 * lh, st, &fit, col, w, h);
                    } else {
                        draw_text(&mut fg_v, &mut fg_i, cx + 12.0, cy + 7.0 + li as f32 * lh, 1.5, line, col, w, h);
                    }
                }
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
        if let Some(r) = tl.prev_bm {
            btn(&mut v, &mut i, &mut fg_v, &mut fg_i, r, "<BM", false);
        }
        if let Some(r) = tl.next_bm {
            btn(&mut v, &mut i, &mut fg_v, &mut fg_i, r, "BM>", false);
        }
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
        // Phase 47: event bookmark markers on the timeline bar.
        let bms = session.replay_bookmarks();
        let active_bm = session.ui.replay.as_ref().and_then(|r| r.active_bookmark);
        for (idx, bm) in bms.iter().enumerate() {
            let mx = bx + bw * (bm.tick as f32 / fin.max(1) as f32).clamp(0.0, 1.0);
            let col = bm.kind.color();
            let is_active = active_bm == Some(idx);
            let pip_h = if is_active { bh + 8.0 } else { bh + 4.0 };
            let pip_w = if is_active { 5.0 } else { 3.5 };
            push_rect(&mut fg_v, &mut fg_i, mx - pip_w * 0.5, by - 4.0, pip_w, pip_h, col, w, h);
            push_rect(&mut fg_v, &mut fg_i, mx - 3.5, by - 6.0, 7.0, 4.0, col, w, h);
        }
        // Hover tooltip for bookmark pins.
        if let Some((cx, cy)) = session.ui.cursor_px
            && let Some(&(_, idx)) = tl.bookmark_pins.iter().rev().find(|(r, _)| rect_hit(*r, cx, cy))
            && let Some(bm) = bms.get(idx)
        {
            let tip = format!("[{}] {}  {}", bm.kind.tag(), crate::controls::fmt_ticks(bm.tick), bm.label);
            let tw = (tip.len() as f32 * 8.0 + 14.0).min(pw);
            let tx = (cx - tw * 0.5).clamp(px, (px + pw - tw).max(px));
            let ty = py - 24.0;
            push_rect(&mut v, &mut i, tx, ty, tw, 20.0, [0.05, 0.08, 0.13, 0.95], w, h);
            push_rect_outline(&mut fg_v, &mut fg_i, tx, ty, tw, 20.0, 1.0, bm.kind.color(), w, h);
            draw_text(&mut fg_v, &mut fg_i, tx + 6.0, ty + 6.0, 1.35, &tip, [0.96, 0.98, 1.0, 1.0], w, h);
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
    if let Some(p) = pal {
        let hdr = [mx - 3.0, my - 18.0, mw + 6.0, 16.0];
        push_rect(&mut v, &mut i, hdr[0], hdr[1], hdr[2], hdr[3], p.header_bg, w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, hdr[0], hdr[1], hdr[2], hdr[3], 1.5, p.border_outer, w, h);
        draw_text(&mut fg_v, &mut fg_i, hdr[0] + 6.0, hdr[1] + 5.0, 1.35, "TACTICAL", p.title_gold, w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, mx - 3.0, my - 2.0, mw + 6.0, mh + 5.0, 2.2, p.border_outer, w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, mx, my, mw, mh, 1.0, p.border_inner, w, h);
        push_brackets(&mut fg_v, &mut fg_i, [mx - 4.0, my - 19.0, mw + 8.0, mh + 23.0], 14.0, 2.2, p.bracket, w, h);
    } else {
        push_rect_outline(&mut fg_v, &mut fg_i, mx - 2.0, my - 2.0, mw + 4.0, mh + 4.0, 2.0, [0.30, 0.44, 0.64, 0.95], w, h);
    }
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

    // 5. Phase 21 in-game chat log + input bar (above bottom-left cluster / minimap)
    let chat_base_y = if layout.classic {
        layout.cluster[1].min(my - 22.0) - 10.0
    } else {
        layout.cluster[1] - 12.0
    };
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

    // 6a. Phase 38 observer/caster panel: per-army economy, army size/value and APM.
    if session.spectate && !session.menu.is_open() {
        let rows = observer_rows(&session.world);
        if let Some(ref fonts) = ttf_fonts {
            draw_observer_panel_ttf(fonts, &mut ttf_v, &mut ttf_i, &rows, pal, w, h);
        } else {
            let mut hb = HudBufs { v: &mut v, i: &mut i, ic_v: &mut ic_v, ic_i: &mut ic_i, fg_v: &mut fg_v, fg_i: &mut fg_i, w, h, pal };
            draw_observer_panel(&mut hb, &rows);
        }
    }

    // 6b. Phase 23/37 tooltip and targeting-mode label at the cursor
    if !session.menu.is_open()
        && let Some(cur) = session.ui.cursor_px
    {
        let mut hb = HudBufs { v: &mut v, i: &mut i, ic_v: &mut ic_v, ic_i: &mut ic_i, fg_v: &mut fg_v, fg_i: &mut fg_i, w, h, pal };
        if let Some(hit) = layout.hit_test_button(cur.0, cur.1) {
            let lines = hud_tooltip(session, &layout, hit);
            // Cluster elements show their tooltip above the whole cluster.
            let anchor = if rect_hit(layout.cluster, cur.0, cur.1) { (cur.0, layout.cluster[1]) } else { cur };
            if let Some(ref fonts) = ttf_fonts {
                draw_tooltip_ttf(fonts, &mut ttf_v, &mut ttf_i, anchor, &lines, pal, w, h);
            } else {
                draw_tooltip(&mut hb, anchor, &lines);
            }
        } else if !layout.over_hud(cur.0, cur.1) {
            let label = match (session.current_order_mode(), session.build_mode.as_deref()) {
                (Some(k), _) => Some((k.tooltip().0.to_uppercase(), k.rgb())),
                (None, Some(t)) => {
                    let fb = session.world.catalog().get(t).map(|u| u.name.as_str()).unwrap_or(t);
                    Some((crate::loc::unit_display_name(t, fb).to_uppercase(), [0.45, 1.0, 0.55]))
                }
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
        push_rect(&mut fg_v, &mut fg_i, px, py, pw, ph, panel_bg_default, w, h);
        push_rect(&mut fg_v, &mut fg_i, px + 1.0, py + 1.0, pw - 2.0, 34.0, hdr_bg_default, w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, px, py, pw, ph, 1.8, brd_default, w, h);
        if let Some(p) = pal {
            push_brackets(&mut fg_v, &mut fg_i, [px - 2.0, py - 2.0, pw + 4.0, ph + 4.0], 16.0, 2.2, p.bracket, w, h);
        }
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
        draw_text(&mut fg_v, &mut fg_i, px + 16.0, py + 12.0, 2.0, title_str, title_gold, w, h);

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

    HudGeometry { mm_v, mm_i, bg_v: v, bg_i: i, ic_v, ic_i, fg_v, fg_i, ttf_v, ttf_i }
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
    pal: Option<crate::ui_kit::ClassicPalette>,
}

fn push_brackets(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    r: [f32; 4],
    len: f32,
    t: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    let l = len.min(r[2] * 0.5).min(r[3] * 0.5);
    for (cx, cy, sx, sy) in [
        (r[0], r[1], 1.0, 1.0),
        (r[0] + r[2], r[1], -1.0, 1.0),
        (r[0] + r[2], r[1] + r[3], -1.0, -1.0),
        (r[0], r[1] + r[3], 1.0, -1.0),
    ] {
        let hx = if sx > 0.0 { cx } else { cx - l };
        let vy = if sy > 0.0 { cy } else { cy - l };
        let hy = if sy > 0.0 { cy } else { cy - t };
        let vx = if sx > 0.0 { cx } else { cx - t };
        push_rect(verts, indices, hx, hy, l, t, color, w, h);
        push_rect(verts, indices, vx, vy, t, l, color, w, h);
    }
}

fn push_classic_chrome(
    v: &mut Vec<HudVertex>,
    i: &mut Vec<u32>,
    fg_v: &mut Vec<HudVertex>,
    fg_i: &mut Vec<u32>,
    r: [f32; 4],
    pal: &crate::ui_kit::ClassicPalette,
    header_h: f32,
    w: f32,
    h: f32,
) {
    push_rect(v, i, r[0], r[1], r[2], r[3], pal.panel_bot, w, h);
    if header_h > 0.0 && r[3] > header_h + 4.0 {
        push_rect(v, i, r[0] + 2.0, r[1] + 2.0, (r[2] - 4.0).max(0.0), header_h, pal.header_bg, w, h);
    }
    push_rect_outline(fg_v, fg_i, r[0], r[1], r[2], r[3], 1.8, pal.border_outer, w, h);
    push_rect_outline(fg_v, fg_i, r[0] + 3.0, r[1] + 3.0, (r[2] - 6.0).max(0.0), (r[3] - 6.0).max(0.0), 1.0, pal.border_inner, w, h);
    push_brackets(fg_v, fg_i, [r[0] - 1.0, r[1] - 1.0, r[2] + 2.0, r[3] + 2.0], 11.0, 2.0, pal.bracket, w, h);
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
    fn panel(&mut self, r: [f32; 4]) {
        if let Some(p) = self.pal {
            push_classic_chrome(self.v, self.i, self.fg_v, self.fg_i, r, &p, 4.0, self.w, self.h);
        } else {
            self.bg(r, [0.04, 0.06, 0.10, 0.86]);
            self.outline(r, 1.2, [0.30, 0.44, 0.64, 0.95]);
        }
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
        let (bg, brd, col) = match self.pal {
            Some(p) => (
                if !enabled { [0.06, 0.07, 0.09, 0.86] } else if active { p.btn_active } else { p.btn },
                if active { p.bracket } else { p.border_outer },
                if enabled { if active { p.title_gold } else { [0.94, 0.97, 1.0, 1.0] } } else { [0.45, 0.50, 0.56, 1.0] },
            ),
            None => (
                if !enabled { [0.07, 0.09, 0.11, 0.85] } else if active { [0.18, 0.48, 0.78, 0.95] } else { [0.11, 0.15, 0.22, 0.92] },
                if active { [0.45, 0.92, 1.0, 1.0] } else { [0.26, 0.34, 0.46, 0.95] },
                if enabled { [0.94, 0.97, 1.0, 1.0] } else { [0.45, 0.50, 0.56, 1.0] },
            ),
        };
        self.bg(r, bg);
        self.outline(r, 1.2, brd);
        let tw = text_w(label, 1.35);
        self.text(r[0] + ((r[2] - tw) * 0.5).max(3.0), r[1] + (r[3] - 6.75) * 0.5, 1.35, label, col);
    }
}

fn text_w(s: &str, scale: f32) -> f32 {
    s.chars().count() as f32 * 4.0 * scale
}

fn trunc(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn fmt_cost(v: i64) -> String {
    let whole = v / SCALE;
    if whole >= 10_000 { format!("{}K", whole / 1000) } else { whole.to_string() }
}

/// Phase 23 bottom command cluster: command card, orders, queue / upgrade, enhancements.
fn draw_command_interface(session: &PlaySession, layout: &HudLayout, b: &mut HudBufs) {
    let keymap = crate::keymap::active();
    let card_keys = crate::settings::card_hotkeys_enabled();
    let (slot_bg, slot_hover, slot_active, slot_brd, slot_brd_active, title_col) = match b.pal {
        Some(p) => (p.btn, p.btn_hover, p.btn_active, p.border_outer, p.bracket, p.title_gold),
        None => (
            [0.09, 0.12, 0.18, 0.95],
            [0.16, 0.24, 0.34, 0.95],
            [0.18, 0.48, 0.78, 0.95],
            [0.24, 0.32, 0.44, 0.95],
            [0.45, 0.92, 1.0, 1.0],
            [1.0, 0.92, 0.55, 1.0],
        ),
    };

    // --- Command card -------------------------------------------------------
    let c = &layout.card;
    b.panel(c.panel);
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
                let bg = if placing { slot_active } else if hover { slot_hover } else { slot_bg };
                b.bg(*r, bg);
                let brd = if placing { slot_brd_active } else { slot_brd };
                b.outline(*r, 1.2, brd);
                if !b.icon(&item.unit_type, [r[0] + 3.0, r[1] + 3.0, r[2] - 6.0, r[3] - 6.0], 0.98) {
                    b.text(r[0] + 4.0, r[1] + 16.0, 1.3, &trunc(&item.unit_type, 8), [0.92, 0.95, 1.0, 1.0]);
                    let fb = session.world.catalog().get(&item.unit_type).map(|t| t.name.trim()).unwrap_or("");
                    let name = crate::loc::unit_display_name(&item.unit_type, fb);
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
    b.panel(layout.orders_panel);
    for btn in &layout.buttons {
        let r = [btn.x, btn.y, btn.w, btn.h];
        let HudAction::Order(kind) = btn.action else {
            continue;
        };
        let hover = session.ui.hud_hover == Some(btn.action);
        let bg = if btn.active { slot_active } else if hover { slot_hover } else { slot_bg };
        b.bg(r, bg);
        let rgb = kind.rgb();
        b.fg([r[0], r[1], 3.0, r[3]], [rgb[0], rgb[1], rgb[2], 0.95]);
        b.outline(r, 1.2, if btn.active { slot_brd_active } else { slot_brd });
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
            format!("{}:{n}", trunc(&kind.short_label(), 6))
        } else {
            kind.short_label()
        };
        b.text(r[0] + 6.0, r[1] + 4.0, 1.35, &label, [0.94, 0.97, 1.0, 1.0]);
        if let Some(k) = kind.key_action().and_then(|a| keymap.label(a)) {
            b.text(r[0] + 6.0, r[1] + 15.5, 1.1, &k, [0.62, 0.72, 0.84, 1.0]);
        }
    }

    // --- Factory queue / upgrade -------------------------------------------
    if let Some(q) = &layout.queue {
        b.panel(q.panel);
        let title = match session.queue_panel_unit().and_then(|id| session.world.unit(id)) {
            Some(u) => {
                let fb = session.world.catalog().get(&u.type_id).map(|t| t.name.trim()).unwrap_or(&u.type_id);
                format!("QUEUE  {}", trunc(&crate::loc::unit_display_name(&u.type_id, fb), 28))
            }
            None => "UPGRADE".to_string(),
        };
        b.text(q.panel[0] + 8.0, q.panel[1] + 7.0, 1.5, &title, title_col);
        if q.entries.is_empty() && session.queue_panel_unit().is_some() {
            b.text(q.panel[0] + 10.0, q.panel[1] + 40.0, 1.35, "EMPTY - CLICK A UNIT ON THE CARD", [0.50, 0.58, 0.66, 1.0]);
        }
        for (k, (r, e)) in q.entries.iter().enumerate() {
            let hover = session.ui.hud_hover == Some(HudAction::QueueItem(k as u16));
            b.bg(*r, if hover { [0.30, 0.12, 0.12, 0.95] } else { slot_bg });
            b.outline(*r, 1.0, if e.progress.is_some() { slot_brd_active } else { slot_brd });
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
            b.bg(r, slot_bg);
            b.bg([r[0], r[1], r[2] * *p as f32 / 1000.0, r[3]], [0.20, 0.55, 0.30, 0.95]);
            b.outline(r, 1.0, slot_brd_active);
            b.text(r[0] + 5.0, r[1] + 8.0, 1.3, &format!("UPGRADING {}%", p / 10), [1.0, 1.0, 1.0, 1.0]);
        }
    }

    // --- Enhancements -------------------------------------------------------
    if let Some(e) = &layout.enh {
        b.panel(e.panel);
        let enh_uid = session.world.unit(e.unit).map(|u| u.type_id.clone()).unwrap_or_default();
        let name = session
            .world
            .catalog()
            .get(&enh_uid)
            .map(|t| crate::loc::unit_display_name(&t.id, &t.name))
            .unwrap_or_default();
        b.text(e.panel[0] + 8.0, e.panel[1] + 7.0, 1.5, &format!("ENHANCEMENTS  {}", trunc(&name, 30)), title_col);
        for (r, slot) in &e.headers {
            b.text(r[0] + 2.0, r[1] + 2.0, 1.3, &crate::command_card::localized_slot_label(slot).to_uppercase(), [0.55, 0.72, 0.90, 1.0]);
        }
        for (r, row, idx) in &e.rows {
            let hover = session.ui.hud_hover == Some(HudAction::EnhanceRow(*idx as u16));
            let (bg, txt) = match row.state {
                EnhState::Installed => ([0.10, 0.30, 0.16, 0.95], [0.70, 1.0, 0.72, 1.0]),
                EnhState::Available => (
                    if hover { slot_hover } else { slot_bg },
                    [0.94, 0.97, 1.0, 1.0],
                ),
                EnhState::Locked => ([0.07, 0.08, 0.10, 0.90], [0.45, 0.50, 0.56, 1.0]),
                EnhState::InProgress(_) => (slot_bg, [1.0, 0.95, 0.70, 1.0]),
            };
            b.bg(*r, bg);
            if let EnhState::InProgress(p) = row.state {
                b.bg([r[0], r[1], r[2] * p as f32 / 1000.0, r[3]], [0.22, 0.50, 0.28, 0.95]);
            }
            b.outline(*r, 1.0, if hover { slot_brd_active } else { slot_brd });
            let cost = match row.state {
                EnhState::Installed => "OK".to_string(),
                EnhState::InProgress(p) => format!("{}%", p / 10),
                _ => format!("M{}", fmt_cost(row.cost_mass)),
            };
            let cw = text_w(&cost, 1.3);
            let max_chars = (((r[2] - cw - 14.0) / (4.0 * 1.3)).floor().max(3.0)) as usize;
            let enh_lbl = crate::loc::enhancement_name(&enh_uid, &row.name, &row.display);
            b.text(r[0] + 5.0, r[1] + 8.0, 1.3, &trunc(&enh_lbl.to_uppercase(), max_chars), txt);
            b.text(r[0] + r[2] - cw - 5.0, r[1] + 8.0, 1.3, &cost, txt);
        }
    }
}

/// Tooltip lines for the HUD element `hit` (localized via [`crate::loc`]).
pub fn hud_tooltip(session: &PlaySession, layout: &HudLayout, hit: HudAction) -> Vec<String> {
    let cat = session.world.catalog();
    let keymap = crate::keymap::active();
    match hit {
        HudAction::CardItem(i) => {
            let Some(item) = layout.card.model.page_items().get(i as usize) else {
                return Vec::new();
            };
            let mut v = crate::command_card::item_tooltip(cat, &item.unit_type, session.card_build_rate());
            v.push(match layout.card.model.source {
                Some(CardSource::Production) => crate::loc::tr("tt.hint_prod", "LMB: Queue  Shift: x5  RMB: Remove"),
                _ => crate::loc::tr("tt.hint_build", "LMB: Place  Drag: Line  Shift: Queue"),
            });
            v
        }
        HudAction::Order(k) => {
            let (title, help) = k.tooltip();
            let mut v = vec![title];
            if let Some(key) = k.key_action().map(|a| keymap.get(a)).filter(|c| !c.is_empty()) {
                let kl = crate::loc::tr("tt.key", "Key");
                v.push(format!("{kl}: {}", crate::keymap::chords_to_string(key)));
            }
            if !help.is_empty() {
                for line in crate::loc::word_wrap(&help, 46).into_iter().take(3) {
                    v.push(line);
                }
            }
            if k.is_targeted() {
                v.push(crate::loc::tr("tt.click_map", "Click on the map to target (Shift to queue)"));
            }
            v
        }
        HudAction::QueueItem(i) => match layout.queue.as_ref().and_then(|q| q.entries.get(i as usize)) {
            Some((_, e)) => {
                let fb = cat.get(&e.unit_type).map(|t| t.name.trim()).unwrap_or(&e.unit_type);
                let name = crate::loc::unit_display_name(&e.unit_type, fb);
                vec![
                    format!("{name} x{}", e.count),
                    crate::loc::tr("tt.hint_queue", "Click: Cancel 1  Shift: Cancel 5"),
                ]
            }
            None => Vec::new(),
        },
        HudAction::Upgrade => match layout.queue.as_ref().and_then(|q| q.upgrade.as_ref()) {
            Some((_, t)) => {
                let mut v = crate::command_card::item_tooltip(cat, t, session.card_build_rate());
                v.insert(0, crate::loc::tr("tt.upgrade_to", "Upgrade selected to"));
                v
            }
            None => Vec::new(),
        },
        HudAction::EnhanceRow(i) => match session.enhancement_panel().and_then(|(uid, rows)| {
            let type_id = session.world.unit(uid).map(|u| u.type_id.clone()).unwrap_or_default();
            rows.get(i as usize).cloned().map(|r| (type_id, r))
        }) {
            Some((type_id, r)) => {
                let rate = session.card_build_rate().max(1);
                let enh_title = crate::loc::enhancement_name(&type_id, &r.name, &r.display);
                let slot_lbl = crate::command_card::localized_slot_label(&r.slot);
                let l_mass = crate::loc::tr("tt.mass", "Mass");
                let l_energy = crate::loc::tr("tt.energy", "Energy");
                let l_time = crate::loc::tr("tt.time", "Time");
                let mut v = vec![
                    format!("{enh_title} ({slot_lbl})"),
                    format!(
                        "{l_mass} {}  {l_energy} {}  {l_time} {:.0}s",
                        r.cost_mass / SCALE,
                        r.cost_energy / SCALE,
                        r.build_work as f64 / rate as f64 / TICK_HZ as f64
                    ),
                ];
                if let Some(desc) = crate::loc::enhancement_description(&type_id, &r.name) {
                    for line in crate::loc::word_wrap(&desc, 48).into_iter().take(4) {
                        v.push(line);
                    }
                }
                v.push(match (r.state, &r.prerequisite) {
                    (EnhState::Installed, _) => crate::loc::tr("tt.installed", "Installed"),
                    (EnhState::InProgress(p), _) => format!("{} {}%", crate::loc::tr("tt.installing", "Installing"), p / 10),
                    (EnhState::Locked, Some(p)) => {
                        let pre_name = crate::loc::enhancement_name(&type_id, p, p);
                        format!("{} {pre_name}", crate::loc::tr("tt.requires", "Requires"))
                    }
                    (EnhState::Locked, None) => crate::loc::tr("tt.slot_occupied", "Slot occupied"),
                    (EnhState::Available, _) => crate::loc::tr("tt.click_install", "Click to install"),
                });
                v
            }
            None => Vec::new(),
        },
        HudAction::CardTab(t) => {
            let entries = crate::loc::tr("tt.entries", "entries");
            let next_tab = crate::loc::tr("tt.next_tab", "Next tab");
            vec![
                format!(
                    "Tech {} ({} {entries})",
                    crate::command_card::tab_label(t),
                    layout.card.model.tab_counts[(t.clamp(1, 4) - 1) as usize]
                ),
                format!("{next_tab}: {}", keymap.label(crate::keymap::Action::CardNextTab).unwrap_or_default()),
            ]
        }
        HudAction::CardGroup(g) => {
            if g == 255 {
                let mut v = vec![crate::loc::tr("tt.filter_all", "All Categories")];
                for line in crate::loc::word_wrap(&crate::loc::tr("tt.filter_all_desc", "Show every buildable unit and structure in this tech level."), 44) {
                    v.push(line);
                }
                v
            } else if let Some(&grp) = layout.card.model.groups.get(g as usize) {
                let (title, desc) = grp.tooltip();
                let mut v = vec![title];
                for line in crate::loc::word_wrap(&desc, 44) {
                    v.push(line);
                }
                v
            } else {
                Vec::new()
            }
        }
        HudAction::CardPage => vec![
            format!("{} {}/{}", crate::loc::tr("tt.page", "Page"), layout.card.model.page + 1, layout.card.model.pages),
            crate::loc::tr("tt.page_hint", "Click to cycle command card page"),
        ],
        _ => Vec::new(),
    }
}

fn push_ttf_rect(
    verts: &mut Vec<HudVertex>,
    idx: &mut Vec<u32>,
    uv: [f32; 2],
    r: [f32; 4],
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    let base = verts.len() as u32;
    verts.extend_from_slice(&[
        HudVertex { pos: to_ndc(r[0], r[1], w, h), uv, color },
        HudVertex { pos: to_ndc(r[0] + r[2], r[1], w, h), uv, color },
        HudVertex { pos: to_ndc(r[0] + r[2], r[1] + r[3], w, h), uv, color },
        HudVertex { pos: to_ndc(r[0], r[1] + r[3], w, h), uv, color },
    ]);
    idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[allow(clippy::too_many_arguments)]
fn draw_tooltip_ttf(
    fonts: &Fonts,
    verts: &mut Vec<HudVertex>,
    idx: &mut Vec<u32>,
    cursor: (f32, f32),
    lines: &[String],
    pal: Option<crate::ui_kit::ClassicPalette>,
    w: f32,
    h: f32,
) {
    if lines.is_empty() {
        return;
    }
    let title_st = ttf_bold(14.5);
    let body_st = ttf_body(13.0);
    let lh = 17.5f32;
    let max_line_w = lines
        .iter()
        .enumerate()
        .map(|(k, l)| fonts.measure(if k == 0 { title_st } else { body_st }, l))
        .fold(0.0f32, f32::max);
    let tw = (max_line_w + 22.0).clamp(150.0, (w - 16.0).min(420.0));
    let th = lines.len() as f32 * lh + 14.0;
    let x = (cursor.0 + 16.0).min(w - tw - 4.0).max(4.0);
    let y = (cursor.1 - th - 10.0).max(52.0);
    let wuv = fonts.white_uv();
    let (bg, brd, hdr_col) = match pal {
        Some(p) => (p.panel_bot, p.border_outer, p.title_gold),
        None => ([0.025, 0.035, 0.055, 0.95], [0.45, 0.80, 0.96, 0.95], [1.0, 0.90, 0.50, 1.0]),
    };
    push_ttf_rect(verts, idx, wuv, [x, y, tw, th], bg, w, h);
    let t = 1.4;
    push_ttf_rect(verts, idx, wuv, [x, y, tw, t], brd, w, h);
    push_ttf_rect(verts, idx, wuv, [x, y + th - t, tw, t], brd, w, h);
    push_ttf_rect(verts, idx, wuv, [x, y, t, th], brd, w, h);
    push_ttf_rect(verts, idx, wuv, [x + tw - t, y, t, th], brd, w, h);
    if lines.len() > 1 {
        push_ttf_rect(verts, idx, wuv, [x + 8.0, y + 7.0 + lh - 1.5, tw - 16.0, 1.0], [brd[0], brd[1], brd[2], 0.35], w, h);
    }
    let last_idx = lines.len().saturating_sub(1);
    for (k, l) in lines.iter().enumerate() {
        let (st, col) = if k == 0 {
            (title_st, hdr_col)
        } else if k == last_idx && lines.len() >= 3 {
            (body_st, [0.62, 0.78, 0.90, 1.0])
        } else {
            (body_st, [0.90, 0.94, 0.98, 1.0])
        };
        let fit = fonts.fit(st, l, tw - 18.0);
        fonts.push_hud_text(verts, idx, x + 10.0, y + 6.0 + k as f32 * lh, st, &fit, col, w, h);
    }
}

/// Phase 38: one observer-panel row (already formatted) plus the army colour swatch.
pub(crate) struct ObserverRow {
    pub rgb: [f32; 3],
    pub text: String,
    /// The same values split per column (name left-aligned, numbers right-aligned).
    pub cols: [String; 6],
    pub defeated: bool,
}

pub(crate) const OBSERVER_HEADER: &str = "ARMY           MASS  ENERGY  UNITS  VALUE  APM";
/// Column titles and their right edges (in TTF ems of 7 px) for the proportional-font panel.
const OBSERVER_COLS: [&str; 6] = ["ARMY", "MASS", "ENERGY", "UNITS", "VALUE", "APM"];
const OBSERVER_COL_X: [f32; 6] = [0.0, 138.0, 196.0, 244.0, 294.0, 330.0];

/// Phase 38: rows of the observer panel, one per army, in army order.
pub(crate) fn observer_rows(world: &crate::skirmish::Skirmish) -> Vec<ObserverRow> {
    world
        .armies()
        .iter()
        .map(|a| {
            let (mobile, _structs, value) = world.army_summary(a.id);
            // income_* is milli-resource per tick at 10 Hz -> resource per second.
            let m = a.income_mass as f64 * 10.0 / crate::sim::SCALE as f64;
            let e = a.income_energy as f64 * 10.0 / crate::sim::SCALE as f64;
            let mut name: String = a.name.chars().take(12).collect();
            if a.defeated {
                name = format!("{} X", name.chars().take(10).collect::<String>());
            }
            let apm = world.apm(a.id);
            ObserverRow {
                rgb: crate::play::army_rgb(world, a.id),
                text: format!("{name:<12} {m:>6.1} {e:>7.0} {mobile:>6} {value:>6} {apm:>4}"),
                cols: [name, format!("{m:.1}"), format!("{e:.0}"), mobile.to_string(), value.to_string(), apm.to_string()],
                defeated: a.defeated,
            }
        })
        .collect()
}

fn observer_panel_rect(rows: usize, row_h: f32, pw: f32, w: f32) -> [f32; 4] {
    let ph = (rows + 1) as f32 * row_h + 12.0;
    [(w - pw - 6.0).max(4.0), 52.0, pw, ph]
}

fn draw_observer_panel_ttf(
    fonts: &Fonts,
    verts: &mut Vec<HudVertex>,
    idx: &mut Vec<u32>,
    rows: &[ObserverRow],
    pal: Option<crate::ui_kit::ClassicPalette>,
    w: f32,
    h: f32,
) {
    if rows.is_empty() {
        return;
    }
    let st = ttf_body(12.5);
    let hdr = ttf_bold(12.5);
    let lh = 16.5f32;
    let [x, y, pw, ph] = observer_panel_rect(rows.len(), lh, OBSERVER_COL_X[5] + 34.0, w);
    let wuv = fonts.white_uv();
    let (bg, brd, hdr_col) = match pal {
        Some(p) => (p.panel_bot, p.border_outer, p.title_gold),
        None => ([0.025, 0.035, 0.055, 0.86], [0.45, 0.80, 0.96, 0.9], [1.0, 0.90, 0.50, 1.0]),
    };
    push_ttf_rect(verts, idx, wuv, [x, y, pw, ph], bg, w, h);
    let t = 1.2;
    push_ttf_rect(verts, idx, wuv, [x, y, pw, t], brd, w, h);
    push_ttf_rect(verts, idx, wuv, [x, y + ph - t, pw, t], brd, w, h);
    push_ttf_rect(verts, idx, wuv, [x, y, t, ph], brd, w, h);
    push_ttf_rect(verts, idx, wuv, [x + pw - t, y, t, ph], brd, w, h);
    let x0 = x + 24.0;
    // Column 0 is left-aligned at x0; the numeric columns are right-aligned at their edge.
    let cell = |verts: &mut Vec<HudVertex>, idx: &mut Vec<u32>, c: usize, ty: f32, style: crate::ui_kit::TextStyle, s: &str, col: [f32; 4]| {
        let cx = if c == 0 { x0 } else { x0 + OBSERVER_COL_X[c] - fonts.measure(style, s) };
        let fit = if c == 0 { fonts.fit(style, s, OBSERVER_COL_X[1] - 52.0) } else { s.to_string() };
        fonts.push_hud_text(verts, idx, cx, ty, style, &fit, col, w, h);
    };
    for (c, title) in OBSERVER_COLS.iter().enumerate() {
        cell(verts, idx, c, y + 6.0, hdr, title, hdr_col);
    }
    for (k, r) in rows.iter().enumerate() {
        let ry = y + 6.0 + (k + 1) as f32 * lh;
        let a = if r.defeated { 0.45 } else { 1.0 };
        push_ttf_rect(verts, idx, wuv, [x + 8.0, ry + 3.0, 10.0, 10.0], [r.rgb[0], r.rgb[1], r.rgb[2], a], w, h);
        for (c, v) in r.cols.iter().enumerate() {
            cell(verts, idx, c, ry, st, v, [0.90, 0.94, 0.98, a]);
        }
    }
}

fn draw_observer_panel(b: &mut HudBufs, rows: &[ObserverRow]) {
    if rows.is_empty() {
        return;
    }
    let ts = 1.4;
    let lh = 5.0 * ts + 6.0;
    let tw = rows.iter().map(|r| text_w(&r.text, ts)).fold(text_w(OBSERVER_HEADER, ts), f32::max);
    let [x, y, pw, ph] = observer_panel_rect(rows.len(), lh, tw + 30.0, b.w);
    b.fg([x, y, pw, ph], [0.02, 0.03, 0.05, 0.86]);
    let (brd, hdr_col) = match b.pal {
        Some(p) => (p.border_outer, p.title_gold),
        None => ([0.45, 0.80, 0.96, 0.9], [1.0, 0.92, 0.55, 1.0]),
    };
    b.outline([x, y, pw, ph], 1.2, brd);
    b.text(x + 22.0, y + 7.0, ts, OBSERVER_HEADER, hdr_col);
    for (k, r) in rows.iter().enumerate() {
        let ry = y + 7.0 + (k + 1) as f32 * lh;
        let a = if r.defeated { 0.45 } else { 1.0 };
        b.fg([x + 7.0, ry, 9.0, 8.0], [r.rgb[0], r.rgb[1], r.rgb[2], a]);
        b.text(x + 22.0, ry, ts, &r.text, [0.90, 0.94, 0.98, a]);
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
    let (brd, hdr_col) = match b.pal {
        Some(p) => (p.border_outer, p.title_gold),
        None => ([0.45, 0.80, 0.96, 0.95], [1.0, 0.92, 0.55, 1.0]),
    };
    b.outline([x, y, tw, th], 1.2, brd);
    for (k, l) in lines.iter().enumerate() {
        let col = if k == 0 { hdr_col } else { [0.88, 0.93, 1.0, 1.0] };
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
    for ch in text.chars() {
        let g = glyph_3x5_char(ch);
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

/// Maps Unicode chars (including German umlauts / Latin-1) onto the 3x5 fallback glyph set.
fn glyph_3x5_char(ch: char) -> [u8; 5] {
    match ch {
        'ä' | 'Ä' => [5, 0, 2, 7, 5],
        'ö' | 'Ö' => [5, 0, 7, 5, 7],
        'ü' | 'Ü' => [5, 0, 5, 5, 7],
        'ß' => [6, 5, 6, 5, 6],
        'á' | 'à' | 'â' | 'ã' | 'å' | 'Á' | 'À' | 'Â' | 'Ã' | 'Å' => glyph_3x5(b'A'),
        'é' | 'è' | 'ê' | 'ë' | 'É' | 'È' | 'Ê' | 'Ë' => glyph_3x5(b'E'),
        'í' | 'ì' | 'î' | 'ï' | 'Í' | 'Ì' | 'Î' | 'Ï' => glyph_3x5(b'I'),
        'ó' | 'ò' | 'ô' | 'õ' | 'ø' | 'Ó' | 'Ò' | 'Ô' | 'Õ' | 'Ø' => glyph_3x5(b'O'),
        'ú' | 'ù' | 'û' | 'Ú' | 'Ù' | 'Û' => glyph_3x5(b'U'),
        'ñ' | 'Ñ' => glyph_3x5(b'N'),
        'ç' | 'Ç' => glyph_3x5(b'C'),
        '–' | '—' => glyph_3x5(b'-'),
        '‘' | '’' => glyph_3x5(b'\''),
        c if c.is_ascii() => glyph_3x5((c as u8).to_ascii_uppercase()),
        _ => [0, 0, 0, 0, 0],
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
                unknown1: 0, unknown2: 0, unknown3: 0, unknown4: 0, post_heightmap_byte: 0,
                shader_raw: Vec::new(), background_raw: Vec::new(), sky_raw: Vec::new(), cubemaps_raw: Vec::new(), tail_bytes: Vec::new(),
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
            props: Vec::new(),
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
    fn observer_panel_lists_every_army_only_while_spectating() {
        let w = test_world(52);
        let map = dummy_map();
        let cam = OrbitCamera::for_map(256.0, 256.0, 10.0, 1.6);
        let mut sess = PlaySession::new(w, 0, true, 1);
        let rows = observer_rows(&sess.world);
        assert_eq!(rows.len(), sess.world.armies().len());
        for (r, a) in rows.iter().zip(sess.world.armies()) {
            assert!(r.text.starts_with(&a.name.chars().take(12).collect::<String>()));
            // Columns line up under the header.
            assert_eq!(r.text.len(), OBSERVER_HEADER.len());
        }
        let on = build_hud_geometry(&sess, &map, &cam, 1280, 800).6.len();
        sess.spectate = false;
        let off = build_hud_geometry(&sess, &map, &cam, 1280, 800).6.len();
        assert!(on > off, "observer panel adds foreground geometry ({on} vs {off})");
        let [x, y, pw, ph] = observer_panel_rect(rows.len(), 13.0, 300.0, 1280.0);
        assert!(x >= 0.0 && x + pw <= 1280.0 && y + ph < 800.0);
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

    #[test]
    fn classic_hud_layout_factions_and_theme_switch() {
        use crate::settings::UiTheme;
        let w = test_world(52);
        let sess = PlaySession::new(w, 0, false, 1);
        let map = dummy_map();
        let cam = OrbitCamera::for_map(256.0, 256.0, 10.0, 1.6);

        let modern = HudLayout::compute_with_theme(&sess, 1280, 800, UiTheme::Modern);
        let classic = HudLayout::compute_with_theme(&sess, 1280, 800, UiTheme::Classic);
        assert!(!modern.classic && classic.classic);
        assert!(modern.minimap[0] > 800.0, "modern minimap on right");
        assert!(classic.minimap[0] < 40.0, "classic minimap on left");
        assert!(classic.minimap[0] + classic.minimap[2] < classic.orders_panel[0], "minimap left of orders");
        assert!(classic.orders_panel[0] + classic.orders_panel[2] < classic.card.panel[0], "orders left of build card");
        assert_eq!(classic.top_panels.len(), 2);

        // Hit-testing order buttons, command card slots/tabs, minimap, and top panels works identically.
        let b0 = &classic.buttons[0];
        assert_eq!(classic.hit_test_button(b0.x + 8.0, b0.y + 8.0), Some(HudAction::Order(OrderKind::Move)));
        let s0 = classic.card.slots[0];
        assert_eq!(classic.hit_test_button(s0[0] + 5.0, s0[1] + 5.0), Some(HudAction::CardItem(0)));
        assert!(classic.over_hud(classic.top_panels[0][0] + 8.0, classic.top_panels[0][1] + 8.0));
        let [mx, my, mw, mh] = classic.minimap;
        let mm = classic.hit_test_minimap(mx + mw * 0.5, my + mh * 0.5).unwrap();
        assert!((mm.0 - 0.5).abs() < 1e-4 && (mm.1 - 0.5).abs() < 1e-4);

        // All 4 classic faction palettes + 1024x768 + 800x600 fit cleanly with finite geometry.
        for fkey in crate::settings::FACTION_KEYS {
            let p = crate::ui_kit::ClassicPalette::for_faction_key(fkey);
            assert!(!p.name.is_empty());
        }
        let narrow = HudLayout::compute_with_theme(&sess, 800, 600, UiTheme::Classic);
        for b in &narrow.buttons {
            assert!(b.x + b.w <= 800.0 && b.y + b.h <= 600.0);
        }

        let (mv, mi, bg_v, bg_i, _ic_v, _ic_i, fg_v, fg_i) =
            build_hud_geometry_with_theme(&sess, &map, &cam, 1280, 800, UiTheme::Classic);
        assert_eq!(mv.len(), 4);
        assert_eq!(mi.len(), 6);
        assert!(!bg_v.is_empty() && !bg_i.is_empty());
        assert!(!fg_v.is_empty() && !fg_i.is_empty());
        assert!(fg_v.iter().all(|vx| vx.pos.iter().all(|c| c.is_finite())));
        assert!(bg_v.iter().all(|vx| vx.pos.iter().all(|c| c.is_finite())));

        // Switching back to Modern restores bottom-right minimap immediately.
        let back = HudLayout::compute_with_theme(&sess, 1280, 800, UiTheme::Modern);
        assert_eq!(back.minimap, modern.minimap);
    }

    #[test]
    fn command_card_and_order_tooltips_cover_every_button_and_localize_to_german() {
        let w = test_world(53);
        let sess = PlaySession::new(w, 0, false, 1);
        let layout = HudLayout::compute(&sess, 1280, 800);

        // Every visible command-card slot, tab, category chip, and order button produces a tooltip.
        for i in 0..layout.card.slots.len() {
            let tt = hud_tooltip(&sess, &layout, HudAction::CardItem(i as u16));
            assert!(tt.len() >= 3, "slot {i} tooltip too short: {tt:?}");
        }
        for &(_, t) in &layout.card.tabs {
            let tt = hud_tooltip(&sess, &layout, HudAction::CardTab(t));
            assert!(!tt.is_empty(), "tab {t} missing tooltip");
        }
        for &(_, g) in &layout.card.groups {
            let tt = hud_tooltip(&sess, &layout, HudAction::CardGroup(g));
            assert!(!tt.is_empty(), "group {g} missing tooltip");
        }
        for b in &layout.buttons {
            let tt = hud_tooltip(&sess, &layout, b.action);
            assert!(tt.len() >= 2, "order {:?} missing help text: {tt:?}", b.action);
        }

        // Switching active language to German localizes order tooltips and unit card stats.
        let prev = crate::loc::active_language();
        crate::loc::set_language("de");
        let ovc = hud_tooltip(&sess, &layout, HudAction::Order(OrderKind::Overcharge));
        assert!(ovc[0].contains("Überladen"), "{ovc:?}");
        assert!(ovc.iter().any(|l| l.contains("Überladungsschuss")), "{ovc:?}");
        let grp_all = hud_tooltip(&sess, &layout, HudAction::CardGroup(255));
        assert!(grp_all[0].contains("Alle Kategorien"), "{grp_all:?}");
        let slot0 = hud_tooltip(&sess, &layout, HudAction::CardItem(0));
        assert!(slot0.iter().any(|l| l.contains("Masse")), "{slot0:?}");
        crate::loc::set_language(&prev);
    }
}
