//! Phase 15/20 skirmish lobby: map list with minimap previews and start spots, slots
//! (you / AI / open) with faction and team, seed, START.
//!
//! The pure state lives here ([`LobbyState`], [`LobbyLaunch`], [`build_match`]); the
//! screen is drawn with the shared [`crate::ui_kit`] and hosted by the front-end shell
//! ([`crate::menu`]), which owns the single winit event loop and hands over to the game
//! on START. `farust play --lobby` opens the shell directly on this screen.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::dds_decode::{RgbaImage, decode_dds_rgba8, resize_rgba_bilinear};
use crate::play::{PlaySession, UnitMeshCache};
use crate::scenario;
use crate::scmap::ScMap;
use crate::skirmish::{ArmyId, ArmySetup, Catalog, MapContext, Skirmish};
use crate::ui_kit::{Align, Canvas, Frame, Rect, WState, body, bold, theme, title};

pub const FACTIONS: [&str; 4] = crate::settings::FACTION_KEYS;
const MAX_SLOTS: usize = 8;
pub const ROW_H: f32 = 24.0;
const SLOT_H: f32 = 30.0;

// ---------------------------------------------------------------------------
// Pure lobby state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LobbyMap {
    pub id: String,
    pub name: String,
    pub size: (u32, u32),
    pub armies: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    Human,
    Ai,
    Open,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub kind: SlotKind,
    pub faction: usize,
    pub team: u8,
}

#[derive(Debug, Clone)]
pub struct MapPreview {
    pub image: Option<RgbaImage>,
    /// Start spots normalized to `0..1` map coordinates, in `ARMY_n` order.
    pub starts: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LobbyAction {
    SelectMap(usize),
    CycleKind(usize),
    CycleFaction(usize),
    CycleTeam(usize),
    SeedDown,
    SeedUp,
    SeedRandom,
    Start,
    Back,
}

/// Everything needed to start the match chosen in the lobby.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LobbyLaunch {
    pub map_id: String,
    pub seed: u64,
    /// `(start spot, faction key, team)` per active slot, in army order.
    pub armies: Vec<(usize, &'static str, u8)>,
    /// Army index controlled by the local human (`None` = spectate all-AI).
    pub human: Option<ArmyId>,
}

/// Installed skirmish maps (with `_save.lua`, `.scmap` and at least 2 armies).
pub fn scan_maps(game_dir: &Path) -> Result<Vec<LobbyMap>, String> {
    let mut maps = Vec::new();
    for files in scenario::list_maps(game_dir)? {
        let Ok(meta) = scenario::parse_scenario_file(&files.scenario_path, &files.id) else {
            continue;
        };
        if files.save_path.is_none()
            || files.scmap_path.is_none()
            || meta.armies.len() < 2
            || !meta.scenario_type.eq_ignore_ascii_case("skirmish")
        {
            continue;
        }
        maps.push(LobbyMap {
            id: files.id,
            name: meta.name,
            size: meta.size,
            armies: meta.armies.len(),
        });
    }
    if maps.is_empty() {
        return Err(format!("no skirmish maps found under {}", game_dir.display()));
    }
    Ok(maps)
}

pub struct LobbyState {
    pub maps: Vec<LobbyMap>,
    pub selected: usize,
    pub scroll: usize,
    pub slots: Vec<Slot>,
    pub seed: u64,
    pub status: String,
    /// Shown on the human slot.
    pub player_name: String,
    /// Faction given to the human slot when slots are created.
    pub default_faction: usize,
    /// Map list rows visible at the last layout (scroll clamping).
    pub visible_rows: usize,
    previews: HashMap<String, MapPreview>,
    game_dir: PathBuf,
}

impl LobbyState {
    #[allow(dead_code)]
    pub fn new(game_dir: &Path, initial_map: &str, seed: u64) -> Result<Self, String> {
        Ok(Self::with_maps(game_dir, scan_maps(game_dir)?, initial_map, seed, "You", 0))
    }

    pub fn with_maps(
        game_dir: &Path,
        maps: Vec<LobbyMap>,
        initial_map: &str,
        seed: u64,
        player_name: &str,
        default_faction: usize,
    ) -> Self {
        let selected = maps
            .iter()
            .position(|m| m.id.eq_ignore_ascii_case(initial_map))
            .unwrap_or(0);
        let mut s = Self {
            maps,
            selected: 0,
            scroll: 0,
            slots: Vec::new(),
            seed,
            status: "Click slot fields to cycle - Enter starts - Esc goes back".into(),
            player_name: player_name.to_string(),
            default_faction: default_faction % FACTIONS.len(),
            visible_rows: 20,
            previews: HashMap::new(),
            game_dir: game_dir.to_path_buf(),
        };
        if !s.maps.is_empty() {
            s.select_map(selected);
        }
        s.ensure_visible();
        s
    }

    pub fn map(&self) -> &LobbyMap {
        &self.maps[self.selected]
    }

    /// Selects a map and resizes the slot table to its start spots (keeps existing slots).
    pub fn select_map(&mut self, idx: usize) {
        self.selected = idx.min(self.maps.len() - 1);
        let n = self.maps[self.selected].armies.min(MAX_SLOTS);
        while self.slots.len() < n {
            let i = self.slots.len();
            self.slots.push(Slot {
                kind: match i {
                    0 => SlotKind::Human,
                    1 => SlotKind::Ai,
                    _ => SlotKind::Open,
                },
                faction: if i == 0 { self.default_faction } else { i % FACTIONS.len() },
                team: i as u8,
            });
        }
        self.slots.truncate(n);
        let id = self.maps[self.selected].id.clone();
        if !self.previews.contains_key(&id) {
            let p = load_preview(&self.game_dir, &id);
            self.previews.insert(id, p);
        }
    }

    pub fn preview(&self) -> Option<&MapPreview> {
        self.previews.get(&self.map().id)
    }

    pub fn apply(&mut self, action: LobbyAction) -> Option<LobbyLaunch> {
        match action {
            LobbyAction::SelectMap(i) => {
                self.select_map(i);
                self.ensure_visible();
            }
            LobbyAction::CycleKind(i) => {
                let next = match self.slots[i].kind {
                    SlotKind::Open => SlotKind::Ai,
                    SlotKind::Ai => SlotKind::Human,
                    SlotKind::Human => SlotKind::Open,
                };
                if next == SlotKind::Human {
                    for s in &mut self.slots {
                        if s.kind == SlotKind::Human {
                            s.kind = SlotKind::Ai;
                        }
                    }
                }
                self.slots[i].kind = next;
            }
            LobbyAction::CycleFaction(i) => {
                self.slots[i].faction = (self.slots[i].faction + 1) % FACTIONS.len();
            }
            LobbyAction::CycleTeam(i) => {
                let n = self.slots.len() as u8;
                self.slots[i].team = (self.slots[i].team + 1) % n.max(2);
            }
            LobbyAction::SeedDown => self.seed = self.seed.wrapping_sub(1),
            LobbyAction::SeedUp => self.seed = self.seed.wrapping_add(1),
            LobbyAction::SeedRandom => {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(1);
                self.seed = (t ^ (t >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) % 1_000_000;
            }
            LobbyAction::Start => match self.launch() {
                Ok(l) => return Some(l),
                Err(e) => self.status = format!("Cannot start: {e}"),
            },
            LobbyAction::Back => {}
        }
        None
    }

    pub fn launch(&self) -> Result<LobbyLaunch, String> {
        let mut armies = Vec::new();
        let mut human = None;
        for (spot, s) in self.slots.iter().enumerate() {
            if s.kind == SlotKind::Open {
                continue;
            }
            if s.kind == SlotKind::Human {
                human = Some(armies.len() as ArmyId);
            }
            armies.push((spot, FACTIONS[s.faction], s.team));
        }
        if armies.len() < 2 {
            return Err("need at least 2 armies (human or AI)".into());
        }
        let first_team = armies[0].2;
        if armies.iter().all(|a| a.2 == first_team) {
            return Err("all armies are on one team".into());
        }
        Ok(LobbyLaunch {
            map_id: self.map().id.clone(),
            seed: self.seed,
            armies,
            human,
        })
    }

    pub fn scroll_by(&mut self, delta: i32) {
        let max = self.maps.len().saturating_sub(self.visible_rows);
        self.scroll = (self.scroll as i64 + delta as i64).clamp(0, max as i64) as usize;
    }

    pub fn ensure_visible(&mut self) {
        let rows = self.visible_rows.max(1);
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + rows {
            self.scroll = self.selected + 1 - rows;
        }
    }

    /// Keyboard navigation of the map list.
    pub fn select_relative(&mut self, delta: i64) {
        let last = self.maps.len() as i64 - 1;
        let i = (self.selected as i64 + delta).clamp(0, last.max(0)) as usize;
        self.apply(LobbyAction::SelectMap(i));
    }
}

/// Minimap (scmap preview DDS, heightmap fallback) plus normalized start spots.
pub(crate) fn load_preview(game_dir: &Path, map_id: &str) -> MapPreview {
    let mut out = MapPreview {
        image: None,
        starts: Vec::new(),
    };
    let Ok(files) = scenario::find_map(game_dir, map_id) else {
        return out;
    };
    let Some(scmap) = files.scmap_path.as_deref().and_then(|p| ScMap::parse_file(p).ok()) else {
        return out;
    };
    out.image = decode_dds_rgba8(&scmap.preview_dds)
        .ok()
        .or_else(|| heightmap_image(&scmap))
        .map(|img| resize_rgba_bilinear(&img, 256, 256));
    if let (Ok(meta), Some(save)) = (
        scenario::parse_scenario_file(&files.scenario_path, &files.id),
        files.save_path.as_deref(),
    ) && let Ok(markers) = scenario::parse_save_markers_file(save, &meta.armies)
    {
        let (mw, mh) = (scmap.map_width.max(1.0), scmap.map_height.max(1.0));
        out.starts = markers
            .start_spots
            .iter()
            .map(|m| (m.position[0] / mw, m.position[2] / mh))
            .collect();
    }
    out
}

fn heightmap_image(scmap: &ScMap) -> Option<RgbaImage> {
    let (w, h) = (scmap.heightmap_width.max(1), scmap.heightmap_height.max(1));
    let (lo, hi) = scmap
        .heightmap
        .iter()
        .fold((u16::MAX, 0u16), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    if scmap.heightmap.len() < (w * h) as usize || hi <= lo {
        return None;
    }
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for &v in &scmap.heightmap[..(w * h) as usize] {
        let t = (v - lo) as f32 / (hi - lo) as f32;
        let g = (40.0 + t * 200.0) as u8;
        pixels.extend_from_slice(&[g / 2, g, g / 3, 255]);
    }
    Some(RgbaImage {
        width: w,
        height: h,
        pixels,
    })
}

// ---------------------------------------------------------------------------
// Layout & drawing (shared UI kit)
// ---------------------------------------------------------------------------

pub fn preview_image_name(map_id: &str) -> String {
    format!("preview:{map_id}")
}

struct LobbyLayout {
    list: Rect,
    list_inner: Rect,
    preview: Rect,
    info: Rect,
    slots: Rect,
    widgets: Vec<(Rect, LobbyAction)>,
}

fn layout(state: &LobbyState, w: f32, h: f32) -> LobbyLayout {
    let f = Frame::new(w, h);
    let c = f.content;
    let list = Rect::new(c.x, c.y, (c.w * 0.32).clamp(300.0, 430.0), c.h);
    let list_inner = Rect::new(list.x + 10.0, list.y + 62.0, list.w - 20.0, list.h - 72.0);
    let rows = ((list_inner.h / ROW_H).floor() as usize).max(3);
    let mut widgets = Vec::new();
    for r in 0..rows {
        let idx = state.scroll + r;
        if idx >= state.maps.len() {
            break;
        }
        widgets.push((
            Rect::new(list_inner.x, list_inner.y + r as f32 * ROW_H, list_inner.w, ROW_H - 2.0),
            LobbyAction::SelectMap(idx),
        ));
    }
    let rx = list.right() + 20.0;
    let rw = c.right() - rx;
    let slots_h = 70.0 + state.slots.len() as f32 * SLOT_H;
    let size = (c.h - slots_h - 20.0 - 24.0).min(rw * 0.5).clamp(120.0, 360.0);
    let preview = Rect::new(rx + 12.0, c.y + 12.0, size, size);
    let info = Rect::new(preview.right() + 20.0, preview.y, (c.right() - preview.right() - 32.0).max(100.0), size);
    let slots = Rect::new(rx, (c.bottom() - slots_h).max(preview.bottom() + 24.0), rw, slots_h);
    let cols = slot_columns(slots);
    for i in 0..state.slots.len() {
        let y = slots.y + 52.0 + i as f32 * SLOT_H;
        widgets.push((Rect::new(cols[0], y, cols[1] - cols[0] - 10.0, SLOT_H - 4.0), LobbyAction::CycleKind(i)));
        widgets.push((Rect::new(cols[1], y, cols[2] - cols[1] - 10.0, SLOT_H - 4.0), LobbyAction::CycleFaction(i)));
        widgets.push((Rect::new(cols[2], y, cols[3] - cols[2], SLOT_H - 4.0), LobbyAction::CycleTeam(i)));
    }
    let sy = info.bottom() - 34.0;
    widgets.push((Rect::new(info.x + 120.0, sy, 34.0, 30.0), LobbyAction::SeedDown));
    widgets.push((Rect::new(info.x + 158.0, sy, 34.0, 30.0), LobbyAction::SeedUp));
    widgets.push((Rect::new(info.x + 196.0, sy, 90.0, 30.0), LobbyAction::SeedRandom));
    widgets.push((f.footer_button(0, false, 160.0), LobbyAction::Back));
    widgets.push((f.footer_button(0, true, 220.0), LobbyAction::Start));
    LobbyLayout { list, list_inner, preview, info, slots, widgets }
}

/// Column x positions: player, faction, team, end.
fn slot_columns(slots: Rect) -> [f32; 4] {
    let x0 = slots.x + 70.0;
    let w = slots.right() - 14.0 - x0;
    [x0, x0 + w * 0.42, x0 + w * 0.74, x0 + w]
}

/// Visible map-list rows for a window size (keeps scrolling consistent with drawing).
pub fn visible_rows(state: &LobbyState, w: f32, h: f32) -> usize {
    let lay = layout(state, w, h);
    ((lay.list_inner.h / ROW_H).floor() as usize).max(3)
}

pub fn widgets(state: &LobbyState, w: f32, h: f32) -> Vec<(Rect, LobbyAction)> {
    layout(state, w, h).widgets
}

pub fn draw(c: &mut Canvas, state: &LobbyState) {
    let (w, h) = (c.w, c.h);
    let lay = layout(state, w, h);
    let f = Frame::new(w, h);
    c.chrome(&f, "SKIRMISH", &format!("{} maps installed", state.maps.len()));

    // Map list
    c.titled_panel(lay.list, "SELECT MAP");
    let li = lay.list_inner;
    c.text(li.x + 6.0, li.y - 22.0, bold(13.0), "NAME", theme::DIM);
    c.text(li.right() - 96.0, li.y - 22.0, bold(13.0), "SPOTS", theme::DIM);
    c.text(li.right() - 40.0, li.y - 22.0, bold(13.0), "SIZE", theme::DIM);
    for (r, a) in &lay.widgets {
        let LobbyAction::SelectMap(idx) = *a else {
            continue;
        };
        let m = &state.maps[idx];
        let sel = idx == state.selected;
        let hov = c.is_hover(*r);
        c.list_row(*r, sel, hov);
        let col = if sel { theme::TEXT } else if hov { [0.78, 0.84, 0.88, 1.0] } else { theme::DIM };
        let name = c.fonts.fit(body(15.0), &m.name, r.w - 120.0);
        c.text_in(Rect::new(r.x + 8.0, r.y, r.w, r.h), body(15.0), &name, col, Align::Left);
        c.text_in(Rect::new(r.right() - 96.0, r.y, 30.0, r.h), body(15.0), &m.armies.to_string(), col, Align::Center);
        c.text_in(Rect::new(r.right() - 50.0, r.y, 44.0, r.h), body(15.0), &format!("{}km", m.size.0 * 5 / 256), col, Align::Right);
    }
    if state.maps.len() > state.visible_rows {
        let track = Rect::new(lay.list.right() - 6.0, li.y, 3.0, li.h);
        c.rect(track, [0.15, 0.2, 0.24, 1.0]);
        let frac = state.visible_rows as f32 / state.maps.len() as f32;
        let off = state.scroll as f32 / state.maps.len() as f32;
        c.rect(Rect::new(track.x, track.y + off * track.h, 3.0, (frac * track.h).max(12.0)), theme::ACCENT);
    }

    // Preview + start spots
    let rp = Rect::new(lay.preview.x - 12.0, lay.preview.y - 12.0, lay.slots.right() - lay.preview.x + 12.0, lay.preview.h + 24.0);
    c.panel(rp);
    let p = lay.preview;
    c.rect(p.inset(-2.0), [0.2, 0.28, 0.34, 1.0]);
    let img = preview_image_name(&state.map().id);
    if !c.image(&img, p, [1.0; 4]) {
        c.rect(p, [0.06, 0.08, 0.07, 1.0]);
        c.text_in(p, bold(16.0), "NO PREVIEW", theme::DIM, Align::Center);
    }
    if let Some(pv) = state.preview() {
        for (i, &(sx, sy)) in pv.starts.iter().enumerate() {
            let cx = p.x + sx.clamp(0.0, 1.0) * p.w;
            let cy = p.y + sy.clamp(0.0, 1.0) * p.h;
            let slot = state.slots.get(i);
            let open = slot.is_none_or(|s| s.kind == SlotKind::Open);
            let col = if open { [0.55, 0.55, 0.55, 0.9] } else { theme::SLOT_COLORS[i % 8] };
            c.rect(Rect::new(cx - 9.0, cy - 9.0, 18.0, 18.0), [0.0, 0.0, 0.0, 0.85]);
            c.rect(Rect::new(cx - 7.0, cy - 7.0, 14.0, 14.0), col);
            c.text_in(Rect::new(cx - 9.0, cy - 9.0, 18.0, 18.0), bold(12.0), &format!("{}", i + 1), [0.0, 0.0, 0.0, 1.0], Align::Center);
        }
    }
    let m = state.map();
    let ix = lay.info.x;
    let iy = lay.info.y;
    let name = c.fonts.fit(title(26.0), &m.name.to_uppercase(), lay.info.w);
    c.text(ix, iy, title(26.0), &name, theme::GOLD);
    c.text(ix, iy + 40.0, body(15.0), &m.id, theme::DIM);
    let active = state.slots.iter().filter(|s| s.kind != SlotKind::Open).count();
    for (k, line) in [
        format!("Map size: {} km ({}x{})", m.size.0 * 5 / 256, m.size.0, m.size.1),
        format!("Start spots: {}", m.armies),
        format!("Armies in match: {active}"),
    ]
    .iter()
    .enumerate()
    {
        c.text(ix, iy + 70.0 + k as f32 * 22.0, body(15.0), line, theme::TEXT);
    }
    c.text_in(Rect::new(ix, lay.info.bottom() - 34.0, 110.0, 30.0), bold(14.0), &format!("SEED {}", state.seed), theme::TEXT, Align::Left);

    // Slots
    let inner = c.titled_panel(lay.slots, "PLAYERS");
    let cols = slot_columns(lay.slots);
    let hy = inner.y - 2.0;
    c.text(lay.slots.x + 16.0, hy, bold(13.0), "SPOT", theme::DIM);
    c.text(cols[0], hy, bold(13.0), "PLAYER", theme::DIM);
    c.text(cols[1], hy, bold(13.0), "FACTION", theme::DIM);
    c.text(cols[2], hy, bold(13.0), "TEAM", theme::DIM);
    for i in 0..state.slots.len() {
        let y = lay.slots.y + 52.0 + i as f32 * SLOT_H;
        c.rect(Rect::new(lay.slots.x + 16.0, y + 5.0, 16.0, 16.0), theme::SLOT_COLORS[i % 8]);
        c.text_in(Rect::new(lay.slots.x + 38.0, y, 24.0, SLOT_H - 4.0), bold(15.0), &format!("{}", i + 1), theme::TEXT, Align::Left);
    }
    for (r, a) in &lay.widgets {
        let hover = c.is_hover(*r);
        let st = WState { hover, ..Default::default() };
        match *a {
            LobbyAction::CycleKind(i) => {
                let label = match state.slots[i].kind {
                    SlotKind::Human => state.player_name.clone(),
                    SlotKind::Ai => "AI: Scripted".to_string(),
                    SlotKind::Open => "Open".to_string(),
                };
                c.cycler(*r, &label, WState { disabled: state.slots[i].kind == SlotKind::Open, ..st });
            }
            LobbyAction::CycleFaction(i) => {
                let s = state.slots[i];
                c.cycler(*r, &faction_label(s.faction), WState { disabled: s.kind == SlotKind::Open, ..st });
            }
            LobbyAction::CycleTeam(i) => {
                let s = state.slots[i];
                c.cycler(*r, &format!("Team {}", s.team + 1), WState { disabled: s.kind == SlotKind::Open, ..st });
            }
            LobbyAction::SeedDown => c.button(*r, "-", st),
            LobbyAction::SeedUp => c.button(*r, "+", st),
            LobbyAction::SeedRandom => c.button(*r, "RANDOM", st),
            LobbyAction::Start => c.button(*r, "START GAME", WState { primary: true, ..st }),
            LobbyAction::Back => c.button(*r, "BACK", st),
            LobbyAction::SelectMap(_) => {}
        }
    }
    let err = state.status.starts_with("Cannot");
    c.footer_status(&f, &state.status, if err { theme::BAD } else { theme::DIM });
}

pub fn faction_label(i: usize) -> String {
    match FACTIONS[i % FACTIONS.len()] {
        "uef" => "UEF".into(),
        "aeon" => "Aeon".into(),
        "cybran" => "Cybran".into(),
        _ => "Seraphim".into(),
    }
}

// ---------------------------------------------------------------------------
// Match construction
// ---------------------------------------------------------------------------

/// Builds the world, session and meshes for a lobby launch.
pub fn build_match(game_dir: &Path, launch: &LobbyLaunch) -> Result<(PlaySession, MapContext, UnitMeshCache), String> {
    let catalog = Catalog::load(game_dir)?;
    let meshes = UnitMeshCache::load(Some(game_dir), &catalog);
    let map = MapContext::load(game_dir, &launch.map_id)?;
    let setups = launch
        .armies
        .iter()
        .map(|&(spot, key, team)| {
            let (name, start) = map
                .starts
                .get(spot)
                .cloned()
                .ok_or_else(|| format!("start spot {} out of range", spot + 1))?;
            let faction = catalog
                .faction_index(key)
                .ok_or_else(|| format!("unknown faction '{key}'"))?;
            Ok(ArmySetup {
                name,
                faction,
                start,
                team,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut world = Skirmish::new(
        catalog,
        map.grid.clone(),
        map.size,
        map.mass_points.clone(),
        &setups,
        launch.seed,
    )?;
    world.set_domain_grids(
        map.naval_grid.clone(),
        map.amphib_grid.clone(),
        map.hydro_points.clone(),
    );
    let session = PlaySession::new(world, launch.human.unwrap_or(0), launch.human.is_none(), 1);
    Ok((session, map, meshes))
}

/// `farust play --lobby`: the front-end shell opened on the skirmish screen.
pub fn run_lobby(
    game_dir: &Path,
    initial_map: &str,
    seed: u64,
    width: u32,
    height: u32,
    snapshot: Option<&Path>,
    autostart: bool,
) -> Result<(), String> {
    crate::menu::run(
        game_dir,
        crate::menu::MenuOptions {
            screen: crate::menu::ScreenId::Skirmish,
            size: Some((width, height)),
            snapshot: snapshot.map(Path::to_path_buf),
            initial_map: Some(initial_map.to_string()),
            seed: Some(seed),
            autostart,
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn state() -> LobbyState {
        let maps = vec![
            LobbyMap { id: "A".into(), name: "Alpha".into(), size: (256, 256), armies: 2 },
            LobbyMap { id: "B".into(), name: "Beta".into(), size: (512, 512), armies: 4 },
        ];
        let mut s = LobbyState::with_maps(Path::new("/nonexistent"), maps, "A", 7, "Tester", 0);
        s.previews.insert("A".into(), MapPreview { image: None, starts: vec![] });
        s.previews.insert("B".into(), MapPreview { image: None, starts: vec![(0.2, 0.2)] });
        s
    }

    #[test]
    fn slots_follow_map_and_launch_builds_army_list() {
        let mut s = state();
        s.select_map(0);
        assert_eq!(s.slots.len(), 2);
        s.apply(LobbyAction::SelectMap(1));
        assert_eq!(s.slots.len(), 4);
        // Spot 3 becomes AI, then human (previous human demoted to AI).
        s.apply(LobbyAction::CycleKind(2));
        s.apply(LobbyAction::CycleKind(2));
        assert_eq!(s.slots[2].kind, SlotKind::Human);
        assert_eq!(s.slots[0].kind, SlotKind::Ai);
        s.apply(LobbyAction::CycleFaction(2));
        let l = s.apply(LobbyAction::Start).expect("launch");
        assert_eq!(l.map_id, "B");
        assert_eq!(l.armies, vec![(0, "uef", 0), (1, "aeon", 1), (2, "seraphim", 2)]);
        assert_eq!(l.human, Some(2));
        // All on one team is rejected.
        s.apply(LobbyAction::CycleTeam(0));
        s.apply(LobbyAction::CycleTeam(2));
        s.apply(LobbyAction::CycleTeam(2));
        s.apply(LobbyAction::CycleTeam(2));
        assert!(s.launch().is_err());
    }

    #[test]
    fn default_faction_applies_to_the_human_slot() {
        let maps = vec![LobbyMap { id: "A".into(), name: "Alpha".into(), size: (256, 256), armies: 2 }];
        let s = LobbyState::with_maps(Path::new("/nonexistent"), maps, "A", 1, "Me", 2);
        assert_eq!(FACTIONS[s.slots[0].faction], "cybran");
        assert_eq!(s.slots[0].kind, SlotKind::Human);
    }

    #[test]
    fn layout_hit_tests_widgets() {
        let mut s = state();
        s.select_map(1);
        let ws = widgets(&s, 1280.0, 800.0);
        assert!(ws.iter().any(|(_, a)| *a == LobbyAction::Start));
        assert!(ws.iter().any(|(_, a)| *a == LobbyAction::CycleTeam(3)));
        for (r, a) in &ws {
            let (x, y) = r.center();
            assert_eq!(crate::ui_kit::hit_test(&ws, x, y), Some(*a), "{a:?} at {r:?}");
            assert!(r.right() <= 1280.0 && r.bottom() <= 800.0, "{a:?} off-screen {r:?}");
        }
        let fonts = crate::ui_kit::Fonts::fallback();
        let store = crate::ui_kit::ImageStore::default();
        let mut c = Canvas::new(1280.0, 800.0, &fonts, &store, Some((30.0, 100.0)));
        draw(&mut c, &s);
        assert!(c.batches.iter().map(|b| b.idx.len()).sum::<usize>() > 600);
    }
}
