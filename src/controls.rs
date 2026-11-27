//! Phase 13 SupCom-style RTS controls & UI polish, plus the Phase 15 replay timeline.
//!
//! Everything here is UI-side: the sim is only ever touched through ordinary `Command`s
//! issued via `Skirmish::issue` (so recorded replays / lockstep stay deterministic), except
//! for replay seeking, which re-simulates the recorded command stream from a pristine clone
//! of the initial world.
//!
//! - Control groups (`Ctrl+1..9` assign, `Ctrl+Shift+digit` add, `1..9` recall, double-tap centers)
//! - Double-click / `Ctrl+click` selects all on-screen own units of the same type
//! - Idle engineer cycling (`.` key / HUD button)
//! - Range rings (weapon / vision / radar / shield) for selected units, drawn on terrain
//! - RMB-drag line formation move with live preview
//! - LMB-drag line construction of structures
//! - `Ctrl+RMB`-drag area reclaim / repair
//! - Hover / selection stats card text
//! - Replay timeline seek planning and army-perspective switching

use std::time::{Duration, Instant};

use crate::camera::OrbitCamera;
use crate::play::{PlaySession, append_beam, append_ring, sample_terrain_y};
use crate::scene::GpuVertex;
use crate::scmap::ScMap;
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{ArmyId, Command, FORMATION_SPACING, IntelVisibility, Role, Skirmish, UnitId};

/// Max seconds between two taps/clicks to count as a double-tap/double-click.
pub const DOUBLE_TAP_SECS: f64 = 0.35;
/// Upper bound on structures laid by one construction drag.
pub const MAX_DRAG_LINE_SITES: usize = 24;
/// Phase 28: upper bound on wall sections laid by one construction drag.
pub const MAX_DRAG_WALL_SITES: usize = 96;
/// Number of control group slots (`0..=9`).
pub const CONTROL_GROUP_SLOTS: usize = 10;

// ---------------------------------------------------------------------------
// Control groups
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ControlGroups {
    sets: [Vec<UnitId>; CONTROL_GROUP_SLOTS],
    last_recall: Option<(usize, f64)>,
}

impl ControlGroups {
    /// Replace group `slot` with `ids` (deduplicated, order preserved).
    pub fn assign(&mut self, slot: usize, ids: &[UnitId]) {
        let Some(set) = self.sets.get_mut(slot) else {
            return;
        };
        set.clear();
        for &id in ids {
            if !set.contains(&id) {
                set.push(id);
            }
        }
        self.last_recall = None;
    }

    /// Append `ids` to group `slot`.
    pub fn add(&mut self, slot: usize, ids: &[UnitId]) {
        let Some(set) = self.sets.get_mut(slot) else {
            return;
        };
        for &id in ids {
            if !set.contains(&id) {
                set.push(id);
            }
        }
    }

    pub fn get(&self, slot: usize) -> &[UnitId] {
        self.sets.get(slot).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Recall group `slot` at time `now` (seconds). Dead units (per `alive`) are pruned.
    /// Returns `(members, double_tap)`; `double_tap` is true when the same slot was recalled
    /// within [`DOUBLE_TAP_SECS`].
    pub fn recall(&mut self, slot: usize, alive: impl Fn(UnitId) -> bool, now: f64) -> (Vec<UnitId>, bool) {
        let Some(set) = self.sets.get_mut(slot) else {
            return (Vec::new(), false);
        };
        set.retain(|&id| alive(id));
        let double = matches!(self.last_recall, Some((s, t)) if s == slot && now - t <= DOUBLE_TAP_SECS && now >= t);
        self.last_recall = if double { None } else { Some((slot, now)) };
        (set.clone(), double && !set.is_empty())
    }
}

/// Maps a digit key index (`1..=9`, `0`) to a control-group slot.
pub fn digit_slot(digit: u8) -> usize {
    (digit as usize) % CONTROL_GROUP_SLOTS
}

// ---------------------------------------------------------------------------
// Per-session UI state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClickRecord {
    pub unit: UnitId,
    pub at: f64,
}

/// Replay timeline / seeking state.
#[derive(Debug, Clone)]
pub struct ReplayUi {
    /// Pristine copy of the world at tick 0 (seeking backwards re-simulates from here).
    pub origin: Box<Skirmish>,
    pub seek_target: Option<u64>,
    /// `None` = observer (no fog), `Some(a)` = fog of war from army `a`'s perspective.
    pub perspective: Option<ArmyId>,
    pub dragging: bool,
    /// Tick under the cursor while dragging the seek bar (committed on release).
    pub drag_tick: Option<u64>,
}

/// What an RMB drag is doing (decided once the drag threshold is crossed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmbDrag {
    Pan,
    Formation,
    Area,
}

#[derive(Debug, Clone, Default)]
pub struct UiState {
    pub groups: ControlGroups,
    pub hover_unit: Option<UnitId>,
    /// Formation drag in world milli `(start, end)`.
    pub formation_drag: Option<((i64, i64), (i64, i64))>,
    pub formation_preview: Vec<(i64, i64)>,
    /// `Ctrl+RMB` area command drag (center, edge) in world milli.
    pub area_drag: Option<((i64, i64), (i64, i64))>,
    /// Construction drag line (start, end) in world milli.
    pub build_drag: Option<((i64, i64), (i64, i64))>,
    /// Phase 18: LMB clicks unload destination for selected air transports.
    pub unload_mode: bool,
    /// Phase 18: index into `available_enhancements_for_selected`.
    pub enhance_choice: usize,
    pub idle_cycle: usize,
    pub last_click: Option<ClickRecord>,
    pub replay: Option<ReplayUi>,
    pub rmb_drag: Option<RmbDrag>,
    pub rmb_down_world: Option<(i64, i64)>,
    pub lmb_down_world: Option<(i64, i64)>,
    // ----- Phase 23 command interface -----
    /// Requested command-card tech tab (`1..=4`, 0 = first non-empty).
    pub card_tab: u8,
    pub card_group: Option<crate::command_card::CardGroup>,
    pub card_page: usize,
    /// Targeting cursor mode of an order button (attack-move / unload keep their flags).
    pub order_mode: Option<crate::command_card::OrderKind>,
    pub bookmarks: crate::command_card::Bookmarks,
    /// Last cursor position in window pixels (HUD hover tooltips, cursor-mode label).
    pub cursor_px: Option<(f32, f32)>,
    /// HUD element under the cursor (redraw when it changes).
    pub hud_hover: Option<crate::hud::HudAction>,
    epoch: Option<Instant>,
}

impl UiState {
    pub fn for_replay(origin: &Skirmish) -> Self {
        Self {
            replay: Some(ReplayUi {
                origin: Box::new(origin.clone()),
                seek_target: None,
                perspective: None,
                dragging: false,
                drag_tick: None,
            }),
            ..Self::default()
        }
    }

    /// Monotonic UI clock in seconds (since first use).
    pub fn now_secs(&mut self) -> f64 {
        let e = *self.epoch.get_or_insert_with(Instant::now);
        e.elapsed().as_secs_f64()
    }
}

// ---------------------------------------------------------------------------
// Pure layout helpers
// ---------------------------------------------------------------------------

fn round_i64(v: f64) -> i64 {
    v.round() as i64
}

/// Line formation slots between `a` and `b` (world milli) for `n` units.
///
/// Units are spread evenly along the drawn line (at least `spacing` apart); if the line is too
/// short for one row, additional rows are stacked behind it (on the `back_hint` side, i.e. where
/// the units come from). Returns `(slots, facing_dir)` where `facing_dir` is the unit normal the
/// formation faces (perpendicular to the line). Returns `None` for degenerate input.
pub fn line_formation_slots(
    a: (i64, i64),
    b: (i64, i64),
    n: usize,
    spacing: i64,
    back_hint: Option<(i64, i64)>,
) -> Option<(Vec<(i64, i64)>, (f64, f64))> {
    if n == 0 {
        return None;
    }
    let spacing = spacing.max(SCALE) as f64;
    let dx = (b.0 - a.0) as f64;
    let dy = (b.1 - a.1) as f64;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return None;
    }
    let (ux, uy) = (dx / len, dy / len);
    // Normal; flip so that "back" points towards the hint (units' centroid).
    let (mut nx, mut ny) = (-uy, ux);
    let mid = ((a.0 + b.0) as f64 * 0.5, (a.1 + b.1) as f64 * 0.5);
    if let Some(h) = back_hint {
        let to_h = (h.0 as f64 - mid.0, h.1 as f64 - mid.1);
        // back = -normal; want back · to_h >= 0 → normal · to_h <= 0
        if nx * to_h.0 + ny * to_h.1 > 0.0 {
            nx = -nx;
            ny = -ny;
        }
    }
    let cols = ((len / spacing).floor() as usize + 1).clamp(1, n);
    let rows = n.div_ceil(cols);
    let step = if cols > 1 { (len / (cols - 1) as f64).max(spacing) } else { 0.0 };
    let mut out = Vec::with_capacity(n);
    for r in 0..rows {
        let k = (n - r * cols).min(cols);
        let row_len = step * (k.saturating_sub(1)) as f64;
        let start = (len - row_len) * 0.5;
        let back = r as f64 * spacing;
        for c in 0..k {
            let t = start + c as f64 * step;
            let x = a.0 as f64 + ux * t - nx * back;
            let y = a.1 as f64 + uy * t - ny * back;
            out.push((round_i64(x), round_i64(y)));
        }
    }
    Some((out, (nx, ny)))
}

/// Assigns `units` (id, position) to `slots` minimizing crossings along the line axis:
/// both lists are sorted by their projection on `axis` (ties broken by id / index).
pub fn assign_slots(units: &[(UnitId, (i64, i64))], slots: &[(i64, i64)], axis: (f64, f64)) -> Vec<(UnitId, (i64, i64))> {
    let proj = |p: (i64, i64)| p.0 as f64 * axis.0 + p.1 as f64 * axis.1;
    let mut us: Vec<_> = units.to_vec();
    us.sort_by(|x, y| proj(x.1).total_cmp(&proj(y.1)).then(x.0.cmp(&y.0)));
    let mut ss: Vec<(usize, (i64, i64))> = slots.iter().copied().enumerate().collect();
    ss.sort_by(|x, y| proj(x.1).total_cmp(&proj(y.1)).then(x.0.cmp(&y.0)));
    us.into_iter().zip(ss).map(|(u, (_, s))| (u.0, s)).collect()
}

/// Construction sites along a drag line from `a` to `b` for a structure with footprint
/// half-size `radius` (milli). Sites are spaced so neighbouring footprints never overlap the
/// `site_is_free` margin (`2r + 1m`, plus a small slack).
pub fn drag_line_sites(a: (i64, i64), b: (i64, i64), radius: i64) -> Vec<(i64, i64)> {
    let step = (2 * radius.max(0) + SCALE + SCALE / 4) as f64;
    let dx = (b.0 - a.0) as f64;
    let dy = (b.1 - a.1) as f64;
    let len = (dx * dx + dy * dy).sqrt();
    if len < step * 0.5 {
        return vec![a];
    }
    let count = ((len / step).floor() as usize + 1).min(MAX_DRAG_LINE_SITES);
    let (ux, uy) = (dx / len, dy / len);
    (0..count)
        .map(|i| {
            let t = i as f64 * step;
            (round_i64(a.0 as f64 + ux * t), round_i64(a.1 as f64 + uy * t))
        })
        .collect()
}

/// Phase 28: wall sections along a drag line from `a` to `b`. Sites snap to the wall's own
/// footprint grid (`2 * radius`, normally the 1 m build grid) and step one footprint at a time
/// along the dominant axis, so the result is a gap-free 8-connected chain without duplicates.
pub fn drag_wall_sites(a: (i64, i64), b: (i64, i64), radius: i64) -> Vec<(i64, i64)> {
    let cell = (2 * radius).max(1);
    let snap = |v: i64| v.div_euclid(cell);
    let (ax, ay) = (snap(a.0), snap(a.1));
    let (bx, by) = (snap(b.0), snap(b.1));
    let (dx, dy) = (bx - ax, by - ay);
    let n = dx.abs().max(dy.abs()).min(MAX_DRAG_WALL_SITES as i64 - 1);
    let span = dx.abs().max(dy.abs()).max(1);
    let centre = |c: i64| c * cell + cell / 2;
    let step = |d: i64, i: i64| (2 * d * i + span).div_euclid(2 * span);
    (0..=n)
        .map(|i| (centre(ax + step(dx, i)), centre(ay + step(dy, i))))
        .collect()
}

/// Replay seek decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeekPlan {
    /// Re-create the world from tick 0 before stepping.
    pub restart: bool,
    /// Ticks to step after the (optional) restart.
    pub steps: u64,
}

/// Converts a timeline fraction to a target tick, clamped to `[0, final_tick]`.
pub fn seek_target_tick(frac: f32, final_tick: u64) -> u64 {
    let f = if frac.is_finite() { frac.clamp(0.0, 1.0) } else { 0.0 };
    ((f as f64 * final_tick as f64).round() as u64).min(final_tick)
}

/// Seeking forward fast-steps; seeking backwards re-simulates from the start.
pub fn seek_plan(current: u64, target: u64) -> SeekPlan {
    if target >= current {
        SeekPlan { restart: false, steps: target - current }
    } else {
        SeekPlan { restart: true, steps: target }
    }
}

/// Next replay perspective: observer → army 0 → … → army n-1 → observer.
pub fn next_perspective(cur: Option<ArmyId>, armies: &[ArmyId]) -> Option<ArmyId> {
    match cur {
        None => armies.first().copied(),
        Some(a) => {
            let idx = armies.iter().position(|&x| x == a);
            idx.and_then(|i| armies.get(i + 1).copied())
        }
    }
}

/// Projects a world point to pixel coordinates; `None` if behind the camera.
pub fn project_to_screen(camera: &OrbitCamera, p: [f32; 3], width: f32, height: f32) -> Option<(f32, f32)> {
    let m = camera.view_proj_matrix();
    let x = m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0];
    let y = m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1];
    let w = m[0][3] * p[0] + m[1][3] * p[1] + m[2][3] * p[2] + m[3][3];
    if w <= 1e-5 {
        return None;
    }
    let (nx, ny) = (x / w, y / w);
    Some(((nx + 1.0) * 0.5 * width, (1.0 - ny) * 0.5 * height))
}

// ---------------------------------------------------------------------------
// Stats card
// ---------------------------------------------------------------------------

/// Sustained DPS of a unit (whole HP per second).
pub fn unit_dps(u: &crate::skirmish::Unit) -> f64 {
    u.weapons
        .iter()
        .map(|w| (w.damage as f64 / SCALE as f64) * TICK_HZ as f64 / w.reset.max(1) as f64)
        .sum()
}

fn fmt_k(v: i64) -> String {
    let whole = v / SCALE;
    if whole.abs() >= 10_000 {
        format!("{}K", whole / 1000)
    } else {
        whole.to_string()
    }
}

/// Text lines for the hover / selection stats card (`header`, then details).
pub fn unit_card_lines(world: &Skirmish, id: UnitId, selection_count: usize) -> Vec<String> {
    let Some(u) = world.unit(id) else {
        return Vec::new();
    };
    let ut = world.catalog().get(&u.type_id);
    let fallback = ut.map(|t| t.name.trim()).unwrap_or("");
    let role_name = crate::loc::unit_display_name(&u.type_id, fallback);
    let name = match crate::loc::unit_proper_name(&u.type_id) {
        Some(proper) if !proper.eq_ignore_ascii_case(&role_name) => format!("{proper} - {role_name}"),
        _ => role_name,
    };
    let mut lines = Vec::new();
    let head = if selection_count > 1 {
        format!("{} ({}) X{}", name, u.type_id, selection_count)
    } else {
        format!("{} ({})", name, u.type_id)
    };
    lines.push(head.chars().take(40).collect());
    let mut hp = format!("HP {}/{}", fmt_k(u.hp), fmt_k(u.max_hp));
    if u.shield_max_hp > 0 {
        let kind = if u.shield_personal { "PSHD" } else { "SHD" };
        let off = if u.shield_on { "" } else { " OFF" };
        hp.push_str(&format!("  {kind} {}/{}{off}", fmt_k(u.shield_hp), fmt_k(u.shield_max_hp)));
    }
    lines.push(hp);
    if !u.weapons.is_empty() {
        lines.push(format!("DPS {:.0}  RNG {}", unit_dps(u), u.max_range / SCALE));
    }
    lines.push(format!("KILLS {}  VET {}  ARMY {}", u.vet_kills, u.vet_level, u.army));
    if let Some(t) = ut {
        lines.push(format!("COST M{} E{}", fmt_k(t.cost_mass), fmt_k(t.cost_energy)));
        let mut flags = Vec::new();
        if u.radar_stealth {
            flags.push("STEALTH".to_string());
        }
        if u.cloaked {
            flags.push("CLOAK".to_string());
        }
        if u.omni_radius > 0 {
            flags.push(format!("OMNI {}", u.omni_radius / SCALE));
        }
        if let Some(ref tr) = t.ext.transport {
            flags.push(format!("CARGO {}/{}", u.cargo.len(), tr.slots()));
        }
        if !flags.is_empty() {
            lines.push(flags.join(" "));
        }
    }
    if !u.enh.installed.is_empty() {
        let names: Vec<&str> = u.enh.installed.iter().map(|(_, n)| n.as_str()).collect();
        let txt: String = format!("ENH {}", names.join(" ")).chars().take(40).collect();
        lines.push(txt);
    }
    if let Some(crate::skirmish::UnitOrder::Enhance(en)) = u.orders.front() {
        lines.push(format!("ENHANCING {en}").chars().take(40).collect());
    }
    if u.carried_by.is_some() {
        lines.push("ABOARD TRANSPORT".to_string());
    }
    if let Some(job) = &u.build {
        lines.push(format!("BUILD {} {}%", job.unit_type, job.progress_permille() / 10));
    }
    if !u.production.is_empty() {
        let mut q: Vec<(String, usize)> = Vec::new();
        for t in &u.production {
            match q.last_mut() {
                Some((last, c)) if last == t => *c += 1,
                _ => q.push((t.clone(), 1)),
            }
        }
        let s: Vec<String> = q.iter().take(3).map(|(t, c)| format!("{t}X{c}")).collect();
        let more = if q.len() > 3 { " .." } else { "" };
        let rpt = if u.repeat_queue { " [RPT]" } else { "" };
        lines.push(format!("QUEUE {}{}{}", s.join(" "), more, rpt));
    } else if u.repeat_queue {
        lines.push("REPEAT QUEUE ON".to_string());
    }
    lines
}

// ---------------------------------------------------------------------------
// PlaySession controller extensions
// ---------------------------------------------------------------------------

impl PlaySession {
    fn own_alive(&self, id: UnitId) -> bool {
        self.world.unit(id).is_some_and(|u| u.army == self.human_army)
    }

    /// `Ctrl+digit` (assign) / `Ctrl+Shift+digit` (add).
    pub fn assign_control_group(&mut self, slot: usize, add: bool) {
        let ids: Vec<UnitId> = self.selected.iter().copied().filter(|&id| self.own_alive(id)).collect();
        if add {
            self.ui.groups.add(slot, &ids);
        } else {
            self.ui.groups.assign(slot, &ids);
        }
        self.status_banner = format!("Group {slot}: {} units", self.ui.groups.get(slot).len());
    }

    /// Recall group `slot`; returns true on a double-tap (caller centers the camera).
    pub fn recall_control_group(&mut self, slot: usize, additive: bool, now: f64) -> bool {
        let world = &self.world;
        let army = self.human_army;
        let (ids, double) = self
            .ui
            .groups
            .recall(slot, |id| world.unit(id).is_some_and(|u| u.army == army), now);
        if ids.is_empty() {
            self.status_banner = format!("Group {slot} empty (Ctrl+{slot} to assign)");
            return false;
        }
        if !additive {
            self.selected.clear();
        }
        for id in ids {
            if !self.selected.contains(&id) {
                self.selected.push(id);
            }
        }
        self.status_banner = format!("Group {slot} ({} units)", self.selected.len());
        double
    }

    /// Select all own, on-screen units sharing `unit`'s type.
    pub fn select_same_type_visible(&mut self, unit: UnitId, camera: &OrbitCamera, scmap: &ScMap, width: f32, height: f32, additive: bool) {
        let Some(tid) = self.world.unit(unit).filter(|u| u.army == self.human_army).map(|u| u.type_id.clone()) else {
            return;
        };
        let scale = SCALE as f32;
        let hits: Vec<UnitId> = self
            .world
            .units()
            .filter(|u| u.army == self.human_army && u.type_id == tid)
            .filter(|u| {
                let wx = u.pos.x as f32 / scale;
                let wz = u.pos.y as f32 / scale;
                let p = [wx, sample_terrain_y(scmap, wx, wz), wz];
                project_to_screen(camera, p, width, height)
                    .is_some_and(|(sx, sy)| sx >= 0.0 && sy >= 0.0 && sx <= width && sy <= height)
            })
            .map(|u| u.id)
            .collect();
        if !additive {
            self.selected.clear();
        }
        for id in hits {
            if !self.selected.contains(&id) {
                self.selected.push(id);
            }
        }
        self.status_banner = format!("Selected {} x {tid}", self.selected.len());
    }

    /// Handles a plain LMB click (no active mode): double-click / Ctrl+click select-same-type,
    /// otherwise ordinary pick.
    pub fn click_select(
        &mut self,
        world_xz: (i64, i64),
        shift: bool,
        ctrl: bool,
        pick_radius: i64,
        now: f64,
        camera: &OrbitCamera,
        scmap: &ScMap,
        width: f32,
        height: f32,
    ) {
        let hit = self.world.unit_at(world_xz, pick_radius, Some(self.human_army), None);
        let double = match (hit, self.ui.last_click) {
            (Some(h), Some(c)) => c.unit == h && now - c.at <= DOUBLE_TAP_SECS && now >= c.at,
            _ => false,
        };
        if let Some(h) = hit
            && (ctrl || double)
        {
            self.select_same_type_visible(h, camera, scmap, width, height, shift);
            self.ui.last_click = None;
            return;
        }
        self.ui.last_click = hit.map(|unit| ClickRecord { unit, at: now });
        self.select_at(world_xz, shift, pick_radius);
    }

    pub fn idle_engineers(&self) -> Vec<UnitId> {
        self.world
            .units()
            .filter(|u| u.army == self.human_army && u.role == Role::Engineer && u.drone_of.is_none() && u.orders.is_empty() && u.build.is_none())
            .map(|u| u.id)
            .collect()
    }

    /// Selects the next idle engineer; returns its id (caller centers the camera).
    pub fn cycle_idle_engineer(&mut self) -> Option<UnitId> {
        let idle = self.idle_engineers();
        if idle.is_empty() {
            self.status_banner = "No idle engineers".to_string();
            return None;
        }
        let i = self.ui.idle_cycle % idle.len();
        self.ui.idle_cycle = self.ui.idle_cycle.wrapping_add(1);
        let id = idle[i];
        self.selected.clear();
        self.selected.push(id);
        self.build_mode = None;
        self.attack_move_mode = false;
        self.status_banner = format!("Idle engineer {}/{}", i + 1, idle.len());
        Some(id)
    }

    fn own_mobiles_with_pos(&self) -> Vec<(UnitId, (i64, i64))> {
        self.selected
            .iter()
            .filter_map(|id| self.world.unit(*id))
            .filter(|u| u.army == self.human_army && u.speed > 0)
            .map(|u| (u.id, (u.pos.x, u.pos.y)))
            .collect()
    }

    pub fn has_mobile_selection(&self) -> bool {
        !self.own_mobiles_with_pos().is_empty()
    }

    fn formation_spacing(&self, units: &[(UnitId, (i64, i64))]) -> i64 {
        let max_r = units
            .iter()
            .filter_map(|(id, _)| self.world.unit(*id))
            .map(|u| u.radius)
            .max()
            .unwrap_or(0);
        (2 * max_r + SCALE).max(FORMATION_SPACING)
    }

    fn formation_assignments(&self, a: (i64, i64), b: (i64, i64)) -> Option<Vec<(UnitId, (i64, i64))>> {
        let units = self.own_mobiles_with_pos();
        if units.is_empty() {
            return None;
        }
        let n = units.len() as i64;
        let centroid = (
            units.iter().map(|u| u.1.0).sum::<i64>() / n,
            units.iter().map(|u| u.1.1).sum::<i64>() / n,
        );
        let spacing = self.formation_spacing(&units);
        let (slots, _facing) = line_formation_slots(a, b, units.len(), spacing, Some(centroid))?;
        let (dx, dy) = ((b.0 - a.0) as f64, (b.1 - a.1) as f64);
        let len = (dx * dx + dy * dy).sqrt().max(1.0);
        Some(assign_slots(&units, &slots, (dx / len, dy / len)))
    }

    /// Updates the live formation preview while RMB-dragging.
    pub fn update_formation_drag(&mut self, a: (i64, i64), b: (i64, i64)) {
        self.ui.formation_drag = Some((a, b));
        self.ui.formation_preview = self
            .formation_assignments(a, b)
            .map(|v| v.into_iter().map(|(_, p)| p).collect())
            .unwrap_or_default();
    }

    /// Commits the formation: one ordinary `Move` per unit to its slot.
    pub fn finish_formation_drag(&mut self, queue: bool) -> bool {
        let Some((a, b)) = self.ui.formation_drag.take() else {
            return false;
        };
        self.ui.formation_preview.clear();
        let Some(assign) = self.formation_assignments(a, b) else {
            return false;
        };
        let size = self.world.size();
        for (id, (x, y)) in &assign {
            self.world.issue(
                self.human_army,
                Command::Move {
                    units: vec![*id],
                    x: (*x).clamp(0, size.0),
                    y: (*y).clamp(0, size.1),
                    queue,
                },
            );
        }
        self.status_banner = format!("Formation move ({} units)", assign.len());
        true
    }

    /// Area command (Ctrl+RMB drag): selected builders reclaim every wreck and repair every
    /// damaged own unit inside the circle, nearest first.
    pub fn issue_area_command(&mut self, center: (i64, i64), edge: (i64, i64), queue: bool) -> bool {
        let builders: Vec<UnitId> = self
            .selected
            .iter()
            .copied()
            .filter(|id| self.world.unit(*id).is_some_and(|u| u.army == self.human_army && self.world.is_constructor(u)))
            .collect();
        if builders.is_empty() {
            self.status_banner = "Area reclaim/repair needs engineers".to_string();
            return false;
        }
        let r = ((edge.0 - center.0) as f64).hypot((edge.1 - center.1) as f64) as i64;
        let r2 = (r as i128) * (r as i128);
        let d2 = |p: (i64, i64)| {
            let dx = (p.0 - center.0) as i128;
            let dy = (p.1 - center.1) as i128;
            dx * dx + dy * dy
        };
        let mut targets: Vec<(i128, bool, UnitId, (i64, i64))> = Vec::new();
        for w in self.world.wrecks() {
            let p = (w.pos.x, w.pos.y);
            if d2(p) <= r2 {
                targets.push((d2(p), true, w.id, p));
            }
        }
        // Phase 34: map props (trees, rocks) in the area are reclaimable too.
        for q in self.world.props() {
            if d2(q.pos) <= r2 {
                targets.push((d2(q.pos), true, 0, q.pos));
            }
        }
        for u in self.world.units() {
            let p = (u.pos.x, u.pos.y);
            if u.army == self.human_army && u.hp < u.max_hp && d2(p) <= r2 && !builders.contains(&u.id) {
                targets.push((d2(p), false, u.id, p));
            }
        }
        targets.sort();
        if targets.is_empty() {
            self.status_banner = "Nothing to reclaim/repair in area".to_string();
            return false;
        }
        for (i, (_, is_wreck, id, p)) in targets.iter().take(32).enumerate() {
            let q = queue || i > 0;
            let cmd = if *is_wreck {
                Command::Reclaim { units: builders.clone(), x: p.0, y: p.1, queue: q }
            } else {
                Command::Repair { units: builders.clone(), target: *id, queue: q }
            };
            self.world.issue(self.human_army, cmd);
        }
        self.status_banner = format!("Area reclaim/repair: {} targets", targets.len().min(32));
        true
    }

    /// Valid sites for the current construction drag (empty if not applicable).
    pub fn build_drag_sites(&self) -> Vec<((i64, i64), bool)> {
        let (Some(btype), Some((a, b))) = (&self.build_mode, self.ui.build_drag) else {
            return Vec::new();
        };
        let Some(t) = self.world.catalog().get(btype) else {
            return Vec::new();
        };
        if matches!(t.role, Role::Extractor | Role::Hydro) {
            return Vec::new();
        }
        let sites = if t.role == Role::Wall {
            drag_wall_sites(a, b, t.radius)
        } else {
            drag_line_sites(a, b, t.radius)
        };
        sites
            .into_iter()
            .map(|p| (p, self.world.site_is_free(btype, p.0, p.1)))
            .collect()
    }

    /// Commits a construction drag: valid sites are queued on the primary builder; any other
    /// selected builders assist it.
    pub fn finish_build_drag(&mut self, queue: bool) -> bool {
        let sites = self.build_drag_sites();
        self.ui.build_drag = None;
        let Some(btype) = self.build_mode.clone() else {
            return false;
        };
        if sites.is_empty() {
            return false;
        }
        let mut builders: Vec<UnitId> = self
            .selected
            .iter()
            .copied()
            .filter(|id| self.world.unit(*id).is_some_and(|u| u.army == self.human_army && self.world.is_constructor(u)))
            .collect();
        if builders.is_empty()
            && let Some(acu) = self.world.commander_of(self.human_army)
        {
            builders.push(acu.id);
        }
        // Phase 23: the primary builder must have the engineering tier for `btype`.
        let Some(primary) = self.builder_for(&btype).or_else(|| builders.first().copied()) else {
            self.status_banner = "No builder alive".to_string();
            return true;
        };
        builders.retain(|&b| b != primary);
        builders.insert(0, primary);
        let mut placed = 0usize;
        for &(p, ok) in &sites {
            if !ok {
                continue;
            }
            self.world.issue(
                self.human_army,
                Command::Build { builder: primary, unit_type: btype.clone(), x: p.0, y: p.1, queue: queue || placed > 0 },
            );
            placed += 1;
        }
        if placed > 0 && builders.len() > 1 {
            self.world.issue(self.human_army, Command::Assist { units: builders[1..].to_vec(), target: primary, queue });
        }
        self.status_banner = format!("Queued {placed}/{} x {btype}", sites.len());
        if !queue {
            self.build_mode = None;
        }
        true
    }

    // ----- replay timeline -----------------------------------------------------

    pub fn is_replay(&self) -> bool {
        self.replay.is_some()
    }

    /// `(current_tick, final_tick)` for replay sessions.
    pub fn replay_progress(&self) -> Option<(u64, u64)> {
        self.replay.as_ref().map(|(r, _)| (self.world.tick(), r.final_tick))
    }

    pub fn request_seek_tick(&mut self, tick: u64) {
        let Some((_, fin)) = self.replay_progress() else {
            return;
        };
        let target = tick.min(fin);
        let cur = self.world.tick();
        let plan = seek_plan(cur, target);
        if plan.restart && let Some(ref ui) = self.ui.replay {
            self.world = (*ui.origin).clone();
            if let Some((_, ref mut cursor)) = self.replay {
                *cursor = 0;
            }
        }
        if let Some(ref mut ui) = self.ui.replay {
            ui.seek_target = Some(target);
        }
        self.status_banner = format!("Seek {}", fmt_ticks(target));
    }

    /// One replay tick without AI / audio side effects. Returns false at the end.
    fn replay_step_quiet(&mut self) -> bool {
        let Some((ref rep, ref mut cursor)) = self.replay else {
            return false;
        };
        let t = self.world.tick();
        if t >= rep.final_tick || self.world.result().is_some() {
            return false;
        }
        while *cursor < rep.commands.len() && rep.commands[*cursor].tick < t {
            *cursor += 1;
        }
        while *cursor < rep.commands.len() && rep.commands[*cursor].tick == t {
            let rc = &rep.commands[*cursor];
            self.world.issue(rc.army, rc.command.clone());
            *cursor += 1;
        }
        self.world.step();
        true
    }

    /// Advances a pending seek within `budget`. Returns true while/when work was done.
    pub fn pump_replay_seek(&mut self, budget: Duration) -> bool {
        let Some(target) = self.ui.replay.as_ref().and_then(|r| r.seek_target) else {
            return false;
        };
        let t0 = Instant::now();
        let mut n = 0u32;
        while self.world.tick() < target {
            if !self.replay_step_quiet() {
                break;
            }
            n += 1;
            if n % 16 == 0 && t0.elapsed() >= budget {
                break;
            }
        }
        if self.world.tick() >= target || self.world.result().is_some() || self.replay_progress().is_some_and(|(c, f)| c >= f) {
            if let Some(ref mut ui) = self.ui.replay {
                ui.seek_target = None;
            }
            self.status_banner = format!("REPLAY @ {}", fmt_ticks(self.world.tick()));
        }
        let _ = self.world.drain_events();
        self.selected.retain(|id| self.world.unit(*id).is_some());
        true
    }

    pub fn seeking(&self) -> bool {
        self.ui.replay.as_ref().is_some_and(|r| r.seek_target.is_some())
    }

    /// Cycles observer → army 0 → … → observer (fog of war from that army).
    pub fn cycle_replay_perspective(&mut self) {
        let armies: Vec<ArmyId> = self.world.armies().iter().map(|a| a.id).collect();
        let Some(ref mut ui) = self.ui.replay else {
            return;
        };
        ui.perspective = next_perspective(ui.perspective, &armies);
        match ui.perspective {
            Some(a) => {
                self.human_army = a;
                self.fog_of_war = true;
                self.status_banner = format!("Perspective: Army {a} (fog of war)");
            }
            None => {
                self.fog_of_war = false;
                self.status_banner = "Perspective: Observer".to_string();
            }
        }
        self.selected.clear();
    }

    pub fn max_speed(&self) -> u32 {
        if self.is_replay() { 16 } else { 8 }
    }

    // ----- Phase 18: transports, enhancements, factory repeat, shields -------

    pub fn selected_transports(&self) -> Vec<UnitId> {
        self.selected
            .iter()
            .copied()
            .filter(|id| {
                self.world.unit(*id).is_some_and(|u| {
                    u.army == self.human_army
                        && u.carried_by.is_none()
                        && self.world.catalog().get(&u.type_id).is_some_and(|t| t.ext.transport.is_some())
                })
            })
            .collect()
    }

    pub fn selected_factories(&self) -> Vec<UnitId> {
        self.selected
            .iter()
            .copied()
            .filter(|id| {
                self.world
                    .unit(*id)
                    .is_some_and(|u| u.army == self.human_army && self.world.is_producer(u))
            })
            .collect()
    }

    pub fn selected_shield_units(&self) -> Vec<UnitId> {
        self.selected
            .iter()
            .copied()
            .filter(|id| {
                self.world
                    .unit(*id)
                    .is_some_and(|u| u.army == self.human_army && u.shield_max_hp > 0)
            })
            .collect()
    }

    pub fn trigger_unload_mode(&mut self) {
        if self.selected_transports().is_empty() {
            self.ui.unload_mode = false;
            self.status_banner = "Select a transport to unload".to_string();
            return;
        }
        self.build_mode = None;
        self.attack_move_mode = false;
        self.ui.unload_mode = !self.ui.unload_mode;
        self.status_banner = if self.ui.unload_mode {
            "Unload: LMB drop site (Shift queue, RMB/Esc cancel)".to_string()
        } else {
            "Cancelled Unload".to_string()
        };
    }

    pub fn handle_left_click_unload(&mut self, world_xz: (i64, i64), queue: bool) -> bool {
        if !self.ui.unload_mode {
            return false;
        }
        let transports = self.selected_transports();
        if !transports.is_empty() {
            self.world.issue(
                self.human_army,
                Command::Unload {
                    transports,
                    x: world_xz.0,
                    y: world_xz.1,
                    queue,
                },
            );
            self.status_banner = "Transport unload ordered".to_string();
        }
        if !queue {
            self.ui.unload_mode = false;
        }
        true
    }

    /// Phase 30: right-clicking an own air staging platform with aircraft selected docks them.
    pub fn handle_right_click_dock(&mut self, world_xz: (i64, i64), queue: bool, pick_radius: i64) -> bool {
        let Some(staging) = self.world.unit_at(world_xz, pick_radius, Some(self.human_army), None) else {
            return false;
        };
        if !self.world.unit(staging).is_some_and(|u| u.role == Role::AirStaging) {
            return false;
        }
        let aircraft: Vec<UnitId> = self
            .selected
            .iter()
            .copied()
            .filter(|&id| {
                self.world.unit(id).is_some_and(|u| {
                    u.army == self.human_army && u.domain == crate::flowfield::NavDomain::Air && u.speed > 0 && u.carried_by.is_none()
                })
            })
            .collect();
        if aircraft.is_empty() {
            return false;
        }
        let n = aircraft.len();
        self.world.issue(self.human_army, Command::Dock { units: aircraft, staging, queue });
        self.status_banner = format!("Docking {n} aircraft at #{staging}");
        true
    }

    /// Right-clicking a friendly air transport with carryable mobiles selected loads them.
    pub fn handle_right_click_board(&mut self, world_xz: (i64, i64), queue: bool, pick_radius: i64) -> bool {
        let Some(transport) = self.world.unit_at(world_xz, pick_radius, Some(self.human_army), None) else {
            return false;
        };
        if !self
            .world
            .unit(transport)
            .is_some_and(|u| self.world.catalog().get(&u.type_id).is_some_and(|t| t.ext.transport.is_some()))
        {
            return false;
        }
        let passengers: Vec<UnitId> = self
            .selected
            .iter()
            .copied()
            .filter(|&id| {
                id != transport
                    && self.world.unit(id).is_some_and(|u| {
                        u.army == self.human_army
                            && u.speed > 0
                            && u.carried_by.is_none()
                            && self.world.catalog().get(&u.type_id).is_some_and(|t| t.ext.transport_class > 0)
                    })
            })
            .collect();
        if passengers.is_empty() {
            return false;
        }
        let n = passengers.len();
        self.world.issue(
            self.human_army,
            Command::Load {
                units: passengers,
                transport,
                queue,
            },
        );
        self.status_banner = format!("Boarding {n} unit(s) onto #{transport}");
        true
    }

    pub fn trigger_toggle_repeat(&mut self) {
        let mut factories = self.selected_factories();
        if factories.is_empty() {
            factories = self
                .world
                .units()
                .filter(|u| u.army == self.human_army && self.world.is_producer(u))
                .map(|u| u.id)
                .collect();
        }
        if factories.is_empty() {
            self.status_banner = "No factory selected or built".to_string();
            return;
        }
        let any_on = factories
            .iter()
            .any(|id| self.world.unit(*id).is_some_and(|u| u.repeat_queue));
        let enabled = !any_on;
        self.world.issue(
            self.human_army,
            Command::SetRepeat { factories, enabled },
        );
        self.status_banner = if enabled {
            "Factory repeat queue ON".to_string()
        } else {
            "Factory repeat queue OFF".to_string()
        };
    }

    pub fn available_enhancements_for_selected(&self) -> Option<(UnitId, Vec<crate::skirmish::EnhancementSpec>)> {
        for &id in &self.selected {
            let Some(u) = self.world.unit(id) else {
                continue;
            };
            if u.army != self.human_army {
                continue;
            }
            let avail: Vec<_> = self
                .world
                .available_enhancements(id)
                .into_iter()
                .filter(|e| !e.is_removal())
                .cloned()
                .collect();
            if !avail.is_empty() {
                return Some((id, avail));
            }
        }
        None
    }

    #[allow(dead_code)]
    pub fn selected_enhancement_label(&self) -> Option<String> {
        let (_, avail) = self.available_enhancements_for_selected()?;
        let spec = &avail[self.ui.enhance_choice % avail.len()];
        let text = if spec.display.is_empty() { &spec.name } else { &spec.display };
        Some(text.chars().take(10).collect())
    }

    pub fn trigger_enhance_selected(&mut self, cycle_only: bool) {
        let Some((id, avail)) = self.available_enhancements_for_selected() else {
            self.status_banner = "No enhancements available for selection".to_string();
            return;
        };
        if cycle_only {
            self.ui.enhance_choice = (self.ui.enhance_choice + 1) % avail.len();
            let spec = &avail[self.ui.enhance_choice];
            let label = if spec.display.is_empty() { &spec.name } else { &spec.display };
            self.status_banner = format!(
                "Enhancement [{}/{}]: {} (M{} E{})",
                self.ui.enhance_choice + 1,
                avail.len(),
                label,
                spec.cost_mass / SCALE,
                spec.cost_energy / SCALE
            );
            return;
        }
        let spec = &avail[self.ui.enhance_choice % avail.len()];
        self.world.issue(
            self.human_army,
            Command::Enhance {
                unit: id,
                name: spec.name.clone(),
            },
        );
        let label = if spec.display.is_empty() { &spec.name } else { &spec.display };
        self.status_banner = format!("Starting enhancement: {label}");
    }

    pub fn trigger_toggle_shield(&mut self) {
        let units = self.selected_shield_units();
        if units.is_empty() {
            self.status_banner = "No shielded units selected".to_string();
            return;
        }
        let any_on = units
            .iter()
            .any(|id| self.world.unit(*id).is_some_and(|u| u.shield_on));
        let enabled = !any_on;
        self.world
            .issue(self.human_army, Command::ToggleShield { units, enabled });
        self.status_banner = if enabled {
            "Shields toggled ON".to_string()
        } else {
            "Shields toggled OFF".to_string()
        };
    }

    /// Test/snapshot hook: select the human army's ACU + mobile units so selection overlays
    /// (range rings, stats card) are visible in offscreen snapshots.
    pub fn preselect_army_for_snapshot(&mut self) {
        let mut ids: Vec<UnitId> = self
            .world
            .units()
            .filter(|u| u.army == self.human_army && !u.role.is_structure())
            .map(|u| u.id)
            .collect();
        ids.sort_by_key(|&id| (self.world.unit(id).map(|u| u.role != Role::Commander), id));
        ids.truncate(24);
        self.selected = ids;
    }

    pub fn update_hover_unit(&mut self, world_xz: Option<(i64, i64)>, pick_radius: i64) -> bool {
        let new = world_xz.and_then(|p| self.world.unit_at(p, pick_radius, None, None)).filter(|&id| {
            !self.fog_of_war
                || self
                    .world
                    .unit(id)
                    .is_some_and(|u| self.world.intel_visibility(self.human_army, u) == IntelVisibility::Visible)
        });
        let changed = new != self.ui.hover_unit;
        self.ui.hover_unit = new;
        changed
    }

    /// Unit shown on the stats card: hovered unit, else the first selected unit.
    pub fn card_unit(&self) -> Option<(UnitId, usize)> {
        if let Some(h) = self.ui.hover_unit.filter(|&id| self.world.unit(id).is_some()) {
            return Some((h, 1));
        }
        self.selected
            .iter()
            .copied()
            .find(|&id| self.world.unit(id).is_some())
            .map(|id| (id, self.selected.len()))
    }
}

pub fn fmt_ticks(t: u64) -> String {
    let s = t / TICK_HZ;
    format!("{:02}:{:02}", s / 60, s % 60)
}

// ---------------------------------------------------------------------------
// World-space overlays (range rings, formation preview, drag-line ghosts)
// ---------------------------------------------------------------------------

fn terrain_ring(
    verts: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    scmap: &ScMap,
    c: (f32, f32),
    r: f32,
    half_w: f32,
    color: [f32; 4],
) {
    if r <= 0.2 {
        return;
    }
    let seg = ((r * 1.2) as u32).clamp(24, 160);
    let pt = |i: u32| {
        let a = (i as f32 / seg as f32) * std::f32::consts::TAU;
        let x = c.0 + r * a.cos();
        let z = c.1 + r * a.sin();
        [x, sample_terrain_y(scmap, x, z) + 0.45, z]
    };
    let mut prev = pt(0);
    for i in 1..=seg {
        let next = pt(i);
        append_beam(verts, indices, prev, next, half_w, color);
        prev = next;
    }
}

/// Appends all Phase 13 world overlays to the dynamic mesh.
pub fn append_ui_overlays(
    session: &PlaySession,
    scmap: &ScMap,
    verts: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    zoom_boost: f32,
) {
    let scale = SCALE as f32;
    let w = 0.16 * zoom_boost.sqrt();
    let world = &session.world;

    // Range rings (dedupe identical rings of the same type at the same spot isn't needed;
    // cap the number of ringed units to keep big selections readable).
    let mut ringed = 0;
    let mut seen_types: Vec<&str> = Vec::new();
    for id in &session.selected {
        let Some(u) = world.unit(*id) else { continue };
        if session.selected.len() > 12 {
            // Large selections: one set of rings per unit type.
            if seen_types.contains(&u.type_id.as_str()) {
                continue;
            }
            seen_types.push(u.type_id.as_str());
        }
        ringed += 1;
        if ringed > 16 {
            break;
        }
        let c = (u.pos.x as f32 / scale, u.pos.y as f32 / scale);
        if u.max_range > 0 {
            terrain_ring(verts, indices, scmap, c, u.max_range as f32 / scale, w, [1.0, 0.30, 0.22, 0.30]);
        }
        if u.vision_radius > 0 {
            terrain_ring(verts, indices, scmap, c, u.vision_radius as f32 / scale, w * 0.8, [0.90, 0.92, 0.95, 0.25]);
        }
        if u.radar_radius > 0 {
            terrain_ring(verts, indices, scmap, c, u.radar_radius as f32 / scale, w, [0.30, 1.0, 0.55, 0.30]);
        }
        if u.shield_radius > 0 {
            terrain_ring(verts, indices, scmap, c, u.shield_radius as f32 / scale, w * 1.3, [0.35, 0.65, 1.0, 0.35]);
        }
    }

    // Formation preview
    if let Some((a, b)) = session.ui.formation_drag {
        let pa = (a.0 as f32 / scale, a.1 as f32 / scale);
        let pb = (b.0 as f32 / scale, b.1 as f32 / scale);
        let ya = sample_terrain_y(scmap, pa.0, pa.1) + 0.5;
        let yb = sample_terrain_y(scmap, pb.0, pb.1) + 0.5;
        append_beam(verts, indices, [pa.0, ya, pa.1], [pb.0, yb, pb.1], w * 1.2, [0.35, 1.0, 0.55, 0.30]);
        for &(sx, sy) in &session.ui.formation_preview {
            let x = sx as f32 / scale;
            let z = sy as f32 / scale;
            let y = sample_terrain_y(scmap, x, z) + 0.35;
            append_ring(verts, indices, [x, y, z], 1.0 * zoom_boost.sqrt(), 0.3 * zoom_boost.sqrt(), [0.35, 1.0, 0.55, 0.30], 12);
        }
    }

    // Area command circle
    if let Some((c, e)) = session.ui.area_drag {
        let r = ((e.0 - c.0) as f32).hypot((e.1 - c.1) as f32) / scale;
        terrain_ring(verts, indices, scmap, (c.0 as f32 / scale, c.1 as f32 / scale), r, w * 1.4, [0.25, 0.90, 0.78, 0.30]);
    }

    // Phase 23 map pings: pulsing ring + beacon in the ping colour.
    for (p, age) in session.chat.active_pings(Instant::now()) {
        let (x, z) = (p.x as f32 / scale, p.z as f32 / scale);
        let y = sample_terrain_y(scmap, x, z);
        let rgb = p.kind.rgb();
        let a = (1.0 - age).clamp(0.15, 1.0);
        let pulse = (age * 8.0).fract();
        let col = [rgb[0], rgb[1], rgb[2], 0.55 * a];
        terrain_ring(verts, indices, scmap, (x, z), 2.0 + 7.0 * pulse * zoom_boost.sqrt(), w * 1.6, [rgb[0], rgb[1], rgb[2], 0.5 * a * (1.0 - pulse)]);
        terrain_ring(verts, indices, scmap, (x, z), 1.6 * zoom_boost.sqrt(), w * 1.4, col);
        append_beam(verts, indices, [x, y + 0.3, z], [x, y + 14.0 * zoom_boost.sqrt(), z], w * 1.5, col);
    }

    // Drag-line construction ghosts
    if session.ui.build_drag.is_some() {
        let radius = session
            .build_mode
            .as_ref()
            .and_then(|b| world.catalog().get(b))
            .map(|t| t.radius as f32 / scale)
            .unwrap_or(2.0);
        for ((x, y), ok) in session.build_drag_sites() {
            let gx = x as f32 / scale;
            let gz = y as f32 / scale;
            let gy = sample_terrain_y(scmap, gx, gz) + 0.25;
            let col = if ok { [0.25, 0.95, 0.45, 0.35] } else { [0.98, 0.22, 0.22, 0.35] };
            append_ring(verts, indices, [gx, gy, gz], radius, 0.5 * zoom_boost.sqrt(), col, 20);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_groups_assign_add_recall_and_double_tap() {
        let mut g = ControlGroups::default();
        g.assign(1, &[5, 6, 6, 7]);
        assert_eq!(g.get(1), &[5, 6, 7]);
        g.add(1, &[7, 8]);
        assert_eq!(g.get(1), &[5, 6, 7, 8]);
        // Unit 6 died.
        let (ids, dbl) = g.recall(1, |id| id != 6, 10.0);
        assert_eq!(ids, vec![5, 7, 8]);
        assert!(!dbl);
        let (_, dbl) = g.recall(1, |_| true, 10.2);
        assert!(dbl, "second tap within window is a double-tap");
        let (_, dbl) = g.recall(1, |_| true, 10.3);
        assert!(!dbl, "a third tap starts a new sequence");
        let (_, dbl) = g.recall(2, |_| true, 10.35);
        assert!(!dbl, "different slot never double-taps");
        let (_, dbl) = g.recall(1, |_| true, 11.0);
        assert!(!dbl);
        g.assign(1, &[]);
        assert!(g.get(1).is_empty());
        assert_eq!(digit_slot(0), 0);
        assert_eq!(digit_slot(9), 9);
        assert!(g.get(42).is_empty());
    }

    #[test]
    fn line_formation_spreads_evenly_and_stacks_rows() {
        let s = 3 * SCALE;
        // 4 units along a 30m line: evenly spaced 10m apart, one row.
        let (slots, facing) = line_formation_slots((0, 0), (30 * SCALE, 0), 4, s, Some((15 * SCALE, -20 * SCALE))).unwrap();
        assert_eq!(slots, vec![(0, 0), (10_000, 0), (20_000, 0), (30_000, 0)]);
        // Units came from -y, so the formation faces +y.
        assert!(facing.1 > 0.99);
        // 7 units on a 6m line → 3 per row, 3 rows, rows stacked behind (toward hint).
        let (slots, _) = line_formation_slots((0, 0), (6 * SCALE, 0), 7, s, Some((3 * SCALE, -50 * SCALE))).unwrap();
        assert_eq!(slots.len(), 7);
        assert_eq!(&slots[0..3], &[(0, 0), (3000, 0), (6000, 0)]);
        assert_eq!(&slots[3..6], &[(0, -3000), (3000, -3000), (6000, -3000)]);
        assert_eq!(slots[6], (3000, -6000), "partial last row is centered");
        // Degenerate cases.
        assert!(line_formation_slots((5, 5), (5, 5), 3, s, None).is_none());
        assert!(line_formation_slots((0, 0), (10, 0), 0, s, None).is_none());
        // Single unit goes to the line center.
        let (one, _) = line_formation_slots((0, 0), (20 * SCALE, 0), 1, s, None).unwrap();
        assert_eq!(one, vec![(10_000, 0)]);
    }

    #[test]
    fn slot_assignment_avoids_crossing() {
        let units = vec![(1, (30_000, -5_000)), (2, (0, -5_000)), (3, (15_000, -5_000))];
        let slots = vec![(0, 0), (15_000, 0), (30_000, 0)];
        let a = assign_slots(&units, &slots, (1.0, 0.0));
        assert_eq!(a, vec![(2, (0, 0)), (3, (15_000, 0)), (1, (30_000, 0))]);
    }

    #[test]
    fn drag_line_spacing_respects_footprints() {
        let r = 2 * SCALE;
        let sites = drag_line_sites((0, 0), (30 * SCALE, 0), r);
        let step = 2 * r + SCALE + SCALE / 4; // 5.25m
        assert_eq!(sites.len(), 6); // 0, 5.25, 10.5, 15.75, 21, 26.25
        for w in sites.windows(2) {
            let d = w[1].0 - w[0].0;
            assert!(d >= 2 * r + SCALE, "neighbours must not overlap site margin");
            assert!((d - step).abs() <= 1);
        }
        // Short drag → single site.
        assert_eq!(drag_line_sites((100, 100), (200, 100), r), vec![(100, 100)]);
        // Capped length.
        assert_eq!(drag_line_sites((0, 0), (10_000 * SCALE, 0), r).len(), MAX_DRAG_LINE_SITES);
        // Diagonal keeps distance.
        let d = drag_line_sites((0, 0), (20 * SCALE, 20 * SCALE), r);
        let dist = (((d[1].0 - d[0].0) as f64).powi(2) + ((d[1].1 - d[0].1) as f64).powi(2)).sqrt();
        assert!((dist - step as f64).abs() < 2.0);
    }

    #[test]
    fn wall_drag_sites_form_a_flush_chain() {
        let r = SCALE / 2;
        // Straight 10 m line: 11 sections, 1 m apart, snapped to cell centres.
        let line = drag_wall_sites((200, 300), (10 * SCALE + 200, 300), r);
        assert_eq!(line.len(), 11);
        assert_eq!(line[0], (500, 500));
        assert!(line.windows(2).all(|p| p[1].0 - p[0].0 == SCALE && p[1].1 == p[0].1));
        // Shallow diagonal: every step moves one cell along x and at most one along z.
        let diag = drag_wall_sites((0, 0), (-9 * SCALE, 4 * SCALE), r);
        assert_eq!(diag.len(), 10);
        assert_eq!(*diag.last().unwrap(), (-9 * SCALE + 500, 4 * SCALE + 500));
        for p in diag.windows(2) {
            assert_eq!(p[1].0 - p[0].0, -SCALE);
            assert!((0..=SCALE).contains(&(p[1].1 - p[0].1)));
        }
        assert_eq!(drag_wall_sites((0, 0), (500 * SCALE, 0), r).len(), MAX_DRAG_WALL_SITES);
        assert_eq!(drag_wall_sites((10, 10), (20, 20), r), vec![(500, 500)]);
    }

    #[test]
    fn seek_target_and_plan() {
        assert_eq!(seek_target_tick(0.0, 3000), 0);
        assert_eq!(seek_target_tick(0.5, 3000), 1500);
        assert_eq!(seek_target_tick(1.7, 3000), 3000);
        assert_eq!(seek_target_tick(-1.0, 3000), 0);
        assert_eq!(seek_target_tick(f32::NAN, 3000), 0);
        assert_eq!(seek_plan(100, 400), SeekPlan { restart: false, steps: 300 });
        assert_eq!(seek_plan(400, 100), SeekPlan { restart: true, steps: 100 });
        assert_eq!(seek_plan(400, 400), SeekPlan { restart: false, steps: 0 });
        assert_eq!(next_perspective(None, &[0, 1]), Some(0));
        assert_eq!(next_perspective(Some(0), &[0, 1]), Some(1));
        assert_eq!(next_perspective(Some(1), &[0, 1]), None);
    }

    #[test]
    fn replay_seek_backward_and_forward_is_deterministic() {
        use crate::replay::ReplayRecorder;
        use crate::skirmish::ArmySetup;
        use crate::skirmish::tests::test_world;
        use crate::skirmish_ai::ScriptedAi;

        let mut world = test_world(7);
        let origin = world.clone();
        let setups = vec![ArmySetup { name: "A".into(), faction: 0, start: (0, 0), team: 0 }];
        let mut rec = ReplayRecorder::new("TEST", 7, setups);
        let mut ais = [ScriptedAi::new(0), ScriptedAi::new(1)];
        let mut hashes = std::collections::BTreeMap::new();
        for _ in 0..400 {
            for ai in &mut ais {
                for cmd in ai.decide(&world) {
                    rec.record(world.tick(), ai.army, &cmd);
                    world.issue(ai.army, cmd);
                }
            }
            world.step();
            hashes.insert(world.tick(), world.state_hash());
        }
        let replay = rec.finish(world.tick(), world.state_hash());
        assert!(!replay.commands.is_empty());

        let mut sess = PlaySession::new_replay(origin, replay, 1);
        for _ in 0..300 {
            sess.step_sim();
        }
        assert_eq!(sess.world.state_hash(), hashes[&300]);
        let pump = |s: &mut PlaySession| {
            while s.seeking() {
                s.pump_replay_seek(Duration::from_secs(5));
            }
        };
        // Backwards: restart from the pristine origin and re-simulate.
        sess.request_seek_tick(120);
        assert_eq!(sess.world.tick(), 0);
        pump(&mut sess);
        assert_eq!(sess.world.tick(), 120);
        assert_eq!(sess.world.state_hash(), hashes[&120]);
        // Forward: fast-step.
        sess.request_seek_tick(350);
        pump(&mut sess);
        assert_eq!(sess.world.state_hash(), hashes[&350]);
        // Clamped past the end; normal stepping resumes identically afterwards.
        sess.request_seek_tick(10_000);
        pump(&mut sess);
        assert_eq!(sess.world.tick(), 400);
        sess.request_seek_tick(200);
        pump(&mut sess);
        sess.step_sim();
        assert_eq!(sess.world.state_hash(), hashes[&201]);
        // Perspective cycling toggles fog.
        sess.cycle_replay_perspective();
        assert!(sess.fog_of_war && sess.human_army == 0);
        sess.cycle_replay_perspective();
        assert_eq!(sess.human_army, 1);
        sess.cycle_replay_perspective();
        assert!(!sess.fog_of_war);
    }

    #[test]
    fn formation_drag_build_drag_and_groups_issue_commands() {
        use crate::skirmish::tests::test_world;
        let mut w = test_world(3);
        let a = w.spawn_unit(0, "BT", (60 * SCALE, 60 * SCALE)).unwrap();
        let b = w.spawn_unit(0, "BT", (62 * SCALE, 60 * SCALE)).unwrap();
        let c = w.spawn_unit(0, "BT", (64 * SCALE, 60 * SCALE)).unwrap();
        w.rebuild_spatial();
        let mut s = PlaySession::new(w, 0, false, 1);
        s.selected = vec![a, b, c];
        let _ = s.world.drain_pending();

        // Control groups
        s.assign_control_group(1, false);
        s.selected.clear();
        assert!(!s.recall_control_group(1, false, 1.0));
        assert_eq!(s.selected, vec![a, b, c]);
        assert!(s.recall_control_group(1, false, 1.1), "double tap");

        // Formation drag → one Move per unit, preview matches the slots.
        s.update_formation_drag((80 * SCALE, 90 * SCALE), (100 * SCALE, 90 * SCALE));
        assert_eq!(s.ui.formation_preview.len(), 3);
        assert!(s.finish_formation_drag(false));
        let cmds = s.world.drain_pending();
        assert_eq!(cmds.len(), 3);
        let mut targets: Vec<(i64, i64)> = cmds
            .iter()
            .map(|(_, c)| match c {
                Command::Move { units, x, y, .. } => {
                    assert_eq!(units.len(), 1);
                    (*x, *y)
                }
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        targets.sort();
        assert_eq!(targets, vec![(80_000, 90_000), (90_000, 90_000), (100_000, 90_000)]);
        assert!(s.ui.formation_drag.is_none() && s.ui.formation_preview.is_empty());

        // Drag-line construction with the ACU.
        let acu = s.world.commander_of(0).unwrap().id;
        s.selected = vec![acu];
        s.trigger_build_mode(Role::Generator);
        s.ui.build_drag = Some(((100 * SCALE, 150 * SCALE), (140 * SCALE, 150 * SCALE)));
        let sites = s.build_drag_sites();
        assert!(sites.len() >= 3);
        let valid = sites.iter().filter(|(_, ok)| *ok).count();
        assert!(s.finish_build_drag(false));
        let cmds = s.world.drain_pending();
        assert_eq!(cmds.len(), valid);
        for (i, (_, c)) in cmds.iter().enumerate() {
            match c {
                Command::Build { builder, queue, .. } => {
                    assert_eq!(*builder, acu);
                    assert_eq!(*queue, i > 0);
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        assert!(s.build_mode.is_none());
    }

    #[test]
    fn project_center_of_view_lands_mid_screen() {
        let cam = OrbitCamera::for_map(256.0, 256.0, 10.0, 1.6);
        let (sx, sy) = project_to_screen(&cam, cam.target, 1600.0, 1000.0).unwrap();
        assert!((sx - 800.0).abs() < 1.0 && (sy - 500.0).abs() < 1.0, "{sx},{sy}");
    }

    #[test]
    fn phase18_repeat_shield_and_card_info_work() {
        use crate::skirmish::tests::test_world;
        let mut w = test_world(180);
        let acu = w.commander_of(0).unwrap().id;
        w.issue(0, Command::Build { builder: acu, unit_type: "RF".into(), x: 64 * SCALE, y: 50 * SCALE, queue: false });
        w.run(350);
        let fac = w.units().find(|u| u.army == 0 && u.role == Role::Factory).unwrap().id;
        let mut s = PlaySession::new(w, 0, false, 1);
        s.selected = vec![fac];
        s.trigger_toggle_repeat();
        s.step_sim();
        assert!(s.world.unit(fac).unwrap().repeat_queue);
        let lines = unit_card_lines(&s.world, fac, 1);
        assert!(lines.iter().any(|l| l.contains("REPEAT")), "{lines:?}");

        s.trigger_toggle_repeat();
        s.step_sim();
        assert!(!s.world.unit(fac).unwrap().repeat_queue);
    }
}
