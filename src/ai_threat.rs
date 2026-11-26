//! Phase 19 AI intel layer: coarse threat / influence map, unit power estimates and
//! navigation reachability.
//!
//! Everything here is integer-only and order-independent so the AI that uses it stays
//! bit-identical on every lockstep peer:
//! - [`Power`]: anti-surface / anti-air fighting strength `sqrt(dps * hp)` (Lanchester
//!   square-law friendly: group strength is the sum of member powers).
//! - [`ThreatMap`]: at most 64x64 cells holding remembered enemy mobile / static / air /
//!   anti-air / economic value layers plus a fresh friendly layer. Cells currently in
//!   vision are overwritten by what is seen; unobserved cells keep a decaying memory.
//! - [`Reach`]: flood-filled bitset of grid cells reachable from a start position.

use std::collections::VecDeque;

use crate::pathfinding::PathGrid;
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{Unit, isqrt};

/// Upper bound on cells per axis (bounded memory & per-think cost on 81 km maps).
pub const MAX_AXIS_CELLS: i64 = 64;
/// Smallest cell edge (milli-units).
pub const MIN_CELL_MILLI: i64 = 16 * SCALE;
/// Remembered mobile ground threat halves every 30 s of not being observed.
pub const MOBILE_HALF_LIFE: u64 = 300;
/// Remembered air units and anti-air halve every 60 s.
pub const AIR_HALF_LIFE: u64 = 600;
/// Remembered structures (defences, economy) halve every 5 minutes.
pub const STATIC_HALF_LIFE: u64 = 3_000;
/// Unknown radar contact: assumed to be roughly one T1 tank.
pub const RADAR_BLIP_POWER: i32 = 40;

/// Visibility levels stored in [`ThreatMap::vis`].
pub const VIS_NONE: u8 = 0;
pub const VIS_RADAR: u8 = 1;
pub const VIS_SIGHT: u8 = 2;

/// Fighting strength split by what it can shoot at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Power {
    pub surface: i32,
    pub aa: i32,
}

impl Power {
    pub fn total(self) -> i32 {
        self.surface.max(self.aa)
    }
}

/// Whole damage per second of one weapon (`damage` milli per shot, `reset` ticks).
pub fn weapon_dps(damage_milli: i64, reset_ticks: u32) -> i64 {
    damage_milli.max(0) * TICK_HZ as i64 / (reset_ticks.max(1) as i64) / SCALE
}

/// `sqrt(dps * hp)` with hp in whole units.
pub fn power_of(dps: i64, hp_milli: i64) -> i32 {
    if dps <= 0 {
        return 0;
    }
    let hp = (hp_milli / SCALE).max(1);
    isqrt((dps as i128) * (hp as i128)).min(i32::MAX as i64) as i32
}

/// Current fighting power of a live unit (uses its current hp).
pub fn unit_power(u: &Unit) -> Power {
    let (mut s, mut a) = (0i64, 0i64);
    for w in &u.weapons {
        let d = weapon_dps(w.damage, w.reset);
        if w.anti_surface {
            s += d;
        }
        if w.anti_air {
            a += d;
        }
    }
    Power {
        surface: power_of(s, u.hp),
        aa: power_of(a, u.hp),
    }
}

/// Integer half-life decay: halves every `half_life` ticks, linear within a half-life.
pub fn decay(v: i32, dt: u64, half_life: u64) -> i32 {
    if v <= 0 {
        return 0;
    }
    if dt == 0 {
        return v;
    }
    let hl = half_life.max(1);
    let halvings = dt / hl;
    if halvings >= 31 {
        return 0;
    }
    let mut x = (v >> halvings) as i64;
    let rem = (dt % hl) as i64;
    x -= x * rem / (2 * hl as i64);
    x.max(0) as i32
}

fn d2(a: (i64, i64), b: (i64, i64)) -> i128 {
    let dx = (a.0 - b.0) as i128;
    let dy = (a.1 - b.1) as i128;
    dx * dx + dy * dy
}

/// Coarse per-army influence grid with decaying memory of enemy sightings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreatMap {
    pub cols: i64,
    pub rows: i64,
    /// Cell edge in milli-units.
    pub cell: i64,
    /// Enemy mobile anti-surface power (land, naval and air-to-ground).
    pub mobile: Vec<i32>,
    /// Enemy static anti-surface defence power (point defence, armed structures).
    pub defense: Vec<i32>,
    /// Enemy anti-air power (mobile and static).
    pub anti_air: Vec<i32>,
    /// Enemy air unit power (signals the need for AA).
    pub air: Vec<i32>,
    /// Enemy economic value (extractors, generators, factories, engineers).
    pub eco: Vec<i32>,
    /// Friendly (team) anti-surface power at the last update.
    pub ally: Vec<i32>,
    /// Friendly anti-air power at the last update.
    pub ally_aa: Vec<i32>,
    /// Visibility at the last update ([`VIS_NONE`], [`VIS_RADAR`], [`VIS_SIGHT`]).
    pub vis: Vec<u8>,
    pub last_update: u64,
}

impl ThreatMap {
    pub fn new(size: (i64, i64)) -> Self {
        let longest = size.0.max(size.1).max(SCALE);
        let cell = ((longest + MAX_AXIS_CELLS - 1) / MAX_AXIS_CELLS).max(MIN_CELL_MILLI);
        let cols = ((size.0.max(1) + cell - 1) / cell).max(1);
        let rows = ((size.1.max(1) + cell - 1) / cell).max(1);
        let n = (cols * rows) as usize;
        Self {
            cols,
            rows,
            cell,
            mobile: vec![0; n],
            defense: vec![0; n],
            anti_air: vec![0; n],
            air: vec![0; n],
            eco: vec![0; n],
            ally: vec![0; n],
            ally_aa: vec![0; n],
            vis: vec![VIS_NONE; n],
            last_update: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.vis.len()
    }

    pub fn cell_xy(&self, p: (i64, i64)) -> (i64, i64) {
        (
            (p.0 / self.cell).clamp(0, self.cols - 1),
            (p.1 / self.cell).clamp(0, self.rows - 1),
        )
    }

    pub fn index(&self, p: (i64, i64)) -> usize {
        let (cx, cy) = self.cell_xy(p);
        (cy * self.cols + cx) as usize
    }

    pub fn center(&self, idx: usize) -> (i64, i64) {
        let i = idx as i64;
        let (cx, cy) = (i % self.cols, i / self.cols);
        (cx * self.cell + self.cell / 2, cy * self.cell + self.cell / 2)
    }

    /// Starts an observation pass at `tick`: decays remembered enemy layers by the time
    /// elapsed since the previous pass and clears visibility and friendly layers.
    pub fn begin(&mut self, tick: u64) {
        let dt = tick.saturating_sub(self.last_update);
        if dt > 0 {
            for v in &mut self.mobile {
                *v = decay(*v, dt, MOBILE_HALF_LIFE);
            }
            for v in &mut self.air {
                *v = decay(*v, dt, AIR_HALF_LIFE);
            }
            for v in &mut self.anti_air {
                *v = decay(*v, dt, AIR_HALF_LIFE);
            }
            for v in &mut self.defense {
                *v = decay(*v, dt, STATIC_HALF_LIFE);
            }
            for v in &mut self.eco {
                *v = decay(*v, dt, STATIC_HALF_LIFE);
            }
        }
        self.last_update = tick;
        self.vis.fill(VIS_NONE);
        self.ally.fill(0);
        self.ally_aa.fill(0);
    }

    /// Calls `f(index, center)` for every cell whose centre lies within `radius` of `p`.
    fn for_cells_in_radius(&self, p: (i64, i64), radius: i64, mut f: impl FnMut(usize, (i64, i64))) {
        let r = radius.max(0);
        let (x0, y0) = self.cell_xy((p.0 - r, p.1 - r));
        let (x1, y1) = self.cell_xy((p.0 + r, p.1 + r));
        let lim = (r as i128) * (r as i128);
        for cy in y0..=y1 {
            for cx in x0..=x1 {
                let c = (cx * self.cell + self.cell / 2, cy * self.cell + self.cell / 2);
                if d2(c, p) <= lim {
                    f((cy * self.cols + cx) as usize, c);
                }
            }
        }
    }

    /// Marks cells covered by a friendly observer's vision or radar.
    pub fn mark_eye(&mut self, p: (i64, i64), vision: i64, radar: i64) {
        let half = self.cell / 2;
        let own = self.index(p);
        self.vis[own] = VIS_SIGHT;
        let mut marks: Vec<(usize, u8)> = Vec::new();
        if radar > 0 {
            self.for_cells_in_radius(p, radar + half, |i, _| marks.push((i, VIS_RADAR)));
        }
        if vision > 0 {
            self.for_cells_in_radius(p, vision + half / 2, |i, _| marks.push((i, VIS_SIGHT)));
        }
        for (i, level) in marks {
            if self.vis[i] < level {
                self.vis[i] = level;
            }
        }
    }

    /// Forgets remembered contacts in currently observed cells. Call after every
    /// [`Self::mark_eye`] and before adding this pass's contacts.
    pub fn clear_observed(&mut self) {
        for i in 0..self.vis.len() {
            match self.vis[i] {
                VIS_SIGHT => {
                    self.mobile[i] = 0;
                    self.defense[i] = 0;
                    self.anti_air[i] = 0;
                    self.air[i] = 0;
                    self.eco[i] = 0;
                }
                VIS_RADAR => {
                    self.mobile[i] = 0;
                    self.air[i] = 0;
                }
                _ => {}
            }
        }
    }

    pub fn visibility(&self, p: (i64, i64)) -> u8 {
        self.vis[self.index(p)]
    }

    /// Records a fully identified enemy unit.
    pub fn add_enemy(&mut self, p: (i64, i64), power: Power, is_air: bool, is_struct: bool, eco_value: i32) {
        let i = self.index(p);
        if is_air {
            self.air[i] = self.air[i].saturating_add(power.total().max(1));
            self.mobile[i] = self.mobile[i].saturating_add(power.surface);
        } else if is_struct {
            self.defense[i] = self.defense[i].saturating_add(power.surface);
        } else {
            self.mobile[i] = self.mobile[i].saturating_add(power.surface);
        }
        self.anti_air[i] = self.anti_air[i].saturating_add(power.aa);
        self.eco[i] = self.eco[i].saturating_add(eco_value.max(0));
    }

    /// Records an unidentified radar contact.
    pub fn add_blip(&mut self, p: (i64, i64)) {
        let i = self.index(p);
        self.mobile[i] = self.mobile[i].saturating_add(RADAR_BLIP_POWER);
    }

    pub fn add_ally(&mut self, p: (i64, i64), power: Power) {
        let i = self.index(p);
        self.ally[i] = self.ally[i].saturating_add(power.surface);
        self.ally_aa[i] = self.ally_aa[i].saturating_add(power.aa);
    }

    /// Sum of `layer` over cells whose centre is within `radius` (+ half a cell) of `p`.
    pub fn sum(&self, layer: &[i32], p: (i64, i64), radius: i64) -> i64 {
        let mut s = 0i64;
        self.for_cells_in_radius(p, radius + self.cell / 2, |i, _| s += layer[i] as i64);
        s
    }

    /// Enemy anti-surface threat (mobile + static) around `p`.
    pub fn surface_threat(&self, p: (i64, i64), radius: i64) -> i64 {
        self.sum(&self.mobile, p, radius) + self.sum(&self.defense, p, radius)
    }
}

/// Bitset of path-grid cells reachable from a seed position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reach {
    pub width: u32,
    pub height: u32,
    pub cell: i64,
    bits: Vec<u64>,
    pub count: usize,
}

impl Reach {
    /// 8-neighbour flood fill from the unblocked cell nearest `from` (within 8 cells).
    pub fn flood(grid: &PathGrid, from: (i64, i64)) -> Self {
        let (w, h) = (grid.width, grid.height);
        let n = (w as usize) * (h as usize);
        let mut r = Self {
            width: w,
            height: h,
            cell: grid.cell_size_milli,
            bits: vec![0; n.div_ceil(64)],
            count: 0,
        };
        let (sx, sz) = grid.world_to_cell(from.0, from.1);
        let mut seed = None;
        'search: for ring in 0..=8i64 {
            for dz in -ring..=ring {
                for dx in -ring..=ring {
                    if dx.abs() != ring && dz.abs() != ring {
                        continue;
                    }
                    let (cx, cz) = (sx as i64 + dx, sz as i64 + dz);
                    if cx < 0 || cz < 0 || cx >= w as i64 || cz >= h as i64 {
                        continue;
                    }
                    if !grid.is_blocked(cx as u32, cz as u32) {
                        seed = Some((cx as u32, cz as u32));
                        break 'search;
                    }
                }
            }
        }
        let Some(seed) = seed else {
            return r;
        };
        let mut q = VecDeque::new();
        r.set(seed.0, seed.1);
        q.push_back(seed);
        while let Some((cx, cz)) = q.pop_front() {
            for (dx, dz) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                let (nx, nz) = (cx as i64 + dx, cz as i64 + dz);
                if nx < 0 || nz < 0 || nx >= w as i64 || nz >= h as i64 {
                    continue;
                }
                let (nx, nz) = (nx as u32, nz as u32);
                if grid.is_blocked(nx, nz) || r.get(nx, nz) {
                    continue;
                }
                r.set(nx, nz);
                q.push_back((nx, nz));
            }
        }
        r
    }

    fn set(&mut self, cx: u32, cz: u32) {
        let i = (cz as usize) * (self.width as usize) + cx as usize;
        if self.bits[i / 64] & (1 << (i % 64)) == 0 {
            self.bits[i / 64] |= 1 << (i % 64);
            self.count += 1;
        }
    }

    pub fn get(&self, cx: u32, cz: u32) -> bool {
        if cx >= self.width || cz >= self.height {
            return false;
        }
        let i = (cz as usize) * (self.width as usize) + cx as usize;
        self.bits[i / 64] & (1 << (i % 64)) != 0
    }

    /// True if the cell of `p`, or one within `slack` cells of it, is reachable.
    pub fn near(&self, p: (i64, i64), slack: i64) -> bool {
        let cs = self.cell.max(1);
        let cx = (p.0 / cs).clamp(0, self.width as i64 - 1);
        let cz = (p.1 / cs).clamp(0, self.height as i64 - 1);
        for dz in -slack..=slack {
            for dx in -slack..=slack {
                let (x, z) = (cx + dx, cz + dz);
                if x >= 0 && z >= 0 && self.get(x as u32, z as u32) {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_halves_per_half_life_and_is_monotonic() {
        assert_eq!(decay(1000, 0, 300), 1000);
        assert_eq!(decay(1000, 300, 300), 500);
        assert_eq!(decay(1000, 600, 300), 250);
        assert_eq!(decay(1000, 150, 300), 750);
        assert_eq!(decay(1000, 300 * 40, 300), 0);
        assert_eq!(decay(-5, 10, 300), 0);
        let mut prev = 1000;
        for dt in (0..2000).step_by(37) {
            let v = decay(1000, dt, 300);
            assert!(v <= prev, "decay must not grow: dt={dt} {v} > {prev}");
            prev = v;
        }
    }

    #[test]
    fn power_follows_dps_and_hp() {
        assert_eq!(weapon_dps(10 * SCALE, 5), 20);
        assert_eq!(power_of(25, 400 * SCALE), 100);
        assert_eq!(power_of(0, 400 * SCALE), 0);
        assert!(power_of(100, 12_000 * SCALE) > power_of(24, 300 * SCALE) * 10);
    }

    #[test]
    fn threat_map_accumulates_clears_seen_cells_and_remembers_unseen() {
        let mut tm = ThreatMap::new((1024 * SCALE, 1024 * SCALE));
        assert_eq!(tm.cols, 64);
        assert_eq!(tm.cell, 16 * SCALE);
        let base = (100 * SCALE, 100 * SCALE);
        let far = (600 * SCALE, 600 * SCALE);
        // Tick 10: an observer near `far` sees two tanks and an extractor.
        tm.begin(10);
        tm.mark_eye(far, 26 * SCALE, 0);
        tm.clear_observed();
        let tank = Power { surface: 85, aa: 0 };
        tm.add_enemy(far, tank, false, false, 0);
        tm.add_enemy(far, tank, false, false, 0);
        tm.add_enemy(far, Power::default(), false, true, 36);
        tm.add_ally(base, Power { surface: 300, aa: 10 });
        assert_eq!(tm.surface_threat(far, 0), 170);
        assert_eq!(tm.sum(&tm.eco, far, 20 * SCALE), 36);
        assert_eq!(tm.sum(&tm.ally, base, 0), 300);
        assert_eq!(tm.visibility(far), VIS_SIGHT);
        assert_eq!(tm.visibility(base), VIS_NONE);

        // Tick 310: nobody looks at `far` any more: mobile memory halves, eco nearly kept.
        tm.begin(310);
        tm.mark_eye(base, 26 * SCALE, 0);
        tm.clear_observed();
        assert_eq!(tm.surface_threat(far, 0), 85);
        assert_eq!(tm.sum(&tm.eco, far, 0), decay(36, 300, STATIC_HALF_LIFE) as i64);
        assert_eq!(tm.sum(&tm.ally, base, 0), 0, "friendly layer is rebuilt every pass");

        // Tick 320: `far` observed again and empty -> memory wiped.
        tm.begin(320);
        tm.mark_eye(far, 26 * SCALE, 0);
        tm.clear_observed();
        assert_eq!(tm.surface_threat(far, 0), 0);
        assert_eq!(tm.sum(&tm.eco, far, 0), 0);

        // Radar only refreshes mobile layers; static memory survives.
        tm.add_enemy(far, Power { surface: 400, aa: 0 }, false, true, 50);
        tm.begin(330);
        tm.mark_eye((far.0 - 60 * SCALE, far.1), 0, 115 * SCALE);
        tm.clear_observed();
        assert_eq!(tm.visibility(far), VIS_RADAR);
        tm.add_blip(far);
        assert_eq!(tm.sum(&tm.defense, far, 0), 400);
        assert_eq!(tm.sum(&tm.mobile, far, 0), RADAR_BLIP_POWER as i64);
    }

    #[test]
    fn small_maps_use_min_cell_and_indices_round_trip() {
        let tm = ThreatMap::new((256 * SCALE, 256 * SCALE));
        assert_eq!((tm.cols, tm.rows, tm.cell), (16, 16, 16 * SCALE));
        let big = ThreatMap::new((4096 * SCALE, 2048 * SCALE));
        assert_eq!((big.cols, big.rows), (64, 32));
        for idx in [0usize, 17, 255] {
            assert_eq!(tm.index(tm.center(idx)), idx);
        }
        assert_eq!(tm.index((-5, 999_999_999)), ((tm.rows - 1) * tm.cols) as usize);
    }

    #[test]
    fn reach_flood_respects_walls() {
        let mut g = PathGrid::new(20, 20, 4 * SCALE);
        for z in 0..20 {
            g.set_blocked(10, z, true);
        }
        let r = Reach::flood(&g, (2 * SCALE, 2 * SCALE));
        assert_eq!(r.count, 200);
        assert!(r.near((30 * SCALE, 30 * SCALE), 0));
        assert!(!r.near((60 * SCALE, 30 * SCALE), 0));
        // Wall cell itself is adjacent to reachable cells.
        assert!(r.near((42 * SCALE, 30 * SCALE), 1));
        g.set_blocked(10, 19, false);
        let r2 = Reach::flood(&g, (2 * SCALE, 2 * SCALE));
        assert_eq!(r2.count, 400 - 19);
    }
}
