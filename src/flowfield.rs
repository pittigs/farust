//! Phase 12 & 16 deterministic multi-domain grid builder and bounded flow-field cache.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use crate::pathfinding::PathGrid;
use crate::scmap::ScMap;
use crate::sim::SCALE;

/// Maximum number of cached flow fields before deterministic LRU eviction.
#[allow(dead_code)]
pub const MAX_FLOW_FIELDS: usize = 64;

/// One flow-field request for [`FlowFieldCache::prefetch`]: domain, its grid and the goal cell.
pub type FlowRequest = (NavDomain, Arc<PathGrid>, (u32, u32));

/// A grid + goal cell to compute on a worker.
type FlowJob = (Arc<PathGrid>, (u32, u32));

/// Memory budget for cached flow fields (see [`FlowFieldCache::capacity_for`]).
pub const FLOW_CACHE_BUDGET_BYTES: usize = 192 << 20;

const GOAL_SEARCH_RADIUS: i32 = 6;
const NAVAL_MIN_DEPTH_MILLI: i64 = 400;

const DIRS: [(i8, i8, u32); 8] = [
    (1, 0, 10),
    (-1, 0, 10),
    (0, 1, 10),
    (0, -1, 10),
    (1, 1, 14),
    (1, -1, 14),
    (-1, 1, 14),
    (-1, -1, 14),
];

/// Navigation domain derived from a unit's blueprint motion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
#[allow(dead_code)]
pub enum NavDomain {
    Land = 0,
    Naval = 1,
    Amphibious = 2,
    Air = 3,
}

impl NavDomain {
    /// Maps a Forged Alliance `RULEUMT_*` motion type string to a [`NavDomain`].
    #[allow(dead_code)]
    pub fn from_motion_type(motion: &str) -> Self {
        match motion {
            "RULEUMT_Air" => Self::Air,
            "RULEUMT_Water" | "RULEUMT_SurfacingSub" => Self::Naval,
            "RULEUMT_Amphibious" | "RULEUMT_AmphibiousFloating" | "RULEUMT_Hover" => {
                Self::Amphibious
            }
            _ => Self::Land,
        }
    }
}

/// Deterministic integer Dijkstra cost-to-goal flow field over a [`PathGrid`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct FlowField {
    pub width: u32,
    pub height: u32,
    pub cell_size_milli: i64,
    pub goal_cell: (u32, u32),
    pub cost: Vec<u32>,
    pub dir: Vec<(i8, i8)>,
}

fn find_nearest_passable_goal(grid: &PathGrid, start: (u32, u32)) -> Option<(u32, u32)> {
    if !grid.is_blocked(start.0, start.1) {
        return Some(start);
    }

    const SIDE: usize = (GOAL_SEARCH_RADIUS as usize) * 2 + 1;
    let mut visited = [false; SIDE * SIDE];
    let mut queue = VecDeque::new();

    let visited_idx = |ox: i32, oz: i32| -> usize {
        ((oz + GOAL_SEARCH_RADIUS) as usize) * SIDE + ((ox + GOAL_SEARCH_RADIUS) as usize)
    };

    visited[visited_idx(0, 0)] = true;
    queue.push_back((0i32, 0i32, 0u32));

    while let Some((ox, oz, dist)) = queue.pop_front() {
        let cx_i = start.0 as i64 + i64::from(ox);
        let cz_i = start.1 as i64 + i64::from(oz);
        if cx_i >= 0 && cz_i >= 0 && cx_i < grid.width as i64 && cz_i < grid.height as i64 {
            let cx = cx_i as u32;
            let cz = cz_i as u32;
            if !grid.is_blocked(cx, cz) {
                return Some((cx, cz));
            }
        }

        if dist < GOAL_SEARCH_RADIUS as u32 {
            for &(dx, dz, _) in &DIRS {
                let nox = ox + i32::from(dx);
                let noz = oz + i32::from(dz);
                if nox.abs() > GOAL_SEARCH_RADIUS || noz.abs() > GOAL_SEARCH_RADIUS {
                    continue;
                }
                let ncx_i = start.0 as i64 + i64::from(nox);
                let ncz_i = start.1 as i64 + i64::from(noz);
                if ncx_i < 0
                    || ncz_i < 0
                    || ncx_i >= grid.width as i64
                    || ncz_i >= grid.height as i64
                {
                    continue;
                }
                let v_idx = visited_idx(nox, noz);
                if !visited[v_idx] {
                    visited[v_idx] = true;
                    queue.push_back((nox, noz, dist + 1));
                }
            }
        }
    }

    None
}

#[inline]
fn step_is_valid(grid: &PathGrid, cx: u32, cz: u32, dx: i8, dz: i8) -> Option<(u32, u32)> {
    let nx_i = cx as i64 + i64::from(dx);
    let nz_i = cz as i64 + i64::from(dz);
    if nx_i < 0 || nz_i < 0 || nx_i >= grid.width as i64 || nz_i >= grid.height as i64 {
        return None;
    }
    let nx = nx_i as u32;
    let nz = nz_i as u32;
    if grid.is_blocked(nx, nz) {
        return None;
    }
    // Prevent diagonal corner cutting when either shared cardinal neighbor is blocked.
    if dx != 0 && dz != 0 && (grid.is_blocked(nx, cz) || grid.is_blocked(cx, nz)) {
        return None;
    }
    Some((nx, nz))
}

impl FlowField {
    /// Computes an integer Dijkstra cost-to-goal flow field outward from `goal_cell`
    /// (falling back to the nearest passable cell within BFS radius 6 if blocked).
    #[allow(dead_code)]
    pub fn compute(grid: &PathGrid, goal_cell: (u32, u32)) -> Self {
        let _p = crate::profiler::scope("flowfield.compute");
        let width = grid.width.max(1);
        let height = grid.height.max(1);
        let cell_size_milli = grid.cell_size_milli.max(1);
        let total_cells = (width as usize) * (height as usize);

        let clamped_goal = (goal_cell.0.min(width - 1), goal_cell.1.min(height - 1));
        let mut cost = vec![u32::MAX; total_cells];
        let mut dir = vec![(0i8, 0i8); total_cells];

        let Some(effective_goal) = find_nearest_passable_goal(grid, clamped_goal) else {
            return Self {
                width,
                height,
                cell_size_milli,
                goal_cell: clamped_goal,
                cost,
                dir,
            };
        };

        // Per-cell bitmask of valid moves (bit k = `DIRS[k]`), shared by both passes.
        let mut moves = vec![0u8; total_cells];
        for cz in 0..height {
            for cx in 0..width {
                let idx = (cz * width + cx) as usize;
                if grid.is_blocked(cx, cz) {
                    continue;
                }
                let mut m = 0u8;
                for (k, &(dx, dz, _)) in DIRS.iter().enumerate() {
                    if step_is_valid(grid, cx, cz, dx, dz).is_some() {
                        m |= 1 << k;
                    }
                }
                moves[idx] = m;
            }
        }
        let w_i = width as i64;
        let offs: [i64; 8] = DIRS.map(|(dx, dz, _)| i64::from(dz) * w_i + i64::from(dx));

        // Dial's bucket queue (edge costs 10/14): yields the exact same shortest-path costs as
        // a binary-heap Dijkstra, so flow directions are bit-identical.
        let goal_idx = effective_goal.1 * width + effective_goal.0;
        cost[goal_idx as usize] = 0;
        const NB: usize = 16;
        let mut buckets: [Vec<u32>; NB] = Default::default();
        buckets[0].push(goal_idx);
        let mut pending = 1usize;
        let mut cur_cost = 0u32;
        while pending > 0 {
            let b = (cur_cost as usize) % NB;
            let mut list = std::mem::take(&mut buckets[b]);
            pending -= list.len();
            for &cur_idx in &list {
                if cost[cur_idx as usize] != cur_cost {
                    continue;
                }
                let m = moves[cur_idx as usize];
                for (k, &(_, _, step_cost)) in DIRS.iter().enumerate() {
                    if m & (1 << k) == 0 {
                        continue;
                    }
                    let n_idx = (cur_idx as i64 + offs[k]) as usize;
                    let new_cost = cur_cost + step_cost;
                    if new_cost < cost[n_idx] {
                        cost[n_idx] = new_cost;
                        buckets[(new_cost as usize) % NB].push(n_idx as u32);
                        pending += 1;
                    }
                }
            }
            list.clear();
            if buckets[b].is_empty() {
                buckets[b] = list;
            }
            cur_cost += 1;
        }

        // Resolve downhill flow directions using fixed `DIRS` priority so ties are invariant.
        for idx in 0..total_cells {
            if cost[idx] == 0 || cost[idx] == u32::MAX {
                continue;
            }
            let m = moves[idx];
            let mut best_total = u32::MAX;
            let mut best_dir = (0i8, 0i8);
            for (k, &(dx, dz, step_cost)) in DIRS.iter().enumerate() {
                if m & (1 << k) == 0 {
                    continue;
                }
                let n_cost = cost[(idx as i64 + offs[k]) as usize];
                if n_cost == u32::MAX {
                    continue;
                }
                let total = n_cost.saturating_add(step_cost);
                if total < best_total {
                    best_total = total;
                    best_dir = (dx, dz);
                }
            }
            dir[idx] = best_dir;
        }

        Self {
            width,
            height,
            cell_size_milli,
            goal_cell: effective_goal,
            cost,
            dir,
        }
    }

    /// Converts world milli-unit coordinates `(x_milli, z_milli)` to grid cell coordinates.
    #[allow(dead_code)]
    pub fn world_to_cell(&self, x_milli: i64, z_milli: i64) -> (u32, u32) {
        let cs = self.cell_size_milli.max(1);
        let cx = (x_milli / cs).clamp(0, (self.width.max(1) as i64) - 1) as u32;
        let cz = (z_milli / cs).clamp(0, (self.height.max(1) as i64) - 1) as u32;
        (cx, cz)
    }

    /// Converts grid cell coordinates `(cx, cz)` to the world milli-unit center of that cell.
    #[allow(dead_code)]
    pub fn cell_to_world(&self, cx: u32, cz: u32) -> (i64, i64) {
        let half = self.cell_size_milli / 2;
        (
            (cx as i64) * self.cell_size_milli + half,
            (cz as i64) * self.cell_size_milli + half,
        )
    }

    /// Returns the next world-space waypoint from `pos` toward `final_goal`.
    /// Returns `final_goal` when already at `goal_cell`, one valid step from `goal_cell`,
    /// or on an unreachable `(0, 0)` cell.
    #[allow(dead_code)]
    pub fn next_waypoint(&self, pos: (i64, i64), final_goal: (i64, i64)) -> (i64, i64) {
        let (cx, cz) = self.world_to_cell(pos.0, pos.1);
        if (cx, cz) == self.goal_cell {
            return final_goal;
        }
        let idx = (cz as usize) * (self.width as usize) + (cx as usize);
        let (dx, dz) = self.dir.get(idx).copied().unwrap_or((0, 0));
        if dx == 0 && dz == 0 {
            return final_goal;
        }
        let nx_i = cx as i64 + i64::from(dx);
        let nz_i = cz as i64 + i64::from(dz);
        if nx_i < 0 || nz_i < 0 || nx_i >= self.width as i64 || nz_i >= self.height as i64 {
            return final_goal;
        }
        let nx = nx_i as u32;
        let nz = nz_i as u32;
        if (nx, nz) == self.goal_cell {
            return final_goal;
        }
        self.cell_to_world(nx, nz)
    }
}

/// Deterministic bounded flow-field cache backed by a `BTreeMap` (max [`MAX_FLOW_FIELDS`]).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FlowFieldCache {
    entries: BTreeMap<(NavDomain, u32, u32), (u64, Arc<FlowField>)>,
}

impl Default for FlowFieldCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowFieldCache {
    /// Creates an empty [`FlowFieldCache`].
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Returns the number of currently cached flow fields.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no flow fields are cached.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears all cached flow fields.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Cache capacity for fields over `grid`: at least [`MAX_FLOW_FIELDS`], growing up to
    /// [`FLOW_CACHE_BUDGET_BYTES`] of field memory (fields are a pure function of grid + goal
    /// cell, so capacity never changes simulation results — only how often they are recomputed).
    pub fn capacity_for(grid: &PathGrid) -> usize {
        let cells = (grid.width.max(1) as usize) * (grid.height.max(1) as usize);
        let per_field = cells * 6 + 64;
        (FLOW_CACHE_BUDGET_BYTES / per_field).clamp(MAX_FLOW_FIELDS, 4096)
    }

    fn evict_to(&mut self, cap: usize) {
        while self.entries.len() > cap {
            let Some(evict_key) = self
                .entries
                .iter()
                .min_by_key(|&(&k, &(last_tick, _))| (last_tick, k))
                .map(|(&k, _)| k)
            else {
                break;
            };
            self.entries.remove(&evict_key);
        }
    }

    /// Looks up `(domain, goal_cell)` or computes and caches a new [`FlowField`],
    /// deterministically evicting the entry with smallest `(last_used_tick, key)` if full.
    #[allow(dead_code)]
    pub fn get_or_compute(
        &mut self,
        domain: NavDomain,
        grid: &PathGrid,
        goal: (i64, i64),
        tick: u64,
    ) -> Arc<FlowField> {
        let goal_cell = grid.world_to_cell(goal.0, goal.1);
        let key = (domain, goal_cell.0, goal_cell.1);

        if let Some((last_used_tick, field)) = self.entries.get_mut(&key) {
            *last_used_tick = tick;
            return Arc::clone(field);
        }

        let cap = Self::capacity_for(grid);
        self.evict_to(cap.saturating_sub(1));

        let field = Arc::new(FlowField::compute(grid, goal_cell));
        self.entries.insert(key, (tick, Arc::clone(&field)));
        field
    }

    /// Computes every missing `(domain, goal_cell)` field of this tick up front — in parallel on
    /// `pool` when given — and marks all of them used at `tick`, so the following sequential
    /// [`Self::get_or_compute`] calls are pure cache hits. Results are identical to computing
    /// lazily because a field depends only on its grid and goal cell.
    pub fn prefetch(
        &mut self,
        wanted: &[FlowRequest],
        tick: u64,
        pool: Option<&crate::jobs::WorkerPool>,
    ) {
        let _p = crate::profiler::scope("flowfield.prefetch");
        let mut missing: Vec<((NavDomain, u32, u32), FlowJob)> = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for (domain, grid, cell) in wanted {
            let key = (*domain, cell.0, cell.1);
            if !seen.insert(key) {
                continue;
            }
            if let Some((last, _)) = self.entries.get_mut(&key) {
                *last = tick;
            } else {
                missing.push((key, (Arc::clone(grid), *cell)));
            }
        }
        if missing.is_empty() {
            return;
        }
        let fields: Vec<FlowField> = match pool {
            Some(pool) if pool.threads() > 1 && missing.len() > 1 => {
                let items: Arc<Vec<FlowJob>> = Arc::new(missing.iter().map(|(_, j)| j.clone()).collect());
                pool.par_map_ordered_arc(items, Arc::new(|_, (g, c): &FlowJob| FlowField::compute(g, *c)))
            }
            _ => missing.iter().map(|(_, (g, c))| FlowField::compute(g, *c)).collect(),
        };
        let cap = wanted
            .iter()
            .map(|(_, g, _)| Self::capacity_for(g))
            .min()
            .unwrap_or(MAX_FLOW_FIELDS)
            .max(seen.len() * 2);
        for ((key, _), field) in missing.into_iter().zip(fields) {
            self.entries.insert(key, (tick, Arc::new(field)));
        }
        self.evict_to(cap);
    }
}

/// Builds `(land_grid, naval_grid, amphib_grid)` from a parsed [`ScMap`].
#[allow(dead_code)]
pub fn build_domain_grids(
    scmap: &ScMap,
    step: u32,
    max_slope_milli: i64,
) -> (PathGrid, PathGrid, PathGrid) {
    let land_grid = PathGrid::from_scmap(scmap, step, max_slope_milli, false);
    let amphib_grid = PathGrid::from_scmap(scmap, step, max_slope_milli, true);

    let step = step.max(1);
    let width = (scmap.heightmap_width / step).max(1);
    let height = (scmap.heightmap_height / step).max(1);
    let cell_size_milli = (step as i64) * SCALE;

    let mut naval_grid = PathGrid::new(width, height, cell_size_milli);
    if !scmap.water.has_water {
        for cz in 0..height {
            for cx in 0..width {
                naval_grid.set_blocked(cx, cz, true);
            }
        }
    } else {
        let height_scale_micro = (scmap.height_scale as f64 * 1_000_000.0).round() as i64;
        let water_milli = (scmap.water.elevation as f64 * (SCALE as f64)).round() as i64;
        let max_seabed_milli = water_milli.saturating_sub(NAVAL_MIN_DEPTH_MILLI);
        let hm_stride = (scmap.heightmap_width + 1) as usize;

        for cz in 0..height {
            for cx in 0..width {
                let hx = cx.saturating_mul(step).min(scmap.heightmap_width) as usize;
                let hz = cz.saturating_mul(step).min(scmap.heightmap_height) as usize;
                let idx = hz * hm_stride + hx;
                let sample = scmap.heightmap.get(idx).copied().unwrap_or(0) as i64;
                let cell_height_milli = (sample * height_scale_micro) / SCALE;
                if cell_height_milli > max_seabed_milli {
                    naval_grid.set_blocked(cx, cz, true);
                }
            }
        }
    }

    (land_grid, naval_grid, amphib_grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scmap::{ScMapLighting, WaterInfo};

    #[test]
    fn flow_field_routes_around_wall_without_corner_cutting() {
        let mut grid = PathGrid::new(10, 10, 4 * SCALE);
        for z in 0..8 {
            grid.set_blocked(5, z, true);
        }
        let ff = FlowField::compute(&grid, (8, 2));
        // Walk from (2, 2) via flow field directions; must reach (8, 2) without stepping into wall.
        let mut cur = (2u32, 2u32);
        let mut steps = 0;
        while cur != ff.goal_cell && steps < 40 {
            assert!(!grid.is_blocked(cur.0, cur.1));
            let d = ff.dir[(cur.1 * ff.width + cur.0) as usize];
            assert_ne!(d, (0, 0), "cell {:?} must have valid downhill dir", cur);
            cur = ((cur.0 as i32 + d.0 as i32) as u32, (cur.1 as i32 + d.1 as i32) as u32);
            steps += 1;
        }
        assert_eq!(cur, (8, 2));
        assert!(steps > 6, "should walk around the z=0..7 wall");
    }

    #[test]
    fn flow_field_is_bit_deterministic_and_handles_blocked_goal() {
        let mut grid = PathGrid::new(8, 8, SCALE);
        grid.set_blocked(4, 4, true);
        let a = FlowField::compute(&grid, (4, 4));
        let b = FlowField::compute(&grid, (4, 4));
        assert_eq!(a, b);
        assert_ne!(a.goal_cell, (4, 4));
        assert!(!grid.is_blocked(a.goal_cell.0, a.goal_cell.1));
    }

    #[test]
    fn flow_field_cache_evicts_deterministically_at_capacity() {
        let grid = PathGrid::new(16, 16, SCALE);
        let mut cache = FlowFieldCache::new();
        for i in 0..75u32 {
            let gx = ((i % 16) as i64) * SCALE;
            let gz = ((i / 16) as i64) * SCALE;
            cache.get_or_compute(NavDomain::Land, &grid, (gx, gz), i as u64);
        }
        assert_eq!(cache.len(), 75.min(FlowFieldCache::capacity_for(&grid)));
        assert!(!cache.is_empty());
    }

    #[test]
    fn build_domain_grids_separates_land_naval_and_amphibious() {
        // 4x4 quads (5x5 vertices): left half submerged (y=5), right half dry (y=15), water at 10.
        let mut heightmap = vec![0u16; 25];
        for z in 0..5 {
            for x in 0..5 {
                heightmap[z * 5 + x] = if x < 2 { 500 } else { 1500 };
            }
        }
        let scmap = ScMap {
            version_major: 2,
            version_minor: 60,
            map_width: 4.0,
            map_height: 4.0,
            preview_dds: vec![],
            heightmap_width: 4,
            heightmap_height: 4,
            height_scale: 0.01,
            heightmap,
            terrain_shader: String::new(),
            background_path: String::new(),
            sky_cubemap_path: String::new(),
            cubemaps: vec![],
            lighting: ScMapLighting::default(),
            water: WaterInfo {
                has_water: true,
                elevation: 10.0,
                elevation_deep: 8.0,
                elevation_abyss: 4.0,
            },
            terrain_textures: None,
        };
        let (land, naval, amphib) = build_domain_grids(&scmap, 1, 20_000);
        // Cell x=0 is at h=5 (underwater), cell x=3 is at h=15 (land).
        assert!(land.is_blocked(0, 1));
        assert!(!land.is_blocked(3, 1));
        assert!(!naval.is_blocked(0, 1));
        assert!(naval.is_blocked(3, 1));
        assert!(!amphib.is_blocked(0, 1));
        assert!(!amphib.is_blocked(3, 1));
    }
}
