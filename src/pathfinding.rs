//! Phase 5 integer-only deterministic A* pathfinding and parallel batch solver.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::jobs::par_map_ordered;
use crate::scmap::ScMap;

pub const CARDINAL_COST: u64 = 1000;
pub const DIAGONAL_COST: u64 = 1414;

#[derive(Debug, Clone)]
pub struct PathGrid {
    pub width: u32,
    pub height: u32,
    pub cell_size_milli: i64,
    blocked: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathRequest {
    pub id: u64,
    pub start: (i64, i64),
    pub goal: (i64, i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct PathResult {
    pub request_id: u64,
    pub waypoints: Vec<(i64, i64)>,
    pub cost: u64,
    pub nodes_expanded: u32,
    pub found: bool,
}

impl PathGrid {
    pub fn new(width: u32, height: u32, cell_size_milli: i64) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let cs = cell_size_milli.max(1);
        Self {
            width: w,
            height: h,
            cell_size_milli: cs,
            blocked: vec![0; (w as usize) * (h as usize)],
        }
    }

    pub fn from_scmap(scmap: &ScMap, step: u32, max_slope_milli: i64, amphibious: bool) -> Self {
        let step = step.max(1);
        let width = (scmap.heightmap_width / step).max(1);
        let height = (scmap.heightmap_height / step).max(1);
        let cell_size_milli = (step as i64) * 1000;

        let mut grid = Self::new(width, height, cell_size_milli);

        let height_scale_micro = (scmap.height_scale as f64 * 1_000_000.0).round() as i64;
        let water_milli = if scmap.water.has_water {
            (scmap.water.elevation as f64 * 1000.0).round() as i64
        } else {
            i64::MIN
        };

        let hm_stride = (scmap.heightmap_width + 1) as usize;
        let cell_height_milli = |cx: u32, cz: u32| -> i64 {
            let hx = (cx * step).min(scmap.heightmap_width) as usize;
            let hz = (cz * step).min(scmap.heightmap_height) as usize;
            let idx = hz * hm_stride + hx;
            let sample = scmap.heightmap.get(idx).copied().unwrap_or(0) as i64;
            (sample * height_scale_micro) / 1000
        };

        for cz in 0..height {
            for cx in 0..width {
                let h0 = cell_height_milli(cx, cz);
                let hx = cell_height_milli(cx + 1, cz);
                let hz = cell_height_milli(cx, cz + 1);

                let under_water = !amphibious && scmap.water.has_water && h0 < water_milli;
                let slope_x = (hx - h0).abs();
                let slope_z = (hz - h0).abs();
                let steep = slope_x.max(slope_z) > max_slope_milli;

                if under_water || steep {
                    grid.set_blocked(cx, cz, true);
                }
            }
        }

        grid
    }

    pub fn set_blocked(&mut self, cx: u32, cz: u32, is_blocked: bool) {
        if cx < self.width && cz < self.height {
            let idx = (cz as usize) * (self.width as usize) + (cx as usize);
            self.blocked[idx] = u8::from(is_blocked);
        }
    }

    pub fn is_blocked(&self, cx: u32, cz: u32) -> bool {
        if cx >= self.width || cz >= self.height {
            return true;
        }
        let idx = (cz as usize) * (self.width as usize) + (cx as usize);
        self.blocked[idx] != 0
    }

    pub fn blocked_count(&self) -> usize {
        self.blocked.iter().filter(|&&b| b != 0).count()
    }

    pub fn world_to_cell(&self, x_milli: i64, z_milli: i64) -> (u32, u32) {
        let cs = self.cell_size_milli.max(1);
        let cx = (x_milli / cs).clamp(0, (self.width as i64) - 1) as u32;
        let cz = (z_milli / cs).clamp(0, (self.height as i64) - 1) as u32;
        (cx, cz)
    }

    pub fn cell_to_world(&self, cx: u32, cz: u32) -> (i64, i64) {
        let half = self.cell_size_milli / 2;
        (
            (cx as i64) * self.cell_size_milli + half,
            (cz as i64) * self.cell_size_milli + half,
        )
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct OpenNode {
    f_cost: u64,
    g_cost: u64,
    cell_idx: u32,
}

impl Ord for OpenNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_cost
            .cmp(&self.f_cost)
            .then_with(|| other.g_cost.cmp(&self.g_cost))
            .then_with(|| other.cell_idx.cmp(&self.cell_idx))
    }
}

impl PartialOrd for OpenNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn octile_h(ax: u32, az: u32, bx: u32, bz: u32) -> u64 {
    let dx = ax.abs_diff(bx) as u64;
    let dz = az.abs_diff(bz) as u64;
    let hi = dx.max(dz);
    let lo = dx.min(dz);
    CARDINAL_COST * hi + (DIAGONAL_COST - CARDINAL_COST) * lo
}

pub fn find_path(grid: &PathGrid, req: &PathRequest) -> PathResult {
    let start_cell = grid.world_to_cell(req.start.0, req.start.1);
    let goal_cell = grid.world_to_cell(req.goal.0, req.goal.1);

    if start_cell == goal_cell {
        let ok = !grid.is_blocked(goal_cell.0, goal_cell.1);
        return PathResult {
            request_id: req.id,
            waypoints: if ok { vec![req.goal] } else { Vec::new() },
            cost: 0,
            nodes_expanded: 1,
            found: ok,
        };
    }

    if grid.is_blocked(start_cell.0, start_cell.1) || grid.is_blocked(goal_cell.0, goal_cell.1) {
        return PathResult {
            request_id: req.id,
            waypoints: Vec::new(),
            cost: 0,
            nodes_expanded: 0,
            found: false,
        };
    }

    let total_cells = (grid.width as usize) * (grid.height as usize);
    let mut g_scores = vec![u64::MAX; total_cells];
    let mut parent = vec![u32::MAX; total_cells];
    let mut closed = vec![0u8; total_cells];

    let start_idx = start_cell.1 * grid.width + start_cell.0;
    let goal_idx = goal_cell.1 * grid.width + goal_cell.0;

    g_scores[start_idx as usize] = 0;

    let mut open = BinaryHeap::new();
    open.push(OpenNode {
        f_cost: octile_h(start_cell.0, start_cell.1, goal_cell.0, goal_cell.1),
        g_cost: 0,
        cell_idx: start_idx,
    });

    const NEIGHBORS: [(i32, i32, u64); 8] = [
        (0, -1, CARDINAL_COST),
        (1, 0, CARDINAL_COST),
        (0, 1, CARDINAL_COST),
        (-1, 0, CARDINAL_COST),
        (1, -1, DIAGONAL_COST),
        (1, 1, DIAGONAL_COST),
        (-1, 1, DIAGONAL_COST),
        (-1, -1, DIAGONAL_COST),
    ];

    let mut nodes_expanded = 0u32;

    while let Some(curr) = open.pop() {
        let c_usize = curr.cell_idx as usize;
        if closed[c_usize] != 0 {
            continue;
        }
        closed[c_usize] = 1;
        nodes_expanded += 1;

        if curr.cell_idx == goal_idx {
            let mut cells = Vec::new();
            let mut walk = goal_idx;
            while walk != start_idx {
                let wx = walk % grid.width;
                let wz = walk / grid.width;
                cells.push((wx, wz));
                walk = parent[walk as usize];
            }
            cells.reverse();

            // Collinear waypoint pruning
            let mut pruned_cells: Vec<(u32, u32)> = Vec::with_capacity(cells.len());
            let mut prev = start_cell;
            for i in 0..cells.len() {
                let curr_c = cells[i];
                if i + 1 < cells.len() {
                    let next_c = cells[i + 1];
                    let dx1 = (curr_c.0 as i64) - (prev.0 as i64);
                    let dz1 = (curr_c.1 as i64) - (prev.1 as i64);
                    let dx2 = (next_c.0 as i64) - (curr_c.0 as i64);
                    let dz2 = (next_c.1 as i64) - (curr_c.1 as i64);
                    if dx1 == dx2 && dz1 == dz2 {
                        prev = curr_c;
                        continue;
                    }
                }
                pruned_cells.push(curr_c);
                prev = curr_c;
            }

            let mut waypoints: Vec<(i64, i64)> = pruned_cells
                .iter()
                .map(|&(cx, cz)| grid.cell_to_world(cx, cz))
                .collect();
            if let Some(last) = waypoints.last_mut() {
                *last = req.goal;
            }

            return PathResult {
                request_id: req.id,
                waypoints,
                cost: curr.g_cost,
                nodes_expanded,
                found: true,
            };
        }

        let cx = (curr.cell_idx % grid.width) as i32;
        let cz = (curr.cell_idx / grid.width) as i32;

        for &(dx, dz, step_cost) in &NEIGHBORS {
            let nx = cx + dx;
            let nz = cz + dz;
            if nx < 0 || nz < 0 || nx >= grid.width as i32 || nz >= grid.height as i32 {
                continue;
            }
            let nux = nx as u32;
            let nuz = nz as u32;
            if grid.is_blocked(nux, nuz) {
                continue;
            }
            // Prevent corner cutting on diagonal moves
            if dx != 0
                && dz != 0
                && (grid.is_blocked(cx as u32, nuz) || grid.is_blocked(nux, cz as u32))
            {
                continue;
            }

            let n_idx = nuz * grid.width + nux;
            let n_usize = n_idx as usize;
            if closed[n_usize] != 0 {
                continue;
            }

            let tentative_g = curr.g_cost + step_cost;
            if tentative_g < g_scores[n_usize] {
                g_scores[n_usize] = tentative_g;
                parent[n_usize] = curr.cell_idx;
                let h = octile_h(nux, nuz, goal_cell.0, goal_cell.1);
                open.push(OpenNode {
                    f_cost: tentative_g + h,
                    g_cost: tentative_g,
                    cell_idx: n_idx,
                });
            }
        }
    }

    PathResult {
        request_id: req.id,
        waypoints: Vec::new(),
        cost: 0,
        nodes_expanded,
        found: false,
    }
}

pub fn solve_paths_batch(
    grid: &PathGrid,
    requests: &[PathRequest],
    threads: usize,
) -> Vec<PathResult> {
    par_map_ordered(requests, threads, |_, req| find_path(grid, req))
}

pub fn path_batch_hash(results: &[PathResult]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mix = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    };
    for r in results {
        mix(&mut h, &r.request_id.to_le_bytes());
        mix(&mut h, &[u8::from(r.found)]);
        mix(&mut h, &r.cost.to_le_bytes());
        mix(&mut h, &r.nodes_expanded.to_le_bytes());
        mix(&mut h, &(r.waypoints.len() as u64).to_le_bytes());
        for &(wx, wz) in &r.waypoints {
            mix(&mut h, &wx.to_le_bytes());
            mix(&mut h, &wz.to_le_bytes());
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_straight_path_and_pruning() {
        let grid = PathGrid::new(16, 16, 1000);
        let req = PathRequest {
            id: 1,
            start: (500, 500),
            goal: (10_500, 500),
        };
        let res = find_path(&grid, &req);
        assert!(res.found);
        assert_eq!(res.cost, 10 * CARDINAL_COST);
        assert_eq!(res.waypoints, vec![(10_500, 500)]);
    }

    #[test]
    fn test_wall_detour_and_no_corner_cutting() {
        let mut grid = PathGrid::new(10, 10, 1000);
        for z in 0..8 {
            grid.set_blocked(5, z, true);
        }
        let req = PathRequest {
            id: 2,
            start: (2500, 2500),
            goal: (7500, 2500),
        };
        let res = find_path(&grid, &req);
        assert!(res.found);
        assert!(res.cost > 5 * CARDINAL_COST);
        assert!(res.waypoints.len() >= 2);
        assert_eq!(*res.waypoints.last().unwrap(), req.goal);
    }

    #[test]
    fn test_unreachable_returns_not_found() {
        let mut grid = PathGrid::new(8, 8, 1000);
        for z in 0..8 {
            grid.set_blocked(4, z, true);
        }
        let req = PathRequest {
            id: 3,
            start: (1500, 1500),
            goal: (6500, 1500),
        };
        let res = find_path(&grid, &req);
        assert!(!res.found);
        assert!(res.waypoints.is_empty());
    }

    #[test]
    fn test_parallel_batch_determinism() {
        let mut grid = PathGrid::new(32, 32, 1000);
        for z in 4..28 {
            if z % 5 != 0 {
                grid.set_blocked(16, z, true);
            }
        }
        let requests: Vec<PathRequest> = (0..32)
            .map(|i| PathRequest {
                id: i as u64,
                start: (1500, (i as i64) * 1000 + 500),
                goal: (29_500, ((31 - i) as i64) * 1000 + 500),
            })
            .collect();

        let single = solve_paths_batch(&grid, &requests, 1);
        let multi = solve_paths_batch(&grid, &requests, 4);
        assert_eq!(single, multi);
        assert_eq!(path_batch_hash(&single), path_batch_hash(&multi));
    }
}
