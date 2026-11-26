//! Phase 10 deterministic integer spatial hash grid.
//!
//! Replaces O(N^2) radius scans (target acquisition, AoE splash, unit separation)
//! with fixed-order cell buckets sorted in ascending `UnitId` order so 1-thread
//! and N-thread runs produce bit-identical results.

use crate::sim::SCALE;

/// Spatial cell size in milli-units (6 world meters for tight unit queries).
pub const SPATIAL_CELL_MILLI: i64 = 6 * SCALE;

/// Cell buckets in insertion (ascending `UnitId`) order plus optional per-team layers so
/// "enemies of team T" queries skip friendly units entirely. Rebuilds clear only the cells
/// touched by the previous rebuild, so their cost is O(units), independent of the map size.
#[derive(Debug, Clone)]
pub struct SpatialGrid {
    width: usize,
    height: usize,
    cell_size: i64,
    cells: Vec<Vec<u32>>,
    touched: Vec<u32>,
    /// Distinct teams present after the last [`Self::rebuild_with_teams`] (ascending).
    teams: Vec<u8>,
    /// Per team value: lazily allocated cell grid + its touched cells.
    team_layers: Vec<Option<(Vec<Vec<u32>>, Vec<u32>)>>,
}

impl SpatialGrid {
    pub fn new(map_width_milli: i64, map_height_milli: i64, cell_size: i64) -> Self {
        let cs = cell_size.max(SCALE);
        let width = ((map_width_milli.max(cs) + cs - 1) / cs).max(1) as usize;
        let height = ((map_height_milli.max(cs) + cs - 1) / cs).max(1) as usize;
        Self {
            width,
            height,
            cell_size: cs,
            cells: vec![Vec::new(); width * height],
            touched: Vec::new(),
            teams: Vec::new(),
            team_layers: Vec::new(),
        }
    }

    #[inline]
    fn cell_coords(&self, x: i64, y: i64) -> (usize, usize) {
        let gx = (x.div_euclid(self.cell_size)).clamp(0, self.width as i64 - 1) as usize;
        let gy = (y.div_euclid(self.cell_size)).clamp(0, self.height as i64 - 1) as usize;
        (gx, gy)
    }

    fn clear_all(&mut self) {
        for c in self.touched.drain(..) {
            self.cells[c as usize].clear();
        }
        for (cells, touched) in self.team_layers.iter_mut().flatten() {
            for c in touched.drain(..) {
                cells[c as usize].clear();
            }
        }
        self.teams.clear();
    }

    /// Clears and inserts `(unit_id, x, y)` pairs. Callers iterating a `BTreeMap`
    /// populate each bucket in strictly ascending `UnitId` order.
    #[allow(dead_code)]
    pub fn rebuild<I>(&mut self, units: I)
    where
        I: IntoIterator<Item = (u32, i64, i64)>,
    {
        self.clear_all();
        for (id, x, y) in units {
            let (gx, gy) = self.cell_coords(x, y);
            let c = gy * self.width + gx;
            if self.cells[c].is_empty() {
                self.touched.push(c as u32);
            }
            self.cells[c].push(id);
        }
    }

    /// Like [`Self::rebuild`] but also fills per-team layers for
    /// [`Self::for_each_enemy_in_radius`]. Bucket order stays the insertion order.
    pub fn rebuild_with_teams<I>(&mut self, units: I)
    where
        I: IntoIterator<Item = (u32, i64, i64, u8)>,
    {
        self.clear_all();
        let ncells = self.width * self.height;
        for (id, x, y, team) in units {
            let (gx, gy) = self.cell_coords(x, y);
            let c = gy * self.width + gx;
            if self.cells[c].is_empty() {
                self.touched.push(c as u32);
            }
            self.cells[c].push(id);
            let t = team as usize;
            if self.team_layers.len() <= t {
                self.team_layers.resize_with(t + 1, || None);
            }
            let (cells, touched) = self.team_layers[t].get_or_insert_with(|| (vec![Vec::new(); ncells], Vec::new()));
            if cells[c].is_empty() {
                touched.push(c as u32);
            }
            cells[c].push(id);
            if let Err(pos) = self.teams.binary_search(&team) {
                self.teams.insert(pos, team);
            }
        }
    }

    /// Collects all candidate unit IDs in cells overlapping `[cx - r, cx + r] x [cy - r, cy + r]`.
    #[allow(dead_code)]
    pub fn query_radius(&self, cx: i64, cy: i64, radius: i64, out: &mut Vec<u32>) {
        out.clear();
        self.for_each_in_radius(cx, cy, radius, |id| out.push(id));
    }

    /// Deterministic ascending-`UnitId` query variant.
    #[allow(dead_code)]
    pub fn query_radius_sorted(&self, cx: i64, cy: i64, radius: i64, out: &mut Vec<u32>) {
        self.query_radius(cx, cy, radius, out);
        out.sort_unstable();
    }

    /// Zero-allocation deterministic cell sweep invoking `f` for every entry in cells
    /// overlapping `[cx - r, cx + r] x [cy - r, cy + r]` (row-major cells, insertion order).
    #[inline]
    pub fn for_each_in_radius<F: FnMut(u32)>(&self, cx: i64, cy: i64, radius: i64, f: F) {
        Self::sweep(&self.cells, self.width, self.cell_rect(cx, cy, radius), f);
    }

    #[inline]
    fn cell_rect(&self, cx: i64, cy: i64, radius: i64) -> (usize, usize, usize, usize) {
        let r = radius.max(0);
        let (x0, y0) = self.cell_coords(cx - r, cy - r);
        let (x1, y1) = self.cell_coords(cx + r, cy + r);
        (x0, y0, x1, y1)
    }

    #[inline]
    fn sweep<F: FnMut(u32)>(cells: &[Vec<u32>], width: usize, (x0, y0, x1, y1): (usize, usize, usize, usize), mut f: F) {
        for gy in y0..=y1 {
            let row = &cells[gy * width + x0..=gy * width + x1];
            for cell in row {
                for &val in cell {
                    f(val);
                }
            }
        }
    }

    /// Visits every entry *not* on `team` in the query square. Visiting order differs from
    /// [`Self::for_each_in_radius`] (team-major), so callers must be order-independent (e.g. a
    /// min with an id tie-break). Without team layers it falls back to the full sweep (callers
    /// still filter by team themselves).
    #[inline]
    pub fn for_each_enemy_in_radius<F: FnMut(u32)>(&self, cx: i64, cy: i64, radius: i64, team: u8, mut f: F) {
        if self.teams.is_empty() {
            return self.for_each_in_radius(cx, cy, radius, f);
        }
        let rect = self.cell_rect(cx, cy, radius);
        for &t in &self.teams {
            if t == team {
                continue;
            }
            if let Some(Some((cells, _))) = self.team_layers.get(t as usize) {
                Self::sweep(cells, self.width, rect, &mut f);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_grid_matches_bruteforce_exactly() {
        let mut grid = SpatialGrid::new(512 * SCALE, 512 * SCALE, SPATIAL_CELL_MILLI);
        let mut pts = Vec::new();
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for id in 1..=500u32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((state >> 32) % 512) as i64 * SCALE;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let y = ((state >> 32) % 512) as i64 * SCALE;
            pts.push((id, x, y));
        }
        grid.rebuild(pts.iter().copied());

        let qx = 240 * SCALE;
        let qy = 260 * SCALE;
        let r = 45 * SCALE;
        let r2 = (r as i128) * (r as i128);

        let mut cand = Vec::new();
        grid.query_radius_sorted(qx, qy, r, &mut cand);
        let got: Vec<u32> = cand
            .into_iter()
            .filter(|id| {
                let (_, x, y) = pts[(*id - 1) as usize];
                let dx = (x - qx) as i128;
                let dy = (y - qy) as i128;
                dx * dx + dy * dy <= r2
            })
            .collect();

        let expected: Vec<u32> = pts
            .iter()
            .filter(|&&(_, x, y)| {
                let dx = (x - qx) as i128;
                let dy = (y - qy) as i128;
                dx * dx + dy * dy <= r2
            })
            .map(|&(id, _, _)| id)
            .collect();

        assert_eq!(got, expected);
        assert!(!got.is_empty());
    }

    #[test]
    fn team_layers_visit_exactly_the_enemies() {
        let mut grid = SpatialGrid::new(256 * SCALE, 256 * SCALE, SPATIAL_CELL_MILLI);
        let mut pts = Vec::new();
        let mut state: u64 = 42;
        for id in 0..400u32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((state >> 33) % 256) as i64 * SCALE;
            let y = ((state >> 13) % 256) as i64 * SCALE;
            pts.push((id, x, y, (id % 3) as u8 * 2));
        }
        grid.rebuild_with_teams(pts.iter().copied());
        for team in [0u8, 2, 4, 9] {
            let mut all = Vec::new();
            grid.for_each_in_radius(100 * SCALE, 120 * SCALE, 40 * SCALE, |i| all.push(i));
            let mut want: Vec<u32> = all.into_iter().filter(|&i| pts[i as usize].3 != team).collect();
            let mut got = Vec::new();
            grid.for_each_enemy_in_radius(100 * SCALE, 120 * SCALE, 40 * SCALE, team, |i| got.push(i));
            want.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, want);
        }
        // Row-major / insertion order within the combined sweep is preserved.
        let mut v = Vec::new();
        grid.query_radius(0, 0, 1_000_000, &mut v);
        assert_eq!(v.len(), 400);
    }
}
