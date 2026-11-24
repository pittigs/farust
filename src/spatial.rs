//! Phase 10 deterministic integer spatial hash grid.
//!
//! Replaces O(N^2) radius scans (target acquisition, AoE splash, unit separation)
//! with fixed-order cell buckets sorted in ascending `UnitId` order so 1-thread
//! and N-thread runs produce bit-identical results.

use crate::sim::SCALE;

/// Spatial cell size in milli-units (16 world meters).
pub const SPATIAL_CELL_MILLI: i64 = 16 * SCALE;

#[derive(Debug, Clone)]
pub struct SpatialGrid {
    width: usize,
    height: usize,
    cell_size: i64,
    cells: Vec<Vec<u32>>,
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
        }
    }

    #[inline]
    fn cell_coords(&self, x: i64, y: i64) -> (usize, usize) {
        let gx = (x.div_euclid(self.cell_size)).clamp(0, self.width as i64 - 1) as usize;
        let gy = (y.div_euclid(self.cell_size)).clamp(0, self.height as i64 - 1) as usize;
        (gx, gy)
    }

    /// Clears and inserts `(unit_id, x, y)` pairs. Callers iterating a `BTreeMap`
    /// populate each bucket in strictly ascending `UnitId` order.
    pub fn rebuild<I>(&mut self, units: I)
    where
        I: IntoIterator<Item = (u32, i64, i64)>,
    {
        for cell in &mut self.cells {
            cell.clear();
        }
        for (id, x, y) in units {
            let (gx, gy) = self.cell_coords(x, y);
            self.cells[gy * self.width + gx].push(id);
        }
    }

    /// Collects all candidate unit IDs in cells overlapping `[cx - r, cx + r] x [cy - r, cy + r]`.
    pub fn query_radius(&self, cx: i64, cy: i64, radius: i64, out: &mut Vec<u32>) {
        out.clear();
        let r = radius.max(0);
        let (x0, y0) = self.cell_coords(cx - r, cy - r);
        let (x1, y1) = self.cell_coords(cx + r, cy + r);
        for gy in y0..=y1 {
            let row_off = gy * self.width;
            for gx in x0..=x1 {
                out.extend_from_slice(&self.cells[row_off + gx]);
            }
        }
    }

    /// Deterministic ascending-`UnitId` query variant.
    pub fn query_radius_sorted(&self, cx: i64, cy: i64, radius: i64, out: &mut Vec<u32>) {
        self.query_radius(cx, cy, radius, out);
        out.sort_unstable();
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
}
