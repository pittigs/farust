//! Phase 29: integer terrain heightfield for the deterministic sim.
//!
//! Heights are quantized once from the `.scmap` heightmap (`u16` samples × `height_scale`)
//! into milli-units; every query is integer bilinear interpolation, so line-of-fire and
//! projectile/terrain collisions are bit-identical across threads and platforms.

use crate::scmap::ScMap;

/// Terrain heights in milli-units on a regular sample grid.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HeightField {
    /// Cells per axis (samples per axis minus one); 0 = flat field at height 0.
    cells_x: u32,
    cells_z: u32,
    /// Distance between samples in milli-units.
    spacing: i64,
    heights: Vec<i64>,
}

/// Step between terrain samples along a line-of-fire test (milli-units).
pub const LOS_STEP: i64 = 1_000;

impl HeightField {
    /// A flat field at height 0 (synthetic maps and tests).
    pub fn flat() -> Self {
        Self::default()
    }

    /// Builds a field from explicit samples (`(cells_x + 1) * (cells_z + 1)`, row-major).
    pub fn from_samples(cells_x: u32, cells_z: u32, spacing: i64, heights: Vec<i64>) -> Self {
        assert_eq!(heights.len(), (cells_x as usize + 1) * (cells_z as usize + 1));
        Self { cells_x, cells_z, spacing: spacing.max(1), heights }
    }

    pub fn from_scmap(scmap: &ScMap) -> Self {
        let (w, h) = (scmap.heightmap_width, scmap.heightmap_height);
        let n = (w as usize + 1) * (h as usize + 1);
        if w == 0 || h == 0 || scmap.heightmap.len() < n {
            return Self::flat();
        }
        // Same quantization as `PathGrid::from_scmap`.
        let scale_micro = (scmap.height_scale as f64 * 1_000_000.0).round() as i64;
        let spacing = ((scmap.map_width as f64 * 1000.0) / w as f64).round() as i64;
        let heights = scmap.heightmap[..n].iter().map(|&s| (s as i64 * scale_micro) / 1000).collect();
        Self::from_samples(w, h, spacing, heights)
    }

    pub fn is_flat(&self) -> bool {
        self.heights.is_empty()
    }

    fn sample(&self, ix: i64, iz: i64) -> i64 {
        let ix = ix.clamp(0, self.cells_x as i64) as usize;
        let iz = iz.clamp(0, self.cells_z as i64) as usize;
        self.heights[iz * (self.cells_x as usize + 1) + ix]
    }

    /// Ground height (milli) at world position `(x, z)` (milli), bilinear, clamped to the map.
    pub fn height_at(&self, x: i64, z: i64) -> i64 {
        if self.is_flat() {
            return 0;
        }
        let s = self.spacing;
        let max_x = self.cells_x as i64 * s;
        let max_z = self.cells_z as i64 * s;
        let (x, z) = (x.clamp(0, max_x), z.clamp(0, max_z));
        let (ix, iz) = (x / s, z / s);
        let (fx, fz) = ((x - ix * s) as i128, (z - iz * s) as i128);
        let s = s as i128;
        let h00 = self.sample(ix, iz) as i128;
        let h10 = self.sample(ix + 1, iz) as i128;
        let h01 = self.sample(ix, iz + 1) as i128;
        let h11 = self.sample(ix + 1, iz + 1) as i128;
        let top = h00 * (s - fx) + h10 * fx;
        let bot = h01 * (s - fx) + h11 * fx;
        ((top * (s - fz) + bot * fz) / (s * s)) as i64
    }

    /// First point on the 3-D segment `a → b` (milli) that dips more than `clearance` below
    /// the terrain, sampled every [`LOS_STEP`]. The first `skip_start` and last `skip_end`
    /// milli-units of the segment's ground distance are not tested (muzzle / impact zone).
    pub fn segment_hit(
        &self,
        a: (i64, i64, i64),
        b: (i64, i64, i64),
        clearance: i64,
        skip_start: i64,
        skip_end: i64,
    ) -> Option<(i64, i64, i64)> {
        if self.is_flat() {
            return None;
        }
        let (dx, dy, dz) = ((b.0 - a.0) as i128, (b.1 - a.1) as i128, (b.2 - a.2) as i128);
        let len = crate::skirmish::isqrt(dx * dx + dy * dy);
        if len <= skip_start + skip_end {
            return None;
        }
        let n = (len + LOS_STEP - 1) / LOS_STEP;
        let len128 = len as i128;
        for i in 1..=n {
            let t = i as i128;
            let along = (len128 * t / n as i128) as i64;
            if along < skip_start || along > len - skip_end {
                continue;
            }
            let p = (
                a.0 + (dx * t / n as i128) as i64,
                a.1 + (dy * t / n as i128) as i64,
                a.2 + (dz * t / n as i128) as i64,
            );
            let g = self.height_at(p.0, p.1);
            if p.2 + clearance < g {
                return Some((p.0, p.1, g));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 8×8 cells of 1 m with a 10 m-high ridge along x = 4.
    fn ridge() -> HeightField {
        let mut h = vec![0i64; 81];
        for z in 0..9 {
            h[z * 9 + 4] = 10_000;
        }
        HeightField::from_samples(8, 8, 1_000, h)
    }

    #[test]
    fn bilinear_heights_and_clamping() {
        let f = ridge();
        assert_eq!(f.height_at(4_000, 2_000), 10_000);
        assert_eq!(f.height_at(3_500, 2_000), 5_000);
        assert_eq!(f.height_at(0, 0), 0);
        assert_eq!(f.height_at(-50_000, 99_000), 0, "clamped to the map edge");
        assert_eq!(HeightField::flat().height_at(123, 456), 0);
    }

    #[test]
    fn segment_over_ridge_hits_it_and_high_shot_clears() {
        let f = ridge();
        let hit = f.segment_hit((1_000, 3_000, 1_000), (7_000, 3_000, 1_000), 250, 500, 500);
        let (x, _, z) = hit.expect("low shot hits the ridge");
        assert!((3_000..=4_000).contains(&x) && z > 1_250, "hit on the near slope at x={x}");
        assert!(f.segment_hit((1_000, 3_000, 12_000), (7_000, 3_000, 12_000), 250, 500, 500).is_none());
        assert!(HeightField::flat().segment_hit((0, 0, -5), (9_000, 0, -5), 0, 0, 0).is_none());
    }
}
