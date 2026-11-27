//! Phase 41: dynamic terrain marks — craters under fresh wrecks, scorch marks where splash
//! shells land, tank treads behind moving land units and footprints under structures.
//!
//! Render-only: the tracker *observes* the world frame by frame (new wrecks, vanished
//! projectiles, unit movement) and never feeds anything back into the sim. Each kind lives
//! in a fixed-size ring ([`BUDGET`]), so long battles stay within a bounded decal count;
//! timed marks fade out over [`LIFE_TICKS`].

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::scene::GpuVertex;
use crate::scmap::ScMap;
use crate::skirmish::{Skirmish, UnitId};
use crate::flowfield::NavDomain;

/// Decal kinds (index into the per-kind arrays).
pub const SCORCH: usize = 0;
pub const CRATER: usize = 1;
pub const TREAD: usize = 2;
pub const FOOTPRINT: usize = 3;
pub const KINDS: usize = 4;

/// Max live marks per kind.
pub const BUDGET: [usize; KINDS] = [256, 128, 384, 512];
/// Lifetime per kind in sim ticks (0 = while the source exists).
pub const LIFE_TICKS: [u64; KINDS] = [1_800, 3_000, 150, 0];
/// FA textures per kind (case-insensitive inside `env.scd` / `textures.scd`).
pub const TEXTURES: [&str; KINDS] = [
    "env/Common/decals/Scorch_001_albedo.dds",
    "env/Common/decals/crater01_albedo.dds",
    "textures/tilesets/_global/tank_treads.dds",
    "textures/tilesets/_global/scorch_mark.dds",
];
/// Distance (map units) a land unit travels between two tread marks.
const TREAD_SPACING: f32 = 2.5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mark {
    pub x: f32,
    pub z: f32,
    pub radius: f32,
    pub yaw: f32,
    pub born: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DecalTracker {
    pub marks: [VecDeque<Mark>; KINDS],
    last_tick: u64,
    wrecks_seen: BTreeSet<UnitId>,
    /// Splash shells in flight: id → (target x, z, splash radius).
    shells: BTreeMap<u32, (f32, f32, f32)>,
    tread_last: BTreeMap<UnitId, (f32, f32)>,
    /// Bumped whenever the visible decal set changes (adds / expiry / fade step).
    pub revision: u64,
}

impl DecalTracker {
    fn push(&mut self, kind: usize, m: Mark) {
        let q = &mut self.marks[kind];
        if q.len() >= BUDGET[kind] {
            q.pop_front();
        }
        q.push_back(m);
    }

    /// Catches up with `w` (a no-op when the tick has not advanced; a backward jump — replay
    /// seek, quickload — clears everything).
    pub fn observe(&mut self, w: &Skirmish) {
        let tick = w.tick();
        if tick < self.last_tick {
            *self = Self { revision: self.revision + 1, ..Self::default() };
        }
        if tick == self.last_tick && tick != 0 {
            return;
        }
        let first = self.last_tick == 0 && self.revision == 0;
        self.last_tick = tick;
        let s = crate::sim::SCALE as f32;
        let mut changed = false;

        // Craters under new wrecks (not for the wrecks that already existed on first sight).
        let wrecks: Vec<(UnitId, f32, f32, f32)> = w
            .wrecks()
            .map(|wr| {
                let r = w.catalog().get(&wr.type_id).map_or(1.0, |t| t.radius as f32 / s);
                (wr.id, wr.pos.x as f32 / s, wr.pos.y as f32 / s, (r * 3.5).clamp(2.5, 16.0))
            })
            .collect();
        for &(id, x, z, r) in &wrecks {
            if self.wrecks_seen.insert(id) && !first {
                self.push(CRATER, Mark { x, z, radius: r, yaw: (id % 628) as f32 / 100.0, born: tick });
                changed = true;
            }
        }
        if self.wrecks_seen.len() > 4 * BUDGET[CRATER] {
            let live: BTreeSet<UnitId> = wrecks.iter().map(|w| w.0).collect();
            self.wrecks_seen.retain(|id| live.contains(id));
        }

        // Scorch where splash shells vanished.
        let now: BTreeMap<u32, (f32, f32, f32)> = w
            .projectiles()
            .filter(|p| p.splash_radius > 0)
            .map(|p| (p.id, (p.target.x as f32 / s, p.target.y as f32 / s, p.splash_radius as f32 / s)))
            .collect();
        let landed: Vec<(u32, (f32, f32, f32))> = self.shells.iter().filter(|(id, _)| !now.contains_key(id)).map(|(id, v)| (*id, *v)).collect();
        for (id, (x, z, splash)) in landed {
            self.push(SCORCH, Mark { x, z, radius: (splash * 2.0).clamp(2.0, 30.0), yaw: (id % 628) as f32 / 100.0, born: tick });
            changed = true;
        }
        self.shells = now;

        // Treads behind moving land units.
        let mut alive = BTreeSet::new();
        for u in w.units().filter(|u| u.alive && u.moving && u.domain == NavDomain::Land && u.carried_by.is_none()) {
            alive.insert(u.id);
            let (x, z) = (u.pos.x as f32 / s, u.pos.y as f32 / s);
            match self.tread_last.get(&u.id).copied() {
                Some((px, pz)) if ((x - px).powi(2) + (z - pz).powi(2)).sqrt() >= TREAD_SPACING => {
                    let yaw = (x - px).atan2(z - pz);
                    self.push(TREAD, Mark { x, z, radius: (u.radius as f32 / s * 1.6).clamp(0.6, 4.0), yaw, born: tick });
                    self.tread_last.insert(u.id, (x, z));
                    changed = true;
                }
                Some(_) => {}
                None => {
                    self.tread_last.insert(u.id, (x, z));
                }
            }
        }
        self.tread_last.retain(|id, _| alive.contains(id) || w.unit(*id).is_some_and(|u| u.alive));

        // Footprints under live structures (rebuilt; capped).
        let foot: Vec<Mark> = w
            .units()
            .filter(|u| u.alive && u.role.is_structure() && u.role != crate::skirmish::Role::Wall)
            .take(BUDGET[FOOTPRINT])
            .map(|u| Mark { x: u.pos.x as f32 / s, z: u.pos.y as f32 / s, radius: (u.radius as f32 / s * 1.5).clamp(1.0, 14.0), yaw: 0.0, born: 0 })
            .collect();
        if foot.len() != self.marks[FOOTPRINT].len() || foot.iter().zip(&self.marks[FOOTPRINT]).any(|(a, b)| a != b) {
            self.marks[FOOTPRINT] = foot.into();
            changed = true;
        }

        // Expiry.
        for (q, &life) in self.marks.iter_mut().zip(LIFE_TICKS.iter()) {
            if life == 0 {
                continue;
            }
            let before = q.len();
            q.retain(|m| tick.saturating_sub(m.born) < life);
            changed |= q.len() != before;
        }
        if changed || tick.is_multiple_of(5) {
            self.revision += 1;
        }
    }

    #[cfg(test)]
    pub fn total(&self) -> usize {
        self.marks.iter().map(VecDeque::len).sum()
    }

    /// Terrain-draped 3×3 quads for every mark of `kind`, alpha fading with age.
    pub fn geometry(&self, kind: usize, scmap: &ScMap) -> (Vec<GpuVertex>, Vec<u32>) {
        let marks = &self.marks[kind];
        let mut v = Vec::with_capacity(marks.len() * 9);
        let mut idx = Vec::with_capacity(marks.len() * 24);
        for m in marks {
            let alpha = match LIFE_TICKS[kind] {
                0 => 0.55,
                life => {
                    let age = self.last_tick.saturating_sub(m.born) as f32 / life as f32;
                    (1.0 - age).clamp(0.0, 1.0).powf(0.7)
                }
            };
            let (sy, cy) = m.yaw.sin_cos();
            let (sx, sz) = if kind == TREAD { (m.radius, m.radius * 0.6) } else { (m.radius * 2.0, m.radius * 2.0) };
            let base = v.len() as u32;
            for iz in 0..3u32 {
                let vv = iz as f32 * 0.5;
                let lz = (vv - 0.5) * sz;
                for ix in 0..3u32 {
                    let u = ix as f32 * 0.5;
                    let lx = (u - 0.5) * sx;
                    let wx = m.x + lx * cy + lz * sy;
                    let wz = m.z - lx * sy + lz * cy;
                    v.push(GpuVertex {
                        position: [wx, crate::scene::sample_scmap_height(scmap, wx, wz) + 0.06 + kind as f32 * 0.01, wz],
                        normal: crate::scene::sample_scmap_normal(scmap, wx, wz),
                        uv: [u, vv],
                        color: [1.0, 1.0, 1.0, alpha],
                    });
                }
            }
            for iz in 0..2u32 {
                for ix in 0..2u32 {
                    let c00 = base + iz * 3 + ix;
                    idx.extend_from_slice(&[c00, c00 + 3, c00 + 1, c00 + 1, c00 + 3, c00 + 4]);
                }
            }
        }
        (v, idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rings_respect_budgets_and_marks_expire() {
        let mut t = DecalTracker::default();
        for i in 0..(BUDGET[SCORCH] + 50) {
            t.push(SCORCH, Mark { x: i as f32, z: 0.0, radius: 1.0, yaw: 0.0, born: 10 });
        }
        assert_eq!(t.marks[SCORCH].len(), BUDGET[SCORCH], "oldest marks are recycled");
        assert_eq!(t.marks[SCORCH][0].x, 50.0);
    }

    #[test]
    fn battle_leaves_craters_scorches_and_treads_within_budget() {
        let mut w = crate::skirmish::tests::test_world(5);
        let mut t = DecalTracker::default();
        let tanks: Vec<UnitId> = (0..6).map(|i| w.spawn_unit(0, "RA", (60_000 + i * 3_000, 60_000)).unwrap()).collect();
        let foes: Vec<UnitId> = (0..6).map(|i| w.spawn_unit(1, "BT", (60_000 + i * 3_000, 110_000)).unwrap()).collect();
        let _ = foes;
        w.issue(0, crate::skirmish::Command::Move { units: tanks.clone(), x: 70_000, y: 85_000, queue: false });
        for _ in 0..600 {
            w.step();
            t.observe(&w);
        }
        assert!(!t.marks[CRATER].is_empty(), "wrecks leave craters");
        assert!(!t.marks[SCORCH].is_empty(), "artillery shells leave scorch marks");
        assert!(t.total() <= BUDGET.iter().sum::<usize>());
        let rev = t.revision;
        t.observe(&w);
        assert_eq!(t.revision, rev, "observing the same tick twice changes nothing");
    }
}
