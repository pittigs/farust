//! Phase 5 parallel render-frame preparation from a read-only simulation snapshot.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::Arc;

use crate::camera::OrbitCamera;
use crate::jobs::{WorkerPool, par_map_ordered};
use crate::scm::ScmMesh;
use crate::viewer::{GpuMeshTemplate, GpuSceneResources, GpuUnitInstance};
use crate::scmap::ScMap;
use crate::sim::SimUnitSnapshot;

const FNV_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

#[derive(Default)]
struct U32Hasher(u64);

impl Hasher for U32Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut v = 0u64;
        for &b in bytes {
            v = (v << 8) | u64::from(b);
        }
        self.0 = v.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = u64::from(i).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type U32Map<V> = HashMap<u32, V, BuildHasherDefault<U32Hasher>>;

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct RenderInstance {
    pub unit_id: u32,
    pub model_matrix: [[f32; 4]; 4],
    pub world_pos: [f32; 3],
    pub health_ratio: f32,
    pub lod: u8,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct PreparedFrame {
    pub instances: Vec<RenderInstance>,
    pub culled_count: usize,
    pub lod_counts: [usize; 3],
    pub digest_fnv1a: u64,
}

#[inline]
fn fnv1a_extend(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn evaluate_unit_instance(
    u: &SimUnitSnapshot,
    scmap: Option<&ScMap>,
    view_proj: &[[f32; 4]; 4],
    camera_eye: &[f32; 3],
    scale: f32,
) -> Option<RenderInstance> {
    let wx = u.x as f32 / scale;
    let wz = u.y as f32 / scale;
    let wy = if let Some(map) = scmap {
        let gx = (wx.round() as i64).clamp(0, map.heightmap_width as i64) as u32;
        let gz = (wz.round() as i64).clamp(0, map.heightmap_height as i64) as u32;
        let terrain_y = map.height_at(gx, gz).unwrap_or(0.0);
        if map.water.has_water {
            terrain_y.max(map.water.elevation)
        } else {
            terrain_y
        }
    } else {
        0.0
    };

    let clip_row = |r: usize| -> f32 {
        view_proj[0][r] * wx + view_proj[1][r] * wy + view_proj[2][r] * wz + view_proj[3][r]
    };

    let clip_x = clip_row(0);
    let clip_y = clip_row(1);
    let clip_z = clip_row(2);
    let clip_w = clip_row(3);

    let m = 4.0_f32;
    let visible = clip_w > 0.0
        && clip_x >= -clip_w - m
        && clip_x <= clip_w + m
        && clip_y >= -clip_w - m
        && clip_y <= clip_w + m
        && clip_z >= -m
        && clip_z <= clip_w + m;

    if !visible {
        return None;
    }

    let dx = wx - camera_eye[0];
    let dy = wy - camera_eye[1];
    let dz = wz - camera_eye[2];
    let dist_sq = dx * dx + dy * dy + dz * dz;

    let lod = if dist_sq < 80.0 * 80.0 {
        0
    } else if dist_sq < 260.0 * 260.0 {
        1
    } else {
        2
    };

    let health_ratio = if u.max_hp > 0 {
        (u.hp as f32 / u.max_hp as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let model_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [wx, wy, wz, 1.0],
    ];

    Some(RenderInstance {
        unit_id: u.id,
        model_matrix,
        world_pos: [wx, wy, wz],
        health_ratio,
        lod,
    })
}

fn finalize_prepared_frame(
    units: &[SimUnitSnapshot],
    results: Vec<Option<RenderInstance>>,
) -> PreparedFrame {
    let mut instances = Vec::with_capacity(units.len());
    let mut culled_count = 0usize;
    let mut lod_counts = [0usize; 3];

    for opt in &results {
        if let Some(inst) = opt {
            lod_counts[inst.lod as usize] += 1;
            instances.push(inst.clone());
        } else {
            culled_count += 1;
        }
    }

    let mut hash = FNV_OFFSET_BASIS;
    hash = fnv1a_extend(hash, &(culled_count as u64).to_le_bytes());

    for (u, opt) in units.iter().zip(results.iter()) {
        if let Some(inst) = opt {
            hash = fnv1a_extend(hash, &inst.unit_id.to_le_bytes());
            hash = fnv1a_extend(hash, &[inst.lod]);
            hash = fnv1a_extend(hash, &u.x.to_le_bytes());
            hash = fnv1a_extend(hash, &u.y.to_le_bytes());
            hash = fnv1a_extend(hash, &u.hp.to_le_bytes());
        }
    }

    PreparedFrame {
        instances,
        culled_count,
        lod_counts,
        digest_fnv1a: hash,
    }
}

pub fn prepare_render_frame(
    units: &[SimUnitSnapshot],
    scmap: Option<&ScMap>,
    view_proj: [[f32; 4]; 4],
    camera_eye: [f32; 3],
    threads: usize,
) -> PreparedFrame {
    let scale = crate::sim::SCALE as f32;
    let results = par_map_ordered(units, threads, |_, u| {
        evaluate_unit_instance(u, scmap, &view_proj, &camera_eye, scale)
    });
    finalize_prepared_frame(units, results)
}

/// Persistent worker-pool variant of [`prepare_render_frame`] yielding identical digests.
pub fn prepare_render_frame_pool(
    units: &Arc<Vec<SimUnitSnapshot>>,
    scmap: Option<&Arc<ScMap>>,
    view_proj: [[f32; 4]; 4],
    camera_eye: [f32; 3],
    pool: &WorkerPool,
) -> PreparedFrame {
    let scale = crate::sim::SCALE as f32;
    let map_opt = scmap.cloned();

    let results = pool.par_map_ordered_arc(
        Arc::clone(units),
        Arc::new(move |_: usize, u: &SimUnitSnapshot| {
            evaluate_unit_instance(u, map_opt.as_deref(), &view_proj, &camera_eye, scale)
        }),
    );

    finalize_prepared_frame(units, results)
}

// ---------------------------------------------------------------------------
// Phase 11 "Living Battlefield" + Phase 17 visual fidelity: blended skeletal animation (walk/idle/
// rest crossfades, death & factory work clips, turret yaw/pitch tracking with barrel recoil), LOD1
// meshes, shield domes with impact ripples, projectile meshes & missile arcs, aircraft banking and
// contrails, ship wakes & sinking wrecks, construction build-up, VFX director, FoW terrain mask and
// strategic icons. Everything here only *reads* the simulation.
// ---------------------------------------------------------------------------

/// Batch-key suffix for LOD1 unit mesh templates.
pub const LOD1_SUFFIX: &str = "#lod1";
/// Batch-key prefix for projectile mesh templates (`proj#<name>`).
pub const PROJ_PREFIX: &str = "proj#";
/// Ticks a turret keeps facing its last target after firing.
const TURRET_HOLD_TICKS: u64 = 40;
/// Crossfade between walk/idle/rest clips.
const ANIM_BLEND_SECS: f32 = 0.2;
/// Barrel recoil return time after the instant kick.
const RECOIL_RETURN_SECS: f32 = 0.45;
/// Shield impact ripple/flash lifetime.
const SHIELD_HIT_SECS: f32 = 0.7;
/// Naval wrecks settle on the sea floor over this many seconds.
const WRECK_SINK_SECS: f32 = 6.0;
/// Death clips are capped / held on their last frame this long before the wreck appears.
const DEATH_MAX_SECS: f32 = 5.0;
const DEATH_HOLD_SECS: f32 = 0.25;
/// Trailing sim ticks a PNG snapshot pre-builds (CPU only) so diff-derived effects have history.
pub const SNAPSHOT_WARMUP_TICKS: u64 = 40;

/// Normalization `(offset, scale)` mapping raw `.scm` space onto the play template space
/// (`p_norm = (p_raw - offset) * scale`): XZ-centered, feet on `y = 0`, footprint `target_span`.
pub fn normalization_for_bounds(min: [f32; 3], max: [f32; 3], target_span: f32) -> ([f32; 3], f32) {
    let cx = (min[0] + max[0]) * 0.5;
    let cz = (min[2] + max[2]) * 0.5;
    let ex = (max[0] - min[0]).abs();
    let ez = (max[2] - min[2]).abs();
    let span = ex.max(ez).max(1e-3);
    ([cx, min[1], cz], target_span / span)
}

/// Maps each vertex' first bone index from a `source` skeleton onto `target` bone indices by name
/// (unmatched bones fall back to `0`, the root).
pub fn remap_vertex_bones(source: &[&str], target: &[&str], vertex_bones: &[u8]) -> Vec<f32> {
    let map = crate::sca::remap_bones_by_name(source, target);
    vertex_bones
        .iter()
        .map(|&b| map.get(usize::from(b)).copied().flatten().unwrap_or(0) as f32)
        .collect()
}

/// Deterministic per-unit animation phase in `[0, 1)` so neighbours don't walk in lock-step.
pub fn unit_anim_phase(id: u32) -> f32 {
    (id.wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32
}

/// `true` when a shot fired at `shot_tick` has not been visualised yet. Without history
/// (`seen == None`), only shots from the last few ticks count, so a fresh renderer (snapshot)
/// still shows the current volley but never replays old ones.
pub fn is_new_shot(shot_tick: u64, seen: Option<u64>, now_tick: u64) -> bool {
    match seen {
        Some(s) => shot_tick > s,
        None => shot_tick + 3 >= now_tick,
    }
}

/// Keys present in `prev` but absent from `now` (units/projectiles that disappeared).
#[cfg(test)]
pub fn vanished<K: Ord + Copy, V: Clone>(prev: &BTreeMap<K, V>, now: &BTreeMap<K, V>) -> Vec<(K, V)> {
    prev.iter()
        .filter(|(k, _)| !now.contains_key(k))
        .map(|(k, v)| (*k, v.clone()))
        .collect()
}

/// Builds the 128x128 RGBA fog-of-war mask (`255` = in vision, `110` = radar only, `0` = unseen)
/// from `(x, z, vision_radius, radar_radius)` circles in world units over a `map_w x map_h` map.
pub fn build_fow_mask(map_w: f32, map_h: f32, circles: &[(f32, f32, f32, f32)]) -> Vec<u8> {
    const N: usize = 128;
    let mut level = vec![0u8; N * N];
    let cw = map_w.max(1.0) / N as f32;
    let ch = map_h.max(1.0) / N as f32;
    for &(x, z, vis, radar) in circles {
        for (r, val) in [(radar, 110u8), (vis, 255u8)] {
            if r <= 0.0 {
                continue;
            }
            let i0 = (((x - r) / cw).floor().max(0.0) as usize).min(N - 1);
            let i1 = (((x + r) / cw).ceil().max(0.0) as usize).min(N - 1);
            let j0 = (((z - r) / ch).floor().max(0.0) as usize).min(N - 1);
            let j1 = (((z + r) / ch).ceil().max(0.0) as usize).min(N - 1);
            let r2 = r * r;
            for j in j0..=j1 {
                let cz = (j as f32 + 0.5) * ch - z;
                for i in i0..=i1 {
                    let cx = (i as f32 + 0.5) * cw - x;
                    if cx * cx + cz * cz <= r2 {
                        let l = &mut level[j * N + i];
                        *l = (*l).max(val);
                    }
                }
            }
        }
    }
    let mut out = vec![0u8; N * N * 4];
    for (i, &l) in level.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&[l, l, l, 255]);
    }
    out
}

/// Render-side trajectory family of a projectile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcKind {
    /// Overcharge bolts / no arc.
    Flat,
    /// Artillery shells: symmetric parabola.
    Ballistic,
    /// Tactical missiles: steep climb, long glide, dive.
    Missile,
    /// Strategic missiles: near-vertical launch, very early apex.
    Strategic,
}

impl ArcKind {
    pub fn of(p: &crate::skirmish::Projectile) -> Self {
        if p.is_overcharge || p.arc_height <= 0 {
            ArcKind::Flat
        } else if p.is_nuke {
            ArcKind::Strategic
        } else if p.is_missile {
            ArcKind::Missile
        } else {
            ArcKind::Ballistic
        }
    }
}

/// Normalized arc height `h(t)` (`0` at both ends, peak `1`) and slope `dh/dt` at flight fraction
/// `t`. Missiles re-time the parabola with `s = t^0.6` (strategic: `t^0.5`) so they climb almost
/// vertically, peak early and dive onto the target.
pub fn arc_profile(kind: ArcKind, t: f32) -> (f32, f32) {
    let t = if t.is_finite() { t.clamp(0.0, 1.0) } else { 0.0 };
    let e = match kind {
        ArcKind::Flat => return (0.0, 0.0),
        ArcKind::Ballistic => 1.0,
        ArcKind::Missile => 0.6,
        ArcKind::Strategic => 0.5,
    };
    let tc = t.max(1e-3);
    let s = tc.powf(e);
    let ds = e * tc.powf(e - 1.0);
    let h = if t <= 0.0 { 0.0 } else { 4.0 * s * (1.0 - s) };
    (h.clamp(0.0, 1.0), (4.0 - 8.0 * s) * ds)
}

/// Bank (roll, radians) for a heading rate in rad/s: aircraft lean into the turn, scaled by the
/// blueprint `Air.BankFactor` and clamped to `max_bank`.
pub fn bank_for_turn(yaw_rate: f32, bank_factor: f32, max_bank: f32) -> f32 {
    if !yaw_rate.is_finite() {
        return 0.0;
    }
    (yaw_rate * 0.32 * bank_factor.clamp(0.5, 6.0)).clamp(-max_bank, max_bank)
}

/// Fraction `0..=1` a wreck in deep water has sunk `age` seconds after it appeared (smoothstep;
/// wrecks without history, `age = inf`, already rest on the sea floor).
pub fn sink_fraction(age: f32, secs: f32) -> f32 {
    if age.is_nan() {
        return 1.0;
    }
    crate::sca::crossfade_weight(age.max(0.0), secs)
}

/// First point where the segment `from -> to` enters the sphere `(c, r)` from outside.
pub fn segment_sphere_entry(from: [f32; 3], to: [f32; 3], c: [f32; 3], r: f32) -> Option<[f32; 3]> {
    let d = sub3(to, from);
    let m = sub3(from, c);
    let a = dot3(d, d);
    let cc = dot3(m, m) - r * r;
    if a < 1e-9 || cc <= 0.0 {
        return None;
    }
    let b = 2.0 * dot3(m, d);
    let disc = b * b - 4.0 * a * cc;
    if disc < 0.0 {
        return None;
    }
    let s = (-b - disc.sqrt()) / (2.0 * a);
    (0.0..=1.0).contains(&s).then(|| [from[0] + d[0] * s, from[1] + d[1] * s, from[2] + d[2] * s])
}

/// Orthonormal `(right, up, forward)` basis for heading `yaw` (+Z turning toward +X), nose-up
/// `pitch` and `roll` (positive lowers the local +X side); equals the
/// `GpuUnitInstance::from_pos_yaw_scale` basis at zero pitch and roll.
pub fn ypr_basis(yaw: f32, pitch: f32, roll: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sy, cy) = yaw.sin_cos();
    let f0 = [sy, 0.0, cy];
    let l0 = [cy, 0.0, -sy];
    let (sb, cb) = roll.sin_cos();
    let l1 = [l0[0] * cb, -sb, l0[2] * cb];
    let u1 = [l0[0] * sb, cb, l0[2] * sb];
    let (sp, cp) = pitch.sin_cos();
    let f2 = [f0[0] * cp + u1[0] * sp, u1[1] * sp, f0[2] * cp + u1[2] * sp];
    let u2 = [u1[0] * cp - f0[0] * sp, u1[1] * cp, u1[2] * cp - f0[2] * sp];
    (l1, u2, f2)
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn add_scaled(a: [f32; 3], d: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] + d[0] * s, a[1] + d[1] * s, a[2] + d[2] * s]
}

/// Moves `cur` toward `target` by at most `step`.
fn approach(cur: f32, target: f32, step: f32) -> f32 {
    if (target - cur).abs() <= step { target } else { cur + step * (target - cur).signum() }
}

/// Render height above the ground of a unit: flight altitude plus a gentle hover bob for aircraft.
fn unit_lift_at(u: &crate::skirmish::Unit, t: f32) -> f32 {
    if u.domain != crate::flowfield::NavDomain::Air || u.altitude <= 0 {
        return 0.0;
    }
    let bob = (t * 1.6 + unit_anim_phase(u.id) * std::f32::consts::TAU).sin() * 0.35;
    u.altitude as f32 / crate::sim::SCALE as f32 + bob
}

/// Points spaced `spacing` apart along `a -> b` (excluding `a`), at most `max`.
fn points_along(a: [f32; 3], b: [f32; 3], spacing: f32, max: usize) -> Vec<[f32; 3]> {
    let d = sub3(b, a);
    let len = dot3(d, d).sqrt();
    let n = ((len / spacing.max(1e-3)).ceil() as usize).clamp(1, max.max(1));
    (1..=n).map(|i| add_scaled(a, d, i as f32 / n as f32)).collect()
}

/// LOD1 template geometry (normalized + bone-remapped onto the LOD0 rig).
struct Lod1Mesh {
    vertices: Vec<crate::scene::GpuVertex>,
    indices: Vec<u32>,
    albedo: Option<crate::dds_decode::RgbaImage>,
    spec_glow: Option<crate::dds_decode::SpecGlowMap>,
}

/// Projectile mesh template (`<Name>_lod0.scm` scaled by `Display.UniformScale`, nose along +Z).
struct ProjMesh {
    vertices: Vec<crate::scene::GpuVertex>,
    indices: Vec<u32>,
    albedo: Option<crate::dds_decode::RgbaImage>,
    /// World-space length along Z at instance scale 1.
    length: f32,
}

/// Per-type render behaviour derived from the blueprint (Phase 17).
#[derive(Debug, Clone)]
struct UnitFxInfo {
    /// Max barrel elevation (radians) and turret slew speeds (rad/s).
    max_pitch: f32,
    yaw_speed: f32,
    pitch_speed: f32,
    /// Has an area (shell-firing) weapon: barrels elevate for a ballistic arc.
    indirect: bool,
    /// Projectile template keys for shells / missiles (`None` = glow tracer fallback).
    shell_proj: Option<String>,
    missile_proj: Option<String>,
    personal_shield: bool,
    bank_factor: f32,
    winged: bool,
    contrail: bool,
}

impl Default for UnitFxInfo {
    fn default() -> Self {
        Self {
            max_pitch: 1.2,
            yaw_speed: 1.6,
            pitch_speed: 1.0,
            indirect: false,
            shell_proj: None,
            missile_proj: None,
            personal_shield: false,
            bank_factor: 2.0,
            winged: true,
            contrail: false,
        }
    }
}

/// Per-unit-type render assets loaded once (in parallel) at startup.
struct UnitRenderAssets {
    id: String,
    lod1_key: Option<String>,
    rig: Option<crate::sca::UnitRig>,
    /// Bone index (as `f32`) per template vertex, matching `play::UnitMeshCache`'s mesh.
    vertex_bones: Vec<f32>,
    lod1: Option<Lod1Mesh>,
    lod_cutoff: f32,
    /// Template height (normalized units, before instance scale) for the build-up clip plane.
    height: f32,
    fx: UnitFxInfo,
}

#[derive(Clone, Copy, Debug)]
struct FxUnit {
    pos: [f32; 3],
    radius: f32,
    acu: bool,
}

/// Render-only per-unit history (crossfades, turret slew, motion, shield hits, last pose).
#[derive(Clone, Debug)]
struct UnitVis {
    seen: u64,
    type_id: String,
    asset_idx: Option<usize>,
    vis_fx: Option<FxUnit>,
    shot_seen: Option<u64>,
    // Clip crossfade.
    mode: Option<crate::sca::AnimMode>,
    prev_mode: crate::sca::AnimMode,
    mode_since: f32,
    // Factory work clip (`AnimationBuild` loop / `AnimationOpen` one-shot).
    work_t: f32,
    work_w: f32,
    // Weapon aim: world yaw + elevation target, slewed relative yaw/pitch, last shot time.
    aim_target: Option<(f32, f32)>,
    aim_yaw: f32,
    aim_pitch: f32,
    aim_init: bool,
    shot_at: f32,
    // Motion: tick-to-tick velocity (world units/s), smoothed heading and bank.
    motion_init: bool,
    last_xy: [f32; 2],
    last_tick: u64,
    last_move_time: f32,
    vel: [f32; 2],
    heading: f32,
    heading_init: bool,
    bank: f32,
    tips: Option<[[f32; 3]; 2]>,
    wake_accum: f32,
    // Shields.
    shield_hp: i64,
    hit_dir: [f32; 3],
    hit_at: f32,
    // Last posed (pre-skin) instance, for the death clip.
    last_inst: Option<GpuUnitInstance>,
    last_inst_frame: u64,
}

impl Default for UnitVis {
    fn default() -> Self {
        Self {
            seen: 0,
            type_id: String::new(),
            asset_idx: None,
            vis_fx: None,
            shot_seen: None,
            mode: None,
            prev_mode: crate::sca::AnimMode::Rest,
            mode_since: f32::NEG_INFINITY,
            work_t: 0.0,
            work_w: 0.0,
            aim_target: None,
            aim_yaw: 0.0,
            aim_pitch: 0.0,
            aim_init: false,
            shot_at: f32::NEG_INFINITY,
            motion_init: false,
            last_xy: [0.0; 2],
            last_tick: 0,
            last_move_time: f32::NEG_INFINITY,
            vel: [0.0; 2],
            heading: 0.0,
            heading_init: false,
            bank: 0.0,
            tips: None,
            wake_accum: 0.0,
            shield_hp: 0,
            hit_dir: [0.0, 1.0, 0.0],
            hit_at: f32::NEG_INFINITY,
            last_inst: None,
            last_inst_frame: 0,
        }
    }
}

impl UnitVis {
    /// Current ground speed (world units/s); `0` when the sim has not moved the unit recently.
    fn speed(&self, now: f32) -> f32 {
        if now - self.last_move_time > 0.6 {
            0.0
        } else {
            (self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1]).sqrt()
        }
    }
}

/// A unit playing its `AnimationDeath` clip before its wreck appears.
#[derive(Clone, Debug)]
struct Dying {
    id: u32,
    type_id: String,
    inst: GpuUnitInstance,
    start: f32,
    clip: usize,
    secs: f32,
}

/// A projectile to draw this frame.
#[derive(Clone, Debug)]
struct ProjDraw {
    key: Option<String>,
    pos: [f32; 3],
    dir: [f32; 3],
    scale: f32,
}

/// One immediate FA `BeamBlueprint` ribbon to draw this frame.
struct BeamDraw {
    a: [f32; 3],
    b: [f32; 3],
    beam: crate::effects::CompiledBeam,
    width_scale: f32,
    tint: Option<[f32; 3]>,
}

/// Active builder beam/spark emission record for one frame.
struct BuilderFx {
    src: [f32; 3],
    dst: [f32; 3],
    rgb: [f32; 3],
    faction: crate::effects::Faction,
}

#[derive(Default)]
struct LivingFrame {
    time: f32,
    last_time: Option<f32>,
    dt: f32,
    tick: u64,
    frame_no: u64,
    primed: bool,
    palette: Vec<[[f32; 4]; 4]>,
    vfx: crate::vfx::VfxSystem,
    spark_accum: f32,
    icons: Vec<([f32; 3], [f32; 4], [f32; 4], f32)>,
    fow_digest: Option<u64>,
    icons_uploaded: bool,
    vfx_tex_uploaded: bool,
    units: U32Map<UnitVis>,
    dying: Vec<Dying>,
    wrecks_seen: U32Map<(f32, u64)>,
    proj_kind: U32Map<Option<String>>,
    proj_shooter: U32Map<String>,
    prev_proj: U32Map<([f32; 3], f32, u64)>,
    proj_draw: Vec<ProjDraw>,
    shields: Vec<GpuUnitInstance>,
    /// Immediate (this frame only) glow billboards `(pos, size, rgba)` and streaks `(a, b, width, rgba)`.
    glows: Vec<([f32; 3], f32, [f32; 4])>,
    streaks: Vec<([f32; 3], [f32; 3], f32, [f32; 4])>,
    /// Immediate FA `BeamBlueprint` ribbons drawn through `vfx_pipeline`.
    beams: Vec<BeamDraw>,
    /// Immediate FA `TrailEmitterBlueprint` projectile ribbons `(tail, head, trail, width_scale)`.
    trails: Vec<([f32; 3], [f32; 3], crate::effects::CompiledTrail, f32)>,
    /// Skeletal poses queued by [`LivingBattlefield::skin_instance`] for this frame, solved in
    /// parallel by [`LivingBattlefield::resolve_pose_jobs`].
    pose_jobs: Vec<PoseJob>,
    /// `(batch index, instance index)` of each queued pose job.
    pose_targets: Vec<(usize, usize)>,
}

/// Everything needed to sample one unit's bone palette off the render thread.
#[derive(Debug, Clone)]
struct PoseJob {
    asset_idx: usize,
    mode: crate::sca::AnimMode,
    prev_mode: crate::sca::AnimMode,
    blending: bool,
    w: f32,
    t: f32,
    phase: f32,
    work_w: f32,
    work_t: f32,
    aim: crate::sca::AimPose,
}

fn compute_pose_palette(assets: &[UnitRenderAssets], j: &PoseJob) -> Vec<[[f32; 4]; 4]> {
    let Some(rig) = assets.get(j.asset_idx).and_then(|a| a.rig.as_ref()) else {
        return Vec::new();
    };
    let mut pose = rig.sample_pose(rig.clip_for(j.mode), rig.clip_time(j.mode, j.t, j.phase), true);
    if j.blending {
        let prev = rig.sample_pose(rig.clip_for(j.prev_mode), rig.clip_time(j.prev_mode, j.t, j.phase), true);
        pose = crate::sca::blend_poses(&prev, &pose, j.w);
    }
    if j.work_w > 1e-3
        && let Some((clip, looping)) = &rig.work
    {
        let wp = rig.sample_pose(Some(clip), j.work_t, *looping);
        pose = crate::sca::blend_poses(&pose, &wp, j.work_w);
    }
    rig.palette_from_pose(&pose, j.aim)
}

fn default_fx() -> &'static UnitFxInfo {
    static DEFAULT_FX: std::sync::OnceLock<UnitFxInfo> = std::sync::OnceLock::new();
    DEFAULT_FX.get_or_init(UnitFxInfo::default)
}

/// Pose jobs below this count are solved inline (thread hand-off would cost more).
const PARALLEL_POSE_MIN: usize = 96;

/// Render-only "living battlefield" state attached to `play::UnitMeshCache`.
#[derive(Default)]
pub struct LivingBattlefield {
    assets: Arc<Vec<UnitRenderAssets>>,
    asset_index: Arc<HashMap<String, usize>>,
    proj_meshes: Arc<BTreeMap<String, ProjMesh>>,
    icon_names: Arc<BTreeMap<String, String>>,
    icon_atlas: Option<Arc<crate::scene::StrategicIconAtlas>>,
    effects: Arc<crate::effects::EffectLibrary>,
    wall_clock: std::cell::Cell<Option<std::time::Instant>>,
    frame: std::cell::RefCell<LivingFrame>,
    /// Lazily created persistent pool for per-frame pose sampling.
    pose_pool: std::sync::OnceLock<Arc<crate::jobs::WorkerPool>>,
}

impl Clone for LivingBattlefield {
    fn clone(&self) -> Self {
        Self {
            assets: Arc::clone(&self.assets),
            asset_index: Arc::clone(&self.asset_index),
            proj_meshes: Arc::clone(&self.proj_meshes),
            icon_names: Arc::clone(&self.icon_names),
            icon_atlas: self.icon_atlas.clone(),
            effects: Arc::clone(&self.effects),
            wall_clock: std::cell::Cell::new(self.wall_clock.get()),
            frame: std::cell::RefCell::new(LivingFrame::default()),
            pose_pool: self.pose_pool.clone(),
        }
    }
}

impl std::fmt::Debug for LivingBattlefield {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LivingBattlefield")
            .field("unit_types", &self.assets.len())
            .field("rigged", &self.rigged_count())
            .field("lod1", &self.assets.iter().filter(|a| a.lod1.is_some()).count())
            .field("projectile_meshes", &self.proj_meshes.len())
            .field("icon_atlas", &self.icon_atlas.is_some())
            .finish()
    }
}

fn lc_norm(p: &str) -> String {
    p.replace('\\', "/").trim_start_matches('/').to_ascii_lowercase()
}

/// `projectiles/<dir>/<name>_proj.bp` -> (`projectiles/<dir>/`, `<name>`), lower-case.
fn projectile_dir_and_name(id: &str) -> Option<(String, String)> {
    let p = lc_norm(id);
    let (dir, file) = p.rsplit_once('/')?;
    let name = file.strip_suffix("_proj.bp")?;
    Some((format!("{dir}/"), name.to_string()))
}

/// Builds projectile mesh templates for every `ProjectileId` that ships a `_lod0.scm`.
fn load_projectile_meshes(game_dir: &Path, ids: &BTreeSet<String>, threads: usize) -> BTreeMap<String, ProjMesh> {
    let mut wanted = BTreeSet::new();
    for id in ids {
        if let Some((dir, name)) = projectile_dir_and_name(id) {
            for suffix in ["_proj.bp", "_lod0.scm", "_albedo.dds"] {
                wanted.insert(format!("{dir}{name}{suffix}"));
            }
        }
    }
    if wanted.is_empty() {
        return BTreeMap::new();
    }
    let scd_path = crate::scd::gamedata_dir(game_dir).join("projectiles.scd");
    let files: BTreeMap<String, Vec<u8>> = crate::scd::read_scd_matching(&scd_path, |n| wanted.contains(&lc_norm(n)))
        .map(|v| v.into_iter().map(|(k, b)| (lc_norm(&k), b)).collect())
        .unwrap_or_default();
    let list: Vec<String> = ids.iter().cloned().collect();
    par_map_ordered(&list, threads, |_, id| {
        let (dir, name) = projectile_dir_and_name(id)?;
        let scm = ScmMesh::parse(files.get(&format!("{dir}{name}_lod0.scm"))?).ok()?;
        if scm.vertices.is_empty() || scm.indices.is_empty() {
            return None;
        }
        let info = files
            .get(&format!("{dir}{name}_proj.bp"))
            .and_then(|b| crate::scene::parse_projectile_blueprint(&String::from_utf8_lossy(b)).ok())
            .map(|bp| crate::scene::parse_projectile_display_info(&bp));
        let s = info.as_ref().map(|i| i.uniform_scale).unwrap_or(0.1);
        let b = scm.bounds?;
        let vertices: Vec<crate::scene::GpuVertex> = scm
            .vertices
            .iter()
            .map(|v| crate::scene::GpuVertex {
                position: [v.position[0] * s, v.position[1] * s, v.position[2] * s],
                normal: normalize3(v.normal),
                uv: v.uv0,
                color: [1.0, 1.0, 1.0, 0.0],
            })
            .collect();
        let albedo = files
            .get(&format!("{dir}{name}_albedo.dds"))
            .and_then(|b| crate::dds_decode::decode_dds_rgba8(b).ok())
            .map(|a| crate::scene::pack_albedo_team_mask(a, None));
        let mesh = ProjMesh {
            vertices,
            indices: scm.indices.iter().map(|&i| u32::from(i)).collect(),
            albedo,
            length: ((b.max[2] - b.min[2]).abs() * s).max(1e-3),
        };
        Some((format!("{PROJ_PREFIX}{name}"), mesh))
    })
    .into_iter()
    .flatten()
    .collect()
}

impl LivingBattlefield {
    /// Loads rigs (`.scm` skeleton + walk/idle/death/work `.sca` clips + weapon bones), LOD1
    /// meshes/textures, projectile meshes and the strategic icon atlas for every catalog unit type.
    /// Archive reads are batched, parsing is parallel.
    pub fn load(game_dir: Option<&Path>, catalog: &crate::skirmish::Catalog) -> Self {
        let icon_names: BTreeMap<String, String> = catalog
            .types
            .iter()
            .map(|(id, t)| (id.clone(), t.strategic_icon.clone()))
            .collect();
        let Some(game_dir) = game_dir else {
            return Self {
                icon_names: Arc::new(icon_names),
                ..Self::default()
            };
        };
        let units_scd = crate::scd::gamedata_dir(game_dir).join("units.scd");
        let ids: Vec<String> = catalog.types.keys().cloned().collect();
        let icon_list: Vec<String> = icon_names.values().cloned().collect();

        let (assets, proj_meshes, atlas, effects) = std::thread::scope(|scope| {
            let atlas_job = scope.spawn(|| crate::scene::load_strategic_icon_atlas(Some(game_dir), &icon_list));
            let effects_ids = ids.clone();
            let effects_job = scope.spawn(move || crate::effects::load_effect_library(game_dir, &effects_ids));

            // Pass 1: blueprints, meshes and LOD1 textures in one archive sweep.
            let mut wanted = BTreeSet::new();
            for id in &ids {
                let l = id.to_ascii_lowercase();
                for suffix in ["_unit.bp", "_lod0.scm", "_lod1.scm", "_lod1_albedo.dds", "_lod1_specteam.dds"] {
                    wanted.insert(format!("units/{l}/{l}{suffix}"));
                }
            }
            let files: BTreeMap<String, Vec<u8>> = crate::scd::read_scd_matching(&units_scd, |n| {
                wanted.contains(&lc_norm(n))
            })
            .map(|v| v.into_iter().map(|(k, b)| (lc_norm(&k), b)).collect())
            .unwrap_or_default();
            let file = |id: &str, suffix: &str| {
                let l = id.to_ascii_lowercase();
                files.get(&format!("units/{l}/{l}{suffix}"))
            };

            let threads = crate::jobs::default_thread_count();
            let infos: Vec<Option<crate::scene::UnitDisplayInfo>> = par_map_ordered(&ids, threads, |_, id| {
                let bytes = file(id, "_unit.bp")?;
                let bp = crate::blueprint::parse_blueprint(&String::from_utf8_lossy(bytes)).ok()?;
                Some(crate::scene::parse_unit_display_info(&bp))
            });

            // Projectile meshes for every referenced `ProjectileId`.
            let proj_ids: BTreeSet<String> = infos
                .iter()
                .flatten()
                .flat_map(|i| i.weapons.iter().filter_map(|w| w.projectile_id.clone()))
                .collect();
            let proj_job = scope.spawn(move || load_projectile_meshes(game_dir, &proj_ids, threads));

            // Pass 2: every referenced walk/idle/death/work clip, parsed once.
            let mut anim_paths = BTreeSet::new();
            for info in infos.iter().flatten() {
                for p in [&info.walk_anim, &info.idle_anim, &info.build_anim, &info.open_anim]
                    .into_iter()
                    .flatten()
                {
                    anim_paths.insert(lc_norm(p));
                }
                for d in &info.death_anims {
                    anim_paths.insert(lc_norm(&d.path));
                }
            }
            let anim_raw: Vec<(String, Vec<u8>)> = if anim_paths.is_empty() {
                Vec::new()
            } else {
                crate::scd::read_scd_matching(&units_scd, |n| anim_paths.contains(&lc_norm(n)))
                    .unwrap_or_default()
            };
            let anims: BTreeMap<String, crate::sca::ScaAnimation> =
                par_map_ordered(&anim_raw, threads, |_, (k, b)| {
                    crate::sca::ScaAnimation::parse(b).ok().map(|a| (lc_norm(k), a))
                })
                .into_iter()
                .flatten()
                .collect();
            let proj_meshes = proj_job.join().unwrap_or_default();
            let proj_key = |id: &Option<String>| {
                let (_, name) = projectile_dir_and_name(id.as_deref()?)?;
                let key = format!("{PROJ_PREFIX}{name}");
                proj_meshes.contains_key(&key).then_some(key)
            };

            let built: Vec<Option<UnitRenderAssets>> = par_map_ordered(&ids, threads, |i, id| {
                let utype = catalog.types.get(id)?;
                let target_span =
                    (utype.size[0].max(utype.size[2]) as f32 / crate::sim::SCALE as f32).max(1.6);
                // Same candidate order as `play::UnitMeshCache` so vertex bone indices line up.
                let (used_lod0, scm) = [("_lod0.scm", true), ("_lod1.scm", false)]
                    .into_iter()
                    .find_map(|(suf, is0)| {
                        let m = ScmMesh::parse(file(id, suf)?).ok()?;
                        (!m.vertices.is_empty() && !m.indices.is_empty() && m.bounds.is_some())
                            .then_some((is0, m))
                    })?;
                let b = scm.bounds?;
                let (norm_offset, norm_scale) = normalization_for_bounds(b.min, b.max, target_span);
                let span_raw = (b.max[0] - b.min[0]).abs().max((b.max[2] - b.min[2]).abs());
                let height = ((b.max[1] - b.min[1]).abs() * norm_scale).max(0.2);
                let vertex_bones: Vec<f32> =
                    scm.vertices.iter().map(|v| f32::from(v.bone_indices[0])).collect();
                let info = infos[i].clone();
                let names: Vec<&str> = scm.bones.iter().map(|b| b.name.as_str()).collect();
                let clip = |p: &Option<String>| {
                    let a = anims.get(&lc_norm(p.as_deref()?))?;
                    crate::sca::AnimClip::new(a.clone(), &names)
                };
                let mut rig = crate::sca::UnitRig::new(scm.bones.clone(), norm_offset, norm_scale);
                let mut fx = UnitFxInfo::default();
                let lod_cutoff = match &info {
                    Some(inf) => {
                        rig.walk = clip(&inf.walk_anim);
                        rig.idle = clip(&inf.idle_anim);
                        rig.walk_rate = (inf.max_speed / inf.walk_rate).clamp(0.2, 4.0);
                        rig.death = inf
                            .death_anims
                            .iter()
                            .filter_map(|d| {
                                Some(crate::sca::OneShotClip { clip: clip(&Some(d.path.clone()))?, rate: d.rate })
                            })
                            .collect();
                        rig.work = clip(&inf.build_anim)
                            .map(|c| (c, true))
                            .or_else(|| clip(&inf.open_anim).map(|c| (c, false)));
                        crate::scene::apply_weapon_bones(&mut rig, &inf.weapons, span_raw);
                        let ws = &inf.weapons;
                        let deg_max = |f: fn(&crate::scene::WeaponDisplayInfo) -> f32, d: f32| {
                            ws.iter().map(f).fold(None, |m: Option<f32>, v| Some(m.map_or(v, |m| m.max(v)))).unwrap_or(d)
                        };
                        fx.max_pitch = deg_max(|w| w.pitch_range_deg, 70.0).clamp(5.0, 89.0).to_radians();
                        fx.yaw_speed = deg_max(|w| w.yaw_speed_deg, 90.0).clamp(30.0, 360.0).to_radians();
                        fx.pitch_speed = deg_max(|w| w.pitch_speed_deg, 60.0).clamp(20.0, 360.0).to_radians();
                        fx.indirect = ws.iter().any(|w| w.splash);
                        fx.shell_proj = ws
                            .iter()
                            .filter(|w| w.splash)
                            .chain(ws.iter())
                            .find_map(|w| proj_key(&w.projectile_id));
                        fx.missile_proj = ws
                            .iter()
                            .filter(|w| {
                                w.projectile_id.as_deref().is_some_and(|p| p.contains("missile") || p.contains("nuke"))
                            })
                            .find_map(|w| proj_key(&w.projectile_id));
                        fx.personal_shield = inf.personal_shield;
                        fx.bank_factor = if inf.bank_factor > 0.0 { inf.bank_factor } else { 2.0 };
                        fx.winged = inf.winged;
                        fx.contrail = inf.winged && (inf.max_airspeed >= 14.0 || utype.tier >= 2);
                        inf.lod0_cutoff.clamp(60.0, 400.0)
                    }
                    None => 150.0,
                };
                let rig = (!scm.bones.is_empty() && rig.is_animated()).then_some(rig);

                let lod1 = if used_lod0 {
                    file(id, "_lod1.scm")
                        .and_then(|bytes| ScmMesh::parse(bytes).ok())
                        .filter(|m| !m.vertices.is_empty() && !m.indices.is_empty())
                        .map(|m| {
                            let src: Vec<&str> = m.bones.iter().map(|b| b.name.as_str()).collect();
                            let vb: Vec<u8> = m.vertices.iter().map(|v| v.bone_indices[0]).collect();
                            let bones = remap_vertex_bones(&src, &names, &vb);
                            let vertices = m
                                .vertices
                                .iter()
                                .zip(bones)
                                .map(|(v, bone)| crate::scene::GpuVertex {
                                    position: [
                                        (v.position[0] - norm_offset[0]) * norm_scale,
                                        (v.position[1] - norm_offset[1]) * norm_scale,
                                        (v.position[2] - norm_offset[2]) * norm_scale,
                                    ],
                                    normal: normalize3(v.normal),
                                    uv: v.uv0,
                                    color: [1.0, 1.0, 1.0, bone],
                                })
                                .collect();
                            let spec = file(id, "_lod1_specteam.dds")
                                .and_then(|b| crate::dds_decode::decode_dds_rgba8(b).ok());
                            let spec_glow = spec.as_ref().map(crate::dds_decode::SpecGlowMap::from_spec_team);
                            let albedo = file(id, "_lod1_albedo.dds")
                                .and_then(|b| crate::dds_decode::decode_dds_rgba8(b).ok())
                                .map(|a| crate::scene::pack_albedo_team_mask(a, spec.as_ref()));
                            Lod1Mesh {
                                vertices,
                                indices: m.indices.iter().map(|&i| u32::from(i)).collect(),
                                albedo,
                                spec_glow,
                            }
                        })
                        .filter(|l| l.albedo.is_some())
                } else {
                    None
                };
                let lod1_key = lod1.is_some().then(|| format!("{id}{LOD1_SUFFIX}"));
                Some(UnitRenderAssets {
                    id: id.clone(),
                    lod1_key,
                    rig,
                    vertex_bones,
                    lod1,
                    lod_cutoff,
                    height,
                    fx,
                })
            });
            let mut assets = Vec::new();
            let mut asset_index = HashMap::new();
            for (id, a) in ids.into_iter().zip(built) {
                if let Some(a) = a {
                    let idx = assets.len();
                    assets.push(a);
                    asset_index.insert(id, idx);
                }
            }
            (
                (assets, asset_index),
                proj_meshes,
                atlas_job.join().ok(),
                effects_job.join().unwrap_or_default(),
            )
        });

        Self {
            assets: Arc::new(assets.0),
            asset_index: Arc::new(assets.1),
            proj_meshes: Arc::new(proj_meshes),
            icon_names: Arc::new(icon_names),
            icon_atlas: atlas.map(Arc::new),
            effects: Arc::new(effects),
            wall_clock: std::cell::Cell::new(None),
            frame: std::cell::RefCell::new(LivingFrame::default()),
            pose_pool: std::sync::OnceLock::new(),
        }
    }

    /// True when the FA `EffectLibrary` (Phase 40 emitters/beams/trails + texture array) loaded.
    pub fn has_effect_library(&self) -> bool {
        self.effects.is_loaded()
    }

    /// True when `type_id` has an FA `BeamBlueprint` weapon beam.
    pub fn has_weapon_beam(&self, type_id: &str) -> bool {
        self.effects
            .unit_weapon_fx(type_id)
            .is_some_and(|w| w.beam.is_some())
    }

    fn asset(&self, type_id: &str) -> Option<&UnitRenderAssets> {
        self.asset_index.get(type_id).and_then(|&i| self.assets.get(i))
    }

    /// Number of unit types with an animated rig (clips, turret or recoil bones).
    pub fn rigged_count(&self) -> usize {
        self.assets.iter().filter(|a| a.rig.is_some()).count()
    }

    /// Switches animation/VFX timing to the wall clock (interactive window). Snapshots keep the
    /// deterministic `tick / TICK_HZ` clock.
    pub fn enable_wall_clock(&self) {
        if self.wall_clock.get().is_none() {
            self.wall_clock.set(Some(std::time::Instant::now()));
        }
    }

    /// The interactive window should redraw continuously so walk cycles & particles stay smooth.
    pub fn wants_continuous_redraw(&self) -> bool {
        self.wall_clock.get().is_some()
    }

    /// Bone index (shader `color.w`) for template vertex `vi` of `type_id` (`0` when unrigged).
    pub fn vertex_bone(&self, type_id: &str, vi: usize) -> f32 {
        self.asset(type_id)
            .and_then(|a| a.vertex_bones.get(vi).copied())
            .unwrap_or(0.0)
    }

    /// Uploads LOD1 templates (`<ID>#lod1`) next to the LOD0 templates, plus projectile mesh
    /// templates (`proj#<name>`).
    pub(crate) fn create_lod_templates(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &GpuSceneResources,
        templates: &mut BTreeMap<String, GpuMeshTemplate>,
    ) {
        for a in self.assets.iter() {
            if let (Some(l), Some(key)) = (&a.lod1, &a.lod1_key)
                && templates.contains_key(&a.id)
            {
                let tpl = resources.create_mesh_template(
                    device,
                    queue,
                    &l.vertices,
                    &l.indices,
                    l.albedo.as_ref(),
                    l.spec_glow.as_ref(),
                    key,
                );
                templates.insert(key.clone(), tpl);
            }
        }
        for (key, m) in self.proj_meshes.iter() {
            let tpl = resources.create_mesh_template(device, queue, &m.vertices, &m.indices, m.albedo.as_ref(), None, key);
            templates.insert(key.clone(), tpl);
        }
    }

    /// Instance batch key: `<ID>#lod1` beyond the blueprint `LODCutoff`, else `<ID>`.
    #[allow(dead_code)]
    pub fn batch_key(&self, type_id: &str, pos: [f32; 3], eye: [f32; 3]) -> String {
        if let Some(a) = self.asset(type_id)
            && let Some(key) = &a.lod1_key
        {
            let d = [pos[0] - eye[0], pos[1] - eye[1], pos[2] - eye[2]];
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > a.lod_cutoff * a.lod_cutoff {
                return key.clone();
            }
        }
        type_id.to_string()
    }

    /// Current render frame time (call after [`Self::begin_frame`]).
    pub fn frame_time(&self) -> f32 {
        self.frame.borrow().time
    }

    /// Render height above terrain using a pre-fetched frame time (avoids per-unit RefCell borrows).
    pub fn unit_lift_with_time(u: &crate::skirmish::Unit, time: f32) -> f32 {
        unit_lift_at(u, time)
    }

    /// `(asset_idx, lod_cutoff_sq, lod1_key)` for per-frame type batch caching.
    pub fn asset_lod_info(&self, type_id: &str) -> Option<(usize, f32, Option<&str>)> {
        let &idx = self.asset_index.get(type_id)?;
        let a = &self.assets[idx];
        Some((idx, a.lod_cutoff * a.lod_cutoff, a.lod1_key.as_deref()))
    }

    fn clock(&self, tick: u64) -> f32 {
        match self.wall_clock.get() {
            Some(start) => start.elapsed().as_secs_f32(),
            None => tick as f32 / crate::sim::TICK_HZ as f32,
        }
    }

    fn fx_info(&self, type_id: &str) -> UnitFxInfo {
        self.asset(type_id).map(|a| a.fx.clone()).unwrap_or_default()
    }

    /// Borrowing form of [`Self::fx_info`] (no per-unit `String` clones on the hot path).
    fn fx_ref(&self, type_id: &str) -> &UnitFxInfo {
        match self.asset(type_id) {
            Some(a) => &a.fx,
            None => default_fx(),
        }
    }

    /// Height above the terrain at which `u` renders (aircraft fly at their blueprint elevation
    /// with a slow bob). Call after [`Self::begin_frame`].
    #[allow(dead_code)]
    pub fn unit_lift(&self, u: &crate::skirmish::Unit) -> f32 {
        unit_lift_at(u, self.frame.borrow().time)
    }

    /// Picks the projectile template + shooter type_id for a freshly spawned projectile from the
    /// weapon blueprint of the unit that fired it.
    fn resolve_projectile_key(&self, world: &crate::skirmish::Skirmish, p: &crate::skirmish::Projectile) -> (Option<String>, Option<String>) {
        let d2 = |u: &crate::skirmish::Unit| {
            let dx = (u.pos.x - p.start.x) as i128;
            let dy = (u.pos.y - p.start.y) as i128;
            dx * dx + dy * dy
        };
        let lim = 4 * crate::sim::SCALE as i128;
        // Phase 55: shells know their shooter; only legacy projectiles need the scan.
        let known = (p.shooter != 0).then(|| world.unit(p.shooter)).flatten();
        let Some(shooter) = known.or_else(|| {
            world
                .units()
                .filter(|u| u.alive && u.army == p.army)
                .min_by_key(|u| (d2(u), u.id))
                .filter(|u| d2(u) <= lim * lim)
        }) else {
            return (None, None);
        };
        let shooter_type = Some(shooter.type_id.clone());
        if p.is_overcharge {
            return (None, shooter_type);
        }
        let fx = self.fx_info(&shooter.type_id);
        let mesh_key = if p.is_missile || p.is_nuke { fx.missile_proj } else { fx.shell_proj };
        (mesh_key, shooter_type)
    }

    /// Starts a render frame: advances the render clock, clears the bone palette / icon list and
    /// turns read-only sim state changes (shots, deaths, impacts, shield hits, construction,
    /// motion) into particles and per-unit render history.
    pub fn begin_frame(
        &self,
        world: &crate::skirmish::Skirmish,
        scmap: &crate::scmap::ScMap,
        viewer: Option<crate::skirmish::ArmyId>,
        zoom_boost: f32,
    ) {
        use crate::flowfield::NavDomain;
        use crate::skirmish::{IntelVisibility, Role};
        let mut fr = self.frame.borrow_mut();
        let f = &mut *fr;
        let tick = world.tick();
        let t = self.clock(tick);
        let dt = match f.last_time {
            Some(lt) => (t - lt).clamp(0.0, 0.25),
            None => 0.0,
        };
        f.time = t;
        f.last_time = Some(t);
        f.dt = dt;
        f.tick = tick;
        f.frame_no += 1;
        let frame_no = f.frame_no;
        f.palette.clear();
        f.pose_jobs.clear();
        f.pose_targets.clear();
        f.icons.clear();
        f.shields.clear();
        f.proj_draw.clear();
        f.glows.clear();
        f.streaks.clear();
        f.beams.clear();
        f.trails.clear();

        let scale = crate::sim::SCALE as f32;
        let zb = zoom_boost.max(1.0);
        let water = scmap.water.has_water.then_some(scmap.water.elevation);
        let ground = |x: i64, z: i64, lift: f32| {
            let wx = x as f32 / scale;
            let wz = z as f32 / scale;
            [wx, crate::play::sample_terrain_y(scmap, wx, wz) + lift, wz]
        };
        let visible = |u: &crate::skirmish::Unit| match viewer {
            Some(a) => world.intel_visibility(a, u) == IntelVisibility::Visible,
            None => true,
        };

        let mut army_rgbs = [[0.7f32; 3]; 8];
        for (idx, rgb) in army_rgbs.iter_mut().enumerate() {
            *rgb = crate::play::army_rgb(world, idx as crate::skirmish::ArmyId);
        }
        let army_rgb_fast = |army: crate::skirmish::ArmyId| -> [f32; 3] {
            army_rgbs
                .get(usize::from(army))
                .copied()
                .unwrap_or_else(|| crate::play::army_rgb(world, army))
        };

        // Active shields: (owner, center, radius, army, rgb).
        let shields: Vec<(u32, [f32; 3], f32, crate::skirmish::ArmyId, [f32; 3])> = world
            .units()
            .filter(|u| u.alive && u.shield_on && u.shield_hp > 0 && u.shield_radius > 0)
            .map(|u| {
                let (c, r) = self.shield_sphere(u, scmap, t, zb);
                (u.id, c, r, u.army, army_rgb_fast(u.army))
            })
            .collect();
        let covering_shield = |target: [f32; 3], shooter_army: crate::skirmish::ArmyId| {
            shields.iter().any(|s| {
                let dx = target[0] - s.1[0];
                let dz = target[2] - s.1[2];
                !world.are_allies(s.3, shooter_army) && dx * dx + dz * dz <= s.2 * s.2
            })
        };

        let _sec = crate::profiler::scope("render.begin.units");
        // Units: deaths -> explosions (ACU -> nuclear shockwave); shots -> muzzle flash + impact
        // (shielded targets flash on the dome instead); motion, aim and work-clip history.
        let mut alive_count = 0usize;
        let mut vanished_fx: Vec<(u32, FxUnit)> = Vec::new();
        let mut builders: Vec<BuilderFx> = Vec::new();
        let mut shots: Vec<([f32; 3], [f32; 3])> = Vec::new();
        for u in world.units() {
            if !u.alive {
                continue;
            }
            alive_count += 1;
            let vis = visible(u);
            let radius = (u.radius as f32 / scale).max(0.6);
            let lift = unit_lift_at(u, t);
            let pos = ground(u.pos.x, u.pos.y, 0.9 * zb + lift);
            let st = f.units.entry(u.id).or_default();
            st.seen = frame_no;
            if st.type_id != u.type_id {
                st.type_id.clear();
                st.type_id.push_str(&u.type_id);
                st.asset_idx = self.asset_index.get(&u.type_id).copied();
            }
            if vis {
                st.vis_fx = Some(FxUnit { pos, radius, acu: u.role == Role::Commander });
            } else if let Some(d) = st.vis_fx.take()
                && f.primed
            {
                vanished_fx.push((u.id, d));
            }
            let asset = st.asset_idx.and_then(|i| self.assets.get(i));
            let fx = match asset {
                Some(a) => &a.fx,
                None => default_fx(),
            };

            // Motion history: tick-to-tick velocity, smoothed heading, bank from the turn rate.
            let xy = [u.pos.x as f32 / scale, u.pos.y as f32 / scale];
            if !st.motion_init {
                st.motion_init = true;
                st.last_xy = xy;
                st.last_tick = tick;
            } else if tick > st.last_tick {
                let secs = (tick - st.last_tick) as f32 / crate::sim::TICK_HZ as f32;
                st.vel = [(xy[0] - st.last_xy[0]) / secs, (xy[1] - st.last_xy[1]) / secs];
                if st.vel[0] != 0.0 || st.vel[1] != 0.0 {
                    st.last_move_time = t;
                }
                st.last_xy = xy;
                st.last_tick = tick;
            }
            let speed = st.speed(t);
            if matches!(u.domain, NavDomain::Air | NavDomain::Naval) {
                let prev_heading = st.heading;
                if speed > 0.3 {
                    let target = st.vel[0].atan2(st.vel[1]);
                    let turn = match u.domain {
                        NavDomain::Naval => 0.9,
                        _ if fx.winged => 1.6,
                        _ => 3.0,
                    };
                    if st.heading_init {
                        st.heading = crate::sca::slew_angle(st.heading, target, turn * dt);
                    } else {
                        st.heading = target;
                        st.heading_init = true;
                    }
                }
                if u.domain == NavDomain::Air {
                    let rate = if dt > 1e-4 { crate::sca::wrap_angle_pi(st.heading - prev_heading) / dt } else { 0.0 };
                    let target_bank = bank_for_turn(rate, fx.bank_factor, if fx.winged { 0.8 } else { 0.35 });
                    st.bank += (target_bank - st.bank) * (1.0 - (-dt * 5.0).exp());
                }
            }

            // Weapon aim target (held after each shot) and recoil timing.
            st.aim_target = None;
            if let Some((tid, shot_tick)) = u.last_shot {
                let seen = if f.primed { st.shot_seen } else { None };
                let fresh = is_new_shot(shot_tick, seen, tick);
                if let Some(tgt) = world.unit(tid) {
                    let tp = ground(tgt.pos.x, tgt.pos.y, 0.8 * zb + unit_lift_at(tgt, t));
                    if shot_tick + TURRET_HOLD_TICKS >= tick {
                        let dist = ((tp[0] - pos[0]).powi(2) + (tp[2] - pos[2]).powi(2)).sqrt();
                        let arc = if fx.indirect && tgt.domain != NavDomain::Air { (dist / 4.0).min(18.0) } else { 0.0 };
                        let pitch = crate::sca::aim_pitch(dist, tp[1] - pos[1], arc, -0.26, fx.max_pitch);
                        st.aim_target = Some((u.turret_yaw_mrad as f32 / 1000.0, pitch));
                    }
                    let dir = [tp[0] - pos[0], tp[1] - pos[1], tp[2] - pos[2]];
                    let len = (dir[0] * dir[0] + dir[2] * dir[2]).sqrt().max(1e-3);
                    let off = (radius * zb * 0.9).min(len * 0.5);
                    let muzzle = [pos[0] + dir[0] / len * off, pos[1] + 0.4 * zb, pos[2] + dir[2] / len * off];
                    let wfx = self.effects.unit_weapon_fx(&u.type_id);
                    if vis
                        && tick.saturating_sub(shot_tick) <= 2
                        && let Some(beam) = wfx.and_then(|w| w.beam.as_ref())
                    {
                        f.beams.push(BeamDraw {
                            a: muzzle,
                            b: tp,
                            beam: beam.clone(),
                            width_scale: zb.sqrt(),
                            tint: None,
                        });
                    }
                    if fresh {
                        st.shot_at = t;
                        shots.push((pos, tp));
                        if vis {
                            if let Some(w) = wfx
                                && !w.muzzle.is_empty()
                            {
                                f.vfx.spawn_emitters(&w.muzzle, muzzle, dir, 0.85 * zb, None, false);
                            } else {
                                f.vfx.spawn_muzzle_flash(muzzle, dir, 0.9 * zb);
                            }
                            if !covering_shield(tp, u.army) {
                                if let Some(w) = wfx
                                    && !w.impact.is_empty()
                                {
                                    let back = [-dir[0], -dir[1], -dir[2]];
                                    f.vfx.spawn_emitters(&w.impact, tp, back, 0.85 * zb, None, false);
                                } else {
                                    f.vfx.spawn_explosion(tp, 0.9 * zb);
                                }
                            }
                        }
                    }
                }
                st.shot_seen = Some(shot_tick);
            } else {
                st.shot_seen = None;
            }

            // Factory work clip: `AnimationBuild` loops (faded in/out), `AnimationOpen` opens while
            // working and closes in reverse when idle.
            if let Some((clip, looping)) = asset.and_then(|a| a.rig.as_ref()).and_then(|r| r.work.as_ref()) {
                let on = u.build.is_some();
                let dur = clip.duration().max(1e-3);
                if *looping {
                    st.work_w = if f.primed { approach(st.work_w, if on { 1.0 } else { 0.0 }, dt / 0.35) } else if on { 1.0 } else { 0.0 };
                    if st.work_w > 0.0 {
                        st.work_t = crate::sca::wrap_anim_time(st.work_t + dt, dur);
                    }
                } else {
                    st.work_t = if f.primed {
                        (st.work_t + if on { dt } else { -dt }).clamp(0.0, dur)
                    } else if on {
                        dur
                    } else {
                        0.0
                    };
                    st.work_w = if st.work_t > 1e-3 { 1.0 } else { 0.0 };
                }
            }

            // Contrails off fast aircraft wingtips; bow & stern wakes off moving ships.
            if vis && u.domain == NavDomain::Air && fx.contrail && speed > 6.0 {
                let (l, _, fw) = ypr_basis(st.heading, 0.0, st.bank);
                let half = radius * zb * 1.15 * 0.8;
                let c = ground(u.pos.x, u.pos.y, lift + 0.25 * zb);
                let tips = [
                    add_scaled(add_scaled(c, l, half), fw, -half * 0.5),
                    add_scaled(add_scaled(c, l, -half), fw, -half * 0.5),
                ];
                if let Some(prev) = st.tips {
                    for k in 0..2 {
                        for p in points_along(prev[k], tips[k], 0.9, 10) {
                            f.vfx.spawn_contrail(p, 0.5 * zb.sqrt());
                        }
                    }
                }
                st.tips = Some(tips);
            } else {
                st.tips = None;
            }
            if vis
                && u.domain == NavDomain::Naval
                && let Some(wl) = water
                && speed > 0.4
            {
                st.wake_accum += if f.primed { dt * (5.0 + speed * 1.2) } else { 4.0 };
                let n = (st.wake_accum.floor() as usize).min(6);
                st.wake_accum -= n as f32;
                let fw = [st.heading.sin(), 0.0, st.heading.cos()];
                let l = [st.heading.cos(), 0.0, -st.heading.sin()];
                let r = radius * zb.powf(0.6);
                let c = [xy[0], wl + 0.06, xy[1]];
                let bow = add_scaled(c, fw, r * 0.95);
                let stern = add_scaled(c, fw, -r * 0.9);
                for i in 0..n {
                    let side = if i % 2 == 0 { 1.0 } else { -1.0 };
                    let out = add_scaled(add_scaled([0.0; 3], l, side * (0.9 + speed * 0.22)), fw, -0.5);
                    f.vfx.spawn_wake(add_scaled(bow, l, side * r * 0.2), out, 0.45 * r);
                    f.vfx.spawn_wake(stern, add_scaled([0.0; 3], fw, -0.6), 0.6 * r);
                }
            }

            if vis
                && u.role.is_builder()
                && let Some(job) = &u.build
            {
                let c = army_rgb_fast(u.army);
                let src = [pos[0], pos[1] + 0.7 * zb, pos[2]];
                let dst = ground(job.x, job.y, 0.6 * zb);
                let fac = crate::effects::Faction::from_unit_id(&u.type_id);
                builders.push(BuilderFx { src, dst, rgb: c, faction: fac });
            }
        }

        drop(_sec);
        let _sec = crate::profiler::scope("render.begin.shields");
        // Shield HP drops -> ripple + flash where the incoming fire crossed the dome.
        for u in world.units().filter(|u| u.alive && u.shield_max_hp > 0) {
            let Some(st) = f.units.get_mut(&u.id) else {
                continue;
            };
            if f.primed && u.shield_hp < st.shield_hp {
                if let Some(s) = shields.iter().find(|s| s.0 == u.id) {
                    let (c, r) = (s.1, s.2);
                    let entry = shots
                        .iter()
                        .filter_map(|(a, b)| segment_sphere_entry(*a, *b, c, r))
                        .next();
                    let dir = match entry {
                        Some(p) => normalize3(sub3(p, c)),
                        None => {
                            let a = unit_anim_phase(u.id ^ tick as u32) * std::f32::consts::TAU;
                            normalize3([a.cos(), 0.8, a.sin()])
                        }
                    };
                    st.hit_dir = dir;
                    st.hit_at = t;
                    if visible(u) {
                        let hit_pos = add_scaled(c, dir, r);
                        let fac = crate::effects::Faction::from_unit_id(&u.type_id);
                        let sh_fx = self.effects.shield_hit_fx(fac);
                        if !sh_fx.is_empty() {
                            f.vfx.spawn_emitters(sh_fx, hit_pos, dir, zb, Some(s.4), false);
                        } else {
                            f.vfx.spawn_shield_hit(hit_pos, s.4);
                        }
                    }
                } else {
                    st.hit_at = t;
                }
            }
            st.shield_hp = u.shield_hp;
        }

        // Shield domes (Fresnel spheres tinted per faction; ripple from the last hit).
        for s in &shields {
            let Some(u) = world.unit(s.0) else {
                continue;
            };
            if !visible(u) {
                continue;
            }
            let st = f.units.get(&u.id);
            let hp = (u.shield_hp as f32 / u.shield_max_hp.max(1) as f32).clamp(0.0, 1.0);
            let age = st.map(|s| t - s.hit_at).unwrap_or(f32::INFINITY);
            let k = if age.is_finite() && age >= 0.0 { (1.0 - age / SHIELD_HIT_SECS).max(0.0) } else { 0.0 };
            let dir = st.map(|s| s.hit_dir).unwrap_or([0.0, 1.0, 0.0]);
            let rgb = s.4;
            let tint = [0.30 + rgb[0] * 0.7, 0.30 + rgb[1] * 0.7, 0.30 + rgb[2] * 0.7, 0.55 + 0.45 * hp];
            let personal = self.fx_ref(&u.type_id).personal_shield;
            f.shields.push(
                GpuUnitInstance::from_pos_yaw_scale(s.1, 0.0, 0.0, s.2, tint, [hp, k, f32::from(u8::from(personal)), 0.0])
                    .with_extra([dir[0], dir[1], dir[2], if k > 0.0 { age } else { 0.0 }]),
            );
        }

        drop(_sec);
        let _sec = crate::profiler::scope("render.begin.gone");
        // Units gone since the last frame: start their death clip (if the type has one and was on
        // screen), otherwise just drop their history. Only scan `f.units` when someone disappeared.
        if f.units.len() > alive_count {
            let mut gone: Vec<u32> = f
                .units
                .iter()
                .filter(|(_, s)| s.seen != frame_no)
                .map(|(&id, _)| id)
                .collect();
            gone.sort_unstable();
            for id in gone {
                let Some(st) = f.units.remove(&id) else {
                    continue;
                };
                if f.primed
                    && let Some(d) = st.vis_fx
                {
                    vanished_fx.push((id, d));
                }
                if !f.primed || st.last_inst_frame + 3 < frame_no {
                    continue;
                }
                if let (Some(inst), Some(rig)) = (
                    st.last_inst,
                    st.asset_idx
                        .and_then(|i| self.assets.get(i))
                        .and_then(|a| a.rig.as_ref()),
                ) && !rig.death.is_empty()
                {
                    let clip = (id as usize) % rig.death.len();
                    let secs = rig.death[clip].play_secs().clamp(0.3, DEATH_MAX_SECS);
                    f.dying.push(Dying { id, type_id: st.type_id, inst, start: t, clip, secs });
                }
            }
        }
        if f.primed && !vanished_fx.is_empty() {
            vanished_fx.sort_unstable_by_key(|(id, _)| *id);
            for (_, d) in vanished_fx {
                if d.acu {
                    if !self.effects.nuke.is_empty() {
                        f.vfx.spawn_emitters(&self.effects.nuke, d.pos, [0.0, 1.0, 0.0], 2.5, None, true);
                    }
                    f.vfx.spawn_acu_nuke(d.pos);
                } else {
                    let rad = d.radius * zb * 1.4;
                    let exp_fx = self.effects.explosion_fx(rad);
                    if !exp_fx.is_empty() {
                        f.vfx.spawn_emitters(exp_fx, d.pos, [0.0, 1.0, 0.0], (rad * 0.45).clamp(0.6, 4.0), None, true);
                    } else {
                        f.vfx.spawn_explosion(d.pos, rad);
                    }
                }
            }
        }

        drop(_sec);
        let _sec = crate::profiler::scope("render.begin.wrecks");
        // Wrecks: remember when each appeared (sinking clock) and bubble while settling in water.
        let mut wreck_count = 0usize;
        let dying_ids: Option<BTreeSet<u32>> = (!f.dying.is_empty())
            .then(|| f.dying.iter().map(|d| d.id).collect());
        for w in world.wrecks() {
            wreck_count += 1;
            let entry = f
                .wrecks_seen
                .entry(w.id)
                .or_insert((if f.primed { t } else { f32::NEG_INFINITY }, frame_no));
            entry.1 = frame_no;
            let since = entry.0;
            if let Some(wl) = water
                && !dying_ids.as_ref().is_some_and(|s| s.contains(&w.id))
                && t - since < WRECK_SINK_SECS
            {
                let wx = w.pos.x as f32 / scale;
                let wz = w.pos.y as f32 / scale;
                if crate::scene::sample_scmap_height(scmap, wx, wz) < wl - 0.5 && (frame_no + u64::from(w.id)) % 2 == 0 {
                    f.vfx.spawn_wake([wx, wl + 0.06, wz], [0.0; 3], 0.9 * zb.sqrt());
                    f.vfx.spawn_bubbles([wx, wl + 0.1, wz]);
                }
            }
        }
        if f.wrecks_seen.len() > wreck_count {
            f.wrecks_seen.retain(|_, (_, sf)| *sf == frame_no);
        }
        // Finished death clips: the wreck (same id) appears now, with a puff of smoke.
        let mut finished = Vec::new();
        f.dying.retain(|d| {
            let keep = t - d.start < d.secs + DEATH_HOLD_SECS;
            if !keep {
                finished.push((d.id, [d.inst.row0[3], d.inst.row1[3], d.inst.row2[3]]));
            }
            keep
        });
        for (id, p) in finished {
            if let Some((s, _)) = f.wrecks_seen.get_mut(&id) {
                *s = t;
            }
            f.vfx.spawn_explosion(add_scaled(p, [0.0, 1.0, 0.0], 0.6 * zb), 0.7 * zb);
        }

        drop(_sec);
        let _sec = crate::profiler::scope("render.begin.proj");
        // Projectiles: arcs, meshes/glows, continuous missile smoke, impacts when they vanish.
        let mut proj_count = 0usize;
        for p in world.projectiles() {
            proj_count += 1;
            let kind = ArcKind::of(p);
            let tt = (p.ticks_lived as f32 / p.total_ticks.max(1) as f32).clamp(0.0, 1.0);
            let (h, dh) = arc_profile(kind, tt);
            let arc_h = p.arc_height as f32 / scale;
            let base = ground(p.pos.x, p.pos.y, 1.2);
            let pos = [base[0], base[1] + h * arc_h, base[2]];
            let hd = [(p.target.x - p.start.x) as f32 / scale, (p.target.y - p.start.y) as f32 / scale];
            let dir = normalize3([hd[0], dh * arc_h, hd[1]]);
            let r = if p.is_nuke { 18.0 } else { (p.splash_radius as f32 / scale).max(0.8) * zb };
            let key = match f.proj_kind.get(&p.id) {
                Some(k) => k.clone(),
                None => {
                    let (k, shooter_tid) = self.resolve_projectile_key(world, p);
                    f.proj_kind.insert(p.id, k.clone());
                    if let Some(tid) = shooter_tid {
                        f.proj_shooter.insert(p.id, tid);
                    }
                    k
                }
            };
            let proj_wfx = f
                .proj_shooter
                .get(&p.id)
                .and_then(|tid| self.effects.unit_weapon_fx(tid));
            let rgb = army_rgb_fast(p.army);
            let hot = [0.55 + rgb[0] * 0.45, 0.55 + rgb[1] * 0.45, 0.55 + rgb[2] * 0.45, 0.95];
            let mesh_len = key.as_ref().and_then(|k| self.proj_meshes.get(k)).map(|m| m.length);
            let (min_len, glow) = if p.is_nuke {
                (5.0, 3.2)
            } else if p.is_missile {
                (2.2, 1.5)
            } else {
                (1.0, 0.9)
            };
            let pscale = zb * mesh_len.map(|l| (min_len / l).max(1.0)).unwrap_or(1.0);
            let len = mesh_len.map(|l| l * pscale).unwrap_or(min_len * zb);
            let tail = add_scaled(pos, dir, -len * 0.5);
            if p.is_overcharge {
                f.glows.push((pos, 2.4 * zb, [0.45, 0.95, 1.0, 0.95]));
                f.streaks.push((add_scaled(pos, dir, -4.0 * zb), pos, 0.9 * zb, [0.35, 0.85, 1.0, 0.85]));
            } else if p.is_missile || p.is_nuke {
                f.glows.push((tail, glow * zb, [1.0, 0.78, 0.40, 0.95]));
                let prev_tail = f
                    .prev_proj
                    .get(&p.id)
                    .map(|(pp, _, _)| add_scaled(*pp, dir, -len * 0.5));
                let fa_trails = proj_wfx.map(|w| w.trail_emitters.as_slice()).unwrap_or(&[]);
                match prev_tail {
                    Some(a) if f.primed => {
                        for q in points_along(a, tail, 0.9 * zb.sqrt(), 16) {
                            if !fa_trails.is_empty() {
                                f.vfx.spawn_emitter(&fa_trails[0], q, [-dir[0], -dir[1], -dir[2]], 0.7 * zb.sqrt(), None, false);
                            } else {
                                f.vfx.spawn_missile_trail(q);
                            }
                        }
                    }
                    _ => {
                        for i in 0..6 {
                            let q = add_scaled(tail, dir, -(i as f32) * 0.8 * zb);
                            if !fa_trails.is_empty() {
                                f.vfx.spawn_emitter(&fa_trails[0], q, [-dir[0], -dir[1], -dir[2]], 0.7 * zb.sqrt(), None, false);
                            } else {
                                f.vfx.spawn_missile_trail(q);
                            }
                        }
                    }
                }
            } else {
                f.glows.push((pos, if key.is_some() { 0.7 } else { 1.1 } * zb, hot));
                if let Some(poly) = proj_wfx.and_then(|w| w.poly_trail.as_ref()) {
                    let tlen = (poly.bp.trail_length * 0.45 * zb).clamp(1.8 * zb, 9.0 * zb);
                    f.trails.push((add_scaled(pos, dir, -tlen), pos, poly.clone(), zb.sqrt()));
                } else {
                    f.streaks.push((add_scaled(pos, dir, -2.6 * zb), pos, 0.45 * zb, [hot[0], hot[1], hot[2], 0.7]));
                }
                if let Some(w) = proj_wfx
                    && !w.trail_emitters.is_empty()
                    && (frame_no + u64::from(p.id)).is_multiple_of(2)
                {
                    f.vfx.spawn_emitter(&w.trail_emitters[0], tail, [-dir[0], -dir[1], -dir[2]], 0.5 * zb.sqrt(), None, false);
                }
            }
            f.prev_proj.insert(p.id, (pos, r, frame_no));
            if key.is_some() {
                f.proj_draw.push(ProjDraw { key, pos, dir, scale: pscale });
            }
        }
        if f.prev_proj.len() > proj_count {
            let mut vanished_proj: Vec<(u32, [f32; 3], f32)> = f
                .prev_proj
                .iter()
                .filter(|(_, (_, _, sf))| *sf != frame_no)
                .map(|(&id, &(pos, r, _))| (id, pos, r))
                .collect();
            vanished_proj.sort_unstable_by_key(|(id, _, _)| *id);
            for (id, pos, r) in vanished_proj {
                f.prev_proj.remove(&id);
                let shooter_tid = f.proj_shooter.remove(&id);
                if f.primed {
                    if r >= 18.0 {
                        if !self.effects.nuke.is_empty() {
                            f.vfx.spawn_emitters(&self.effects.nuke, pos, [0.0, 1.0, 0.0], 2.5, None, true);
                        }
                        f.vfx.spawn_acu_nuke(pos);
                    } else if shields.iter().any(|s| {
                        let dx = pos[0] - s.1[0];
                        let dz = pos[2] - s.1[2];
                        dx * dx + dz * dz <= s.2 * s.2
                    }) {
                        // Absorbed by a dome: the ripple/flash comes from the shield HP diff.
                    } else if let Some(imp) = shooter_tid
                        .as_deref()
                        .and_then(|tid| self.effects.unit_weapon_fx(tid))
                        .filter(|w| !w.impact.is_empty())
                        .map(|w| w.impact.as_slice())
                    {
                        f.vfx.spawn_emitters(imp, pos, [0.0, 1.0, 0.0], (r * 0.65).clamp(0.6, 3.5), None, true);
                    } else {
                        let exp_fx = self.effects.explosion_fx(r);
                        if !exp_fx.is_empty() {
                            f.vfx.spawn_emitters(exp_fx, pos, [0.0, 1.0, 0.0], (r * 0.55).clamp(0.6, 3.5), None, true);
                        } else {
                            f.vfx.spawn_explosion(pos, r);
                        }
                    }
                }
            }
        }
        if f.proj_kind.len() > proj_count {
            f.proj_kind.retain(|id, _| f.prev_proj.contains_key(id));
        }
        if f.proj_shooter.len() > proj_count {
            f.proj_shooter.retain(|id, _| f.prev_proj.contains_key(id));
        }

        // Build beams (`BeamBlueprint`) and build sparks (`EmitterBlueprint`, ~24/s per active builder).
        let spark_n = if f.primed {
            f.spark_accum += dt * 24.0;
            let s = f.spark_accum.floor();
            f.spark_accum -= s;
            s as usize
        } else {
            6
        };
        for b in &builders {
            if let Some(beam) = self.effects.build_beam_fx(b.faction) {
                f.beams.push(BeamDraw {
                    a: b.src,
                    b: b.dst,
                    beam: beam.clone(),
                    width_scale: zb.sqrt(),
                    tint: Some(b.rgb),
                });
            }
            let bfx = self.effects.build_sparks_fx(b.faction);
            if !bfx.is_empty() {
                if spark_n > 0 {
                    f.vfx.spawn_emitters(bfx, b.dst, [0.0, 1.0, 0.0], 0.55 * zb.sqrt(), Some(b.rgb), false);
                }
            } else {
                for _ in 0..spark_n.min(8) {
                    f.vfx.spawn_build_spark(b.dst, b.rgb);
                }
            }
        }

        drop(_sec);
        let _sec = crate::profiler::scope("render.begin.vfx");
        // A fresh renderer (snapshot) advances newly seeded particles into mid-life.
        f.vfx.step(if f.primed { dt } else { 0.05 });
        f.primed = true;
    }

    /// Shield sphere `(center, radius)`: bubbles sit on the generator (lower half in the ground),
    /// personal shields wrap the unit and scale with its render size.
    fn shield_sphere(&self, u: &crate::skirmish::Unit, scmap: &crate::scmap::ScMap, t: f32, zb: f32) -> ([f32; 3], f32) {
        let scale = crate::sim::SCALE as f32;
        let wx = u.pos.x as f32 / scale;
        let wz = u.pos.y as f32 / scale;
        let gy = crate::play::sample_terrain_y(scmap, wx, wz) + unit_lift_at(u, t);
        let r = u.shield_radius as f32 / scale;
        if self.fx_ref(&u.type_id).personal_shield {
            let rr = r.max((u.radius as f32 / scale) * 1.3) * (zb * 1.15).max(1.0);
            ([wx, gy + rr * 0.35, wz], rr)
        } else {
            ([wx, gy, wz], r)
        }
    }

    /// Poses a unit: aircraft bank/bob along their flight heading, ships sway, then animated rigs
    /// queue a blended clip pose with turret aim + recoil. The returned instance is pushed by the
    /// caller at `batch_vecs[batch_idx][slot]` (`target`); [`Self::resolve_pose_jobs`] later appends
    /// the bone palettes (in queue order) and sets `extra.x`. `target = None` marks a frustum-culled
    /// unit: render history still advances, but no pose is sampled or uploaded.
    pub fn skin_instance(
        &self,
        u: &crate::skirmish::Unit,
        body_yaw: f32,
        mut inst: GpuUnitInstance,
        target: Option<(usize, usize)>,
    ) -> GpuUnitInstance {
        use crate::flowfield::NavDomain;
        use crate::sca::{AimPose, AnimMode};
        let mut fr = self.frame.borrow_mut();
        let f = &mut *fr;
        let (t, dt, frame_no) = (f.time, f.dt, f.frame_no);
        let Some(st) = f.units.get_mut(&u.id) else {
            return inst;
        };

        // 1. Hull orientation: flight heading + bank for aircraft, slow roll/pitch sway for ships.
        let mut yaw = body_yaw;
        let reorient = match u.domain {
            NavDomain::Air => {
                if st.heading_init {
                    yaw = st.heading;
                }
                Some((0.0, st.bank))
            }
            NavDomain::Naval => {
                if st.heading_init {
                    yaw = st.heading;
                }
                let ph = unit_anim_phase(u.id) * std::f32::consts::TAU;
                Some(((t * 0.7 + ph).sin() * 0.012, (t * 0.9 + ph * 1.3).sin() * 0.03))
            }
            _ => None,
        };
        if let Some((pitch, roll)) = reorient {
            let pos = [inst.row0[3], inst.row1[3], inst.row2[3]];
            let s = (inst.row0[0] * inst.row0[0] + inst.row1[0] * inst.row1[0] + inst.row2[0] * inst.row2[0]).sqrt();
            let (l, up, fw) = ypr_basis(yaw, pitch, roll);
            inst = GpuUnitInstance::from_basis(pos, l, up, fw, [s; 3], inst.team_color, inst.flags).with_extra(inst.extra);
        }
        st.last_inst = Some(inst);
        st.last_inst_frame = frame_no;

        let Some(asset_idx) = st.asset_idx else {
            return inst;
        };
        let asset = &self.assets[asset_idx];
        let Some(rig) = asset.rig.as_ref() else {
            return inst;
        };
        let fx = &asset.fx;

        // 2. Clip crossfade bookkeeping.
        let mode = rig.mode_for(u.moving);
        if st.mode != Some(mode) {
            match st.mode {
                Some(prev) => {
                    st.prev_mode = prev;
                    st.mode_since = t;
                }
                None => {
                    st.prev_mode = mode;
                    st.mode_since = f32::NEG_INFINITY;
                }
            }
            st.mode = Some(mode);
        }
        let w = crate::sca::crossfade_weight(t - st.mode_since, ANIM_BLEND_SECS);

        // 3. Turret slew toward the held target (relative to the hull), recoil kick.
        let (ty, tp) = st
            .aim_target
            .map(|(yw, p)| (crate::sca::wrap_angle_pi(yw - yaw), p))
            .unwrap_or((0.0, 0.0));
        if st.aim_init {
            st.aim_yaw = crate::sca::slew_angle(st.aim_yaw, ty, fx.yaw_speed * dt);
            st.aim_pitch = approach(st.aim_pitch, tp, fx.pitch_speed * dt);
        } else {
            st.aim_yaw = ty;
            st.aim_pitch = tp;
            st.aim_init = true;
        }
        let aim = AimPose {
            yaw: st.aim_yaw,
            pitch: st.aim_pitch,
            recoil: crate::sca::recoil_curve(t - st.shot_at, RECOIL_RETURN_SECS),
        };
        let (work_w, work_t) = (st.work_w, st.work_t);
        let blending = w < 1.0 && st.prev_mode != mode;
        if mode == AnimMode::Rest && !blending && work_w <= 1e-3 && aim.is_neutral() {
            return inst;
        }
        // Off-screen (culled) units keep their animation state but sample no pose.
        let Some((batch_idx, slot)) = target else {
            return inst;
        };

        // 4. Pose = blend(prev clip, clip) -> blend(work clip) -> aim/recoil -> palette (deferred).
        f.pose_jobs.push(PoseJob {
            asset_idx,
            mode,
            prev_mode: st.prev_mode,
            blending,
            w,
            t,
            phase: unit_anim_phase(u.id),
            work_w,
            work_t,
            aim,
        });
        f.pose_targets.push((batch_idx, slot));
        inst
    }

    /// Samples every queued pose (in parallel for large battles), appends the palettes in queue
    /// order — the same layout the inline path produced — and patches each instance's `extra.x`.
    pub fn resolve_pose_jobs(&self, batch_vecs: &mut [Vec<GpuUnitInstance>]) {
        let _p = crate::profiler::scope("render.poses");
        let (jobs, targets) = {
            let mut f = self.frame.borrow_mut();
            (std::mem::take(&mut f.pose_jobs), std::mem::take(&mut f.pose_targets))
        };
        if jobs.is_empty() {
            return;
        }
        let palettes: Vec<Vec<[[f32; 4]; 4]>> = if jobs.len() >= PARALLEL_POSE_MIN {
            let pool = self
                .pose_pool
                .get_or_init(|| Arc::new(crate::jobs::WorkerPool::new(crate::jobs::default_thread_count())));
            let assets = Arc::clone(&self.assets);
            pool.par_map_ordered_arc(Arc::new(jobs), Arc::new(move |_: usize, j: &PoseJob| compute_pose_palette(&assets, j)))
        } else {
            jobs.iter().map(|j| compute_pose_palette(&self.assets, j)).collect()
        };
        let mut f = self.frame.borrow_mut();
        for ((batch_idx, slot), mats) in targets.into_iter().zip(palettes) {
            if mats.is_empty() {
                continue;
            }
            let base = f.palette.len();
            f.palette.extend_from_slice(&mats);
            if let Some(inst) = batch_vecs.get_mut(batch_idx).and_then(|v| v.get_mut(slot)) {
                inst.extra[0] = base as f32 + 1.0;
            }
        }
    }

    /// Wreck render adjustment `(dy, extra_tilt)`; `None` hides the wreck while its unit is still
    /// playing a death clip. Wrecks over deep water sink from the surface to the sea floor.
    pub fn wreck_pose(&self, w: &crate::skirmish::Wreck, scmap: &crate::scmap::ScMap) -> Option<(f32, f32)> {
        let f = self.frame.borrow();
        if f.dying.iter().any(|d| d.id == w.id) {
            return None;
        }
        if !scmap.water.has_water {
            return Some((0.0, 0.0));
        }
        let scale = crate::sim::SCALE as f32;
        let (wx, wz) = (w.pos.x as f32 / scale, w.pos.y as f32 / scale);
        let floor = crate::scene::sample_scmap_height(scmap, wx, wz);
        let wl = scmap.water.elevation;
        if floor >= wl - 0.5 {
            return Some((0.0, 0.0));
        }
        let since = f.wrecks_seen.get(&w.id).map(|&(s, _)| s).unwrap_or(f32::NEG_INFINITY);
        let s = sink_fraction(f.time - since, WRECK_SINK_SECS);
        Some(((floor - wl) * s, 0.35 * s))
    }

    /// Appends this frame's extra mesh instances: death-clip poses, construction build-up ghosts
    /// (`extra.w` = progress, `extra.y` = template height) and projectile meshes.
    #[allow(clippy::too_many_arguments)]
    pub fn append_frame_instances(
        &self,
        world: &crate::skirmish::Skirmish,
        scmap: &crate::scmap::ScMap,
        viewer: Option<crate::skirmish::ArmyId>,
        zoom_boost: f32,
        strategic_zoom: bool,
        batches: &mut BTreeMap<String, Vec<GpuUnitInstance>>,
        has_texture: &dyn Fn(&str) -> bool,
    ) {
        let mut fr = self.frame.borrow_mut();
        let f = &mut *fr;
        let t = f.time;
        if !strategic_zoom {
            // Death clips.
            for d in &f.dying {
                let Some(rig) = self.asset(&d.type_id).and_then(|a| a.rig.as_ref()) else {
                    continue;
                };
                let Some(clip) = rig.death.get(d.clip) else {
                    continue;
                };
                let ct = ((t - d.start) * clip.rate).clamp(0.0, clip.clip.duration());
                let pose = rig.sample_pose(Some(&clip.clip), ct, false);
                let mats = rig.palette_from_pose(&pose, crate::sca::AimPose::default());
                let mut inst = d.inst;
                if !mats.is_empty() {
                    inst.extra[0] = f.palette.len() as f32 + 1.0;
                    f.palette.extend_from_slice(&mats);
                }
                batches.entry(d.type_id.clone()).or_default().push(inst);
            }

            // Construction ghosts rising out of the ground (deduplicated per site).
            let scale = crate::sim::SCALE as f32;
            let zb = zoom_boost.max(1.0);
            let mut sites = BTreeSet::new();
            for u in world.units() {
                let Some(job) = u.build.as_ref().filter(|_| u.alive) else {
                    continue;
                };
                if viewer.is_some_and(|a| world.intel_visibility(a, u) != crate::skirmish::IntelVisibility::Visible) {
                    continue;
                }
                let Some(jt) = world.catalog().get(&job.unit_type) else {
                    continue;
                };
                if !sites.insert((job.x, job.y, job.unit_type.clone())) {
                    continue;
                }
                let prog = (job.progress_permille() as f32 / 1000.0).clamp(0.02, 0.98);
                let (wx, wz) = (job.x as f32 / scale, job.y as f32 / scale);
                let wy = crate::play::sample_terrain_y(scmap, wx, wz);
                let rs = if jt.role.is_structure() { zb.powf(0.45) } else { zb * 1.15 };
                let rgb = crate::play::army_rgb(world, u.army);
                let height = self.asset(&job.unit_type).map(|a| a.height).unwrap_or(2.0);
                let tex = if has_texture(&job.unit_type) { 1.0 } else { 0.0 };
                batches.entry(job.unit_type.clone()).or_default().push(
                    GpuUnitInstance::from_pos_yaw_scale([wx, wy, wz], 0.0, 0.0, rs, [rgb[0], rgb[1], rgb[2], 0.45], [tex, 0.0, 0.0, 0.85])
                        .with_extra([0.0, height, 0.0, prog]),
                );
            }
        }

        // Projectile meshes oriented along their flight direction.
        for p in &f.proj_draw {
            let Some(key) = &p.key else {
                continue;
            };
            let yaw = p.dir[0].atan2(p.dir[2]);
            let pitch_up = p.dir[1].clamp(-1.0, 1.0).asin();
            let (l, up, fw) = ypr_basis(yaw, pitch_up, 0.0);
            batches.entry(key.clone()).or_default().push(GpuUnitInstance::from_basis(
                p.pos,
                l,
                up,
                fw,
                [p.scale; 3],
                [0.9, 0.9, 0.9, 0.6],
                [1.0, 0.0, 0.0, 0.0],
            ));
        }
    }

    /// Queues a strategic icon for `u` (far zoom). Returns `false` when no atlas is loaded.
    pub fn push_strategic_icon(
        &self,
        u: &crate::skirmish::Unit,
        pos: [f32; 3],
        rgb: [f32; 3],
        selected: bool,
    ) -> bool {
        let Some(atlas) = &self.icon_atlas else {
            return false;
        };
        let name = self
            .icon_names
            .get(&u.type_id)
            .map(String::as_str)
            .unwrap_or("icon_land1_directfire");
        let uv = atlas.uv(name, selected);
        let px = if u.role == crate::skirmish::Role::Commander {
            30.0
        } else if u.role.is_structure() {
            20.0
        } else {
            22.0
        };
        let c = if selected { [1.0, 1.0, 1.0, 1.0] } else { [rgb[0], rgb[1], rgb[2], 1.0] };
        self.frame.borrow_mut().icons.push((pos, uv, c, px));
        true
    }

    /// Pushes this frame's bone palette, particles + projectile glows, shield domes, strategic
    /// icons and FoW mask to the GPU.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload(
        &self,
        device: &wgpu::Device,
        resources: &mut GpuSceneResources,
        world: &crate::skirmish::Skirmish,
        scmap: &crate::scmap::ScMap,
        fow_viewer: Option<crate::skirmish::ArmyId>,
        camera: &OrbitCamera,
        width: u32,
        height: u32,
    ) {
        let queue = resources.queue();
        let mut f = self.frame.borrow_mut();
        let f = &mut *f;

        if !f.palette.is_empty() {
            resources.set_bone_palette(device, &queue, &f.palette);
        }

        if !f.vfx_tex_uploaded
            && let Some(ref arr) = self.effects.texture_array
        {
            resources.set_vfx_texture_array(device, &queue, arr.width, arr.height, &arr.layers);
            f.vfx_tex_uploaded = true;
        }

        let view = camera.view_matrix();
        let right = [view[0][0], view[1][0], view[2][0]];
        let up = [view[0][1], view[1][1], view[2][1]];
        let (mut vv, mut vi) = f.vfx.build_billboards(right, up);
        let eye = camera.eye_position();
        for &(a, b, w, c) in &f.streaks {
            crate::vfx::append_streak(&mut vv, &mut vi, a, b, w, c, eye);
        }
        for &(p, s, c) in &f.glows {
            crate::vfx::append_glow(&mut vv, &mut vi, p, s, c, right, up);
        }
        for bd in &f.beams {
            crate::vfx::append_fa_beam(&mut vv, &mut vi, bd.a, bd.b, &bd.beam, bd.width_scale, f.time, eye, bd.tint);
        }
        for (tail, head, trail, wscale) in &f.trails {
            crate::vfx::append_fa_trail(&mut vv, &mut vi, *tail, *head, trail, *wscale, f.time, eye);
        }
        resources.set_vfx(device, &vv, &vi);
        resources.set_shield_instances(device, &f.shields);

        if let Some(atlas) = &self.icon_atlas {
            if !f.icons_uploaded {
                resources.set_strategic_icons_texture(device, &queue, &atlas.image);
                f.icons_uploaded = true;
            }
            let (iv, ii) = build_icon_quads(&f.icons, &camera.view_proj_matrix(), width, height);
            resources.set_hud_strategic_icons(device, &iv, &ii);
        }

        let mask = match fow_viewer {
            Some(army) => {
                let scale = crate::sim::SCALE as f32;
                let circles: Vec<(f32, f32, f32, f32)> = world
                    .units()
                    .filter(|u| u.alive && world.are_allies(army, u.army))
                    .map(|u| {
                        (
                            u.pos.x as f32 / scale,
                            u.pos.y as f32 / scale,
                            u.vision_radius as f32 / scale,
                            u.radar_radius as f32 / scale,
                        )
                    })
                    .collect();
                build_fow_mask(scmap.heightmap_width as f32, scmap.heightmap_height as f32, &circles)
            }
            None => vec![255u8; 128 * 128 * 4],
        };
        let digest = fnv1a_extend(FNV_OFFSET_BASIS, &mask);
        if f.fow_digest != Some(digest) {
            resources.update_fow_texture(&queue, &mask);
            f.fow_digest = Some(digest);
        }
    }
}

/// CPU-only snapshot warm-up: builds (and discards) the render frame for each of the last
/// [`SNAPSHOT_WARMUP_TICKS`] ticks before `final_tick`, so diff-derived effects (death clips,
/// shield ripples, recoil, banking, contrails, wakes, sinking wrecks, trails) have history when
/// the PNG frame is rendered. Never touches the simulation.
pub fn snapshot_warm_up(
    session: &crate::play::PlaySession,
    scmap: &ScMap,
    meshes: &crate::play::UnitMeshCache,
    aspect: f32,
    final_tick: u64,
) {
    session.decals.borrow_mut().observe(&session.world);
    if session.world.tick() + SNAPSHOT_WARMUP_TICKS < final_tick {
        return;
    }
    let cam = crate::play::battle_snapshot_camera(&session.world, scmap, aspect);
    let _ = crate::play::build_skirmish_render_frame(session, scmap, meshes, &cam);
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-6 { [v[0] / l, v[1] / l, v[2] / l] } else { [0.0, 1.0, 0.0] }
}

/// Projects world-space icon anchors to NDC quads (`px` pixels wide) for the HUD pass.
fn build_icon_quads(
    icons: &[([f32; 3], [f32; 4], [f32; 4], f32)],
    vp: &[[f32; 4]; 4],
    width: u32,
    height: u32,
) -> (Vec<crate::viewer::HudVertex>, Vec<u32>) {
    let mut v = Vec::with_capacity(icons.len() * 4);
    let mut idx = Vec::with_capacity(icons.len() * 6);
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    for &(p, uv, color, px) in icons {
        let cx = vp[0][0] * p[0] + vp[1][0] * p[1] + vp[2][0] * p[2] + vp[3][0];
        let cy = vp[0][1] * p[0] + vp[1][1] * p[1] + vp[2][1] * p[2] + vp[3][1];
        let cw = vp[0][3] * p[0] + vp[1][3] * p[1] + vp[2][3] * p[2] + vp[3][3];
        if cw <= 1e-4 {
            continue;
        }
        let (nx, ny) = (cx / cw, cy / cw);
        if nx.abs() > 1.1 || ny.abs() > 1.1 {
            continue;
        }
        let hx = px / w;
        let hy = px / h;
        let base = v.len() as u32;
        for (dx, dy, u, vv) in [
            (-hx, hy, uv[0], uv[1]),
            (hx, hy, uv[2], uv[1]),
            (hx, -hy, uv[2], uv[3]),
            (-hx, -hy, uv[0], uv[3]),
        ] {
            v.push(crate::viewer::HudVertex {
                pos: [nx + dx, ny + dy],
                uv: [u, vv],
                color,
            });
        }
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (v, idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SCALE;

    fn make_unit(id: u32, wx: f32, wz: f32, hp: i64, max_hp: i64) -> SimUnitSnapshot {
        SimUnitSnapshot {
            id,
            blueprint_id: "UAL0001".to_string(),
            x: (wx * SCALE as f32).round() as i64,
            y: (wz * SCALE as f32).round() as i64,
            hp,
            max_hp,
        }
    }

    #[test]
    fn test_deterministic_parallel_vs_single_thread_512_units() {
        let view_proj = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.2, 1.0],
            [0.0, 0.0, 10.0, 50.0],
        ];
        let camera_eye = [0.0, 20.0, -50.0];

        let units: Vec<SimUnitSnapshot> = (0..512u32)
            .map(|i| {
                let wx = ((i as i32 % 32) - 16) as f32 * 8.0;
                let wz = ((i as i32 / 32) - 4) as f32 * 25.0;
                make_unit(i + 1, wx, wz, 50 + (i as i64 % 50), 100)
            })
            .collect();

        let frame_1 = prepare_render_frame(&units, None, view_proj, camera_eye, 1);
        let frame_4 = prepare_render_frame(&units, None, view_proj, camera_eye, 4);

        assert_eq!(frame_1, frame_4);
        assert_eq!(frame_1.digest_fnv1a, frame_4.digest_fnv1a);
        assert!(!frame_1.instances.is_empty());
        assert!(frame_1.culled_count > 0);
        assert_eq!(
            frame_1.instances.len() + frame_1.culled_count,
            units.len()
        );
    }

    #[test]
    fn test_frustum_culling_behind_camera_and_visible() {
        let view_proj = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let camera_eye = [0.0, 0.0, 0.0];

        let units = vec![
            make_unit(1, 0.0, 20.0, 100, 100),
            make_unit(2, 0.0, -10.0, 100, 100),
            make_unit(3, 0.0, 0.0, 100, 100),
            make_unit(4, 100.0, 10.0, 100, 100),
        ];

        let frame = prepare_render_frame(&units, None, view_proj, camera_eye, 2);
        assert_eq!(frame.instances.len(), 1);
        assert_eq!(frame.culled_count, 3);
        assert_eq!(frame.instances[0].unit_id, 1);
    }

    #[test]
    fn test_lod_assignment_by_distance() {
        let view_proj = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1000.0],
        ];
        let camera_eye = [0.0, 0.0, 0.0];

        let units = vec![
            make_unit(10, 50.0, 0.0, 80, 100),
            make_unit(20, 150.0, 0.0, 50, 100),
            make_unit(30, 300.0, 0.0, 25, 100),
        ];

        let frame = prepare_render_frame(&units, None, view_proj, camera_eye, 1);
        assert_eq!(frame.culled_count, 0);
        assert_eq!(frame.instances.len(), 3);
        assert_eq!(frame.instances[0].lod, 0);
        assert_eq!(frame.instances[1].lod, 1);
        assert_eq!(frame.instances[2].lod, 2);
        assert_eq!(frame.lod_counts, [1, 1, 1]);
        assert!((frame.instances[0].health_ratio - 0.8).abs() < 1e-6);
    }

    #[test]
    fn living_normalization_matches_template_space() {
        let (off, s) = normalization_for_bounds([-2.0, 1.0, -4.0], [2.0, 5.0, 4.0], 4.0);
        assert_eq!(off, [0.0, 1.0, 0.0]);
        assert!((s - 0.5).abs() < 1e-6);
        // Degenerate bounds never divide by zero.
        let (_, s2) = normalization_for_bounds([0.0; 3], [0.0; 3], 2.0);
        assert!(s2.is_finite() && s2 > 0.0);
    }

    #[test]
    fn living_lod1_vertex_bones_remap_by_name() {
        let lod1 = ["Root", "Leg_R", "Leg_L"];
        let lod0 = ["root", "Torso", "leg_l", "LEG_R"];
        let out = remap_vertex_bones(&lod1, &lod0, &[0, 1, 2, 7]);
        assert_eq!(out, vec![0.0, 3.0, 2.0, 0.0]);
    }

    #[test]
    fn living_shot_and_death_diffing() {
        assert!(is_new_shot(100, None, 102));
        assert!(!is_new_shot(90, None, 102));
        assert!(is_new_shot(101, Some(100), 140));
        assert!(!is_new_shot(100, Some(100), 101));

        let mut prev = BTreeMap::new();
        prev.insert(1u32, 10);
        prev.insert(2u32, 20);
        prev.insert(3u32, 30);
        let mut now = BTreeMap::new();
        now.insert(2u32, 21);
        now.insert(4u32, 40);
        assert_eq!(vanished(&prev, &now), vec![(1, 10), (3, 30)]);

        for id in [0u32, 1, 7, 123_456, u32::MAX] {
            let p = unit_anim_phase(id);
            assert!((0.0..1.0).contains(&p));
        }
        assert_ne!(unit_anim_phase(1), unit_anim_phase(2));
    }

    #[test]
    fn living_fow_mask_marks_vision_and_radar() {
        // 256x256 map, 2 world units per cell; vision r=10 at (64,64), radar r=40 there too.
        let mask = build_fow_mask(256.0, 256.0, &[(64.0, 64.0, 10.0, 40.0)]);
        assert_eq!(mask.len(), 128 * 128 * 4);
        let at = |x: f32, z: f32| mask[((z / 2.0) as usize * 128 + (x / 2.0) as usize) * 4];
        assert_eq!(at(64.0, 64.0), 255);
        assert_eq!(at(64.0 + 25.0, 64.0), 110);
        assert_eq!(at(200.0, 200.0), 0);
        assert!(mask.chunks(4).all(|p| p[3] == 255));
        // Circles hanging off the map edge clamp safely.
        let edge = build_fow_mask(256.0, 256.0, &[(-5.0, 300.0, 30.0, 0.0)]);
        assert_eq!(edge.len(), 128 * 128 * 4);
    }

    #[test]
    fn living_icon_quads_project_and_cull() {
        let cam = crate::camera::OrbitCamera::for_map(256.0, 256.0, 0.0, 1.0);
        let vp = cam.view_proj_matrix();
        let icons = vec![
            (cam.target, [0.0, 0.0, 0.5, 0.5], [1.0; 4], 20.0),
            ([cam.target[0] + 1.0e5, 0.0, cam.target[2]], [0.0; 4], [1.0; 4], 20.0),
        ];
        let (v, i) = build_icon_quads(&icons, &vp, 400, 400);
        assert_eq!(v.len(), 4);
        assert_eq!(i, vec![0, 1, 2, 0, 2, 3]);
        let w = v[1].pos[0] - v[0].pos[0];
        assert!((w - 0.1).abs() < 1e-4, "20px of 400px = 0.1 NDC, got {w}");
        assert!(v[0].pos[0].abs() < 0.1 && v[0].pos[1].abs() < 0.1);
    }
}
