//! Phases 6–10 playable skirmish viewer & interactive controller.
//!
//! Features:
//! - Baked real Forged Alliance `.scmap` terrain strata (`env.scd`) + water overlay
//! - Hardware-instanced `.scm` unit meshes with `_Albedo` + `_SpecTeam` team-color masks
//! - Strategic zoom crossfade (`distance > 340` switches to crisp FA role icons)
//! - In-window 2D HUD (`src/hud.rs`): economy gauges, clickable build/produce palette, minimap
//! - Full Phase 8 gameplay: Engineers, wrecks/reclaim, repair, assist, Overcharge, ballistic
//!   artillery shells, Point Defense, Radar & Fog-of-War (`F2`)
//! - Phase 9 replay playback (`PlaySession::new_replay`)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::dds_decode::{RgbaImage, decode_dds_rgba8};
use crate::hud::{self, HudAction, HudLayout};
use crate::replay::Replay;
use crate::scd;
use crate::scm::ScmMesh;
use crate::scmap::ScMap;
use crate::scene::{self, GpuVertex, LoadedScene};
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{
    ArmyId, Catalog, Command, GameResult, IntelVisibility, MapContext, Role, Skirmish, UnitId,
    UnitOrder,
};
use crate::skirmish_ai::ScriptedAi;
use crate::viewer::{self, DEPTH_FORMAT, GpuMeshTemplate, GpuSceneResources, GpuUnitInstance, MSAA_SAMPLES};

// ---------------------------------------------------------------------------
// Terrain sampling & mouse-ray picking
// ---------------------------------------------------------------------------

pub fn sample_terrain_y(scmap: &ScMap, wx: f32, wz: f32) -> f32 {
    let hm_w = scmap.heightmap_width;
    let hm_h = scmap.heightmap_height;
    if hm_w == 0 || hm_h == 0 {
        return 0.0;
    }
    let mw = if scmap.map_width > 0.0 {
        scmap.map_width
    } else {
        hm_w as f32
    };
    let mh = if scmap.map_height > 0.0 {
        scmap.map_height
    } else {
        hm_h as f32
    };
    let gx = ((wx / mw) * hm_w as f32).clamp(0.0, hm_w as f32);
    let gz = ((wz / mh) * hm_h as f32).clamp(0.0, hm_h as f32);
    let x0 = (gx.floor() as u32).min(hm_w);
    let z0 = (gz.floor() as u32).min(hm_h);
    let x1 = (x0 + 1).min(hm_w);
    let z1 = (z0 + 1).min(hm_h);
    let tx = (gx - x0 as f32).clamp(0.0, 1.0);
    let tz = (gz - z0 as f32).clamp(0.0, 1.0);

    let h00 = scmap.height_at(x0, z0).unwrap_or(0.0);
    let h10 = scmap.height_at(x1, z0).unwrap_or(h00);
    let h01 = scmap.height_at(x0, z1).unwrap_or(h00);
    let h11 = scmap.height_at(x1, z1).unwrap_or(h00);

    let h0 = h00 + (h10 - h00) * tx;
    let h1 = h01 + (h11 - h01) * tx;
    let terrain = h0 + (h1 - h0) * tz;
    if scmap.water.has_water {
        terrain.max(scmap.water.elevation)
    } else {
        terrain
    }
}

pub fn pick_terrain_point(
    camera: &OrbitCamera,
    scmap: &ScMap,
    px: f32,
    py: f32,
    width: u32,
    height: u32,
) -> Option<[f32; 3]> {
    if width == 0 || height == 0 {
        return None;
    }
    let ndc_x = (2.0 * px / width as f32) - 1.0;
    let ndc_y = 1.0 - (2.0 * py / height as f32);
    let (eye, dir) = camera.screen_ray(ndc_x, ndc_y);
    let (min_h, max_h, avg_h) = scmap.height_stats();
    let surface_min = if scmap.water.has_water {
        min_h.max(scmap.water.elevation)
    } else {
        min_h
    };
    let surface_max = max_h.max(surface_min + 0.5);

    if dir[1].abs() <= 1e-6 {
        return camera.ray_plane_y(ndc_x, ndc_y, avg_h);
    }

    let t_a = (surface_max - eye[1]) / dir[1];
    let t_b = (surface_min - eye[1]) / dir[1];
    let mut t_start = t_a.min(t_b).max(1.0);
    let mut t_end = t_a.max(t_b).max(t_start + 1.0);
    let max_t = camera.z_far.max(camera.distance * 4.0);
    t_end = t_end.min(max_t);
    t_start = t_start.min(t_end);

    let mw = scmap.map_width.max(1.0);
    let mh = scmap.map_height.max(1.0);
    let eval = |t: f32| -> (f32, [f32; 3]) {
        let p = [eye[0] + dir[0] * t, eye[1] + dir[1] * t, eye[2] + dir[2] * t];
        let ty = sample_terrain_y(scmap, p[0].clamp(0.0, mw), p[2].clamp(0.0, mh));
        (p[1] - ty, p)
    };

    const STEPS: usize = 64;
    let mut prev_t = t_start;
    let (mut prev_diff, _) = eval(prev_t);
    if prev_diff <= 0.0 {
        return camera.ray_plane_y(ndc_x, ndc_y, surface_max);
    }

    for i in 1..=STEPS {
        let frac = i as f32 / STEPS as f32;
        let t = t_start + (t_end - t_start) * frac;
        let (diff, _) = eval(t);
        if diff <= 0.0 && prev_diff >= 0.0 {
            let mut lo = prev_t;
            let mut hi = t;
            for _ in 0..8 {
                let mid = 0.5 * (lo + hi);
                if eval(mid).0 > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let (_, mut hit) = eval(0.5 * (lo + hi));
            hit[0] = hit[0].clamp(0.0, mw);
            hit[2] = hit[2].clamp(0.0, mh);
            hit[1] = sample_terrain_y(scmap, hit[0], hit[2]);
            return Some(hit);
        }
        prev_t = t;
        prev_diff = diff;
    }

    let mut fallback = camera.ray_plane_y(ndc_x, ndc_y, camera.target[1])?;
    fallback[0] = fallback[0].clamp(0.0, mw);
    fallback[2] = fallback[2].clamp(0.0, mh);
    fallback[1] = sample_terrain_y(scmap, fallback[0], fallback[2]);
    Some(fallback)
}

fn world_f32_to_milli(p: [f32; 3]) -> (i64, i64) {
    (
        (p[0] as f64 * SCALE as f64).round() as i64,
        (p[2] as f64 * SCALE as f64).round() as i64,
    )
}

fn pick_radius_for_camera(camera: &OrbitCamera) -> i64 {
    let meters = (camera.distance * 0.025).clamp(2.5, 16.0) as f64;
    (meters * SCALE as f64).round() as i64
}

// ---------------------------------------------------------------------------
// Real .scm mesh + unit Albedo/SpecTeam texture cache
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NormalizedMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub albedo: Option<RgbaImage>,
}

#[derive(Debug, Clone, Default)]
pub struct UnitMeshCache {
    by_type: BTreeMap<String, NormalizedMesh>,
    fallback: Vec<(Role, NormalizedMesh)>,
    /// Phase 11 render-only rigs/clips, LOD1 meshes, VFX director, FoW mask & strategic icons.
    pub(crate) living: crate::render_prep::LivingBattlefield,
}

impl UnitMeshCache {
    /// Loads normalized `.scm` meshes (`LOD1`/`LOD0`) and `_Albedo`+`_SpecTeam` textures from
    /// `units.scd` (with `textures.scd` fallback) in parallel, concurrently with [`LivingBattlefield::load`].
    pub fn load(game_dir: Option<&Path>, catalog: &Catalog) -> Self {
        let entries: Vec<(String, Role, f32)> = catalog
            .types
            .iter()
            .map(|(id, utype)| {
                let target_span = (utype.size[0].max(utype.size[2]) as f32 / SCALE as f32).max(1.6);
                (id.clone(), utype.role, target_span)
            })
            .collect();

        let (by_type, living) = std::thread::scope(|scope| {
            let living_job = scope.spawn(|| crate::render_prep::LivingBattlefield::load(game_dir, catalog));

            let by_type: BTreeMap<String, NormalizedMesh> = if let Some(dir) = game_dir {
                let gamedata = scd::gamedata_dir(dir);
                let units_scd = gamedata.join("units.scd");
                let norm_key = |n: &str| n.replace('\\', "/").trim_start_matches('/').to_ascii_lowercase();
                let mut wanted = std::collections::HashSet::with_capacity(entries.len() * 4);
                for (id, _, _) in &entries {
                    let l = id.to_ascii_lowercase();
                    for suf in ["_lod0.scm", "_lod1.scm", "_albedo.dds", "_specteam.dds"] {
                        wanted.insert(format!("units/{l}/{l}{suf}"));
                    }
                }
                let mut files: std::collections::HashMap<String, Vec<u8>> =
                    scd::read_scd_matching(&units_scd, |n| wanted.contains(&norm_key(n)))
                        .map(|v| v.into_iter().map(|(k, b)| (norm_key(&k), b)).collect())
                        .unwrap_or_default();

                let missing_albedo: std::collections::HashSet<String> = entries
                    .iter()
                    .filter_map(|(id, _, _)| {
                        let l = id.to_ascii_lowercase();
                        let p = format!("units/{l}/{l}_albedo.dds");
                        (!files.contains_key(&p)).then_some(p)
                    })
                    .collect();
                if !missing_albedo.is_empty() {
                    let textures_scd = gamedata.join("textures.scd");
                    if let Ok(extra) =
                        scd::read_scd_matching(&textures_scd, |n| missing_albedo.contains(&norm_key(n)))
                    {
                        for (k, b) in extra {
                            files.entry(norm_key(&k)).or_insert(b);
                        }
                    }
                }

                let threads = crate::jobs::default_thread_count();
                crate::jobs::par_map_ordered(&entries, threads, |_, (id, role, target_span)| {
                    let l = id.to_ascii_lowercase();
                    let mut loaded = None;
                    for suf in ["_lod0.scm", "_lod1.scm"] {
                        if let Some(bytes) = files.get(&format!("units/{l}/{l}{suf}"))
                            && let Ok(scm) = ScmMesh::parse(bytes)
                            && let Some(mut norm) = normalize_scm(&scm, *target_span)
                        {
                            norm.albedo = files
                                .get(&format!("units/{l}/{l}_albedo.dds"))
                                .and_then(|b| crate::dds_decode::decode_dds_rgba8(b).ok())
                                .map(|albedo| {
                                    let spec = files
                                        .get(&format!("units/{l}/{l}_specteam.dds"))
                                        .and_then(|b| crate::dds_decode::decode_dds_rgba8(b).ok());
                                    scene::pack_albedo_team_mask(albedo, spec.as_ref())
                                });
                            loaded = Some(norm);
                            break;
                        }
                    }
                    (id.clone(), loaded.unwrap_or_else(|| procedural_role_mesh(*role, *target_span)))
                })
                .into_iter()
                .collect()
            } else {
                entries
                    .iter()
                    .map(|(id, role, target_span)| (id.clone(), procedural_role_mesh(*role, *target_span)))
                    .collect()
            };

            let living = living_job
                .join()
                .unwrap_or_else(|_| crate::render_prep::LivingBattlefield::load(None, catalog));
            (by_type, living)
        });

        let fallback = [
            Role::Commander,
            Role::Factory,
            Role::Extractor,
            Role::Generator,
            Role::PointDefense,
            Role::Radar,
            Role::Engineer,
            Role::Mobile,
            Role::Wall,
        ]
        .into_iter()
        .map(|r| (r, procedural_role_mesh(r, match r { Role::Factory => 5.0, Role::Wall => 1.0, _ => 1.8 })))
        .collect();

        Self { by_type, fallback, living }
    }

    pub fn mesh_for(&self, type_id: &str, role: Role) -> &NormalizedMesh {
        self.by_type.get(type_id).unwrap_or_else(|| {
            let target = match role {
                Role::AirFactory | Role::NavalFactory => Role::Factory,
                Role::Hydro | Role::MassStorage | Role::EnergyStorage => Role::Generator,
                Role::AirStaging | Role::EngStation | Role::Orbital => Role::Factory,
                Role::Shield | Role::AntiAir | Role::MissileSilo | Role::MissileDefense => Role::PointDefense,
                Role::Experimental => Role::Mobile,
                r => r,
            };
            &self
                .fallback
                .iter()
                .find(|(r, _)| *r == target)
                .unwrap_or(&self.fallback[0])
                .1
        })
    }
}

fn normalize_scm(scm: &ScmMesh, target_span: f32) -> Option<NormalizedMesh> {
    if scm.vertices.is_empty() || scm.indices.is_empty() {
        return None;
    }
    let bounds = scm.bounds?;
    let cx = (bounds.min[0] + bounds.max[0]) * 0.5;
    let cz = (bounds.min[2] + bounds.max[2]) * 0.5;
    let min_y = bounds.min[1];
    let ex = (bounds.max[0] - bounds.min[0]).abs();
    let ez = (bounds.max[2] - bounds.min[2]).abs();
    let span = ex.max(ez).max(1e-3);
    let s = target_span / span;

    let mut positions = Vec::with_capacity(scm.vertices.len());
    let mut normals = Vec::with_capacity(scm.vertices.len());
    let mut uvs = Vec::with_capacity(scm.vertices.len());
    for v in &scm.vertices {
        positions.push([
            (v.position[0] - cx) * s,
            (v.position[1] - min_y) * s,
            (v.position[2] - cz) * s,
        ]);
        normals.push(normalize_or_up(v.normal));
        uvs.push(v.uv0);
    }
    let indices = scm.indices.iter().map(|&i| u32::from(i)).collect();
    Some(NormalizedMesh {
        positions,
        normals,
        uvs,
        indices,
        albedo: None,
    })
}

fn procedural_role_mesh(role: Role, span: f32) -> NormalizedMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    let half = span * 0.5;
    match role {
        Role::Commander => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half * 0.75, span * 0.85, half * 0.75]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 0.85, 0.0], [half * 0.9, span * 0.35, half * 0.65]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 1.2, 0.0], [half * 0.42, span * 0.35, half * 0.42]);
        }
        Role::Factory => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [-half * 0.72, 0.0, 0.0], [half * 0.25, span * 0.45, half * 0.95]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [half * 0.72, 0.0, 0.0], [half * 0.25, span * 0.45, half * 0.95]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, -half * 0.72], [half * 0.95, span * 0.38, half * 0.25]);
        }
        Role::Extractor => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half * 0.8, span * 0.55, half * 0.8]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 0.55, 0.0], [half * 0.35, span * 0.85, half * 0.35]);
        }
        Role::Generator => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half * 0.85, span * 0.45, half * 0.85]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 0.45, 0.0], [half * 0.55, span * 0.45, half * 0.55]);
        }
        Role::PointDefense => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half * 0.7, span * 0.65, half * 0.7]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 0.65, half * 0.3], [half * 0.28, span * 0.28, half * 0.9]);
        }
        Role::Radar => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half * 0.6, span * 0.5, half * 0.6]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 0.5, 0.0], [half * 0.18, span * 1.1, half * 0.18]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 1.45, 0.0], [half * 0.75, span * 0.25, half * 0.22]);
        }
        Role::Engineer => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half * 0.75, span * 0.32, half * 0.85]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 0.32, half * 0.3], [half * 0.35, span * 0.35, half * 0.35]);
        }
        Role::Wall => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half, span * 0.9, half]);
        }
        Role::Mobile | Role::Experimental => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half * 0.75, span * 0.35, half]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 0.35, half * 0.1], [half * 0.4, span * 0.25, half * 0.65]);
        }
        Role::AirFactory | Role::NavalFactory | Role::Hydro | Role::MassStorage | Role::EnergyStorage | Role::Shield | Role::AntiAir | Role::MissileSilo | Role::MissileDefense | Role::AirStaging | Role::EngStation | Role::Orbital => {
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, 0.0, 0.0], [half * 0.75, span * 0.55, half * 0.75]);
            append_local_box(&mut positions, &mut normals, &mut uvs, &mut indices, [0.0, span * 0.55, 0.0], [half * 0.45, span * 0.35, half * 0.45]);
        }
    }
    NormalizedMesh {
        positions,
        normals,
        uvs,
        indices,
        albedo: None,
    }
}

fn append_local_box(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
    base_center: [f32; 3],
    half_extents_with_full_h: [f32; 3],
) {
    let hx = half_extents_with_full_h[0];
    let h = half_extents_with_full_h[1];
    let hz = half_extents_with_full_h[2];
    let [cx, cy, cz] = base_center;
    let x0 = cx - hx;
    let x1 = cx + hx;
    let y0 = cy;
    let y1 = cy + h;
    let z0 = cz - hz;
    let z1 = cz + hz;

    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 1.0, 0.0], [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]]),
        ([0.0, -1.0, 0.0], [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),
        ([0.0, 0.0, 1.0], [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),
        ([0.0, 0.0, -1.0], [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]]),
        ([1.0, 0.0, 0.0], [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]]),
        ([-1.0, 0.0, 0.0], [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]]),
    ];

    for (n, q) in faces {
        let base = positions.len() as u32;
        positions.extend_from_slice(&q);
        normals.extend_from_slice(&[n, n, n, n]);
        uvs.extend_from_slice(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

// ---------------------------------------------------------------------------
// PlaySession — pure controller state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PlaySession {
    pub world: Skirmish,
    pub ais: Vec<ScriptedAi>,
    pub human_army: ArmyId,
    pub spectate: bool,
    pub fog_of_war: bool,
    pub selected: Vec<UnitId>,
    pub build_mode: Option<String>,
    pub attack_move_mode: bool,
    pub paused: bool,
    pub speed: u32,
    /// Start with audio muted (`--mute`); F3 toggles at runtime.
    pub muted: bool,
    /// Phase 14 audio (started lazily by the window loop; clones are silent).
    pub audio: crate::audio_play::SessionAudio,
    pub hover_world: Option<[f32; 3]>,
    pub drag_box_world: Option<([f32; 3], [f32; 3])>,
    pub status_banner: String,
    pub(crate) replay: Option<(Replay, usize)>,
    pub ui: crate::controls::UiState,
    /// Phase 15 lockstep: when set, `step_sim` is gated on network turn bundles.
    pub net: Option<crate::net::NetHandle>,
    /// Phase 21: in-memory record of every command applied (human, AI, network) — the
    /// source of saves and the automatic match replay. `None` for replay playback.
    pub recorder: Option<crate::savegame::MatchRecorder>,
    /// Phase 21: pending time-sliced re-simulation of a loaded save.
    pub load: Option<Box<crate::savegame::LoadJob>>,
    /// Phase 21: UI-side score sampler (read-only on the world).
    pub stats: crate::stats::ScoreSampler,
    /// Phase 21: chat log + input line.
    pub chat: crate::chat::ChatState,
    /// Phase 21: modal overlay (pause menu, load list, score screen).
    pub menu: hud::MatchMenu,
    /// The end-of-game screen was shown (or dismissed with "continue watching").
    pub end_shown: bool,
    /// Write the automatic replay on exit / game end (`--no-record` disables).
    pub auto_record: bool,
}

impl PlaySession {
    pub fn new(world: Skirmish, human_army: ArmyId, spectate: bool, speed: u32) -> Self {
        let ais: Vec<ScriptedAi> = world
            .armies()
            .iter()
            .filter(|a| spectate || a.id != human_army)
            .map(|a| ScriptedAi::new(a.id))
            .collect();
        let mut selected = Vec::new();
        if !spectate && let Some(acu) = world.commander_of(human_army) {
            selected.push(acu.id);
        }
        let mut recorder = crate::savegame::MatchRecorder::default();
        for ai in &ais {
            recorder.ai_event(world.tick(), ai.army, true);
        }
        let mut stats = crate::stats::ScoreSampler::default();
        stats.prime(&world);
        Self {
            world,
            ais,
            human_army,
            spectate,
            fog_of_war: false,
            selected,
            build_mode: None,
            attack_move_mode: false,
            paused: false,
            speed: speed.clamp(1, 8),
            muted: false,
            audio: Default::default(),
            hover_world: None,
            drag_box_world: None,
            status_banner: if spectate {
                "Spectator AI active (F1 take control, F2 FOW)".to_string()
            } else {
                "B:Fac M:Mex P:PGen D:PD N:Rad | G/T/Y/U:Build | F1:AI F2:FOW".to_string()
            },
            replay: None,
            ui: crate::controls::UiState::default(),
            net: None,
            recorder: Some(recorder),
            load: None,
            stats,
            chat: Default::default(),
            menu: Default::default(),
            end_shown: false,
            auto_record: !crate::savegame::recording_disabled(),
        }
    }

    pub fn new_replay(world: Skirmish, replay: Replay, speed: u32) -> Self {
        let ui = crate::controls::UiState::for_replay(&world);
        let mut stats = crate::stats::ScoreSampler::default();
        stats.prime(&world);
        Self {
            world,
            ais: Vec::new(),
            human_army: 0,
            spectate: true,
            fog_of_war: false,
            selected: Vec::new(),
            build_mode: None,
            attack_move_mode: false,
            paused: false,
            speed: speed.clamp(1, 8),
            muted: false,
            audio: Default::default(),
            hover_world: None,
            drag_box_world: None,
            status_banner: format!("REPLAY ({} cmds)", replay.commands.len()),
            replay: Some((replay, 0)),
            ui,
            net: None,
            recorder: None,
            load: None,
            stats,
            chat: Default::default(),
            menu: Default::default(),
            end_shown: false,
            auto_record: false,
        }
    }

    pub fn toggle_spectate(&mut self) {
        if self.replay.is_some() {
            return;
        }
        self.spectate = !self.spectate;
        let tick = self.world.tick();
        if self.spectate {
            if !self.ais.iter().any(|ai| ai.army == self.human_army) {
                self.ais.push(ScriptedAi::new(self.human_army));
                self.ais.sort_by_key(|ai| ai.army);
                if let Some(rec) = self.recorder.as_mut() {
                    rec.ai_event(tick, self.human_army, true);
                }
            }
            self.build_mode = None;
            self.attack_move_mode = false;
            self.status_banner = "AI assisting Army 0 (F1 to take back control)".to_string();
        } else {
            if self.ais.iter().any(|ai| ai.army == self.human_army)
                && let Some(rec) = self.recorder.as_mut()
            {
                rec.ai_event(tick, self.human_army, false);
            }
            self.ais.retain(|ai| ai.army != self.human_army);
            self.status_banner = "Manual command active".to_string();
        }
    }

    /// Whether the window loop should advance the simulation right now. Networked matches
    /// always run (the lockstep core applies synchronized pauses); single player stops for
    /// the pause key, modal menus and pending loads.
    pub fn sim_active(&self) -> bool {
        if self.world.result().is_some() || self.loading() {
            return false;
        }
        if self.net.is_some() {
            return true;
        }
        !self.paused && !self.menu.blocks_sim()
    }

    /// `Space` / HUD pause button: local pause in single player, synchronized pause request
    /// in networked matches.
    pub fn toggle_pause(&mut self) {
        if let Some(net) = self.net.clone() {
            let want = !net.is_paused();
            match net.request_pause(want) {
                Ok(()) => {
                    self.status_banner = if want { "PAUSE REQUESTED" } else { "RESUME REQUESTED" }.to_string()
                }
                Err(e) => self.chat.system(&e.to_uppercase()),
            }
            return;
        }
        self.paused = !self.paused;
    }

    /// `+`/`-`: local speed, or a host-only synchronized speed request when networked.
    pub fn change_speed(&mut self, faster: bool) {
        let target = if faster { self.speed * 2 } else { self.speed / 2 }.clamp(1, self.max_speed());
        if let Some(net) = self.net.clone() {
            match net.request_speed(target) {
                Ok(()) => self.status_banner = format!("SPEED {target}X REQUESTED"),
                Err(e) => self.chat.system(&e.to_uppercase()),
            }
            return;
        }
        self.speed = target;
    }

    /// Sends queued chat lines: over the network when connected, else echoed locally.
    pub fn flush_local_chat(&mut self) {
        if self.net.is_some() {
            return;
        }
        let name = self
            .world
            .army(self.human_army)
            .map(|a| a.name.clone())
            .unwrap_or_else(|| "YOU".to_string());
        for (allies, text) in std::mem::take(&mut self.chat.outbox) {
            self.chat.push(Some(self.human_army), &name, &text, allies);
        }
    }

    /// True once the match is over for this player (victory/defeat/draw) and the score
    /// screen has not been shown yet.
    pub fn end_of_game_due(&self) -> bool {
        if self.end_shown || self.is_replay() || self.loading() {
            return false;
        }
        self.world.result().is_some()
            || (!self.spectate && self.world.army(self.human_army).is_some_and(|a| a.defeated))
    }

    /// Single-player bookkeeping around one `world.step()`: record the commands about to be
    /// applied (all sources), step, then checkpoint + score sampling.
    fn step_recorded(&mut self) {
        let counts = match self.recorder.as_mut() {
            Some(rec) => rec.record_pending(&self.world),
            None => Vec::new(),
        };
        self.world.step();
        if let Some(rec) = self.recorder.as_mut() {
            rec.after_step(&self.world);
        }
        self.stats.observe(&self.world, &counts);
    }

    pub fn step_sim(&mut self) {
        if self.world.result().is_some() {
            return;
        }
        if self.loading() {
            return;
        }
        if let Some(net) = self.net.clone() {
            let _p = crate::profiler::scope("net.step_session");
            net.step_session(self);
            return;
        }
        if let Some((ref rep, ref mut cursor)) = self.replay {
            if self.world.tick() >= rep.final_tick {
                self.paused = true;
                return;
            }
            let t = self.world.tick();
            while *cursor < rep.commands.len() && rep.commands[*cursor].tick == t {
                let rc = &rep.commands[*cursor];
                self.world.issue(rc.army, rc.command.clone());
                *cursor += 1;
            }
            self.world.step();
            self.selected.retain(|id| self.world.unit(*id).is_some());
            return;
        }
        {
            let _p = crate::profiler::scope("ai.think");
            for ai in &mut self.ais {
                ai.think(&mut self.world);
            }
        }
        self.step_recorded();
        self.selected.retain(|id| self.world.unit(*id).is_some());
    }

    pub fn select_at(&mut self, world_xz: (i64, i64), additive: bool, pick_radius: i64) {
        let hit = self
            .world
            .unit_at(world_xz, pick_radius, Some(self.human_army), None);
        if !additive {
            self.selected.clear();
        }
        if let Some(id) = hit
            && !self.selected.contains(&id)
        {
            self.selected.push(id);
        }
    }

    pub fn select_box(&mut self, a: (i64, i64), b: (i64, i64), additive: bool) {
        let mut hits = self.world.units_in_rect(self.human_army, a, b);
        let has_mobile_or_acu = hits.iter().any(|id| {
            self.world
                .unit(*id)
                .is_some_and(|u| !u.role.is_structure())
        });
        if has_mobile_or_acu {
            hits.retain(|id| {
                self.world
                    .unit(*id)
                    .is_some_and(|u| !u.role.is_structure())
            });
        }
        if !additive {
            self.selected.clear();
        }
        for id in hits {
            if !self.selected.contains(&id) {
                self.selected.push(id);
            }
        }
    }

    pub fn select_commander(&mut self) {
        self.selected.clear();
        if let Some(acu) = self.world.commander_of(self.human_army) {
            self.selected.push(acu.id);
        }
    }

    pub fn select_factories(&mut self) {
        self.selected = self
            .world
            .units()
            .filter(|u| u.army == self.human_army && u.role == Role::Factory)
            .map(|u| u.id)
            .collect();
    }

    pub fn select_army(&mut self) {
        self.selected = self
            .world
            .units()
            .filter(|u| u.army == self.human_army && u.role == Role::Mobile)
            .map(|u| u.id)
            .collect();
    }

    pub fn issue_left_click_action(
        &mut self,
        world_xz: (i64, i64),
        queue: bool,
        pick_radius: i64,
    ) -> bool {
        if self.handle_left_click_unload(world_xz, queue) {
            return true;
        }
        if self.handle_left_click_order(world_xz, queue, pick_radius) {
            return true;
        }
        if self.place_template(world_xz, queue) {
            return true;
        }
        if let Some(ref btype) = self.build_mode.clone() {
            let Some(builder_id) = self.builder_for(btype) else {
                self.build_mode = None;
                self.status_banner = "No builder alive".to_string();
                return true;
            };
            let target_role = self.world.catalog().get(btype).map(|t| t.role);
            let target = if target_role == Some(Role::Extractor) {
                match self.world.snap_extractor(world_xz.0, world_xz.1) {
                    Some((_, p)) => p,
                    None => {
                        self.status_banner = "No free mass point nearby".to_string();
                        return true;
                    }
                }
            } else if target_role == Some(Role::Hydro) {
                match self.world.snap_hydro(world_xz.0, world_xz.1) {
                    Some((_, p)) => p,
                    None => {
                        self.status_banner = "No free hydrocarbon point nearby".to_string();
                        return true;
                    }
                }
            } else {
                if !self.world.site_is_free(btype, world_xz.0, world_xz.1) {
                    self.status_banner = "Blocked build site".to_string();
                    return true;
                }
                world_xz
            };
            self.world.issue(
                self.human_army,
                Command::Build {
                    builder: builder_id,
                    unit_type: btype.clone(),
                    x: target.0,
                    y: target.1,
                    queue,
                },
            );
            // Phase 23: the other selected builders assist the primary one.
            let helpers: Vec<UnitId> = self.selected_builders().into_iter().filter(|&b| b != builder_id).collect();
            if !helpers.is_empty() {
                self.world.issue(self.human_army, Command::Assist { units: helpers, target: builder_id, queue: true });
            }
            self.status_banner = format!("Ordered build: {btype}");
            if !queue {
                self.build_mode = None;
            }
            return true;
        }

        if self.attack_move_mode {
            let units = self.selected_mobiles();
            let enemy = self.world.unit_at(world_xz, pick_radius, None, Some(self.human_army));
            if let (false, Some(target)) = (units.is_empty(), enemy) {
                self.world.issue(self.human_army, Command::Attack { units, target, queue });
                self.status_banner = format!("Attack target #{target}");
            } else if !units.is_empty() {
                self.world.issue(
                    self.human_army,
                    Command::AttackMove {
                        units,
                        x: world_xz.0,
                        y: world_xz.1,
                        queue,
                    },
                );
                self.status_banner = "Attack-move ordered".to_string();
            }
            if !queue {
                self.attack_move_mode = false;
            }
            return true;
        }

        false
    }

    pub fn issue_right_click(&mut self, world_xz: (i64, i64), queue: bool, pick_radius: i64) {
        if self.build_mode.is_some() || self.attack_move_mode || self.ui.unload_mode || self.ui.order_mode.is_some() {
            self.build_mode = None;
            self.clear_order_mode();
            self.status_banner = "Cancelled active mode".to_string();
            return;
        }
        if self.selected.is_empty() {
            return;
        }

        let factories: Vec<UnitId> = self
            .selected
            .iter()
            .copied()
            .filter(|id| {
                self.world
                    .unit(*id)
                    .is_some_and(|u| u.army == self.human_army && u.role == Role::Factory)
            })
            .collect();
        let mobiles = self.selected_mobiles();
        let builders: Vec<UnitId> = self
            .selected
            .iter()
            .copied()
            .filter(|id| {
                self.world
                    .unit(*id)
                    .is_some_and(|u| u.army == self.human_army && u.role.is_builder())
            })
            .collect();

        if mobiles.is_empty() && !factories.is_empty() {
            for f in factories {
                self.world.issue(
                    self.human_army,
                    Command::Rally {
                        factory: f,
                        x: world_xz.0,
                        y: world_xz.1,
                    },
                );
            }
            self.status_banner = "Factory rally point updated".to_string();
            return;
        }

        if let Some(target) = self
            .world
            .unit_at(world_xz, pick_radius, None, Some(self.human_army))
        {
            self.world.issue(
                self.human_army,
                Command::Attack {
                    units: mobiles,
                    target,
                    queue,
                },
            );
            self.status_banner = format!("Attack target #{target}");
            return;
        }

        if self.handle_right_click_board(world_xz, queue, pick_radius) {
            return;
        }
        if self.handle_right_click_dock(world_xz, queue, pick_radius) {
            return;
        }

        if !builders.is_empty() {
            if let Some((ally_id, ally_hp, ally_max_hp, ally_role)) = self
                .world
                .unit_at(world_xz, pick_radius, Some(self.human_army), None)
                .and_then(|id| self.world.unit(id))
                .map(|u| (u.id, u.hp, u.max_hp, u.role))
            {
                if ally_hp < ally_max_hp {
                    self.world.issue(
                        self.human_army,
                        Command::Repair {
                            units: builders,
                            target: ally_id,
                            queue,
                        },
                    );
                    self.status_banner = format!("Repairing #{ally_id}");
                    return;
                } else if ally_role == Role::Factory || ally_role.is_builder() {
                    self.world.issue(
                        self.human_army,
                        Command::Assist {
                            units: builders,
                            target: ally_id,
                            queue,
                        },
                    );
                    self.status_banner = format!("Assisting #{ally_id}");
                    return;
                }
            }
            if let Some((_, wp)) = self.world.wreck_at(world_xz, pick_radius + 4 * SCALE) {
                self.world.issue(
                    self.human_army,
                    Command::Reclaim {
                        units: builders,
                        x: wp.0,
                        y: wp.1,
                        queue,
                    },
                );
                self.status_banner = "Reclaiming wreck".to_string();
                return;
            }
        }

        self.world.issue(
            self.human_army,
            Command::Move {
                units: mobiles,
                x: world_xz.0,
                y: world_xz.1,
                queue,
            },
        );
        self.status_banner = "Move ordered".to_string();
    }

    pub fn trigger_build_mode(&mut self, role: Role) {
        let Some(f) = self.world.faction(self.human_army).cloned() else {
            return;
        };
        let tid = match role {
            Role::Factory => f.factory,
            Role::Extractor => f.extractor,
            Role::Generator => f.generator,
            Role::PointDefense => f.point_defense,
            Role::Radar => f.radar,
            _ => return,
        };
        self.attack_move_mode = false;
        self.ui.unload_mode = false;
        self.build_mode = Some(tid.clone());
        self.status_banner = format!("Place {tid} (LMB place, Shift+LMB queue, RMB cancel)");
    }

    pub fn trigger_produce_engineer(&mut self, count: u32) {
        let Some(f) = self.world.faction(self.human_army).cloned() else {
            return;
        };
        self.trigger_produce_type(f.engineer, count);
    }

    pub fn trigger_produce(&mut self, slot: usize, count: u32) {
        let Some(f) = self.world.faction(self.human_army).cloned() else {
            return;
        };
        let Some(utype) = f.land_units.get(slot).cloned() else {
            return;
        };
        self.trigger_produce_type(utype, count);
    }

    fn trigger_produce_type(&mut self, utype: String, count: u32) {
        let mut factories: Vec<UnitId> = self
            .selected
            .iter()
            .copied()
            .filter(|id| {
                self.world
                    .unit(*id)
                    .is_some_and(|u| u.army == self.human_army && u.role == Role::Factory)
            })
            .collect();
        if factories.is_empty() {
            factories = self
                .world
                .units()
                .filter(|u| u.army == self.human_army && u.role == Role::Factory)
                .map(|u| u.id)
                .collect();
        }
        if factories.is_empty() {
            self.status_banner = "No Land Factory built yet (press B to build)".to_string();
            return;
        }
        for fid in &factories {
            self.world.issue(
                self.human_army,
                Command::Produce {
                    factory: *fid,
                    unit_type: utype.clone(),
                    count,
                },
            );
        }
        self.status_banner = format!("Queued {count}x {utype}");
    }

    pub fn trigger_overcharge_nearest(&mut self) {
        let Some(acu) = self.world.commander_of(self.human_army) else {
            return;
        };
        let acu_id = acu.id;
        let acu_pos = (acu.pos.x, acu.pos.y);
        if let Some(target) = self.world.unit_at(acu_pos, 26 * SCALE, None, Some(self.human_army)) {
            self.world.issue(
                self.human_army,
                Command::Overcharge {
                    unit: acu_id,
                    target,
                },
            );
            self.status_banner = format!("OVERCHARGE -> #{target}!");
        } else {
            self.status_banner = "No enemy in Overcharge range".to_string();
        }
    }

    pub fn trigger_stop(&mut self) {
        self.build_mode = None;
        self.attack_move_mode = false;
        self.ui.unload_mode = false;
        if !self.selected.is_empty() {
            let units = self.selected.clone();
            self.world.issue(self.human_army, Command::Stop { units });
            self.status_banner = "Stopped selected units".to_string();
        }
    }

    pub fn trigger_hud_action(&mut self, action: HudAction, shift: bool) {
        let count = if shift { 5 } else { 1 };
        match action {
            HudAction::BuildFactory => self.trigger_build_mode(Role::Factory),
            HudAction::BuildExtractor => self.trigger_build_mode(Role::Extractor),
            HudAction::BuildGenerator => self.trigger_build_mode(Role::Generator),
            HudAction::BuildPointDefense => self.trigger_build_mode(Role::PointDefense),
            HudAction::BuildRadar => self.trigger_build_mode(Role::Radar),
            HudAction::ProduceEngineer => self.trigger_produce_engineer(count),
            HudAction::ProduceTank => self.trigger_produce(0, count),
            HudAction::ProduceBot => self.trigger_produce(1, count),
            HudAction::ProduceArtillery => self.trigger_produce(2, count),
            HudAction::AttackMove => {
                self.build_mode = None;
                self.ui.unload_mode = false;
                self.ui.order_mode = None;
                self.attack_move_mode = !self.attack_move_mode;
                self.status_banner = if self.attack_move_mode {
                    "Attack: LMB enemy = attack, ground = attack-move".to_string()
                } else {
                    "Cancelled Attack-Move".to_string()
                };
            }
            HudAction::Stop => self.trigger_stop(),
            HudAction::TogglePause => self.toggle_pause(),
            HudAction::ToggleFog => {
                self.fog_of_war = !self.fog_of_war;
                self.status_banner = if self.fog_of_war {
                    "Fog of War + Radar Intel ON".to_string()
                } else {
                    "Full Map Reveal ON".to_string()
                };
            }
            HudAction::ToggleAi => self.toggle_spectate(),
            HudAction::IdleEngineer => {
                self.cycle_idle_engineer();
            }
            HudAction::UnloadTransport => self.trigger_unload_mode(),
            HudAction::ToggleRepeat => self.trigger_toggle_repeat(),
            HudAction::EnhanceNext => self.trigger_enhance_selected(shift),
            HudAction::ToggleShield => self.trigger_toggle_shield(),
            HudAction::Order(kind) => self.trigger_order(kind),
            HudAction::CardItem(idx) => self.trigger_card_item(idx as usize, shift, false),
            HudAction::CardTab(t) => self.set_card_tab(t),
            HudAction::CardGroup(g) => self.set_card_group(g),
            HudAction::CardPage => self.next_card_page(),
            HudAction::QueueItem(idx) => self.trigger_queue_cancel(idx as usize, shift),
            HudAction::Upgrade => self.trigger_upgrade(),
            HudAction::EnhanceRow(idx) => self.trigger_enhance_row(idx as usize),
        }
    }

    /// Right-click on a HUD element (card entry: dequeue one, queue entry: cancel one).
    /// Returns `true` if the element reacts to RMB.
    pub fn trigger_hud_secondary(&mut self, action: HudAction, shift: bool) -> bool {
        match action {
            HudAction::CardItem(idx) => {
                self.trigger_card_item(idx as usize, shift, true);
                true
            }
            HudAction::QueueItem(idx) => {
                self.trigger_queue_cancel(idx as usize, shift);
                true
            }
            _ => false,
        }
    }

    fn selected_mobiles(&self) -> Vec<UnitId> {
        self.selected
            .iter()
            .copied()
            .filter(|id| {
                self.world
                    .unit(*id)
                    .is_some_and(|u| u.army == self.human_army && u.speed > 0)
            })
            .collect()
    }

    pub fn window_title(&self, map_name: &str) -> String {
        let secs = self.world.tick() / TICK_HZ;
        let time = format!("{:02}:{:02}", secs / 60, secs % 60);
        let speed_tag = if self.paused {
            "PAUSED".to_string()
        } else if self.spectate {
            format!("{}x AI", self.speed)
        } else {
            format!("{}x", self.speed)
        };
        let end_tag = match self.world.result() {
            Some(GameResult::Victory(a)) => format!(" | VICTORY: Army {a}"),
            Some(GameResult::Draw) => " | DRAW".to_string(),
            None => String::new(),
        };
        let Some(a) = self.world.army(self.human_army) else {
            return format!("FARust — {map_name} [{time}]{end_tag}");
        };
        format!(
            "FARust — {map_name} [{time} {speed_tag}] | M {}/{} E {}/{} | Sel {}{end_tag}",
            a.mass / SCALE,
            a.mass_cap / SCALE,
            a.energy / SCALE,
            a.energy_cap / SCALE,
            self.selected.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Colors, Instanced Unit Batches & Dynamic Overlays (Tracers, Projectiles, Icons)
// ---------------------------------------------------------------------------

pub(crate) fn army_rgb(world: &Skirmish, army: ArmyId) -> [f32; 3] {
    let faction_key = world
        .army(army)
        .and_then(|a| world.catalog().factions.get(a.faction))
        .map(|f| f.key.as_str())
        .unwrap_or("");
    match faction_key {
        "uef" => [0.22, 0.56, 0.98],
        "cybran" => [0.94, 0.24, 0.22],
        "aeon" => [0.20, 0.84, 0.44],
        "seraphim" => [0.96, 0.80, 0.22],
        _ => match army % 4 {
            0 => [0.22, 0.56, 0.98],
            1 => [0.94, 0.24, 0.22],
            2 => [0.20, 0.84, 0.44],
            _ => [0.96, 0.80, 0.22],
        },
    }
}

fn army_color(world: &Skirmish, army: ArmyId) -> [f32; 4] {
    let c = army_rgb(world, army);
    [c[0], c[1], c[2], 0.45]
}

fn unit_yaw(world: &Skirmish, u: &crate::skirmish::Unit) -> f32 {
    // Walking units face their path; turrets (GPU-skinned) track the target independently.
    if u.moving
        && let Some(UnitOrder::Move { x, y } | UnitOrder::AttackMove { x, y } | UnitOrder::Patrol { x, y }) =
            u.orders.front()
        && (*x != u.pos.x || *y != u.pos.y)
    {
        return ((x - u.pos.x) as f32).atan2((y - u.pos.y) as f32);
    }
    if let Some((tid, _)) = u.last_shot
        && let Some(t) = world.unit(tid)
    {
        return ((t.pos.x - u.pos.x) as f32).atan2((t.pos.y - u.pos.y) as f32);
    }
    match u.orders.front() {
        Some(UnitOrder::Move { x, y })
        | Some(UnitOrder::AttackMove { x, y })
        | Some(UnitOrder::Patrol { x, y })
        | Some(UnitOrder::LaunchMissile { x, y })
        | Some(UnitOrder::Build { x, y, .. })
        | Some(UnitOrder::Reclaim { x, y }) => {
            ((x - u.pos.x) as f32).atan2((y - u.pos.y) as f32)
        }
        Some(UnitOrder::Attack(tid))
        | Some(UnitOrder::Repair(tid))
        | Some(UnitOrder::Assist(tid))
        | Some(UnitOrder::Overcharge(tid)) => world
            .unit(*tid)
            .map(|t| ((t.pos.x - u.pos.x) as f32).atan2((t.pos.y - u.pos.y) as f32))
            .unwrap_or(0.0),
        Some(UnitOrder::Unload { x, y }) => ((x - u.pos.x) as f32).atan2((y - u.pos.y) as f32),
        Some(UnitOrder::Board(_) | UnitOrder::Enhance(_)) => 0.0,
        Some(UnitOrder::AttackGround { x, y }) => ((x - u.pos.x) as f32).atan2((y - u.pos.y) as f32),
        Some(UnitOrder::Ferry { from, to }) => {
            let g = if u.cargo.is_empty() { *from } else { *to };
            ((g.0 - u.pos.x) as f32).atan2((g.1 - u.pos.y) as f32)
        }
        Some(UnitOrder::Guard(tid)) | Some(UnitOrder::Dock(tid)) => world
            .unit(*tid)
            .map(|t| ((t.pos.x - u.pos.x) as f32).atan2((t.pos.y - u.pos.y) as f32))
            .unwrap_or(0.0),
        Some(UnitOrder::Upgrade(_)) | None => 0.0,
    }
}

pub(crate) fn build_skirmish_render_frame(
    session: &PlaySession,
    scmap: &ScMap,
    meshes: &UnitMeshCache,
    camera: &OrbitCamera,
) -> (BTreeMap<String, Vec<GpuUnitInstance>>, Vec<GpuVertex>, Vec<u32>) {
    let _p = crate::profiler::scope("render.build_frame");
    let mut batches: BTreeMap<String, Vec<GpuUnitInstance>> = BTreeMap::new();
    let mut verts = Vec::with_capacity(8192);
    let mut indices = Vec::with_capacity(16384);
    let scale = SCALE as f32;
    let zoom_boost = (camera.distance / 320.0).clamp(1.0, 3.2);
    let strategic_zoom = camera.distance > 340.0;
    let world = &session.world;
    let selected_set = &session.selected;
    let fow_viewer = session.fog_of_war.then_some(session.human_army);
    {
        let _p = crate::profiler::scope("render.begin_frame");
        meshes.living.begin_frame(world, scmap, fow_viewer, zoom_boost);
    }
    let frame_time = meshes.living.frame_time();
    let eye = camera.eye_position();
    // CPU frustum culling of unit/wreck mesh instances, bone poses and bars (generous margin so
    // shadows of units just off-screen still land in view).
    let frustum = viewer::Frustum::from_view_proj(&camera.view_proj_matrix());

    let mut army_colors = [[0.7, 0.7, 0.7, 0.85]; 8];
    let mut army_is_cybran = [false; 8];
    for idx in 0..8usize {
        let a = idx as ArmyId;
        army_colors[idx] = army_color(world, a);
        army_is_cybran[idx] = world
            .army(a)
            .and_then(|ar| world.catalog().factions.get(ar.faction))
            .is_some_and(|f| f.key == "cybran");
    }
    let sel_lookup: Option<std::collections::HashSet<UnitId>> =
        (selected_set.len() > 8).then(|| selected_set.iter().copied().collect());
    let is_unit_selected = |id: UnitId| -> bool {
        if selected_set.is_empty() {
            false
        } else if let Some(ref set) = sel_lookup {
            set.contains(&id)
        } else {
            selected_set.contains(&id)
        }
    };

    // 1. Unclaimed mass deposit pads + emerald locator rings
    for (idx, &(mx, mz)) in world.mass_points().iter().enumerate() {
        if world.mass_claim(idx).is_some() {
            continue;
        }
        let wx = mx as f32 / scale;
        let wz = mz as f32 / scale;
        let wy = sample_terrain_y(scmap, wx, wz) + 0.10;
        let s = zoom_boost.sqrt();
        append_box_world(&mut verts, &mut indices, [wx, wy - 0.06, wz], 0.85 * s, 0.12, [0.018, 0.022, 0.028, 0.45]);
        append_ring(
            &mut verts,
            &mut indices,
            [wx, wy + 0.08, wz],
            1.15 * s,
            0.26 * s,
            [0.14, 0.86, 0.40, 0.22],
            12,
        );
    }

    // 2. Reclaimable wrecks (charred real unit .scm meshes; box fallback in strategic zoom or missing type)
    for w in world.wrecks() {
        let Some((sink_dy, sink_tilt)) = meshes.living.wreck_pose(w, scmap) else {
            continue;
        };
        let wx = w.pos.x as f32 / scale;
        let wz = w.pos.y as f32 / scale;
        let wy = sample_terrain_y(scmap, wx, wz) + sink_dy;
        if strategic_zoom || w.type_id.is_empty() {
            let sz = 0.48 * zoom_boost.sqrt();
            append_box_world(&mut verts, &mut indices, [wx, wy, wz], sz, sz * 0.35, [0.05, 0.045, 0.042, 0.25]);
        } else {
            let use_tex = if meshes.mesh_for(&w.type_id, Role::Mobile).albedo.is_some() {
                1.0
            } else {
                0.0
            };
            let yaw = ((w.id.wrapping_mul(2654435761) & 1023) as f32 / 1024.0) * std::f32::consts::TAU;
            let tilt = (((w.id % 5) as f32) - 2.0) * 0.07 + sink_tilt;
            let render_scale = zoom_boost * 1.10;
            if !frustum.sphere_visible([wx, wy, wz], 6.0 * render_scale + 4.0) {
                continue;
            }
            batches.entry(w.type_id.clone()).or_default().push(
                GpuUnitInstance::from_pos_yaw_scale(
                    [wx, wy - 0.12 * render_scale, wz],
                    yaw,
                    tilt,
                    render_scale,
                    [0.45, 0.42, 0.40, 0.12],
                    [use_tex, 0.0, 1.0, 0.10],
                ),
            );
        }
    }

    // 3. Units (instanced real .scm meshes at close/mid zoom, crisp strategic icons when zoomed far out)
    struct TypeBatchCache {
        use_tex: f32,
        lod_cutoff_sq: f32,
        lod0_idx: usize,
        lod1_idx: Option<usize>,
    }
    fn intern_batch(
        key: &str,
        keys: &mut Vec<String>,
        vecs: &mut Vec<Vec<GpuUnitInstance>>,
        map: &mut std::collections::HashMap<String, usize>,
    ) -> usize {
        if let Some(&idx) = map.get(key) {
            idx
        } else {
            let idx = keys.len();
            keys.push(key.to_string());
            vecs.push(Vec::new());
            map.insert(key.to_string(), idx);
            idx
        }
    }
    let mut batch_keys: Vec<String> = Vec::with_capacity(32);
    let mut batch_vecs: Vec<Vec<GpuUnitInstance>> = Vec::with_capacity(32);
    let mut batch_key_to_idx: std::collections::HashMap<String, usize> = std::collections::HashMap::with_capacity(32);
    let mut type_cache: std::collections::HashMap<&str, TypeBatchCache> = std::collections::HashMap::with_capacity(32);

    let units_span = crate::profiler::scope("render.units");
    for u in world.units() {
        let vis = if session.fog_of_war {
            world.intel_visibility(session.human_army, u)
        } else {
            IntelVisibility::Visible
        };
        if vis == IntelVisibility::Hidden {
            continue;
        }
        let wx = u.pos.x as f32 / scale;
        let wz = u.pos.y as f32 / scale;
        let wy = sample_terrain_y(scmap, wx, wz) + crate::render_prep::LivingBattlefield::unit_lift_with_time(u, frame_time);
        let color = army_colors
            .get(usize::from(u.army))
            .copied()
            .unwrap_or_else(|| army_color(world, u.army));

        if vis == IntelVisibility::RadarBlip {
            let r = 1.8 * zoom_boost.sqrt();
            append_box_world(&mut verts, &mut indices, [wx, wy + 0.6, wz], r, r * 1.4, [1.0, 0.3, 0.25, 0.2]);
            continue;
        }

        let render_scale = if u.role.is_structure() {
            zoom_boost.powf(0.45)
        } else {
            zoom_boost * 1.15
        };
        let cull_r = (u.radius as f32 / scale).max(1.0) * render_scale * 2.5 + 6.0;
        let on_screen = frustum.sphere_visible([wx, wy, wz], cull_r);
        let is_sel = is_unit_selected(u.id);

        if strategic_zoom
            && !meshes.living.push_strategic_icon(
                u,
                [wx, wy + 0.5, wz],
                [color[0], color[1], color[2]],
                is_sel,
            )
        {
            let s = (u.radius as f32 / scale).max(1.2) * (zoom_boost * 0.75);
            let h = if u.role == Role::Commander { s * 1.6 } else if u.role.is_structure() { s * 0.9 } else { s * 1.1 };
            append_box_world(&mut verts, &mut indices, [wx, wy + 0.3, wz], s, h, color);
        } else if !strategic_zoom {
            let yaw = if u.role.is_structure() { 0.0 } else { unit_yaw(world, u) };
            if on_screen {
                let (use_tex, lod_cutoff_sq, lod0_idx, lod1_idx) = if let Some(c) = type_cache.get(u.type_id.as_str()) {
                    (c.use_tex, c.lod_cutoff_sq, c.lod0_idx, c.lod1_idx)
                } else {
                    let use_tex = if meshes.mesh_for(&u.type_id, u.role).albedo.is_some() {
                        1.0
                    } else {
                        0.0
                    };
                    let lod0_idx = intern_batch(&u.type_id, &mut batch_keys, &mut batch_vecs, &mut batch_key_to_idx);
                    let (lod_cutoff_sq, lod1_idx) = match meshes.living.asset_lod_info(&u.type_id) {
                        Some((_, cutoff_sq, Some(l1_key))) => (
                            cutoff_sq,
                            Some(intern_batch(l1_key, &mut batch_keys, &mut batch_vecs, &mut batch_key_to_idx)),
                        ),
                        _ => (f32::INFINITY, None),
                    };
                    type_cache.insert(
                        u.type_id.as_str(),
                        TypeBatchCache { use_tex, lod_cutoff_sq, lod0_idx, lod1_idx },
                    );
                    (use_tex, lod_cutoff_sq, lod0_idx, lod1_idx)
                };
                let batch_idx = if let Some(l1) = lod1_idx {
                    let dx = wx - eye[0];
                    let dy = wy - eye[1];
                    let dz = wz - eye[2];
                    if dx * dx + dy * dy + dz * dz > lod_cutoff_sq { l1 } else { lod0_idx }
                } else {
                    lod0_idx
                };
                let inst = GpuUnitInstance::from_pos_yaw_scale(
                    [wx, wy, wz],
                    yaw,
                    0.0,
                    render_scale,
                    color,
                    [use_tex, 0.0, 0.0, 0.85],
                );
                let slot = batch_vecs[batch_idx].len();
                let inst = meshes.living.skin_instance(u, yaw, inst, Some((batch_idx, slot)));
                batch_vecs[batch_idx].push(inst);
            } else {
                let inst = GpuUnitInstance::from_pos_yaw_scale(
                    [wx, wy, wz],
                    yaw,
                    0.0,
                    render_scale,
                    color,
                    [0.0, 0.0, 0.0, 0.85],
                );
                meshes.living.skin_instance(u, yaw, inst, None);
            }
        }

        let radius_m = (u.radius as f32 / scale) * render_scale;

        if is_sel {
            append_ring(
                &mut verts,
                &mut indices,
                [wx, wy + 0.18, wz],
                radius_m.max(1.2) + 0.55 * zoom_boost.sqrt(),
                0.35 * zoom_boost.sqrt(),
                [0.25, 1.0, 0.45, 0.25],
                16,
            );
        }

        if on_screen && (is_sel || u.hp < u.max_hp || u.role == Role::Commander) {
            let bar_y = wy + (radius_m * 1.25).max(2.4) + 1.2;
            let bar_w = (radius_m * 1.6).clamp(2.2, 8.0);
            let bar_h = 0.45 * zoom_boost.sqrt();
            let frac = (u.hp as f32 / u.max_hp.max(1) as f32).clamp(0.0, 1.0);
            let hp_col = if frac > 0.55 {
                [0.22, 0.92, 0.35, 0.10]
            } else if frac > 0.25 {
                [0.95, 0.78, 0.18, 0.10]
            } else {
                [0.95, 0.22, 0.20, 0.10]
            };
            append_bar_billboard(&mut verts, &mut indices, [wx, bar_y, wz], bar_w, bar_h, frac, hp_col);
        }

        if let Some(ref job) = u.build {
            let prog = (job.progress_permille() as f32 / 1000.0).clamp(0.0, 1.0);
            let bar_y = wy + (radius_m * 1.25).max(2.4) + 2.0;
            let bar_w = (radius_m * 1.6).clamp(2.4, 8.0);
            append_bar_billboard(
                &mut verts,
                &mut indices,
                [wx, bar_y, wz],
                bar_w,
                0.4 * zoom_boost.sqrt(),
                prog,
                [0.20, 0.85, 1.0, 0.15],
            );
            if u.role.is_builder() && !meshes.living.has_effect_library() {
                let bx = job.x as f32 / scale;
                let bz = job.y as f32 / scale;
                let by = sample_terrain_y(scmap, bx, bz) + 1.2;
                append_beam(
                    &mut verts,
                    &mut indices,
                    [wx, wy + 1.6 * render_scale, wz],
                    [bx, by, bz],
                    0.22 * zoom_boost.sqrt(),
                    [1.0, 0.85, 0.25, 0.80],
                );
            }
        }

        if let Some((tid, shot_tick)) = u.last_shot
            && world.tick().saturating_sub(shot_tick) <= 2
            && !meshes.living.has_weapon_beam(&u.type_id)
            && let Some(t) = world.unit(tid)
        {
            let tx = t.pos.x as f32 / scale;
            let tz = t.pos.y as f32 / scale;
            let ty = sample_terrain_y(scmap, tx, tz) + 1.2 + crate::render_prep::LivingBattlefield::unit_lift_with_time(t, frame_time);
            let is_cybran = army_is_cybran.get(usize::from(u.army)).copied().unwrap_or(false);
            let tracer_col = if is_cybran {
                [1.0, 0.28, 0.22, 0.95]
            } else {
                [0.45, 0.88, 1.0, 0.95]
            };
            append_beam(
                &mut verts,
                &mut indices,
                [wx, wy + 1.3 * render_scale, wz],
                [tx, ty, tz],
                0.28 * zoom_boost.sqrt(),
                tracer_col,
            );
        }

        if is_sel {
            let mut prev = [wx, wy + 0.35, wz];
            for ord in &u.orders {
                let (dest, col) = match ord {
                    UnitOrder::Move { x, y } => ((*x, *y), [0.25, 0.95, 0.45, 0.2]),
                    UnitOrder::AttackMove { x, y } => ((*x, *y), [1.0, 0.55, 0.15, 0.2]),
                    UnitOrder::Patrol { x, y } => ((*x, *y), [0.40, 0.70, 1.0, 0.2]),
                    UnitOrder::LaunchMissile { x, y } => ((*x, *y), [1.0, 0.35, 0.20, 0.2]),
                    UnitOrder::Build { x, y, .. } => ((*x, *y), [1.0, 0.88, 0.25, 0.2]),
                    UnitOrder::Reclaim { x, y } => ((*x, *y), [0.25, 0.90, 0.78, 0.2]),
                    UnitOrder::Upgrade(_) => continue,
                    UnitOrder::Unload { x, y } => ((*x, *y), [0.85, 0.80, 0.30, 0.2]),
                    UnitOrder::Board(_) | UnitOrder::Enhance(_) => continue,
                    UnitOrder::AttackGround { x, y } => ((*x, *y), [1.0, 0.45, 0.10, 0.2]),
                    UnitOrder::Ferry { from, to } => (if u.cargo.is_empty() { *from } else { *to }, [0.85, 0.80, 0.30, 0.2]),
                    UnitOrder::Guard(tid) => match world.unit(*tid) {
                        Some(t) => ((t.pos.x, t.pos.y), [0.45, 1.0, 0.55, 0.2]),
                        None => continue,
                    },
                    UnitOrder::Dock(tid) => match world.unit(*tid) {
                        Some(t) => ((t.pos.x, t.pos.y), [0.40, 0.85, 1.0, 0.2]),
                        None => continue,
                    },
                    UnitOrder::Attack(tid) | UnitOrder::Overcharge(tid) => {
                        if let Some(t) = world.unit(*tid) {
                            ((t.pos.x, t.pos.y), [1.0, 0.22, 0.22, 0.2])
                        } else {
                            continue;
                        }
                    }
                    UnitOrder::Repair(tid) | UnitOrder::Assist(tid) => {
                        if let Some(t) = world.unit(*tid) {
                            ((t.pos.x, t.pos.y), [0.30, 0.75, 1.0, 0.2])
                        } else {
                            continue;
                        }
                    }
                };
                let dx = dest.0 as f32 / scale;
                let dz = dest.1 as f32 / scale;
                let dy = sample_terrain_y(scmap, dx, dz) + 0.35;
                let next = [dx, dy, dz];
                append_beam(&mut verts, &mut indices, prev, next, 0.22 * zoom_boost.sqrt(), col);
                prev = next;
            }
            if u.role == Role::Factory
                && let Some((rx, ry)) = u.rally
            {
                let dx = rx as f32 / scale;
                let dz = ry as f32 / scale;
                let dy = sample_terrain_y(scmap, dx, dz) + 0.35;
                append_beam(
                    &mut verts,
                    &mut indices,
                    [wx, wy + 0.35, wz],
                    [dx, dy, dz],
                    0.22 * zoom_boost.sqrt(),
                    [0.25, 0.85, 1.0, 0.2],
                );
            }
        }
    }

    // 4. Death clips, construction build-up ghosts & projectile meshes (glows/trails: VFX pass)
    drop(units_span);
    meshes.living.resolve_pose_jobs(&mut batch_vecs);
    for (key, vec) in batch_keys.into_iter().zip(batch_vecs) {
        if vec.is_empty() {
            continue;
        }
        match batches.entry(key) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(vec);
            }
            std::collections::btree_map::Entry::Occupied(mut e) => {
                e.get_mut().extend(vec);
            }
        }
    }
    let _pfi = crate::profiler::scope("render.frame_instances");
    meshes.living.append_frame_instances(
        world,
        scmap,
        fow_viewer,
        zoom_boost,
        strategic_zoom,
        &mut batches,
        &|id| meshes.mesh_for(id, Role::Mobile).albedo.is_some(),
    );

    // 5. Build placement ghost
    if let (Some(btype), Some(hover), None) = (&session.build_mode, session.hover_world, session.ui.build_drag) {
        let mut hx = world_f32_to_milli(hover);
        let is_mex = world
            .catalog()
            .get(btype)
            .is_some_and(|t| t.role == Role::Extractor);
        let valid = if is_mex {
            if let Some((_, p)) = world.snap_extractor(hx.0, hx.1) {
                hx = p;
                true
            } else {
                false
            }
        } else {
            world.site_is_free(btype, hx.0, hx.1)
        };
        let gx = hx.0 as f32 / scale;
        let gz = hx.1 as f32 / scale;
        let gy = sample_terrain_y(scmap, gx, gz) + 0.25;
        let radius = world
            .catalog()
            .get(btype)
            .map(|t| t.radius as f32 / scale)
            .unwrap_or(2.0);
        let gcol = if valid {
            [0.25, 0.95, 0.45, 0.35]
        } else {
            [0.98, 0.22, 0.22, 0.35]
        };
        append_ring(&mut verts, &mut indices, [gx, gy, gz], radius, 0.5 * zoom_boost.sqrt(), gcol, 20);
    }

    // 6. Box selection rectangle
    if let Some((a, b)) = session.drag_box_world {
        let x0 = a[0].min(b[0]);
        let x1 = a[0].max(b[0]);
        let z0 = a[2].min(b[2]);
        let z1 = a[2].max(b[2]);
        if (x1 - x0) > 0.5 || (z1 - z0) > 0.5 {
            let y00 = sample_terrain_y(scmap, x0, z0) + 0.5;
            let y10 = sample_terrain_y(scmap, x1, z0) + 0.5;
            let y11 = sample_terrain_y(scmap, x1, z1) + 0.5;
            let y01 = sample_terrain_y(scmap, x0, z1) + 0.5;
            let col = [0.35, 1.0, 0.55, 0.25];
            let w = 0.28 * zoom_boost.sqrt();
            append_beam(&mut verts, &mut indices, [x0, y00, z0], [x1, y10, z0], w, col);
            append_beam(&mut verts, &mut indices, [x1, y10, z0], [x1, y11, z1], w, col);
            append_beam(&mut verts, &mut indices, [x1, y11, z1], [x0, y01, z1], w, col);
            append_beam(&mut verts, &mut indices, [x0, y01, z1], [x0, y00, z0], w, col);
        }
    }

    // 7. Phase 13 overlays: range rings, formation preview, area circle, drag-line ghosts
    crate::controls::append_ui_overlays(session, scmap, &mut verts, &mut indices, zoom_boost);

    (batches, verts, indices)
}

#[allow(dead_code)]
pub fn build_skirmish_dynamic_mesh(
    session: &PlaySession,
    scmap: &ScMap,
    meshes: &UnitMeshCache,
    camera: &OrbitCamera,
) -> (Vec<GpuVertex>, Vec<u32>) {
    let (_batches, verts, indices) = build_skirmish_render_frame(session, scmap, meshes, camera);
    (verts, indices)
}

fn append_box_world(
    verts: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    base_center: [f32; 3],
    half_xz: f32,
    height: f32,
    color: [f32; 4],
) {
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut uvs = Vec::new();
    let mut idx = Vec::new();
    append_local_box(
        &mut pos,
        &mut nrm,
        &mut uvs,
        &mut idx,
        base_center,
        [half_xz, height, half_xz],
    );
    let base = verts.len() as u32;
    for ((p, n), uv) in pos.into_iter().zip(nrm).zip(uvs) {
        verts.push(GpuVertex {
            position: p,
            normal: n,
            uv,
            color,
        });
    }
    indices.extend(idx.into_iter().map(|i| base + i));
}

pub(crate) fn append_ring(
    verts: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    center: [f32; 3],
    outer_r: f32,
    thickness: f32,
    color: [f32; 4],
    segments: u32,
) {
    let inner_r = (outer_r - thickness).max(0.05);
    let base = verts.len() as u32;
    let n = [0.0, 1.0, 0.0];
    for i in 0..segments {
        let ang = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let (s, c) = ang.sin_cos();
        verts.push(GpuVertex {
            position: [center[0] + inner_r * c, center[1], center[2] + inner_r * s],
            normal: n,
            uv: [0.5, 0.5],
            color,
        });
        verts.push(GpuVertex {
            position: [center[0] + outer_r * c, center[1], center[2] + outer_r * s],
            normal: n,
            uv: [0.5, 0.5],
            color,
        });
    }
    for i in 0..segments {
        let next = (i + 1) % segments;
        let i0 = base + i * 2;
        let o0 = i0 + 1;
        let i1 = base + next * 2;
        let o1 = i1 + 1;
        indices.extend_from_slice(&[i0, o0, i1, i1, o0, o1]);
    }
}

fn append_bar_billboard(
    verts: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    center: [f32; 3],
    width: f32,
    height: f32,
    fill_frac: f32,
    fill_color: [f32; 4],
) {
    let x0 = center[0] - width * 0.5;
    let x1 = center[0] + width * 0.5;
    let xf = x0 + width * fill_frac.clamp(0.0, 1.0);
    let y0 = center[1];
    let y1 = center[1] + height;
    let z = center[2];
    let n = [0.0, 1.0, 0.0];

    let bg = [0.08, 0.08, 0.10, 0.0];
    let b = verts.len() as u32;
    verts.extend_from_slice(&[
        GpuVertex { position: [x0, y0, z], normal: n, uv: [0.5, 0.5], color: bg },
        GpuVertex { position: [x1, y0, z], normal: n, uv: [0.5, 0.5], color: bg },
        GpuVertex { position: [x1, y1, z], normal: n, uv: [0.5, 0.5], color: bg },
        GpuVertex { position: [x0, y1, z], normal: n, uv: [0.5, 0.5], color: bg },
    ]);
    indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);

    if xf > x0 + 0.01 {
        let f = verts.len() as u32;
        verts.extend_from_slice(&[
            GpuVertex { position: [x0, y0, z + 0.05], normal: n, uv: [0.5, 0.5], color: fill_color },
            GpuVertex { position: [xf, y0, z + 0.05], normal: n, uv: [0.5, 0.5], color: fill_color },
            GpuVertex { position: [xf, y1, z + 0.05], normal: n, uv: [0.5, 0.5], color: fill_color },
            GpuVertex { position: [x0, y1, z + 0.05], normal: n, uv: [0.5, 0.5], color: fill_color },
        ]);
        indices.extend_from_slice(&[f, f + 1, f + 2, f, f + 2, f + 3]);
    }
}

pub(crate) fn append_beam(
    verts: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    a: [f32; 3],
    b: [f32; 3],
    half_width: f32,
    color: [f32; 4],
) {
    let dx = b[0] - a[0];
    let dz = b[2] - a[2];
    let len = (dx * dx + dz * dz).sqrt();
    if len < 1e-4 {
        return;
    }
    let px = (-dz / len) * half_width;
    let pz = (dx / len) * half_width;
    let n = [0.0, 1.0, 0.0];
    let base = verts.len() as u32;
    verts.extend_from_slice(&[
        GpuVertex { position: [a[0] - px, a[1], a[2] - pz], normal: n, uv: [0.5, 0.5], color },
        GpuVertex { position: [a[0] + px, a[1], a[2] + pz], normal: n, uv: [0.5, 0.5], color },
        GpuVertex { position: [b[0] + px, b[1], b[2] + pz], normal: n, uv: [0.5, 0.5], color },
        GpuVertex { position: [b[0] - px, b[1], b[2] - pz], normal: n, uv: [0.5, 0.5], color },
    ]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn normalize_or_up(v: [f32; 3]) -> [f32; 3] {
    let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if l2.is_finite() && l2 > 1e-12 {
        let inv = 1.0 / l2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0, 1.0, 0.0]
    }
}

// ---------------------------------------------------------------------------
// GPU Upload Helpers, Scene, and Camera
// ---------------------------------------------------------------------------

pub fn initial_army_camera(world: &Skirmish, scmap: &ScMap, army: ArmyId, aspect: f32) -> OrbitCamera {
    let scale = SCALE as f32;
    let focus_xz = world
        .commander_of(army)
        .map(|u| (u.pos.x as f32 / scale, u.pos.y as f32 / scale))
        .or_else(|| world.army(army).map(|a| (a.start.0 as f32 / scale, a.start.1 as f32 / scale)))
        .unwrap_or((scmap.map_width * 0.5, scmap.map_height * 0.5));
    let fy = sample_terrain_y(scmap, focus_xz.0, focus_xz.1);
    let span = scmap.map_width.max(scmap.map_height).max(128.0);
    let mut cam = OrbitCamera::for_map(scmap.map_width, scmap.map_height, fy, aspect);
    cam.target = [focus_xz.0, fy, focus_xz.1];
    cam.yaw = 0.55;
    cam.pitch = 0.95;
    cam.distance = 220.0_f32.clamp(12.0, span * 1.2);
    cam.max_distance = span * 3.5;
    cam
}

pub fn battle_snapshot_camera(world: &Skirmish, scmap: &ScMap, aspect: f32) -> OrbitCamera {
    let scale = SCALE as f32;
    let water_y = if scmap.water.has_water {
        scmap.water.elevation
    } else {
        f32::NEG_INFINITY
    };
    let units: Vec<_> = world.units().collect();

    // Pick a unit on dry land right in the thickest active battle cluster.
    let focus = units
        .iter()
        .filter_map(|u| {
            let ux = u.pos.x as f32 / scale;
            let uz = u.pos.y as f32 / scale;
            if sample_terrain_y(scmap, ux, uz) < water_y + 0.5 {
                return None;
            }
            let recent_shot = u
                .last_shot
                .is_some_and(|(_, tick)| tick + 80 >= world.tick());
            let mut min_enemy_d2 = f32::MAX;
            let mut enemies_near = 0u64;
            let mut allies_near = 0u64;
            for o in &units {
                let dx = o.pos.x as f32 / scale - ux;
                let dz = o.pos.y as f32 / scale - uz;
                let d2 = dx * dx + dz * dz;
                if o.army != u.army {
                    if d2 < min_enemy_d2 {
                        min_enemy_d2 = d2;
                    }
                    if d2 <= 36.0 * 36.0 {
                        enemies_near += 1;
                    }
                } else if d2 <= 28.0 * 28.0 {
                    allies_near += 1;
                }
            }
            // Also check active combat near this unit vs nearest enemy distance.
            let closeness = (5000 - (min_enemy_d2.sqrt() as u64).min(5000)) as u64;
            let score = (u64::from(recent_shot) << 44)
                | (enemies_near.min(63) << 36)
                | (allies_near.min(63) << 28)
                | closeness;
            Some((score, (ux, uz)))
        })
        .max_by_key(|(score, _)| *score)
        .map(|(_, c)| c);

    let Some((cx, cz)) = focus else {
        return OrbitCamera::for_map(scmap.map_width, scmap.map_height, 20.0, aspect);
    };
    let cy = sample_terrain_y(scmap, cx, cz).max(water_y);

    let span = scmap.map_width.max(scmap.map_height).max(256.0);
    let mut cam = OrbitCamera::for_map(scmap.map_width, scmap.map_height, cy, aspect);
    cam.target = [cx, cy, cz];
    cam.yaw = 0.50;
    cam.pitch = 0.44;
    cam.distance = 44.0_f32.clamp(28.0, span * 0.25);
    cam
}

pub(crate) fn build_terrain_loaded_scene(
    game_dir: Option<&Path>,
    map: &MapContext,
    camera: OrbitCamera,
) -> (LoadedScene, Option<RgbaImage>) {
    let terrain_splat = game_dir.and_then(|d| scene::load_terrain_splat_data(d, &map.scmap));
    let water = game_dir.and_then(|d| scene::load_water_gpu_data(d, &map.scmap));
    let sky_cube = game_dir.and_then(|d| scene::load_sky_cubemap(d, &map.scmap));
    let prop_batches = game_dir
        .map(|d| scene::load_map_props(d, &map.scmap))
        .unwrap_or_default();
    let decal_batches = game_dir
        .map(|d| scene::load_map_decals(d, &map.scmap))
        .unwrap_or_default();

    let baked = if terrain_splat.is_some() {
        None
    } else {
        game_dir.and_then(|d| scene::bake_terrain_albedo(d, &map.scmap, 1024))
    };
    let textured = terrain_splat.is_some() || baked.is_some();
    let (vertices, indices, overlay_vertices, overlay_indices) =
        scene::build_terrain_and_overlay_meshes_opts(&map.scmap, None, textured, water.is_none());
    let minimap_img = decode_dds_rgba8(&map.scmap.preview_dds)
        .ok()
        .or_else(|| baked.clone());
    let albedo = baked.unwrap_or_else(|| RgbaImage {
        width: 1,
        height: 1,
        pixels: vec![255, 255, 255, 255],
    });
    let fc = map.scmap.lighting.fog_color;
    let scene = LoadedScene {
        title: format!("{} ({})", map.map_name, map.map_id),
        vertices,
        indices,
        overlay_vertices,
        overlay_indices,
        albedo,
        initial_camera: camera,
        clear_color: [
            f64::from(fc[0].clamp(0.05, 1.0)),
            f64::from(fc[1].clamp(0.05, 1.0)),
            f64::from(fc[2].clamp(0.05, 1.0)),
            1.0,
        ],
        terrain_splat,
        water,
        sky_cube,
        prop_batches,
        decal_batches,
    };
    (scene, minimap_img)
}

pub(crate) fn upload_unit_templates(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &GpuSceneResources,
    meshes: &UnitMeshCache,
) -> BTreeMap<String, GpuMeshTemplate> {
    let mut templates = BTreeMap::new();
    for (id, nm) in &meshes.by_type {
        let verts: Vec<GpuVertex> = nm
            .positions
            .iter()
            .zip(nm.normals.iter())
            .zip(nm.uvs.iter())
            .enumerate()
            .map(|(vi, ((&p, &n), &uv))| GpuVertex {
                position: p,
                normal: n,
                uv,
                color: [1.0, 1.0, 1.0, meshes.living.vertex_bone(id, vi)],
            })
            .collect();
        let tpl = resources.create_mesh_template(
            device,
            queue,
            &verts,
            &nm.indices,
            nm.albedo.as_ref(),
            id,
        );
        templates.insert(id.clone(), tpl);
    }
    meshes.living.create_lod_templates(device, queue, resources, &mut templates);
    templates
}

pub(crate) fn upload_frame_resources(
    device: &wgpu::Device,
    resources: &mut GpuSceneResources,
    templates: &BTreeMap<String, GpuMeshTemplate>,
    session: &PlaySession,
    map: &MapContext,
    meshes: &UnitMeshCache,
    camera: &OrbitCamera,
    width: u32,
    height: u32,
) {
    let _p = crate::profiler::scope("render.prep_upload");
    let (batches, dyn_v, dyn_i) = build_skirmish_render_frame(session, &map.scmap, meshes, camera);
    {
        let _p = crate::profiler::scope("render.upload_units");
        let gpu_batches: Vec<(&GpuMeshTemplate, &[GpuUnitInstance])> = batches
            .iter()
            .filter_map(|(tid, insts)| templates.get(tid).map(|tpl| (tpl, insts.as_slice())))
            .collect();
        resources.set_unit_batches(device, gpu_batches);
        resources.set_dynamic(device, &dyn_v, &dyn_i);
    }
    let fow_viewer = session.fog_of_war.then_some(session.human_army);
    {
        let _p = crate::profiler::scope("render.upload_living");
        meshes.living.upload(device, resources, &session.world, &map.scmap, fow_viewer, camera, width, height);
    }

    let _p = crate::profiler::scope("render.hud");
    let mut g = hud::build_hud_geometry_ex(session, map, camera, width, height, crate::settings::ui_theme());
    crate::profiler::append_overlay(&mut g.bg_v, &mut g.bg_i, &mut g.fg_v, &mut g.fg_i, width as f32, height as f32);
    resources.set_hud(device, &g.mm_v, &g.mm_i, &g.bg_v, &g.bg_i, &g.ic_v, &g.ic_i, &g.fg_v, &g.fg_i);
    resources.set_hud_ttf(device, &g.ttf_v, &g.ttf_i);
}

pub(crate) fn upload_hud_icons(
    game_dir: Option<&Path>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &mut GpuSceneResources,
    session: &PlaySession,
) {
    // Phase 23/37: grid atlas of every catalog unit icon, TrueType HUD font atlas, and FA localization tables.
    let Some(dir) = game_dir else {
        crate::command_card::set_atlas_ids(None);
        hud::set_hud_fonts(None);
        return;
    };
    crate::loc::init(dir);
    let hud_fonts = std::sync::Arc::new(crate::ui_kit::Fonts::load(dir));
    if hud_fonts.has_faces() {
        resources.set_hud_font_texture(device, queue, &hud_fonts.atlas);
        hud::set_hud_fonts(Some(hud_fonts));
    } else {
        hud::set_hud_fonts(None);
    }
    let ids = crate::command_card::atlas_icon_ids(session.world.catalog());
    match crate::command_card::load_icon_atlas(dir, &ids) {
        Some((atlas, found_ids)) => {
            resources.set_hud_icons_texture(device, queue, &atlas);
            crate::command_card::set_atlas_ids(Some(found_ids));
        }
        None => crate::command_card::set_atlas_ids(None),
    }
}

// ---------------------------------------------------------------------------
// Offscreen PNG snapshot renderer
// ---------------------------------------------------------------------------

pub fn render_skirmish_snapshot(
    game_dir: Option<&Path>,
    session: &PlaySession,
    map: &MapContext,
    meshes: &UnitMeshCache,
    width: u32,
    height: u32,
    out_path: &Path,
    camera_override: Option<OrbitCamera>,
) -> Result<String, String> {
    if width == 0 || height == 0 {
        return Err("render dimensions must be non-zero".to_string());
    }
    let aspect = width as f32 / height as f32;
    let mut camera = camera_override
        .unwrap_or_else(|| initial_army_camera(&session.world, &map.scmap, session.human_army, aspect));
    camera.set_aspect(width, height);
    // Test hook: `FARUST_SNAPSHOT_SELECT=1` pre-selects the human army so selection overlays
    // (range rings, stats card) appear in offscreen snapshots. Phase 23: `=acu`, `=engineer`,
    // `=factory` (with a queue) or `=enhance` (ACU mid-enhancement) select one unit class for
    // the command card / queue / enhancement panels; `FARUST_SNAPSHOT_HOVER=card` hovers the
    // first card entry (tooltip), `FARUST_SNAPSHOT_TAB=N` picks the card tech tab.
    if let Some(dir) = game_dir {
        crate::loc::init(dir);
    }
    let preselected;
    let session = if let Ok(sel) = std::env::var("FARUST_SNAPSHOT_SELECT") {
        let mut s = session.clone();
        if !crate::command_card::snapshot_select(&mut s, sel.trim()) {
            s.preselect_army_for_snapshot();
        }
        if let Some(t) = std::env::var("FARUST_SNAPSHOT_TAB").ok().and_then(|v| v.trim().parse::<u8>().ok()) {
            s.set_card_tab(t);
        }
        if let Ok(hov) = std::env::var("FARUST_SNAPSHOT_HOVER") {
            let layout = HudLayout::compute(&s, width, height);
            let hspec = hov.trim().to_ascii_lowercase();
            let (kind, idx) = match hspec.split_once(':') {
                Some((k, n)) => (k, n.parse::<usize>().unwrap_or(0)),
                None => (hspec.as_str(), 0),
            };
            let target_rect = match kind {
                "card" => layout.card.slots.get(idx).copied(),
                "order" => layout.buttons.get(idx).map(|b| [b.x, b.y, b.w, b.h]),
                "enhance" => layout.enh.as_ref().and_then(|e| e.rows.get(idx).map(|(r, _, _)| *r)),
                "group" => layout.card.groups.get(idx).map(|(r, _)| *r),
                "tab" => layout.card.tabs.get(idx).map(|(r, _)| *r),
                _ => None,
            };
            if let Some(r) = target_rect {
                s.ui.cursor_px = Some((r[0] + r[2] * 0.5, r[1] + r[3] * 0.5));
                s.ui.hud_hover = layout.hit_test_button(r[0] + r[2] * 0.5, r[1] + r[3] * 0.5);
            }
        }
        let scale = SCALE as f32;
        let pts: Vec<(f32, f32)> = s
            .selected
            .iter()
            .filter_map(|id| s.world.unit(*id))
            .map(|u| (u.pos.x as f32 / scale, u.pos.y as f32 / scale))
            .collect();
        if !pts.is_empty() {
            let n = pts.len() as f32;
            let cx = pts.iter().map(|p| p.0).sum::<f32>() / n;
            let cz = pts.iter().map(|p| p.1).sum::<f32>() / n;
            camera.target = [cx, sample_terrain_y(&map.scmap, cx, cz), cz];
            camera.distance = camera.distance.max(110.0);
        }
        preselected = s;
        &preselected
    } else {
        session
    };
    let (scene, minimap_img) = build_terrain_loaded_scene(game_dir, map, camera.clone());

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no compatible Vulkan/GPU adapter found".to_string())?;
    let adapter_info = adapter.get_info();
    let (device, queue) = pollster::block_on(
        adapter.request_device(&wgpu::DeviceDescriptor::default(), None),
    )
    .map_err(|e| format!("failed to create GPU device: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut resources = GpuSceneResources::new(&device, &queue, &scene, format, MSAA_SAMPLES, &camera);
    if let Some(ref mm) = minimap_img {
        resources.set_minimap_texture(&device, &queue, mm);
    }
    upload_hud_icons(game_dir, &device, &queue, &mut resources, session);
    let templates = upload_unit_templates(&device, &queue, &resources, meshes);
    upload_frame_resources(
        &device,
        &mut resources,
        &templates,
        session,
        map,
        meshes,
        &camera,
        width,
        height,
    );

    let (_msaa_tex, msaa_view) = viewer::create_attachment(
        &device,
        width,
        height,
        format,
        MSAA_SAMPLES,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "play_snap_msaa",
    );
    let (resolve_tex, resolve_view) = viewer::create_attachment(
        &device,
        width,
        height,
        format,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        "play_snap_resolve",
    );
    let (_depth_tex, depth_view) = viewer::create_attachment(
        &device,
        width,
        height,
        DEPTH_FORMAT,
        MSAA_SAMPLES,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "play_snap_depth",
    );

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded_bpr = width * 4;
    let padded_bpr = unpadded_bpr.div_ceil(align) * align;
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("play_snap_readback"),
        size: u64::from(padded_bpr) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("play_snap_encoder"),
    });
    resources.encode_draw(
        &mut encoder,
        &msaa_view,
        Some(&resolve_view),
        &depth_view,
        scene.clear_color,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &resolve_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("readback channel failed: {e}"))?
        .map_err(|e| format!("GPU buffer map failed: {e}"))?;

    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded_bpr * height) as usize);
    for row in 0..height as usize {
        let start = row * padded_bpr as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded_bpr as usize]);
    }
    drop(mapped);
    readback_buf.unmap();

    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let file = File::create(out_path)
        .map_err(|e| format!("failed to create {}: {e}", out_path.display()))?;
    let mut png_enc = png::Encoder::new(BufWriter::new(file), width, height);
    png_enc.set_color(png::ColorType::Rgba);
    png_enc.set_depth(png::BitDepth::Eight);
    let mut writer = png_enc
        .write_header()
        .map_err(|e| format!("png header error: {e}"))?;
    writer
        .write_image_data(&pixels)
        .map_err(|e| format!("png write error: {e}"))?;

    Ok(format!("{} ({:?})", adapter_info.name, adapter_info.backend))
}

// ---------------------------------------------------------------------------
// Interactive playable window (`winit` + `wgpu`)
// ---------------------------------------------------------------------------

pub fn run_play_window(
    game_dir: Option<&Path>,
    session: PlaySession,
    map: MapContext,
    meshes: UnitMeshCache,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("failed to create event loop: {e}"))?;
    let mut app = PlayApp::new(game_dir, session, map, meshes, width, height);
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("event loop error: {e}"))?;
    if let Some(err) = app.init_err {
        return Err(err);
    }
    Ok(())
}

struct PlayGpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    resources: GpuSceneResources,
    templates: BTreeMap<String, GpuMeshTemplate>,
    camera: OrbitCamera,
    clear_color: [f64; 4],
    modifiers: ModifiersState,
    cursor_px: Option<(f64, f64)>,
    lmb_down_px: Option<(f64, f64)>,
    rmb_down_px: Option<(f64, f64)>,
    rmb_last_px: Option<(f64, f64)>,
    rmb_dragged: bool,
    mmb_drag: bool,
    mmb_last_px: Option<(f64, f64)>,
    last_step: Instant,
    accum_secs: f64,
    dirty_dynamic: bool,
    /// Phase 23: window focus (edge scrolling) and the OS cursor currently shown.
    focused: bool,
    cursor_icon: winit::window::CursorIcon,
    /// Phase 34: every map prop batch as loaded, and how many reclaimed props the uploaded
    /// batches already hide (re-filtered at most once a second).
    prop_cpu: Vec<crate::scene::PropMeshBatch>,
    props_hidden: usize,
    props_filtered_at: Instant,
}

/// Phase 34: `batches` without the props whose `.scmap` (x, z) position bits are in `gone`.
fn filter_reclaimed_props(batches: &[crate::scene::PropMeshBatch], gone: &[(u32, u32)]) -> Vec<crate::scene::PropMeshBatch> {
    let gone: std::collections::BTreeSet<(u32, u32)> = gone.iter().copied().collect();
    batches
        .iter()
        .map(|b| {
            let mut b = b.clone();
            b.instances.retain(|i| !gone.contains(&(i.row0[3].to_bits(), i.row2[3].to_bits())));
            b
        })
        .collect()
}

/// How the player asked to leave a running match (pause menu / end-of-game screen).
/// A host shell (main menu, lobby) polls [`PlayApp::take_exit`] after each event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PlayExit {
    /// Return to the main menu / lobby (standalone windows simply exit).
    ToMenu,
    /// Quit the program.
    Quit,
}

pub(crate) struct PlayApp {
    game_dir: Option<std::path::PathBuf>,
    session: PlaySession,
    map: MapContext,
    meshes: UnitMeshCache,
    init_size: (u32, u32),
    state: Option<PlayGpuState>,
    init_err: Option<String>,
    /// Set when the player asked to leave the match; see [`PlayExit`].
    pub(crate) exit: Option<PlayExit>,
    /// True when hosted inside another app (menu/lobby) on the same event loop: a
    /// `PlayExit::ToMenu` request must then NOT call `event_loop.exit()`.
    pub(crate) embedded: bool,
}

impl PlayApp {
    /// Builds the windowed play app (window is created on `resumed`); also used by the
    /// in-window lobby (`lobby.rs`) to switch into the game on the same event loop.
    pub(crate) fn new(
        game_dir: Option<&Path>,
        session: PlaySession,
        map: MapContext,
        meshes: UnitMeshCache,
        width: u32,
        height: u32,
    ) -> Self {
        meshes.living.enable_wall_clock();
        PlayApp {
            game_dir: game_dir.map(Path::to_path_buf),
            session,
            map,
            meshes,
            init_size: (width.max(480), height.max(360)),
            state: None,
            init_err: None,
            exit: None,
            embedded: false,
        }
    }

    /// Takes the pending exit request, if the player asked to leave the match.
    #[allow(dead_code)]
    pub(crate) fn take_exit(&mut self) -> Option<PlayExit> {
        self.exit.take()
    }

    pub(crate) fn take_init_err(&mut self) -> Option<String> {
        self.init_err.take()
    }

    fn init_window(&mut self, event_loop: &ActiveEventLoop) -> Result<PlayGpuState, String> {
        let title = self.session.window_title(&self.map.map_name);
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(self.init_size.0, self.init_size.1));
        let attrs = crate::settings::apply_window_attrs(attrs);
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .map_err(|e| format!("failed to create window: {e}"))?,
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("failed to create wgpu surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| "no compatible GPU adapter for window surface".to_string())?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                ..wgpu::DeviceDescriptor::default()
            },
            None,
        ))
        .map_err(|e| format!("failed to request GPU device: {e}"))?;

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: crate::settings::present_mode(),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let aspect = width as f32 / height as f32;
        let camera = initial_army_camera(
            &self.session.world,
            &self.map.scmap,
            self.session.human_army,
            aspect,
        );
        let (terrain_scene, minimap_img) =
            build_terrain_loaded_scene(self.game_dir.as_deref(), &self.map, camera.clone());
        let clear_color = terrain_scene.clear_color;
        let prop_cpu = terrain_scene.prop_batches.clone();

        let mut resources =
            GpuSceneResources::new(&device, &queue, &terrain_scene, format, MSAA_SAMPLES, &camera);
        resources.enable_gpu_timer(&device, &queue);
        if let Some(ref mm) = minimap_img {
            resources.set_minimap_texture(&device, &queue, mm);
        }
        upload_hud_icons(
            self.game_dir.as_deref(),
            &device,
            &queue,
            &mut resources,
            &self.session,
        );
        let templates = upload_unit_templates(&device, &queue, &resources, &self.meshes);
        upload_frame_resources(
            &device,
            &mut resources,
            &templates,
            &self.session,
            &self.map,
            &self.meshes,
            &camera,
            width,
            height,
        );

        let (_msaa_tex, msaa_view) = viewer::create_attachment(
            &device,
            width,
            height,
            format,
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "play_win_msaa",
        );
        let (_depth_tex, depth_view) = viewer::create_attachment(
            &device,
            width,
            height,
            DEPTH_FORMAT,
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "play_win_depth",
        );

        Ok(PlayGpuState {
            window,
            surface,
            device,
            queue,
            config,
            msaa_view,
            depth_view,
            resources,
            templates,
            camera,
            clear_color,
            modifiers: ModifiersState::empty(),
            cursor_px: None,
            lmb_down_px: None,
            rmb_down_px: None,
            rmb_last_px: None,
            rmb_dragged: false,
            mmb_drag: false,
            mmb_last_px: None,
            last_step: Instant::now(),
            accum_secs: 0.0,
            dirty_dynamic: false,
            focused: true,
            cursor_icon: winit::window::CursorIcon::Default,
            prop_cpu,
            props_hidden: 0,
            props_filtered_at: Instant::now(),
        })
    }

    fn center_on_selection_or_commander(&mut self) {
        let Some(st) = self.state.as_mut() else {
            return;
        };
        let scale = SCALE as f32;
        let sel: Vec<(f32, f32)> = self
            .session
            .selected
            .iter()
            .filter_map(|id| self.session.world.unit(*id))
            .map(|u| (u.pos.x as f32 / scale, u.pos.y as f32 / scale))
            .collect();
        let pos = if sel.is_empty() {
            self.session
                .world
                .commander_of(self.session.human_army)
                .map(|u| (u.pos.x as f32 / scale, u.pos.y as f32 / scale))
        } else {
            let n = sel.len() as f32;
            Some((sel.iter().map(|p| p.0).sum::<f32>() / n, sel.iter().map(|p| p.1).sum::<f32>() / n))
        };
        if let Some((wx, wz)) = pos {
            let wy = sample_terrain_y(&self.map.scmap, wx, wz);
            st.camera.target = [wx, wy, wz];
            st.resources.update_camera(&st.queue, &st.camera);
            st.dirty_dynamic = true;
            st.window.request_redraw();
        }
    }

    fn leave_match(&mut self, event_loop: &ActiveEventLoop, target: PlayExit) {
        let _ = self.session.write_auto_replay(&self.map.map_id);
        self.exit = Some(target);
        if !self.embedded || target == PlayExit::Quit {
            event_loop.exit();
        }
    }

    fn quick_save(&mut self) {
        let path = crate::savegame::data_dir("saves").join(crate::savegame::QUICKSAVE_FILE);
        match self.session.save_game(&self.map.map_id, &self.map.map_name, &path) {
            Ok(_) => {
                let msg = format!("SAVED QUICKSAVE ({})", crate::controls::fmt_ticks(self.session.world.tick()));
                self.session.status_banner = msg.clone();
                self.session.chat.system(&msg);
            }
            Err(e) => {
                let msg = format!("SAVE FAILED: {e}");
                self.session.status_banner = msg.clone();
                self.session.chat.system(&msg);
            }
        }
    }

    fn load_save_file(&mut self, path: &Path) {
        if self.session.net.is_some() {
            self.session.status_banner = "LOAD DISABLED IN NETWORKED MATCHES".to_string();
            return;
        }
        match crate::savegame::SaveGame::read(path).and_then(|save| {
            let origin = crate::savegame::origin_world(self.session.world.catalog().clone(), &self.map, &save.replay)?;
            save.into_session(origin)
        }) {
            Ok(mut loaded) => {
                loaded.muted = self.session.muted;
                loaded.auto_record = self.session.auto_record;
                self.session = loaded;
            }
            Err(e) => {
                let msg = format!("LOAD FAILED: {e}");
                self.session.status_banner = msg.clone();
                self.session.chat.system(&msg);
            }
        }
    }

    fn switch_to_replay(&mut self, path: &Path) {
        match Replay::load(path).and_then(|rep| {
            let origin = crate::savegame::origin_world(self.session.world.catalog().clone(), &self.map, &rep)?;
            Ok((origin, rep))
        }) {
            Ok((origin, rep)) => {
                let muted = self.session.muted;
                let mut sess = PlaySession::new_replay(origin, rep, 1);
                sess.muted = muted;
                self.session = sess;
            }
            Err(e) => {
                self.session.status_banner = format!("REPLAY ERROR: {e}");
            }
        }
    }

    fn handle_match_menu_action(&mut self, event_loop: &ActiveEventLoop, act: hud::MatchMenuAction) {
        match act {
            hud::MatchMenuAction::Resume => {
                self.session.menu = hud::MatchMenu::None;
            }
            hud::MatchMenuAction::QuickSave => {
                self.quick_save();
                self.session.menu = hud::MatchMenu::None;
            }
            hud::MatchMenuAction::OpenLoadList => {
                let entries = crate::savegame::list_saves(Some(&self.map.map_id), 24);
                self.session.menu = hud::MatchMenu::LoadList { entries, selected: 0 };
            }
            hud::MatchMenuAction::OpenScore => {
                self.session.menu = hud::MatchMenu::Score {
                    series: crate::stats::Series::MassIncome,
                };
            }
            hud::MatchMenuAction::SelectLoad(idx) => {
                if let hud::MatchMenu::LoadList { selected, .. } = &mut self.session.menu {
                    *selected = idx;
                }
            }
            hud::MatchMenuAction::LoadSelected(path) => {
                self.load_save_file(&path);
            }
            hud::MatchMenuAction::SwitchScoreSeries(series) => {
                if let hud::MatchMenu::Score { series: s } = &mut self.session.menu {
                    *s = series;
                }
            }
            hud::MatchMenuAction::WatchReplay => {
                match self.session.write_auto_replay(&self.map.map_id) {
                    Ok(Some(p)) => self.switch_to_replay(&p),
                    Ok(None) => {
                        if let Some(p) = self.session.recorder.as_ref().and_then(|r| r.replay_path.clone()) {
                            self.switch_to_replay(&p);
                        } else {
                            self.session.status_banner = "NO REPLAY RECORDED".to_string();
                        }
                    }
                    Err(e) => self.session.status_banner = format!("REPLAY ERROR: {e}"),
                }
            }
            hud::MatchMenuAction::ContinueWatching => {
                self.session.end_shown = true;
                self.session.menu = hud::MatchMenu::None;
            }
            hud::MatchMenuAction::QuitToMenu => self.leave_match(event_loop, PlayExit::ToMenu),
            hud::MatchMenuAction::QuitGame => self.leave_match(event_loop, PlayExit::Quit),
        }
    }
}

impl ApplicationHandler for PlayApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            match self.init_window(event_loop) {
                Ok(st) => {
                    st.window.request_redraw();
                    self.state = Some(st);
                }
                Err(e) => {
                    self.init_err = Some(e);
                    event_loop.exit();
                }
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(st) = self.state.as_ref() {
            let _p = crate::profiler::scope("audio.frame");
            crate::audio_play::frame(
                &mut self.session,
                Some(&self.map.scmap),
                self.game_dir.as_deref(),
                &st.camera,
            );
        }
        let Some(st) = self.state.as_mut() else {
            return;
        };
        let now = Instant::now();
        let dt = now.duration_since(st.last_step).as_secs_f64().min(0.25);
        st.last_step = now;

        if self.session.loading() {
            self.session.pump_load(Duration::from_millis(24));
            st.accum_secs = 0.0;
            st.dirty_dynamic = true;
            st.window
                .set_title(&self.session.window_title(&self.map.map_name));
            st.window.request_redraw();
            event_loop.set_control_flow(ControlFlow::Poll);
            return;
        }

        if self.session.seeking() {
            self.session.pump_replay_seek(Duration::from_millis(24));
            st.accum_secs = 0.0;
            st.dirty_dynamic = true;
            st.window
                .set_title(&self.session.window_title(&self.map.map_name));
            st.window.request_redraw();
            event_loop.set_control_flow(ControlFlow::Poll);
            return;
        }

        if self.session.sim_active() {
            let step_interval = 1.0 / (TICK_HZ as f64 * self.session.speed as f64);
            st.accum_secs += dt;
            let mut stepped = false;
            let mut steps = 0;
            while st.accum_secs >= step_interval && steps < 8 {
                st.accum_secs -= step_interval;
                {
                    let _p = crate::profiler::scope("sim.step_sim");
                    self.session.step_sim();
                }
                stepped = true;
                steps += 1;
            }
            if stepped {
                if self.session.end_of_game_due() {
                    self.session.end_shown = true;
                    self.session.menu = hud::MatchMenu::Score {
                        series: crate::stats::Series::MassIncome,
                    };
                    let _ = self.session.write_auto_replay(&self.map.map_id);
                }
                st.dirty_dynamic = true;
                st.window
                    .set_title(&self.session.window_title(&self.map.map_name));
                st.window.request_redraw();
            }
        } else {
            st.accum_secs = 0.0;
        }
        if self.session.end_of_game_due() {
            self.session.end_shown = true;
            self.session.menu = hud::MatchMenu::Score {
                series: crate::stats::Series::MassIncome,
            };
            let _ = self.session.write_auto_replay(&self.map.map_id);
            st.dirty_dynamic = true;
            st.window.request_redraw();
        }
        if self.meshes.living.wants_continuous_redraw() {
            st.dirty_dynamic = true;
            st.window.request_redraw();
        }
        // Phase 23: animated map pings keep redrawing until they expire.
        if !self.session.chat.pings.is_empty() {
            self.session.chat.prune_pings(now);
            st.dirty_dynamic = true;
            st.window.request_redraw();
        }
        // Phase 23: edge scrolling (Options -> Edge Scroll).
        if st.focused
            && !self.session.menu.is_open()
            && !self.session.chat.is_typing()
            && !st.mmb_drag
            && st.rmb_down_px.is_none()
            && crate::settings::edge_scroll_enabled()
            && let Some((cx, cy)) = st.cursor_px
        {
            let (dx, dy) = crate::command_card::edge_scroll_dir(
                cx as f32,
                cy as f32,
                st.config.width as f32,
                st.config.height as f32,
                4.0,
            );
            if dx != 0.0 || dy != 0.0 {
                let speed = 1.1 * dt as f32;
                st.camera.pan(dx * speed, dy * speed);
                st.camera.target[0] = st.camera.target[0].clamp(0.0, self.map.scmap.map_width);
                st.camera.target[2] = st.camera.target[2].clamp(0.0, self.map.scmap.map_height);
                st.resources.update_camera(&st.queue, &st.camera);
                st.dirty_dynamic = true;
                st.window.request_redraw();
            }
        }

        event_loop.set_control_flow(ControlFlow::WaitUntil(now + Duration::from_millis(16)));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        crate::audio_play::window_event(&mut self.session, &event);
        let Some(st) = self.state.as_mut() else {
            return;
        };
        if st.window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                self.leave_match(event_loop, PlayExit::Quit);
            }
            WindowEvent::ModifiersChanged(mods) => {
                st.modifiers = mods.state();
            }
            WindowEvent::Focused(f) => {
                st.focused = f;
            }
            WindowEvent::CursorLeft { .. } => {
                st.cursor_px = None;
                self.session.ui.cursor_px = None;
                if self.session.ui.hud_hover.take().is_some() {
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                }
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width > 0 && new_size.height > 0 {
                    st.config.width = new_size.width;
                    st.config.height = new_size.height;
                    st.surface.configure(&st.device, &st.config);
                    let (_, msaa_view) = viewer::create_attachment(
                        &st.device,
                        new_size.width,
                        new_size.height,
                        st.config.format,
                        MSAA_SAMPLES,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                        "play_win_msaa",
                    );
                    let (_, depth_view) = viewer::create_attachment(
                        &st.device,
                        new_size.width,
                        new_size.height,
                        DEPTH_FORMAT,
                        MSAA_SAMPLES,
                        wgpu::TextureUsages::RENDER_ATTACHMENT,
                        "play_win_depth",
                    );
                    st.msaa_view = msaa_view;
                    st.depth_view = depth_view;
                    st.camera.set_aspect(new_size.width, new_size.height);
                    // Window re-config: apply the active shadow quality live.
                    st.resources
                        .set_shadow_quality(&st.device, crate::settings::shadow_quality());
                    st.resources.update_camera(&st.queue, &st.camera);
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cur = (position.x, position.y);
                st.cursor_px = Some(cur);
                let hit = pick_terrain_point(
                    &st.camera,
                    &self.map.scmap,
                    cur.0 as f32,
                    cur.1 as f32,
                    st.config.width,
                    st.config.height,
                );
                let hud_layout =
                    HudLayout::compute(&self.session, st.config.width, st.config.height);
                if let Some(ref mut rui) = self.session.ui.replay
                    && rui.dragging
                    && let Some(ref tl) = hud_layout.timeline
                {
                    let fin = self.session.replay_progress().map(|p| p.1).unwrap_or(0);
                    if let Some(ref mut rui) = self.session.ui.replay {
                        rui.drag_tick = Some(crate::controls::seek_target_tick(tl.frac_at(cur.0 as f32), fin));
                    }
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                    return;
                }
                let over_hud = hud_layout.over_hud(cur.0 as f32, cur.1 as f32);
                // Phase 23: tooltips / cursor-mode label follow the cursor.
                self.session.ui.cursor_px = Some((cur.0 as f32, cur.1 as f32));
                let hud_hover = hud_layout.hit_test_button(cur.0 as f32, cur.1 as f32);
                if hud_hover != self.session.ui.hud_hover
                    || self.session.current_order_mode().is_some()
                    || self.session.build_mode.is_some()
                {
                    self.session.ui.hud_hover = hud_hover;
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                }
                let hover_xz = if over_hud { None } else { hit.map(world_f32_to_milli) };
                if self
                    .session
                    .update_hover_unit(hover_xz, pick_radius_for_camera(&st.camera))
                {
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                }
                if self.session.build_mode.is_some() || st.lmb_down_px.is_some() {
                    self.session.hover_world = hit;
                    if let (Some(start_px), Some(cur_w)) = (st.lmb_down_px, hit) {
                        let d = (cur.0 - start_px.0).hypot(cur.1 - start_px.1);
                        if d > 6.0
                            && self.session.build_mode.is_some()
                            && let Some(start_xz) = self.session.ui.lmb_down_world
                        {
                            self.session.ui.build_drag = Some((start_xz, world_f32_to_milli(cur_w)));
                        }
                        if d > 6.0
                            && self.session.build_mode.is_none()
                            && !self.session.attack_move_mode
                            && let Some(start_w) = pick_terrain_point(
                                &st.camera,
                                &self.map.scmap,
                                start_px.0 as f32,
                                start_px.1 as f32,
                                st.config.width,
                                st.config.height,
                            )
                        {
                            self.session.drag_box_world = Some((start_w, cur_w));
                        }
                    }
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                }

                if let Some(down) = st.rmb_down_px {
                    if (cur.0 - down.0).hypot(cur.1 - down.1) > 6.0 {
                        st.rmb_dragged = true;
                        if self.session.ui.rmb_drag.is_none() {
                            let free_mode = self.session.build_mode.is_none()
                                && !self.session.attack_move_mode
                                && !self.session.is_replay()
                                && !st.modifiers.alt_key()
                                && self.session.ui.rmb_down_world.is_some();
                            let has_builder = self.session.selected.iter().any(|id| {
                                self.session.world.unit(*id).is_some_and(|u| {
                                    u.army == self.session.human_army && u.role.is_builder()
                                })
                            });
                            self.session.ui.rmb_drag = Some(if free_mode && st.modifiers.control_key() && has_builder {
                                crate::controls::RmbDrag::Area
                            } else if free_mode && self.session.has_mobile_selection() {
                                crate::controls::RmbDrag::Formation
                            } else {
                                crate::controls::RmbDrag::Pan
                            });
                        }
                    }
                    let drag_mode = self.session.ui.rmb_drag;
                    if matches!(
                        drag_mode,
                        Some(crate::controls::RmbDrag::Formation | crate::controls::RmbDrag::Area)
                    ) {
                        if let (Some(a), Some(cur_w)) = (self.session.ui.rmb_down_world, hit) {
                            let b = world_f32_to_milli(cur_w);
                            if drag_mode == Some(crate::controls::RmbDrag::Formation) {
                                self.session.update_formation_drag(a, b);
                            } else {
                                self.session.ui.area_drag = Some((a, b));
                            }
                            st.dirty_dynamic = true;
                            st.window.request_redraw();
                        }
                    } else if st.rmb_dragged
                        && let Some((px, py)) = st.rmb_last_px
                    {
                        let dx = (cur.0 - px) as f32;
                        let dy = (cur.1 - py) as f32;
                        st.camera.pan(-dx * 0.0018, dy * 0.0018);
                        st.resources.update_camera(&st.queue, &st.camera);
                        st.dirty_dynamic = true;
                        st.window.request_redraw();
                    }
                    st.rmb_last_px = Some(cur);
                }

                if st.mmb_drag {
                    if let Some((px, py)) = st.mmb_last_px {
                        let dx = (cur.0 - px) as f32;
                        let dy = (cur.1 - py) as f32;
                        st.camera.orbit(-dx * 0.006, dy * 0.006);
                        st.resources.update_camera(&st.queue, &st.camera);
                        st.dirty_dynamic = true;
                        st.window.request_redraw();
                    }
                    st.mmb_last_px = Some(cur);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let cur = st.cursor_px.unwrap_or((
                    st.config.width as f64 * 0.5,
                    st.config.height as f64 * 0.5,
                ));
                let shift = st.modifiers.shift_key();
                let hud_layout =
                    HudLayout::compute(&self.session, st.config.width, st.config.height);
                if self.session.loading() {
                    return;
                }
                if pressed && button == MouseButton::Left
                    && let Some(ml) = hud::MatchMenuLayout::compute(&self.session, st.config.width, st.config.height)
                {
                    if let Some(act) = ml.hit(cur.0 as f32, cur.1 as f32) {
                        self.handle_match_menu_action(event_loop, act);
                    }
                    if let Some(st) = self.state.as_mut() {
                        st.dirty_dynamic = true;
                        st.window.set_title(&self.session.window_title(&self.map.map_name));
                        st.window.request_redraw();
                    }
                    return;
                }

                // Phase 23 map pings: Alt+LMB alert, Alt+Shift+LMB attack, Alt+RMB move.
                if pressed
                    && st.modifiers.alt_key()
                    && matches!(button, MouseButton::Left | MouseButton::Right)
                    && !self.session.is_replay()
                {
                    let (px, py) = (cur.0 as f32, cur.1 as f32);
                    let target = if let Some((u, v)) = hud_layout.hit_test_minimap(px, py) {
                        Some(((u as f64 * self.map.size.0 as f64) as i64, (v as f64 * self.map.size.1 as f64) as i64))
                    } else if hud_layout.over_hud(px, py) {
                        None
                    } else {
                        pick_terrain_point(&st.camera, &self.map.scmap, px, py, st.config.width, st.config.height)
                            .map(world_f32_to_milli)
                    };
                    if let Some((x, z)) = target {
                        let kind = match (button, shift) {
                            (MouseButton::Right, _) => crate::chat::PingKind::Move,
                            (_, true) => crate::chat::PingKind::Attack,
                            _ => crate::chat::PingKind::Alert,
                        };
                        self.session.chat.send_ping(kind, x, z);
                        self.session.flush_local_chat();
                        self.session.status_banner = format!("{} ping sent to allies", kind.label());
                        st.dirty_dynamic = true;
                        st.window.request_redraw();
                        return;
                    }
                }

                match button {
                    MouseButton::Left => {
                        if !pressed
                            && let Some(ref mut rui) = self.session.ui.replay
                            && rui.dragging
                        {
                            rui.dragging = false;
                            if let Some(t) = rui.drag_tick.take() {
                                self.session.request_seek_tick(t);
                            }
                            st.dirty_dynamic = true;
                            st.window.request_redraw();
                            return;
                        }
                        if pressed
                            && let Some(ref tl) = hud_layout.timeline
                            && let Some(th) = tl.hit(cur.0 as f32, cur.1 as f32)
                        {
                            let max_speed = self.session.max_speed();
                            match th {
                                hud::TimelineHit::Seek(frac) => {
                                    let fin = self.session.replay_progress().map(|p| p.1).unwrap_or(0);
                                    if let Some(ref mut rui) = self.session.ui.replay {
                                        rui.dragging = true;
                                        rui.drag_tick = Some(crate::controls::seek_target_tick(frac, fin));
                                    }
                                }
                                hud::TimelineHit::TogglePause => self.session.paused = !self.session.paused,
                                hud::TimelineHit::Slower => {
                                    self.session.speed = (self.session.speed / 2).clamp(1, max_speed)
                                }
                                hud::TimelineHit::Faster => {
                                    self.session.speed = (self.session.speed * 2).clamp(1, max_speed)
                                }
                                hud::TimelineHit::Perspective => self.session.cycle_replay_perspective(),
                            }
                            st.dirty_dynamic = true;
                            st.window.request_redraw();
                            return;
                        }
                        if pressed {
                            if let Some(act) = hud_layout.hit_test_button(cur.0 as f32, cur.1 as f32) {
                                self.session.trigger_hud_action(act, shift);
                                st.dirty_dynamic = true;
                                st.window.request_redraw();
                                return;
                            }
                            if let Some((u, v)) = hud_layout.hit_test_minimap(cur.0 as f32, cur.1 as f32) {
                                let wx = u * self.map.scmap.map_width;
                                let wz = v * self.map.scmap.map_height;
                                let wy = sample_terrain_y(&self.map.scmap, wx, wz);
                                st.camera.target = [wx, wy, wz];
                                st.resources.update_camera(&st.queue, &st.camera);
                                st.dirty_dynamic = true;
                                st.window.request_redraw();
                                return;
                            }
                            if hud_layout.over_hud(cur.0 as f32, cur.1 as f32) {
                                return;
                            }
                            st.lmb_down_px = Some(cur);
                            self.session.ui.lmb_down_world = pick_terrain_point(
                                &st.camera,
                                &self.map.scmap,
                                cur.0 as f32,
                                cur.1 as f32,
                                st.config.width,
                                st.config.height,
                            )
                            .map(world_f32_to_milli);
                        } else if let Some(down) = st.lmb_down_px.take() {
                            self.session.drag_box_world = None;
                            self.session.ui.lmb_down_world = None;
                            let drag_dist = (cur.0 - down.0).hypot(cur.1 - down.1);
                            if self.session.ui.build_drag.is_some() {
                                if !self.session.finish_build_drag(shift)
                                    && let Some(hit) = pick_terrain_point(
                                        &st.camera,
                                        &self.map.scmap,
                                        cur.0 as f32,
                                        cur.1 as f32,
                                        st.config.width,
                                        st.config.height,
                                    )
                                {
                                    let pick_r = pick_radius_for_camera(&st.camera);
                                    self.session.issue_left_click_action(world_f32_to_milli(hit), shift, pick_r);
                                }
                                st.dirty_dynamic = true;
                                st.window.request_redraw();
                                return;
                            }
                            if let Some(hit) = pick_terrain_point(
                                &st.camera,
                                &self.map.scmap,
                                cur.0 as f32,
                                cur.1 as f32,
                                st.config.width,
                                st.config.height,
                            ) {
                                let pick_r = pick_radius_for_camera(&st.camera);
                                let world_xz = world_f32_to_milli(hit);
                                if drag_dist <= 6.0 || self.session.build_mode.is_some() || self.session.attack_move_mode {
                                    if !self.session.issue_left_click_action(world_xz, shift, pick_r) {
                                        let now = self.session.ui.now_secs();
                                        self.session.click_select(
                                            world_xz,
                                            shift,
                                            st.modifiers.control_key(),
                                            pick_r,
                                            now,
                                            &st.camera,
                                            &self.map.scmap,
                                            st.config.width as f32,
                                            st.config.height as f32,
                                        );
                                    }
                                } else if let Some(start_hit) = pick_terrain_point(
                                    &st.camera,
                                    &self.map.scmap,
                                    down.0 as f32,
                                    down.1 as f32,
                                    st.config.width,
                                    st.config.height,
                                ) && !self.session.finish_waypoint_drag(world_f32_to_milli(start_hit), world_xz, pick_r)
                                {
                                    self.session.select_box(
                                        world_f32_to_milli(start_hit),
                                        world_xz,
                                        shift,
                                    );
                                }
                            }
                            st.dirty_dynamic = true;
                            st.window
                                .set_title(&self.session.window_title(&self.map.map_name));
                            st.window.request_redraw();
                        }
                    }
                    MouseButton::Right => {
                        if pressed {
                            if let Some(act) = hud_layout.hit_test_button(cur.0 as f32, cur.1 as f32) {
                                self.session.trigger_hud_secondary(act, shift);
                                st.dirty_dynamic = true;
                                st.window.request_redraw();
                                return;
                            }
                            if let Some((u, v)) = hud_layout.hit_test_minimap(cur.0 as f32, cur.1 as f32) {
                                let wx = (u as f64 * self.map.size.0 as f64).round() as i64;
                                let wz = (v as f64 * self.map.size.1 as f64).round() as i64;
                                self.session.issue_right_click((wx, wz), shift, 4 * SCALE);
                                st.dirty_dynamic = true;
                                st.window.request_redraw();
                                return;
                            }
                            if hud_layout.over_hud(cur.0 as f32, cur.1 as f32) {
                                return;
                            }
                            st.rmb_down_px = Some(cur);
                            st.rmb_last_px = Some(cur);
                            st.rmb_dragged = false;
                            self.session.ui.rmb_drag = None;
                            self.session.ui.rmb_down_world = pick_terrain_point(
                                &st.camera,
                                &self.map.scmap,
                                cur.0 as f32,
                                cur.1 as f32,
                                st.config.width,
                                st.config.height,
                            )
                            .map(world_f32_to_milli);
                        } else {
                            let was_drag = st.rmb_dragged;
                            // A release without a world press (HUD / minimap) issues nothing.
                            if st.rmb_down_px.take().is_none() {
                                return;
                            }
                            st.rmb_last_px = None;
                            st.rmb_dragged = false;
                            self.session.ui.rmb_down_world = None;
                            match self.session.ui.rmb_drag.take() {
                                Some(crate::controls::RmbDrag::Formation) => {
                                    self.session.finish_formation_drag(shift);
                                    st.dirty_dynamic = true;
                                    st.window.request_redraw();
                                    return;
                                }
                                Some(crate::controls::RmbDrag::Area) => {
                                    if let Some((c, e)) = self.session.ui.area_drag.take() {
                                        self.session.issue_area_command(c, e, shift);
                                    }
                                    st.dirty_dynamic = true;
                                    st.window.request_redraw();
                                    return;
                                }
                                _ => {}
                            }
                            if !was_drag
                                && let Some(hit) = pick_terrain_point(
                                    &st.camera,
                                    &self.map.scmap,
                                    cur.0 as f32,
                                    cur.1 as f32,
                                    st.config.width,
                                    st.config.height,
                                )
                            {
                                let pick_r = pick_radius_for_camera(&st.camera);
                                self.session.issue_right_click(
                                    world_f32_to_milli(hit),
                                    shift,
                                    pick_r,
                                );
                                st.dirty_dynamic = true;
                                st.window
                                    .set_title(&self.session.window_title(&self.map.map_name));
                                st.window.request_redraw();
                            }
                        }
                    }
                    MouseButton::Middle => {
                        st.mmb_drag = pressed;
                        st.mmb_last_px = if pressed { Some(cur) } else { None };
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                if steps.abs() > 0.0 {
                    st.camera.zoom(steps);
                    st.resources.update_camera(&st.queue, &st.camera);
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let shift = st.modifiers.shift_key();
                let ctrl = st.modifiers.control_key();
                if self.session.chat.is_typing() {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => {
                            self.session.chat.input = None;
                        }
                        PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter) => {
                            self.session.chat.submit();
                            self.session.flush_local_chat();
                        }
                        PhysicalKey::Code(KeyCode::Backspace) => {
                            self.session.chat.backspace();
                        }
                        _ => {
                            if let Some(text) = &event.text {
                                self.session.chat.type_text(text.as_str());
                            }
                        }
                    }
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                    return;
                }
                let mut cam_changed = false;
                let mut ui_changed = false;
                // Phase 23: every binding goes through the (remappable) key map.
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                let keymap = crate::keymap::active();
                let card_active = crate::settings::card_hotkeys_enabled() && self.session.card_active();
                let Some(action) = keymap.lookup(code, ctrl, shift, st.modifiers.alt_key(), card_active) else {
                    return;
                };
                use crate::command_card::OrderKind as OK;
                use crate::keymap::Action as KA;
                match action {
                    KA::CancelOrMenu => {
                        if self.session.build_mode.is_some()
                            || self.session.current_order_mode().is_some()
                            || self.session.ui.template_mode.is_some()
                        {
                            self.session.build_mode = None;
                            self.session.clear_order_mode();
                            self.session.ui.build_drag = None;
                            self.session.status_banner = "Cancelled active mode".to_string();
                            ui_changed = true;
                        } else if self.session.ui.formation_drag.is_some() || self.session.ui.area_drag.is_some() {
                            self.session.ui.formation_drag = None;
                            self.session.ui.formation_preview.clear();
                            self.session.ui.area_drag = None;
                            self.session.ui.rmb_drag = Some(crate::controls::RmbDrag::Pan);
                            ui_changed = true;
                        } else {
                            self.session.menu = match &self.session.menu {
                                hud::MatchMenu::None => hud::MatchMenu::Pause,
                                hud::MatchMenu::LoadList { .. } => hud::MatchMenu::Pause,
                                hud::MatchMenu::Score { .. } => {
                                    self.session.end_shown = true;
                                    hud::MatchMenu::None
                                }
                                hud::MatchMenu::Pause => hud::MatchMenu::None,
                            };
                            ui_changed = true;
                        }
                    }
                    KA::ChatAll | KA::ChatAllies if !self.session.menu.is_open() => {
                        self.session.chat.open(action == KA::ChatAllies);
                        ui_changed = true;
                    }
                    KA::ChatAll | KA::ChatAllies => {}
                    KA::ScoreScreen => {
                        self.session.menu = if matches!(self.session.menu, hud::MatchMenu::Score { .. }) {
                            hud::MatchMenu::None
                        } else {
                            hud::MatchMenu::Score { series: crate::stats::Series::MassIncome }
                        };
                        ui_changed = true;
                    }
                    KA::QuickSave => {
                        self.quick_save();
                        if let Some(st) = self.state.as_mut() {
                            st.dirty_dynamic = true;
                            st.window.set_title(&self.session.window_title(&self.map.map_name));
                            st.window.request_redraw();
                        }
                        return;
                    }
                    KA::QuickLoad => {
                        let path = crate::savegame::data_dir("saves").join(crate::savegame::QUICKSAVE_FILE);
                        if path.is_file() {
                            self.load_save_file(&path);
                        } else {
                            let entries = crate::savegame::list_saves(Some(&self.map.map_id), 24);
                            self.session.menu = hud::MatchMenu::LoadList { entries, selected: 0 };
                        }
                        if let Some(st) = self.state.as_mut() {
                            st.dirty_dynamic = true;
                            st.window.set_title(&self.session.window_title(&self.map.map_name));
                            st.window.request_redraw();
                        }
                        return;
                    }
                    KA::Group(digit) => {
                        let slot = crate::controls::digit_slot(digit);
                        if st.modifiers.control_key() {
                            self.session.assign_control_group(slot, shift);
                        } else {
                            let now = self.session.ui.now_secs();
                            if self.session.recall_control_group(slot, shift, now) {
                                self.center_on_selection_or_commander();
                            }
                        }
                        let Some(st) = self.state.as_mut() else {
                            return;
                        };
                        st.dirty_dynamic = true;
                        st.window
                            .set_title(&self.session.window_title(&self.map.map_name));
                        st.window.request_redraw();
                        return;
                    }
                    KA::SelectAcu => {
                        self.session.select_commander();
                        ui_changed = true;
                    }
                    KA::SelectFactories => {
                        self.session.select_factories();
                        ui_changed = true;
                    }
                    KA::SelectArmy => {
                        self.session.select_army();
                        ui_changed = true;
                    }
                    KA::IdleEngineer => {
                        if self.session.cycle_idle_engineer().is_some() {
                            self.center_on_selection_or_commander();
                        }
                        let Some(st) = self.state.as_mut() else {
                            return;
                        };
                        st.dirty_dynamic = true;
                        st.window.request_redraw();
                        return;
                    }
                    KA::ReplayPerspective => {
                        self.session.cycle_replay_perspective();
                        ui_changed = true;
                    }
                    KA::ReplayBack | KA::ReplayForward | KA::ReplayStart if self.session.is_replay() => {
                        let cur = self.session.world.tick();
                        let jump = 30 * TICK_HZ;
                        let target = match action {
                            KA::ReplayBack => cur.saturating_sub(jump),
                            KA::ReplayForward => cur + jump,
                            _ => 0,
                        };
                        self.session.request_seek_tick(target);
                        ui_changed = true;
                    }
                    KA::ReplayBack | KA::ReplayForward | KA::ReplayStart => {}
                    KA::CenterSelection => {
                        self.center_on_selection_or_commander();
                        return;
                    }
                    KA::ToggleRepeat => {
                        self.session.trigger_hud_action(HudAction::ToggleRepeat, shift);
                        ui_changed = true;
                    }
                    KA::ResetCamera => {
                        let aspect = st.camera.aspect;
                        let (_, _, avg_h) = self.map.scmap.height_stats();
                        let fy = if self.map.scmap.water.has_water {
                            self.map.scmap.water.elevation
                        } else {
                            avg_h
                        };
                        st.camera = OrbitCamera::for_map(
                            self.map.scmap.map_width,
                            self.map.scmap.map_height,
                            fy,
                            aspect,
                        );
                        cam_changed = true;
                    }
                    KA::PanUp => {
                        st.camera.pan(0.0, 0.06);
                        cam_changed = true;
                    }
                    KA::PanDown => {
                        st.camera.pan(0.0, -0.06);
                        cam_changed = true;
                    }
                    KA::PanLeft => {
                        st.camera.pan(-0.06, 0.0);
                        cam_changed = true;
                    }
                    KA::PanRight => {
                        st.camera.pan(0.06, 0.0);
                        cam_changed = true;
                    }
                    KA::OrbitLeft => {
                        st.camera.orbit(-0.10, 0.0);
                        cam_changed = true;
                    }
                    KA::OrbitRight => {
                        st.camera.orbit(0.10, 0.0);
                        cam_changed = true;
                    }
                    KA::AttackMove => {
                        self.session.trigger_hud_action(HudAction::AttackMove, shift);
                        ui_changed = true;
                    }
                    KA::Stop => {
                        self.session.trigger_hud_action(HudAction::Stop, shift);
                        ui_changed = true;
                    }
                    KA::BuildFactory => {
                        self.session.trigger_hud_action(HudAction::BuildFactory, shift);
                        ui_changed = true;
                    }
                    KA::BuildExtractor => {
                        self.session.trigger_hud_action(HudAction::BuildExtractor, shift);
                        ui_changed = true;
                    }
                    KA::BuildGenerator => {
                        self.session.trigger_hud_action(HudAction::BuildGenerator, shift);
                        ui_changed = true;
                    }
                    KA::BuildPointDefense => {
                        self.session.trigger_hud_action(HudAction::BuildPointDefense, shift);
                        ui_changed = true;
                    }
                    KA::BuildRadar => {
                        self.session.trigger_hud_action(HudAction::BuildRadar, shift);
                        ui_changed = true;
                    }
                    KA::ProduceEngineer => {
                        self.session.trigger_hud_action(HudAction::ProduceEngineer, shift);
                        ui_changed = true;
                    }
                    KA::ProduceTank => {
                        self.session.trigger_hud_action(HudAction::ProduceTank, shift);
                        ui_changed = true;
                    }
                    KA::ProduceBot => {
                        self.session.trigger_hud_action(HudAction::ProduceBot, shift);
                        ui_changed = true;
                    }
                    KA::ProduceArtillery => {
                        self.session.trigger_hud_action(HudAction::ProduceArtillery, shift);
                        ui_changed = true;
                    }
                    KA::OverchargeNearest => {
                        self.session.trigger_overcharge_nearest();
                        ui_changed = true;
                    }
                    KA::UnloadMode => {
                        self.session.trigger_hud_action(HudAction::UnloadTransport, shift);
                        ui_changed = true;
                    }
                    KA::Enhance | KA::EnhanceCycle => {
                        self.session.trigger_hud_action(HudAction::EnhanceNext, action == KA::EnhanceCycle);
                        ui_changed = true;
                    }
                    KA::ToggleShield => {
                        self.session.trigger_hud_action(HudAction::ToggleShield, shift);
                        ui_changed = true;
                    }
                    KA::Pause => {
                        self.session.trigger_hud_action(HudAction::TogglePause, shift);
                        ui_changed = true;
                    }
                    KA::SpeedUp | KA::SpeedDown => {
                        self.session.change_speed(action == KA::SpeedUp);
                        ui_changed = true;
                    }
                    KA::ToggleAi => {
                        self.session.trigger_hud_action(HudAction::ToggleAi, false);
                        ui_changed = true;
                    }
                    KA::ToggleFog => {
                        self.session.trigger_hud_action(HudAction::ToggleFog, false);
                        ui_changed = true;
                    }
                    KA::ToggleProfiler => {
                        crate::profiler::toggle_overlay();
                        ui_changed = true;
                    }
                    KA::OrderMove
                    | KA::OrderPatrol
                    | KA::OrderAssist
                    | KA::OrderGuard
                    | KA::OrderRepair
                    | KA::OrderReclaim
                    | KA::OrderLoad
                    | KA::OrderOvercharge
                    | KA::OrderLaunchMissile => {
                        let kind = match action {
                            KA::OrderMove => OK::Move,
                            KA::OrderPatrol => OK::Patrol,
                            KA::OrderAssist => OK::Assist,
                            KA::OrderGuard => OK::Guard,
                            KA::OrderRepair => OK::Repair,
                            KA::OrderReclaim => OK::Reclaim,
                            KA::OrderLoad => OK::Load,
                            KA::OrderOvercharge => OK::Overcharge,
                            _ => OK::LaunchMissile,
                        };
                        if !self.session.is_replay() {
                            self.session.trigger_order(kind);
                        }
                        ui_changed = true;
                    }
                    KA::CardNextTab => {
                        self.session.cycle_card_tab();
                        ui_changed = true;
                    }
                    KA::HotbuildFactory | KA::HotbuildDefense | KA::HotbuildEconomy | KA::HotbuildIntel => {
                        use crate::roster::BuildGroup as G;
                        let g = match action {
                            KA::HotbuildFactory => G::Factory,
                            KA::HotbuildDefense => G::Defense,
                            KA::HotbuildEconomy => G::Economy,
                            _ => G::Intel,
                        };
                        if !self.session.is_replay() {
                            self.session.hotbuild(g);
                        }
                        ui_changed = true;
                    }
                    KA::TemplateSave => {
                        self.session.save_template_from_selection();
                        ui_changed = true;
                    }
                    KA::TemplateCycle => {
                        if !self.session.is_replay() {
                            self.session.cycle_template();
                        }
                        ui_changed = true;
                    }
                    KA::CardSlot(i) => {
                        self.session.trigger_card_item(i as usize, shift, false);
                        ui_changed = true;
                    }
                    KA::BookmarkSave(i) | KA::BookmarkRecall(i) => {
                        let save = matches!(action, KA::BookmarkSave(_));
                        if self.session.apply_bookmark(i as usize, save, &mut st.camera) {
                            cam_changed = true;
                        }
                        ui_changed = true;
                    }
                }
                if cam_changed {
                    st.resources.update_camera(&st.queue, &st.camera);
                    st.dirty_dynamic = true;
                    st.window.request_redraw();
                }
                if ui_changed {
                    st.dirty_dynamic = true;
                    st.window
                        .set_title(&self.session.window_title(&self.map.map_name));
                    st.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Phase 23: OS cursor follows the targeting mode.
                use winit::window::CursorIcon as CI;
                use crate::command_card::OrderKind as OK;
                let want = match (self.session.current_order_mode(), self.session.build_mode.is_some()) {
                    (Some(OK::Attack | OK::Overcharge | OK::LaunchMissile), _) => CI::Crosshair,
                    (Some(OK::Move | OK::Patrol | OK::Load | OK::Unload), _) => CI::Move,
                    (Some(OK::Assist | OK::Guard | OK::Repair), _) => CI::Copy,
                    (Some(OK::Reclaim), _) => CI::Grab,
                    (None, true) => CI::Cell,
                    _ => CI::Default,
                };
                if want != st.cursor_icon {
                    st.cursor_icon = want;
                    st.window.set_cursor(want);
                }
                st.device.poll(wgpu::Maintain::Poll);
                st.resources.poll_gpu_timer();
                // Phase 48: mission messages / objective changes → banner, camera requests.
                if let Some(rt) = self.session.world.mission() {
                    let mut banner = None;
                    if rt.log.len() > self.session.ui.mission_said {
                        banner = rt.log.last().map(|(_, t)| t.clone());
                    }
                    let states: Vec<crate::mission::ObjectiveState> =
                        rt.mission.objectives.iter().map(|o| rt.objectives[&o.id]).collect();
                    for (i, o) in rt.mission.objectives.iter().enumerate() {
                        if self.session.ui.mission_objectives.get(i).is_some_and(|s| *s != states[i]) {
                            banner = Some(format!("Objective {:?}: {}", states[i], o.text));
                        }
                    }
                    let cam = rt.camera;
                    self.session.ui.mission_said = rt.log.len();
                    self.session.ui.mission_objectives = states;
                    if let Some(b) = banner {
                        self.session.status_banner = b;
                    }
                    if let Some((t, at, zoom)) = cam
                        && self.session.ui.mission_camera != Some(t)
                    {
                        self.session.ui.mission_camera = Some(t);
                        let (wx, wz) = (at.0 as f32 / 1000.0, at.1 as f32 / 1000.0);
                        st.camera.target = [wx, sample_terrain_y(&self.map.scmap, wx, wz), wz];
                        if let Some(z) = zoom {
                            st.camera.distance = z as f32;
                        }
                        st.resources.update_camera(&st.queue, &st.camera);
                        st.dirty_dynamic = true;
                    }
                }
                // Phase 34: hide reclaimed trees / rocks (also restores them after a seek back).
                let gone = self.session.world.props_gone();
                if gone.len() != st.props_hidden && st.props_filtered_at.elapsed().as_secs_f32() >= 1.0 {
                    let batches = filter_reclaimed_props(&st.prop_cpu, gone);
                    st.resources.update_prop_batches(&st.device, &st.queue, &batches);
                    st.props_hidden = gone.len();
                    st.props_filtered_at = Instant::now();
                }
                if crate::profiler::overlay_visible() {
                    st.dirty_dynamic = true;
                }
                if st.dirty_dynamic {
                    upload_frame_resources(
                        &st.device,
                        &mut st.resources,
                        &st.templates,
                        &self.session,
                        &self.map,
                        &self.meshes,
                        &st.camera,
                        st.config.width,
                        st.config.height,
                    );
                    st.dirty_dynamic = false;
                }
                let frame = match st.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                        st.surface.configure(&st.device, &st.config);
                        return;
                    }
                    Err(_) => return,
                };
                let resolve_view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    st.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("play_win_encoder"),
                        });
                {
                    let _p = crate::profiler::scope("render.encode_submit");
                    st.resources.encode_draw(
                        &mut encoder,
                        &st.msaa_view,
                        Some(&resolve_view),
                        &st.depth_view,
                        st.clear_color,
                    );
                    st.queue.submit(Some(encoder.finish()));
                    st.resources.gpu_timer_after_submit();
                }
                {
                    let _p = crate::profiler::scope("render.present");
                    frame.present();
                }
                crate::profiler::end_frame(None);
                if crate::profiler::overlay_visible() {
                    st.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn reclaimed_props_are_filtered_out_of_their_batches() {
        let inst = |x: f32, z: f32| crate::viewer::GpuUnitInstance::from_basis([x, 0.0, z], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0; 3], [1.0; 4], [0.0; 4]);
        let batch = crate::scene::PropMeshBatch {
            label: "t".into(),
            vertices: vec![],
            indices: vec![],
            albedo: None,
            alpha_cutout: false,
            instances: vec![inst(10.5, 20.25), inst(30.0, 40.0)],
            bounds_min: [0.0; 3],
            bounds_max: [1.0; 3],
        };
        let out = super::filter_reclaimed_props(&[batch.clone()], &[(10.5f32.to_bits(), 20.25f32.to_bits())]);
        assert_eq!(out[0].instances.len(), 1);
        assert_eq!(out[0].instances[0].row0[3], 30.0);
        assert_eq!(super::filter_reclaimed_props(&[batch], &[])[0].instances.len(), 2);
    }

    use super::*;
    use crate::scmap::{ScMap, WaterInfo};
    use crate::skirmish::tests::test_world;

    fn test_scmap() -> ScMap {
        ScMap {
            version_major: 2,
            version_minor: 60,
            map_width: 256.0,
            map_height: 256.0,
            preview_dds: vec![],
            heightmap_width: 16,
            heightmap_height: 16,
            height_scale: 0.01,
            heightmap: vec![2000; 17 * 17],
            terrain_shader: String::new(),
            background_path: String::new(),
            sky_cubemap_path: String::new(),
            cubemaps: vec![],
            lighting: crate::scmap::ScMapLighting::default(),
            water: WaterInfo {
                has_water: true,
                elevation: 10.0,
                elevation_deep: 8.0,
                elevation_abyss: 2.0,
            },
            unknown1: 0, unknown2: 0, unknown3: 0, unknown4: 0, post_heightmap_byte: 0,
            shader_raw: Vec::new(), background_raw: Vec::new(), sky_raw: Vec::new(), cubemaps_raw: Vec::new(), tail_bytes: Vec::new(),
            terrain_textures: None,
        }
    }

    #[test]
    fn selection_and_box_prefer_mobiles_over_structures() {
        let mut w = test_world(101);
        let acu = w.commander_of(0).unwrap().id;
        w.issue(0, Command::Build { builder: acu, unit_type: "RF".into(), x: 64 * SCALE, y: 50 * SCALE, queue: false });
        w.run(350);
        let fac = w.units().find(|u| u.army == 0 && u.role == Role::Factory).unwrap().id;
        w.issue(0, Command::Produce { factory: fac, unit_type: "RT".into(), count: 1 });
        w.run(250);

        let mut sess = PlaySession::new(w, 0, false, 1);
        assert_eq!(sess.selected, vec![acu]);

        sess.select_factories();
        assert_eq!(sess.selected, vec![fac]);

        sess.select_box((0, 0), (200 * SCALE, 200 * SCALE), false);
        assert!(sess.selected.contains(&acu));
        assert!(!sess.selected.contains(&fac));

        sess.select_army();
        assert!(!sess.selected.contains(&acu));
        assert!(!sess.selected.is_empty());
    }

    #[test]
    fn right_click_orders_move_attack_rally_and_stop() {
        let w = test_world(102);
        let acu0 = w.commander_of(0).unwrap().id;
        let acu1 = w.commander_of(1).unwrap().id;
        let p1 = w.unit(acu1).map(|u| (u.pos.x, u.pos.y)).unwrap();

        let mut sess = PlaySession::new(w, 0, false, 1);
        sess.issue_right_click((80 * SCALE, 80 * SCALE), false, 3 * SCALE);
        sess.step_sim();
        assert!(!sess.world.unit(acu0).unwrap().orders.is_empty());

        sess.issue_right_click(p1, false, 3 * SCALE);
        sess.step_sim();
        assert_eq!(
            sess.world.unit(acu0).unwrap().orders.front(),
            Some(&UnitOrder::Attack(acu1))
        );

        sess.trigger_stop();
        sess.step_sim();
        assert!(sess.world.unit(acu0).unwrap().orders.is_empty());
    }

    #[test]
    fn build_mode_extractor_snaps_and_queues_and_builds_render_frame() {
        let w = test_world(103);
        let scmap = test_scmap();
        let cache = UnitMeshCache::load(None, w.catalog());
        let mut sess = PlaySession::new(w, 0, false, 1);

        sess.trigger_build_mode(Role::Extractor);
        assert!(sess.issue_left_click_action((41 * SCALE, 39 * SCALE), true, 3 * SCALE));
        assert!(sess.build_mode.is_some());
        sess.trigger_build_mode(Role::Factory);
        assert!(sess.issue_left_click_action((72 * SCALE, 52 * SCALE), true, 3 * SCALE));
        sess.build_mode = None;

        for _ in 0..400 {
            sess.step_sim();
        }
        assert!(sess.world.units().any(|u| u.army == 0 && u.role == Role::Extractor));
        assert!(sess.world.units().any(|u| u.army == 0 && u.role == Role::Factory));

        sess.select_factories();
        sess.issue_right_click((120 * SCALE, 120 * SCALE), false, 3 * SCALE);
        sess.trigger_produce(0, 1);
        sess.step_sim();

        let cam = initial_army_camera(&sess.world, &scmap, 0, 1.6);
        let (batches, v, i) = build_skirmish_render_frame(&sess, &scmap, &cache, &cam);
        assert!(!batches.is_empty());
        assert!(!v.is_empty() && !i.is_empty());
        assert!(v.iter().all(|vx| vx.position.iter().all(|c| c.is_finite())));
        assert!(i.iter().all(|&ix| (ix as usize) < v.len()));
    }
}
