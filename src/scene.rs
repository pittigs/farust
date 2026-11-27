//! Scene builders turning Forged Alliance unit meshes/textures and `.scmap` terrains
//! into GPU-ready vertex/index/texture buffers.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use bytemuck::{Pod, Zeroable};

use crate::blueprint::{self, Value};
use crate::camera::OrbitCamera;
use crate::dds_decode::{
    RgbaImage, SpecGlowMap, decode_dds_cubemap_rgba8, decode_dds_rgba8, resize_rgba_bilinear,
};
use crate::jobs;
use crate::scd;
use crate::scenario;
use crate::scm::ScmMesh;
use crate::scmap::{ScMap, ScMapDecal, ScMapLighting, ScMapProp};
use crate::viewer::GpuUnitInstance;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// rgb = vertex color multiplier, a = specular intensity (0.0..=1.0).
    pub color: [f32; 4],
}

/// Multi-stratum terrain textures & blend maps for per-pixel GPU terrain splatting.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TerrainSplatData {
    pub tm0: RgbaImage,
    pub tm1: RgbaImage,
    pub strata_albedo: Vec<RgbaImage>,
    pub strata_normal: Vec<RgbaImage>,
    pub albedo_scales: [f32; 10],
    pub albedo_active: [u32; 10],
    pub normal_scales: [f32; 9],
    pub normal_active: [u32; 9],
    pub lighting: ScMapLighting,
    pub map_size: [f32; 2],
    pub water_elevation: f32,
    pub water_deep: f32,
    pub water_abyss: f32,
}

/// Depth-encoded water surface mesh, wave normals, and water ramp for GPU water shading.
#[derive(Debug, Clone)]
pub struct WaterGpuData {
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    pub waves_normal: RgbaImage,
    pub water_ramp: RgbaImage,
    pub fresnel_bias: f32,
    pub fresnel_power: f32,
    pub surface_color: [f32; 3],
}

/// Instanced environment prop batch (pine trees, boulders, logs) parsed from `.scmap`,
/// partitioned into an `8x8` world XZ grid with conservative world AABBs for frustum culling.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PropMeshBatch {
    pub label: String,
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    pub albedo: Option<RgbaImage>,
    pub alpha_cutout: bool,
    pub instances: Vec<GpuUnitInstance>,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

/// Terrain-draped decal geometry grouped by albedo texture path from `.scmap`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DecalBatch {
    pub label: String,
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    pub albedo: Option<RgbaImage>,
}

/// Packed `32x32` strategic icon texture atlas (`16` columns) with `:rest` and `:sel` UV rects.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StrategicIconAtlas {
    pub image: RgbaImage,
    pub uv_rects: BTreeMap<String, [f32; 4]>,
    pub default_uv: [f32; 4],
}

impl StrategicIconAtlas {
    #[allow(dead_code)]
    pub fn uv(&self, name: &str, selected: bool) -> [f32; 4] {
        let base = normalize_icon_base_name(name);
        let suffix = if selected { "sel" } else { "rest" };
        let key = format!("{base}:{suffix}");
        if let Some(&rect) = self.uv_rects.get(&key) {
            return rect;
        }
        if selected {
            let rest_key = format!("{base}:rest");
            if let Some(&rect) = self.uv_rects.get(&rest_key) {
                return rect;
            }
        }
        self.default_uv
    }
}

/// Faction-themed HUD color palette.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub struct FactionTheme {
    pub name: &'static str,
    pub primary: [f32; 4],
    pub border: [f32; 4],
    pub accent: [f32; 4],
    pub panel_bg: [f32; 4],
}

/// Returns the HUD color theme for `"uef"`, `"cybran"`, `"aeon"`, `"seraphim"`,
/// falling back to a neutral steel theme for unrecognized faction keys.
#[allow(dead_code)]
pub fn faction_theme(faction_key: &str) -> FactionTheme {
    match faction_key.trim().to_ascii_lowercase().as_str() {
        "uef" => FactionTheme {
            name: "UEF",
            primary: [0.18, 0.52, 0.94, 1.0],
            border: [0.34, 0.66, 0.98, 0.95],
            accent: [0.52, 0.88, 1.00, 1.0],
            panel_bg: [0.05, 0.08, 0.14, 0.86],
        },
        "cybran" => FactionTheme {
            name: "Cybran",
            primary: [0.88, 0.20, 0.18, 1.0],
            border: [0.95, 0.36, 0.28, 0.95],
            accent: [1.00, 0.56, 0.24, 1.0],
            panel_bg: [0.12, 0.05, 0.06, 0.86],
        },
        "aeon" => FactionTheme {
            name: "Aeon",
            primary: [0.18, 0.78, 0.44, 1.0],
            border: [0.32, 0.90, 0.58, 0.95],
            accent: [0.90, 0.86, 0.38, 1.0],
            panel_bg: [0.05, 0.11, 0.08, 0.86],
        },
        "seraphim" => FactionTheme {
            name: "Seraphim",
            primary: [0.92, 0.74, 0.20, 1.0],
            border: [0.98, 0.84, 0.36, 0.95],
            accent: [0.45, 0.88, 0.96, 1.0],
            panel_bg: [0.12, 0.09, 0.04, 0.86],
        },
        _ => FactionTheme {
            name: "Neutral",
            primary: [0.45, 0.56, 0.70, 1.0],
            border: [0.58, 0.68, 0.82, 0.95],
            accent: [0.82, 0.88, 0.96, 1.0],
            panel_bg: [0.07, 0.09, 0.12, 0.86],
        },
    }
}

#[derive(Debug, Clone)]
pub struct LoadedScene {
    pub title: String,
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    /// Optional secondary geometry (water plane, map markers) drawn untextured
    /// so baked terrain textures stay clean.
    pub overlay_vertices: Vec<GpuVertex>,
    pub overlay_indices: Vec<u32>,
    pub albedo: RgbaImage,
    /// Optional linear `(specular, glow)` map extracted from `*_SpecTeam.dds` (Phase 42).
    pub spec_glow: Option<SpecGlowMap>,
    pub initial_camera: OrbitCamera,
    pub clear_color: [f64; 4],
    pub terrain_splat: Option<TerrainSplatData>,
    pub water: Option<WaterGpuData>,
    pub sky_cube: Option<[RgbaImage; 6]>,
    pub prop_batches: Vec<PropMeshBatch>,
    pub decal_batches: Vec<DecalBatch>,
}

#[allow(dead_code)]
pub fn load_unit_scene(game_dir: &Path, unit_id: &str, aspect: f32) -> Result<LoadedScene, String> {
    load_unit_scene_posed(game_dir, unit_id, aspect, None, 0.0)
}

/// Per-weapon render metadata (Phase 17): aim/recoil bones and the projectile blueprint.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WeaponDisplayInfo {
    pub label: String,
    pub yaw_bone: Option<String>,
    pub pitch_bone: Option<String>,
    /// `RackBones[].RackBone` (slid back by `recoil_distance` on fire).
    pub rack_bones: Vec<String>,
    /// `RackBones[].MuzzleBones[]` plus `TurretBoneMuzzle`.
    pub muzzle_bones: Vec<String>,
    /// `RackRecoilDistance` in raw mesh units along the rack bone's local Z (negative = backwards).
    pub recoil_distance: f32,
    /// `TurretPitchRange` in degrees (`90` when absent).
    pub pitch_range_deg: f32,
    /// `TurretYawSpeed` / `TurretPitchSpeed` in degrees per second.
    pub yaw_speed_deg: f32,
    pub pitch_speed_deg: f32,
    /// Normalized lower-case `ProjectileId` path (`projectiles/x/x_proj.bp`).
    pub projectile_id: Option<String>,
    /// `DamageRadius > 0` (area weapons fire ballistic shells in the sim).
    pub splash: bool,
}

/// One `Display.AnimationDeath` entry: clip path and mid playback rate.
#[derive(Debug, Clone, PartialEq)]
pub struct DeathAnimInfo {
    pub path: String,
    pub rate: f32,
}

/// Blueprint `Display` fields the in-game renderer needs for skeletal animation and LOD switching.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitDisplayInfo {
    pub walk_anim: Option<String>,
    pub idle_anim: Option<String>,
    pub walk_rate: f32,
    pub max_speed: f32,
    pub lod0_cutoff: f32,
    /// `Display.AnimationDeath[]` clips (played before the wreck appears).
    pub death_anims: Vec<DeathAnimInfo>,
    /// Looping `Display.AnimationBuild` (factory arms while producing).
    pub build_anim: Option<String>,
    /// One-shot `Display.AnimationOpen` (held open while working, reversed when idle).
    pub open_anim: Option<String>,
    pub weapons: Vec<WeaponDisplayInfo>,
    /// `Defense.Shield.PersonalShield`.
    pub personal_shield: bool,
    /// `Air.BankFactor` (`0` for non-air units), `Air.Winged`, `Air.MaxAirspeed`.
    pub bank_factor: f32,
    pub winged: bool,
    pub max_airspeed: f32,
}

fn norm_asset_path(s: &str) -> Option<String> {
    let p = s.trim().replace('\\', "/").trim_start_matches('/').to_string();
    (!p.is_empty()).then_some(p)
}

/// Extracts the weapon aim/recoil bones and projectile ids from `Weapon = { ... }`.
pub fn parse_weapon_display_info(bp: &Value) -> Vec<WeaponDisplayInfo> {
    let Some(list) = bp.get("Weapon") else {
        return Vec::new();
    };
    list.table_list()
        .into_iter()
        .map(|w| {
            let s = |k: &str| w.get_str(k).map(str::trim).filter(|v| !v.is_empty()).map(str::to_owned);
            let f = |k: &str, d: f64| w.get_f64(k).filter(|v| v.is_finite()).unwrap_or(d) as f32;
            let mut rack_bones = Vec::new();
            let mut muzzle_bones = Vec::new();
            if let Some(racks) = w.get("RackBones") {
                for r in racks.table_list() {
                    if let Some(rb) = r.get_str("RackBone").map(str::trim).filter(|v| !v.is_empty()) {
                        rack_bones.push(rb.to_string());
                    }
                    if let Some(m) = r.get("MuzzleBones") {
                        muzzle_bones.extend(m.string_list());
                    }
                }
            }
            if let Some(m) = s("TurretBoneMuzzle")
                && !muzzle_bones.iter().any(|b| b.eq_ignore_ascii_case(&m))
            {
                muzzle_bones.push(m);
            }
            WeaponDisplayInfo {
                label: s("Label").unwrap_or_default(),
                yaw_bone: s("TurretBoneYaw"),
                pitch_bone: s("TurretBonePitch"),
                rack_bones,
                muzzle_bones,
                recoil_distance: f("RackRecoilDistance", 0.0),
                pitch_range_deg: f("TurretPitchRange", 90.0).abs(),
                yaw_speed_deg: f("TurretYawSpeed", 90.0).abs(),
                pitch_speed_deg: f("TurretPitchSpeed", 60.0).abs(),
                projectile_id: s("ProjectileId").and_then(|p| norm_asset_path(&p)).map(|p| p.to_ascii_lowercase()),
                splash: f("DamageRadius", 0.0) > 0.0,
            }
        })
        .collect()
}

/// Wires every weapon's yaw/pitch/rack bones (and `RackRecoilDistance`) into `rig`.
pub fn apply_weapon_bones(rig: &mut crate::sca::UnitRig, weapons: &[WeaponDisplayInfo], span_raw: f32) {
    let yaw: Vec<&str> = weapons.iter().filter_map(|w| w.yaw_bone.as_deref()).collect();
    let pitch: Vec<&str> = weapons.iter().filter_map(|w| w.pitch_bone.as_deref()).collect();
    let recoil: Vec<(&str, f32)> = weapons
        .iter()
        .flat_map(|w| w.rack_bones.iter().map(move |b| (b.as_str(), w.recoil_distance)))
        .collect();
    rig.set_weapon_bones(&yaw, &pitch, &recoil, span_raw);
}

/// Extracts `Display.AnimationWalk/AnimationIdle/AnimationWalkRate`, `Physics.MaxSpeed` and
/// the first `Display.Mesh.LODs[].LODCutoff` from a parsed unit blueprint.
pub fn parse_unit_display_info(bp: &Value) -> UnitDisplayInfo {
    let display = bp.get("Display");
    let anim_str = |key: &str| display.and_then(|d| d.get_str(key)).and_then(norm_asset_path);
    let death_anims = display
        .and_then(|d| d.get("AnimationDeath"))
        .map(|list| {
            list.table_list()
                .into_iter()
                .filter_map(|e| {
                    let path = e.get_str("Animation").and_then(norm_asset_path)?;
                    let lo = e.get_f64("AnimationRateMin").filter(|v| v.is_finite() && *v > 0.0).unwrap_or(1.0);
                    let hi = e.get_f64("AnimationRateMax").filter(|v| v.is_finite() && *v > 0.0).unwrap_or(lo);
                    Some(DeathAnimInfo { path, rate: ((lo + hi) * 0.5).clamp(0.1, 10.0) as f32 })
                })
                .collect()
        })
        .unwrap_or_default();
    let air = bp.get("Air");
    let air_f = |k: &str| air.and_then(|a| a.get_f64(k)).filter(|v| v.is_finite()).unwrap_or(0.0) as f32;
    let personal_shield = bp
        .get("Defense")
        .and_then(|d| d.get("Shield"))
        .and_then(|s| s.get("PersonalShield"))
        .is_some_and(|v| matches!(v, Value::Bool(true)));
    let walk_rate = display
        .and_then(|d| d.get_f64("AnimationWalkRate"))
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.0) as f32;
    let max_speed = bp
        .get("Physics")
        .and_then(|p| p.get_f64("MaxSpeed"))
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(1.0) as f32;
    let lod0_cutoff = display
        .and_then(|d| d.get("Mesh"))
        .and_then(|m| m.get("LODs"))
        .and_then(|l| l.table_list().into_iter().next())
        .and_then(|l| l.get_f64("LODCutoff"))
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(150.0) as f32;
    UnitDisplayInfo {
        walk_anim: anim_str("AnimationWalk"),
        idle_anim: anim_str("AnimationIdle"),
        walk_rate,
        max_speed,
        lod0_cutoff,
        death_anims,
        build_anim: anim_str("AnimationBuild"),
        open_anim: anim_str("AnimationOpen"),
        weapons: parse_weapon_display_info(bp),
        personal_shield,
        bank_factor: air_f("BankFactor").abs(),
        winged: air.and_then(|a| a.get("Winged")).is_some_and(|v| matches!(v, Value::Bool(true))),
        max_airspeed: air_f("MaxAirspeed").max(0.0),
    }
}

/// Render-side projectile mesh info from a `*_proj.bp` (`Display.UniformScale`, `MISSILE` category).
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectileDisplayInfo {
    pub uniform_scale: f32,
    pub missile: bool,
}

/// Parses a `ProjectileBlueprint { ... }` file (same pure-data grammar as unit blueprints).
pub fn parse_projectile_blueprint(text: &str) -> Result<Value, String> {
    blueprint::parse_blueprint(&text.replacen("ProjectileBlueprint", "UnitBlueprint", 1))
}

pub fn parse_projectile_display_info(bp: &Value) -> ProjectileDisplayInfo {
    let uniform_scale = bp
        .get("Display")
        .and_then(|d| d.get_f64("UniformScale"))
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.0) as f32;
    let missile = bp
        .get("Categories")
        .map(Value::string_list)
        .unwrap_or_default()
        .iter()
        .any(|c| c.eq_ignore_ascii_case("MISSILE"));
    ProjectileDisplayInfo { uniform_scale, missile }
}

/// Like [`load_unit_scene`] but CPU-poses the mesh with a skeletal clip: `anim` is `walk`, `idle`,
/// or a substring of a `.sca` file name in the unit's folder; `anim_time` is in clip seconds.
pub fn load_unit_scene_posed(
    game_dir: &Path,
    unit_id: &str,
    aspect: f32,
    anim: Option<&str>,
    anim_time: f32,
) -> Result<LoadedScene, String> {
    let unit_id = unit_id.trim();
    if unit_id.is_empty() {
        return Err("Unit ID must not be empty".to_string());
    }

    let gamedata = scd::gamedata_dir(game_dir);
    let units_scd = gamedata.join("units.scd");

    let direct_bp = format!("units/{unit_id}/{unit_id}_unit.bp");
    let (actual_bp_path, bp_bytes) = match scd::read_scd_file_ci(&units_scd, &direct_bp) {
        Ok(found) => found,
        Err(_) => {
            let entries = scd::list_scd(&units_scd)
                .map_err(|e| format!("Failed to open {}: {e}", units_scd.display()))?;
            let suffix = format!("/{unit_id}_unit.bp");
            let root_name = format!("{unit_id}_unit.bp");
            let matched = entries
                .iter()
                .find(|entry| {
                    if entry.is_dir {
                        return false;
                    }
                    let norm = entry.name.replace('\\', "/");
                    ends_with_ci(&norm, &suffix) || norm.eq_ignore_ascii_case(&root_name)
                })
                .ok_or_else(|| {
                    format!(
                        "Unit blueprint '{unit_id}_unit.bp' not found in {}",
                        units_scd.display()
                    )
                })?;
            scd::read_scd_file_ci(&units_scd, &matched.name)
                .map_err(|e| format!("Failed to read blueprint '{}': {e}", matched.name))?
        }
    };

    let bp_text = String::from_utf8_lossy(&bp_bytes);
    let bp = blueprint::parse_blueprint(&bp_text)?;

    let display = bp.get("Display");
    let raw_scale = display
        .and_then(|d| d.get_f64("UniformScale"))
        .unwrap_or(1.0) as f32;
    let scale = if raw_scale.is_finite() && raw_scale > 0.0 {
        raw_scale
    } else {
        1.0
    };

    let desc = bp
        .get_str("Description")
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("Unit");
    let categories = bp
        .get("Categories")
        .map(Value::string_list)
        .unwrap_or_default();

    let lod0 = display
        .and_then(|d| d.get("Mesh"))
        .and_then(|m| m.get("LODs"))
        .and_then(|lods| lods.table_list().into_iter().next());
    let mesh_name = lod0.and_then(|l| l.get_str("MeshName"));
    let albedo_name = lod0.and_then(|l| l.get_str("AlbedoName"));
    let secondary_name = lod0.and_then(|l| l.get_str("SecondaryName"));

    let norm_bp_path = actual_bp_path.replace('\\', "/");
    let norm_bp_trimmed = norm_bp_path.trim_start_matches('/');
    let base = strip_suffix_ci(norm_bp_trimmed, "_unit.bp").unwrap_or(norm_bp_trimmed);
    let bp_dir = base.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");

    let mut scm_candidates = Vec::with_capacity(2);
    if let Some(name) = mesh_name {
        let resolved = resolve_bp_asset_path(bp_dir, name);
        if !resolved.is_empty() {
            scm_candidates.push(resolved);
        }
    }
    let default_scm = format!("{base}_lod0.scm");
    if !scm_candidates
        .iter()
        .any(|c| c.eq_ignore_ascii_case(&default_scm))
    {
        scm_candidates.push(default_scm);
    }

    let mesh_archives = ["units.scd", "meshes.scd", "props.scd"];
    let scm_bytes = find_asset_in_scds(&gamedata, &mesh_archives, &scm_candidates).ok_or_else(
        || {
            format!(
                "Mesh not found for unit '{unit_id}' (tried {:?} in {:?})",
                scm_candidates, mesh_archives
            )
        },
    )?;
    let mesh = ScmMesh::parse(&scm_bytes)?;

    let mut albedo_candidates = Vec::with_capacity(2);
    if let Some(name) = albedo_name {
        let resolved = resolve_bp_asset_path(bp_dir, name);
        if !resolved.is_empty() {
            albedo_candidates.push(resolved);
        }
    }
    let default_albedo = format!("{base}_Albedo.dds");
    if !albedo_candidates
        .iter()
        .any(|c| c.eq_ignore_ascii_case(&default_albedo))
    {
        albedo_candidates.push(default_albedo);
    }

    let tex_archives = ["units.scd", "textures.scd", "env.scd"];
    let mut albedo = decode_first_dds(&gamedata, &tex_archives, &albedo_candidates).unwrap_or_else(
        || RgbaImage {
            width: 1,
            height: 1,
            pixels: vec![180, 185, 190, 64],
        },
    );

    let mut spec_candidates = Vec::with_capacity(2);
    if let Some(name) = secondary_name {
        let resolved = resolve_bp_asset_path(bp_dir, name);
        if !resolved.is_empty() {
            spec_candidates.push(resolved);
        }
    }
    let default_spec = format!("{base}_SpecTeam.dds");
    if !spec_candidates
        .iter()
        .any(|c| c.eq_ignore_ascii_case(&default_spec))
    {
        spec_candidates.push(default_spec);
    }

    let spec_team = decode_first_dds(&gamedata, &["units.scd"], &spec_candidates);
    let spec_glow = spec_team.as_ref().map(SpecGlowMap::from_spec_team);
    let mut spec_applied = false;
    if let Some(spec_img) = spec_team
        && spec_img.width == albedo.width
        && spec_img.height == albedo.height
    {
        let team_rgb = pick_team_color(&categories, unit_id);
        let pixel_count = (albedo.width as usize * albedo.height as usize)
            .min(albedo.pixels.len() / 4)
            .min(spec_img.pixels.len() / 4);
        for i in 0..pixel_count {
            let mask = (spec_img.pixels[i * 4 + 3] as f32 / 255.0) * 0.85;
            for c in 0..3 {
                let base_c = albedo.pixels[i * 4 + c] as f32;
                let team_c = team_rgb[c] as f32;
                let blended = base_c + (team_c - base_c) * mask;
                albedo.pixels[i * 4 + c] = blended.round().clamp(0.0, 255.0) as u8;
            }
            // FA SpecTeam.dds G channel = Phong specular intensity.
            let spec_strength =
                ((spec_img.pixels[i * 4 + 1] as u16 * 180 / 255) + 40).min(255) as u8;
            albedo.pixels[i * 4 + 3] = spec_strength;
        }
        spec_applied = true;
    }

    if !spec_applied {
        let pixel_count = albedo.pixels.len() / 4;
        for i in 0..pixel_count {
            albedo.pixels[i * 4 + 3] = 64;
        }
    }

    let mut title_suffix = String::new();
    let pose = match anim {
        Some(sel) => {
            let info = parse_unit_display_info(&bp);
            let names: Vec<&str> = mesh.bones.iter().map(|b| b.name.as_str()).collect();
            // `+`-separated tokens: `aim:YAW:PITCH[:RECOIL]` (degrees, degrees, 0..1) poses the weapons;
            // anything else picks the clip (`walk`, `idle`, `death`, `open`, `build`, `.sca` substring).
            let mut aim = crate::sca::AimPose::default();
            let mut clip_sel = None;
            for tok in sel.split('+').map(str::trim).filter(|t| !t.is_empty()) {
                if let Some(rest) = tok.strip_prefix("aim") {
                    let v: Vec<f32> = rest
                        .split(':')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.trim().parse::<f32>().map_err(|_| format!("invalid aim value '{s}' in '{tok}'")))
                        .collect::<Result<_, _>>()?;
                    aim.yaw = v.first().copied().unwrap_or(0.0).to_radians();
                    aim.pitch = v.get(1).copied().unwrap_or(0.0).to_radians();
                    aim.recoil = v.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                    title_suffix.push_str(&format!(
                        " [aim yaw {:.0} pitch {:.0} recoil {:.2}]",
                        aim.yaw.to_degrees(),
                        aim.pitch.to_degrees(),
                        aim.recoil
                    ));
                } else {
                    clip_sel = Some(tok);
                }
            }
            let mut rig = crate::sca::UnitRig::new(mesh.bones.clone(), [0.0; 3], scale);
            let span_raw = mesh
                .bounds
                .map(|b| (b.max[0] - b.min[0]).abs().max((b.max[2] - b.min[2]).abs()))
                .unwrap_or(1.0);
            apply_weapon_bones(&mut rig, &info.weapons, span_raw);
            let (clip, t, looping) = match clip_sel {
                Some(c) => {
                    let (path, clip) = resolve_unit_clip(&gamedata, bp_dir, unit_id, &info, c)?;
                    let clip = crate::sca::AnimClip::new(clip, &names)
                        .ok_or_else(|| format!("animation '{path}' shares no bones with the unit skeleton"))?;
                    let one_shot = matches!(c.to_ascii_lowercase().as_str(), "death" | "open")
                        || path.to_ascii_lowercase().contains("death");
                    let t = if one_shot {
                        anim_time.clamp(0.0, clip.duration())
                    } else {
                        crate::sca::wrap_anim_time(anim_time, clip.duration())
                    };
                    title_suffix = format!(" [{path} @ {t:.2}s]{title_suffix}");
                    (Some(clip), t, !one_shot)
                }
                None => (None, 0.0, true),
            };
            let local = rig.sample_pose(clip.as_ref(), t, looping);
            Some(rig.palette_from_pose(&local, aim))
        }
        None => None,
    };

    let vertices: Vec<GpuVertex> = mesh
        .vertices
        .iter()
        .map(|v| {
            let p = [
                v.position[0] * scale,
                v.position[1] * scale,
                v.position[2] * scale,
            ];
            let (position, normal) = match pose
                .as_ref()
                .and_then(|pal| pal.get(usize::from(v.bone_indices[0])))
            {
                Some(m) => (
                    crate::sca::skin_point(m, p),
                    normalize_or_up(crate::sca::skin_dir(m, v.normal)),
                ),
                None => (p, normalize_or_up(v.normal)),
            };
            GpuVertex {
                position,
                normal,
                uv: v.uv0,
                color: [1.0, 1.0, 1.0, 1.0],
            }
        })
        .collect();

    let indices: Vec<u32> = mesh.indices.iter().map(|&idx| idx as u32).collect();

    let (bounds_min, bounds_max) = if let Some(b) = &mesh.bounds {
        (
            [b.min[0] * scale, b.min[1] * scale, b.min[2] * scale],
            [b.max[0] * scale, b.max[1] * scale, b.max[2] * scale],
        )
    } else if !vertices.is_empty() {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in &vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(v.position[axis]);
                max[axis] = max[axis].max(v.position[axis]);
            }
        }
        (min, max)
    } else {
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    };

    let initial_camera = OrbitCamera::for_bounds(bounds_min, bounds_max, aspect);
    let title = format!("{} — {}{title_suffix}", unit_id.to_uppercase(), desc);

    Ok(LoadedScene {
        title,
        vertices,
        indices,
        overlay_vertices: Vec::new(),
        overlay_indices: Vec::new(),
        albedo,
        spec_glow,
        initial_camera,
        clear_color: [0.055, 0.070, 0.095, 1.0],
        terrain_splat: None,
        water: None,
        sky_cube: None,
        prop_batches: Vec::new(),
        decal_batches: Vec::new(),
    })
}

pub fn load_map_scene(game_dir: &Path, map_name: &str, aspect: f32) -> Result<LoadedScene, String> {
    let map_files = scenario::find_map(game_dir, map_name)?;
    let meta = scenario::parse_scenario_file(&map_files.scenario_path, &map_files.id)?;
    let markers = map_files
        .save_path
        .as_deref()
        .and_then(|save_path| scenario::parse_save_markers_file(save_path, &meta.armies).ok());

    let scmap_path = map_files
        .scmap_path
        .as_deref()
        .ok_or_else(|| format!("Map '{}' has no .scmap file", map_files.id))?;
    let scmap = ScMap::parse_file(scmap_path)?;

    let title = if meta.name.trim().is_empty() {
        meta.id.clone()
    } else {
        format!("{} ({})", meta.name, meta.id)
    };
    Ok(build_map_scene_from_parts(
        game_dir,
        title,
        &scmap,
        markers.as_ref(),
        aspect,
    ))
}

/// Builds a [`LoadedScene`] from an already-parsed [`ScMap`] and optional [`scenario::SaveMarkers`].
pub fn build_map_scene_from_parts(
    game_dir: &Path,
    title: String,
    scmap: &ScMap,
    markers: Option<&scenario::SaveMarkers>,
    aspect: f32,
) -> LoadedScene {
    let terrain_splat = load_terrain_splat_data(game_dir, scmap);
    let water = load_water_gpu_data(game_dir, scmap);
    let sky_cube = load_sky_cubemap(game_dir, scmap);
    let prop_batches = load_map_props(game_dir, scmap);
    let decal_batches = load_map_decals(game_dir, scmap);

    let baked = if terrain_splat.is_some() {
        None
    } else {
        bake_terrain_albedo(game_dir, scmap, 1024)
    };
    let textured = terrain_splat.is_some() || baked.is_some();
    let (vertices, indices, overlay_vertices, overlay_indices) =
        build_terrain_and_overlay_meshes_opts(scmap, markers, textured, water.is_none());

    let albedo = baked.unwrap_or_else(|| RgbaImage {
        width: 1,
        height: 1,
        pixels: vec![255, 255, 255, 255],
    });

    let (_, _, avg_h) = scmap.height_stats();
    let focus_y = if scmap.water.has_water {
        scmap.water.elevation
    } else {
        avg_h
    };
    let initial_camera = OrbitCamera::for_map(scmap.map_width, scmap.map_height, focus_y, aspect);
    let fc = scmap.lighting.fog_color;

    LoadedScene {
        title,
        vertices,
        indices,
        overlay_vertices,
        overlay_indices,
        albedo,
        spec_glow: None,
        initial_camera,
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
    }
}

#[allow(dead_code)]
pub fn build_terrain_mesh(
    scmap: &ScMap,
    markers: Option<&scenario::SaveMarkers>,
) -> (Vec<GpuVertex>, Vec<u32>) {
    let (mut verts, mut idx, ov_v, ov_i) = build_terrain_and_overlay_meshes(scmap, markers, false);
    let base = verts.len() as u32;
    verts.extend(ov_v);
    idx.extend(ov_i.into_iter().map(|i| base + i));
    (verts, idx)
}

/// Builds `(terrain_vertices, terrain_indices, overlay_vertices, overlay_indices)` where
/// water and beacons are placed in the overlay buffer so baked terrain strata textures
/// do not color the water surface.
pub fn build_terrain_and_overlay_meshes(
    scmap: &ScMap,
    markers: Option<&scenario::SaveMarkers>,
    neutral_terrain_tint: bool,
) -> (Vec<GpuVertex>, Vec<u32>, Vec<GpuVertex>, Vec<u32>) {
    build_terrain_and_overlay_meshes_opts(scmap, markers, neutral_terrain_tint, true)
}

/// Variant allowing callers using [`WaterGpuData`] to omit the legacy flat opaque water quad.
pub fn build_terrain_and_overlay_meshes_opts(
    scmap: &ScMap,
    markers: Option<&scenario::SaveMarkers>,
    neutral_terrain_tint: bool,
    include_water_quad: bool,
) -> (Vec<GpuVertex>, Vec<u32>, Vec<GpuVertex>, Vec<u32>) {
    let hm_w = scmap.heightmap_width;
    let hm_h = scmap.heightmap_height;
    if hm_w == 0 || hm_h == 0 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let max_dim = hm_w.max(hm_h);
    let step = (max_dim / 512).max(1);
    let cells_x = hm_w.div_ceil(step);
    let cells_z = hm_h.div_ceil(step);
    let nx = cells_x + 1;
    let nz = cells_z + 1;

    let scale_x = if scmap.map_width > 0.0 {
        scmap.map_width / hm_w as f32
    } else {
        1.0
    };
    let scale_z = if scmap.map_height > 0.0 {
        scmap.map_height / hm_h as f32
    } else {
        1.0
    };

    let (min_h, max_h, _) = scmap.height_stats();
    let land_base = if scmap.water.has_water {
        scmap.water.elevation
    } else {
        min_h
    };
    let deep_floor = if scmap.water.has_water {
        if scmap.water.elevation_abyss < scmap.water.elevation {
            scmap.water.elevation_abyss.min(min_h)
        } else {
            min_h.min(scmap.water.elevation - 10.0)
        }
    } else {
        min_h
    };

    let mut vertices = Vec::with_capacity((nx * nz) as usize);
    let mut indices = Vec::with_capacity((cells_x * cells_z * 6) as usize);
    let mut ov_vertices = Vec::with_capacity(64);
    let mut ov_indices = Vec::with_capacity(64);

    for iz in 0..nz {
        let z = (iz * step).min(hm_h);
        let z_up = z.saturating_sub(step);
        let z_down = (z + step).min(hm_h);
        let wz = z as f32 * scale_z;
        let v_coord = z as f32 / hm_h as f32;

        for ix in 0..nx {
            let x = (ix * step).min(hm_w);
            let x_left = x.saturating_sub(step);
            let x_right = (x + step).min(hm_w);
            let wx = x as f32 * scale_x;
            let u_coord = x as f32 / hm_w as f32;

            let y = scmap.height_at(x, z).unwrap_or(0.0);
            let h_left = scmap.height_at(x_left, z).unwrap_or(y);
            let h_right = scmap.height_at(x_right, z).unwrap_or(y);
            let h_up = scmap.height_at(x, z_up).unwrap_or(y);
            let h_down = scmap.height_at(x, z_down).unwrap_or(y);

            let nx_v = h_left - h_right;
            let ny_v = 2.0 * step as f32;
            let nz_v = h_up - h_down;
            let normal = normalize_or_up([nx_v, ny_v, nz_v]);

            let color = if neutral_terrain_tint {
                [1.0, 1.0, 1.0, 0.08]
            } else {
                terrain_vertex_color(
                    y,
                    normal,
                    scmap.water.has_water,
                    scmap.water.elevation,
                    deep_floor,
                    land_base,
                    max_h,
                )
            };

            vertices.push(GpuVertex {
                position: [wx, y, wz],
                normal,
                uv: [u_coord, v_coord],
                color,
            });
        }
    }

    for iz in 0..cells_z {
        for ix in 0..cells_x {
            let c00 = iz * nx + ix;
            let c10 = c00 + 1;
            let c01 = (iz + 1) * nx + ix;
            let c11 = c01 + 1;
            indices.extend_from_slice(&[c00, c01, c10, c10, c01, c11]);
        }
    }

    if include_water_quad && scmap.water.has_water {
        let w = if scmap.map_width > 0.0 {
            scmap.map_width
        } else {
            hm_w as f32
        };
        let h = if scmap.map_height > 0.0 {
            scmap.map_height
        } else {
            hm_h as f32
        };
        let wy = scmap.water.elevation;
        let water_color = [0.09, 0.28, 0.46, 0.80];
        let base_idx = ov_vertices.len() as u32;

        ov_vertices.extend_from_slice(&[
            GpuVertex {
                position: [0.0, wy, 0.0],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 0.0],
                color: water_color,
            },
            GpuVertex {
                position: [w, wy, 0.0],
                normal: [0.0, 1.0, 0.0],
                uv: [1.0, 0.0],
                color: water_color,
            },
            GpuVertex {
                position: [0.0, wy, h],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 1.0],
                color: water_color,
            },
            GpuVertex {
                position: [w, wy, h],
                normal: [0.0, 1.0, 0.0],
                uv: [1.0, 1.0],
                color: water_color,
            },
        ]);
        ov_indices.extend_from_slice(&[
            base_idx,
            base_idx + 2,
            base_idx + 1,
            base_idx + 1,
            base_idx + 2,
            base_idx + 3,
        ]);
    }

    if let Some(sm) = markers {
        let map_span = scmap
            .map_width
            .max(scmap.map_height)
            .max(hm_w.max(hm_h) as f32);
        let radius = (map_span * 0.008).clamp(2.0, 14.0);
        let min_marker_y = if scmap.water.has_water {
            scmap.water.elevation
        } else {
            0.0
        };

        let spawn_color = [0.98, 0.80, 0.16, 0.55];
        for spot in &sm.start_spots {
            let base_y = spot.position[1].max(min_marker_y);
            append_pyramid(
                &mut ov_vertices,
                &mut ov_indices,
                [spot.position[0], base_y, spot.position[2]],
                radius,
                radius * 2.8,
                spawn_color,
            );
        }

        let mass_radius = radius * 0.45;
        let mass_color = [0.22, 0.88, 0.38, 0.40];
        for mass in &sm.mass_points {
            let base_y = mass.position[1].max(min_marker_y);
            append_pyramid(
                &mut ov_vertices,
                &mut ov_indices,
                [mass.position[0], base_y, mass.position[2]],
                mass_radius,
                mass_radius * 2.8,
                mass_color,
            );
        }
    }

    (vertices, indices, ov_vertices, ov_indices)
}

#[inline]
fn sample_rgba_bilinear(img: &RgbaImage, u: f32, v: f32) -> [f32; 4] {
    let w = img.width as i32;
    let h = img.height as i32;
    if w <= 0 || h <= 0 || img.pixels.len() < 4 {
        return [1.0, 1.0, 1.0, 1.0];
    }
    let fx = u.rem_euclid(1.0) * w as f32 - 0.5;
    let fy = v.rem_euclid(1.0) * h as f32 - 0.5;
    let x0 = (fx.floor() as i32).rem_euclid(w) as usize;
    let y0 = (fy.floor() as i32).rem_euclid(h) as usize;
    let x1 = (x0 + 1) % (w as usize);
    let y1 = (y0 + 1) % (h as usize);
    let tx = fx - fx.floor();
    let ty = fy - fy.floor();

    let at = |x: usize, y: usize| -> [f32; 4] {
        let i = (y * (w as usize) + x) * 4;
        [
            img.pixels[i] as f32 / 255.0,
            img.pixels[i + 1] as f32 / 255.0,
            img.pixels[i + 2] as f32 / 255.0,
            img.pixels[i + 3] as f32 / 255.0,
        ]
    };
    let c00 = at(x0, y0);
    let c10 = at(x1, y0);
    let c01 = at(x0, y1);
    let c11 = at(x1, y1);
    let mut out = [0.0_f32; 4];
    for c in 0..4 {
        let a = c00[c] + (c10[c] - c00[c]) * tx;
        let b = c01[c] + (c11[c] - c01[c]) * tx;
        out[c] = a + (b - a) * ty;
    }
    out
}

/// Decodes and blends real Forged Alliance `.scmap` terrain strata (`env.scd`) with
/// the embedded texture blend maps (`texturemap0` / `texturemap1`) into an `out_size x out_size` texture.
pub fn bake_terrain_albedo(game_dir: &Path, scmap: &ScMap, out_size: u32) -> Option<RgbaImage> {
    let tt = scmap.terrain_textures.as_ref()?;
    let tm0 = decode_dds_rgba8(&tt.texturemap0_dds).ok()?;
    let tm1 = (!tt.texturemap1_dds.is_empty())
        .then(|| decode_dds_rgba8(&tt.texturemap1_dds).ok())
        .flatten();

    let gamedata = scd::gamedata_dir(game_dir);
    let raw_layers: Vec<Option<Vec<u8>>> = tt
        .strata_albedo
        .iter()
        .map(|s| {
            let trimmed = s.path.trim().trim_start_matches('/').replace('\\', "/");
            if trimmed.is_empty() {
                None
            } else {
                find_asset_in_scds(&gamedata, &["env.scd", "textures.scd"], &[trimmed])
            }
        })
        .collect();

    let threads = jobs::default_thread_count();
    let layers: Vec<Option<RgbaImage>> = jobs::par_map_ordered(&raw_layers, threads, |_, opt| {
        opt.as_deref().and_then(|b| decode_dds_rgba8(b).ok())
    });

    if layers.first().and_then(|o| o.as_ref()).is_none() {
        return None;
    }

    let size = out_size.max(64);
    let map_span = scmap.map_width.max(scmap.map_height).max(256.0);
    let tile_repeat = |idx: usize| -> f32 {
        let raw = tt.strata_albedo.get(idx).map(|s| s.scale).unwrap_or(4.0);
        if raw.is_finite() && raw > 0.05 {
            (map_span / (raw.max(1.5) * 3.0)).clamp(16.0, 96.0)
        } else {
            32.0
        }
    };
    let repeats: Vec<f32> = (0..10).map(tile_repeat).collect();
    let row_indices: Vec<u32> = (0..size).collect();
    let tm1_ref = tm1.as_ref();
    let layers_ref = &layers;
    let repeats_ref = &repeats;

    let rows = jobs::par_map_ordered(&row_indices, threads, |_, &py| {
        let mut row = Vec::with_capacity(size as usize * 4);
        let v = (py as f32 + 0.5) / size as f32;
        for px in 0..size {
            let u = (px as f32 + 0.5) / size as f32;
            let b0 = sample_rgba_bilinear(&tm0, u, v);
            let b1 = tm1_ref.map(|t| sample_rgba_bilinear(t, u, v)).unwrap_or([0.0; 4]);

            let mut rgb = [0.28_f32, 0.34, 0.22];
            if let Some(ref base) = layers_ref[0] {
                let s = sample_rgba_bilinear(base, u * repeats_ref[0], v * repeats_ref[0]);
                rgb = [s[0], s[1], s[2]];
            }
            for ch in 0..4 {
                if let Some(ref layer) = layers_ref[1 + ch] {
                    let w = b0[ch];
                    if w > 0.002 {
                        let s = sample_rgba_bilinear(
                            layer,
                            u * repeats_ref[1 + ch],
                            v * repeats_ref[1 + ch],
                        );
                        for k in 0..3 {
                            rgb[k] = rgb[k] + (s[k] - rgb[k]) * w;
                        }
                    }
                }
            }
            for ch in 0..4 {
                if let Some(ref layer) = layers_ref[5 + ch] {
                    let w = b1[ch];
                    if w > 0.002 {
                        let s = sample_rgba_bilinear(
                            layer,
                            u * repeats_ref[5 + ch],
                            v * repeats_ref[5 + ch],
                        );
                        for k in 0..3 {
                            rgb[k] = rgb[k] + (s[k] - rgb[k]) * w;
                        }
                    }
                }
            }
            if let Some(ref mac) = layers_ref[9] {
                let s = sample_rgba_bilinear(mac, u * 1.5, v * 1.5);
                for k in 0..3 {
                    rgb[k] = (rgb[k] * (0.78 + 0.44 * s[k])).clamp(0.0, 1.0);
                }
            }

            row.push((rgb[0] * 255.0).round().clamp(0.0, 255.0) as u8);
            row.push((rgb[1] * 255.0).round().clamp(0.0, 255.0) as u8);
            row.push((rgb[2] * 255.0).round().clamp(0.0, 255.0) as u8);
            row.push(255);
        }
        row
    });

    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for row in rows {
        pixels.extend_from_slice(&row);
    }
    Some(RgbaImage {
        width: size,
        height: size,
        pixels,
    })
}

/// Loads `{unit_id}_Albedo.dds` + `{unit_id}_SpecTeam.dds` from `units.scd` and packs
/// `(r, g, b, team_mask)` into one RGBA8 texture for hardware-instanced team-colored units.
#[allow(dead_code)]
pub fn load_unit_albedo_and_team_mask(game_dir: &Path, unit_id: &str) -> Option<RgbaImage> {
    let gamedata = scd::gamedata_dir(game_dir);
    let albedo_paths = [
        format!("units/{unit_id}/{unit_id}_Albedo.dds"),
        format!("units/{unit_id}/{unit_id}_albedo.dds"),
    ];
    let albedo = decode_first_dds(&gamedata, &["units.scd", "textures.scd"], &albedo_paths)?;
    let spec_paths = [
        format!("units/{unit_id}/{unit_id}_SpecTeam.dds"),
        format!("units/{unit_id}/{unit_id}_specteam.dds"),
    ];
    let spec = decode_first_dds(&gamedata, &["units.scd"], &spec_paths);
    Some(pack_albedo_team_mask(albedo, spec.as_ref()))
}

/// Packs `(albedo.rgb, spec_team.a)` into one RGBA8 image (nearest-resampled team mask).
pub fn pack_albedo_team_mask(mut albedo: RgbaImage, spec: Option<&RgbaImage>) -> RgbaImage {
    let n = (albedo.width as usize * albedo.height as usize).min(albedo.pixels.len() / 4);
    if let Some(sp) = spec
        && sp.width > 0
        && sp.height > 0
    {
        for i in 0..n {
            let x = i % albedo.width as usize;
            let y = i / albedo.width as usize;
            let sx = (x * sp.width as usize / albedo.width as usize).min(sp.width as usize - 1);
            let sy = (y * sp.height as usize / albedo.height as usize).min(sp.height as usize - 1);
            let si = (sy * sp.width as usize + sx) * 4;
            albedo.pixels[i * 4 + 3] = sp.pixels[si + 3];
        }
    } else {
        for i in 0..n {
            albedo.pixels[i * 4 + 3] = 72;
        }
    }
    albedo
}

fn normalize_tangent_normal_image(mut img: RgbaImage) -> RgbaImage {
    let px_count = img.pixels.len() / 4;
    if px_count == 0 {
        return RgbaImage {
            width: 4,
            height: 4,
            pixels: [128u8, 128, 255, 255].repeat(16),
        };
    }
    let sample_step = (px_count / 64).max(1);
    let mut sum_b: u64 = 0;
    let mut samples: u64 = 0;
    for i in (0..px_count).step_by(sample_step) {
        sum_b += u64::from(img.pixels[i * 4 + 2]);
        samples += 1;
    }
    let avg_b = sum_b / samples.max(1);
    if avg_b < 80 {
        // FA DXT5/DXT3 normal map convention stores tangent X in Alpha and Y in Green.
        for px in img.pixels.chunks_exact_mut(4) {
            let nx = (f32::from(px[3]) / 255.0) * 2.0 - 1.0;
            let ny = (f32::from(px[1]) / 255.0) * 2.0 - 1.0;
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            px[0] = ((nx * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            px[1] = ((ny * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            px[2] = ((nz * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            px[3] = 255;
        }
    } else {
        for px in img.pixels.chunks_exact_mut(4) {
            px[3] = 255;
        }
    }
    img
}

/// Loads all 10 albedo strata, 9 normal strata, and blend maps (`texturemap0`/`texturemap1`)
/// for hardware GPU splatting.
pub fn load_terrain_splat_data(game_dir: &Path, scmap: &ScMap) -> Option<TerrainSplatData> {
    let tt = scmap.terrain_textures.as_ref()?;
    let tm0 = decode_dds_rgba8(&tt.texturemap0_dds).ok()?;
    let tm1 = (!tt.texturemap1_dds.is_empty())
        .then(|| decode_dds_rgba8(&tt.texturemap1_dds).ok())
        .flatten()
        .unwrap_or_else(|| RgbaImage {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0, 0],
        });

    let gamedata = scd::gamedata_dir(game_dir);
    let mut raw_paths = Vec::with_capacity(19);
    for s in tt.strata_albedo.iter().take(10) {
        let p = s.path.trim().trim_start_matches('/').replace('\\', "/");
        raw_paths.push((false, p));
    }
    while raw_paths.len() < 10 {
        raw_paths.push((false, String::new()));
    }
    for s in tt.strata_normal.iter().take(9) {
        let p = s.path.trim().trim_start_matches('/').replace('\\', "/");
        raw_paths.push((true, p));
    }
    while raw_paths.len() < 19 {
        raw_paths.push((true, String::new()));
    }

    let threads = jobs::default_thread_count();
    let decoded: Vec<Option<RgbaImage>> = jobs::par_map_ordered(&raw_paths, threads, |_, (is_n, p)| {
        if p.is_empty() {
            return None;
        }
        let bytes = find_asset_in_scds(&gamedata, &["env.scd", "textures.scd"], std::slice::from_ref(p))?;
        let img = decode_dds_rgba8(&bytes).ok()?;
        if *is_n {
            Some(resize_rgba_bilinear(&normalize_tangent_normal_image(img), 256, 256))
        } else {
            Some(resize_rgba_bilinear(&img, 512, 512))
        }
    });

    if decoded[0].is_none() {
        return None;
    }

    let neutral_albedo = RgbaImage {
        width: 512,
        height: 512,
        pixels: [128u8, 128, 128, 255].repeat(512 * 512),
    };
    let neutral_normal = RgbaImage {
        width: 256,
        height: 256,
        pixels: [128u8, 128, 255, 255].repeat(256 * 256),
    };

    let mut strata_albedo = Vec::with_capacity(10);
    let mut albedo_scales = [0.0_f32; 10];
    let mut albedo_active = [0u32; 10];
    for i in 0..10 {
        if let Some(ref img) = decoded[i] {
            strata_albedo.push(img.clone());
            let raw = tt.strata_albedo.get(i).map(|s| s.scale).unwrap_or(4.0);
            albedo_scales[i] = if raw.is_finite() && raw > 0.05 { raw } else { 4.0 };
            albedo_active[i] = 1;
        } else {
            strata_albedo.push(neutral_albedo.clone());
        }
    }

    let mut strata_normal = Vec::with_capacity(9);
    let mut normal_scales = [0.0_f32; 9];
    let mut normal_active = [0u32; 9];
    for i in 0..9 {
        if let Some(ref img) = decoded[10 + i] {
            strata_normal.push(img.clone());
            let raw = tt.strata_normal.get(i).map(|s| s.scale).unwrap_or(4.0);
            normal_scales[i] = if raw.is_finite() && raw > 0.05 { raw } else { 4.0 };
            normal_active[i] = 1;
        } else {
            strata_normal.push(neutral_normal.clone());
        }
    }

    let mw = if scmap.map_width > 0.0 {
        scmap.map_width
    } else {
        scmap.heightmap_width.max(1) as f32
    };
    let mh = if scmap.map_height > 0.0 {
        scmap.map_height
    } else {
        scmap.heightmap_height.max(1) as f32
    };

    Some(TerrainSplatData {
        tm0,
        tm1,
        strata_albedo,
        strata_normal,
        albedo_scales,
        albedo_active,
        normal_scales,
        normal_active,
        lighting: scmap.lighting.clone(),
        map_size: [mw, mh],
        water_elevation: if scmap.water.has_water {
            scmap.water.elevation
        } else {
            -1000.0
        },
        water_deep: scmap.water.elevation_deep,
        water_abyss: scmap.water.elevation_abyss,
    })
}

/// Builds a depth-encoded water mesh plus real wave normal & water color ramp textures from `.scmap`.
pub fn load_water_gpu_data(game_dir: &Path, scmap: &ScMap) -> Option<WaterGpuData> {
    if !scmap.water.has_water || scmap.heightmap_width == 0 || scmap.heightmap_height == 0 {
        return None;
    }
    let mw = if scmap.map_width > 0.0 {
        scmap.map_width
    } else {
        scmap.heightmap_width as f32
    };
    let mh = if scmap.map_height > 0.0 {
        scmap.map_height
    } else {
        scmap.heightmap_height as f32
    };
    let grid_n = 128u32;
    let nx = grid_n + 1;
    let elev = scmap.water.elevation;

    let mut vertices = Vec::with_capacity((nx * nx) as usize);
    let mut depths = Vec::with_capacity((nx * nx) as usize);
    for iz in 0..nx {
        let v = iz as f32 / grid_n as f32;
        let wz = v * mh;
        let hz = (v * scmap.heightmap_height as f32)
            .round()
            .clamp(0.0, scmap.heightmap_height as f32) as u32;
        for ix in 0..nx {
            let u = ix as f32 / grid_n as f32;
            let wx = u * mw;
            let hx = (u * scmap.heightmap_width as f32)
                .round()
                .clamp(0.0, scmap.heightmap_width as f32) as u32;
            let h = scmap.height_at(hx, hz).unwrap_or(elev);
            let depth = (elev - h).max(0.0);
            depths.push(elev - h);
            vertices.push(GpuVertex {
                position: [wx, elev, wz],
                normal: [0.0, 1.0, 0.0],
                uv: [u, v],
                color: [depth, 0.0, 0.0, 1.0],
            });
        }
    }
    let mut indices = Vec::with_capacity((grid_n * grid_n * 6) as usize);
    for iz in 0..grid_n {
        for ix in 0..grid_n {
            let c00 = (iz * nx + ix) as usize;
            let c10 = c00 + 1;
            let c01 = c00 + nx as usize;
            let c11 = c01 + 1;
            if depths[c00] > -2.0
                || depths[c10] > -2.0
                || depths[c01] > -2.0
                || depths[c11] > -2.0
            {
                indices.extend_from_slice(&[
                    c00 as u32, c01 as u32, c10 as u32, c10 as u32, c01 as u32, c11 as u32,
                ]);
            }
        }
    }

    let gamedata = scd::gamedata_dir(game_dir);
    let wp = scmap.terrain_textures.as_ref().map(|t| &t.water_params);
    let wave_candidates: Vec<String> = wp
        .and_then(|w| w.wave_textures.first())
        .map(|s| vec![s.trim().trim_start_matches('/').replace('\\', "/"), "textures/engine/waves.dds".to_string()])
        .unwrap_or_else(|| vec!["textures/engine/waves.dds".to_string()]);
    let waves_normal = decode_first_dds(&gamedata, &["textures.scd", "env.scd"], &wave_candidates)
        .map(normalize_tangent_normal_image)
        .unwrap_or_else(|| RgbaImage {
            width: 4,
            height: 4,
            pixels: [128u8, 128, 255, 255].repeat(16),
        });

    let ramp_candidates: Vec<String> = wp
        .map(|w| {
            vec![
                w.water_ramp.trim().trim_start_matches('/').replace('\\', "/"),
                "textures/engine/waterramp_eg.dds".to_string(),
            ]
        })
        .unwrap_or_else(|| vec!["textures/engine/waterramp_eg.dds".to_string()]);
    let water_ramp = decode_first_dds(&gamedata, &["textures.scd", "env.scd"], &ramp_candidates)
        .unwrap_or_else(|| RgbaImage {
            width: 4,
            height: 1,
            pixels: vec![
                62, 132, 148, 255, 36, 98, 128, 255, 18, 62, 98, 255, 10, 36, 68, 255,
            ],
        });

    Some(WaterGpuData {
        vertices,
        indices,
        waves_normal,
        water_ramp,
        fresnel_bias: wp.map(|w| w.fresnel_bias).unwrap_or(0.06),
        fresnel_power: wp.map(|w| w.fresnel_power).unwrap_or(1.8),
        surface_color: wp.map(|w| w.surface_color).unwrap_or([0.06, 0.09, 0.12]),
    })
}

/// Decodes the map's 6-face sky cubemap (`sky_cubemap_path` or `<default>` cubemap) from `textures.scd`.
pub fn load_sky_cubemap(game_dir: &Path, scmap: &ScMap) -> Option<[RgbaImage; 6]> {
    let gamedata = scd::gamedata_dir(game_dir);
    let mut candidates = Vec::new();
    let sky = scmap.sky_cubemap_path.trim().trim_start_matches('/').replace('\\', "/");
    if !sky.is_empty() {
        candidates.push(sky);
    }
    for (_, p) in &scmap.cubemaps {
        let trimmed = p.trim().trim_start_matches('/').replace('\\', "/");
        if !trimmed.is_empty() {
            candidates.push(trimmed);
        }
    }
    let bytes = find_asset_in_scds(&gamedata, &["textures.scd", "env.scd"], &candidates)?;
    decode_dds_cubemap_rgba8(&bytes).ok()
}

fn resolve_rel_vfs_path(base_dir: &str, raw: &str) -> String {
    let norm = raw.trim().replace('\\', "/");
    if norm.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = if norm.starts_with('/') {
        Vec::new()
    } else {
        base_dir
            .split('/')
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    };
    for seg in norm.trim_start_matches('/').split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other.to_string()),
        }
    }
    parts.join("/")
}

#[allow(dead_code)]
pub(crate) fn sample_scmap_height(scmap: &ScMap, wx: f32, wz: f32) -> f32 {
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
    h0 + (h1 - h0) * tz
}

#[allow(dead_code)]
pub(crate) fn sample_scmap_normal(scmap: &ScMap, wx: f32, wz: f32) -> [f32; 3] {
    let hm_w = scmap.heightmap_width;
    let hm_h = scmap.heightmap_height;
    if hm_w == 0 || hm_h == 0 {
        return [0.0, 1.0, 0.0];
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
    let eps_x = (mw / hm_w as f32).clamp(0.25, 2.0);
    let eps_z = (mh / hm_h as f32).clamp(0.25, 2.0);

    let x_l = (wx - eps_x).clamp(0.0, mw);
    let x_r = (wx + eps_x).clamp(0.0, mw);
    let z_d = (wz - eps_z).clamp(0.0, mh);
    let z_u = (wz + eps_z).clamp(0.0, mh);

    let dx = (x_r - x_l).max(1e-4);
    let dz = (z_u - z_d).max(1e-4);
    let dh_dx = (sample_scmap_height(scmap, x_r, wz) - sample_scmap_height(scmap, x_l, wz)) / dx;
    let dh_dz = (sample_scmap_height(scmap, wx, z_u) - sample_scmap_height(scmap, wx, z_d)) / dz;
    normalize_or_up([-dh_dx, 1.0, -dh_dz])
}

/// Loads terrain-draped albedo decals from `scmap`, grouped by texture path (top 24 by count),
/// subdivided into `2x2` terrain-sampled quads lifted `+0.08m` above the terrain.
#[allow(dead_code)]
pub fn load_map_decals(game_dir: &Path, scmap: &ScMap) -> Vec<DecalBatch> {
    let Some(ref tt) = scmap.terrain_textures else {
        return Vec::new();
    };
    if tt.decals.is_empty() {
        return Vec::new();
    }

    let mut grouped: BTreeMap<String, (String, Vec<&ScMapDecal>)> = BTreeMap::new();
    for d in &tt.decals {
        if !d.position[0].is_finite()
            || !d.position[2].is_finite()
            || !d.scale[0].is_finite()
            || !d.scale[2].is_finite()
            || d.scale[0] <= 0.01
            || d.scale[2] <= 0.01
        {
            continue;
        }
        let Some(raw_tex) = d.textures.iter().map(|s| s.trim()).find(|s| !s.is_empty()) else {
            continue;
        };
        let tex_norm = raw_tex.trim_start_matches(['/', '\\']).replace('\\', "/");
        if tex_norm.is_empty() {
            continue;
        }
        let is_albedo = d.decal_type == 1
            || ends_with_ci(&tex_norm, "_albedo.dds")
            || !ends_with_ci(&tex_norm, "_normals.dds");
        if !is_albedo {
            continue;
        }
        grouped
            .entry(tex_norm.to_ascii_lowercase())
            .or_insert_with(|| (tex_norm, Vec::new()))
            .1
            .push(d);
    }

    let mut groups: Vec<(String, Vec<&ScMapDecal>)> = grouped.into_values().collect();
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
    groups.truncate(24);

    let gamedata = scd::gamedata_dir(game_dir);
    let threads = jobs::default_thread_count();
    let built = jobs::par_map_ordered(&groups, threads, |_, (tex_path, decals)| -> Option<DecalBatch> {
        let albedo = decode_first_dds(
            &gamedata,
            &["env.scd", "textures.scd"],
            std::slice::from_ref(tex_path),
        );
        let mut vertices = Vec::with_capacity(decals.len() * 9);
        let mut indices = Vec::with_capacity(decals.len() * 24);

        for d in decals {
            let yaw = if d.rotation[1].is_finite() {
                d.rotation[1]
            } else {
                0.0
            };
            let (sy, cy) = yaw.sin_cos();
            let cx = d.position[0];
            let cz = d.position[2];
            let sx = d.scale[0];
            let sz = d.scale[2];

            let Ok(base) = u32::try_from(vertices.len()) else {
                break;
            };
            for iz in 0..3u32 {
                let v = iz as f32 * 0.5;
                let lz = (v - 0.5) * sz;
                for ix in 0..3u32 {
                    let u = ix as f32 * 0.5;
                    let lx = (u - 0.5) * sx;
                    let wx = cx + lx * cy + lz * sy;
                    let wz = cz - lx * sy + lz * cy;
                    let wy = sample_scmap_height(scmap, wx, wz) + 0.08;
                    let normal = sample_scmap_normal(scmap, wx, wz);
                    vertices.push(GpuVertex {
                        position: [wx, wy, wz],
                        normal,
                        uv: [u, v],
                        color: [1.0, 1.0, 1.0, 1.0],
                    });
                }
            }
            for iz in 0..2u32 {
                for ix in 0..2u32 {
                    let c00 = base + iz * 3 + ix;
                    let c10 = c00 + 1;
                    let c01 = c00 + 3;
                    let c11 = c01 + 1;
                    indices.extend_from_slice(&[c00, c01, c10, c10, c01, c11]);
                }
            }
        }

        if vertices.is_empty() || indices.is_empty() {
            return None;
        }
        Some(DecalBatch {
            label: tex_path.clone(),
            vertices,
            indices,
            albedo,
        })
    });

    built.into_iter().flatten().collect()
}

struct PropChunk<'a> {
    chunk_idx: usize,
    refs: Vec<&'a ScMapProp>,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}

fn partition_prop_chunks<'a>(
    refs: &[&'a ScMapProp],
    map_width: f32,
    map_height: f32,
) -> Vec<PropChunk<'a>> {
    let mw = map_width.max(1.0);
    let mh = map_height.max(1.0);
    let mut buckets: [Vec<&'a ScMapProp>; 64] = std::array::from_fn(|_| Vec::new());

    for &p in refs {
        if !p.position[0].is_finite() || !p.position[1].is_finite() || !p.position[2].is_finite() {
            continue;
        }
        let cx = ((p.position[0] / mw) * 8.0).floor().clamp(0.0, 7.0) as usize;
        let cz = ((p.position[2] / mh) * 8.0).floor().clamp(0.0, 7.0) as usize;
        buckets[cz * 8 + cx].push(p);
    }

    let mut chunks = Vec::new();
    for (chunk_idx, chunk_refs) in buckets.into_iter().enumerate() {
        if chunk_refs.is_empty() {
            continue;
        }
        let mut min_p = [f32::INFINITY; 3];
        let mut max_p = [f32::NEG_INFINITY; 3];
        for p in &chunk_refs {
            min_p[0] = min_p[0].min(p.position[0]);
            min_p[1] = min_p[1].min(p.position[1]);
            min_p[2] = min_p[2].min(p.position[2]);
            max_p[0] = max_p[0].max(p.position[0]);
            max_p[1] = max_p[1].max(p.position[1]);
            max_p[2] = max_p[2].max(p.position[2]);
        }
        chunks.push(PropChunk {
            chunk_idx,
            refs: chunk_refs,
            bounds_min: [min_p[0] - 8.0, min_p[1] - 4.0, min_p[2] - 8.0],
            bounds_max: [max_p[0] + 8.0, max_p[1] + 18.0, max_p[2] + 8.0],
        });
    }
    chunks
}

/// Loads unique `.scmap` prop meshes & textures from `env.scd` and builds spatially chunked
/// (`8x8` world XZ grid) hardware-instanced batches with world-space AABBs.
pub fn load_map_props(game_dir: &Path, scmap: &ScMap) -> Vec<PropMeshBatch> {
    let Some(ref tt) = scmap.terrain_textures else {
        return Vec::new();
    };
    if tt.props.is_empty() {
        return Vec::new();
    }
    let mut grouped: BTreeMap<String, (String, Vec<&ScMapProp>)> = BTreeMap::new();
    for p in &tt.props {
        let clean = p.blueprint_path.trim().trim_start_matches('/').replace('\\', "/");
        if clean.is_empty() {
            continue;
        }
        grouped
            .entry(clean.to_ascii_lowercase())
            .or_insert_with(|| (clean, Vec::new()))
            .1
            .push(p);
    }

    let mw = if scmap.map_width > 0.0 {
        scmap.map_width
    } else {
        scmap.heightmap_width.max(1) as f32
    };
    let mh = if scmap.map_height > 0.0 {
        scmap.map_height
    } else {
        scmap.heightmap_height.max(1) as f32
    };

    let entries: Vec<(String, Vec<&ScMapProp>)> = grouped.into_values().collect();
    let gamedata = scd::gamedata_dir(game_dir);
    let env_scd = gamedata.join("env.scd");
    let threads = jobs::default_thread_count();

    let built = jobs::par_map_ordered(&entries, threads, |_, (bp_rel, refs)| -> Vec<PropMeshBatch> {
        let Ok((actual_bp, bp_bytes)) = scd::read_scd_file_ci(&env_scd, bp_rel) else {
            return Vec::new();
        };
        let bp_text = String::from_utf8_lossy(&bp_bytes);
        let Ok(root) = blueprint::parse_prop_blueprint(&bp_text) else {
            return Vec::new();
        };

        let display = root.get("Display");
        let uniform_scale = display
            .and_then(|d| d.get_f64("UniformScale"))
            .unwrap_or(1.0) as f32;
        let lod0 = display
            .and_then(|d| d.get("Mesh"))
            .and_then(|m| m.get("LODs"))
            .and_then(|l| l.table_list().into_iter().next());

        let bp_norm = actual_bp.replace('\\', "/");
        let bp_dir = bp_norm.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
        let base_stem = strip_suffix_ci(&bp_norm, "_prop.bp")
            .or_else(|| strip_suffix_ci(&bp_norm, ".bp"))
            .unwrap_or(&bp_norm);

        let mut mesh_cands = Vec::new();
        if let Some(mname) = lod0.and_then(|l| l.get_str("MeshName")) {
            let p = resolve_rel_vfs_path(bp_dir, mname);
            if !p.is_empty() {
                mesh_cands.push(p);
            }
        }
        mesh_cands.push(format!("{base_stem}_lod0.scm"));
        mesh_cands.push(format!("{base_stem}_lod1.scm"));

        let Some(scm_bytes) = find_asset_in_scds(&gamedata, &["env.scd"], &mesh_cands) else {
            return Vec::new();
        };
        let Ok(mesh) = ScmMesh::parse(&scm_bytes) else {
            return Vec::new();
        };
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            return Vec::new();
        }

        let mut alb_cands = Vec::new();
        if let Some(aname) = lod0.and_then(|l| l.get_str("AlbedoName")) {
            let p = resolve_rel_vfs_path(bp_dir, aname);
            if !p.is_empty() {
                alb_cands.push(p);
            }
        }
        alb_cands.push(format!("{base_stem}_albedo.dds"));
        let albedo = decode_first_dds(&gamedata, &["env.scd", "textures.scd"], &alb_cands);

        let shader_name = lod0.and_then(|l| l.get_str("ShaderName")).unwrap_or("");
        let alpha_cutout = shader_name.to_ascii_lowercase().contains("alpha")
            || bp_norm.to_ascii_lowercase().contains("tree");

        let vertices: Vec<GpuVertex> = mesh
            .vertices
            .iter()
            .map(|v| GpuVertex {
                position: [
                    v.position[0] * uniform_scale,
                    v.position[1] * uniform_scale,
                    v.position[2] * uniform_scale,
                ],
                normal: normalize_or_up(v.normal),
                uv: v.uv0,
                color: [1.0, 1.0, 1.0, 0.2],
            })
            .collect();
        let indices: Vec<u32> = mesh.indices.iter().map(|&i| u32::from(i)).collect();

        let use_tex = if albedo.is_some() { 1.0 } else { 0.0 };
        let cut = if alpha_cutout { 1.0 } else { 0.0 };

        partition_prop_chunks(refs, mw, mh)
            .into_iter()
            .map(|chunk| {
                let instances = chunk
                    .refs
                    .iter()
                    .map(|p| {
                        GpuUnitInstance::from_basis(
                            p.position,
                            p.rot_x,
                            p.rot_y,
                            p.rot_z,
                            p.scale,
                            [0.95, 0.95, 0.95, 0.12],
                            [use_tex, cut, 0.0, 0.0],
                        )
                    })
                    .collect();
                PropMeshBatch {
                    label: format!("{bp_norm}#c{:02}", chunk.chunk_idx),
                    vertices: vertices.clone(),
                    indices: indices.clone(),
                    albedo: albedo.clone(),
                    alpha_cutout,
                    instances,
                    bounds_min: chunk.bounds_min,
                    bounds_max: chunk.bounds_max,
                }
            })
            .collect()
    });

    built.into_iter().flatten().collect()
}

#[allow(dead_code)]
const CORE_STRATEGIC_ICONS: &[&str] = &[
    "icon_commander_generic",
    "icon_land1_engineer",
    "icon_land1_directfire",
    "icon_land1_artillery",
    "icon_land2_directfire",
    "icon_bot1_directfire",
    "icon_bot3_directfire",
    "icon_fighter1_antiair",
    "icon_bomber1_directfire",
    "icon_gunship2_directfire",
    "icon_ship1_intel",
    "icon_ship2_antinavy",
    "icon_factory1_land",
    "icon_factory1_air",
    "icon_factory1_naval",
    "icon_structure1_mass",
    "icon_structure1_energy",
    "icon_structure2_shield",
    "icon_structure2_missile",
    "icon_structure1_antiair",
    "icon_structure1_intel",
    "icon_experimental_generic",
];

#[allow(dead_code)]
fn normalize_icon_base_name(raw: &str) -> String {
    let s = raw.trim().replace('\\', "/");
    let leaf = s.rsplit('/').next().unwrap_or(&s);
    let no_ext = strip_suffix_ci(leaf, ".dds").unwrap_or(leaf);
    let base = strip_suffix_ci(no_ext, "_rest")
        .or_else(|| strip_suffix_ci(no_ext, "_selected"))
        .or_else(|| strip_suffix_ci(no_ext, "_over"))
        .unwrap_or(no_ext);
    base.to_ascii_lowercase()
}

#[allow(dead_code)]
fn strategic_icon_sdf(name: &str, dx: f32, dy: f32) -> f32 {
    let adx = dx.abs();
    let ady = dy.abs();
    if name.contains("commander") {
        (adx + ady - 12.0).min(adx.max(ady) - 7.5)
    } else if name.contains("experimental") {
        (adx + ady - 13.0).min(adx.max(ady) - 9.0)
    } else if name.contains("factory") {
        adx.max(ady) - 10.5
    } else if name.contains("fighter") || name.contains("bomber") || name.contains("gunship") {
        let half_w = ((dy + 10.5) / 20.0) * 11.0;
        (-dy - 10.5).max(dy - 9.5).max(adx - half_w)
    } else if name.contains("ship") || name.contains("naval") {
        let half_w = ((10.5 - dy) / 18.0) * 11.5;
        (-dy - 7.5).max(dy - 9.5).max(adx - half_w)
    } else if name.contains("engineer") {
        (adx - 3.5).max(ady - 10.5).min((ady - 3.5).max(adx - 10.5))
    } else if name.contains("artillery") || name.contains("missile") {
        (adx * 1.45 + ady) - 11.5
    } else if name.contains("structure")
        || name.contains("shield")
        || name.contains("intel")
        || name.contains("mass")
        || name.contains("energy")
        || name.contains("antiair")
    {
        adx.max(ady).max((adx + ady) * 0.72) - 9.5
    } else {
        (adx + ady) - 10.5
    }
}

#[allow(dead_code)]
fn render_fallback_strategic_icon(name: &str, selected: bool) -> RgbaImage {
    const CELL: usize = 32;
    let mut pixels = vec![0u8; CELL * CELL * 4];
    for y in 0..CELL {
        let dy = y as f32 - 15.5;
        for x in 0..CELL {
            let dx = x as f32 - 15.5;
            let adx = dx.abs();
            let ady = dy.abs();
            let sd = strategic_icon_sdf(name, dx, dy);
            let mut inner_cut = false;
            if sd <= -2.5 && name.contains("commander") {
                inner_cut = adx <= 1.2 && ady <= 4.2 || ady <= 1.2 && adx <= 4.2;
            } else if sd <= -2.5 && name.contains("factory") {
                inner_cut = adx <= 5.0 && ady <= 1.5;
            } else if sd <= -2.5 && name.contains("shield") {
                inner_cut = adx + ady <= 4.0;
            }

            let rgba: [u8; 4] = if inner_cut {
                if selected {
                    [255, 224, 96, 255]
                } else {
                    [36, 44, 56, 255]
                }
            } else if sd <= -1.5 {
                if selected {
                    [255, 255, 255, 255]
                } else {
                    [232, 238, 245, 255]
                }
            } else if sd <= 0.5 {
                if selected {
                    [255, 224, 96, 255]
                } else {
                    [16, 20, 26, 255]
                }
            } else if selected && adx.max(ady) >= 12.0 && adx.max(ady) <= 14.5 && adx >= 8.0 && ady >= 8.0 {
                [255, 232, 110, 255]
            } else {
                [0, 0, 0, 0]
            };

            let off = (y * CELL + x) * 4;
            pixels[off..off + 4].copy_from_slice(&rgba);
        }
    }
    RgbaImage {
        width: CELL as u32,
        height: CELL as u32,
        pixels,
    }
}

/// Loads FA strategic icon DDS textures (`_rest.dds` & `_selected.dds`) from `textures.scd`
/// and packs them into a `16`-column `32x32` RGBA atlas with procedural fallback icons.
#[allow(dead_code)]
pub fn load_strategic_icon_atlas(
    game_dir: Option<&Path>,
    icon_names: &[String],
) -> StrategicIconAtlas {
    const CELL: u32 = 32;
    const COLS: u32 = 16;

    let mut unique = BTreeSet::new();
    for &core in CORE_STRATEGIC_ICONS {
        unique.insert(core.to_string());
    }
    for raw in icon_names {
        let norm = normalize_icon_base_name(raw);
        if !norm.is_empty() {
            unique.insert(norm);
        }
    }

    let dds_map: BTreeMap<String, Vec<u8>> = game_dir
        .and_then(|dir| {
            let textures_scd = scd::gamedata_dir(dir).join("textures.scd");
            scd::read_scd_matching(&textures_scd, |entry_name| {
                let norm = entry_name.replace('\\', "/").to_ascii_lowercase();
                let trimmed = norm.trim_start_matches('/');
                trimmed.starts_with("textures/ui/common/game/strategicicons/")
                    && trimmed.ends_with(".dds")
            })
            .ok()
        })
        .map(|entries| {
            entries
                .into_iter()
                .map(|(k, v)| (k.replace('\\', "/").trim_start_matches('/').to_ascii_lowercase(), v))
                .collect()
        })
        .unwrap_or_default();

    let mut slots_meta = Vec::with_capacity(unique.len() * 2);
    for name in unique {
        let rest_path = format!("textures/ui/common/game/strategicicons/{name}_rest.dds");
        let sel_path = format!("textures/ui/common/game/strategicicons/{name}_selected.dds");
        slots_meta.push((name.clone(), false, format!("{name}:rest"), rest_path));
        slots_meta.push((name.clone(), true, format!("{name}:sel"), sel_path));
    }

    let threads = jobs::default_thread_count();
    let tiles: Vec<RgbaImage> = jobs::par_map_ordered(&slots_meta, threads, |_, (name, selected, _, rel_path)| {
        if let Some(bytes) = dds_map.get(rel_path)
            && let Ok(img) = decode_dds_rgba8(bytes)
        {
            return resize_rgba_bilinear(&img, CELL, CELL);
        }
        render_fallback_strategic_icon(name, *selected)
    });

    let total_slots = slots_meta.len().max(1) as u32;
    let rows = total_slots.div_ceil(COLS);
    let atlas_w = COLS * CELL;
    let atlas_h = rows * CELL;
    let mut pixels = vec![0u8; (atlas_w as usize) * (atlas_h as usize) * 4];
    let mut uv_rects = BTreeMap::new();

    let pad_u = 0.5 / atlas_w as f32;
    let pad_v = 0.5 / atlas_h as f32;

    for (idx, ((_, _, key, _), tile)) in slots_meta.into_iter().zip(tiles.into_iter()).enumerate() {
        let idx_u32 = idx as u32;
        let col = idx_u32 % COLS;
        let row = idx_u32 / COLS;
        let dst_x = (col * CELL) as usize;
        let dst_y = (row * CELL) as usize;
        let row_bytes = CELL as usize * 4;
        for y in 0..CELL as usize {
            let src_off = y * row_bytes;
            let dst_off = ((dst_y + y) * atlas_w as usize + dst_x) * 4;
            pixels[dst_off..dst_off + row_bytes].copy_from_slice(&tile.pixels[src_off..src_off + row_bytes]);
        }
        let u0 = (col * CELL) as f32 / atlas_w as f32 + pad_u;
        let v0 = (row * CELL) as f32 / atlas_h as f32 + pad_v;
        let u1 = ((col + 1) * CELL) as f32 / atlas_w as f32 - pad_u;
        let v1 = ((row + 1) * CELL) as f32 / atlas_h as f32 - pad_v;
        uv_rects.insert(key, [u0, v0, u1, v1]);
    }

    let default_uv = uv_rects
        .get("icon_land1_directfire:rest")
        .or_else(|| uv_rects.values().next())
        .copied()
        .unwrap_or([0.0, 0.0, 1.0, 1.0]);

    StrategicIconAtlas {
        image: RgbaImage {
            width: atlas_w,
            height: atlas_h,
            pixels,
        },
        uv_rects,
        default_uv,
    }
}

/// Loads FA `64x64` HUD unit icons from `textures.scd` into a horizontal `(64*N)x64` atlas.
#[allow(dead_code)]
pub fn load_hud_icon_atlas(game_dir: &Path, unit_ids: &[&str]) -> Option<RgbaImage> {
    if unit_ids.is_empty() {
        return None;
    }
    let gamedata = scd::gamedata_dir(game_dir);
    let cell = 64usize;
    let total_w = unit_ids.len() * cell;
    let mut pixels = vec![0u8; total_w * cell * 4];
    let mut found_any = false;

    for (slot, id) in unit_ids.iter().enumerate() {
        let up = id.to_ascii_uppercase();
        let candidates = [
            format!("textures/ui/common/icons/units/{up}_icon.dds"),
            format!("textures/ui/common/icons/units/{}_icon.dds", id.to_ascii_lowercase()),
        ];
        if let Some(img) = decode_first_dds(&gamedata, &["textures.scd"], &candidates) {
            let icon = resize_rgba_bilinear(&img, cell as u32, cell as u32);
            found_any = true;
            for y in 0..cell {
                let src_row = &icon.pixels[y * cell * 4..(y + 1) * cell * 4];
                let dst_start = (y * total_w + slot * cell) * 4;
                pixels[dst_start..dst_start + cell * 4].copy_from_slice(src_row);
            }
        }
    }

    if !found_any {
        return None;
    }
    Some(RgbaImage {
        width: total_w as u32,
        height: cell as u32,
        pixels,
    })
}

fn terrain_vertex_color(
    y: f32,
    normal: [f32; 3],
    has_water: bool,
    water_elevation: f32,
    deep_floor: f32,
    land_base: f32,
    max_h: f32,
) -> [f32; 4] {
    if has_water && y < water_elevation {
        let span = (water_elevation - deep_floor).max(1.0);
        let t = ((y - deep_floor) / span).clamp(0.0, 1.0);
        let rgb = lerp3([0.04, 0.10, 0.18], [0.16, 0.34, 0.38], t);
        [rgb[0], rgb[1], rgb[2], 0.05]
    } else if has_water && y < water_elevation + 1.2 {
        [0.70, 0.64, 0.46, 0.08]
    } else {
        let t = ((y - land_base) / (max_h - land_base).max(1.0)).clamp(0.0, 1.0);
        let base_rgb = if t < 0.5 {
            lerp3([0.25, 0.39, 0.20], [0.48, 0.44, 0.32], t * 2.0)
        } else {
            lerp3([0.48, 0.44, 0.32], [0.68, 0.66, 0.62], (t - 0.5) * 2.0)
        };
        let cliff = ((0.78 - normal[1]) / 0.35).clamp(0.0, 1.0);
        let rgb = lerp3(base_rgb, [0.32, 0.31, 0.29], cliff);
        [rgb[0], rgb[1], rgb[2], 0.10]
    }
}

fn append_pyramid(
    vertices: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    base_center: [f32; 3],
    radius: f32,
    height: f32,
    color: [f32; 4],
) {
    let [cx, cy, cz] = base_center;
    let apex = [cx, cy + height, cz];
    let corners = [
        [cx - radius, cy, cz - radius],
        [cx - radius, cy, cz + radius],
        [cx + radius, cy, cz + radius],
        [cx + radius, cy, cz - radius],
    ];

    for side in 0..4 {
        let b0 = corners[side];
        let b1 = corners[(side + 1) % 4];
        let e1 = [b0[0] - apex[0], b0[1] - apex[1], b0[2] - apex[2]];
        let e2 = [b1[0] - apex[0], b1[1] - apex[1], b1[2] - apex[2]];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let normal = normalize_or_up(cross);
        let idx = vertices.len() as u32;
        vertices.extend_from_slice(&[
            GpuVertex {
                position: apex,
                normal,
                uv: [0.5, 0.0],
                color,
            },
            GpuVertex {
                position: b0,
                normal,
                uv: [0.0, 1.0],
                color,
            },
            GpuVertex {
                position: b1,
                normal,
                uv: [1.0, 1.0],
                color,
            },
        ]);
        indices.extend_from_slice(&[idx, idx + 1, idx + 2]);
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn normalize_or_up(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq.is_finite() && len_sq > 1e-12 {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn pick_team_color(categories: &[String], unit_id: &str) -> [u8; 3] {
    for cat in categories {
        if cat.eq_ignore_ascii_case("AEON") {
            return [36, 178, 95];
        }
        if cat.eq_ignore_ascii_case("CYBRAN") {
            return [218, 44, 38];
        }
        if cat.eq_ignore_ascii_case("SERAPHIM") {
            return [232, 182, 48];
        }
        if cat.eq_ignore_ascii_case("UEF") {
            return [48, 112, 230];
        }
    }

    let mut chars = unit_id.chars();
    let first = chars.next().map(|c| c.to_ascii_uppercase());
    let second = chars.next().map(|c| c.to_ascii_uppercase());

    if matches!(first, Some('U' | 'D' | 'X')) && second == Some('A') {
        [36, 178, 95]
    } else if second == Some('R') {
        [218, 44, 38]
    } else if second == Some('S') {
        [232, 182, 48]
    } else {
        [48, 112, 230]
    }
}

/// Resolves `walk` / `idle` / `.sca` name substring to a parsed clip from `units.scd`.
fn resolve_unit_clip(
    gamedata: &Path,
    bp_dir: &str,
    unit_id: &str,
    info: &UnitDisplayInfo,
    sel: &str,
) -> Result<(String, crate::sca::ScaAnimation), String> {
    let units_scd = gamedata.join("units.scd");
    let wanted = sel.trim().to_ascii_lowercase();
    let path = match wanted.as_str() {
        "walk" => info
            .walk_anim
            .clone()
            .ok_or_else(|| format!("{unit_id} has no Display.AnimationWalk"))?,
        "idle" => info
            .idle_anim
            .clone()
            .ok_or_else(|| format!("{unit_id} has no Display.AnimationIdle"))?,
        "death" => info
            .death_anims
            .first()
            .map(|d| d.path.clone())
            .ok_or_else(|| format!("{unit_id} has no Display.AnimationDeath"))?,
        "open" => info
            .open_anim
            .clone()
            .ok_or_else(|| format!("{unit_id} has no Display.AnimationOpen"))?,
        "build" => info
            .build_anim
            .clone()
            .ok_or_else(|| format!("{unit_id} has no Display.AnimationBuild"))?,
        _ => {
            let dir = if bp_dir.is_empty() {
                format!("units/{}", unit_id.to_ascii_lowercase())
            } else {
                bp_dir.to_ascii_lowercase()
            };
            let mut hits: Vec<String> = scd::list_scd(&units_scd)
                .map_err(|e| format!("{}: {e}", units_scd.display()))?
                .into_iter()
                .filter(|e| !e.is_dir)
                .map(|e| e.name.replace('\\', "/"))
                .filter(|n| {
                    let l = n.trim_start_matches('/').to_ascii_lowercase();
                    l.starts_with(&dir) && l.ends_with(".sca") && l.contains(&wanted)
                })
                .collect();
            hits.sort();
            hits.into_iter()
                .next()
                .ok_or_else(|| format!("no .sca matching '{sel}' under {dir}/"))?
        }
    };
    let (found, bytes) = scd::read_scd_file_ci(&units_scd, &path)
        .map_err(|e| format!("animation '{path}': {e}"))?;
    let anim = crate::sca::ScaAnimation::parse(&bytes).map_err(|e| format!("{found}: {e}"))?;
    Ok((found, anim))
}

fn resolve_bp_asset_path(bp_dir: &str, raw_path: &str) -> String {
    let norm = raw_path.trim().replace('\\', "/");
    if norm.is_empty() {
        return String::new();
    }
    if let Some(stripped) = norm.strip_prefix('/') {
        stripped.to_string()
    } else if norm.contains('/') || bp_dir.is_empty() {
        norm
    } else {
        format!("{bp_dir}/{norm}")
    }
}

fn strip_suffix_ci<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    if s.len() >= suffix.len() {
        let split_at = s.len() - suffix.len();
        if s.is_char_boundary(split_at) && s[split_at..].eq_ignore_ascii_case(suffix) {
            return Some(&s[..split_at]);
        }
    }
    None
}

fn ends_with_ci(s: &str, suffix: &str) -> bool {
    strip_suffix_ci(s, suffix).is_some()
}

fn find_asset_in_scds(
    gamedata: &Path,
    archives: &[&str],
    candidates: &[String],
) -> Option<Vec<u8>> {
    for candidate in candidates {
        for archive in archives {
            let scd_path = gamedata.join(archive);
            if let Ok((_, bytes)) = scd::read_scd_file_ci(&scd_path, candidate) {
                return Some(bytes);
            }
        }
    }
    None
}

/// Phase 41: the FA textures of the dynamic decal kinds ([`crate::decals::TEXTURES`]).
pub fn load_dynamic_decal_textures(game_dir: Option<&Path>) -> Vec<Option<RgbaImage>> {
    let Some(dir) = game_dir else {
        return vec![None; crate::decals::KINDS];
    };
    let gamedata = scd::gamedata_dir(dir);
    crate::decals::TEXTURES
        .iter()
        .map(|p| decode_first_dds(&gamedata, &["env.scd", "textures.scd"], &[p.to_string()]))
        .collect()
}

pub(crate) fn decode_first_dds(
    gamedata: &Path,
    archives: &[&str],
    candidates: &[String],
) -> Option<RgbaImage> {
    for candidate in candidates {
        for archive in archives {
            let scd_path = gamedata.join(archive);
            if let Ok((_, bytes)) = scd::read_scd_file_ci(&scd_path, candidate)
                && let Ok(img) = decode_dds_rgba8(&bytes)
            {
                return Some(img);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    fn make_synthetic_scmap_bytes(
        hm_w: u32,
        hm_h: u32,
        height_scale: f32,
        heights: &[u16],
        has_water: bool,
        water_elev: f32,
    ) -> Vec<u8> {
        assert_eq!(heights.len(), ((hm_w + 1) * (hm_h + 1)) as usize);
        let mut buf = Vec::with_capacity(2048);

        buf.extend_from_slice(b"Map\x1a");
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&0xBEEF_000Du32.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&(hm_w as f32).to_le_bytes());
        buf.extend_from_slice(&(hm_h as f32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());

        let mut preview_dds = Vec::with_capacity(132);
        preview_dds.extend_from_slice(b"DDS ");
        preview_dds.extend_from_slice(&124u32.to_le_bytes());
        preview_dds.extend_from_slice(&0x1007u32.to_le_bytes());
        preview_dds.extend_from_slice(&1u32.to_le_bytes());
        preview_dds.extend_from_slice(&1u32.to_le_bytes());
        preview_dds.extend_from_slice(&4u32.to_le_bytes());
        preview_dds.extend_from_slice(&0u32.to_le_bytes());
        preview_dds.extend_from_slice(&0u32.to_le_bytes());
        preview_dds.extend_from_slice(&[0u8; 44]);
        preview_dds.extend_from_slice(&32u32.to_le_bytes());
        preview_dds.extend_from_slice(&0x41u32.to_le_bytes());
        preview_dds.extend_from_slice(&0u32.to_le_bytes());
        preview_dds.extend_from_slice(&32u32.to_le_bytes());
        preview_dds.extend_from_slice(&0x00ff_0000u32.to_le_bytes());
        preview_dds.extend_from_slice(&0x0000_ff00u32.to_le_bytes());
        preview_dds.extend_from_slice(&0x0000_00ffu32.to_le_bytes());
        preview_dds.extend_from_slice(&0xff00_0000u32.to_le_bytes());
        preview_dds.extend_from_slice(&0x1000u32.to_le_bytes());
        preview_dds.extend_from_slice(&[0u8; 16]);
        preview_dds.extend_from_slice(&[32u8, 64, 96, 255]);
        assert_eq!(preview_dds.len(), 132);

        buf.extend_from_slice(&(preview_dds.len() as u32).to_le_bytes());
        buf.extend_from_slice(&preview_dds);

        // SCMAP minor version 60
        buf.extend_from_slice(&60i32.to_le_bytes());
        buf.extend_from_slice(&hm_w.to_le_bytes());
        buf.extend_from_slice(&hm_h.to_le_bytes());
        buf.extend_from_slice(&height_scale.to_le_bytes());
        for &h in heights {
            buf.extend_from_slice(&h.to_le_bytes());
        }

        buf.push(0u8);
        buf.extend_from_slice(b"TTerrain\0");
        buf.extend_from_slice(b"\0");
        buf.extend_from_slice(b"\0");
        buf.extend_from_slice(&0u32.to_le_bytes());

        for _ in 0..23 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }

        buf.push(u8::from(has_water));
        buf.extend_from_slice(&water_elev.to_le_bytes());
        buf.extend_from_slice(&(water_elev - 2.0).to_le_bytes());
        buf.extend_from_slice(&(water_elev - 4.0).to_le_bytes());

        buf.resize(buf.len() + 1024, 0u8);
        buf
    }

    #[test]
    fn gpu_vertex_layout_is_48_bytes() {
        assert_eq!(size_of::<GpuVertex>(), 48);
        let v = GpuVertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.25, 0.75],
            color: [1.0, 0.5, 0.25, 0.8],
        };
        assert_eq!(bytemuck::bytes_of(&v).len(), 48);
    }

    #[test]
    fn build_terrain_mesh_2x2_with_water_and_markers() {
        let heights: [u16; 9] = [2, 5, 8, 3, 6, 12, 4, 10, 20];
        let raw = make_synthetic_scmap_bytes(2, 2, 1.0, &heights, true, 5.0);

        let tmp_path = std::env::temp_dir().join(format!(
            "farust_scene_test_{}_2x2.scmap",
            std::process::id()
        ));
        std::fs::write(&tmp_path, &raw).expect("write temp scmap");
        let scmap_res = ScMap::parse_file(&tmp_path);
        let _ = std::fs::remove_file(&tmp_path);
        let scmap = scmap_res.expect("parse synthetic 2x2 scmap");

        let (verts_plain, indices_plain) = build_terrain_mesh(&scmap, None);
        assert_eq!(verts_plain.len(), 13);
        assert_eq!(indices_plain.len(), 30);

        let markers = scenario::SaveMarkers {
            start_spots: vec![scenario::Marker {
                name: "ARMY_1".to_string(),
                marker_type: "Blank Marker".to_string(),
                position: [1.0, 6.0, 1.0],
            }],
            mass_points: vec![scenario::Marker {
                name: "Mass 01".to_string(),
                marker_type: "Mass".to_string(),
                position: [0.5, 2.0, 1.5],
            }],
            hydro_points: Vec::new(),
            total_markers: 2,
        };

        let (vertices, indices) = build_terrain_mesh(&scmap, Some(&markers));
        assert!(!vertices.is_empty());
        assert!(!indices.is_empty());
        assert_eq!(indices.len() % 3, 0);

        assert_eq!(vertices.len(), verts_plain.len() + 24);
        assert_eq!(indices.len(), indices_plain.len() + 24);

        for &idx in &indices {
            assert!(
                (idx as usize) < vertices.len(),
                "index {idx} out of bounds for {} vertices",
                vertices.len()
            );
        }

        for (i, v) in vertices.iter().enumerate() {
            assert!(
                v.normal[1] > 0.0,
                "vertex {i} expected upward normal[1] > 0.0, got {:?}",
                v.normal
            );
            let len = (v.normal[0] * v.normal[0]
                + v.normal[1] * v.normal[1]
                + v.normal[2] * v.normal[2])
                .sqrt();
            assert!((len - 1.0).abs() < 1e-4, "vertex {i} normal not unit length: {len}");
        }

        let water_count = vertices
            .iter()
            .filter(|v| v.color == [0.09, 0.28, 0.46, 0.80])
            .count();
        assert_eq!(water_count, 4, "expected 4 water surface vertices");

        let spawn_count = vertices
            .iter()
            .filter(|v| v.color == [0.98, 0.80, 0.16, 0.55])
            .count();
        assert_eq!(spawn_count, 12, "expected 12 gold spawn beacon vertices");

        let mass_count = vertices
            .iter()
            .filter(|v| v.color == [0.22, 0.88, 0.38, 0.40])
            .count();
        assert_eq!(mass_count, 12, "expected 12 green mass marker vertices");
    }

    #[test]
    fn faction_color_and_path_helpers() {
        assert_eq!(pick_team_color(&["AEON".to_string()], "uel0001"), [36, 178, 95]);
        assert_eq!(pick_team_color(&[], "ual0001"), [36, 178, 95]);
        assert_eq!(pick_team_color(&[], "xas0204"), [36, 178, 95]);
        assert_eq!(pick_team_color(&[], "url0107"), [218, 44, 38]);
        assert_eq!(pick_team_color(&[], "xsl0001"), [232, 182, 48]);
        assert_eq!(pick_team_color(&[], "uel0001"), [48, 112, 230]);

        assert_eq!(
            strip_suffix_ci("units/UAL0001/UAL0001_Unit.BP", "_unit.bp"),
            Some("units/UAL0001/UAL0001")
        );
        assert_eq!(
            resolve_bp_asset_path("units/UAL0001", "/units/UAL0001/UAL0001_lod0.scm"),
            "units/UAL0001/UAL0001_lod0.scm"
        );
        assert_eq!(
            resolve_bp_asset_path("units/UAL0001", "UAL0001_lod0.scm"),
            "units/UAL0001/UAL0001_lod0.scm"
        );
    }

    #[test]
    fn strategic_icon_atlas_fallback_generation_and_uv_lookup() {
        let custom = vec![
            "icon_custom_cruiser".to_string(),
            "ICON_COMMANDER_GENERIC".to_string(),
            "  ".to_string(),
        ];
        let atlas = load_strategic_icon_atlas(None, &custom);

        assert_eq!(atlas.image.width, 16 * 32);
        assert!(atlas.image.height >= 32 && atlas.image.height % 32 == 0);
        assert_eq!(
            atlas.image.pixels.len(),
            (atlas.image.width * atlas.image.height * 4) as usize
        );

        for &name in CORE_STRATEGIC_ICONS {
            let rest_key = format!("{name}:rest");
            let sel_key = format!("{name}:sel");
            assert!(atlas.uv_rects.contains_key(&rest_key), "missing {rest_key}");
            assert!(atlas.uv_rects.contains_key(&sel_key), "missing {sel_key}");
            let r_uv = atlas.uv(name, false);
            let s_uv = atlas.uv(name, true);
            assert_ne!(r_uv, s_uv);
            for uv in [r_uv, s_uv] {
                assert!(uv[0] >= 0.0 && uv[0] < uv[2] && uv[2] <= 1.0);
                assert!(uv[1] >= 0.0 && uv[1] < uv[3] && uv[3] <= 1.0);
            }
        }

        assert!(atlas.uv_rects.contains_key("icon_custom_cruiser:rest"));
        assert!(atlas.uv_rects.contains_key("icon_custom_cruiser:sel"));
        assert_eq!(atlas.uv("unknown_unit_icon", false), atlas.default_uv);

        let tile_rest = render_fallback_strategic_icon("icon_commander_generic", false);
        let tile_sel = render_fallback_strategic_icon("icon_commander_generic", true);
        assert_ne!(tile_rest.pixels, tile_sel.pixels);
        assert_eq!(tile_rest.pixels[(15 * 32 + 15) * 4 + 3], 255);
        assert_eq!(tile_rest.pixels[3], 0);
    }

    #[test]
    fn decal_quad_construction_drapes_rotates_and_caps_top_24() {
        use crate::scmap::{ScMapWaterParams, TerrainTextures, WaterInfo};

        let hm_w = 4u32;
        let hm_h = 4u32;
        let mut heights = Vec::with_capacity(25);
        for iz in 0..=hm_h {
            for ix in 0..=hm_w {
                heights.push((ix * 4 + iz * 2) as u16);
            }
        }

        let decals = vec![
            ScMapDecal {
                id: 1,
                decal_type: 1,
                textures: vec!["/env/common/decals/crater_albedo.dds".to_string()],
                scale: [8.0, 1.0, 4.0],
                position: [32.0, 0.0, 32.0],
                rotation: [0.0, 0.0, 0.0],
                cutoff_lod: 500.0,
                near_cutoff_lod: 0.0,
            },
            ScMapDecal {
                id: 2,
                decal_type: 1,
                textures: vec!["/env/common/decals/crater_albedo.dds".to_string()],
                scale: [8.0, 1.0, 4.0],
                position: [32.0, 0.0, 32.0],
                rotation: [0.0, std::f32::consts::FRAC_PI_2, 0.0],
                cutoff_lod: 500.0,
                near_cutoff_lod: 0.0,
            },
            ScMapDecal {
                id: 3,
                decal_type: 2,
                textures: vec!["/env/common/decals/crater_normals.dds".to_string()],
                scale: [8.0, 1.0, 4.0],
                position: [32.0, 0.0, 32.0],
                rotation: [0.0, 0.0, 0.0],
                cutoff_lod: 500.0,
                near_cutoff_lod: 0.0,
            },
            ScMapDecal {
                id: 4,
                decal_type: 0,
                textures: vec!["".to_string(), "/env/common/decals/scorch.dds".to_string()],
                scale: [6.0, 1.0, 6.0],
                position: [24.0, 0.0, 24.0],
                rotation: [0.0, std::f32::consts::FRAC_PI_4, 0.0],
                cutoff_lod: 500.0,
                near_cutoff_lod: 0.0,
            },
        ];

        let mut scmap = ScMap {
            version_major: 2,
            version_minor: 60,
            map_width: 64.0,
            map_height: 64.0,
            preview_dds: Vec::new(),
            heightmap_width: hm_w,
            heightmap_height: hm_h,
            height_scale: 1.0,
            heightmap: heights,
            terrain_shader: "TTerrain".to_string(),
            background_path: String::new(),
            sky_cubemap_path: String::new(),
            cubemaps: Vec::new(),
            lighting: ScMapLighting::default(),
            water: WaterInfo {
                has_water: false,
                elevation: 0.0,
                elevation_deep: 0.0,
                elevation_abyss: 0.0,
            },
            unknown1: 0, unknown2: 0, unknown3: 0, unknown4: 0, post_heightmap_byte: 0,
            shader_raw: Vec::new(), background_raw: Vec::new(), sky_raw: Vec::new(), cubemaps_raw: Vec::new(), tail_bytes: Vec::new(),
            terrain_textures: Some(TerrainTextures {
                strata_albedo: Vec::new(),
                strata_normal: Vec::new(),
                decals,
                texturemap0_dds: Vec::new(),
                texturemap1_dds: Vec::new(),
                normal_map_dds: Vec::new(),
                water_params: ScMapWaterParams::default(),
                props: Vec::new(),
                props_tail_range: None,
                props_dirty: false,
            }),
        };

        let batches = load_map_decals(Path::new("/nonexistent_farust_test"), &scmap);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].label, "env/common/decals/crater_albedo.dds");
        assert_eq!(batches[0].vertices.len(), 18);
        assert_eq!(batches[0].indices.len(), 48);
        assert_eq!(batches[1].label, "env/common/decals/scorch.dds");
        assert_eq!(batches[1].vertices.len(), 9);
        assert_eq!(batches[1].indices.len(), 24);

        for batch in &batches {
            for &idx in &batch.indices {
                assert!((idx as usize) < batch.vertices.len());
            }
            for v in &batch.vertices {
                let expected_y = sample_scmap_height(&scmap, v.position[0], v.position[2]) + 0.08;
                assert!((v.position[1] - expected_y).abs() < 1e-4);
                let nlen = (v.normal[0] * v.normal[0]
                    + v.normal[1] * v.normal[1]
                    + v.normal[2] * v.normal[2])
                    .sqrt();
                assert!((nlen - 1.0).abs() < 1e-4);
                assert!(v.normal[1] > 0.8);
                assert!(v.normal[0] < -0.1);
            }
        }

        let rot_verts = &batches[0].vertices[9..18];
        let min_x = rot_verts.iter().map(|v| v.position[0]).fold(f32::INFINITY, f32::min);
        let max_x = rot_verts.iter().map(|v| v.position[0]).fold(f32::NEG_INFINITY, f32::max);
        let min_z = rot_verts.iter().map(|v| v.position[2]).fold(f32::INFINITY, f32::min);
        let max_z = rot_verts.iter().map(|v| v.position[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((min_x - 30.0).abs() < 1e-4 && (max_x - 34.0).abs() < 1e-4);
        assert!((min_z - 28.0).abs() < 1e-4 && (max_z - 36.0).abs() < 1e-4);

        let mut many_decals = Vec::new();
        for tex_idx in 0..26u32 {
            let count = 26 - tex_idx;
            for copy in 0..count {
                many_decals.push(ScMapDecal {
                    id: tex_idx * 100 + copy,
                    decal_type: 1,
                    textures: vec![format!("/env/decals/d_{tex_idx:02}_albedo.dds")],
                    scale: [4.0, 1.0, 4.0],
                    position: [16.0, 0.0, 16.0],
                    rotation: [0.0, 0.0, 0.0],
                    cutoff_lod: 500.0,
                    near_cutoff_lod: 0.0,
                });
            }
        }
        scmap.terrain_textures.as_mut().expect("terrain_textures").decals = many_decals;
        let capped = load_map_decals(Path::new("/nonexistent_farust_test"), &scmap);
        assert_eq!(capped.len(), 24);
        assert_eq!(capped[0].label, "env/decals/d_00_albedo.dds");
        assert_eq!(capped[23].label, "env/decals/d_23_albedo.dds");
    }

    #[test]
    fn faction_theme_returns_distinct_valid_palettes() {
        let uef = faction_theme("uef");
        let cybran = faction_theme("Cybran");
        let aeon = faction_theme("AEON");
        let seraphim = faction_theme("seraphim");
        let neutral = faction_theme("unknown");

        assert_eq!(uef.name, "UEF");
        assert_eq!(cybran.name, "Cybran");
        assert_eq!(aeon.name, "Aeon");
        assert_eq!(seraphim.name, "Seraphim");
        assert_eq!(neutral.name, "Neutral");

        let themes = [uef, cybran, aeon, seraphim, neutral];
        for (i, t) in themes.iter().enumerate() {
            for rgba in [t.primary, t.border, t.accent, t.panel_bg] {
                for c in rgba {
                    assert!(c.is_finite() && (0.0..=1.0).contains(&c));
                }
            }
            for other in themes.iter().skip(i + 1) {
                assert_ne!(t.primary, other.primary);
                assert_ne!(t.panel_bg, other.panel_bg);
            }
        }
    }

    #[test]
    fn prop_chunk_partitioning_8x8_and_aabb_bounds() {
        let make_prop = |x: f32, y: f32, z: f32| ScMapProp {
            blueprint_path: "/env/trees/pine_prop.bp".to_string(),
            position: [x, y, z],
            rot_x: [1.0, 0.0, 0.0],
            rot_y: [0.0, 1.0, 0.0],
            rot_z: [0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        };
        let props = [
            make_prop(10.0, 5.0, 12.0),
            make_prop(100.0, 8.0, 70.0),
            make_prop(112.0, 14.0, 80.0),
            make_prop(250.0, 20.0, 240.0),
        ];
        let refs: Vec<&ScMapProp> = props.iter().collect();
        let chunks = partition_prop_chunks(&refs, 256.0, 256.0);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].chunk_idx, 0);
        assert_eq!(chunks[0].bounds_min, [2.0, 1.0, 4.0]);
        assert_eq!(chunks[0].bounds_max, [18.0, 23.0, 20.0]);

        assert_eq!(chunks[1].chunk_idx, 2 * 8 + 3);
        assert_eq!(chunks[1].refs.len(), 2);
        assert_eq!(chunks[1].bounds_min, [92.0, 4.0, 62.0]);
        assert_eq!(chunks[1].bounds_max, [120.0, 32.0, 88.0]);

        assert_eq!(chunks[2].chunk_idx, 7 * 8 + 7);
        assert_eq!(chunks[2].bounds_min, [242.0, 16.0, 232.0]);
        assert_eq!(chunks[2].bounds_max, [258.0, 38.0, 248.0]);
    }

    #[test]
    fn parses_unit_display_anim_info() {
        let bp = blueprint::parse_blueprint(
            r#"UnitBlueprint {
                Display = {
                    AnimationWalk = '/units/uel0001/uel0001_a001.sca',
                    AnimationWalkRate = 1.7,
                    Mesh = { LODs = { { LODCutoff = 175 }, { LODCutoff = 400 } } },
                },
                Physics = { MaxSpeed = 1.7 },
            }"#,
        )
        .unwrap();
        let info = parse_unit_display_info(&bp);
        assert_eq!(info.walk_anim.as_deref(), Some("units/uel0001/uel0001_a001.sca"));
        assert_eq!(info.idle_anim, None);
        assert!((info.walk_rate - 1.7).abs() < 1e-6);
        assert!((info.max_speed - 1.7).abs() < 1e-6);
        assert!((info.lod0_cutoff - 175.0).abs() < 1e-6);

        let empty = parse_unit_display_info(&blueprint::parse_blueprint("UnitBlueprint { }").unwrap());
        assert_eq!(empty.walk_anim, None);
        assert_eq!(empty.walk_rate, 1.0);
        assert_eq!(empty.lod0_cutoff, 150.0);
        assert!(empty.weapons.is_empty() && empty.death_anims.is_empty());
        assert!(!empty.personal_shield && empty.bank_factor == 0.0 && !empty.winged);
    }

    #[test]
    fn parses_weapon_bones_death_clips_shield_and_air() {
        let bp = blueprint::parse_blueprint(
            r#"UnitBlueprint {
                Air = { BankFactor = 2, Winged = true, MaxAirspeed = 15 },
                Defense = { Shield = { PersonalShield = true, ShieldSize = 3 } },
                Display = {
                    AnimationBuild = '/units/urb0101/urb0101_abuild.sca',
                    AnimationDeath = {
                        { Animation = '/units/uel0106/uel0106_Adeath01.sca', AnimationRateMax = 1.25, AnimationRateMin = 0.75 },
                        { Animation = '/units/uel0106/uel0106_Adeath02.sca' },
                    },
                },
                Weapon = {
                    {
                        DamageRadius = 0,
                        Label = 'MainGun',
                        ProjectileId = '/projectiles/TDFGauss01/TDFGauss01_proj.bp',
                        RackBones = { { MuzzleBones = { 'Turret_Muzzle' }, RackBone = 'Turret_Barrel' } },
                        RackRecoilDistance = -2,
                        TurretBoneMuzzle = 'Turret_Muzzle',
                        TurretBonePitch = 'Turret_Barrel',
                        TurretBoneYaw = 'Turret',
                        TurretPitchRange = 45,
                        TurretYawSpeed = 100,
                    },
                    { DamageRadius = 2, Label = 'Arty', ProjectileId = '/projectiles/TIFMortar01/TIFMortar01_proj.bp' },
                },
            }"#,
        )
        .unwrap();
        let info = parse_unit_display_info(&bp);
        assert_eq!(info.weapons.len(), 2);
        let w = &info.weapons[0];
        assert_eq!(w.label, "MainGun");
        assert_eq!(w.yaw_bone.as_deref(), Some("Turret"));
        assert_eq!(w.pitch_bone.as_deref(), Some("Turret_Barrel"));
        assert_eq!(w.rack_bones, vec!["Turret_Barrel".to_string()]);
        assert_eq!(w.muzzle_bones, vec!["Turret_Muzzle".to_string()]);
        assert_eq!(w.recoil_distance, -2.0);
        assert_eq!(w.pitch_range_deg, 45.0);
        assert_eq!(w.yaw_speed_deg, 100.0);
        assert_eq!(w.projectile_id.as_deref(), Some("projectiles/tdfgauss01/tdfgauss01_proj.bp"));
        assert!(!w.splash && info.weapons[1].splash);
        assert_eq!(info.weapons[1].pitch_range_deg, 90.0);

        assert_eq!(info.death_anims.len(), 2);
        assert_eq!(info.death_anims[0].path, "units/uel0106/uel0106_Adeath01.sca");
        assert!((info.death_anims[0].rate - 1.0).abs() < 1e-6);
        assert_eq!(info.death_anims[1].rate, 1.0);
        assert_eq!(info.build_anim.as_deref(), Some("units/urb0101/urb0101_abuild.sca"));
        assert_eq!(info.open_anim, None);
        assert!(info.personal_shield);
        assert_eq!(info.bank_factor, 2.0);
        assert!(info.winged);
        assert_eq!(info.max_airspeed, 15.0);

        let proj = parse_projectile_blueprint(
            "ProjectileBlueprint { Categories = { 'UEF', 'MISSILE' }, Display = { UniformScale = 0.15 } }",
        )
        .unwrap();
        let p = parse_projectile_display_info(&proj);
        assert!(p.missile && (p.uniform_scale - 0.15).abs() < 1e-6);
    }
}
