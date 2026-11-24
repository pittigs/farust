//! Scene builders turning Forged Alliance unit meshes/textures and `.scmap` terrains
//! into GPU-ready vertex/index/texture buffers.

use std::path::Path;
use bytemuck::{Pod, Zeroable};

use crate::blueprint::{self, Value};
use crate::camera::OrbitCamera;
use crate::dds_decode::{RgbaImage, decode_dds_rgba8};
use crate::scd;
use crate::scenario;
use crate::scm::ScmMesh;
use crate::scmap::ScMap;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// rgb = vertex color multiplier, a = specular intensity (0.0..=1.0).
    pub color: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct LoadedScene {
    pub title: String,
    pub vertices: Vec<GpuVertex>,
    pub indices: Vec<u32>,
    pub albedo: RgbaImage,
    pub initial_camera: OrbitCamera,
    pub clear_color: [f64; 4],
}

pub fn load_unit_scene(game_dir: &Path, unit_id: &str, aspect: f32) -> Result<LoadedScene, String> {
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
            let spec_strength =
                ((spec_img.pixels[i * 4] as u16 * 180 / 255) + 40).min(255) as u8;
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

    let vertices: Vec<GpuVertex> = mesh
        .vertices
        .iter()
        .map(|v| GpuVertex {
            position: [
                v.position[0] * scale,
                v.position[1] * scale,
                v.position[2] * scale,
            ],
            normal: normalize_or_up(v.normal),
            uv: v.uv0,
            color: [1.0, 1.0, 1.0, 1.0],
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
    let title = format!("{} — {}", unit_id.to_uppercase(), desc);

    Ok(LoadedScene {
        title,
        vertices,
        indices,
        albedo,
        initial_camera,
        clear_color: [0.055, 0.070, 0.095, 1.0],
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

    let (vertices, indices) = build_terrain_mesh(&scmap, markers.as_ref());

    let albedo = RgbaImage {
        width: 1,
        height: 1,
        pixels: vec![255, 255, 255, 255],
    };

    let (_, _, avg_h) = scmap.height_stats();
    let focus_y = if scmap.water.has_water {
        scmap.water.elevation
    } else {
        avg_h
    };
    let initial_camera = OrbitCamera::for_map(scmap.map_width, scmap.map_height, focus_y, aspect);

    let title = if meta.name.trim().is_empty() {
        meta.id.clone()
    } else {
        format!("{} ({})", meta.name, meta.id)
    };

    Ok(LoadedScene {
        title,
        vertices,
        indices,
        albedo,
        initial_camera,
        clear_color: [0.42, 0.58, 0.78, 1.0],
    })
}

pub fn build_terrain_mesh(
    scmap: &ScMap,
    markers: Option<&scenario::SaveMarkers>,
) -> (Vec<GpuVertex>, Vec<u32>) {
    let hm_w = scmap.heightmap_width;
    let hm_h = scmap.heightmap_height;
    if hm_w == 0 || hm_h == 0 {
        return (Vec::new(), Vec::new());
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

    let mut vertices = Vec::with_capacity((nx * nz) as usize + 64);
    let mut indices = Vec::with_capacity((cells_x * cells_z * 6) as usize + 64);

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

            let color = terrain_vertex_color(
                y,
                normal,
                scmap.water.has_water,
                scmap.water.elevation,
                deep_floor,
                land_base,
                max_h,
            );

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

    if scmap.water.has_water {
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
        let base_idx = vertices.len() as u32;

        vertices.extend_from_slice(&[
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
        indices.extend_from_slice(&[
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
                &mut vertices,
                &mut indices,
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
                &mut vertices,
                &mut indices,
                [mass.position[0], base_y, mass.position[2]],
                mass_radius,
                mass_radius * 2.8,
                mass_color,
            );
        }
    }

    (vertices, indices)
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

fn decode_first_dds(
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
}
