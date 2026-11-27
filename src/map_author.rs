//! Phase 50: Map authoring tools — `.scmap` v2.60 round-trip verification,
//! scripted map editing (`map-edit`), and an interactive GPU map editor
//! (`map-editor`) built on [`crate::viewer`] and [`crate::ui_kit`].
//!
//! All writes target user map folders (`$XDG_DATA_HOME/farust/maps/<NEW_ID>/`
//! by default) and never modify the read-only FA game install.
//!
//! Note: heightmap edits update the `.scmap` elevation grid (and geometric
//! terrain normals recomputed by FARust's renderer), while the embedded DXT5
//! normal-map DDS in the `.scmap` tail is preserved verbatim.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Modifiers, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::play::pick_terrain_point;
use crate::scenario::{self, MapSource, Marker, SaveMarkers, ScenarioMeta};
use crate::scd;
use crate::scene::{
    self, GpuVertex, LoadedScene, build_map_scene_from_parts, build_terrain_and_overlay_meshes_opts,
    load_map_props,
};
use crate::scmap::{BrushMode, ScMap, ScMapProp};
use crate::ui_kit::{
    self, Align, Canvas, Fonts, ImageStore, Rect, UiRenderer, WState, body, bold, hit_test, theme,
};
use crate::viewer::{self, DEPTH_FORMAT, GpuSceneResources, MSAA_SAMPLES, create_attachment};

const DEFAULT_SCRIPT_LUA: &str = "local ScenarioUtils = import('/lua/sim/ScenarioUtilities.lua')\n\nfunction OnPopulate()\n\tScenarioUtils.InitializeArmies()\nend\n\nfunction OnStart(self)\nend\n";

const DEFAULT_PROP_BLUEPRINT: &str = "/env/evergreen/props/trees/pine06_prop.bp";

// ---------------------------------------------------------------------------
// Round-trip verification
// ---------------------------------------------------------------------------

/// Result of verifying `write(parse(bytes)) == bytes` and `parse(write(map))`.
#[derive(Debug, Clone)]
pub struct RoundtripReport {
    pub id: String,
    pub byte_len: usize,
    pub byte_identical: bool,
    pub first_diff: Option<(usize, u8, u8)>,
    pub reparse_ok: bool,
    pub version_minor: i32,
    pub heightmap_size: (u32, u32),
    pub prop_count: usize,
}

/// Verifies byte-exact `.scmap` round-trip (`write(parse(b)) == b`) and re-parse identity.
pub fn roundtrip_scmap_bytes(id: &str, bytes: &[u8]) -> Result<RoundtripReport, String> {
    let parsed = ScMap::parse(bytes).map_err(|e| format!("{id}: parse failed: {e}"))?;
    let written = parsed.write();
    let first_diff = if written == bytes {
        None
    } else {
        let min_len = bytes.len().min(written.len());
        let diff_pos = (0..min_len).find(|&i| bytes[i] != written[i]);
        match diff_pos {
            Some(i) => Some((i, bytes[i], written[i])),
            None => Some((
                min_len,
                bytes.get(min_len).copied().unwrap_or(0),
                written.get(min_len).copied().unwrap_or(0),
            )),
        }
    };
    let reparsed = ScMap::parse(&written).map_err(|e| format!("{id}: reparse failed: {e}"))?;
    let prop_count = parsed
        .terrain_textures
        .as_ref()
        .map(|t| t.props.len())
        .unwrap_or(0);
    let reparsed_props = reparsed
        .terrain_textures
        .as_ref()
        .map(|t| t.props.len())
        .unwrap_or(0);
    let reparse_ok = reparsed.version_major == parsed.version_major
        && reparsed.version_minor == parsed.version_minor
        && reparsed.heightmap_width == parsed.heightmap_width
        && reparsed.heightmap_height == parsed.heightmap_height
        && reparsed.heightmap == parsed.heightmap
        && reparsed.water.has_water == parsed.water.has_water
        && reparsed.water.elevation.to_bits() == parsed.water.elevation.to_bits()
        && reparsed_props == prop_count;

    Ok(RoundtripReport {
        id: id.to_string(),
        byte_len: bytes.len(),
        byte_identical: first_diff.is_none(),
        first_diff,
        reparse_ok,
        version_minor: parsed.version_minor,
        heightmap_size: (parsed.heightmap_width, parsed.heightmap_height),
        prop_count,
    })
}

pub fn cmd_map_roundtrip(args: &[String]) -> Result<(), String> {
    let all = args.iter().any(|a| a == "--all");
    let include_user = args.iter().any(|a| a == "--include-user");
    let positionals: Vec<&str> = args
        .iter()
        .skip(2)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect();

    if all {
        let game_dir = positionals
            .first()
            .map(PathBuf::from)
            .unwrap_or_else(scd::default_game_dir);
        let maps = scenario::list_maps(&game_dir)?;
        let selected: Vec<_> = maps
            .into_iter()
            .filter(|m| include_user || m.source == MapSource::Stock)
            .collect();
        if selected.is_empty() {
            return Err(format!("no maps found in {}", game_dir.display()));
        }
        let mut identical = 0usize;
        let mut reparse_ok = 0usize;
        let mut total_bytes = 0usize;
        let mut failures = Vec::new();
        for m in &selected {
            let Some(scmap_path) = m.scmap_path.as_deref() else {
                failures.push(format!("{}: missing .scmap file", m.id));
                continue;
            };
            let bytes = std::fs::read(scmap_path)
                .map_err(|e| format!("{}: failed to read {}: {e}", m.id, scmap_path.display()))?;
            match roundtrip_scmap_bytes(&m.id, &bytes) {
                Ok(rep) => {
                    total_bytes += rep.byte_len;
                    if rep.byte_identical {
                        identical += 1;
                    } else if let Some((off, exp, got)) = rep.first_diff {
                        failures.push(format!(
                            "{}: mismatch at byte {off} (orig=0x{exp:02x}, written=0x{got:02x}, orig_len={})",
                            rep.id, rep.byte_len
                        ));
                    }
                    if rep.reparse_ok {
                        reparse_ok += 1;
                    } else {
                        failures.push(format!("{}: reparse field mismatch", rep.id));
                    }
                }
                Err(e) => failures.push(e),
            }
        }
        let scope = if include_user { "maps" } else { "stock maps" };
        println!(
            "map-roundtrip: {identical}/{} {scope} byte-identical ({reparse_ok}/{} reparse ok, {:.2} MiB)",
            selected.len(),
            selected.len(),
            total_bytes as f64 / (1024.0 * 1024.0)
        );
        if !failures.is_empty() {
            return Err(format!(
                "{} round-trip failure(s):\n{}",
                failures.len(),
                failures.join("\n")
            ));
        }
        return Ok(());
    }

    let Some(&map_name) = positionals.first() else {
        return Err("usage: farust map-roundtrip [--all | <MAP>] [GAME_DIR]".to_string());
    };
    let game_dir = positionals
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(scd::default_game_dir);
    let m = scenario::find_map(&game_dir, map_name)?;
    let scmap_path = m
        .scmap_path
        .as_deref()
        .ok_or_else(|| format!("map '{}' has no .scmap file", m.id))?;
    let bytes = std::fs::read(scmap_path)
        .map_err(|e| format!("failed to read {}: {e}", scmap_path.display()))?;
    let rep = roundtrip_scmap_bytes(&m.id, &bytes)?;
    println!(
        "{}: {} ({} bytes, v2.{}, heightmap {}x{}, props {}, reparse={})",
        rep.id,
        if rep.byte_identical { "byte-identical" } else { "MISMATCH" },
        rep.byte_len,
        rep.version_minor,
        rep.heightmap_size.0,
        rep.heightmap_size.1,
        rep.prop_count,
        if rep.reparse_ok { "ok" } else { "FAIL" }
    );
    if !rep.byte_identical {
        if let Some((off, exp, got)) = rep.first_diff {
            return Err(format!(
                "{}: first differing byte at offset {off} (orig=0x{exp:02x}, written=0x{got:02x})",
                rep.id
            ));
        }
        return Err(format!("{}: output differed from input", rep.id));
    }
    if !rep.reparse_ok {
        return Err(format!("{}: reparse did not match original fields", rep.id));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Editable map document
// ---------------------------------------------------------------------------

/// In-memory editable FA map bundle (`.scmap` + `_scenario.lua` + `_save.lua` + `_script.lua`).
#[derive(Debug, Clone)]
pub struct MapDocument {
    pub src_id: String,
    pub meta: ScenarioMeta,
    pub scmap: ScMap,
    pub markers: SaveMarkers,
    pub scenario_text: String,
    pub save_text: String,
    pub script_text: Option<String>,
    pub prop_palette: Vec<String>,
}

impl MapDocument {
    pub fn load(game_dir: &Path, map_name: &str) -> Result<Self, String> {
        let map_files = scenario::find_map(game_dir, map_name)?;
        let scenario_bytes = std::fs::read(&map_files.scenario_path)
            .map_err(|e| format!("failed to read {}: {e}", map_files.scenario_path.display()))?;
        let scenario_text = String::from_utf8_lossy(&scenario_bytes).into_owned();
        let meta = scenario::parse_scenario(&scenario_text, &map_files.id)?;

        let scmap_path = map_files
            .scmap_path
            .as_deref()
            .ok_or_else(|| format!("map '{}' has no .scmap file", map_files.id))?;
        let scmap = ScMap::parse_file(scmap_path)?;

        let save_path = map_files
            .save_path
            .as_deref()
            .ok_or_else(|| format!("map '{}' has no _save.lua file", map_files.id))?;
        let save_bytes = std::fs::read(save_path)
            .map_err(|e| format!("failed to read {}: {e}", save_path.display()))?;
        let save_text = String::from_utf8_lossy(&save_bytes).into_owned();
        let markers = scenario::parse_save_markers(&save_text, &meta.armies)?;

        let script_text = find_script_file(&map_files.dir)
            .and_then(|p| std::fs::read(p).ok())
            .map(|b| String::from_utf8_lossy(&b).into_owned());

        let prop_palette = build_prop_palette(&scmap);
        Ok(Self {
            src_id: map_files.id,
            meta,
            scmap,
            markers,
            scenario_text,
            save_text,
            script_text,
            prop_palette,
        })
    }

    /// Applies a terrain heightmap brush and re-snaps affected markers & props to terrain height.
    pub fn apply_brush(
        &mut self,
        x: f32,
        z: f32,
        radius: f32,
        strength: f32,
        mode: BrushMode,
    ) -> usize {
        let changed = self.scmap.apply_brush(x, z, radius, strength, mode);
        if changed > 0 {
            let snap_r = radius + 4.0;
            let snap_r2 = snap_r * snap_r;
            for list in [
                &mut self.markers.start_spots,
                &mut self.markers.mass_points,
                &mut self.markers.hydro_points,
            ] {
                for m in list.iter_mut() {
                    let dx = m.position[0] - x;
                    let dz = m.position[2] - z;
                    if dx * dx + dz * dz <= snap_r2 {
                        m.position[1] = self.scmap.sample_height(m.position[0], m.position[2]);
                    }
                }
            }
            // Re-snap any props sitting inside the sculpted area so they don't float or sink.
            if let Some(tt) = self.scmap.terrain_textures.as_ref()
                && tt.props_tail_range.is_some()
            {
                let mut moved: Vec<(usize, f32)> = Vec::new();
                for (i, p) in tt.props.iter().enumerate() {
                    let dx = p.position[0] - x;
                    let dz = p.position[2] - z;
                    if dx * dx + dz * dz <= snap_r2 {
                        let ny = self.scmap.sample_height(p.position[0], p.position[2]);
                        if (ny - p.position[1]).abs() > 1e-4 {
                            moved.push((i, ny));
                        }
                    }
                }
                if !moved.is_empty()
                    && let Some(tt) = self.scmap.terrain_textures.as_mut()
                {
                    for (i, ny) in moved {
                        tt.props[i].position[1] = ny;
                    }
                    tt.props_dirty = true;
                }
            }
        }
        changed
    }

    pub fn add_mass(&mut self, x: f32, z: f32) -> String {
        let (x, z) = self.clamp_xz(x, z);
        let y = self.scmap.sample_height(x, z);
        let name = next_marker_name("Mass", &self.markers.mass_points);
        self.markers.mass_points.push(Marker {
            name: name.clone(),
            marker_type: "Mass".to_string(),
            position: [x, y, z],
        });
        self.markers.total_markers += 1;
        name
    }

    pub fn remove_mass_near(&mut self, x: f32, z: f32, max_dist: f32) -> Option<Marker> {
        let removed = remove_marker_near(&mut self.markers.mass_points, x, z, max_dist)?;
        self.markers.total_markers = self.markers.total_markers.saturating_sub(1);
        Some(removed)
    }

    pub fn add_hydro(&mut self, x: f32, z: f32) -> String {
        let (x, z) = self.clamp_xz(x, z);
        let y = self.scmap.sample_height(x, z);
        let name = next_marker_name("Hydrocarbon", &self.markers.hydro_points);
        self.markers.hydro_points.push(Marker {
            name: name.clone(),
            marker_type: "Hydrocarbon".to_string(),
            position: [x, y, z],
        });
        self.markers.total_markers += 1;
        name
    }

    pub fn remove_hydro_near(&mut self, x: f32, z: f32, max_dist: f32) -> Option<Marker> {
        let removed = remove_marker_near(&mut self.markers.hydro_points, x, z, max_dist)?;
        self.markers.total_markers = self.markers.total_markers.saturating_sub(1);
        Some(removed)
    }

    /// Moves a 1-based start spot (`ARMY_n`, or index `n-1`) to `(x, z)`.
    pub fn move_start(&mut self, army_num_1based: usize, x: f32, z: f32) -> Result<String, String> {
        if army_num_1based == 0 {
            return Err("start spot index is 1-based (e.g. 1 for ARMY_1)".to_string());
        }
        let (x, z) = self.clamp_xz(x, z);
        let y = self.scmap.sample_height(x, z);
        let target_name = format!("ARMY_{army_num_1based}");
        let idx = self
            .markers
            .start_spots
            .iter()
            .position(|m| m.name.eq_ignore_ascii_case(&target_name))
            .or_else(|| {
                (army_num_1based - 1 < self.markers.start_spots.len())
                    .then_some(army_num_1based - 1)
            })
            .ok_or_else(|| {
                format!(
                    "start spot {army_num_1based} ({target_name}) not found (map has {} start spots)",
                    self.markers.start_spots.len()
                )
            })?;
        self.markers.start_spots[idx].position = [x, y, z];
        Ok(self.markers.start_spots[idx].name.clone())
    }

    pub fn add_prop(&mut self, blueprint_path: &str, x: f32, z: f32, scale: f32) -> bool {
        let (x, z) = self.clamp_xz(x, z);
        let y = self.scmap.sample_height(x, z);
        let s = scale.clamp(0.1, 20.0);
        let prop = ScMapProp {
            blueprint_path: blueprint_path.to_string(),
            position: [x, y, z],
            rot_x: [1.0, 0.0, 0.0],
            rot_y: [0.0, 1.0, 0.0],
            rot_z: [0.0, 0.0, 1.0],
            scale: [s, s, s],
        };
        let ok = self.scmap.add_prop(prop);
        if ok && !self.prop_palette.iter().any(|p| p.eq_ignore_ascii_case(blueprint_path)) {
            self.prop_palette.push(blueprint_path.to_string());
        }
        ok
    }

    pub fn remove_prop_near(&mut self, x: f32, z: f32, max_dist: f32) -> Option<ScMapProp> {
        self.scmap.remove_prop_near(x, z, max_dist)
    }

    fn clamp_xz(&self, x: f32, z: f32) -> (f32, f32) {
        let mw = self.scmap.map_width.max(1.0);
        let mh = self.scmap.map_height.max(1.0);
        (x.clamp(0.0, mw), z.clamp(0.0, mh))
    }

    /// Writes a complete FA map folder (`<dest_root>/<new_id>/`) with `.scmap`,
    /// `_scenario.lua`, `_save.lua`, and `_script.lua`. Never writes inside `game_dir`.
    pub fn save_as(
        &self,
        dest_root: &Path,
        new_id: &str,
        display_name: Option<&str>,
        game_dir: Option<&Path>,
    ) -> Result<PathBuf, String> {
        validate_map_id(new_id)?;
        let out_dir = dest_root.join(new_id);
        if let Some(gd) = game_dir {
            let gd_norm = std::fs::canonicalize(gd).unwrap_or_else(|_| gd.to_path_buf());
            let dest_norm = std::fs::canonicalize(dest_root).unwrap_or_else(|_| dest_root.to_path_buf());
            if dest_norm.starts_with(&gd_norm) {
                return Err(format!(
                    "refusing to write edited map inside read-only game install '{}'",
                    gd.display()
                ));
            }
        }
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("failed to create '{}': {e}", out_dir.display()))?;

        // 1. Write .scmap
        let scmap_out = out_dir.join(format!("{new_id}.scmap"));
        self.scmap.write_file(&scmap_out)?;

        // 2. Write _scenario.lua
        let scenario_out_text = scenario::rewrite_scenario_for_clone(
            &self.scenario_text,
            &self.src_id,
            new_id,
            display_name,
        );
        let scenario_out = out_dir.join(format!("{new_id}_scenario.lua"));
        std::fs::write(&scenario_out, scenario_out_text)
            .map_err(|e| format!("failed to write '{}': {e}", scenario_out.display()))?;

        // 3. Write _save.lua
        let save_id_rewritten = scenario::rewrite_map_id(&self.save_text, &self.src_id, new_id);
        let save_out_text = scenario::rewrite_save_markers(
            &save_id_rewritten,
            &self.meta.armies,
            &self.markers.start_spots,
            &self.markers.mass_points,
            &self.markers.hydro_points,
        )?;
        let save_out = out_dir.join(format!("{new_id}_save.lua"));
        std::fs::write(&save_out, save_out_text)
            .map_err(|e| format!("failed to write '{}': {e}", save_out.display()))?;

        // 4. Write _script.lua
        let script_out_text = match &self.script_text {
            Some(s) => scenario::rewrite_map_id(s, &self.src_id, new_id),
            None => DEFAULT_SCRIPT_LUA.to_string(),
        };
        let script_out = out_dir.join(format!("{new_id}_script.lua"));
        std::fs::write(&script_out, script_out_text)
            .map_err(|e| format!("failed to write '{}': {e}", script_out.display()))?;

        Ok(out_dir)
    }
}

pub fn validate_map_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.len() > 64 {
        return Err("map id must be 1..=64 ASCII characters".to_string());
    }
    if !id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(format!(
            "invalid map id '{id}': only ASCII letters, digits, '_' and '-' are allowed"
        ));
    }
    Ok(())
}

fn find_script_file(dir: &Path) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut matches: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.to_ascii_lowercase().ends_with("_script.lua"))
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

fn build_prop_palette(scmap: &ScMap) -> Vec<String> {
    let mut uniq = BTreeSet::new();
    if let Some(tt) = &scmap.terrain_textures {
        for p in &tt.props {
            let trimmed = p.blueprint_path.trim();
            if !trimmed.is_empty() {
                uniq.insert(trimmed.to_string());
            }
        }
    }
    let mut out: Vec<String> = uniq.into_iter().collect();
    if out.is_empty() {
        out.push(DEFAULT_PROP_BLUEPRINT.to_string());
    }
    out
}

fn next_marker_name(prefix: &str, existing: &[Marker]) -> String {
    let mut n = existing.len() + 1;
    loop {
        let cand = format!("{prefix} {n:02}");
        if !existing.iter().any(|m| m.name.eq_ignore_ascii_case(&cand)) {
            return cand;
        }
        n += 1;
    }
}

fn remove_marker_near(list: &mut Vec<Marker>, x: f32, z: f32, max_dist: f32) -> Option<Marker> {
    let max_d2 = max_dist.max(0.0) * max_dist.max(0.0);
    let (idx, _) = list
        .iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let dx = m.position[0] - x;
            let dz = m.position[2] - z;
            let d2 = dx * dx + dz * dz;
            (d2 <= max_d2).then_some((i, d2))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    Some(list.remove(idx))
}

fn find_marker_near(list: &[Marker], x: f32, z: f32, max_dist: f32) -> Option<usize> {
    let max_d2 = max_dist.max(0.0) * max_dist.max(0.0);
    list.iter()
        .enumerate()
        .filter_map(|(i, m)| {
            let dx = m.position[0] - x;
            let dz = m.position[2] - z;
            let d2 = dx * dx + dz * dz;
            (d2 <= max_d2).then_some((i, d2))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

// ---------------------------------------------------------------------------
// Scripted CLI: `farust map-edit`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum EditOp {
    Brush {
        x: f32,
        z: f32,
        r: f32,
        s: f32,
        mode: BrushMode,
    },
    AddMass(f32, f32),
    DelMass(f32, f32, f32),
    AddHydro(f32, f32),
    DelHydro(f32, f32, f32),
    MoveStart(usize, f32, f32),
    AddProp(String, f32, f32, f32),
    DelProp(f32, f32, f32),
}

fn parse_f32_csv(spec: &str, expected: usize, flag: &str) -> Result<Vec<f32>, String> {
    let parts: Vec<&str> = spec.split(',').collect();
    if parts.len() != expected {
        return Err(format!("{flag} expects {expected} comma-separated numbers, got '{spec}'"));
    }
    let mut vals = Vec::with_capacity(expected);
    for p in parts {
        let v: f32 = p
            .trim()
            .parse()
            .map_err(|_| format!("invalid number '{p}' in {flag} '{spec}'"))?;
        if !v.is_finite() {
            return Err(format!("non-finite number '{p}' in {flag}"));
        }
        vals.push(v);
    }
    Ok(vals)
}

pub fn cmd_map_edit(args: &[String]) -> Result<(), String> {
    if args.len() < 4 {
        return Err(
            "usage: farust map-edit <SRC_MAP> <NEW_ID> [GAME_DIR] [--name TITLE] [--out-dir DIR] \
             [--raise X,Z,R,H] [--lower X,Z,R,H] [--smooth X,Z,R,S] [--add-mass X,Z] \
             [--del-mass X,Z,R] [--add-hydro X,Z] [--del-hydro X,Z,R] [--move-start N,X,Z] \
             [--add-prop BP,X,Z[,SCALE]] [--del-prop X,Z,R]"
                .to_string(),
        );
    }
    let src_map = &args[2];
    let new_id = &args[3];
    validate_map_id(new_id)?;

    let mut game_dir: Option<PathBuf> = None;
    let mut out_dir_root: Option<PathBuf> = None;
    let mut display_name: Option<String> = None;
    let mut ops: Vec<EditOp> = Vec::new();

    let mut i = 4usize;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: usize, name: &str| -> Result<&str, String> {
            args.get(idx + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("missing value after {name}"))
        };
        match flag {
            "--name" => {
                display_name = Some(next_val(i, flag)?.to_string());
                i += 2;
            }
            "--out-dir" => {
                out_dir_root = Some(PathBuf::from(next_val(i, flag)?));
                i += 2;
            }
            "--raise" | "--lower" | "--smooth" => {
                let v = parse_f32_csv(next_val(i, flag)?, 4, flag)?;
                let mode = match flag {
                    "--raise" => BrushMode::Raise,
                    "--lower" => BrushMode::Lower,
                    _ => BrushMode::Smooth,
                };
                ops.push(EditOp::Brush { x: v[0], z: v[1], r: v[2], s: v[3], mode });
                i += 2;
            }
            "--add-mass" => {
                let v = parse_f32_csv(next_val(i, flag)?, 2, flag)?;
                ops.push(EditOp::AddMass(v[0], v[1]));
                i += 2;
            }
            "--del-mass" => {
                let v = parse_f32_csv(next_val(i, flag)?, 3, flag)?;
                ops.push(EditOp::DelMass(v[0], v[1], v[2]));
                i += 2;
            }
            "--add-hydro" => {
                let v = parse_f32_csv(next_val(i, flag)?, 2, flag)?;
                ops.push(EditOp::AddHydro(v[0], v[1]));
                i += 2;
            }
            "--del-hydro" => {
                let v = parse_f32_csv(next_val(i, flag)?, 3, flag)?;
                ops.push(EditOp::DelHydro(v[0], v[1], v[2]));
                i += 2;
            }
            "--move-start" => {
                let raw = next_val(i, flag)?;
                let parts: Vec<&str> = raw.split(',').collect();
                if parts.len() != 3 {
                    return Err(format!("--move-start expects N,X,Z, got '{raw}'"));
                }
                let n: usize = parts[0]
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid army index '{}' in --move-start", parts[0]))?;
                let x: f32 = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid X '{}' in --move-start", parts[1]))?;
                let z: f32 = parts[2]
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid Z '{}' in --move-start", parts[2]))?;
                ops.push(EditOp::MoveStart(n, x, z));
                i += 2;
            }
            "--add-prop" => {
                let raw = next_val(i, flag)?;
                let parts: Vec<&str> = raw.split(',').collect();
                if parts.len() != 3 && parts.len() != 4 {
                    return Err(format!("--add-prop expects BP,X,Z[,SCALE], got '{raw}'"));
                }
                let bp = parts[0].trim().to_string();
                let x: f32 = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid X '{}' in --add-prop", parts[1]))?;
                let z: f32 = parts[2]
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid Z '{}' in --add-prop", parts[2]))?;
                let scale: f32 = if parts.len() == 4 {
                    parts[3]
                        .trim()
                        .parse()
                        .map_err(|_| format!("invalid scale '{}' in --add-prop", parts[3]))?
                } else {
                    1.0
                };
                ops.push(EditOp::AddProp(bp, x, z, scale));
                i += 2;
            }
            "--del-prop" => {
                let v = parse_f32_csv(next_val(i, flag)?, 3, flag)?;
                ops.push(EditOp::DelProp(v[0], v[1], v[2]));
                i += 2;
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown map-edit flag '{other}'"));
            }
            pos => {
                if game_dir.is_none() {
                    game_dir = Some(PathBuf::from(pos));
                } else {
                    return Err(format!("unexpected extra argument '{pos}'"));
                }
                i += 1;
            }
        }
    }

    let game_dir = game_dir.unwrap_or_else(scd::default_game_dir);
    let dest_root = out_dir_root.unwrap_or_else(scenario::farust_maps_dir);
    let mut doc = MapDocument::load(&game_dir, src_map)?;

    let mut samples_changed = 0usize;
    let mut mass_added = 0usize;
    let mut mass_removed = 0usize;
    let mut hydro_added = 0usize;
    let mut hydro_removed = 0usize;
    let mut starts_moved = 0usize;
    let mut props_added = 0usize;
    let mut props_removed = 0usize;

    for op in ops {
        match op {
            EditOp::Brush { x, z, r, s, mode } => {
                samples_changed += doc.apply_brush(x, z, r, s, mode);
            }
            EditOp::AddMass(x, z) => {
                let _ = doc.add_mass(x, z);
                mass_added += 1;
            }
            EditOp::DelMass(x, z, r) => {
                if doc.remove_mass_near(x, z, r).is_some() {
                    mass_removed += 1;
                }
            }
            EditOp::AddHydro(x, z) => {
                let _ = doc.add_hydro(x, z);
                hydro_added += 1;
            }
            EditOp::DelHydro(x, z, r) => {
                if doc.remove_hydro_near(x, z, r).is_some() {
                    hydro_removed += 1;
                }
            }
            EditOp::MoveStart(n, x, z) => {
                let _ = doc.move_start(n, x, z)?;
                starts_moved += 1;
            }
            EditOp::AddProp(bp, x, z, scale) => {
                if !doc.add_prop(&bp, x, z, scale) {
                    return Err("map has no editable prop section in .scmap".to_string());
                }
                props_added += 1;
            }
            EditOp::DelProp(x, z, r) => {
                if doc.remove_prop_near(x, z, r).is_some() {
                    props_removed += 1;
                }
            }
        }
    }

    let out_dir = doc.save_as(&dest_root, new_id, display_name.as_deref(), Some(&game_dir))?;
    println!(
        "map-edit: '{}' -> '{}' ({})",
        doc.src_id,
        new_id,
        out_dir.display()
    );
    println!(
        "  terrain samples edited: {samples_changed} | mass: {} (+{mass_added}/-{mass_removed}) | hydro: {} (+{hydro_added}/-{hydro_removed}) | starts moved: {starts_moved} | props: {} (+{props_added}/-{props_removed})",
        doc.markers.mass_points.len(),
        doc.markers.hydro_points.len(),
        doc.scmap
            .terrain_textures
            .as_ref()
            .map(|t| t.props.len())
            .unwrap_or(0)
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive GPU map editor (`farust map-editor`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorTool {
    Brush,
    Smooth,
    Mass,
    Hydro,
    Start,
    Prop,
}

impl EditorTool {
    pub const ALL: [EditorTool; 6] = [
        EditorTool::Brush,
        EditorTool::Smooth,
        EditorTool::Mass,
        EditorTool::Hydro,
        EditorTool::Start,
        EditorTool::Prop,
    ];

    pub fn label(self) -> &'static str {
        match self {
            EditorTool::Brush => "1:Brush",
            EditorTool::Smooth => "2:Smooth",
            EditorTool::Mass => "3:Mass",
            EditorTool::Hydro => "4:Hydro",
            EditorTool::Start => "5:Start",
            EditorTool::Prop => "6:Prop",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorAction {
    SelectTool(EditorTool),
    RadiusDown,
    RadiusUp,
    StrengthDown,
    StrengthUp,
    CycleStart,
    CycleProp,
    FocusSaveId,
    Save,
}

pub struct EditorState {
    pub doc: MapDocument,
    pub tool: EditorTool,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub active_start_idx: usize,
    pub active_prop_idx: usize,
    pub save_id: String,
    pub save_id_focused: bool,
    pub status: String,
    pub status_error: bool,
    pub hover_world: Option<[f32; 3]>,
    pub dragging_mass: Option<usize>,
    pub terrain_dirty: bool,
    pub overlay_dirty: bool,
    pub props_dirty: bool,
    pub unsaved_changes: usize,
}

impl EditorState {
    pub fn new(doc: MapDocument, default_save_id: Option<String>) -> Self {
        let save_id = default_save_id.unwrap_or_else(|| format!("{}_EDIT", doc.src_id));
        let status = format!(
            "Loaded {} ({}x{}) - LMB/RMB edit, MMB orbit, WASD pan, Wheel zoom, Ctrl+S save",
            doc.src_id, doc.scmap.map_width as u32, doc.scmap.map_height as u32
        );
        Self {
            doc,
            tool: EditorTool::Brush,
            brush_radius: 24.0,
            brush_strength: 1.5,
            active_start_idx: 0,
            active_prop_idx: 0,
            save_id,
            save_id_focused: false,
            status,
            status_error: false,
            hover_world: None,
            dragging_mass: None,
            terrain_dirty: false,
            overlay_dirty: false,
            props_dirty: false,
            unsaved_changes: 0,
        }
    }

    pub fn active_start_label(&self) -> &str {
        self.doc
            .markers
            .start_spots
            .get(self.active_start_idx)
            .map(|m| m.name.as_str())
            .unwrap_or("ARMY_1")
    }

    pub fn active_prop_short(&self) -> &str {
        let full = self
            .doc
            .prop_palette
            .get(self.active_prop_idx)
            .map(String::as_str)
            .unwrap_or(DEFAULT_PROP_BLUEPRINT);
        full.rsplit('/').next().unwrap_or(full)
    }

    pub fn apply_action(&mut self, action: EditorAction, dest_root: &Path, game_dir: &Path) {
        self.save_id_focused = matches!(action, EditorAction::FocusSaveId);
        match action {
            EditorAction::SelectTool(t) => {
                self.tool = t;
                self.dragging_mass = None;
                self.status_error = false;
                self.status = match t {
                    EditorTool::Brush => "Brush: LMB raise, RMB lower, [/] radius, -/= strength".into(),
                    EditorTool::Smooth => "Smooth: LMB smooth heightmap, [/] radius".into(),
                    EditorTool::Mass => "Mass: LMB place or drag existing mass point, RMB delete nearest".into(),
                    EditorTool::Hydro => "Hydro: LMB place hydrocarbon marker, RMB delete nearest".into(),
                    EditorTool::Start => format!("Start: LMB move {}, RMB cycle start slot", self.active_start_label()),
                    EditorTool::Prop => format!("Prop: LMB place {}, RMB delete nearest prop", self.active_prop_short()),
                };
            }
            EditorAction::RadiusDown => {
                self.brush_radius = (self.brush_radius - 4.0).clamp(4.0, 160.0);
            }
            EditorAction::RadiusUp => {
                self.brush_radius = (self.brush_radius + 4.0).clamp(4.0, 160.0);
            }
            EditorAction::StrengthDown => {
                self.brush_strength = (self.brush_strength - 0.5).clamp(0.25, 25.0);
            }
            EditorAction::StrengthUp => {
                self.brush_strength = (self.brush_strength + 0.5).clamp(0.25, 25.0);
            }
            EditorAction::CycleStart => {
                let n = self.doc.markers.start_spots.len().max(1);
                self.active_start_idx = (self.active_start_idx + 1) % n;
                self.status = format!("Active start spot: {}", self.active_start_label());
                self.status_error = false;
            }
            EditorAction::CycleProp => {
                let n = self.doc.prop_palette.len().max(1);
                self.active_prop_idx = (self.active_prop_idx + 1) % n;
                self.status = format!("Active prop: {}", self.active_prop_short());
                self.status_error = false;
            }
            EditorAction::FocusSaveId => {}
            EditorAction::Save => {
                let id = self.save_id.trim().to_string();
                match self.doc.save_as(dest_root, &id, None, Some(game_dir)) {
                    Ok(dir) => {
                        self.unsaved_changes = 0;
                        self.status_error = false;
                        self.status = format!("Saved '{}' to {}", id, dir.display());
                    }
                    Err(e) => {
                        self.status_error = true;
                        self.status = format!("Save failed: {e}");
                    }
                }
            }
        }
    }

    /// Handles a left-click (initial press) at world position `pos`.
    pub fn on_left_press(&mut self, pos: [f32; 3]) {
        match self.tool {
            EditorTool::Brush => {
                let n = self.doc.apply_brush(pos[0], pos[2], self.brush_radius, self.brush_strength, BrushMode::Raise);
                if n > 0 {
                    self.terrain_dirty = true;
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                }
            }
            EditorTool::Smooth => {
                let alpha = (self.brush_strength / 4.0).clamp(0.15, 1.0);
                let n = self.doc.apply_brush(pos[0], pos[2], self.brush_radius, alpha, BrushMode::Smooth);
                if n > 0 {
                    self.terrain_dirty = true;
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                }
            }
            EditorTool::Mass => {
                let pick_r = (self.doc.scmap.map_width.max(self.doc.scmap.map_height) * 0.015).clamp(4.0, 18.0);
                if let Some(idx) = find_marker_near(&self.doc.markers.mass_points, pos[0], pos[2], pick_r) {
                    self.dragging_mass = Some(idx);
                    let name = self.doc.markers.mass_points[idx].name.clone();
                    self.status = format!("Dragging mass marker {name}");
                    self.status_error = false;
                } else {
                    let name = self.doc.add_mass(pos[0], pos[2]);
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                    self.status = format!("Placed {name} at ({:.1}, {:.1})", pos[0], pos[2]);
                    self.status_error = false;
                }
            }
            EditorTool::Hydro => {
                let name = self.doc.add_hydro(pos[0], pos[2]);
                self.overlay_dirty = true;
                self.unsaved_changes += 1;
                self.status = format!("Placed {name} at ({:.1}, {:.1})", pos[0], pos[2]);
                self.status_error = false;
            }
            EditorTool::Start => {
                let idx_1 = self.active_start_idx + 1;
                match self.doc.move_start(idx_1, pos[0], pos[2]) {
                    Ok(name) => {
                        self.overlay_dirty = true;
                        self.unsaved_changes += 1;
                        self.status = format!("Moved {name} to ({:.1}, {:.1})", pos[0], pos[2]);
                        self.status_error = false;
                    }
                    Err(e) => {
                        self.status = e;
                        self.status_error = true;
                    }
                }
            }
            EditorTool::Prop => {
                let bp = self
                    .doc
                    .prop_palette
                    .get(self.active_prop_idx)
                    .cloned()
                    .unwrap_or_else(|| DEFAULT_PROP_BLUEPRINT.to_string());
                if self.doc.add_prop(&bp, pos[0], pos[2], 1.0) {
                    self.props_dirty = true;
                    self.unsaved_changes += 1;
                    self.status = format!("Placed prop {} at ({:.1}, {:.1})", self.active_prop_short(), pos[0], pos[2]);
                    self.status_error = false;
                } else {
                    self.status = "Map has no editable prop section".to_string();
                    self.status_error = true;
                }
            }
        }
    }

    /// Handles continuous left-drag at world position `pos`.
    pub fn on_left_drag(&mut self, pos: [f32; 3]) {
        match self.tool {
            EditorTool::Brush => {
                let n = self.doc.apply_brush(
                    pos[0],
                    pos[2],
                    self.brush_radius,
                    self.brush_strength * 0.45,
                    BrushMode::Raise,
                );
                if n > 0 {
                    self.terrain_dirty = true;
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                }
            }
            EditorTool::Smooth => {
                let alpha = (self.brush_strength / 5.0).clamp(0.1, 0.8);
                let n = self.doc.apply_brush(pos[0], pos[2], self.brush_radius, alpha, BrushMode::Smooth);
                if n > 0 {
                    self.terrain_dirty = true;
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                }
            }
            EditorTool::Mass => {
                if let Some(idx) = self.dragging_mass
                    && idx < self.doc.markers.mass_points.len()
                {
                    let (x, z) = self.doc.clamp_xz(pos[0], pos[2]);
                    let y = self.doc.scmap.sample_height(x, z);
                    self.doc.markers.mass_points[idx].position = [x, y, z];
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                }
            }
            _ => {}
        }
    }

    /// Handles right-click / right-drag at world position `pos`.
    pub fn on_right_action(&mut self, pos: [f32; 3], is_drag: bool) {
        let pick_r = (self.doc.scmap.map_width.max(self.doc.scmap.map_height) * 0.02).clamp(6.0, 24.0);
        match self.tool {
            EditorTool::Brush => {
                let scale = if is_drag { 0.45 } else { 1.0 };
                let n = self.doc.apply_brush(
                    pos[0],
                    pos[2],
                    self.brush_radius,
                    self.brush_strength * scale,
                    BrushMode::Lower,
                );
                if n > 0 {
                    self.terrain_dirty = true;
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                }
            }
            EditorTool::Mass if !is_drag => {
                if let Some(m) = self.doc.remove_mass_near(pos[0], pos[2], pick_r) {
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                    self.status = format!("Deleted mass marker {}", m.name);
                    self.status_error = false;
                }
            }
            EditorTool::Hydro if !is_drag => {
                if let Some(m) = self.doc.remove_hydro_near(pos[0], pos[2], pick_r) {
                    self.overlay_dirty = true;
                    self.unsaved_changes += 1;
                    self.status = format!("Deleted hydro marker {}", m.name);
                    self.status_error = false;
                }
            }
            EditorTool::Start if !is_drag => {
                let n = self.doc.markers.start_spots.len().max(1);
                self.active_start_idx = (self.active_start_idx + 1) % n;
                self.status = format!("Active start spot: {}", self.active_start_label());
                self.status_error = false;
            }
            EditorTool::Prop if !is_drag => {
                if let Some(p) = self.doc.remove_prop_near(pos[0], pos[2], pick_r) {
                    self.props_dirty = true;
                    self.unsaved_changes += 1;
                    let short = p.blueprint_path.rsplit('/').next().unwrap_or(&p.blueprint_path);
                    self.status = format!("Deleted prop {short}");
                    self.status_error = false;
                }
            }
            _ => {}
        }
    }

    /// Draws the top toolbar + bottom status strip with [`ui_kit::Canvas`] and returns
    /// clickable widget hit rects.
    pub fn build_toolbar(&self, c: &mut Canvas) -> (Rect, Vec<(Rect, EditorAction)>) {
        let mut hits: Vec<(Rect, EditorAction)> = Vec::new();
        let bar = Rect::new(8.0, 8.0, (c.w - 16.0).max(320.0), 48.0);
        c.panel(bar);

        let mut x = bar.x + 10.0;
        let y = bar.y + 9.0;
        let bh = 30.0;

        // Tool buttons
        for tool in EditorTool::ALL {
            let bw = 78.0;
            let r = Rect::new(x, y, bw, bh);
            c.button(
                r,
                tool.label(),
                WState {
                    hover: c.is_hover(r),
                    active: self.tool == tool,
                    ..Default::default()
                },
            );
            hits.push((r, EditorAction::SelectTool(tool)));
            x += bw + 6.0;
        }

        // Contextual controls
        x += 6.0;
        match self.tool {
            EditorTool::Brush | EditorTool::Smooth => {
                let r_dn = Rect::new(x, y, 26.0, bh);
                c.button(r_dn, "-", WState { hover: c.is_hover(r_dn), ..Default::default() });
                hits.push((r_dn, EditorAction::RadiusDown));
                x += 28.0;

                let r_lbl = Rect::new(x, y, 68.0, bh);
                c.text_in(r_lbl, body(13.0), &format!("R:{:.0}", self.brush_radius), theme::TEXT, Align::Center);
                x += 70.0;

                let r_up = Rect::new(x, y, 26.0, bh);
                c.button(r_up, "+", WState { hover: c.is_hover(r_up), ..Default::default() });
                hits.push((r_up, EditorAction::RadiusUp));
                x += 34.0;

                let s_dn = Rect::new(x, y, 26.0, bh);
                c.button(s_dn, "-", WState { hover: c.is_hover(s_dn), ..Default::default() });
                hits.push((s_dn, EditorAction::StrengthDown));
                x += 28.0;

                let s_lbl = Rect::new(x, y, 68.0, bh);
                c.text_in(s_lbl, body(13.0), &format!("S:{:.1}", self.brush_strength), theme::TEXT, Align::Center);
                x += 70.0;

                let s_up = Rect::new(x, y, 26.0, bh);
                c.button(s_up, "+", WState { hover: c.is_hover(s_up), ..Default::default() });
                hits.push((s_up, EditorAction::StrengthUp));
            }
            EditorTool::Start => {
                let r = Rect::new(x, y, 130.0, bh);
                c.cycler(r, self.active_start_label(), WState { hover: c.is_hover(r), ..Default::default() });
                hits.push((r, EditorAction::CycleStart));
            }
            EditorTool::Prop => {
                let r = Rect::new(x, y, 175.0, bh);
                c.cycler(r, self.active_prop_short(), WState { hover: c.is_hover(r), ..Default::default() });
                hits.push((r, EditorAction::CycleProp));
            }
            EditorTool::Mass | EditorTool::Hydro => {
                let info = format!(
                    "Mass:{} Hydro:{}",
                    self.doc.markers.mass_points.len(),
                    self.doc.markers.hydro_points.len()
                );
                c.text_in(Rect::new(x, y, 160.0, bh), body(13.0), &info, theme::DIM, Align::Left);
            }
        }

        // Right-aligned save ID field + SAVE button
        let save_btn = Rect::new(bar.right() - 88.0, y, 78.0, bh);
        let id_w = 220.0;
        let id_field = Rect::new(save_btn.x - id_w - 8.0, y, id_w, bh);
        if id_field.x > bar.x + 500.0 {
            c.text_field(
                id_field,
                &self.save_id,
                self.save_id_focused,
                "NEW_MAP_ID",
                WState { hover: c.is_hover(id_field), ..Default::default() },
            );
            hits.push((id_field, EditorAction::FocusSaveId));
        }
        let save_lbl = if self.unsaved_changes > 0 { "SAVE*" } else { "SAVE" };
        c.button(
            save_btn,
            save_lbl,
            WState {
                hover: c.is_hover(save_btn),
                primary: true,
                ..Default::default()
            },
        );
        hits.push((save_btn, EditorAction::Save));

        // Bottom status strip
        let foot = Rect::new(8.0, c.h - 34.0, (c.w - 16.0).max(320.0), 26.0);
        c.rect(foot, theme::PANEL);
        c.outline(foot, 1.0, theme::BORDER);
        let status_col = if self.status_error { theme::BAD } else { theme::TEXT };
        let fitted = c.fonts.fit(body(13.0), &self.status, foot.w - 220.0);
        c.text_in(Rect::new(foot.x + 10.0, foot.y, foot.w - 220.0, foot.h), body(13.0), &fitted, status_col, Align::Left);
        if let Some(p) = self.hover_world {
            let coord = format!("X:{:.1} Y:{:.1} Z:{:.1}", p[0], p[1], p[2]);
            c.text_in(Rect::new(foot.right() - 210.0, foot.y, 200.0, foot.h), bold(13.0), &coord, theme::ACCENT, Align::Right);
        }

        (bar, hits)
    }

    /// Builds dynamic 3D editor geometry: hydrocarbon beacons, active start-spot
    /// highlight ring, and a terrain-hugging brush/cursor ring.
    pub fn build_dynamic_overlay(&self) -> (Vec<GpuVertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut idx = Vec::new();
        let span = self.doc.scmap.map_width.max(self.doc.scmap.map_height).max(256.0);
        let base_r = (span * 0.008).clamp(2.0, 14.0);
        let min_y = if self.doc.scmap.water.has_water {
            self.doc.scmap.water.elevation
        } else {
            0.0
        };

        // Hydrocarbon beacons (cyan pyramids)
        let hydro_r = base_r * 0.65;
        for h in &self.doc.markers.hydro_points {
            let y = h.position[1].max(min_y);
            append_editor_pyramid(&mut verts, &mut idx, [h.position[0], y, h.position[2]], hydro_r, hydro_r * 2.8, [0.16, 0.78, 0.96, 0.50]);
        }

        // Active start-spot ring when Start tool is active
        if self.tool == EditorTool::Start
            && let Some(spot) = self.doc.markers.start_spots.get(self.active_start_idx)
        {
            append_terrain_ring(
                &mut verts,
                &mut idx,
                &self.doc.scmap,
                spot.position[0],
                spot.position[2],
                base_r * 1.6,
                [1.0, 0.92, 0.32, 0.85],
            );
        }

        // Cursor / brush ring
        if let Some(hp) = self.hover_world {
            let (r, col) = match self.tool {
                EditorTool::Brush => (self.brush_radius, [0.35, 0.95, 0.55, 0.85]),
                EditorTool::Smooth => (self.brush_radius, [0.45, 0.82, 1.0, 0.85]),
                EditorTool::Mass => (base_r * 0.7, [0.22, 0.92, 0.42, 0.85]),
                EditorTool::Hydro => (base_r * 0.9, [0.20, 0.85, 0.98, 0.85]),
                EditorTool::Start => (base_r * 1.3, [0.98, 0.82, 0.20, 0.85]),
                EditorTool::Prop => (base_r * 0.7, [0.85, 0.65, 0.35, 0.85]),
            };
            append_terrain_ring(&mut verts, &mut idx, &self.doc.scmap, hp[0], hp[2], r, col);
        }

        (verts, idx)
    }
}

fn append_editor_pyramid(
    verts: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    base_center: [f32; 3],
    radius: f32,
    height: f32,
    color: [f32; 4],
) {
    let [cx, cy, cz] = base_center;
    let apex = [cx, cy + height, cz];
    let ring = [
        [cx - radius, cy, cz - radius],
        [cx + radius, cy, cz - radius],
        [cx + radius, cy, cz + radius],
        [cx - radius, cy, cz + radius],
    ];
    for side in 0..4 {
        let p0 = ring[side];
        let p1 = ring[(side + 1) % 4];
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [apex[0] - p0[0], apex[1] - p0[1], apex[2] - p0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-5);
        let normal = [n[0] / len, n[1] / len, n[2] / len];
        let base = verts.len() as u32;
        verts.extend_from_slice(&[
            GpuVertex { position: p0, normal, uv: [0.0, 0.0], color },
            GpuVertex { position: p1, normal, uv: [1.0, 0.0], color },
            GpuVertex { position: apex, normal, uv: [0.5, 1.0], color },
        ]);
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

fn append_terrain_ring(
    verts: &mut Vec<GpuVertex>,
    indices: &mut Vec<u32>,
    scmap: &ScMap,
    cx: f32,
    cz: f32,
    radius: f32,
    color: [f32; 4],
) {
    let segs = 36usize;
    let thick = (radius * 0.08).clamp(0.45, 2.4);
    let r_in = (radius - thick * 0.5).max(0.2);
    let r_out = radius + thick * 0.5;
    let min_y = if scmap.water.has_water { scmap.water.elevation } else { f32::NEG_INFINITY };
    let base = verts.len() as u32;
    for i in 0..=segs {
        let a = (i as f32 / segs as f32) * std::f32::consts::TAU;
        let (sa, ca) = a.sin_cos();
        let ix = cx + ca * r_in;
        let iz = cz + sa * r_in;
        let ox = cx + ca * r_out;
        let oz = cz + sa * r_out;
        let iy = scmap.sample_height(ix, iz).max(min_y) + 0.35;
        let oy = scmap.sample_height(ox, oz).max(min_y) + 0.35;
        verts.push(GpuVertex { position: [ix, iy, iz], normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0], color });
        verts.push(GpuVertex { position: [ox, oy, oz], normal: [0.0, 1.0, 0.0], uv: [1.0, 0.0], color });
    }
    for i in 0..segs as u32 {
        let a = base + i * 2;
        indices.extend_from_slice(&[a, a + 1, a + 2, a + 2, a + 1, a + 3]);
    }
}

pub fn cmd_map_editor(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err(
            "usage: farust map-editor <MAP> [GAME_DIR] [--out NEW_ID] [--size WxH] [--snapshot OUT.png]"
                .to_string(),
        );
    }
    let map_name = &args[2];
    let mut game_dir: Option<PathBuf> = None;
    let mut out_id: Option<String> = None;
    let mut snapshot: Option<PathBuf> = None;
    let mut size: (u32, u32) = (1280, 800);

    let mut i = 3usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out_id = Some(
                    args.get(i + 1)
                        .ok_or_else(|| "missing value after --out".to_string())?
                        .clone(),
                );
                i += 2;
            }
            "--snapshot" => {
                snapshot = Some(PathBuf::from(
                    args.get(i + 1)
                        .ok_or_else(|| "missing value after --snapshot".to_string())?,
                ));
                i += 2;
            }
            "--size" => {
                let raw = args
                    .get(i + 1)
                    .ok_or_else(|| "missing value after --size".to_string())?;
                let (w, h) = raw
                    .split_once(['x', 'X'])
                    .ok_or_else(|| format!("invalid --size '{raw}'"))?;
                size = (
                    w.parse().map_err(|_| format!("invalid width in '{raw}'"))?,
                    h.parse().map_err(|_| format!("invalid height in '{raw}'"))?,
                );
                i += 2;
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown map-editor flag '{other}'"));
            }
            pos => {
                if game_dir.is_none() {
                    game_dir = Some(PathBuf::from(pos));
                } else {
                    return Err(format!("unexpected extra argument '{pos}'"));
                }
                i += 1;
            }
        }
    }

    let game_dir = game_dir.unwrap_or_else(scd::default_game_dir);
    let doc = MapDocument::load(&game_dir, map_name)?;
    let aspect = size.0.max(1) as f32 / size.1.max(1) as f32;
    let title = format!("FARust Map Editor — {}", doc.src_id);
    let scene = build_map_scene_from_parts(&game_dir, title, &doc.scmap, Some(&doc.markers), aspect);
    let state = EditorState::new(doc, out_id);

    if let Some(out_path) = snapshot {
        let gpu = render_editor_snapshot(&game_dir, &scene, &state, size.0, size.1, &out_path)?;
        println!(
            "rendered map-editor '{}' on {} -> {}",
            state.doc.src_id,
            gpu,
            out_path.display()
        );
        return Ok(());
    }

    println!(
        "opening map editor '{}' — 1..6 tools, LMB/RMB edit, MMB orbit, WASD pan, Wheel zoom, Ctrl+S save, Esc quit",
        state.doc.src_id
    );
    run_editor_window(game_dir, scene, state, size.0, size.1)
}

fn render_editor_snapshot(
    game_dir: &Path,
    scene: &LoadedScene,
    state: &EditorState,
    width: u32,
    height: u32,
    out_path: &Path,
) -> Result<String, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no compatible GPU adapter found".to_string())?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        .map_err(|e| format!("failed to create GPU device: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut camera = scene.initial_camera.clone();
    camera.set_aspect(width, height);
    let mut resources = GpuSceneResources::new(&device, &queue, scene, format, MSAA_SAMPLES, &camera);
    let (dv, di) = state.build_dynamic_overlay();
    resources.set_dynamic(&device, &dv, &di);

    let (_msaa_tex, msaa_view) = create_attachment(
        &device,
        width,
        height,
        format,
        MSAA_SAMPLES,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "editor_snap_msaa",
    );
    let (resolve_tex, resolve_view) = create_attachment(
        &device,
        width,
        height,
        format,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        "editor_snap_resolve",
    );
    let (_depth_tex, depth_view) = create_attachment(
        &device,
        width,
        height,
        DEPTH_FORMAT,
        MSAA_SAMPLES,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
        "editor_snap_depth",
    );

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("editor_snap_enc") });
    resources.encode_draw(&mut enc, &msaa_view, Some(&resolve_view), &depth_view, scene.clear_color);
    queue.submit(Some(enc.finish()));

    // Composite ui_kit toolbar on top
    let fonts = Fonts::load(game_dir);
    let images = ImageStore::default();
    let mut ui = UiRenderer::new(&device, &queue, format, &fonts);
    let mut canvas = Canvas::new(width as f32, height as f32, &fonts, &images, None);
    let _ = state.build_toolbar(&mut canvas);
    ui.draw_overlay(&device, &queue, &resolve_view, &canvas);

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bpr = width * 4;
    let padded = bpr.div_ceil(align) * align;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("editor_snap_rb"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc2.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &resolve_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit(Some(enc2.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("readback channel failed: {e}"))?
        .map_err(|e| format!("GPU map failed: {e}"))?;
    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((bpr * height) as usize);
    for row in 0..height as usize {
        let s = row * padded as usize;
        pixels.extend_from_slice(&mapped[s..s + bpr as usize]);
    }
    drop(mapped);
    readback.unmap();
    ui_kit::save_png(&crate::dds_decode::RgbaImage { width, height, pixels }, out_path)?;
    Ok(format!("{} ({:?})", info.name, info.backend))
}

struct EditorGpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    resources: GpuSceneResources,
    ui: UiRenderer,
    fonts: Fonts,
    images: ImageStore,
    camera: OrbitCamera,
    textured_terrain: bool,
    has_gpu_water: bool,
}

struct MapEditorApp {
    game_dir: PathBuf,
    dest_root: PathBuf,
    scene: LoadedScene,
    editor: EditorState,
    init_size: (u32, u32),
    gpu: Option<EditorGpu>,
    init_err: Option<String>,
    cursor: (f32, f32),
    last_drag_cursor: Option<(f64, f64)>,
    lmb_down: bool,
    rmb_down: bool,
    mmb_down: bool,
    modifiers: Modifiers,
    toolbar_rect: Rect,
    toolbar_hits: Vec<(Rect, EditorAction)>,
}

fn run_editor_window(
    game_dir: PathBuf,
    scene: LoadedScene,
    editor: EditorState,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("failed to create event loop: {e}"))?;
    let mut app = MapEditorApp {
        game_dir,
        dest_root: scenario::farust_maps_dir(),
        scene,
        editor,
        init_size: (width.max(640), height.max(480)),
        gpu: None,
        init_err: None,
        cursor: (0.0, 0.0),
        last_drag_cursor: None,
        lmb_down: false,
        rmb_down: false,
        mmb_down: false,
        modifiers: Modifiers::default(),
        toolbar_rect: Rect::default(),
        toolbar_hits: Vec::new(),
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("event loop error: {e}"))?;
    if let Some(err) = app.init_err {
        return Err(err);
    }
    Ok(())
}

impl MapEditorApp {
    fn init_window(&mut self, event_loop: &ActiveEventLoop) -> Result<EditorGpu, String> {
        let attrs = Window::default_attributes()
            .with_title(self.scene.title.clone())
            .with_inner_size(PhysicalSize::new(self.init_size.0, self.init_size.1));
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
        .ok_or_else(|| "no compatible GPU adapter for editor window".to_string())?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
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
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut camera = self.scene.initial_camera.clone();
        camera.set_aspect(width, height);
        let resources = GpuSceneResources::new(&device, &queue, &self.scene, format, MSAA_SAMPLES, &camera);
        let (_, msaa_view) = create_attachment(
            &device,
            width,
            height,
            format,
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "ed_msaa",
        );
        let (_, depth_view) = create_attachment(
            &device,
            width,
            height,
            DEPTH_FORMAT,
            MSAA_SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            "ed_depth",
        );
        let fonts = Fonts::load(&self.game_dir);
        let ui = UiRenderer::new(&device, &queue, format, &fonts);
        Ok(EditorGpu {
            window,
            surface,
            device,
            queue,
            config,
            msaa_view,
            depth_view,
            resources,
            ui,
            fonts,
            images: ImageStore::default(),
            camera,
            textured_terrain: self.scene.terrain_splat.is_some() || self.scene.albedo.width > 1,
            has_gpu_water: self.scene.water.is_some(),
        })
    }

    fn sync_dirty_gpu(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        if self.editor.terrain_dirty || self.editor.overlay_dirty {
            let (verts, idx, ov_v, ov_i) = build_terrain_and_overlay_meshes_opts(
                &self.editor.doc.scmap,
                Some(&self.editor.doc.markers),
                gpu.textured_terrain,
                !gpu.has_gpu_water,
            );
            if self.editor.terrain_dirty {
                gpu.resources.update_terrain_mesh(&gpu.device, &verts, &idx);
                self.editor.terrain_dirty = false;
            }
            gpu.resources.update_static_overlay(&gpu.device, &ov_v, &ov_i);
            self.editor.overlay_dirty = false;
        }
        if self.editor.props_dirty {
            let batches = load_map_props(&self.game_dir, &self.editor.doc.scmap);
            gpu.resources.update_prop_batches(&gpu.device, &gpu.queue, &batches);
            self.editor.props_dirty = false;
        }
    }
}

impl ApplicationHandler for MapEditorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        match self.init_window(event_loop) {
            Ok(gpu) => {
                gpu.window.request_redraw();
                self.gpu = Some(gpu);
            }
            Err(err) => {
                self.init_err = Some(err);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if self.gpu.is_none() {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ModifiersChanged(m) => self.modifiers = m,
            WindowEvent::Resized(new_size) => {
                let gpu = self.gpu.as_mut().unwrap();
                let w = new_size.width.max(1);
                let h = new_size.height.max(1);
                gpu.config.width = w;
                gpu.config.height = h;
                gpu.surface.configure(&gpu.device, &gpu.config);
                let (_, mv) = create_attachment(
                    &gpu.device,
                    w,
                    h,
                    gpu.config.format,
                    MSAA_SAMPLES,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                    "ed_msaa",
                );
                let (_, dv) = create_attachment(
                    &gpu.device,
                    w,
                    h,
                    DEPTH_FORMAT,
                    MSAA_SAMPLES,
                    wgpu::TextureUsages::RENDER_ATTACHMENT,
                    "ed_depth",
                );
                gpu.msaa_view = mv;
                gpu.depth_view = dv;
                gpu.camera.set_aspect(w, h);
                gpu.resources.update_camera(&gpu.queue, &gpu.camera);
                gpu.window.request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let (cx, cy) = self.cursor;
                match button {
                    MouseButton::Left => {
                        self.lmb_down = pressed;
                        if pressed {
                            if let Some(act) = hit_test(&self.toolbar_hits, cx, cy) {
                                self.editor.apply_action(act, &self.dest_root, &self.game_dir);
                                self.lmb_down = false;
                            } else if !self.toolbar_rect.contains(cx, cy) {
                                self.editor.save_id_focused = false;
                                if self.modifiers.state().alt_key() {
                                    self.last_drag_cursor = Some((f64::from(cx), f64::from(cy)));
                                } else if let Some(wp) = self.editor.hover_world {
                                    self.editor.on_left_press(wp);
                                }
                            }
                        } else {
                            self.editor.dragging_mass = None;
                            self.last_drag_cursor = None;
                        }
                    }
                    MouseButton::Right => {
                        self.rmb_down = pressed;
                        if pressed && !self.toolbar_rect.contains(cx, cy) {
                            if self.modifiers.state().shift_key() {
                                self.last_drag_cursor = Some((f64::from(cx), f64::from(cy)));
                            } else if let Some(wp) = self.editor.hover_world {
                                self.editor.on_right_action(wp, false);
                            }
                        } else if !pressed {
                            self.last_drag_cursor = None;
                        }
                    }
                    MouseButton::Middle => {
                        self.mmb_down = pressed;
                        self.last_drag_cursor = pressed.then_some((f64::from(cx), f64::from(cy)));
                    }
                    _ => {}
                }
                self.sync_dirty_gpu();
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cur = (position.x, position.y);
                self.cursor = (cur.0 as f32, cur.1 as f32);
                let mut cam_changed = false;
                if let Some((px, py)) = self.last_drag_cursor {
                    let dx = (cur.0 - px) as f32;
                    let dy = (cur.1 - py) as f32;
                    let gpu = self.gpu.as_mut().unwrap();
                    if self.mmb_down || (self.lmb_down && self.modifiers.state().alt_key()) {
                        gpu.camera.orbit(-dx * 0.006, dy * 0.006);
                        cam_changed = true;
                    } else if self.rmb_down && self.modifiers.state().shift_key() {
                        gpu.camera.pan(-dx * 0.0018, dy * 0.0018);
                        cam_changed = true;
                    }
                    if cam_changed {
                        gpu.resources.update_camera(&gpu.queue, &gpu.camera);
                        self.last_drag_cursor = Some(cur);
                    }
                }
                if !cam_changed {
                    let gpu = self.gpu.as_ref().unwrap();
                    if self.toolbar_rect.contains(self.cursor.0, self.cursor.1) {
                        self.editor.hover_world = None;
                    } else {
                        self.editor.hover_world = pick_terrain_point(
                            &gpu.camera,
                            &self.editor.doc.scmap,
                            self.cursor.0,
                            self.cursor.1,
                            gpu.config.width,
                            gpu.config.height,
                        );
                    }
                    if let Some(wp) = self.editor.hover_world {
                        if self.lmb_down && !self.modifiers.state().alt_key() {
                            self.editor.on_left_drag(wp);
                        } else if self.rmb_down && !self.modifiers.state().shift_key() {
                            self.editor.on_right_action(wp, true);
                        }
                    }
                }
                self.sync_dirty_gpu();
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                if steps.abs() > 0.0 {
                    let gpu = self.gpu.as_mut().unwrap();
                    gpu.camera.zoom(steps);
                    gpu.resources.update_camera(&gpu.queue, &gpu.camera);
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if self.editor.save_id_focused {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape | KeyCode::Enter) => {
                            self.editor.save_id_focused = false;
                        }
                        PhysicalKey::Code(KeyCode::Backspace) => {
                            self.editor.save_id.pop();
                        }
                        _ => {
                            if let Key::Character(ref ch) = event.logical_key {
                                for c in ch.chars() {
                                    if (c.is_ascii_alphanumeric() || c == '_' || c == '-')
                                        && self.editor.save_id.len() < 64
                                    {
                                        self.editor.save_id.push(c);
                                    }
                                }
                            }
                        }
                    }
                    if let Some(gpu) = &self.gpu {
                        gpu.window.request_redraw();
                    }
                    return;
                }

                let ctrl = self.modifiers.state().control_key();
                let mut cam_changed = false;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        event_loop.exit();
                        return;
                    }
                    PhysicalKey::Code(KeyCode::KeyS) if ctrl => {
                        self.editor.apply_action(EditorAction::Save, &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::Digit1) => {
                        self.editor.apply_action(EditorAction::SelectTool(EditorTool::Brush), &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::Digit2) => {
                        self.editor.apply_action(EditorAction::SelectTool(EditorTool::Smooth), &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::Digit3) => {
                        self.editor.apply_action(EditorAction::SelectTool(EditorTool::Mass), &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::Digit4) => {
                        self.editor.apply_action(EditorAction::SelectTool(EditorTool::Hydro), &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::Digit5) => {
                        self.editor.apply_action(EditorAction::SelectTool(EditorTool::Start), &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::Digit6) => {
                        self.editor.apply_action(EditorAction::SelectTool(EditorTool::Prop), &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::BracketLeft) => {
                        self.editor.apply_action(EditorAction::RadiusDown, &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::BracketRight) => {
                        self.editor.apply_action(EditorAction::RadiusUp, &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::Minus) => {
                        self.editor.apply_action(EditorAction::StrengthDown, &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::Equal) => {
                        self.editor.apply_action(EditorAction::StrengthUp, &self.dest_root, &self.game_dir);
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        let gpu = self.gpu.as_mut().unwrap();
                        let aspect = gpu.camera.aspect;
                        gpu.camera = self.scene.initial_camera.clone();
                        gpu.camera.aspect = aspect;
                        cam_changed = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyW | KeyCode::ArrowUp) => {
                        self.gpu.as_mut().unwrap().camera.pan(0.0, 0.06);
                        cam_changed = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyS | KeyCode::ArrowDown) => {
                        self.gpu.as_mut().unwrap().camera.pan(0.0, -0.06);
                        cam_changed = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyA | KeyCode::ArrowLeft) => {
                        self.gpu.as_mut().unwrap().camera.pan(-0.06, 0.0);
                        cam_changed = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyD | KeyCode::ArrowRight) => {
                        self.gpu.as_mut().unwrap().camera.pan(0.06, 0.0);
                        cam_changed = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) => {
                        self.gpu.as_mut().unwrap().camera.orbit(-0.10, 0.0);
                        cam_changed = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyE) => {
                        self.gpu.as_mut().unwrap().camera.orbit(0.10, 0.0);
                        cam_changed = true;
                    }
                    _ => {}
                }
                if cam_changed {
                    let gpu = self.gpu.as_mut().unwrap();
                    gpu.resources.update_camera(&gpu.queue, &gpu.camera);
                }
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                let (dv, di) = self.editor.build_dynamic_overlay();
                let gpu = self.gpu.as_mut().unwrap();
                gpu.resources.set_dynamic(&gpu.device, &dv, &di);

                let frame = match gpu.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                        gpu.surface.configure(&gpu.device, &gpu.config);
                        return;
                    }
                    Err(_) => return,
                };
                let resolve_view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("editor_3d_enc"),
                    });
                gpu.resources.encode_draw(
                    &mut encoder,
                    &gpu.msaa_view,
                    Some(&resolve_view),
                    &gpu.depth_view,
                    self.scene.clear_color,
                );
                gpu.queue.submit(Some(encoder.finish()));

                let mut canvas = Canvas::new(
                    gpu.config.width as f32,
                    gpu.config.height as f32,
                    &gpu.fonts,
                    &gpu.images,
                    Some(self.cursor),
                );
                let (bar, hits) = self.editor.build_toolbar(&mut canvas);
                self.toolbar_rect = bar;
                self.toolbar_hits = hits;
                gpu.ui.draw_overlay(&gpu.device, &gpu.queue, &resolve_view, &canvas);
                frame.present();
            }
            _ => {}
        }
    }
}

// Keep unused imports clean if needed.
#[allow(dead_code)]
fn _assert_imports_used() {
    let _ = viewer::MSAA_SAMPLES;
    let _ = scene::load_map_scene;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scmap::{ScMapLighting, WaterInfo};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            static SEQ: AtomicUsize = AtomicUsize::new(0);
            let n = SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "farust_map_author_{prefix}_{}_{}",
                std::process::id(),
                n
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create temp dir");
            Self(dir)
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn synthetic_scmap(w: u32, h: u32) -> ScMap {
        let count = (w as usize + 1) * (h as usize + 1);
        ScMap {
            version_major: 2,
            version_minor: 60,
            map_width: 256.0,
            map_height: 256.0,
            preview_dds: b"DDS test".to_vec(),
            heightmap_width: w,
            heightmap_height: h,
            height_scale: 0.25,
            heightmap: vec![80; count],
            terrain_shader: "TTerrain".to_string(),
            background_path: "/env/bg.dds".to_string(),
            sky_cubemap_path: "/env/sky.dds".to_string(),
            cubemaps: vec![("default".into(), "/env/sky.dds".into())],
            lighting: ScMapLighting::default(),
            water: WaterInfo {
                has_water: true,
                elevation: 10.0,
                elevation_deep: 8.0,
                elevation_abyss: 2.0,
            },
            terrain_textures: None,
            unknown1: 1,
            unknown2: 2,
            unknown3: 3,
            unknown4: 4,
            post_heightmap_byte: 1,
            shader_raw: Vec::new(),
            background_raw: Vec::new(),
            sky_raw: Vec::new(),
            cubemaps_raw: Vec::new(),
            tail_bytes: b"TAIL_BLOB".to_vec(),
        }
    }

    fn write_synthetic_map_dir(root: &Path, id: &str) {
        let dir = root.join("maps").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        let scmap = synthetic_scmap(16, 16);
        scmap.write_file(&dir.join(format!("{id}.scmap"))).unwrap();
        let scenario = format!(
            "version = 3\nScenarioInfo = {{\n    name = 'Synthetic Test',\n    type = 'skirmish',\n    size = {{256, 256}},\n    map = '/maps/{id}/{id}.scmap',\n    save = '/maps/{id}/{id}_save.lua',\n    script = '/maps/{id}/{id}_script.lua',\n    Configurations = {{\n        ['standard'] = {{\n            teams = {{ {{ name = 'FFA', armies = {{'ARMY_1', 'ARMY_2'}} }} }},\n        }},\n    }},\n}}\n"
        );
        let save = "Scenario = {\n    MasterChain = {\n        ['_MASTERCHAIN_'] = {\n            Markers = {\n                ['ARMY_1'] = {\n                    ['type'] = STRING( 'Blank Marker' ),\n                    ['position'] = VECTOR3( 32.0, 20.0, 32.0 ),\n                },\n                ['ARMY_2'] = {\n                    ['type'] = STRING( 'Blank Marker' ),\n                    ['position'] = VECTOR3( 224.0, 20.0, 224.0 ),\n                },\n                ['Mass 01'] = {\n                    ['type'] = STRING( 'Mass' ),\n                    ['position'] = VECTOR3( 64.0, 20.0, 64.0 ),\n                },\n                ['PathNode 01'] = {\n                    ['type'] = STRING( 'Land Path Node' ),\n                    ['position'] = VECTOR3( 128.0, 20.0, 128.0 ),\n                },\n            },\n        },\n    },\n}\n";
        std::fs::write(dir.join(format!("{id}_scenario.lua")), scenario).unwrap();
        std::fs::write(dir.join(format!("{id}_save.lua")), save).unwrap();
        std::fs::write(dir.join(format!("{id}_script.lua")), DEFAULT_SCRIPT_LUA).unwrap();
    }

    #[test]
    fn roundtrip_scmap_bytes_reports_identity_and_detects_diffs() {
        let map = synthetic_scmap(8, 8);
        let bytes = map.write();
        let rep = roundtrip_scmap_bytes("SYNTH", &bytes).expect("roundtrip");
        assert!(rep.byte_identical);
        assert!(rep.reparse_ok);
        assert_eq!(rep.first_diff, None);
        assert_eq!(rep.heightmap_size, (8, 8));
    }

    #[test]
    fn document_edit_save_and_reload_roundtrips_cleanly() {
        let game = TempDirGuard::new("game");
        let user_out = TempDirGuard::new("user_maps");
        write_synthetic_map_dir(&game.0, "SYN_001");

        let mut doc = MapDocument::load(&game.0, "SYN_001").expect("load synthetic");
        let changed = doc.apply_brush(128.0, 128.0, 40.0, 12.0, BrushMode::Raise);
        assert!(changed > 0);
        let new_mass = doc.add_mass(128.0, 128.0);
        assert_eq!(new_mass, "Mass 02");
        let new_hydro = doc.add_hydro(96.0, 160.0);
        assert_eq!(new_hydro, "Hydrocarbon 01");
        let moved = doc.move_start(1, 48.0, 48.0).expect("move ARMY_1");
        assert_eq!(moved, "ARMY_1");

        // Refuses writing inside read-only game install
        assert!(doc.save_as(&game.0.join("maps"), "BAD_WRITE", None, Some(&game.0)).is_err());
        // Rejects unsafe map id
        assert!(doc.save_as(&user_out.0, "../evil", None, Some(&game.0)).is_err());

        let saved_dir = doc
            .save_as(&user_out.0, "SYN_EDITED", Some("Edited Synth"), Some(&game.0))
            .expect("save edited map");
        assert!(saved_dir.join("SYN_EDITED.scmap").is_file());
        assert!(saved_dir.join("SYN_EDITED_scenario.lua").is_file());
        assert!(saved_dir.join("SYN_EDITED_save.lua").is_file());
        assert!(saved_dir.join("SYN_EDITED_script.lua").is_file());

        // Reload via scenario::list_maps_in (same discovery path as real user maps)
        let listed = scenario::list_maps_in(&user_out.0, &game.0, MapSource::Farust, None).expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "SYN_EDITED");
        let meta = scenario::parse_scenario_file(&listed[0].scenario_path, "SYN_EDITED").expect("meta");
        assert_eq!(meta.name, "Edited Synth");
        assert_eq!(meta.map_vfs_path, "/maps/SYN_EDITED/SYN_EDITED.scmap");
        assert_eq!(meta.save_vfs_path, "/maps/SYN_EDITED/SYN_EDITED_save.lua");

        let markers = scenario::parse_save_markers_file(listed[0].save_path.as_deref().unwrap(), &meta.armies)
            .expect("markers");
        assert_eq!(markers.start_spots.len(), 2);
        assert!((markers.start_spots[0].position[0] - 48.0).abs() < 1e-3);
        assert_eq!(markers.mass_points.len(), 2);
        assert_eq!(markers.hydro_points.len(), 1);
        assert_eq!(markers.total_markers, 6); // 2 starts + 2 mass + 1 hydro + 1 kept PathNode

        let reloaded_scmap = ScMap::parse_file(listed[0].scmap_path.as_deref().unwrap()).expect("reparse scmap");
        assert_eq!(reloaded_scmap.heightmap, doc.scmap.heightmap);
        assert!(reloaded_scmap.sample_height(128.0, 128.0) > 25.0);
    }

    #[test]
    fn editor_state_toolbar_and_tools_work_without_gpu() {
        let game = TempDirGuard::new("ed_game");
        let out = TempDirGuard::new("ed_out");
        write_synthetic_map_dir(&game.0, "ED_001");
        let doc = MapDocument::load(&game.0, "ED_001").expect("load");
        let mut st = EditorState::new(doc, Some("ED_001_CUSTOM".into()));
        st.hover_world = Some([128.0, 20.0, 128.0]);

        let fonts = Fonts::fallback();
        let images = ImageStore::default();
        let mut canvas = Canvas::new(1280.0, 800.0, &fonts, &images, None);
        let (bar, hits) = st.build_toolbar(&mut canvas);
        assert!(bar.w > 300.0);
        assert!(hits.iter().any(|(_, a)| *a == EditorAction::Save));
        assert!(hits.iter().any(|(_, a)| *a == EditorAction::SelectTool(EditorTool::Mass)));

        // Brush raise + smooth
        st.on_left_press([128.0, 20.0, 128.0]);
        assert!(st.terrain_dirty);
        assert!(st.unsaved_changes >= 1);

        // Place mass & delete mass
        st.apply_action(EditorAction::SelectTool(EditorTool::Mass), &out.0, &game.0);
        let before_mass = st.doc.markers.mass_points.len();
        st.on_left_press([150.0, 20.0, 150.0]);
        assert_eq!(st.doc.markers.mass_points.len(), before_mass + 1);
        st.on_right_action([150.0, 20.0, 150.0], false);
        assert_eq!(st.doc.markers.mass_points.len(), before_mass);

        // Place hydro & move start
        st.apply_action(EditorAction::SelectTool(EditorTool::Hydro), &out.0, &game.0);
        st.on_left_press([180.0, 20.0, 90.0]);
        assert_eq!(st.doc.markers.hydro_points.len(), 1);
        st.apply_action(EditorAction::SelectTool(EditorTool::Start), &out.0, &game.0);
        st.on_left_press([60.0, 20.0, 60.0]);
        assert!((st.doc.markers.start_spots[0].position[0] - 60.0).abs() < 1e-3);

        // Dynamic overlay builds valid triangles
        let (dv, di) = st.build_dynamic_overlay();
        assert!(!dv.is_empty() && !di.is_empty() && di.len() % 3 == 0);
        assert!(di.iter().all(|&i| (i as usize) < dv.len()));

        // Save via toolbar action
        st.apply_action(EditorAction::Save, &out.0, &game.0);
        assert!(!st.status_error, "unexpected error: {}", st.status);
        assert_eq!(st.unsaved_changes, 0);
        assert!(out.0.join("ED_001_CUSTOM/ED_001_CUSTOM.scmap").is_file());
    }
}
