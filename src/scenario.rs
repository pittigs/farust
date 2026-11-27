use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ScenarioMeta {
    pub id: String,
    pub name: String,
    pub description: String,
    pub scenario_type: String,
    pub size: (u32, u32),
    pub map_vfs_path: String,
    pub save_vfs_path: String,
    pub armies: Vec<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Marker {
    pub name: String,
    pub marker_type: String,
    pub position: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct SaveMarkers {
    pub start_spots: Vec<Marker>,
    pub mass_points: Vec<Marker>,
    pub hydro_points: Vec<Marker>,
    pub total_markers: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MapFiles {
    pub id: String,
    pub dir: PathBuf,
    pub scenario_path: PathBuf,
    pub save_path: Option<PathBuf>,
    pub scmap_path: Option<PathBuf>,
    /// Phase 24: where the map was discovered.
    pub source: MapSource,
}

/// Phase 24: origin of a discovered map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum MapSource {
    /// `<game_dir>/maps`.
    #[default]
    Stock,
    /// FA's own user map folder (`Documents/My Games/.../Maps`, Proton prefix on Linux).
    FaUser,
    /// `$XDG_DATA_HOME/farust/maps`.
    Farust,
    /// `$FARUST_MAPS_DIR`.
    Env,
}

impl MapSource {
    pub fn label(self) -> &'static str {
        match self {
            MapSource::Stock => "stock",
            MapSource::FaUser => "fa-user",
            MapSource::Farust => "farust",
            MapSource::Env => "env",
        }
    }
}

/// Resolves a Forged Alliance virtual filesystem path (e.g. `/maps/SCMP_001/SCMP_001.scmap`)
/// case-insensitively on disk relative to `game_dir`.
pub fn resolve_vfs_path(game_dir: &Path, vfs_path: &str) -> Option<PathBuf> {
    let normalized = vfs_path.replace('\\', "/");
    let trimmed = normalized.trim_start_matches('/');
    let mut segments = trimmed.split('/').filter(|s| !s.is_empty()).peekable();
    segments.peek()?;

    let mut current = game_dir.to_path_buf();
    for seg in segments {
        let exact = current.join(seg);
        if exact.exists() {
            current = exact;
            continue;
        }
        let entries = std::fs::read_dir(&current).ok()?;
        let mut matched: Option<PathBuf> = None;
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().eq_ignore_ascii_case(seg) {
                matched = Some(entry.path());
                break;
            }
        }
        current = matched?;
    }

    Some(current)
}

/// Discovers all maps: stock maps under `<game_dir>/maps` first, then the user map
/// folders of [`user_map_dirs`] (loose folders and `.scd` / `.zip` archives). Ids are unique
/// case-insensitively; the first source wins. Sorted by id.
pub fn list_maps(game_dir: &Path) -> Result<Vec<MapFiles>, String> {
    let maps_dir = resolve_vfs_path(game_dir, "maps")
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| game_dir.join("maps"));
    let mut maps = list_maps_in(&maps_dir, game_dir, MapSource::Stock, None)?;
    let cache = map_cache_dir();
    // Unit tests must not see the developer's real user map folders.
    let user_dirs = if cfg!(test) { Vec::new() } else { user_map_dirs(game_dir) };
    for (source, dir) in user_dirs {
        if dir == maps_dir || !dir.is_dir() {
            continue;
        }
        if let Ok(found) = list_maps_in(&dir, game_dir, source, Some(&cache)) {
            maps.extend(found);
        }
    }
    let mut seen = HashSet::new();
    maps.retain(|m| seen.insert(m.id.to_ascii_lowercase()));
    maps.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(maps)
}

/// `$XDG_DATA_HOME` (absolute) or `~/.local/share`.
pub fn xdg_data_home() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Extraction cache for archived user maps: `$XDG_CACHE_HOME/farust/maps` (`~/.cache/...`).
pub fn map_cache_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| std::env::temp_dir())
        .join("farust")
        .join("maps")
}

const FA_USER_MAPS: &str =
    "compatdata/9420/pfx/drive_c/users/steamuser/Documents/My Games/Gas Powered Games/Supreme Commander Forged Alliance/Maps";

/// User map folders in discovery order: FA's user folder (Proton prefix next to the
/// install's `steamapps`, and under `~/.local/share/Steam`), `$XDG_DATA_HOME/farust/maps`,
/// then `$FARUST_MAPS_DIR`. Missing folders are included (callers skip them).
pub fn user_map_dirs(game_dir: &Path) -> Vec<(MapSource, PathBuf)> {
    let mut out: Vec<(MapSource, PathBuf)> = Vec::new();
    if let Some(steamapps) = game_dir.parent().and_then(Path::parent) {
        out.push((MapSource::FaUser, steamapps.join(FA_USER_MAPS)));
    }
    if let Some(home) = std::env::var_os("HOME") {
        out.push((MapSource::FaUser, PathBuf::from(home).join(".local/share/Steam/steamapps").join(FA_USER_MAPS)));
    }
    out.push((MapSource::Farust, xdg_data_home().join("farust").join("maps")));
    if let Some(d) = std::env::var_os("FARUST_MAPS_DIR") {
        out.push((MapSource::Env, PathBuf::from(d)));
    }
    let mut seen = HashSet::new();
    out.retain(|(_, p)| seen.insert(p.clone()));
    out
}

/// Maps directly inside `root`: every sub-folder holding a `*_scenario.lua`, and (when
/// `cache` is given) every `.scd` / `.zip` archive, extracted once into `cache`.
pub fn list_maps_in(root: &Path, game_dir: &Path, source: MapSource, cache: Option<&Path>) -> Result<Vec<MapFiles>, String> {
    let entries = std::fs::read_dir(root)
        .map_err(|e| format!("failed to read maps directory {}: {e}", root.display()))?;

    let mut subdirs = Vec::new();
    let mut archives = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| format!("failed to read entry in {}: {e}", root.display()))?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            subdirs.push((name, path));
        } else if ends_with_ignore_ascii_case(&name, ".scd") || ends_with_ignore_ascii_case(&name, ".zip") {
            archives.push(path);
        }
    }
    subdirs.sort();
    archives.sort();
    if let Some(cache) = cache {
        for a in archives {
            match extract_map_archive(&a, cache) {
                Ok(dirs) => subdirs.extend(dirs),
                Err(e) => eprintln!("warning: skipping map archive {}: {e}", a.display()),
            }
        }
    }

    let mut maps = Vec::with_capacity(subdirs.len());
    for (id, dir) in subdirs {
        let files = list_dir_files_sorted(&dir)?;
        let Some(scenario_path) = find_file_by_suffix(&files, "_scenario.lua") else {
            continue;
        };

        let meta = match parse_scenario_file(&scenario_path, &id) {
            Ok(m) => m,
            Err(e) if source == MapSource::Stock => return Err(e),
            Err(e) => {
                eprintln!("warning: skipping user map {}: {e}", dir.display());
                continue;
            }
        };
        let local = |vfs: &str| -> Option<PathBuf> {
            let base = vfs.replace('\\', "/");
            let base = base.rsplit('/').next().unwrap_or("");
            files
                .iter()
                .find(|(n, _)| !base.is_empty() && n.eq_ignore_ascii_case(base))
                .map(|(_, p)| p.clone())
        };
        let vfs = |vfs: &str| resolve_vfs_path(game_dir, vfs).filter(|p| p.is_file());

        let (scmap_path, save_path) = if source == MapSource::Stock {
            (
                vfs(&meta.map_vfs_path).or_else(|| find_file_by_suffix(&files, ".scmap")),
                vfs(&meta.save_vfs_path).or_else(|| find_file_by_suffix(&files, "_save.lua")),
            )
        } else {
            // User maps: prefer the map's own folder over same-named stock paths.
            (
                local(&meta.map_vfs_path)
                    .or_else(|| find_file_by_suffix(&files, ".scmap"))
                    .or_else(|| vfs(&meta.map_vfs_path)),
                local(&meta.save_vfs_path)
                    .or_else(|| find_file_by_suffix(&files, "_save.lua"))
                    .or_else(|| vfs(&meta.save_vfs_path)),
            )
        };

        maps.push(MapFiles {
            id,
            dir,
            scenario_path,
            save_path,
            scmap_path,
            source,
        });
    }

    maps.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(maps)
}

/// Extracts a zipped map (`.scd` / `.zip`) into `cache/<archive stem>/` (re-extracting when
/// the archive's size or mtime changed) and returns `(id, dir)` of every folder inside
/// that holds a `*_scenario.lua`. Entry paths are sanitized (no absolute / `..` paths).
pub fn extract_map_archive(archive: &Path, cache: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    use std::io::Read;
    let stem = archive
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .ok_or_else(|| "archive has no name".to_string())?;
    let meta = std::fs::metadata(archive).map_err(|e| e.to_string())?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let stamp = format!("{} {}", meta.len(), mtime);
    let out = cache.join(&stem);
    let stamp_path = out.join(".farust-source");
    if std::fs::read_to_string(&stamp_path).ok().as_deref() != Some(stamp.as_str()) {
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
        let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
        let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("not a zip: {e}"))?;
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
            let Some(rel) = entry.enclosed_name() else {
                continue;
            };
            let target = out.join(rel);
            if entry.is_dir() {
                std::fs::create_dir_all(&target).map_err(|e| e.to_string())?;
                continue;
            }
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let mut buf = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            std::fs::write(&target, buf).map_err(|e| e.to_string())?;
        }
        std::fs::write(&stamp_path, &stamp).map_err(|e| e.to_string())?;
    }
    let mut found = Vec::new();
    collect_scenario_dirs(&out, &stem, 0, &mut found);
    found.sort();
    Ok(found)
}

fn collect_scenario_dirs(dir: &Path, name: &str, depth: usize, out: &mut Vec<(String, PathBuf)>) {
    let Ok(files) = list_dir_files_sorted(dir) else {
        return;
    };
    if find_file_by_suffix(&files, "_scenario.lua").is_some() {
        out.push((name.to_string(), dir.to_path_buf()));
        return;
    }
    if depth >= 3 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subs: Vec<(String, PathBuf)> = rd
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .collect();
    subs.sort();
    for (n, p) in subs {
        collect_scenario_dirs(&p, &n, depth + 1, out);
    }
}

/// Finds a map by folder id (case-insensitive) among all discovered maps.
pub fn find_map(game_dir: &Path, name: &str) -> Result<MapFiles, String> {
    let maps = list_maps(game_dir)?;
    maps.into_iter()
        .find(|m| m.id.eq_ignore_ascii_case(name))
        .ok_or_else(|| format!("map '{name}' not found in {} or the user map folders", game_dir.display()))
}

pub fn parse_scenario_file(path: &Path, id: &str) -> Result<ScenarioMeta, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read scenario file {}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    parse_scenario(&text, id)
}

pub fn parse_scenario(text: &str, id: &str) -> Result<ScenarioMeta, String> {
    let mut parser = Parser::new(text)?;
    let root = parser.parse_top_level()?;
    let info = root
        .get("ScenarioInfo")
        .and_then(|v| v.as_table())
        .ok_or_else(|| format!("missing ScenarioInfo table for scenario '{id}'"))?;

    let name = info
        .get("name")
        .and_then(|v| v.as_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| id.to_string());

    let description = info
        .get("description")
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let scenario_type = info
        .get("type")
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let size = extract_size(info.get("size"));

    let map_vfs_path = info
        .get("map")
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let save_vfs_path = info
        .get("save")
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let armies = extract_armies(info);

    Ok(ScenarioMeta {
        id: id.to_string(),
        name,
        description,
        scenario_type,
        size,
        map_vfs_path,
        save_vfs_path,
        armies,
    })
}

pub fn parse_save_markers_file(path: &Path, armies: &[String]) -> Result<SaveMarkers, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read save file {}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    parse_save_markers(&text, armies)
}

/// Extracts markers exclusively from `MasterChain -> ['_MASTERCHAIN_'] -> Markers`.
pub fn parse_save_markers(text: &str, armies: &[String]) -> Result<SaveMarkers, String> {
    let mut scanner = Lexer::new(text);
    let brace_pos = scanner
        .find_table_value_start("MasterChain")?
        .ok_or_else(|| "missing MasterChain table in save file".to_string())?;

    let mut parser = Parser::new(&scanner.src[brace_pos..])?;
    let master_chain = parser.parse_table()?;

    let master_inner = master_chain
        .get("_MASTERCHAIN_")
        .and_then(|v| v.as_table())
        .ok_or_else(|| "missing '_MASTERCHAIN_' table inside MasterChain".to_string())?;

    let markers_table = master_inner
        .get("Markers")
        .and_then(|v| v.as_table())
        .ok_or_else(|| "missing 'Markers' table inside '_MASTERCHAIN_'".to_string())?;

    let mut start_spots = Vec::new();
    let mut mass_points = Vec::new();
    let mut hydro_points = Vec::new();
    let mut total_markers = 0usize;

    for (key, val) in &markers_table.entries {
        let Some(marker_name) = key.as_ref() else {
            continue;
        };
        let Some(marker_table) = val.as_table() else {
            continue;
        };

        let marker_type = marker_table
            .get("type")
            .and_then(|v| v.as_string())
            .unwrap_or_default();

        let position = marker_table
            .get("position")
            .and_then(|v| v.as_vec3())
            .unwrap_or([0.0, 0.0, 0.0]);

        total_markers += 1;

        let is_start =
            marker_name.starts_with("ARMY_") || armies.iter().any(|a| a == marker_name);

        if is_start {
            start_spots.push(Marker {
                name: marker_name.clone(),
                marker_type,
                position,
            });
        } else if marker_type == "Mass" {
            mass_points.push(Marker {
                name: marker_name.clone(),
                marker_type,
                position,
            });
        } else if marker_type == "Hydrocarbon" {
            hydro_points.push(Marker {
                name: marker_name.clone(),
                marker_type,
                position,
            });
        }
    }

    start_spots.sort_by(|a, b| a.name.cmp(&b.name));
    mass_points.sort_by(|a, b| a.name.cmp(&b.name));
    hydro_points.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(SaveMarkers {
        start_spots,
        mass_points,
        hydro_points,
        total_markers,
    })
}

/// Directory edited maps are written to: `$XDG_DATA_HOME/farust/maps`.
pub fn farust_maps_dir() -> PathBuf {
    xdg_data_home().join("farust").join("maps")
}

/// Replaces every ASCII-case-insensitive occurrence of `old_id` with `new_id`
/// in scenario/save/script text when cloning a map under a new id.
pub fn rewrite_map_id(text: &str, old_id: &str, new_id: &str) -> String {
    if old_id.is_empty() || old_id.eq_ignore_ascii_case(new_id) && old_id == new_id {
        return text.to_string();
    }
    let needle = old_id.as_bytes();
    let hay = text.as_bytes();
    if needle.len() > hay.len() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < hay.len() {
        if i + needle.len() <= hay.len() && hay[i..i + needle.len()].eq_ignore_ascii_case(needle) {
            out.push_str(new_id);
            i += needle.len();
        } else {
            let ch = text[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Rewrites a `_scenario.lua` file for a cloned/edited map under `new_id`,
/// updating VFS paths (`map`, `save`, `script`) and optionally the display `name`.
pub fn rewrite_scenario_for_clone(
    text: &str,
    old_id: &str,
    new_id: &str,
    new_name: Option<&str>,
) -> String {
    let mut out = rewrite_map_id(text, old_id, new_id);
    // Normalize `map = '...'`, `save = '...'`, `script = '...'` lines in case the
    // source scenario referenced differently named files.
    let targets = [
        ("map", format!("/maps/{new_id}/{new_id}.scmap")),
        ("save", format!("/maps/{new_id}/{new_id}_save.lua")),
        ("script", format!("/maps/{new_id}/{new_id}_script.lua")),
    ];
    let mut lines: Vec<String> = Vec::new();
    for line in out.split_inclusive('\n') {
        let trimmed = line.trim_start();
        let indent_len = line.len() - trimmed.len();
        let indent = &line[..indent_len];
        let eol = if line.ends_with("\r\n") {
            "\r\n"
        } else if line.ends_with('\n') {
            "\n"
        } else {
            ""
        };
        let mut replaced = None;
        for (key, val) in &targets {
            if let Some(rest) = trimmed.strip_prefix(key) {
                let after = rest.trim_start();
                if after.starts_with('=') {
                    let comma = if trimmed.trim_end().ends_with(',') { "," } else { "" };
                    replaced = Some(format!("{indent}{key} = '{val}'{comma}{eol}"));
                    break;
                }
            }
        }
        if replaced.is_none()
            && let Some(title) = new_name
            && let Some(rest) = trimmed.strip_prefix("name")
            && rest.trim_start().starts_with('=')
        {
            let escaped = title.replace('\\', "\\\\").replace('\'', "\\'");
            let comma = if trimmed.trim_end().ends_with(',') { "," } else { "" };
            replaced = Some(format!("{indent}name = '{escaped}'{comma}{eol}"));
        }
        lines.push(replaced.unwrap_or_else(|| line.to_string()));
    }
    out = lines.concat();
    out
}

fn scan_matching_brace(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = open;
    while i < bytes.len() {
        match bytes[i] {
            b'-' if bytes.get(i + 1) == Some(&b'-') => {
                // Lua comment: long `--[[ ]]` or to end of line.
                let mut j = i + 2;
                if bytes.get(j) == Some(&b'[') {
                    let mut eq = 0usize;
                    j += 1;
                    while bytes.get(j) == Some(&b'=') {
                        eq += 1;
                        j += 1;
                    }
                    if bytes.get(j) == Some(&b'[') {
                        let close: Vec<u8> = std::iter::once(b']')
                            .chain(std::iter::repeat_n(b'=', eq))
                            .chain(*b"]")
                            .collect();
                        let mut k = j + 1;
                        let mut found = None;
                        while k + close.len() <= bytes.len() {
                            if &bytes[k..k + close.len()] == close.as_slice() {
                                found = Some(k + close.len());
                                break;
                            }
                            k += 1;
                        }
                        i = found?;
                        continue;
                    }
                }
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'\'' | b'"' => {
                let q = bytes[i];
                i += 1;
                while i < bytes.len() && bytes[i] != q {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
            }
            b'{' => {
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth -= 1;
                i += 1;
                if depth == 0 {
                    return Some(i - 1);
                }
            }
            _ => i += 1,
        }
    }
    None
}

fn skip_ws_and_lua_comments(bytes: &[u8], mut i: usize, end: usize) -> usize {
    while i < end {
        if bytes[i].is_ascii_whitespace() || bytes[i] == b',' || bytes[i] == b';' {
            i += 1;
            continue;
        }
        if bytes[i] == b'-' && bytes.get(i + 1) == Some(&b'-') {
            let mut j = i + 2;
            if bytes.get(j) == Some(&b'[') {
                let mut eq = 0usize;
                j += 1;
                while j < end && bytes[j] == b'=' {
                    eq += 1;
                    j += 1;
                }
                if j < end && bytes[j] == b'[' {
                    let close: Vec<u8> = std::iter::once(b']')
                        .chain(std::iter::repeat_n(b'=', eq))
                        .chain(*b"]")
                        .collect();
                    let mut k = j + 1;
                    let mut found = end;
                    while k + close.len() <= end {
                        if &bytes[k..k + close.len()] == close.as_slice() {
                            found = k + close.len();
                            break;
                        }
                        k += 1;
                    }
                    i = found;
                    continue;
                }
            }
            while i < end && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        break;
    }
    i
}

fn marker_entry(name: &str, marker_type: &str, pos: [f32; 3]) -> String {
    let escaped = name.replace('\\', "\\\\").replace('\'', "\\'");
    let (color, prop, extra) = match marker_type {
        "Mass" => (
            "ff808080",
            "/env/common/props/markers/M_Mass_prop.bp",
            "                    ['size'] = FLOAT( 1.000000 ),\n                    ['resource'] = BOOLEAN( true ),\n                    ['amount'] = FLOAT( 100.000000 ),\n                    ['editorIcon'] = STRING( '/textures/editor/marker_mass.bmp' ),\n",
        ),
        "Hydrocarbon" => (
            "ff808080",
            "/env/common/props/markers/M_Hydrocarbon_prop.bp",
            "                    ['size'] = FLOAT( 1.000000 ),\n                    ['resource'] = BOOLEAN( true ),\n                    ['amount'] = FLOAT( 100.000000 ),\n                    ['editorIcon'] = STRING( '/textures/editor/marker_hydro.bmp' ),\n",
        ),
        _ => ("ff800080", "/env/common/props/markers/M_Blank_prop.bp", ""),
    };
    format!(
        "                ['{escaped}'] = {{\n{extra}                    ['color'] = STRING( '{color}' ),\n                    ['type'] = STRING( '{marker_type}' ),\n                    ['prop'] = STRING( '{prop}' ),\n                    ['orientation'] = VECTOR3( 0, 0, 0 ),\n                    ['position'] = VECTOR3( {:.4}, {:.4}, {:.4} ),\n                }},\n",
        pos[0], pos[1], pos[2]
    )
}

/// Regenerates the start/resource markers of a `_save.lua` text from the
/// given lists. All other markers and all text outside the `Markers` table
/// are kept verbatim.
pub fn rewrite_save_markers(
    text: &str,
    armies: &[String],
    starts: &[Marker],
    mass: &[Marker],
    hydro: &[Marker],
) -> Result<String, String> {
    let src = text.strip_prefix('\u{FEFF}').unwrap_or(text);
    let prefix_len = text.len() - src.len();
    let bytes = src.as_bytes();

    // Locate `MasterChain = { ... ['_MASTERCHAIN_'] = { ... Markers = { ... } } }`
    // using the comment/string-aware Lua lexer.
    let mc_open = Lexer::new(src)
        .find_table_value_start("MasterChain")?
        .ok_or_else(|| "missing MasterChain in save text".to_string())?;
    let mc_close = scan_matching_brace(bytes, mc_open)
        .ok_or_else(|| "unterminated MasterChain table".to_string())?;
    let inner_open = Lexer::new(&src[mc_open + 1..mc_close])
        .find_table_value_start("_MASTERCHAIN_")?
        .map(|p| mc_open + 1 + p)
        .ok_or_else(|| "missing '_MASTERCHAIN_' in MasterChain".to_string())?;
    let inner_close = scan_matching_brace(bytes, inner_open)
        .ok_or_else(|| "unterminated '_MASTERCHAIN_' table".to_string())?;
    let open = Lexer::new(&src[inner_open + 1..inner_close])
        .find_table_value_start("Markers")?
        .map(|p| inner_open + 1 + p)
        .ok_or_else(|| "missing Markers in '_MASTERCHAIN_'".to_string())?;
    let close = scan_matching_brace(bytes, open)
        .ok_or_else(|| "unterminated Markers table".to_string())?;

    // Split the interior into top-level `['name'] = { ... },` or `name = { ... },` entries.
    let mut kept: Vec<String> = Vec::new();
    let mut i = open + 1;
    while i < close {
        i = skip_ws_and_lua_comments(bytes, i, close);
        if i >= close {
            break;
        }
        let entry_start = i;
        // Parse key name (`['name']`, `["name"]`, or bare identifier).
        let name = if bytes[i] == b'[' {
            let mut lex = Lexer::new(&src[i + 1..close]);
            let _ = lex.skip_whitespace_and_comments();
            let k = match lex.next_token().ok().flatten() {
                Some(Token::Str(s)) => s,
                Some(Token::Ident(s)) => s,
                Some(Token::Num(n)) => n.to_string(),
                _ => String::new(),
            };
            let _ = lex.skip_whitespace_and_comments();
            if bytes.get(i + 1 + lex.pos) == Some(&b']') {
                i = i + 1 + lex.pos + 1;
            }
            k
        } else if is_ident_start(bytes[i]) {
            let ks = i;
            i += 1;
            while i < close && is_ident_continue(bytes[i]) {
                i += 1;
            }
            src[ks..i].to_string()
        } else {
            String::new()
        };

        // Advance past `=` to the entry's opening `{`.
        i = skip_ws_and_lua_comments(bytes, i, close);
        if i < close && bytes[i] == b'=' {
            i += 1;
            i = skip_ws_and_lua_comments(bytes, i, close);
        }
        let (val_open, entry_end) = if i < close && bytes[i] == b'{' {
            let eopen = i;
            match scan_matching_brace(bytes, eopen) {
                Some(eclose) => {
                    let mut e = eclose + 1;
                    if bytes.get(e) == Some(&b',') {
                        e += 1;
                    }
                    (Some((eopen, eclose)), e)
                }
                None => (None, close),
            }
        } else {
            // Non-table fallback: advance to next comma or end.
            let mut e = i;
            while e < close && bytes[e] != b',' {
                e += 1;
            }
            if e < close && bytes[e] == b',' {
                e += 1;
            }
            (None, e)
        };
        if entry_end <= entry_start {
            break;
        }
        let entry = &src[entry_start..entry_end];
        let ty = val_open
            .and_then(|(vo, vc)| Parser::new(&src[vo..=vc]).ok()?.parse_table().ok())
            .and_then(|tbl| tbl.get("type").and_then(|v| v.as_string()))
            .unwrap_or_default();
        let is_start = name.starts_with("ARMY_") || armies.iter().any(|a| a == &name);
        if !is_start && ty != "Mass" && ty != "Hydrocarbon" {
            kept.push(format!("                {}", entry.trim_start()));
            if !entry.trim_end().ends_with(',') {
                kept.last_mut().unwrap().push(',');
            }
            kept.last_mut().unwrap().push('\n');
        }
        i = entry_end;
    }

    let mut interior = String::new();
    interior.push('\n');
    for m in starts {
        interior.push_str(&marker_entry(&m.name, "Blank Marker", m.position));
    }
    for m in mass {
        interior.push_str(&marker_entry(&m.name, "Mass", m.position));
    }
    for m in hydro {
        interior.push_str(&marker_entry(&m.name, "Hydrocarbon", m.position));
    }
    for k in kept {
        interior.push_str(&k);
    }
    interior.push_str("            ");
    Ok(format!(
        "{}{}{}{}",
        &text[..prefix_len],
        &src[..=open],
        interior,
        &src[close..]
    ))
}

fn list_dir_files_sorted(dir: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read directory {}: {e}", dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|e| format!("failed to read entry in {}: {e}", dir.display()))?;
        let path = entry.path();
        if path.is_file() {
            let name = entry.file_name().to_string_lossy().into_owned();
            files.push((name, path));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn find_file_by_suffix(files: &[(String, PathBuf)], suffix: &str) -> Option<PathBuf> {
    files.iter().find_map(|(name, path)| {
        if ends_with_ignore_ascii_case(name, suffix) {
            Some(path.clone())
        } else {
            None
        }
    })
}

fn ends_with_ignore_ascii_case(s: &str, suffix: &str) -> bool {
    let sb = s.as_bytes();
    let tb = suffix.as_bytes();
    sb.len() >= tb.len() && sb[sb.len() - tb.len()..].eq_ignore_ascii_case(tb)
}

fn extract_size(val: Option<&LuaVal>) -> (u32, u32) {
    let Some(table) = val.and_then(|v| v.as_table()) else {
        return (0, 0);
    };
    let mut dims = [0u32; 2];
    let mut count = 0;
    for (_, entry_val) in &table.entries {
        if let Some(n) = entry_val.as_f64() {
            dims[count] = n.max(0.0).round() as u32;
            count += 1;
            if count == 2 {
                return (dims[0], dims[1]);
            }
        }
    }
    (0, 0)
}

fn extract_armies(info: &LuaTable) -> Vec<String> {
    let mut armies = Vec::new();
    let mut seen = HashSet::new();

    let Some(teams) = info
        .get("Configurations")
        .and_then(|v| v.as_table())
        .and_then(|cfg| cfg.get("standard"))
        .and_then(|v| v.as_table())
        .and_then(|std_cfg| std_cfg.get("teams"))
        .and_then(|v| v.as_table())
    else {
        return armies;
    };

    for (_, team_val) in &teams.entries {
        let Some(team_table) = team_val.as_table() else {
            continue;
        };
        let Some(team_armies) = team_table.get("armies").and_then(|v| v.as_table()) else {
            continue;
        };
        for (_, army_val) in &team_armies.entries {
            if let Some(army_name) = army_val.as_string()
                && seen.insert(army_name.clone())
            {
                armies.push(army_name);
            }
        }
    }

    armies
}

#[derive(Debug, Clone)]
struct LuaTable {
    entries: Vec<(Option<String>, LuaVal)>,
}

impl LuaTable {
    fn get(&self, key: &str) -> Option<&LuaVal> {
        self.entries
            .iter()
            .rev()
            .find_map(|(k, v)| match k.as_deref() {
                Some(name) if name == key => Some(v),
                _ => None,
            })
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum LuaVal {
    Nil,
    Bool(bool),
    Num(f64),
    Str(String),
    Ident(String),
    Table(LuaTable),
    Call(String, Vec<LuaVal>),
}

impl LuaVal {
    fn as_string(&self) -> Option<String> {
        match self {
            LuaVal::Str(s) => Some(s.clone()),
            LuaVal::Call(name, args) if name.eq_ignore_ascii_case("STRING") => {
                args.first().and_then(|v| v.as_string())
            }
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            LuaVal::Num(n) => Some(*n),
            LuaVal::Call(name, args) if name.eq_ignore_ascii_case("FLOAT") => {
                args.first().and_then(|v| v.as_f64())
            }
            _ => None,
        }
    }

    fn as_table(&self) -> Option<&LuaTable> {
        match self {
            LuaVal::Table(t) => Some(t),
            LuaVal::Call(_, args) if args.len() == 1 => args[0].as_table(),
            _ => None,
        }
    }

    fn as_vec3(&self) -> Option<[f32; 3]> {
        match self {
            LuaVal::Call(name, args)
                if name.eq_ignore_ascii_case("VECTOR3") && args.len() >= 3 =>
            {
                Some([
                    args[0].as_f64()? as f32,
                    args[1].as_f64()? as f32,
                    args[2].as_f64()? as f32,
                ])
            }
            LuaVal::Table(t) => {
                let mut nums = Vec::with_capacity(3);
                for (_, v) in &t.entries {
                    if let Some(n) = v.as_f64() {
                        nums.push(n as f32);
                        if nums.len() == 3 {
                            return Some([nums[0], nums[1], nums[2]]);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Str(String),
    Num(f64),
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Eq,
    Comma,
    Semi,
    Dot,
    DotDot,
    Minus,
    Plus,
}

struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        let src = input.strip_prefix('\u{FEFF}').unwrap_or(input);
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn parse_long_bracket(&self, start: usize) -> Option<(usize, usize)> {
        if *self.bytes.get(start)? != b'[' {
            return None;
        }
        let mut i = start + 1;
        while self.bytes.get(i) == Some(&b'=') {
            i += 1;
        }
        if self.bytes.get(i) == Some(&b'[') {
            Some((i - start - 1, i + 1))
        } else {
            None
        }
    }

    fn scan_long_bracket_end(&self, content_start: usize, eq_count: usize) -> Option<(usize, usize)> {
        let mut i = content_start;
        while i < self.bytes.len() {
            if self.bytes[i] == b']' {
                let mut j = i + 1;
                let mut eq = 0;
                while eq < eq_count && self.bytes.get(j) == Some(&b'=') {
                    j += 1;
                    eq += 1;
                }
                if eq == eq_count && self.bytes.get(j) == Some(&b']') {
                    return Some((i, j + 1));
                }
            }
            i += 1;
        }
        None
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<(), String> {
        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos >= self.bytes.len() {
                return Ok(());
            }
            if self.bytes[self.pos] == b'#' {
                self.pos += 1;
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            if self.bytes[self.pos] == b'-' && self.bytes.get(self.pos + 1) == Some(&b'-') {
                self.pos += 2;
                if let Some((eq_count, content_start)) = self.parse_long_bracket(self.pos) {
                    let (_, after_close) = self
                        .scan_long_bracket_end(content_start, eq_count)
                        .ok_or_else(|| "unterminated Lua block comment".to_string())?;
                    self.pos = after_close;
                } else {
                    while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                continue;
            }
            return Ok(());
        }
    }

    fn skip_quoted_string(&mut self, quote: u8) -> Result<(), String> {
        self.pos += 1;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == quote {
                self.pos += 1;
                return Ok(());
            }
            if b == b'\\' {
                self.pos += 1;
                let ch = self.src[self.pos..]
                    .chars()
                    .next()
                    .ok_or_else(|| "unterminated escape in Lua string".to_string())?;
                self.pos += ch.len_utf8();
                continue;
            }
            if b < 0x80 {
                self.pos += 1;
            } else if let Some(ch) = self.src[self.pos..].chars().next() {
                self.pos += ch.len_utf8();
            }
        }
        Err("unterminated Lua string".to_string())
    }

    fn read_quoted_string(&mut self, quote: u8) -> Result<String, String> {
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == quote {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let escaped = self.src[self.pos..]
                    .chars()
                    .next()
                    .ok_or_else(|| "unterminated escape in Lua string".to_string())?;
                self.pos += escaped.len_utf8();
                match escaped {
                    '\\' => out.push('\\'),
                    '\'' => out.push('\''),
                    '"' => out.push('"'),
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'a' => out.push('\x07'),
                    'b' => out.push('\x08'),
                    'f' => out.push('\x0C'),
                    'v' => out.push('\x0B'),
                    '\n' => out.push('\n'),
                    '\r' => {
                        if self.bytes.get(self.pos) == Some(&b'\n') {
                            self.pos += 1;
                        }
                        out.push('\n');
                    }
                    '0'..='9' => {
                        let mut val = (escaped as u8 - b'0') as u32;
                        for _ in 0..2 {
                            if let Some(&d) = self.bytes.get(self.pos)
                                && d.is_ascii_digit()
                            {
                                let next_val = val * 10 + (d - b'0') as u32;
                                if next_val <= 255 {
                                    val = next_val;
                                    self.pos += 1;
                                    continue;
                                }
                            }
                            break;
                        }
                        if let Some(c) = char::from_u32(val) {
                            out.push(c);
                        }
                    }
                    other => out.push(other),
                }
                continue;
            }
            if b < 0x80 {
                out.push(b as char);
                self.pos += 1;
            } else if let Some(ch) = self.src[self.pos..].chars().next() {
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }
        Err("unterminated Lua string".to_string())
    }

    fn read_long_string(&mut self, eq_count: usize, content_start: usize) -> Result<String, String> {
        let (content_end, after_close) = self
            .scan_long_bracket_end(content_start, eq_count)
            .ok_or_else(|| "unterminated Lua long string".to_string())?;
        let mut s = &self.src[content_start..content_end];
        if let Some(rest) = s.strip_prefix("\r\n") {
            s = rest;
        } else if let Some(rest) = s.strip_prefix('\n') {
            s = rest;
        }
        self.pos = after_close;
        Ok(s.to_string())
    }

    fn read_number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        if self.bytes[self.pos] == b'0'
            && matches!(self.bytes.get(self.pos + 1), Some(b'x' | b'X'))
        {
            self.pos += 2;
            let hex_start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_hexdigit() {
                self.pos += 1;
            }
            if self.pos == hex_start {
                return Err("invalid hex literal in Lua".to_string());
            }
            let hex_str = &self.src[hex_start..self.pos];
            let val = u64::from_str_radix(hex_str, 16)
                .map_err(|e| format!("invalid hex literal '{hex_str}': {e}"))?;
            return Ok(val as f64);
        }

        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.bytes.get(self.pos) == Some(&b'.') && self.bytes.get(self.pos + 1) != Some(&b'.') {
            self.pos += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exp_start = self.pos;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos == exp_start {
                return Err("invalid exponent in Lua number".to_string());
            }
        }

        let num_str = &self.src[start..self.pos];
        num_str
            .parse::<f64>()
            .map_err(|e| format!("invalid number '{num_str}': {e}"))
    }

    fn next_token(&mut self) -> Result<Option<Token>, String> {
        self.skip_whitespace_and_comments()?;
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let b = self.bytes[self.pos];
        match b {
            b'{' => {
                self.pos += 1;
                Ok(Some(Token::LBrace))
            }
            b'}' => {
                self.pos += 1;
                Ok(Some(Token::RBrace))
            }
            b']' => {
                self.pos += 1;
                Ok(Some(Token::RBracket))
            }
            b'(' => {
                self.pos += 1;
                Ok(Some(Token::LParen))
            }
            b')' => {
                self.pos += 1;
                Ok(Some(Token::RParen))
            }
            b'=' => {
                self.pos += 1;
                Ok(Some(Token::Eq))
            }
            b',' => {
                self.pos += 1;
                Ok(Some(Token::Comma))
            }
            b';' => {
                self.pos += 1;
                Ok(Some(Token::Semi))
            }
            b'+' => {
                self.pos += 1;
                Ok(Some(Token::Plus))
            }
            b'-' => {
                self.pos += 1;
                Ok(Some(Token::Minus))
            }
            b'\'' | b'"' => Ok(Some(Token::Str(self.read_quoted_string(b)?))),
            b'[' => {
                if let Some((eq_count, content_start)) = self.parse_long_bracket(self.pos) {
                    Ok(Some(Token::Str(
                        self.read_long_string(eq_count, content_start)?,
                    )))
                } else {
                    self.pos += 1;
                    Ok(Some(Token::LBracket))
                }
            }
            b'.' => {
                if self.bytes.get(self.pos + 1) == Some(&b'.') {
                    self.pos += 2;
                    Ok(Some(Token::DotDot))
                } else if self.bytes.get(self.pos + 1).is_some_and(|c| c.is_ascii_digit()) {
                    Ok(Some(Token::Num(self.read_number()?)))
                } else {
                    self.pos += 1;
                    Ok(Some(Token::Dot))
                }
            }
            _ if b.is_ascii_digit() => Ok(Some(Token::Num(self.read_number()?))),
            _ if is_ident_start(b) => {
                let start = self.pos;
                self.pos += 1;
                while self.pos < self.bytes.len() && is_ident_continue(self.bytes[self.pos]) {
                    self.pos += 1;
                }
                Ok(Some(Token::Ident(self.src[start..self.pos].to_string())))
            }
            _ => {
                let ch = self.src[self.pos..].chars().next().unwrap_or('\0');
                Err(format!("unexpected character '{ch}' at byte {}", self.pos))
            }
        }
    }

    fn consume_eq_open_brace(&mut self) -> Result<Option<usize>, String> {
        self.skip_whitespace_and_comments()?;
        if self.bytes.get(self.pos) == Some(&b'=') {
            self.pos += 1;
            self.skip_whitespace_and_comments()?;
            if self.bytes.get(self.pos) == Some(&b'{') {
                return Ok(Some(self.pos));
            }
        }
        Ok(None)
    }

    fn find_table_value_start(&mut self, target_key: &str) -> Result<Option<usize>, String> {
        loop {
            self.skip_whitespace_and_comments()?;
            if self.pos >= self.bytes.len() {
                return Ok(None);
            }
            let b = self.bytes[self.pos];
            if b == b'\'' || b == b'"' {
                self.skip_quoted_string(b)?;
                continue;
            }
            if let Some((eq_count, content_start)) = self.parse_long_bracket(self.pos) {
                let (_, after_close) = self
                    .scan_long_bracket_end(content_start, eq_count)
                    .ok_or_else(|| "unterminated Lua long string".to_string())?;
                self.pos = after_close;
                continue;
            }
            if b == b'[' {
                self.pos += 1;
                self.skip_whitespace_and_comments()?;
                if self.pos < self.bytes.len()
                    && (self.bytes[self.pos] == b'\'' || self.bytes[self.pos] == b'"')
                {
                    let q = self.bytes[self.pos];
                    let s = self.read_quoted_string(q)?;
                    if s == target_key {
                        self.skip_whitespace_and_comments()?;
                        if self.bytes.get(self.pos) == Some(&b']') {
                            self.pos += 1;
                            if let Some(brace_pos) = self.consume_eq_open_brace()? {
                                return Ok(Some(brace_pos));
                            }
                        }
                    }
                }
                continue;
            }
            if is_ident_start(b) {
                let start = self.pos;
                self.pos += 1;
                while self.pos < self.bytes.len() && is_ident_continue(self.bytes[self.pos]) {
                    self.pos += 1;
                }
                if &self.src[start..self.pos] == target_key
                    && let Some(brace_pos) = self.consume_eq_open_brace()?
                {
                    return Ok(Some(brace_pos));
                }
                continue;
            }
            if b.is_ascii_digit()
                || (b == b'.' && self.bytes.get(self.pos + 1).is_some_and(|c| c.is_ascii_digit()))
            {
                let _ = self.read_number()?;
                continue;
            }
            if b < 0x80 {
                self.pos += 1;
            } else if let Some(ch) = self.src[self.pos..].chars().next() {
                self.pos += ch.len_utf8();
            }
        }
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

struct Parser<'a> {
    lexer: Lexer<'a>,
    cur: Option<Token>,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Result<Self, String> {
        let mut lexer = Lexer::new(src);
        let cur = lexer.next_token()?;
        Ok(Self { lexer, cur })
    }

    fn advance(&mut self) -> Result<Option<Token>, String> {
        let next = self.lexer.next_token()?;
        Ok(std::mem::replace(&mut self.cur, next))
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        if self.cur.as_ref() == Some(expected) {
            self.advance()?;
            Ok(())
        } else {
            Err(format!("expected {expected:?}, found {:?}", self.cur))
        }
    }

    fn consume_dotted_ident(&mut self, name: &mut String) -> Result<(), String> {
        while self.cur == Some(Token::Dot) {
            self.advance()?;
            match self.cur.clone() {
                Some(Token::Ident(sub)) => {
                    self.advance()?;
                    name.push('.');
                    name.push_str(&sub);
                }
                other => {
                    return Err(format!("expected identifier after '.', found {other:?}"));
                }
            }
        }
        Ok(())
    }

    fn parse_top_level(&mut self) -> Result<LuaTable, String> {
        let mut entries = Vec::new();
        while let Some(tok) = self.cur.clone() {
            match tok {
                Token::Semi => {
                    self.advance()?;
                }
                Token::Ident(mut name) => {
                    self.advance()?;
                    if name == "local" {
                        if let Some(Token::Ident(local_name)) = self.cur.clone() {
                            self.advance()?;
                            if self.cur == Some(Token::Eq) {
                                self.advance()?;
                                let val = self.parse_expr()?;
                                entries.push((Some(local_name), val));
                            }
                        }
                        continue;
                    }
                    self.consume_dotted_ident(&mut name)?;
                    if self.cur == Some(Token::Eq) {
                        self.advance()?;
                        let val = self.parse_expr()?;
                        entries.push((Some(name), val));
                    } else if self.cur == Some(Token::LParen) || self.cur == Some(Token::LBrace) {
                        let _ = self.parse_ident_postfix(name)?;
                    } else {
                        return Err(format!(
                            "expected '=' after top-level identifier '{name}'"
                        ));
                    }
                }
                other => {
                    return Err(format!("unexpected top-level token: {other:?}"));
                }
            }
        }
        Ok(LuaTable { entries })
    }

    fn parse_table(&mut self) -> Result<LuaTable, String> {
        self.expect(&Token::LBrace)?;
        let mut entries = Vec::new();
        loop {
            if self.cur == Some(Token::RBrace) {
                self.advance()?;
                break;
            }
            if self.cur.is_none() {
                return Err("unterminated Lua table '{'".to_string());
            }

            let entry = match self.cur.clone() {
                Some(Token::LBracket) => {
                    self.advance()?;
                    let key = self.parse_bracket_key()?;
                    self.expect(&Token::RBracket)?;
                    self.expect(&Token::Eq)?;
                    let val = self.parse_expr()?;
                    (Some(key), val)
                }
                Some(Token::Ident(name)) if !matches!(name.as_str(), "true" | "false" | "nil") => {
                    self.advance()?;
                    if self.cur == Some(Token::Eq) {
                        self.advance()?;
                        let val = self.parse_expr()?;
                        (Some(name), val)
                    } else {
                        let primary = self.parse_ident_postfix(name)?;
                        let val = self.parse_expr_tail(primary)?;
                        (None, val)
                    }
                }
                _ => {
                    let val = self.parse_expr()?;
                    (None, val)
                }
            };
            entries.push(entry);

            match self.cur {
                Some(Token::Comma | Token::Semi) => {
                    self.advance()?;
                }
                Some(Token::RBrace) => {
                    self.advance()?;
                    break;
                }
                _ => {
                    return Err(format!(
                        "expected ',', ';', or '}}' in Lua table, found {:?}",
                        self.cur
                    ));
                }
            }
        }
        Ok(LuaTable { entries })
    }

    fn parse_bracket_key(&mut self) -> Result<String, String> {
        let val = self.parse_expr()?;
        match val {
            LuaVal::Str(s) => Ok(s),
            LuaVal::Num(n) => Ok(n.to_string()),
            LuaVal::Ident(s) => Ok(s),
            LuaVal::Bool(b) => Ok(b.to_string()),
            other => Err(format!("unsupported table bracket key: {other:?}")),
        }
    }

    fn parse_expr(&mut self) -> Result<LuaVal, String> {
        let primary = self.parse_primary()?;
        self.parse_expr_tail(primary)
    }

    fn parse_expr_tail(&mut self, mut lhs: LuaVal) -> Result<LuaVal, String> {
        while self.cur == Some(Token::DotDot) {
            self.advance()?;
            let rhs = self.parse_primary()?;
            lhs = match (lhs, rhs) {
                (LuaVal::Str(mut a), LuaVal::Str(b)) => {
                    a.push_str(&b);
                    LuaVal::Str(a)
                }
                (LuaVal::Str(mut a), LuaVal::Num(n)) => {
                    a.push_str(&n.to_string());
                    LuaVal::Str(a)
                }
                (LuaVal::Num(n), LuaVal::Str(b)) => LuaVal::Str(format!("{n}{b}")),
                (a, b) => match (a.as_string(), b.as_string()) {
                    (Some(mut sa), Some(sb)) => {
                        sa.push_str(&sb);
                        LuaVal::Str(sa)
                    }
                    _ => return Err("cannot concatenate non-string Lua values".to_string()),
                },
            };
        }
        Ok(lhs)
    }

    fn parse_primary(&mut self) -> Result<LuaVal, String> {
        match self.cur.clone() {
            Some(Token::Str(s)) => {
                self.advance()?;
                Ok(LuaVal::Str(s))
            }
            Some(Token::Num(n)) => {
                self.advance()?;
                Ok(LuaVal::Num(n))
            }
            Some(Token::Minus) => {
                self.advance()?;
                match self.parse_primary()? {
                    LuaVal::Num(n) => Ok(LuaVal::Num(-n)),
                    other => Err(format!("cannot apply unary '-' to {other:?}")),
                }
            }
            Some(Token::Plus) => {
                self.advance()?;
                match self.parse_primary()? {
                    LuaVal::Num(n) => Ok(LuaVal::Num(n)),
                    other => Err(format!("cannot apply unary '+' to {other:?}")),
                }
            }
            Some(Token::LBrace) => Ok(LuaVal::Table(self.parse_table()?)),
            Some(Token::LParen) => {
                self.advance()?;
                let val = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(val)
            }
            Some(Token::Ident(name)) => {
                self.advance()?;
                match name.as_str() {
                    "true" => Ok(LuaVal::Bool(true)),
                    "false" => Ok(LuaVal::Bool(false)),
                    "nil" => Ok(LuaVal::Nil),
                    _ => self.parse_ident_postfix(name),
                }
            }
            other => Err(format!("unexpected token in expression: {other:?}")),
        }
    }

    fn parse_ident_postfix(&mut self, mut name: String) -> Result<LuaVal, String> {
        self.consume_dotted_ident(&mut name)?;
        if self.cur == Some(Token::LParen) {
            self.advance()?;
            let mut args = Vec::new();
            if self.cur != Some(Token::RParen) {
                loop {
                    args.push(self.parse_expr()?);
                    if self.cur == Some(Token::Comma) {
                        self.advance()?;
                        if self.cur == Some(Token::RParen) {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Token::RParen)?;
            Ok(LuaVal::Call(name, args))
        } else if self.cur == Some(Token::LBrace) {
            let table = self.parse_table()?;
            Ok(LuaVal::Call(name, vec![LuaVal::Table(table)]))
        } else {
            Ok(LuaVal::Ident(name))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TempDirGuard(PathBuf);

    impl TempDirGuard {
        fn new(prefix: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let dir = std::env::temp_dir().join(format!("farust_scenario_{prefix}_{pid}_{seq}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("create temp test directory");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn parse_scenario_standard_and_helpers() {
        let lua = r#"
--[[                                                                           ]]--
--[[  File     :  /maps/SCMP_001/SCMP_001_scenario.lua                         ]]--
--[[                                                                           ]]--
# optional hash comment
version = 3 -- trailing line comment
ScenarioInfo = {
    name = STRING("Seton's Clutch"),
    description = "Line1\n\t\"Quoted\" \'Single\' \\Slash\r",
    preview = '',
    map_version = 1,
    type = 'skirmish',
    starts = true,
    size = {1024, 1024},
    reclaim = {0, 0},
    map = '/maps/SCMP_001/SCMP_001.scmap',
    save = '/maps/SCMP_001/SCMP_001_save.lua',
    Configurations = {
        ['standard'] = {
            teams = {
                {
                    name = 'FFA',
                    armies = {'ARMY_1', 'ARMY_2'},
                },
                {
                    name = 'EXTRA',
                    armies = {'ARMY_2', 'ARMY_3'},
                },
            },
        },
    },
}
"#;
        let meta = parse_scenario(lua, "SCMP_001").expect("parse scenario");
        assert_eq!(meta.id, "SCMP_001");
        assert_eq!(meta.name, "Seton's Clutch");
        assert_eq!(meta.description, "Line1\n\t\"Quoted\" 'Single' \\Slash\r");
        assert_eq!(meta.scenario_type, "skirmish");
        assert_eq!(meta.size, (1024, 1024));
        assert_eq!(meta.map_vfs_path, "/maps/SCMP_001/SCMP_001.scmap");
        assert_eq!(meta.save_vfs_path, "/maps/SCMP_001/SCMP_001_save.lua");
        assert_eq!(meta.armies, vec!["ARMY_1", "ARMY_2", "ARMY_3"]);
    }

    #[test]
    fn parse_scenario_name_fallback_and_lua_syntax() {
        let lua = r#"
version = 3;
ScenarioInfo = {
    ["description"] = [[Raw \n multiline
description]],
    type = 'campaign',
    size = { [1] = 256, [2] = 512 },
    map = '/maps/X1CA_001/X1CA_001.scmap',
    save = '/maps/X1CA_001/X1CA_001_save.lua',
    extra_rect = RECTANGLE(0, 0, 256, 512),
    extra_group = GROUP { count = FLOAT(3.5), enabled = BOOLEAN(false) },
    joined = 'PartA' .. 'PartB',
    Configurations = {
        ['standard'] = {
            teams = {
                [1] = {
                    name = 'FFA',
                    armies = { [1] = 'Player', [2] = 'Aeon' },
                },
            },
        },
    },
}
"#;
        let meta = parse_scenario(lua, "X1CA_001").expect("parse scenario");
        assert_eq!(meta.name, "X1CA_001");
        assert_eq!(meta.description, "Raw \\n multiline\ndescription");
        assert_eq!(meta.scenario_type, "campaign");
        assert_eq!(meta.size, (256, 512));
        assert_eq!(meta.armies, vec!["Player", "Aeon"]);
    }

    #[test]
    fn rewrite_save_markers_regenerates_and_keeps_others() {
        let lua = "Scenario = {\n    MasterChain = {\n        ['_MASTERCHAIN_'] = {\n            Markers = {\n                ['ARMY_1'] = {\n                    ['type'] = STRING( 'Blank Marker' ),\n                    ['position'] = VECTOR3( 10, 5, 20 ),\n                },\n                ['MASS_1'] = {\n                    ['type'] = STRING( 'Mass' ),\n                    ['position'] = VECTOR3( 30, 5, 40 ),\n                },\n                ['Transport Marker 01'] = {\n                    ['type'] = STRING( 'Transport Marker' ),\n                    ['position'] = VECTOR3( 1, 2, 3 ),\n                },\n            },\n        },\n    },\n}\n";
        let armies = vec!["ARMY_1".to_string(), "ARMY_2".to_string()];
        let starts = vec![
            Marker { name: "ARMY_1".into(), marker_type: "Blank Marker".into(), position: [100.0, 9.0, 200.0] },
            Marker { name: "ARMY_2".into(), marker_type: "Blank Marker".into(), position: [300.0, 9.0, 400.0] },
        ];
        let mass = vec![Marker { name: "MASS_900".into(), marker_type: "Mass".into(), position: [50.0, 7.0, 60.0] }];
        let out = rewrite_save_markers(lua, &armies, &starts, &mass, &[]).expect("rewrite");
        assert!(out.contains("Transport Marker 01"), "other markers kept verbatim");
        assert!(out.starts_with("Scenario = {\n    MasterChain = {"));
        let parsed = parse_save_markers(&out, &armies).expect("reparse rewritten save");
        assert_eq!(parsed.start_spots.len(), 2);
        assert_eq!(parsed.start_spots[0].position, [100.0, 9.0, 200.0]);
        assert_eq!(parsed.mass_points.len(), 1);
        assert_eq!(parsed.mass_points[0].name, "MASS_900");
        assert_eq!(parsed.total_markers, 4);
        assert_eq!(
            rewrite_map_id("map = '/maps/scmp_001/SCMP_001.scmap'", "SCMP_001", "NEW1"),
            "map = '/maps/NEW1/NEW1.scmap'"
        );
    }

    #[test]
    fn rewrite_save_markers_handles_comments_bare_keys_and_plain_types() {
        let lua = r#"
--[[ Fake MasterChain = { ['_MASTERCHAIN_'] = { Markers = { } } } ]]--
Scenario = {
    note = "MasterChain = { fake }",
    MasterChain = {
        ['_MASTERCHAIN_'] = {
            Markers = {
                -- existing comment inside Markers
                ARMY_1 = {
                    ['type'] = 'Blank Marker',
                    ['position'] = VECTOR3( 1, 2, 3 ),
                },
                ['Mass 01'] = {
                    ['type'] = 'Mass',
                    ['position'] = VECTOR3( 4, 5, 6 ),
                },
                ['Hydro 01'] = {
                    ['type'] = STRING( 'Hydrocarbon' ),
                    ['position'] = VECTOR3( 7, 8, 9 ),
                },
                ['Land Path Node 01'] = {
                    ['type'] = STRING( 'Land Path Node' ),
                    ['position'] = VECTOR3( 10, 11, 12 ),
                },
            },
        },
    },
}
"#;
        let armies = vec!["ARMY_1".to_string()];
        let starts = vec![Marker {
            name: "ARMY_1".into(),
            marker_type: "Blank Marker".into(),
            position: [55.0, 12.0, 66.0],
        }];
        let mass = vec![Marker {
            name: "Mass 02".into(),
            marker_type: "Mass".into(),
            position: [77.0, 12.0, 88.0],
        }];
        let hydro = vec![Marker {
            name: "Hydro 02".into(),
            marker_type: "Hydrocarbon".into(),
            position: [99.0, 12.0, 111.0],
        }];
        let out = rewrite_save_markers(lua, &armies, &starts, &mass, &hydro).expect("rewrite");
        let parsed = parse_save_markers(&out, &armies).expect("reparse");
        assert_eq!(parsed.start_spots.len(), 1);
        assert_eq!(parsed.start_spots[0].position, [55.0, 12.0, 66.0]);
        assert_eq!(parsed.mass_points.len(), 1);
        assert_eq!(parsed.mass_points[0].name, "Mass 02");
        assert_eq!(parsed.hydro_points.len(), 1);
        assert_eq!(parsed.hydro_points[0].name, "Hydro 02");
        assert_eq!(parsed.total_markers, 4); // 1 start + 1 mass + 1 hydro + 1 kept path node
        assert!(out.contains("Land Path Node 01"));

        let scen = "ScenarioInfo = {\n    name = 'Old Name',\n    map = '/maps/old_id/old.scmap',\n    save = '/maps/old_id/old_save.lua',\n    script = '/maps/old_id/old_script.lua',\n}\n";
        let cloned = rewrite_scenario_for_clone(scen, "OLD_ID", "NEW_ID", Some("New Name"));
        let meta = parse_scenario(&cloned, "NEW_ID").expect("parse cloned scenario");
        assert_eq!(meta.name, "New Name");
        assert_eq!(meta.map_vfs_path, "/maps/NEW_ID/NEW_ID.scmap");
        assert_eq!(meta.save_vfs_path, "/maps/NEW_ID/NEW_ID_save.lua");
        assert!(cloned.contains("script = '/maps/NEW_ID/NEW_ID_script.lua',"));
    }

    #[test]
    fn parse_save_markers_reads_only_masterchain() {
        let lua = r#"
--[[                                                                           ]]--
--[[  Dummy comment with fake MasterChain = { ['_MASTERCHAIN_'] = {          ]]--
--[[      Markers = { ['ARMY_99'] = { ['type'] = STRING('Blank Marker') } }   ]]--
--[[  } }                                                                      ]]--
Scenario = {
    next_area_id = '1',
    note = "MasterChain = { fake string }",
    Areas = {
        ['AREA_1'] = {
            ['rectangle'] = RECTANGLE( 0, 0, 512, 512 ),
        },
    },
    EarlyDummyChains = {
        ['FakeChain'] = {
            Markers = {
                ['ARMY_98'] = {
                    ['type'] = STRING( 'Blank Marker' ),
                    ['position'] = VECTOR3( 999.0, 0.0, 999.0 ),
                },
            },
        },
    },
    MasterChain = {
        ['_MASTERCHAIN_'] = {
            Markers = {
                ['ARMY_2'] = {
                    ['type'] = STRING( 'Blank Marker' ),
                    ['position'] = VECTOR3( 400.0, 20.0, 410.0 ),
                },
                ['ARMY_1'] = {
                    ['type'] = 'Blank Marker',
                    ['position'] = { 50.0, 20.0, 60.0 },
                },
                ['Player'] = {
                    ['type'] = STRING( 'Blank Marker' ),
                    ['position'] = VECTOR3( -12.5, 18.25, 34.5 ),
                },
                ['Mass 02'] = {
                    ['size'] = FLOAT( 1.000000 ),
                    ['resource'] = BOOLEAN( true ),
                    ['type'] = STRING( 'Mass' ),
                    ['orientation'] = VECTOR3( 0, -0, 0 ),
                    ['position'] = VECTOR3( 220.0, 15.0, 230.0 ),
                },
                ['Mass 01'] = {
                    ['type'] = 'Mass',
                    ['position'] = VECTOR3( 110.0, 15.0, 120.0 ),
                },
                ['Hydro 01'] = {
                    ['type'] = STRING( 'Hydrocarbon' ),
                    ['position'] = VECTOR3( 300.0, 16.0, 310.0 ),
                },
                ['Land Path Node 01'] = {
                    ['type'] = STRING( 'Land Path Node' ),
                    ['position'] = VECTOR3( 10.0, 10.0, 10.0 ),
                },
            },
        },
    },
    Chains = {
        ['Attack_Chain_1'] = {
            Markers = {
                ['ARMY_77'] = {
                    ['type'] = STRING( 'Blank Marker' ),
                    ['position'] = VECTOR3( 777.0, 0.0, 777.0 ),
                },
                ['Mass 99'] = {
                    ['type'] = STRING( 'Mass' ),
                    ['position'] = VECTOR3( 888.0, 0.0, 888.0 ),
                },
                ['Hydro 99'] = {
                    ['type'] = STRING( 'Hydrocarbon' ),
                    ['position'] = VECTOR3( 999.0, 0.0, 999.0 ),
                },
            },
        },
    },
    Armies = {
        ['ARMY_1'] = {
            Units = GROUP {
                ['UNIT_1'] = {
                    Position = { 1.0, 2.0, 3.0 },
                },
            },
        },
    },
}
"#;
        let armies = vec!["Player".to_string()];
        let markers = parse_save_markers(lua, &armies).expect("parse save markers");

        assert_eq!(markers.total_markers, 7);

        let start_names: Vec<&str> = markers.start_spots.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(start_names, vec!["ARMY_1", "ARMY_2", "Player"]);
        assert_eq!(markers.start_spots[0].position, [50.0, 20.0, 60.0]);
        assert_eq!(markers.start_spots[2].position, [-12.5, 18.25, 34.5]);

        let mass_names: Vec<&str> = markers.mass_points.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(mass_names, vec!["Mass 01", "Mass 02"]);
        assert_eq!(markers.mass_points[0].marker_type, "Mass");
        assert_eq!(markers.mass_points[0].position, [110.0, 15.0, 120.0]);

        let hydro_names: Vec<&str> = markers.hydro_points.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(hydro_names, vec!["Hydro 01"]);
        assert_eq!(markers.hydro_points[0].marker_type, "Hydrocarbon");
        assert_eq!(markers.hydro_points[0].position, [300.0, 16.0, 310.0]);
    }

    #[test]
    fn resolve_vfs_path_case_insensitive_in_temp_dir() {
        let guard = TempDirGuard::new("vfs");
        let map_dir = guard.path().join("Maps").join("SCMP_001");
        std::fs::create_dir_all(&map_dir).expect("create Maps/SCMP_001");
        let scmap = map_dir.join("ScMp_001.scmap");
        std::fs::write(&scmap, b"SCMAP").expect("write scmap");

        let resolved1 = resolve_vfs_path(guard.path(), "/maps/scmp_001/SCMP_001.SCMAP")
            .expect("resolve unix-style vfs path");
        assert_eq!(resolved1, scmap);

        let resolved2 = resolve_vfs_path(guard.path(), "\\MAPS\\\\ScMp_001\\scmp_001.scmap")
            .expect("resolve windows-style vfs path");
        assert_eq!(resolved2, scmap);

        assert!(resolve_vfs_path(guard.path(), "/maps/scmp_001/missing.scmap").is_none());
        assert!(resolve_vfs_path(guard.path(), "").is_none());
        assert!(resolve_vfs_path(guard.path(), "/").is_none());
    }

    #[test]
    fn list_maps_find_map_cp1252_and_fallbacks() {
        let guard = TempDirGuard::new("list");
        let maps_root = guard.path().join("maps");

        // Map 1: X1CA_001 with CP1252 bytes and stale VFS paths requiring suffix fallback
        let x1ca_dir = maps_root.join("X1CA_001");
        std::fs::create_dir_all(&x1ca_dir).expect("create X1CA_001");
        let mut cp1252_bytes = Vec::new();
        cp1252_bytes.extend_from_slice(b"version = 3\nScenarioInfo = {\n    name = 'Operation");
        cp1252_bytes.push(0x92); // CP1252 right single quote
        cp1252_bytes.extend_from_slice(b"s Dawn',\n    description = ");
        cp1252_bytes.push(b'"');
        cp1252_bytes.push(0x93);
        cp1252_bytes.extend_from_slice(b"CP1252");
        cp1252_bytes.push(0x94);
        cp1252_bytes.extend_from_slice(
            b"\",\n    type = 'campaign',\n    size = {512, 512},\n    map = '/maps/OLD_LOC/old.scmap',\n    save = '/maps/OLD_LOC/old_save.lua',\n}\n",
        );
        std::fs::write(x1ca_dir.join("x1ca_001_SCENARIO.LUA"), &cp1252_bytes)
            .expect("write cp1252 scenario");
        std::fs::write(x1ca_dir.join("X1CA_001.SCMAP"), b"scmap").expect("write fallback scmap");
        std::fs::write(
            x1ca_dir.join("X1CA_001_SAVE.LUA"),
            b"Scenario = { MasterChain = { ['_MASTERCHAIN_'] = { Markers = { ['ARMY_1'] = { ['type'] = 'Blank Marker', ['position'] = VECTOR3(1, 2, 3) } } } } }",
        )
        .expect("write fallback save");

        // Map 2: SCMP_002 with valid VFS paths
        let scmp_dir = maps_root.join("SCMP_002");
        std::fs::create_dir_all(&scmp_dir).expect("create SCMP_002");
        std::fs::write(
            scmp_dir.join("SCMP_002_scenario.lua"),
            b"ScenarioInfo = { name = 'Canis', type = 'skirmish', size = {256, 256}, map = '/maps/scmp_002/SCMP_002.scmap', save = '/maps/scmp_002/SCMP_002_save.lua' }",
        )
        .expect("write scmp_002 scenario");
        std::fs::write(scmp_dir.join("scmp_002.scmap"), b"scmap2").expect("write scmp_002 scmap");
        std::fs::write(scmp_dir.join("scmp_002_save.lua"), b"Scenario = {}")
            .expect("write scmp_002 save");

        // Empty non-map subdir should be skipped
        std::fs::create_dir_all(maps_root.join("unused_dir")).expect("create unused_dir");

        let maps = list_maps(guard.path()).expect("list_maps");
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[0].id, "SCMP_002");
        assert_eq!(maps[1].id, "X1CA_001");
        assert!(maps[0].scmap_path.is_some());
        assert!(maps[0].save_path.is_some());
        assert_eq!(
            maps[1].scmap_path.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()),
            Some("X1CA_001.SCMAP")
        );
        assert_eq!(
            maps[1].save_path.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()),
            Some("X1CA_001_SAVE.LUA")
        );

        let found = find_map(guard.path(), "x1ca_001").expect("find_map case-insensitive");
        assert_eq!(found.id, "X1CA_001");
        let parsed_save = parse_save_markers_file(
            found.save_path.as_deref().expect("save path"),
            &[],
        )
        .expect("parse save file");
        assert_eq!(parsed_save.start_spots.len(), 1);
        assert_eq!(parsed_save.start_spots[0].name, "ARMY_1");
        assert_eq!(parsed_save.start_spots[0].position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn discovers_user_maps_loose_and_zipped() {
        use std::io::Write;
        let guard = TempDirGuard::new("usermaps");
        let root = guard.path().join("Maps");
        let cache = guard.path().join("cache");
        let loose = root.join("UserMap");
        std::fs::create_dir_all(&loose).unwrap();
        std::fs::write(
            loose.join("UserMap_scenario.lua"),
            b"ScenarioInfo = { name = 'User', type = 'skirmish', size = {256, 256}, map = '/maps/UserMap/UserMap.scmap', save = '/maps/UserMap/UserMap_save.lua' }",
        )
        .unwrap();
        std::fs::write(loose.join("usermap.scmap"), b"x").unwrap();
        std::fs::write(loose.join("UserMap_save.lua"), b"Scenario = {}").unwrap();

        let opts = zip::write::SimpleFileOptions::default();
        let mut zw = zip::ZipWriter::new(std::fs::File::create(root.join("Pack.zip")).unwrap());
        zw.start_file("ZipMap/ZipMap_scenario.lua", opts).unwrap();
        zw.write_all(b"ScenarioInfo = { name = 'Zipped', type = 'skirmish', size = {512, 512}, map = '/maps/ZipMap/ZipMap.scmap', save = '/maps/ZipMap/ZipMap_save.lua' }").unwrap();
        zw.start_file("ZipMap/ZipMap.scmap", opts).unwrap();
        zw.write_all(b"y").unwrap();
        zw.start_file("../evil.txt", opts).unwrap();
        zw.write_all(b"no").unwrap();
        zw.finish().unwrap();
        std::fs::write(root.join("broken.scd"), b"not a zip").unwrap();

        let maps = list_maps_in(&root, guard.path(), MapSource::FaUser, Some(&cache)).unwrap();
        let ids: Vec<&str> = maps.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, vec!["UserMap", "ZipMap"]);
        assert!(maps.iter().all(|m| m.source == MapSource::FaUser));
        assert_eq!(maps[0].scmap_path.as_ref().unwrap().file_name().unwrap(), "usermap.scmap");
        assert!(maps[1].dir.starts_with(&cache));
        assert!(maps[1].scmap_path.is_some() && maps[1].save_path.is_none());
        assert!(!guard.path().join("evil.txt").exists() && !cache.join("evil.txt").exists());
        // Second listing reuses the extraction.
        let again = list_maps_in(&root, guard.path(), MapSource::FaUser, Some(&cache)).unwrap();
        assert_eq!(again.len(), 2);
        // Without a cache archives are ignored.
        assert_eq!(list_maps_in(&root, guard.path(), MapSource::Env, None).unwrap().len(), 1);

        let game = guard.path().join("steamapps/common/FA");
        let dirs = user_map_dirs(&game);
        assert_eq!(dirs[0], (MapSource::FaUser, guard.path().join("steamapps").join(FA_USER_MAPS)));
        assert!(dirs.iter().any(|(s, p)| *s == MapSource::Farust && p.ends_with("farust/maps")));
    }
}
