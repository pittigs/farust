//! Phase 24 & Phase 51 blueprint mods, mod manager, conflict detection and lobby mod sync.
//!
//! A mod is a directory (default location `$XDG_DATA_HOME/farust/mods/<name>/`, override
//! with `FARUST_MODS_DIR`) holding
//!
//! * `mod_info.lua` — top-level `name`, `uid`, `version`, `author` and `description`
//!   assignments are read (FA syntax, e.g. `name = "Heavy Tanks"`); no Lua runs;
//! * `units/**/*.bp` — `UnitBlueprint { ... }` files. The blueprint id is the `BlueprintId`
//!   field if present, otherwise the file name without `.bp` / `_unit.bp` (case-insensitive).
//!
//! Merge semantics, applied mod by mod in the order they are enabled and file by file in
//! sorted path order:
//!
//! * `Merge = true`: the blueprint is deep-merged into the existing one with
//!   [`crate::blueprint::merge_into`] (tables recurse, scalars replace, positional table
//!   lists like `Weapon` merge by index, positional scalar lists like `Categories` replace;
//!   `AddCategories` / `DelCategories` edit the category list). Merging into an unknown id
//!   is an error.
//! * otherwise: an existing id is replaced wholesale, a new id adds a unit. New units join
//!   the roster through their categories (e.g. `BUILTBYTIER1FACTORY UEF MOBILE LAND`).
//!
//! Mods are enabled persistently via the in-game Mod Manager / `farust mods enable <SPEC>`
//! (`mods=uid1,uid2` in `settings.ini`), or per process via `--mod <DIR|NAME>` / `FARUST_MODS`.
//! During multiplayer handshakes, missing mods are automatically streamed from the host,
//! verified against the host's [`ModIdentity`] content hash, cached in a per-user downloaded
//! mods directory (never into the game install directory), and activated before match start.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::RwLock;

use crate::blueprint::{self, Value};

/// Maximum wire size for a single streamed mod bundle (32 MiB).
pub const MAX_MOD_BUNDLE_BYTES: usize = 32 * 1024 * 1024;
/// Maximum number of blueprint files allowed in one streamed mod bundle.
pub const MAX_MOD_BUNDLE_FILES: usize = 4096;
const MOD_BUNDLE_MAGIC: &[u8; 8] = b"FARMD1\0\0";

/// Identity of a loaded mod as recorded in replays, saves and net handshakes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModIdentity {
    pub name: String,
    pub uid: String,
    pub version: String,
    pub hash: u64,
}

impl std::fmt::Display for ModIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} v{} ({}) #{:016x}", self.name, self.version, self.uid, self.hash)
    }
}

/// Parsed fields from a `mod_info.lua` file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModInfo {
    pub name: Option<String>,
    pub uid: Option<String>,
    pub version: Option<String>,
    pub author: Option<String>,
    pub description: Option<String>,
}

/// One loaded mod: metadata plus its blueprint files `(relative path, text)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mod {
    pub name: String,
    pub uid: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub mod_info: String,
    pub dir: Option<PathBuf>,
    pub files: Vec<(String, String)>,
    pub hash: u64,
}

/// What applying one mod changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub added: Vec<String>,
    pub replaced: Vec<String>,
    pub merged: Vec<String>,
}

/// A conflict or load error among a set of enabled mods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModConflict {
    /// Two or more enabled mods modify the same blueprint id.
    BlueprintOverride {
        blueprint_id: String,
        /// `(mod_name, mode)` pairs (`"merge"` or `"replace/add"`).
        mods: Vec<(String, String)>,
    },
    /// Two or more enabled mods declare the same UID.
    DuplicateUid {
        uid: String,
        mods: Vec<String>,
    },
    /// A mod failed to load or parse a blueprint.
    LoadError {
        spec: String,
        error: String,
    },
}

impl std::fmt::Display for ModConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModConflict::BlueprintOverride { blueprint_id, mods } => {
                let list: Vec<String> = mods.iter().map(|(n, m)| format!("{n} ({m})")).collect();
                write!(f, "blueprint {blueprint_id} overridden by {}", list.join(" vs "))
            }
            ModConflict::DuplicateUid { uid, mods } => {
                write!(f, "duplicate uid '{uid}' shared by {}", mods.join(", "))
            }
            ModConflict::LoadError { spec, error } => {
                write!(f, "mod '{spec}' failed to load: {error}")
            }
        }
    }
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

/// Strips a Lua `--` line comment outside single/double/long-bracket strings.
fn strip_lua_line_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut i = 0;
    let mut quote: Option<u8> = None;
    let mut in_long = false;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = quote {
            if b == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if b == q {
                quote = None;
            }
            i += 1;
        } else if in_long {
            if b == b']' && bytes.get(i + 1) == Some(&b']') {
                in_long = false;
                i += 2;
            } else {
                i += 1;
            }
        } else if b == b'"' || b == b'\'' {
            quote = Some(b);
            i += 1;
        } else if b == b'[' && bytes.get(i + 1) == Some(&b'[') {
            in_long = true;
            i += 2;
        } else if b == b'-' && bytes.get(i + 1) == Some(&b'-') {
            return &line[..i];
        } else {
            i += 1;
        }
    }
    line
}

/// Parses `name`, `uid`, `version`, `author` and `description` from `mod_info.lua` text.
pub fn parse_mod_info_full(text: &str) -> ModInfo {
    let mut info = ModInfo::default();
    for raw in text.lines() {
        let line = strip_lua_line_comment(raw).trim();
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let v = v.trim().trim_end_matches([',', ';']).trim();
        let v = v
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .or_else(|| v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
            .or_else(|| v.strip_prefix("[[").and_then(|s| s.strip_suffix("]]")))
            .unwrap_or(v)
            .to_string();
        match k.trim() {
            "name" if info.name.is_none() => info.name = Some(v),
            "uid" if info.uid.is_none() => info.uid = Some(v),
            "version" if info.version.is_none() => info.version = Some(v),
            "author" if info.author.is_none() => info.author = Some(v),
            "description" if info.description.is_none() => info.description = Some(v),
            _ => {}
        }
    }
    info
}

/// Reads `name`, `uid` and `version` from `mod_info.lua` text.
pub fn parse_mod_info(text: &str) -> (Option<String>, Option<String>, Option<String>) {
    let info = parse_mod_info_full(text);
    (info.name, info.uid, info.version)
}

impl Mod {
    /// Builds a mod from in-memory `mod_info.lua` text and `(path, text)` blueprint files.
    pub fn from_memory(mod_info: &str, files: Vec<(String, String)>, fallback_name: &str) -> Self {
        let info = parse_mod_info_full(mod_info);
        let mut files = files;
        files.sort();
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        fnv(&mut h, b"mod_info.lua\0");
        fnv(&mut h, mod_info.as_bytes());
        for (p, t) in &files {
            fnv(&mut h, b"\0");
            fnv(&mut h, p.as_bytes());
            fnv(&mut h, b"\0");
            fnv(&mut h, t.as_bytes());
        }
        let name = info.name.filter(|s| !s.is_empty()).unwrap_or_else(|| fallback_name.to_string());
        Self {
            uid: info.uid.filter(|s| !s.is_empty()).unwrap_or_else(|| name.clone()),
            name,
            version: info.version.filter(|s| !s.is_empty()).unwrap_or_else(|| "1".to_string()),
            author: info.author.unwrap_or_default(),
            description: info.description.unwrap_or_default(),
            mod_info: mod_info.to_string(),
            dir: None,
            files,
            hash: h,
        }
    }

    /// Loads a mod directory (`mod_info.lua` optional; `units/**/*.bp`).
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        if !dir.is_dir() {
            return Err(format!("mod directory {} does not exist", dir.display()));
        }
        let info = std::fs::read(dir.join("mod_info.lua"))
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
        let mut files = Vec::new();
        let units = find_ci(dir, "units");
        if let Some(units) = units {
            collect_bp(&units, "units", &mut files)?;
        }
        let fallback = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "mod".into());
        let mut m = Self::from_memory(&info, files, &fallback);
        m.dir = Some(dir.to_path_buf());
        Ok(m)
    }

    pub fn identity(&self) -> ModIdentity {
        ModIdentity { name: self.name.clone(), uid: self.uid.clone(), version: self.version.clone(), hash: self.hash }
    }

    /// Parsed `(blueprint id, root)` pairs of this mod.
    pub fn blueprints(&self) -> Result<Vec<(String, Value)>, String> {
        let mut out = Vec::new();
        for (path, text) in &self.files {
            let root = blueprint::parse_blueprint(text).map_err(|e| format!("mod '{}': {path}: {e}", self.name))?;
            let id = root
                .get_str("BlueprintId")
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| blueprint::unit_id_from_path(path));
            out.push((id.to_ascii_uppercase(), root));
        }
        Ok(out)
    }
}

fn find_ci(dir: &Path, name: &str) -> Option<PathBuf> {
    let exact = dir.join(name);
    if exact.is_dir() {
        return Some(exact);
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().eq_ignore_ascii_case(name) && e.path().is_dir())
        .map(|e| e.path())
}

fn collect_bp(dir: &Path, rel: &str, out: &mut Vec<(String, String)>) -> Result<(), String> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().into_owned();
        let p = e.path();
        let r = format!("{rel}/{name}");
        if p.is_dir() {
            collect_bp(&p, &r, out)?;
        } else if name.to_ascii_lowercase().ends_with(".bp") {
            let bytes = std::fs::read(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
            out.push((r, String::from_utf8_lossy(&bytes).into_owned()));
        }
    }
    Ok(())
}

/// Applies `mods` in order to `roots` (keyed by upper-case blueprint id).
pub fn apply(roots: &mut BTreeMap<String, Value>, mods: &[Mod]) -> Result<Vec<MergeReport>, String> {
    let mut reports = Vec::with_capacity(mods.len());
    for m in mods {
        let mut rep = MergeReport::default();
        for (id, mut root) in m.blueprints()? {
            let merge = root.get_bool("Merge").unwrap_or(false);
            if merge {
                let Some(base) = roots.get_mut(&id) else {
                    return Err(format!("mod '{}': Merge = true for unknown blueprint {id}", m.name));
                };
                blueprint::merge_into(base, &root);
                rep.merged.push(id);
            } else {
                blueprint::strip_control_keys(&mut root);
                if roots.insert(id.clone(), root).is_some() {
                    rep.replaced.push(id);
                } else {
                    rep.added.push(id);
                }
            }
        }
        reports.push(rep);
    }
    Ok(reports)
}

/// Detects conflicts (blueprint override collisions, duplicate UIDs, blueprint parse errors)
/// among a slice of enabled mods.
pub fn detect_conflicts(mods: &[Mod]) -> Vec<ModConflict> {
    let mut out = Vec::new();
    // Duplicate UIDs
    let mut by_uid: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for m in mods {
        by_uid.entry(m.uid.to_ascii_lowercase()).or_default().push(m.name.clone());
    }
    for (uid, names) in by_uid {
        if names.len() >= 2 {
            out.push(ModConflict::DuplicateUid { uid, mods: names });
        }
    }
    // Blueprint collisions and parse errors
    let mut by_bp: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for m in mods {
        match m.blueprints() {
            Ok(bps) => {
                let mut seen_in_mod: BTreeMap<String, String> = BTreeMap::new();
                for (id, root) in bps {
                    let mode = if root.get_bool("Merge").unwrap_or(false) { "merge" } else { "replace/add" };
                    seen_in_mod.entry(id).or_insert_with(|| mode.to_string());
                }
                for (id, mode) in seen_in_mod {
                    by_bp.entry(id).or_default().push((m.name.clone(), mode));
                }
            }
            Err(e) => {
                out.push(ModConflict::LoadError { spec: m.name.clone(), error: e });
            }
        }
    }
    for (blueprint_id, list) in by_bp {
        if list.len() >= 2 {
            out.push(ModConflict::BlueprintOverride { blueprint_id, mods: list });
        }
    }
    out
}

/// Resolves `specs` and returns `(loaded_mods, conflicts)` including any load failures.
pub fn detect_conflicts_for_specs(specs: &[String]) -> (Vec<Mod>, Vec<ModConflict>) {
    let mut loaded = Vec::new();
    let mut conflicts = Vec::new();
    for s in specs {
        match resolve(s) {
            Ok(m) => loaded.push(m),
            Err(e) => conflicts.push(ModConflict::LoadError { spec: s.clone(), error: e }),
        }
    }
    conflicts.extend(detect_conflicts(&loaded));
    (loaded, conflicts)
}

// ---------------------------------------------------------------------------
// Directories & discovery (strictly user data/cache; never FA install dir)
// ---------------------------------------------------------------------------

/// `$FARUST_MODS_DIR`, else `$XDG_DATA_HOME/farust/mods` (`~/.local/share/farust/mods`).
pub fn mods_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("FARUST_MODS_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(d);
    }
    crate::scenario::xdg_data_home().join("farust").join("mods")
}

/// Per-user cache directory for mods automatically downloaded from lobby hosts.
///
/// Precedence:
/// 1. `$FARUST_MODS_CACHE_DIR`
/// 2. `$FARUST_MODS_DIR/.downloaded` (when `FARUST_MODS_DIR` is overridden, keeping test/CLI
///    sandboxes isolated from the user's global cache)
/// 3. `$XDG_CACHE_HOME/farust/mods` (fallback `~/.cache/farust/mods`, else
///    `$XDG_DATA_HOME/farust/mods_cache`).
pub fn downloaded_mods_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("FARUST_MODS_CACHE_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(d);
    }
    if let Some(d) = std::env::var_os("FARUST_MODS_DIR").filter(|v| !v.is_empty()) {
        return PathBuf::from(d).join(".downloaded");
    }
    if let Some(c) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(c).join("farust").join("mods");
    }
    if let Some(h) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(h).join(".cache").join("farust").join("mods");
    }
    crate::scenario::xdg_data_home().join("farust").join("mods_cache")
}

/// Every mod directory in [`mods_dir`], sorted by folder name (hidden `.` dirs skipped).
pub fn discover() -> Vec<(PathBuf, Result<Mod, String>)> {
    discover_in(&mods_dir())
}

/// Every mod directory inside `root`, sorted by folder name (hidden `.` dirs skipped).
pub fn discover_in(root: &Path) -> Vec<(PathBuf, Result<Mod, String>)> {
    let Ok(rd) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with('.'))
        })
        .collect();
    dirs.sort();
    dirs.into_iter()
        .map(|d| {
            let m = Mod::load_dir(&d);
            (d, m)
        })
        .collect()
}

/// Resolves a `--mod` spec: an existing directory, or a folder / mod name / uid in [`mods_dir`].
pub fn resolve(spec: &str) -> Result<Mod, String> {
    let p = Path::new(spec);
    if p.is_dir() {
        return Mod::load_dir(p);
    }
    for (dir, m) in discover() {
        let folder = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if folder.eq_ignore_ascii_case(spec) {
            return m;
        }
        if let Ok(m) = m
            && (m.name.eq_ignore_ascii_case(spec) || m.uid.eq_ignore_ascii_case(spec))
        {
            return Ok(m);
        }
    }
    Err(format!("mod '{spec}' not found (not a directory, nor a mod in {})", mods_dir().display()))
}

/// Searches `extra_mods`, optionally [`mods_dir`], and `cache_root` for a mod whose full
/// [`ModIdentity`] (name, uid, version, content hash) matches `want`.
pub fn find_local_mod(
    want: &ModIdentity,
    extra_mods: &[Mod],
    cache_root: &Path,
    search_mods_dir: bool,
) -> Option<Mod> {
    if let Some(m) = extra_mods.iter().find(|m| m.identity() == *want) {
        return Some(m.clone());
    }
    if search_mods_dir {
        for (_, res) in discover() {
            if let Ok(m) = res
                && m.identity() == *want
            {
                return Some(m);
            }
        }
    }
    for (_, res) in discover_in(cache_root) {
        if let Ok(m) = res
            && m.identity() == *want
        {
            return Some(m);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Wire bundle codec, path safety & safe cache installer (Phase 51)
// ---------------------------------------------------------------------------

/// Validates a relative file path carried inside a streamed mod bundle.
///
/// Rejects empty paths, absolute paths, `..` / `.` segments, backslashes, drive colons,
/// NUL/control characters, and non-normal OS path components.
pub fn validate_bundle_rel_path(path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Err("mod bundle contains an empty file path".to_string());
    }
    if path.len() > 512 {
        return Err(format!("mod bundle path too long ({} bytes)", path.len()));
    }
    if path.starts_with('/') || path.contains('\\') || path.contains('\0') || path.contains(':') {
        return Err(format!("mod bundle contains unsafe path '{path}'"));
    }
    if path.bytes().any(|b| b < 0x20 || b == 0x7f) {
        return Err(format!("mod bundle path contains control characters: '{path}'"));
    }
    for seg in path.split('/') {
        if seg.is_empty() || seg == "." || seg == ".." {
            return Err(format!("mod bundle contains invalid path segment in '{path}'"));
        }
    }
    let p = Path::new(path);
    if p.is_absolute() || !p.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(format!("mod bundle path is not strictly relative: '{path}'"));
    }
    Ok(())
}

/// Encodes `m` into a length-prefixed binary bundle capped at [`MAX_MOD_BUNDLE_BYTES`].
pub fn encode_mod_bundle(m: &Mod) -> Result<Vec<u8>, String> {
    if m.files.len() > MAX_MOD_BUNDLE_FILES {
        return Err(format!("mod '{}' has {} files (max {MAX_MOD_BUNDLE_FILES})", m.name, m.files.len()));
    }
    let mut out = Vec::with_capacity(256 + m.mod_info.len());
    out.extend_from_slice(MOD_BUNDLE_MAGIC);
    crate::replay::write_str(&mut out, &m.name);
    crate::replay::write_str(&mut out, &m.uid);
    crate::replay::write_str(&mut out, &m.version);
    out.extend_from_slice(&m.hash.to_le_bytes());
    let info_bytes = m.mod_info.as_bytes();
    out.extend_from_slice(&(info_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(info_bytes);
    out.extend_from_slice(&(m.files.len() as u32).to_le_bytes());
    for (rel, text) in &m.files {
        validate_bundle_rel_path(rel)?;
        crate::replay::write_str(&mut out, rel);
        let bytes = text.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
        if out.len() > MAX_MOD_BUNDLE_BYTES {
            return Err(format!(
                "mod '{}' bundle exceeds {} MiB limit",
                m.name,
                MAX_MOD_BUNDLE_BYTES / (1024 * 1024)
            ));
        }
    }
    if out.len() > MAX_MOD_BUNDLE_BYTES {
        return Err(format!(
            "mod '{}' bundle exceeds {} MiB limit",
            m.name,
            MAX_MOD_BUNDLE_BYTES / (1024 * 1024)
        ));
    }
    Ok(out)
}

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, max_total: usize) -> Result<&'a [u8], String> {
    let n = crate::replay::read_u32(data, pos)? as usize;
    if n > max_total || *pos + n > data.len() {
        return Err(format!("mod bundle truncated (wanted {n} bytes at {})", *pos));
    }
    let slice = &data[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

/// Decodes a mod bundle and verifies path safety, size cap, and exact [`ModIdentity`]
/// (including content hash) against `expected`.
pub fn decode_mod_bundle(data: &[u8], expected: &ModIdentity) -> Result<Mod, String> {
    if data.len() > MAX_MOD_BUNDLE_BYTES {
        return Err(format!(
            "mod bundle size {} exceeds {} MiB limit",
            data.len(),
            MAX_MOD_BUNDLE_BYTES / (1024 * 1024)
        ));
    }
    if data.len() < MOD_BUNDLE_MAGIC.len() || &data[..8] != MOD_BUNDLE_MAGIC {
        return Err("invalid mod bundle header".to_string());
    }
    let mut pos = 8;
    let hdr_name = crate::replay::read_str(data, &mut pos)?;
    let hdr_uid = crate::replay::read_str(data, &mut pos)?;
    let hdr_version = crate::replay::read_str(data, &mut pos)?;
    let hdr_hash = crate::replay::read_u64(data, &mut pos)?;

    let info_bytes = read_bytes(data, &mut pos, MAX_MOD_BUNDLE_BYTES)?;
    let mod_info = String::from_utf8(info_bytes.to_vec())
        .map_err(|_| "mod bundle mod_info.lua is not valid UTF-8".to_string())?;

    let file_count = crate::replay::read_u32(data, &mut pos)? as usize;
    if file_count > MAX_MOD_BUNDLE_FILES {
        return Err(format!("mod bundle lists {file_count} files (max {MAX_MOD_BUNDLE_FILES})"));
    }
    let mut files = Vec::with_capacity(file_count);
    for _ in 0..file_count {
        let rel = crate::replay::read_str(data, &mut pos)?;
        validate_bundle_rel_path(&rel)?;
        let content_bytes = read_bytes(data, &mut pos, MAX_MOD_BUNDLE_BYTES)?;
        let content = String::from_utf8(content_bytes.to_vec())
            .map_err(|_| format!("mod bundle file '{rel}' is not valid UTF-8"))?;
        files.push((rel, content));
    }
    if pos != data.len() {
        return Err(format!("mod bundle has {} trailing bytes", data.len() - pos));
    }

    let decoded = Mod::from_memory(&mod_info, files, &hdr_name);
    if decoded.hash != hdr_hash || decoded.hash != expected.hash {
        return Err(format!(
            "mod '{}' hash mismatch: expected #{:016x}, got #{:016x} (header #{:016x})",
            expected.name, expected.hash, decoded.hash, hdr_hash
        ));
    }
    if decoded.uid != expected.uid
        || decoded.name != expected.name
        || decoded.version != expected.version
        || hdr_uid != expected.uid
        || hdr_version != expected.version
    {
        return Err(format!(
            "mod identity mismatch: expected {}, got {}",
            expected,
            decoded.identity()
        ));
    }
    // Also verify blueprints parse cleanly before installing.
    decoded.blueprints()?;
    Ok(decoded)
}

fn sanitize_cache_segment(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .take(40)
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() { "mod".to_string() } else { trimmed.to_string() }
}

/// Installs a hash-verified [`Mod`] into `cache_root/<uid>-<hash>/` (never touching any game
/// install directory) and re-verifies the content hash from disk.
pub fn install_cached_mod(m: &Mod, cache_root: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(cache_root)
        .map_err(|e| format!("cannot create mod cache dir {}: {e}", cache_root.display()))?;
    let folder = format!("{}-{:016x}", sanitize_cache_segment(&m.uid), m.hash);
    let dest = cache_root.join(&folder);
    if dest.is_dir()
        && let Ok(existing) = Mod::load_dir(&dest)
        && existing.identity() == m.identity()
    {
        return Ok(dest);
    }

    let tmp = cache_root.join(format!(".{folder}.tmp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)
        .map_err(|e| format!("cannot create temp mod dir {}: {e}", tmp.display()))?;
    std::fs::write(tmp.join("mod_info.lua"), m.mod_info.as_bytes())
        .map_err(|e| format!("cannot write mod_info.lua: {e}"))?;
    for (rel, text) in &m.files {
        validate_bundle_rel_path(rel)?;
        let target = tmp.join(rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        std::fs::write(&target, text.as_bytes())
            .map_err(|e| format!("cannot write {}: {e}", target.display()))?;
    }
    if dest.exists() {
        let _ = std::fs::remove_dir_all(&dest);
    }
    std::fs::rename(&tmp, &dest)
        .map_err(|e| format!("cannot finalize cached mod {}: {e}", dest.display()))?;
    let reloaded = Mod::load_dir(&dest)?;
    if reloaded.hash != m.hash {
        let _ = std::fs::remove_dir_all(&dest);
        return Err(format!(
            "cached mod '{}' hash verification failed after write: expected #{:016x}, got #{:016x}",
            m.name, m.hash, reloaded.hash
        ));
    }
    Ok(dest)
}

// ---------------------------------------------------------------------------
// Process-wide enabled / active mod runtime state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RuntimeMods {
    cli_specs: Option<Vec<String>>,
    active_override: Option<Vec<Mod>>,
}

static RUNTIME: RwLock<RuntimeMods> = RwLock::new(RuntimeMods { cli_specs: None, active_override: None });

/// Sets the process-wide enabled mod specs from CLI `--mod`. Empty clears CLI override so
/// `FARUST_MODS` or `settings.ini` can apply.
pub fn set_enabled(specs: Vec<String>) {
    if let Ok(mut g) = RUNTIME.write() {
        g.cli_specs = if specs.is_empty() { None } else { Some(specs) };
        g.active_override = None;
    }
}

/// Sets the process-wide enabled mod specs explicitly (even when empty, e.g. from the Mod
/// Manager screen when the user disables all mods).
pub fn set_runtime_enabled(specs: Vec<String>) {
    if let Ok(mut g) = RUNTIME.write() {
        g.cli_specs = Some(specs);
        g.active_override = None;
    }
}

/// Directly activates already-loaded mods (used after lobby mod sync so downloaded mods take
/// effect immediately in exact host order).
pub fn set_active_mods(mods: Vec<Mod>) {
    if let Ok(mut g) = RUNTIME.write() {
        g.active_override = Some(mods);
    }
}

/// Clears any lobby-synced active mod override so normal enabled specs take effect again.
#[allow(dead_code)]
pub fn clear_active_override() {
    if let Ok(mut g) = RUNTIME.write() {
        g.active_override = None;
    }
}

/// Effective enabled mod specs in precedence order:
/// 1. Explicit `--mod` CLI args / Mod Manager runtime set
/// 2. `FARUST_MODS` env var (empty string disables persisted mods)
/// 3. `mods=...` persisted in `settings.ini` (skipped under `#[cfg(test)]` unless active settings are set)
pub fn enabled_specs() -> Vec<String> {
    if let Ok(g) = RUNTIME.read()
        && let Some(ref v) = g.cli_specs
    {
        return v.clone();
    }
    if let Ok(s) = std::env::var("FARUST_MODS") {
        return s
            .split([',', ':'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
    }
    if let Some(s) = crate::settings::active() {
        return s.mods;
    }
    #[cfg(not(test))]
    {
        if let Some(p) = crate::settings::settings_path() {
            return crate::settings::Settings::load_from(&p).0.mods;
        }
    }
    Vec::new()
}

/// The currently enabled mods (either lobby-synced override or resolved from [`enabled_specs`]).
pub fn active() -> Result<Vec<Mod>, String> {
    if let Ok(g) = RUNTIME.read()
        && let Some(ref ov) = g.active_override
    {
        return Ok(ov.clone());
    }
    enabled_specs().iter().map(|s| resolve(s)).collect()
}

/// Identities of the enabled mods (empty if none or if loading failed).
pub fn active_identities() -> Vec<ModIdentity> {
    active().map(|m| m.iter().map(Mod::identity).collect()).unwrap_or_default()
}

fn describe(ids: &[ModIdentity]) -> String {
    if ids.is_empty() {
        "no mods (stock FA content)".to_string()
    } else {
        ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
    }
}

/// Errors unless `theirs` (recorded / remote content) equals `ours` (loaded content).
pub fn check_match(theirs: &[ModIdentity], ours: &[ModIdentity], what: &str) -> Result<(), String> {
    if theirs == ours {
        return Ok(());
    }
    Err(format!(
        "content mismatch: {what} uses {}, but this game has {} loaded (enable the same mods with --mod)",
        describe(theirs),
        describe(ours)
    ))
}

/// Binary codec shared by replays, lobbies and the lockstep handshake.
pub fn encode_identities(out: &mut Vec<u8>, ids: &[ModIdentity]) {
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    for m in ids {
        crate::replay::write_str(out, &m.name);
        crate::replay::write_str(out, &m.uid);
        crate::replay::write_str(out, &m.version);
        out.extend_from_slice(&m.hash.to_le_bytes());
    }
}

pub fn decode_identities(data: &[u8], pos: &mut usize) -> Result<Vec<ModIdentity>, String> {
    let n = crate::replay::read_u32(data, pos)? as usize;
    if n > 256 {
        return Err(format!("header lists {n} mods (max 256)"));
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ModIdentity {
            name: crate::replay::read_str(data, pos)?,
            uid: crate::replay::read_str(data, pos)?,
            version: crate::replay::read_str(data, pos)?,
            hash: crate::replay::read_u64(data, pos)?,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Persisted enable / disable helpers (used by CLI and Mod Manager)
// ---------------------------------------------------------------------------

/// Canonical spec to store in `settings.ini` for `m` (`uid` when discoverable in
/// [`mods_dir`], otherwise its directory path or fallback spec).
pub fn canonical_spec_for(m: &Mod, fallback: &str) -> String {
    let md = mods_dir();
    if let Some(ref d) = m.dir
        && d.starts_with(&md)
    {
        return m.uid.clone();
    }
    if discover().into_iter().any(|(_, r)| r.is_ok_and(|x| x.uid.eq_ignore_ascii_case(&m.uid))) {
        return m.uid.clone();
    }
    if let Some(ref d) = m.dir {
        return d.canonicalize().unwrap_or_else(|_| d.clone()).display().to_string();
    }
    fallback.to_string()
}

/// Enables `spec` in `settings_path` (persisted in `settings.ini`), returning `(mod, conflicts)`.
pub fn enable_in_settings(spec: &str, settings_path: &Path) -> Result<(Mod, Vec<ModConflict>), String> {
    let m = resolve(spec)?;
    // Verify blueprints parse cleanly.
    m.blueprints()?;
    let canon = canonical_spec_for(&m, spec);
    let (mut settings, _) = crate::settings::Settings::load_from(settings_path);
    let already = settings.mods.iter().any(|existing| {
        existing.eq_ignore_ascii_case(&canon)
            || existing.eq_ignore_ascii_case(&m.uid)
            || existing.eq_ignore_ascii_case(&m.name)
            || resolve(existing).is_ok_and(|em| em.uid.eq_ignore_ascii_case(&m.uid))
    });
    if !already {
        settings.mods.push(canon);
        settings.save_to(settings_path)?;
    }
    let (_, conflicts) = detect_conflicts_for_specs(&settings.mods);
    Ok((m, conflicts))
}

/// Disables `spec` in `settings_path` (persisted in `settings.ini`), returning `true` if removed.
pub fn disable_in_settings(spec: &str, settings_path: &Path) -> Result<bool, String> {
    let (mut settings, _) = crate::settings::Settings::load_from(settings_path);
    let target_uid = resolve(spec).ok().map(|m| m.uid);
    let before = settings.mods.len();
    settings.mods.retain(|existing| {
        if existing.eq_ignore_ascii_case(spec) {
            return false;
        }
        if let Some(ref uid) = target_uid {
            if existing.eq_ignore_ascii_case(uid) {
                return false;
            }
            if let Ok(em) = resolve(existing)
                && em.uid.eq_ignore_ascii_case(uid)
            {
                return false;
            }
        } else if let Ok(em) = resolve(existing)
            && (em.name.eq_ignore_ascii_case(spec)
                || em.uid.eq_ignore_ascii_case(spec)
                || em
                    .dir
                    .as_ref()
                    .and_then(|d| d.file_name())
                    .is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case(spec)))
        {
            return false;
        }
        true
    });
    let removed = settings.mods.len() != before;
    if removed {
        settings.save_to(settings_path)?;
    }
    Ok(removed)
}

/// `farust mods [enable <SPEC> | disable <SPEC> | conflicts | GAME_DIR]`.
pub fn cmd_mods(args: &[String]) -> Result<(), String> {
    if let Some(sub) = args.get(2).map(String::as_str) {
        match sub {
            "enable" => {
                let spec = args
                    .get(3)
                    .ok_or_else(|| "usage: farust mods enable <SPEC>".to_string())?;
                let sp = crate::settings::settings_path()
                    .ok_or_else(|| "cannot determine settings.ini path (set HOME or XDG_CONFIG_HOME)".to_string())?;
                let (m, conflicts) = enable_in_settings(spec, &sp)?;
                println!("enabled mod: {} (saved to {})", m.identity(), sp.display());
                if !conflicts.is_empty() {
                    println!("warning: {} conflict(s) detected:", conflicts.len());
                    for c in &conflicts {
                        println!("  ! {c}");
                    }
                }
                return Ok(());
            }
            "disable" => {
                let spec = args
                    .get(3)
                    .ok_or_else(|| "usage: farust mods disable <SPEC>".to_string())?;
                let sp = crate::settings::settings_path()
                    .ok_or_else(|| "cannot determine settings.ini path (set HOME or XDG_CONFIG_HOME)".to_string())?;
                if disable_in_settings(spec, &sp)? {
                    println!("disabled mod '{spec}' (saved to {})", sp.display());
                } else {
                    println!("mod '{spec}' was not enabled in {}", sp.display());
                }
                return Ok(());
            }
            "conflicts" => {
                let specs = enabled_specs();
                let (loaded, mut conflicts) = detect_conflicts_for_specs(&specs);
                // Also verify merge targets against stock blueprints if game_dir is readable.
                let game_dir = args
                    .iter()
                    .skip(3)
                    .find(|a| !a.starts_with("--"))
                    .map(PathBuf::from)
                    .unwrap_or_else(crate::scd::default_game_dir);
                if let Ok(mut roots) = crate::skirmish::Catalog::read_blueprints(&game_dir) {
                    for m in &loaded {
                        if let Err(e) = apply(&mut roots, std::slice::from_ref(m)) {
                            conflicts.push(ModConflict::LoadError { spec: m.name.clone(), error: e });
                        }
                    }
                }
                if conflicts.is_empty() {
                    println!("no conflicts among {} enabled mod(s)", loaded.len());
                } else {
                    println!("{} conflict(s) among {} enabled mod(s):", conflicts.len(), specs.len());
                    for c in &conflicts {
                        println!("  ! {c}");
                    }
                }
                return Ok(());
            }
            _ => {}
        }
    }

    let game_dir = args
        .iter()
        .skip(2)
        .find(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(crate::scd::default_game_dir);
    let stock = crate::skirmish::Catalog::read_blueprints(&game_dir).ok();
    let specs = enabled_specs();
    let (enabled, conflicts) = detect_conflicts_for_specs(&specs);
    let enabled_ids: Vec<ModIdentity> = enabled.iter().map(Mod::identity).collect();
    let mut listed: Vec<Mod> = Vec::new();
    println!("mods dir:   {}", mods_dir().display());
    println!("cache dir:  {}", downloaded_mods_dir().display());
    let found = discover();
    if found.is_empty() && enabled.is_empty() {
        println!("  (no mods installed)");
    }
    for (dir, m) in found {
        match m {
            Ok(m) => listed.push(m),
            Err(e) => println!("  {}  ERROR {e}", dir.display()),
        }
    }
    for m in &enabled {
        if !listed.iter().any(|l| l.identity() == m.identity()) {
            listed.push(m.clone());
        }
    }
    for m in &listed {
        let on = enabled_ids.contains(&m.identity());
        println!(
            "  [{}] {}  {} blueprint file(s)  {}",
            if on { "enabled" } else { "  off  " },
            m.identity(),
            m.files.len(),
            m.dir.as_ref().map(|d| d.display().to_string()).unwrap_or_default()
        );
        if !m.author.is_empty() || !m.description.is_empty() {
            println!(
                "      author: {}  {}",
                if m.author.is_empty() { "-" } else { &m.author },
                m.description
            );
        }
        let Some(stock) = &stock else {
            continue;
        };
        let mut roots = stock.clone();
        match apply(&mut roots, std::slice::from_ref(m)) {
            Ok(rep) => {
                let rep = &rep[0];
                let infos: BTreeMap<String, crate::roster::BpInfo> = roots
                    .iter()
                    .map(|(id, r)| (id.clone(), crate::roster::BpInfo::from_blueprint(id, r)))
                    .collect();
                let r = crate::roster::derive(&infos, &BTreeMap::new());
                let st = |id: &String| match r.entries.get(id).map(|e| &e.status) {
                    Some(crate::roster::Status::Playable) => format!("{id} (playable)"),
                    Some(crate::roster::Status::Excluded(m)) => format!("{id} (excluded: {m})"),
                    _ => format!("{id} (not buildable)"),
                };
                println!(
                    "      added: {}  merged: {}  replaced: {}",
                    if rep.added.is_empty() { "-".into() } else { rep.added.iter().map(st).collect::<Vec<_>>().join(", ") },
                    if rep.merged.is_empty() { "-".into() } else { rep.merged.join(", ") },
                    if rep.replaced.is_empty() { "-".into() } else { rep.replaced.join(", ") },
                );
            }
            Err(e) => println!("      ERROR {e}"),
        }
    }
    if !conflicts.is_empty() {
        println!("conflicts ({}):", conflicts.len());
        for c in &conflicts {
            println!("  ! {c}");
        }
    }
    println!("manage: farust mods <enable|disable> <SPEC> | farust mods conflicts | farust --mod <DIR|NAME> ...");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const INFO: &str = "name = \"Heavy Tanks\" -- comment\nuid = 'heavy-tanks-001'\nversion = 2\nauthor = \"Jane Doe\"\ndescription = \"Doubles tank HP and firepower\"\n";

    fn base() -> BTreeMap<String, Value> {
        let mut m = BTreeMap::new();
        m.insert(
            "UEL0201".to_string(),
            blueprint::parse_blueprint("UnitBlueprint { Categories = {'UEF','MOBILE','LAND'}, Defense = { MaxHealth = 300 }, Weapon = { { Label = 'Gun', Damage = 10, MaxRadius = 18, RateOfFire = 1.0, MuzzleVelocity = 20 } } }").unwrap(),
        );
        m
    }

    fn sample_mod() -> Mod {
        Mod::from_memory(
            INFO,
            vec![
                ("units/UEL0201/UEL0201_unit.bp".into(), "UnitBlueprint { Merge = true, Defense = { MaxHealth = 600 }, Weapon = { { Damage = 25, MaxRadius = 28, RateOfFire = 2.0, MuzzleVelocity = 45, DamageRadius = 3.0 } } }".into()),
                ("units/MOD0001/MOD0001_unit.bp".into(), "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','TECH1','BUILTBYTIER1FACTORY'}, Defense = { MaxHealth = 99 } }".into()),
                ("units/other.bp".into(), "UnitBlueprint { BlueprintId = 'uel0106', Categories = {'UEF'} }".into()),
            ],
            "fallback",
        )
    }

    #[test]
    fn parses_mod_info() {
        let (n, u, v) = parse_mod_info(INFO);
        assert_eq!(n.as_deref(), Some("Heavy Tanks"));
        assert_eq!(u.as_deref(), Some("heavy-tanks-001"));
        assert_eq!(v.as_deref(), Some("2"));
        let full = parse_mod_info_full(INFO);
        assert_eq!(full.author.as_deref(), Some("Jane Doe"));
        assert_eq!(full.description.as_deref(), Some("Doubles tank HP and firepower"));
        // Comments inside strings are preserved.
        let tricky = parse_mod_info_full("name = \"A -- B\" -- real comment\nuid = 'u--1'\n");
        assert_eq!(tricky.name.as_deref(), Some("A -- B"));
        assert_eq!(tricky.uid.as_deref(), Some("u--1"));
    }

    #[test]
    fn in_memory_mod_merges_adds_and_replaces() {
        let mut roots = base();
        let m = sample_mod();
        let rep = apply(&mut roots, std::slice::from_ref(&m)).unwrap();
        assert_eq!(rep[0].merged, vec!["UEL0201"]);
        assert_eq!(rep[0].added, vec!["MOD0001", "UEL0106"]);
        let t = &roots["UEL0201"];
        assert_eq!(t.get("Defense").unwrap().get_f64("MaxHealth"), Some(600.0));
        let w0 = t.get("Weapon").unwrap().table_list()[0];
        assert_eq!(w0.get_f64("Damage"), Some(25.0));
        assert_eq!(w0.get_f64("MaxRadius"), Some(28.0));
        assert_eq!(w0.get_f64("RateOfFire"), Some(2.0));
        assert_eq!(w0.get_f64("MuzzleVelocity"), Some(45.0));
        assert_eq!(w0.get_f64("DamageRadius"), Some(3.0));
        assert_eq!(w0.get_str("Label"), Some("Gun"));
        assert!(roots["UEL0106"].get("BlueprintId").is_none());
        // Re-adding replaces wholesale.
        let m2 = Mod::from_memory("name='r'", vec![("units/x/UEL0201_unit.bp".into(), "UnitBlueprint { Categories = {'UEF'} }".into())], "r");
        let rep = apply(&mut roots, &[m2]).unwrap();
        assert_eq!(rep[0].replaced, vec!["UEL0201"]);
        assert!(roots["UEL0201"].get("Defense").is_none());
    }

    #[test]
    fn mod_unit_joins_the_derived_roster() {
        let mut roots: BTreeMap<String, Value> = [
            ("UEL0001", "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','COMMANDER'}, Economy = { BuildableCategory = {'BUILTBYCOMMANDER UEF'} } }"),
            ("UEB0101", "UnitBlueprint { Categories = {'UEF','STRUCTURE','LAND','FACTORY','TECH1','BUILTBYCOMMANDER'}, Economy = { BuildableCategory = {'BUILTBYTIER1FACTORY UEF MOBILE'} } }"),
            ("UEL0201", "UnitBlueprint { Categories = {'UEF','MOBILE','LAND','TECH1'} }"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), blueprint::parse_blueprint(v).unwrap()))
        .collect();
        apply(&mut roots, &[sample_mod()]).unwrap();
        let infos: BTreeMap<String, crate::roster::BpInfo> =
            roots.iter().map(|(k, v)| (k.clone(), crate::roster::BpInfo::from_blueprint(k, v))).collect();
        let r = crate::roster::derive(&infos, &BTreeMap::new());
        assert_eq!(r.entries["MOD0001"].status, crate::roster::Status::Playable);
        assert!(r.can("UEB0101", "MOD0001", crate::roster::BuildKind::Produce));
    }

    #[test]
    fn merge_into_unknown_is_an_error() {
        let m = Mod::from_memory("", vec![("units/x_unit.bp".into(), "UnitBlueprint { Merge = true }".into())], "bad");
        let err = apply(&mut base(), &[m]).unwrap_err();
        assert!(err.contains("unknown blueprint X"), "{err}");
    }

    #[test]
    fn identity_hash_tracks_content_and_codec_roundtrips() {
        let a = sample_mod().identity();
        let b = sample_mod().identity();
        assert_eq!(a, b);
        let mut files = sample_mod().files;
        files[0].1.push(' ');
        let c = Mod::from_memory(INFO, files, "x").identity();
        assert_ne!(a.hash, c.hash);
        assert_eq!(a.name, "Heavy Tanks");
        let mut buf = Vec::new();
        encode_identities(&mut buf, &[a.clone(), c.clone()]);
        let mut pos = 0;
        assert_eq!(decode_identities(&buf, &mut pos).unwrap(), vec![a.clone(), c.clone()]);
        assert!(check_match(&[a.clone()], &[a.clone()], "replay").is_ok());
        let e = check_match(&[a], &[], "replay").unwrap_err();
        assert!(e.contains("Heavy Tanks") && e.contains("no mods"), "{e}");
    }

    #[test]
    fn loads_example_mod_dir() {
        let dir = std::env::temp_dir().join(format!("farust-mod-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("units/ABC0001")).unwrap();
        std::fs::write(dir.join("mod_info.lua"), INFO).unwrap();
        std::fs::write(dir.join("units/ABC0001/ABC0001_unit.bp"), "UnitBlueprint { Categories = {'UEF'} }").unwrap();
        std::fs::write(dir.join("units/readme.txt"), "ignored").unwrap();
        let m = Mod::load_dir(&dir).unwrap();
        assert_eq!(m.files.len(), 1);
        assert_eq!(m.blueprints().unwrap()[0].0, "ABC0001");
        assert_eq!(m.identity().uid, "heavy-tanks-001");
        assert_eq!(m.author, "Jane Doe");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bundle_roundtrip_path_safety_and_hash_verification() {
        let m = sample_mod();
        let id = m.identity();
        let bytes = encode_mod_bundle(&m).unwrap();
        let decoded = decode_mod_bundle(&bytes, &id).unwrap();
        assert_eq!(decoded.identity(), id);
        assert_eq!(decoded.files, m.files);
        assert_eq!(decoded.author, m.author);
        assert_eq!(decoded.description, m.description);

        // Tampering with expected hash fails verification cleanly.
        let mut wrong_id = id.clone();
        wrong_id.hash ^= 0xFF;
        let err = decode_mod_bundle(&bytes, &wrong_id).unwrap_err();
        assert!(err.contains("hash mismatch"), "{err}");

        // Tampering with payload bytes fails hash verification.
        let mut tampered = bytes.clone();
        let last = tampered.len() - 2;
        tampered[last] ^= 0x01;
        assert!(decode_mod_bundle(&tampered, &id).is_err());

        // Path safety checks reject traversal, absolute paths, backslashes, NUL, drive letters.
        for bad in [
            "",
            "/etc/passwd",
            "../evil.bp",
            "units/../../evil.bp",
            "units/./a.bp",
            "units//a.bp",
            "units\\UEL0201.bp",
            "C:units/UEL0201.bp",
            "units/a\0b.bp",
        ] {
            assert!(validate_bundle_rel_path(bad).is_err(), "accepted unsafe path '{bad}'");
            let bad_mod = Mod::from_memory(INFO, vec![(bad.into(), "UnitBlueprint {}".into())], "bad");
            assert!(encode_mod_bundle(&bad_mod).is_err());
        }
        assert!(validate_bundle_rel_path("units/UEL0201/UEL0201_unit.bp").is_ok());

        // Install into cache dir and verify round-trip reload matches hash.
        let cache = std::env::temp_dir().join(format!("farust-mod-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache);
        let installed = install_cached_mod(&decoded, &cache).unwrap();
        assert!(installed.starts_with(&cache));
        let found = find_local_mod(&id, &[], &cache, false).expect("cached mod found");
        assert_eq!(found.identity(), id);
        let _ = std::fs::remove_dir_all(&cache);
    }

    #[test]
    fn conflict_detection_catches_overrides_duplicate_uids_and_parse_errors() {
        let a = Mod::from_memory(
            "name = 'Mod A'\nuid = 'mod-a'\nversion = 1\n",
            vec![("units/UEL0201/UEL0201_unit.bp".into(), "UnitBlueprint { Merge = true, Defense = { MaxHealth = 400 } }".into())],
            "a",
        );
        let b = Mod::from_memory(
            "name = 'Mod B'\nuid = 'mod-b'\nversion = 1\n",
            vec![("units/UEL0201/UEL0201_unit.bp".into(), "UnitBlueprint { Categories = {'UEF'} }".into())],
            "b",
        );
        let c = Mod::from_memory(
            "name = 'Mod C'\nuid = 'mod-a'\nversion = 2\n",
            vec![("units/URL0107/URL0107_unit.bp".into(), "UnitBlueprint { Merge = true }".into())],
            "c",
        );
        let broken = Mod::from_memory(
            "name = 'Broken'\nuid = 'mod-broken'\nversion = 1\n",
            vec![("units/BAD/BAD_unit.bp".into(), "UnitBlueprint { unclosed ".into())],
            "broken",
        );
        // No conflicts when single valid mod.
        assert!(detect_conflicts(std::slice::from_ref(&a)).is_empty());

        let conflicts = detect_conflicts(&[a, b, c, broken]);
        assert!(conflicts.iter().any(|k| matches!(k, ModConflict::BlueprintOverride { blueprint_id, mods } if blueprint_id == "UEL0201" && mods.len() == 2)));
        assert!(conflicts.iter().any(|k| matches!(k, ModConflict::DuplicateUid { uid, .. } if uid == "mod-a")));
        assert!(conflicts.iter().any(|k| matches!(k, ModConflict::LoadError { spec, .. } if spec == "Broken")));
    }

    #[test]
    fn settings_enable_disable_persists_mods() {
        let root = std::env::temp_dir().join(format!("farust-mod-settings-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mod_dir = root.join("my_mod");
        std::fs::create_dir_all(mod_dir.join("units/UEL0201")).unwrap();
        std::fs::write(mod_dir.join("mod_info.lua"), INFO).unwrap();
        std::fs::write(
            mod_dir.join("units/UEL0201/UEL0201_unit.bp"),
            "UnitBlueprint { Merge = true, Defense = { MaxHealth = 500 } }",
        )
        .unwrap();
        let ini = root.join("settings.ini");

        let (m, conflicts) = enable_in_settings(&mod_dir.display().to_string(), &ini).unwrap();
        assert_eq!(m.uid, "heavy-tanks-001");
        assert!(conflicts.is_empty());
        let (loaded, warns) = crate::settings::Settings::load_from(&ini);
        assert!(warns.is_empty(), "{warns:?}");
        assert_eq!(loaded.mods.len(), 1);

        // Enabling again does not duplicate.
        enable_in_settings(&mod_dir.display().to_string(), &ini).unwrap();
        let (loaded2, _) = crate::settings::Settings::load_from(&ini);
        assert_eq!(loaded2.mods.len(), 1);

        // Disabling by uid removes it.
        assert!(disable_in_settings("heavy-tanks-001", &ini).unwrap());
        let (loaded3, _) = crate::settings::Settings::load_from(&ini);
        assert!(loaded3.mods.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn mod_weapon_and_ability_override_changes_sim_outcome() {
        use crate::pathfinding::PathGrid;
        use crate::sim::SCALE;
        use crate::skirmish::{ArmySetup, Command, Role, Skirmish, UnitType};

        let base_bp = r#"UnitBlueprint {
            Description = 'Light Tank',
            Categories = {'UEF','MOBILE','LAND','TECH1','DIRECTFIRE'},
            Defense = { MaxHealth = 250 },
            Economy = { BuildCostMass = 50, BuildCostEnergy = 250, BuildTime = 200 },
            Intel = { VisionRadius = 20 },
            Physics = { MaxSpeed = 3.5, MotionType = 'RULEUMT_Land' },
            Weapon = {
                {
                    Label = 'MainGun',
                    Damage = 15,
                    MaxRadius = 18,
                    RateOfFire = 0.5,
                    MuzzleVelocity = 25,
                    DamageRadius = 0,
                },
            },
        }"#;
        let mod_patch = Mod::from_memory(
            "name = 'Super Gun'\nuid = 'super-gun-1'\nversion = 1\n",
            vec![(
                "units/UEL0201/UEL0201_unit.bp".into(),
                r#"UnitBlueprint {
                    Merge = true,
                    Defense = {
                        Shield = { ShieldMaxHealth = 200, ShieldSize = 4, ShieldRegenRate = 10 },
                    },
                    Intel = { RadarStealth = true },
                    Weapon = {
                        {
                            Damage = 90,
                            MaxRadius = 32,
                            RateOfFire = 2.0,
                            MuzzleVelocity = 50,
                            DamageRadius = 3.0,
                            Buffs = {
                                { BuffType = 'STUN', Duration = 2.0, Radius = 2.5, Add = { OnImpact = true } },
                            },
                        },
                    },
                }"#
                .into(),
            )],
            "super-gun",
        );

        let mut stock_roots = BTreeMap::new();
        stock_roots.insert("UEL0201".into(), blueprint::parse_blueprint(base_bp).unwrap());
        let mut modded_roots = stock_roots.clone();
        apply(&mut modded_roots, &[mod_patch]).unwrap();

        let stock_ty = UnitType::from_blueprint("UEL0201", Role::Mobile, &stock_roots["UEL0201"]);
        let mod_ty = UnitType::from_blueprint("UEL0201", Role::Mobile, &modded_roots["UEL0201"]);
        // Verify weapon parameters & abilities all reflect blueprint data overrides.
        assert_eq!(stock_ty.weapons[0].damage, 15 * SCALE);
        assert_eq!(mod_ty.weapons[0].damage, 90 * SCALE);
        assert_eq!(mod_ty.weapons[0].range, 32 * SCALE);
        assert_eq!(mod_ty.weapons[0].splash_radius, 3 * SCALE);
        assert_eq!(mod_ty.weapons[0].muzzle_velocity, 5 * SCALE);
        assert!(mod_ty.weapons[0].reset < stock_ty.weapons[0].reset);
        assert_eq!(mod_ty.weapons[0].stun, Some((20, 2500)));
        assert_eq!(mod_ty.shield_max_hp, 200 * SCALE);
        assert!(mod_ty.ext.intel.radar_stealth);

        // Run identical 1v1 tank duel with stock vs modded UEL0201 and verify sim hash & outcome differ.
        let run_duel = |ty: UnitType| {
            let mut cat = crate::skirmish::tests::test_catalog();
            cat.types.insert("UEL0201".into(), ty);
            let grid = PathGrid::new(64, 64, 4 * SCALE);
            let setups = vec![
                ArmySetup { name: "ARMY_1".into(), faction: 0, start: (40 * SCALE, 40 * SCALE), team: 0 },
                ArmySetup { name: "ARMY_2".into(), faction: 1, start: (200 * SCALE, 200 * SCALE), team: 1 },
            ];
            let mut w = Skirmish::new(cat, grid, (256 * SCALE, 256 * SCALE), vec![], &setups, 42).unwrap();
            let u0 = w.spawn_unit(0, "UEL0201", (100 * SCALE, 100 * SCALE)).unwrap();
            let u1 = w.spawn_unit(1, "URL0107", (125 * SCALE, 100 * SCALE)).unwrap();
            w.issue(0, Command::Attack { units: vec![u0], target: u1, queue: false });
            w.issue(1, Command::Attack { units: vec![u1], target: u0, queue: false });
            w.run(120);
            (w.state_hash(), w.unit(u1).is_none())
        };
        let (h_stock, enemy_dead_stock) = run_duel(stock_ty);
        let (h_mod, enemy_dead_mod) = run_duel(mod_ty);
        assert_ne!(h_stock, h_mod, "modded weapon override must change sim hash");
        assert!(enemy_dead_mod, "modded heavy weapon should destroy enemy quickly");
        let _ = enemy_dead_stock;
    }
}
