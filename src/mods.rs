//! Phase 24 blueprint mods.
//!
//! A mod is a directory (default location `$XDG_DATA_HOME/farust/mods/<name>/`, override
//! with `FARUST_MODS_DIR`) holding
//!
//! * `mod_info.lua` — only the top-level `name`, `uid` and `version` assignments are read
//!   (FA syntax, e.g. `name = "Heavy Tanks"`); everything else is ignored, no Lua runs;
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
//! Mods are enabled per process with the global `--mod <DIR|NAME>` flag (repeatable) or
//! `FARUST_MODS=a,b`. Each enabled mod's identity (name, uid, version and an FNV-1a 64
//! hash over its `mod_info.lua` and blueprint files) is written into replay / save
//! headers and the lockstep handshake; content mismatches are rejected with a clear error.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::blueprint::{self, Value};

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

/// One loaded mod: metadata plus its blueprint files `(relative path, text)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mod {
    pub name: String,
    pub uid: String,
    pub version: String,
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

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

/// Reads `name`, `uid` and `version` from `mod_info.lua` text.
pub fn parse_mod_info(text: &str) -> (Option<String>, Option<String>, Option<String>) {
    let (mut name, mut uid, mut version) = (None, None, None);
    for line in text.lines() {
        let line = line.split("--").next().unwrap_or("").trim();
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let v = v.trim().trim_end_matches([',', ';']).trim();
        let v = v
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .or_else(|| v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
            .unwrap_or(v)
            .to_string();
        match k.trim() {
            "name" if name.is_none() => name = Some(v),
            "uid" if uid.is_none() => uid = Some(v),
            "version" if version.is_none() => version = Some(v),
            _ => {}
        }
    }
    (name, uid, version)
}

impl Mod {
    /// Builds a mod from in-memory `mod_info.lua` text and `(path, text)` blueprint files.
    pub fn from_memory(mod_info: &str, files: Vec<(String, String)>, fallback_name: &str) -> Self {
        let (name, uid, version) = parse_mod_info(mod_info);
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
        let name = name.filter(|s| !s.is_empty()).unwrap_or_else(|| fallback_name.to_string());
        Self {
            uid: uid.filter(|s| !s.is_empty()).unwrap_or_else(|| name.clone()),
            name,
            version: version.filter(|s| !s.is_empty()).unwrap_or_else(|| "1".to_string()),
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

/// `$FARUST_MODS_DIR`, else `$XDG_DATA_HOME/farust/mods` (`~/.local/share/farust/mods`).
pub fn mods_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("FARUST_MODS_DIR") {
        return PathBuf::from(d);
    }
    crate::scenario::xdg_data_home().join("farust").join("mods")
}

/// Every mod directory in [`mods_dir`], sorted by folder name.
pub fn discover() -> Vec<(PathBuf, Result<Mod, String>)> {
    let dir = mods_dir();
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    dirs.into_iter().map(|d| {
        let m = Mod::load_dir(&d);
        (d, m)
    }).collect()
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

static ENABLED: OnceLock<Vec<String>> = OnceLock::new();
static ACTIVE: OnceLock<Result<Vec<Mod>, String>> = OnceLock::new();

/// Sets the process-wide enabled mod specs (from `--mod`). Must run before the first
/// catalog load; later calls are ignored.
pub fn set_enabled(specs: Vec<String>) {
    let _ = ENABLED.set(specs);
}

fn enabled_specs() -> Vec<String> {
    if let Some(v) = ENABLED.get()
        && !v.is_empty()
    {
        return v.clone();
    }
    std::env::var("FARUST_MODS")
        .map(|s| s.split([',', ':']).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect())
        .unwrap_or_default()
}

/// The enabled mods, loaded once per process.
pub fn active() -> Result<&'static [Mod], String> {
    ACTIVE
        .get_or_init(|| enabled_specs().iter().map(|s| resolve(s)).collect())
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
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

/// Binary codec shared by replays and the lockstep handshake.
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

/// `farust mods [GAME_DIR]`: installed mods in [`mods_dir`] plus the enabled ones, with
/// identity and a dry-run of their blueprint changes against the stock roster.
pub fn cmd_mods(args: &[String]) -> Result<(), String> {
    let game_dir = args
        .iter()
        .skip(2)
        .find(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(crate::scd::default_game_dir);
    let stock = crate::skirmish::Catalog::read_blueprints(&game_dir).ok();
    let enabled = active()?;
    let enabled_ids: Vec<ModIdentity> = enabled.iter().map(Mod::identity).collect();
    let mut listed: Vec<Mod> = Vec::new();
    println!("mods dir: {}", mods_dir().display());
    let found = discover();
    if found.is_empty() {
        println!("  (no mods installed)");
    }
    for (dir, m) in found {
        match m {
            Ok(m) => listed.push(m),
            Err(e) => println!("  {}  ERROR {e}", dir.display()),
        }
    }
    for m in enabled {
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
    println!("enable with: farust --mod <DIR|NAME> <subcommand> ...  (or FARUST_MODS=a,b)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const INFO: &str = "name = \"Heavy Tanks\" -- comment\nuid = 'heavy-tanks-001'\nversion = 2\ndescription = \"x\"\n";

    fn base() -> BTreeMap<String, Value> {
        let mut m = BTreeMap::new();
        m.insert(
            "UEL0201".to_string(),
            blueprint::parse_blueprint("UnitBlueprint { Categories = {'UEF','MOBILE','LAND'}, Defense = { MaxHealth = 300 }, Weapon = { { Label = 'Gun', Damage = 10 } } }").unwrap(),
        );
        m
    }

    fn sample_mod() -> Mod {
        Mod::from_memory(
            INFO,
            vec![
                ("units/UEL0201/UEL0201_unit.bp".into(), "UnitBlueprint { Merge = true, Defense = { MaxHealth = 600 }, Weapon = { { Damage = 25 } } }".into()),
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
        assert_eq!(t.get("Weapon").unwrap().table_list()[0].get_f64("Damage"), Some(25.0));
        assert_eq!(t.get("Weapon").unwrap().table_list()[0].get_str("Label"), Some("Gun"));
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
        let _ = std::fs::remove_dir_all(&dir);
    }
}
