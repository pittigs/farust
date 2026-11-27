//! Phase 56: per-OS user directories and Forged Alliance install discovery.
//!
//! Linux keeps the XDG layout (`$XDG_CONFIG_HOME`, `$XDG_DATA_HOME`, `$XDG_CACHE_HOME`,
//! falling back to `~/.config`, `~/.local/share`, `~/.cache`). Windows uses `%APPDATA%`
//! (config + data) and `%LOCALAPPDATA%` (cache); macOS uses `~/Library/Application Support`
//! and `~/Library/Caches`. An explicitly set absolute `XDG_*` variable wins on every OS
//! (handy for tests and portable installs).
//!
//! The game install is found (in order) via `$FA_PATH`, then every Steam library listed in
//! `steamapps/libraryfolders.vdf` of the known Steam roots, then GOG / retail / Wine-prefix
//! folders. A folder counts as an install when it has `gamedata/units.scd` (any case).

use std::path::{Path, PathBuf};

/// Steam's folder name for Supreme Commander: Forged Alliance (app 9420).
pub const FA_STEAM_DIR: &str = "Supreme Commander Forged Alliance";

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn env_abs(var: &str) -> Option<PathBuf> {
    env_path(var).filter(|p| p.is_absolute())
}

/// The user's home directory (`HOME`, or `USERPROFILE` on Windows).
pub fn home_dir() -> Option<PathBuf> {
    env_path("HOME").or_else(|| if cfg!(windows) { env_path("USERPROFILE") } else { None })
}

/// Base directory for configuration (`farust/` is appended by callers).
pub fn config_home() -> Option<PathBuf> {
    if let Some(d) = env_abs("XDG_CONFIG_HOME") {
        return Some(d);
    }
    if cfg!(windows) {
        env_path("APPDATA").or_else(|| home_dir().map(|h| h.join("AppData").join("Roaming")))
    } else if cfg!(target_os = "macos") {
        home_dir().map(|h| h.join("Library").join("Application Support"))
    } else {
        home_dir().map(|h| h.join(".config"))
    }
}

/// Base directory for user data (saves, replays, maps, mods).
pub fn data_home() -> PathBuf {
    if let Some(d) = env_abs("XDG_DATA_HOME") {
        return d;
    }
    let base = if cfg!(windows) {
        env_path("APPDATA").or_else(|| home_dir().map(|h| h.join("AppData").join("Roaming")))
    } else if cfg!(target_os = "macos") {
        home_dir().map(|h| h.join("Library").join("Application Support"))
    } else {
        home_dir().map(|h| h.join(".local").join("share"))
    };
    base.unwrap_or_else(|| PathBuf::from("."))
}

/// Base directory for caches (extracted map archives, downloaded mods).
pub fn cache_home() -> PathBuf {
    if let Some(d) = env_abs("XDG_CACHE_HOME") {
        return d;
    }
    let base = if cfg!(windows) {
        env_path("LOCALAPPDATA").or_else(|| home_dir().map(|h| h.join("AppData").join("Local")))
    } else if cfg!(target_os = "macos") {
        home_dir().map(|h| h.join("Library").join("Caches"))
    } else {
        home_dir().map(|h| h.join(".cache"))
    };
    base.unwrap_or_else(std::env::temp_dir)
}

/// FA's own per-user folder (`Documents/My Games/Gas Powered Games/...`) on a native
/// Windows install; `None` elsewhere (Proton prefixes are derived from the install).
pub fn fa_documents_dir() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    home_dir().map(|h| {
        h.join("Documents")
            .join("My Games")
            .join("Gas Powered Games")
            .join("Supreme Commander Forged Alliance")
    })
}

/// Library roots listed in a Steam `libraryfolders.vdf` (`"path" "…"` entries; `\\` unescaped).
pub fn steam_library_paths(vdf: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for line in vdf.lines() {
        let mut parts = line.split('"').filter(|s| !s.trim().is_empty());
        if parts.next().is_some_and(|k| k.eq_ignore_ascii_case("path"))
            && let Some(v) = parts.next()
        {
            let p = PathBuf::from(v.replace("\\\\", "\\"));
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// Steam installation roots to look in on this OS.
pub fn steam_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if cfg!(windows) {
        for var in ["ProgramFiles(x86)", "ProgramFiles"] {
            if let Some(p) = env_path(var) {
                roots.push(p.join("Steam"));
            }
        }
        roots.push(PathBuf::from(r"C:\Program Files (x86)\Steam"));
    } else if let Some(h) = home_dir() {
        if cfg!(target_os = "macos") {
            roots.push(h.join("Library").join("Application Support").join("Steam"));
        } else {
            roots.push(h.join(".local").join("share").join("Steam"));
            roots.push(h.join(".steam").join("steam"));
            roots.push(h.join(".steam").join("root"));
            roots.push(h.join(".var/app/com.valvesoftware.Steam/.local/share/Steam"));
            roots.push(h.join("snap/steam/common/.local/share/Steam"));
        }
    }
    // `~/.steam/steam` and `~/.steam/root` are usually symlinks to the same install.
    let mut seen = Vec::new();
    roots.retain(|r| {
        let key = std::fs::canonicalize(r).unwrap_or_else(|_| r.clone());
        let fresh = !seen.contains(&key);
        seen.push(key);
        fresh
    });
    roots
}

/// Every place an FA install may live, most likely first (existing or not).
pub fn fa_install_candidates() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.contains(&p) {
            out.push(p);
        }
    };
    for root in steam_roots() {
        push(root.join("steamapps").join("common").join(FA_STEAM_DIR));
        if let Ok(vdf) = std::fs::read_to_string(root.join("steamapps").join("libraryfolders.vdf")) {
            for lib in steam_library_paths(&vdf) {
                push(lib.join("steamapps").join("common").join(FA_STEAM_DIR));
            }
        }
    }
    if cfg!(windows) {
        for drive in ["C:", "D:"] {
            push(PathBuf::from(format!(r"{drive}\GOG Games\Supreme Commander - Forged Alliance")));
        }
        for var in ["ProgramFiles(x86)", "ProgramFiles"] {
            if let Some(p) = env_path(var) {
                push(p.join("GOG Galaxy").join("Games").join("Supreme Commander Forged Alliance"));
                push(p.join("THQ").join("Gas Powered Games").join("Supreme Commander - Forged Alliance"));
            }
        }
    } else if let Some(h) = home_dir() {
        push(h.join("GOG Games").join("Supreme Commander Forged Alliance"));
        let wine = h.join(".wine").join("drive_c");
        push(wine.join("GOG Games").join("Supreme Commander - Forged Alliance"));
        push(wine.join("Program Files (x86)/THQ/Gas Powered Games/Supreme Commander - Forged Alliance"));
    }
    out
}

/// `dir` holds a Forged Alliance install (`gamedata/units.scd`, case-insensitive).
pub fn is_fa_install(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|e| {
        e.file_name().to_string_lossy().eq_ignore_ascii_case("gamedata")
            && std::fs::read_dir(e.path()).is_ok_and(|g| {
                g.flatten().any(|f| f.file_name().to_string_lossy().eq_ignore_ascii_case("units.scd"))
            })
    })
}

/// The first existing FA install among [`fa_install_candidates`].
pub fn discover_fa_install() -> Option<PathBuf> {
    fa_install_candidates().into_iter().find(|p| is_fa_install(p))
}

/// Game directory: `$FA_PATH`, else the discovered install, else the most likely Steam
/// location (so error messages name a sensible path).
pub fn default_game_dir() -> PathBuf {
    if let Some(d) = env_path("FA_PATH") {
        return d;
    }
    discover_fa_install()
        .or_else(|| fa_install_candidates().into_iter().next())
        .unwrap_or_else(|| PathBuf::from(FA_STEAM_DIR))
}

/// `farust platform`: prints the directories and install discovery for this machine.
pub fn cmd_platform() -> Result<(), String> {
    println!("os:        {} / {}", std::env::consts::OS, std::env::consts::ARCH);
    println!("config:    {}", config_home().map(|p| p.join("farust").display().to_string()).unwrap_or_else(|| "(none)".into()));
    println!("data:      {}", data_home().join("farust").display());
    println!("cache:     {}", cache_home().join("farust").display());
    if let Some(d) = fa_documents_dir() {
        println!("fa user:   {}", d.display());
    }
    match env_path("FA_PATH") {
        Some(p) => println!("FA_PATH:   {} ({})", p.display(), if is_fa_install(&p) { "ok" } else { "no gamedata/units.scd" }),
        None => println!("FA_PATH:   (unset)"),
    }
    println!("candidates:");
    for c in fa_install_candidates() {
        println!("  [{}] {}", if is_fa_install(&c) { "x" } else { " " }, c.display());
    }
    println!("game dir:  {}", default_game_dir().display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_steam_library_folders_including_windows_escapes() {
        let vdf = r#""libraryfolders"
{
	"0"
	{
		"path"		"/home/user/.local/share/Steam"
		"label"		""
		"apps" { "9420" "123" }
	}
	"1"
	{
		"path"		"D:\\SteamLibrary"
	}
	"2"
	{
		"PATH"		"/mnt/games/SteamLibrary"
	}
}"#;
        assert_eq!(
            steam_library_paths(vdf),
            vec![
                PathBuf::from("/home/user/.local/share/Steam"),
                PathBuf::from(r"D:\SteamLibrary"),
                PathBuf::from("/mnt/games/SteamLibrary"),
            ]
        );
        assert!(steam_library_paths("garbage \"label\" \"x\"").is_empty());
    }

    #[test]
    fn install_detection_is_case_insensitive_and_needs_units_scd() {
        let root = std::env::temp_dir().join(format!("farust-platform-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("GameData")).unwrap();
        assert!(!is_fa_install(&root), "no units.scd yet");
        std::fs::write(root.join("GameData").join("Units.SCD"), b"x").unwrap();
        assert!(is_fa_install(&root));
        assert!(!is_fa_install(&root.join("missing")));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn candidates_cover_steam_and_dirs_are_absolute_with_home() {
        let c = fa_install_candidates();
        assert!(c.iter().any(|p| p.ends_with(Path::new("steamapps/common").join(FA_STEAM_DIR))));
        if home_dir().is_some() {
            assert!(data_home().is_absolute());
            assert!(cache_home().is_absolute());
            assert!(config_home().is_some_and(|p| p.is_absolute()));
        }
    }
}
