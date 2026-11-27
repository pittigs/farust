//! Phase 20 user settings: `$XDG_CONFIG_HOME/farust/settings.ini` (fallback
//! `~/.config/farust/settings.ini`), a flat `key=value` file with `#`/`;` comments.
//!
//! Unknown keys are ignored, invalid values fall back to their defaults (each reported as
//! a warning), and [`Settings::to_ini`] always writes every key, so a load → save round
//! trip normalizes the file. The front end ([`crate::menu`]) edits and saves it; the
//! active copy is published process-wide ([`set_active`]) so window creation in `play.rs`
//! can honor fullscreen / vsync without threading the struct through every constructor.

use std::path::{Path, PathBuf};
use std::sync::RwLock;

pub const FILE_NAME: &str = "settings.ini";
pub const FACTION_KEYS: [&str; 4] = ["uef", "aeon", "cybran", "seraphim"];
/// Resolution presets offered by the options screen (any valid `WxH` is accepted in the file).
pub const RESOLUTIONS: [(u32, u32); 9] = [
    (1024, 768),
    (1280, 720),
    (1280, 800),
    (1366, 768),
    (1600, 900),
    (1680, 1050),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
];
pub const MAX_NAME_LEN: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowQuality {
    Off,
    Low,
    Medium,
    High,
}

impl ShadowQuality {
    pub const ALL: [ShadowQuality; 4] = [Self::Off, Self::Low, Self::Medium, Self::High];

    pub fn key(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|q| q.key().eq_ignore_ascii_case(s.trim()))
    }

    /// Shadow-map edge length a renderer should use (`0` = shadows disabled).
    pub fn map_size(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Low => 1024,
            Self::Medium => 2048,
            Self::High => 4096,
        }
    }
}

/// Front-end & in-match HUD visual theme (`modern` = default FARust gunmetal/cyan,
/// `classic` = old-school Forged Alliance bottom console + faction skins).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiTheme {
    #[default]
    Modern,
    Classic,
}

impl UiTheme {
    pub const ALL: [UiTheme; 2] = [Self::Modern, Self::Classic];

    pub fn key(self) -> &'static str {
        match self {
            Self::Modern => "modern",
            Self::Classic => "classic",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Modern => "Modern",
            Self::Classic => "Classic (FA)",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "modern" => Some(Self::Modern),
            "classic" | "fa" | "oldschool" => Some(Self::Classic),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub resolution: (u32, u32),
    pub fullscreen: bool,
    pub vsync: bool,
    pub shadow_quality: ShadowQuality,
    /// Volumes in percent, `0..=100` (100 = the audio engine's stock mix).
    pub master_volume: u8,
    pub music_volume: u8,
    pub sfx_volume: u8,
    /// Index into [`FACTION_KEYS`].
    pub default_faction: usize,
    pub player_name: String,
    pub edge_scroll: bool,
    /// Last address used in "Join game" (`host:port`).
    pub last_host: String,
    /// Phase 23: FA grid hotkeys (Q W E R T / A S D F G / Z X C V B) on the command card
    /// while a builder or factory is selected (off = legacy single-key bindings win).
    pub card_hotkeys: bool,
    /// Phase 27: UI layout & skin theme (`modern` or `classic`).
    pub ui_theme: UiTheme,
    /// Phase 37: localization language code (`us`, `de`, `fr`, ...).
    pub language: String,
    /// Phase 26: lightweight TCP rendezvous + relay server (`HOST:PORT`).
    pub rendezvous_server: String,
    /// Phase 23: remappable key bindings (`key.<action>=Ctrl+Shift+K`).
    pub keys: crate::keymap::KeyMap,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            resolution: (1280, 800),
            fullscreen: false,
            vsync: true,
            shadow_quality: ShadowQuality::High,
            master_volume: 100,
            music_volume: 80,
            sfx_volume: 100,
            default_faction: 0,
            player_name: "Commander".to_string(),
            edge_scroll: true,
            last_host: "127.0.0.1:15000".to_string(),
            card_hotkeys: false,
            ui_theme: UiTheme::Modern,
            language: "us".to_string(),
            rendezvous_server: crate::rendezvous::DEFAULT_ADDR.to_string(),
            keys: crate::keymap::KeyMap::default(),
        }
    }
}

/// Every key written to / understood in `settings.ini`, in file order (followed by one
/// `key.<action>` line per bindable action, see [`crate::keymap`]).
pub const KEYS: [&str; 15] = [
    "resolution",
    "fullscreen",
    "vsync",
    "shadow_quality",
    "master_volume",
    "music_volume",
    "sfx_volume",
    "default_faction",
    "player_name",
    "edge_scroll",
    "last_host",
    "card_hotkeys",
    "ui_theme",
    "language",
    "rendezvous_server",
];

fn parse_bool(v: &str) -> Result<bool, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("'{other}' is not a boolean (true/false)")),
    }
}

fn parse_percent(v: &str) -> Result<u8, String> {
    let n: u32 = v.trim().parse().map_err(|_| format!("'{}' is not a number 0..100", v.trim()))?;
    if n > 100 {
        return Err(format!("{n} is out of range 0..100"));
    }
    Ok(n as u8)
}

pub fn parse_resolution(v: &str) -> Result<(u32, u32), String> {
    let (w, h) = v
        .trim()
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("'{}' is not WIDTHxHEIGHT", v.trim()))?;
    let w: u32 = w.trim().parse().map_err(|_| format!("bad width '{w}'"))?;
    let h: u32 = h.trim().parse().map_err(|_| format!("bad height '{h}'"))?;
    if !(640..=7680).contains(&w) || !(480..=4320).contains(&h) {
        return Err(format!("{w}x{h} outside 640x480..7680x4320"));
    }
    Ok((w, h))
}

/// Player names: 1..=20 printable ASCII characters (no `=`), trimmed.
pub fn validate_name(v: &str) -> Result<String, String> {
    let t = v.trim();
    if t.is_empty() {
        return Err("player name is empty".into());
    }
    if t.len() > MAX_NAME_LEN {
        return Err(format!("player name longer than {MAX_NAME_LEN} characters"));
    }
    if !t.bytes().all(|b| (0x20..0x7f).contains(&b) && b != b'=') {
        return Err("player name must be printable ASCII without '='".into());
    }
    Ok(t.to_string())
}

/// `host:port` with a non-empty host and a valid non-zero port.
pub fn validate_host_addr(v: &str) -> Result<String, String> {
    let t = v.trim();
    let (host, port) = t
        .rsplit_once(':')
        .ok_or_else(|| format!("'{t}' must be HOST:PORT"))?;
    if host.is_empty() || host.contains(char::is_whitespace) {
        return Err(format!("'{t}' has an empty or invalid host"));
    }
    match port.parse::<u16>() {
        Ok(p) if p > 0 => Ok(t.to_string()),
        _ => Err(format!("'{port}' is not a valid port 1..65535")),
    }
}

pub fn faction_index(key: &str) -> Option<usize> {
    FACTION_KEYS.iter().position(|k| k.eq_ignore_ascii_case(key.trim()))
}

/// Validates a localization language code (`us`, `de`, `fr`, `en_US`, ...).
pub fn validate_language(v: &str) -> Result<String, String> {
    let t = v.trim();
    if !(2..=12).contains(&t.len())
        || !t.bytes().all(|b| b.is_ascii_alphabetic() || b == b'_' || b == b'-')
    {
        return Err(format!("'{t}' is not a valid language code (e.g. us, de, fr)"));
    }
    let norm = crate::loc::normalize_lang(t);
    if !(2..=4).contains(&norm.len()) || !norm.bytes().all(|b| b.is_ascii_alphabetic()) {
        return Err(format!("'{t}' is not a valid language code"));
    }
    Ok(norm)
}

impl Settings {
    /// Applies one `key=value` pair; returns an error (and leaves the field unchanged)
    /// for unknown keys or invalid values.
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let v = value.trim();
        match key.trim().to_ascii_lowercase().as_str() {
            "resolution" => self.resolution = parse_resolution(v)?,
            "fullscreen" => self.fullscreen = parse_bool(v)?,
            "vsync" => self.vsync = parse_bool(v)?,
            "shadow_quality" => {
                self.shadow_quality =
                    ShadowQuality::parse(v).ok_or_else(|| format!("'{v}' is not off/low/medium/high"))?
            }
            "master_volume" => self.master_volume = parse_percent(v)?,
            "music_volume" => self.music_volume = parse_percent(v)?,
            "sfx_volume" => self.sfx_volume = parse_percent(v)?,
            "default_faction" => {
                self.default_faction =
                    faction_index(v).ok_or_else(|| format!("'{v}' is not uef/aeon/cybran/seraphim"))?
            }
            "player_name" => self.player_name = validate_name(v)?,
            "edge_scroll" => self.edge_scroll = parse_bool(v)?,
            "last_host" => self.last_host = validate_host_addr(v)?,
            "card_hotkeys" => self.card_hotkeys = parse_bool(v)?,
            "ui_theme" => {
                self.ui_theme =
                    UiTheme::parse(v).ok_or_else(|| format!("'{v}' is not modern/classic"))?
            }
            "language" => self.language = validate_language(v)?,
            "rendezvous_server" => self.rendezvous_server = validate_host_addr(v)?,
            other => match other.strip_prefix("key.") {
                Some(id) => self.keys.set_from_setting(id, v)?,
                None => return Err(format!("unknown key '{other}'")),
            },
        }
        Ok(())
    }

    /// Value of `key` exactly as written to the file.
    pub fn get(&self, key: &str) -> Option<String> {
        let b = |v: bool| if v { "true" } else { "false" }.to_string();
        Some(match key {
            "resolution" => format!("{}x{}", self.resolution.0, self.resolution.1),
            "fullscreen" => b(self.fullscreen),
            "vsync" => b(self.vsync),
            "shadow_quality" => self.shadow_quality.key().to_string(),
            "master_volume" => self.master_volume.to_string(),
            "music_volume" => self.music_volume.to_string(),
            "sfx_volume" => self.sfx_volume.to_string(),
            "default_faction" => FACTION_KEYS[self.default_faction.min(3)].to_string(),
            "player_name" => self.player_name.clone(),
            "edge_scroll" => b(self.edge_scroll),
            "last_host" => self.last_host.clone(),
            "card_hotkeys" => b(self.card_hotkeys),
            "ui_theme" => self.ui_theme.key().to_string(),
            "language" => self.language.clone(),
            "rendezvous_server" => self.rendezvous_server.clone(),
            other => {
                let id = other.strip_prefix("key.")?;
                let action = crate::keymap::action_by_id(id)?;
                crate::keymap::chords_to_string(self.keys.get(action))
            }
        })
    }

    /// Parses `settings.ini` text on top of the defaults. Returns the settings plus one
    /// warning per ignored line (unknown key, invalid value, missing `=`).
    pub fn parse(text: &str) -> (Self, Vec<String>) {
        let mut s = Self::default();
        let mut warnings = Vec::new();
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') || line.starts_with('[') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                warnings.push(format!("line {}: missing '='", n + 1));
                continue;
            };
            if let Err(e) = s.set(k, v) {
                warnings.push(format!("line {}: {e}", n + 1));
            }
        }
        (s, warnings)
    }

    pub fn to_ini(&self) -> String {
        let mut out = String::from("# FARust settings (written by the options screen)\n");
        for k in KEYS {
            out.push_str(k);
            out.push('=');
            out.push_str(&self.get(k).unwrap_or_default());
            out.push('\n');
        }
        out.push_str("# Key bindings: key.<action>=Ctrl+Shift+K (comma-separated alternatives, empty = unbound)\n");
        for (k, v) in self.keys.to_settings() {
            out.push_str(&k);
            out.push('=');
            out.push_str(&v);
            out.push('\n');
        }
        out
    }

    pub fn load_from(path: &Path) -> (Self, Vec<String>) {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (Self::default(), Vec::new()),
            Err(e) => (Self::default(), vec![format!("cannot read {}: {e}", path.display())]),
        }
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let tmp = path.with_extension("ini.tmp");
        std::fs::write(&tmp, self.to_ini()).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("cannot replace {}: {e}", path.display()))
    }

    #[allow(dead_code)]
    pub fn faction_key(&self) -> &'static str {
        FACTION_KEYS[self.default_faction.min(FACTION_KEYS.len() - 1)]
    }

    /// `(master, music, sfx)` as linear gains `0.0..=1.0`.
    #[allow(dead_code)]
    pub fn volume_gains(&self) -> (f32, f32, f32) {
        (
            self.master_volume as f32 / 100.0,
            self.music_volume as f32 / 100.0,
            self.sfx_volume as f32 / 100.0,
        )
    }
}

fn env_dir(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// `$XDG_CONFIG_HOME/farust` or `~/.config/farust`.
pub fn config_dir() -> Option<PathBuf> {
    env_dir("XDG_CONFIG_HOME")
        .or_else(|| env_dir("HOME").map(|h| h.join(".config")))
        .map(|d| d.join("farust"))
}

/// `$XDG_DATA_HOME/farust` or `~/.local/share/farust`.
#[allow(dead_code)]
pub fn data_dir() -> Option<PathBuf> {
    env_dir("XDG_DATA_HOME")
        .or_else(|| env_dir("HOME").map(|h| h.join(".local").join("share")))
        .map(|d| d.join("farust"))
}

pub fn settings_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join(FILE_NAME))
}

// ---------------------------------------------------------------------------
// Process-wide active settings (window-creation hooks)
// ---------------------------------------------------------------------------

static ACTIVE: RwLock<Option<Settings>> = RwLock::new(None);

/// Publishes the settings window creation should honor (`None` = stock behavior, used by
/// the plain CLI commands).
pub fn set_active(s: Option<Settings>) {
    if let Ok(mut g) = ACTIVE.write() {
        *g = s;
    }
}

pub fn active() -> Option<Settings> {
    ACTIVE.read().ok().and_then(|g| g.clone())
}

/// Window hook: borderless fullscreen when the active settings ask for it.
pub fn apply_window_attrs(attrs: winit::window::WindowAttributes) -> winit::window::WindowAttributes {
    match active() {
        Some(s) if s.fullscreen => attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))),
        _ => attrs,
    }
}

/// Surface present mode for the active settings (`Fifo` when none are active).
pub fn present_mode() -> wgpu::PresentMode {
    match active() {
        Some(s) if !s.vsync => wgpu::PresentMode::AutoNoVsync,
        Some(_) => wgpu::PresentMode::AutoVsync,
        None => wgpu::PresentMode::Fifo,
    }
}

/// Shadow quality for the active settings (`Medium` = stock 2048px when none are active).
pub fn shadow_quality() -> ShadowQuality {
    active().map(|s| s.shadow_quality).unwrap_or(ShadowQuality::Medium)
}

/// Whether camera edge scrolling is enabled (defaults to on without active settings).
pub fn edge_scroll_enabled() -> bool {
    active().is_none_or(|s| s.edge_scroll)
}

/// Settings for in-match input: the active settings, else (plain CLI `play`) the user's
/// `settings.ini` read once, else defaults. Tests always see defaults.
pub fn input_settings() -> Settings {
    if let Some(s) = active() {
        return s;
    }
    #[cfg(test)]
    {
        Settings::default()
    }
    #[cfg(not(test))]
    {
        static FILE: std::sync::OnceLock<Settings> = std::sync::OnceLock::new();
        FILE.get_or_init(|| settings_path().map(|p| Settings::load_from(&p).0).unwrap_or_default())
            .clone()
    }
}

/// Whether the command card grid hotkeys are enabled (`card_hotkeys`, or
/// `FARUST_CARD_HOTKEYS=1`).
pub fn card_hotkeys_enabled() -> bool {
    input_settings().card_hotkeys || std::env::var("FARUST_CARD_HOTKEYS").is_ok_and(|v| v == "1")
}

/// Active UI visual theme (`FARUST_UI_THEME` env var overrides `settings.ini`, default `Modern`).
pub fn ui_theme() -> UiTheme {
    if let Ok(v) = std::env::var("FARUST_UI_THEME")
        && let Some(t) = UiTheme::parse(&v)
    {
        return t;
    }
    input_settings().ui_theme
}

/// Active localization language (`FARUST_LANG` env var overrides `settings.ini`, default `"us"`).
pub fn language() -> String {
    if let Ok(v) = std::env::var("FARUST_LANG")
        && let Ok(code) = validate_language(&v)
    {
        return code;
    }
    input_settings().language
}

/// Active rendezvous server address (`FARUST_RENDEZVOUS` env var overrides `settings.ini`).
pub fn effective_rendezvous(override_s: Option<&Settings>) -> String {
    if let Ok(v) = std::env::var("FARUST_RENDEZVOUS") {
        let t = v.trim();
        if !t.is_empty() {
            return t.to_string();
        }
    }
    override_s
        .map(|s| s.rendezvous_server.clone())
        .unwrap_or_else(|| input_settings().rendezvous_server)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_every_key() {
        let s = Settings {
            resolution: (1920, 1080),
            fullscreen: true,
            vsync: false,
            shadow_quality: ShadowQuality::Low,
            master_volume: 35,
            music_volume: 0,
            sfx_volume: 100,
            default_faction: 2,
            player_name: "Red Leader 7".into(),
            edge_scroll: false,
            last_host: "192.168.1.20:15400".into(),
            card_hotkeys: true,
            ui_theme: UiTheme::Classic,
            language: "de".into(),
            rendezvous_server: "relay.example.org:15050".into(),
            keys: {
                let mut k = crate::keymap::KeyMap::default();
                k.set_from_setting("build_factory", "Ctrl+Shift+K").unwrap();
                k.set_from_setting("toggle_repeat", "").unwrap();
                k
            },
        };
        let text = s.to_ini();
        for k in KEYS {
            assert!(text.contains(&format!("{k}=")), "{k} missing in\n{text}");
        }
        assert!(text.contains("\nkey.build_factory=Ctrl+Shift+K\n"), "{text}");
        assert!(text.contains("\nkey.toggle_repeat=\n"), "{text}");
        assert!(text.contains("\nkey.bookmark_save_1=Ctrl+F7\n"), "{text}");
        assert_eq!(s.get("key.build_factory").as_deref(), Some("Ctrl+Shift+K"));
        let (back, warnings) = Settings::parse(&text);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(back, s);
        // Defaults round-trip too.
        assert_eq!(Settings::parse(&Settings::default().to_ini()).0, Settings::default());
    }

    #[test]
    fn invalid_values_fall_back_to_defaults_with_warnings() {
        let text = "# comment\n\
                    resolution = 100x50\n\
                    fullscreen=maybe\n\
                    master_volume=140\n\
                    music_volume= 25 \n\
                    default_faction=CYBRAN\n\
                    player_name=\n\
                    last_host=nohost\n\
                    shadow_quality=ultra\n\
                    bogus=1\n\
                    no equals sign\n\
                    vsync=off\n";
        let (s, w) = Settings::parse(text);
        let d = Settings::default();
        assert_eq!(s.resolution, d.resolution);
        assert_eq!(s.fullscreen, d.fullscreen);
        assert_eq!(s.master_volume, d.master_volume);
        assert_eq!(s.music_volume, 25);
        assert_eq!(s.default_faction, 2);
        assert_eq!(s.player_name, d.player_name);
        assert_eq!(s.last_host, d.last_host);
        assert_eq!(s.shadow_quality, d.shadow_quality);
        assert!(!s.vsync);
        assert_eq!(w.len(), 8, "{w:?}");
        assert!(w.iter().any(|m| m.contains("unknown key 'bogus'")));
        assert!(w.iter().any(|m| m.starts_with("line 11") && m.contains("missing '='")));
    }

    #[test]
    fn validators() {
        assert_eq!(parse_resolution("2560X1440"), Ok((2560, 1440)));
        assert!(parse_resolution("1280").is_err());
        assert!(parse_resolution("99999x10").is_err());
        assert_eq!(validate_name("  Ace  ").as_deref(), Ok("Ace"));
        assert!(validate_name("a=b").is_err());
        assert!(validate_name(&"x".repeat(MAX_NAME_LEN + 1)).is_err());
        assert!(validate_host_addr("example.org:15000").is_ok());
        assert!(validate_host_addr("[::1]:15000").is_ok());
        assert!(validate_host_addr("host:0").is_err());
        assert!(validate_host_addr(":15000").is_err());
        assert!(validate_host_addr("host:99999").is_err());
        assert_eq!(ShadowQuality::parse("MEDIUM"), Some(ShadowQuality::Medium));
    }

    #[test]
    fn shadow_quality_hook_defaults_and_follows_active() {
        // No active settings: stock Medium (2048px) so plain CLI keeps old behavior.
        let prev = active();
        set_active(None);
        assert_eq!(shadow_quality(), ShadowQuality::Medium);
        assert_eq!(shadow_quality().map_size(), 2048);
        let mut low = Settings::default();
        low.shadow_quality = ShadowQuality::Low;
        set_active(Some(low));
        assert_eq!(shadow_quality(), ShadowQuality::Low);
        assert_eq!(shadow_quality().map_size(), 1024);
        let mut off = Settings::default();
        off.shadow_quality = ShadowQuality::Off;
        set_active(Some(off));
        assert_eq!(shadow_quality().map_size(), 0);
        set_active(prev);
    }

    #[test]
    fn save_and_load_file() {
        let dir = std::env::temp_dir().join(format!("farust-settings-{}", std::process::id()));
        let path = dir.join("sub").join(FILE_NAME);
        let mut s = Settings::default();
        s.player_name = "Saver".into();
        s.sfx_volume = 12;
        s.save_to(&path).unwrap();
        let (back, w) = Settings::load_from(&path);
        assert!(w.is_empty());
        assert_eq!(back, s);
        let _ = std::fs::remove_dir_all(&dir);
        // Missing file = defaults, no warnings.
        let (d, w) = Settings::load_from(&dir.join("missing.ini"));
        assert_eq!(d, Settings::default());
        assert!(w.is_empty());
    }

    #[test]
    fn ui_theme_and_rendezvous_round_trip_and_defaults() {
        assert_eq!(UiTheme::parse("CLASSIC"), Some(UiTheme::Classic));
        assert_eq!(UiTheme::parse("fa"), Some(UiTheme::Classic));
        assert_eq!(UiTheme::parse("modern"), Some(UiTheme::Modern));
        assert_eq!(UiTheme::parse("unknown"), None);

        let (s, w) = Settings::parse("ui_theme=retro\nrendezvous_server=badaddr\n");
        assert_eq!(s.ui_theme, UiTheme::Modern);
        assert_eq!(s.rendezvous_server, crate::rendezvous::DEFAULT_ADDR);
        assert_eq!(w.len(), 2, "{w:?}");

        let (c, cw) = Settings::parse("ui_theme=classic\nlanguage=DE\nrendezvous_server=10.0.0.5:15050\n");
        assert!(cw.is_empty(), "{cw:?}");
        assert_eq!(c.ui_theme, UiTheme::Classic);
        assert_eq!(c.language, "de");
        assert_eq!(c.rendezvous_server, "10.0.0.5:15050");
        assert_eq!(c.get("ui_theme").as_deref(), Some("classic"));
        assert_eq!(c.get("language").as_deref(), Some("de"));
        assert_eq!(c.get("rendezvous_server").as_deref(), Some("10.0.0.5:15050"));

        let (bad, bw) = Settings::parse("language=123!!\n");
        assert_eq!(bad.language, "us");
        assert_eq!(bw.len(), 1, "{bw:?}");
        assert_eq!(validate_language("en_US").as_deref(), Ok("us"));
        assert_eq!(validate_language("de-DE").as_deref(), Ok("de"));
        assert_eq!(validate_language("fr").as_deref(), Ok("fr"));
    }
}
