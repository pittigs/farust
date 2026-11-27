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

/// Phase 42 anti-aliasing mode (`off`, `fxaa`, `msaa4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AaMode {
    Off,
    Fxaa,
    #[default]
    Msaa4,
}

impl AaMode {
    pub const ALL: [AaMode; 3] = [Self::Off, Self::Fxaa, Self::Msaa4];

    pub fn key(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fxaa => "fxaa",
            Self::Msaa4 => "msaa4",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Fxaa => "FXAA",
            Self::Msaa4 => "MSAA 4x",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "0" => Some(Self::Off),
            "fxaa" => Some(Self::Fxaa),
            "msaa4" | "msaa" | "msaa4x" | "4x" => Some(Self::Msaa4),
            _ => None,
        }
    }

    /// MSAA sample count for the 3D HDR scene target (`4` for `Msaa4`, `1` otherwise).
    pub fn sample_count(self) -> u32 {
        match self {
            Self::Msaa4 => 4,
            Self::Off | Self::Fxaa => 1,
        }
    }
}

/// Gamma bounds in hundredths (`80..=140` => `0.80..=1.40`, default `100` => `1.00`).
pub const GAMMA_MIN_X100: u16 = 80;
pub const GAMMA_MAX_X100: u16 = 140;
pub const GAMMA_DEFAULT_X100: u16 = 100;

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
    /// Phase 39 accessibility: colour-blind safe army palette, swapped mouse buttons and
    /// on-screen captions for announcer voice lines.
    pub colorblind: bool,
    pub mouse_swap: bool,
    pub subtitles: bool,
    /// Phase 42 HDR post-processing: bloom, anti-aliasing (`off|fxaa|msaa4`), gamma (`80..=140` => `0.80..=1.40`).
    pub bloom: bool,
    pub aa: AaMode,
    pub gamma_x100: u16,
    /// Phase 51: persisted enabled blueprint mods (`mods=uid1,uid2`).
    pub mods: Vec<String>,
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
            colorblind: false,
            mouse_swap: false,
            subtitles: true,
            bloom: true,
            aa: AaMode::Msaa4,
            gamma_x100: GAMMA_DEFAULT_X100,
            mods: Vec::new(),
            keys: crate::keymap::KeyMap::default(),
        }
    }
}

/// Every key written to / understood in `settings.ini`, in file order (followed by one
/// `key.<action>` line per bindable action, see [`crate::keymap`]).
pub const KEYS: [&str; 22] = [
    "resolution",
    "fullscreen",
    "vsync",
    "shadow_quality",
    "bloom",
    "aa",
    "gamma",
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
    "colorblind",
    "mouse_swap",
    "subtitles",
    "mods",
];

fn parse_bool(v: &str) -> Result<bool, String> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!("'{other}' is not a boolean (true/false)")),
    }
}

pub fn parse_gamma_x100(v: &str) -> Result<u16, String> {
    let f: f32 = v
        .trim()
        .parse()
        .map_err(|_| format!("'{}' is not a gamma number 0.8..1.4", v.trim()))?;
    if !f.is_finite() || !(0.8..=1.4).contains(&f) {
        return Err(format!("{f} is out of gamma range 0.8..1.4"));
    }
    Ok((f * 100.0).round().clamp(GAMMA_MIN_X100 as f32, GAMMA_MAX_X100 as f32) as u16)
}

pub fn format_gamma_x100(g: u16) -> String {
    let clamped = g.clamp(GAMMA_MIN_X100, GAMMA_MAX_X100);
    format!("{:.2}", clamped as f32 / 100.0)
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

/// Parses a comma-separated mod spec list (`mods=uid1,uid2`), deduplicating case-insensitively.
pub fn parse_mod_list(v: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for raw in v.split(',') {
        let item = raw.trim();
        if item.is_empty() {
            continue;
        }
        if item.bytes().any(|b| b < 0x20 || b == 0x7f || b == b'=') {
            return Err(format!("invalid mod entry '{item}'"));
        }
        if !out.iter().any(|x| x.eq_ignore_ascii_case(item)) {
            out.push(item.to_string());
        }
    }
    Ok(out)
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
            "colorblind" => self.colorblind = parse_bool(v)?,
            "mouse_swap" => self.mouse_swap = parse_bool(v)?,
            "subtitles" => self.subtitles = parse_bool(v)?,
            "bloom" => self.bloom = parse_bool(v)?,
            "aa" => {
                self.aa =
                    AaMode::parse(v).ok_or_else(|| format!("'{v}' is not off/fxaa/msaa4"))?
            }
            "gamma" => self.gamma_x100 = parse_gamma_x100(v)?,
            "mods" => self.mods = parse_mod_list(v)?,
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
            "colorblind" => b(self.colorblind),
            "mouse_swap" => b(self.mouse_swap),
            "subtitles" => b(self.subtitles),
            "bloom" => if self.bloom { "on" } else { "off" }.to_string(),
            "aa" => self.aa.key().to_string(),
            "gamma" => format_gamma_x100(self.gamma_x100),
            "mods" => self.mods.join(","),
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

/// `$XDG_CONFIG_HOME/farust` or `~/.config/farust` (Windows `%APPDATA%\farust`, macOS
/// `~/Library/Application Support/farust`; see [`crate::platform::config_home`]).
pub fn config_dir() -> Option<PathBuf> {
    crate::platform::config_home().map(|d| d.join("farust"))
}

/// `$XDG_DATA_HOME/farust` or `~/.local/share/farust` (per-OS, see [`crate::platform::data_home`]).
#[allow(dead_code)]
pub fn data_dir() -> Option<PathBuf> {
    Some(crate::platform::data_home().join("farust"))
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
    ACCESS_FLAGS.store(0, std::sync::atomic::Ordering::Relaxed);
    RENDER_POST_CACHE.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Phase 39: cached accessibility flags (bit 0 = known, 1 = colour-blind, 2 = mouse swap,
/// 3 = subtitles). Read per unit per frame, so never clone the settings on the hot path.
static ACCESS_FLAGS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Phase 42: cached HDR post-processing settings (bit 0 = known, bit 1 = bloom,
/// bits 2..=3 = AaMode, bits 8..=15 = gamma_x100). Read per frame without locking/cloning.
static RENDER_POST_CACHE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn render_post_packed() -> u32 {
    use std::sync::atomic::Ordering::Relaxed;
    let p = RENDER_POST_CACHE.load(Relaxed);
    if p & 1 != 0 {
        return p;
    }
    let s = input_settings();
    let bloom = std::env::var("FARUST_BLOOM")
        .ok()
        .and_then(|v| parse_bool(&v).ok())
        .unwrap_or(s.bloom);
    let aa = std::env::var("FARUST_AA")
        .ok()
        .and_then(|v| AaMode::parse(&v))
        .unwrap_or(s.aa);
    let gamma_x100 = std::env::var("FARUST_GAMMA")
        .ok()
        .and_then(|v| parse_gamma_x100(&v).ok())
        .unwrap_or(s.gamma_x100)
        .clamp(GAMMA_MIN_X100, GAMMA_MAX_X100);
    let aa_bits: u32 = match aa {
        AaMode::Off => 0,
        AaMode::Fxaa => 1,
        AaMode::Msaa4 => 2,
    };
    let packed = 1 | (u32::from(bloom) << 1) | (aa_bits << 2) | (u32::from(gamma_x100) << 8);
    RENDER_POST_CACHE.store(packed, Relaxed);
    packed
}

/// Phase 42: lock-free cached `(bloom_enabled, aa_mode, gamma)` for the renderer hot path.
pub fn render_post() -> (bool, AaMode, f32) {
    let p = render_post_packed();
    let bloom = (p & 2) != 0;
    let aa = match (p >> 2) & 3 {
        0 => AaMode::Off,
        1 => AaMode::Fxaa,
        _ => AaMode::Msaa4,
    };
    let gamma = ((p >> 8) & 0xFF).clamp(u32::from(GAMMA_MIN_X100), u32::from(GAMMA_MAX_X100)) as f32 / 100.0;
    (bloom, aa, gamma)
}

#[allow(dead_code)]
pub fn bloom_enabled() -> bool {
    render_post().0
}

#[allow(dead_code)]
pub fn aa_mode() -> AaMode {
    render_post().1
}

#[allow(dead_code)]
pub fn gamma() -> f32 {
    render_post().2
}

fn access_flags() -> u8 {
    use std::sync::atomic::Ordering::Relaxed;
    let f = ACCESS_FLAGS.load(Relaxed);
    if f & 1 != 0 {
        return f;
    }
    let s = input_settings();
    let cb = s.colorblind || std::env::var("FARUST_COLORBLIND").is_ok_and(|v| v == "1");
    let f = 1 | u8::from(cb) << 1 | u8::from(s.mouse_swap) << 2 | u8::from(s.subtitles) << 3;
    ACCESS_FLAGS.store(f, Relaxed);
    f
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

/// Phase 39: colour-blind safe army colours (`FARUST_COLORBLIND=1` overrides).
pub fn colorblind() -> bool {
    access_flags() & 2 != 0
}

/// Phase 39: left / right mouse buttons swapped in matches.
pub fn mouse_swap() -> bool {
    access_flags() & 4 != 0
}

/// Phase 39: captions for announcer voice lines.
pub fn subtitles() -> bool {
    access_flags() & 8 != 0
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
            colorblind: true,
            mouse_swap: true,
            subtitles: false,
            bloom: false,
            aa: AaMode::Fxaa,
            gamma_x100: 125,
            mods: vec!["heavy-tanks-001".into(), "extra-units".into()],
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

    #[test]
    fn phase42_hdr_post_settings_round_trip_and_cache() {
        assert_eq!(AaMode::parse("OFF"), Some(AaMode::Off));
        assert_eq!(AaMode::parse("fxaa"), Some(AaMode::Fxaa));
        assert_eq!(AaMode::parse("MSAA4"), Some(AaMode::Msaa4));
        assert_eq!(AaMode::parse("taa"), None);
        assert_eq!(AaMode::Off.sample_count(), 1);
        assert_eq!(AaMode::Fxaa.sample_count(), 1);
        assert_eq!(AaMode::Msaa4.sample_count(), 4);

        assert_eq!(parse_gamma_x100("0.8"), Ok(80));
        assert_eq!(parse_gamma_x100("1.00"), Ok(100));
        assert_eq!(parse_gamma_x100("1.4"), Ok(140));
        assert!(parse_gamma_x100("0.79").is_err());
        assert!(parse_gamma_x100("1.41").is_err());
        assert!(parse_gamma_x100("nan").is_err());

        let (s, w) = Settings::parse("bloom=off\naa=fxaa\ngamma=1.15\n");
        assert!(w.is_empty(), "{w:?}");
        assert!(!s.bloom);
        assert_eq!(s.aa, AaMode::Fxaa);
        assert_eq!(s.gamma_x100, 115);
        assert_eq!(s.get("bloom").as_deref(), Some("off"));
        assert_eq!(s.get("aa").as_deref(), Some("fxaa"));
        assert_eq!(s.get("gamma").as_deref(), Some("1.15"));

        let (bad, bw) = Settings::parse("bloom=maybe\naa=ssaa\ngamma=2.5\n");
        assert_eq!(bw.len(), 3, "{bw:?}");
        assert!(bad.bloom);
        assert_eq!(bad.aa, AaMode::Msaa4);
        assert_eq!(bad.gamma_x100, GAMMA_DEFAULT_X100);

        let prev = active();
        set_active(Some(s));
        let (bl, aa, gam) = render_post();
        assert!(!bl);
        assert_eq!(aa, AaMode::Fxaa);
        assert!((gam - 1.15).abs() < 1e-5);
        set_active(prev);
    }
}
