//! Phase 20 front-end shell: Main Menu, Skirmish, Multiplayer (Host/Join/Lobby), Replays,
//! and Options on a single `winit` event loop that hands over cleanly to `play::PlayApp`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::WindowId;

use crate::lobby::{self, FACTIONS, LobbyAction, LobbyMap, LobbyState, faction_label};
use crate::net::{NetHandle, world_from_handshake};
use crate::net_lobby::{ClientPoll, HOST_ID, HostPoll, NetLobbyClient, NetLobbyHost, NetLobbyState, NetSlotKind};
use crate::play::{PlayApp, PlayExit, PlaySession, UnitMeshCache};
use crate::replay::Replay;
use crate::savegame;
use crate::settings::{
    self, FACTION_KEYS, RESOLUTIONS, Settings, ShadowQuality, UiTheme, validate_host_addr, validate_name,
};
use crate::skirmish::{Catalog, GameOptions, MapContext, VictoryCondition};
use crate::skirmish_ai::{Difficulty, ScriptedAi};
use crate::ui_kit::{
    Align, Canvas, ClassicPalette, Fonts, Frame, ImageStore, Rect, WState, WindowGpu, body, bold, render_png,
    theme, title,
};

const UNIT_CAPS: [u32; 6] = [250, 500, 1000, 1500, 2000, 0];

// ---------------------------------------------------------------------------
// Public entry types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScreenId {
    #[default]
    Main,
    Skirmish,
    Multiplayer,
    MpLobby,
    Replays,
    Options,
    /// Phase 23: Options -> Keys (key remapping).
    Keys,
}

impl FromStr for ScreenId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "main" | "menu" => Ok(ScreenId::Main),
            "skirmish" | "lobby" => Ok(ScreenId::Skirmish),
            "multiplayer" | "mp" => Ok(ScreenId::Multiplayer),
            "mp-host" | "mp-lobby" | "mplobby" | "host" => Ok(ScreenId::MpLobby),
            "replays" | "replay" => Ok(ScreenId::Replays),
            "options" | "settings" => Ok(ScreenId::Options),
            "keys" | "keybinds" | "keybindings" | "controls" => Ok(ScreenId::Keys),
            other => Err(format!(
                "unknown menu screen '{other}' (main, skirmish, multiplayer, mp-lobby, replays, options, keys)"
            )),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MenuOptions {
    pub screen: ScreenId,
    pub size: Option<(u32, u32)>,
    pub snapshot: Option<PathBuf>,
    pub initial_map: Option<String>,
    pub seed: Option<u64>,
    pub autostart: bool,
    /// Optional Steam-compatible `+connect_lobby <CODE|HOST:PORT>` auto-join target.
    pub connect_lobby: Option<String>,
}

pub fn run(game_dir: &Path, opts: MenuOptions) -> Result<(), String> {
    let mut state = MenuState::new(game_dir, &opts)?;

    if let Some(ref out_path) = opts.snapshot {
        let (w, h) = opts.size.unwrap_or(state.settings.resolution);
        let adapter = render_png(w, h, out_path, &state.fonts, &state.images, |c| state.draw(c))?;
        println!("menu snapshot ({:?}) {}x{} on {} -> {}", state.screen, w, h, adapter, out_path.display());
        return Ok(());
    }

    if opts.autostart && state.screen == ScreenId::Skirmish
        && let Some(ref mut lob) = state.lobby
        && let Ok(launch) = lob.launch()
    {
        let (mut session, map, meshes) = build_skirmish_from_menu(
            game_dir,
            &launch,
            state.ai_difficulty,
            state.game_options,
        )?;
        let (w, h) = opts.size.unwrap_or(state.settings.resolution);
        session.muted = state.settings.master_volume == 0;
        return crate::play::run_play_window(Some(game_dir), session, map, meshes, w, h);
    }

    let size = opts.size.unwrap_or(state.settings.resolution);
    let event_loop = EventLoop::new().map_err(|e| format!("failed to create event loop: {e}"))?;
    let mut app = ShellApp {
        state,
        active: ActiveMode::Menu(None),
        size,
        init_err: None,
        hover: None,
        modifiers: ModifiersState::empty(),
        lmb_down: false,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("menu event loop error: {e}"))?;
    if let Some(err) = app.init_err {
        return Err(err);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// State model (all 5 screens + network lobby)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ReplayItem {
    path: PathBuf,
    name: String,
    replay: Replay,
}

enum MpSession {
    None,
    Host(Box<NetLobbyHost>),
    Client(Box<NetLobbyClient>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FocusField {
    None,
    MpHostAddr,
    MpRendezvous,
    MpJoinAddr,
    MpChat,
    OptPlayerName,
    OptLastHost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    // Main menu
    Go(ScreenId),
    Quit,
    /// Phase 39: launch the built-in tutorial mission.
    StartTutorial,
    /// Phase 49: launch the built-in Forged Alliance campaign operation.
    StartCampaign,
    /// Phase 39 accessibility toggles.
    ToggleColorblind,
    ToggleMouseSwap,
    ToggleSubtitles,
    // Skirmish extras
    Skirmish(LobbyAction),
    CycleAiDiff,
    CycleVictory,
    CycleUnitCap,
    // Multiplayer Hub
    Focus(FocusField),
    ToggleMpUseCode,
    HostLobby,
    JoinLobby,
    BackToMain,
    // Active MP lobby
    LeaveMpLobby,
    MpCopyInviteCode,
    MpHostNextMap,
    MpHostPrevMap,
    MpHostSeedRandom,
    MpHostMoveSlot(usize),
    MpHostCycleKind(usize),
    MpHostCycleFaction(usize),
    MpHostCycleTeam(usize),
    MpHostKick(u8),
    MpClientToggleReady,
    MpClientCycleFaction,
    MpSendChat,
    MpStartMatch,
    // Replays
    SelectReplay(usize),
    VerifyReplay,
    WatchReplay,
    RefreshReplays,
    // Options
    CycleResolution,
    ToggleFullscreen,
    ToggleVsync,
    CycleShadow,
    CycleUiTheme,
    CycleLanguage,
    SliderMaster(u8),
    SliderMusic(u8),
    SliderSfx(u8),
    CycleDefaultFaction,
    ToggleEdgeScroll,
    SaveOptions,
    ResetOptions,
    // Phase 23 key bindings
    ToggleCardHotkeys,
    /// Start capturing a key for row `i` of [`crate::keymap::actions`].
    KeyRow(usize),
    KeysReset,
    KeysBack,
    KeysScroll(i32),
}

struct MenuState {
    game_dir: PathBuf,
    screen: ScreenId,
    settings: Settings,
    draft: Settings,
    fonts: Fonts,
    images: ImageStore,
    lobby: Option<LobbyState>,
    ai_difficulty: Difficulty,
    game_options: GameOptions,
    replays: Vec<ReplayItem>,
    replay_selected: usize,
    replay_scroll: usize,
    /// Phase 23 Keys screen: action being rebound (`add` = keep existing chords).
    keys_capture: Option<(crate::keymap::Action, bool)>,
    keys_scroll: usize,
    /// Shift held at the last click (Shift+click adds an alternative binding).
    shift_held: bool,
    mp_session: MpSession,
    mp_host_addr: String,
    mp_rendezvous: String,
    mp_use_code: bool,
    mp_join_addr: String,
    mp_chat_input: String,
    mp_map_idx: usize,
    snapshot_mode: bool,
    focus: FocusField,
    status: String,
    status_err: bool,
    map_count: usize,
}

impl MenuState {
    fn new(game_dir: &Path, opts: &MenuOptions) -> Result<Self, String> {
        #[cfg(not(test))]
        let settings = {
            let s = settings::settings_path()
                .map(|p| Settings::load_from(&p).0)
                .unwrap_or_default();
            settings::set_active(Some(s.clone()));
            s
        };
        #[cfg(test)]
        let settings = Settings::default();

        #[cfg(not(test))]
        crate::loc::init(game_dir);

        let fonts = Fonts::load(game_dir);
        let mut images = ImageStore::default();
        images.load_scd_textures(
            game_dir,
            &[
                ("bg", "textures/ui/common/scx_menu/campaign-select/bg.dds"),
                ("logo", "textures/ui/common/menus/main02/sc-logo-lg_bmp.dds"),
                ("bg_classic", "textures/ui/common/menus/background-colossus_bmp.dds"),
                ("bg_classic_sub", "textures/ui/common/menus/background-paint01_bmp.dds"),
            ],
        );
        if !images.contains("bg") {
            images.load_scd_textures(game_dir, &[("bg", "textures/ui/common/marketing/splash.dds")]);
        }

        let seed = opts.seed.unwrap_or(crate::sim::DEFAULT_SEED);
        let initial_map = opts.initial_map.as_deref().unwrap_or("SCMP_001");
        let lobby = lobby::scan_maps(game_dir).ok().map(|maps| {
            LobbyState::with_maps(
                game_dir,
                maps,
                initial_map,
                seed,
                &settings.player_name,
                settings.default_faction,
            )
        });
        let map_count = lobby.as_ref().map(|l| l.maps.len()).unwrap_or(0);
        let replays = scan_replays();
        let rendezvous = settings::effective_rendezvous(Some(&settings));

        let mut st = Self {
            game_dir: game_dir.to_path_buf(),
            screen: opts.screen,
            settings: settings.clone(),
            draft: settings.clone(),
            fonts,
            images,
            lobby,
            ai_difficulty: Difficulty::Normal,
            game_options: GameOptions::default(),
            replays,
            replay_selected: 0,
            replay_scroll: 0,
            keys_capture: None,
            keys_scroll: 0,
            shift_held: false,
            mp_session: MpSession::None,
            mp_host_addr: "0.0.0.0:15000".to_string(),
            mp_rendezvous: rendezvous,
            mp_use_code: true,
            mp_join_addr: settings.last_host.clone(),
            mp_chat_input: String::new(),
            mp_map_idx: 0,
            snapshot_mode: opts.snapshot.is_some(),
            focus: FocusField::None,
            status: "Ready".to_string(),
            status_err: false,
            map_count,
        };
        st.ensure_lobby_preview_image();
        if let Some(ref target) = opts.connect_lobby {
            st.mp_join_addr = target.clone();
            if let Err(e) = st.start_join_lobby() {
                st.screen = ScreenId::Multiplayer;
                st.set_status(e, true);
            }
        } else if st.screen == ScreenId::MpLobby {
            let bind_addr = if opts.snapshot.is_some() {
                "127.0.0.1:0"
            } else {
                &st.mp_host_addr.clone()
            };
            if let Err(e) = st.start_host_lobby(bind_addr) {
                st.status = e;
                st.status_err = true;
            }
        }
        Ok(st)
    }

    fn set_status(&mut self, msg: impl Into<String>, err: bool) {
        self.status = msg.into();
        self.status_err = err;
    }

    fn ensure_lobby_preview_image(&mut self) {
        if let Some(ref lob) = self.lobby {
            let id = lob.map().id.clone();
            let key = lobby::preview_image_name(&id);
            if !self.images.contains(&key)
                && let Some(pv) = lob.preview()
                && let Some(ref img) = pv.image
            {
                self.images.insert(&key, img.clone());
            }
        }
        let mp_id = match &self.mp_session {
            MpSession::Host(h) => Some(h.state.map_id.clone()),
            MpSession::Client(c) => c.state.as_ref().map(|s| s.map_id.clone()),
            MpSession::None => None,
        };
        if let Some(id) = mp_id {
            let key = lobby::preview_image_name(&id);
            if !self.images.contains(&key) {
                let pv = lobby::load_preview(&self.game_dir, &id);
                if let Some(img) = pv.image {
                    self.images.insert(&key, img);
                }
            }
        }
    }

    fn available_maps(&self) -> Vec<LobbyMap> {
        self.lobby
            .as_ref()
            .map(|l| l.maps.clone())
            .unwrap_or_else(|| vec![LobbyMap {
                id: "SCMP_001".into(),
                name: "Burial Mounds".into(),
                size: (256, 256),
                armies: 2,
            }])
    }

    fn start_host_lobby(&mut self, addr: &str) -> Result<(), String> {
        let maps = self.available_maps();
        let idx = self.mp_map_idx % maps.len();
        let m = &maps[idx];
        let seed = self.lobby.as_ref().map(|l| l.seed).unwrap_or(crate::sim::DEFAULT_SEED);
        let st = NetLobbyState::new(
            &m.id,
            &m.name,
            m.armies,
            seed,
            m.armies.clamp(2, 8) as u8,
            &self.settings.player_name,
            self.settings.default_faction as u8,
        );
        let mut host = NetLobbyHost::bind(addr, st)?;
        self.mp_host_addr = host.local_addr.to_string();
        let mut status_msg = format!("Hosting lobby on {}", self.mp_host_addr);
        if self.mp_use_code && (!self.snapshot_mode || std::env::var_os("FARUST_RENDEZVOUS").is_some()) {
            match host.register_invite(&self.mp_rendezvous, Duration::from_millis(250)) {
                Ok(code) => status_msg = format!("Hosting on {} - Invite Code: {code}", self.mp_host_addr),
                Err(_) => status_msg = format!("Hosting on {} (direct IP; rendezvous offline)", self.mp_host_addr),
            }
        }
        self.mp_session = MpSession::Host(Box::new(host));
        self.screen = ScreenId::MpLobby;
        self.focus = FocusField::MpChat;
        self.ensure_lobby_preview_image();
        self.set_status(status_msg, false);
        Ok(())
    }

    fn start_join_lobby(&mut self) -> Result<(), String> {
        let raw = self.mp_join_addr.trim().to_string();
        if crate::rendezvous::looks_like_code(&raw) {
            let code = crate::rendezvous::normalize_code(&raw)
                .ok_or_else(|| format!("invalid invite code '{raw}'"))?;
            let rv = if !self.mp_rendezvous.trim().is_empty() {
                self.mp_rendezvous.trim().to_string()
            } else {
                settings::effective_rendezvous(Some(&self.settings))
            };
            let client = NetLobbyClient::connect_code(&rv, &code, false, &self.settings.player_name);
            self.mp_session = MpSession::Client(Box::new(client));
            self.screen = ScreenId::MpLobby;
            self.focus = FocusField::MpChat;
            self.set_status(format!("Resolving invite code {code} via {rv}..."), false);
            return Ok(());
        }
        let addr = validate_host_addr(&raw)?;
        self.settings.last_host = addr.clone();
        self.draft.last_host = addr.clone();
        if let Some(path) = settings::settings_path() {
            let _ = self.settings.save_to(&path);
        }
        let client = NetLobbyClient::connect(&addr, &self.settings.player_name);
        self.mp_session = MpSession::Client(Box::new(client));
        self.screen = ScreenId::MpLobby;
        self.focus = FocusField::MpChat;
        self.set_status(format!("Connecting to {addr}..."), false);
        Ok(())
    }

    fn close_mp_session(&mut self) {
        self.mp_session = MpSession::None;
        self.mp_chat_input.clear();
        self.focus = FocusField::None;
    }

    fn navigate(&mut self, to: ScreenId) {
        self.focus = FocusField::None;
        let from = self.screen;
        self.screen = to;
        self.keys_capture = None;
        match to {
            // Options <-> Keys share one draft (unsaved key edits survive the round trip).
            ScreenId::Options if from == ScreenId::Keys => {
                self.set_status("Key bindings kept in the draft - click SAVE & APPLY", false);
            }
            ScreenId::Options => {
                self.draft = self.settings.clone();
                self.set_status("Edit settings and click SAVE & APPLY", false);
            }
            ScreenId::Keys => {
                if from != ScreenId::Options {
                    self.draft = self.settings.clone();
                }
                self.set_status("Click a binding and press a key (Shift+click adds an alternative)", false);
            }
            ScreenId::Replays => {
                self.replays = scan_replays();
                if self.replay_selected >= self.replays.len() {
                    self.replay_selected = 0;
                }
                self.set_status(format!("{} replay file(s) found", self.replays.len()), false);
            }
            ScreenId::Skirmish => {
                if let Some(ref mut lob) = self.lobby {
                    lob.player_name = self.settings.player_name.clone();
                }
                self.ensure_lobby_preview_image();
                self.set_status("Configure map, teams, AI & rules", false);
            }
            ScreenId::Multiplayer => {
                self.mp_join_addr = self.settings.last_host.clone();
                self.mp_rendezvous = settings::effective_rendezvous(Some(&self.settings));
                self.set_status("Host with invite code / IP or enter friend's code / IP:port", false);
            }
            _ => {}
        }
    }

    fn preview_lang(&self) -> String {
        if let Ok(v) = std::env::var("FARUST_LANG")
            && let Ok(code) = settings::validate_language(&v)
        {
            return code;
        }
        if matches!(self.screen, ScreenId::Options | ScreenId::Keys) {
            self.draft.language.clone()
        } else {
            self.settings.language.clone()
        }
    }

    fn preview_classic_palette(&self) -> Option<ClassicPalette> {
        let env_theme = std::env::var("FARUST_UI_THEME").ok().and_then(|v| UiTheme::parse(&v));
        let (theme_mode, faction_idx) = if matches!(self.screen, ScreenId::Options | ScreenId::Keys) {
            (env_theme.unwrap_or(self.draft.ui_theme), self.draft.default_faction)
        } else {
            (env_theme.unwrap_or(self.settings.ui_theme), self.settings.default_faction)
        };
        if theme_mode == UiTheme::Classic {
            Some(ClassicPalette::for_faction_key(FACTION_KEYS[faction_idx % FACTION_KEYS.len()]))
        } else {
            None
        }
    }

    fn draw(&self, c: &mut Canvas) {
        c.set_classic(self.preview_classic_palette());
        match self.screen {
            ScreenId::Main => self.draw_main(c),
            ScreenId::Skirmish => self.draw_skirmish(c),
            ScreenId::Multiplayer => self.draw_mp_hub(c),
            ScreenId::MpLobby => self.draw_mp_lobby(c),
            ScreenId::Replays => self.draw_replays(c),
            ScreenId::Options => self.draw_options(c),
            ScreenId::Keys => self.draw_keys(c),
        }
    }

    fn widgets(&self, w: f32, h: f32) -> Vec<(Rect, Action)> {
        match self.screen {
            ScreenId::Main => self.main_widgets(w, h),
            ScreenId::Skirmish => self.skirmish_widgets(w, h),
            ScreenId::Multiplayer => self.mp_hub_widgets(w, h),
            ScreenId::MpLobby => self.mp_lobby_widgets(w, h),
            ScreenId::Replays => self.replay_widgets(w, h),
            ScreenId::Options => self.options_widgets(w, h),
            ScreenId::Keys => self.keys_widgets(w, h),
        }
    }

    // ---- Main Menu -----------------------------------------------------------

    fn main_widgets(&self, w: f32, h: f32) -> Vec<(Rect, Action)> {
        let bx = (w * 0.08).max(48.0);
        let bw = 280.0;
        let bh = 42.0;
        let gap = 10.0;
        let by0 = (h * 0.30).max(186.0);
        let items = [
            ("CAMPAIGN", Action::StartCampaign),
            ("SKIRMISH", Action::Go(ScreenId::Skirmish)),
            ("TUTORIAL", Action::StartTutorial),
            ("MULTIPLAYER", Action::Go(ScreenId::Multiplayer)),
            ("REPLAYS", Action::Go(ScreenId::Replays)),
            ("OPTIONS", Action::Go(ScreenId::Options)),
            ("QUIT TO DESKTOP", Action::Quit),
        ];
        items
            .into_iter()
            .enumerate()
            .map(|(i, (_, act))| (Rect::new(bx, by0 + i as f32 * (bh + gap), bw, bh), act))
            .collect()
    }

    fn draw_main(&self, c: &mut Canvas) {
        let (w, h) = (c.w, c.h);
        c.backdrop(0.70);
        let bx = (w * 0.08).max(48.0);
        if c.has_image("logo") {
            c.image("logo", Rect::new(bx, 34.0, 420.0, 108.0), [1.0; 4]);
        } else {
            c.text(bx, 48.0, title(46.0), "FORGED ALLIANCE", theme::GOLD);
        }
        c.text(bx + 4.0, 148.0, bold(16.0), "FARUST DETERMINISTIC RTS ENGINE", theme::ACCENT);

        let lang = self.preview_lang();
        let tr = |k: &str, fb: &str| crate::loc::tr_lang(&lang, k, fb);
        let nav_box = Rect::new(bx - 16.0, (h * 0.30).max(186.0) - 18.0, 312.0, 390.0);
        c.panel(nav_box);
        let labels = [
            tr("ui.campaign", "CAMPAIGN"),
            tr("ui.skirmish", "SKIRMISH"),
            tr("ui.tutorial", "TUTORIAL"),
            tr("ui.multiplayer", "MULTIPLAYER"),
            tr("ui.replays", "REPLAYS"),
            tr("ui.options", "OPTIONS"),
            tr("ui.quit_desktop", "QUIT TO DESKTOP"),
        ];
        for (i, (r, _)) in self.main_widgets(w, h).into_iter().enumerate() {
            let st = WState {
                hover: c.is_hover(r),
                primary: i == 0,
                ..Default::default()
            };
            c.button(r, &labels[i], st);
        }

        // Right-hand profile & engine overview card
        let rx = (w - 430.0).max(bx + 340.0);
        if rx + 380.0 <= w - 24.0 {
            let rp = Rect::new(rx, nav_box.y, 380.0, nav_box.h);
            let inner = c.titled_panel(rp, &tr("ui.profile", "COMMANDER PROFILE"));
            let mut y = inner.y + 4.0;
            for (k, v) in [
                ("Commander", self.settings.player_name.clone()),
                ("Faction", faction_label(self.settings.default_faction)),
                ("Resolution", format!("{}x{}", self.settings.resolution.0, self.settings.resolution.1)),
                ("Installed Maps", self.map_count.to_string()),
                ("Local Replays", self.replays.len().to_string()),
                ("Audio Volume", format!("{}%", self.settings.master_volume)),
            ] {
                c.text(inner.x, y, bold(14.0), k, theme::DIM);
                c.text_in(Rect::new(inner.x + 150.0, y - 2.0, inner.w - 150.0, 22.0), body(15.0), &v, theme::TEXT, Align::Right);
                y += 30.0;
            }
            c.text(inner.x, inner.bottom() - 44.0, body(13.0), "In-match: F6 Quicksave | F9 Quickload", theme::DIM);
            c.text(inner.x, inner.bottom() - 22.0, body(13.0), "Enter Chat | Esc Pause Menu | Tab Score", theme::DIM);
        }
    }

    // ---- Skirmish (Lobby + AI Difficulty / Victory / Unit Cap) ---------------

    fn skirmish_widgets(&self, w: f32, h: f32) -> Vec<(Rect, Action)> {
        let Some(ref lob) = self.lobby else {
            let f = Frame::new(w, h);
            return vec![(f.footer_button(0, false, 160.0), Action::BackToMain)];
        };
        let mut out: Vec<(Rect, Action)> = lobby::widgets(lob, w, h)
            .into_iter()
            .map(|(r, a)| (r, Action::Skirmish(a)))
            .collect();
        // Extra cyclers placed neatly in the info block above the seed bar
        let c = Frame::new(w, h).content;
        let list_w = (c.w * 0.32).clamp(300.0, 430.0);
        let rx = c.x + list_w + 20.0;
        let rw = c.right() - rx;
        let slots_h = 70.0 + lob.slots.len() as f32 * 30.0;
        let size = (c.h - slots_h - 20.0 - 24.0).min(rw * 0.5).clamp(120.0, 360.0);
        let preview_right = rx + 12.0 + size;
        let info_x = preview_right + 20.0;
        let info_w = (c.right() - preview_right - 32.0).max(100.0);
        let info_y = c.y + 12.0;
        let seed_y = info_y + size - 34.0;
        let mut cy = (info_y + 142.0).min(seed_y - 98.0);
        let cw = (info_w - 110.0).max(90.0);
        if info_w >= 210.0 && cy + 88.0 <= seed_y {
            for act in [Action::CycleAiDiff, Action::CycleVictory, Action::CycleUnitCap] {
                out.push((Rect::new(info_x + 106.0, cy, cw, 26.0), act));
                cy += 30.0;
            }
        }
        out
    }

    fn draw_skirmish(&self, c: &mut Canvas) {
        let Some(ref lob) = self.lobby else {
            let f = Frame::new(c.w, c.h);
            c.chrome(&f, "SKIRMISH", "No maps found");
            c.text_in(f.content, bold(18.0), "Install Forged Alliance maps or set FA_PATH", theme::BAD, Align::Center);
            c.button(f.footer_button(0, false, 160.0), "BACK", WState { hover: c.is_hover(f.footer_button(0, false, 160.0)), ..Default::default() });
            return;
        };
        lobby::draw(c, lob);
        for (r, act) in self.skirmish_widgets(c.w, c.h) {
            let st = WState { hover: c.is_hover(r), ..Default::default() };
            match act {
                Action::CycleAiDiff => {
                    c.text_in(Rect::new(r.x - 106.0, r.y, 100.0, r.h), bold(13.0), "AI LEVEL", theme::DIM, Align::Left);
                    c.cycler(r, self.ai_difficulty.name(), st);
                }
                Action::CycleVictory => {
                    c.text_in(Rect::new(r.x - 106.0, r.y, 100.0, r.h), bold(13.0), "VICTORY", theme::DIM, Align::Left);
                    c.cycler(r, self.game_options.victory.label(), st);
                }
                Action::CycleUnitCap => {
                    c.text_in(Rect::new(r.x - 106.0, r.y, 100.0, r.h), bold(13.0), "UNIT CAP", theme::DIM, Align::Left);
                    let lbl = if self.game_options.unit_cap == 0 {
                        "Unlimited".to_string()
                    } else {
                        self.game_options.unit_cap.to_string()
                    };
                    c.cycler(r, &lbl, st);
                }
                _ => {}
            }
        }
    }

    // ---- Multiplayer Hub -----------------------------------------------------

    fn mp_hub_widgets(&self, w: f32, h: f32) -> Vec<(Rect, Action)> {
        let f = Frame::new(w, h);
        let c = f.content;
        let col_w = ((c.w - 28.0) * 0.5).min(540.0);
        let left_x = c.x + (c.w - (col_w * 2.0 + 28.0)) * 0.5;
        let right_x = left_x + col_w + 28.0;
        let py = c.y + 28.0;
        let ph = (c.h - 56.0).clamp(360.0, 420.0);
        let top = py + 42.0;
        vec![
            (Rect::new(left_x + 24.0, top + 56.0, col_w - 48.0, 32.0), Action::Focus(FocusField::MpHostAddr)),
            (Rect::new(left_x + 204.0, top + 98.0, (col_w - 228.0).max(96.0), 26.0), Action::ToggleMpUseCode),
            (Rect::new(left_x + 24.0, top + 154.0, col_w - 48.0, 32.0), Action::Focus(FocusField::MpRendezvous)),
            (Rect::new(left_x + 24.0, py + ph - 58.0, col_w - 48.0, 40.0), Action::HostLobby),
            (Rect::new(right_x + 24.0, top + 56.0, col_w - 48.0, 34.0), Action::Focus(FocusField::MpJoinAddr)),
            (Rect::new(right_x + 24.0, py + ph - 58.0, col_w - 48.0, 40.0), Action::JoinLobby),
            (f.footer_button(0, false, 160.0), Action::BackToMain),
        ]
    }

    fn draw_mp_hub(&self, c: &mut Canvas) {
        let f = Frame::new(c.w, c.h);
        c.chrome(&f, "MULTIPLAYER", "TCP Deterministic Lockstep (v4) - Direct + Invite Code Relay");
        let cnt = f.content;
        let col_w = ((cnt.w - 28.0) * 0.5).min(540.0);
        let left_x = cnt.x + (cnt.w - (col_w * 2.0 + 28.0)) * 0.5;
        let right_x = left_x + col_w + 28.0;
        let py = cnt.y + 28.0;
        let ph = (cnt.h - 56.0).clamp(360.0, 420.0);

        let left_r = Rect::new(left_x, py, col_w, ph);
        let lin = c.titled_panel(left_r, "HOST LOBBY");
        c.text(lin.x + 12.0, lin.y + 4.0, body(15.0), &format!("Host player: {}", self.settings.player_name), theme::TEXT);
        c.text(lin.x + 12.0, lin.y + 34.0, bold(13.0), "BIND ADDRESS (IP:PORT)", theme::DIM);
        c.text(lin.x + 12.0, lin.y + 102.0, bold(13.0), "INVITE CODE RELAY", theme::DIM);
        c.text(lin.x + 12.0, lin.y + 132.0, bold(13.0), "RENDEZVOUS SERVER (HOST:PORT)", theme::DIM);

        let right_r = Rect::new(right_x, py, col_w, ph);
        let rin = c.titled_panel(right_r, "JOIN GAME");
        c.text(rin.x + 12.0, rin.y + 4.0, body(15.0), "Join by invite code (XXXX-XXXX) or direct IP:port.", theme::TEXT);
        c.text(rin.x + 12.0, rin.y + 34.0, bold(13.0), "HOST ADDRESS OR INVITE CODE", theme::DIM);
        c.text(rin.x + 12.0, rin.y + 106.0, body(13.0), "Direct TCP tried first; relays through rendezvous if NAT blocks.", theme::DIM);
        c.text(rin.x + 12.0, rin.y + 128.0, body(13.0), "Self-host relay: `farust rendezvous 0.0.0.0:15050`", theme::ACCENT);

        for (r, act) in self.mp_hub_widgets(c.w, c.h) {
            let st = WState { hover: c.is_hover(r), ..Default::default() };
            match act {
                Action::Focus(FocusField::MpHostAddr) => {
                    c.text_field(r, &self.mp_host_addr, self.focus == FocusField::MpHostAddr, "0.0.0.0:15000", st);
                }
                Action::ToggleMpUseCode => c.checkbox(r, self.mp_use_code, st),
                Action::Focus(FocusField::MpRendezvous) => {
                    c.text_field(r, &self.mp_rendezvous, self.focus == FocusField::MpRendezvous, crate::rendezvous::DEFAULT_ADDR, st);
                }
                Action::HostLobby => c.button(r, "OPEN MULTIPLAYER LOBBY", WState { primary: true, ..st }),
                Action::Focus(FocusField::MpJoinAddr) => {
                    c.text_field(r, &self.mp_join_addr, self.focus == FocusField::MpJoinAddr, "XXXX-XXXX or 127.0.0.1:15000", st);
                }
                Action::JoinLobby => c.button(r, "CONNECT TO HOST", WState { primary: true, ..st }),
                Action::BackToMain => c.button(r, "BACK", st),
                _ => {}
            }
        }
        c.footer_status(&f, &self.status, if self.status_err { theme::BAD } else { theme::DIM });
    }

    // ---- Multiplayer Lobby Room ----------------------------------------------

    fn mp_lobby_widgets(&self, w: f32, h: f32) -> Vec<(Rect, Action)> {
        let f = Frame::new(w, h);
        let c = f.content;
        let mut out = vec![(f.footer_button(0, false, 160.0), Action::LeaveMpLobby)];
        let left_w = (c.w * 0.56).max(380.0);
        let right_x = c.x + left_w + 18.0;
        let right_w = (c.right() - right_x).max(240.0);

        match &self.mp_session {
            MpSession::Host(host) => {
                let st = &host.state;
                let mw = left_w - 24.0;
                if host.invite_code.is_some() {
                    out.push((Rect::new(c.x + left_w - 134.0, c.y + 8.0, 122.0, 24.0), Action::MpCopyInviteCode));
                }
                out.push((Rect::new(c.x + 12.0, c.y + 44.0, mw - 120.0, 28.0), Action::MpHostNextMap));
                out.push((Rect::new(c.x + 12.0 + mw - 112.0, c.y + 44.0, 112.0, 28.0), Action::MpHostSeedRandom));
                let _ = Action::MpHostPrevMap;
                for i in 0..st.slots.len() {
                    let y = c.y + 116.0 + i as f32 * 32.0;
                    let sx = c.x + 48.0;
                    let sw = mw - 36.0;
                    out.push((Rect::new(sx, y, sw * 0.36, 26.0), Action::MpHostCycleKind(i)));
                    out.push((Rect::new(sx + sw * 0.38, y, sw * 0.24, 26.0), Action::MpHostCycleFaction(i)));
                    out.push((Rect::new(sx + sw * 0.64, y, sw * 0.16, 26.0), Action::MpHostCycleTeam(i)));
                    if let NetSlotKind::Peer(pid) = st.slots[i].kind {
                        if pid == HOST_ID {
                            out.push((Rect::new(sx + sw * 0.82, y, sw * 0.18, 26.0), Action::MpHostMoveSlot(i)));
                        } else {
                            out.push((Rect::new(sx + sw * 0.82, y, sw * 0.18, 26.0), Action::MpHostKick(pid)));
                        }
                    }
                }
                let can_start = st.start_check().is_ok();
                if can_start || true {
                    out.push((f.footer_button(0, true, 220.0), Action::MpStartMatch));
                }
            }
            MpSession::Client(client) => {
                out.push((Rect::new(c.x + 12.0, c.bottom() - 48.0, 160.0, 32.0), Action::MpClientCycleFaction));
                let ready_rect = f.footer_button(0, true, 220.0);
                out.push((ready_rect, Action::MpClientToggleReady));
                let _ = client;
            }
            MpSession::None => {}
        }

        // Chat box input & send button on the lower-right panel
        let chat_y = c.y + 210.0;
        let chat_h = (c.bottom() - chat_y).max(160.0);
        let inp_y = chat_y + chat_h - 42.0;
        out.push((Rect::new(right_x + 12.0, inp_y, right_w - 104.0, 30.0), Action::Focus(FocusField::MpChat)));
        out.push((Rect::new(right_x + right_w - 84.0, inp_y, 72.0, 30.0), Action::MpSendChat));
        out
    }

    fn draw_mp_lobby(&self, c: &mut Canvas) {
        let f = Frame::new(c.w, c.h);
        let subtitle = match &self.mp_session {
            MpSession::Host(h) => match &h.invite_code {
                Some(code) => format!("CODE: {code}  |  Hosting on {}", h.local_addr),
                None => format!("Hosting on {}", h.local_addr),
            },
            MpSession::Client(cl) => match cl.via_relay {
                Some(true) => format!("Relayed ({})", cl.addr),
                Some(false) => format!("Direct ({})", cl.addr),
                None => format!("Connecting to {}", cl.addr),
            },
            MpSession::None => "Disconnected".to_string(),
        };
        c.chrome(&f, "MULTIPLAYER LOBBY", &subtitle);

        let cnt = f.content;
        let left_w = (cnt.w * 0.56).max(380.0);
        let right_x = cnt.x + left_w + 18.0;
        let right_w = (cnt.right() - right_x).max(240.0);
        c.panel(Rect::new(cnt.x, cnt.y, left_w, cnt.h));

        let (lobby_state, chat_log, is_host, my_id) = match &self.mp_session {
            MpSession::Host(h) => (Some(&h.state), &h.chat[..], true, Some(HOST_ID)),
            MpSession::Client(cl) => (cl.state.as_ref(), &cl.chat[..], false, cl.peer_id),
            MpSession::None => (None, &[][..], false, None),
        };

        if let Some(st) = lobby_state {
            let max_map_w = if is_host { left_w - 156.0 } else { left_w - 28.0 };
            let hdr_txt = c.fonts.fit(
                bold(16.0),
                &format!("MAP: {} ({})   SEED: {}", st.map_name, st.map_id, st.seed),
                max_map_w,
            );
            c.text(cnt.x + 14.0, cnt.y + 12.0, bold(16.0), &hdr_txt, theme::GOLD);
            c.text(cnt.x + 14.0, cnt.y + 88.0, bold(13.0), "SLOTS & PLAYERS", theme::DIM);

            let mw = left_w - 24.0;
            for (i, slot) in st.slots.iter().enumerate() {
                let y = cnt.y + 116.0 + i as f32 * 32.0;
                c.rect(Rect::new(cnt.x + 14.0, y + 4.0, 18.0, 18.0), theme::SLOT_COLORS[i % 8]);
                c.text_in(Rect::new(cnt.x + 14.0, y, 18.0, 26.0), bold(12.0), &format!("{}", i + 1), [0.0, 0.0, 0.0, 1.0], Align::Center);

                if !is_host {
                    let sx = cnt.x + 48.0;
                    let sw = mw - 36.0;
                    let who = match slot.kind {
                        NetSlotKind::Open => "Open".to_string(),
                        NetSlotKind::Closed => "Closed".to_string(),
                        NetSlotKind::Ai => "AI: Scripted".to_string(),
                        NetSlotKind::Peer(pid) => {
                            let p = st.peer(pid);
                            let tag = if p.is_some_and(|p| p.ready) { "[READY]" } else { "[...]" };
                            format!("{} {tag}", p.map(|p| p.name.as_str()).unwrap_or("Peer"))
                        }
                    };
                    c.text_in(Rect::new(sx, y, sw * 0.36, 26.0), body(14.0), &who, theme::TEXT, Align::Left);
                    c.text_in(Rect::new(sx + sw * 0.38, y, sw * 0.24, 26.0), body(14.0), &faction_label(slot.faction as usize), theme::TEXT, Align::Center);
                    c.text_in(Rect::new(sx + sw * 0.64, y, sw * 0.16, 26.0), body(14.0), &format!("T{}", slot.team + 1), theme::DIM, Align::Center);
                    let _ = my_id;
                }
            }

            // Right top: Map preview + peer list
            let prev_panel = Rect::new(right_x, cnt.y, right_w, 196.0);
            c.titled_panel(prev_panel, &st.map_id);
            let img_r = Rect::new(right_x + 12.0, cnt.y + 40.0, 144.0, 144.0);
            let key = lobby::preview_image_name(&st.map_id);
            if !c.image(&key, img_r, [1.0; 4]) {
                c.rect(img_r, [0.06, 0.08, 0.09, 1.0]);
            }
            let mut py = cnt.y + 42.0;
            for p in &st.peers {
                let tag = if p.ready { "READY" } else { "WAIT" };
                let col = if p.ready { theme::GOOD } else { theme::GOLD };
                let relay_tag = match &self.mp_session {
                    MpSession::Host(h) if p.id != HOST_ID => {
                        if h.peer_is_relayed(p.id) == Some(true) { " [R]" } else { " [D]" }
                    }
                    _ => "",
                };
                let shown = format!("{}{}", p.name, relay_tag);
                c.text(right_x + 168.0, py, body(14.0), &c.fonts.fit(body(14.0), &shown, right_w - 235.0), theme::TEXT);
                c.text_in(Rect::new(right_x + right_w - 64.0, py - 2.0, 52.0, 20.0), bold(12.0), tag, col, Align::Right);
                py += 22.0;
            }
        } else {
            c.text_in(Rect::new(cnt.x, cnt.y + 120.0, left_w, 40.0), bold(16.0), "Connecting to host lobby...", theme::ACCENT, Align::Center);
        }

        // Right bottom: Chat panel
        let chat_y = cnt.y + 210.0;
        let chat_h = (cnt.bottom() - chat_y).max(160.0);
        let cin = c.titled_panel(Rect::new(right_x, chat_y, right_w, chat_h), "LOBBY CHAT");
        let lines = chat_log.iter().rev().take(7).collect::<Vec<_>>();
        for (idx, line) in lines.into_iter().rev().enumerate() {
            let y = cin.y + idx as f32 * 20.0;
            let text = if line.from.is_empty() {
                format!("* {}", line.text)
            } else {
                format!("{}: {}", line.from, line.text)
            };
            let col = if line.from.is_empty() { theme::GOLD } else { theme::TEXT };
            let fit = c.fonts.fit(body(13.0), &text, cin.w);
            c.text(cin.x, y, body(13.0), &fit, col);
        }

        // Widgets
        for (r, act) in self.mp_lobby_widgets(c.w, c.h) {
            let st = WState { hover: c.is_hover(r), ..Default::default() };
            match act {
                Action::LeaveMpLobby => c.button(r, "LEAVE", st),
                Action::MpCopyInviteCode => c.button(r, "COPY CODE", WState { primary: true, ..st }),
                Action::MpHostNextMap => {
                    let lbl = lobby_state.map(|s| format!("Map: {}", s.map_name)).unwrap_or_default();
                    c.cycler(r, &lbl, st);
                }
                Action::MpHostSeedRandom => c.button(r, "NEW SEED", st),
                Action::MpHostCycleKind(i) => {
                    if let Some(ls) = lobby_state {
                        let lbl = match ls.slots[i].kind {
                            NetSlotKind::Open => "Open".to_string(),
                            NetSlotKind::Closed => "Closed".to_string(),
                            NetSlotKind::Ai => "AI: Scripted".to_string(),
                            NetSlotKind::Peer(pid) => {
                                let p = ls.peer(pid);
                                let ready = if p.is_some_and(|x| x.ready) { " [R]" } else { "" };
                                format!("{}{}", p.map(|x| x.name.as_str()).unwrap_or("Peer"), ready)
                            }
                        };
                        c.cycler(r, &lbl, st);
                    }
                }
                Action::MpHostCycleFaction(i) => {
                    if let Some(ls) = lobby_state {
                        c.cycler(r, &faction_label(ls.slots[i].faction as usize), st);
                    }
                }
                Action::MpHostCycleTeam(i) => {
                    if let Some(ls) = lobby_state {
                        c.cycler(r, &format!("T{}", ls.slots[i].team + 1), st);
                    }
                }
                Action::MpHostMoveSlot(_) => c.button(r, "MOVE", st),
                Action::MpHostKick(_) => c.button(r, "KICK", st),
                Action::MpStartMatch => {
                    let ok = lobby_state.is_some_and(|s| s.start_check().is_ok());
                    c.button(r, "START MATCH", WState { primary: ok, disabled: !ok, ..st });
                }
                Action::MpClientCycleFaction => {
                    let fidx = match &self.mp_session {
                        MpSession::Client(cl) => cl
                            .peer_id
                            .and_then(|id| cl.state.as_ref().and_then(|s| s.slot_of(id).map(|si| s.slots[si].faction)))
                            .unwrap_or(self.settings.default_faction as u8),
                        _ => 0,
                    };
                    c.cycler(r, &faction_label(fidx as usize), st);
                }
                Action::MpClientToggleReady => {
                    let rdy = match &self.mp_session {
                        MpSession::Client(cl) => cl.ready,
                        _ => false,
                    };
                    c.button(r, if rdy { "UNREADY" } else { "READY" }, WState { primary: !rdy, active: rdy, ..st });
                }
                Action::Focus(FocusField::MpChat) => {
                    c.text_field(r, &self.mp_chat_input, self.focus == FocusField::MpChat, "Type chat & Enter...", st);
                }
                Action::MpSendChat => c.button(r, "SEND", st),
                _ => {}
            }
        }

        let bar_msg = match lobby_state {
            Some(ls) if is_host => match ls.start_check() {
                Ok(()) => "All players ready - click START MATCH".to_string(),
                Err(why) => format!("Lobby: {why}"),
            },
            _ => self.status.clone(),
        };
        c.footer_status(&f, &bar_msg, if self.status_err { theme::BAD } else { theme::DIM });
    }

    // ---- Replay Browser ------------------------------------------------------

    fn replay_widgets(&self, w: f32, h: f32) -> Vec<(Rect, Action)> {
        let f = Frame::new(w, h);
        let c = f.content;
        let list_w = (c.w * 0.44).clamp(320.0, 560.0);
        let mut out = Vec::new();
        let row_y0 = c.y + 52.0;
        let max_rows = ((c.h - 68.0) / 28.0).floor().max(1.0) as usize;
        for r in 0..max_rows {
            let idx = self.replay_scroll + r;
            if idx >= self.replays.len() {
                break;
            }
            out.push((
                Rect::new(c.x + 12.0, row_y0 + r as f32 * 28.0, list_w - 24.0, 25.0),
                Action::SelectReplay(idx),
            ));
        }
        let has_sel = self.replays.get(self.replay_selected).is_some();
        out.push((f.footer_button(0, false, 160.0), Action::BackToMain));
        out.push((f.footer_button(1, false, 160.0), Action::RefreshReplays));
        if has_sel {
            out.push((f.footer_button(1, true, 190.0), Action::VerifyReplay));
            out.push((f.footer_button(0, true, 190.0), Action::WatchReplay));
        }
        out
    }

    fn draw_replays(&self, c: &mut Canvas) {
        let f = Frame::new(c.w, c.h);
        c.chrome(&f, "REPLAYS", &format!("{} saved replays", self.replays.len()));
        let cnt = f.content;
        let list_w = (cnt.w * 0.44).clamp(320.0, 560.0);
        let list_r = Rect::new(cnt.x, cnt.y, list_w, cnt.h);
        c.titled_panel(list_r, "RECORDED MATCHES");

        if self.replays.is_empty() {
            c.text(cnt.x + 16.0, cnt.y + 62.0, body(15.0), "No .farust-replay files found.", theme::DIM);
            c.text(cnt.x + 16.0, cnt.y + 88.0, body(14.0), "Matches auto-record to fa-unpack/replays/.", theme::DIM);
        }

        let mut actions_seen = false;
        for (r, act) in self.replay_widgets(c.w, c.h) {
            let st = WState { hover: c.is_hover(r), ..Default::default() };
            match act {
                Action::SelectReplay(idx) => {
                    actions_seen = true;
                    let it = &self.replays[idx];
                    let sel = idx == self.replay_selected;
                    c.list_row(r, sel, st.hover);
                    let secs = it.replay.final_tick / crate::sim::TICK_HZ;
                    let left = c.fonts.fit(body(14.0), &it.name, r.w - 130.0);
                    let right = format!("{}  {:02}:{:02}", it.replay.header.map_id, secs / 60, secs % 60);
                    c.text_in(Rect::new(r.x + 8.0, r.y, r.w - 130.0, r.h), body(14.0), &left, if sel { theme::TEXT } else { theme::DIM }, Align::Left);
                    c.text_in(Rect::new(r.right() - 122.0, r.y, 114.0, r.h), body(13.0), &right, theme::ACCENT, Align::Right);
                }
                Action::BackToMain => c.button(r, "BACK", st),
                Action::RefreshReplays => c.button(r, "REFRESH", st),
                Action::VerifyReplay => c.button(r, "VERIFY HASH", st),
                Action::WatchReplay => c.button(r, "WATCH REPLAY", WState { primary: true, ..st }),
                _ => {}
            }
        }
        let _ = actions_seen;

        let rx = cnt.x + list_w + 20.0;
        let rw = (cnt.right() - rx).max(200.0);
        let det_in = c.titled_panel(Rect::new(rx, cnt.y, rw, cnt.h), "REPLAY DETAILS");
        if let Some(it) = self.replays.get(self.replay_selected) {
            let rep = &it.replay;
            let secs = rep.final_tick / crate::sim::TICK_HZ;
            let cap = if rep.header.options.unit_cap == 0 {
                "Unlimited".to_string()
            } else {
                rep.header.options.unit_cap.to_string()
            };
            let mut y = det_in.y + 4.0;
            for (k, v) in [
                ("File", it.path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()),
                ("Map", rep.header.map_id.clone()),
                ("Duration", format!("{:02}:{:02} ({} ticks)", secs / 60, secs % 60, rep.final_tick)),
                ("Victory", rep.header.options.victory.label().to_string()),
                ("Unit Cap", cap),
                ("Seed", rep.header.seed.to_string()),
                ("Commands", rep.commands.len().to_string()),
                ("Checkpoints", rep.checkpoints.len().to_string()),
                ("Final Hash", format!("{:016x}", rep.final_hash)),
            ] {
                c.text(det_in.x, y, bold(14.0), k, theme::DIM);
                c.text_in(Rect::new(det_in.x + 120.0, y - 2.0, det_in.w - 120.0, 22.0), body(14.0), &c.fonts.fit(body(14.0), &v, det_in.w - 124.0), theme::TEXT, Align::Left);
                y += 26.0;
            }
            y += 10.0;
            c.text(det_in.x, y, bold(14.0), "ARMIES", theme::GOLD);
            y += 24.0;
            for (i, a) in rep.header.setups.iter().enumerate() {
                c.rect(Rect::new(det_in.x, y + 3.0, 12.0, 12.0), theme::SLOT_COLORS[i % 8]);
                c.text(det_in.x + 20.0, y, body(14.0), &format!("{}  ({})  Team {}", a.name, faction_label(a.faction), a.team + 1), theme::TEXT);
                y += 22.0;
            }
        } else {
            c.text(det_in.x, det_in.y + 12.0, body(15.0), "Select a replay on the left.", theme::DIM);
        }
        c.footer_status(&f, &self.status, if self.status_err { theme::BAD } else { theme::GOOD });
    }

    // ---- Options -------------------------------------------------------------

    fn options_widgets(&self, w: f32, h: f32) -> Vec<(Rect, Action)> {
        let f = Frame::new(w, h);
        let c = f.content;
        let col_w = ((c.w - 24.0) * 0.5).max(320.0);
        let lx = c.x;
        let rx = c.x + col_w + 24.0;
        let lw = 150.0;
        let ctrl_w = (col_w - lw - 28.0).max(160.0);
        let mut y = c.y + 54.0;
        let step = 42.0;
        let mut out = vec![
            (Rect::new(lx + 14.0 + lw, y, ctrl_w, 30.0), Action::CycleResolution),
            (Rect::new(lx + 14.0 + lw, y + step, ctrl_w, 30.0), Action::ToggleFullscreen),
            (Rect::new(lx + 14.0 + lw, y + step * 2.0, ctrl_w, 30.0), Action::ToggleVsync),
            (Rect::new(lx + 14.0 + lw, y + step * 3.0, ctrl_w, 30.0), Action::CycleShadow),
            (Rect::new(lx + 14.0 + lw, y + step * 4.0, ctrl_w, 30.0), Action::SliderMaster(self.draft.master_volume)),
            (Rect::new(lx + 14.0 + lw, y + step * 5.0, ctrl_w, 30.0), Action::SliderMusic(self.draft.music_volume)),
            (Rect::new(lx + 14.0 + lw, y + step * 6.0, ctrl_w, 30.0), Action::SliderSfx(self.draft.sfx_volume)),
            (Rect::new(lx + 14.0 + lw, y + step * 7.0, ctrl_w, 30.0), Action::CycleUiTheme),
            (Rect::new(lx + 14.0 + lw, y + step * 8.0, ctrl_w, 30.0), Action::CycleLanguage),
        ];
        let _ = &mut y;
        let rctrl_w = (col_w - lw - 28.0).max(160.0);
        out.extend([
            (Rect::new(rx + 14.0 + lw, y, rctrl_w, 30.0), Action::Focus(FocusField::OptPlayerName)),
            (Rect::new(rx + 14.0 + lw, y + step, rctrl_w, 30.0), Action::CycleDefaultFaction),
            (Rect::new(rx + 14.0 + lw, y + step * 2.0, rctrl_w, 30.0), Action::ToggleEdgeScroll),
            (Rect::new(rx + 14.0 + lw, y + step * 3.0, rctrl_w, 30.0), Action::Focus(FocusField::OptLastHost)),
            (Rect::new(rx + 14.0 + lw, y + step * 4.0, rctrl_w, 30.0), Action::ToggleCardHotkeys),
            (Rect::new(rx + 14.0 + lw, y + step * 5.0, rctrl_w, 30.0), Action::Go(ScreenId::Keys)),
            (Rect::new(rx + 14.0 + lw, y + step * 6.0, rctrl_w, 30.0), Action::ToggleColorblind),
            (Rect::new(rx + 14.0 + lw, y + step * 7.0, rctrl_w, 30.0), Action::ToggleMouseSwap),
            (Rect::new(rx + 14.0 + lw, y + step * 8.0, rctrl_w, 30.0), Action::ToggleSubtitles),
            (f.footer_button(0, false, 160.0), Action::BackToMain),
            (f.footer_button(1, false, 160.0), Action::ResetOptions),
            (f.footer_button(0, true, 220.0), Action::SaveOptions),
        ]);
        out
    }

    fn draw_options(&self, c: &mut Canvas) {
        let lang = self.preview_lang();
        let tr = |k: &str, fb: &str| crate::loc::tr_lang(&lang, k, fb);
        let f = Frame::new(c.w, c.h);
        let path_lbl = settings::settings_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "settings.ini".to_string());
        c.chrome(&f, &tr("ui.options", "OPTIONS"), &path_lbl);
        let cnt = f.content;
        let col_w = ((cnt.w - 24.0) * 0.5).max(320.0);
        let lx = cnt.x;
        let rx = cnt.x + col_w + 24.0;
        let lw = 150.0;
        c.titled_panel(Rect::new(lx, cnt.y, col_w, cnt.h), &tr("ui.graphics_audio", "GRAPHICS & AUDIO"));
        c.titled_panel(Rect::new(rx, cnt.y, col_w, cnt.h), &tr("ui.gameplay_controls", "GAMEPLAY & CONTROLS"));

        for (r, act) in self.options_widgets(c.w, c.h) {
            let st = WState { hover: c.is_hover(r), ..Default::default() };
            let mut label = |txt: &str| {
                c.text_in(Rect::new(r.x - lw, r.y, lw - 8.0, r.h), bold(14.0), txt, theme::DIM, Align::Left);
            };
            match act {
                Action::CycleResolution => {
                    label(&tr("ui.resolution", "Resolution"));
                    c.cycler(r, &format!("{}x{}", self.draft.resolution.0, self.draft.resolution.1), st);
                }
                Action::ToggleFullscreen => {
                    label(&tr("ui.fullscreen", "Fullscreen"));
                    c.checkbox(r, self.draft.fullscreen, st);
                }
                Action::ToggleVsync => {
                    label(&tr("ui.vsync", "VSync"));
                    c.checkbox(r, self.draft.vsync, st);
                }
                Action::CycleShadow => {
                    label(&tr("ui.shadows", "Shadows"));
                    c.cycler(r, self.draft.shadow_quality.key(), st);
                }
                Action::CycleUiTheme => {
                    label(&tr("ui.ui_theme", "UI Theme"));
                    c.cycler(r, self.draft.ui_theme.label(), st);
                }
                Action::CycleLanguage => {
                    label(&tr("ui.language", "Language"));
                    let shown = crate::loc::language_display_name(&lang);
                    c.cycler(r, &shown, st);
                }
                Action::SliderMaster(_) => {
                    label(&tr("ui.master_volume", "Master Volume"));
                    c.slider(r, self.draft.master_volume as f32 / 100.0, &format!("{}%", self.draft.master_volume), st);
                }
                Action::SliderMusic(_) => {
                    label(&tr("ui.music_volume", "Music Volume"));
                    c.slider(r, self.draft.music_volume as f32 / 100.0, &format!("{}%", self.draft.music_volume), st);
                }
                Action::SliderSfx(_) => {
                    label(&tr("ui.sfx_volume", "SFX Volume"));
                    c.slider(r, self.draft.sfx_volume as f32 / 100.0, &format!("{}%", self.draft.sfx_volume), st);
                }
                Action::Focus(FocusField::OptPlayerName) => {
                    label(&tr("ui.player_name", "Player Name"));
                    c.text_field(r, &self.draft.player_name, self.focus == FocusField::OptPlayerName, "Commander", st);
                }
                Action::CycleDefaultFaction => {
                    label(&tr("ui.default_faction", "Default Faction"));
                    c.cycler(r, &faction_label(self.draft.default_faction), st);
                }
                Action::ToggleEdgeScroll => {
                    label(&tr("ui.edge_scroll", "Edge Scroll"));
                    c.checkbox(r, self.draft.edge_scroll, st);
                }
                Action::Focus(FocusField::OptLastHost) => {
                    label(&tr("ui.default_host", "Default Host"));
                    c.text_field(r, &self.draft.last_host, self.focus == FocusField::OptLastHost, "127.0.0.1:15000", st);
                }
                Action::ToggleCardHotkeys => {
                    label(&tr("ui.card_hotkeys", "Card Grid Keys"));
                    c.checkbox(r, self.draft.card_hotkeys, st);
                }
                Action::ToggleColorblind => {
                    label(&tr("ui.colorblind", "Colour-blind Mode"));
                    c.checkbox(r, self.draft.colorblind, st);
                }
                Action::ToggleMouseSwap => {
                    label(&tr("ui.mouse_swap", "Swap Mouse Buttons"));
                    c.checkbox(r, self.draft.mouse_swap, st);
                }
                Action::ToggleSubtitles => {
                    label(&tr("ui.subtitles", "Subtitles"));
                    c.checkbox(r, self.draft.subtitles, st);
                }
                Action::Go(ScreenId::Keys) => {
                    label(&tr("ui.key_bindings", "Key Bindings"));
                    let n = self.draft.keys.conflicts().len();
                    let txt = if n > 0 {
                        format!("EDIT KEYS ({n} CONFLICTS)")
                    } else if self.draft.keys.is_default() {
                        "EDIT KEYS (DEFAULT)".to_string()
                    } else {
                        "EDIT KEYS (CUSTOM)".to_string()
                    };
                    c.button(r, &txt, st);
                }
                Action::BackToMain => c.button(r, &tr("ui.back", "BACK"), st),
                Action::ResetOptions => c.button(r, &tr("ui.defaults", "DEFAULTS"), st),
                Action::SaveOptions => c.button(r, &tr("ui.save_apply", "SAVE & APPLY"), WState { primary: true, ..st }),
                _ => {}
            }
        }

        // Mouse legend in lower half of right panel (keys: see EDIT KEYS)
        let ky0 = cnt.y + 54.0 + 42.0 * 9.3;
        c.text(rx + 14.0, ky0, bold(14.0), "MOUSE", theme::GOLD);
        let binds = [
            "LMB / Box : Select units   Dbl-click : Select same type",
            "RMB / Drag : Order / Formation   Ctrl+RMB drag : Area",
            "Card: LMB queue/place, Shift x5, RMB remove one",
            "Alt+LMB : Alert ping   Alt+Shift+LMB : Attack ping",
            "Alt+RMB : Move ping   Edge of screen : Scroll",
        ];
        for (i, line) in binds.iter().enumerate() {
            let y = ky0 + 24.0 + i as f32 * 22.0;
            if y + 20.0 < cnt.bottom() - 8.0 {
                c.text(rx + 14.0, y, body(13.0), line, theme::TEXT);
            }
        }

        c.footer_status(&f, &self.status, if self.status_err { theme::BAD } else { theme::GOOD });
    }

    // ---- Interaction handling ------------------------------------------------

    fn click(&mut self, w: f32, h: f32, px: f32, py: f32) -> MenuEvent {
        let ws = self.widgets(w, h);
        let Some((rect, act)) = ws.iter().rev().find(|(r, _)| r.contains(px, py)).cloned() else {
            self.focus = FocusField::None;
            return MenuEvent::None;
        };
        if !matches!(act, Action::Focus(_)) {
            self.focus = FocusField::None;
        }
        match act {
            Action::Go(s) => self.navigate(s),
            Action::StartCampaign => return MenuEvent::StartCampaign,
            Action::StartTutorial => return MenuEvent::StartTutorial,
            Action::ToggleColorblind => self.draft.colorblind = !self.draft.colorblind,
            Action::ToggleMouseSwap => self.draft.mouse_swap = !self.draft.mouse_swap,
            Action::ToggleSubtitles => self.draft.subtitles = !self.draft.subtitles,
            Action::Quit => return MenuEvent::Quit,
            Action::BackToMain => self.navigate(ScreenId::Main),
            Action::Skirmish(la) => {
                if matches!(la, LobbyAction::Back) {
                    self.navigate(ScreenId::Main);
                    return MenuEvent::None;
                }
                if let Some(ref mut lob) = self.lobby
                    && let Some(launch) = lob.apply(la)
                {
                    return MenuEvent::StartSkirmish(launch);
                }
                self.ensure_lobby_preview_image();
            }
            Action::CycleAiDiff => {
                let i = Difficulty::ALL.iter().position(|d| *d == self.ai_difficulty).unwrap_or(0);
                self.ai_difficulty = Difficulty::ALL[(i + 1) % Difficulty::ALL.len()];
                self.set_status(
                    format!("AI {}: {}", self.ai_difficulty.name(), self.ai_difficulty.describe()),
                    false,
                );
            }
            Action::CycleVictory => {
                let i = VictoryCondition::ALL.iter().position(|v| *v == self.game_options.victory).unwrap_or(0);
                self.game_options.victory = VictoryCondition::ALL[(i + 1) % VictoryCondition::ALL.len()];
            }
            Action::CycleUnitCap => {
                let i = UNIT_CAPS.iter().position(|c| *c == self.game_options.unit_cap).unwrap_or(0);
                self.game_options.unit_cap = UNIT_CAPS[(i + 1) % UNIT_CAPS.len()];
            }
            Action::Focus(ff) => self.focus = ff,
            Action::ToggleMpUseCode => self.mp_use_code = !self.mp_use_code,
            Action::HostLobby => {
                let addr = self.mp_host_addr.clone();
                if let Err(e) = self.start_host_lobby(&addr) {
                    self.set_status(e, true);
                }
            }
            Action::JoinLobby => {
                if let Err(e) = self.start_join_lobby() {
                    self.set_status(e, true);
                }
            }
            Action::LeaveMpLobby => {
                self.close_mp_session();
                self.navigate(ScreenId::Multiplayer);
            }
            Action::MpCopyInviteCode => {
                if let MpSession::Host(ref h) = self.mp_session
                    && let Some(ref code) = h.invite_code
                {
                    let copied = copy_to_clipboard(code);
                    let suffix = if copied { " (copied to clipboard)" } else { "" };
                    self.set_status(format!("Share invite code: {code}{suffix}"), false);
                }
            }
            Action::MpHostNextMap | Action::MpHostPrevMap => {
                let maps = self.available_maps();
                let delta = if matches!(act, Action::MpHostNextMap) { 1 } else { maps.len() - 1 };
                self.mp_map_idx = (self.mp_map_idx + delta) % maps.len();
                let m = &maps[self.mp_map_idx];
                if let MpSession::Host(ref mut host) = self.mp_session {
                    host.edit(|s| s.set_map(&m.id, &m.name, m.armies));
                }
                self.ensure_lobby_preview_image();
            }
            Action::MpHostSeedRandom => {
                if let MpSession::Host(ref mut host) = self.mp_session {
                    let t = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(1);
                    let seed = (t ^ (t >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15) % 1_000_000;
                    host.edit(|s| {
                        s.seed = seed;
                        Vec::new()
                    });
                }
            }
            Action::MpHostCycleKind(i) => {
                if let MpSession::Host(ref mut host) = self.mp_session {
                    host.edit(|s| {
                        s.cycle_kind(i);
                        Vec::new()
                    });
                }
            }
            Action::MpHostCycleFaction(i) => {
                if let MpSession::Host(ref mut host) = self.mp_session {
                    host.edit(|s| {
                        s.cycle_faction(i);
                        Vec::new()
                    });
                }
            }
            Action::MpHostCycleTeam(i) => {
                if let MpSession::Host(ref mut host) = self.mp_session {
                    host.edit(|s| {
                        s.cycle_team(i);
                        Vec::new()
                    });
                }
            }
            Action::MpHostMoveSlot(i) => {
                if let MpSession::Host(ref mut host) = self.mp_session {
                    host.edit(|s| {
                        s.move_slot(i);
                        Vec::new()
                    });
                }
            }
            Action::MpHostKick(pid) => {
                if let MpSession::Host(ref mut host) = self.mp_session {
                    host.kick(pid);
                }
            }
            Action::MpClientToggleReady => {
                if let MpSession::Client(ref mut cl) = self.mp_session {
                    let next = !cl.ready;
                    cl.set_ready(next);
                }
            }
            Action::MpClientCycleFaction => {
                if let MpSession::Client(ref mut cl) = self.mp_session {
                    let next = cl
                        .peer_id
                        .and_then(|id| cl.state.as_ref().and_then(|s| s.slot_of(id).map(|si| s.slots[si].faction)))
                        .map(|f| (f + 1) % FACTION_KEYS.len() as u8)
                        .unwrap_or(0);
                    cl.set_faction(next);
                }
            }
            Action::MpSendChat => self.submit_mp_chat(),
            Action::MpStartMatch => {
                if let MpSession::Host(ref mut host) = self.mp_session {
                    if let Err(e) = host.begin_start() {
                        self.set_status(format!("Cannot start: {e}"), true);
                    } else {
                        self.set_status("Starting networked lockstep match...", false);
                    }
                }
            }
            Action::SelectReplay(idx) => {
                if idx < self.replays.len() {
                    self.replay_selected = idx;
                    self.set_status("Ready to verify or watch replay", false);
                }
            }
            Action::RefreshReplays => {
                self.replays = scan_replays();
                if self.replay_selected >= self.replays.len() {
                    self.replay_selected = 0;
                }
                self.set_status(format!("Refreshed ({} replays)", self.replays.len()), false);
            }
            Action::VerifyReplay => {
                if let Some(it) = self.replays.get(self.replay_selected) {
                    let t0 = Instant::now();
                    let res = Catalog::load(&self.game_dir).and_then(|cat| {
                        let map = MapContext::load(&self.game_dir, &it.replay.header.map_id)?;
                        it.replay.verify(cat, &map)
                    });
                    match res {
                        Ok(r) => self.set_status(
                            format!("VERIFIED OK ({:.0}ms): {} ticks, hash {:016x}", t0.elapsed().as_secs_f64() * 1000.0, r.ticks, r.final_hash),
                            false,
                        ),
                        Err(e) => self.set_status(format!("Verify failed: {e}"), true),
                    }
                }
            }
            Action::WatchReplay => {
                if let Some(it) = self.replays.get(self.replay_selected).cloned() {
                    return MenuEvent::StartReplay(it.replay);
                }
            }
            Action::CycleResolution => {
                let i = RESOLUTIONS.iter().position(|r| *r == self.draft.resolution).unwrap_or(0);
                self.draft.resolution = RESOLUTIONS[(i + 1) % RESOLUTIONS.len()];
            }
            Action::ToggleFullscreen => self.draft.fullscreen = !self.draft.fullscreen,
            Action::ToggleVsync => self.draft.vsync = !self.draft.vsync,
            Action::CycleShadow => {
                let i = ShadowQuality::ALL.iter().position(|q| *q == self.draft.shadow_quality).unwrap_or(0);
                self.draft.shadow_quality = ShadowQuality::ALL[(i + 1) % ShadowQuality::ALL.len()];
            }
            Action::CycleUiTheme => {
                let i = UiTheme::ALL.iter().position(|t| *t == self.draft.ui_theme).unwrap_or(0);
                self.draft.ui_theme = UiTheme::ALL[(i + 1) % UiTheme::ALL.len()];
                self.set_status(
                    format!("Theme: {} (click SAVE & APPLY to keep)", self.draft.ui_theme.label()),
                    false,
                );
            }
            Action::CycleLanguage => {
                let langs = crate::loc::available_languages();
                let i = langs.iter().position(|(c, _)| c == &self.draft.language).unwrap_or(0);
                let (next_code, next_name) = &langs[(i + 1) % langs.len()];
                self.draft.language = next_code.clone();
                self.set_status(
                    format!("Language: {next_name} (click SAVE & APPLY to keep)"),
                    false,
                );
            }
            Action::SliderMaster(_) | Action::SliderMusic(_) | Action::SliderSfx(_) => {
                let track_w = (rect.w - 56.0).max(20.0);
                let val = (((px - rect.x) / track_w).clamp(0.0, 1.0) * 100.0).round() as u8;
                match act {
                    Action::SliderMaster(_) => self.draft.master_volume = val,
                    Action::SliderMusic(_) => self.draft.music_volume = val,
                    _ => self.draft.sfx_volume = val,
                }
            }
            Action::CycleDefaultFaction => {
                self.draft.default_faction = (self.draft.default_faction + 1) % FACTIONS.len();
            }
            Action::ToggleEdgeScroll => self.draft.edge_scroll = !self.draft.edge_scroll,
            Action::ToggleCardHotkeys => self.draft.card_hotkeys = !self.draft.card_hotkeys,
            Action::KeyRow(i) => {
                if let Some(info) = crate::keymap::actions().get(i) {
                    self.keys_capture = Some((info.action, self.shift_held));
                    let how = if self.shift_held { "add an alternative for" } else { "rebind" };
                    self.set_status(
                        format!("Press a key to {how} '{}' (Esc cancel, Del unbind, Backspace default)", info.label),
                        false,
                    );
                }
            }
            Action::KeysReset => {
                self.draft.keys = crate::keymap::KeyMap::default();
                self.keys_capture = None;
                self.set_status("Default key bindings restored (click SAVE & APPLY)", false);
            }
            Action::KeysBack => self.navigate(ScreenId::Options),
            Action::KeysScroll(d) => self.scroll_keys(d, h),
            Action::ResetOptions => {
                self.draft = Settings::default();
                self.set_status("Defaults restored (click SAVE & APPLY)", false);
            }
            Action::SaveOptions => {
                if let Err(e) = self.save_options() {
                    self.set_status(e, true);
                } else {
                    self.set_status("Settings saved and applied", false);
                    return MenuEvent::SettingsChanged;
                }
            }
        }
        MenuEvent::None
    }

    fn drag_slider(&mut self, w: f32, h: f32, px: f32, py: f32) -> bool {
        if self.screen != ScreenId::Options {
            return false;
        }
        for (r, act) in self.options_widgets(w, h) {
            if matches!(act, Action::SliderMaster(_) | Action::SliderMusic(_) | Action::SliderSfx(_))
                && r.contains(px, py)
            {
                let track_w = (r.w - 56.0).max(20.0);
                let val = (((px - r.x) / track_w).clamp(0.0, 1.0) * 100.0).round() as u8;
                match act {
                    Action::SliderMaster(_) => self.draft.master_volume = val,
                    Action::SliderMusic(_) => self.draft.music_volume = val,
                    Action::SliderSfx(_) => self.draft.sfx_volume = val,
                    _ => {}
                }
                return true;
            }
        }
        false
    }

    // ---- Phase 23 Keys screen --------------------------------------------------

    /// Rows per column of the key list.
    fn keys_rows_per_col(h: f32) -> usize {
        let f = Frame::new(1280.0, h);
        (((f.content.h - 58.0) / 28.0).floor().max(1.0)) as usize
    }

    fn scroll_keys(&mut self, d: i32, h: f32) {
        let total = crate::keymap::actions().len();
        let per_page = 2 * Self::keys_rows_per_col(h).max(1);
        let max = total.saturating_sub(per_page.min(total));
        self.keys_scroll = (self.keys_scroll as i64 + d as i64).clamp(0, max as i64) as usize;
    }

    fn keys_widgets(&self, w: f32, h: f32) -> Vec<(Rect, Action)> {
        let f = Frame::new(w, h);
        let c = f.content;
        let col_w = ((c.w - 24.0) * 0.5).max(300.0);
        let per_col = Self::keys_rows_per_col(h);
        let total = crate::keymap::actions().len();
        let max_scroll = total.saturating_sub((2 * per_col).min(total));
        let start = self.keys_scroll.min(max_scroll);
        let mut out = Vec::new();
        for k in 0..2 * per_col {
            let i = start + k;
            if i >= total {
                break;
            }
            let (col, row) = (k / per_col, k % per_col);
            let x = c.x + col as f32 * (col_w + 24.0);
            let y = c.y + 44.0 + row as f32 * 28.0;
            let bw = (col_w * 0.42).max(150.0);
            out.push((Rect::new(x + col_w - bw - 12.0, y, bw, 24.0), Action::KeyRow(i)));
        }
        out.extend([
            (f.footer_button(0, false, 160.0), Action::KeysBack),
            (f.footer_button(1, false, 160.0), Action::KeysReset),
            (f.footer_button(0, true, 220.0), Action::SaveOptions),
        ]);
        if start > 0 {
            out.push((Rect::new(c.right() - 120.0, c.y + 4.0, 52.0, 24.0), Action::KeysScroll(-(per_col as i32))));
        }
        if start < max_scroll {
            out.push((Rect::new(c.right() - 62.0, c.y + 4.0, 52.0, 24.0), Action::KeysScroll(per_col as i32)));
        }
        out
    }

    fn draw_keys(&self, c: &mut Canvas) {
        let f = Frame::new(c.w, c.h);
        c.chrome(&f, "KEY BINDINGS", "Click a binding, press a key - Shift+click adds - Del unbinds - Backspace default");
        let cnt = f.content;
        let col_w = ((cnt.w - 24.0) * 0.5).max(300.0);
        let total = crate::keymap::actions().len();
        let conflicts = self.draft.keys.conflicts();
        c.titled_panel(Rect::new(cnt.x, cnt.y, cnt.w, cnt.h), &format!("{total} ACTIONS - {} CONFLICTS", conflicts.len()));
        for (r, act) in self.keys_widgets(c.w, c.h) {
            let st = WState { hover: c.is_hover(r), ..Default::default() };
            match act {
                Action::KeyRow(i) => {
                    let info = &crate::keymap::actions()[i];
                    let capturing = self.keys_capture.is_some_and(|(a, _)| a == info.action);
                    let clash = self.draft.keys.conflicts_with(info.action);
                    let row = Rect::new(r.x - (col_w - r.w - 12.0) + 2.0, r.y, col_w - 4.0, r.h);
                    c.list_row(row, capturing, st.hover);
                    let lab_col = if clash.is_empty() { theme::TEXT } else { theme::BAD };
                    let lab_w = row.w - r.w - 20.0;
                    let label = c.fonts.fit(body(14.0), &info.label, lab_w);
                    c.text_in(Rect::new(row.x + 10.0, row.y, lab_w, row.h), body(14.0), &label, lab_col, Align::Left);
                    let chords = self.draft.keys.get(info.action);
                    let txt = if capturing {
                        "PRESS A KEY...".to_string()
                    } else if chords.is_empty() {
                        "-".to_string()
                    } else {
                        crate::keymap::chords_to_string(chords).replace(',', ", ")
                    };
                    c.button(r, &txt, WState { active: capturing, ..st });
                    if !clash.is_empty() {
                        c.outline(r, 2.0, theme::BAD);
                    }
                }
                Action::KeysBack => c.button(r, "BACK", st),
                Action::KeysReset => c.button(r, "DEFAULTS", st),
                Action::SaveOptions => c.button(r, "SAVE & APPLY", WState { primary: true, ..st }),
                Action::KeysScroll(d) => c.button(r, if d < 0 { "UP" } else { "DOWN" }, st),
                _ => {}
            }
        }
        c.footer_status(&f, &self.status, if self.status_err { theme::BAD } else { theme::GOOD });
    }

    /// Key press while capturing on the Keys screen; returns `true` if consumed.
    fn capture_key(&mut self, code: KeyCode, mods: ModifiersState) -> bool {
        let Some((action, add)) = self.keys_capture else {
            return false;
        };
        let label = crate::keymap::info(action).map(|i| i.label.clone()).unwrap_or_default();
        match code {
            KeyCode::ShiftLeft
            | KeyCode::ShiftRight
            | KeyCode::ControlLeft
            | KeyCode::ControlRight
            | KeyCode::AltLeft
            | KeyCode::AltRight
            | KeyCode::SuperLeft
            | KeyCode::SuperRight => return true,
            KeyCode::Escape => {
                self.keys_capture = None;
                self.set_status("Cancelled", false);
                return true;
            }
            KeyCode::Delete => {
                self.keys_capture = None;
                self.draft.keys.set(action, Vec::new());
                self.set_status(format!("'{label}' unbound"), false);
                return true;
            }
            KeyCode::Backspace => {
                self.keys_capture = None;
                self.draft.keys.reset(action);
                self.set_status(format!("'{label}' reset to default"), false);
                return true;
            }
            _ => {}
        }
        if crate::keymap::key_name(code).is_none() {
            self.set_status("That key cannot be bound", true);
            return true;
        }
        let chord = crate::keymap::KeyChord {
            key: code,
            ctrl: mods.control_key(),
            shift: mods.shift_key(),
            alt: mods.alt_key(),
        };
        self.keys_capture = None;
        let mut v = if add { self.draft.keys.get(action).to_vec() } else { Vec::new() };
        if !v.contains(&chord) {
            v.push(chord);
        }
        self.draft.keys.set(action, v);
        let clash: Vec<String> = self
            .draft
            .keys
            .conflicts_with(action)
            .into_iter()
            .filter_map(|a| crate::keymap::info(a).map(|i| i.label.clone()))
            .collect();
        let ks = crate::keymap::chord_to_string(&chord);
        if clash.is_empty() {
            self.set_status(format!("'{label}' = {ks} (SAVE & APPLY to keep)"), false);
        } else {
            self.set_status(format!("CONFLICT: {ks} is also bound to {}", clash.join(", ")), true);
        }
        true
    }

    fn save_options(&mut self) -> Result<(), String> {
        let name = validate_name(&self.draft.player_name)?;
        let last_host = validate_host_addr(&self.draft.last_host)?;
        self.draft.player_name = name;
        self.draft.last_host = last_host;
        if let Some(p) = settings::settings_path() {
            self.draft.save_to(&p)?;
        }
        self.settings = self.draft.clone();
        settings::set_active(Some(self.settings.clone()));
        crate::loc::set_language(&self.settings.language);
        Ok(())
    }

    fn submit_mp_chat(&mut self) {
        let txt = std::mem::take(&mut self.mp_chat_input);
        match &mut self.mp_session {
            MpSession::Host(h) => h.send_chat(&txt),
            MpSession::Client(cl) => cl.send_chat(&txt),
            MpSession::None => {}
        }
    }

    fn active_field_mut(&mut self) -> Option<&mut String> {
        match self.focus {
            FocusField::MpHostAddr => Some(&mut self.mp_host_addr),
            FocusField::MpRendezvous => Some(&mut self.mp_rendezvous),
            FocusField::MpJoinAddr => Some(&mut self.mp_join_addr),
            FocusField::MpChat => Some(&mut self.mp_chat_input),
            FocusField::OptPlayerName => Some(&mut self.draft.player_name),
            FocusField::OptLastHost => Some(&mut self.draft.last_host),
            FocusField::None => None,
        }
    }
}

fn copy_to_clipboard(text: &str) -> bool {
    use std::io::Write;
    use std::process::{Command, Stdio};
    for (bin, args) in [
        ("wl-copy", &[][..]),
        ("xclip", &["-selection", "clipboard"][..]),
        ("xsel", &["-b"][..]),
    ] {
        if let Ok(mut child) = Command::new(bin)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            if child.wait().is_ok_and(|s| s.success()) {
                return true;
            }
        }
    }
    false
}

enum MenuEvent {
    None,
    Quit,
    SettingsChanged,
    StartSkirmish(crate::lobby::LobbyLaunch),
    StartReplay(Replay),
    StartCampaign,
    StartTutorial,
}

// ---------------------------------------------------------------------------
// Single winit event loop hosting Menu + PlayApp
// ---------------------------------------------------------------------------

enum ActiveMode {
    Menu(Option<WindowGpu>),
    Match(Box<PlayApp>),
}

struct ShellApp {
    state: MenuState,
    active: ActiveMode,
    size: (u32, u32),
    init_err: Option<String>,
    hover: Option<(f32, f32)>,
    modifiers: ModifiersState,
    lmb_down: bool,
}

impl ShellApp {
    fn open_menu_window(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let gpu = WindowGpu::create(event_loop, "FARust — Forged Alliance", self.size, &self.state.fonts)?;
        gpu.window.request_redraw();
        self.active = ActiveMode::Menu(Some(gpu));
        Ok(())
    }

    fn enter_match(&mut self, event_loop: &ActiveEventLoop, session: PlaySession, map: MapContext, meshes: UnitMeshCache) {
        let sz = match &self.active {
            ActiveMode::Menu(Some(gpu)) => (gpu.config.width, gpu.config.height),
            _ => self.size,
        };
        // Drop the menu window cleanly before opening the match window on the same event loop.
        self.active = ActiveMode::Menu(None);
        let mut play = Box::new(PlayApp::new(Some(&self.state.game_dir), session, map, meshes, sz.0, sz.1));
        play.embedded = true;
        play.resumed(event_loop);
        if let Some(e) = play.take_init_err() {
            self.state.set_status(format!("Match launch failed: {e}"), true);
            if let Err(err) = self.open_menu_window(event_loop) {
                self.init_err = Some(err);
                event_loop.exit();
            }
            return;
        }
        self.active = ActiveMode::Match(play);
    }

    fn handle_menu_event(&mut self, event_loop: &ActiveEventLoop, ev: MenuEvent) {
        match ev {
            MenuEvent::None => {}
            MenuEvent::Quit => event_loop.exit(),
            MenuEvent::SettingsChanged => {
                self.size = self.state.settings.resolution;
                if let ActiveMode::Menu(Some(ref mut gpu)) = self.active {
                    gpu.reconfigure();
                    let _ = gpu.window.request_inner_size(winit::dpi::PhysicalSize::new(self.size.0, self.size.1));
                    let fs = self.state.settings.fullscreen
                        .then_some(winit::window::Fullscreen::Borderless(None));
                    gpu.window.set_fullscreen(fs);
                }
            }
            MenuEvent::StartSkirmish(launch) => {
                match build_skirmish_from_menu(
                    &self.state.game_dir,
                    &launch,
                    self.state.ai_difficulty,
                    self.state.game_options,
                ) {
                    Ok((mut sess, map, meshes)) => {
                        sess.muted = self.state.settings.master_volume == 0;
                        self.enter_match(event_loop, sess, map, meshes);
                    }
                    Err(e) => self.state.set_status(format!("Start failed: {e}"), true),
                }
            }
            MenuEvent::StartCampaign => match build_campaign_from_menu(&self.state.game_dir) {
                Ok((mut sess, map, meshes)) => {
                    sess.muted = self.state.settings.master_volume == 0;
                    self.enter_match(event_loop, sess, map, meshes);
                }
                Err(e) => self.state.set_status(format!("Campaign failed: {e}"), true),
            },
            MenuEvent::StartTutorial => match build_tutorial_from_menu(&self.state.game_dir) {
                Ok((mut sess, map, meshes)) => {
                    sess.muted = self.state.settings.master_volume == 0;
                    self.enter_match(event_loop, sess, map, meshes);
                }
                Err(e) => self.state.set_status(format!("Tutorial failed: {e}"), true),
            },
            MenuEvent::StartReplay(rep) => {
                match build_replay_from_menu(&self.state.game_dir, rep) {
                    Ok((mut sess, map, meshes)) => {
                        sess.muted = self.state.settings.master_volume == 0;
                        self.enter_match(event_loop, sess, map, meshes);
                    }
                    Err(e) => self.state.set_status(format!("Replay failed: {e}"), true),
                }
            }
        }
    }

    fn poll_mp(&mut self, event_loop: &ActiveEventLoop) -> bool {
        let mut dirty = false;
        let mut host_launch = None;
        let mut client_launch = None;
        let mut fail_msg = None;

        match &mut self.state.mp_session {
            MpSession::Host(host) => match host.poll() {
                HostPoll::Idle => {}
                HostPoll::Changed => dirty = true,
                HostPoll::Launch(l) => {
                    host_launch = Some(l);
                    dirty = true;
                }
                HostPoll::Failed(e) => {
                    fail_msg = Some(e);
                    dirty = true;
                }
            },
            MpSession::Client(client) => match client.poll() {
                ClientPoll::Idle => {}
                ClientPoll::Changed => dirty = true,
                ClientPoll::Launch(hs, transport) => {
                    client_launch = Some((hs, transport));
                    dirty = true;
                }
                ClientPoll::Closed(e) => {
                    fail_msg = Some(e);
                    dirty = true;
                }
            },
            MpSession::None => {}
        }

        if dirty {
            self.state.ensure_lobby_preview_image();
        }
        if let Some(msg) = fail_msg {
            self.state.close_mp_session();
            self.state.screen = ScreenId::Multiplayer;
            self.state.set_status(msg, true);
        }
        if let Some(launch) = host_launch {
            self.state.mp_session = MpSession::None;
            match start_net_host_match(&self.state.game_dir, launch, &self.state.settings) {
                Ok((sess, map, meshes)) => self.enter_match(event_loop, sess, map, meshes),
                Err(e) => {
                    self.state.screen = ScreenId::Multiplayer;
                    self.state.set_status(format!("Net launch error: {e}"), true);
                }
            }
            return true;
        }
        if let Some((hs, transport)) = client_launch {
            self.state.mp_session = MpSession::None;
            match start_net_client_match(&self.state.game_dir, hs, transport, &self.state.settings) {
                Ok((sess, map, meshes)) => self.enter_match(event_loop, sess, map, meshes),
                Err(e) => {
                    self.state.screen = ScreenId::Multiplayer;
                    self.state.set_status(format!("Net join error: {e}"), true);
                }
            }
            return true;
        }
        dirty
    }
}

impl ApplicationHandler for ShellApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if matches!(self.active, ActiveMode::Menu(None))
            && let Err(e) = self.open_menu_window(event_loop)
        {
            self.init_err = Some(e);
            event_loop.exit();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let ActiveMode::Match(ref mut play) = self.active {
            play.about_to_wait(event_loop);
            if let Some(exit) = play.take_exit() {
                match exit {
                    PlayExit::ToMenu => {
                        self.state.close_mp_session();
                        self.state.navigate(self.state.screen);
                        if let Err(e) = self.open_menu_window(event_loop) {
                            self.init_err = Some(e);
                            event_loop.exit();
                        }
                    }
                    PlayExit::Quit => event_loop.exit(),
                }
            }
            return;
        }

        let mp_dirty = self.poll_mp(event_loop);
        if let ActiveMode::Menu(Some(ref gpu)) = self.active {
            if mp_dirty {
                gpu.window.request_redraw();
            }
            let wait_ms = if matches!(self.state.mp_session, MpSession::None) { 100 } else { 20 };
            event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(wait_ms)));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        if let ActiveMode::Match(ref mut play) = self.active {
            play.window_event(event_loop, window_id, event);
            if let Some(exit) = play.take_exit() {
                match exit {
                    PlayExit::ToMenu => {
                        self.state.close_mp_session();
                        self.state.navigate(self.state.screen);
                        if let Err(e) = self.open_menu_window(event_loop) {
                            self.init_err = Some(e);
                            event_loop.exit();
                        }
                    }
                    PlayExit::Quit => event_loop.exit(),
                }
            }
            return;
        }

        let Some(gpu) = (match &mut self.active {
            ActiveMode::Menu(Some(g)) => Some(g),
            _ => None,
        }) else {
            return;
        };
        if gpu.window.id() != window_id {
            return;
        }

        let (w, h) = gpu.size();
        let mut redraw = false;

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return;
            }
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::Resized(s) => {
                gpu.resize(s.width, s.height);
                if let Some(ref mut lob) = self.state.lobby {
                    lob.visible_rows = lobby::visible_rows(lob, s.width as f32, s.height as f32);
                    lob.ensure_visible();
                }
                redraw = true;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x as f32, position.y as f32);
                self.hover = Some(p);
                if self.lmb_down {
                    self.state.drag_slider(w, h, p.0, p.1);
                }
                redraw = true;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y.round() as i32,
                    MouseScrollDelta::PixelDelta(p) => -(p.y / 24.0).round() as i32,
                };
                if d != 0 {
                    match self.state.screen {
                        ScreenId::Skirmish => {
                            if let Some(ref mut lob) = self.state.lobby {
                                lob.scroll_by(d);
                                redraw = true;
                            }
                        }
                        ScreenId::Keys => {
                            self.state.scroll_keys(d * 2, h);
                            redraw = true;
                        }
                        ScreenId::Replays => {
                            let max = self.state.replays.len().saturating_sub(12);
                            self.state.replay_scroll =
                                (self.state.replay_scroll as i64 + d as i64).clamp(0, max as i64) as usize;
                            redraw = true;
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                let pressed = state == ElementState::Pressed;
                self.lmb_down = pressed;
                self.state.shift_held = self.modifiers.shift_key();
                if pressed && let Some((px, py)) = self.hover {
                    let ev = self.state.click(w, h, px, py);
                    self.handle_menu_event(event_loop, ev);
                    redraw = true;
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if self.state.screen == ScreenId::Keys
                    && let PhysicalKey::Code(code) = event.physical_key
                    && self.state.capture_key(code, self.modifiers)
                {
                    gpu.window.request_redraw();
                    return;
                }
                if self.state.focus != FocusField::None {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => {
                            self.state.focus = FocusField::None;
                            redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter) => {
                            if self.state.focus == FocusField::MpChat {
                                self.state.submit_mp_chat();
                            } else {
                                self.state.focus = FocusField::None;
                            }
                            redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::Backspace) => {
                            if let Some(f) = self.state.active_field_mut() {
                                f.pop();
                            }
                            redraw = true;
                        }
                        _ => {
                            if let Some(t) = &event.text
                                && let Some(f) = self.state.active_field_mut()
                            {
                                for ch in t.chars() {
                                    if ch.is_ascii_graphic() || ch == ' ' {
                                        if f.len() < 48 {
                                            f.push(ch);
                                        }
                                    }
                                }
                                redraw = true;
                            }
                        }
                    }
                } else {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => {
                            match self.state.screen {
                                ScreenId::Main => {
                                    event_loop.exit();
                                    return;
                                }
                                ScreenId::MpLobby => {
                                    self.state.close_mp_session();
                                    self.state.navigate(ScreenId::Multiplayer);
                                }
                                ScreenId::Keys => self.state.navigate(ScreenId::Options),
                                _ => self.state.navigate(ScreenId::Main),
                            }
                            redraw = true;
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) if self.state.screen == ScreenId::Skirmish => {
                            if let Some(ref mut lob) = self.state.lobby {
                                lob.select_relative(-1);
                                self.state.ensure_lobby_preview_image();
                                redraw = true;
                            }
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) if self.state.screen == ScreenId::Skirmish => {
                            if let Some(ref mut lob) = self.state.lobby {
                                lob.select_relative(1);
                                self.state.ensure_lobby_preview_image();
                                redraw = true;
                            }
                        }
                        PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter)
                            if self.state.screen == ScreenId::Skirmish =>
                        {
                            if let Some(ref mut lob) = self.state.lobby
                                && let Some(launch) = lob.apply(LobbyAction::Start)
                            {
                                self.handle_menu_event(event_loop, MenuEvent::StartSkirmish(launch));
                                return;
                            }
                            redraw = true;
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let mut canvas = Canvas::new(w, h, &self.state.fonts, &self.state.images, self.hover);
                self.state.draw(&mut canvas);
                gpu.render(&canvas);
            }
            _ => {}
        }

        if redraw
            && let ActiveMode::Menu(Some(ref g)) = self.active
        {
            g.window.request_redraw();
        }
    }
}

// ---------------------------------------------------------------------------
// Match builders
// ---------------------------------------------------------------------------

fn build_skirmish_from_menu(
    game_dir: &Path,
    launch: &crate::lobby::LobbyLaunch,
    ai_diff: Difficulty,
    game_options: GameOptions,
) -> Result<(PlaySession, MapContext, UnitMeshCache), String> {
    let (mut session, map, meshes) = lobby::build_match(game_dir, launch)?;
    session.world.set_options(game_options);
    session.ais = session
        .world
        .armies()
        .iter()
        .filter(|a| session.spectate || a.id != session.human_army)
        .map(|a| ScriptedAi::with_difficulty(a.id, ai_diff))
        .collect();
    if let Some(ref mut rec) = session.recorder {
        rec.ai_events.clear();
        for ai in &session.ais {
            rec.ai_event(0, ai.army, true);
        }
    }
    Ok((session, map, meshes))
}

fn build_mission_from_source(
    game_dir: &Path,
    source: &str,
) -> Result<(PlaySession, MapContext, UnitMeshCache), String> {
    let m = crate::mission::Mission::parse(source)?;
    let mut catalog = Catalog::load(game_dir)?;
    m.ensure_catalog_types(&mut catalog, game_dir)?;
    let map = MapContext::load(game_dir, &m.map)?;
    let meshes = UnitMeshCache::load(Some(game_dir), &catalog);
    let world = crate::mission::build_world(&m, catalog, &map, crate::sim::DEFAULT_SEED)?;
    let human = m.human_army().unwrap_or(0) as crate::skirmish::ArmyId;
    let mut session = PlaySession::new(world, human, false, 1);
    session.ais = crate::mission::ai_armies(&m, None)?
        .into_iter()
        .map(|(a, d)| crate::skirmish_ai::ScriptedAi::with_difficulty(a, d))
        .collect();
    Ok((session, map, meshes))
}

/// Phase 39: the built-in tutorial mission as a ready-to-play session.
fn build_tutorial_from_menu(game_dir: &Path) -> Result<(PlaySession, MapContext, UnitMeshCache), String> {
    build_mission_from_source(game_dir, crate::mission::TUTORIAL)
}

/// Phase 49: Forged Alliance campaign Operation 1 (`X1CA_001`) as a ready-to-play session.
fn build_campaign_from_menu(game_dir: &Path) -> Result<(PlaySession, MapContext, UnitMeshCache), String> {
    build_mission_from_source(game_dir, crate::mission::CAMPAIGN)
}

fn build_replay_from_menu(
    game_dir: &Path,
    rep: Replay,
) -> Result<(PlaySession, MapContext, UnitMeshCache), String> {
    let mut catalog = Catalog::load(game_dir)?;
    if let Some(src) = &rep.header.mission {
        crate::mission::ensure_mission_types_src(src, &mut catalog, game_dir)?;
    }
    let map = MapContext::load(game_dir, &rep.header.map_id)?;
    let meshes = UnitMeshCache::load(Some(game_dir), &catalog);
    let world = savegame::origin_world(catalog, &map, &rep)?;
    Ok((PlaySession::new_replay(world, rep, 1), map, meshes))
}

fn start_net_host_match(
    game_dir: &Path,
    launch: crate::net_lobby::NetHostLaunch,
    settings: &Settings,
) -> Result<(PlaySession, MapContext, UnitMeshCache), String> {
    let catalog = Catalog::load(game_dir)?;
    let map = MapContext::load(game_dir, &launch.state.map_id)?;
    let (hs, host) = launch.into_transport(&map.starts)?;
    let meshes = UnitMeshCache::load(Some(game_dir), &catalog);
    let world = world_from_handshake(catalog, &map, &hs)?;
    let mut session = PlaySession::new(world, 0, false, 1);
    session.muted = settings.master_volume == 0;
    NetHandle::start(&session.world, &hs, Box::new(host))?.attach(&mut session);
    Ok((session, map, meshes))
}

fn start_net_client_match(
    game_dir: &Path,
    hs: crate::net::Handshake,
    client: crate::net::LockstepClient,
    settings: &Settings,
) -> Result<(PlaySession, MapContext, UnitMeshCache), String> {
    let catalog = Catalog::load(game_dir)?;
    let map = MapContext::load(game_dir, &hs.map_id)?;
    let meshes = UnitMeshCache::load(Some(game_dir), &catalog);
    let world = world_from_handshake(catalog, &map, &hs)?;
    let mut session = PlaySession::new(world, hs.assigned_army, false, 1);
    session.muted = settings.master_volume == 0;
    NetHandle::start(&session.world, &hs, Box::new(client))?.attach(&mut session);
    Ok((session, map, meshes))
}

fn scan_replays() -> Vec<ReplayItem> {
    let mut dirs = Vec::new();
    let d_replays = savegame::data_dir("replays");
    if d_replays.is_dir() {
        dirs.push(d_replays);
    }
    let cwd = PathBuf::from(".");
    if cwd.is_dir() {
        dirs.push(cwd);
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for dir in dirs {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in rd.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(savegame::REPLAY_EXT) {
                continue;
            }
            let key = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if !seen.insert(key) {
                continue;
            }
            if let Ok(replay) = Replay::load(&path) {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "replay".into());
                out.push(ReplayItem { path, name, replay });
            }
        }
    }
    out.sort_by(|a, b| b.name.cmp(&a.name));
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::{ReplayHeader, ReplayRecorder};
    use crate::skirmish::ArmySetup;
    use crate::ui_kit::hit_test;

    #[test]
    fn screen_id_parses_all_variants() {
        assert_eq!(ScreenId::from_str("main").unwrap(), ScreenId::Main);
        assert_eq!(ScreenId::from_str("skirmish").unwrap(), ScreenId::Skirmish);
        assert_eq!(ScreenId::from_str("lobby").unwrap(), ScreenId::Skirmish);
        assert_eq!(ScreenId::from_str("multiplayer").unwrap(), ScreenId::Multiplayer);
        assert_eq!(ScreenId::from_str("mp-lobby").unwrap(), ScreenId::MpLobby);
        assert_eq!(ScreenId::from_str("replays").unwrap(), ScreenId::Replays);
        assert_eq!(ScreenId::from_str("options").unwrap(), ScreenId::Options);
        assert_eq!(ScreenId::from_str("keys").unwrap(), ScreenId::Keys);
        assert!(ScreenId::from_str("bogus").is_err());
    }

    #[test]
    fn all_menu_screens_draw_and_hit_test_cleanly() {
        let mut st = MenuState::new(
            Path::new("/nonexistent"),
            &MenuOptions::default(),
        )
        .unwrap();
        let rep = ReplayRecorder::new(
            "SCMP_001",
            42,
            vec![
                ArmySetup { name: "A".into(), faction: 0, start: (0, 0), team: 0 },
                ArmySetup { name: "B".into(), faction: 1, start: (1000, 1000), team: 1 },
            ],
        )
        .finish(600, 0x1234);
        st.replays.push(ReplayItem {
            path: PathBuf::from("/tmp/test.farust-replay"),
            name: "test".into(),
            replay: rep,
        });
        st.lobby = Some(LobbyState::with_maps(
            Path::new("/nonexistent"),
            vec![LobbyMap { id: "SCMP_001".into(), name: "Burial Mounds".into(), size: (256, 256), armies: 2 }],
            "SCMP_001",
            42,
            "Tester",
            0,
        ));
        st.start_host_lobby("127.0.0.1:0").unwrap();

        for screen in [
            ScreenId::Main,
            ScreenId::Skirmish,
            ScreenId::Multiplayer,
            ScreenId::MpLobby,
            ScreenId::Replays,
            ScreenId::Options,
            ScreenId::Keys,
        ] {
            st.screen = screen;
            let mut c = Canvas::new(1280.0, 800.0, &st.fonts, &st.images, Some((200.0, 200.0)));
            st.draw(&mut c);
            let idx_count: usize = c.batches.iter().map(|b| b.idx.len()).sum();
            assert!(idx_count > 100, "{screen:?} drew too few indices ({idx_count})");
            let ws = st.widgets(1280.0, 800.0);
            assert!(!ws.is_empty(), "{screen:?} has no widgets");
            for (r, _) in &ws {
                assert!(r.right() <= 1280.5 && r.bottom() <= 800.5, "{screen:?} offscreen {r:?}");
                let (cx, cy) = r.center();
                assert!(hit_test(&ws, cx, cy).is_some(), "{screen:?} missed hit at {r:?}");
            }
        }
        st.close_mp_session();
    }

    #[test]
    fn options_and_skirmish_cycles_update_state() {
        let mut st = MenuState::new(Path::new("/nonexistent"), &MenuOptions::default()).unwrap();
        st.navigate(ScreenId::Options);
        let ws = st.options_widgets(1280.0, 800.0);
        let find = |pred: fn(&Action) -> bool| ws.iter().find(|(_, a)| pred(a)).unwrap().0.center();
        let (rx, ry) = find(|a| matches!(a, Action::CycleResolution));
        let before = st.draft.resolution;
        st.click(1280.0, 800.0, rx, ry);
        assert_ne!(st.draft.resolution, before);

        let (vx, vy) = find(|a| matches!(a, Action::ToggleVsync));
        let bv = st.draft.vsync;
        st.click(1280.0, 800.0, vx, vy);
        assert_ne!(st.draft.vsync, bv);

        st.navigate(ScreenId::Skirmish);
        st.lobby = Some(LobbyState::with_maps(
            Path::new("/nonexistent"),
            vec![LobbyMap { id: "A".into(), name: "Alpha".into(), size: (256, 256), armies: 2 }],
            "A",
            1,
            "You",
            0,
        ));
        let sws = st.skirmish_widgets(1280.0, 800.0);
        let (dx, dy) = sws.iter().find(|(_, a)| matches!(a, Action::CycleAiDiff)).unwrap().0.center();
        let d0 = st.ai_difficulty;
        st.click(1280.0, 800.0, dx, dy);
        assert_ne!(st.ai_difficulty, d0);
        let (wx, wy) = sws.iter().find(|(_, a)| matches!(a, Action::CycleVictory)).unwrap().0.center();
        let v0 = st.game_options.victory;
        st.click(1280.0, 800.0, wx, wy);
        assert_ne!(st.game_options.victory, v0);
        let _ = ReplayHeader { map_id: "A".into(), seed: 1, setups: vec![], options: st.game_options, mods: vec![], rules: crate::skirmish::RULES_VERSION, mission: None };
    }

    #[test]
    fn keys_screen_rebinds_detects_conflicts_and_keeps_draft() {
        use crate::keymap::{Action as KA, KeyChord};
        let mut st = MenuState::new(Path::new("/nonexistent"), &MenuOptions::default()).unwrap();
        st.navigate(ScreenId::Options);
        st.draft.card_hotkeys = true;
        st.navigate(ScreenId::Keys);
        assert!(st.draft.card_hotkeys, "Options -> Keys keeps the draft");
        let ws = st.keys_widgets(1280.0, 800.0);
        let idx = crate::keymap::actions().iter().position(|a| a.action == KA::BuildFactory).unwrap();
        let (r, _) = ws.iter().find(|(_, a)| *a == Action::KeyRow(idx)).copied().expect("row visible");
        st.click(1280.0, 800.0, r.center().0, r.center().1);
        assert_eq!(st.keys_capture, Some((KA::BuildFactory, false)));
        // Modifier presses alone keep waiting.
        assert!(st.capture_key(KeyCode::ControlLeft, ModifiersState::CONTROL));
        assert!(st.keys_capture.is_some());
        assert!(st.capture_key(KeyCode::KeyK, ModifiersState::CONTROL | ModifiersState::SHIFT));
        assert_eq!(st.draft.keys.get(KA::BuildFactory), &[crate::keymap::parse_chord("Ctrl+Shift+K").unwrap()]);
        assert!(!st.status_err);
        // Binding Stop to A conflicts with attack.
        st.keys_capture = Some((KA::Stop, false));
        assert!(st.capture_key(KeyCode::KeyA, ModifiersState::empty()));
        assert!(st.status_err && st.status.contains("CONFLICT"), "{}", st.status);
        assert_eq!(st.draft.keys.conflicts_with(KA::Stop), vec![KA::AttackMove]);
        // Shift+click adds an alternative; Backspace resets; Delete unbinds.
        st.keys_capture = Some((KA::Stop, true));
        assert!(st.capture_key(KeyCode::KeyI, ModifiersState::empty()));
        assert_eq!(st.draft.keys.get(KA::Stop), &[KeyChord::plain(KeyCode::KeyA), KeyChord::plain(KeyCode::KeyI)]);
        st.keys_capture = Some((KA::Stop, false));
        assert!(st.capture_key(KeyCode::Backspace, ModifiersState::empty()));
        assert_eq!(st.draft.keys.get(KA::Stop), &[KeyChord::plain(KeyCode::KeyS)]);
        st.keys_capture = Some((KA::ToggleFog, false));
        assert!(st.capture_key(KeyCode::Delete, ModifiersState::empty()));
        assert!(st.draft.keys.get(KA::ToggleFog).is_empty());
        assert!(!st.capture_key(KeyCode::KeyQ, ModifiersState::empty()), "not capturing");
        // Back to Options keeps the key edits; DEFAULTS restores them.
        st.navigate(ScreenId::Options);
        assert!(st.draft.keys.get(KA::ToggleFog).is_empty());
        st.navigate(ScreenId::Keys);
        let reset = st.keys_widgets(1280.0, 800.0).into_iter().find(|(_, a)| *a == Action::KeysReset).unwrap().0;
        st.click(1280.0, 800.0, reset.center().0, reset.center().1);
        assert!(st.draft.keys.is_default());
        // Scrolling reveals the last rows.
        st.scroll_keys(10_000, 800.0);
        let last = crate::keymap::actions().len() - 1;
        assert!(st.keys_widgets(1280.0, 800.0).iter().any(|(_, a)| *a == Action::KeyRow(last)));
    }

    #[test]
    fn classic_theme_draws_and_hit_tests_all_screens() {
        let mut st = MenuState::new(Path::new("/nonexistent"), &MenuOptions::default()).unwrap();
        st.settings.ui_theme = UiTheme::Classic;
        st.draft.ui_theme = UiTheme::Classic;
        st.lobby = Some(LobbyState::with_maps(
            Path::new("/nonexistent"),
            vec![LobbyMap { id: "SCMP_001".into(), name: "Burial Mounds".into(), size: (256, 256), armies: 2 }],
            "SCMP_001",
            42,
            "Tester",
            0,
        ));
        st.start_host_lobby("127.0.0.1:0").unwrap();
        for fidx in 0..FACTION_KEYS.len() {
            st.settings.default_faction = fidx;
            st.draft.default_faction = fidx;
            for screen in [
                ScreenId::Main,
                ScreenId::Skirmish,
                ScreenId::Multiplayer,
                ScreenId::MpLobby,
                ScreenId::Replays,
                ScreenId::Options,
                ScreenId::Keys,
            ] {
                st.screen = screen;
                let mut c = Canvas::new(1280.0, 800.0, &st.fonts, &st.images, Some((200.0, 200.0)));
                st.draw(&mut c);
                assert!(c.classic.is_some());
                let idx_count: usize = c.batches.iter().map(|b| b.idx.len()).sum();
                assert!(idx_count > 100, "{screen:?} faction {fidx} drew too few indices");
                let ws = st.widgets(1280.0, 800.0);
                for (r, _) in &ws {
                    assert!(r.right() <= 1280.5 && r.bottom() <= 800.5, "{screen:?} offscreen {r:?}");
                    let (cx, cy) = r.center();
                    assert!(hit_test(&ws, cx, cy).is_some(), "{screen:?} missed hit at {r:?}");
                }
            }
        }
        st.close_mp_session();
    }

    #[test]
    fn options_language_cycler_switches_and_draws_german() {
        let mut st = MenuState::new(Path::new("/nonexistent"), &MenuOptions::default()).unwrap();
        st.navigate(ScreenId::Options);
        assert_eq!(st.draft.language, "us");
        let ws = st.options_widgets(1280.0, 800.0);
        let (cx, cy) = ws.iter().find(|(_, a)| matches!(a, Action::CycleLanguage)).unwrap().0.center();
        st.click(1280.0, 800.0, cx, cy);
        assert_eq!(st.draft.language, "de");
        let mut c = Canvas::new(1280.0, 800.0, &st.fonts, &st.images, None);
        st.draw(&mut c);
        assert!(c.batches.iter().map(|b| b.idx.len()).sum::<usize>() > 100);
    }

    #[test]
    fn main_menu_campaign_and_tutorial_buttons_dispatch_events() {
        let mut st = MenuState::new(Path::new("/nonexistent"), &MenuOptions::default()).unwrap();
        let ws = st.widgets(1280.0, 800.0);
        let (cx, cy) = ws
            .iter()
            .find(|(_, a)| matches!(a, Action::StartCampaign))
            .expect("CAMPAIGN button on main menu")
            .0
            .center();
        assert!(matches!(st.click(1280.0, 800.0, cx, cy), MenuEvent::StartCampaign));
        let (tx, ty) = ws
            .iter()
            .find(|(_, a)| matches!(a, Action::StartTutorial))
            .expect("TUTORIAL button on main menu")
            .0
            .center();
        assert!(matches!(st.click(1280.0, 800.0, tx, ty), MenuEvent::StartTutorial));
    }
}
