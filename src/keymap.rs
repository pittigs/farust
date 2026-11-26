//! Phase 23 remappable key bindings.
//!
//! A [`KeyMap`] maps every in-match [`Action`] to zero or more [`KeyChord`]s (`key` plus
//! `Ctrl`/`Shift`/`Alt`). Defaults equal the hard-wired bindings of earlier phases. Bindings
//! are stored in `settings.ini` as `key.<action>=Ctrl+Shift+K` (several chords separated by
//! `,`; an empty value unbinds) and edited on the Options → Keys screen.
//!
//! Lookup is exact on the modifiers first; actions marked [`Flex::Shift`] also match with
//! an extra `Shift` (which then means "×5" / "queue"), and [`Flex::Any`] actions (control
//! groups) ignore `Ctrl`/`Shift` entirely (`Ctrl` assigns, `Ctrl+Shift` adds, `Shift`
//! recalls additively). Command-card grid slots live in their own [`Ctx::Card`] context,
//! active only while card hotkeys are enabled and a builder or factory is selected, so they
//! never conflict with the global bindings.

use std::sync::OnceLock;

use winit::keyboard::KeyCode;

/// One in-match action that can be bound to keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    CancelOrMenu,
    ChatAll,
    ChatAllies,
    ScoreScreen,
    QuickSave,
    QuickLoad,
    SelectAcu,
    SelectFactories,
    SelectArmy,
    IdleEngineer,
    ReplayPerspective,
    ReplayBack,
    ReplayForward,
    ReplayStart,
    CenterSelection,
    ResetCamera,
    PanUp,
    PanDown,
    PanLeft,
    PanRight,
    OrbitLeft,
    OrbitRight,
    AttackMove,
    Stop,
    BuildFactory,
    BuildExtractor,
    BuildGenerator,
    BuildPointDefense,
    BuildRadar,
    ProduceEngineer,
    ProduceTank,
    ProduceBot,
    ProduceArtillery,
    OverchargeNearest,
    UnloadMode,
    Enhance,
    EnhanceCycle,
    ToggleShield,
    ToggleRepeat,
    Pause,
    SpeedUp,
    SpeedDown,
    ToggleAi,
    ToggleFog,
    ToggleProfiler,
    OrderMove,
    OrderPatrol,
    OrderAssist,
    OrderGuard,
    OrderRepair,
    OrderReclaim,
    OrderLoad,
    OrderOvercharge,
    OrderLaunchMissile,
    CardNextTab,
    /// Control group `0..=9`.
    Group(u8),
    /// Camera bookmark `0..4` (save / recall).
    BookmarkSave(u8),
    BookmarkRecall(u8),
    /// Command card grid slot `0..15` (row-major, 5 columns).
    CardSlot(u8),
}

/// Binding context: card slots only conflict with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ctx {
    Global,
    Card,
}

/// How strictly the modifiers must match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flex {
    Exact,
    /// An extra `Shift` still matches (×5 / queue).
    Shift,
    /// `Ctrl` and `Shift` are ignored (control groups).
    Any,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyChord {
    pub key: KeyCode,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl KeyChord {
    pub const fn plain(key: KeyCode) -> Self {
        Self { key, ctrl: false, shift: false, alt: false }
    }
}

#[derive(Debug, Clone)]
pub struct ActionInfo {
    pub action: Action,
    /// Settings id (`key.<id>`).
    pub id: String,
    pub label: String,
    pub ctx: Ctx,
    pub flex: Flex,
    pub defaults: Vec<KeyChord>,
}

/// Number of camera bookmarks.
pub const BOOKMARKS: usize = 4;
/// Command card grid: 5 columns × 3 rows.
pub const CARD_SLOTS: usize = 15;

/// Default card slot keys (FA grid layout).
pub const CARD_GRID_KEYS: [&str; CARD_SLOTS] = ["Q", "W", "E", "R", "T", "A", "S", "D", "F", "G", "Z", "X", "C", "V", "B"];

const BOOKMARK_KEYS: [&str; BOOKMARKS] = ["F7", "F8", "F10", "F11"];

/// Every bindable action in display order.
pub fn actions() -> &'static [ActionInfo] {
    static TABLE: OnceLock<Vec<ActionInfo>> = OnceLock::new();
    TABLE.get_or_init(build_table)
}

fn build_table() -> Vec<ActionInfo> {
    use Action as A;
    use Flex::{Any, Exact, Shift};
    let fixed: &[(Action, &str, &str, Flex, &str)] = &[
        (A::CancelOrMenu, "cancel_menu", "Cancel mode / pause menu", Exact, "Escape"),
        (A::ChatAll, "chat_all", "Chat (all)", Exact, "Enter"),
        (A::ChatAllies, "chat_allies", "Chat (allies)", Exact, "Shift+Enter"),
        (A::ScoreScreen, "score", "Score screen", Shift, "Tab"),
        (A::QuickSave, "quicksave", "Quicksave", Exact, "F6"),
        (A::QuickLoad, "quickload", "Quickload", Exact, "F9"),
        (A::SelectAcu, "select_acu", "Select commander", Shift, "F"),
        (A::SelectFactories, "select_factories", "Select land factories", Shift, "H"),
        (A::SelectArmy, "select_army", "Select all mobile units", Shift, "J"),
        (A::IdleEngineer, "idle_engineer", "Next idle engineer", Shift, "Period"),
        (A::ReplayPerspective, "replay_view", "Replay: cycle perspective", Shift, "V"),
        (A::ReplayBack, "replay_back", "Replay: back 30s", Shift, "BracketLeft"),
        (A::ReplayForward, "replay_forward", "Replay: forward 30s", Shift, "BracketRight"),
        (A::ReplayStart, "replay_start", "Replay: restart", Shift, "Home"),
        (A::CenterSelection, "center", "Center on selection", Shift, "C"),
        (A::ResetCamera, "reset_camera", "Reset camera", Exact, "R"),
        (A::PanUp, "pan_up", "Pan up", Shift, "ArrowUp,W"),
        (A::PanDown, "pan_down", "Pan down", Shift, "ArrowDown"),
        (A::PanLeft, "pan_left", "Pan left", Shift, "ArrowLeft"),
        (A::PanRight, "pan_right", "Pan right", Shift, "ArrowRight"),
        (A::OrbitLeft, "orbit_left", "Orbit left", Shift, "Q"),
        (A::OrbitRight, "orbit_right", "Orbit right", Shift, "E"),
        (A::AttackMove, "attack", "Attack / attack-move", Shift, "A"),
        (A::Stop, "stop", "Stop", Shift, "S"),
        (A::BuildFactory, "build_factory", "Build land factory", Shift, "B"),
        (A::BuildExtractor, "build_mex", "Build mass extractor", Shift, "M"),
        (A::BuildGenerator, "build_pgen", "Build power generator", Shift, "P"),
        (A::BuildPointDefense, "build_pd", "Build point defense", Shift, "D"),
        (A::BuildRadar, "build_radar", "Build radar", Shift, "N"),
        (A::ProduceEngineer, "produce_engineer", "Produce engineer (Shift x5)", Shift, "G"),
        (A::ProduceTank, "produce_tank", "Produce tank (Shift x5)", Shift, "T"),
        (A::ProduceBot, "produce_bot", "Produce bot (Shift x5)", Shift, "Y"),
        (A::ProduceArtillery, "produce_arty", "Produce artillery (Shift x5)", Shift, "U"),
        (A::OverchargeNearest, "overcharge_nearest", "Overcharge nearest enemy", Shift, "O"),
        (A::UnloadMode, "unload", "Transport unload mode", Shift, "L"),
        (A::Enhance, "enhance", "Start selected enhancement", Exact, "K"),
        (A::EnhanceCycle, "enhance_cycle", "Cycle enhancement choice", Exact, "Shift+K"),
        (A::ToggleShield, "toggle_shield", "Toggle shields", Shift, "X"),
        (A::ToggleRepeat, "toggle_repeat", "Toggle factory repeat", Shift, "Z,Ctrl+R"),
        (A::Pause, "pause", "Pause", Shift, "Space"),
        (A::SpeedUp, "speed_up", "Game speed +", Shift, "Equal,NumpadAdd"),
        (A::SpeedDown, "speed_down", "Game speed -", Shift, "Minus,NumpadSubtract"),
        (A::ToggleAi, "toggle_ai", "Toggle AI control", Exact, "F1"),
        (A::ToggleFog, "toggle_fog", "Toggle fog of war", Exact, "F2"),
        (A::ToggleProfiler, "toggle_profiler", "Toggle profiler overlay", Exact, "F12"),
        (A::OrderMove, "order_move", "Order: move", Exact, "Ctrl+M"),
        (A::OrderPatrol, "order_patrol", "Order: patrol", Exact, "Ctrl+P"),
        (A::OrderAssist, "order_assist", "Order: assist", Exact, "Ctrl+A"),
        (A::OrderGuard, "order_guard", "Order: guard", Exact, "Ctrl+G"),
        (A::OrderRepair, "order_repair", "Order: repair", Exact, "Ctrl+E"),
        (A::OrderReclaim, "order_reclaim", "Order: reclaim", Exact, "Ctrl+C"),
        (A::OrderLoad, "order_load", "Order: load into transport", Exact, "Ctrl+L"),
        (A::OrderOvercharge, "order_overcharge", "Order: overcharge target", Exact, "Ctrl+O"),
        (A::OrderLaunchMissile, "order_missile", "Order: launch missile", Exact, "Ctrl+N"),
        (A::CardNextTab, "card_next_tab", "Command card: next tech tab", Exact, "Backquote"),
    ];
    let mut out: Vec<ActionInfo> = fixed
        .iter()
        .map(|&(action, id, label, flex, keys)| ActionInfo {
            action,
            id: id.to_string(),
            label: label.to_string(),
            ctx: Ctx::Global,
            flex,
            defaults: parse_chord_list(keys).expect("valid default binding"),
        })
        .collect();
    for d in [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 0] {
        out.push(ActionInfo {
            action: A::Group(d),
            id: format!("group_{d}"),
            label: format!("Control group {d} (Ctrl assign)"),
            ctx: Ctx::Global,
            flex: Any,
            defaults: vec![KeyChord::plain(parse_key_name(&format!("Digit{d}")).unwrap())],
        });
    }
    for (i, k) in BOOKMARK_KEYS.iter().enumerate() {
        let key = parse_key_name(k).unwrap();
        out.push(ActionInfo {
            action: A::BookmarkSave(i as u8),
            id: format!("bookmark_save_{}", i + 1),
            label: format!("Save camera bookmark {}", i + 1),
            ctx: Ctx::Global,
            flex: Exact,
            defaults: vec![KeyChord { ctrl: true, ..KeyChord::plain(key) }],
        });
        out.push(ActionInfo {
            action: A::BookmarkRecall(i as u8),
            id: format!("bookmark_{}", i + 1),
            label: format!("Recall camera bookmark {}", i + 1),
            ctx: Ctx::Global,
            flex: Exact,
            defaults: vec![KeyChord::plain(key)],
        });
    }
    for (i, k) in CARD_GRID_KEYS.iter().enumerate() {
        out.push(ActionInfo {
            action: A::CardSlot(i as u8),
            id: format!("card_slot_{}", i + 1),
            label: format!("Card slot {} (row {}, col {})", i + 1, i / 5 + 1, i % 5 + 1),
            ctx: Ctx::Card,
            flex: Shift,
            defaults: vec![KeyChord::plain(parse_key_name(k).unwrap())],
        });
    }
    out
}

pub fn info(action: Action) -> Option<&'static ActionInfo> {
    actions().iter().find(|a| a.action == action)
}

pub fn action_by_id(id: &str) -> Option<Action> {
    let id = id.trim();
    actions().iter().find(|a| a.id.eq_ignore_ascii_case(id)).map(|a| a.action)
}

// ---------------------------------------------------------------------------
// Key names
// ---------------------------------------------------------------------------

const NAMED_KEYS: &[(&str, KeyCode)] = &[
    ("Escape", KeyCode::Escape),
    ("Enter", KeyCode::Enter),
    ("Tab", KeyCode::Tab),
    ("Space", KeyCode::Space),
    ("Backspace", KeyCode::Backspace),
    ("Delete", KeyCode::Delete),
    ("Insert", KeyCode::Insert),
    ("Home", KeyCode::Home),
    ("End", KeyCode::End),
    ("PageUp", KeyCode::PageUp),
    ("PageDown", KeyCode::PageDown),
    ("ArrowUp", KeyCode::ArrowUp),
    ("ArrowDown", KeyCode::ArrowDown),
    ("ArrowLeft", KeyCode::ArrowLeft),
    ("ArrowRight", KeyCode::ArrowRight),
    ("Period", KeyCode::Period),
    ("Comma", KeyCode::Comma),
    ("Slash", KeyCode::Slash),
    ("Backslash", KeyCode::Backslash),
    ("Semicolon", KeyCode::Semicolon),
    ("Quote", KeyCode::Quote),
    ("Backquote", KeyCode::Backquote),
    ("BracketLeft", KeyCode::BracketLeft),
    ("BracketRight", KeyCode::BracketRight),
    ("Minus", KeyCode::Minus),
    ("Equal", KeyCode::Equal),
    ("NumpadAdd", KeyCode::NumpadAdd),
    ("NumpadSubtract", KeyCode::NumpadSubtract),
    ("NumpadMultiply", KeyCode::NumpadMultiply),
    ("NumpadDivide", KeyCode::NumpadDivide),
    ("NumpadEnter", KeyCode::NumpadEnter),
    ("NumpadDecimal", KeyCode::NumpadDecimal),
    ("Numpad0", KeyCode::Numpad0),
    ("Numpad1", KeyCode::Numpad1),
    ("Numpad2", KeyCode::Numpad2),
    ("Numpad3", KeyCode::Numpad3),
    ("Numpad4", KeyCode::Numpad4),
    ("Numpad5", KeyCode::Numpad5),
    ("Numpad6", KeyCode::Numpad6),
    ("Numpad7", KeyCode::Numpad7),
    ("Numpad8", KeyCode::Numpad8),
    ("Numpad9", KeyCode::Numpad9),
    ("Digit0", KeyCode::Digit0),
    ("Digit1", KeyCode::Digit1),
    ("Digit2", KeyCode::Digit2),
    ("Digit3", KeyCode::Digit3),
    ("Digit4", KeyCode::Digit4),
    ("Digit5", KeyCode::Digit5),
    ("Digit6", KeyCode::Digit6),
    ("Digit7", KeyCode::Digit7),
    ("Digit8", KeyCode::Digit8),
    ("Digit9", KeyCode::Digit9),
    ("F1", KeyCode::F1),
    ("F2", KeyCode::F2),
    ("F3", KeyCode::F3),
    ("F4", KeyCode::F4),
    ("F5", KeyCode::F5),
    ("F6", KeyCode::F6),
    ("F7", KeyCode::F7),
    ("F8", KeyCode::F8),
    ("F9", KeyCode::F9),
    ("F10", KeyCode::F10),
    ("F11", KeyCode::F11),
    ("F12", KeyCode::F12),
    ("A", KeyCode::KeyA),
    ("B", KeyCode::KeyB),
    ("C", KeyCode::KeyC),
    ("D", KeyCode::KeyD),
    ("E", KeyCode::KeyE),
    ("F", KeyCode::KeyF),
    ("G", KeyCode::KeyG),
    ("H", KeyCode::KeyH),
    ("I", KeyCode::KeyI),
    ("J", KeyCode::KeyJ),
    ("K", KeyCode::KeyK),
    ("L", KeyCode::KeyL),
    ("M", KeyCode::KeyM),
    ("N", KeyCode::KeyN),
    ("O", KeyCode::KeyO),
    ("P", KeyCode::KeyP),
    ("Q", KeyCode::KeyQ),
    ("R", KeyCode::KeyR),
    ("S", KeyCode::KeyS),
    ("T", KeyCode::KeyT),
    ("U", KeyCode::KeyU),
    ("V", KeyCode::KeyV),
    ("W", KeyCode::KeyW),
    ("X", KeyCode::KeyX),
    ("Y", KeyCode::KeyY),
    ("Z", KeyCode::KeyZ),
];

/// Settings name of a bindable key (`None` for keys that cannot be bound).
pub fn key_name(key: KeyCode) -> Option<&'static str> {
    NAMED_KEYS.iter().find(|(_, k)| *k == key).map(|(n, _)| *n)
}

/// Parses a key name (case-insensitive; `Esc`, `Return`, `Up`, `1`, `KeyA` aliases accepted).
pub fn parse_key_name(s: &str) -> Option<KeyCode> {
    let s = s.trim();
    let alias = match s.to_ascii_lowercase().as_str() {
        "esc" => "Escape",
        "return" => "Enter",
        "up" => "ArrowUp",
        "down" => "ArrowDown",
        "left" => "ArrowLeft",
        "right" => "ArrowRight",
        "." => "Period",
        "," => "Comma",
        "`" => "Backquote",
        "[" => "BracketLeft",
        "]" => "BracketRight",
        "=" | "plus" => "Equal",
        "-" => "Minus",
        _ => s,
    };
    let digit;
    let alias = if alias.len() == 1 && alias.as_bytes()[0].is_ascii_digit() {
        digit = format!("Digit{alias}");
        digit.as_str()
    } else if alias.len() == 4 && alias.is_ascii() && alias[..3].eq_ignore_ascii_case("key") {
        &alias[3..]
    } else {
        alias
    };
    NAMED_KEYS
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(alias))
        .map(|(_, k)| *k)
}

/// `Ctrl+Shift+K` style chord parser.
pub fn parse_chord(s: &str) -> Result<KeyChord, String> {
    let parts: Vec<&str> = s.split('+').map(str::trim).collect();
    let (key_part, mods) = match parts.split_last() {
        Some((k, m)) if !k.is_empty() => (*k, m),
        _ => return Err(format!("'{s}' has no key")),
    };
    let mut c = KeyChord::plain(parse_key_name(key_part).ok_or_else(|| format!("unknown key '{key_part}'"))?);
    for m in mods {
        match m.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => c.ctrl = true,
            "shift" => c.shift = true,
            "alt" => c.alt = true,
            other => return Err(format!("unknown modifier '{other}'")),
        }
    }
    Ok(c)
}

/// Comma-separated chord list; an empty string is an empty (unbound) list.
pub fn parse_chord_list(s: &str) -> Result<Vec<KeyChord>, String> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let c = parse_chord(part)?;
        if !out.contains(&c) {
            out.push(c);
        }
    }
    Ok(out)
}

pub fn chord_to_string(c: &KeyChord) -> String {
    let mut s = String::new();
    if c.ctrl {
        s.push_str("Ctrl+");
    }
    if c.shift {
        s.push_str("Shift+");
    }
    if c.alt {
        s.push_str("Alt+");
    }
    s.push_str(key_name(c.key).unwrap_or("?"));
    s
}

pub fn chords_to_string(v: &[KeyChord]) -> String {
    v.iter().map(chord_to_string).collect::<Vec<_>>().join(",")
}

/// Short HUD label of a chord (`^` = Ctrl, `+` = Shift prefix kept readable).
pub fn short_label(c: &KeyChord) -> String {
    let base = match key_name(c.key).unwrap_or("?") {
        n if n.starts_with("Digit") => n[5..].to_string(),
        "Period" => ".".to_string(),
        "Backquote" => "`".to_string(),
        "Escape" => "ESC".to_string(),
        "Space" => "SPC".to_string(),
        n => n.to_ascii_uppercase(),
    };
    let mut s = String::new();
    if c.ctrl {
        s.push('^');
    }
    if c.alt {
        s.push_str("A-");
    }
    if c.shift {
        s.push_str("S-");
    }
    s.push_str(&base);
    s
}

// ---------------------------------------------------------------------------
// KeyMap
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyMap {
    /// Parallel to [`actions`].
    binds: Vec<Vec<KeyChord>>,
}

impl Default for KeyMap {
    fn default() -> Self {
        Self { binds: actions().iter().map(|a| a.defaults.clone()).collect() }
    }
}

/// A chord bound to two actions of the same context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict {
    pub a: Action,
    pub b: Action,
    pub chord: KeyChord,
}

fn index_of(action: Action) -> Option<usize> {
    actions().iter().position(|a| a.action == action)
}

impl KeyMap {
    pub fn get(&self, action: Action) -> &[KeyChord] {
        index_of(action).and_then(|i| self.binds.get(i)).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn set(&mut self, action: Action, chords: Vec<KeyChord>) {
        if let Some(i) = index_of(action) {
            self.binds[i] = chords;
        }
    }

    pub fn reset(&mut self, action: Action) {
        if let Some(i) = index_of(action) {
            self.binds[i] = actions()[i].defaults.clone();
        }
    }

    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Applies one `key.<id>` settings value.
    pub fn set_from_setting(&mut self, id: &str, value: &str) -> Result<(), String> {
        let action = action_by_id(id).ok_or_else(|| format!("unknown key action '{id}'"))?;
        let chords = parse_chord_list(value)?;
        self.set(action, chords);
        Ok(())
    }

    /// `(settings key, value)` for every action, in table order.
    pub fn to_settings(&self) -> Vec<(String, String)> {
        actions()
            .iter()
            .zip(&self.binds)
            .map(|(a, b)| (format!("key.{}", a.id), chords_to_string(b)))
            .collect()
    }

    /// First chord of `action` as a short HUD label.
    pub fn label(&self, action: Action) -> Option<String> {
        self.get(action).first().map(short_label)
    }

    /// Resolves a key press. `card_active` enables the [`Ctx::Card`] slot bindings, which
    /// then take precedence over global ones.
    pub fn lookup(&self, key: KeyCode, ctrl: bool, shift: bool, alt: bool, card_active: bool) -> Option<Action> {
        let ctxs: &[Ctx] = if card_active { &[Ctx::Card, Ctx::Global] } else { &[Ctx::Global] };
        for &ctx in ctxs {
            // Pass 1: exact modifiers. Pass 2: Shift-flexible. Pass 3: modifier-agnostic.
            for pass in 0..3 {
                for (info, chords) in actions().iter().zip(&self.binds) {
                    if info.ctx != ctx {
                        continue;
                    }
                    let hit = chords.iter().any(|c| {
                        c.key == key
                            && match pass {
                                0 => c.ctrl == ctrl && c.shift == shift && c.alt == alt,
                                1 => info.flex == Flex::Shift && c.ctrl == ctrl && c.alt == alt && !c.shift && shift,
                                _ => info.flex == Flex::Any && c.alt == alt,
                            }
                    });
                    if hit {
                        return Some(info.action);
                    }
                }
            }
        }
        None
    }

    /// All duplicate chords within one context.
    pub fn conflicts(&self) -> Vec<Conflict> {
        let table = actions();
        let mut out = Vec::new();
        for i in 0..table.len() {
            for j in i + 1..table.len() {
                if table[i].ctx != table[j].ctx {
                    continue;
                }
                for c in &self.binds[i] {
                    if self.binds[j].contains(c) {
                        out.push(Conflict { a: table[i].action, b: table[j].action, chord: *c });
                    }
                }
            }
        }
        out
    }

    /// Actions sharing a chord with `action`.
    pub fn conflicts_with(&self, action: Action) -> Vec<Action> {
        self.conflicts()
            .into_iter()
            .filter_map(|c| {
                if c.a == action {
                    Some(c.b)
                } else if c.b == action {
                    Some(c.a)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Key map of the active settings (or `settings.ini` for the plain CLI, see
/// [`crate::settings::input_settings`]).
pub fn active() -> KeyMap {
    crate::settings::input_settings().keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chords_parse_and_serialize_round_trip() {
        let c = parse_chord("Ctrl+Shift+K").unwrap();
        assert_eq!(c, KeyChord { key: KeyCode::KeyK, ctrl: true, shift: true, alt: false });
        assert_eq!(chord_to_string(&c), "Ctrl+Shift+K");
        assert_eq!(parse_chord("alt+f7").unwrap(), KeyChord { alt: true, ..KeyChord::plain(KeyCode::F7) });
        assert_eq!(parse_chord("1").unwrap().key, KeyCode::Digit1);
        assert_eq!(parse_chord("KeyA").unwrap().key, KeyCode::KeyA);
        assert_eq!(parse_chord("Esc").unwrap().key, KeyCode::Escape);
        assert!(parse_chord("Ctrl+").is_err());
        assert!(parse_chord("Hyper+K").is_err());
        assert!(parse_chord("Ctrl+Banana").is_err());
        let list = parse_chord_list("Z, Ctrl+R ,Z").unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(chords_to_string(&list), "Z,Ctrl+R");
        assert!(parse_chord_list("").unwrap().is_empty());
        // Every key name round-trips.
        for (n, k) in NAMED_KEYS {
            assert_eq!(parse_key_name(n), Some(*k), "{n}");
            assert_eq!(key_name(*k), Some(*n));
        }
    }

    #[test]
    fn defaults_match_legacy_bindings_and_are_conflict_free() {
        let m = KeyMap::default();
        assert!(m.conflicts().is_empty(), "{:?}", m.conflicts());
        let ids: Vec<&str> = actions().iter().map(|a| a.id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate action ids");
        let k = |key| m.lookup(key, false, false, false, false);
        assert_eq!(k(KeyCode::KeyB), Some(Action::BuildFactory));
        assert_eq!(k(KeyCode::KeyM), Some(Action::BuildExtractor));
        assert_eq!(k(KeyCode::KeyT), Some(Action::ProduceTank));
        assert_eq!(k(KeyCode::KeyW), Some(Action::PanUp));
        assert_eq!(k(KeyCode::ArrowUp), Some(Action::PanUp));
        assert_eq!(k(KeyCode::KeyR), Some(Action::ResetCamera));
        assert_eq!(k(KeyCode::F9), Some(Action::QuickLoad));
        assert_eq!(k(KeyCode::F7), Some(Action::BookmarkRecall(0)));
        assert_eq!(k(KeyCode::F10), Some(Action::BookmarkRecall(2)));
        assert_eq!(m.lookup(KeyCode::F8, true, false, false, false), Some(Action::BookmarkSave(1)));
        // Ctrl+R repeat vs. plain R camera reset.
        assert_eq!(m.lookup(KeyCode::KeyR, true, false, false, false), Some(Action::ToggleRepeat));
        // Shift = x5 on produce keys; Shift+K is its own action.
        assert_eq!(m.lookup(KeyCode::KeyT, false, true, false, false), Some(Action::ProduceTank));
        assert_eq!(m.lookup(KeyCode::KeyK, false, true, false, false), Some(Action::EnhanceCycle));
        assert_eq!(m.lookup(KeyCode::KeyK, false, false, false, false), Some(Action::Enhance));
        assert_eq!(m.lookup(KeyCode::Enter, false, true, false, false), Some(Action::ChatAllies));
        // Control groups ignore Ctrl/Shift.
        assert_eq!(m.lookup(KeyCode::Digit3, true, true, false, false), Some(Action::Group(3)));
        assert_eq!(m.lookup(KeyCode::Digit0, false, false, false, false), Some(Action::Group(0)));
        // Exact actions do not fire with extra modifiers.
        assert_eq!(m.lookup(KeyCode::F1, false, true, false, false), None);
        assert_eq!(m.lookup(KeyCode::KeyB, false, false, true, false), None);
        // Card context overrides the global grid letters only while active.
        assert_eq!(m.lookup(KeyCode::KeyQ, false, false, false, true), Some(Action::CardSlot(0)));
        assert_eq!(m.lookup(KeyCode::KeyB, false, true, false, true), Some(Action::CardSlot(14)));
        assert_eq!(m.lookup(KeyCode::KeyQ, false, false, false, false), Some(Action::OrbitLeft));
        // Non-card keys still work with the card active.
        assert_eq!(m.lookup(KeyCode::KeyM, false, false, false, true), Some(Action::BuildExtractor));
    }

    #[test]
    fn settings_round_trip_and_conflicts() {
        let mut m = KeyMap::default();
        m.set_from_setting("build_factory", "Ctrl+Shift+K").unwrap();
        m.set_from_setting("toggle_fog", "").unwrap();
        assert!(m.set_from_setting("nope", "K").is_err());
        assert!(m.set_from_setting("stop", "Ctrl+Nope").is_err());
        assert_eq!(m.get(Action::BuildFactory), &[parse_chord("Ctrl+Shift+K").unwrap()]);
        assert!(m.get(Action::ToggleFog).is_empty());
        assert_eq!(m.lookup(KeyCode::KeyB, false, false, false, false), None);
        assert_eq!(m.lookup(KeyCode::KeyK, true, true, false, false), Some(Action::BuildFactory));

        let mut back = KeyMap::default();
        for (k, v) in m.to_settings() {
            back.set_from_setting(k.strip_prefix("key.").unwrap(), &v).unwrap();
        }
        assert_eq!(back, m);
        assert!(!back.is_default());

        // Conflict: bind stop to B as well as ... the factory key moved away, so bind it to S's twin.
        m.set(Action::Stop, vec![KeyChord::plain(KeyCode::KeyA)]);
        let c = m.conflicts();
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].chord, KeyChord::plain(KeyCode::KeyA));
        assert_eq!(m.conflicts_with(Action::Stop), vec![Action::AttackMove]);
        // Card slots never conflict with global keys.
        m.set(Action::CardSlot(0), vec![KeyChord::plain(KeyCode::KeyM)]);
        assert_eq!(m.conflicts().len(), 1);
        m.reset(Action::Stop);
        m.reset(Action::CardSlot(0));
        assert!(m.conflicts().is_empty());
        assert_eq!(short_label(&parse_chord("Ctrl+F7").unwrap()), "^F7");
        assert_eq!(short_label(&parse_chord("Digit4").unwrap()), "4");
    }
}
