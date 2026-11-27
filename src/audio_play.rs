//! Phase 14 game-side audio glue.
//!
//! Maps a [`PlaySession`] onto Forged Alliance XACT cues (names taken from the original
//! `.xsb` SoundBanks): sim events (weapons, impacts, explosions, missiles, ACU death),
//! unit lifecycle diffs (construction start / finish, factory roll-out, deaths), selection
//! voices, order acknowledgements, UI toggles, and adaptive music + ambient loops.
//! It only *reads* sim state (plus draining the audio copy of the event queue), so it
//! can never influence determinism. Also hosts the headless `audio render-mix` /
//! `audio cues` / `audio coverage` tooling.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::camera::OrbitCamera;
use crate::flowfield::NavDomain;
use crate::play::{self, PlaySession};
use crate::scmap::ScMap;
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{ArmyId, Catalog, Command, GameResult, Role, SimEvent, Skirmish, UnitId};
use crate::xact::{
    self, AudioEngine, Bus, CueId, CueSpec, Listener, OUTPUT_SAMPLE_RATE, TriggerOutcome, WaveBank,
};

// Cue playback policies (volume, cooldown s, max instances, steal priority).
const SPEC_WEAPON: CueSpec = CueSpec::sfx(0.55, 0.06, 4, 1.0);
const SPEC_IMPACT: CueSpec = CueSpec::sfx(0.5, 0.05, 4, 1.2);
const SPEC_EXPLOSION: CueSpec = CueSpec::sfx(0.8, 0.08, 4, 2.0);
const SPEC_EXPLOSION_LARGE: CueSpec = CueSpec::sfx(1.0, 0.12, 3, 3.0);
const SPEC_NUKE: CueSpec = CueSpec::sfx(1.4, 0.5, 2, 10.0);
const SPEC_DEATH: CueSpec = CueSpec::sfx(0.8, 0.08, 4, 2.5);
const SPEC_CONSTRUCT: CueSpec = CueSpec::sfx(0.6, 0.25, 3, 1.5);
const SPEC_ACTIVATE: CueSpec = CueSpec::sfx(0.7, 0.2, 3, 2.0);
const SPEC_ROLLOUT: CueSpec = CueSpec::sfx(0.5, 0.2, 3, 1.2);
const SPEC_MISSILE: CueSpec = CueSpec::sfx(0.8, 0.2, 3, 3.0);
const SPEC_INTERCEPT: CueSpec = CueSpec::sfx(0.7, 0.1, 3, 2.0);
const SPEC_SHIELD: CueSpec = CueSpec::sfx(0.35, 0.15, 2, 0.8);
const SPEC_ENHANCE: CueSpec = CueSpec::sfx(0.8, 0.3, 2, 3.0);
const SPEC_GATE_IN: CueSpec = CueSpec::sfx(0.9, 0.5, 2, 5.0);
const SPEC_SELECT: CueSpec = CueSpec::voice(0.9, 0.15);
const SPEC_UI: CueSpec = CueSpec::ui(0.6, 0.04);
/// Phase 24: looping construction / movement sounds (no cooldown; capped by the book).
const SPEC_LOOP: CueSpec = CueSpec::sfx(0.32, 0.0, 8, 0.9);
/// Unit loops audible at once (below [`xact::MAX_LOOP_VOICES`] so one-shots keep room).
const LOOP_LIMIT: usize = 6;
/// Loops only for units within this camera-eye distance (world units).
const LOOP_RANGE: f32 = 260.0;

const MUSIC_BASE: &str = "Base_Building";
const MUSIC_BATTLE: &str = "Battle";
const AMBIENT_LOOP: &str = "AMB_Planet_Rumble_zoom";
const UI_ORDER_MOVE: &str = "UI_Opt_Mini_Button_Click";
const UI_ORDER_ATTACK: &str = "UI_Action_MouseDown";
const UI_ORDER_BUILD: &str = "UI_Main_IG_Click";
const UI_ORDER_PRODUCE: &str = "UI_MFD_Click";
const UI_ORDER_STOP: &str = "UI_Menu_Cancel_02";
const UI_MODE_TOGGLE: &str = "UI_Tab_Click_01";
const UI_TOGGLE: &str = "UI_Menu_Select_01";
const UI_VOLUME: &str = "UI_Menu_Rollover_Sml";
const WEAPON_EXCLUDE: &[&str] = &["_charge", "_loop", "_stop", "_start", "_rof", "_flight", "_water", "_oc"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fac {
    Uef,
    Cybran,
    Aeon,
    Seraphim,
}

impl Fac {
    const ALL: [Fac; 4] = [Fac::Uef, Fac::Cybran, Fac::Aeon, Fac::Seraphim];

    fn from_key(key: &str) -> Self {
        let k = key.to_ascii_lowercase();
        if k.contains("cyb") {
            Fac::Cybran
        } else if k.contains("aeon") {
            Fac::Aeon
        } else if k.contains("sera") {
            Fac::Seraphim
        } else {
            Fac::Uef
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Fac::Uef => "UE",
            Fac::Cybran => "UR",
            Fac::Aeon => "UA",
            Fac::Seraphim => "XS",
        }
    }

    fn select_name(self) -> &'static str {
        match self {
            Fac::Uef => "UEF",
            Fac::Cybran => "Cybran",
            Fac::Aeon => "Aeon",
            Fac::Seraphim => "Seraphim",
        }
    }

    /// Generic faction weapon when the firing unit type has no own weapon cue.
    fn weapon(self, heavy: bool, anti_air: bool) -> &'static str {
        match (self, anti_air, heavy) {
            (Fac::Uef, true, _) => "UEL0104_Railgun",
            (Fac::Uef, false, true) => "UEL0202_Cannon_Dbl",
            (Fac::Uef, false, false) => "UEL0201_Cannon_Sgl",
            (Fac::Cybran, true, _) => "URL0104_Nano_Dart",
            (Fac::Cybran, false, true) => "URL0202_Cannon_Dbl",
            (Fac::Cybran, false, false) => "URL0107_Laser_Auto",
            (Fac::Aeon, true, _) => "UAL0205_AA_Fizz",
            (Fac::Aeon, false, true) => "UAL0203_Laser_Hvy",
            (Fac::Aeon, false, false) => "UAL0101_Laser_Lgt",
            (Fac::Seraphim, true, _) => "XSL0104_AA_Autocannon",
            (Fac::Seraphim, false, true) => "XSL0201_Cannon_Spectra",
            (Fac::Seraphim, false, false) => "XSL0101_PhasicAutogun",
        }
    }

    fn nuke_impact(self) -> &'static str {
        match self {
            Fac::Uef => "UEF_Nuke_Impact",
            Fac::Cybran => "Cybran_Nuke_Impact",
            Fac::Aeon | Fac::Seraphim => "Aeon_Nuke_Impact",
        }
    }

    fn destroy(self, structure: bool, tier: u8) -> &'static str {
        match (self, structure, tier) {
            (Fac::Uef, true, _) => "UEB_Destroy_Med_Land",
            (Fac::Uef, false, 4..) => "UEL_Destroy_Huge",
            (Fac::Uef, false, 2..) => "UEL_Destroy_Med_Land",
            (Fac::Uef, false, _) => "UEL_Destroy_Sml_Land",
            (Fac::Cybran, true, _) => "URB_Destroy_Land",
            (Fac::Cybran, false, 4..) => "URL_Destroy_Huge",
            (Fac::Cybran, false, _) => "URL_Destroy_Land",
            (Fac::Aeon, true, _) => "UAB_Destroy_Land",
            (Fac::Aeon, false, 4..) => "UAL_Destroy_Huge",
            (Fac::Aeon, false, _) => "UAL_Destroy_Land",
            (Fac::Seraphim, true, _) => "XSL_Destroy_Land_Lrg",
            (Fac::Seraphim, false, 4..) => "XSB_Destroy_Huge",
            (Fac::Seraphim, false, 2..) => "XSL_Destroy_Land_Med",
            (Fac::Seraphim, false, _) => "XSL_Destroy_Land_Sml",
        }
    }

    /// ACU warp-in (Seraphim has none of its own; Aeon/Cybran reuse UEF's as well).
    fn gate_in(self) -> &'static str {
        match self {
            Fac::Uef | Fac::Seraphim => "UEL0001_Gate_In",
            Fac::Cybran => "URL0001_Gate_In",
            Fac::Aeon => "UAL0001_Gate_In",
        }
    }

    fn builder_fallbacks(self) -> [String; 2] {
        let p = self.prefix();
        [format!("{p}L0105_Construct_Start"), format!("{p}L0001_Construct_Start")]
    }
}

/// FA selection-voice category (`<Faction>_Select_<Category>` cues in Interface /
/// SeraphimSelect SoundBanks).
fn select_category(role: Role, domain: NavDomain, tier: u8) -> &'static str {
    match role {
        Role::Commander => "Commander",
        Role::Factory | Role::AirFactory | Role::NavalFactory => "Factory",
        Role::Extractor | Role::Generator | Role::Hydro | Role::MassStorage | Role::EnergyStorage => "Resource",
        Role::PointDefense => "Gun",
        Role::Radar => "Radar",
        Role::Shield | Role::MissileSilo | Role::MissileDefense | Role::AntiAir | Role::Wall => "Structure",
        Role::Engineer => "Vehicle",
        Role::Experimental => "Bot",
        Role::Mobile => match domain {
            NavDomain::Air => "Air",
            NavDomain::Naval => "Naval",
            _ if tier >= 3 => "Bot",
            _ => "Tank",
        },
    }
}

fn selection_rank(role: Role) -> u8 {
    match role {
        Role::Commander => 5,
        Role::Experimental => 4,
        r if r.is_factory() => 3,
        r if r.is_structure() => 1,
        _ => 2,
    }
}

fn order_cue(cmd: &Command) -> &'static str {
    match cmd {
        Command::Move { .. }
        | Command::FormationMove { .. }
        | Command::Patrol { .. }
        | Command::Rally { .. }
        | Command::AttackMove { .. } => UI_ORDER_MOVE,
        Command::Attack { .. } | Command::Overcharge { .. } | Command::LaunchMissile { .. } => UI_ORDER_ATTACK,
        Command::Build { .. }
        | Command::Reclaim { .. }
        | Command::Repair { .. }
        | Command::Assist { .. }
        | Command::Upgrade { .. } => UI_ORDER_BUILD,
        Command::Produce { .. } => UI_ORDER_PRODUCE,
        Command::Stop { .. } | Command::ToggleShield { .. } => UI_ORDER_STOP,
        Command::Load { .. } | Command::Unload { .. } => UI_ORDER_MOVE,
        Command::Enhance { .. } => UI_ORDER_BUILD,
        Command::SetRepeat { .. } | Command::CancelProduce { .. } => UI_ORDER_PRODUCE,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UiFlags {
    build: bool,
    attack_move: bool,
    paused: bool,
    spectate: bool,
    fog: bool,
}

#[derive(Debug, Clone)]
struct Tracked {
    army: ArmyId,
    role: Role,
    tier: u8,
    building: bool,
    pos: [f32; 3],
    stamp: u32,
}

/// Phase 24: kind of a looping unit sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoopKind {
    /// Builder / factory working on a job (`*_Construct_Loop`).
    Construct,
    /// Mobile unit moving (`*_Move_Loop`).
    Move,
}

/// A loop a unit wants this frame; `score` ranks audibility (higher = closer).
#[derive(Debug, Clone, PartialEq)]
pub struct LoopWant {
    pub unit: UnitId,
    pub kind: LoopKind,
    pub cue: String,
    pub pos: [f32; 3],
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LoopActive {
    kind: LoopKind,
    cue: String,
    voice: u64,
}

/// What to do with the mixer this frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoopPlan {
    pub start: Vec<LoopWant>,
    pub stop: Vec<u64>,
    pub update: Vec<(u64, [f32; 3])>,
}

/// Phase 24 bookkeeping of looping unit sounds: which unit owns which loop voice. Pure
/// (no engine access) so the start/stop logic is unit-testable.
#[derive(Debug, Clone, Default)]
pub struct LoopBook {
    active: BTreeMap<UnitId, LoopActive>,
}

impl LoopBook {
    /// Score bonus keeping already-playing loops (avoids flapping at the limit).
    const KEEP_BONUS: f32 = 40.0;

    /// Reconciles running loops with `wants` (several per unit allowed; construction wins
    /// over movement), keeping at most `limit` by score. `alive(voice)` reports whether a
    /// voice still plays (stolen / failed voices are forgotten and may restart).
    pub fn plan(&mut self, mut wants: Vec<LoopWant>, limit: usize, alive: impl Fn(u64) -> bool) -> LoopPlan {
        self.active.retain(|_, a| alive(a.voice));
        wants.sort_by(|a, b| (a.unit, a.kind).cmp(&(b.unit, b.kind)));
        wants.dedup_by_key(|w| w.unit);
        let eff = |w: &LoopWant, active: &BTreeMap<UnitId, LoopActive>| {
            let keep = active.get(&w.unit).is_some_and(|a| a.kind == w.kind && a.cue == w.cue);
            w.score + if keep { Self::KEEP_BONUS } else { 0.0 }
        };
        wants.sort_by(|a, b| {
            eff(b, &self.active)
                .total_cmp(&eff(a, &self.active))
                .then(a.unit.cmp(&b.unit))
        });
        wants.truncate(limit);
        let mut plan = LoopPlan::default();
        let keep: BTreeMap<UnitId, &LoopWant> = wants.iter().map(|w| (w.unit, w)).collect();
        let stale: Vec<UnitId> = self
            .active
            .iter()
            .filter(|(u, a)| keep.get(u).is_none_or(|w| w.kind != a.kind || w.cue != a.cue))
            .map(|(u, _)| *u)
            .collect();
        for u in stale {
            if let Some(a) = self.active.remove(&u) {
                plan.stop.push(a.voice);
            }
        }
        for w in wants {
            match self.active.get(&w.unit) {
                Some(a) => plan.update.push((a.voice, w.pos)),
                None => plan.start.push(w),
            }
        }
        plan
    }

    /// Records that `w` started on mixer voice `voice`.
    pub fn started(&mut self, w: &LoopWant, voice: u64) {
        self.active.insert(w.unit, LoopActive { kind: w.kind, cue: w.cue.clone(), voice });
    }

    /// Forgets every loop, returning the voices to stop.
    pub fn clear(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.active).into_values().map(|a| a.voice).collect()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.active.len()
    }
}

/// Per-window audio state; owns an [`AudioEngine`] handle.
pub struct GameAudio {
    engine: AudioEngine,
    cues: HashMap<String, Option<CueId>>,
    dyn_names: HashMap<String, Option<String>>,
    units: HashMap<UnitId, Tracked>,
    stamp: u32,
    primed: bool,
    last_tick: u64,
    selection: Vec<UnitId>,
    ui: Option<UiFlags>,
    pending_seen: usize,
    heat: f32,
    calm_secs: f32,
    battle: bool,
    music: Option<u64>,
    track: usize,
    ambient: Option<u64>,
    music_retry: u32,
    ambient_retry: u32,
    result_seen: bool,
    /// Phase 24: running construction / movement loops.
    loops: LoopBook,
    /// Played-cue histogram (render-mix log / cue mapping verification).
    pub played: BTreeMap<String, u32>,
}

impl GameAudio {
    /// Realtime audio for an interactive window; `None` when no FA `sounds/` exist.
    pub fn new(game_dir: Option<&Path>, muted: bool) -> Option<Self> {
        let dir = game_dir?;
        if !dir.join("sounds").is_dir() {
            return None;
        }
        let ga = Self::with_engine(AudioEngine::new(Some(dir), muted))?;
        println!(
            "audio:    {} FA cues indexed{} | F3 mute, F4/F5 volume",
            ga.engine.cue_count(),
            if muted { " (muted)" } else { "" }
        );
        Some(ga)
    }

    /// Offline (no device) audio for headless rendering.
    pub fn offline(game_dir: &Path) -> Option<Self> {
        Self::with_engine(AudioEngine::offline(Some(game_dir)))
    }

    fn with_engine(engine: AudioEngine) -> Option<Self> {
        if engine.cue_count() == 0 {
            return None;
        }
        Some(Self {
            engine,
            cues: HashMap::new(),
            dyn_names: HashMap::new(),
            units: HashMap::new(),
            stamp: 0,
            primed: false,
            last_tick: 0,
            selection: Vec::new(),
            ui: None,
            pending_seen: 0,
            heat: 0.0,
            calm_secs: 0.0,
            battle: false,
            music: None,
            track: 0,
            ambient: None,
            music_retry: 0,
            ambient_retry: 0,
            result_seen: false,
            loops: LoopBook::default(),
            played: BTreeMap::new(),
        })
    }

    fn cue(&mut self, name: &str, spec: CueSpec) -> Option<CueId> {
        if let Some(c) = self.cues.get(name) {
            return *c;
        }
        let id = self.engine.resolve(name, spec);
        self.cues.insert(name.to_string(), id);
        id
    }

    /// Triggers cue `name`; returns false if the cue does not exist / decode.
    fn play(&mut self, name: &str, spec: CueSpec, pos: Option<[f32; 3]>, volume: f32) -> bool {
        let Some(id) = self.cue(name, spec) else {
            return false;
        };
        if let TriggerOutcome::Played(_) = self.engine.trigger(id, pos, volume) {
            *self.played.entry(name.to_string()).or_insert(0) += 1;
        }
        true
    }

    /// Plays the first resolvable cue among `names`.
    fn play_first<S: AsRef<str>>(&mut self, names: &[S], spec: CueSpec, pos: Option<[f32; 3]>, volume: f32) {
        for n in names {
            if self.play(n.as_ref(), spec, pos, volume) {
                return;
            }
        }
    }

    /// Per-unit-type weapon cue (first `<TYPE>_*` cue in a `*Weapon*` SoundBank).
    fn weapon_cue_for(&mut self, type_id: &str) -> Option<String> {
        let key = format!("w:{type_id}");
        if let Some(n) = self.dyn_names.get(&key) {
            return n.clone();
        }
        let prefix = format!("{}_", type_id.to_ascii_uppercase());
        let found = self
            .engine
            .with_library_ref(|lib| lib.find_prefixed(&prefix, "weapon", WEAPON_EXCLUDE))
            .flatten();
        self.dyn_names.insert(key, found.clone());
        found
    }

    fn faction(world: &Skirmish, army: ArmyId) -> Fac {
        world.faction(army).map(|f| Fac::from_key(&f.key)).unwrap_or(Fac::Uef)
    }

    /// Adopts the current world state without playing anything (headless fast-forward).
    pub fn prime(&mut self, s: &mut PlaySession) {
        let _ = s.world.drain_audio_events();
        for v in self.loops.clear() {
            self.engine.fade_out(v, 0.1);
        }
        self.stamp = self.stamp.wrapping_add(1);
        self.units.clear();
        for u in s.world.units().filter(|u| u.alive) {
            self.units.insert(
                u.id,
                Tracked {
                    army: u.army,
                    role: u.role,
                    tier: u.tier,
                    building: u.build.is_some(),
                    pos: [0.0; 3],
                    stamp: self.stamp,
                },
            );
        }
        self.selection = s.selected.clone();
        self.last_tick = s.world.tick();
        self.pending_seen = s.world.pending_view().len();
        self.primed = true;
    }

    /// Call once per frame *before* stepping the sim: selection voices, order
    /// acknowledgements (commands the player queued since last frame) and UI toggles.
    pub fn before_step(&mut self, s: &PlaySession) {
        let flags = UiFlags {
            build: s.build_mode.is_some(),
            attack_move: s.attack_move_mode,
            paused: s.paused,
            spectate: s.spectate,
            fog: s.fog_of_war,
        };
        if let Some(prev) = self.ui
            && prev != flags
        {
            let cue = if prev.build != flags.build || prev.attack_move != flags.attack_move {
                UI_MODE_TOGGLE
            } else {
                UI_TOGGLE
            };
            self.play(cue, SPEC_UI, None, 1.0);
        }
        self.ui = Some(flags);

        if s.selected != self.selection {
            let best = s
                .selected
                .iter()
                .filter(|id| !self.selection.contains(id))
                .filter_map(|id| s.world.unit(*id))
                .max_by_key(|u| selection_rank(u.role));
            if let Some(u) = best {
                let fac = Self::faction(&s.world, u.army);
                let cat = select_category(u.role, u.domain, u.tier);
                let names = [
                    format!("{}_Select_{cat}", fac.select_name()),
                    format!("{}_Select_Vehicle", fac.select_name()),
                    UI_TOGGLE.to_string(),
                ];
                self.play_first(&names, SPEC_SELECT, None, 1.0);
            }
            self.selection = s.selected.clone();
        }

        let pending = s.world.pending_view();
        if pending.len() < self.pending_seen {
            self.pending_seen = 0;
        }
        let ack = pending[self.pending_seen..]
            .iter()
            .find(|(army, _)| *army == s.human_army)
            .map(|(_, cmd)| order_cue(cmd));
        self.pending_seen = pending.len();
        if let Some(cue) = ack {
            self.play(cue, SPEC_UI, None, 1.0);
        }
    }

    /// Call once per frame *after* stepping: updates the listener from `cam`, turns the
    /// audio event stream and unit lifecycle changes into cues, and drives music.
    pub fn after_step(&mut self, s: &mut PlaySession, cam: &OrbitCamera) {
        self.engine.set_listener(Listener {
            view_proj: cam.view_proj_matrix(),
            eye: cam.eye_position(),
        });
        let tick = s.world.tick();
        let dt = tick.saturating_sub(self.last_tick) as f32 / TICK_HZ as f32;
        if self.primed && tick == self.last_tick {
            // No sim progress since last frame (paused / input-only redraw): skip the scan.
            self.update_music(0.0, cam);
            return;
        }
        if self.primed && (tick < self.last_tick || tick - self.last_tick > 40) {
            // Replay seek / fast-forward: resync silently instead of replaying a backlog.
            self.prime(s);
            self.update_music(0.0, cam);
            return;
        }
        let ground = cam.target[1];
        let scale = SCALE as f32;
        let to_world = |x: i64, y: i64, alt: i64| [x as f32 / scale, ground + alt as f32 / scale, y as f32 / scale];

        // ---- unit lifecycle diff -------------------------------------------------
        self.stamp = self.stamp.wrapping_add(1);
        let stamp = self.stamp;
        let primed = self.primed;
        let last_tick = self.last_tick;
        let mut shooters: Vec<(i64, i64, ArmyId, String)> = Vec::new();
        let mut starts: Vec<(String, ArmyId, Role, [f32; 3])> = Vec::new();
        let mut births: Vec<(String, ArmyId, Role, [f32; 3])> = Vec::new();
        for u in s.world.units() {
            if !u.alive {
                continue;
            }
            let p = to_world(u.pos.x, u.pos.y, u.altitude);
            if primed && u.last_shot.is_some_and(|(_, t)| t >= last_tick) {
                shooters.push((u.pos.x, u.pos.y, u.army, u.type_id.clone()));
            }
            let building = u.build.is_some();
            match self.units.get_mut(&u.id) {
                Some(t) => {
                    if building && !t.building {
                        starts.push((u.type_id.clone(), u.army, u.role, p));
                    }
                    t.building = building;
                    t.pos = p;
                    t.stamp = stamp;
                }
                None => {
                    if primed || u.role == Role::Commander {
                        births.push((u.type_id.clone(), u.army, u.role, p));
                    }
                    self.units.insert(
                        u.id,
                        Tracked {
                            army: u.army,
                            role: u.role,
                            tier: u.tier,
                            building,
                            pos: p,
                            stamp,
                        },
                    );
                }
            }
        }
        let mut deaths: Vec<Tracked> = Vec::new();
        self.units.retain(|_, t| {
            let keep = t.stamp == stamp;
            if !keep {
                deaths.push(t.clone());
            }
            keep
        });

        // ---- sim events ------------------------------------------------------------
        let events = s.world.drain_audio_events();
        if primed {
            for ev in &events {
                self.on_event(&s.world, ev, &shooters, &to_world);
            }
        }
        for (type_id, army, role, p) in starts {
            let fac = Self::faction(&s.world, army);
            let own = format!("{}_Construct_Start", type_id.to_ascii_uppercase());
            if role.is_factory() {
                let names = [own, format!("{}B0101_Construct_Start", fac.prefix())];
                self.play_first(&names, SPEC_CONSTRUCT, Some(p), 1.0);
            } else if role.is_builder() {
                let [a, b] = fac.builder_fallbacks();
                self.play_first(&[own, a, b], SPEC_CONSTRUCT, Some(p), 1.0);
            }
        }
        for (type_id, army, role, p) in births {
            let fac = Self::faction(&s.world, army);
            let upper = type_id.to_ascii_uppercase();
            if !primed {
                // Initial commanders warp in.
                let names = [format!("{upper}_Gate_In"), fac.gate_in().to_string()];
                self.play_first(&names, SPEC_GATE_IN, Some(p), 1.0);
            } else if role.is_structure() {
                let names = [format!("{upper}_Activate"), format!("{}B0101_Activate", fac.prefix())];
                self.play_first(&names, SPEC_ACTIVATE, Some(p), 1.0);
            } else {
                self.play(&format!("{upper}_Move_Start"), SPEC_ROLLOUT, Some(p), 1.0);
            }
        }
        for d in deaths {
            if d.role == Role::Commander {
                continue; // SimEvent::AcuDied plays the nuke-scale blast.
            }
            let fac = Self::faction(&s.world, d.army);
            let cue = fac.destroy(d.role.is_structure(), d.tier);
            self.play(cue, SPEC_DEATH, Some(d.pos), 1.0);
            self.heat += 0.3;
        }

        self.update_loops(&s.world, cam, &to_world);

        // ---- game end, music, ambient -----------------------------------------------
        if !self.result_seen
            && let Some(res) = s.world.result()
        {
            self.result_seen = true;
            let won = s.spectate || matches!(res, GameResult::Victory(a) if a == s.human_army);
            if let Some(id) = self.music.take() {
                self.engine.fade_out(id, 1.5);
            }
            let cue = if won { "UI_END_Game_Victory" } else { "UI_END_Game_Fail" };
            if self.engine.play_stream(cue, 0, Bus::Music, 1.0, false, 0.3).is_some() {
                *self.played.entry(cue.to_string()).or_insert(0) += 1;
            }
        }
        self.update_music(dt, cam);

        self.last_tick = tick;
        self.primed = true;
    }

    /// First resolvable cue among `names`, cached under `key`.
    fn loop_cue(&mut self, key: String, names: &[String]) -> Option<String> {
        if let Some(n) = self.dyn_names.get(&key) {
            return n.clone();
        }
        let found = names.iter().find(|n| self.cue(n, SPEC_LOOP).is_some()).cloned();
        self.dyn_names.insert(key, found.clone());
        found
    }

    /// Phase 24: starts / moves / stops looping construction and movement sounds for the
    /// closest working or moving units (at most [`LOOP_LIMIT`]).
    fn update_loops(&mut self, world: &Skirmish, cam: &OrbitCamera, to_world: &dyn Fn(i64, i64, i64) -> [f32; 3]) {
        let eye = cam.eye_position();
        let mut wants = Vec::new();
        if world.result().is_none() {
            for u in world.units() {
                if !u.alive || u.carried_by.is_some() {
                    continue;
                }
                let kind = if u.build.is_some() && (u.role.is_builder() || u.role.is_factory()) {
                    LoopKind::Construct
                } else if u.moving && !u.role.is_structure() {
                    LoopKind::Move
                } else {
                    continue;
                };
                let p = to_world(u.pos.x, u.pos.y, u.altitude);
                let d = ((p[0] - eye[0]).powi(2) + (p[1] - eye[1]).powi(2) + (p[2] - eye[2]).powi(2)).sqrt();
                if d > LOOP_RANGE {
                    continue;
                }
                let upper = u.type_id.to_ascii_uppercase();
                let cue = match kind {
                    LoopKind::Construct => {
                        let fac = Self::faction(world, u.army);
                        let fallback = if u.role.is_factory() {
                            format!("{}B0101_Construct_Loop", fac.prefix())
                        } else {
                            format!("{}L0105_Construct_Loop", fac.prefix())
                        };
                        self.loop_cue(format!("cl:{upper}"), &[format!("{upper}_Construct_Loop"), fallback])
                    }
                    LoopKind::Move => self.loop_cue(format!("ml:{upper}"), &[format!("{upper}_Move_Loop")]),
                };
                if let Some(cue) = cue {
                    wants.push(LoopWant { unit: u.id, kind, cue, pos: p, score: -d });
                }
            }
        }
        let engine = self.engine.clone();
        let plan = self.loops.plan(wants, LOOP_LIMIT, |v| engine.voice_active(v));
        for v in plan.stop {
            self.engine.fade_out(v, 0.25);
        }
        for (v, p) in plan.update {
            self.engine.set_voice_spatial(v, Some(p), 1.0);
        }
        for w in plan.start {
            let Some(id) = self.cue(&w.cue, SPEC_LOOP) else {
                continue;
            };
            if let TriggerOutcome::Played(v) = self.engine.start_loop(id, Some(w.pos), 1.0) {
                *self.played.entry(w.cue.clone()).or_insert(0) += 1;
                self.loops.started(&w, v);
            }
        }
    }

    fn on_event(
        &mut self,
        world: &Skirmish,
        ev: &SimEvent,
        shooters: &[(i64, i64, ArmyId, String)],
        to_world: &dyn Fn(i64, i64, i64) -> [f32; 3],
    ) {
        match ev {
            SimEvent::WeaponFired {
                from,
                army,
                heavy,
                anti_air,
                ..
            } => {
                let near = 4 * SCALE;
                let shooter = shooters
                    .iter()
                    .filter(|(x, y, a, _)| a == army && (x - from.0).abs() <= near && (y - from.1).abs() <= near)
                    .min_by_key(|(x, y, _, _)| (x - from.0).abs() + (y - from.1).abs())
                    .map(|(_, _, _, t)| t.clone());
                let own = shooter.and_then(|t| self.weapon_cue_for(&t));
                let fallback = Self::faction(world, *army).weapon(*heavy, *anti_air);
                let pos = Some(to_world(from.0, from.1, from.2));
                match own {
                    Some(n) => self.play_first(&[n.as_str(), fallback], SPEC_WEAPON, pos, 1.0),
                    None => self.play_first(&[fallback], SPEC_WEAPON, pos, 1.0),
                }
                self.heat += 0.02;
            }
            SimEvent::Explosion { pos, radius, is_nuke } => {
                let p = Some(to_world(pos.0, pos.1, pos.2));
                if *is_nuke {
                    self.play("UEF_Nuke_Impact", SPEC_NUKE, p, 1.0);
                    self.heat += 6.0;
                } else if *radius <= 5 * SCALE / 2 {
                    self.play("Impact_Land_Gen_UEF", SPEC_IMPACT, p, 1.0);
                    self.heat += 0.05;
                } else if *radius < 6 * SCALE {
                    self.play("Explosion_Medium", SPEC_EXPLOSION, p, 1.0);
                    self.heat += 0.15;
                } else {
                    self.play("Explosion_Large_01", SPEC_EXPLOSION_LARGE, p, 1.0);
                    self.heat += 0.4;
                }
            }
            SimEvent::AcuDied { pos, army } => {
                let cue = Self::faction(world, *army).nuke_impact();
                self.play(cue, SPEC_NUKE, Some(to_world(pos.0, pos.1, pos.2)), 1.6);
                self.heat += 8.0;
            }
            SimEvent::ShieldHit { pos, .. } => {
                self.play("Impact_Land_Gen_AEON", SPEC_SHIELD, Some(to_world(pos.0, pos.1, pos.2)), 1.0);
            }
            SimEvent::MissileLaunched { pos, is_nuke } => {
                let cue = if *is_nuke { "UEB2305_Nuke_Fire" } else { "UEB2108_Missile_Cruise" };
                self.play(cue, SPEC_MISSILE, Some(to_world(pos.0, pos.1, pos.2)), 1.0);
            }
            SimEvent::MissileIntercepted { pos } => {
                self.play("Expl_Anti_Nuke", SPEC_INTERCEPT, Some(to_world(pos.0, pos.1, pos.2)), 1.0);
            }
            SimEvent::UpgradeComplete { pos, .. } => {
                self.play("GLB_Enhance_Stop", SPEC_ENHANCE, Some(to_world(pos.0, pos.1, pos.2)), 1.0);
            }
            // Construction is voiced from the unit lifecycle diff (start / activate).
            SimEvent::BuildSpark { .. } => {}
        }
    }

    /// Adaptive music: `Base_Building` playlist, crossfading to `Battle` when combat
    /// intensity rises and back after ~20 s of calm; ambient rumble scales with zoom.
    fn update_music(&mut self, dt: f32, cam: &OrbitCamera) {
        self.heat *= 0.5f32.powf(dt / 8.0);
        self.calm_secs = if self.heat < 1.0 { self.calm_secs + dt } else { 0.0 };
        let want_battle = if self.battle {
            !(self.heat < 1.0 && self.calm_secs > 20.0)
        } else {
            self.heat > 4.0
        };
        self.music_retry = self.music_retry.saturating_sub(1);
        self.ambient_retry = self.ambient_retry.saturating_sub(1);
        let music_live = self.music.is_some_and(|id| self.engine.voice_active(id));
        if !self.result_seen && (want_battle != self.battle || !music_live) && self.music_retry == 0 {
            if let Some(id) = self.music.take() {
                self.engine.fade_out(id, 2.5);
            }
            self.battle = want_battle;
            let cue = if want_battle { MUSIC_BATTLE } else { MUSIC_BASE };
            self.music = self.engine.play_stream(cue, self.track, Bus::Music, 1.0, false, 2.0);
            self.track += 1;
            if self.music.is_some() {
                *self.played.entry(cue.to_string()).or_insert(0) += 1;
            } else {
                self.music_retry = 300;
            }
        }
        let amb_vol = ((cam.distance - 60.0) / 400.0).clamp(0.15, 1.0);
        match self.ambient {
            Some(id) if self.engine.voice_active(id) => self.engine.set_voice_volume(id, amb_vol),
            _ if self.ambient_retry == 0 => {
                self.ambient = self.engine.play_stream(AMBIENT_LOOP, 0, Bus::Ambient, amb_vol, true, 3.0);
                if self.ambient.is_none() {
                    self.ambient_retry = 300;
                } else {
                    *self.played.entry(AMBIENT_LOOP.to_string()).or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }

    /// F3 toggles mute, F4 / F5 lower / raise master volume (`[`/`]` are replay seek).
    /// Returns true if consumed.
    pub fn handle_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::F3 => {
                let muted = !self.engine.is_muted();
                self.engine.set_muted(muted);
                self.music_retry = 0;
                self.ambient_retry = 0;
                println!("audio:    {}", if muted { "muted" } else { "unmuted" });
                if !muted {
                    self.play(UI_VOLUME, SPEC_UI, None, 1.0);
                }
                true
            }
            KeyCode::F4 | KeyCode::F5 => {
                let delta = if key == KeyCode::F5 { 0.1 } else { -0.1 };
                let v = self.engine.adjust_master(delta);
                println!("audio:    master volume {:.0}%", v * 100.0);
                self.play(UI_VOLUME, SPEC_UI, None, 1.0);
                true
            }
            _ => false,
        }
    }

    /// Window-loop hook: forwards audio hotkeys to [`GameAudio::handle_key`].
    pub fn handle_window_event(&mut self, event: &WindowEvent) {
        if let WindowEvent::KeyboardInput { event, .. } = event
            && event.state == ElementState::Pressed
            && !event.repeat
            && let PhysicalKey::Code(code) = event.physical_key
        {
            self.handle_key(code);
        }
    }
}

/// Audio slot stored on `PlaySession::audio`; started lazily on the first window
/// frame (needs `game_dir` and the `muted` flag). Cloned sessions are silent.
#[derive(Default)]
pub enum SessionAudio {
    #[default]
    Uninit,
    Off,
    On(Box<GameAudio>),
}

impl Clone for SessionAudio {
    fn clone(&self) -> Self {
        match self {
            SessionAudio::Uninit => SessionAudio::Uninit,
            _ => SessionAudio::Off,
        }
    }
}

impl std::fmt::Debug for SessionAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            SessionAudio::Uninit => "SessionAudio::Uninit",
            SessionAudio::Off => "SessionAudio::Off",
            SessionAudio::On(_) => "SessionAudio::On",
        })
    }
}

/// Window-loop hook, once per `about_to_wait`: voices the sim progress since the last
/// call (events, unit lifecycle), then selection / order / UI changes made by input.
pub fn frame(s: &mut PlaySession, game_dir: Option<&Path>, cam: &OrbitCamera) {
    let mut slot = std::mem::take(&mut s.audio);
    if let SessionAudio::Uninit = slot {
        slot = match GameAudio::new(game_dir, s.muted) {
            Some(ga) => SessionAudio::On(Box::new(ga)),
            None => SessionAudio::Off,
        };
    }
    if let SessionAudio::On(ga) = &mut slot {
        ga.after_step(s, cam);
        ga.before_step(s);
    }
    s.audio = slot;
}

/// Window-event hook: audio hotkeys (F3 mute, F4/F5 volume).
pub fn window_event(s: &mut PlaySession, event: &WindowEvent) {
    if let SessionAudio::On(ga) = &mut s.audio {
        ga.handle_window_event(event);
    }
}

// ---------------------------------------------------------------------------
// Headless tooling
// ---------------------------------------------------------------------------

/// Summary statistics of an interleaved stereo `f32` buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcmStats {
    pub seconds: f64,
    pub peak: f32,
    pub rms: f32,
    pub active_rms: f32,
    pub silent_fraction: f32,
    pub clipped: usize,
}

pub fn pcm_stats(pcm: &[f32]) -> PcmStats {
    let frames = pcm.len() / 2;
    let peak = pcm.iter().fold(0.0f32, |a, s| a.max(s.abs()));
    let sum_sq: f64 = pcm.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let win = (OUTPUT_SAMPLE_RATE / 10) as usize * 2;
    let (mut silent, mut windows, mut active_sq, mut active_n) = (0usize, 0usize, 0f64, 0usize);
    for w in pcm.chunks(win) {
        let e: f64 = w.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        windows += 1;
        if (e / w.len().max(1) as f64).sqrt() < 0.001 {
            silent += 1;
        } else {
            active_sq += e;
            active_n += w.len();
        }
    }
    PcmStats {
        seconds: frames as f64 / f64::from(OUTPUT_SAMPLE_RATE),
        peak,
        rms: (sum_sq / pcm.len().max(1) as f64).sqrt() as f32,
        active_rms: (active_sq / active_n.max(1) as f64).sqrt() as f32,
        silent_fraction: silent as f32 / windows.max(1) as f32,
        clipped: pcm.iter().filter(|s| s.abs() >= 0.999).count(),
    }
}

fn dbfs(v: f32) -> f32 {
    20.0 * v.max(1e-9).log10()
}

/// Runs `session` headless for `ticks`, feeding every frame through the same
/// [`GameAudio`] mapping and mixer as the window, writes a 48 kHz S16 stereo WAV and
/// prints level statistics plus the cue histogram. Also re-simulates without audio and
/// checks the state hash is identical (audio never feeds back into the sim).
pub fn render_mix(
    game_dir: &Path,
    session: PlaySession,
    scmap: &ScMap,
    from_tick: u64,
    ticks: u64,
    out: &Path,
) -> Result<(), String> {
    let mut ga = GameAudio::offline(game_dir)
        .ok_or_else(|| format!("no FA sound banks under {}", game_dir.join("sounds").display()))?;
    let mut reference = session.clone();
    let mut s = session;
    let fpt = (OUTPUT_SAMPLE_RATE as u64 / TICK_HZ) as usize;
    let mut pcm: Vec<f32> = Vec::with_capacity(ticks as usize * fpt * 2 + 4 * OUTPUT_SAMPLE_RATE as usize * 2);
    let aspect = 16.0 / 9.0;
    let mut cam = play::initial_army_camera(&s.world, scmap, 0, aspect);
    let t0 = std::time::Instant::now();
    let mut stepped = 0u64;
    while stepped < from_tick && s.world.result().is_none() {
        s.step_sim();
        stepped += 1;
    }
    if stepped > 0 {
        ga.prime(&mut s);
        cam = play::battle_snapshot_camera(&s.world, scmap, aspect);
        cam.distance = 90.0;
    }
    let window_start = stepped;
    for t in 0..ticks {
        // Exercise selection voices (UI-only state; the sim never sees selection).
        match t {
            20 => s.select_commander(),
            900 => s.select_factories(),
            1800 => s.select_army(),
            _ => {}
        }
        ga.before_step(&s);
        if s.world.result().is_none() {
            s.step_sim();
            stepped += 1;
        }
        if t % 50 == 0 && (t >= 300 || window_start > 0) {
            cam = play::battle_snapshot_camera(&s.world, scmap, aspect);
            cam.distance = 90.0;
        }
        ga.after_step(&mut s, &cam);
        pcm.extend(ga.engine.render(fpt));
        if s.world.result().is_some() && t + window_start > stepped + 40 {
            break;
        }
    }
    pcm.extend(ga.engine.render(2 * OUTPUT_SAMPLE_RATE as usize));
    let mix_ms = t0.elapsed().as_secs_f64() * 1000.0;

    for _ in 0..stepped {
        if reference.world.result().is_some() {
            break;
        }
        reference.step_sim();
    }
    let st = pcm_stats(&pcm);
    let pcm16: Vec<i16> = pcm.iter().map(|&v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16).collect();
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    std::fs::write(out, xact::wav_bytes_i16(OUTPUT_SAMPLE_RATE, 2, &pcm16))
        .map_err(|e| format!("write {}: {e}", out.display()))?;
    let ms = ga.engine.stats();

    println!(
        "render-mix: {} (ticks {window_start}..{} rendered, mixed in {mix_ms:.0} ms)",
        out.display(),
        stepped
    );
    println!(
        "levels:   duration {:.1}s | peak {:.3} ({:.1} dBFS) | rms {:.1} dBFS | active rms {:.1} dBFS | silent windows {:.0}% | clipped samples {}",
        st.seconds,
        st.peak,
        dbfs(st.peak),
        dbfs(st.rms),
        dbfs(st.active_rms),
        st.silent_fraction * 100.0,
        st.clipped
    );
    println!(
        "mixer:    triggered {} | cooldown-dropped {} | voice-limit-dropped {} | stolen {} | inaudible {} | peak voices {} | min limiter gain {:.2}",
        ms.triggered, ms.dropped_cooldown, ms.dropped_voice_limit, ms.stolen, ms.dropped_inaudible, ms.peak_voices, ms.min_limiter_gain
    );
    println!("cues played ({} distinct):", ga.played.len());
    let mut counts: Vec<(&String, &u32)> = ga.played.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (name, n) in counts {
        println!("  {n:>6}  {name}");
    }
    let (ha, hb) = (s.world.hash_hex(), reference.world.hash_hex());
    println!(
        "determinism: tick {} hash with audio {ha} vs without {hb} -> {}",
        s.world.tick(),
        if ha == hb { "identical" } else { "MISMATCH" }
    );
    if ha != hb {
        return Err("audio changed the simulation state hash".to_string());
    }
    Ok(())
}

/// `farust audio cues`: prints the semantic slot -> FA cue mapping and whether each
/// cue resolves in the install (plus per-unit-type weapon cue coverage).
pub fn print_cue_mapping(game_dir: &Path, catalog: Option<&Catalog>) {
    let Some(mut ga) = GameAudio::offline(game_dir) else {
        println!("no FA sound banks under {}", game_dir.join("sounds").display());
        return;
    };
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut add = |slot: &str, name: &str| rows.push((slot.to_string(), name.to_string()));
    for f in Fac::ALL {
        let fs = f.select_name();
        for (heavy, aa, slot) in [(false, false, "weapon"), (true, false, "weapon heavy"), (false, true, "weapon AA")] {
            add(&format!("{fs} {slot} (fallback)"), f.weapon(heavy, aa));
        }
        add(&format!("{fs} ACU death"), f.nuke_impact());
        for (st, tier, slot) in [(false, 1, "T1"), (false, 2, "T2/T3"), (false, 4, "T4"), (true, 1, "structure")] {
            add(&format!("{fs} {slot} destroyed"), f.destroy(st, tier));
        }
        let [b1, b2] = f.builder_fallbacks();
        add(&format!("{fs} construction start"), &b1);
        add(&format!("{fs} construction start (ACU)"), &b2);
        add(&format!("{fs} factory build start"), &format!("{}B0101_Construct_Start", f.prefix()));
        add(&format!("{fs} structure complete (fallback)"), &format!("{}B0101_Activate", f.prefix()));
        add(&format!("{fs} ACU warp-in"), f.gate_in());
        for cat in ["Commander", "Factory", "Resource", "Gun", "Radar", "Structure", "Vehicle", "Tank", "Bot", "Air", "Naval"] {
            add(&format!("{fs} select {cat}"), &format!("{fs}_Select_{cat}"));
        }
    }
    for (slot, name) in [
        ("impact (small)", "Impact_Land_Gen_UEF"),
        ("explosion (medium)", "Explosion_Medium"),
        ("explosion (large)", "Explosion_Large_01"),
        ("nuke impact", "UEF_Nuke_Impact"),
        ("shield hit", "Impact_Land_Gen_AEON"),
        ("tactical missile launch", "UEB2108_Missile_Cruise"),
        ("nuke launch", "UEB2305_Nuke_Fire"),
        ("missile intercepted", "Expl_Anti_Nuke"),
        ("upgrade complete", "GLB_Enhance_Stop"),
        ("order: move/patrol/rally", UI_ORDER_MOVE),
        ("order: attack/overcharge", UI_ORDER_ATTACK),
        ("order: build/repair/assist/reclaim", UI_ORDER_BUILD),
        ("order: produce", UI_ORDER_PRODUCE),
        ("order: stop", UI_ORDER_STOP),
        ("UI: build/attack-move mode", UI_MODE_TOGGLE),
        ("UI: pause/spectate/FOW toggle", UI_TOGGLE),
        ("UI: volume/unmute", UI_VOLUME),
        ("music: base building", MUSIC_BASE),
        ("music: battle", MUSIC_BATTLE),
        ("music: victory", "UI_END_Game_Victory"),
        ("music: defeat", "UI_END_Game_Fail"),
        ("ambient loop", AMBIENT_LOOP),
    ] {
        add(slot, name);
    }
    let mut missing = 0;
    for (slot, name) in &rows {
        let info = ga
            .engine
            .with_library_ref(|lib| lib.cue(name).map(|c| (c.soundbank.clone(), c.waves.len())))
            .flatten();
        match info {
            Some((bank, n)) if n > 0 => println!("  {slot:<38} {name:<28} {bank}.xsb, {n} wave(s)"),
            _ => {
                missing += 1;
                println!("  {slot:<38} {name:<28} MISSING");
            }
        }
    }
    println!("{} mapped cue slots, {missing} missing", rows.len());

    if let Some(cat) = catalog {
        let mut ids: Vec<String> = cat
            .factions
            .iter()
            .flat_map(|f| f.land_units.iter().cloned().chain([f.commander.clone()]))
            .collect();
        ids.sort();
        ids.dedup();
        let mut own = 0;
        for id in &ids {
            match ga.weapon_cue_for(id) {
                Some(n) => {
                    own += 1;
                    println!("  weapon {id:<10} -> {n}");
                }
                None => println!("  weapon {id:<10} -> faction fallback"),
            }
        }
        println!("{own}/{} roster unit types have their own FA weapon cue", ids.len());
    }
}

/// `farust audio coverage`: decodes every wave of every `.xwb` (recursively, incl.
/// `Voice/`) and tallies formats / decode results.
pub fn print_decode_coverage(sounds_dir: &Path) -> Result<(), String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            let mut v: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            v.sort();
            for p in v {
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().and_then(|s| s.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("xwb")) {
                    out.push(p);
                }
            }
        }
    }
    let mut files = Vec::new();
    walk(sounds_dir, &mut files);
    if files.is_empty() {
        return Err(format!("no .xwb files under {}", sounds_dir.display()));
    }
    // (format, bits, channels) -> (waves, decoded, failed, seconds)
    let mut tally: BTreeMap<(String, u16, u16), (usize, usize, usize, f64)> = BTreeMap::new();
    let mut silent = 0usize;
    let mut bank_errors = Vec::new();
    for p in &files {
        let wb = match WaveBank::open_lazy(p) {
            Ok(wb) => wb,
            Err(e) => {
                bank_errors.push(e);
                continue;
            }
        };
        for (i, e) in wb.entries.iter().enumerate() {
            let slot = tally
                .entry((xact::format_name(e.format_tag).to_string(), e.bits_per_sample, e.channels))
                .or_default();
            slot.0 += 1;
            match wb.read_entry(i, Some(p)).and_then(|raw| xact::decode_wave_data(e, &raw)) {
                Ok(clip) => {
                    slot.1 += 1;
                    slot.3 += clip.duration_secs();
                    if clip.samples.iter().all(|s| s.abs() < 1e-4) {
                        silent += 1;
                    }
                }
                Err(_) => slot.2 += 1,
            }
        }
    }
    println!("{}: {} wavebanks", sounds_dir.display(), files.len());
    let (mut tw, mut td) = (0, 0);
    for ((fmt, bits, ch), (n, ok, bad, secs)) in &tally {
        tw += n;
        td += ok;
        println!("  {fmt:<9} {bits:>2}-bit {ch}ch: {n:>5} waves, {ok:>5} decoded, {bad:>3} failed, {:.1} min audio", secs / 60.0);
    }
    println!("total: {td}/{tw} waves decoded ({silent} all-silent), {} unreadable banks", bank_errors.len());
    for e in &bank_errors {
        println!("  bank error: {e}");
    }
    Ok(())
}

/// `farust audio play-cue <CUE>`: realtime smoke test of the output sink.
pub fn play_cue_realtime(game_dir: &Path, cue: &str, muted: bool) -> Result<(), String> {
    let engine = AudioEngine::new(Some(game_dir), muted);
    let id = engine
        .resolve(cue, SPEC_UI)
        .ok_or_else(|| format!("cue '{cue}' not found / not decodable"))?;
    let t0 = std::time::Instant::now();
    let outcome = engine.trigger(id, None, 1.0);
    std::thread::sleep(std::time::Duration::from_millis(1500));
    println!(
        "play-cue {cue}: {outcome:?} | output sink {} | muted {} | {:.1}s",
        if engine.output_available() { "ok" } else { "unavailable (silent)" },
        engine.is_muted(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_factions_roles_and_orders_to_fa_cue_names() {
        assert_eq!(Fac::from_key("seraphim"), Fac::Seraphim);
        assert_eq!(Fac::from_key("cybran").weapon(true, false), "URL0202_Cannon_Dbl");
        assert_eq!(Fac::from_key("uef").destroy(false, 4), "UEL_Destroy_Huge");
        assert_eq!(select_category(Role::Mobile, NavDomain::Air, 1), "Air");
        assert_eq!(select_category(Role::Commander, NavDomain::Land, 1), "Commander");
        assert_eq!(order_cue(&Command::Stop { units: vec![] }), UI_ORDER_STOP);
        assert!(selection_rank(Role::Commander) > selection_rank(Role::Factory));
        let st = pcm_stats(&[0.5, -0.5, 0.0, 0.0]);
        assert!((st.peak - 0.5).abs() < 1e-6 && st.clipped == 0);
    }

    fn want(unit: UnitId, kind: LoopKind, cue: &str, score: f32) -> LoopWant {
        LoopWant { unit, kind, cue: cue.into(), pos: [unit as f32, 0.0, 0.0], score }
    }

    #[test]
    fn loop_book_starts_updates_and_stops_with_unit_state() {
        let mut book = LoopBook::default();
        // Unit 1 builds and moves (construction wins), unit 2 moves, unit 3 is far away.
        let p = book.plan(
            vec![
                want(1, LoopKind::Move, "A_Move_Loop", -10.0),
                want(1, LoopKind::Construct, "A_Construct_Loop", -10.0),
                want(2, LoopKind::Move, "B_Move_Loop", -20.0),
                want(3, LoopKind::Move, "C_Move_Loop", -200.0),
            ],
            2,
            |_| true,
        );
        let started: Vec<(UnitId, LoopKind)> = p.start.iter().map(|w| (w.unit, w.kind)).collect();
        assert_eq!(started, vec![(1, LoopKind::Construct), (2, LoopKind::Move)]);
        assert!(p.stop.is_empty() && p.update.is_empty());
        for (i, w) in p.start.iter().enumerate() {
            book.started(w, 100 + i as u64);
        }
        // Same state: positions update, nothing restarts.
        let p = book.plan(
            vec![want(1, LoopKind::Construct, "A_Construct_Loop", -10.0), want(2, LoopKind::Move, "B_Move_Loop", -20.0)],
            2,
            |_| true,
        );
        assert_eq!(p.update.len(), 2);
        assert!(p.start.is_empty() && p.stop.is_empty());
        // Unit 1 finished building and now moves: construct loop stops, move loop starts.
        // Unit 2 stopped moving: its loop stops. A closer unit 3 does not steal while
        // within the keep bonus of an existing loop.
        let p = book.plan(
            vec![want(1, LoopKind::Move, "A_Move_Loop", -10.0), want(3, LoopKind::Move, "C_Move_Loop", -5.0)],
            2,
            |_| true,
        );
        let mut stops = p.stop.clone();
        stops.sort();
        assert_eq!(stops, vec![100, 101]);
        assert_eq!(p.start.iter().map(|w| w.unit).collect::<Vec<_>>(), vec![3, 1]);
        for (i, w) in p.start.iter().enumerate() {
            book.started(w, 200 + i as u64);
        }
        assert_eq!(book.len(), 2);
        // A stolen voice is forgotten and restarted on the next plan.
        let p = book.plan(vec![want(3, LoopKind::Move, "C_Move_Loop", -5.0)], 2, |v| v != 200);
        assert_eq!(p.start.len(), 1);
        assert_eq!(p.stop, vec![201]);
        // Limit respected; clear stops the rest.
        assert!(book.len() <= 2);
        book.started(&p.start[0], 300);
        assert_eq!(book.clear(), vec![300]);
        assert_eq!(book.len(), 0);
    }
}
