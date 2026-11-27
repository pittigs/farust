//! Phase 14 & Phase 43 game-side audio glue.
//!
//! Maps a [`PlaySession`] onto Forged Alliance XACT cues (names taken from the original
//! `.xsb` SoundBanks, including `Voice/<lang>/XGG.xsb`):
//! - Sim events (weapons, impacts, explosions, missiles, ACU death) & unit lifecycle diffs
//! - Phase 24 layered XACT cues and looping unit construction/movement sounds (`LoopBook`)
//! - Phase 43 tactical & economy announcer (`AnnouncerQueue`: per-event cooldowns, priority
//!   queue, single-voice playback, render-side detection for the local focus army & shared
//!   allied intel), data-driven per-map biome ambient loops (`MapAmbience`), and smoothed
//!   combat-intensity adaptive music (`MusicDirector`: hysteresis + crossfade + ducking).
//!
//! Audio only *reads* sim state (plus draining the audio copy of the event queue), so it
//! can never influence simulation determinism.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::camera::OrbitCamera;
use crate::flowfield::NavDomain;
use crate::play::{self, PlaySession};
use crate::scmap::ScMap;
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{
    ArmyId, Catalog, Command, GameResult, IntelVisibility, Role, SimEvent, Skirmish, UnitId,
};
use crate::xact::{
    self, AudioEngine, Bus, CueId, CueLibrary, CueSpec, Listener, OUTPUT_SAMPLE_RATE,
    TriggerOutcome, WaveBank,
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
/// Phase 43: announcer callout voice spec (high priority so SFX never steals it; queue
/// enforces single-instance playback and per-event cooldowns).
const SPEC_ANNOUNCER: CueSpec = CueSpec {
    bus: Bus::Voice,
    volume: 1.05,
    cooldown_secs: 0.0,
    max_instances: 1,
    priority: 50.0,
    replace_oldest: false,
};
/// Phase 49: scripted campaign dialogue voice cues (`say "..." voice CUE`); highest voice
/// priority, queued sequentially on `Bus::Voice`, purely render-side.
const SPEC_DIALOGUE: CueSpec = CueSpec {
    bus: Bus::Voice,
    volume: 1.10,
    cooldown_secs: 0.0,
    max_instances: 1,
    priority: 80.0,
    replace_oldest: false,
};
/// Phase 24: looping construction / movement sounds (no cooldown; capped by the book).
const SPEC_LOOP: CueSpec = CueSpec::sfx(0.32, 0.0, 8, 0.9);
/// Unit loops audible at once (below [`xact::MAX_LOOP_VOICES`] so one-shots keep room).
const LOOP_LIMIT: usize = 6;
/// Loops only for units within this camera-eye distance (world units).
const LOOP_RANGE: f32 = 260.0;

const MUSIC_BASE: &str = "Base_Building";
const MUSIC_BATTLE: &str = "Battle";
const AMBIENT_RUMBLE: &str = "AMB_Planet_Rumble_zoom";
const AMBIENT_VOLCANIC: &str = "Gen_Fire_Loop";
const AMBIENT_TUNDRA: &str = "AMB_SER_OP_Briefing";
const AMBIENT_WATER: &str = "UEL0203_Move_Water_Lp";
const AMBIENT_ARID: &str = "Rumble1";

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
        Role::Shield | Role::MissileSilo | Role::MissileDefense | Role::AntiAir | Role::Wall | Role::AirStaging => "Structure",
        Role::EngStation => "Factory",
        Role::Orbital => "Structure",
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
        Command::Load { .. } | Command::Unload { .. } | Command::Dock { .. } => UI_ORDER_MOVE,
        Command::SetAutoRefuel { .. } | Command::SetFireState { .. } | Command::SetBuildPaused { .. } => UI_ORDER_STOP,
        Command::AttackGround { .. } => UI_ORDER_ATTACK,
        Command::Guard { .. } | Command::MoveWaypoint { .. } | Command::Ferry { .. } | Command::Ping { .. } => UI_ORDER_MOVE,
        Command::Enhance { .. } => UI_ORDER_BUILD,
        Command::SetRepeat { .. } | Command::CancelProduce { .. } => UI_ORDER_PRODUCE,
    }
}

// ---------------------------------------------------------------------------
// Phase 43: Announcer events, real FA XGG cue mapping, cooldowns & priority queue
// ---------------------------------------------------------------------------

/// Tactical & economy announcer events derived render-side from `Skirmish`.
///
/// Standalone Forged Alliance ships skirmish computer/HQ callouts in `Voice/<lang>/XGG.xsb`
/// (`Computer_*`, `XGG_Computer_CV01_*`, `XGG_HQ_GD1_*`). Vanilla SupCom 1's separate
/// `COMPUTER_<FAC>_VO` banks (which held "Additional Mass/Energy storage required") are
/// absent in standalone FA; those two events try the authentic `vo_computer.lua` cue names
/// when installed and report `MISSING` otherwise rather than inventing names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AnnouncerEvent {
    StrategicLaunchDetected,
    EnemyNukeInbound,
    CommanderUnderAttack,
    ExperimentalDetected,
    StrategicMissileDestroyed,
    BaseUnderAttack,
    StructureLost,
    ShieldDown,
    EnergyStorageEmpty,
    MassStorageEmpty,
    EnergyStorageLow,
    MassStorageLow,
    UnitCapReached,
    TransportFull,
    MassStorageFull,
    EnergyStorageFull,
}

const EXP_DETECTED_VARIANTS: &[&str] = &[
    "XGG_HQ_GD1_04179", // Fatboy (UEL0401)
    "XGG_HQ_GD1_04180", // Atlantis (UES0401)
    "XGG_HQ_GD1_04181", // Mavor (UEB2401)
    "XGG_HQ_GD1_04182", // Monkeylord (URL0402)
    "XGG_HQ_GD1_04183", // Soul Ripper (URA0401)
    "XGG_HQ_GD1_04184", // Scathis (URL0401)
    "XGG_HQ_GD1_04185", // Colossus (UAL0401)
    "XGG_HQ_GD1_04186", // CZAR (UAA0310)
    "XGG_HQ_GD1_04187", // Tempest (UAS0401)
    "XGG_HQ_GD1_04188", // Ythotha (XSL0401)
    "XGG_HQ_GD1_04189", // Ahwassa (XSA0402)
    "XGG_HQ_GD1_04190", // Yolona Oss (XSB2401)
];

const BASE_UNDER_ATTACK_VARIANTS: &[&str] = &[
    "XGG_HQ_GD1_04151", // T1 ground units inbound
    "XGG_HQ_GD1_04152", // T2 ground units inbound
    "XGG_HQ_GD1_04153", // T3 ground units inbound
    "XGG_HQ_GD1_04142", // Large land force east
    "XGG_HQ_GD1_04143", // Large land force west
    "XGG_HQ_GD1_04144", // Large land force north
    "XGG_HQ_GD1_04145", // Large land force south
    "XGG_HQ_GD1_04155", // Enemy air south
    "XGG_HQ_GD1_04156", // Enemy air north
    "XGG_HQ_GD1_04157", // Enemy air east
    "XGG_HQ_GD1_04158", // Enemy air west
    "XGG_HQ_GD1_04169", // Enemy fleet north
    "XGG_HQ_GD1_04170", // Enemy fleet east
    "XGG_HQ_GD1_04171", // Enemy fleet south
    "XGG_HQ_GD1_04172", // Enemy fleet west
];

const STRUCTURE_LOST_VARIANTS: &[&str] = &[
    "XGG_HQ_GD1_04214", // Factory lost
    "XGG_HQ_GD1_04215", // Power generator lost
    "XGG_HQ_GD1_04216", // Mass extractor lost
    "XGG_HQ_GD1_04218", // Radar lost
    "XGG_HQ_GD1_04219", // Mass storage lost
    "XGG_HQ_GD1_04220", // Energy storage lost
    "XGG_HQ_GD1_04221", // Shield lost
    "XGG_HQ_GD1_04213", // Artillery / defense lost
];

impl AnnouncerEvent {
    pub const ALL: [AnnouncerEvent; 16] = [
        AnnouncerEvent::StrategicLaunchDetected,
        AnnouncerEvent::EnemyNukeInbound,
        AnnouncerEvent::CommanderUnderAttack,
        AnnouncerEvent::ExperimentalDetected,
        AnnouncerEvent::StrategicMissileDestroyed,
        AnnouncerEvent::BaseUnderAttack,
        AnnouncerEvent::StructureLost,
        AnnouncerEvent::ShieldDown,
        AnnouncerEvent::EnergyStorageEmpty,
        AnnouncerEvent::MassStorageEmpty,
        AnnouncerEvent::EnergyStorageLow,
        AnnouncerEvent::MassStorageLow,
        AnnouncerEvent::UnitCapReached,
        AnnouncerEvent::TransportFull,
        AnnouncerEvent::MassStorageFull,
        AnnouncerEvent::EnergyStorageFull,
    ];

    /// Phase 39: localized on-screen caption of the voice line.
    pub fn caption(self) -> String {
        let fallback = match self {
            AnnouncerEvent::StrategicLaunchDetected => "Strategic launch detected",
            AnnouncerEvent::EnemyNukeInbound => "Nuclear missile inbound",
            AnnouncerEvent::CommanderUnderAttack => "Commander under attack",
            AnnouncerEvent::ExperimentalDetected => "Enemy experimental detected",
            AnnouncerEvent::StrategicMissileDestroyed => "Strategic missile destroyed",
            AnnouncerEvent::BaseUnderAttack => "Base under attack",
            AnnouncerEvent::StructureLost => "Structure lost",
            AnnouncerEvent::ShieldDown => "Shield down",
            AnnouncerEvent::EnergyStorageEmpty => "Energy depleted",
            AnnouncerEvent::MassStorageEmpty => "Mass depleted",
            AnnouncerEvent::EnergyStorageLow => "Low energy",
            AnnouncerEvent::MassStorageLow => "Low mass",
            AnnouncerEvent::UnitCapReached => "Unit cap reached",
            AnnouncerEvent::TransportFull => "Transport full",
            AnnouncerEvent::MassStorageFull => "Mass storage full",
            AnnouncerEvent::EnergyStorageFull => "Energy storage full",
        };
        format!("[{}]", crate::loc::tr(&format!("caption.{}", self.slug()), fallback))
    }

    pub const fn slug(self) -> &'static str {
        match self {
            AnnouncerEvent::StrategicLaunchDetected => "strategic_launch_detected",
            AnnouncerEvent::EnemyNukeInbound => "enemy_nuke_inbound",
            AnnouncerEvent::CommanderUnderAttack => "commander_under_attack",
            AnnouncerEvent::ExperimentalDetected => "experimental_detected",
            AnnouncerEvent::StrategicMissileDestroyed => "strategic_missile_destroyed",
            AnnouncerEvent::BaseUnderAttack => "base_under_attack",
            AnnouncerEvent::StructureLost => "structure_lost",
            AnnouncerEvent::ShieldDown => "shield_down",
            AnnouncerEvent::EnergyStorageEmpty => "energy_storage_empty",
            AnnouncerEvent::MassStorageEmpty => "mass_storage_empty",
            AnnouncerEvent::EnergyStorageLow => "energy_storage_low",
            AnnouncerEvent::MassStorageLow => "mass_storage_low",
            AnnouncerEvent::UnitCapReached => "unit_cap_reached",
            AnnouncerEvent::TransportFull => "transport_full",
            AnnouncerEvent::MassStorageFull => "mass_storage_full",
            AnnouncerEvent::EnergyStorageFull => "energy_storage_full",
        }
    }

    /// Higher urgency wins when several events wait for the voice channel.
    pub const fn priority(self) -> u8 {
        match self {
            AnnouncerEvent::StrategicLaunchDetected => 100,
            AnnouncerEvent::EnemyNukeInbound => 98,
            AnnouncerEvent::CommanderUnderAttack => 95,
            AnnouncerEvent::ExperimentalDetected => 85,
            AnnouncerEvent::StrategicMissileDestroyed => 80,
            AnnouncerEvent::BaseUnderAttack => 75,
            AnnouncerEvent::StructureLost => 70,
            AnnouncerEvent::ShieldDown => 60,
            AnnouncerEvent::EnergyStorageEmpty => 58,
            AnnouncerEvent::MassStorageEmpty => 55,
            AnnouncerEvent::EnergyStorageLow => 48,
            AnnouncerEvent::MassStorageLow => 45,
            AnnouncerEvent::UnitCapReached => 40,
            AnnouncerEvent::TransportFull => 38,
            AnnouncerEvent::MassStorageFull => 35,
            AnnouncerEvent::EnergyStorageFull => 30,
        }
    }

    /// Per-event minimum replay interval in seconds (mirrors FA `VOReplayTime` ~15–60 s).
    pub const fn cooldown_secs(self) -> f64 {
        match self {
            AnnouncerEvent::StrategicLaunchDetected => 10.0,
            AnnouncerEvent::EnemyNukeInbound => 20.0,
            AnnouncerEvent::CommanderUnderAttack => 20.0,
            AnnouncerEvent::ExperimentalDetected => 45.0,
            AnnouncerEvent::StrategicMissileDestroyed => 10.0,
            AnnouncerEvent::BaseUnderAttack => 30.0,
            AnnouncerEvent::StructureLost => 25.0,
            AnnouncerEvent::ShieldDown => 25.0,
            AnnouncerEvent::EnergyStorageEmpty => 40.0,
            AnnouncerEvent::MassStorageEmpty => 40.0,
            AnnouncerEvent::EnergyStorageLow => 40.0,
            AnnouncerEvent::MassStorageLow => 40.0,
            AnnouncerEvent::UnitCapReached => 60.0,
            AnnouncerEvent::TransportFull => 20.0,
            AnnouncerEvent::MassStorageFull => 60.0,
            AnnouncerEvent::EnergyStorageFull => 60.0,
        }
    }

    /// Primary candidate cue(s) in FA's `XGG.xsb` (or SupCom 1's `COMPUTER_*_VO` if merged).
    pub const fn default_cues(self) -> &'static [&'static str] {
        match self {
            AnnouncerEvent::StrategicLaunchDetected => &["Computer_Computer_MissileLaunch_01351"],
            AnnouncerEvent::EnemyNukeInbound => &["Computer_Computer_MissileLaunch_01358"],
            AnnouncerEvent::CommanderUnderAttack => &["Computer_Computer_Commanders_01314"],
            AnnouncerEvent::ExperimentalDetected => &["XGG_HQ_GD1_04179"],
            AnnouncerEvent::StrategicMissileDestroyed => &["Computer_Computer_MissileLaunch_01356"],
            AnnouncerEvent::BaseUnderAttack => &["XGG_HQ_GD1_04151"],
            AnnouncerEvent::StructureLost => &["XGG_HQ_GD1_04214"],
            AnnouncerEvent::ShieldDown => &["XGG_Computer_CV01_04768"],
            AnnouncerEvent::EnergyStorageEmpty => &["XGG_Computer_CV01_04777", "XGG_Computer_CV01_04776"],
            AnnouncerEvent::MassStorageEmpty => &["XGG_Computer_CV01_04773", "XGG_Computer_CV01_04772"],
            AnnouncerEvent::EnergyStorageLow => &["XGG_Computer_CV01_04775", "XGG_Computer_CV01_04774"],
            AnnouncerEvent::MassStorageLow => &["XGG_Computer_CV01_04771", "XGG_Computer_CV01_04770"],
            AnnouncerEvent::UnitCapReached => &["Computer_Computer_CommandCap_01298"],
            AnnouncerEvent::TransportFull => &["Computer_TransportIsFull", "XGG_Computer_CV01_04743"],
            AnnouncerEvent::MassStorageFull => &[
                "UEFComputer_Resources_01313",
                "AeonComputer_Resources_01313",
                "CybranComputer_Resources_01313",
            ],
            AnnouncerEvent::EnergyStorageFull => &[
                "UEFComputer_Resources_01312",
                "AeonComputer_Resources_01312",
                "CybranComputer_Resources_01312",
            ],
        }
    }

    /// Contextual cue pool for events that choose unit/domain/role-specific callouts.
    pub const fn variant_cues(self) -> &'static [&'static str] {
        match self {
            AnnouncerEvent::ExperimentalDetected => EXP_DETECTED_VARIANTS,
            AnnouncerEvent::BaseUnderAttack => BASE_UNDER_ATTACK_VARIANTS,
            AnnouncerEvent::StructureLost => STRUCTURE_LOST_VARIANTS,
            _ => &[],
        }
    }
}

/// Specific `XGG_HQ_GD1_*` experimental-sighted line for `type_id` (falls back by faction).
pub fn experimental_sighted_cue(type_id: &str) -> &'static str {
    let up = type_id.to_ascii_uppercase();
    match up.as_str() {
        "UEL0401" => "XGG_HQ_GD1_04179", // Fatboy
        "UES0401" => "XGG_HQ_GD1_04180", // Atlantis
        "UEB2401" => "XGG_HQ_GD1_04181", // Mavor
        "URL0402" => "XGG_HQ_GD1_04182", // Monkeylord
        "URA0401" => "XGG_HQ_GD1_04183", // Soul Ripper
        "URL0401" => "XGG_HQ_GD1_04184", // Scathis
        "UAL0401" => "XGG_HQ_GD1_04185", // Colossus
        "UAA0310" => "XGG_HQ_GD1_04186", // CZAR
        "UAS0401" => "XGG_HQ_GD1_04187", // Tempest
        "XSL0401" => "XGG_HQ_GD1_04188", // Ythotha
        "XSA0402" => "XGG_HQ_GD1_04189", // Ahwassa
        "XSB2401" => "XGG_HQ_GD1_04190", // Yolona Oss
        _ if up.starts_with("UR") || up.starts_with("XR") => "XGG_HQ_GD1_04182",
        _ if up.starts_with("UA") || up.starts_with("XA") => "XGG_HQ_GD1_04185",
        _ if up.starts_with("XS") => "XGG_HQ_GD1_04188",
        _ => "XGG_HQ_GD1_04179",
    }
}

/// Contextual `XGG_HQ_GD1_*` inbound-attack cues from attacker domain, tech tier, and
/// offset from the defending army's base `(dx_east, dz_south)`.
pub fn base_under_attack_cues(domain: NavDomain, tier: u8, dx_east: i64, dz_south: i64) -> [&'static str; 2] {
    let east_west = dx_east.abs() >= dz_south.abs();
    match domain {
        NavDomain::Air => {
            let dir = if east_west {
                if dx_east >= 0 { "XGG_HQ_GD1_04157" } else { "XGG_HQ_GD1_04158" }
            } else if dz_south >= 0 {
                "XGG_HQ_GD1_04155"
            } else {
                "XGG_HQ_GD1_04156"
            };
            [dir, "XGG_HQ_GD1_04151"]
        }
        NavDomain::Naval => {
            let dir = if east_west {
                if dx_east >= 0 { "XGG_HQ_GD1_04170" } else { "XGG_HQ_GD1_04172" }
            } else if dz_south >= 0 {
                "XGG_HQ_GD1_04171"
            } else {
                "XGG_HQ_GD1_04169"
            };
            [dir, "XGG_HQ_GD1_04151"]
        }
        NavDomain::Land | NavDomain::Amphibious => {
            let by_tier = match tier {
                3.. => "XGG_HQ_GD1_04153",
                2 => "XGG_HQ_GD1_04152",
                _ => "XGG_HQ_GD1_04151",
            };
            let by_dir = if east_west {
                if dx_east >= 0 { "XGG_HQ_GD1_04142" } else { "XGG_HQ_GD1_04143" }
            } else if dz_south >= 0 {
                "XGG_HQ_GD1_04145"
            } else {
                "XGG_HQ_GD1_04144"
            };
            [by_tier, by_dir]
        }
    }
}

/// Specific `XGG_HQ_GD1_*` structure-lost cue for `role`.
pub fn structure_lost_cue(role: Role) -> Option<&'static str> {
    match role {
        Role::Factory | Role::AirFactory | Role::NavalFactory => Some("XGG_HQ_GD1_04214"),
        Role::Generator | Role::Hydro => Some("XGG_HQ_GD1_04215"),
        Role::Extractor => Some("XGG_HQ_GD1_04216"),
        Role::Radar => Some("XGG_HQ_GD1_04218"),
        Role::MassStorage => Some("XGG_HQ_GD1_04219"),
        Role::EnergyStorage => Some("XGG_HQ_GD1_04220"),
        Role::Shield => Some("XGG_HQ_GD1_04221"),
        Role::PointDefense | Role::AntiAir | Role::MissileSilo | Role::MissileDefense => {
            Some("XGG_HQ_GD1_04213")
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct PendingCallout {
    event: AnnouncerEvent,
    candidates: Vec<String>,
    enqueued_at: f64,
    seq: u64,
}

/// Pure priority queue + per-event cooldown gate ensuring at most one announcer callout
/// speaks at a time. Game-data-independent so it can be unit-tested directly.
#[derive(Debug, Clone, Default)]
pub struct AnnouncerQueue {
    pending: Vec<PendingCallout>,
    last_played: HashMap<AnnouncerEvent, f64>,
    last_voice_end: Option<f64>,
    was_voice_active: bool,
    next_seq: u64,
    pub played_count: u32,
}

impl AnnouncerQueue {
    pub const MAX_PENDING: usize = 6;
    pub const MAX_QUEUE_AGE_SECS: f64 = 12.0;
    pub const MIN_GAP_SECS: f64 = 0.35;

    /// Queues `event` if not on cooldown; coalesces duplicate pending events.
    pub fn enqueue<S: Into<String>>(
        &mut self,
        event: AnnouncerEvent,
        candidates: impl IntoIterator<Item = S>,
        now_secs: f64,
    ) -> bool {
        if let Some(&last) = self.last_played.get(&event)
            && now_secs - last < event.cooldown_secs()
        {
            return false;
        }
        let cands: Vec<String> = candidates.into_iter().map(Into::into).collect();
        if cands.is_empty() {
            return false;
        }
        if let Some(existing) = self.pending.iter_mut().find(|p| p.event == event) {
            existing.candidates = cands;
            return false;
        }
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.pending.push(PendingCallout {
            event,
            candidates: cands,
            enqueued_at: now_secs,
            seq,
        });
        if self.pending.len() > Self::MAX_PENDING
            && let Some((drop_idx, _)) = self
                .pending
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| (p.event.priority(), Reverse(p.seq)))
        {
            self.pending.remove(drop_idx);
        }
        true
    }

    /// Drops pending callouts without resetting cooldown history (used on replay seek).
    pub fn clear_pending(&mut self) {
        self.pending.clear();
        self.was_voice_active = false;
    }

    #[cfg(test)]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Advances the queue at `now_secs`. While `voice_active` is true (or within the
    /// post-line breath gap), nothing new starts. Otherwise pops highest-priority non-stale
    /// callouts until `try_play` succeeds (unresolvable `MISSING` events still record their
    /// cooldown and immediately yield to the next queued callout).
    pub fn poll<T>(
        &mut self,
        now_secs: f64,
        voice_active: bool,
        mut try_play: impl FnMut(AnnouncerEvent, &[String]) -> Option<T>,
    ) -> Option<(AnnouncerEvent, T)> {
        if self.was_voice_active && !voice_active {
            self.last_voice_end = Some(now_secs);
        }
        self.was_voice_active = voice_active;

        let last_played = &self.last_played;
        self.pending.retain(|p| {
            now_secs - p.enqueued_at <= Self::MAX_QUEUE_AGE_SECS
                && !last_played
                    .get(&p.event)
                    .is_some_and(|&t| now_secs - t < p.event.cooldown_secs())
        });

        if voice_active {
            return None;
        }
        if let Some(end) = self.last_voice_end
            && now_secs - end < Self::MIN_GAP_SECS
        {
            return None;
        }

        while !self.pending.is_empty() {
            let best_idx = self
                .pending
                .iter()
                .enumerate()
                .max_by_key(|(_, p)| (p.event.priority(), Reverse(p.seq)))
                .map(|(i, _)| i)?;
            let item = self.pending.remove(best_idx);
            self.last_played.insert(item.event, now_secs);
            if let Some(tok) = try_play(item.event, &item.candidates) {
                self.was_voice_active = true;
                self.played_count += 1;
                return Some((item.event, tok));
            }
        }
        None
    }
}

/// Edge-triggered economy stockpile watcher with hold timers and hysteresis bands so
/// initial full storage at tick 0 or momentary flickers never spam announcer alerts.
#[derive(Debug, Clone, Default)]
pub struct EconWatch {
    mass_dipped: bool,
    energy_dipped: bool,
    mass_empty_hold: f32,
    mass_low_hold: f32,
    mass_full_hold: f32,
    energy_empty_hold: f32,
    energy_low_hold: f32,
    energy_full_hold: f32,
    mass_empty_latched: bool,
    mass_low_latched: bool,
    mass_full_latched: bool,
    energy_empty_latched: bool,
    energy_low_latched: bool,
    energy_full_latched: bool,
}

impl EconWatch {
    const GRACE_TICKS: u64 = 200; // 20 s opening build grace period
    const EMPTY_HOLD_SECS: f32 = 1.5;
    const LOW_HOLD_SECS: f32 = 2.0;
    const FULL_HOLD_SECS: f32 = 4.0;

    /// Resets latches to match an existing world snapshot (e.g. after replay seek).
    pub fn seed(&mut self, mass: i64, mass_cap: i64, energy: i64, energy_cap: i64) {
        *self = Self::default();
        let mr = mass as f32 / mass_cap.max(1) as f32;
        let er = energy as f32 / energy_cap.max(1) as f32;
        self.mass_dipped = mr < 0.90;
        self.energy_dipped = er < 0.90;
        self.mass_empty_latched = mass <= SCALE;
        self.mass_low_latched = mr <= 0.10;
        self.mass_full_latched = mr >= 0.99;
        self.energy_empty_latched = energy <= 5 * SCALE;
        self.energy_low_latched = er <= 0.10;
        self.energy_full_latched = er >= 0.99;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        tick: u64,
        dt: f32,
        mass: i64,
        mass_cap: i64,
        income_mass: i64,
        energy: i64,
        energy_cap: i64,
        income_energy: i64,
    ) -> Vec<AnnouncerEvent> {
        let mut out = Vec::new();
        if mass_cap <= 0 || energy_cap <= 0 || dt <= 0.0 {
            return out;
        }
        let mr = mass as f32 / mass_cap as f32;
        let er = energy as f32 / energy_cap as f32;
        if mr < 0.90 {
            self.mass_dipped = true;
        }
        if er < 0.90 {
            self.energy_dipped = true;
        }
        // Release latches past hysteresis bands.
        if mr > 0.08 {
            self.mass_empty_latched = false;
        }
        if mr > 0.22 {
            self.mass_low_latched = false;
        }
        if mr < 0.85 {
            self.mass_full_latched = false;
        }
        if er > 0.08 {
            self.energy_empty_latched = false;
        }
        if er > 0.22 {
            self.energy_low_latched = false;
        }
        if er < 0.85 {
            self.energy_full_latched = false;
        }
        if tick < Self::GRACE_TICKS {
            return out;
        }

        // Mass empty / low / full
        if mass <= SCALE {
            self.mass_empty_hold += dt;
            self.mass_low_hold = 0.0;
            if !self.mass_empty_latched && self.mass_empty_hold >= Self::EMPTY_HOLD_SECS {
                self.mass_empty_latched = true;
                self.mass_low_latched = true;
                out.push(AnnouncerEvent::MassStorageEmpty);
            }
        } else {
            self.mass_empty_hold = 0.0;
            if mr <= 0.10 {
                self.mass_low_hold += dt;
                if !self.mass_low_latched && self.mass_low_hold >= Self::LOW_HOLD_SECS {
                    self.mass_low_latched = true;
                    out.push(AnnouncerEvent::MassStorageLow);
                }
            } else {
                self.mass_low_hold = 0.0;
            }
        }
        if self.mass_dipped && mr >= 0.99 && income_mass > 0 {
            self.mass_full_hold += dt;
            if !self.mass_full_latched && self.mass_full_hold >= Self::FULL_HOLD_SECS {
                self.mass_full_latched = true;
                out.push(AnnouncerEvent::MassStorageFull);
            }
        } else {
            self.mass_full_hold = 0.0;
        }

        // Energy empty / low / full
        if energy <= 5 * SCALE {
            self.energy_empty_hold += dt;
            self.energy_low_hold = 0.0;
            if !self.energy_empty_latched && self.energy_empty_hold >= Self::EMPTY_HOLD_SECS {
                self.energy_empty_latched = true;
                self.energy_low_latched = true;
                out.push(AnnouncerEvent::EnergyStorageEmpty);
            }
        } else {
            self.energy_empty_hold = 0.0;
            if er <= 0.10 {
                self.energy_low_hold += dt;
                if !self.energy_low_latched && self.energy_low_hold >= Self::LOW_HOLD_SECS {
                    self.energy_low_latched = true;
                    out.push(AnnouncerEvent::EnergyStorageLow);
                }
            } else {
                self.energy_low_hold = 0.0;
            }
        }
        if self.energy_dipped && er >= 0.99 && income_energy > 0 && tick >= 450 {
            self.energy_full_hold += dt;
            if !self.energy_full_latched && self.energy_full_hold >= Self::FULL_HOLD_SECS {
                self.energy_full_latched = true;
                out.push(AnnouncerEvent::EnergyStorageFull);
            }
        } else {
            self.energy_full_hold = 0.0;
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Phase 43: Data-driven per-map ambient loops (`MapBiome` / `MapAmbience`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapBiome {
    Temperate,
    Oceanic,
    Volcanic,
    Arid,
    Tundra,
}

impl MapBiome {
    pub const ALL: [MapBiome; 5] = [
        MapBiome::Temperate,
        MapBiome::Oceanic,
        MapBiome::Volcanic,
        MapBiome::Arid,
        MapBiome::Tundra,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            MapBiome::Temperate => "temperate",
            MapBiome::Oceanic => "oceanic",
            MapBiome::Volcanic => "volcanic",
            MapBiome::Arid => "arid",
            MapBiome::Tundra => "tundra",
        }
    }

    pub const fn cues(self, has_coastal_water: bool) -> (&'static str, Option<&'static str>) {
        match self {
            MapBiome::Temperate => (
                AMBIENT_RUMBLE,
                if has_coastal_water { Some(AMBIENT_WATER) } else { None },
            ),
            MapBiome::Oceanic => (AMBIENT_RUMBLE, Some(AMBIENT_WATER)),
            MapBiome::Volcanic => (AMBIENT_RUMBLE, Some(AMBIENT_VOLCANIC)),
            MapBiome::Arid => (AMBIENT_RUMBLE, Some(AMBIENT_ARID)),
            MapBiome::Tundra => (AMBIENT_TUNDRA, Some(AMBIENT_RUMBLE)),
        }
    }
}

/// Map-specific dual-layer ambient loop configuration derived from `.scmap` terrain
/// strata, skybox and heightmap water coverage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapAmbience {
    pub biome: MapBiome,
    pub water_fraction: f32,
    pub primary_cue: &'static str,
    pub detail_cue: Option<&'static str>,
}

impl Default for MapAmbience {
    fn default() -> Self {
        Self {
            biome: MapBiome::Temperate,
            water_fraction: 0.0,
            primary_cue: AMBIENT_RUMBLE,
            detail_cue: None,
        }
    }
}

impl MapAmbience {
    /// Classifies `scmap` into a [`MapBiome`] from its `.scmap` stratum paths and water
    /// coverage, picking a primary planetary loop + optional surface/orbital detail loop.
    pub fn from_scmap(scmap: &ScMap) -> Self {
        let water_fraction = if scmap.water.has_water && !scmap.heightmap.is_empty() {
            let stride = (scmap.heightmap.len() / 4096).max(1);
            let (mut wet, mut total) = (0usize, 0usize);
            for &h in scmap.heightmap.iter().step_by(stride) {
                total += 1;
                if (h as f32) * scmap.height_scale < scmap.water.elevation {
                    wet += 1;
                }
            }
            wet as f32 / total.max(1) as f32
        } else {
            0.0
        };

        let (mut volcanic, mut tundra, mut arid, mut lush) = (0u32, 0u32, 0u32, 0u32);
        if let Some(ref tt) = scmap.terrain_textures {
            for (i, s) in tt.strata_albedo.iter().enumerate() {
                let p = s.path.to_ascii_lowercase();
                if p.is_empty() || p.contains("macrotexture") || p.contains("/common/") {
                    continue;
                }
                let w = if i < 5 { 2 } else { 1 };
                if p.contains("/lava/") {
                    volcanic += w * 2;
                } else if p.contains("/crystalline/") || p.contains("/tundra/") || p.contains("snow") || p.contains("ice") {
                    tundra += w;
                } else if p.contains("/desert/") || p.contains("/redrocks/") || p.contains("/red/") {
                    arid += w;
                } else if p.contains("/evergreen") || p.contains("/tropical") || p.contains("/swamp") || p.contains("/paradise") {
                    lush += w;
                }
            }
        }
        let sky = scmap.sky_cubemap_path.to_ascii_lowercase();
        if sky.contains("lava") || sky.contains("brimstone") {
            volcanic += 2;
        } else if sky.contains("white") || sky.contains("ice") || sky.contains("tundra") {
            tundra += 1;
        } else if sky.contains("desert") || sky.contains("red") {
            arid += 1;
        }

        let biome = if volcanic > 0 && volcanic >= lush && volcanic >= arid {
            MapBiome::Volcanic
        } else if water_fraction >= 0.25 {
            MapBiome::Oceanic
        } else if tundra > 0 && tundra >= lush && tundra >= arid {
            MapBiome::Tundra
        } else if arid > 0 && arid > lush {
            MapBiome::Arid
        } else {
            MapBiome::Temperate
        };

        let (primary_cue, detail_cue) = biome.cues(water_fraction >= 0.04);
        Self {
            biome,
            water_fraction,
            primary_cue,
            detail_cue,
        }
    }

    /// `(primary_vol, detail_vol)` for `cam_distance` (world units from camera target).
    pub fn gains(&self, cam_distance: f32) -> (f32, f32) {
        let zoom_out = ((cam_distance - 50.0) / 380.0).clamp(0.0, 1.0);
        let near_ground = (1.0 - (cam_distance - 40.0) / 300.0).clamp(0.0, 1.0);
        match self.biome {
            MapBiome::Tundra => (0.35 + 0.25 * near_ground, 0.12 + 0.78 * zoom_out),
            MapBiome::Volcanic => (0.18 + 0.75 * zoom_out, 0.08 + 0.42 * near_ground),
            MapBiome::Oceanic => {
                let wf = self.water_fraction.clamp(0.25, 1.0);
                (0.18 + 0.75 * zoom_out, (0.08 + 0.36 * near_ground) * wf.sqrt())
            }
            MapBiome::Arid => (0.18 + 0.78 * zoom_out, 0.06 + 0.28 * near_ground),
            MapBiome::Temperate => (
                0.15 + 0.80 * zoom_out,
                if self.detail_cue.is_some() { 0.05 + 0.24 * near_ground } else { 0.0 },
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 43: Smoothed combat-intensity adaptive music (`MusicDirector`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MusicMode {
    Peace,
    Battle,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MusicTransition {
    pub mode: MusicMode,
    pub cue: &'static str,
    pub variant: usize,
    pub fade_out_secs: f32,
    pub fade_in_secs: f32,
}

/// Smoothed combat-volume tracker driving `Base_Building` ↔ `Battle` playlist crossfades
/// with Schmitt-trigger hysteresis, minimum battle dwell and calm hold timers.
#[derive(Debug, Clone)]
pub struct MusicDirector {
    mode: MusicMode,
    smoothed_rate: f32,
    pending_impulse: f32,
    calm_secs: f32,
    mode_dwell_secs: f32,
    peace_track: usize,
    battle_track: usize,
    pub peak_intensity: f32,
    pub transitions: u32,
}

impl Default for MusicDirector {
    fn default() -> Self {
        Self::new()
    }
}

impl MusicDirector {
    /// Combat impulse/sec corresponding to `intensity() == 1.0`.
    pub const FULL_BATTLE_RATE: f32 = 2.4;
    /// Upward hysteresis threshold entering `Battle`.
    pub const ENTER_BATTLE_INTENSITY: f32 = 0.45;
    /// Downward hysteresis threshold starting the return-to-peace hold timer.
    pub const EXIT_BATTLE_INTENSITY: f32 = 0.15;
    /// Seconds intensity must stay at/below [`Self::EXIT_BATTLE_INTENSITY`] before leaving `Battle`
    /// (mirrors FA `UserMusic.lua` `PeaceTimer = 20s`).
    pub const PEACE_HOLD_SECS: f32 = 18.0;
    /// Minimum seconds spent in `Battle` before a return to `Base_Building` is allowed.
    pub const MIN_BATTLE_DWELL_SECS: f32 = 10.0;
    const ATTACK_TAU: f32 = 2.5;
    const RELEASE_TAU: f32 = 8.0;

    pub fn new() -> Self {
        Self {
            mode: MusicMode::Peace,
            smoothed_rate: 0.0,
            pending_impulse: 0.0,
            calm_secs: 0.0,
            mode_dwell_secs: 0.0,
            peace_track: 0,
            battle_track: 0,
            peak_intensity: 0.0,
            transitions: 0,
        }
    }

    pub fn mode(&self) -> MusicMode {
        self.mode
    }

    /// Smoothed combat intensity in `0.0..=1.0`.
    pub fn intensity(&self) -> f32 {
        (self.smoothed_rate / Self::FULL_BATTLE_RATE).clamp(0.0, 1.0)
    }

    /// Adds unpositioned combat impulse (e.g. strategic nuke / ACU death).
    pub fn add_combat(&mut self, amount: f32) {
        self.pending_impulse += amount.max(0.0);
    }

    /// Adds world-space combat impulse weighted by camera proximity and focus-army involvement.
    pub fn add_spatial_combat(
        &mut self,
        amount: f32,
        pos_xz: [f32; 2],
        cam_xz: [f32; 2],
        focus_involved: bool,
    ) {
        let d = ((pos_xz[0] - cam_xz[0]).powi(2) + (pos_xz[1] - cam_xz[1]).powi(2)).sqrt();
        let near_cam = (1.0 - (d - 110.0).max(0.0) / 260.0).clamp(0.25, 1.0);
        let w = if focus_involved { near_cam.max(0.75) } else { near_cam };
        self.pending_impulse += amount.max(0.0) * w;
    }

    /// Music stream volume, swelling slightly in intense combat and ducking under announcer VO.
    pub fn target_volume(&self, announcer_speaking: bool) -> f32 {
        let base = match self.mode {
            MusicMode::Peace => 0.92,
            MusicMode::Battle => 0.88 + 0.12 * self.intensity(),
        };
        if announcer_speaking { base * 0.62 } else { base }
    }

    /// Steps smoothing and playlist state. When `sim_dt == 0.0` (paused), combat state freezes
    /// and only a finished/unstarted stream triggers a track start.
    pub fn step(&mut self, sim_dt: f32, real_dt: f32, music_live: bool) -> Option<MusicTransition> {
        let impulse = std::mem::take(&mut self.pending_impulse);
        if sim_dt > 0.0 {
            let inst_rate = impulse / sim_dt.max(0.05);
            let dt = real_dt.max(sim_dt.min(0.25));
            let tau = if inst_rate > self.smoothed_rate {
                Self::ATTACK_TAU
            } else {
                Self::RELEASE_TAU
            };
            let alpha = 1.0 - (-dt / tau).exp();
            self.smoothed_rate += (inst_rate - self.smoothed_rate) * alpha;
            let intensity = self.intensity();
            self.peak_intensity = self.peak_intensity.max(intensity);
            self.mode_dwell_secs += dt;
            if intensity <= Self::EXIT_BATTLE_INTENSITY {
                self.calm_secs += dt;
            } else {
                self.calm_secs = 0.0;
            }
        }

        let intensity = self.intensity();
        let want_mode = match self.mode {
            MusicMode::Peace => {
                if intensity >= Self::ENTER_BATTLE_INTENSITY {
                    MusicMode::Battle
                } else {
                    MusicMode::Peace
                }
            }
            MusicMode::Battle => {
                if intensity <= Self::EXIT_BATTLE_INTENSITY
                    && self.calm_secs >= Self::PEACE_HOLD_SECS
                    && self.mode_dwell_secs >= Self::MIN_BATTLE_DWELL_SECS
                {
                    MusicMode::Peace
                } else {
                    MusicMode::Battle
                }
            }
        };

        if want_mode == self.mode && music_live {
            return None;
        }
        let mode_changed = want_mode != self.mode;
        self.mode = want_mode;
        if mode_changed {
            self.mode_dwell_secs = 0.0;
            self.calm_secs = 0.0;
            self.transitions += 1;
        }
        let (cue, variant, fade_out_secs, fade_in_secs) = match want_mode {
            MusicMode::Battle => {
                let v = self.battle_track;
                self.battle_track = self.battle_track.wrapping_add(1);
                (MUSIC_BATTLE, v, 1.8, 1.2)
            }
            MusicMode::Peace => {
                let v = self.peace_track;
                self.peace_track = self.peace_track.wrapping_add(1);
                (
                    MUSIC_BASE,
                    v,
                    if mode_changed { 3.0 } else { 2.0 },
                    if mode_changed { 2.5 } else { 2.0 },
                )
            }
        };
        Some(MusicTransition {
            mode: want_mode,
            cue,
            variant,
            fade_out_secs,
            fade_in_secs,
        })
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
    hp: i64,
    shield_hp: i64,
    transport_full: bool,
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
        wants.sort_by_key(|a| (a.unit, a.kind));
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
    audio_clock: f64,
    selection: Vec<UnitId>,
    ui: Option<UiFlags>,
    pending_seen: usize,
    music_dir: MusicDirector,
    music: Option<u64>,
    music_retry: u32,
    ambience: MapAmbience,
    ambient_primary: Option<u64>,
    ambient_detail: Option<u64>,
    ambient_retry: u32,
    announcer: AnnouncerQueue,
    announcer_voice: Option<u64>,
    /// Phase 49: index into `MissionRuntime::voices` already enqueued and sequential VO queue.
    mission_voices_seen: usize,
    dialogue_queue: std::collections::VecDeque<String>,
    dialogue_voice: Option<u64>,
    econ_watch: EconWatch,
    seen_experimentals: HashSet<UnitId>,
    enemy_nukes: HashMap<u32, bool>,
    at_cap_prev: bool,
    result_seen: bool,
    /// Phase 24: running construction / movement loops.
    loops: LoopBook,
    /// Played-cue histogram (render-mix log / cue mapping verification).
    pub played: BTreeMap<String, u32>,
    /// Announcer events actually voiced during this session.
    pub announced: BTreeMap<AnnouncerEvent, u32>,
    /// Phase 39: captions of announcer lines played since the last frame.
    pub captions: Vec<String>,
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
            audio_clock: 0.0,
            selection: Vec::new(),
            ui: None,
            pending_seen: 0,
            music_dir: MusicDirector::new(),
            music: None,
            music_retry: 0,
            ambience: MapAmbience::default(),
            ambient_primary: None,
            ambient_detail: None,
            ambient_retry: 0,
            announcer: AnnouncerQueue::default(),
            announcer_voice: None,
            mission_voices_seen: 0,
            dialogue_queue: std::collections::VecDeque::new(),
            dialogue_voice: None,
            econ_watch: EconWatch::default(),
            seen_experimentals: HashSet::new(),
            enemy_nukes: HashMap::new(),
            at_cap_prev: false,
            result_seen: false,
            loops: LoopBook::default(),
            played: BTreeMap::new(),
            announced: BTreeMap::new(),
            captions: Vec::new(),
        })
    }

    /// Configures map-specific ambient loops from `.scmap` biome & water coverage.
    pub fn set_map(&mut self, scmap: &ScMap) {
        let next = MapAmbience::from_scmap(scmap);
        if next != self.ambience {
            if let Some(id) = self.ambient_primary.take() {
                self.engine.fade_out(id, 1.0);
            }
            if let Some(id) = self.ambient_detail.take() {
                self.engine.fade_out(id, 1.0);
            }
            self.ambience = next;
            self.ambient_retry = 0;
        }
    }

    fn focus_army(s: &PlaySession) -> ArmyId {
        s.ui.replay
            .as_ref()
            .and_then(|r| r.perspective)
            .unwrap_or(s.human_army)
    }

    fn is_transport_full(world: &Skirmish, u: &crate::skirmish::Unit) -> bool {
        if u.cargo.is_empty() {
            return false;
        }
        let Some(spec) = world.catalog().get(&u.type_id).and_then(|t| t.ext.transport.as_ref()) else {
            return false;
        };
        let classes = world.cargo_classes(u.id);
        !spec.fits(&classes, 1)
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

    /// Adopts the current world state without playing anything (headless fast-forward / seek).
    pub fn prime(&mut self, s: &mut PlaySession) {
        let _ = s.world.drain_audio_events();
        for v in self.loops.clear() {
            self.engine.fade_out(v, 0.1);
        }
        self.announcer.clear_pending();
        self.dialogue_queue.clear();
        self.mission_voices_seen = s.world.mission().map_or(0, |rt| rt.voices.len());
        let focus = Self::focus_army(s);
        self.stamp = self.stamp.wrapping_add(1);
        self.units.clear();
        self.seen_experimentals.clear();
        for u in s.world.units().filter(|u| u.alive) {
            self.units.insert(
                u.id,
                Tracked {
                    army: u.army,
                    role: u.role,
                    tier: u.tier,
                    building: u.build.is_some(),
                    hp: u.hp,
                    shield_hp: u.shield_hp,
                    transport_full: Self::is_transport_full(&s.world, u),
                    pos: [0.0; 3],
                    stamp: self.stamp,
                },
            );
            if (u.role == Role::Experimental || u.tier >= 4)
                && !s.world.are_allies(focus, u.army)
                && s.world.intel_visibility(focus, u) != IntelVisibility::Hidden
            {
                self.seen_experimentals.insert(u.id);
            }
        }
        self.enemy_nukes = s
            .world
            .projectiles()
            .filter(|p| p.is_nuke && !s.world.are_allies(focus, p.army))
            .map(|p| (p.id, p.ticks_lived * 2 >= p.total_ticks.max(1)))
            .collect();
        self.at_cap_prev = s.world.at_unit_cap(focus);
        if let Some(a) = s.world.army(focus) {
            self.econ_watch.seed(a.mass, a.mass_cap, a.energy, a.energy_cap);
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
        let mut load_onto_full = false;
        let mut ack: Option<&'static str> = None;
        for (army, cmd) in &pending[self.pending_seen..] {
            if *army == s.human_army {
                if ack.is_none() {
                    ack = Some(order_cue(cmd));
                }
                if let Command::Load { transport, .. } = cmd
                    && let Some(tu) = s.world.unit(*transport)
                    && Self::is_transport_full(&s.world, tu)
                {
                    load_onto_full = true;
                }
            }
        }
        self.pending_seen = pending.len();
        if let Some(cue) = ack {
            self.play(cue, SPEC_UI, None, 1.0);
        }
        if load_onto_full {
            self.announcer.enqueue(
                AnnouncerEvent::TransportFull,
                AnnouncerEvent::TransportFull.default_cues().iter().copied(),
                self.audio_clock,
            );
        }
    }

    /// Call once per frame *after* stepping: updates the listener from `cam`, turns the
    /// audio event stream and unit lifecycle changes into cues, runs the announcer, and
    /// drives adaptive music + map ambience.
    pub fn after_step(&mut self, s: &mut PlaySession, cam: &OrbitCamera) {
        self.engine.set_listener(Listener {
            view_proj: cam.view_proj_matrix(),
            eye: cam.eye_position(),
        });
        let tick = s.world.tick();
        let dt = tick.saturating_sub(self.last_tick) as f32 / TICK_HZ as f32;
        let real_dt = dt / s.speed.max(1) as f32;
        if self.primed && tick == self.last_tick {
            // No sim progress since last frame (paused / input-only redraw): skip the scan.
            self.pump_mission_dialogue(&s.world);
            self.pump_announcer();
            self.update_music_and_ambience(0.0, 0.0, cam);
            return;
        }
        if self.primed && (tick < self.last_tick || tick - self.last_tick > 40) {
            // Replay seek / fast-forward: resync silently instead of replaying a backlog.
            self.prime(s);
            self.update_music_and_ambience(0.0, 0.0, cam);
            return;
        }
        self.audio_clock += f64::from(real_dt);
        let focus = Self::focus_army(s);
        let ground = cam.target[1];
        let cam_xz = [cam.target[0], cam.target[2]];
        let scale = SCALE as f32;
        let to_world = |x: i64, y: i64, alt: i64| [x as f32 / scale, ground + alt as f32 / scale, y as f32 / scale];

        // ---- unit lifecycle & announcer damage/intel diff --------------------------
        self.stamp = self.stamp.wrapping_add(1);
        let stamp = self.stamp;
        let primed = self.primed;
        let last_tick = self.last_tick;
        let mut shooters: Vec<(i64, i64, ArmyId, String)> = Vec::new();
        let mut starts: Vec<(String, ArmyId, Role, [f32; 3])> = Vec::new();
        let mut births: Vec<(String, ArmyId, Role, [f32; 3])> = Vec::new();
        let mut commander_hit = false;
        let mut shield_collapsed = false;
        let mut base_hit_pos: Option<(i64, i64)> = None;
        let mut sighted_exp: Option<String> = None;
        let mut transport_became_full = false;

        for u in s.world.units() {
            if !u.alive {
                continue;
            }
            let p = to_world(u.pos.x, u.pos.y, u.altitude);
            if primed && u.last_shot.is_some_and(|(_, t)| t >= last_tick) {
                shooters.push((u.pos.x, u.pos.y, u.army, u.type_id.clone()));
            }
            let building = u.build.is_some();
            let tfull = if u.army == focus { Self::is_transport_full(&s.world, u) } else { false };
            match self.units.get_mut(&u.id) {
                Some(t) => {
                    if building && !t.building {
                        starts.push((u.type_id.clone(), u.army, u.role, p));
                    }
                    if primed && u.army == focus {
                        if u.role == Role::Commander && u.hp < t.hp {
                            commander_hit = true;
                        } else if u.role.is_structure() && u.role != Role::Wall && u.hp < t.hp {
                            base_hit_pos.get_or_insert((u.pos.x, u.pos.y));
                        }
                        if u.shield_max_hp > 0 && t.shield_hp > 0 && u.shield_hp == 0 {
                            shield_collapsed = true;
                        }
                        if tfull && !t.transport_full {
                            transport_became_full = true;
                        }
                    }
                    t.building = building;
                    t.hp = u.hp;
                    t.shield_hp = u.shield_hp;
                    t.transport_full = tfull;
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
                            hp: u.hp,
                            shield_hp: u.shield_hp,
                            transport_full: tfull,
                            pos: p,
                            stamp,
                        },
                    );
                }
            }

            if primed
                && (u.role == Role::Experimental || u.tier >= 4)
                && !s.world.are_allies(focus, u.army)
                && !self.seen_experimentals.contains(&u.id)
                && s.world.intel_visibility(focus, u) != IntelVisibility::Hidden
            {
                self.seen_experimentals.insert(u.id);
                sighted_exp.get_or_insert_with(|| u.type_id.clone());
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
        let mut intercepted_any = false;
        if primed {
            for ev in &events {
                if matches!(ev, SimEvent::MissileIntercepted { .. }) {
                    intercepted_any = true;
                }
                self.on_event(&s.world, focus, cam_xz, ev, &shooters, &to_world);
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
                // Initial commanders warp in; also seed initial economy state cleanly.
                let names = [format!("{upper}_Gate_In"), fac.gate_in().to_string()];
                self.play_first(&names, SPEC_GATE_IN, Some(p), 1.0);
            } else if role.is_structure() {
                let names = [format!("{upper}_Activate"), format!("{}B0101_Activate", fac.prefix())];
                self.play_first(&names, SPEC_ACTIVATE, Some(p), 1.0);
            } else {
                self.play(&format!("{upper}_Move_Start"), SPEC_ROLLOUT, Some(p), 1.0);
            }
        }
        let mut lost_structure_role: Option<Role> = None;
        for d in deaths {
            if d.role == Role::Commander {
                continue; // SimEvent::AcuDied plays the nuke-scale blast.
            }
            let fac = Self::faction(&s.world, d.army);
            let cue = fac.destroy(d.role.is_structure(), d.tier);
            self.play(cue, SPEC_DEATH, Some(d.pos), 1.0);
            let death_weight = match d.tier {
                4.. => 2.2,
                3 => 0.9,
                2 => 0.55,
                _ => 0.35,
            };
            self.music_dir.add_spatial_combat(
                death_weight,
                [d.pos[0], d.pos[2]],
                cam_xz,
                s.world.are_allies(focus, d.army),
            );
            if primed && d.army == focus && d.role.is_structure() && d.role != Role::Wall {
                lost_structure_role.get_or_insert(d.role);
                base_hit_pos.get_or_insert(((d.pos[0] * scale) as i64, (d.pos[2] * scale) as i64));
            }
        }

        if !primed
            && let Some(a) = s.world.army(focus)
        {
            self.econ_watch.seed(a.mass, a.mass_cap, a.energy, a.energy_cap);
        }

        // ---- Phase 49 mission dialogue & Phase 43 announcer (render-side only) ------
        self.pump_mission_dialogue(&s.world);
        if primed && s.world.result().is_none() {
            self.detect_announcer_events(
                &s.world,
                focus,
                dt,
                commander_hit,
                shield_collapsed,
                base_hit_pos,
                lost_structure_role,
                sighted_exp.as_deref(),
                transport_became_full,
                intercepted_any,
            );
            self.pump_announcer();
        }

        self.update_loops(&s.world, cam, &to_world);

        // ---- game end, adaptive music, map ambience --------------------------------
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
        self.update_music_and_ambience(dt, real_dt, cam);

        self.last_tick = tick;
        self.primed = true;
    }

    #[allow(clippy::too_many_arguments)]
    fn detect_announcer_events(
        &mut self,
        world: &Skirmish,
        focus: ArmyId,
        sim_dt: f32,
        commander_hit: bool,
        shield_collapsed: bool,
        base_hit_pos: Option<(i64, i64)>,
        lost_structure_role: Option<Role>,
        sighted_exp: Option<&str>,
        transport_became_full: bool,
        intercepted_any: bool,
    ) {
        let now = self.audio_clock;

        // 1. Strategic missile launch / inbound / intercepted
        let mut live_nukes: HashMap<u32, bool> = HashMap::new();
        for p in world.projectiles() {
            if !p.is_nuke || world.are_allies(focus, p.army) {
                continue;
            }
            let prev_inbound = self.enemy_nukes.get(&p.id).copied();
            if prev_inbound.is_none() {
                self.announcer.enqueue(
                    AnnouncerEvent::StrategicLaunchDetected,
                    AnnouncerEvent::StrategicLaunchDetected.default_cues().iter().copied(),
                    now,
                );
            }
            let halfway = p.ticks_lived * 2 >= p.total_ticks.max(1);
            if halfway && prev_inbound == Some(false) {
                self.announcer.enqueue(
                    AnnouncerEvent::EnemyNukeInbound,
                    AnnouncerEvent::EnemyNukeInbound.default_cues().iter().copied(),
                    now,
                );
            }
            live_nukes.insert(p.id, prev_inbound.unwrap_or(false) || halfway);
        }
        if intercepted_any && self.enemy_nukes.keys().any(|id| !live_nukes.contains_key(id)) {
            self.announcer.enqueue(
                AnnouncerEvent::StrategicMissileDestroyed,
                AnnouncerEvent::StrategicMissileDestroyed.default_cues().iter().copied(),
                now,
            );
        }
        self.enemy_nukes = live_nukes;

        // 2. Commander under attack
        if commander_hit {
            self.announcer.enqueue(
                AnnouncerEvent::CommanderUnderAttack,
                AnnouncerEvent::CommanderUnderAttack.default_cues().iter().copied(),
                now,
            );
        }

        // 3. Enemy experimental detected (vision / radar / allied shared intel)
        if let Some(exp_type) = sighted_exp {
            let specific = experimental_sighted_cue(exp_type);
            self.announcer.enqueue(
                AnnouncerEvent::ExperimentalDetected,
                [specific, "XGG_HQ_GD1_04179"],
                now,
            );
        }

        // 4. Base under attack (contextual domain / tech / compass callout)
        if let Some((bx, by)) = base_hit_pos {
            let base_origin = world.army(focus).map(|a| a.start).unwrap_or((bx, by));
            let near_enemy = world
                .units()
                .filter(|u| u.alive && !world.are_allies(focus, u.army))
                .min_by_key(|u| (u.pos.x - bx).abs() + (u.pos.y - by).abs())
                .filter(|u| (u.pos.x - bx).abs() <= 70 * SCALE && (u.pos.y - by).abs() <= 70 * SCALE);
            let (dom, tier, ax, ay) = near_enemy
                .map(|u| (u.domain, u.tier, u.pos.x, u.pos.y))
                .unwrap_or((NavDomain::Land, 1, bx, by));
            let cues = base_under_attack_cues(dom, tier, ax - base_origin.0, ay - base_origin.1);
            self.announcer.enqueue(AnnouncerEvent::BaseUnderAttack, cues, now);
        }

        // 5. Key base structure lost
        if let Some(role) = lost_structure_role
            && let Some(cue) = structure_lost_cue(role)
        {
            self.announcer.enqueue(AnnouncerEvent::StructureLost, [cue, "XGG_HQ_GD1_04214"], now);
        }

        // 6. Shield collapsed
        if shield_collapsed {
            self.announcer.enqueue(
                AnnouncerEvent::ShieldDown,
                AnnouncerEvent::ShieldDown.default_cues().iter().copied(),
                now,
            );
        }

        // 7. Transport full & unit cap reached
        if transport_became_full {
            self.announcer.enqueue(
                AnnouncerEvent::TransportFull,
                AnnouncerEvent::TransportFull.default_cues().iter().copied(),
                now,
            );
        }
        let at_cap = world.at_unit_cap(focus);
        if at_cap && !self.at_cap_prev {
            self.announcer.enqueue(
                AnnouncerEvent::UnitCapReached,
                AnnouncerEvent::UnitCapReached.default_cues().iter().copied(),
                now,
            );
        }
        self.at_cap_prev = at_cap;

        // 8. Economy storage alerts (focus army only)
        if let Some(a) = world.army(focus)
            && !a.defeated
        {
            for ev in self.econ_watch.observe(
                world.tick(),
                sim_dt,
                a.mass,
                a.mass_cap,
                a.income_mass,
                a.energy,
                a.energy_cap,
                a.income_energy,
            ) {
                self.announcer.enqueue(ev, ev.default_cues().iter().copied(), now);
            }
        }
    }

    fn pump_mission_dialogue(&mut self, world: &Skirmish) {
        if let Some(rt) = world.mission() {
            if rt.voices.len() < self.mission_voices_seen {
                self.mission_voices_seen = rt.voices.len();
                self.dialogue_queue.clear();
            }
            for (_, cue) in &rt.voices[self.mission_voices_seen..] {
                if self.dialogue_queue.len() >= 8 {
                    self.dialogue_queue.pop_front();
                }
                self.dialogue_queue.push_back(cue.clone());
            }
            self.mission_voices_seen = rt.voices.len();
        } else {
            self.mission_voices_seen = 0;
        }
        let dlg_active = self.dialogue_voice.is_some_and(|v| self.engine.voice_active(v));
        if !dlg_active {
            self.dialogue_voice = None;
        }
        let ann_active = self.announcer_voice.is_some_and(|v| self.engine.voice_active(v));
        if dlg_active || ann_active {
            return;
        }
        while let Some(name) = self.dialogue_queue.pop_front() {
            let id = match self.cues.get(name.as_str()) {
                Some(c) => *c,
                None => {
                    let resolved = self.engine.resolve(&name, SPEC_DIALOGUE);
                    self.cues.insert(name.clone(), resolved);
                    resolved
                }
            };
            if let Some(cid) = id
                && let TriggerOutcome::Played(vid) = self.engine.trigger(cid, None, 1.0)
            {
                *self.played.entry(name).or_insert(0) += 1;
                self.dialogue_voice = Some(vid);
                break;
            }
        }
    }

    fn pump_announcer(&mut self) {
        let dlg_active = self.dialogue_voice.is_some_and(|v| self.engine.voice_active(v));
        let voice_active =
            dlg_active || self.announcer_voice.is_some_and(|v| self.engine.voice_active(v));
        if !self.announcer_voice.is_some_and(|v| self.engine.voice_active(v)) {
            self.announcer_voice = None;
        }
        let now = self.audio_clock;
        let engine = self.engine.clone();
        let cues_cache = &mut self.cues;
        let played = &mut self.played;
        let outcome = self.announcer.poll(now, voice_active, |_ev, candidates| {
            for name in candidates {
                let id = match cues_cache.get(name.as_str()) {
                    Some(c) => *c,
                    None => {
                        let resolved = engine.resolve(name, SPEC_ANNOUNCER);
                        cues_cache.insert(name.clone(), resolved);
                        resolved
                    }
                };
                if let Some(cid) = id
                    && let TriggerOutcome::Played(vid) = engine.trigger(cid, None, 1.0)
                {
                    *played.entry(name.clone()).or_insert(0) += 1;
                    return Some(vid);
                }
            }
            None
        });
        if let Some((ev, vid)) = outcome {
            self.announcer_voice = Some(vid);
            *self.announced.entry(ev).or_insert(0) += 1;
            self.captions.push(ev.caption());
        }
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
        focus: ArmyId,
        cam_xz: [f32; 2],
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
                let wpos = to_world(from.0, from.1, from.2);
                match own {
                    Some(n) => self.play_first(&[n.as_str(), fallback], SPEC_WEAPON, Some(wpos), 1.0),
                    None => self.play_first(&[fallback], SPEC_WEAPON, Some(wpos), 1.0),
                }
                self.music_dir.add_spatial_combat(
                    if *heavy { 0.11 } else { 0.05 },
                    [wpos[0], wpos[2]],
                    cam_xz,
                    world.are_allies(focus, *army),
                );
            }
            SimEvent::Explosion { pos, radius, is_nuke } => {
                let wpos = to_world(pos.0, pos.1, pos.2);
                let p = Some(wpos);
                if *is_nuke {
                    self.play("UEF_Nuke_Impact", SPEC_NUKE, p, 1.0);
                    self.music_dir.add_combat(6.0);
                } else if *radius <= 5 * SCALE / 2 {
                    self.play("Impact_Land_Gen_UEF", SPEC_IMPACT, p, 1.0);
                    self.music_dir.add_spatial_combat(0.06, [wpos[0], wpos[2]], cam_xz, false);
                } else if *radius < 6 * SCALE {
                    self.play("Explosion_Medium", SPEC_EXPLOSION, p, 1.0);
                    self.music_dir.add_spatial_combat(0.20, [wpos[0], wpos[2]], cam_xz, false);
                } else {
                    self.play("Explosion_Large_01", SPEC_EXPLOSION_LARGE, p, 1.0);
                    self.music_dir.add_spatial_combat(0.50, [wpos[0], wpos[2]], cam_xz, false);
                }
            }
            SimEvent::AcuDied { pos, army } => {
                let cue = Self::faction(world, *army).nuke_impact();
                self.play(cue, SPEC_NUKE, Some(to_world(pos.0, pos.1, pos.2)), 1.6);
                self.music_dir.add_combat(8.0);
            }
            SimEvent::ShieldHit { pos, army } => {
                let wpos = to_world(pos.0, pos.1, pos.2);
                self.play("Impact_Land_Gen_AEON", SPEC_SHIELD, Some(wpos), 1.0);
                self.music_dir.add_spatial_combat(
                    0.06,
                    [wpos[0], wpos[2]],
                    cam_xz,
                    world.are_allies(focus, *army),
                );
            }
            SimEvent::MissileLaunched { pos, is_nuke } => {
                let wpos = to_world(pos.0, pos.1, pos.2);
                let cue = if *is_nuke { "UEB2305_Nuke_Fire" } else { "UEB2108_Missile_Cruise" };
                self.play(cue, SPEC_MISSILE, Some(wpos), 1.0);
                if *is_nuke {
                    self.music_dir.add_combat(5.0);
                } else {
                    self.music_dir.add_spatial_combat(0.45, [wpos[0], wpos[2]], cam_xz, false);
                }
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

    /// Drives `Base_Building` ↔ `Battle` music crossfades (with announcer ducking) and
    /// per-map dual-layer ambient loops scaled by camera zoom.
    fn update_music_and_ambience(&mut self, sim_dt: f32, real_dt: f32, cam: &OrbitCamera) {
        self.music_retry = self.music_retry.saturating_sub(1);
        self.ambient_retry = self.ambient_retry.saturating_sub(1);

        let announcer_speaking = self.announcer_voice.is_some_and(|v| self.engine.voice_active(v))
            || self.dialogue_voice.is_some_and(|v| self.engine.voice_active(v));
        let music_live = self.music.is_some_and(|id| self.engine.voice_active(id));
        if !self.result_seen {
            if let Some(tr) = self.music_dir.step(sim_dt, real_dt, music_live) {
                if self.music_retry == 0 {
                    if let Some(id) = self.music.take() {
                        self.engine.fade_out(id, tr.fade_out_secs);
                    }
                    let vol = self.music_dir.target_volume(announcer_speaking);
                    self.music = self.engine.play_stream(tr.cue, tr.variant, Bus::Music, vol, false, tr.fade_in_secs);
                    if self.music.is_some() {
                        *self.played.entry(tr.cue.to_string()).or_insert(0) += 1;
                    } else {
                        self.music_retry = 300;
                    }
                }
            } else if let Some(id) = self.music {
                self.engine.set_voice_volume(id, self.music_dir.target_volume(announcer_speaking));
            }
        }

        let (pri_vol, det_vol) = self.ambience.gains(cam.distance);
        let pri_cue = self.ambience.primary_cue;
        match self.ambient_primary {
            Some(id) if self.engine.voice_active(id) => self.engine.set_voice_volume(id, pri_vol),
            _ if self.ambient_retry == 0 => {
                self.ambient_primary = self.engine.play_stream(pri_cue, 0, Bus::Ambient, pri_vol, true, 2.5);
                if self.ambient_primary.is_some() {
                    *self.played.entry(pri_cue.to_string()).or_insert(0) += 1;
                } else {
                    self.ambient_retry = 300;
                }
            }
            _ => {}
        }
        if let Some(det_cue) = self.ambience.detail_cue {
            match self.ambient_detail {
                Some(id) if self.engine.voice_active(id) => self.engine.set_voice_volume(id, det_vol),
                _ if self.ambient_retry == 0 => {
                    self.ambient_detail = self.engine.play_stream(det_cue, 0, Bus::Ambient, det_vol, true, 2.5);
                    if self.ambient_detail.is_some() {
                        *self.played.entry(det_cue.to_string()).or_insert(0) += 1;
                    }
                }
                _ => {}
            }
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
/// call (events, unit lifecycle, announcer, adaptive music, map ambience), then selection /
/// order / UI changes made by input.
pub fn frame(s: &mut PlaySession, scmap: Option<&ScMap>, game_dir: Option<&Path>, cam: &OrbitCamera) {
    let mut slot = std::mem::take(&mut s.audio);
    if let SessionAudio::Uninit = slot {
        slot = match GameAudio::new(game_dir, s.muted) {
            Some(mut ga) => {
                if let Some(m) = scmap {
                    ga.set_map(m);
                }
                SessionAudio::On(Box::new(ga))
            }
            None => SessionAudio::Off,
        };
    }
    if let SessionAudio::On(ga) = &mut slot {
        if let Some(m) = scmap {
            ga.set_map(m);
        }
        ga.after_step(s, cam);
        ga.before_step(s);
        let caps = std::mem::take(&mut ga.captions);
        if crate::settings::subtitles()
            && let Some(c) = caps.last()
        {
            s.status_banner = c.clone();
        }
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
    ga.set_map(scmap);
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
    println!(
        "ambience: biome {} (water {:.0}%, primary={}, detail={}) | music: mode={:?} peak_intensity={:.2} transitions={}",
        ga.ambience.biome.label(),
        ga.ambience.water_fraction * 100.0,
        ga.ambience.primary_cue,
        ga.ambience.detail_cue.unwrap_or("-"),
        ga.music_dir.mode(),
        ga.music_dir.peak_intensity,
        ga.music_dir.transitions,
    );
    let ann_summary: Vec<String> = ga
        .announced
        .iter()
        .map(|(ev, n)| format!("{}x{n}", ev.slug()))
        .collect();
    println!(
        "announcer: {} callout(s) ({})",
        ga.announcer.played_count,
        if ann_summary.is_empty() { "none".to_string() } else { ann_summary.join(", ") }
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

/// Resolves a candidate cue or `"Bank#Index"` wave ref against `lib` for coverage reporting.
fn resolve_cue_entry(lib: &mut CueLibrary, name: &str) -> Option<(String, usize)> {
    if let Some(c) = lib.cue(name)
        && !c.waves.is_empty()
    {
        return Some((format!("{}.xsb", c.soundbank), c.waves.len()));
    }
    if let Some((wb, idx)) = xact::parse_wave_ref(name)
        && let Some((bname, _)) = lib.wave_entry(wb, idx)
    {
        return Some((format!("{bname}.xwb#{idx}"), 1));
    }
    None
}

/// `farust audio cues [--all]`: prints the semantic slot -> FA cue mapping and whether each
/// cue resolves in the install (plus per-unit-type weapon cue coverage, or every indexed
/// cue across all SoundBanks when `--all` is given).
pub fn print_cue_mapping(game_dir: &Path, catalog: Option<&Catalog>, all: bool) {
    let Some(mut ga) = GameAudio::offline(game_dir) else {
        println!("no FA sound banks under {}", game_dir.join("sounds").display());
        return;
    };
    if all {
        let mut list: Vec<(String, String, usize)> = ga
            .engine
            .with_library_ref(|lib| {
                lib.cues()
                    .map(|c| (c.soundbank.clone(), c.name.clone(), c.waves.len()))
                    .collect()
            })
            .unwrap_or_default();
        list.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        println!("all indexed cues ({}):", list.len());
        for (bank, name, n) in &list {
            println!("  {bank:<18} {name:<42} {n} wave(s)");
        }
        return;
    }

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
        ("ambient: planetary rumble", AMBIENT_RUMBLE),
        ("ambient: volcanic fire loop", AMBIENT_VOLCANIC),
        ("ambient: tundra/alien pad", AMBIENT_TUNDRA),
        ("ambient: surface water loop", AMBIENT_WATER),
        ("ambient: arid rumble", AMBIENT_ARID),
    ] {
        add(slot, name);
    }
    for ev in AnnouncerEvent::ALL {
        add(&format!("announcer: {}", ev.slug()), ev.default_cues()[0]);
    }
    let mut missing = 0;
    for (slot, name) in &rows {
        let info = ga
            .engine
            .with_library_ref(|lib| lib.cue(name).map(|c| (c.soundbank.clone(), c.waves.len())))
            .flatten();
        match info {
            Some((bank, n)) if n > 0 => println!("  {slot:<38} {name:<38} {bank}.xsb, {n} wave(s)"),
            _ => {
                missing += 1;
                println!("  {slot:<38} {name:<38} MISSING");
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
/// `Voice/`), tallies formats / decode results, and reports Phase 43 announcer, biome
/// ambience, and adaptive music cue coverage.
pub fn print_decode_coverage(sounds_dir: &Path) -> Result<(), String> {
    let files: Vec<PathBuf> = xact::collect_sound_paths(sounds_dir)
        .into_iter()
        .filter(|p| p.extension().and_then(|s| s.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("xwb")))
        .collect();
    if files.is_empty() {
        return Err(format!("no .xwb files under {}", sounds_dir.display()));
    }
    // (format, bits, channels) -> (waves, decoded, failed, seconds)
    type FormatTally = BTreeMap<(String, u16, u16), (usize, usize, usize, f64)>;
    let mut tally: FormatTally = BTreeMap::new();
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

    // Phase 43 announcer, map ambience & adaptive music coverage
    let mut lib = CueLibrary::scan(sounds_dir);
    println!("announcer events ({} indexed cues):", lib.len());
    let (mut mapped, mut missing) = (0usize, 0usize);
    for ev in AnnouncerEvent::ALL {
        let resolved = ev
            .default_cues()
            .iter()
            .find_map(|&c| resolve_cue_entry(&mut lib, c).map(|(src, n)| (c, src, n)));
        let variants = ev.variant_cues();
        let var_ok = variants
            .iter()
            .filter(|&&v| resolve_cue_entry(&mut lib, v).is_some())
            .count();
        match resolved {
            Some((cue, src, n)) => {
                mapped += 1;
                let extra = if variants.is_empty() {
                    String::new()
                } else {
                    format!(" [variants {var_ok}/{}]", variants.len())
                };
                println!(
                    "  {:<28} -> {:<38} ({src}, {n} wave(s), cd {:.0}s, prio {}){extra}",
                    ev.slug(),
                    cue,
                    ev.cooldown_secs(),
                    ev.priority()
                );
            }
            None => {
                missing += 1;
                println!(
                    "  {:<28} -> {:<38} MISSING",
                    ev.slug(),
                    ev.default_cues()[0]
                );
            }
        }
    }
    println!(
        "announcer summary: {mapped}/{} events mapped, {missing} MISSING",
        AnnouncerEvent::ALL.len()
    );

    println!("ambience & adaptive music:");
    for b in MapBiome::ALL {
        let (pri, det) = b.cues(true);
        let pri_ok = resolve_cue_entry(&mut lib, pri).is_some();
        let det_str = match det {
            Some(d) if resolve_cue_entry(&mut lib, d).is_some() => format!(" + {d}"),
            Some(d) => format!(" + {d} (MISSING)"),
            None => String::new(),
        };
        println!(
            "  biome {:<10} -> {pri}{}{det_str}",
            b.label(),
            if pri_ok { "" } else { " (MISSING)" }
        );
    }
    for (slot, cue) in [("playlist peace", MUSIC_BASE), ("playlist battle", MUSIC_BATTLE)] {
        match resolve_cue_entry(&mut lib, cue) {
            Some((src, n)) => println!("  {slot:<16} -> {cue} ({src}, {n} tracks)"),
            None => println!("  {slot:<16} -> {cue} MISSING"),
        }
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
    use crate::scmap::{ScMapLighting, ScMapWaterParams, TerrainStratum, TerrainTextures, WaterInfo};

    #[test]
    fn maps_factions_roles_and_orders_to_fa_cue_names() {
        assert_eq!(Fac::from_key("seraphim"), Fac::Seraphim);
        assert_eq!(Fac::from_key("cybran").weapon(true, false), "URL0202_Cannon_Dbl");
        assert_eq!(Fac::from_key("uef").destroy(false, 4), "UEL_Destroy_Huge");
        assert_eq!(select_category(Role::Mobile, NavDomain::Air, 1), "Air");
        assert_eq!(select_category(Role::Commander, NavDomain::Land, 1), "Commander");
        assert_eq!(order_cue(&Command::Stop { units: vec![] }), UI_ORDER_STOP);
        assert!(selection_rank(Role::Commander) > selection_rank(Role::Factory));
        assert_eq!(experimental_sighted_cue("URL0402"), "XGG_HQ_GD1_04182");
        assert_eq!(experimental_sighted_cue("XSB2401"), "XGG_HQ_GD1_04190");
        assert_eq!(structure_lost_cue(Role::Extractor), Some("XGG_HQ_GD1_04216"));
        assert_eq!(base_under_attack_cues(NavDomain::Air, 2, 100, 10)[0], "XGG_HQ_GD1_04157");
        assert_eq!(base_under_attack_cues(NavDomain::Naval, 2, -5, -80)[0], "XGG_HQ_GD1_04169");
        assert_eq!(base_under_attack_cues(NavDomain::Land, 3, 10, 50)[0], "XGG_HQ_GD1_04153");
        let st = pcm_stats(&[0.5, -0.5, 0.0, 0.0]);
        assert!((st.peak - 0.5).abs() < 1e-6 && st.clipped == 0);
    }

    fn want(unit: UnitId, kind: LoopKind, cue: &str, score: f32) -> LoopWant {
        LoopWant { unit, kind, cue: cue.into(), pos: [unit as f32, 0.0, 0.0], score }
    }

    #[test]
    fn loop_book_starts_updates_and_stops_with_unit_state() {
        let mut book = LoopBook::default();
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
        let p = book.plan(
            vec![want(1, LoopKind::Construct, "A_Construct_Loop", -10.0), want(2, LoopKind::Move, "B_Move_Loop", -20.0)],
            2,
            |_| true,
        );
        assert_eq!(p.update.len(), 2);
        assert!(p.start.is_empty() && p.stop.is_empty());
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
        let p = book.plan(vec![want(3, LoopKind::Move, "C_Move_Loop", -5.0)], 2, |v| v != 200);
        assert_eq!(p.start.len(), 1);
        assert_eq!(p.stop, vec![201]);
        assert!(book.len() <= 2);
        book.started(&p.start[0], 300);
        assert_eq!(book.clear(), vec![300]);
        assert_eq!(book.len(), 0);
    }

    #[test]
    fn announcer_queue_enforces_priority_single_voice_cooldowns_and_expiry() {
        let mut q = AnnouncerQueue::default();
        // Low-priority economy alert queued before urgent combat alerts: urgent wins first.
        assert!(q.enqueue(AnnouncerEvent::MassStorageLow, ["mass_low"], 10.0));
        assert!(q.enqueue(AnnouncerEvent::BaseUnderAttack, ["base_atk"], 10.1));
        assert!(q.enqueue(AnnouncerEvent::StrategicLaunchDetected, ["nuke_det"], 10.2));
        // Duplicate pending event coalesces rather than adding a second entry.
        assert!(!q.enqueue(AnnouncerEvent::BaseUnderAttack, ["base_atk_2"], 10.2));
        assert_eq!(q.pending_len(), 3);

        // First poll starts highest-priority (`StrategicLaunchDetected`).
        let (ev1, cue1) = q
            .poll(10.2, false, |_, c| Some(c[0].clone()))
            .expect("first line plays");
        assert_eq!((ev1, cue1.as_str()), (AnnouncerEvent::StrategicLaunchDetected, "nuke_det"));

        // While voice is active, nothing else overlaps.
        assert!(q.poll(11.0, true, |_, c| Some(c[0].clone())).is_none());
        // Within the 10s cooldown of StrategicLaunchDetected, re-enqueue is rejected.
        assert!(!q.enqueue(AnnouncerEvent::StrategicLaunchDetected, ["nuke_det"], 15.0));

        // Voice finishes at t=12.0 -> brief 0.35s breath gap holds at 12.0 and 12.2.
        assert!(q.poll(12.0, false, |_, c| Some(c[0].clone())).is_none());
        assert!(q.poll(12.2, false, |_, c| Some(c[0].clone())).is_none());
        // At t=12.4, next highest priority (`BaseUnderAttack`, updated candidate) plays.
        let (ev2, cue2) = q
            .poll(12.4, false, |_, c| Some(c[0].clone()))
            .expect("second line plays");
        assert_eq!((ev2, cue2.as_str()), (AnnouncerEvent::BaseUnderAttack, "base_atk_2"));

        // If `MassStorageFull` (MISSING cue) is higher than nothing else or queued alongside
        // `MassStorageLow`, unresolvable cues skip cleanly without blocking real ones.
        q.clear_pending();
        assert!(q.enqueue(AnnouncerEvent::MassStorageFull, ["missing_full"], 30.0));
        assert!(q.enqueue(AnnouncerEvent::EnergyStorageFull, ["missing_efull"], 30.0));
        // Wait: MassStorageLow was queued at 10.0 and cleared; re-enqueue at 30.0 alongside missing.
        assert!(q.enqueue(AnnouncerEvent::MassStorageLow, ["mass_low"], 30.0));
        let (ev3, _) = q
            .poll(30.0, false, |_, c| (!c[0].starts_with("missing")).then(|| c[0].clone()))
            .expect("skips missing");
        assert_eq!(ev3, AnnouncerEvent::MassStorageLow);
        // Missing cue still entered cooldown when popped after MassStorageLow finishes:
        let _ = q.poll(32.0, false, |_, _| None::<String>); // marks end of voice
        assert!(q.poll(32.5, false, |_, _| None::<String>).is_none()); // drains missing entries onto cooldown
        assert!(!q.enqueue(AnnouncerEvent::MassStorageFull, ["missing_full"], 50.0), "60s cooldown active");
        assert!(q.enqueue(AnnouncerEvent::MassStorageFull, ["missing_full"], 95.0), "cooldown expired");

        // Stale queued items (> 12s old) expire instead of playing late.
        assert!(q.poll(110.0, false, |_, c| Some(c[0].clone())).is_none());
        assert_eq!(q.pending_len(), 0);
    }

    #[test]
    fn econ_watch_hysteresis_ignores_initial_full_and_latches_stalls() {
        let mut w = EconWatch::default();
        let cap = 1000 * SCALE;
        // Starting at 100% storage never triggers `MassStorageFull` before dipping < 90%.
        for t in 200..300 {
            assert!(w.observe(t, 0.1, cap, cap, 10 * SCALE, cap, cap, 50 * SCALE).is_empty());
        }
        // Spend mass down to 5% (50 mass) for >= 2.0s -> fires MassStorageLow once.
        let mut fired = Vec::new();
        for t in 300..350 {
            fired.extend(w.observe(t, 0.1, 50 * SCALE, cap, 2 * SCALE, 500 * SCALE, cap, 20 * SCALE));
        }
        assert_eq!(fired, vec![AnnouncerEvent::MassStorageLow]);
        // Recover above 22% (300 mass), then fill back to 100% for >= 4.0s -> MassStorageFull fires once.
        let _ = w.observe(360, 0.1, 300 * SCALE, cap, 5 * SCALE, 500 * SCALE, cap, 20 * SCALE);
        let mut full_fired = Vec::new();
        for t in 361..420 {
            full_fired.extend(w.observe(t, 0.1, cap, cap, 5 * SCALE, 500 * SCALE, cap, 20 * SCALE));
        }
        assert_eq!(full_fired, vec![AnnouncerEvent::MassStorageFull]);
    }

    #[test]
    fn music_director_switches_with_hysteresis_dwell_and_ducks_for_announcer() {
        let mut md = MusicDirector::new();
        // Starts on Base_Building track 0.
        let tr0 = md.step(0.1, 0.1, false).expect("initial peace track");
        assert_eq!((tr0.mode, tr0.cue, tr0.variant), (MusicMode::Peace, MUSIC_BASE, 0));

        // Light skirmishing stays below ENTER_BATTLE_INTENSITY (0.45).
        for _ in 0..20 {
            md.add_combat(0.03);
            assert!(md.step(0.1, 0.1, true).is_none());
        }
        assert_eq!(md.mode(), MusicMode::Peace);

        // Sustained heavy combat pushes smoothed intensity >= 0.45 -> crossfades to Battle.
        let mut switched_to_battle = None;
        for _ in 0..50 {
            md.add_spatial_combat(0.45, [100.0, 100.0], [100.0, 100.0], true);
            if let Some(tr) = md.step(0.1, 0.1, true) {
                switched_to_battle = Some(tr);
                break;
            }
        }
        let tr_b = switched_to_battle.expect("switched to battle");
        assert_eq!((tr_b.mode, tr_b.cue, tr_b.variant), (MusicMode::Battle, MUSIC_BATTLE, 0));
        assert!(md.target_volume(true) < md.target_volume(false) * 0.7, "ducks under announcer");

        // Pausing (sim_dt == 0) mid-battle never decays intensity back to peace.
        for _ in 0..400 {
            assert!(md.step(0.0, 0.1, true).is_none());
        }
        assert_eq!(md.mode(), MusicMode::Battle);

        // Silence: must decay below EXIT_BATTLE_INTENSITY (0.15) AND hold calm >= 18s.
        for _ in 0..150 {
            assert!(md.step(0.1, 0.1, true).is_none(), "15s of silence is still holding battle");
        }
        let mut back_to_peace = None;
        for _ in 0..250 {
            if let Some(tr) = md.step(0.1, 0.1, true) {
                back_to_peace = Some(tr);
                break;
            }
        }
        let tr_p = back_to_peace.expect("returned to peace after hold");
        assert_eq!((tr_p.mode, tr_p.cue, tr_p.variant), (MusicMode::Peace, MUSIC_BASE, 1));
    }

    fn synth_scmap(stratum_path: &str, has_water: bool, water_elev: f32, heights: Vec<u16>) -> ScMap {
        ScMap {
            version_major: 2,
            version_minor: 60,
            map_width: 512.0,
            map_height: 512.0,
            preview_dds: Vec::new(),
            heightmap_width: 4,
            heightmap_height: 4,
            height_scale: 1.0 / 128.0,
            heightmap: heights,
            terrain_shader: String::new(),
            background_path: String::new(),
            sky_cubemap_path: String::new(),
            cubemaps: Vec::new(),
            lighting: ScMapLighting::default(),
            water: WaterInfo {
                has_water,
                elevation: water_elev,
                elevation_deep: water_elev - 5.0,
                elevation_abyss: water_elev - 15.0,
            },
            unknown1: 0, unknown2: 0, unknown3: 0, unknown4: 0, post_heightmap_byte: 0,
            shader_raw: Vec::new(), background_raw: Vec::new(), sky_raw: Vec::new(), cubemaps_raw: Vec::new(), tail_bytes: Vec::new(),
            terrain_textures: (!stratum_path.is_empty()).then(|| TerrainTextures {
                strata_albedo: vec![TerrainStratum { path: stratum_path.into(), scale: 16.0 }],
                strata_normal: Vec::new(),
                decals: Vec::new(),
                texturemap0_dds: Vec::new(),
                texturemap1_dds: Vec::new(),
                normal_map_dds: Vec::new(),
                water_params: ScMapWaterParams::default(),
                props: Vec::new(),
                props_tail_range: None,
                props_dirty: false,
            }),
        }
    }

    #[test]
    fn map_ambience_selects_loops_from_scmap_biome_and_water() {
        let lava = MapAmbience::from_scmap(&synth_scmap("/env/lava/layers/lav_rock_albedo.dds", false, 0.0, vec![3000; 25]));
        assert_eq!((lava.biome, lava.primary_cue, lava.detail_cue), (MapBiome::Volcanic, AMBIENT_RUMBLE, Some(AMBIENT_VOLCANIC)));
        // Zooming in boosts surface fire detail; zooming out boosts planetary rumble.
        assert!(lava.gains(50.0).1 > lava.gains(400.0).1);
        assert!(lava.gains(400.0).0 > lava.gains(50.0).0);

        let ocean = MapAmbience::from_scmap(&synth_scmap("/env/evergreen/layers/grass.dds", true, 25.0, vec![1000; 25]));
        assert_eq!((ocean.biome, ocean.detail_cue), (MapBiome::Oceanic, Some(AMBIENT_WATER)));

        let tundra = MapAmbience::from_scmap(&synth_scmap("/env/tundra/layers/snow.dds", false, 0.0, vec![3000; 25]));
        assert_eq!((tundra.biome, tundra.primary_cue), (MapBiome::Tundra, AMBIENT_TUNDRA));

        let desert = MapAmbience::from_scmap(&synth_scmap("/env/desert/layers/sand.dds", false, 0.0, vec![3000; 25]));
        assert_eq!((desert.biome, desert.detail_cue), (MapBiome::Arid, Some(AMBIENT_ARID)));

        let plain = MapAmbience::from_scmap(&synth_scmap("", false, 0.0, vec![3000; 25]));
        assert_eq!((plain.biome, plain.primary_cue, plain.detail_cue), (MapBiome::Temperate, AMBIENT_RUMBLE, None));
    }

    #[test]
    fn mission_dialogue_queues_render_side_without_affecting_world_hash() {
        let mut w = crate::skirmish::tests::test_world(7);
        w.set_options(crate::skirmish::GameOptions {
            victory: crate::skirmish::VictoryCondition::Sandbox,
            ..Default::default()
        });
        let src = r#"
mission "Voice Test"
map TESTMAP
army 0 faction red human
army 1 faction blue ai none
when at 0:00 do say "HQ briefing" voice X01_HQ_M01_02881; say "Fletcher reply" voice X01_Fletcher_M01_03419
"#;
        w.set_mission(crate::mission::Mission::parse(src).unwrap());
        w.step();
        let hash_before = w.state_hash();
        assert_eq!(w.mission().unwrap().voices.len(), 2);

        let engine = AudioEngine::new(None, true);
        let mut ga = GameAudio {
            engine,
            cues: HashMap::new(),
            dyn_names: HashMap::new(),
            units: HashMap::new(),
            stamp: 0,
            primed: false,
            last_tick: 0,
            audio_clock: 0.0,
            selection: Vec::new(),
            ui: None,
            pending_seen: 0,
            music_dir: MusicDirector::new(),
            music: None,
            music_retry: 0,
            ambience: MapAmbience::default(),
            ambient_primary: None,
            ambient_detail: None,
            ambient_retry: 0,
            announcer: AnnouncerQueue::default(),
            announcer_voice: None,
            mission_voices_seen: 0,
            dialogue_queue: std::collections::VecDeque::new(),
            dialogue_voice: None,
            econ_watch: EconWatch::default(),
            seen_experimentals: HashSet::new(),
            enemy_nukes: HashMap::new(),
            at_cap_prev: false,
            result_seen: false,
            loops: LoopBook::default(),
            played: BTreeMap::new(),
            announced: BTreeMap::new(),
            captions: Vec::new(),
        };
        ga.pump_mission_dialogue(&w);
        assert_eq!(ga.mission_voices_seen, 2);
        assert!(ga.dialogue_queue.is_empty(), "unresolvable offline cues drain cleanly");
        assert_eq!(w.state_hash(), hash_before, "dialogue playback never mutates sim state");

        let gd = crate::scd::default_game_dir();
        if let Some(mut real_ga) = GameAudio::offline(&gd) {
            real_ga.pump_mission_dialogue(&w);
            assert_eq!(real_ga.played.get("X01_HQ_M01_02881"), Some(&1));
            assert_eq!(real_ga.dialogue_queue.len(), 1, "second line waits until first finishes");
            assert_eq!(w.state_hash(), hash_before);
        }
    }
}
