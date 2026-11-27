//! Phases 6, 8, 10, 12 & 16 multi-army deterministic skirmish world.
//!
//! Fixed-tick (10 Hz) integer-only RTS core reading unit stats directly from
//! original `*_unit.bp` files:
//! - Multi-army & team-based economies (mass/energy stockpiles, storage caps, income/drain,
//!   Hydrocarbon plants, Mass/Energy Storage, and adjacency bonuses)
//! - Commanders, T1/T2/T3 Engineers (build, repair, assist, wreck reclaim, Overcharge)
//! - Full Tech 1->2->3 factory/extractor/shield upgrade chains & T4 Experimentals
//! - Multi-domain units & movement (`Land`, `Naval`, `Amphibious`, `Air`)
//! - Energy-draining bubble shields, Tactical & Strategic Missiles + TMD/SMD defense,
//!   and unit veterancy
//! - Deterministic `SpatialGrid`, bounded `FlowFieldCache` group routing, and parallel
//!   deterministic target acquisition with bit-identical `1t == Nt` state hashes.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use crate::blueprint::{self, Value};
use crate::combat;
use crate::economy::ConstructionProject;
use crate::flowfield::{self, FlowFieldCache, NavDomain};
use crate::jobs::WorkerPool;
use crate::mods;
use crate::pathfinding::{self, PathGrid, PathRequest};
use crate::roster;
use crate::scd;
use crate::scenario;
use crate::scmap::ScMap;
use crate::sim::{SCALE, TICK_HZ, Vec2};
use crate::spatial::{SPATIAL_CELL_MILLI, SpatialGrid};
use crate::terrain::HeightField;

pub type ArmyId = u8;
pub type UnitId = u32;

/// Hash header; updated for Phase 12/16 state.
pub const SKIRMISH_HASH_MAGIC: &[u8] = b"FARUSTSKM3";
/// Default headless game length cap: 60 minutes of game time.
pub const DEFAULT_MAX_TICKS: u64 = 36_000;
/// Path grid resolution in heightmap samples per cell.
pub const GRID_STEP: u32 = 4;
/// Steepest walkable rise between adjacent grid samples (milli-units).
pub const GRID_MAX_SLOPE_MILLI: i64 = 3_600;
/// Extra acquisition radius beyond weapon range for attack-move (milli).
pub const ACQUIRE_BONUS: i64 = 10 * SCALE;
/// Spacing between units in a formation (milli).
pub const FORMATION_SPACING: i64 = 3 * SCALE;
/// Snap radius for placing an extractor onto a mass marker or hydro onto hydro marker (milli).
pub const MASS_SNAP: i64 = 8 * SCALE;
/// Energy cost of one ACU Overcharge shot (5000 energy in FA).
pub const OVERCHARGE_ENERGY_COST: i64 = 5_000 * SCALE;
/// Damage of an ACU Overcharge shot.
pub const OVERCHARGE_DAMAGE: i64 = 12_000 * SCALE;
/// Splash radius of an ACU Overcharge shot.
pub const OVERCHARGE_SPLASH: i64 = 2_500;
/// ACU Overcharge range.
pub const OVERCHARGE_RANGE: i64 = 22 * SCALE;
/// Shield downtime after collapse or energy stall (50 ticks = 5s).
pub const SHIELD_RECHARGE_DELAY: u32 = 50;
/// Phase 18: Forged Alliance default per-army unit cap.
pub const DEFAULT_UNIT_CAP: u32 = 1000;
/// Phase 18: anti-nuke interceptor cost (projectile blueprint `TIMMissileIntercerptor01`).
pub const ANTI_NUKE_COST_MASS: i64 = 3_600 * SCALE;
pub const ANTI_NUKE_COST_ENERGY: i64 = 360_000 * SCALE;
/// Phase 18: extra reach (beyond both footprints) at which a unit boards a transport.
pub const TRANSPORT_PICKUP_REACH: i64 = 4 * SCALE;
/// Phase 18: jammer false blips re-roll their positions every this many ticks.
pub const JAMMER_BLIP_PERIOD: u64 = 50;

/// Phase 18 victory condition (FA lobby option).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum VictoryCondition {
    /// An army is defeated when its Commander dies (FA default).
    #[default]
    Assassination,
    /// Defeated when every structure and engineer (incl. the Commander) is dead.
    Supremacy,
    /// Defeated when every unit is dead.
    Annihilation,
    /// Nobody is ever defeated; the game runs until the tick limit.
    Sandbox,
}

impl VictoryCondition {
    pub const ALL: [VictoryCondition; 4] = [
        VictoryCondition::Assassination,
        VictoryCondition::Supremacy,
        VictoryCondition::Annihilation,
        VictoryCondition::Sandbox,
    ];
    pub fn code(self) -> u8 {
        match self {
            VictoryCondition::Assassination => 0,
            VictoryCondition::Supremacy => 1,
            VictoryCondition::Annihilation => 2,
            VictoryCondition::Sandbox => 3,
        }
    }
    pub fn from_code(c: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.code() == c)
    }
    pub fn label(self) -> &'static str {
        match self {
            VictoryCondition::Assassination => "assassination",
            VictoryCondition::Supremacy => "supremacy",
            VictoryCondition::Annihilation => "annihilation",
            VictoryCondition::Sandbox => "sandbox",
        }
    }
    /// Parses a CLI / lobby name (case-insensitive, `demoralization` = assassination).
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        if s == "demoralization" {
            return Some(VictoryCondition::Assassination);
        }
        Self::ALL.into_iter().find(|v| v.label() == s)
    }
}

/// Phase 18 game options. Stored in the replay header; `Default` reproduces the
/// pre-Phase-18 rules exactly (assassination, FA unit cap 1000).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GameOptions {
    pub victory: VictoryCondition,
    /// Max live units (+ started constructions) per army; `0` = unlimited.
    pub unit_cap: u32,
}

impl Default for GameOptions {
    fn default() -> Self {
        Self {
            victory: VictoryCondition::Assassination,
            unit_cap: DEFAULT_UNIT_CAP,
        }
    }
}

/// Phase 18 blueprint `Intel` block (stealth, cloak, omni, jammer).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IntelSpec {
    pub radar_stealth: bool,
    pub cloak: bool,
    /// `RadarStealthFieldRadius` (milli): grants radar stealth to own units inside.
    pub stealth_field_radius: i64,
    /// `CloakFieldRadius` (milli): grants cloak to own units inside.
    pub cloak_field_radius: i64,
    pub omni_radius: i64,
    /// `JammerBlips` false radar contacts within `JamRadius` (min, max) milli.
    pub jammer_blips: u32,
    pub jam_radius: (i64, i64),
    /// `StealthWaitTime`: stealth/cloak only after standing still this long (Selen).
    pub stealth_wait_ticks: u32,
    /// Flags only come from enhancements (Commanders/SCUs ship them disabled).
    pub gated: bool,
}

/// Phase 18 air-transport capacity (attach points counted from the `.scm` skeleton).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TransportSpec {
    /// Small / medium / large attach points.
    pub small: u8,
    pub medium: u8,
    pub large: u8,
    /// Small slots taken by one class-2 / class-3 passenger (`Class2AttachSize` / `Class3AttachSize`).
    pub class2_size: u8,
    pub class3_size: u8,
    /// `CANTRANSPORTCOMMANDER` category.
    pub carries_commander: bool,
}

impl TransportSpec {
    /// Whether a passenger of `class` fits next to `cargo_classes` already aboard.
    pub fn fits(&self, cargo_classes: &[u8], class: u8) -> bool {
        if !(1..=3).contains(&class) {
            return false;
        }
        let cost = |c: u8| match c {
            1 => 1u32,
            2 => self.class2_size.max(1) as u32,
            _ => self.class3_size.max(1) as u32,
        };
        let used: u32 = cargo_classes.iter().map(|&c| cost(c)).sum();
        let mediums = cargo_classes.iter().filter(|&&c| c == 2).count() as u32;
        let larges = cargo_classes.iter().filter(|&&c| c == 3).count() as u32;
        used + cost(class) <= self.small as u32
            && (class != 2 || mediums < self.medium as u32)
            && (class != 3 || larges < self.large as u32)
    }
    /// Total small-slot capacity.
    pub fn slots(&self) -> u32 {
        self.small as u32
    }
}

/// Phase 18 ACU/SCU enhancement from blueprint `Enhancements`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EnhancementSpec {
    /// Blueprint key, e.g. `AdvancedEngineering`.
    pub name: String,
    /// Display name with the `<LOC ..>` tag stripped.
    pub display: String,
    /// `LCH`, `RCH` or `Back`.
    pub slot: String,
    pub prerequisite: Option<String>,
    /// `RemoveEnhancements`: non-empty marks a removal enhancement.
    pub removes: Vec<String>,
    pub cost_mass: i64,
    pub cost_energy: i64,
    pub build_work: i64,
    /// Effects (0 / false = none).
    pub new_build_rate: i64,
    pub add_health: i64,
    pub regen_per_tick: i64,
    pub engineering_tier: u8,
    pub new_max_radius: i64,
    pub new_rate_of_fire_milli: i64,
    pub damage_add: i64,
    pub new_damage_radius: i64,
    pub vision_radius: i64,
    pub omni_radius: i64,
    pub mass_per_tick: i64,
    pub energy_per_tick: i64,
    pub upkeep_per_tick: i64,
    pub shield_max_hp: i64,
    pub shield_radius: i64,
    pub shield_regen_per_tick: i64,
    pub personal_shield: bool,
    pub radar_stealth: bool,
    pub cloak: bool,
    pub jammer_radius: i64,
}

impl EnhancementSpec {
    pub fn is_removal(&self) -> bool {
        !self.removes.is_empty()
    }
}

/// Phase 18 blueprint data beyond the core stats.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnitTypeExt {
    pub intel: IntelSpec,
    /// Present for air transports.
    pub transport: Option<TransportSpec>,
    /// `Transport.TransportClass` of a land unit (1..=3); 0 = cannot be carried.
    pub transport_class: u8,
    pub enhancements: Vec<EnhancementSpec>,
    /// Missile defense: intercepts tactical (TMD) and/or strategic (SMD) missiles.
    pub intercept_tactical: bool,
    pub intercept_strategic: bool,
    /// Ticks between two interceptions.
    pub intercept_reset: u32,
    /// Interceptor missile storage (`MaxProjectileStorage`, SMD); 0 = no ammo needed.
    pub ammo_cap: u32,
    /// Phase 24: structure can only be placed on water (`Physics.BuildOnLayerCaps`).
    pub water_only: bool,
    /// Phase 30: flight time on a full tank (`Physics.FuelUseTime`, ticks); 0 = unlimited.
    pub fuel_ticks: u32,
    /// Phase 30: `Physics.FuelRechargeRate` (fuel seconds regained per second, milli).
    pub fuel_recharge_milli: i64,
    /// Phase 30: present on air staging platforms.
    pub staging: Option<StagingSpec>,
    /// Phase 31: pod drones an engineering station keeps alive (`Economy.EngineeringPods`
    /// `PodUnitID`, upper-case); empty = the station works itself (Cybran Hive).
    pub pods: Vec<String>,
    /// Phase 32: unit this structure turns into the moment it is finished
    /// (`Economy.BuildUnit`, e.g. Megalith crab eggs).
    pub hatch: Option<String>,
    /// Phase 32: non-structure that produces units (Fatboy, carriers, CZAR, Tempest).
    pub mobile_factory: bool,
    /// Phase 32: non-engineer that places structures (Megalith laying eggs).
    pub mobile_builder: bool,
    /// Phase 33: death weapon (`DeathWeapon` / `DeathImpact` / `WeaponCategory = 'Death'`):
    /// damage and radius dealt to every unit around the wreck ("Volatile").
    pub death_damage: i64,
    pub death_radius: i64,
    /// Phase 33: `Buffs.Stun` — (ticks, radius, on EMP impact, on death).
    pub stun: Option<(u32, i64, bool, bool)>,
    /// Phase 33: `UNTARGETABLE` category (orbital satellites).
    pub untargetable: bool,
    /// Phase 33: satellite an orbital launch centre keeps in orbit (resolved by the catalog).
    pub satellite: Option<String>,
}

/// Phase 30: air staging platform data (`Transport.DockingSlots`, `AI.Refueling*`,
/// `Transport.RepairRate`, `AI.RepairConsume*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagingSpec {
    pub slots: u8,
    /// Docked refuel speed = `FuelRechargeRate × multiplier`.
    pub refuel_multiplier: i64,
    /// Repair per second in ‰ of max health.
    pub repair_permille: i64,
    /// Resource cost per second of repairing (milli).
    pub repair_energy: i64,
    pub repair_mass: i64,
    /// Auto-refuel search radius (`AI.StagingPlatformScanRadius`, milli).
    pub scan_radius: i64,
}

/// Runtime enhancement state of one unit (effects already folded in).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EnhancementState {
    /// Installed `(slot, enhancement)` pairs, sorted by slot.
    pub installed: Vec<(String, String)>,
    /// Build rate override (0 = blueprint).
    pub build_rate: i64,
    /// Engineering tech level granted (0/1 = T1).
    pub engineering_tier: u8,
    pub regen_per_tick: i64,
    pub mass_per_tick: i64,
    pub energy_per_tick: i64,
    /// Non-shield maintenance (stealth, cloak, jammer, teleporter) per tick.
    pub upkeep_per_tick: i64,
    pub radar_stealth: bool,
    pub cloak: bool,
    pub jammer_radius: i64,
    /// Max-HP bonus currently applied (so replacing an enhancement applies the delta).
    pub health_bonus: i64,
    /// Upkeep could not be paid this tick: intel effects are offline.
    pub stalled: bool,
}

/// Strips a `<LOC key>` prefix from a blueprint display string.
fn strip_loc(s: &str) -> String {
    match s.find('>') {
        Some(i) if s.starts_with("<LOC") => s[i + 1..].to_string(),
        _ => s.to_string(),
    }
}

/// Weapons skipped during standard weapon extraction (Overcharge/missiles/death use explicit paths).
const EXCLUDED_WEAPONS: &[&str] = &[
    "OverCharge",
    "AutoOverCharge",
    "DeathWeapon",
    "DeathImpact",
    "ChronoDampener",
    "MLG",
    "AntiTorpedo",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    Commander,
    Factory,
    Extractor,
    Generator,
    PointDefense,
    Radar,
    Engineer,
    Mobile,
    AirFactory,
    NavalFactory,
    Hydro,
    MassStorage,
    EnergyStorage,
    Shield,
    AntiAir,
    MissileSilo,
    MissileDefense,
    Experimental,
    /// Phase 28: wall section — an inert structure that blocks land/amphibious pathing.
    Wall,
    /// Phase 30: air staging platform — docks, refuels and repairs aircraft.
    AirStaging,
    /// Phase 31: engineering station (UEF Kennel, Cybran Hive) — automatically assists and
    /// repairs allied work in its radius, itself or through its pod drones.
    EngStation,
    /// Phase 33: orbital launch centre (Novax) — keeps its satellite in orbit.
    Orbital,
}

impl Role {
    pub fn is_structure(self) -> bool {
        matches!(
            self,
            Role::Factory
                | Role::AirFactory
                | Role::NavalFactory
                | Role::Extractor
                | Role::Generator
                | Role::Hydro
                | Role::MassStorage
                | Role::EnergyStorage
                | Role::PointDefense
                | Role::Radar
                | Role::Shield
                | Role::AntiAir
                | Role::MissileSilo
                | Role::MissileDefense
                | Role::Wall
                | Role::AirStaging
                | Role::EngStation
                | Role::Orbital
        )
    }
    pub fn is_factory(self) -> bool {
        matches!(self, Role::Factory | Role::AirFactory | Role::NavalFactory)
    }
    pub fn is_builder(self) -> bool {
        matches!(self, Role::Commander | Role::Engineer)
    }
    pub fn code(self) -> u8 {
        match self {
            Role::Commander => 0,
            Role::Factory => 1,
            Role::Extractor => 2,
            Role::Generator => 3,
            Role::PointDefense => 4,
            Role::Radar => 5,
            Role::Engineer => 6,
            Role::Mobile => 7,
            Role::AirFactory => 8,
            Role::NavalFactory => 9,
            Role::Hydro => 10,
            Role::MassStorage => 11,
            Role::EnergyStorage => 12,
            Role::Shield => 13,
            Role::AntiAir => 14,
            Role::MissileSilo => 15,
            Role::MissileDefense => 16,
            Role::Experimental => 17,
            Role::Wall => 18,
            Role::AirStaging => 19,
            Role::EngStation => 20,
            Role::Orbital => 21,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntelVisibility {
    Visible,
    RadarBlip,
    Hidden,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeaponType {
    pub label: String,
    pub damage: i64,
    pub range: i64,
    pub reset: u32,
    pub splash_radius: i64,
    pub muzzle_velocity: i64,
    pub anti_air: bool,
    pub anti_surface: bool,
    pub turret_yaw_bone: Option<String>,
    pub turret_pitch_bone: Option<String>,
    /// Phase 29: turret slew rate (milliradians per tick); 0 = fixed / instant aim.
    pub turret_yaw_speed: i64,
    /// Phase 29: fire only when the turret is within this angle of the target (mrad).
    pub firing_tolerance: i64,
    /// Phase 29: instant beam / hitscan weapon (no projectile; terrain-checked line of fire).
    pub beam: bool,
    /// Phase 29: aim ahead of moving targets (`LeadTarget`).
    pub lead_target: bool,
    /// Phase 33: `DamageType = 'EMP'` — applies the unit's `Buffs.Stun` on impact.
    pub emp: bool,
    /// Phase 33: `WeaponCategory = 'Kamikaze'` — detonates the unit on firing.
    pub kamikaze: bool,
    /// Phase 33: weapon `Buffs` STUN applied on impact (ticks, radius).
    pub stun: Option<(u32, i64)>,
}

/// Phase 33: the first `BuffType = 'STUN'` entry of a weapon's `Buffs` list that applies
/// on impact, as (ticks, radius milli).
fn weapon_stun(w: &Value) -> Option<(u32, i64)> {
    w.get("Buffs")?.table_list().into_iter().find_map(|b| {
        let on_impact = b.get("Add").and_then(|a| a.get_bool("OnImpact")).unwrap_or(true);
        (b.get_str("BuffType") == Some("STUN") && on_impact).then(|| {
            let ticks = (b.get_f64("Duration").unwrap_or(0.0).max(0.0) * TICK_HZ as f64).round() as u32;
            let radius = (b.get_f64("Radius").unwrap_or(0.0).max(0.0) * SCALE as f64).round() as i64;
            (ticks, radius)
        })
    }).filter(|s| s.0 > 0)
}

/// Phase 33: a death effect to resolve after cleanup: (army, position, death damage, death
/// radius, on-death stun).
type DeathFx = (ArmyId, (i64, i64), i64, i64, Option<(u32, i64)>);

/// Phase 29: spatial reach (beyond half a shell's per-tick segment) searched for units the
/// shell passes through; hit boxes wider than this are only hit near their centre line.
const DIRECT_HIT_REACH: i64 = 5 * SCALE;

/// Phase 29: a line of fire may graze the terrain by this much (milli) before it is blocked.
pub const LOF_CLEARANCE: i64 = SCALE / 4;

/// Phase 29: sim rules generation. Replays / saves record it and refuse to re-simulate
/// under different rules (1 = Phases ≤ 28; 2 = Phase 29 ballistics, turrets and structure
/// footprints blocking pathing).
/// 3 = Phase 30 air fuel & staging; 4 = Phase 31 engineering stations & pod drones;
/// 5 = Phase 32 mobile factories, mobile staging & hatching eggs; 6 = Phase 33 Novax
/// satellite, EMP stun, death explosions & kamikaze weapons; 7 = Phase 34 prop reclaim;
/// 8 = Phase 35 fire states, ground fire, guard & build pause; 9 = Phase 36 ferries;
/// 10 = Phase 53 pings as sim commands (allied AIs react).
pub const RULES_VERSION: u32 = 10;

/// Phase 29: milliradians per degree, scaled by 1000 (17.453 mrad / deg).
const MRAD_PER_DEG_MILLI: i64 = 17_453;

/// Quantized unit type derived once from a blueprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitType {
    pub id: String,
    pub name: String,
    pub role: Role,
    pub tier: u8,
    pub domain: NavDomain,
    pub strategic_icon: String,
    pub upgrades_to: Option<String>,
    pub max_hp: i64,
    pub speed_per_tick: i64,
    pub elevation: i64,
    pub weapons: Vec<WeaponType>,
    /// Milli-mass / milli-energy.
    pub cost_mass: i64,
    pub cost_energy: i64,
    /// Work units: `BuildTime * TICK_HZ` (one builder adds `build_rate`/tick).
    pub build_work: i64,
    pub build_rate: i64,
    pub build_range: i64,
    /// Milli-units per tick.
    pub mass_per_tick: i64,
    pub energy_per_tick: i64,
    pub storage_mass: i64,
    pub storage_energy: i64,
    pub vision_radius: i64,
    pub radar_radius: i64,
    pub shield_max_hp: i64,
    pub shield_radius: i64,
    pub shield_energy_upkeep: i64,
    pub shield_regen_per_tick: i64,
    pub vet_thresholds: [u32; 5],
    pub missile_cost_mass: i64,
    pub missile_cost_energy: i64,
    pub missile_damage: i64,
    pub missile_splash: i64,
    pub missile_range: i64,
    /// Phase 33: stun carried by this silo's missiles (Cybran nukes).
    pub missile_stun: Option<(u32, i64)>,
    pub intercept_radius: i64,
    /// Footprint half-size (milli) and render size (milli).
    pub radius: i64,
    pub size: [i64; 3],
    /// Phase 18: intel II, transport, enhancement and interception data.
    pub ext: UnitTypeExt,
}

fn default_strategic_icon(role: Role, tier: u8) -> &'static str {
    match role {
        Role::Commander => "icon_commander_generic",
        Role::Factory => match tier {
            3 => "icon_factory3_land",
            2 => "icon_factory2_land",
            _ => "icon_factory1_land",
        },
        Role::AirFactory => match tier {
            3 => "icon_factory3_air",
            2 => "icon_factory2_air",
            _ => "icon_factory1_air",
        },
        Role::NavalFactory => match tier {
            3 => "icon_factory3_naval",
            2 => "icon_factory2_naval",
            _ => "icon_factory1_naval",
        },
        Role::Extractor => match tier {
            3 => "icon_structure3_mass",
            2 => "icon_structure2_mass",
            _ => "icon_structure1_mass",
        },
        Role::Generator | Role::Hydro | Role::EnergyStorage => match tier {
            3 => "icon_structure3_energy",
            2 => "icon_structure2_energy",
            _ => "icon_structure1_energy",
        },
        Role::MassStorage => "icon_structure1_mass",
        Role::PointDefense => "icon_structure1_directfire",
        Role::Radar => "icon_structure1_intel",
        Role::Shield => "icon_structure2_shield",
        Role::AntiAir => "icon_structure1_antiair",
        Role::MissileSilo => "icon_structure2_missile",
        Role::MissileDefense => "icon_structure2_antimissile",
        Role::Engineer => "icon_land1_engineer",
        Role::Experimental => "icon_experimental_generic",
        Role::Mobile => "icon_land1_directfire",
        Role::Wall => "icon_structure_wall",
        Role::AirStaging => "icon_structure2_air",
        Role::EngStation => "icon_structure2_engineer",
        Role::Orbital => "icon_experimental_generic",
    }
}

impl UnitType {
    pub fn max_range(&self) -> i64 {
        self.weapons.iter().map(|w| w.range).max().unwrap_or(0)
    }

    #[allow(dead_code)]
    pub fn can_fire_aa(&self) -> bool {
        self.weapons.iter().any(|w| w.anti_air)
    }

    #[allow(dead_code)]
    pub fn can_fire_surface(&self) -> bool {
        self.weapons.iter().any(|w| w.anti_surface)
    }

    /// Build a type from a parsed blueprint.
    pub fn from_blueprint(id: &str, role: Role, root: &Value) -> Self {
        let s = blueprint::summarize(id, "", root);
        let q = |v: f64| (v * SCALE as f64).round() as i64;
        let per_tick = |v: f64| q(v) / TICK_HZ as i64;
        let econ = root.get("Economy");
        let econ_f = |k: &str| econ.and_then(|e| e.get_f64(k)).unwrap_or(0.0);
        let intel = root.get("Intel");
        let intel_f = |k: &str| intel.and_then(|e| e.get_f64(k)).unwrap_or(0.0);
        let phys = root.get("Physics");
        let display = root.get("Display");
        let defense = root.get("Defense");
        let shield = defense.and_then(|d| d.get("Shield"));

        let cats: Vec<String> = root
            .get("Categories")
            .map(Value::string_list)
            .unwrap_or_default();
        let has_cat = |c: &str| cats.iter().any(|x| x.eq_ignore_ascii_case(c));
        let tier = if role == Role::Experimental || has_cat("EXPERIMENTAL") {
            4
        } else if has_cat("TECH3") {
            3
        } else if has_cat("TECH2") {
            2
        } else {
            1
        };

        let motion = phys.and_then(|p| p.get_str("MotionType")).unwrap_or("");
        let mut domain = NavDomain::from_motion_type(motion);
        if has_cat("AIR") && !role.is_structure() {
            domain = NavDomain::Air;
        }
        let elevation = if domain == NavDomain::Air {
            q(phys.and_then(|p| p.get_f64("Elevation")).unwrap_or(16.0).clamp(4.0, 25.0))
        } else {
            0
        };

        let raw_weapons: Vec<&Value> = root
            .get("Weapon")
            .map(|w| w.table_list())
            .unwrap_or_default();

        let mut missile_damage = 0i64;
        let mut missile_stun: Option<(u32, i64)> = None;
        let mut missile_splash = 0i64;
        let mut missile_range = 0i64;
        let mut intercept_radius = 0i64;

        let weapons = raw_weapons
            .iter()
            .filter_map(|t| {
                let label = t.get_str("Label").unwrap_or("").to_string();
                if EXCLUDED_WEAPONS.iter().any(|x| label.contains(x)) {
                    return None;
                }
                let r = t.get_f64("MaxRadius")?;
                let nuke_dmg = t.get_f64("NukeInnerRingDamage").unwrap_or(0.0);
                let is_nuke = label.contains("Nuke") || t.get("NukeInnerRingDamage").is_some();
                let is_tac_missile = role == Role::MissileSilo
                    && (label.contains("Missile") || has_cat("SILO"));

                if is_nuke || is_tac_missile {
                    missile_stun = missile_stun.or(weapon_stun(t));
                    let md = t.get_f64("Damage").unwrap_or(0.0).max(nuke_dmg).max(if is_nuke { 70_000.0 } else { 6_000.0 });
                    let mr = t.get_f64("NukeInnerRingRadius").or_else(|| t.get_f64("DamageRadius")).unwrap_or(if is_nuke { 25.0 } else { 4.0 });
                    missile_damage = q(md);
                    missile_splash = q(mr.max(2.0));
                    missile_range = q(r.min(2_000.0));
                    return None;
                }
                if role == Role::MissileDefense && (label.contains("AntiNuke") || t.get_str("RangeCategory") == Some("UWRC_Countermeasure") || r >= 25.0) {
                    intercept_radius = intercept_radius.max(q(r));
                    // Phase 18: countermeasure turrets only intercept missiles (no AA fire).
                    if label.contains("AntiNuke") || t.get_str("RangeCategory") == Some("UWRC_Countermeasure") {
                        return None;
                    }
                }
                if label.contains("TacMissile") || label.contains("TacNuke") {
                    return None;
                }

                let mut d = t.get_f64("Damage").unwrap_or(0.0);
                if d <= 0.0 && role != Role::PointDefense && role != Role::AntiAir && tier >= 2 {
                    // Fallback for salvo weapons without inline Damage parsed cleanly
                    d = if tier == 4 { 400.0 } else { 80.0 };
                }
                let rof = t.get_f64("RateOfFire").unwrap_or(1.0);
                if d <= 0.0 || r <= 0.0 || rof <= 0.0 {
                    return None;
                }

                let layers_str = t
                    .get("FireTargetLayerCapsTable")
                    .map(|tbl| {
                        ["Land", "Air", "Water", "Seabed"]
                            .iter()
                            .filter_map(|k| tbl.get_str(k))
                            .collect::<Vec<_>>()
                            .join("|")
                    })
                    .unwrap_or_default();
                let range_cat = t.get_str("RangeCategory").unwrap_or("");
                let explicit_aa = role == Role::AntiAir
                    || range_cat == "UWRC_AntiAir"
                    || label.contains("AA")
                    || layers_str.contains("Air");
                let only_aa = role == Role::AntiAir
                    || range_cat == "UWRC_AntiAir"
                    || (layers_str.contains("Air")
                        && !layers_str.contains("Land")
                        && !layers_str.contains("Water")
                        && !layers_str.contains("Seabed"));
                let anti_air = explicit_aa;
                let anti_surface = !only_aa;

                let splash_radius = t.get_f64("DamageRadius").map(q).unwrap_or(0).max(0);
                let muzzle_velocity = t.get_f64("MuzzleVelocity").map(per_tick).unwrap_or(0).max(0);
                let turret_yaw_bone = t.get_str("TurretBoneYaw").map(ToOwned::to_owned);
                let turret_pitch_bone = t.get_str("TurretBonePitch").map(ToOwned::to_owned);
                let deg_to_mrad = |deg: f64| ((deg * MRAD_PER_DEG_MILLI as f64) / 1000.0).round() as i64;
                let turreted = t.get_bool("Turreted").unwrap_or(turret_yaw_bone.is_some()) && turret_yaw_bone.is_some();
                let turret_yaw_speed = if turreted {
                    (deg_to_mrad(t.get_f64("TurretYawSpeed").unwrap_or(90.0).max(1.0)) / TICK_HZ as i64).max(1)
                } else {
                    0
                };
                let firing_tolerance = deg_to_mrad(t.get_f64("FiringTolerance").unwrap_or(2.0).clamp(0.0, 180.0)).max(5);
                let beam = t.get("BeamLifetime").is_some()
                    || t.get_bool("ContinuousBeam").unwrap_or(false)
                    || muzzle_velocity <= 0;
                let lead_target = t.get_bool("LeadTarget").unwrap_or(true);
                let emp = t.get_str("DamageType") == Some("EMP");
                let stun = weapon_stun(t);
                let kamikaze = t.get_str("WeaponCategory") == Some("Kamikaze");
                Some(WeaponType {
                    label,
                    damage: q(d),
                    range: q(r),
                    reset: combat::cooldown_reset_ticks(TICK_HZ, q(rof)),
                    splash_radius,
                    muzzle_velocity,
                    anti_air,
                    anti_surface,
                    turret_yaw_bone,
                    turret_pitch_bone,
                    turret_yaw_speed,
                    firing_tolerance,
                    beam: beam && !kamikaze,
                    lead_target,
                    emp,
                    kamikaze,
                    stun,
                })
            })
            .collect();

        if role == Role::MissileSilo && missile_damage == 0 {
            if tier >= 3 {
                missile_damage = 70_000 * SCALE;
                missile_splash = 24 * SCALE;
                missile_range = 1_024 * SCALE;
            } else {
                missile_damage = 6_000 * SCALE;
                missile_splash = 4 * SCALE;
                missile_range = 256 * SCALE;
            }
        }
        if role == Role::MissileDefense && intercept_radius == 0 {
            intercept_radius = if tier >= 3 { 90 * SCALE } else { 30 * SCALE };
        }

        let size_x = root.get_f64("SizeX").unwrap_or(1.0).max(0.2);
        let size_y = root.get_f64("SizeY").unwrap_or(1.0).max(0.2);
        let size_z = root.get_f64("SizeZ").unwrap_or(1.0).max(0.2);
        let footprint = root.get("Footprint");
        let fp = |k: &str, dflt: f64| footprint.and_then(|f| f.get_f64(k)).unwrap_or(dflt);
        let half = if role.is_structure() {
            fp("SizeX", size_x).max(fp("SizeZ", size_z)) * 0.5
        } else {
            size_x.max(size_z) * 0.5
        };
        let name = s
            .description
            .as_deref()
            .map(|d| match d.find('>') {
                Some(i) if d.starts_with("<LOC") => d[i + 1..].to_string(),
                _ => d.to_string(),
            })
            .unwrap_or_else(|| id.to_string());
        let strategic_icon = display
            .and_then(|d| d.get_str("StrategicIconName"))
            .unwrap_or_else(|| default_strategic_icon(role, tier))
            .to_string();
        let upgrades_to = root
            .get("General")
            .and_then(|g| g.get_str("UpgradesTo"))
            .map(|s| s.to_ascii_uppercase());

        let shield_max_hp = shield.and_then(|s| s.get_f64("ShieldMaxHealth")).map(q).unwrap_or(0);
        let shield_radius = if shield_max_hp > 0 {
            let dia = shield.and_then(|s| s.get_f64("ShieldSize")).unwrap_or(6.0);
            q((dia * 0.5).max(3.0))
        } else {
            0
        };
        let shield_energy_upkeep = if shield_max_hp > 0 {
            per_tick(econ_f("MaintenanceConsumptionPerSecondEnergy").max(50.0))
        } else {
            0
        };
        let shield_regen_per_tick = if shield_max_hp > 0 {
            per_tick(shield.and_then(|s| s.get_f64("ShieldRegenRate")).unwrap_or(40.0)).max(1)
        } else {
            0
        };

        let vet = root.get("Veteran");
        let v_lvl = |k: &str, dflt: u32| -> u32 {
            vet.and_then(|v| v.get_f64(k))
                .map(|x| x.round().max(1.0) as u32)
                .unwrap_or(dflt)
        };
        let vet_thresholds = [
            v_lvl("Level1", 5),
            v_lvl("Level2", 10),
            v_lvl("Level3", 25),
            v_lvl("Level4", 50),
            v_lvl("Level5", 100),
        ];

        let default_build_range = match role {
            Role::Commander => 10.0,
            Role::Engineer => 6.0,
            // Phase 31: stations work within their guard radius.
            Role::EngStation => root.get("AI").and_then(|a| a.get_f64("GuardScanRadius")).unwrap_or(15.0),
            _ => 0.0,
        };
        let vision_default = match role {
            Role::Commander => 26.0,
            Role::PointDefense => 24.0,
            Role::Radar => 20.0,
            _ => 18.0,
        };
        let missile_cost_mass = if role == Role::MissileSilo {
            if tier >= 3 { 12_000 * SCALE } else { 250 * SCALE }
        } else {
            0
        };
        let missile_cost_energy = if role == Role::MissileSilo {
            if tier >= 3 { 1_350_000 * SCALE } else { 5_000 * SCALE }
        } else {
            0
        };
        let ext = UnitTypeExt::from_blueprint(role, tier, domain, root, &cats, &raw_weapons);
        // Phase 18: air transports fly at their `Air.MaxAirspeed` (Physics.MaxSpeed is a
        // ground placeholder for aircraft).
        let speed_per_tick = if role.is_structure() {
            0
        } else if ext.transport.is_some() && domain == NavDomain::Air {
            root.get("Air")
                .and_then(|a| a.get_f64("MaxAirspeed"))
                .or(s.max_speed)
                .map(per_tick)
                .unwrap_or(0)
                .max(0)
        } else {
            s.max_speed.map(per_tick).unwrap_or(0).max(0)
        };

        Self {
            id: id.to_string(),
            name,
            role,
            tier,
            domain,
            strategic_icon,
            upgrades_to,
            max_hp: s.health.map(q).unwrap_or(SCALE).max(1),
            speed_per_tick,
            elevation,
            weapons,
            cost_mass: s.build_cost_mass.map(q).unwrap_or(1).max(1),
            cost_energy: s.build_cost_energy.map(q).unwrap_or(1).max(1),
            build_work: (s.build_time.unwrap_or(1.0).max(1.0) * TICK_HZ as f64).round() as i64,
            build_rate: econ_f("BuildRate").round().max(0.0) as i64,
            build_range: q(econ_f("MaxBuildDistance").max(default_build_range)),
            mass_per_tick: per_tick(econ_f("ProductionPerSecondMass")),
            energy_per_tick: per_tick(econ_f("ProductionPerSecondEnergy")),
            storage_mass: q(econ_f("StorageMass")),
            storage_energy: q(econ_f("StorageEnergy")),
            vision_radius: q(intel_f("VisionRadius").max(vision_default)),
            radar_radius: q(intel_f("RadarRadius").max(0.0)),
            shield_max_hp,
            shield_radius,
            shield_energy_upkeep,
            shield_regen_per_tick,
            vet_thresholds,
            missile_cost_mass,
            missile_cost_energy,
            missile_damage,
            missile_splash,
            missile_range,
            missile_stun,
            intercept_radius,
            radius: q(half).max(SCALE / 4),
            size: [q(size_x), q(size_y), q(size_z)],
            ext,
        }
    }
}

impl UnitTypeExt {
    /// Phase 18 blueprint extraction: `Intel`, `Transport`, `Enhancements` and missile
    /// countermeasure data. Pure; mesh-derived transport capacity is filled in by
    /// [`Catalog::load`].
    pub fn from_blueprint(
        role: Role,
        tier: u8,
        domain: NavDomain,
        root: &Value,
        cats: &[String],
        raw_weapons: &[&Value],
    ) -> Self {
        let q = |v: f64| (v * SCALE as f64).round() as i64;
        let per_tick = |v: f64| q(v) / TICK_HZ as i64;
        let has_cat = |c: &str| cats.iter().any(|x| x.eq_ignore_ascii_case(c));
        let intel = root.get("Intel");
        let i_f = |k: &str| intel.and_then(|t| t.get_f64(k)).unwrap_or(0.0);
        let i_b = |k: &str| intel.and_then(|t| t.get_bool(k)).unwrap_or(false);
        let jam = intel.and_then(|t| t.get("JamRadius"));
        let jam_f = |k: &str| jam.and_then(|t| t.get_f64(k)).unwrap_or(0.0);
        let has_enh = root.get("Enhancements").is_some();
        let radar_stealth = i_b("RadarStealth");
        let cloak = i_b("Cloak");
        let mut out = UnitTypeExt {
            intel: IntelSpec {
                radar_stealth,
                cloak,
                stealth_field_radius: q(i_f("RadarStealthFieldRadius").max(0.0)),
                cloak_field_radius: q(i_f("CloakFieldRadius").max(0.0)),
                omni_radius: q(i_f("OmniRadius").max(0.0)),
                jammer_blips: i_f("JammerBlips").round().clamp(0.0, 32.0) as u32,
                jam_radius: (q(jam_f("Min").max(0.0)), q(jam_f("Max").max(jam_f("Min")).max(0.0))),
                stealth_wait_ticks: if radar_stealth || cloak {
                    (i_f("StealthWaitTime").max(0.0) * TICK_HZ as f64).round() as u32
                } else {
                    0
                },
                gated: role == Role::Commander || has_enh,
            },
            ..UnitTypeExt::default()
        };

        if role.is_structure() {
            let caps = root.get("Physics").and_then(|p| p.get("BuildOnLayerCaps"));
            let layer = |k: &str| caps.and_then(|c| c.get_bool(k)).unwrap_or(false);
            out.water_only = layer("LAYER_Water") && !layer("LAYER_Land");
        }

        let transport = root.get("Transport");
        let t_f = |k: &str| transport.and_then(|t| t.get_f64(k));
        let t_class = t_f("TransportClass").unwrap_or(1.0).round() as i64;
        if domain == NavDomain::Air && has_cat("TRANSPORTATION") && t_class >= 10 {
            let (small, medium, large) = if tier >= 2 { (12, 6, 3) } else { (6, 2, 1) };
            out.transport = Some(TransportSpec {
                small,
                medium,
                large,
                class2_size: t_f("Class2AttachSize").unwrap_or(2.0).clamp(1.0, 8.0) as u8,
                class3_size: t_f("Class3AttachSize").unwrap_or(4.0).clamp(1.0, 16.0) as u8,
                carries_commander: has_cat("CANTRANSPORTCOMMANDER"),
            });
        }
        if domain == NavDomain::Air && !role.is_structure() {
            let fuel = root.get("Physics").and_then(|p| p.get_f64("FuelUseTime")).unwrap_or(0.0);
            if fuel > 0.0 {
                out.fuel_ticks = (fuel * TICK_HZ as f64).round().clamp(1.0, u32::MAX as f64 / 2.0) as u32;
                out.fuel_recharge_milli = (root
                    .get("Physics")
                    .and_then(|p| p.get_f64("FuelRechargeRate"))
                    .unwrap_or(5.0)
                    .max(0.1)
                    * 1000.0)
                    .round() as i64;
            }
        }
        for w in raw_weapons {
            let label = w.get_str("Label").unwrap_or("");
            if label == "DeathWeapon"
                || label == "DeathImpact"
                || w.get_str("WeaponCategory") == Some("Death")
                || w.get_bool("FireOnDeath").unwrap_or(false)
            {
                let d = w.get_f64("Damage").unwrap_or(0.0).max(w.get_f64("NukeInnerRingDamage").unwrap_or(0.0));
                let r = w.get_f64("DamageRadius").unwrap_or(0.0).max(w.get_f64("NukeInnerRingRadius").unwrap_or(0.0));
                if d > 0.0 && r > 0.0 && q(d) > out.death_damage {
                    out.death_damage = q(d);
                    out.death_radius = q(r);
                }
            }
        }
        if let Some(st) = root.get("Buffs").and_then(|b| b.get("Stun"))
            && st.get_str("BuffType") == Some("STUN")
        {
            let add = st.get("Add");
            let flag = |k: &str| add.and_then(|a| a.get_bool(k)).unwrap_or(false);
            let ticks = (st.get_f64("Duration").unwrap_or(0.0).max(0.0) * TICK_HZ as f64).round() as u32;
            let radius = q(st.get_f64("Radius").unwrap_or(0.0).max(0.0));
            if ticks > 0 {
                out.stun = Some((ticks, radius, flag("OnImpact"), flag("OnDeath")));
            }
        }
        out.untargetable = has_cat("UNTARGETABLE");
        if role.is_structure() {
            out.hatch = root
                .get("Economy")
                .and_then(|e| e.get_str("BuildUnit"))
                .map(|id| id.to_ascii_uppercase());
        }
        if role == Role::EngStation {
            out.pods = root
                .get("Economy")
                .and_then(|e| e.get("EngineeringPods"))
                .map(|p| p.table_list())
                .unwrap_or_default()
                .iter()
                .filter_map(|p| p.get_str("PodUnitID"))
                .map(|id| id.to_ascii_uppercase())
                .collect();
        }
        if role == Role::AirStaging || has_cat("AIRSTAGINGPLATFORM") {
            let ai = root.get("AI");
            let ai_f = |k: &str, d: f64| ai.and_then(|a| a.get_f64(k)).unwrap_or(d);
            out.staging = Some(StagingSpec {
                slots: t_f("DockingSlots").unwrap_or(4.0).clamp(1.0, 32.0) as u8,
                refuel_multiplier: ai_f("RefuelingMultiplier", 50.0).clamp(1.0, 1000.0).round() as i64,
                repair_permille: (t_f("RepairRate").unwrap_or(0.1).clamp(0.0, 1.0) * 1000.0).round() as i64,
                repair_energy: (ai_f("RepairConsumeEnergy", 5.0).max(0.0) * SCALE as f64).round() as i64,
                repair_mass: (ai_f("RepairConsumeMass", 0.5).max(0.0) * SCALE as f64).round() as i64,
                scan_radius: (ai_f("StagingPlatformScanRadius", 300.0).clamp(10.0, 2_000.0) * SCALE as f64).round() as i64,
            });
        }
        if matches!(domain, NavDomain::Land | NavDomain::Amphibious)
            && !role.is_structure()
            && role != Role::Experimental
            && !has_cat("EXPERIMENTAL")
            && !has_cat("CANNOTBETRANSPORTED")
            && (1..=3).contains(&t_class)
        {
            out.transport_class = t_class as u8;
        }

        if role == Role::MissileDefense {
            for w in raw_weapons {
                let only = w.get_str("TargetRestrictOnlyAllow").unwrap_or("").to_ascii_uppercase();
                let label = w.get_str("Label").unwrap_or("");
                let counter = w.get_str("RangeCategory") == Some("UWRC_Countermeasure") || label.contains("AntiNuke");
                if !counter {
                    continue;
                }
                let strategic = only.contains("STRATEGIC") || label.contains("AntiNuke");
                if strategic {
                    out.intercept_strategic = true;
                    out.ammo_cap = out.ammo_cap.max(w.get_f64("MaxProjectileStorage").unwrap_or(5.0).clamp(1.0, 20.0) as u32);
                } else {
                    out.intercept_tactical = true;
                }
                let rof = w.get_f64("RateOfFire").unwrap_or(0.5).max(0.01);
                let reset = combat::cooldown_reset_ticks(TICK_HZ, q(rof));
                out.intercept_reset = out.intercept_reset.max(reset.min(600));
            }
            if !out.intercept_strategic && !out.intercept_tactical {
                if tier >= 3 {
                    out.intercept_strategic = true;
                    out.ammo_cap = 5;
                } else {
                    out.intercept_tactical = true;
                }
            }
            if out.intercept_reset == 0 {
                out.intercept_reset = 20;
            }
        }

        if let Some(enh) = root.get("Enhancements") {
            for (name, e) in enh.keyed_tables() {
                if name == "Slots" {
                    continue;
                }
                let f = |k: &str| e.get_f64(k).unwrap_or(0.0);
                let shield_hp = q(f("ShieldMaxHealth").max(0.0));
                let shield_size = e.get_f64("ShieldSize");
                let personal = e.get_bool("PersonalShield").unwrap_or(false)
                    || (shield_hp > 0 && shield_size.is_none_or(|s| s <= 4.0));
                let cat_adds = e.get_str("BuildableCategoryAdds").unwrap_or("").to_ascii_uppercase();
                let cloak_enh = name.contains("Cloak");
                let stealth_enh = name.contains("Stealth") || cloak_enh;
                out.enhancements.push(EnhancementSpec {
                    name: name.to_string(),
                    display: strip_loc(e.get_str("Name").unwrap_or(name)),
                    slot: e.get_str("Slot").unwrap_or("Back").to_string(),
                    prerequisite: e.get_str("Prerequisite").filter(|p| !p.is_empty()).map(str::to_string),
                    removes: e.get("RemoveEnhancements").map(Value::string_list).unwrap_or_default(),
                    cost_mass: q(f("BuildCostMass")).max(1),
                    cost_energy: q(f("BuildCostEnergy")).max(1),
                    build_work: (f("BuildTime").max(0.1) * TICK_HZ as f64).round().max(1.0) as i64,
                    new_build_rate: f("NewBuildRate").round().max(0.0) as i64,
                    add_health: q(f("NewHealth").max(0.0)),
                    regen_per_tick: per_tick(f("NewRegenRate").max(0.0)),
                    engineering_tier: if cat_adds.contains("TIER3") {
                        3
                    } else if cat_adds.contains("TIER2") {
                        2
                    } else {
                        0
                    },
                    new_max_radius: q(f("NewMaxRadius").max(0.0)),
                    new_rate_of_fire_milli: q(f("NewRateOfFire").max(0.0)),
                    damage_add: q(f("ZephyrDamageMod") + f("NewDamageMod") + f("AdditionalDamage")),
                    new_damage_radius: q(f("NewDamageRadius").max(0.0)),
                    vision_radius: q(f("NewVisionRadius").max(0.0)),
                    omni_radius: q(f("NewOmniRadius").max(0.0)),
                    mass_per_tick: per_tick(f("ProductionPerSecondMass").max(0.0)),
                    energy_per_tick: per_tick(f("ProductionPerSecondEnergy").max(0.0)),
                    upkeep_per_tick: per_tick(f("MaintenanceConsumptionPerSecondEnergy").max(0.0)),
                    shield_max_hp: shield_hp,
                    shield_radius: if shield_hp > 0 {
                        q((shield_size.unwrap_or(3.0) * 0.5).max(1.5))
                    } else {
                        0
                    },
                    shield_regen_per_tick: if shield_hp > 0 {
                        per_tick(e.get_f64("ShieldRegenRate").unwrap_or(40.0)).max(1)
                    } else {
                        0
                    },
                    personal_shield: personal,
                    radar_stealth: stealth_enh,
                    cloak: cloak_enh,
                    jammer_radius: q(f("NewJammerRadius").max(0.0)),
                });
            }
        }
        out
    }

    pub fn enhancement(&self, name: &str) -> Option<&EnhancementSpec> {
        self.enhancements.iter().find(|e| e.name == name)
    }
}

/// Counts `(small, medium, large)` transport attach points from `.scm` bone names
/// (`Attachpoint*`, `*Attachpoint_Med*`, `*Attachpoint_Lrg*`).
pub fn transport_attach_counts<'a>(bones: impl IntoIterator<Item = &'a str>) -> (u8, u8, u8) {
    let (mut s, mut m, mut l) = (0u32, 0u32, 0u32);
    for b in bones {
        let lb = b.to_ascii_lowercase();
        if !lb.contains("attachpoint") {
            continue;
        }
        if lb.contains("lrg") {
            l += 1;
        } else if lb.contains("med") {
            m += 1;
        } else {
            s += 1;
        }
    }
    (s.min(64) as u8, m.min(64) as u8, l.min(64) as u8)
}

/// One playable faction with full T1–T4 combined-arms unit and structure rosters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Faction {
    pub key: String,
    pub commander: String,
    pub factory: String,
    pub extractor: String,
    pub generator: String,
    pub point_defense: String,
    pub radar: String,
    pub engineer: String,
    pub land_units: Vec<String>,
    // Phase 12 extended rosters
    pub air_factory: String,
    pub naval_factory: String,
    pub hydro: String,
    pub mass_storage: String,
    pub energy_storage: String,
    pub shield: String,
    pub anti_air: String,
    pub tac_missile: String,
    pub strat_missile: String,
    pub tac_defense: String,
    pub strat_defense: String,
    pub experimental: String,
    pub t2_land_factory: String,
    pub t3_land_factory: String,
    pub t2_mex: String,
    pub t3_mex: String,
    pub t2_pgen: String,
    pub t3_pgen: String,
    pub t3_shield: String,
    pub t2_engineer: String,
    pub t3_engineer: String,
    pub t2_land_units: Vec<String>,
    pub t3_land_units: Vec<String>,
    pub air_units: Vec<String>,
    pub naval_units: Vec<String>,
}

impl Faction {
    pub fn can_build(&self, unit_type: &str) -> bool {
        [
            &self.factory,
            &self.air_factory,
            &self.naval_factory,
            &self.extractor,
            &self.generator,
            &self.hydro,
            &self.mass_storage,
            &self.energy_storage,
            &self.point_defense,
            &self.radar,
            &self.shield,
            &self.t3_shield,
            &self.anti_air,
            &self.tac_missile,
            &self.strat_missile,
            &self.tac_defense,
            &self.strat_defense,
            &self.t2_pgen,
            &self.t3_pgen,
            &self.experimental,
        ]
        .iter()
        .any(|t| !t.is_empty() && **t == unit_type)
    }

    pub fn can_produce(&self, unit_type: &str) -> bool {
        self.engineer == unit_type
            || (!self.t2_engineer.is_empty() && self.t2_engineer == unit_type)
            || (!self.t3_engineer.is_empty() && self.t3_engineer == unit_type)
            || self.land_units.iter().any(|u| u == unit_type)
            || self.t2_land_units.iter().any(|u| u == unit_type)
            || self.t3_land_units.iter().any(|u| u == unit_type)
            || self.air_units.iter().any(|u| u == unit_type)
            || self.naval_units.iter().any(|u| u == unit_type)
    }
}

/// Core 10 T1 blueprint ids per faction `[commander, factory, mex, pgen, pd, radar, engineer, tank, bot, artillery]`.
pub const FACTION_TABLE: &[(&str, &[&str])] = &[
    (
        "uef",
        &[
            "UEL0001", "UEB0101", "UEB1103", "UEB1101", "UEB2101", "UEB3101", "UEL0105", "UEL0201",
            "UEL0106", "UEL0103",
        ],
    ),
    (
        "cybran",
        &[
            "URL0001", "URB0101", "URB1103", "URB1101", "URB2101", "URB3101", "URL0105", "URL0107",
            "URL0106", "URL0103",
        ],
    ),
    (
        "aeon",
        &[
            "UAL0001", "UAB0101", "UAB1103", "UAB1101", "UAB2101", "UAB3101", "UAL0105", "UAL0201",
            "UAL0106", "UAL0103",
        ],
    ),
    (
        "seraphim",
        &[
            "XSL0001", "XSB0101", "XSB1103", "XSB1101", "XSB2101", "XSB3101", "XSL0105", "XSL0201",
            "XSL0101", "XSL0103",
        ],
    ),
];

struct ExtendedFactionDef {
    air_fac: &'static str,
    naval_fac: &'static str,
    hydro: &'static str,
    m_sto: &'static str,
    e_sto: &'static str,
    shield: &'static str,
    aa: &'static str,
    tac_msl: &'static str,
    strat_msl: &'static str,
    tmd: &'static str,
    smd: &'static str,
    exp: &'static str,
    t2_fac: &'static str,
    t3_fac: &'static str,
    t2_air_fac: &'static str,
    t3_air_fac: &'static str,
    t2_nav_fac: &'static str,
    t3_nav_fac: &'static str,
    t2_mex: &'static str,
    t3_mex: &'static str,
    t2_pgen: &'static str,
    t3_pgen: &'static str,
    t3_shield: &'static str,
    t2_aa: &'static str,
    t2_eng: &'static str,
    t3_eng: &'static str,
    t2_land: &'static [&'static str],
    t3_land: &'static [&'static str],
    air: &'static [&'static str],
    naval: &'static [&'static str],
}

const EXTENDED_FACTIONS: [ExtendedFactionDef; 4] = [
    ExtendedFactionDef {
        air_fac: "UEB0102",
        naval_fac: "UEB0103",
        hydro: "UEB1102",
        m_sto: "UEB1106",
        e_sto: "UEB1105",
        shield: "UEB4202",
        aa: "UEB2104",
        tac_msl: "UEB2108",
        strat_msl: "UEB2305",
        tmd: "UEB4201",
        smd: "UEB4302",
        exp: "UEL0401",
        t2_fac: "UEB0201",
        t3_fac: "UEB0301",
        t2_air_fac: "UEB0202",
        t3_air_fac: "UEB0302",
        t2_nav_fac: "UEB0203",
        t3_nav_fac: "UEB0303",
        t2_mex: "UEB1202",
        t3_mex: "UEB1302",
        t2_pgen: "UEB1201",
        t3_pgen: "UEB1301",
        t3_shield: "UEB4301",
        t2_aa: "UEB2204",
        t2_eng: "UEL0208",
        t3_eng: "UEL0309",
        t2_land: &["UEL0202"],
        t3_land: &["UEL0303"],
        air: &["UEA0102", "UEA0103", "UEA0203", "UEA0304"],
        naval: &["UES0103", "UES0201", "UES0302"],
    },
    ExtendedFactionDef {
        air_fac: "URB0102",
        naval_fac: "URB0103",
        hydro: "URB1102",
        m_sto: "URB1106",
        e_sto: "URB1105",
        shield: "URB4202",
        aa: "URB2104",
        tac_msl: "URB2108",
        strat_msl: "URB2305",
        tmd: "URB4201",
        smd: "URB4302",
        exp: "URL0402",
        t2_fac: "URB0201",
        t3_fac: "URB0301",
        t2_air_fac: "URB0202",
        t3_air_fac: "URB0302",
        t2_nav_fac: "URB0203",
        t3_nav_fac: "URB0303",
        t2_mex: "URB1202",
        t3_mex: "URB1302",
        t2_pgen: "URB1201",
        t3_pgen: "URB1301",
        t3_shield: "URB4204",
        t2_aa: "URB2204",
        t2_eng: "URL0208",
        t3_eng: "URL0309",
        t2_land: &["URL0202"],
        t3_land: &["URL0303"],
        air: &["URA0102", "URA0103", "URA0203", "URA0304"],
        naval: &["URS0103", "URS0201", "URS0302"],
    },
    ExtendedFactionDef {
        air_fac: "UAB0102",
        naval_fac: "UAB0103",
        hydro: "UAB1102",
        m_sto: "UAB1106",
        e_sto: "UAB1105",
        shield: "UAB4202",
        aa: "UAB2104",
        tac_msl: "UAB2108",
        strat_msl: "UAB2305",
        tmd: "UAB4201",
        smd: "UAB4302",
        exp: "UAL0401",
        t2_fac: "UAB0201",
        t3_fac: "UAB0301",
        t2_air_fac: "UAB0202",
        t3_air_fac: "UAB0302",
        t2_nav_fac: "UAB0203",
        t3_nav_fac: "UAB0303",
        t2_mex: "UAB1202",
        t3_mex: "UAB1302",
        t2_pgen: "UAB1201",
        t3_pgen: "UAB1301",
        t3_shield: "UAB4301",
        t2_aa: "UAB2204",
        t2_eng: "UAL0208",
        t3_eng: "UAL0309",
        t2_land: &["UAL0202"],
        t3_land: &["UAL0303"],
        air: &["UAA0102", "UAA0103", "UAA0203", "UAA0304"],
        naval: &["UAS0103", "UAS0201", "UAS0302"],
    },
    ExtendedFactionDef {
        air_fac: "XSB0102",
        naval_fac: "XSB0103",
        hydro: "XSB1102",
        m_sto: "XSB1106",
        e_sto: "XSB1105",
        shield: "XSB4202",
        aa: "XSB2104",
        tac_msl: "XSB2108",
        strat_msl: "XSB2305",
        tmd: "XSB4201",
        smd: "XSB4302",
        exp: "XSL0401",
        t2_fac: "XSB0201",
        t3_fac: "XSB0301",
        t2_air_fac: "XSB0202",
        t3_air_fac: "XSB0302",
        t2_nav_fac: "XSB0203",
        t3_nav_fac: "XSB0303",
        t2_mex: "XSB1202",
        t3_mex: "XSB1302",
        t2_pgen: "XSB1201",
        t3_pgen: "XSB1301",
        t3_shield: "XSB4301",
        t2_aa: "XSB2204",
        t2_eng: "XSL0208",
        t3_eng: "XSL0309",
        t2_land: &["XSL0202"],
        t3_land: &["XSL0303"],
        air: &["XSA0102", "XSA0103", "XSA0203", "XSA0304"],
        naval: &["XSS0103", "XSS0201", "XSS0302"],
    },
];

/// Phase 18 per-faction additions (same order as [`FACTION_TABLE`]):
/// `(factory-producible units, engineer-buildable structures)`.
/// T1/T2 air transports, the Cybran mobile stealth field and stealth field generators.
const PHASE18_EXTRAS: [(&[&str], &[&str]); 4] = [
    (&["UEA0107", "UEA0104"], &["UEB4203"]),
    (&["URA0107", "URA0104", "URL0306"], &["URB4203"]),
    (&["UAA0107", "UAA0104"], &["UAB4203"]),
    (&["XSA0107", "XSA0104"], &["XSB4203"]),
];

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    pub types: BTreeMap<String, UnitType>,
    pub factions: Vec<Faction>,
    /// Phase 18: extra factory-producible ids per faction index (transports, ...).
    pub extra_producible: Vec<Vec<String>>,
    /// Phase 18: extra engineer-buildable ids per faction index (stealth field generators).
    pub extra_buildable: Vec<Vec<String>>,
    /// Phase 24: data-driven roster derived from every blueprint's build categories.
    pub roster: Arc<roster::Roster>,
    /// Phase 24: enabled blueprint mods (recorded in replay / save / handshake headers).
    pub mods: Vec<mods::ModIdentity>,
}

impl Catalog {
    /// [`Faction::can_build`] plus Phase 18 extras.
    pub fn faction_can_build(&self, faction: usize, unit_type: &str) -> bool {
        self.factions.get(faction).is_some_and(|f| f.can_build(unit_type))
            || self
                .extra_buildable
                .get(faction)
                .is_some_and(|v| v.iter().any(|t| t == unit_type) && self.types.contains_key(unit_type))
    }

    /// [`Faction::can_produce`] plus Phase 18 extras.
    pub fn faction_can_produce(&self, faction: usize, unit_type: &str) -> bool {
        self.factions.get(faction).is_some_and(|f| f.can_produce(unit_type))
            || self
                .extra_producible
                .get(faction)
                .is_some_and(|v| v.iter().any(|t| t == unit_type) && self.types.contains_key(unit_type))
    }

    /// Air transports a faction can produce.
    #[allow(dead_code)]
    pub fn faction_transports(&self, faction: usize) -> Vec<String> {
        self.extra_producible
            .get(faction)
            .map(|v| {
                v.iter()
                    .filter(|t| self.types.get(*t).is_some_and(|u| u.ext.transport.is_some()))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn faction_index(&self, key: &str) -> Option<usize> {
        self.factions
            .iter()
            .position(|f| f.key.eq_ignore_ascii_case(key))
    }

    pub fn get(&self, id: &str) -> Option<&UnitType> {
        self.types.get(id)
    }

    /// Phase 24 build card: everything blueprint `builder_type` can build (engineers, ACUs,
    /// SACUs: `BuildKind::Build`), produce (factories, gates: `BuildKind::Produce`) or
    /// upgrade into (`BuildKind::Upgrade`), with tech level and card group. Sorted by
    /// `(kind, group, tier, id)`. ACU options include every engineering-suite tier; filter
    /// with [`Skirmish::build_options`] for what a given unit may issue right now.
    pub fn build_options(&self, builder_type: &str) -> &[roster::BuildOption] {
        self.roster.options_of(builder_type)
    }

    /// Phase 24: `builder_type` lists `unit_type` as a structure / experimental to place.
    pub fn builder_can_build(&self, builder_type: &str, unit_type: &str) -> bool {
        self.roster.can(builder_type, unit_type, roster::BuildKind::Build) && self.types.contains_key(unit_type)
    }

    /// Phase 24: factory blueprint `factory_type` can queue `unit_type`.
    pub fn builder_can_produce(&self, factory_type: &str, unit_type: &str) -> bool {
        self.roster.can(factory_type, unit_type, roster::BuildKind::Produce) && self.types.contains_key(unit_type)
    }

    /// Phase 24: every playable unit type of faction index `faction`, ordered by
    /// `(tier, id)`.
    pub fn faction_roster(&self, faction: usize) -> Vec<&UnitType> {
        let key = self.factions.get(faction).map(|f| f.key.as_str()).unwrap_or("");
        let Some(ri) = roster::FACTION_CATS.iter().position(|(k, _)| *k == key) else {
            return Vec::new();
        };
        let mut v: Vec<&UnitType> = self
            .roster
            .playable
            .get(ri)
            .map(|ids| ids.iter().filter_map(|id| self.types.get(id)).collect())
            .unwrap_or_default();
        v.sort_by(|a, b| (a.tier, &a.id).cmp(&(b.tier, &b.id)));
        v
    }

    /// Phase 24: build-card group of a unit type (economy, defense, land, air, ...).
    pub fn build_group(&self, id: &str) -> Option<roster::BuildGroup> {
        self.roster.entries.get(id).and_then(|e| e.group)
    }

    /// Phase 24: derives the data-driven roster from parsed blueprints and adds every
    /// playable type that the legacy tables did not load.
    fn extend_from_roster(&mut self, roots: &BTreeMap<String, Value>) {
        let infos: BTreeMap<String, roster::BpInfo> = roots
            .iter()
            .map(|(id, r)| (id.clone(), roster::BpInfo::from_blueprint(id, r)))
            .collect();
        let fixed: BTreeMap<String, Role> = self.types.iter().map(|(id, t)| (id.to_ascii_uppercase(), t.role)).collect();
        let r = roster::derive(&infos, &fixed);
        for (id, e) in &r.entries {
            if e.status != roster::Status::Playable || self.types.contains_key(id) {
                continue;
            }
            let (Some(role), Some(root)) = (e.role, roots.get(id)) else {
                continue;
            };
            let mut t = UnitType::from_blueprint(id, role, root);
            if role == Role::Generator {
                // Mass fabricators trade energy for mass.
                let upkeep = root
                    .get("Economy")
                    .and_then(|e| e.get_f64("MaintenanceConsumptionPerSecondEnergy"))
                    .unwrap_or(0.0)
                    .max(0.0);
                t.energy_per_tick -= (upkeep * SCALE as f64).round() as i64 / TICK_HZ as i64;
            }
            self.types.insert(id.clone(), t);
        }
        // Phase 32: mobile units with produce / build options become mobile factories /
        // mobile builders; eggs' hatchlings must be loadable.
        for t in self.types.values_mut() {
            if t.role.is_structure() || t.role.is_builder() {
                continue;
            }
            let opts = r.options_of(&t.id.to_ascii_uppercase());
            t.ext.mobile_factory = opts.iter().any(|o| o.kind == roster::BuildKind::Produce);
            t.ext.mobile_builder = opts.iter().any(|o| o.kind == roster::BuildKind::Build);
        }
        let hatchlings: Vec<String> = self.types.values().filter_map(|t| t.ext.hatch.clone()).collect();
        for id in hatchlings {
            if !self.types.contains_key(&id)
                && let Some(root) = roots.get(&id)
                && let Some(Ok(role)) = Some(roster::infer_role(&infos[&id]))
            {
                self.types.insert(id.clone(), UnitType::from_blueprint(&id, role, root));
            }
        }
        // Phase 33: an orbital launch centre's satellite is its faction's `SATELLITE` unit.
        let launchers: Vec<String> = self.types.values().filter(|t| t.role == Role::Orbital).map(|t| t.id.clone()).collect();
        for lid in launchers {
            let fac = infos.get(&lid.to_ascii_uppercase()).and_then(roster::BpInfo::faction);
            let sat = infos
                .values()
                .filter(|b| b.has("SATELLITE") && b.faction() == fac)
                .map(|b| b.id.clone())
                .min();
            if let Some(sid) = sat
                && let Some(root) = roots.get(&sid)
            {
                if !self.types.contains_key(&sid) {
                    let role = roster::infer_role(&infos[&sid]).unwrap_or(Role::Mobile);
                    self.types.insert(sid.clone(), UnitType::from_blueprint(&sid, role, root));
                }
                self.types.get_mut(&lid).unwrap().ext.satellite = Some(sid);
            }
        }
        // Phase 31: pod drones are spawned by their stations, never built: load them too.
        let pods: Vec<String> = self.types.values().flat_map(|t| t.ext.pods.iter().cloned()).collect();
        for id in pods {
            if !self.types.contains_key(&id)
                && let Some(root) = roots.get(&id)
            {
                self.types.insert(id.clone(), UnitType::from_blueprint(&id, Role::Engineer, root));
            }
        }
        self.roster = Arc::new(r);
    }

    /// Load every faction in [`FACTION_TABLE`] and extended Phase 12 units from `units.scd`,
    /// apply the process-wide enabled mods ([`mods::active`]) and add the Phase 24
    /// data-driven roster.
    pub fn load(game_dir: &Path) -> Result<Self, String> {
        Self::load_with_mods(game_dir, mods::active()?)
    }

    /// Every `*_unit.bp` in `units.scd`, keyed by upper-case blueprint id.
    pub fn read_blueprints(game_dir: &Path) -> Result<BTreeMap<String, Value>, String> {
        let units_scd = scd::gamedata_dir(game_dir).join("units.scd");
        let files = scd::read_scd_matching(&units_scd, |n| n.to_ascii_lowercase().ends_with("_unit.bp"))
            .map_err(|e| format!("cannot read {}: {e}", units_scd.display()))?;
        let mut roots = BTreeMap::new();
        for (name, bytes) in files {
            let id = blueprint::unit_id_from_path(&name).to_ascii_uppercase();
            if let Ok(root) = blueprint::parse_blueprint(&String::from_utf8_lossy(&bytes)) {
                roots.entry(id).or_insert(root);
            }
        }
        Ok(roots)
    }

    /// [`Catalog::load`] with an explicit mod list (applied in order).
    pub fn load_with_mods(game_dir: &Path, mod_list: &[mods::Mod]) -> Result<Self, String> {
        let mut roots = Self::read_blueprints(game_dir)?;
        mods::apply(&mut roots, mod_list)?;
        let mut cat = Self::from_roots(game_dir, &roots)?;
        cat.mods = mod_list.iter().map(mods::Mod::identity).collect();
        Ok(cat)
    }

    /// Builds the catalog from parsed (and possibly modded) blueprints.
    pub fn from_roots(game_dir: &Path, roots: &BTreeMap<String, Value>) -> Result<Self, String> {
        let units_scd = scd::gamedata_dir(game_dir).join("units.scd");
        let mut cat = Catalog::default();

        let load_id = |types: &mut BTreeMap<String, UnitType>, id: &str, role: Role| -> Result<(), String> {
            if types.contains_key(id) {
                return Ok(());
            }
            let root = roots
                .get(&id.to_ascii_uppercase())
                .ok_or_else(|| format!("cannot read units/{id}/{id}_unit.bp: not found in {}", units_scd.display()))?;
            types.insert(id.to_string(), UnitType::from_blueprint(id, role, root));
            Ok(())
        };

        for (fi, (key, ids)) in FACTION_TABLE.iter().enumerate() {
            for (i, id) in ids.iter().enumerate() {
                let role = match i {
                    0 => Role::Commander,
                    1 => Role::Factory,
                    2 => Role::Extractor,
                    3 => Role::Generator,
                    4 => Role::PointDefense,
                    5 => Role::Radar,
                    6 => Role::Engineer,
                    _ => Role::Mobile,
                };
                load_id(&mut cat.types, id, role)?;
            }

            let ext = &EXTENDED_FACTIONS[fi];
            for (id, role) in [
                (ext.air_fac, Role::AirFactory),
                (ext.naval_fac, Role::NavalFactory),
                (ext.hydro, Role::Hydro),
                (ext.m_sto, Role::MassStorage),
                (ext.e_sto, Role::EnergyStorage),
                (ext.shield, Role::Shield),
                (ext.aa, Role::AntiAir),
                (ext.tac_msl, Role::MissileSilo),
                (ext.strat_msl, Role::MissileSilo),
                (ext.tmd, Role::MissileDefense),
                (ext.smd, Role::MissileDefense),
                (ext.exp, Role::Experimental),
                (ext.t2_fac, Role::Factory),
                (ext.t3_fac, Role::Factory),
                (ext.t2_air_fac, Role::AirFactory),
                (ext.t3_air_fac, Role::AirFactory),
                (ext.t2_nav_fac, Role::NavalFactory),
                (ext.t3_nav_fac, Role::NavalFactory),
                (ext.t2_mex, Role::Extractor),
                (ext.t3_mex, Role::Extractor),
                (ext.t2_pgen, Role::Generator),
                (ext.t3_pgen, Role::Generator),
                (ext.t3_shield, Role::Shield),
                (ext.t2_aa, Role::AntiAir),
                (ext.t2_eng, Role::Engineer),
                (ext.t3_eng, Role::Engineer),
            ] {
                let _ = load_id(&mut cat.types, id, role);
            }
            for &id in ext
                .t2_land
                .iter()
                .chain(ext.t3_land.iter())
                .chain(ext.air.iter())
                .chain(ext.naval.iter())
            {
                let _ = load_id(&mut cat.types, id, Role::Mobile);
            }

            cat.factions.push(Faction {
                key: (*key).to_string(),
                commander: ids[0].to_string(),
                factory: ids[1].to_string(),
                extractor: ids[2].to_string(),
                generator: ids[3].to_string(),
                point_defense: ids[4].to_string(),
                radar: ids[5].to_string(),
                engineer: ids[6].to_string(),
                land_units: ids[7..].iter().map(|s| (*s).to_string()).collect(),
                air_factory: ext.air_fac.into(),
                naval_factory: ext.naval_fac.into(),
                hydro: ext.hydro.into(),
                mass_storage: ext.m_sto.into(),
                energy_storage: ext.e_sto.into(),
                shield: ext.shield.into(),
                anti_air: ext.aa.into(),
                tac_missile: ext.tac_msl.into(),
                strat_missile: ext.strat_msl.into(),
                tac_defense: ext.tmd.into(),
                strat_defense: ext.smd.into(),
                experimental: ext.exp.into(),
                t2_land_factory: ext.t2_fac.into(),
                t3_land_factory: ext.t3_fac.into(),
                t2_mex: ext.t2_mex.into(),
                t3_mex: ext.t3_mex.into(),
                t2_pgen: ext.t2_pgen.into(),
                t3_pgen: ext.t3_pgen.into(),
                t3_shield: ext.t3_shield.into(),
                t2_engineer: ext.t2_eng.into(),
                t3_engineer: ext.t3_eng.into(),
                t2_land_units: ext.t2_land.iter().map(|s| (*s).to_string()).collect(),
                t3_land_units: ext.t3_land.iter().map(|s| (*s).to_string()).collect(),
                air_units: ext.air.iter().map(|s| (*s).to_string()).collect(),
                naval_units: ext.naval.iter().map(|s| (*s).to_string()).collect(),
            });

            // Phase 18 extras (missing blueprints are skipped, never fatal).
            let (prod, build) = PHASE18_EXTRAS[fi];
            let mut p = Vec::new();
            for &id in prod {
                if load_id(&mut cat.types, id, Role::Mobile).is_ok() {
                    p.push(id.to_string());
                }
            }
            let mut b = Vec::new();
            for &id in build {
                if load_id(&mut cat.types, id, Role::Radar).is_ok() {
                    b.push(id.to_string());
                }
            }
            cat.extra_producible.push(p);
            cat.extra_buildable.push(b);
        }
        cat.extend_from_roster(roots);

        // Transport capacity comes from the mesh skeleton's attach points, like the engine.
        let transport_ids: Vec<String> = cat
            .types
            .values()
            .filter(|t| t.ext.transport.is_some())
            .map(|t| t.id.clone())
            .collect();
        for id in transport_ids {
            let inner = format!("units/{id}/{id}_lod0.scm");
            let Ok((_, bytes)) = scd::read_scd_file_ci(&units_scd, &inner) else {
                continue;
            };
            let Ok(mesh) = crate::scm::ScmMesh::parse(&bytes) else {
                continue;
            };
            let (s, m, l) = transport_attach_counts(mesh.bones.iter().map(|b| b.name.as_str()));
            if let Some(spec) = cat.types.get_mut(&id).and_then(|t| t.ext.transport.as_mut())
                && s > 0
            {
                spec.small = s;
                spec.medium = m;
                spec.large = l;
            }
        }
        Ok(cat)
    }
}

/// Orders a unit can hold. The queue head is the active order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnitOrder {
    Move { x: i64, y: i64 },
    AttackMove { x: i64, y: i64 },
    Attack(UnitId),
    Build { unit_type: String, x: i64, y: i64 },
    Reclaim { x: i64, y: i64 },
    Repair(UnitId),
    Assist(UnitId),
    Overcharge(UnitId),
    Upgrade(String),
    LaunchMissile { x: i64, y: i64 },
    Patrol { x: i64, y: i64 },
    /// Phase 18: walk to and board this air transport.
    Board(UnitId),
    /// Phase 18: (transport) fly to `(x, y)` and drop all cargo.
    Unload { x: i64, y: i64 },
    /// Phase 18: (ACU/SCU) install the named blueprint enhancement in place.
    Enhance(String),
    /// Phase 30: (aircraft) fly to this air staging platform, dock, refuel and repair.
    Dock(UnitId),
    /// Phase 35: keep firing at a map position (FA ground fire).
    AttackGround { x: i64, y: i64 },
    /// Phase 35: follow and protect a friendly unit.
    Guard(UnitId),
    /// Phase 36: (air transport) shuttle idle land units from `from` to `to`, forever.
    Ferry { from: (i64, i64), to: (i64, i64) },
}

/// Phase 35: automatic engagement rule of a unit (FA `RULEUCC_RetaliateToggle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FireState {
    /// Engage anything in range (FA default).
    #[default]
    FireAtWill,
    /// Only shoot back at the unit that last hit it.
    ReturnFire,
    /// Never fire automatically; explicit attack / ground-fire orders only.
    HoldFire,
}

impl FireState {
    pub fn code(self) -> u8 {
        match self {
            FireState::FireAtWill => 0,
            FireState::ReturnFire => 1,
            FireState::HoldFire => 2,
        }
    }
    pub fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(FireState::FireAtWill),
            1 => Some(FireState::ReturnFire),
            2 => Some(FireState::HoldFire),
            _ => None,
        }
    }
    pub fn next(self) -> Self {
        match self {
            FireState::FireAtWill => FireState::ReturnFire,
            FireState::ReturnFire => FireState::HoldFire,
            FireState::HoldFire => FireState::FireAtWill,
        }
    }
}

/// Player / AI input. Applied at the start of the next tick in issue order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Move { units: Vec<UnitId>, x: i64, y: i64, queue: bool },
    AttackMove { units: Vec<UnitId>, x: i64, y: i64, queue: bool },
    Attack { units: Vec<UnitId>, target: UnitId, queue: bool },
    Stop { units: Vec<UnitId> },
    Build { builder: UnitId, unit_type: String, x: i64, y: i64, queue: bool },
    Produce { factory: UnitId, unit_type: String, count: u32 },
    Rally { factory: UnitId, x: i64, y: i64 },
    Reclaim { units: Vec<UnitId>, x: i64, y: i64, queue: bool },
    Repair { units: Vec<UnitId>, target: UnitId, queue: bool },
    Assist { units: Vec<UnitId>, target: UnitId, queue: bool },
    Overcharge { unit: UnitId, target: UnitId },
    Upgrade { unit: UnitId, target_type: String },
    LaunchMissile { unit: UnitId, x: i64, y: i64 },
    FormationMove {
        units: Vec<UnitId>,
        x: i64,
        y: i64,
        facing_mrad: i64,
        spacing: i64,
        queue: bool,
    },
    Patrol { units: Vec<UnitId>, x: i64, y: i64, queue: bool },
    ToggleShield { units: Vec<UnitId>, enabled: bool },
    /// Phase 18: `units` board air transport `transport`.
    Load { units: Vec<UnitId>, transport: UnitId, queue: bool },
    /// Phase 30: aircraft dock at an own air staging platform.
    Dock { units: Vec<UnitId>, staging: UnitId, queue: bool },
    /// Phase 30: toggle automatic return to air staging on low fuel.
    SetAutoRefuel { units: Vec<UnitId>, enabled: bool },
    /// Phase 35: engagement rule.
    SetFireState { units: Vec<UnitId>, state: FireState },
    /// Phase 35: fire at a map position until given another order.
    AttackGround { units: Vec<UnitId>, x: i64, y: i64, queue: bool },
    /// Phase 35: follow and protect a friendly unit.
    Guard { units: Vec<UnitId>, target: UnitId, queue: bool },
    /// Phase 35: pause / resume construction and production.
    SetBuildPaused { units: Vec<UnitId>, paused: bool },
    /// Phase 36: move the `index`-th queued Move / AttackMove / Patrol point of each unit.
    MoveWaypoint { units: Vec<UnitId>, index: u32, x: i64, y: i64 },
    /// Phase 36: set up a ferry route for air transports.
    Ferry { transports: Vec<UnitId>, from: (i64, i64), to: (i64, i64), queue: bool },
    /// Phase 53: map ping for allies (0 alert / 1 move / 2 attack); allied AIs react.
    Ping { kind: u8, x: i64, y: i64 },
    /// Phase 18: `transports` fly to `(x, y)` and drop their cargo.
    Unload { transports: Vec<UnitId>, x: i64, y: i64, queue: bool },
    /// Phase 18: install ACU/SCU enhancement `name` (blueprint `Enhancements` key).
    Enhance { unit: UnitId, name: String },
    /// Phase 18: factory repeat-build toggle (finished units are re-queued).
    SetRepeat { factories: Vec<UnitId>, enabled: bool },
    /// Phase 23: remove up to `count` queued `unit_type` entries from the back of a factory
    /// queue; when fewer are queued, the in-progress job of that type is cancelled too.
    CancelProduce { factory: UnitId, unit_type: String, count: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildJob {
    pub unit_type: String,
    pub x: i64,
    pub y: i64,
    pub project: ConstructionProject,
    pub mass_point: Option<usize>,
}

impl BuildJob {
    /// Progress in per-mille.
    pub fn progress_permille(&self) -> i64 {
        (self.project.progress.min(self.project.work_total) * 1000) / self.project.work_total.max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeaponState {
    pub damage: i64,
    pub range_sq: i128,
    pub reset: u32,
    pub cooldown: u32,
    pub splash_radius: i64,
    pub muzzle_velocity: i64,
    pub anti_air: bool,
    pub anti_surface: bool,
    /// Phase 29: copied from [`WeaponType`].
    pub turret_yaw_speed: i64,
    pub firing_tolerance: i64,
    pub beam: bool,
    pub lead_target: bool,
    pub emp: bool,
    pub kamikaze: bool,
    pub stun: Option<(u32, i64)>,
    /// Phase 29: current turret yaw (mrad, `atan2(dx, dz)` convention).
    pub yaw: i64,
}

impl WeaponState {
    pub fn from_type(w: &WeaponType) -> Self {
        Self {
            damage: w.damage,
            range_sq: (w.range as i128) * (w.range as i128),
            reset: w.reset,
            cooldown: 0,
            splash_radius: w.splash_radius,
            muzzle_velocity: w.muzzle_velocity,
            anti_air: w.anti_air,
            anti_surface: w.anti_surface,
            turret_yaw_speed: w.turret_yaw_speed,
            firing_tolerance: w.firing_tolerance,
            beam: w.beam,
            lead_target: w.lead_target,
            emp: w.emp,
            kamikaze: w.kamikaze,
            stun: w.stun,
            yaw: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Unit {
    pub id: UnitId,
    pub army: ArmyId,
    pub type_id: String,
    pub role: Role,
    pub tier: u8,
    pub domain: NavDomain,
    pub pos: Vec2,
    pub altitude: i64,
    pub hp: i64,
    pub max_hp: i64,
    pub alive: bool,
    pub weapons: Vec<WeaponState>,
    pub speed: i64,
    pub radius: i64,
    pub max_range: i64,
    pub can_fire_aa: bool,
    pub can_fire_surface: bool,
    pub vision_radius: i64,
    pub radar_radius: i64,
    pub shield_hp: i64,
    pub shield_max_hp: i64,
    pub shield_radius: i64,
    pub shield_on: bool,
    pub shield_recharge_cd: u32,
    pub vet_kills: u32,
    pub vet_level: u8,
    pub hydro_point: Option<usize>,
    pub silo_missiles: u32,
    pub silo_progress: i64,
    pub adjacency_bonus_pct: i64,
    pub turret_yaw_mrad: i64,
    pub moving: bool,
    pub orders: VecDeque<UnitOrder>,
    path: VecDeque<(i64, i64)>,
    path_goal: Option<(i64, i64)>,
    pub build: Option<BuildJob>,
    pub production: VecDeque<String>,
    pub rally: Option<(i64, i64)>,
    pub mass_point: Option<usize>,
    /// Last target fired at and the tick it happened (render tracers).
    pub last_shot: Option<(UnitId, u64)>,
    last_hit_by: Option<ArmyId>,
    last_hit_by_unit: Option<UnitId>,
    // ----- Phase 18 -----
    /// Air transport carrying this unit (removed from the spatial grid while set).
    pub carried_by: Option<UnitId>,
    /// Passengers of an air transport, in boarding order.
    pub cargo: Vec<UnitId>,
    /// Factory repeat-build: completed units are re-queued.
    pub repeat_queue: bool,
    pub enh: EnhancementState,
    /// Effective intel state this tick (blueprint, fields, enhancements, stalls).
    pub radar_stealth: bool,
    pub cloaked: bool,
    pub omni_radius: i64,
    /// Shield behaviour (bubble vs. personal) and per-tick upkeep / regen.
    pub shield_personal: bool,
    pub shield_upkeep: i64,
    pub shield_regen: i64,
    /// Missile defense: ticks until the next interception.
    pub intercept_cd: u32,
    still_ticks: u32,
    /// Armies (bit = army id) whose omni sensors (or allies') reveal this cloaked unit.
    omni_mask: u8,
    /// Phase 29: hit-box height (milli, blueprint `SizeY`) for line-of-fire tests.
    pub height: i64,
    /// Phase 29: ground displacement during the last tick (milli), used to lead targets.
    pub vel: (i64, i64),
    /// Phase 30: remaining flight time (ticks; only meaningful when the type has fuel).
    pub fuel: u32,
    /// Phase 30: return to the nearest free air staging platform when fuel runs low.
    pub auto_refuel: bool,
    /// Phase 31: the engineering station this pod drone belongs to.
    pub drone_of: Option<UnitId>,
    /// Phase 31: (station) ticks spent rebuilding a missing pod drone.
    pub pod_timer: u32,
    /// Phase 33: ticks left stunned (no moving, firing or building).
    pub stunned: u32,
    /// Phase 33: never auto- or manually targeted (orbital satellites).
    pub untargetable: bool,
    /// Phase 35: automatic engagement rule.
    pub fire_state: FireState,
    /// Phase 35: construction / production paused.
    pub build_paused: bool,
    /// Phase 54: cached type economy `[mass/tick, energy/tick, mass storage, energy
    /// storage]` (refreshed on upgrade) so per-tick passes avoid catalog lookups.
    eco: [i64; 4],
}

impl Unit {
    #[allow(dead_code)]
    pub fn is_idle(&self) -> bool {
        self.orders.is_empty() && self.build.is_none() && self.production.is_empty()
    }

    /// Whether army `army` may target this unit (not aboard a transport, and not cloaked
    /// unless revealed by `army`'s or its allies' omni sensors).
    pub fn targetable_by(&self, army: ArmyId) -> bool {
        !self.untargetable && self.carried_by.is_none() && (!self.cloaked || (self.omni_mask >> (army & 7)) & 1 == 1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wreck {
    pub id: UnitId,
    pub type_id: String,
    pub pos: Vec2,
    pub mass_left: i64,
}

#[derive(Debug, Clone, Copy)]
struct UnitSpatialSnap {
    id: UnitId,
    army: ArmyId,
    team: u8,
    domain: NavDomain,
    x: i64,
    y: i64,
    radius: i64,
    speed: i64,
    is_struct: bool,
    /// Phase 28: wall sections are only auto-targeted when nothing else is in range.
    is_wall: bool,
    /// Phase 29: hit-box height (milli) for direct-fire collisions.
    height: i64,
    /// Phase 33: orbital satellites are never acquired as targets.
    untargetable: bool,
    can_fire_aa: bool,
    can_fire_surface: bool,
    max_range: i64,
    /// Phase 18: cloaked, and the armies whose omni reveals it.
    cloaked: bool,
    omni_mask: u8,
}

impl UnitSpatialSnap {
    /// Phase 18: cloak blocks automatic target acquisition unless omni reveals it.
    fn hidden_from(&self, army: ArmyId) -> bool {
        self.untargetable || (self.cloaked && (self.omni_mask >> (army & 7)) & 1 == 0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projectile {
    pub id: u32,
    pub army: ArmyId,
    pub start: Vec2,
    pub pos: Vec2,
    pub target: Vec2,
    pub speed: i64,
    pub damage: i64,
    pub splash_radius: i64,
    pub arc_height: i64,
    pub ticks_lived: u32,
    pub total_ticks: u32,
    pub is_overcharge: bool,
    pub is_missile: bool,
    pub is_nuke: bool,
    pub target_air: bool,
    /// Phase 29: straight-flying single-target direct-fire shell.
    pub direct: bool,
    /// Phase 29: firing unit (excluded from collisions; 0 = none).
    pub shooter: UnitId,
    /// Phase 29: current / start / aim height (milli) of a direct shell.
    pub z: i64,
    pub start_z: i64,
    pub target_z: i64,
    /// Phase 33: EMP stun (ticks, radius) applied around the impact point.
    pub stun_ticks: u32,
    pub stun_radius: i64,
}

/// Visual/audio telemetry emitted during simulation ticks (excluded from `state_hash()`).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum SimEvent {
    WeaponFired {
        from: (i64, i64, i64),
        to: (i64, i64, i64),
        army: ArmyId,
        heavy: bool,
        anti_air: bool,
    },
    Explosion {
        pos: (i64, i64, i64),
        radius: i64,
        is_nuke: bool,
    },
    AcuDied {
        pos: (i64, i64, i64),
        army: ArmyId,
    },
    BuildSpark {
        pos: (i64, i64, i64),
        army: ArmyId,
    },
    ShieldHit {
        pos: (i64, i64, i64),
        army: ArmyId,
    },
    MissileLaunched {
        pos: (i64, i64, i64),
        is_nuke: bool,
    },
    MissileIntercepted {
        pos: (i64, i64, i64),
    },
    UpgradeComplete {
        pos: (i64, i64, i64),
        army: ArmyId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Army {
    pub id: ArmyId,
    pub name: String,
    pub faction: usize,
    pub team: u8,
    pub mass: i64,
    pub energy: i64,
    pub mass_cap: i64,
    pub energy_cap: i64,
    /// Income of the last tick (milli per tick).
    pub income_mass: i64,
    pub income_energy: i64,
    pub start: (i64, i64),
    pub defeated: bool,
    pub units_built: u32,
    pub units_lost: u32,
    pub kills: u32,
    pub damage_dealt: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Victory(ArmyId),
    Draw,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmySetup {
    pub name: String,
    pub faction: usize,
    pub start: (i64, i64),
    pub team: u8,
}

/// Phase 34: a reclaimable map prop (tree, rock, log) from the `.scmap` prop list with its
/// blueprint's `ReclaimMassMax` / `ReclaimEnergyMax`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapProp {
    pub pos: (i64, i64),
    pub mass: i64,
    pub energy: i64,
    /// Original `.scmap` position bits (x, z) so the renderer can hide reclaimed props.
    pub src_xz: (u32, u32),
}

/// Phase 34: what a reclaim order is eating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReclaimKind {
    Wreck,
    Prop,
    Unit,
}

/// Phase 36: idle land units this close to a ferry's pick-up point board it.
pub const FERRY_RADIUS: i64 = 12 * SCALE;

/// Phase 34: snap radius for prop reclaim orders.
pub const PROP_SNAP: i64 = 3 * SCALE;

/// Map-derived context shared by headless runs, the AI and the viewer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MapContext {
    pub map_id: String,
    pub map_name: String,
    pub scmap: ScMap,
    pub grid: PathGrid,
    pub naval_grid: PathGrid,
    pub amphib_grid: PathGrid,
    pub size: (i64, i64),
    /// Start spots in milli-units, indexed like `ARMY_1..ARMY_n`.
    pub starts: Vec<(String, (i64, i64))>,
    /// Land mass markers in milli-units.
    pub mass_points: Vec<(i64, i64)>,
    /// Hydrocarbon markers in milli-units.
    pub hydro_points: Vec<(i64, i64)>,
    pub has_water: bool,
    /// Phase 34: reclaimable props (empty when their blueprints cannot be read).
    pub props: Vec<MapProp>,
}

impl MapContext {
    /// Phase 34: reclaimable props of `scmap`, values from the prop blueprints in `env.scd`
    /// (case-insensitive paths, cached per blueprint).
    pub fn load_props(game_dir: &Path, scmap: &ScMap) -> Vec<MapProp> {
        let Some(tt) = scmap.terrain_textures.as_ref() else {
            return Vec::new();
        };
        let env = scd::gamedata_dir(game_dir).join("env.scd");
        let mut values: BTreeMap<String, (i64, i64)> = BTreeMap::new();
        let q = |v: f64| (v * SCALE as f64).round() as i64;
        let mut out = Vec::new();
        for p in &tt.props {
            let key = p.blueprint_path.trim().trim_start_matches('/').replace('\\', "/").to_ascii_lowercase();
            let (mass, energy) = *values.entry(key.clone()).or_insert_with(|| {
                scd::read_scd_file_ci(&env, &key)
                    .ok()
                    .and_then(|(_, b)| blueprint::parse_any_blueprint(&String::from_utf8_lossy(&b)).ok().map(|(_, v)| v))
                    .and_then(|root| {
                        let e = root.get("Economy")?;
                        Some((q(e.get_f64("ReclaimMassMax").unwrap_or(0.0).max(0.0)), q(e.get_f64("ReclaimEnergyMax").unwrap_or(0.0).max(0.0))))
                    })
                    .unwrap_or((0, 0))
            });
            if mass > 0 || energy > 0 {
                out.push(MapProp {
                    pos: (q(p.position[0] as f64), q(p.position[2] as f64)),
                    mass,
                    energy,
                    src_xz: (p.position[0].to_bits(), p.position[2].to_bits()),
                });
            }
        }
        out
    }

    pub fn load(game_dir: &Path, map: &str) -> Result<Self, String> {
        let files = scenario::find_map(game_dir, map)?;
        let meta = scenario::parse_scenario_file(&files.scenario_path, &files.id)?;
        let save = files
            .save_path
            .as_deref()
            .ok_or_else(|| format!("map '{}' has no _save.lua", files.id))?;
        let markers = scenario::parse_save_markers_file(save, &meta.armies)?;
        let scmap_path = files
            .scmap_path
            .as_deref()
            .ok_or_else(|| format!("map '{}' has no .scmap", files.id))?;
        let scmap = ScMap::parse_file(scmap_path)?;
        let (grid, naval_grid, amphib_grid) =
            flowfield::build_domain_grids(&scmap, GRID_STEP, GRID_MAX_SLOPE_MILLI);
        let q = |v: f32| (v as f64 * SCALE as f64).round() as i64;
        let starts = markers
            .start_spots
            .iter()
            .map(|m| (m.name.clone(), (q(m.position[0]), q(m.position[2]))))
            .collect::<Vec<_>>();
        if starts.len() < 2 {
            return Err(format!("map '{}' has fewer than 2 start spots", files.id));
        }
        let water = scmap.water.has_water.then_some(scmap.water.elevation);
        let mass_points = markers
            .mass_points
            .iter()
            .filter(|m| water.is_none_or(|w| m.position[1] > w))
            .map(|m| (q(m.position[0]), q(m.position[2])))
            .collect();
        let hydro_points = markers
            .hydro_points
            .iter()
            .filter(|m| water.is_none_or(|w| m.position[1] > w))
            .map(|m| (q(m.position[0]), q(m.position[2])))
            .collect();
        let has_water = scmap.water.has_water && naval_grid.blocked_count() < (naval_grid.width * naval_grid.height) as usize;
        let size = (q(scmap.map_width), q(scmap.map_height));
        let props = Self::load_props(game_dir, &scmap);
        Ok(Self {
            map_id: files.id.clone(),
            map_name: meta.name.clone(),
            scmap,
            grid,
            naval_grid,
            amphib_grid,
            size,
            starts,
            mass_points,
            hydro_points,
            has_water,
            props,
        })
    }
}

/// Deterministic multi-army skirmish world.
#[derive(Debug, Clone)]
pub struct Skirmish {
    tick: u64,
    seed: u64,
    rng: u64,
    next_id: UnitId,
    next_project: u64,
    next_projectile_id: u32,
    armies: Vec<Army>,
    units: BTreeMap<UnitId, Unit>,
    wrecks: BTreeMap<UnitId, Wreck>,
    projectiles: BTreeMap<u32, Projectile>,
    grid: Arc<PathGrid>,
    naval_grid: Arc<PathGrid>,
    amphib_grid: Arc<PathGrid>,
    /// Phase 28: terrain-only land / amphibious grids; live walls are stamped on top of
    /// these into `grid` / `amphib_grid` by [`Skirmish::sync_walls`].
    base_grid: Arc<PathGrid>,
    base_amphib: Arc<PathGrid>,
    base_naval: Arc<PathGrid>,
    /// Phase 28/29: live structures (walls and, since Phase 29, every structure) per
    /// land-grid cell (row-major, `base_grid` layout).
    wall_cells: Vec<u16>,
    walls_dirty: bool,
    /// Phase 29: integer terrain heights (flat on synthetic maps).
    terrain: Arc<HeightField>,
    /// Phase 34: reclaimable map props still standing (key = index in the map's list).
    props: BTreeMap<u32, MapProp>,
    /// Phase 34: `.scmap` position bits of props reclaimed so far (render hides them).
    props_gone: Vec<(u32, u32)>,
    /// Phase 48: running mission script (triggers evaluated after every tick's cleanup).
    mission: Option<Box<crate::mission::MissionRuntime>>,
    /// Phase 53: recent map pings `(tick, army, kind, position)`, oldest first (≤ 32).
    pings: VecDeque<(u64, ArmyId, u8, (i64, i64))>,
    flow_cache: FlowFieldCache,
    spatial: Arc<SpatialGrid>,
    spatial_snap: Arc<Vec<UnitSpatialSnap>>,
    pool: Option<Arc<WorkerPool>>,
    size: (i64, i64),
    mass_points: Vec<(i64, i64)>,
    mass_claims: Vec<Option<UnitId>>,
    hydro_points: Vec<(i64, i64)>,
    hydro_claims: Vec<Option<UnitId>>,
    pending: Vec<(ArmyId, Command)>,
    events: Vec<SimEvent>,
    /// Independent copy of `events` for the audio engine (so VFX and audio can both drain).
    audio_events: Vec<SimEvent>,
    catalog: Catalog,
    shots_fired: u64,
    kills: u64,
    paths_solved: u64,
    commands_applied: u64,
    commands_rejected: u64,
    result: Option<GameResult>,
    /// Phase 18 game options (victory condition, unit cap).
    options: GameOptions,
}

/// Integer square root (floor) for non-negative `i128`.
pub fn isqrt(v: i128) -> i64 {
    if v <= 0 {
        return 0;
    }
    let mut x = (v as f64).sqrt() as i128;
    while x * x > v {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= v {
        x += 1;
    }
    x as i64
}

fn dist_sq(a: (i64, i64), b: (i64, i64)) -> i128 {
    combat::dist_sq(a.0, a.1, b.0, b.1)
}

#[allow(dead_code)]
impl Skirmish {
    pub fn new(
        catalog: Catalog,
        grid: PathGrid,
        size: (i64, i64),
        mass_points: Vec<(i64, i64)>,
        setups: &[ArmySetup],
        seed: u64,
    ) -> Result<Self, String> {
        if setups.len() < 2 || setups.len() > 8 {
            return Err("a skirmish needs 2..=8 armies".into());
        }
        let claims = vec![None; mass_points.len()];
        let spatial = Arc::new(SpatialGrid::new(size.0, size.1, SPATIAL_CELL_MILLI));
        let grid_arc = Arc::new(grid);
        let mut w = Self {
            tick: 0,
            seed,
            rng: seed ^ 0x9E37_79B9_7F4A_7C15,
            next_id: 1,
            next_project: 0,
            next_projectile_id: 1,
            armies: Vec::new(),
            units: BTreeMap::new(),
            wrecks: BTreeMap::new(),
            projectiles: BTreeMap::new(),
            wall_cells: vec![0; (grid_arc.width as usize) * (grid_arc.height as usize)],
            walls_dirty: false,
            grid: Arc::clone(&grid_arc),
            naval_grid: Arc::clone(&grid_arc),
            amphib_grid: Arc::clone(&grid_arc),
            base_grid: Arc::clone(&grid_arc),
            base_amphib: Arc::clone(&grid_arc),
            base_naval: grid_arc,
            terrain: Arc::new(HeightField::flat()),
            props: BTreeMap::new(),
            props_gone: Vec::new(),
            mission: None,
            pings: VecDeque::new(),
            flow_cache: FlowFieldCache::new(),
            spatial,
            spatial_snap: Arc::new(Vec::new()),
            pool: None,
            size,
            mass_points,
            mass_claims: claims,
            hydro_points: Vec::new(),
            hydro_claims: Vec::new(),
            pending: Vec::new(),
            events: Vec::new(),
            audio_events: Vec::new(),
            catalog,
            shots_fired: 0,
            kills: 0,
            paths_solved: 0,
            commands_applied: 0,
            commands_rejected: 0,
            result: None,
            options: GameOptions::default(),
        };
        for (i, s) in setups.iter().enumerate() {
            let faction = w
                .catalog
                .factions
                .get(s.faction)
                .ok_or_else(|| format!("unknown faction index {}", s.faction))?
                .clone();
            w.armies.push(Army {
                id: i as ArmyId,
                name: s.name.clone(),
                faction: s.faction,
                team: s.team,
                mass: 0,
                energy: 0,
                mass_cap: 0,
                energy_cap: 0,
                income_mass: 0,
                income_energy: 0,
                start: s.start,
                defeated: false,
                units_built: 0,
                units_lost: 0,
                kills: 0,
                damage_dealt: 0,
            });
            w.spawn_unit(i as ArmyId, &faction.commander, s.start)
                .ok_or_else(|| format!("commander {} missing from catalog", faction.commander))?;
        }
        w.recompute_caps();
        for a in &mut w.armies {
            a.mass = a.mass_cap;
            a.energy = a.energy_cap;
        }
        w.rebuild_spatial();
        Ok(w)
    }

    /// Convenience: set up armies on `map` at the given start-spot indices.
    pub fn on_map(
        catalog: Catalog,
        map: &MapContext,
        armies: &[(usize, usize)],
        seed: u64,
    ) -> Result<Self, String> {
        let setups = armies
            .iter()
            .enumerate()
            .map(|(i, &(spot, faction))| {
                let (name, start) = map
                    .starts
                    .get(spot)
                    .cloned()
                    .ok_or_else(|| format!("start spot {} out of range", spot + 1))?;
                Ok(ArmySetup {
                    name,
                    faction,
                    start,
                    team: i as u8,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut w = Self::new(
            catalog,
            map.grid.clone(),
            map.size,
            map.mass_points.clone(),
            &setups,
            seed,
        )?;
        w.attach_map(map);
        Ok(w)
    }

    pub fn set_domain_grids(
        &mut self,
        naval_grid: PathGrid,
        amphib_grid: PathGrid,
        hydro_points: Vec<(i64, i64)>,
    ) {
        self.naval_grid = Arc::new(naval_grid);
        self.amphib_grid = Arc::new(amphib_grid);
        self.base_amphib = Arc::clone(&self.amphib_grid);
        self.base_naval = Arc::clone(&self.naval_grid);
        self.hydro_claims = vec![None; hydro_points.len()];
        self.hydro_points = hydro_points;
        self.flow_cache.clear();
        if self.wall_cells.iter().any(|&c| c > 0) {
            // Re-stamp existing walls onto the new amphibious grid.
            self.wall_cells.iter_mut().for_each(|c| *c = 0);
            self.walls_dirty = true;
            self.sync_walls();
        }
    }

    /// Phase 29: terrain heights used for line-of-fire and projectile collisions.
    pub fn set_terrain(&mut self, terrain: HeightField) {
        self.terrain = Arc::new(terrain);
    }

    pub fn terrain(&self) -> &HeightField {
        &self.terrain
    }

    /// Attaches the naval / amphibious grids, hydro markers and terrain of `map` (the
    /// common tail of every map-based world construction).
    pub fn attach_map(&mut self, map: &MapContext) {
        self.set_domain_grids(map.naval_grid.clone(), map.amphib_grid.clone(), map.hydro_points.clone());
        self.set_terrain(HeightField::from_scmap(&map.scmap));
        self.set_props(map.props.clone());
    }

    /// Phase 53: recent map pings `(tick, army, kind, position)`, oldest first.
    pub fn pings(&self) -> impl Iterator<Item = &(u64, ArmyId, u8, (i64, i64))> {
        self.pings.iter()
    }

    /// Phase 48: runs `mission` in this world from now on.
    pub fn set_mission(&mut self, mission: crate::mission::Mission) {
        self.mission = Some(Box::new(crate::mission::MissionRuntime::new(mission)));
    }

    pub fn mission(&self) -> Option<&crate::mission::MissionRuntime> {
        self.mission.as_deref()
    }

    /// Phase 48: the mission script text (stored in replay headers).
    pub fn mission_source(&self) -> Option<String> {
        self.mission.as_ref().map(|m| m.mission.source.clone())
    }

    /// Phase 34: reclaimable props (tests and map attachment).
    pub fn set_props(&mut self, props: Vec<MapProp>) {
        self.props = props.into_iter().enumerate().map(|(i, p)| (i as u32, p)).collect();
        self.props_gone.clear();
    }

    pub fn props(&self) -> impl Iterator<Item = &MapProp> {
        self.props.values()
    }

    /// Phase 34: `.scmap` position bits of every reclaimed prop, in reclaim order.
    pub fn props_gone(&self) -> &[(u32, u32)] {
        &self.props_gone
    }

    /// Phase 34: nearest standing prop to `p` within `max_dist` (ties by index).
    pub fn prop_at(&self, p: (i64, i64), max_dist: i64) -> Option<(u32, (i64, i64))> {
        let max_sq = (max_dist as i128) * (max_dist as i128);
        self.props
            .iter()
            .map(|(&i, q)| (dist_sq(p, q.pos), i, q.pos))
            .filter(|(d, _, _)| *d <= max_sq)
            .min()
            .map(|(_, i, pos)| (i, pos))
    }

    /// Phase 34: what a reclaim order at `p` works on: a wreck (snap 16 m) first, else a prop
    /// (snap [`PROP_SNAP`]), else a live unit (walls, structures, land units — never
    /// commanders or aircraft) whose footprint covers `p`.
    fn reclaim_target(&self, p: (i64, i64)) -> Option<(ReclaimKind, u32, (i64, i64))> {
        self.wreck_at(p, MASS_SNAP * 2)
            .map(|(id, wp)| (ReclaimKind::Wreck, id, wp))
            .or_else(|| self.prop_at(p, PROP_SNAP).map(|(i, pp)| (ReclaimKind::Prop, i, pp)))
            .or_else(|| {
                self.units
                    .values()
                    .filter(|u| {
                        u.alive
                            && u.role != Role::Commander
                            && u.domain != NavDomain::Air
                            && u.carried_by.is_none()
                            && dist_sq(p, (u.pos.x, u.pos.y)) <= ((u.radius + SCALE) as i128).pow(2)
                    })
                    .map(|u| (dist_sq(p, (u.pos.x, u.pos.y)), u.id, (u.pos.x, u.pos.y)))
                    .min()
                    .map(|(_, id, up)| (ReclaimKind::Unit, id, up))
            })
    }

    /// Phase 34: nearest reclaimable mass to `p` within `r` (wrecks, then mass-bearing props).
    pub fn reclaimable_mass_near(&self, p: (i64, i64), r: i64) -> Option<(i64, i64)> {
        let max_sq = (r as i128) * (r as i128);
        self.wreck_at(p, r).map(|(_, wp)| wp).or_else(|| {
            self.props
                .iter()
                .filter(|(_, q)| q.mass > 0)
                .map(|(&i, q)| (dist_sq(p, q.pos), i, q.pos))
                .filter(|(d, _, _)| *d <= max_sq)
                .min()
                .map(|(_, _, pos)| pos)
        })
    }

    /// Phase 28: grid cells covered by a square footprint of half-size `r` centred on `p`.
    /// Walls block every cell they touch (so 1 m sections form solid lines); other
    /// structures (Phase 29) block the cells whose centre lies inside their footprint, so
    /// small structures and the 1 m gaps between buildings do not seal off whole cells.
    fn footprint_cells(grid: &PathGrid, p: (i64, i64), r: i64, wall: bool) -> Vec<(u32, u32)> {
        let r = r.max(1);
        let (x0, z0) = grid.world_to_cell(p.0 - r, p.1 - r);
        let (x1, z1) = grid.world_to_cell(p.0 + r - 1, p.1 + r - 1);
        let half = grid.cell_size_milli / 2;
        let cs = grid.cell_size_milli;
        let inside = |c: u32, lo: i64, hi: i64| {
            let m = c as i64 * cs + half;
            m >= lo && m < hi
        };
        (z0..=z1)
            .flat_map(|z| (x0..=x1).map(move |x| (x, z)))
            .filter(|&(x, z)| wall || (inside(x, p.0 - r, p.0 + r) && inside(z, p.1 - r, p.1 + r)))
            .collect()
    }

    /// Phase 28: whether a live wall section covers the grid cell containing `p`.
    pub fn is_wall_cell(&self, p: (i64, i64)) -> bool {
        let g = &self.base_grid;
        if p.0 < 0 || p.1 < 0 {
            return false;
        }
        let (cx, cz) = g.world_to_cell(p.0, p.1);
        self.wall_cells
            .get((cz as usize) * (g.width as usize) + cx as usize)
            .is_some_and(|&c| c > 0)
    }

    /// Phase 28: number of grid cells currently blocked by walls.
    pub fn wall_cell_count(&self) -> usize {
        self.wall_cells.iter().filter(|&&c| c > 0).count()
    }

    fn stamp_walls(base: &PathGrid, cells: &[u16]) -> PathGrid {
        let mut g = base.clone();
        for (i, _) in cells.iter().enumerate().filter(|(_, c)| **c > 0) {
            let w = g.width as usize;
            g.set_blocked((i % w) as u32, (i / w) as u32, true);
        }
        g
    }

    /// Phase 28: re-derives the wall overlay from live `Role::Wall` units when a wall was
    /// spawned or died. On change, the land / amphibious grids are re-stamped, flow fields
    /// and cached paths are dropped (so movers re-route around the new layout), and ground
    /// units standing on a freshly walled cell are pushed to the nearest open cell.
    /// A no-op (no hash-visible side effects) when the wall layout is unchanged.
    fn sync_walls(&mut self) {
        if !self.walls_dirty {
            return;
        }
        self.walls_dirty = false;
        let mut cells = vec![0u16; self.wall_cells.len()];
        let w = self.base_grid.width as usize;
        for u in self.units.values().filter(|u| u.alive && u.role.is_structure() && u.carried_by.is_none()) {
            for (cx, cz) in Self::footprint_cells(&self.base_grid, (u.pos.x, u.pos.y), u.radius, u.role == Role::Wall) {
                let c = &mut cells[(cz as usize) * w + cx as usize];
                *c = c.saturating_add(1);
            }
        }
        if cells == self.wall_cells {
            return;
        }
        let newly: Vec<bool> = cells.iter().zip(&self.wall_cells).map(|(&n, &o)| n > 0 && o == 0).collect();
        // Bounding box (cells) of newly blocked cells: only paths near it can be invalid.
        let mut bb: Option<(u32, u32, u32, u32)> = None;
        for (i, _) in newly.iter().enumerate().filter(|(_, n)| **n) {
            let (x, z) = ((i % w) as u32, (i / w) as u32);
            bb = Some(bb.map_or((x, z, x, z), |(a, b, c, d)| (a.min(x), b.min(z), c.max(x), d.max(z))));
        }
        self.wall_cells = cells;
        self.grid = Arc::new(Self::stamp_walls(&self.base_grid, &self.wall_cells));
        let same_dims = |g: &PathGrid| g.width == self.base_grid.width && g.height == self.base_grid.height;
        self.amphib_grid = if same_dims(&self.base_amphib) {
            Arc::new(Self::stamp_walls(&self.base_amphib, &self.wall_cells))
        } else {
            Arc::clone(&self.base_amphib)
        };
        self.naval_grid = if same_dims(&self.base_naval) {
            Arc::new(Self::stamp_walls(&self.base_naval, &self.wall_cells))
        } else {
            Arc::clone(&self.base_naval)
        };
        self.flow_cache.clear();
        let Some((bx0, bz0, bx1, bz1)) = bb else {
            return; // only unblocking: existing paths stay valid
        };
        let cs = self.base_grid.cell_size_milli;
        let (lo_x, lo_z) = (bx0 as i64 * cs, bz0 as i64 * cs);
        let (hi_x, hi_z) = ((bx1 as i64 + 1) * cs, (bz1 as i64 + 1) * cs);
        let mut evict = Vec::new();
        for u in self.units.values_mut() {
            if u.domain == NavDomain::Air {
                continue;
            }
            // Re-route units whose remaining path (incl. current position) overlaps the
            // newly blocked area's bounding box.
            let mut ax = (u.pos.x, u.pos.x);
            let mut az = (u.pos.y, u.pos.y);
            for &(x, z) in &u.path {
                ax = (ax.0.min(x), ax.1.max(x));
                az = (az.0.min(z), az.1.max(z));
            }
            if ax.1 >= lo_x - cs && ax.0 <= hi_x + cs && az.1 >= lo_z - cs && az.0 <= hi_z + cs {
                u.path.clear();
                u.path_goal = None;
            }
            if u.alive && u.speed > 0 && u.carried_by.is_none() {
                let (cx, cz) = self.base_grid.world_to_cell(u.pos.x, u.pos.y);
                if newly[(cz as usize) * w + cx as usize] {
                    evict.push(u.id);
                }
            }
        }
        for id in evict {
            let p = self.units[&id].pos;
            let np = self.walkable_near(id, (p.x, p.y));
            let u = self.units.get_mut(&id).unwrap();
            u.pos.x = np.0;
            u.pos.y = np.1;
        }
    }

    /// Attach or detach a persistent `WorkerPool` for batched A* pathfinding.
    pub fn set_worker_pool(&mut self, pool: Option<Arc<WorkerPool>>) {
        self.pool = pool;
    }

    /// Phase 18: set the game options (call before the first tick; recorded in replays).
    pub fn set_options(&mut self, options: GameOptions) {
        self.options = options;
    }

    /// Phase 18 builder form of [`Skirmish::set_options`].
    pub fn with_options(mut self, options: GameOptions) -> Self {
        self.options = options;
        self
    }

    pub fn options(&self) -> GameOptions {
        self.options
    }

    pub fn drain_events(&mut self) -> Vec<SimEvent> {
        std::mem::take(&mut self.events)
    }

    /// Drains the audio copy of the event stream (never affects `state_hash()`).
    pub fn drain_audio_events(&mut self) -> Vec<SimEvent> {
        std::mem::take(&mut self.audio_events)
    }

    fn emit_event(&mut self, ev: SimEvent) {
        if self.audio_events.len() < 1024 {
            self.audio_events.push(ev.clone());
        }
        if self.events.len() < 512 {
            self.events.push(ev);
        }
    }

    pub fn are_allies(&self, a: ArmyId, b: ArmyId) -> bool {
        if a == b {
            return true;
        }
        match (self.army(a), self.army(b)) {
            (Some(aa), Some(bb)) => aa.team == bb.team,
            _ => false,
        }
    }

    // ----- read-only queries -------------------------------------------------

    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn armies(&self) -> &[Army] {
        &self.armies
    }
    pub fn army(&self, id: ArmyId) -> Option<&Army> {
        self.armies.get(id as usize)
    }
    pub fn faction(&self, army: ArmyId) -> Option<&Faction> {
        self.army(army)
            .and_then(|a| self.catalog.factions.get(a.faction))
    }
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }
    pub fn units(&self) -> impl Iterator<Item = &Unit> {
        self.units.values().filter(|u| u.alive)
    }
    pub fn unit(&self, id: UnitId) -> Option<&Unit> {
        self.units.get(&id).filter(|u| u.alive)
    }
    pub fn wrecks(&self) -> impl Iterator<Item = &Wreck> {
        self.wrecks.values()
    }
    pub fn projectiles(&self) -> impl Iterator<Item = &Projectile> {
        self.projectiles.values()
    }
    pub fn mass_points(&self) -> &[(i64, i64)] {
        &self.mass_points
    }
    pub fn hydro_points(&self) -> &[(i64, i64)] {
        &self.hydro_points
    }
    pub fn mass_claim(&self, idx: usize) -> Option<UnitId> {
        self.mass_claims.get(idx).copied().flatten()
    }
    pub fn hydro_claim(&self, idx: usize) -> Option<UnitId> {
        self.hydro_claims.get(idx).copied().flatten()
    }
    pub fn size(&self) -> (i64, i64) {
        self.size
    }
    pub fn grid(&self) -> &PathGrid {
        &self.grid
    }
    pub fn domain_grid(&self, domain: NavDomain) -> &PathGrid {
        match domain {
            NavDomain::Land => &self.grid,
            NavDomain::Naval => &self.naval_grid,
            NavDomain::Amphibious | NavDomain::Air => &self.amphib_grid,
        }
    }
    pub fn shots_fired(&self) -> u64 {
        self.shots_fired
    }
    pub fn kills(&self) -> u64 {
        self.kills
    }
    pub fn paths_solved(&self) -> u64 {
        self.paths_solved
    }
    pub fn commands_applied(&self) -> u64 {
        self.commands_applied
    }
    pub fn commands_rejected(&self) -> u64 {
        self.commands_rejected
    }
    pub fn result(&self) -> Option<GameResult> {
        self.result
    }
    pub fn commander_of(&self, army: ArmyId) -> Option<&Unit> {
        self.units()
            .find(|u| u.army == army && u.role == Role::Commander)
    }

    pub fn desync_summary(&self) -> String {
        let mut s = format!("tick={} units={}", self.tick, self.units.len());
        for a in &self.armies {
            let cnt = self.units().filter(|u| u.army == a.id).count();
            let hp_sum: i64 = self.units().filter(|u| u.army == a.id).map(|u| u.hp / SCALE).sum();
            s.push_str(&format!(
                " | A{}: u={} hp={} M={} E={}",
                a.id,
                cnt,
                hp_sum,
                a.mass / SCALE,
                a.energy / SCALE
            ));
        }
        s
    }

    /// Own live units whose centre lies in the axis-aligned rectangle.
    pub fn units_in_rect(&self, army: ArmyId, a: (i64, i64), b: (i64, i64)) -> Vec<UnitId> {
        let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
        let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
        self.units()
            .filter(|u| u.army == army && u.carried_by.is_none())
            .filter(|u| u.pos.x >= x0 && u.pos.x <= x1 && u.pos.y >= y0 && u.pos.y <= y1)
            .map(|u| u.id)
            .collect()
    }

    /// Nearest live unit near `p` (within `u.radius + pick_radius`), optionally
    /// restricted to `army_filter` or excluding `hostile_to` allies.
    pub fn unit_at(
        &self,
        p: (i64, i64),
        pick_radius: i64,
        army_filter: Option<ArmyId>,
        hostile_to: Option<ArmyId>,
    ) -> Option<UnitId> {
        let mut best: Option<(i128, UnitId)> = None;
        let host_team = hostile_to.and_then(|a| self.army(a).map(|x| x.team));
        self.spatial
            .for_each_in_radius(p.0, p.1, pick_radius + 10 * SCALE, |idx| {
                let u = &self.spatial_snap[idx as usize];
                if army_filter.is_some_and(|a| u.army != a)
                    || host_team.is_some_and(|t| u.team == t)
                {
                    return;
                }
                let reach = (u.radius + pick_radius) as i128;
                let d2 = dist_sq((u.x, u.y), p);
                if d2 <= reach * reach
                    && best.is_none_or(|(bd, bid)| d2 < bd || (d2 == bd && u.id < bid))
                {
                    best = Some((d2, u.id));
                }
            });
        best.map(|(_, id)| id)
    }

    /// Nearest wreck to `p` within `max_dist`.
    pub fn wreck_at(&self, p: (i64, i64), max_dist: i64) -> Option<(UnitId, (i64, i64))> {
        let max_sq = (max_dist as i128) * (max_dist as i128);
        self.wrecks
            .values()
            .filter(|w| w.mass_left > 0)
            .map(|w| (dist_sq(p, (w.pos.x, w.pos.y)), w.id, (w.pos.x, w.pos.y)))
            .filter(|(d, _, _)| *d <= max_sq)
            .min()
            .map(|(_, id, pos)| (id, pos))
    }

    /// Fog-of-War & Radar classification from `viewer_army`'s perspective (shares allied intel).
    ///
    /// Phase 18 Intel II: omni reveals everything in range; cloaked units are invisible to
    /// vision; radar-stealthed units never show as radar blips; units aboard an enemy
    /// transport are hidden (the transport itself is what you see).
    pub fn intel_visibility(&self, viewer_army: ArmyId, target: &Unit) -> IntelVisibility {
        if self.are_allies(viewer_army, target.army) {
            return IntelVisibility::Visible;
        }
        if target.carried_by.is_some() {
            return IntelVisibility::Hidden;
        }
        let tp = (target.pos.x, target.pos.y);
        let mut radar_hit = false;
        for u in self.units() {
            if !self.are_allies(viewer_army, u.army) || u.carried_by.is_some() {
                continue;
            }
            let d2 = dist_sq((u.pos.x, u.pos.y), tp);
            if u.omni_radius > 0 {
                let omni = (u.omni_radius + target.radius) as i128;
                if d2 <= omni * omni {
                    return IntelVisibility::Visible;
                }
            }
            let vis = (u.vision_radius + target.radius) as i128;
            if d2 <= vis * vis && !target.cloaked {
                return IntelVisibility::Visible;
            }
            if u.radar_radius > 0 && !target.radar_stealth {
                let rad = (u.radar_radius + target.radius) as i128;
                if d2 <= rad * rad {
                    radar_hit = true;
                }
            }
        }
        if radar_hit {
            IntelVisibility::RadarBlip
        } else {
            IntelVisibility::Hidden
        }
    }

    /// Phase 18: false radar contacts created by enemy jammers, as seen by `viewer_army`.
    ///
    /// Pure query (never touches sim state or the RNG): each jammer projects
    /// `JammerBlips` blips at hash-derived offsets within its `JamRadius`, re-rolled every
    /// [`JAMMER_BLIP_PERIOD`] ticks. A blip shows only where the viewer's team has radar
    /// coverage but no vision.
    pub fn jammer_blips(&self, viewer_army: ArmyId) -> Vec<(i64, i64)> {
        let mut out = Vec::new();
        let sensors: Vec<(i64, i64, i64, i64)> = self
            .units()
            .filter(|u| self.are_allies(viewer_army, u.army) && u.carried_by.is_none())
            .map(|u| (u.pos.x, u.pos.y, u.vision_radius, u.radar_radius))
            .collect();
        let epoch = self.tick / JAMMER_BLIP_PERIOD;
        for u in self.units() {
            if self.are_allies(viewer_army, u.army) || u.carried_by.is_some() {
                continue;
            }
            let spec = self.catalog.get(&u.type_id).map(|t| &t.ext.intel);
            let (mut n, (mut rmin, mut rmax)) = spec.map(|i| (i.jammer_blips, i.jam_radius)).unwrap_or((0, (0, 0)));
            if u.enh.jammer_radius > 0 && !u.enh.stalled {
                n = n.max(3);
                rmax = rmax.max(u.enh.jammer_radius);
                rmin = rmin.min(rmax / 2);
            }
            if n == 0 || rmax <= 0 {
                continue;
            }
            for i in 0..n {
                let mut h: u64 = 0xCBF2_9CE4_8422_2325;
                Self::mix(&mut h, &u.id.to_le_bytes());
                Self::mix(&mut h, &i.to_le_bytes());
                Self::mix(&mut h, &epoch.to_le_bytes());
                let angle = (h % 6283) as i64;
                let span = (rmax - rmin).max(1);
                let r = rmin + ((h >> 16) % span as u64) as i64;
                let (c, s) = Self::cos_sin_mrad(angle);
                let p = self.clamp_to_map(u.pos.x + r * c / SCALE, u.pos.y + r * s / SCALE);
                let mut radar = false;
                let mut seen = false;
                for &(sx, sy, vis, rad) in &sensors {
                    let d2 = dist_sq((sx, sy), p);
                    if d2 <= (vis as i128) * (vis as i128) {
                        seen = true;
                        break;
                    }
                    if rad > 0 && d2 <= (rad as i128) * (rad as i128) {
                        radar = true;
                    }
                }
                if radar && !seen {
                    out.push(p);
                }
            }
        }
        out
    }

    /// Phase 24 build card for a live unit: the data-driven [`Catalog::build_options`] of
    /// its type, minus ACU `Build` options above its current engineering-suite tier.
    /// Every returned option is accepted by the matching `Command` (Build / Produce /
    /// Upgrade) unless placement, resources or the unit cap stop it.
    pub fn build_options(&self, id: UnitId) -> Vec<&roster::BuildOption> {
        let Some(u) = self.unit(id) else {
            return Vec::new();
        };
        let max_tier = if u.role == Role::Commander { u.enh.engineering_tier.max(1) } else { 4 };
        self.catalog
            .build_options(&u.type_id)
            .iter()
            .filter(|o| self.catalog.types.contains_key(&o.id))
            .filter(|o| o.kind != roster::BuildKind::Build || o.tier <= max_tier)
            .collect()
    }

    /// Phase 18: units counted against the unit cap (live units plus started constructions).
    pub fn army_unit_count(&self, army: ArmyId) -> u32 {
        self.units
            .values()
            .filter(|u| u.alive && u.army == army)
            .map(|u| 1 + u32::from(Self::job_spawns_unit(u)))
            .sum()
    }

    /// Whether `u`'s current construction job will spawn a new unit (not an in-place
    /// upgrade or enhancement).
    fn job_spawns_unit(u: &Unit) -> bool {
        u.build.is_some() && !matches!(u.orders.front(), Some(UnitOrder::Upgrade(_) | UnitOrder::Enhance(_)))
    }

    /// Phase 18: whether `army` is at its unit cap.
    pub fn at_unit_cap(&self, army: ArmyId) -> bool {
        self.options.unit_cap > 0 && self.army_unit_count(army) >= self.options.unit_cap
    }

    /// Phase 18: enhancements unit `id` could start right now (blueprint order).
    pub fn available_enhancements(&self, id: UnitId) -> Vec<&EnhancementSpec> {
        let Some(u) = self.unit(id) else {
            return Vec::new();
        };
        let Some(t) = self.catalog.get(&u.type_id) else {
            return Vec::new();
        };
        t.ext
            .enhancements
            .iter()
            .filter(|e| Self::enhancement_allowed(&u.enh, e))
            .collect()
    }

    /// Slot rules: a slot holds one enhancement; a new one needs an empty slot (and no
    /// prerequisite) or must directly follow the prerequisite currently installed there.
    /// Removals need their prerequisite installed.
    fn enhancement_allowed(state: &EnhancementState, e: &EnhancementSpec) -> bool {
        let cur = state
            .installed
            .iter()
            .find(|(s, _)| *s == e.slot)
            .map(|(_, n)| n.as_str());
        if e.is_removal() {
            return e.prerequisite.as_deref().is_some_and(|p| cur == Some(p));
        }
        match (cur, e.prerequisite.as_deref()) {
            (Some(c), _) if c == e.name => false,
            (None, None) => true,
            (Some(c), Some(p)) => c == p,
            _ => false,
        }
    }

    /// Phase 18: transport classes of the passengers aboard `transport`.
    pub fn cargo_classes(&self, transport: UnitId) -> Vec<u8> {
        self.units
            .get(&transport)
            .map(|t| {
                t.cargo
                    .iter()
                    .filter_map(|c| self.units.get(c))
                    .filter(|c| c.alive)
                    .map(|c| self.catalog.get(&c.type_id).map(|ct| ct.ext.transport_class).unwrap_or(1))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Phase 18: whether `passenger` may board `transport` right now (capacity, classes,
    /// commander rule, same army).
    pub fn can_board(&self, passenger: UnitId, transport: UnitId) -> bool {
        let (Some(p), Some(t)) = (self.unit(passenger), self.unit(transport)) else {
            return false;
        };
        if p.army != t.army || p.carried_by.is_some() || t.carried_by.is_some() || passenger == transport {
            return false;
        }
        let Some(spec) = self.catalog.get(&t.type_id).and_then(|x| x.ext.transport.as_ref()) else {
            return false;
        };
        let class = self.catalog.get(&p.type_id).map(|x| x.ext.transport_class).unwrap_or(0);
        if class == 0 || (p.role == Role::Commander && !spec.carries_commander) {
            return false;
        }
        spec.fits(&self.cargo_classes(transport), class)
    }

    /// Nearest unclaimed mass point to `p` within `max_dist` (milli).
    pub fn nearest_free_mass_point(&self, p: (i64, i64), max_dist: i64) -> Option<(usize, (i64, i64))> {
        let max_sq = (max_dist as i128) * (max_dist as i128);
        self.mass_points
            .iter()
            .enumerate()
            .filter(|(i, _)| self.mass_claims[*i].is_none())
            .map(|(i, &m)| (dist_sq(p, m), i, m))
            .filter(|(d, _, _)| *d <= max_sq)
            .min()
            .map(|(_, i, m)| (i, m))
    }

    /// Nearest unclaimed hydrocarbon point to `p` within `max_dist` (milli).
    pub fn nearest_free_hydro_point(&self, p: (i64, i64), max_dist: i64) -> Option<(usize, (i64, i64))> {
        let max_sq = (max_dist as i128) * (max_dist as i128);
        self.hydro_points
            .iter()
            .enumerate()
            .filter(|(i, _)| self.hydro_claims.get(*i).copied().flatten().is_none())
            .map(|(i, &h)| (dist_sq(p, h), i, h))
            .filter(|(d, _, _)| *d <= max_sq)
            .min()
            .map(|(_, i, h)| (i, h))
    }

    /// Whether a structure of `type_id` may be placed at `(x, y)`.
    pub fn site_is_free(&self, type_id: &str, x: i64, y: i64) -> bool {
        let Some(t) = self.catalog.get(type_id) else {
            return false;
        };
        let r = t.radius;
        if x - r < 0 || y - r < 0 || x + r > self.size.0 || y + r > self.size.1 {
            return false;
        }
        let check_grid = if t.role == Role::NavalFactory || t.ext.water_only || (t.domain == NavDomain::Naval && !t.role.is_structure()) {
            &self.naval_grid
        } else {
            &self.grid
        };
        let (cx, cy) = check_grid.world_to_cell(x, y);
        if check_grid.is_blocked(cx, cy) {
            return false;
        }
        // Phase 28: wall sections may sit flush against each other and other structures.
        let is_wall = t.role == Role::Wall;
        let margin_with = |other_wall: bool| if is_wall || other_wall { 0 } else { SCALE };
        let overlaps_structure = self.units().any(|u| {
            u.role.is_structure() && {
                let lim = (u.radius + r + margin_with(u.role == Role::Wall)) as i128;
                dist_sq((u.pos.x, u.pos.y), (x, y)) < lim * lim
            }
        });
        let overlaps_site = self.units().any(|u| {
            u.build.as_ref().is_some_and(|j| {
                u.role.is_builder() && {
                    let other_t = self.catalog.get(&j.unit_type);
                    let other = other_t.map(|t| t.radius).unwrap_or(0);
                    let lim = (other + r + margin_with(other_t.is_some_and(|t| t.role == Role::Wall))) as i128;
                    dist_sq((j.x, j.y), (x, y)) < lim * lim
                }
            })
        });
        if overlaps_structure || overlaps_site {
            return false;
        }
        if t.role != Role::Extractor {
            let keep = (r + 2 * SCALE) as i128;
            if self.mass_points.iter().any(|&m| dist_sq(m, (x, y)) < keep * keep) {
                return false;
            }
        }
        if t.role != Role::Hydro {
            let keep = (r + 2 * SCALE) as i128;
            if self.hydro_points.iter().any(|&h| dist_sq(h, (x, y)) < keep * keep) {
                return false;
            }
        }
        true
    }

    /// Snap a requested extractor site to the nearest free mass point.
    pub fn snap_extractor(&self, x: i64, y: i64) -> Option<(usize, (i64, i64))> {
        self.nearest_free_mass_point((x, y), MASS_SNAP)
    }

    /// Snap a requested hydro site to the nearest free hydrocarbon marker.
    pub fn snap_hydro(&self, x: i64, y: i64) -> Option<(usize, (i64, i64))> {
        self.nearest_free_hydro_point((x, y), MASS_SNAP * 2)
    }

    // ----- input ---------------------------------------------------------------

    /// Queue a command for the next tick.
    pub fn issue(&mut self, army: ArmyId, cmd: Command) {
        self.pending.push((army, cmd));
    }

    /// Applies an external command directly (used by replay / lockstep turn sync).
    pub fn apply_command_direct(&mut self, army: ArmyId, cmd: Command) -> bool {
        let ok = self
            .army(army)
            .is_some_and(|a| !a.defeated)
            && self.apply_command(army, cmd);
        if ok {
            self.commands_applied += 1;
        } else {
            self.commands_rejected += 1;
        }
        ok
    }

    pub fn pending_commands(&self) -> usize {
        self.pending.len()
    }

    /// Commands queued for the next tick (read-only; used for order acknowledgements).
    pub fn pending_view(&self) -> &[(ArmyId, Command)] {
        &self.pending
    }

    pub fn drain_pending(&mut self) -> Vec<(ArmyId, Command)> {
        std::mem::take(&mut self.pending)
    }

    fn owned_mobiles(&self, army: ArmyId, ids: &[UnitId]) -> Vec<UnitId> {
        let mut v: Vec<UnitId> = ids
            .iter()
            .copied()
            .filter(|id| {
                self.unit(*id)
                    .is_some_and(|u| u.army == army && u.speed > 0 && u.carried_by.is_none())
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    fn owned_builders(&self, army: ArmyId, ids: &[UnitId]) -> Vec<UnitId> {
        let mut v: Vec<UnitId> = ids
            .iter()
            .copied()
            .filter(|id| {
                self.unit(*id)
                    .is_some_and(|u| u.army == army && u.role.is_builder() && u.carried_by.is_none())
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Formation slot `i` of `n` around `(x, y)` (square grid, row-major).
    pub fn formation_slot(i: usize, n: usize, x: i64, y: i64) -> (i64, i64) {
        if n <= 1 {
            return (x, y);
        }
        let cols = (isqrt(n as i128) as usize).max(1);
        let cols = if cols * cols < n { cols + 1 } else { cols };
        let rows = n.div_ceil(cols);
        let c = (i % cols) as i64;
        let r = (i / cols) as i64;
        let ox = (c * 2 - (cols as i64 - 1)) * FORMATION_SPACING / 2;
        let oy = (r * 2 - (rows as i64 - 1)) * FORMATION_SPACING / 2;
        (x + ox, y + oy)
    }

    /// Pure integer `(cos, sin)` scaled by `SCALE` (1000) for `angle_mrad` (milliradians).
    fn cos_sin_mrad(angle_mrad: i64) -> (i64, i64) {
        const TWO_PI: i64 = 6283;
        const PI: i64 = 3142;
        const HALF_PI: i64 = 1571;
        let mut a = angle_mrad.rem_euclid(TWO_PI);
        if a > PI {
            a -= TWO_PI;
        }
        let (x, sin_sign, cos_sign) = if a >= 0 {
            if a <= HALF_PI {
                (a, 1i64, 1i64)
            } else {
                (PI - a, 1i64, -1i64)
            }
        } else if a >= -HALF_PI {
            (-a, -1i64, 1i64)
        } else {
            (PI + a, -1i64, -1i64)
        };
        let eval_sin = |u: i64| -> i64 {
            let s = SCALE as i128;
            let u128 = u as i128;
            let u2 = u128 * u128 / s;
            let u3 = u2 * u128 / s;
            let u5 = u3 * u2 / s;
            let u7 = u5 * u2 / s;
            ((u128 - u3 / 6 + u5 / 120 - u7 / 5040) as i64).clamp(0, SCALE)
        };
        let s = eval_sin(x) * sin_sign;
        let c = eval_sin((HALF_PI - x).max(0)) * cos_sign;
        (c, s)
    }

    /// Rotated formation slot around `(x, y)` oriented by `facing_mrad`.
    pub fn rotated_formation_slot(
        i: usize,
        n: usize,
        x: i64,
        y: i64,
        facing_mrad: i64,
        spacing: i64,
    ) -> (i64, i64) {
        if n <= 1 {
            return (x, y);
        }
        let sp = spacing.max(2 * SCALE);
        let cols = (isqrt(n as i128) as usize).max(1);
        let cols = if cols * cols < n { cols + 1 } else { cols };
        let rows = n.div_ceil(cols);
        let c = (i % cols) as i64;
        let r = (i / cols) as i64;
        let lx = (c * 2 - (cols as i64 - 1)) * sp / 2;
        let ly = (r * 2 - (rows as i64 - 1)) * sp / 2;
        let (cos_a, sin_a) = Self::cos_sin_mrad(facing_mrad);
        let rx = ((lx as i128 * cos_a as i128 - ly as i128 * sin_a as i128) / SCALE as i128) as i64;
        let ry = ((lx as i128 * sin_a as i128 + ly as i128 * cos_a as i128) / SCALE as i128) as i64;
        (x + rx, y + ry)
    }

    fn clamp_to_map(&self, x: i64, y: i64) -> (i64, i64) {
        (x.clamp(0, self.size.0), y.clamp(0, self.size.1))
    }

    fn push_order(&mut self, id: UnitId, order: UnitOrder, queue: bool) {
        let cancel_job = !queue
            && self
                .units
                .get(&id)
                .is_some_and(|u| u.role.is_builder() && u.build.is_some());
        if cancel_job {
            self.cancel_build(id);
        }
        if let Some(u) = self.units.get_mut(&id) {
            if !queue {
                u.orders.clear();
                u.path.clear();
                u.path_goal = None;
            }
            u.orders.push_back(order);
        }
    }

    fn cancel_build(&mut self, id: UnitId) {
        let job = self.units.get_mut(&id).and_then(|u| u.build.take());
        if let Some(j) = job {
            if let Some(mp) = j.mass_point
                && self.mass_claims.get(mp).copied().flatten() == Some(id)
            {
                self.mass_claims[mp] = None;
            }
            if let Some(hp) = self.hydro_claims.iter().position(|&c| c == Some(id)) {
                self.hydro_claims[hp] = None;
            }
        }
    }

    fn apply_commands(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        for (army, cmd) in pending {
            self.apply_command_direct(army, cmd);
        }
    }

    fn apply_command(&mut self, army: ArmyId, cmd: Command) -> bool {
        match cmd {
            Command::Move { units, x, y, queue } => {
                let ids = self.owned_mobiles(army, &units);
                let n = ids.len();
                for (i, id) in ids.iter().enumerate() {
                    let (sx, sy) = Self::formation_slot(i, n, x, y);
                    let (sx, sy) = self.clamp_to_map(sx, sy);
                    self.push_order(*id, UnitOrder::Move { x: sx, y: sy }, queue);
                }
                n > 0
            }
            Command::FormationMove {
                units,
                x,
                y,
                facing_mrad,
                spacing,
                queue,
            } => {
                let ids = self.owned_mobiles(army, &units);
                let n = ids.len();
                for (i, id) in ids.iter().enumerate() {
                    let (sx, sy) =
                        Self::rotated_formation_slot(i, n, x, y, facing_mrad, spacing);
                    let (sx, sy) = self.clamp_to_map(sx, sy);
                    self.push_order(*id, UnitOrder::Move { x: sx, y: sy }, queue);
                }
                n > 0
            }
            Command::AttackMove { units, x, y, queue } => {
                let ids = self.owned_mobiles(army, &units);
                let n = ids.len();
                for (i, id) in ids.iter().enumerate() {
                    let (sx, sy) = Self::formation_slot(i, n, x, y);
                    let (sx, sy) = self.clamp_to_map(sx, sy);
                    self.push_order(*id, UnitOrder::AttackMove { x: sx, y: sy }, queue);
                }
                n > 0
            }
            Command::Patrol { units, x, y, queue } => {
                let ids = self.owned_mobiles(army, &units);
                let n = ids.len();
                for (i, id) in ids.iter().enumerate() {
                    let (sx, sy) = Self::formation_slot(i, n, x, y);
                    let (sx, sy) = self.clamp_to_map(sx, sy);
                    self.push_order(*id, UnitOrder::Patrol { x: sx, y: sy }, queue);
                }
                n > 0
            }
            Command::Attack { units, target, queue } => {
                let valid = self
                    .unit(target)
                    .is_some_and(|t| !self.are_allies(army, t.army) && t.targetable_by(army));
                if !valid {
                    return false;
                }
                let mut v: Vec<UnitId> = units
                    .iter()
                    .copied()
                    .filter(|id| {
                        self.unit(*id)
                            .is_some_and(|u| u.army == army && !u.weapons.is_empty() && u.carried_by.is_none())
                    })
                    .collect();
                v.sort_unstable();
                v.dedup();
                for id in &v {
                    self.push_order(*id, UnitOrder::Attack(target), queue);
                }
                !v.is_empty()
            }
            Command::Stop { units } => {
                let mut any = false;
                let mut ids: Vec<UnitId> = units;
                ids.sort_unstable();
                ids.dedup();
                for id in ids {
                    if self.unit(id).is_some_and(|u| u.army == army) {
                        self.cancel_build(id);
                        if let Some(u) = self.units.get_mut(&id) {
                            u.orders.clear();
                            u.path.clear();
                            u.path_goal = None;
                            u.production.clear();
                        }
                        any = true;
                    }
                }
                any
            }
            Command::Build { builder, unit_type, x, y, queue } => {
                let Some(b) = self.unit(builder) else {
                    return false;
                };
                if b.army != army || !self.is_constructor(b) || b.carried_by.is_some() {
                    return false;
                }
                let Some(fi) = self.army(army).map(|a| a.faction) else {
                    return false;
                };
                if !self.catalog.faction_can_build(fi, &unit_type)
                    && !self.catalog.builder_can_build(&b.type_id, &unit_type)
                {
                    return false;
                }
                // Phase 18: Commanders build up to their engineering-suite tech level.
                if b.role == Role::Commander
                    && self
                        .catalog
                        .get(&unit_type)
                        .is_some_and(|t| t.tier > b.enh.engineering_tier.max(1))
                {
                    return false;
                }
                if self.at_unit_cap(army) {
                    return false;
                }
                let target_role = self.catalog.get(&unit_type).map(|t| t.role);
                let (x, y) = if target_role == Some(Role::Extractor) {
                    match self.snap_extractor(x, y) {
                        Some((_, p)) => p,
                        None => return false,
                    }
                } else if target_role == Some(Role::Hydro) {
                    match self.snap_hydro(x, y) {
                        Some((_, p)) => p,
                        None => return false,
                    }
                } else {
                    self.clamp_to_map(x, y)
                };
                self.push_order(builder, UnitOrder::Build { unit_type, x, y }, queue);
                true
            }
            Command::Produce { factory, unit_type, count } => {
                let fi = self.army(army).map(|a| a.faction).unwrap_or(usize::MAX);
                let ok = self.unit(factory).is_some_and(|u| {
                    u.army == army
                        && self.is_producer(u)
                        && (self.catalog.faction_can_produce(fi, &unit_type)
                            || self.catalog.builder_can_produce(&u.type_id, &unit_type))
                });
                if !ok || count == 0 || self.at_unit_cap(army) {
                    return false;
                }
                if let Some(u) = self.units.get_mut(&factory) {
                    for _ in 0..count.min(50) {
                        u.production.push_back(unit_type.clone());
                    }
                }
                true
            }
            Command::Rally { factory, x, y } => {
                let p = self.clamp_to_map(x, y);
                let producer = self.unit(factory).is_some_and(|u| self.is_producer(u));
                match self.units.get_mut(&factory) {
                    Some(u) if u.alive && u.army == army && producer => {
                        u.rally = Some(p);
                        true
                    }
                    _ => false,
                }
            }
            Command::Reclaim { units, x, y, queue } => {
                let p = self.clamp_to_map(x, y);
                let ids = self.owned_builders(army, &units);
                for id in &ids {
                    self.push_order(*id, UnitOrder::Reclaim { x: p.0, y: p.1 }, queue);
                }
                !ids.is_empty()
            }
            Command::Repair { units, target, queue } => {
                if !self
                    .unit(target)
                    .is_some_and(|t| self.are_allies(army, t.army) && t.carried_by.is_none())
                {
                    return false;
                }
                let ids = self.owned_builders(army, &units);
                for id in &ids {
                    self.push_order(*id, UnitOrder::Repair(target), queue);
                }
                !ids.is_empty()
            }
            Command::Assist { units, target, queue } => {
                if !self
                    .unit(target)
                    .is_some_and(|t| self.are_allies(army, t.army) && (t.role.is_factory() || t.role.is_builder()))
                {
                    return false;
                }
                let ids: Vec<UnitId> = self
                    .owned_builders(army, &units)
                    .into_iter()
                    .filter(|&id| id != target)
                    .collect();
                for id in &ids {
                    self.push_order(*id, UnitOrder::Assist(target), queue);
                }
                !ids.is_empty()
            }
            Command::Overcharge { unit, target } => {
                let ok_u = self
                    .unit(unit)
                    .is_some_and(|u| u.army == army && u.role == Role::Commander && u.carried_by.is_none());
                let ok_t = self
                    .unit(target)
                    .is_some_and(|t| !self.are_allies(army, t.army) && t.targetable_by(army));
                if !ok_u || !ok_t {
                    return false;
                }
                self.push_order(unit, UnitOrder::Overcharge(target), false);
                true
            }
            Command::Upgrade { unit, target_type } => {
                let Some(u) = self.unit(unit) else {
                    return false;
                };
                if u.army != army || self.catalog.get(&target_type).is_none() || u.carried_by.is_some() {
                    return false;
                }
                let expected = self
                    .catalog
                    .get(&u.type_id)
                    .and_then(|t| t.upgrades_to.as_ref());
                if expected.is_none_or(|e| !e.eq_ignore_ascii_case(&target_type)) {
                    return false;
                }
                let resolved_id = self
                    .catalog
                    .types
                    .keys()
                    .find(|k| k.eq_ignore_ascii_case(&target_type))
                    .cloned()
                    .unwrap_or(target_type);
                self.push_order(unit, UnitOrder::Upgrade(resolved_id), false);
                true
            }
            Command::LaunchMissile { unit, x, y } => {
                let Some(u) = self.unit(unit) else {
                    return false;
                };
                if u.army != army || u.role != Role::MissileSilo || u.silo_missiles == 0 {
                    return false;
                }
                let p = self.clamp_to_map(x, y);
                if let Some(u_mut) = self.units.get_mut(&unit) {
                    u_mut.orders.push_back(UnitOrder::LaunchMissile { x: p.0, y: p.1 });
                }
                true
            }
            Command::ToggleShield { units, enabled } => {
                let mut any = false;
                let mut ids = units;
                ids.sort_unstable();
                ids.dedup();
                for id in ids {
                    if let Some(u) = self.units.get_mut(&id)
                        && u.alive
                        && u.army == army
                        && u.shield_max_hp > 0
                    {
                        u.shield_on = enabled;
                        if !enabled {
                            u.shield_hp = 0;
                        }
                        any = true;
                    }
                }
                any
            }
            Command::Load { units, transport, queue } => {
                let ok_t = self.unit(transport).is_some_and(|t| {
                    t.army == army
                        && t.carried_by.is_none()
                        && self.catalog.get(&t.type_id).is_some_and(|x| x.ext.transport.is_some())
                });
                if !ok_t {
                    return false;
                }
                let mut ids: Vec<UnitId> = units
                    .iter()
                    .copied()
                    .filter(|&id| {
                        id != transport
                            && self.unit(id).is_some_and(|u| {
                                u.army == army
                                    && u.carried_by.is_none()
                                    && u.speed > 0
                                    && self.catalog.get(&u.type_id).is_some_and(|x| x.ext.transport_class > 0)
                            })
                    })
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                if ids.is_empty() {
                    return false;
                }
                for id in &ids {
                    self.push_order(*id, UnitOrder::Board(transport), queue);
                }
                // An idle transport comes to meet its passengers halfway.
                let n = ids.len() as i64;
                let cx = ids.iter().filter_map(|id| self.unit(*id)).map(|u| u.pos.x).sum::<i64>() / n;
                let cy = ids.iter().filter_map(|id| self.unit(*id)).map(|u| u.pos.y).sum::<i64>() / n;
                if let Some(t) = self.units.get_mut(&transport)
                    && t.orders.is_empty()
                {
                    let mx = (t.pos.x + cx) / 2;
                    let my = (t.pos.y + cy) / 2;
                    t.orders.push_back(UnitOrder::Move { x: mx, y: my });
                }
                true
            }
            Command::Dock { units, staging, queue } => {
                if !self.is_staging_of(staging, army) {
                    return false;
                }
                let mut ids: Vec<UnitId> = units
                    .iter()
                    .copied()
                    .filter(|&id| self.unit(id).is_some_and(|u| u.army == army && u.domain == NavDomain::Air && u.speed > 0 && u.carried_by.is_none()))
                    .collect();
                ids.sort_unstable();
                ids.dedup();
                if ids.is_empty() {
                    return false;
                }
                for id in ids {
                    self.push_order(id, UnitOrder::Dock(staging), queue);
                }
                true
            }
            Command::SetFireState { units, state } => {
                let mut any = false;
                for id in units {
                    if let Some(u) = self.units.get_mut(&id).filter(|u| u.army == army && u.alive) {
                        u.fire_state = state;
                        any = true;
                    }
                }
                any
            }
            Command::SetBuildPaused { units, paused } => {
                let mut any = false;
                for id in units {
                    if let Some(u) = self.units.get_mut(&id).filter(|u| u.army == army && u.alive) {
                        u.build_paused = paused;
                        any = true;
                    }
                }
                any
            }
            Command::AttackGround { units, x, y, queue } => {
                let p = self.clamp_to_map(x, y);
                let mut v: Vec<UnitId> = units
                    .iter()
                    .copied()
                    .filter(|id| self.unit(*id).is_some_and(|u| u.army == army && u.can_fire_surface && u.carried_by.is_none()))
                    .collect();
                v.sort_unstable();
                v.dedup();
                for id in &v {
                    self.push_order(*id, UnitOrder::AttackGround { x: p.0, y: p.1 }, queue);
                }
                !v.is_empty()
            }
            Command::Guard { units, target, queue } => {
                if !self.unit(target).is_some_and(|t| t.alive && self.are_allies(army, t.army)) {
                    return false;
                }
                let mut v: Vec<UnitId> = units
                    .iter()
                    .copied()
                    .filter(|&id| id != target && self.unit(id).is_some_and(|u| u.army == army && u.speed > 0 && u.carried_by.is_none()))
                    .collect();
                v.sort_unstable();
                v.dedup();
                for id in &v {
                    self.push_order(*id, UnitOrder::Guard(target), queue);
                }
                !v.is_empty()
            }
            Command::Ping { kind, x, y } => {
                if kind > 2 {
                    return false;
                }
                let p = self.clamp_to_map(x, y);
                if self.pings.len() >= 32 {
                    self.pings.pop_front();
                }
                self.pings.push_back((self.tick, army, kind, p));
                true
            }
            Command::MoveWaypoint { units, index, x, y } => {
                let p = self.clamp_to_map(x, y);
                let mut any = false;
                for id in units {
                    let Some(u) = self.units.get_mut(&id).filter(|u| u.army == army && u.alive) else { continue };
                    let Some(o) = u.orders.get_mut(index as usize) else { continue };
                    match o {
                        UnitOrder::Move { x, y } | UnitOrder::AttackMove { x, y } | UnitOrder::Patrol { x, y } => {
                            (*x, *y) = p;
                            any = true;
                            if index == 0 {
                                u.path.clear();
                                u.path_goal = None;
                            }
                        }
                        _ => {}
                    }
                }
                any
            }
            Command::Ferry { transports, from, to, queue } => {
                let (from, to) = (self.clamp_to_map(from.0, from.1), self.clamp_to_map(to.0, to.1));
                let mut v: Vec<UnitId> = transports
                    .iter()
                    .copied()
                    .filter(|id| {
                        self.unit(*id).is_some_and(|u| {
                            u.army == army && u.carried_by.is_none() && self.catalog.get(&u.type_id).is_some_and(|t| t.ext.transport.is_some())
                        })
                    })
                    .collect();
                v.sort_unstable();
                v.dedup();
                for id in &v {
                    self.push_order(*id, UnitOrder::Ferry { from, to }, queue);
                }
                !v.is_empty()
            }
            Command::SetAutoRefuel { units, enabled } => {
                let mut any = false;
                for id in units {
                    if let Some(u) = self.units.get_mut(&id).filter(|u| u.army == army && u.alive) {
                        u.auto_refuel = enabled;
                        any = true;
                    }
                }
                any
            }
            Command::Unload { transports, x, y, queue } => {
                let p = self.clamp_to_map(x, y);
                let mut ids = transports;
                ids.sort_unstable();
                ids.dedup();
                let ids: Vec<UnitId> = ids
                    .into_iter()
                    .filter(|&id| {
                        self.unit(id).is_some_and(|t| {
                            t.army == army && self.catalog.get(&t.type_id).is_some_and(|x| x.ext.transport.is_some())
                        })
                    })
                    .collect();
                let n = ids.len();
                for (i, id) in ids.iter().enumerate() {
                    let (sx, sy) = Self::formation_slot(i, n, p.0, p.1);
                    let (sx, sy) = self.clamp_to_map(sx, sy);
                    self.push_order(*id, UnitOrder::Unload { x: sx, y: sy }, queue);
                }
                n > 0
            }
            Command::Enhance { unit, name } => {
                let Some(u) = self.unit(unit) else {
                    return false;
                };
                if u.army != army || u.carried_by.is_some() {
                    return false;
                }
                let Some(spec) = self.catalog.get(&u.type_id).and_then(|t| t.ext.enhancement(&name)) else {
                    return false;
                };
                if !Self::enhancement_allowed(&u.enh, spec) {
                    return false;
                }
                self.push_order(unit, UnitOrder::Enhance(name), false);
                true
            }
            Command::SetRepeat { factories, enabled } => {
                let mut any = false;
                let mut ids = factories;
                ids.sort_unstable();
                ids.dedup();
                for id in ids {
                    let producer = self.unit(id).is_some_and(|u| self.is_producer(u));
                    if let Some(u) = self.units.get_mut(&id)
                        && u.alive
                        && u.army == army
                        && producer
                    {
                        u.repeat_queue = enabled;
                        any = true;
                    }
                }
                any
            }
            Command::CancelProduce { factory, unit_type, count } => {
                let producer = self.unit(factory).is_some_and(|u| self.is_producer(u));
                let Some(u) = self.units.get_mut(&factory) else {
                    return false;
                };
                if !u.alive || u.army != army || !producer || count == 0 {
                    return false;
                }
                let mut left = count.min(500);
                let mut i = u.production.len();
                while i > 0 && left > 0 {
                    i -= 1;
                    if u.production[i] == unit_type {
                        u.production.remove(i);
                        left -= 1;
                    }
                }
                let cancel_active = left > 0
                    && !matches!(u.orders.front(), Some(UnitOrder::Upgrade(_)))
                    && u.build.as_ref().is_some_and(|j| j.unit_type == unit_type);
                if cancel_active {
                    self.cancel_build(factory);
                    left -= 1;
                }
                left < count.min(500)
            }
        }
    }

    // ----- simulation ----------------------------------------------------------

    fn next_rng(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn spawn_unit(&mut self, army: ArmyId, type_id: &str, pos: (i64, i64)) -> Option<UnitId> {
        let t = self.catalog.get(type_id)?.clone();
        // Phase 32: a finished egg hatches straight into its unit.
        if let Some(h) = t.ext.hatch.as_deref()
            && h != t.id
            && self.catalog.get(h).is_some()
        {
            return self.spawn_unit(army, h, pos);
        }
        let id = self.next_id;
        self.next_id += 1;
        let weapons: Vec<WeaponState> = t
            .weapons
            .iter()
            .map(WeaponState::from_type)
            .collect();
        let can_aa = weapons.iter().any(|w| w.anti_air);
        let can_surf = weapons.iter().any(|w| w.anti_surface);
        let shield_init = t.shield_max_hp;
        if t.role.is_structure() {
            self.walls_dirty = true;
        }
        let spawn_pods = t.ext.pods.clone();
        self.units.insert(
            id,
            Unit {
                id,
                army,
                type_id: t.id.clone(),
                role: t.role,
                tier: t.tier,
                domain: t.domain,
                pos: Vec2 { x: pos.0, y: pos.1 },
                altitude: if t.domain == NavDomain::Air { t.elevation } else { 0 },
                hp: t.max_hp,
                max_hp: t.max_hp,
                alive: true,
                weapons,
                speed: t.speed_per_tick,
                radius: t.radius,
                max_range: t.max_range(),
                can_fire_aa: can_aa,
                can_fire_surface: can_surf,
                vision_radius: t.vision_radius,
                radar_radius: t.radar_radius,
                shield_hp: shield_init,
                shield_max_hp: t.shield_max_hp,
                shield_radius: t.shield_radius,
                shield_on: t.shield_max_hp > 0,
                shield_recharge_cd: 0,
                vet_kills: 0,
                vet_level: 0,
                hydro_point: None,
                silo_missiles: 0,
                silo_progress: 0,
                adjacency_bonus_pct: 0,
                turret_yaw_mrad: 0,
                moving: false,
                orders: VecDeque::new(),
                path: VecDeque::new(),
                path_goal: None,
                build: None,
                production: VecDeque::new(),
                rally: None,
                mass_point: None,
                last_shot: None,
                last_hit_by: None,
                last_hit_by_unit: None,
                carried_by: None,
                cargo: Vec::new(),
                repeat_queue: false,
                enh: EnhancementState::default(),
                radar_stealth: false,
                cloaked: false,
                omni_radius: t.ext.intel.omni_radius,
                shield_personal: false,
                shield_upkeep: t.shield_energy_upkeep,
                shield_regen: t.shield_regen_per_tick,
                intercept_cd: 0,
                still_ticks: 0,
                omni_mask: 0,
                height: t.size[1].max(SCALE / 4),
                vel: (0, 0),
                fuel: t.ext.fuel_ticks,
                auto_refuel: t.ext.fuel_ticks > 0,
                drone_of: None,
                pod_timer: 0,
                stunned: 0,
                untargetable: t.ext.untargetable,
                fire_state: FireState::FireAtWill,
                build_paused: false,
                eco: [t.mass_per_tick, t.energy_per_tick, t.storage_mass, t.storage_energy],
            },
        );
        // Phase 31: a finished engineering station comes with its pod drones.
        for pod in spawn_pods {
            self.spawn_pod(id, &pod);
        }
        // Phase 33: an orbital launch centre puts its satellite into orbit.
        if let Some(sat) = t.ext.satellite.clone() {
            self.spawn_pod(id, &sat);
        }
        Some(id)
    }

    /// Phase 31: spawns pod drone `pod_type` docked at station `station`.
    fn spawn_pod(&mut self, station: UnitId, pod_type: &str) -> Option<UnitId> {
        let (army, pos) = self.units.get(&station).map(|s| (s.army, (s.pos.x, s.pos.y)))?;
        let pid = self.spawn_unit(army, pod_type, pos)?;
        let p = self.units.get_mut(&pid).unwrap();
        p.drone_of = Some(station);
        p.auto_refuel = false;
        Some(pid)
    }

    /// Phase 31: live pod drones of station `sid`.
    pub fn pods_of(&self, sid: UnitId) -> Vec<UnitId> {
        self.units.values().filter(|u| u.alive && u.drone_of == Some(sid)).map(|u| u.id).collect()
    }

    /// Phase 31: work an engineering station at `p` with radius `r` should pick up for
    /// `army`: the nearest allied construction (builder site, factory, in-place upgrade) in
    /// range, else the nearest damaged allied surface unit — ties by id.
    fn station_target(&self, sid: UnitId, army: ArmyId, p: (i64, i64), r: i64) -> Option<UnitOrder> {
        let lim = (r as i128) * (r as i128);
        let mut assist: Option<(i128, UnitId)> = None;
        let mut repair: Option<(i128, UnitId)> = None;
        for u in self.units.values() {
            if !u.alive || u.id == sid || u.carried_by.is_some() || u.drone_of.is_some() || !self.are_allies(army, u.army) {
                continue;
            }
            if let Some(j) = &u.build {
                let site = if u.role.is_builder() { (j.x, j.y) } else { (u.pos.x, u.pos.y) };
                let d = dist_sq(p, site);
                if d <= lim && assist.is_none_or(|(bd, bid)| (d, u.id) < (bd, bid)) {
                    assist = Some((d, u.id));
                }
            } else if u.hp < u.max_hp && u.domain != NavDomain::Air {
                let d = dist_sq(p, (u.pos.x, u.pos.y));
                if d <= lim && repair.is_none_or(|(bd, bid)| (d, u.id) < (bd, bid)) {
                    repair = Some((d, u.id));
                }
            }
        }
        assist.map(|(_, id)| UnitOrder::Assist(id)).or(repair.map(|(_, id)| UnitOrder::Repair(id)))
    }

    /// Phase 31 engineering stations (ascending id): pods die with their station; stations
    /// rebuild missing pods (pod build time at the station's rate, paid on completion);
    /// every second an idle-or-finished station (or each of its pods) re-targets the nearest
    /// allied construction / damaged unit in its radius. Pods fly out to work and return to
    /// the station when idle; work outside the radius is dropped.
    fn step_eng_stations(&mut self) {
        if !self.units.values().any(|u| u.role == Role::EngStation || u.drone_of.is_some()) {
            return;
        }
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        for &id in &ids {
            let Some(parent) = self.units[&id].drone_of else { continue };
            if !self.units.get(&parent).is_some_and(|s| s.alive) {
                self.units.get_mut(&id).unwrap().hp = 0;
            }
        }
        let retarget = self.tick.is_multiple_of(TICK_HZ);
        for &sid in &ids {
            let Some(s) = self.units.get(&sid).filter(|s| s.alive && s.role == Role::EngStation && s.carried_by.is_none()) else {
                continue;
            };
            let (army, sp, upgrading) = (s.army, (s.pos.x, s.pos.y), matches!(s.orders.front(), Some(UnitOrder::Upgrade(_))));
            let Some(t) = self.catalog.get(&s.type_id) else { continue };
            let (pods_spec, radius, rate) = (t.ext.pods.clone(), t.build_range, t.build_rate.max(1));
            // Rebuild missing pods.
            let live = self.pods_of(sid);
            if !upgrading && live.len() < pods_spec.len() {
                let pod_type = pods_spec[live.len()].clone();
                let (work, cm, ce) = self.catalog.get(&pod_type).map_or((1, 0, 0), |p| (p.build_work, p.cost_mass, p.cost_energy));
                let st = self.units.get_mut(&sid).unwrap();
                st.pod_timer = st.pod_timer.saturating_add(1);
                let a = &self.armies[army as usize];
                if st.pod_timer as i64 * rate >= work && a.mass >= cm && a.energy >= ce {
                    st.pod_timer = 0;
                    self.armies[army as usize].mass -= cm;
                    self.armies[army as usize].energy -= ce;
                    self.spawn_pod(sid, &pod_type);
                }
            }
            if !retarget || upgrading {
                continue;
            }
            let target = self.station_target(sid, army, sp, radius);
            let workers: Vec<UnitId> = if pods_spec.is_empty() { vec![sid] } else { self.pods_of(sid) };
            for wid in workers {
                let w = self.units.get_mut(&wid).unwrap();
                let busy = matches!(w.orders.front(), Some(UnitOrder::Assist(t) | UnitOrder::Repair(t)) if Some(*t) == target.as_ref().and_then(|o| match o { UnitOrder::Assist(x) | UnitOrder::Repair(x) => Some(*x), _ => None }));
                if busy {
                    continue;
                }
                w.orders.clear();
                w.path.clear();
                w.path_goal = None;
                match &target {
                    Some(o) => w.orders.push_back(o.clone()),
                    None if wid != sid && dist_sq((w.pos.x, w.pos.y), sp) > (2 * SCALE as i128).pow(2) => {
                        w.orders.push_back(UnitOrder::Move { x: sp.0, y: sp.1 });
                    }
                    None => {}
                }
            }
        }
    }

    pub fn rebuild_spatial(&mut self) {
        let mut snaps = Vec::with_capacity(self.units.len());
        for u in self.units.values().filter(|u| u.alive && u.carried_by.is_none()) {
            let team = self.armies.get(u.army as usize).map(|a| a.team).unwrap_or(u.army);
            snaps.push(UnitSpatialSnap {
                id: u.id,
                army: u.army,
                team,
                domain: u.domain,
                x: u.pos.x,
                y: u.pos.y,
                radius: u.radius,
                speed: u.speed,
                is_struct: u.role.is_structure(),
                is_wall: u.role == Role::Wall,
                height: u.height,
                untargetable: u.untargetable,
                can_fire_aa: u.can_fire_aa,
                can_fire_surface: u.can_fire_surface,
                max_range: u.max_range,
                cloaked: u.cloaked,
                omni_mask: u.omni_mask,
            });
        }
        let spatial = Arc::make_mut(&mut self.spatial);
        spatial.rebuild_with_teams(
            snaps
                .iter()
                .enumerate()
                .map(|(i, s)| (i as u32, s.x, s.y, s.team)),
        );
        self.spatial_snap = Arc::new(snaps);
    }

    fn recompute_caps(&mut self) {
        let mut caps = vec![(0i64, 0i64); self.armies.len()];
        for u in self.units.values().filter(|u| u.alive) {
            caps[u.army as usize].0 += u.eco[2];
            caps[u.army as usize].1 += u.eco[3];
        }
        for (a, (m, e)) in self.armies.iter_mut().zip(caps) {
            a.mass_cap = m;
            a.energy_cap = e;
        }
    }

    fn recompute_adjacency(&mut self) {
        let structs: Vec<(UnitId, ArmyId, Role, i64, i64, i64)> = self
            .units
            .values()
            .filter(|u| u.alive && u.role.is_structure())
            .map(|u| (u.id, u.army, u.role, u.pos.x, u.pos.y, u.radius))
            .collect();

        for (id, army, role, x, y, r) in &structs {
            let mut bonus = 0i64;
            for (oid, oarmy, orole, ox, oy, orad) in &structs {
                if oid == id || oarmy != army {
                    continue;
                }
                let max_d = r + orad + 6 * SCALE;
                if dist_sq((*x, *y), (*ox, *oy)) > (max_d as i128) * (max_d as i128) {
                    continue;
                }
                match (role, orole) {
                    (Role::Extractor, Role::MassStorage) => bonus += 12,
                    (Role::Generator | Role::Hydro, Role::EnergyStorage) => bonus += 12,
                    (Role::Factory | Role::AirFactory | Role::NavalFactory, Role::Generator | Role::Hydro) => {
                        bonus += 10;
                    }
                    _ => {}
                }
            }
            if let Some(u) = self.units.get_mut(id) {
                u.adjacency_bonus_pct = bonus.min(50);
            }
        }
    }

    /// Advance one fixed tick. Every phase is strictly ordered by ascending ID.
    pub fn step(&mut self) {
        if self.result.is_some() {
            return;
        }
        use crate::profiler::scope;
        let _p = scope("sim.tick");
        self.tick += 1;
        let _ = self.next_rng();
        { let _p = scope("sim.commands"); self.apply_commands(); }
        for u in self.units.values_mut() {
            u.stunned = u.stunned.saturating_sub(1);
        }
        if self.tick == 1 || self.tick % 10 == 0 {
            let _p = scope("sim.adjacency");
            self.recompute_adjacency();
        }
        { let _p = scope("sim.economy"); self.step_income(); self.step_shields(); self.step_enhancements(); self.step_silos(); }
        { let _p = scope("sim.construction"); self.step_eng_stations(); self.step_construction(); self.sync_walls(); }
        { let _p = scope("sim.spatial"); self.rebuild_spatial(); }
        // Phase 29: per-tick ground velocity (for target leading) = position delta over the
        // movement, separation and transport passes.
        for u in self.units.values_mut() {
            u.vel = (u.pos.x, u.pos.y);
        }
        { let _p = scope("sim.movement"); self.step_movement(); }
        { let _p = scope("sim.separation"); self.step_separation(); }
        { let _p = scope("sim.transport"); self.step_transport(); self.step_ferries(); self.step_staging(); }
        for u in self.units.values_mut() {
            u.vel = (u.pos.x - u.vel.0, u.pos.y - u.vel.1);
        }
        { let _p = scope("sim.intel"); self.step_intel(); }
        { let _p = scope("sim.spatial"); self.rebuild_spatial(); }
        { let _p = scope("sim.weapons"); self.step_weapons(); }
        { let _p = scope("sim.projectiles"); self.step_projectiles(); }
        { let _p = scope("sim.cleanup"); self.step_cleanup(); }
        if let Some(mut m) = self.mission.take() {
            m.step(self);
            if let Some(r) = m.outcome {
                self.result = Some(r);
            }
            self.mission = Some(m);
        }
        { let _p = scope("sim.spatial"); self.rebuild_spatial(); }
    }

    pub fn run(&mut self, n: u64) {
        for _ in 0..n {
            if self.result.is_some() {
                break;
            }
            self.step();
        }
    }

    fn step_income(&mut self) {
        // Storage caps and income in one pass over the cached per-unit economy.
        let mut caps = vec![(0i64, 0i64); self.armies.len()];
        let mut income = vec![(0i64, 0i64); self.armies.len()];
        for u in self.units.values().filter(|u| u.alive) {
            let a = u.army as usize;
            caps[a].0 += u.eco[2];
            caps[a].1 += u.eco[3];
            if u.eco[0] != 0 || u.eco[1] != 0 {
                let mult = 100 + u.adjacency_bonus_pct;
                income[a].0 += u.eco[0] * mult / 100;
                income[a].1 += u.eco[1] * mult / 100;
            }
            // Phase 18: Resource Allocation enhancements.
            income[a].0 += u.enh.mass_per_tick;
            income[a].1 += u.enh.energy_per_tick;
        }
        for (a, (m, e)) in self.armies.iter_mut().zip(caps) {
            a.mass_cap = m;
            a.energy_cap = e;
        }
        let mut overflow = vec![(0i64, 0i64); self.armies.len()];
        for (i, (a, (m, e))) in self.armies.iter_mut().zip(income).enumerate() {
            a.income_mass = m;
            a.income_energy = e;
            let (want_m, want_e) = (a.mass + m, a.energy + e);
            a.mass = (a.mass + m).min(a.mass_cap.max(a.mass));
            a.energy = (a.energy + e).min(a.energy_cap.max(a.energy));
            a.mass = a.mass.min(a.mass_cap.max(0)).max(0);
            a.energy = a.energy.min(a.energy_cap.max(0)).max(0);
            overflow[i] = ((want_m - a.mass).clamp(0, m.max(0)), (want_e - a.energy).clamp(0, e.max(0)));
        }
        self.share_overflow(&overflow);
    }

    /// Phase 18 team economy: income that did not fit in a full storage flows to allied
    /// (same team, undefeated) armies, split evenly in ascending army order; whatever no
    /// ally can store is lost, exactly like FA's resource overflow sharing.
    fn share_overflow(&mut self, overflow: &[(i64, i64)]) {
        for (src, &(om, oe)) in overflow.iter().enumerate() {
            if om <= 0 && oe <= 0 {
                continue;
            }
            let team = self.armies[src].team;
            let allies: Vec<usize> = self
                .armies
                .iter()
                .enumerate()
                .filter(|(i, a)| *i != src && a.team == team && !a.defeated)
                .map(|(i, _)| i)
                .collect();
            if allies.is_empty() {
                continue;
            }
            for (left0, is_mass) in [(om, true), (oe, false)] {
                let mut left = left0;
                let k = allies.len() as i64;
                for (n, &ai) in allies.iter().enumerate() {
                    if left <= 0 {
                        break;
                    }
                    // Even split; what a full ally cannot take rolls over to the next.
                    let share = left / (k - n as i64).max(1);
                    let a = &mut self.armies[ai];
                    let (cur, cap) = if is_mass { (&mut a.mass, a.mass_cap) } else { (&mut a.energy, a.energy_cap) };
                    let room = (cap - *cur).max(0);
                    let give = share.min(room).max(0);
                    *cur += give;
                    left -= give;
                }
            }
        }
    }

    fn step_shields(&mut self) {
        let ids: Vec<UnitId> = self.units.values().filter(|u| u.shield_max_hp > 0).map(|u| u.id).collect();
        for id in ids {
            let (army, max_sh, upkeep, regen, on, cur_sh, cd) = match self.units.get(&id) {
                Some(u) if u.alive && u.shield_max_hp > 0 => (
                    u.army,
                    u.shield_max_hp,
                    u.shield_upkeep,
                    u.shield_regen,
                    u.shield_on,
                    u.shield_hp,
                    u.shield_recharge_cd,
                ),
                _ => continue,
            };
            if !on {
                continue;
            }
            if cd > 0 {
                if let Some(u) = self.units.get_mut(&id) {
                    u.shield_recharge_cd -= 1;
                    if u.shield_recharge_cd == 0 {
                        u.shield_hp = (max_sh / 5).max(1);
                    }
                }
                continue;
            }
            let a = &mut self.armies[army as usize];
            if upkeep > 0 && a.energy < upkeep {
                // Energy stall collapses shield
                if let Some(u) = self.units.get_mut(&id) {
                    u.shield_hp = 0;
                    u.shield_recharge_cd = SHIELD_RECHARGE_DELAY;
                }
                continue;
            }
            a.energy -= upkeep;
            if cur_sh < max_sh
                && let Some(u) = self.units.get_mut(&id)
            {
                u.shield_hp = (u.shield_hp + regen).min(max_sh);
            }
        }
    }

    fn step_silos(&mut self) {
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        for id in ids {
            let (army, missiles, progress, cm, ce, pos) = match self.units.get(&id) {
                Some(u) if u.alive && u.role == Role::MissileSilo && u.silo_missiles < 4 => {
                    let t = self.catalog.get(&u.type_id);
                    (
                        u.army,
                        u.silo_missiles,
                        u.silo_progress,
                        t.map(|x| x.missile_cost_mass).unwrap_or(250 * SCALE),
                        t.map(|x| x.missile_cost_energy).unwrap_or(5_000 * SCALE),
                        (u.pos.x, u.pos.y, u.altitude),
                    )
                }
                // Phase 18: strategic missile defense builds its interceptor ammo.
                Some(u)
                    if u.alive
                        && u.role == Role::MissileDefense
                        && self
                            .catalog
                            .get(&u.type_id)
                            .is_some_and(|t| t.ext.ammo_cap > u.silo_missiles) =>
                {
                    (
                        u.army,
                        u.silo_missiles,
                        u.silo_progress,
                        ANTI_NUKE_COST_MASS,
                        ANTI_NUKE_COST_ENERGY,
                        (u.pos.x, u.pos.y, u.altitude),
                    )
                }
                _ => continue,
            };
            let _ = missiles;
            let step_m = (cm / 150).max(SCALE);
            let step_e = (ce / 150).max(10 * SCALE);
            let a = &mut self.armies[army as usize];
            if a.mass >= step_m && a.energy >= step_e {
                a.mass -= step_m;
                a.energy -= step_e;
                let next_p = progress + 1;
                if next_p >= 150 {
                    if let Some(u) = self.units.get_mut(&id) {
                        u.silo_missiles += 1;
                        u.silo_progress = 0;
                    }
                    self.emit_event(SimEvent::BuildSpark { pos, army });
                } else if let Some(u) = self.units.get_mut(&id) {
                    u.silo_progress = next_p;
                }
            }
        }
    }

    fn new_job(
        &mut self,
        builder_rate: i64,
        unit_type: &str,
        x: i64,
        y: i64,
        mp: Option<usize>,
    ) -> Option<BuildJob> {
        let t = self.catalog.get(unit_type)?;
        let project = ConstructionProject::new(
            self.next_project,
            t.cost_mass,
            t.cost_energy,
            t.build_work,
            builder_rate.max(1),
        );
        self.next_project += 1;
        Some(BuildJob {
            unit_type: unit_type.to_string(),
            x,
            y,
            project,
            mass_point: mp,
        })
    }

    fn progress_job_with_rate(
        armies: &mut [Army],
        army: ArmyId,
        job: &mut BuildJob,
        effective_rate: i64,
    ) -> bool {
        let saved = job.project.build_rate;
        job.project.build_rate = effective_rate.max(1);
        let a = &mut armies[army as usize];
        let mut stock = (a.mass, a.energy);
        let done = job.project.step(&mut stock);
        a.mass = stock.0;
        a.energy = stock.1;
        job.project.build_rate = saved;
        done
    }

    fn complete_upgrade(&mut self, id: UnitId, target_type: &str) {
        let Some(new_t) = self.catalog.get(target_type).cloned() else {
            return;
        };
        let Some(u) = self.units.get_mut(&id) else {
            return;
        };
        let old_max = u.max_hp.max(1);
        let new_hp = ((u.hp as i128 * new_t.max_hp as i128) / old_max as i128)
            .max(1)
            .min(new_t.max_hp as i128) as i64;
        u.type_id = new_t.id.clone();
        u.role = new_t.role;
        u.tier = new_t.tier;
        u.max_hp = new_t.max_hp;
        u.hp = new_hp;
        u.shield_max_hp = new_t.shield_max_hp;
        u.shield_radius = new_t.shield_radius;
        u.shield_upkeep = new_t.shield_energy_upkeep;
        u.shield_regen = new_t.shield_regen_per_tick;
        u.shield_personal = false;
        u.omni_radius = new_t.ext.intel.omni_radius;
        if new_t.shield_max_hp > 0 {
            u.shield_on = true;
            u.shield_hp = new_t.shield_max_hp;
        }
        u.max_range = new_t.max_range();
        u.height = new_t.size[1].max(SCALE / 4);
        u.eco = [new_t.mass_per_tick, new_t.energy_per_tick, new_t.storage_mass, new_t.storage_energy];
        if new_t.role.is_structure() {
            self.walls_dirty = true;
        }
        u.vision_radius = new_t.vision_radius;
        u.radar_radius = new_t.radar_radius;
        u.weapons = new_t
            .weapons
            .iter()
            .map(WeaponState::from_type)
            .collect();
        u.can_fire_aa = u.weapons.iter().any(|w| w.anti_air);
        u.can_fire_surface = u.weapons.iter().any(|w| w.anti_surface);
        let ev_pos = (u.pos.x, u.pos.y, u.altitude);
        let ev_army = u.army;
        self.emit_event(SimEvent::UpgradeComplete { pos: ev_pos, army: ev_army });
    }

    /// Phase 18: enhancement maintenance (stealth / cloak / jammer / teleporter energy) and
    /// enhancement regeneration. Inert for units without enhancements.
    fn step_enhancements(&mut self) {
        let ids: Vec<UnitId> = self
            .units
            .values()
            .filter(|u| u.alive && (u.enh.upkeep_per_tick > 0 || u.enh.regen_per_tick > 0))
            .map(|u| u.id)
            .collect();
        for id in ids {
            let (army, upkeep, regen) = {
                let u = &self.units[&id];
                (u.army, u.enh.upkeep_per_tick, u.enh.regen_per_tick)
            };
            let mut stalled = false;
            if upkeep > 0 {
                let a = &mut self.armies[army as usize];
                if a.energy >= upkeep {
                    a.energy -= upkeep;
                } else {
                    stalled = true;
                }
            }
            let u = self.units.get_mut(&id).unwrap();
            u.enh.stalled = stalled;
            if regen > 0 && u.hp < u.max_hp {
                u.hp = (u.hp + regen).min(u.max_hp);
            }
        }
    }

    /// Phase 18: finish enhancement `name` on unit `id` (install or remove), then refold
    /// every installed enhancement's effects into the unit.
    fn complete_enhancement(&mut self, id: UnitId, name: &str) {
        let Some(spec) = self
            .units
            .get(&id)
            .and_then(|u| self.catalog.get(&u.type_id))
            .and_then(|t| t.ext.enhancement(name))
            .cloned()
        else {
            return;
        };
        let Some(u) = self.units.get_mut(&id) else {
            return;
        };
        if spec.is_removal() {
            u.enh.installed.retain(|(s, n)| *s != spec.slot && !spec.removes.contains(n));
        } else {
            u.enh.installed.retain(|(s, _)| *s != spec.slot);
            u.enh.installed.push((spec.slot.clone(), spec.name.clone()));
        }
        u.enh.installed.sort();
        let ev = (u.pos.x, u.pos.y, u.altitude + 2 * SCALE);
        let army = u.army;
        self.refresh_enhancement_effects(id);
        self.emit_event(SimEvent::UpgradeComplete { pos: ev, army });
    }

    /// Recomputes enhancement-derived stats (build rate, tech level, HP bonus, regen,
    /// production, upkeep, intel, primary-weapon mods, vision/omni, shields) from the
    /// installed set. HP bonuses apply as deltas so veterancy bonuses survive.
    fn refresh_enhancement_effects(&mut self, id: UnitId) {
        let Some(t) = self.units.get(&id).and_then(|u| self.catalog.get(&u.type_id)).cloned() else {
            return;
        };
        let u = self.units.get_mut(&id).unwrap();
        let specs: Vec<&EnhancementSpec> = u
            .enh
            .installed
            .iter()
            .filter_map(|(_, n)| t.ext.enhancement(n))
            .collect();
        let mut st = EnhancementState {
            installed: u.enh.installed.clone(),
            stalled: u.enh.stalled,
            ..EnhancementState::default()
        };
        let (mut range, mut rof, mut dmg_add, mut dmg_radius) = (0i64, 0i64, 0i64, 0i64);
        let mut vision = t.vision_radius;
        let mut omni = t.ext.intel.omni_radius;
        let mut shield: Option<&EnhancementSpec> = None;
        let mut health_bonus = 0i64;
        for s in &specs {
            st.build_rate = st.build_rate.max(s.new_build_rate);
            st.engineering_tier = st.engineering_tier.max(s.engineering_tier);
            health_bonus += s.add_health;
            st.regen_per_tick += s.regen_per_tick;
            st.mass_per_tick += s.mass_per_tick;
            st.energy_per_tick += s.energy_per_tick;
            if s.shield_max_hp > 0 {
                shield = Some(s);
            } else {
                st.upkeep_per_tick += s.upkeep_per_tick;
            }
            st.radar_stealth |= s.radar_stealth;
            st.cloak |= s.cloak;
            st.jammer_radius = st.jammer_radius.max(s.jammer_radius);
            range = range.max(s.new_max_radius);
            rof = rof.max(s.new_rate_of_fire_milli);
            dmg_add += s.damage_add;
            dmg_radius = dmg_radius.max(s.new_damage_radius);
            vision = vision.max(s.vision_radius);
            omni = omni.max(s.omni_radius);
        }
        let delta = health_bonus - u.enh.health_bonus;
        u.max_hp = (u.max_hp + delta).max(1);
        if delta > 0 {
            u.hp += delta;
        }
        u.hp = u.hp.clamp(1, u.max_hp);
        st.health_bonus = health_bonus;

        let old_cd: Vec<u32> = u.weapons.iter().map(|w| w.cooldown).collect();
        let old_yaw: Vec<i64> = u.weapons.iter().map(|w| w.yaw).collect();
        u.weapons = t
            .weapons
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let mut r = w.range;
                let mut reset = w.reset;
                let mut damage = w.damage;
                let mut splash = w.splash_radius;
                if i == 0 {
                    r = r.max(range);
                    if rof > 0 {
                        reset = combat::cooldown_reset_ticks(TICK_HZ, rof);
                    }
                    damage = (damage + dmg_add).max(1);
                    splash = splash.max(dmg_radius);
                }
                WeaponState {
                    damage,
                    range_sq: (r as i128) * (r as i128),
                    reset,
                    cooldown: old_cd.get(i).copied().unwrap_or(0).min(reset),
                    splash_radius: splash,
                    yaw: old_yaw.get(i).copied().unwrap_or(0),
                    ..WeaponState::from_type(w)
                }
            })
            .collect();
        u.max_range = u.weapons.iter().map(|w| isqrt(w.range_sq)).max().unwrap_or(0);
        u.can_fire_aa = u.weapons.iter().any(|w| w.anti_air);
        u.can_fire_surface = u.weapons.iter().any(|w| w.anti_surface);
        u.vision_radius = vision;
        u.omni_radius = omni;

        match shield {
            Some(s) => {
                let fresh = u.shield_max_hp != s.shield_max_hp;
                u.shield_max_hp = s.shield_max_hp;
                u.shield_radius = s.shield_radius;
                u.shield_regen = s.shield_regen_per_tick;
                u.shield_upkeep = s.upkeep_per_tick;
                u.shield_personal = s.personal_shield;
                if fresh {
                    u.shield_on = true;
                    u.shield_hp = s.shield_max_hp;
                    u.shield_recharge_cd = 0;
                }
            }
            None => {
                u.shield_max_hp = t.shield_max_hp;
                u.shield_radius = t.shield_radius;
                u.shield_regen = t.shield_regen_per_tick;
                u.shield_upkeep = t.shield_energy_upkeep;
                u.shield_personal = false;
                u.shield_hp = u.shield_hp.min(t.shield_max_hp);
                u.shield_on = t.shield_max_hp > 0;
            }
        }
        u.enh = st;
    }

    /// `(build_rate, build_range)` of a unit including Phase 18 enhancement overrides.
    fn unit_build_stats(&self, u: &Unit) -> Option<(i64, i64)> {
        self.catalog.get(&u.type_id).map(|t| {
            let rate = if u.enh.build_rate > 0 { u.enh.build_rate } else { t.build_rate };
            (rate, t.build_range)
        })
    }

    fn step_construction(&mut self) {
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        // Phase 18 unit cap: live units + started constructions per army (only when capped).
        let mut cap_counts: Vec<u32> = vec![0; self.armies.len()];
        if self.options.unit_cap > 0 {
            for u in self.units.values().filter(|u| u.alive) {
                cap_counts[u.army as usize] += 1 + u32::from(Self::job_spawns_unit(u));
            }
        }
        let cap = self.options.unit_cap;
        let capped = |counts: &[u32], army: ArmyId| cap > 0 && counts[army as usize] >= cap;

        // Pre-compute allied build-rate assist contributions in ascending ID order.
        let mut assist_bonus: BTreeMap<UnitId, i64> = BTreeMap::new();
        for &id in &ids {
            let Some(u) = self.units.get(&id) else { continue };
            if !u.alive || !(u.role.is_builder() || u.role == Role::EngStation) || u.carried_by.is_some() {
                continue;
            }
            if let Some(UnitOrder::Assist(target_id)) = u.orders.front()
                && let Some(target) = self.units.get(target_id).filter(|t| t.alive && self.are_allies(u.army, t.army))
            {
                let (br, brange) = self.unit_build_stats(u).unwrap_or((0, 0));
                let reach = (brange + target.radius + u.radius) as i128;
                if dist_sq((u.pos.x, u.pos.y), (target.pos.x, target.pos.y)) <= reach * reach {
                    *assist_bonus.entry(*target_id).or_insert(0) += br;
                }
            }
        }

        for id in ids {
            // Phase 54: plain mobile units with nothing to build skip the whole pass.
            if self.units.get(&id).is_some_and(|u| {
                u.role == Role::Mobile
                    && u.build.is_none()
                    && u.production.is_empty()
                    && !matches!(
                        u.orders.front(),
                        Some(UnitOrder::Build { .. } | UnitOrder::Repair(_) | UnitOrder::Assist(_) | UnitOrder::Reclaim { .. } | UnitOrder::Upgrade(_) | UnitOrder::Enhance(_))
                    )
            }) {
                continue;
            }
            let (alive, role, army, pos, radius, rate, head, adj, silocnt, type_id) =
                match self.units.get(&id) {
                    Some(u) => (
                        u.alive && u.carried_by.is_none(),
                        u.role,
                        u.army,
                        (u.pos.x, u.pos.y),
                        u.radius,
                        self.unit_build_stats(u),
                        u.orders.front().cloned(),
                        u.adjacency_bonus_pct,
                        u.silo_missiles,
                        u.type_id.clone(),
                    ),
                    None => continue,
                };
            if !alive || self.units[&id].stunned > 0 || self.units[&id].build_paused {
                continue;
            }

            // Missile launch execution for MissileSilo
            if role == Role::MissileSilo
                && let Some(UnitOrder::LaunchMissile { x, y }) = head
            {
                self.units.get_mut(&id).unwrap().orders.pop_front();
                if silocnt > 0 {
                    self.units.get_mut(&id).unwrap().silo_missiles -= 1;
                    let ut = self.catalog.get(&type_id);
                    let is_nuke = ut.is_some_and(|t| t.tier >= 3);
                    let dmg = ut.map(|t| t.missile_damage).unwrap_or(6_000 * SCALE);
                    let spl = ut.map(|t| t.missile_splash).unwrap_or(4 * SCALE);
                    let missile_stun = ut.and_then(|t| t.missile_stun);
                    self.spawn_projectile_full(
                        army,
                        Vec2 { x: pos.0, y: pos.1 },
                        Vec2 { x, y },
                        if is_nuke { 8 * SCALE / TICK_HZ as i64 } else { 14 * SCALE / TICK_HZ as i64 },
                        dmg,
                        spl,
                        false,
                        true,
                        is_nuke,
                        false,
                    );
                    if let Some((ticks, radius)) = missile_stun
                        && let Some(p) = self.projectiles.get_mut(&(self.next_projectile_id - 1))
                    {
                        p.stun_ticks = ticks;
                        p.stun_radius = radius;
                    }
                    self.emit_event(SimEvent::MissileLaunched {
                        pos: (pos.0, pos.1, 4 * SCALE),
                        is_nuke,
                    });
                }
                continue;
            }

            // In-place upgrades (Factories, Extractors, Shields, Radars)
            if let Some(UnitOrder::Upgrade(target_type)) = head.as_ref()
                && !role.is_builder()
            {
                let upg_rate = rate.map(|r| r.0).unwrap_or(15).max(15)
                    + assist_bonus.get(&id).copied().unwrap_or(0);
                if self.units[&id].build.is_none() {
                    let job = self.new_job(upg_rate, target_type, pos.0, pos.1, None);
                    self.units.get_mut(&id).unwrap().build = job;
                }
                if let Some(mut job) = self.units.get_mut(&id).unwrap().build.take() {
                    if Self::progress_job_with_rate(&mut self.armies, army, &mut job, upg_rate) {
                        let done_tid = job.unit_type.clone();
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        self.complete_upgrade(id, &done_tid);
                    } else {
                        self.units.get_mut(&id).unwrap().build = Some(job);
                    }
                }
                continue;
            }

            let Some((build_rate, build_range)) = rate else {
                continue;
            };
            let assisted_rate = build_rate * (100 + adj) / 100 + assist_bonus.get(&id).copied().unwrap_or(0);

            let (mobile_factory, mobile_builder) =
                self.catalog.get(&type_id).map_or((false, false), |t| (t.ext.mobile_factory, t.ext.mobile_builder));
            if role.is_factory() || mobile_factory {
                if self.units[&id].build.is_none() && !capped(&cap_counts, army) {
                    let next = self.units.get_mut(&id).unwrap().production.pop_front();
                    if let Some(tid) = next {
                        let job = self.new_job(build_rate, &tid, pos.0, pos.1, None);
                        if job.is_some() && cap > 0 {
                            cap_counts[army as usize] += 1;
                        }
                        self.units.get_mut(&id).unwrap().build = job;
                    }
                }
                let mut job = match self.units.get_mut(&id).unwrap().build.take() {
                    Some(j) => j,
                    None => continue,
                };
                if Self::progress_job_with_rate(&mut self.armies, army, &mut job, assisted_rate) {
                    let exit = (pos.0, (pos.1 + radius + 2 * SCALE).min(self.size.1));
                    // Phase 18 repeat-build: the finished unit goes back to the queue's end.
                    if let Some(f) = self.units.get_mut(&id)
                        && f.repeat_queue
                    {
                        f.production.push_back(job.unit_type.clone());
                    }
                    if let Some(nid) = self.spawn_unit(army, &job.unit_type, exit) {
                        // Phase 29: roll off onto open ground when the exit is built over.
                        let free = self.walkable_near(nid, exit);
                        if let Some(nu) = self.units.get_mut(&nid) {
                            nu.pos = Vec2 { x: free.0, y: free.1 };
                        }
                        self.armies[army as usize].units_built += 1;
                        let rally = self.units[&id].rally;
                        let u = self.units.get_mut(&nid).unwrap();
                        match rally {
                            Some((rx, ry)) => u.orders.push_back(UnitOrder::Move { x: rx, y: ry }),
                            None => u.orders.push_back(UnitOrder::Move {
                                x: exit.0,
                                y: (exit.1 + 4 * SCALE).min(self.size.1),
                            }),
                        }
                    }
                } else {
                    if self.tick % 4 == 0 {
                        self.emit_event(SimEvent::BuildSpark {
                            pos: (pos.0, pos.1, 2 * SCALE),
                            army,
                        });
                    }
                    self.units.get_mut(&id).unwrap().build = Some(job);
                }
            } else if role.is_builder() || role == Role::EngStation || mobile_builder {
                match head {
                    Some(UnitOrder::Build { unit_type, x, y }) => {
                        let target_r = self.catalog.get(&unit_type).map(|t| t.radius).unwrap_or(0);
                        let reach = (build_range + target_r + radius) as i128;
                        if dist_sq(pos, (x, y)) > reach * reach {
                            continue;
                        }
                        if self.units[&id].build.is_none() && capped(&cap_counts, army) {
                            // Phase 18 unit cap: wait in place until a slot frees up.
                            continue;
                        }
                        if self.units[&id].build.is_none() {
                            if cap > 0 {
                                cap_counts[army as usize] += 1;
                            }
                            let target_role = self.catalog.get(&unit_type).map(|t| t.role);
                            let is_mex = target_role == Some(Role::Extractor);
                            let is_hydro = target_role == Some(Role::Hydro);
                            let mp = if is_mex {
                                self.mass_points
                                    .iter()
                                    .position(|&m| m == (x, y))
                                    .filter(|&i| self.mass_claims[i].is_none())
                            } else {
                                None
                            };
                            let hp = if is_hydro {
                                self.hydro_points
                                    .iter()
                                    .position(|&h| h == (x, y))
                                    .filter(|&i| self.hydro_claims.get(i).copied().flatten().is_none())
                            } else {
                                None
                            };
                            let valid = if is_mex {
                                mp.is_some()
                            } else if is_hydro {
                                hp.is_some()
                            } else {
                                self.site_is_free(&unit_type, x, y)
                            };
                            if !valid {
                                self.units.get_mut(&id).unwrap().orders.pop_front();
                                continue;
                            }
                            if let Some(i) = mp {
                                self.mass_claims[i] = Some(id);
                            }
                            if let Some(i) = hp {
                                self.hydro_claims[i] = Some(id);
                            }
                            let job = self.new_job(build_rate, &unit_type, x, y, mp);
                            self.units.get_mut(&id).unwrap().build = job;
                        }
                        let Some(mut job) = self.units.get_mut(&id).unwrap().build.take() else {
                            continue;
                        };
                        if Self::progress_job_with_rate(&mut self.armies, army, &mut job, assisted_rate)
                        {
                            if let Some(sid) = self.spawn_unit(army, &job.unit_type, (job.x, job.y))
                            {
                                self.armies[army as usize].units_built += 1;
                                if let Some(i) = job.mass_point {
                                    self.mass_claims[i] = Some(sid);
                                    self.units.get_mut(&sid).unwrap().mass_point = Some(i);
                                }
                                if let Some(i) = self.hydro_claims.iter().position(|&c| c == Some(id)) {
                                    self.hydro_claims[i] = Some(sid);
                                    self.units.get_mut(&sid).unwrap().hydro_point = Some(i);
                                }
                            }
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                        } else {
                            if self.tick % 4 == 0 {
                                self.emit_event(SimEvent::BuildSpark {
                                    pos: (job.x, job.y, 2 * SCALE),
                                    army,
                                });
                            }
                            self.units.get_mut(&id).unwrap().build = Some(job);
                        }
                    }
                    Some(UnitOrder::Reclaim { x, y }) => {
                        let reach = build_range + 4 * SCALE;
                        match self.reclaim_target((x, y)) {
                            Some((ReclaimKind::Wreck, wid, wpos)) if dist_sq(pos, wpos) <= (reach as i128) * (reach as i128) => {
                                let gain = (build_rate.max(1) * 400).max(SCALE / 5);
                                let w = self.wrecks.get_mut(&wid).unwrap();
                                let harvested = w.mass_left.min(gain);
                                w.mass_left -= harvested;
                                let a = &mut self.armies[army as usize];
                                a.mass = (a.mass + harvested).min(a.mass_cap.max(a.mass));
                                if w.mass_left <= 0 {
                                    self.wrecks.remove(&wid);
                                }
                            }
                            Some((ReclaimKind::Unit, tid, tpos)) if tid != id && dist_sq(pos, tpos) <= (reach as i128) * (reach as i128) => {
                                // Phase 34: a live unit drains back into its mass cost; fully
                                // reclaimed units vanish (no wreck, no kill credit).
                                let gain = (build_rate.max(1) * 400).max(SCALE / 5);
                                let cost = self.units.get(&tid).and_then(|t| self.catalog.get(&t.type_id)).map_or(SCALE, |t| t.cost_mass.max(SCALE));
                                let t = self.units.get_mut(&tid).unwrap();
                                let hp_loss = ((t.max_hp as i128 * gain as i128) / cost as i128).max(1) as i64;
                                let taken = hp_loss.min(t.hp);
                                let value = ((cost as i128 * taken as i128) / t.max_hp.max(1) as i128) as i64;
                                t.hp -= taken;
                                let gone = t.hp <= 0;
                                if gone {
                                    t.alive = false;
                                    t.hp = 0;
                                }
                                let structure = t.role.is_structure();
                                let a = &mut self.armies[army as usize];
                                a.mass = (a.mass + value).min(a.mass_cap.max(a.mass));
                                if gone {
                                    self.units.get_mut(&id).unwrap().orders.pop_front();
                                    if structure {
                                        self.walls_dirty = true;
                                    }
                                }
                            }
                            Some((ReclaimKind::Prop, pid, ppos)) if dist_sq(pos, ppos) <= (reach as i128) * (reach as i128) => {
                                // Phase 34: props drain mass and energy at the wreck rate
                                // (energy ten times faster, matching its lower value).
                                let gain = (build_rate.max(1) * 400).max(SCALE / 5);
                                let pr = self.props.get_mut(&pid).unwrap();
                                let (dm, de) = (pr.mass.min(gain), pr.energy.min(gain * 10));
                                pr.mass -= dm;
                                pr.energy -= de;
                                let done = pr.mass <= 0 && pr.energy <= 0;
                                let src = pr.src_xz;
                                let a = &mut self.armies[army as usize];
                                a.mass = (a.mass + dm).min(a.mass_cap.max(a.mass));
                                a.energy = (a.energy + de).min(a.energy_cap.max(a.energy));
                                if done {
                                    self.props.remove(&pid);
                                    self.props_gone.push(src);
                                }
                            }
                            Some(_) => {}
                            None => {
                                self.units.get_mut(&id).unwrap().orders.pop_front();
                            }
                        }
                    }
                    Some(UnitOrder::Repair(tid)) => {
                        let tinfo = self
                            .unit(tid)
                            .filter(|t| self.are_allies(army, t.army))
                            .map(|t| ((t.pos.x, t.pos.y), t.radius, t.hp, t.max_hp, t.type_id.clone()));
                        let Some((tpos, tr, thp, tmax, ttype)) = tinfo else {
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                            continue;
                        };
                        if thp >= tmax {
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                            continue;
                        }
                        let reach = (build_range + tr + radius) as i128;
                        if dist_sq(pos, tpos) > reach * reach {
                            continue;
                        }
                        let (cost_m, cost_e, work_total) = self
                            .catalog
                            .get(&ttype)
                            .map(|t| (t.cost_mass, t.cost_energy, t.build_work.max(1)))
                            .unwrap_or((100, 100, 100));
                        let need_hp = tmax - thp;
                        let step_work = build_rate.max(1);
                        let mut dummy = ConstructionProject::new(0, cost_m, cost_e, work_total, step_work);
                        let mut stock = (self.armies[army as usize].mass, self.armies[army as usize].energy);
                        let _ = dummy.step(&mut stock);
                        self.armies[army as usize].mass = stock.0;
                        self.armies[army as usize].energy = stock.1;
                        let heal = ((tmax as i128 * dummy.progress as i128) / work_total as i128).max(1) as i64;
                        let actual_heal = heal.min(need_hp);
                        if let Some(t) = self.units.get_mut(&tid) {
                            t.hp = (t.hp + actual_heal).min(t.max_hp);
                            if t.hp >= t.max_hp {
                                self.units.get_mut(&id).unwrap().orders.pop_front();
                            }
                        }
                    }
                    Some(UnitOrder::Assist(tid)) => {
                        if !self.unit(tid).is_some_and(|t| self.are_allies(army, t.army)) {
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                        }
                    }
                    Some(UnitOrder::Enhance(name)) => {
                        // Phase 18: in-place ACU/SCU enhancement, paid like construction.
                        if self.units[&id].build.is_none() {
                            let spec = self
                                .catalog
                                .get(&type_id)
                                .and_then(|t| t.ext.enhancement(&name))
                                .filter(|s| Self::enhancement_allowed(&self.units[&id].enh, s))
                                .map(|s| (s.cost_mass, s.cost_energy, s.build_work));
                            let Some((cm, ce, work)) = spec else {
                                self.units.get_mut(&id).unwrap().orders.pop_front();
                                continue;
                            };
                            let project = ConstructionProject::new(self.next_project, cm, ce, work, build_rate.max(1));
                            self.next_project += 1;
                            self.units.get_mut(&id).unwrap().build = Some(BuildJob {
                                unit_type: name.clone(),
                                x: pos.0,
                                y: pos.1,
                                project,
                                mass_point: None,
                            });
                        }
                        let Some(mut job) = self.units.get_mut(&id).unwrap().build.take() else {
                            continue;
                        };
                        if Self::progress_job_with_rate(&mut self.armies, army, &mut job, assisted_rate) {
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                            self.complete_enhancement(id, &name);
                        } else {
                            if self.tick % 4 == 0 {
                                self.emit_event(SimEvent::BuildSpark {
                                    pos: (pos.0, pos.1, 2 * SCALE),
                                    army,
                                });
                            }
                            self.units.get_mut(&id).unwrap().build = Some(job);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// Integer step toward `goal`, capped at `speed`, never overshooting.
    fn approach(pos: &mut Vec2, goal: (i64, i64), speed: i64) {
        let dx = goal.0 - pos.x;
        let dy = goal.1 - pos.y;
        let d = isqrt((dx as i128) * (dx as i128) + (dy as i128) * (dy as i128));
        if d <= speed {
            pos.x = goal.0;
            pos.y = goal.1;
            return;
        }
        pos.x += ((dx as i128) * (speed as i128) / (d as i128)) as i64;
        pos.y += ((dy as i128) * (speed as i128) / (d as i128)) as i64;
    }

    fn advance_along_path(&mut self, id: UnitId, goal: (i64, i64)) -> bool {
        let u = self.units.get_mut(&id).unwrap();
        u.moving = true;
        if (u.pos.x, u.pos.y) == goal {
            u.path.clear();
            u.path_goal = None;
            u.moving = false;
            return true;
        }
        let speed = u.speed;
        if speed <= 0 {
            return false;
        }
        if let Some(back) = u.path.back_mut() {
            *back = goal;
        } else {
            u.path.push_back(goal);
        }
        let mut budget = speed;
        while budget > 0 {
            let Some(&wp) = u.path.front() else {
                break;
            };
            let before = u.pos;
            Self::approach(&mut u.pos, wp, budget);
            let moved = isqrt(dist_sq((before.x, before.y), (u.pos.x, u.pos.y)));
            budget -= moved.max(1);
            if (u.pos.x, u.pos.y) == wp {
                u.path.pop_front();
            } else {
                break;
            }
        }
        if (u.pos.x, u.pos.y) == goal {
            u.path.clear();
            u.path_goal = None;
            true
        } else {
            false
        }
    }

    /// Nearest live enemy of `army` to `p` within `radius`, filtered by air/surface weapon capability.
    fn nearest_enemy_filtered(
        &self,
        army: ArmyId,
        p: (i64, i64),
        radius: i64,
        can_aa: bool,
        can_surface: bool,
    ) -> Option<UnitId> {
        let team = self.armies.get(army as usize).map(|a| a.team).unwrap_or(army);
        Self::nearest_enemy_in(&self.spatial, &self.spatial_snap, team, army, p, radius, can_aa, can_surface)
    }

    /// Pure spatial-snapshot form of [`Self::nearest_enemy_filtered`] (usable from workers).
    #[allow(clippy::too_many_arguments)]
    fn nearest_enemy_in(
        spatial: &SpatialGrid,
        snap: &[UnitSpatialSnap],
        team: u8,
        army: ArmyId,
        p: (i64, i64),
        radius: i64,
        can_aa: bool,
        can_surface: bool,
    ) -> Option<UnitId> {
        let lim = (radius as i128) * (radius as i128);
        let mut best_mobile: Option<(i128, UnitId)> = None;
        let mut best_struct: Option<(i128, UnitId)> = None;
        let mut best_wall: Option<(i128, UnitId)> = None;
        spatial
            .for_each_enemy_in_radius(p.0, p.1, radius + 6 * SCALE, team, |idx| {
                let u = &snap[idx as usize];
                if u.team == team || u.hidden_from(army) {
                    return;
                }
                let is_air = u.domain == NavDomain::Air;
                if (is_air && !can_aa) || (!is_air && !can_surface) {
                    return;
                }
                let d = dist_sq(p, (u.x, u.y));
                if d > lim {
                    return;
                }
                let slot = if u.is_wall {
                    &mut best_wall
                } else if u.is_struct {
                    &mut best_struct
                } else {
                    &mut best_mobile
                };
                if slot.is_none_or(|(bd, bid)| d < bd || (d == bd && u.id < bid)) {
                    *slot = Some((d, u.id));
                }
            });
        best_mobile.or(best_struct).or(best_wall).map(|(_, id)| id)
    }

    /// Attack-move/patrol acquisition for every eligible mover, solved up front on the pool.
    /// Pass 1 of [`Self::step_movement`] only pops orders (positions and the spatial snapshot
    /// are unchanged), so these answers equal the lazily computed ones.
    fn prefetch_attack_move_targets(&self) -> BTreeMap<UnitId, Option<UnitId>> {
        /// (unit, team, army, position, acquire radius, can_aa, can_surface)
        type AcquireReq = (UnitId, u8, ArmyId, (i64, i64), i64, bool, bool);
        let _p = crate::profiler::scope("move.acquire");
        let reqs: Vec<AcquireReq> = self
            .units
            .values()
            .filter(|u| u.alive && u.speed > 0 && u.carried_by.is_none() && u.max_range > 0)
            .filter(|u| {
                u.orders
                    .iter()
                    .take(4)
                    .any(|o| matches!(o, UnitOrder::AttackMove { .. } | UnitOrder::Patrol { .. }))
            })
            .map(|u| {
                let team = self.armies.get(u.army as usize).map(|a| a.team).unwrap_or(u.army);
                (u.id, team, u.army, (u.pos.x, u.pos.y), u.max_range + ACQUIRE_BONUS, u.can_fire_aa, u.can_fire_surface)
            })
            .collect();
        let solve = |sp: &SpatialGrid, sn: &[UnitSpatialSnap], r: &AcquireReq| {
            Self::nearest_enemy_in(sp, sn, r.1, r.2, r.3, r.4, r.5, r.6)
        };
        let results: Vec<Option<UnitId>> = match self.pool.as_ref() {
            Some(pool) if pool.threads() > 1 && reqs.len() >= 128 => {
                let sp = Arc::clone(&self.spatial);
                let sn = Arc::clone(&self.spatial_snap);
                let items = Arc::new(reqs.clone());
                pool.par_map_ordered_arc(items, Arc::new(move |_: usize, r: &AcquireReq| solve(&sp, &sn, r)))
            }
            _ => reqs.iter().map(|r| solve(&self.spatial, &self.spatial_snap, r)).collect(),
        };
        reqs.iter().map(|r| r.0).zip(results).collect()
    }

    fn nearest_enemy(&self, army: ArmyId, p: (i64, i64), radius: i64) -> Option<UnitId> {
        self.nearest_enemy_filtered(army, p, radius, true, true)
    }

    fn in_weapon_range(&self, u: &Unit, target: UnitId) -> bool {
        self.unit(target).is_some_and(|t| {
            if !t.targetable_by(u.army) {
                return false;
            }
            let reach = (u.max_range + t.radius) as i128;
            dist_sq((u.pos.x, u.pos.y), (t.pos.x, t.pos.y)) <= reach * reach
        })
    }

    /// Determines a unit's desired movement `(goal, tolerance)` if it needs to move this tick.
    fn pop_completed_and_goal(
        &mut self,
        id: UnitId,
        pre_acquired: Option<&BTreeMap<UnitId, Option<UnitId>>>,
    ) -> Option<((i64, i64), i64)> {
        for _ in 0..4 {
            let Some(u) = self.units.get(&id) else {
                return None;
            };
            if !u.alive || u.speed <= 0 || u.carried_by.is_some() || u.stunned > 0 {
                return None;
            }
            let Some(head) = u.orders.front().cloned() else {
                return None;
            };
            let army = u.army;
            let pos = (u.pos.x, u.pos.y);
            let can_aa = u.can_fire_aa;
            let can_surf = u.can_fire_surface;
            match head {
                UnitOrder::Move { x, y } => {
                    if pos == (x, y) {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    }
                    return Some(((x, y), SCALE));
                }
                UnitOrder::AttackMove { x, y } | UnitOrder::Patrol { x, y } => {
                    let acquire = u.max_range + ACQUIRE_BONUS;
                    let holding = u.fire_state != FireState::FireAtWill;
                    let found = match pre_acquired.and_then(|m| m.get(&id)) {
                        _ if holding => None,
                        Some(&hit) => hit,
                        None if u.max_range > 0 => self.nearest_enemy_filtered(army, pos, acquire, can_aa, can_surf),
                        None => None,
                    };
                    if u.max_range > 0
                        && let Some(e) = found
                    {
                        if self.in_weapon_range(&self.units[&id], e) && u.domain != NavDomain::Air {
                            return None;
                        }
                        let ep = self.unit(e).map(|t| (t.pos.x, t.pos.y)).unwrap();
                        return Some((ep, 4 * SCALE));
                    }
                    if pos == (x, y) {
                        if matches!(head, UnitOrder::Patrol { .. }) {
                            // Reverse patrol toward second queued patrol or start position
                            let u_mut = self.units.get_mut(&id).unwrap();
                            if u_mut.orders.len() > 1 {
                                let front = u_mut.orders.pop_front().unwrap();
                                u_mut.orders.push_back(front);
                                continue;
                            }
                            let home = self.armies[army as usize].start;
                            u_mut.orders.push_back(UnitOrder::Patrol { x: home.0, y: home.1 });
                            let front = u_mut.orders.pop_front().unwrap();
                            u_mut.orders.push_back(front);
                            continue;
                        }
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    }
                    return Some(((x, y), SCALE));
                }
                UnitOrder::Attack(t) | UnitOrder::Overcharge(t) => {
                    let Some(tp) = self.unit(t).filter(|x| x.targetable_by(army)).map(|t| (t.pos.x, t.pos.y)) else {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    };
                    let in_r = if matches!(head, UnitOrder::Overcharge(_)) {
                        let reach = (OVERCHARGE_RANGE + self.unit(t).unwrap().radius) as i128;
                        dist_sq(pos, tp) <= reach * reach
                    } else {
                        self.in_weapon_range(&self.units[&id], t)
                    };
                    if in_r && self.units[&id].domain != NavDomain::Air {
                        return None;
                    }
                    return Some((tp, 4 * SCALE));
                }
                UnitOrder::Build { unit_type, x, y } => {
                    let target_r = self.catalog.get(&unit_type).map(|t| t.radius).unwrap_or(0);
                    let range = self
                        .catalog
                        .get(&u.type_id)
                        .map(|t| t.build_range)
                        .unwrap_or(0);
                    let reach = (range + target_r + u.radius) as i128;
                    if dist_sq(pos, (x, y)) <= reach * reach {
                        return None;
                    }
                    return Some(((x, y), SCALE));
                }
                UnitOrder::Reclaim { x, y } => {
                    let range = self
                        .catalog
                        .get(&u.type_id)
                        .map(|t| t.build_range)
                        .unwrap_or(0);
                    let reach = range + 4 * SCALE;
                    let target = self.reclaim_target((x, y)).map(|(_, _, p)| p).unwrap_or((x, y));
                    if dist_sq(pos, target) <= (reach as i128) * (reach as i128) {
                        return None;
                    }
                    return Some((target, 2 * SCALE));
                }
                UnitOrder::Repair(tid) | UnitOrder::Assist(tid) => {
                    let Some(t) = self.unit(tid) else {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    };
                    let range = self
                        .catalog
                        .get(&u.type_id)
                        .map(|ut| ut.build_range)
                        .unwrap_or(0);
                    let reach = (range + t.radius + u.radius) as i128;
                    let tp = (t.pos.x, t.pos.y);
                    if dist_sq(pos, tp) <= reach * reach {
                        return None;
                    }
                    return Some((tp, 3 * SCALE));
                }
                UnitOrder::Upgrade(_) | UnitOrder::LaunchMissile { .. } | UnitOrder::Enhance(_) => return None,
                UnitOrder::Board(tid) => {
                    if !self.can_board(id, tid) {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    }
                    let t = &self.units[&tid];
                    let reach = (t.radius + u.radius + TRANSPORT_PICKUP_REACH) as i128;
                    let tp = (t.pos.x, t.pos.y);
                    if dist_sq(pos, tp) <= reach * reach {
                        return None;
                    }
                    return Some((tp, 2 * SCALE));
                }
                UnitOrder::Unload { x, y } => {
                    if pos == (x, y) {
                        return None;
                    }
                    return Some(((x, y), SCALE));
                }
                UnitOrder::Ferry { from, to } => {
                    let goal = if u.cargo.is_empty() { from } else { to };
                    if pos == goal {
                        return None;
                    }
                    return Some((goal, SCALE));
                }
                UnitOrder::AttackGround { x, y } => {
                    let reach = (u.max_range - SCALE).max(SCALE) as i128;
                    if dist_sq(pos, (x, y)) <= reach * reach {
                        return None;
                    }
                    return Some(((x, y), SCALE));
                }
                UnitOrder::Guard(gid) => {
                    let Some(g) = self.unit(gid).filter(|g| g.alive && self.are_allies(army, g.army)) else {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    };
                    let gp = (g.pos.x, g.pos.y);
                    // Engage threats near the guarded unit (unless holding fire), else follow.
                    if u.max_range > 0 && u.fire_state == FireState::FireAtWill {
                        let threat = self.nearest_enemy_filtered(army, gp, u.max_range + ACQUIRE_BONUS, can_aa, can_surf);
                        if let Some(e) = threat {
                            if self.in_weapon_range(&self.units[&id], e) && u.domain != NavDomain::Air {
                                return None;
                            }
                            let ep = self.unit(e).map(|t| (t.pos.x, t.pos.y)).unwrap();
                            return Some((ep, SCALE));
                        }
                    }
                    let near = (g.radius + u.radius + 5 * SCALE) as i128;
                    if dist_sq(pos, gp) <= near * near {
                        return None;
                    }
                    return Some((gp, 3 * SCALE));
                }
                UnitOrder::Dock(sid) => {
                    if !self.is_staging_of(sid, army) {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    }
                    let s = &self.units[&sid];
                    let sp = (s.pos.x, s.pos.y);
                    if pos == sp {
                        return None;
                    }
                    return Some((sp, SCALE));
                }
            }
        }
        None
    }

    /// Phase 30/32: whether `sid` is a live air staging platform (structure, or a mobile
    /// one such as the Fatboy, Atlantis or CZAR) owned by `army`.
    fn is_staging_of(&self, sid: UnitId, army: ArmyId) -> bool {
        self.unit(sid).is_some_and(|s| {
            s.alive && s.army == army && s.carried_by.is_none() && self.catalog.get(&s.type_id).is_some_and(|t| t.ext.staging.is_some())
        })
    }

    /// Phase 32: unit `u` can queue unit production (factory structures and mobile factories).
    pub fn is_producer(&self, u: &Unit) -> bool {
        u.role.is_factory() || self.catalog.get(&u.type_id).is_some_and(|t| t.ext.mobile_factory)
    }

    /// Phase 32: unit `u` can place structures (engineers, commanders, mobile builders).
    pub fn is_constructor(&self, u: &Unit) -> bool {
        u.role.is_builder() || self.catalog.get(&u.type_id).is_some_and(|t| t.ext.mobile_builder)
    }

    /// Phase 30: docking slots of staging platform `sid` taken by docked aircraft.
    fn docked_at(&self, sid: UnitId) -> Vec<UnitId> {
        self.units
            .values()
            .filter(|u| u.alive && u.carried_by == Some(sid) && matches!(u.orders.front(), Some(UnitOrder::Dock(s)) if *s == sid))
            .map(|u| u.id)
            .collect()
    }

    /// Phase 30 air fuel & staging (ascending unit id throughout):
    /// 1. aircraft whose staging platform died are released;
    /// 2. airborne aircraft burn one tick of fuel and fly at half speed when empty;
    /// 3. every second, low-fuel aircraft with auto-refuel queue a `Dock` at the nearest own
    ///    platform with a free slot within its scan radius;
    /// 4. aircraft that reached their platform dock (if a slot is free);
    /// 5. docked aircraft refuel (`FuelRechargeRate × RefuelingMultiplier`) and are repaired
    ///    (`RepairRate`, paid per second in energy/mass), then undock and resume orders.
    fn step_staging(&mut self) {
        if !self.units.values().any(|u| u.domain == NavDomain::Air) {
            return;
        }
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        // 1. Release aircraft from dead platforms.
        for &id in &ids {
            let Some(sid) = self.units[&id].carried_by else { continue };
            let staged = matches!(self.units[&id].orders.front(), Some(UnitOrder::Dock(s)) if *s == sid);
            if staged && !self.units.get(&sid).is_some_and(|s| s.alive) {
                let u = self.units.get_mut(&id).unwrap();
                u.carried_by = None;
                u.orders.pop_front();
            }
        }
        // 2. Fuel burn.
        for &id in &ids {
            if self.units[&id].domain != NavDomain::Air {
                continue;
            }
            let (fuel_ticks, base_speed) = match self.units.get(&id).and_then(|u| self.catalog.get(&u.type_id)) {
                Some(t) => (t.ext.fuel_ticks, t.speed_per_tick),
                None => continue,
            };
            let u = self.units.get_mut(&id).unwrap();
            if fuel_ticks == 0 || !u.alive || u.carried_by.is_some() || u.domain != NavDomain::Air || u.drone_of.is_some() {
                continue;
            }
            u.fuel = u.fuel.saturating_sub(1);
            u.speed = if u.fuel == 0 { base_speed / 2 } else { base_speed };
        }
        // 3. Auto-refuel.
        if self.tick.is_multiple_of(TICK_HZ) {
            let platforms: Vec<(UnitId, ArmyId, (i64, i64), StagingSpec)> = self
                .units
                .values()
                .filter(|s| s.alive && s.carried_by.is_none())
                .filter_map(|s| self.catalog.get(&s.type_id).and_then(|t| t.ext.staging).map(|sp| (s.id, s.army, (s.pos.x, s.pos.y), sp)))
                .collect();
            if !platforms.is_empty() {
                let mut load: BTreeMap<UnitId, u32> = BTreeMap::new();
                for u in self.units.values().filter(|u| u.alive) {
                    if let Some(UnitOrder::Dock(s)) = u.orders.front() {
                        *load.entry(*s).or_insert(0) += 1;
                    }
                }
                for &id in &ids {
                    let Some(u) = self.units.get(&id).filter(|u| u.domain == NavDomain::Air) else { continue };
                    let fuel_ticks = self.catalog.get(&u.type_id).map_or(0, |t| t.ext.fuel_ticks);
                    if !u.alive || !u.auto_refuel || fuel_ticks == 0 || u.carried_by.is_some()
                        || matches!(u.orders.front(), Some(UnitOrder::Dock(_)))
                        || u.fuel as u64 * 5 > fuel_ticks as u64
                    {
                        continue;
                    }
                    let p = (u.pos.x, u.pos.y);
                    let best = platforms
                        .iter()
                        .filter(|(sid, a, sp_pos, sp)| {
                            *a == u.army
                                && load.get(sid).copied().unwrap_or(0) < sp.slots as u32
                                && dist_sq(p, *sp_pos) <= (sp.scan_radius as i128) * (sp.scan_radius as i128)
                        })
                        .min_by_key(|(sid, _, sp_pos, _)| (dist_sq(p, *sp_pos), *sid))
                        .map(|(sid, ..)| *sid);
                    if let Some(sid) = best {
                        *load.entry(sid).or_insert(0) += 1;
                        let u = self.units.get_mut(&id).unwrap();
                        u.orders.push_front(UnitOrder::Dock(sid));
                        u.path.clear();
                        u.path_goal = None;
                    }
                }
            }
        }
        // 4. Docking on arrival; 5. servicing.
        for &id in &ids {
            let Some(u) = self.units.get(&id) else { continue };
            let Some(&UnitOrder::Dock(sid)) = u.orders.front() else { continue };
            if !u.alive {
                continue;
            }
            let Some((spos, sradius, spec)) = self
                .units
                .get(&sid)
                .and_then(|s| self.catalog.get(&s.type_id).and_then(|t| t.ext.staging).map(|sp| ((s.pos.x, s.pos.y), s.radius, sp)))
            else {
                continue;
            };
            let (army, docked) = (u.army, u.carried_by == Some(sid));
            if !docked {
                let reach = (sradius + 3 * SCALE) as i128;
                if dist_sq((u.pos.x, u.pos.y), spos) > reach * reach || self.docked_at(sid).len() >= spec.slots as usize {
                    continue;
                }
                let u = self.units.get_mut(&id).unwrap();
                u.carried_by = Some(sid);
                u.pos = Vec2 { x: spos.0, y: spos.1 };
                u.path.clear();
                u.path_goal = None;
                u.moving = false;
                continue;
            }
            self.units.get_mut(&id).unwrap().pos = Vec2 { x: spos.0, y: spos.1 };
            let (fuel_ticks, recharge, base_speed) = self
                .catalog
                .get(&self.units[&id].type_id)
                .map_or((0, 0, 0), |t| (t.ext.fuel_ticks, t.ext.fuel_recharge_milli, t.speed_per_tick));
            // `FuelRechargeRate` fuel-seconds per second = fuel ticks per tick.
            let refuel = ((recharge * spec.refuel_multiplier) / 1000).clamp(1, u32::MAX as i64 / 4) as u32;
            let (hp, max_hp) = (self.units[&id].hp, self.units[&id].max_hp);
            let mut heal = 0;
            if hp < max_hp {
                let e = spec.repair_energy / TICK_HZ as i64;
                let m = spec.repair_mass / TICK_HZ as i64;
                let a = &mut self.armies[army as usize];
                if a.energy >= e && a.mass >= m {
                    a.energy -= e;
                    a.mass -= m;
                    heal = (max_hp * spec.repair_permille / 1000 / TICK_HZ as i64).max(1);
                }
            }
            let u = self.units.get_mut(&id).unwrap();
            u.fuel = u.fuel.saturating_add(refuel).min(fuel_ticks);
            u.hp = (u.hp + heal).min(u.max_hp);
            u.speed = base_speed;
            if u.fuel >= fuel_ticks && u.hp >= u.max_hp {
                u.carried_by = None;
                u.orders.pop_front();
                u.pos = Vec2 { x: spos.0, y: (spos.1 + sradius + 2 * SCALE).min(self.size.1) };
            }
        }
    }

    fn step_movement(&mut self) {
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        for u in self.units.values_mut() {
            u.moving = false;
        }

        // Pass 1: Collect movement targets grouped by (domain, goal_cell) + small-group A* requests.
        let mut active_goals: Vec<(UnitId, NavDomain, (i64, i64))> = Vec::with_capacity(ids.len());
        let mut group_counts: BTreeMap<(NavDomain, u32, u32), usize> = BTreeMap::new();

        let pre_acquired = self.prefetch_attack_move_targets();
        let _pg = crate::profiler::scope("move.goals");
        for id in ids {
            let Some((mut goal, _tol)) = self.pop_completed_and_goal(id, Some(&pre_acquired)) else {
                continue;
            };
            let domain = self.units[&id].domain;
            // Phase 28: a goal inside a wall (e.g. attacking a wall section) is reached from
            // the nearest open cell instead of walking straight through the wall line.
            if domain != NavDomain::Air && self.is_wall_cell(goal) {
                goal = self.walkable_near(id, goal);
            }
            if domain != NavDomain::Air {
                let gcell = self.domain_grid(domain).world_to_cell(goal.0, goal.1);
                *group_counts.entry((domain, gcell.0, gcell.1)).or_insert(0) += 1;
            }
            active_goals.push((id, domain, goal));
        }

        // Pass 2: Flow-field routing for large groups (>= 6 units sharing goal cell),
        // parallel WorkerPool A* per domain for small groups (< 6), direct flight for Air.
        drop(_pg);
        let _pf = crate::profiler::scope("move.route");
        let mut astar_by_domain: [Vec<(UnitId, PathRequest, (i64, i64))>; 3] =
            [Vec::new(), Vec::new(), Vec::new()];
        let cur_tick = self.tick;

        // Flow fields depend only on (domain grid, goal cell): compute this tick's missing
        // ones up front (in parallel when a pool is attached); the loop below then only hits.
        let wanted: Vec<flowfield::FlowRequest> = group_counts
            .iter()
            .filter(|&(_, &n)| n >= 6)
            .map(|(&(domain, gx, gy), _)| {
                let g = match domain {
                    NavDomain::Land => Arc::clone(&self.grid),
                    NavDomain::Naval => Arc::clone(&self.naval_grid),
                    NavDomain::Amphibious | NavDomain::Air => Arc::clone(&self.amphib_grid),
                };
                (domain, g, (gx, gy))
            })
            .collect();
        if !wanted.is_empty() {
            self.flow_cache.prefetch(&wanted, cur_tick, self.pool.as_deref());
        }

        for &(id, domain, goal) in &active_goals {
            if domain == NavDomain::Air {
                continue;
            }
            let dgrid = self.domain_grid(domain);
            let gcell = dgrid.world_to_cell(goal.0, goal.1);
            let count = group_counts.get(&(domain, gcell.0, gcell.1)).copied().unwrap_or(0);
            if count >= 6 {
                let grid_clone = match domain {
                    NavDomain::Land => Arc::clone(&self.grid),
                    NavDomain::Naval => Arc::clone(&self.naval_grid),
                    NavDomain::Amphibious | NavDomain::Air => Arc::clone(&self.amphib_grid),
                };
                let ff = self.flow_cache.get_or_compute(domain, &grid_clone, goal, cur_tick);
                let pos = (self.units[&id].pos.x, self.units[&id].pos.y);
                let wp = ff.next_waypoint(pos, goal);
                if let Some(u) = self.units.get_mut(&id) {
                    u.path.clear();
                    u.path.push_back(wp);
                    if wp != goal {
                        u.path.push_back(goal);
                    }
                    u.path_goal = Some(goal);
                }
            } else {
                let u = &self.units[&id];
                let stale = u.path_goal.is_none_or(|g| dist_sq(g, goal) > (4 * SCALE as i128) * (4 * SCALE as i128));
                if stale {
                    let dom_idx = match domain {
                        NavDomain::Land => 0,
                        NavDomain::Naval => 1,
                        NavDomain::Amphibious | NavDomain::Air => 2,
                    };
                    astar_by_domain[dom_idx].push((
                        id,
                        PathRequest {
                            id: id as u64,
                            start: (u.pos.x, u.pos.y),
                            goal,
                        },
                        goal,
                    ));
                }
            }
        }

        for (dom_idx, batch) in astar_by_domain.into_iter().enumerate() {
            if batch.is_empty() {
                continue;
            }
            let dgrid = match dom_idx {
                0 => &self.grid,
                1 => &self.naval_grid,
                _ => &self.amphib_grid,
            };
            let reqs: Vec<PathRequest> = batch.iter().map(|(_, r, _)| r.clone()).collect();
            let results = if reqs.len() >= 4
                && let Some(ref pool) = self.pool
                && pool.threads() > 1
            {
                pathfinding::solve_paths_batch_pool(dgrid, Arc::new(reqs), pool)
            } else {
                reqs.iter()
                    .map(|req| pathfinding::find_path(dgrid, req))
                    .collect()
            };
            self.paths_solved += results.len() as u64;
            for ((uid, _, goal), res) in batch.into_iter().zip(results) {
                let mut path: VecDeque<(i64, i64)> = if res.found {
                    res.waypoints.into_iter().collect()
                } else {
                    VecDeque::new()
                };
                path.pop_back();
                path.push_back(goal);
                if let Some(u) = self.units.get_mut(&uid) {
                    u.path = path;
                    u.path_goal = Some(goal);
                }
            }
        }

        // Pass 3: Advance every moving unit cleanly along its path in ascending UnitId order.
        drop(_pf);
        let _pa = crate::profiler::scope("move.advance");
        for (id, domain, goal) in active_goals {
            if domain == NavDomain::Air {
                let u = self.units.get_mut(&id).unwrap();
                u.moving = true;
                Self::approach(&mut u.pos, goal, u.speed);
                if (u.pos.x, u.pos.y) == goal
                    && matches!(
                        u.orders.front(),
                        Some(UnitOrder::Move { .. } | UnitOrder::AttackMove { .. })
                    )
                {
                    u.orders.pop_front();
                }
                continue;
            }
            if self.advance_along_path(id, goal)
                && let Some(u) = self.units.get_mut(&id)
                && matches!(
                    u.orders.front(),
                    Some(UnitOrder::Move { .. } | UnitOrder::AttackMove { .. })
                )
            {
                u.orders.pop_front();
            }
        }
    }

    /// Deterministic parallel pairwise repulsion for crowded mobile units via `SpatialGrid`.
    fn step_separation(&mut self) {
        let mobile_indices: Vec<usize> = self
            .spatial_snap
            .iter()
            .enumerate()
            .filter(|(_, u)| u.speed > 0 && !u.is_struct)
            .map(|(i, _)| i)
            .collect();
        if mobile_indices.is_empty() {
            return;
        }

        let compute_delta = |idx: usize, snap: &[UnitSpatialSnap], grid: &SpatialGrid| -> (UnitId, i64, i64) {
            let u = &snap[idx];
            let mut push_x: i64 = 0;
            let mut push_y: i64 = 0;
            let u_air = u.domain == NavDomain::Air;
            grid.for_each_in_radius(u.x, u.y, u.radius + 3 * SCALE, |oidx| {
                let o = &snap[oidx as usize];
                if o.id == u.id || o.is_struct || ((o.domain == NavDomain::Air) != u_air) {
                    return;
                }
                let min_sep = u.radius + o.radius + SCALE / 2;
                let dx = u.x - o.x;
                let dy = u.y - o.y;
                let d2 = (dx as i128) * (dx as i128) + (dy as i128) * (dy as i128);
                if d2 < (min_sep as i128) * (min_sep as i128) {
                    if d2 == 0 {
                        let sign = if u.id < o.id { -1 } else { 1 };
                        push_x += sign * (SCALE / 4);
                    } else {
                        let d = isqrt(d2).max(1);
                        let push = (min_sep - d) / 4;
                        push_x += ((dx as i128) * (push as i128) / (d as i128)) as i64;
                        push_y += ((dy as i128) * (push as i128) / (d as i128)) as i64;
                    }
                }
            });
            if push_x != 0 || push_y != 0 {
                let cap = (u.speed / 2).max(SCALE / 10);
                (u.id, push_x.clamp(-cap, cap), push_y.clamp(-cap, cap))
            } else {
                (u.id, 0, 0)
            }
        };

        let deltas: Vec<(UnitId, i64, i64)> = if mobile_indices.len() >= 128
            && let Some(ref pool) = self.pool
            && pool.threads() > 1
        {
            let snap_arc = Arc::clone(&self.spatial_snap);
            let grid_arc = Arc::clone(&self.spatial);
            pool.par_map_ordered_arc(
                Arc::new(mobile_indices),
                Arc::new(move |_: usize, &idx: &usize| compute_delta(idx, &snap_arc, &grid_arc)),
            )
        } else {
            let snap = &self.spatial_snap;
            let grid = &self.spatial;
            mobile_indices
                .iter()
                .map(|&idx| compute_delta(idx, snap, grid))
                .collect()
        };

        let (max_x, max_y) = self.size;
        for (id, dx, dy) in deltas {
            if dx == 0 && dy == 0 {
                continue;
            }
            let Some(u) = self.units.get(&id) else { continue };
            let np = ((u.pos.x + dx).clamp(0, max_x), (u.pos.y + dy).clamp(0, max_y));
            // Phase 28: crowd pushes never shove ground units into a wall cell.
            if u.domain != NavDomain::Air && self.is_wall_cell(np) && !self.is_wall_cell((u.pos.x, u.pos.y)) {
                continue;
            }
            let u = self.units.get_mut(&id).unwrap();
            u.pos.x = np.0;
            u.pos.y = np.1;
        }
    }

    /// Phase 18 air transports: boarding (ascending passenger id, capacity-checked),
    /// carrying (cargo rides along, outside the spatial grid) and unloading at the
    /// `Unload` destination onto the nearest walkable cells. Inert without transports.
    fn step_transport(&mut self) {
        // 1. Boarding.
        let boarders: Vec<(UnitId, UnitId)> = self
            .units
            .values()
            .filter(|u| u.alive && u.carried_by.is_none())
            .filter_map(|u| match u.orders.front() {
                Some(UnitOrder::Board(t)) => Some((u.id, *t)),
                _ => None,
            })
            .collect();
        for (pid, tid) in boarders {
            if !self.can_board(pid, tid) {
                if let Some(u) = self.units.get_mut(&pid) {
                    u.orders.pop_front();
                }
                continue;
            }
            let (pp, pr) = {
                let p = &self.units[&pid];
                ((p.pos.x, p.pos.y), p.radius)
            };
            let (tp, tr) = {
                let t = &self.units[&tid];
                ((t.pos.x, t.pos.y), t.radius)
            };
            let reach = (tr + pr + TRANSPORT_PICKUP_REACH) as i128;
            if dist_sq(pp, tp) > reach * reach {
                continue;
            }
            self.cancel_build(pid);
            if let Some(p) = self.units.get_mut(&pid) {
                p.carried_by = Some(tid);
                p.orders.clear();
                p.path.clear();
                p.path_goal = None;
                p.moving = false;
            }
            if let Some(t) = self.units.get_mut(&tid) {
                t.cargo.push(pid);
            }
        }

        // 2. Carry + unload, ascending transport id.
        let transports: Vec<UnitId> = self
            .units
            .values()
            .filter(|u| u.alive && (!u.cargo.is_empty() || matches!(u.orders.front(), Some(UnitOrder::Unload { .. }))))
            .map(|u| u.id)
            .collect();
        for tid in transports {
            let alive_cargo: Vec<UnitId> = self.units[&tid]
                .cargo
                .iter()
                .copied()
                .filter(|c| self.units.get(c).is_some_and(|u| u.alive))
                .collect();
            let (tpos, talt, head) = {
                let t = self.units.get_mut(&tid).unwrap();
                t.cargo = alive_cargo.clone();
                ((t.pos.x, t.pos.y), t.altitude, t.orders.front().cloned())
            };
            for &c in &alive_cargo {
                let u = self.units.get_mut(&c).unwrap();
                u.pos = Vec2 { x: tpos.0, y: tpos.1 };
                u.altitude = (talt - SCALE).max(0);
            }
            if let Some(UnitOrder::Unload { x, y }) = head
                && tpos == (x, y)
            {
                let n = alive_cargo.len();
                for (i, &c) in alive_cargo.iter().enumerate() {
                    let (sx, sy) = Self::formation_slot(i, n, x, y);
                    let (sx, sy) = self.walkable_near(c, self.clamp_to_map(sx, sy));
                    let u = self.units.get_mut(&c).unwrap();
                    u.carried_by = None;
                    u.pos = Vec2 { x: sx, y: sy };
                    u.altitude = if u.domain == NavDomain::Air { u.altitude } else { 0 };
                    u.orders.clear();
                    u.path.clear();
                    u.path_goal = None;
                }
                let t = self.units.get_mut(&tid).unwrap();
                t.cargo.clear();
                t.orders.pop_front();
                if n > 0 {
                    self.emit_event(SimEvent::BuildSpark {
                        pos: (x, y, SCALE),
                        army: self.units[&tid].army,
                    });
                }
            }
        }
    }

    /// Nearest unblocked cell centre (spiral search, 4 rings) for dropping unit `id` at `p`.
    fn walkable_near(&self, id: UnitId, p: (i64, i64)) -> (i64, i64) {
        let domain = self.units.get(&id).map(|u| u.domain).unwrap_or(NavDomain::Land);
        if domain == NavDomain::Air {
            return p;
        }
        let grid = self.domain_grid(domain);
        let (cx, cy) = grid.world_to_cell(p.0, p.1);
        if !grid.is_blocked(cx, cy) {
            return p;
        }
        for ring in 1..=4i64 {
            for dy in -ring..=ring {
                for dx in -ring..=ring {
                    if dx.abs() != ring && dy.abs() != ring {
                        continue;
                    }
                    let (nx, ny) = (cx as i64 + dx, cy as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= grid.width as i64 || ny >= grid.height as i64 {
                        continue;
                    }
                    if !grid.is_blocked(nx as u32, ny as u32) {
                        let cell = grid.cell_size_milli;
                        return self.clamp_to_map(nx * cell + cell / 2, ny * cell + cell / 2);
                    }
                }
            }
        }
        p
    }

    /// Phase 18 Intel II: per-tick radar stealth / cloak state (blueprint flags,
    /// enhancement grants, `StealthWaitTime`, stealth & cloak fields, upkeep stalls) and
    /// the omni reveal mask of every cloaked unit. Deterministic ascending-id passes.
    fn step_intel(&mut self) {
        let mut fields: Vec<(ArmyId, i64, i64, i64, bool)> = Vec::new();
        let mut omni: Vec<(ArmyId, i64, i64, i64)> = Vec::new();
        for u in self.units.values_mut() {
            if !u.alive {
                continue;
            }
            if u.moving {
                u.still_ticks = 0;
            } else {
                u.still_ticks = u.still_ticks.saturating_add(1);
            }
            let spec = self.catalog.get(&u.type_id).map(|t| &t.ext.intel);
            let enh_on = !u.enh.stalled;
            let (mut rs, mut cl) = match spec {
                Some(i) if !i.gated => (i.radar_stealth, i.cloak),
                _ => (false, false),
            };
            rs |= enh_on && u.enh.radar_stealth;
            cl |= enh_on && u.enh.cloak;
            if let Some(i) = spec
                && i.stealth_wait_ticks > 0
                && u.still_ticks < i.stealth_wait_ticks
            {
                rs = false;
                cl = false;
            }
            u.radar_stealth = rs;
            u.cloaked = cl;
            u.omni_mask = 0;
            if u.carried_by.is_some() {
                continue;
            }
            if let Some(i) = spec
                && !i.gated
            {
                if i.stealth_field_radius > 0 {
                    fields.push((u.army, u.pos.x, u.pos.y, i.stealth_field_radius, false));
                }
                if i.cloak_field_radius > 0 {
                    fields.push((u.army, u.pos.x, u.pos.y, i.cloak_field_radius, true));
                }
            }
            if u.omni_radius > 0 {
                omni.push((u.army, u.pos.x, u.pos.y, u.omni_radius));
            }
        }
        if !fields.is_empty() {
            for u in self.units.values_mut() {
                if !u.alive || u.carried_by.is_some() {
                    continue;
                }
                for &(army, fx, fy, r, cloak) in &fields {
                    if army == u.army && dist_sq((fx, fy), (u.pos.x, u.pos.y)) <= (r as i128) * (r as i128) {
                        if cloak {
                            u.cloaked = true;
                        }
                        u.radar_stealth = true;
                    }
                }
            }
        }
        if !self.units.values().any(|u| u.alive && u.cloaked) {
            return;
        }
        let teams: Vec<u8> = self.armies.iter().map(|a| a.team).collect();
        let n_armies = teams.len().min(8);
        for u in self.units.values_mut() {
            if !u.alive || !u.cloaked {
                continue;
            }
            let mut mask = 0u8;
            for &(army, ox, oy, r) in &omni {
                let reach = (r + u.radius) as i128;
                if dist_sq((ox, oy), (u.pos.x, u.pos.y)) <= reach * reach {
                    let team = teams.get(army as usize).copied().unwrap_or(army);
                    for (a, &t) in teams.iter().enumerate().take(n_armies) {
                        if t == team {
                            mask |= 1 << a;
                        }
                    }
                }
            }
            u.omni_mask = mask;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn_projectile_full(
        &mut self,
        army: ArmyId,
        start: Vec2,
        target: Vec2,
        speed: i64,
        damage: i64,
        splash_radius: i64,
        is_overcharge: bool,
        is_missile: bool,
        is_nuke: bool,
        target_air: bool,
    ) {
        let id = self.next_projectile_id;
        self.next_projectile_id += 1;
        let d = isqrt(dist_sq((start.x, start.y), (target.x, target.y))).max(1);
        let spd = speed.max(SCALE / 2);
        let max_ticks = if is_nuke { 400 } else if is_missile { 250 } else { 120 };
        let total_ticks = ((d + spd - 1) / spd).clamp(1, max_ticks) as u32;
        let arc_height = if is_overcharge {
            0
        } else if is_nuke {
            95 * SCALE
        } else if is_missile {
            35 * SCALE
        } else {
            (d / 4).min(18 * SCALE)
        };
        self.projectiles.insert(
            id,
            Projectile {
                id,
                army,
                start,
                pos: start,
                target,
                speed: spd,
                damage,
                splash_radius,
                arc_height,
                ticks_lived: 0,
                total_ticks,
                is_overcharge,
                is_missile,
                is_nuke,
                target_air,
                direct: false,
                shooter: 0,
                z: 0,
                start_z: 0,
                target_z: 0,
                stun_ticks: 0,
                stun_radius: 0,
            },
        );
    }

    fn apply_shield_and_damage(
        &mut self,
        target_id: UnitId,
        damage: i64,
        attacker_army: ArmyId,
        attacker_unit: Option<UnitId>,
        shield_ids: &[UnitId],
    ) {
        let Some((t_army, tx, ty, t_alt)) = self
            .units
            .get(&target_id)
            .filter(|t| t.alive)
            .map(|t| (t.army, t.pos.x, t.pos.y, t.altitude))
        else {
            return;
        };

        // Check active allied bubble shields covering (tx, ty) in deterministic ID order
        // (Phase 18: personal shields only cover their owner).
        // `shield_ids` (ascending) holds every live unit with a shield radius; callers build it
        // once per phase since neither phase creates units or changes shield radii.
        let shield_holder = shield_ids
            .iter()
            .filter_map(|id| self.units.get(id))
            .find(|s| {
                s.alive
                    && s.shield_on
                    && s.shield_hp > 0
                    && s.shield_radius > 0
                    && (!s.shield_personal || s.id == target_id)
                    && self.are_allies(s.army, t_army)
                    && dist_sq((s.pos.x, s.pos.y), (tx, ty))
                        <= (s.shield_radius as i128) * (s.shield_radius as i128)
            })
            .map(|s| s.id);

        let mut remaining = damage;
        if let Some(sid) = shield_holder
            && let Some(sh) = self.units.get_mut(&sid)
        {
            let absorbed = sh.shield_hp.min(remaining);
            sh.shield_hp -= absorbed;
            remaining -= absorbed;
            if sh.shield_hp <= 0 {
                sh.shield_hp = 0;
                sh.shield_recharge_cd = SHIELD_RECHARGE_DELAY;
            }
            let sh_army = sh.army;
            let sh_pos = (tx, ty, t_alt + 2 * SCALE);
            self.armies[attacker_army as usize].damage_dealt += absorbed;
            self.emit_event(SimEvent::ShieldHit { pos: sh_pos, army: sh_army });
        }

        if remaining > 0
            && let Some(t) = self.units.get_mut(&target_id)
        {
            self.armies[attacker_army as usize].damage_dealt += remaining;
            combat::apply_hit(&mut t.hp, remaining);
            t.last_hit_by = Some(attacker_army);
            t.last_hit_by_unit = attacker_unit;
        }
    }

    /// Live units with a shield radius, ascending (see [`Self::apply_shield_and_damage`]).
    fn shield_candidates(&self) -> Vec<UnitId> {
        self.units
            .values()
            .filter(|s| s.alive && s.shield_radius > 0)
            .map(|s| s.id)
            .collect()
    }

    fn nearest_snap_target(
        idx: usize,
        snap: &[UnitSpatialSnap],
        grid: &SpatialGrid,
    ) -> (UnitId, Option<UnitId>) {
        let u = &snap[idx];
        let lim_r = u.max_range + 8 * SCALE;
        let lim_sq = (lim_r as i128) * (lim_r as i128);
        let p = (u.x, u.y);
        let mut best_mobile: Option<(i128, UnitId)> = None;
        let mut best_struct: Option<(i128, UnitId)> = None;
        let mut best_wall: Option<(i128, UnitId)> = None;
        grid.for_each_enemy_in_radius(p.0, p.1, lim_r + 6 * SCALE, u.team, |oidx| {
            let o = &snap[oidx as usize];
            if o.team == u.team || o.hidden_from(u.army) {
                return;
            }
            let o_air = o.domain == NavDomain::Air;
            if (o_air && !u.can_fire_aa) || (!o_air && !u.can_fire_surface) {
                return;
            }
            let reach = (u.max_range + o.radius) as i128;
            let d = dist_sq(p, (o.x, o.y));
            if d > lim_sq || d > reach * reach {
                return;
            }
            let slot = if o.is_wall {
                &mut best_wall
            } else if o.is_struct {
                &mut best_struct
            } else {
                &mut best_mobile
            };
            if slot.is_none_or(|(bd, bid)| d < bd || (d == bd && o.id < bid)) {
                *slot = Some((d, o.id));
            }
        });
        (u.id, best_mobile.or(best_struct).or(best_wall).map(|(_, id)| id))
    }

    fn step_weapons(&mut self) {
        // Pass A (Phase 16): Parallel deterministic target acquisition across armed units.
        let _pa = crate::profiler::scope("weapons.acquire");
        let armed_snaps: Vec<usize> = self
            .spatial_snap
            .iter()
            .enumerate()
            .filter(|(_, s)| s.max_range > 0)
            .map(|(i, _)| i)
            .collect();

        let acquired: Vec<(UnitId, Option<UnitId>)> = if armed_snaps.len() >= 128
            && let Some(ref pool) = self.pool
            && pool.threads() > 1
        {
            let snap_arc = Arc::clone(&self.spatial_snap);
            let grid_arc = Arc::clone(&self.spatial);
            pool.par_map_ordered_arc(
                Arc::new(armed_snaps),
                Arc::new(move |_: usize, &sidx: &usize| {
                    Self::nearest_snap_target(sidx, &snap_arc, &grid_arc)
                }),
            )
        } else {
            let snap = &self.spatial_snap;
            let grid = &self.spatial;
            armed_snaps
                .iter()
                .map(|&sidx| Self::nearest_snap_target(sidx, snap, grid))
                .collect()
        };

        let mut target_map: BTreeMap<UnitId, UnitId> = BTreeMap::new();
        for (uid, opt_tid) in acquired {
            if let Some(tid) = opt_tid {
                target_map.insert(uid, tid);
            }
        }

        // Pass B: Strictly sequential resolution in ascending UnitId order.
        drop(_pa);
        let _pb = crate::profiler::scope("weapons.resolve");
        let shield_ids = self.shield_candidates();
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        for id in ids {
            let (army, pos, alt, role, explicit, overcharge, has_guns) = match self.units.get(&id) {
                Some(u) if u.alive && u.carried_by.is_none() && u.stunned == 0 => (
                    u.army,
                    (u.pos.x, u.pos.y),
                    u.altitude,
                    u.role,
                    match u.orders.front() {
                        Some(UnitOrder::Attack(t)) => Some(*t),
                        _ => None,
                    },
                    match u.orders.front() {
                        Some(UnitOrder::Overcharge(t)) => Some(*t),
                        _ => None,
                    },
                    !u.weapons.is_empty(),
                ),
                _ => continue,
            };

            if role == Role::Commander
                && let Some(tid) = overcharge
                && let Some(t) = self.unit(tid).filter(|t| t.targetable_by(army))
            {
                let tp = (t.pos.x, t.pos.y);
                let t_alt = t.altitude;
                let reach = (OVERCHARGE_RANGE + t.radius) as i128;
                if dist_sq(pos, tp) <= reach * reach
                    && self.armies[army as usize].energy >= OVERCHARGE_ENERGY_COST
                {
                    self.armies[army as usize].energy -= OVERCHARGE_ENERGY_COST;
                    let tick = self.tick;
                    if let Some(me) = self.units.get_mut(&id) {
                        me.last_shot = Some((tid, tick));
                        me.orders.pop_front();
                    }
                    self.shots_fired += 1;
                    self.spawn_projectile_full(
                        army,
                        Vec2 { x: pos.0, y: pos.1 },
                        Vec2 { x: tp.0, y: tp.1 },
                        25 * SCALE / TICK_HZ as i64,
                        OVERCHARGE_DAMAGE,
                        OVERCHARGE_SPLASH,
                        true,
                        false,
                        false,
                        false,
                    );
                    self.emit_event(SimEvent::WeaponFired {
                        from: (pos.0, pos.1, alt + 2 * SCALE),
                        to: (tp.0, tp.1, t_alt),
                        army,
                        heavy: true,
                        anti_air: false,
                    });
                    continue;
                }
            }

            if !has_guns {
                continue;
            }
            // Phase 35: ground fire — shoot at the position (splash shells, direct shells into
            // the ground, terrain-checked beams).
            if let Some(UnitOrder::AttackGround { x, y }) = self.units[&id].orders.front().cloned() {
                self.fire_at_ground(id, (x, y), &shield_ids);
                continue;
            }
            let (fire_state, last_attacker) = {
                let u = &self.units[&id];
                (u.fire_state, u.last_hit_by_unit)
            };
            let target = explicit
                .filter(|t| self.in_weapon_range(&self.units[&id], *t))
                .or_else(|| match fire_state {
                    FireState::FireAtWill => target_map.get(&id).copied().filter(|t| self.unit(*t).is_some()),
                    FireState::ReturnFire => last_attacker.filter(|t| {
                        self.unit(*t).is_some_and(|e| e.alive && e.targetable_by(army) && !self.are_allies(army, e.army))
                            && self.in_weapon_range(&self.units[&id], *t)
                    }),
                    FireState::HoldFire => None,
                });
            let tinfo = target.and_then(|t| {
                self.unit(t).map(|u| {
                    (t, (u.pos.x, u.pos.y), u.radius, u.altitude, u.domain == NavDomain::Air, u.height, u.vel)
                })
            });
            let my_height = self.units[&id].height;
            let n = self.units[&id].weapons.len();
            for wi in 0..n {
                // Phase 29: turrets slew toward the current target (also while reloading)
                // and only fire once within `FiringTolerance`.
                let mut aimed = true;
                if let Some((_, tp, _, _, t_air, _, _)) = tinfo {
                    let me = self.units.get_mut(&id).unwrap();
                    let ws = &mut me.weapons[wi];
                    if (t_air && ws.anti_air) || (!t_air && ws.anti_surface) {
                        let desired = combat::atan2_mrad(tp.0 - pos.0, tp.1 - pos.1);
                        if ws.turret_yaw_speed > 0 {
                            let diff = combat::wrap_mrad(desired - ws.yaw);
                            ws.yaw = combat::wrap_mrad(ws.yaw + diff.clamp(-ws.turret_yaw_speed, ws.turret_yaw_speed));
                            aimed = combat::wrap_mrad(desired - ws.yaw).abs() <= ws.firing_tolerance;
                        } else {
                            ws.yaw = desired;
                        }
                    }
                }
                let w = self.units[&id].weapons[wi].clone();
                if w.cooldown > 0 {
                    self.units.get_mut(&id).unwrap().weapons[wi].cooldown -= 1;
                    continue;
                }
                let Some((tid, tp, tr, t_alt, t_air, t_height, t_vel)) = tinfo else {
                    continue;
                };
                if (t_air && !w.anti_air) || (!t_air && !w.anti_surface) || !aimed {
                    continue;
                }
                let alive = self.units.get(&tid).is_some_and(|t| t.alive);
                let reach = isqrt(w.range_sq) + tr;
                let reach_sq = (reach as i128) * (reach as i128);
                if combat::can_fire(0, alive, Some(reach_sq), dist_sq(pos, tp), Some(w.damage)) {
                    let tick = self.tick;
                    let yaw_mrad = w.yaw;
                    {
                        let me = self.units.get_mut(&id).unwrap();
                        me.weapons[wi].cooldown = w.reset;
                        me.last_shot = Some((tid, tick));
                        me.turret_yaw_mrad = yaw_mrad;
                    }
                    self.shots_fired += 1;
                    self.emit_event(SimEvent::WeaponFired {
                        from: (pos.0, pos.1, alt + SCALE),
                        to: (tp.0, tp.1, t_alt + SCALE / 2),
                        army,
                        heavy: w.splash_radius > 0 || w.damage >= 100 * SCALE,
                        anti_air: t_air,
                    });
                    let stun = w.stun.or_else(|| {
                        w.emp
                            .then(|| self.catalog.get(&self.units[&id].type_id).and_then(|t| t.ext.stun).filter(|s| s.2).map(|s| (s.0, s.1)))
                            .flatten()
                    });
                    if w.kamikaze {
                        // Phase 33: detonate in place — damage enemies around, then die.
                        self.splash_damage(army, pos, w.splash_radius.max(SCALE), w.damage, Some(id), false, &shield_ids);
                        self.units.get_mut(&id).unwrap().hp = 0;
                        self.emit_event(SimEvent::Explosion { pos: (pos.0, pos.1, SCALE / 2), radius: w.splash_radius.max(2 * SCALE), is_nuke: false });
                        break;
                    }
                    if w.splash_radius > 0 {
                        let spd = if w.muzzle_velocity > 0 {
                            w.muzzle_velocity
                        } else {
                            14 * SCALE / TICK_HZ as i64
                        };
                        self.spawn_projectile_full(
                            army,
                            Vec2 { x: pos.0, y: pos.1 },
                            Vec2 { x: tp.0, y: tp.1 },
                            spd,
                            w.damage,
                            w.splash_radius,
                            false,
                            false,
                            false,
                            t_air,
                        );
                        if let Some((ticks, radius)) = stun
                            && let Some(p) = self.projectiles.get_mut(&(self.next_projectile_id - 1))
                        {
                            p.stun_ticks = ticks;
                            p.stun_radius = radius;
                        }
                    } else if t_air {
                        // Anti-air stays hitscan (FA AA is homing / flak).
                        self.apply_shield_and_damage(tid, w.damage, army, Some(id), &shield_ids);
                    } else {
                        // Phase 29: surface direct fire from the muzzle to the target's centre.
                        let muzzle = (pos.0, pos.1, self.terrain.height_at(pos.0, pos.1) + alt + (my_height * 3 / 4).max(SCALE / 2));
                        let centre = |p: (i64, i64)| (p.0, p.1, self.terrain.height_at(p.0, p.1) + t_alt + t_height / 2);
                        if w.beam {
                            // Instant beam: the terrain between muzzle and target soaks the shot.
                            let aim = centre(tp);
                            match self.terrain.segment_hit(muzzle, aim, LOF_CLEARANCE, SCALE, tr + SCALE) {
                                Some(_) => {}
                                None => {
                                    self.apply_shield_and_damage(tid, w.damage, army, Some(id), &shield_ids);
                                    if let Some((ticks, radius)) = stun {
                                        self.apply_stun(army, tp, radius, ticks);
                                    }
                                }
                            }
                        } else {
                            // Projectile: lead the target by its last-tick velocity.
                            let spd = w.muzzle_velocity.max(SCALE / 2);
                            let mut aim2 = tp;
                            if w.lead_target && t_vel != (0, 0) {
                                for _ in 0..2 {
                                    let t = isqrt(dist_sq(pos, aim2)) / spd;
                                    aim2 = self.clamp_to_map(tp.0 + t_vel.0 * t, tp.1 + t_vel.1 * t);
                                }
                            }
                            self.spawn_direct_projectile(army, id, muzzle, centre(aim2), spd, w.damage);
                            if let Some((ticks, radius)) = stun
                                && let Some(p) = self.projectiles.get_mut(&(self.next_projectile_id - 1))
                            {
                                p.stun_ticks = ticks;
                                p.stun_radius = radius;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Phase 35: every ready weapon of `id` in range of ground point `g` fires at it.
    fn fire_at_ground(&mut self, id: UnitId, g: (i64, i64), shield_ids: &[UnitId]) {
        let Some((army, pos, alt, height)) = self.units.get(&id).map(|u| (u.army, (u.pos.x, u.pos.y), u.altitude, u.height)) else {
            return;
        };
        let _ = shield_ids;
        let n = self.units[&id].weapons.len();
        for wi in 0..n {
            let w = self.units[&id].weapons[wi].clone();
            if w.cooldown > 0 {
                self.units.get_mut(&id).unwrap().weapons[wi].cooldown -= 1;
                continue;
            }
            if !w.anti_surface || w.kamikaze || dist_sq(pos, g) > w.range_sq {
                continue;
            }
            let tick = self.tick;
            {
                let me = self.units.get_mut(&id).unwrap();
                me.weapons[wi].cooldown = w.reset;
                me.weapons[wi].yaw = combat::atan2_mrad(g.0 - pos.0, g.1 - pos.1);
                me.turret_yaw_mrad = me.weapons[wi].yaw;
                me.last_shot = Some((0, tick));
            }
            self.shots_fired += 1;
            let gz = self.terrain.height_at(g.0, g.1);
            self.emit_event(SimEvent::WeaponFired {
                from: (pos.0, pos.1, alt + SCALE),
                to: (g.0, g.1, gz),
                army,
                heavy: w.splash_radius > 0,
                anti_air: false,
            });
            if w.splash_radius > 0 {
                let spd = if w.muzzle_velocity > 0 { w.muzzle_velocity } else { 14 * SCALE / TICK_HZ as i64 };
                self.spawn_projectile_full(army, Vec2 { x: pos.0, y: pos.1 }, Vec2 { x: g.0, y: g.1 }, spd, w.damage, w.splash_radius, false, false, false, false);
            } else if !w.beam {
                let muzzle = (pos.0, pos.1, self.terrain.height_at(pos.0, pos.1) + alt + (height * 3 / 4).max(SCALE / 2));
                self.spawn_direct_projectile(army, id, muzzle, (g.0, g.1, gz), w.muzzle_velocity.max(SCALE / 2), w.damage);
            }
        }
    }

    /// Phase 29: a straight-flying direct-fire shell from `start` to `target` (3-D, milli).
    fn spawn_direct_projectile(
        &mut self,
        army: ArmyId,
        shooter: UnitId,
        start: (i64, i64, i64),
        target: (i64, i64, i64),
        speed: i64,
        damage: i64,
    ) {
        let id = self.next_projectile_id;
        self.next_projectile_id += 1;
        let (dx, dy, dz) = ((target.0 - start.0) as i128, (target.1 - start.1) as i128, (target.2 - start.2) as i128);
        let d = isqrt(dx * dx + dy * dy + dz * dz).max(1);
        let spd = speed.max(1);
        let total_ticks = ((d + spd - 1) / spd).clamp(1, 120) as u32;
        self.projectiles.insert(
            id,
            Projectile {
                id,
                army,
                start: Vec2 { x: start.0, y: start.1 },
                pos: Vec2 { x: start.0, y: start.1 },
                target: Vec2 { x: target.0, y: target.1 },
                speed,
                damage,
                splash_radius: 0,
                arc_height: 0,
                ticks_lived: 0,
                total_ticks,
                is_overcharge: false,
                is_missile: false,
                is_nuke: false,
                target_air: false,
                direct: true,
                shooter,
                z: start.2,
                start_z: start.2,
                target_z: target.2,
                stun_ticks: 0,
                stun_radius: 0,
            },
        );
    }

    /// Phase 29: advances direct-fire shell `pid` one tick. The shell hits the first thing on
    /// its path this tick — terrain (more than [`LOF_CLEARANCE`] below the ground) or a
    /// non-allied surface unit whose hit box (footprint radius × height) it passes through,
    /// walls included — and otherwise flies on; reaching its aim point without a hit is a miss.
    fn step_direct_projectile(&mut self, pid: u32, shield_ids: &[UnitId]) {
        let Some(p) = self.projectiles.get(&pid).cloned() else { return };
        let tt = p.total_ticks.max(1);
        let t1 = (p.ticks_lived + 1).min(tt);
        let lerp = |a: i64, b: i64| a + ((b - a) as i128 * t1 as i128 / tt as i128) as i64;
        let a = (p.pos.x, p.pos.y, p.z);
        let b = (lerp(p.start.x, p.target.x), lerp(p.start.y, p.target.y), lerp(p.start_z, p.target_z));
        let last = t1 >= tt;
        let (sx, sy, sz) = ((b.0 - a.0) as i128, (b.1 - a.1) as i128, (b.2 - a.2) as i128);
        let seg2 = sx * sx + sy * sy;
        let seg = isqrt(seg2);

        // Terrain: parameter along the segment in 1/1000ths.
        let skip_start = if p.ticks_lived == 0 { SCALE } else { 0 };
        let skip_end = if last { SCALE * 3 / 2 } else { 0 };
        let ground = self.terrain.segment_hit(a, b, LOF_CLEARANCE, skip_start, skip_end).map(|h| {
            let along = isqrt(dist_sq((a.0, a.1), (h.0, h.1)));
            (along as i128 * 1000 / (seg as i128).max(1), h)
        });

        // Units: closest approach of the segment to each candidate's centre, from the
        // spatial snapshot (positions as of this tick's weapons pass). Candidates are tried in
        // (t, id) order and the first one still alive takes the hit.
        let mid = ((a.0 + b.0) / 2, (a.1 + b.1) / 2);
        let team = self.armies.get(p.army as usize).map(|a| a.team).unwrap_or(p.army);
        let terrain = &self.terrain;
        let snap = &self.spatial_snap;
        let mut hits: Vec<(i128, UnitId)> = Vec::new();
        self.spatial.for_each_enemy_in_radius(mid.0, mid.1, seg / 2 + DIRECT_HIT_REACH, team, |idx| {
            let u = &snap[idx as usize];
            if u.team == team || u.domain == NavDomain::Air || u.id == p.shooter {
                return;
            }
            let (px, py) = ((u.x - a.0) as i128, (u.y - a.1) as i128);
            let t = if seg2 == 0 { 1000 } else { ((px * sx + py * sy) * 1000 / seg2).clamp(0, 1000) };
            let c = (a.0 + (sx * t / 1000) as i64, a.1 + (sy * t / 1000) as i64);
            let r = u.radius as i128;
            if dist_sq(c, (u.x, u.y)) > r * r {
                return;
            }
            let z = a.2 + (sz * t / 1000) as i64;
            let base = terrain.height_at(u.x, u.y);
            if z > base + u.height + LOF_CLEARANCE || z + SCALE < base {
                return;
            }
            hits.push((t, u.id));
        });
        hits.sort_unstable();
        let best = hits
            .into_iter()
            .find(|&(_, id)| self.units.get(&id).is_some_and(|u| u.alive && u.carried_by.is_none()));

        match (best, ground) {
            (Some((tu, uid)), g) if g.is_none_or(|(tg, _)| tu <= tg) => {
                self.projectiles.remove(&pid);
                let shooter = self.units.contains_key(&p.shooter).then_some(p.shooter);
                self.apply_shield_and_damage(uid, p.damage, p.army, shooter, shield_ids);
                if p.stun_ticks > 0 {
                    let up = self.units.get(&uid).map_or((b.0, b.1), |u| (u.pos.x, u.pos.y));
                    self.apply_stun(p.army, up, p.stun_radius, p.stun_ticks);
                }
            }
            (_, Some(_)) => {
                self.projectiles.remove(&pid);
            }
            _ if last => {
                self.projectiles.remove(&pid);
            }
            _ => {
                let q = self.projectiles.get_mut(&pid).unwrap();
                q.ticks_lived = t1;
                q.pos = Vec2 { x: b.0, y: b.1 };
                q.z = b.2;
            }
        }
    }

    fn step_projectiles(&mut self) {
        let shield_ids = self.shield_candidates();
        // Missile Defense interception check for incoming tactical & strategic missiles.
        // Phase 18: TMD only stops tactical missiles, SMD only strategic ones (and spends an
        // interceptor from its ammo); each defense engages one missile per reload.
        for u in self.units.values_mut() {
            if u.intercept_cd > 0 {
                u.intercept_cd -= 1;
            }
        }
        let def_sites: Vec<(UnitId, ArmyId, i64, i64, i64, bool, bool)> = self
            .units
            .values()
            .filter(|u| u.alive && u.role == Role::MissileDefense && u.carried_by.is_none())
            .filter_map(|u| {
                let t = self.catalog.get(&u.type_id)?;
                let rad = t.intercept_radius;
                let ammo_ok = t.ext.ammo_cap == 0 || u.silo_missiles > 0;
                let ready = u.intercept_cd == 0 && ammo_ok;
                (rad > 0 && ready).then_some((
                    u.id,
                    u.army,
                    u.pos.x,
                    u.pos.y,
                    rad,
                    t.ext.intercept_tactical,
                    t.ext.intercept_strategic,
                ))
            })
            .collect();
        let mut site_used = vec![false; def_sites.len()];

        // Phase 29: direct-fire shells fly and collide on their own.
        let direct: Vec<u32> = self.projectiles.values().filter(|p| p.direct).map(|p| p.id).collect();
        for pid in direct {
            self.step_direct_projectile(pid, &shield_ids);
        }
        let pids: Vec<u32> = self.projectiles.values().filter(|p| !p.direct).map(|p| p.id).collect();
        let mut exploded = Vec::new();
        let mut intercepted = Vec::new();
        for pid in pids {
            let p = self.projectiles.get_mut(&pid).unwrap();
            p.ticks_lived += 1;
            let done = p.ticks_lived >= p.total_ticks
                || dist_sq((p.pos.x, p.pos.y), (p.target.x, p.target.y))
                    <= (p.speed as i128) * (p.speed as i128);
            if done {
                p.pos = p.target;
                exploded.push(pid);
            } else {
                let t_num = p.ticks_lived as i128;
                let t_den = p.total_ticks.max(1) as i128;
                p.pos.x = p.start.x + ((p.target.x - p.start.x) as i128 * t_num / t_den) as i64;
                p.pos.y = p.start.y + ((p.target.y - p.start.y) as i128 * t_num / t_den) as i64;

                if p.is_missile {
                    let parm = p.army;
                    let px = p.pos.x;
                    let py = p.pos.y;
                    let is_nuke = p.is_nuke;
                    let team_map = &self.armies;
                    let p_team = team_map.get(parm as usize).map(|a| a.team).unwrap_or(parm);
                    for (si, &(did, darm, dx, dy, drad, vs_tac, vs_strat)) in def_sites.iter().enumerate() {
                        if site_used[si] || (is_nuke && !vs_strat) || (!is_nuke && !vs_tac) {
                            continue;
                        }
                        let d_team = team_map.get(darm as usize).map(|a| a.team).unwrap_or(darm);
                        if d_team != p_team && dist_sq((px, py), (dx, dy)) <= (drad as i128) * (drad as i128) {
                            site_used[si] = true;
                            intercepted.push((pid, did, px, py));
                            break;
                        }
                    }
                }
            }
        }

        for (pid, did, px, py) in intercepted {
            self.projectiles.remove(&pid);
            let (reset, ammo) = self
                .units
                .get(&did)
                .and_then(|d| self.catalog.get(&d.type_id))
                .map(|t| (t.ext.intercept_reset, t.ext.ammo_cap > 0))
                .unwrap_or((20, false));
            if let Some(d) = self.units.get_mut(&did) {
                d.intercept_cd = reset;
                if ammo {
                    d.silo_missiles = d.silo_missiles.saturating_sub(1);
                }
            }
            self.emit_event(SimEvent::MissileIntercepted {
                pos: (px, py, 14 * SCALE),
            });
        }

        let mut cand = Vec::new();
        for pid in exploded {
            let Some(p) = self.projectiles.remove(&pid) else { continue };
            self.emit_event(SimEvent::Explosion {
                pos: (p.target.x, p.target.y, SCALE / 2),
                radius: p.splash_radius.max(2 * SCALE),
                is_nuke: p.is_nuke,
            });
            let r = p.splash_radius.max(SCALE);
            cand.clear();
            self.spatial
                .for_each_in_radius(p.target.x, p.target.y, r + 4 * SCALE, |idx| {
                    cand.push(self.spatial_snap[idx as usize].id);
                });
            cand.sort_unstable();
            for &uid in &cand {
                let Some((t_army, tx, ty, tr, t_alt)) = self
                    .units
                    .get(&uid)
                    .filter(|t| t.alive)
                    .map(|t| (t.army, t.pos.x, t.pos.y, t.radius, t.altitude))
                else {
                    continue;
                };
                if self.are_allies(p.army, t_army) {
                    continue;
                }
                if t_alt > 4 * SCALE && !p.target_air && !p.is_nuke {
                    continue;
                }
                let reach = (r + tr) as i128;
                if dist_sq((p.target.x, p.target.y), (tx, ty)) <= reach * reach {
                    self.apply_shield_and_damage(uid, p.damage, p.army, None, &shield_ids);
                }
            }
            if p.stun_ticks > 0 {
                self.apply_stun(p.army, (p.target.x, p.target.y), p.stun_radius, p.stun_ticks);
            }
        }
    }

    /// Phase 33: stuns every non-allied mobile surface unit within `radius` of `p` for
    /// `ticks` (FA `Buffs.Stun`, `TargetAllow = 'MOBILE'`). Ascending id; never shortens.
    fn apply_stun(&mut self, army: ArmyId, p: (i64, i64), radius: i64, ticks: u32) {
        if ticks == 0 {
            return;
        }
        let lim = (radius.max(SCALE) as i128).pow(2);
        let ids: Vec<UnitId> = self
            .units
            .values()
            .filter(|u| {
                u.alive
                    && u.speed > 0
                    && !u.role.is_structure()
                    && u.domain != NavDomain::Air
                    && u.carried_by.is_none()
                    && !self.are_allies(army, u.army)
                    && dist_sq(p, (u.pos.x, u.pos.y)) <= lim
            })
            .map(|u| u.id)
            .collect();
        for id in ids {
            let u = self.units.get_mut(&id).unwrap();
            u.stunned = u.stunned.max(ticks);
            u.path.clear();
            u.path_goal = None;
        }
    }

    /// Phase 33: area damage at `p` (`radius` + target hit box) to surface units — enemies of
    /// `army` only, or everyone when `friendly_fire` (death explosions). Ascending id.
    #[allow(clippy::too_many_arguments)]
    fn splash_damage(
        &mut self,
        army: ArmyId,
        p: (i64, i64),
        radius: i64,
        damage: i64,
        attacker: Option<UnitId>,
        friendly_fire: bool,
        shield_ids: &[UnitId],
    ) {
        let hits: Vec<UnitId> = self
            .units
            .values()
            .filter(|u| {
                u.alive
                    && u.hp > 0
                    && Some(u.id) != attacker
                    && u.carried_by.is_none()
                    && !u.untargetable
                    && u.altitude <= 4 * SCALE
                    && (friendly_fire || !self.are_allies(army, u.army))
                    && dist_sq(p, (u.pos.x, u.pos.y)) <= ((radius + u.radius) as i128).pow(2)
            })
            .map(|u| u.id)
            .collect();
        for uid in hits {
            self.apply_shield_and_damage(uid, damage, army, attacker, shield_ids);
        }
    }

    /// Phase 36 ferries (ascending transport id): at the pick-up point an empty-ish
    /// transport loads idle own land units within [`FERRY_RADIUS`] (ascending id, capacity
    /// checked) and leaves as soon as it carries anyone; at the drop-off point it unloads
    /// in formation and heads back. The route repeats until another order replaces it.
    fn step_ferries(&mut self) {
        type Route = (UnitId, (i64, i64), (i64, i64));
        let ferries: Vec<Route> = self
            .units
            .values()
            .filter(|u| u.alive && u.carried_by.is_none())
            .filter_map(|u| match u.orders.front() {
                Some(UnitOrder::Ferry { from, to }) => Some((u.id, *from, *to)),
                _ => None,
            })
            .collect();
        for (tid, from, to) in ferries {
            let (tpos, army, cargo) = {
                let t = &self.units[&tid];
                ((t.pos.x, t.pos.y), t.army, t.cargo.clone())
            };
            if cargo.is_empty() && tpos == from {
                let lim = (FERRY_RADIUS as i128).pow(2);
                let waiting: Vec<UnitId> = self
                    .units
                    .values()
                    .filter(|u| {
                        u.alive
                            && u.army == army
                            && u.domain != NavDomain::Air
                            && u.carried_by.is_none()
                            && u.orders.is_empty()
                            && u.build.is_none()
                            && dist_sq((u.pos.x, u.pos.y), from) <= lim
                    })
                    .map(|u| u.id)
                    .collect();
                for pid in waiting {
                    if !self.can_board(pid, tid) {
                        continue;
                    }
                    let p = self.units.get_mut(&pid).unwrap();
                    p.carried_by = Some(tid);
                    p.path.clear();
                    p.path_goal = None;
                    p.moving = false;
                    self.units.get_mut(&tid).unwrap().cargo.push(pid);
                }
            } else if !cargo.is_empty() && tpos == to {
                let n = cargo.len();
                for (i, &c) in cargo.iter().enumerate() {
                    let (sx, sy) = Self::formation_slot(i, n, to.0, to.1);
                    let (sx, sy) = self.walkable_near(c, self.clamp_to_map(sx, sy));
                    if let Some(u) = self.units.get_mut(&c) {
                        u.carried_by = None;
                        u.pos = Vec2 { x: sx, y: sy };
                        u.altitude = 0;
                        u.orders.clear();
                        u.path.clear();
                        u.path_goal = None;
                    }
                }
                let t = self.units.get_mut(&tid).unwrap();
                t.cargo.clear();
                t.path.clear();
                t.path_goal = None;
                self.emit_event(SimEvent::BuildSpark { pos: (to.0, to.1, SCALE), army });
            }
        }
    }

    fn step_cleanup(&mut self) {
        // Phase 18: passengers die with their transport (FA behaviour).
        let doomed: Vec<(Vec<UnitId>, Option<ArmyId>, Option<UnitId>)> = self
            .units
            .values()
            .filter(|u| u.alive && u.hp <= 0 && !u.cargo.is_empty())
            .map(|u| (u.cargo.clone(), u.last_hit_by, u.last_hit_by_unit))
            .collect();
        for (cargo, by, by_unit) in doomed {
            for c in cargo {
                if let Some(p) = self.units.get_mut(&c)
                    && p.alive
                {
                    p.hp = 0;
                    p.last_hit_by = by;
                    p.last_hit_by_unit = by_unit;
                }
            }
        }
        let mut dead_commanders = Vec::new();
        let mut death_fx: Vec<DeathFx> = Vec::new();
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        for id in &ids {
            let (army, killer, killer_unit, role, pos, type_id, died) = {
                let u = self.units.get_mut(id).unwrap();
                if u.alive && u.hp <= 0 {
                    u.hp = 0;
                    u.alive = false;
                    self.kills += 1;
                    (
                        u.army,
                        u.last_hit_by,
                        u.last_hit_by_unit,
                        u.role,
                        u.pos,
                        u.type_id.clone(),
                        true,
                    )
                } else {
                    (0, None, None, Role::Mobile, Vec2 { x: 0, y: 0 }, String::new(), false)
                }
            };
            if died {
                if let Some(t) = self.catalog.get(&type_id) {
                    let stun = t.ext.stun.filter(|s| s.3).map(|s| (s.0, s.1));
                    if t.ext.death_damage > 0 || stun.is_some() {
                        death_fx.push((army, (pos.x, pos.y), t.ext.death_damage, t.ext.death_radius, stun));
                    }
                }
                self.armies[army as usize].units_lost += 1;
                if let Some(k) = killer {
                    self.armies[k as usize].kills += 1;
                }
                if let Some(ku_id) = killer_unit
                    && let Some(ku) = self.units.get_mut(&ku_id)
                    && ku.alive
                {
                    ku.vet_kills += 1;
                    let thresholds = self
                        .catalog
                        .get(&ku.type_id)
                        .map(|t| t.vet_thresholds)
                        .unwrap_or([5, 10, 25, 50, 100]);
                    while (ku.vet_level as usize) < 5
                        && ku.vet_kills >= thresholds[ku.vet_level as usize]
                    {
                        ku.vet_level += 1;
                        let boost = ku.max_hp / 10;
                        ku.max_hp += boost;
                        ku.hp = (ku.hp + boost).min(ku.max_hp);
                    }
                }
                if role == Role::Commander {
                    self.emit_event(SimEvent::AcuDied {
                        pos: (pos.x, pos.y, 2 * SCALE),
                        army,
                    });
                    if self.options.victory == VictoryCondition::Assassination {
                        dead_commanders.push(army);
                    }
                } else if let Some(ut) = self.catalog.get(&type_id) {
                    let is_struct = ut.role.is_structure();
                    let ev_rad = (ut.radius * 2).max(2 * SCALE);
                    let mass_left = (ut.cost_mass * 81) / 100;
                    if !is_struct {
                        self.emit_event(SimEvent::Explosion {
                            pos: (pos.x, pos.y, SCALE),
                            radius: ev_rad,
                            is_nuke: false,
                        });
                    }
                    if mass_left > 0 {
                        self.wrecks.insert(
                            *id,
                            Wreck {
                                id: *id,
                                type_id,
                                pos,
                                mass_left,
                            },
                        );
                    }
                }
            }
        }
        // Phase 33: death weapons hit everything around the wreck (chain deaths resolve next
        // tick); on-death EMP stuns enemy mobiles.
        if !death_fx.is_empty() {
            let shield_ids = self.shield_candidates();
            for (army, p, dmg, radius, stun) in death_fx {
                if dmg > 0 {
                    self.splash_damage(army, p, radius, dmg, None, true, &shield_ids);
                    self.emit_event(SimEvent::Explosion { pos: (p.0, p.1, SCALE), radius: radius.max(2 * SCALE), is_nuke: radius >= 15 * SCALE });
                }
                if let Some((ticks, r)) = stun {
                    self.apply_stun(army, p, r, ticks);
                }
            }
        }
        // Phase 18 victory conditions: supremacy (structures + engineers) / annihilation.
        let victory = self.options.victory;
        if matches!(victory, VictoryCondition::Supremacy | VictoryCondition::Annihilation) {
            for ai in 0..self.armies.len() {
                let army = ai as ArmyId;
                if self.armies[ai].defeated {
                    continue;
                }
                let holds_on = self.units.values().any(|u| {
                    u.alive
                        && u.army == army
                        && u.role != Role::Wall
                        && (victory == VictoryCondition::Annihilation || u.role.is_structure() || u.role.is_builder())
                });
                if !holds_on {
                    dead_commanders.push(army);
                }
            }
        }
        for army in dead_commanders {
            self.armies[army as usize].defeated = true;
            for u in self.units.values_mut() {
                if u.army == army && u.alive {
                    u.alive = false;
                    u.hp = 0;
                    self.armies[army as usize].units_lost += 1;
                }
            }
        }
        for i in 0..self.mass_claims.len() {
            if let Some(holder) = self.mass_claims[i]
                && !self.units.get(&holder).is_some_and(|u| u.alive)
            {
                self.mass_claims[i] = None;
            }
        }
        for i in 0..self.hydro_claims.len() {
            if let Some(holder) = self.hydro_claims[i]
                && !self.units.get(&holder).is_some_and(|u| u.alive)
            {
                self.hydro_claims[i] = None;
            }
        }
        if self.units.values().any(|u| !u.alive && u.role.is_structure()) {
            self.walls_dirty = true;
        }
        self.units.retain(|_, u| u.alive);
        let live: Vec<UnitId> = self.units.keys().copied().collect();
        for u in self.units.values_mut() {
            if !u.cargo.is_empty() {
                u.cargo.retain(|c| live.binary_search(c).is_ok());
            }
        }

        let alive_armies: Vec<(ArmyId, u8)> = self
            .armies
            .iter()
            .filter(|a| !a.defeated)
            .map(|a| (a.id, a.team))
            .collect();
        let mut distinct_teams: Vec<u8> = alive_armies.iter().map(|&(_, t)| t).collect();
        distinct_teams.sort_unstable();
        distinct_teams.dedup();
        self.result = match distinct_teams.len() {
            _ if victory == VictoryCondition::Sandbox => None,
            0 => Some(GameResult::Draw),
            1 => Some(GameResult::Victory(alive_armies[0].0)),
            _ => None,
        };
    }

    // ----- hashing -------------------------------------------------------------

    fn mix(h: &mut u64, bytes: &[u8]) {
        for b in bytes {
            *h ^= *b as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    /// FNV-1a 64 over every piece of tick state, integers only.
    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        let m = Self::mix;
        m(&mut h, SKIRMISH_HASH_MAGIC);
        m(&mut h, &self.seed.to_le_bytes());
        m(&mut h, &self.tick.to_le_bytes());
        m(&mut h, &self.rng.to_le_bytes());
        m(&mut h, &self.next_id.to_le_bytes());
        m(&mut h, &self.next_project.to_le_bytes());
        m(&mut h, &self.next_projectile_id.to_le_bytes());
        m(&mut h, &self.shots_fired.to_le_bytes());
        m(&mut h, &self.kills.to_le_bytes());
        m(&mut h, &(self.props.len() as u64).to_le_bytes());
        if let Some(ms) = &self.mission {
            m(&mut h, &ms.digest().to_le_bytes());
        }
        for p in self.props.values() {
            m(&mut h, &p.mass.to_le_bytes());
            m(&mut h, &p.energy.to_le_bytes());
        }
        for a in &self.armies {
            m(&mut h, &[a.id, a.team, a.defeated as u8]);
            for v in [a.mass, a.energy, a.mass_cap, a.energy_cap, a.damage_dealt] {
                m(&mut h, &v.to_le_bytes());
            }
            for v in [a.units_built, a.units_lost, a.kills] {
                m(&mut h, &v.to_le_bytes());
            }
        }
        for c in &self.mass_claims {
            m(&mut h, &c.unwrap_or(0).to_le_bytes());
        }
        for c in &self.hydro_claims {
            m(&mut h, &c.unwrap_or(0).to_le_bytes());
        }
        for (id, u) in &self.units {
            m(&mut h, &id.to_le_bytes());
            m(&mut h, &[u.army, u.role.code(), u.tier, u.vet_level, u8::from(u.shield_on)]);
            m(&mut h, &(u.type_id.len() as u32).to_le_bytes());
            m(&mut h, u.type_id.as_bytes());
            for v in [u.hp, u.pos.x, u.pos.y, u.altitude, u.shield_hp] {
                m(&mut h, &v.to_le_bytes());
            }
            m(&mut h, &u.silo_missiles.to_le_bytes());
            m(&mut h, &u.fuel.to_le_bytes());
            m(&mut h, &[u8::from(u.auto_refuel)]);
            m(&mut h, &u.drone_of.unwrap_or(0).to_le_bytes());
            m(&mut h, &u.pod_timer.to_le_bytes());
            m(&mut h, &u.stunned.to_le_bytes());
            m(&mut h, &[u.fire_state.code(), u8::from(u.build_paused)]);
            m(&mut h, &(u.orders.len() as u32).to_le_bytes());
            m(&mut h, &(u.production.len() as u32).to_le_bytes());
            m(&mut h, &(u.path.len() as u32).to_le_bytes());
            if let Some(j) = &u.build {
                m(&mut h, &j.project.progress.to_le_bytes());
            }
            for w in &u.weapons {
                m(&mut h, &w.cooldown.to_le_bytes());
            }
            // Phase 18 state, mixed only when non-default so pre-Phase-18 hashes are kept.
            if u.carried_by.is_some()
                || !u.cargo.is_empty()
                || u.repeat_queue
                || !u.enh.installed.is_empty()
                || u.intercept_cd > 0
            {
                m(&mut h, b"P18U");
                m(&mut h, &u.carried_by.unwrap_or(0).to_le_bytes());
                m(&mut h, &(u.cargo.len() as u32).to_le_bytes());
                for c in &u.cargo {
                    m(&mut h, &c.to_le_bytes());
                }
                m(&mut h, &[u8::from(u.repeat_queue), u8::from(u.cloaked), u8::from(u.radar_stealth)]);
                for (slot, name) in &u.enh.installed {
                    m(&mut h, slot.as_bytes());
                    m(&mut h, name.as_bytes());
                }
                for v in [u.max_hp, u.enh.build_rate, u.intercept_cd as i64] {
                    m(&mut h, &v.to_le_bytes());
                }
            }
        }
        if self.options != GameOptions::default() {
            m(&mut h, b"P18O");
            m(&mut h, &[self.options.victory.code()]);
            m(&mut h, &self.options.unit_cap.to_le_bytes());
        }
        for (wid, w) in &self.wrecks {
            m(&mut h, &wid.to_le_bytes());
            m(&mut h, &(w.type_id.len() as u32).to_le_bytes());
            m(&mut h, w.type_id.as_bytes());
            m(&mut h, &w.pos.x.to_le_bytes());
            m(&mut h, &w.pos.y.to_le_bytes());
            m(&mut h, &w.mass_left.to_le_bytes());
        }
        for (pid, p) in &self.projectiles {
            m(&mut h, &pid.to_le_bytes());
            m(&mut h, &[p.army, u8::from(p.is_overcharge), u8::from(p.is_missile), u8::from(p.is_nuke)]);
            for v in [p.pos.x, p.pos.y, p.damage] {
                m(&mut h, &v.to_le_bytes());
            }
        }
        match self.result {
            None => m(&mut h, &[0]),
            Some(GameResult::Draw) => m(&mut h, &[1]),
            Some(GameResult::Victory(a)) => m(&mut h, &[2, a]),
        }
        h
    }

    pub fn hash_hex(&self) -> String {
        format!("{:016x}", self.state_hash())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn ty(id: &str, role: Role, hp: i64, speed: i64, dmg: i64, range: i64) -> UnitType {
        UnitType {
            id: id.into(),
            name: id.into(),
            role,
            tier: 1,
            domain: if role.is_structure() {
                NavDomain::Land
            } else {
                NavDomain::Land
            },
            strategic_icon: default_strategic_icon(role, 1).into(),
            upgrades_to: None,
            max_hp: hp * SCALE,
            speed_per_tick: speed * SCALE / 10,
            elevation: 0,
            weapons: if dmg > 0 {
                vec![WeaponType {
                    label: "Gun".into(),
                    damage: dmg * SCALE,
                    range: range * SCALE,
                    reset: 10,
                    splash_radius: 0,
                    muzzle_velocity: 0,
                    anti_air: role == Role::AntiAir || role == Role::Commander,
                    anti_surface: role != Role::AntiAir,
                    turret_yaw_bone: None,
                    turret_pitch_bone: None,
                    turret_yaw_speed: 0,
                    firing_tolerance: 35,
                    beam: true,
                    lead_target: true,
                    emp: false,
                    kamikaze: false,
                    stun: None,
                }]
            } else {
                vec![]
            },
            cost_mass: 50 * SCALE,
            cost_energy: 250 * SCALE,
            build_work: 200,
            build_rate: if role == Role::Mobile || role == Role::PointDefense || role == Role::Radar {
                0
            } else if role == Role::Engineer {
                10
            } else {
                20
            },
            build_range: match role {
                Role::Commander => 10 * SCALE,
                Role::Engineer => 6 * SCALE,
                _ => 0,
            },
            mass_per_tick: match role {
                Role::Commander => 100,
                Role::Extractor => 200,
                _ => 0,
            },
            energy_per_tick: match role {
                Role::Commander => 2000,
                Role::Generator => 2000,
                Role::Hydro => 10_000,
                _ => 0,
            },
            storage_mass: if role == Role::Commander { 650 * SCALE } else if role == Role::MassStorage { 500 * SCALE } else { 0 },
            storage_energy: if role == Role::Commander { 4000 * SCALE } else if role == Role::EnergyStorage { 5000 * SCALE } else { 0 },
            vision_radius: 26 * SCALE,
            radar_radius: if role == Role::Radar { 115 * SCALE } else { 0 },
            shield_max_hp: if role == Role::Shield { 500 * SCALE } else { 0 },
            shield_radius: if role == Role::Shield { 15 * SCALE } else { 0 },
            shield_energy_upkeep: if role == Role::Shield { 10 * SCALE } else { 0 },
            shield_regen_per_tick: if role == Role::Shield { SCALE } else { 0 },
            vet_thresholds: [2, 5, 10, 20, 40],
            missile_cost_mass: if role == Role::MissileSilo { 10 * SCALE } else { 0 },
            missile_cost_energy: if role == Role::MissileSilo { 50 * SCALE } else { 0 },
            missile_damage: if role == Role::MissileSilo { 1000 * SCALE } else { 0 },
            missile_splash: if role == Role::MissileSilo { 4 * SCALE } else { 0 },
            missile_range: if role == Role::MissileSilo { 200 * SCALE } else { 0 },
            missile_stun: None,
            intercept_radius: if role == Role::MissileDefense { 30 * SCALE } else { 0 },
            radius: if role.is_structure() { 3 * SCALE } else { SCALE / 2 },
            size: [SCALE; 3],
            ext: UnitTypeExt {
                transport_class: if matches!(role, Role::Mobile | Role::Engineer | Role::Commander) { 1 } else { 0 },
                intercept_tactical: role == Role::MissileDefense,
                intercept_reset: if role == Role::MissileDefense { 20 } else { 0 },
                ..UnitTypeExt::default()
            },
        }
    }

    pub(crate) fn test_catalog() -> Catalog {
        let mut c = Catalog::default();
        for (k, ids, air_id, sh_id, fac2_id, t2u_id) in [
            ("red", ["RC", "RF", "RM", "RP", "RPD", "RRAD", "RENG", "RT", "RA"], "RAIR", "RSH", "RF2", "RT2"),
            ("blue", ["BC", "BF", "BM", "BP", "BPD", "BRAD", "BENG", "BT", "BA"], "BAIR", "BSH", "BF2", "BT2"),
        ] {
            let mut f1 = ty(ids[1], Role::Factory, 1000, 0, 0, 0);
            f1.upgrades_to = Some(fac2_id.into());
            c.types.insert(ids[0].into(), ty(ids[0], Role::Commander, 2000, 2, 20, 20));
            c.types.insert(ids[1].into(), f1);
            let mut f2 = ty(fac2_id, Role::Factory, 2200, 0, 0, 0);
            f2.tier = 2;
            c.types.insert(fac2_id.into(), f2);
            c.types.insert(ids[2].into(), ty(ids[2], Role::Extractor, 300, 0, 0, 0));
            c.types.insert(ids[3].into(), ty(ids[3], Role::Generator, 300, 0, 0, 0));
            c.types.insert(ids[4].into(), ty(ids[4], Role::PointDefense, 600, 0, 50, 26));
            c.types.insert(ids[5].into(), ty(ids[5], Role::Radar, 10, 0, 0, 0));
            c.types.insert(ids[6].into(), ty(ids[6], Role::Engineer, 150, 2, 0, 0));
            c.types.insert(ids[7].into(), ty(ids[7], Role::Mobile, 200, 4, 25, 16));
            let mut arty = ty(ids[8], Role::Mobile, 150, 3, 120, 30);
            arty.weapons[0].splash_radius = 2 * SCALE;
            arty.weapons[0].muzzle_velocity = 14 * SCALE / TICK_HZ as i64;
            arty.weapons[0].reset = 50;
            c.types.insert(ids[8].into(), arty);

            let mut air = ty(air_id, Role::Mobile, 180, 12, 30, 22);
            air.domain = NavDomain::Air;
            air.elevation = 16 * SCALE;
            air.weapons[0].anti_air = true;
            c.types.insert(air_id.into(), air);
            c.types.insert(sh_id.into(), ty(sh_id, Role::Shield, 400, 0, 0, 0));
            let mut t2 = ty(t2u_id, Role::Mobile, 500, 4, 60, 22);
            t2.tier = 2;
            c.types.insert(t2u_id.into(), t2);

            c.factions.push(Faction {
                key: k.into(),
                commander: ids[0].into(),
                factory: ids[1].into(),
                extractor: ids[2].into(),
                generator: ids[3].into(),
                point_defense: ids[4].into(),
                radar: ids[5].into(),
                engineer: ids[6].into(),
                land_units: vec![ids[7].into(), ids[8].into()],
                air_factory: String::new(),
                naval_factory: String::new(),
                hydro: String::new(),
                mass_storage: String::new(),
                energy_storage: String::new(),
                shield: sh_id.into(),
                anti_air: String::new(),
                tac_missile: String::new(),
                strat_missile: String::new(),
                tac_defense: String::new(),
                strat_defense: String::new(),
                experimental: String::new(),
                t2_land_factory: fac2_id.into(),
                t3_land_factory: String::new(),
                t2_mex: String::new(),
                t3_mex: String::new(),
                t2_pgen: String::new(),
                t3_pgen: String::new(),
                t3_shield: String::new(),
                t2_engineer: String::new(),
                t3_engineer: String::new(),
                t2_land_units: vec![t2u_id.into()],
                t3_land_units: vec![],
                air_units: vec![air_id.into()],
                naval_units: vec![],
            });
        }
        c
    }

    pub(crate) fn test_world(seed: u64) -> Skirmish {
        let grid = PathGrid::new(64, 64, 4 * SCALE);
        let mass = vec![(40 * SCALE, 40 * SCALE), (60 * SCALE, 40 * SCALE), (210 * SCALE, 210 * SCALE)];
        Skirmish::new(
            test_catalog(),
            grid,
            (256 * SCALE, 256 * SCALE),
            mass,
            &[
                ArmySetup { name: "A".into(), faction: 0, start: (50 * SCALE, 50 * SCALE), team: 0 },
                ArmySetup { name: "B".into(), faction: 1, start: (200 * SCALE, 200 * SCALE), team: 1 },
            ],
            seed,
        )
        .unwrap()
    }

    #[test]
    fn bubble_shield_absorbs_damage_and_upgrades_preserve_position() {
        let mut w = test_world(99);
        let sh = w.spawn_unit(0, "RSH", (60 * SCALE, 50 * SCALE)).unwrap();
        let defender = w.spawn_unit(0, "RF", (64 * SCALE, 50 * SCALE)).unwrap();
        let attacker = w.spawn_unit(1, "BT", (72 * SCALE, 50 * SCALE)).unwrap();
        w.rebuild_spatial();
        w.issue(1, Command::Attack { units: vec![attacker], target: defender, queue: false });
        w.run(30);
        assert!(w.unit(sh).unwrap().shield_hp < w.unit(sh).unwrap().shield_max_hp, "shield absorbs hits");
        assert_eq!(w.unit(defender).unwrap().hp, w.unit(defender).unwrap().max_hp, "shielded factory unscathed");

        w.issue(0, Command::Upgrade { unit: defender, target_type: "RF2".into() });
        w.run(260);
        let upg = w.unit(defender).unwrap();
        assert_eq!(upg.type_id, "RF2");
        assert_eq!(upg.tier, 2);
    }

    #[test]
    fn air_domain_bypasses_wall_and_large_group_uses_flowfield() {
        let mut grid = PathGrid::new(64, 64, 4 * SCALE);
        for z in 0..64 {
            grid.set_blocked(32, z, true);
        }
        let mut w = Skirmish::new(
            test_catalog(),
            grid,
            (256 * SCALE, 256 * SCALE),
            vec![],
            &[
                ArmySetup { name: "A".into(), faction: 0, start: (40 * SCALE, 40 * SCALE), team: 0 },
                ArmySetup { name: "B".into(), faction: 1, start: (220 * SCALE, 220 * SCALE), team: 1 },
            ],
            7,
        )
        .unwrap();
        let fighter = w.spawn_unit(0, "RAIR", (40 * SCALE, 100 * SCALE)).unwrap();
        w.issue(0, Command::Move { units: vec![fighter], x: 200 * SCALE, y: 100 * SCALE, queue: false });
        w.run(150);
        let f = w.unit(fighter).unwrap();
        assert!(f.pos.x > 160 * SCALE, "air unit flies cleanly over blocked wall");
        assert_eq!(f.altitude, 16 * SCALE);
    }

    /// Phase 28 fixture: open 256 m map (64×64 cells of 4 m) with a 1 m wall type `RWALL`.
    fn wall_world() -> Skirmish {
        let mut cat = test_catalog();
        let mut wall = ty("RWALL", Role::Wall, 4000, 0, 0, 0);
        wall.radius = SCALE / 2;
        cat.types.insert("RWALL".into(), wall);
        Skirmish::new(
            cat,
            PathGrid::new(64, 64, 4 * SCALE),
            (256 * SCALE, 256 * SCALE),
            vec![],
            &[
                ArmySetup { name: "A".into(), faction: 0, start: (20 * SCALE, 20 * SCALE), team: 0 },
                ArmySetup { name: "B".into(), faction: 1, start: (236 * SCALE, 20 * SCALE), team: 1 },
            ],
            11,
        )
        .unwrap()
    }

    #[test]
    fn wall_line_blocks_land_pathing_until_destroyed() {
        let mut w = wall_world();
        // A 1 m-spaced wall line along x = 128 m, from z = 40 m to z = 200 m.
        let mut walls = Vec::new();
        for z in 40..200 {
            walls.push(w.spawn_unit(0, "RWALL", (128 * SCALE + SCALE / 2, z * SCALE + SCALE / 2)).unwrap());
        }
        let tank = w.spawn_unit(0, "RT", (100 * SCALE, 120 * SCALE)).unwrap();
        w.step();
        assert_eq!(w.wall_cell_count(), 40, "160 sections stamp a 40-cell line");
        assert!(w.grid().is_blocked(32, 30));
        w.issue(0, Command::Move { units: vec![tank], x: 156 * SCALE, y: 120 * SCALE, queue: false });
        let mut max_detour = 0i64;
        for _ in 0..1200 {
            w.step();
            let u = w.unit(tank).unwrap();
            assert!(!w.is_wall_cell((u.pos.x, u.pos.y)), "tank entered a wall cell at {:?}", u.pos);
            max_detour = max_detour.max((u.pos.y - 120 * SCALE).abs());
            if u.orders.is_empty() {
                break;
            }
        }
        let u = w.unit(tank).unwrap();
        assert_eq!((u.pos.x, u.pos.y), (156 * SCALE, 120 * SCALE), "tank reaches the far side");
        assert!(max_detour >= 70 * SCALE, "tank walks around the wall end (detour {max_detour})");

        // Destroying the walls re-opens the grid and frees the cells.
        for id in &walls {
            w.units.get_mut(id).unwrap().hp = 0;
        }
        w.step();
        w.step();
        assert_eq!(w.wall_cell_count(), 0);
        assert!(!w.grid().is_blocked(32, 30));
        w.issue(0, Command::Move { units: vec![tank], x: 100 * SCALE, y: 120 * SCALE, queue: false });
        for _ in 0..200 {
            w.step();
            let u = w.unit(tank).unwrap();
            assert!((u.pos.y - 120 * SCALE).abs() <= 2 * SCALE, "straight line once walls are gone");
        }
    }

    #[test]
    fn wall_spawned_on_unit_evicts_it_and_large_groups_reroute() {
        let mut w = wall_world();
        let tank = w.spawn_unit(0, "RT", (50 * SCALE + 200, 50 * SCALE + 200)).unwrap();
        w.spawn_unit(0, "RWALL", (50 * SCALE + SCALE / 2, 50 * SCALE + SCALE / 2)).unwrap();
        w.step();
        let u = w.unit(tank).unwrap();
        assert!(!w.is_wall_cell((u.pos.x, u.pos.y)), "tank pushed off the new wall cell");

        // A flow-field group (>= 6 movers) also respects a wall line built mid-route.
        let mut w = wall_world();
        let group: Vec<UnitId> = (0..8)
            .map(|i| w.spawn_unit(0, "RT", (90 * SCALE + i * 2 * SCALE, 120 * SCALE)).unwrap())
            .collect();
        w.issue(0, Command::Move { units: group.clone(), x: 170 * SCALE, y: 120 * SCALE, queue: false });
        w.run(5);
        for z in 60..180 {
            w.spawn_unit(0, "RWALL", (130 * SCALE + SCALE / 2, z * SCALE + SCALE / 2)).unwrap();
        }
        for _ in 0..1500 {
            w.step();
            for id in &group {
                let u = w.unit(*id).unwrap();
                assert!(!w.is_wall_cell((u.pos.x, u.pos.y)), "group member {id} crossed the wall");
            }
        }
        assert!(group.iter().all(|id| w.unit(*id).unwrap().pos.x > 150 * SCALE), "group gets around the wall");
    }

    #[test]
    fn walls_are_last_resort_targets_and_do_not_hold_off_defeat() {
        let mut w = wall_world();
        let pd = w.spawn_unit(0, "RPD", (120 * SCALE, 150 * SCALE)).unwrap();
        let wall = w.spawn_unit(1, "RWALL", (124 * SCALE + SCALE / 2, 150 * SCALE + SCALE / 2)).unwrap();
        let tank = w.spawn_unit(1, "BT", (140 * SCALE, 150 * SCALE)).unwrap();
        w.run(20);
        assert!(w.unit(tank).is_none_or(|t| t.hp < t.max_hp), "PD shoots the tank first");
        assert_eq!(w.unit(wall).unwrap().hp, w.unit(wall).unwrap().max_hp, "wall ignored while a tank is in range");
        let _ = pd;

        // Annihilation: an army that only has walls left is defeated.
        let mut w = wall_world().with_options(GameOptions { victory: VictoryCondition::Annihilation, ..GameOptions::default() });
        w.spawn_unit(1, "RWALL", (200 * SCALE + SCALE / 2, 200 * SCALE + SCALE / 2)).unwrap();
        let blue_acu = w.commander_of(1).unwrap().id;
        w.units.get_mut(&blue_acu).unwrap().hp = 0;
        w.run(3);
        assert!(w.army(1).unwrap().defeated, "walls alone do not keep an army alive");
        assert_eq!(w.result(), Some(GameResult::Victory(0)));
    }

    #[test]
    fn walls_place_flush_but_other_structures_keep_their_margin() {
        let mut w = wall_world();
        w.spawn_unit(0, "RWALL", (100 * SCALE + SCALE / 2, 100 * SCALE + SCALE / 2)).unwrap();
        assert!(w.site_is_free("RWALL", 101 * SCALE + SCALE / 2, 100 * SCALE + SCALE / 2), "adjacent wall section");
        assert!(!w.site_is_free("RWALL", 100 * SCALE + SCALE / 2, 100 * SCALE + SCALE / 2), "same spot");
        // PD (3 m half-size) may touch a wall but not overlap it.
        assert!(w.site_is_free("RPD", 104 * SCALE + SCALE / 2, 100 * SCALE + SCALE / 2));
        assert!(!w.site_is_free("RPD", 103 * SCALE, 100 * SCALE + SCALE / 2));
        w.spawn_unit(0, "RPD", (60 * SCALE, 60 * SCALE)).unwrap();
        assert!(!w.site_is_free("RPD", 66 * SCALE + SCALE / 2, 60 * SCALE), "PDs keep the 1 m margin");
    }

    /// Phase 29 fixture: `wall_world` plus a static projectile gun `RGUN` (25 dmg, 26 m,
    /// 3 m/tick shells, fixed mount) and an optional 10 m ridge along x = 128 m.
    fn ballistic_world(ridge: bool, gun: impl Fn(&mut WeaponType)) -> Skirmish {
        let mut w = wall_world();
        let mut g = ty("RGUN", Role::Mobile, 500, 0, 25, 26);
        g.weapons[0].beam = false;
        g.weapons[0].muzzle_velocity = 3 * SCALE;
        gun(&mut g.weapons[0]);
        w.catalog.types.insert("RGUN".into(), g);
        if ridge {
            let n = 257usize;
            let mut h = vec![0i64; n * n];
            for z in 0..n {
                h[z * n + 128] = 10 * SCALE;
            }
            w.set_terrain(HeightField::from_samples(256, 256, SCALE, h));
        }
        w
    }

    #[test]
    fn ridge_soaks_shells_and_beams_but_open_ground_does_not() {
        for beam in [false, true] {
            for ridge in [true, false] {
                let mut w = ballistic_world(ridge, |wt| wt.beam = beam);
                w.spawn_unit(0, "RGUN", (116 * SCALE, 150 * SCALE)).unwrap();
                let target = w.spawn_unit(1, "BT", (139 * SCALE, 150 * SCALE)).unwrap();
                w.run(60);
                assert!(w.shots_fired() > 0, "gun fires (beam={beam}, ridge={ridge})");
                let t = w.unit(target);
                let hurt = t.is_none_or(|t| t.hp < t.max_hp);
                assert_eq!(hurt, !ridge, "beam={beam} ridge={ridge}: target hurt={hurt}");
            }
        }
    }

    #[test]
    fn turrets_slew_before_firing_within_tolerance() {
        // 50 mrad/tick: turning from +z (yaw 0) to face -z (yaw π) takes ~63 ticks.
        let mut w = ballistic_world(false, |wt| {
            wt.turret_yaw_speed = 50;
            wt.firing_tolerance = 35;
        });
        let gun = w.spawn_unit(0, "RGUN", (100 * SCALE, 150 * SCALE)).unwrap();
        w.spawn_unit(1, "BT", (100 * SCALE, 130 * SCALE)).unwrap();
        w.run(55);
        assert_eq!(w.shots_fired(), 0, "still slewing");
        assert!(combat::wrap_mrad(w.unit(gun).unwrap().weapons[0].yaw).abs() > 2_500, "turned most of the way");
        w.run(20);
        assert!(w.shots_fired() > 0, "fires once aimed");
    }

    #[test]
    fn shells_hit_the_first_enemy_on_their_path_and_lead_movers() {
        let mut w = ballistic_world(false, |_| {});
        w.spawn_unit(0, "RGUN", (100 * SCALE, 150 * SCALE)).unwrap();
        let wall = w.spawn_unit(1, "RWALL", (110 * SCALE, 150 * SCALE)).unwrap();
        let tank = w.spawn_unit(1, "BT", (120 * SCALE, 150 * SCALE)).unwrap();
        w.run(40);
        let wl = w.unit(wall).unwrap();
        assert!(wl.hp < wl.max_hp, "wall in the line of fire soaks the shells");
        let t = w.unit(tank).unwrap();
        assert_eq!(t.hp, t.max_hp, "tank behind the wall is untouched");

        // A crossing target is still hit thanks to target leading; without leading the
        // slow shells land behind it.
        let mut hits = [0i64; 2];
        for (i, lead) in [true, false].into_iter().enumerate() {
            let mut w = ballistic_world(false, |wt| {
                wt.lead_target = lead;
                wt.muzzle_velocity = SCALE;
            });
            w.spawn_unit(0, "RGUN", (100 * SCALE, 100 * SCALE)).unwrap();
            let runner = w.spawn_unit(1, "BT", (120 * SCALE, 85 * SCALE)).unwrap();
            w.issue(1, Command::Move { units: vec![runner], x: 120 * SCALE, y: 200 * SCALE, queue: false });
            w.run(70);
            hits[i] = w.unit(runner).map_or(i64::MAX, |u| u.max_hp - u.hp);
        }
        assert!(hits[0] > hits[1], "leading lands more damage ({} vs {})", hits[0], hits[1]);
    }

    #[test]
    fn structures_block_pathing_and_factories_roll_units_off() {
        let mut w = wall_world();
        let fac = w.spawn_unit(0, "RF", (128 * SCALE, 120 * SCALE)).unwrap();
        w.step();
        assert!(w.is_wall_cell((128 * SCALE, 120 * SCALE)), "factory footprint blocks its centre cell");
        let tank = w.spawn_unit(0, "RT", (110 * SCALE, 120 * SCALE)).unwrap();
        w.issue(0, Command::Move { units: vec![tank], x: 146 * SCALE, y: 120 * SCALE, queue: false });
        for _ in 0..400 {
            w.step();
            let u = w.unit(tank).unwrap();
            assert!(!w.is_wall_cell((u.pos.x, u.pos.y)), "tank walked into the factory at {:?}", u.pos);
        }
        assert_eq!(w.unit(tank).unwrap().pos, Vec2 { x: 146 * SCALE, y: 120 * SCALE });

        // Wall the factory's exit side; the produced unit still rolls off onto open ground.
        for x in 120..137 {
            w.spawn_unit(0, "RWALL", (x * SCALE + SCALE / 2, 125 * SCALE + SCALE / 2)).unwrap();
        }
        w.armies[0].mass = 10_000 * SCALE;
        w.armies[0].energy = 100_000 * SCALE;
        w.issue(0, Command::Produce { factory: fac, unit_type: "RT".into(), count: 1 });
        let before: Vec<UnitId> = w.units().map(|u| u.id).collect();
        w.run(60);
        let fresh: Vec<&Unit> = w.units().filter(|u| !before.contains(&u.id) && u.role == Role::Mobile).collect();
        assert_eq!(fresh.len(), 1, "one tank produced");
        assert!(!w.is_wall_cell((fresh[0].pos.x, fresh[0].pos.y)), "rolled off onto an open cell");
    }

    /// Phase 30 fixture: `wall_world` with a 10 s-fuel fighter (`RAIR`, 12 m/s) and an air
    /// staging platform `RSTAGE` (2 slots).
    fn staging_world() -> Skirmish {
        let mut w = wall_world();
        let air = w.catalog.types.get_mut("RAIR").unwrap();
        air.ext.fuel_ticks = 100;
        air.ext.fuel_recharge_milli = 5_000;
        let mut st = ty("RSTAGE", Role::AirStaging, 500, 0, 0, 0);
        st.ext.staging = Some(StagingSpec {
            slots: 2,
            refuel_multiplier: 2,
            repair_permille: 100,
            repair_energy: 5 * SCALE,
            repair_mass: SCALE / 2,
            scan_radius: 300 * SCALE,
        });
        w.catalog.types.insert("RSTAGE".into(), st);
        w.armies[0].mass = 1_000 * SCALE;
        w.armies[0].energy = 10_000 * SCALE;
        w
    }

    #[test]
    fn aircraft_burn_fuel_and_fly_at_half_speed_when_empty() {
        let mut w = staging_world();
        let f = w.spawn_unit(0, "RAIR", (40 * SCALE, 120 * SCALE)).unwrap();
        w.issue(0, Command::SetAutoRefuel { units: vec![f], enabled: false });
        w.issue(0, Command::Patrol { units: vec![f], x: 220 * SCALE, y: 120 * SCALE, queue: false });
        let full = w.unit(f).unwrap().speed;
        w.run(50);
        assert_eq!(w.unit(f).unwrap().fuel, 50);
        assert_eq!(w.unit(f).unwrap().speed, full);
        w.run(60);
        let u = w.unit(f).unwrap();
        assert_eq!(u.fuel, 0);
        assert_eq!(u.speed, full / 2, "out of fuel: half speed");
    }

    #[test]
    fn low_fuel_aircraft_dock_refuel_repair_and_resume() {
        let mut w = staging_world();
        let st = w.spawn_unit(0, "RSTAGE", (60 * SCALE, 60 * SCALE)).unwrap();
        let f = w.spawn_unit(0, "RAIR", (60 * SCALE, 90 * SCALE)).unwrap();
        w.issue(0, Command::Move { units: vec![f], x: 200 * SCALE, y: 200 * SCALE, queue: false });
        w.issue(0, Command::Move { units: vec![f], x: 60 * SCALE, y: 200 * SCALE, queue: true });
        w.step();
        w.units.get_mut(&f).unwrap().hp /= 2;
        let energy0 = w.armies[0].energy;
        let (mut docked, mut ticks) = (false, 0);
        while ticks < 400 {
            w.step();
            ticks += 1;
            let u = w.unit(f).unwrap();
            if u.carried_by == Some(st) {
                docked = true;
                assert_eq!(u.pos, w.unit(st).unwrap().pos);
            } else if docked {
                break;
            }
        }
        assert!(docked, "low-fuel fighter docked at the platform");
        let u = w.unit(f).unwrap();
        assert_eq!((u.fuel, u.hp), (100, u.max_hp), "undocks refuelled and repaired");
        assert!(w.armies[0].energy < energy0, "repair costs energy");
        assert!(matches!(u.orders.front(), Some(UnitOrder::Move { .. })), "resumes its orders: {:?}", u.orders);
    }

    #[test]
    fn staging_slots_are_limited_and_dead_platforms_release_aircraft() {
        let mut w = staging_world();
        let st = w.spawn_unit(0, "RSTAGE", (60 * SCALE, 60 * SCALE)).unwrap();
        let fs: Vec<UnitId> = (0..3).map(|i| w.spawn_unit(0, "RAIR", (60 * SCALE + i * SCALE, 70 * SCALE)).unwrap()).collect();
        for &f in &fs {
            w.units.get_mut(&f).unwrap().hp = 10 * SCALE;
        }
        w.issue(0, Command::Dock { units: fs.clone(), staging: st, queue: false });
        w.run(10);
        let docked = fs.iter().filter(|f| w.unit(**f).unwrap().carried_by == Some(st)).count();
        assert_eq!(docked, 2, "only two docking slots");
        assert!(!w.apply_command_direct(1, Command::Dock { units: vec![], staging: st, queue: false }), "enemy cannot dock");

        w.units.get_mut(&st).unwrap().hp = 0;
        w.run(2);
        for &f in &fs {
            let u = w.unit(f).unwrap();
            assert!(u.alive && u.carried_by.is_none(), "aircraft released when the platform dies");
            assert!(!matches!(u.orders.front(), Some(UnitOrder::Dock(_))));
        }
    }

    /// Phase 31 fixture: a Hive-like station `RHIVE` (works itself, 15 m radius) and a
    /// Kennel-like `RKEN` with one pod drone `RPOD` (upgrades to `RKEN2` with two pods).
    fn station_world() -> Skirmish {
        let mut w = wall_world();
        let mut hive = ty("RHIVE", Role::EngStation, 500, 0, 0, 0);
        hive.build_rate = 20;
        hive.build_range = 15 * SCALE;
        hive.radius = SCALE / 2;
        let mut pod = ty("RPOD", Role::Engineer, 6, 10, 0, 0);
        pod.domain = NavDomain::Air;
        pod.elevation = SCALE;
        pod.build_rate = 20;
        pod.build_range = 4 * SCALE;
        pod.cost_mass = 25 * SCALE;
        pod.build_work = 800;
        let mut ken = hive.clone();
        ken.id = "RKEN".into();
        ken.ext.pods = vec!["RPOD".into()];
        ken.upgrades_to = Some("RKEN2".into());
        let mut ken2 = ken.clone();
        ken2.id = "RKEN2".into();
        ken2.tier = 3;
        ken2.upgrades_to = None;
        ken2.ext.pods = vec!["RPOD".into(), "RPOD".into()];
        for t in [hive, pod, ken, ken2] {
            w.catalog.types.insert(t.id.clone(), t);
        }
        w.catalog.types.get_mut("RT").unwrap().build_work = 2_000;
        w.armies[0].mass = 5_000 * SCALE;
        w.armies[0].energy = 50_000 * SCALE;
        w
    }

    /// Ticks until factory `fac` (with `RT` queued) rolls out a new unit.
    fn ticks_to_produce(w: &mut Skirmish, fac: UnitId) -> u64 {
        let before: Vec<UnitId> = w.units().map(|u| u.id).collect();
        w.issue(0, Command::Produce { factory: fac, unit_type: "RT".into(), count: 1 });
        for t in 1..=600 {
            w.step();
            if w.units().any(|u| !before.contains(&u.id) && u.role == Role::Mobile) {
                return t;
            }
        }
        panic!("factory never finished");
    }

    #[test]
    fn hive_assists_construction_and_repairs_in_its_radius() {
        let mut alone = station_world();
        let f1 = alone.spawn_unit(0, "RF", (100 * SCALE, 100 * SCALE)).unwrap();
        let solo = ticks_to_produce(&mut alone, f1);

        let mut w = station_world();
        let fac = w.spawn_unit(0, "RF", (100 * SCALE, 100 * SCALE)).unwrap();
        let hive = w.spawn_unit(0, "RHIVE", (108 * SCALE, 100 * SCALE)).unwrap();
        let helped = ticks_to_produce(&mut w, fac);
        assert!(helped * 10 < solo * 8, "hive speeds up production ({helped} vs {solo} ticks)");

        // Repairs the nearest damaged ally in range, ignores one outside it.
        let near = w.spawn_unit(0, "RPD", (112 * SCALE, 110 * SCALE)).unwrap();
        let far = w.spawn_unit(0, "RPD", (160 * SCALE, 100 * SCALE)).unwrap();
        for id in [near, far] {
            let u = w.units.get_mut(&id).unwrap();
            u.hp = u.max_hp / 2;
        }
        w.run(200);
        assert_eq!(w.unit(near).unwrap().hp, w.unit(near).unwrap().max_hp, "near PD repaired");
        assert_eq!(w.unit(far).unwrap().hp, w.unit(far).unwrap().max_hp / 2, "far PD untouched");
        assert!(w.unit(hive).unwrap().pos == Vec2 { x: 108 * SCALE, y: 100 * SCALE }, "hive never moves");
    }

    #[test]
    fn kennel_pods_fly_out_to_work_return_rebuild_and_die_with_the_station() {
        let mut w = station_world();
        let ken = w.spawn_unit(0, "RKEN", (100 * SCALE, 100 * SCALE)).unwrap();
        let pods = w.pods_of(ken);
        assert_eq!(pods.len(), 1, "station spawns with its pod");
        let pod = pods[0];
        assert_eq!(w.unit(pod).unwrap().drone_of, Some(ken));

        let hurt = w.spawn_unit(0, "RPD", (110 * SCALE, 108 * SCALE)).unwrap();
        w.units.get_mut(&hurt).unwrap().hp = 100 * SCALE;
        let mut flew = false;
        for _ in 0..300 {
            w.step();
            let p = w.unit(pod).unwrap();
            flew |= dist_sq((p.pos.x, p.pos.y), (100 * SCALE, 100 * SCALE)) > (4 * SCALE as i128).pow(2);
        }
        assert!(flew, "pod flew out to the damaged PD");
        assert_eq!(w.unit(hurt).unwrap().hp, w.unit(hurt).unwrap().max_hp, "pod repaired it");
        let p = w.unit(pod).unwrap();
        assert!(dist_sq((p.pos.x, p.pos.y), (100 * SCALE, 100 * SCALE)) <= (2 * SCALE as i128).pow(2), "pod returned home");

        // A destroyed pod is rebuilt (paid), the upgrade adds a second one.
        w.units.get_mut(&pod).unwrap().hp = 0;
        let mass0 = w.armies[0].mass;
        w.run(30);
        assert!(w.pods_of(ken).is_empty(), "rebuild takes time");
        w.run(20);
        assert_eq!(w.pods_of(ken).len(), 1, "pod rebuilt");
        assert!(w.armies[0].mass <= mass0 - 25 * SCALE + 100 * SCALE, "rebuild paid");
        w.issue(0, Command::Upgrade { unit: ken, target_type: "RKEN2".into() });
        w.run(400);
        assert_eq!(w.unit(ken).unwrap().type_id, "RKEN2");
        assert_eq!(w.pods_of(ken).len(), 2, "upgraded station keeps two pods");

        let all = w.pods_of(ken);
        w.units.get_mut(&ken).unwrap().hp = 0;
        w.run(3);
        assert!(all.iter().all(|p| w.unit(*p).is_none()), "pods die with their station");
    }

    /// Phase 32 fixture: a Fatboy-like mobile factory + air staging `RFAT`, a Megalith-like
    /// mobile builder `RMEGA` and an egg `REGG` that hatches into an engineer.
    fn mobile_factory_world() -> Skirmish {
        let mut w = staging_world();
        let mut fat = ty("RFAT", Role::Experimental, 5_000, 2, 0, 0);
        fat.build_rate = 40;
        fat.radius = 3 * SCALE;
        fat.ext.mobile_factory = true;
        fat.ext.staging = Some(StagingSpec { slots: 2, refuel_multiplier: 1, repair_permille: 0, repair_energy: 0, repair_mass: 0, scan_radius: 300 * SCALE });
        let mut mega = ty("RMEGA", Role::Experimental, 5_000, 2, 0, 0);
        mega.build_rate = 40;
        mega.build_range = SCALE;
        mega.ext.mobile_builder = true;
        let mut egg = ty("REGG", Role::Factory, 10, 0, 0, 0);
        egg.radius = SCALE / 2;
        egg.ext.hatch = Some("RENG".into());
        for t in [fat, mega, egg] {
            w.catalog.types.insert(t.id.clone(), t);
        }
        w.catalog.extra_buildable.resize(2, Vec::new());
        w.catalog.extra_buildable[0].push("REGG".into());
        w
    }

    #[test]
    fn mobile_factories_produce_on_the_move_and_dock_aircraft() {
        let mut w = mobile_factory_world();
        let fat = w.spawn_unit(0, "RFAT", (60 * SCALE, 60 * SCALE)).unwrap();
        assert!(w.is_producer(w.unit(fat).unwrap()));
        w.issue(0, Command::Move { units: vec![fat], x: 160 * SCALE, y: 60 * SCALE, queue: false });
        w.issue(0, Command::Produce { factory: fat, unit_type: "RT".into(), count: 1 });
        let before: Vec<UnitId> = w.units().map(|u| u.id).collect();
        let mut made = None;
        for _ in 0..100 {
            w.step();
            if let Some(u) = w.units().find(|u| !before.contains(&u.id) && u.type_id == "RT") {
                made = Some((u.pos, w.unit(fat).unwrap().pos));
                break;
            }
        }
        let (tank, fat_pos) = made.expect("the moving factory produced a tank");
        assert!(fat_pos.x > 60 * SCALE, "it was moving");
        assert!(dist_sq((tank.x, tank.y), (fat_pos.x, fat_pos.y)) <= (8 * SCALE as i128).pow(2), "tank rolls off next to it");

        // Mobile air staging: a docked fighter rides along.
        let f = w.spawn_unit(0, "RAIR", (fat_pos.x, fat_pos.y + 5 * SCALE)).unwrap();
        w.units.get_mut(&f).unwrap().hp = SCALE;
        w.issue(0, Command::Dock { units: vec![f], staging: fat, queue: false });
        w.run(5);
        assert_eq!(w.unit(f).unwrap().carried_by, Some(fat));
        w.run(10);
        assert_eq!(w.unit(f).unwrap().pos, w.unit(fat).unwrap().pos, "docked fighter follows the platform");
    }

    #[test]
    fn mobile_builders_lay_eggs_that_hatch_into_units() {
        let mut w = mobile_factory_world();
        let mega = w.spawn_unit(0, "RMEGA", (80 * SCALE, 80 * SCALE)).unwrap();
        assert!(w.is_constructor(w.unit(mega).unwrap()));
        assert!(w.apply_command_direct(0, Command::Build { builder: mega, unit_type: "REGG".into(), x: 84 * SCALE, y: 80 * SCALE, queue: false }));
        w.run(40);
        assert!(w.units().all(|u| u.type_id != "REGG"), "the egg never lingers as a structure");
        let hatched: Vec<&Unit> = w.units().filter(|u| u.type_id == "RENG" && u.army == 0).collect();
        assert_eq!(hatched.len(), 1, "egg hatched into an engineer");
        assert!(dist_sq((hatched[0].pos.x, hatched[0].pos.y), (84 * SCALE, 80 * SCALE)) <= (2 * SCALE as i128).pow(2));
    }

    #[test]
    fn emp_hits_stun_targets_and_they_cannot_move_or_fire() {
        let mut w = wall_world();
        let mut emp = ty("REMP", Role::Mobile, 500, 0, 5, 20);
        emp.weapons[0].stun = Some((30, 3 * SCALE));
        emp.weapons[0].reset = 200;
        w.catalog.types.insert("REMP".into(), emp);
        w.spawn_unit(0, "REMP", (100 * SCALE, 100 * SCALE)).unwrap();
        let tank = w.spawn_unit(1, "BT", (115 * SCALE, 100 * SCALE)).unwrap();
        w.issue(1, Command::Move { units: vec![tank], x: 115 * SCALE, y: 200 * SCALE, queue: false });
        w.step();
        let stunned = w.unit(tank).unwrap().stunned;
        assert!(stunned > 20, "EMP hit stuns the tank ({stunned})");
        let p0 = w.unit(tank).unwrap().pos;
        let shots0 = w.shots_fired();
        w.run(20);
        assert_eq!(w.unit(tank).unwrap().pos, p0, "stunned tank does not move");
        assert_eq!(w.shots_fired(), shots0, "nobody fires: the emitter is reloading, the tank is stunned");
        w.run(20);
        assert!(w.unit(tank).unwrap().pos.y > p0.y, "moves again once the stun wears off");
    }

    #[test]
    fn death_weapons_hit_friend_and_foe_and_kamikazes_detonate_once() {
        let mut w = wall_world();
        let mut boom_t = ty("RBOOM", Role::Generator, 100, 0, 0, 0);
        boom_t.ext.death_damage = 100 * SCALE;
        boom_t.ext.death_radius = 6 * SCALE;
        w.catalog.types.insert("RBOOM".into(), boom_t);
        let boom = w.spawn_unit(0, "RBOOM", (100 * SCALE, 100 * SCALE)).unwrap();
        let own = w.spawn_unit(0, "RENG", (104 * SCALE, 100 * SCALE)).unwrap();
        let foe = w.spawn_unit(1, "BENG", (100 * SCALE, 104 * SCALE)).unwrap();
        let far = w.spawn_unit(1, "BENG", (100 * SCALE, 120 * SCALE)).unwrap();
        w.units.get_mut(&boom).unwrap().hp = 0;
        w.step();
        let dmg = |w: &Skirmish, id| w.unit(id).map_or(i64::MAX, |u| u.max_hp - u.hp);
        assert_eq!(dmg(&w, own), 100 * SCALE, "friendly unit caught in the blast");
        assert_eq!(dmg(&w, foe), 100 * SCALE, "enemy unit caught in the blast");
        assert_eq!(dmg(&w, far), 0, "outside the radius");

        let mut w = wall_world();
        let mut beetle = ty("RBEETLE", Role::Mobile, 100, 4, 400, 3);
        beetle.weapons[0].kamikaze = true;
        beetle.weapons[0].splash_radius = 3 * SCALE;
        beetle.weapons[0].reset = 1;
        w.catalog.types.insert("RBEETLE".into(), beetle);
        let b = w.spawn_unit(0, "RBEETLE", (100 * SCALE, 100 * SCALE)).unwrap();
        let victim = w.spawn_unit(1, "BPD", (110 * SCALE, 100 * SCALE)).unwrap();
        w.issue(0, Command::Attack { units: vec![b], target: victim, queue: false });
        w.run(40);
        assert!(w.unit(b).is_none(), "the kamikaze died in its own blast");
        let v = w.unit(victim).unwrap();
        assert_eq!(v.max_hp - v.hp, 400 * SCALE, "exactly one detonation");
    }

    #[test]
    fn orbital_centre_launches_an_untargetable_satellite_that_dies_with_it() {
        let mut w = wall_world();
        let mut sat = ty("RSAT", Role::Mobile, 100, 10, 75, 30);
        sat.domain = NavDomain::Air;
        sat.elevation = 25 * SCALE;
        sat.ext.untargetable = true;
        let mut novax = ty("RNOVAX", Role::Orbital, 9_000, 0, 0, 0);
        novax.ext.satellite = Some("RSAT".into());
        w.catalog.types.insert("RSAT".into(), sat);
        w.catalog.types.insert("RNOVAX".into(), novax);
        let centre = w.spawn_unit(0, "RNOVAX", (60 * SCALE, 60 * SCALE)).unwrap();
        let sats = w.pods_of(centre);
        assert_eq!(sats.len(), 1, "satellite launched");
        let s = sats[0];
        assert!(w.unit(s).unwrap().untargetable);

        // Enemy AA right below never engages it, and explicit attacks are refused.
        let aa = w.spawn_unit(1, "BAIR", (60 * SCALE, 60 * SCALE)).unwrap();
        w.units.get_mut(&aa).unwrap().speed = 0;
        let hp0 = w.unit(s).unwrap().hp;
        w.issue(1, Command::Attack { units: vec![aa], target: s, queue: false });
        w.run(30);
        assert_eq!(w.unit(s).unwrap().hp, hp0, "satellite untouched");

        w.units.get_mut(&centre).unwrap().hp = 0;
        w.run(3);
        assert!(w.unit(s).is_none(), "satellite falls with its launch centre");
    }

    #[test]
    fn engineers_reclaim_trees_and_rocks_for_mass_and_energy_deterministically() {
        let run = |reclaim: bool| {
            let mut w = wall_world();
            w.set_props(vec![
                MapProp { pos: (100 * SCALE, 100 * SCALE), mass: 4 * SCALE, energy: 40 * SCALE, src_xz: (1, 2) },
                MapProp { pos: (104 * SCALE, 100 * SCALE), mass: 5 * SCALE, energy: 0, src_xz: (3, 4) },
                MapProp { pos: (200 * SCALE, 200 * SCALE), mass: 5 * SCALE, energy: 0, src_xz: (5, 6) },
            ]);
            w.armies[0].mass = 0;
            w.armies[0].energy = 0;
            let eng = w.spawn_unit(0, "RENG", (90 * SCALE, 100 * SCALE)).unwrap();
            assert_eq!(w.prop_at((101 * SCALE, 100 * SCALE), PROP_SNAP).map(|p| p.0), Some(0));
            let (m0, e0) = (w.armies[0].mass, w.armies[0].energy);
            if reclaim {
                w.issue(0, Command::Reclaim { units: vec![eng], x: 100 * SCALE, y: 100 * SCALE, queue: false });
                w.issue(0, Command::Reclaim { units: vec![eng], x: 104 * SCALE, y: 100 * SCALE, queue: true });
            }
            w.run(120);
            (w.armies[0].mass - m0, w.armies[0].energy - e0, w.props().count(), w.props_gone().to_vec(), w.state_hash())
        };
        let (dm, de, left, gone, h1) = run(true);
        let (cm, ce, ..) = run(false);
        assert_eq!(dm - cm, 9 * SCALE, "exactly both props' mass on top of income");
        assert_eq!(de - ce, 40 * SCALE, "exactly the tree's energy on top of income");
        assert_eq!(left, 1, "the far rock still stands");
        assert_eq!(gone, vec![(1, 2), (3, 4)], "reclaimed props reported in order for the renderer");
        assert_eq!(run(true).4, h1, "prop reclaim is deterministic");
    }

    #[test]
    fn engineers_reclaim_live_walls_for_their_mass_cost() {
        let mut w = wall_world();
        w.catalog.types.get_mut("RWALL").unwrap().cost_mass = 20 * SCALE;
        let wall = w.spawn_unit(0, "RWALL", (100 * SCALE + SCALE / 2, 100 * SCALE + SCALE / 2)).unwrap();
        let eng = w.spawn_unit(0, "RENG", (94 * SCALE, 100 * SCALE)).unwrap();
        w.step();
        assert!(w.is_wall_cell((100 * SCALE, 100 * SCALE)));
        w.armies[0].mass = 0;
        let mut control = w.clone();
        w.issue(0, Command::Reclaim { units: vec![eng], x: 100 * SCALE + SCALE / 2, y: 100 * SCALE + SCALE / 2, queue: false });
        w.run(30);
        control.run(30);
        assert!(w.unit(wall).is_none(), "wall reclaimed away");
        assert!(w.wrecks().next().is_none(), "reclaimed units leave no wreck");
        assert_eq!(w.armies[0].mass - control.armies[0].mass, 20 * SCALE, "refunds the wall's mass cost");
        assert!(!w.is_wall_cell((100 * SCALE, 100 * SCALE)), "path grid re-opened");
        assert!(w.unit(eng).unwrap().orders.is_empty());
    }

    #[test]
    fn fire_states_hold_return_and_fire_at_will() {
        let setup = |a_state: FireState, b_state: FireState| {
            let mut w = wall_world();
            let a = w.spawn_unit(0, "RT", (100 * SCALE, 100 * SCALE)).unwrap();
            let b = w.spawn_unit(1, "BT", (112 * SCALE, 100 * SCALE)).unwrap();
            w.issue(0, Command::SetFireState { units: vec![a], state: a_state });
            w.issue(1, Command::SetFireState { units: vec![b], state: b_state });
            w.step();
            (w, a, b)
        };
        let hurt = |w: &Skirmish, id: UnitId| w.unit(id).is_none_or(|u| u.hp < u.max_hp);
        let (mut w, a, b) = setup(FireState::HoldFire, FireState::HoldFire);
        w.run(30);
        assert!(!hurt(&w, a) && !hurt(&w, b), "both holding fire");
        let (mut w, a, b) = setup(FireState::ReturnFire, FireState::HoldFire);
        w.run(30);
        assert!(!hurt(&w, b), "return-fire waits to be shot at");
        w.issue(1, Command::Attack { units: vec![b], target: a, queue: false });
        w.run(30);
        assert!(hurt(&w, a) && hurt(&w, b), "attacked, it shoots back");
        let (mut w, _a, b) = setup(FireState::FireAtWill, FireState::HoldFire);
        w.run(30);
        assert!(hurt(&w, b), "fire at will engages on its own");
    }

    #[test]
    fn ground_fire_hits_what_stands_at_the_point_even_when_holding_fire() {
        let mut w = wall_world();
        let arty = w.spawn_unit(0, "RA", (100 * SCALE, 100 * SCALE)).unwrap();
        let target = w.spawn_unit(1, "BENG", (120 * SCALE, 100 * SCALE)).unwrap();
        w.issue(0, Command::SetFireState { units: vec![arty], state: FireState::HoldFire });
        w.run(10);
        assert_eq!(w.shots_fired(), 0);
        w.issue(0, Command::AttackGround { units: vec![arty], x: 120 * SCALE, y: 100 * SCALE, queue: false });
        w.run(40);
        assert!(w.shots_fired() >= 1, "ground fire shoots");
        assert!(w.unit(target).is_none_or(|u| u.hp < u.max_hp), "the shell lands on the unit at the point");
        assert!(matches!(w.unit(arty).unwrap().orders.front(), Some(UnitOrder::AttackGround { .. })), "ground fire persists");
    }

    #[test]
    fn guards_follow_and_defend_and_paused_factories_stop() {
        let mut w = wall_world();
        let vip = w.spawn_unit(0, "RENG", (60 * SCALE, 60 * SCALE)).unwrap();
        let guard = w.spawn_unit(0, "RT", (50 * SCALE, 60 * SCALE)).unwrap();
        w.issue(0, Command::Guard { units: vec![guard], target: vip, queue: false });
        w.issue(0, Command::Move { units: vec![vip], x: 140 * SCALE, y: 60 * SCALE, queue: false });
        w.run(250);
        let (g, v) = (w.unit(guard).unwrap().pos, w.unit(vip).unwrap().pos);
        assert!(dist_sq((g.x, g.y), (v.x, v.y)) <= (12 * SCALE as i128).pow(2), "guard kept up ({g:?} vs {v:?})");
        let raider = w.spawn_unit(1, "BENG", (v.x + 10 * SCALE, v.y)).unwrap();
        w.run(60);
        assert!(w.unit(raider).is_none_or(|u| u.hp < u.max_hp), "guard engaged the threat");

        let mut w = wall_world();
        let fac = w.spawn_unit(0, "RF", (100 * SCALE, 100 * SCALE)).unwrap();
        w.catalog.types.get_mut("RT").unwrap().build_work = 2_000;
        w.issue(0, Command::Produce { factory: fac, unit_type: "RT".into(), count: 1 });
        w.run(20);
        let p1 = w.unit(fac).unwrap().build.as_ref().unwrap().project.progress;
        w.issue(0, Command::SetBuildPaused { units: vec![fac], paused: true });
        w.run(30);
        let p2 = w.unit(fac).unwrap().build.as_ref().unwrap().project.progress;
        assert!(p2 <= p1 + 20, "paused factory makes no progress ({p1} -> {p2})");
        w.issue(0, Command::SetBuildPaused { units: vec![fac], paused: false });
        w.run(20);
        assert!(w.unit(fac).unwrap().build.as_ref().is_none_or(|b| b.project.progress > p2), "resumes");
    }

    #[test]
    fn ferries_shuttle_idle_land_units_between_two_points() {
        let mut w = wall_world();
        let mut tr = ty("RTRANS", Role::Mobile, 300, 10, 0, 0);
        tr.domain = NavDomain::Air;
        tr.elevation = 8 * SCALE;
        tr.ext.transport = Some(TransportSpec { small: 6, medium: 2, large: 1, class2_size: 2, class3_size: 4, carries_commander: false });
        w.catalog.types.insert("RTRANS".into(), tr);
        let t = w.spawn_unit(0, "RTRANS", (40 * SCALE, 40 * SCALE)).unwrap();
        let pax: Vec<UnitId> = (0..3).map(|i| w.spawn_unit(0, "RT", (60 * SCALE + i * 2 * SCALE, 60 * SCALE)).unwrap()).collect();
        let busy = w.spawn_unit(0, "RT", (61 * SCALE, 62 * SCALE)).unwrap();
        w.issue(0, Command::Move { units: vec![busy], x: 30 * SCALE, y: 200 * SCALE, queue: false });
        w.issue(0, Command::Ferry { transports: vec![t], from: (60 * SCALE, 60 * SCALE), to: (180 * SCALE, 160 * SCALE), queue: false });
        let mut delivered = false;
        for _ in 0..400 {
            w.step();
            if pax.iter().all(|p| {
                let u = w.unit(*p).unwrap();
                u.carried_by.is_none() && dist_sq((u.pos.x, u.pos.y), (180 * SCALE, 160 * SCALE)) <= (10 * SCALE as i128).pow(2)
            }) {
                delivered = true;
                break;
            }
        }
        assert!(delivered, "all idle units at the pick-up point were ferried across");
        assert!(w.unit(busy).unwrap().pos.y > 70 * SCALE, "a unit with its own orders was not picked up");
        assert!(matches!(w.unit(t).unwrap().orders.front(), Some(UnitOrder::Ferry { .. })), "the route repeats");
    }

    #[test]
    fn isqrt_exact() {
        for v in [0i128, 1, 2, 3, 4, 15, 16, 17, 999_999, 1_000_000, (1i128 << 62) + 7] {
            let r = isqrt(v) as i128;
            assert!(r * r <= v && (r + 1) * (r + 1) > v, "v={v}");
        }
    }

    #[test]
    fn starts_with_full_storage_and_commanders() {
        let w = test_world(1);
        assert_eq!(w.units().count(), 2);
        assert_eq!(w.armies()[0].mass, 650 * SCALE);
        assert_eq!(w.armies()[0].energy, 4000 * SCALE);
        assert!(w.commander_of(0).is_some() && w.commander_of(1).is_some());
    }

    #[test]
    fn commander_builds_factory_and_factory_produces() {
        let mut w = test_world(2);
        let acu = w.commander_of(0).unwrap().id;
        w.issue(0, Command::Build { builder: acu, unit_type: "RF".into(), x: 70 * SCALE, y: 50 * SCALE, queue: false });
        w.run(400);
        let fac = w.units().find(|u| u.role == Role::Factory && u.army == 0).map(|u| u.id);
        let fac = fac.expect("factory built");
        w.issue(0, Command::Produce { factory: fac, unit_type: "RT".into(), count: 2 });
        w.run(300);
        let tanks = w.units().filter(|u| u.army == 0 && u.role == Role::Mobile).count();
        assert_eq!(tanks, 2);
        assert_eq!(w.armies()[0].units_built, 3);
    }

    #[test]
    fn extractor_snaps_to_mass_and_adds_income() {
        let mut w = test_world(3);
        let acu = w.commander_of(0).unwrap().id;
        w.issue(0, Command::Build { builder: acu, unit_type: "RM".into(), x: 41 * SCALE, y: 39 * SCALE, queue: false });
        w.run(200);
        let mex = w.units().find(|u| u.role == Role::Extractor).expect("mex");
        assert_eq!((mex.pos.x, mex.pos.y), (40 * SCALE, 40 * SCALE));
        assert_eq!(w.mass_claim(0), Some(mex.id));
        assert_eq!(w.armies()[0].income_mass, 300);
    }

    #[test]
    fn rejects_foreign_and_invalid_commands() {
        let mut w = test_world(4);
        let enemy_acu = w.commander_of(1).unwrap().id;
        w.issue(0, Command::Move { units: vec![enemy_acu], x: 0, y: 0, queue: false });
        w.issue(0, Command::Build { builder: enemy_acu, unit_type: "RF".into(), x: 0, y: 0, queue: false });
        w.issue(0, Command::Build { builder: w.commander_of(0).unwrap().id, unit_type: "BF".into(), x: 0, y: 0, queue: false });
        w.step();
        assert_eq!(w.commands_rejected(), 3);
        assert_eq!(w.commands_applied(), 0);
    }

    #[test]
    fn move_order_arrives_in_formation() {
        let mut w = test_world(5);
        let acu = w.commander_of(0).unwrap().id;
        w.issue(0, Command::Move { units: vec![acu], x: 80 * SCALE, y: 90 * SCALE, queue: false });
        w.run(400);
        let u = w.unit(acu).unwrap();
        assert_eq!((u.pos.x, u.pos.y), (80 * SCALE, 90 * SCALE));
        assert!(u.orders.is_empty());
        assert_eq!(Skirmish::formation_slot(0, 4, 0, 0), (-1500, -1500));
        assert_eq!(Skirmish::formation_slot(3, 4, 0, 0), (1500, 1500));
    }

    #[test]
    fn engineer_build_repair_assist_reclaim_and_overcharge() {
        let mut w = test_world(8);
        let acu = w.commander_of(0).unwrap().id;
        w.issue(0, Command::Build { builder: acu, unit_type: "RF".into(), x: 64 * SCALE, y: 50 * SCALE, queue: false });
        w.run(250);
        let fac = w.units().find(|u| u.role == Role::Factory && u.army == 0).unwrap().id;
        w.issue(0, Command::Produce { factory: fac, unit_type: "RENG".into(), count: 1 });
        w.run(200);
        let eng = w
            .units()
            .find(|u| u.role == Role::Engineer && u.army == 0)
            .expect("engineer produced")
            .id;

        w.issue(0, Command::Build { builder: eng, unit_type: "RPD".into(), x: 76 * SCALE, y: 50 * SCALE, queue: false });
        w.issue(0, Command::Build { builder: eng, unit_type: "RRAD".into(), x: 52 * SCALE, y: 62 * SCALE, queue: true });
        w.run(500);
        assert!(w.units().any(|u| u.role == Role::PointDefense));
        assert!(w.units().any(|u| u.role == Role::Radar));

        let enemy_acu = w.commander_of(1).unwrap().id;
        w.units.get_mut(&enemy_acu).unwrap().pos = Vec2 { x: 120 * SCALE, y: 50 * SCALE };
        w.rebuild_spatial();
        assert_eq!(
            w.intel_visibility(0, w.unit(enemy_acu).unwrap()),
            IntelVisibility::RadarBlip
        );

        let tank = w.spawn_unit(1, "BT", (82 * SCALE, 50 * SCALE)).unwrap();
        w.run(120);
        assert!(w.unit(tank).is_none(), "PointDefense kills nearby enemy tank");
        assert!(w.wrecks().count() >= 1, "Dead tank leaves reclaimable wreck");

        let before_m = w.armies()[0].mass;
        w.issue(0, Command::Reclaim { units: vec![eng], x: 82 * SCALE, y: 50 * SCALE, queue: false });
        w.run(200);
        assert!(w.armies()[0].mass >= before_m || w.wrecks().count() == 0);
    }

    #[test]
    fn artillery_launches_ballistic_aoe_projectile() {
        let mut w = test_world(9);
        let arty = w.spawn_unit(0, "RA", (50 * SCALE, 50 * SCALE)).unwrap();
        let t1 = w.spawn_unit(1, "BT", (75 * SCALE, 50 * SCALE)).unwrap();
        let t2 = w.spawn_unit(1, "BT", (76 * SCALE, 50 * SCALE)).unwrap();
        w.issue(0, Command::Attack { units: vec![arty], target: t1, queue: false });
        w.run(5);
        assert!(w.projectiles().count() >= 1, "artillery should spawn ballistic shell");
        w.run(30);
        assert!(w.unit(t1).unwrap().hp < w.unit(t1).unwrap().max_hp);
        assert!(w.unit(t2).unwrap().hp < w.unit(t2).unwrap().max_hp, "AoE splash should hit adjacent unit");
    }

    #[test]
    fn assassination_ends_game() {
        let mut w = test_world(6);
        let a = w.commander_of(0).unwrap().id;
        let b = w.commander_of(1).unwrap().id;
        w.issue(0, Command::Attack { units: vec![a], target: b, queue: false });
        w.run(20_000);
        let r = w.result().expect("game over");
        assert!(matches!(r, GameResult::Victory(_) | GameResult::Draw));
        assert!(w.shots_fired() > 0);
    }

    #[test]
    fn parallel_skirmish_matches_single_thread_hash() {
        let run_with = |threads: usize| {
            let mut w = test_world(77);
            if threads > 1 {
                w.set_worker_pool(Some(Arc::new(WorkerPool::new(threads))));
            }
            for i in 0..32i64 {
                let u0 = w.spawn_unit(0, "RT", (20 * SCALE + (i % 8) * 3 * SCALE, 20 * SCALE + (i / 8) * 3 * SCALE)).unwrap();
                let u1 = w.spawn_unit(1, "BT", (220 * SCALE + (i % 8) * 3 * SCALE, 220 * SCALE + (i / 8) * 3 * SCALE)).unwrap();
                w.issue(0, Command::AttackMove { units: vec![u0], x: 200 * SCALE, y: 200 * SCALE, queue: false });
                w.issue(1, Command::AttackMove { units: vec![u1], x: 50 * SCALE, y: 50 * SCALE, queue: false });
            }
            w.run(120);
            w.hash_hex()
        };
        assert_eq!(run_with(1), run_with(4));
    }
}
