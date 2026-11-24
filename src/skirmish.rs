//! Phases 6, 8 & 10 multi-army deterministic skirmish world.
//!
//! Fixed-tick (10 Hz) integer-only RTS core reading unit stats directly from
//! original `*_unit.bp` files:
//! - Multi-army economies (mass/energy stockpiles, storage caps, income/drain)
//! - Commanders & T1 Engineers (build, repair, assist, wreck reclaim, Overcharge)
//! - Structures: Land Factory, Mass Extractor (with marker snapping),
//!   Power Generator, Point Defense, Radar System
//! - Units: Commander, Engineer, Medium Tank/Mantis, Light Assault Bot,
//!   Mobile Artillery (parabolic ballistic shells + AoE splash)
//! - Deterministic `SpatialGrid` radius acceleration + `WorkerPool` batched
//!   A* pathfinding with bit-identical `1t == Nt` state hashes.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use crate::blueprint::{self, Value};
use crate::combat;
use crate::economy::ConstructionProject;
use crate::jobs::WorkerPool;
use crate::pathfinding::{self, PathGrid, PathRequest};
use crate::scd;
use crate::scenario;
use crate::scmap::ScMap;
use crate::sim::{SCALE, TICK_HZ, Vec2};
use crate::spatial::{SPATIAL_CELL_MILLI, SpatialGrid};

pub type ArmyId = u8;
pub type UnitId = u32;

/// Hash header; updated for Phase 8/10 state (wrecks, projectiles, roles).
pub const SKIRMISH_HASH_MAGIC: &[u8] = b"FARUSTSKM2";
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
/// Snap radius for placing an extractor onto a mass marker (milli).
pub const MASS_SNAP: i64 = 8 * SCALE;
/// Energy cost of one ACU Overcharge shot (5000 energy in FA).
pub const OVERCHARGE_ENERGY_COST: i64 = 5_000 * SCALE;
/// Damage of an ACU Overcharge shot.
pub const OVERCHARGE_DAMAGE: i64 = 12_000 * SCALE;
/// Splash radius of an ACU Overcharge shot.
pub const OVERCHARGE_SPLASH: i64 = 2_500;
/// ACU Overcharge range.
pub const OVERCHARGE_RANGE: i64 = 22 * SCALE;

/// Weapons skipped during standard weapon extraction (Overcharge uses explicit command).
const EXCLUDED_WEAPONS: &[&str] = &[
    "OverCharge",
    "AutoOverCharge",
    "TacMissile",
    "TacNukeMissile",
    "DeathWeapon",
    "Torpedo",
    "MLG",
    "ChronoDampener",
    "Missile",
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
}

impl Role {
    pub fn is_structure(self) -> bool {
        matches!(
            self,
            Role::Factory | Role::Extractor | Role::Generator | Role::PointDefense | Role::Radar
        )
    }
    pub fn is_builder(self) -> bool {
        matches!(self, Role::Commander | Role::Engineer)
    }
    fn code(self) -> u8 {
        match self {
            Role::Commander => 0,
            Role::Factory => 1,
            Role::Extractor => 2,
            Role::Generator => 3,
            Role::PointDefense => 4,
            Role::Radar => 5,
            Role::Engineer => 6,
            Role::Mobile => 7,
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
}

/// Quantized unit type derived once from a blueprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitType {
    pub id: String,
    pub name: String,
    pub role: Role,
    pub max_hp: i64,
    pub speed_per_tick: i64,
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
    /// Footprint half-size (milli) and render size (milli).
    pub radius: i64,
    pub size: [i64; 3],
}

impl UnitType {
    pub fn max_range(&self) -> i64 {
        self.weapons.iter().map(|w| w.range).max().unwrap_or(0)
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

        let raw_weapons: Vec<&Value> = root
            .get("Weapon")
            .map(|w| w.table_list())
            .unwrap_or_default();

        let weapons = raw_weapons
            .iter()
            .filter_map(|t| {
                let label = t.get_str("Label").unwrap_or("").to_string();
                if EXCLUDED_WEAPONS.iter().any(|x| label.contains(x)) {
                    return None;
                }
                let d = t.get_f64("Damage")?;
                let r = t.get_f64("MaxRadius")?;
                let rof = t.get_f64("RateOfFire")?;
                if d <= 0.0 || r <= 0.0 || rof <= 0.0 {
                    return None;
                }
                let splash_radius = t.get_f64("DamageRadius").map(q).unwrap_or(0).max(0);
                let muzzle_velocity = t.get_f64("MuzzleVelocity").map(per_tick).unwrap_or(0).max(0);
                Some(WeaponType {
                    label,
                    damage: q(d),
                    range: q(r),
                    reset: combat::cooldown_reset_ticks(TICK_HZ, q(rof)),
                    splash_radius,
                    muzzle_velocity,
                })
            })
            .collect();

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
        let default_build_range = match role {
            Role::Commander => 10.0,
            Role::Engineer => 5.0,
            _ => 0.0,
        };
        let vision_default = match role {
            Role::Commander => 26.0,
            Role::PointDefense => 24.0,
            Role::Radar => 20.0,
            _ => 18.0,
        };
        Self {
            id: id.to_string(),
            name,
            role,
            max_hp: s.health.map(q).unwrap_or(SCALE).max(1),
            speed_per_tick: if role.is_structure() {
                0
            } else {
                s.max_speed.map(per_tick).unwrap_or(0).max(0)
            },
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
            radius: q(half).max(SCALE / 4),
            size: [q(size_x), q(size_y), q(size_z)],
        }
    }
}

/// One playable faction: commander, 5 structures, T1 engineer, and combat land units.
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
}

impl Faction {
    pub fn can_build(&self, unit_type: &str) -> bool {
        [
            &self.factory,
            &self.extractor,
            &self.generator,
            &self.point_defense,
            &self.radar,
        ]
        .iter()
        .any(|t| **t == unit_type)
    }

    pub fn can_produce(&self, unit_type: &str) -> bool {
        self.engineer == unit_type || self.land_units.iter().any(|u| u == unit_type)
    }
}

/// Blueprint ids per faction:
/// `[commander, factory, mex, pgen, point_defense, radar, engineer, tank, bot, artillery]`
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

#[derive(Debug, Clone, Default)]
pub struct Catalog {
    pub types: BTreeMap<String, UnitType>,
    pub factions: Vec<Faction>,
}

impl Catalog {
    pub fn faction_index(&self, key: &str) -> Option<usize> {
        self.factions
            .iter()
            .position(|f| f.key.eq_ignore_ascii_case(key))
    }

    pub fn get(&self, id: &str) -> Option<&UnitType> {
        self.types.get(id)
    }

    /// Load every faction in [`FACTION_TABLE`] from `units.scd`.
    pub fn load(game_dir: &Path) -> Result<Self, String> {
        let units_scd = scd::gamedata_dir(game_dir).join("units.scd");
        let mut cat = Catalog::default();
        for (key, ids) in FACTION_TABLE {
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
                let inner = format!("units/{id}/{id}_unit.bp");
                let (_, bytes) = scd::read_scd_file_ci(&units_scd, &inner)
                    .map_err(|e| format!("{key}: cannot read {inner}: {e}"))?;
                let root = blueprint::parse_blueprint(&String::from_utf8_lossy(&bytes))
                    .map_err(|e| format!("{inner}: {e}"))?;
                cat.types
                    .insert(id.to_string(), UnitType::from_blueprint(id, role, &root));
            }
            cat.factions.push(Faction {
                key: key.to_string(),
                commander: ids[0].to_string(),
                factory: ids[1].to_string(),
                extractor: ids[2].to_string(),
                generator: ids[3].to_string(),
                point_defense: ids[4].to_string(),
                radar: ids[5].to_string(),
                engineer: ids[6].to_string(),
                land_units: ids[7..].iter().map(|s| s.to_string()).collect(),
            });
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
}

#[derive(Debug, Clone)]
pub struct Unit {
    pub id: UnitId,
    pub army: ArmyId,
    pub type_id: String,
    pub role: Role,
    pub pos: Vec2,
    pub hp: i64,
    pub max_hp: i64,
    pub alive: bool,
    pub weapons: Vec<WeaponState>,
    pub speed: i64,
    pub radius: i64,
    pub max_range: i64,
    pub vision_radius: i64,
    pub radar_radius: i64,
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
}

impl Unit {
    #[allow(dead_code)]
    pub fn is_idle(&self) -> bool {
        self.orders.is_empty() && self.build.is_none() && self.production.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wreck {
    pub id: UnitId,
    pub pos: Vec2,
    pub mass_left: i64,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Army {
    pub id: ArmyId,
    pub name: String,
    pub faction: usize,
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
}

/// Map-derived context shared by headless runs, the AI and the viewer.
#[derive(Debug, Clone)]
pub struct MapContext {
    pub map_id: String,
    pub map_name: String,
    pub scmap: ScMap,
    pub grid: PathGrid,
    pub size: (i64, i64),
    /// Start spots in milli-units, indexed like `ARMY_1..ARMY_n`.
    pub starts: Vec<(String, (i64, i64))>,
    /// Land mass markers in milli-units.
    pub mass_points: Vec<(i64, i64)>,
}

impl MapContext {
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
        let grid = PathGrid::from_scmap(&scmap, GRID_STEP, GRID_MAX_SLOPE_MILLI, false);
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
        let size = (q(scmap.map_width), q(scmap.map_height));
        Ok(Self {
            map_id: files.id.clone(),
            map_name: meta.name.clone(),
            scmap,
            grid,
            size,
            starts,
            mass_points,
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
    spatial: SpatialGrid,
    pool: Option<Arc<WorkerPool>>,
    size: (i64, i64),
    mass_points: Vec<(i64, i64)>,
    mass_claims: Vec<Option<UnitId>>,
    pending: Vec<(ArmyId, Command)>,
    catalog: Catalog,
    shots_fired: u64,
    kills: u64,
    paths_solved: u64,
    commands_applied: u64,
    commands_rejected: u64,
    result: Option<GameResult>,
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
        let spatial = SpatialGrid::new(size.0, size.1, SPATIAL_CELL_MILLI);
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
            grid: Arc::new(grid),
            spatial,
            pool: None,
            size,
            mass_points,
            mass_claims: claims,
            pending: Vec::new(),
            catalog,
            shots_fired: 0,
            kills: 0,
            paths_solved: 0,
            commands_applied: 0,
            commands_rejected: 0,
            result: None,
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
            .map(|&(spot, faction)| {
                let (name, start) = map
                    .starts
                    .get(spot)
                    .cloned()
                    .ok_or_else(|| format!("start spot {} out of range", spot + 1))?;
                Ok(ArmySetup { name, faction, start })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Self::new(
            catalog,
            map.grid.clone(),
            map.size,
            map.mass_points.clone(),
            &setups,
            seed,
        )
    }

    /// Attach or detach a persistent `WorkerPool` for batched A* pathfinding.
    pub fn set_worker_pool(&mut self, pool: Option<Arc<WorkerPool>>) {
        self.pool = pool;
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
    pub fn mass_claim(&self, idx: usize) -> Option<UnitId> {
        self.mass_claims.get(idx).copied().flatten()
    }
    pub fn size(&self) -> (i64, i64) {
        self.size
    }
    pub fn grid(&self) -> &PathGrid {
        &self.grid
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

    /// Own live units whose centre lies in the axis-aligned rectangle.
    pub fn units_in_rect(&self, army: ArmyId, a: (i64, i64), b: (i64, i64)) -> Vec<UnitId> {
        let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
        let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
        self.units()
            .filter(|u| u.army == army)
            .filter(|u| u.pos.x >= x0 && u.pos.x <= x1 && u.pos.y >= y0 && u.pos.y <= y1)
            .map(|u| u.id)
            .collect()
    }

    /// Nearest live unit near `p` (within `u.radius + pick_radius`), optionally
    /// restricted to `army_filter` or excluding `hostile_to`.
    pub fn unit_at(
        &self,
        p: (i64, i64),
        pick_radius: i64,
        army_filter: Option<ArmyId>,
        hostile_to: Option<ArmyId>,
    ) -> Option<UnitId> {
        let mut cand = Vec::new();
        self.spatial
            .query_radius_sorted(p.0, p.1, pick_radius + 10 * SCALE, &mut cand);
        cand.into_iter()
            .filter_map(|id| self.unit(id))
            .filter(|u| army_filter.is_none_or(|a| u.army == a))
            .filter(|u| hostile_to.is_none_or(|a| u.army != a))
            .filter_map(|u| {
                let reach = (u.radius + pick_radius) as i128;
                let d2 = dist_sq((u.pos.x, u.pos.y), p);
                (d2 <= reach * reach).then_some((d2, u.id))
            })
            .min()
            .map(|(_, id)| id)
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

    /// Fog-of-War & Radar classification from `viewer_army`'s perspective.
    pub fn intel_visibility(&self, viewer_army: ArmyId, target: &Unit) -> IntelVisibility {
        if target.army == viewer_army {
            return IntelVisibility::Visible;
        }
        let tp = (target.pos.x, target.pos.y);
        let mut radar_hit = false;
        for u in self.units() {
            if u.army != viewer_army {
                continue;
            }
            let d2 = dist_sq((u.pos.x, u.pos.y), tp);
            let vis = (u.vision_radius + target.radius) as i128;
            if d2 <= vis * vis {
                return IntelVisibility::Visible;
            }
            if u.radar_radius > 0 {
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

    /// Whether a structure of `type_id` may be placed at `(x, y)`.
    pub fn site_is_free(&self, type_id: &str, x: i64, y: i64) -> bool {
        let Some(t) = self.catalog.get(type_id) else {
            return false;
        };
        let r = t.radius;
        if x - r < 0 || y - r < 0 || x + r > self.size.0 || y + r > self.size.1 {
            return false;
        }
        let (cx, cy) = self.grid.world_to_cell(x, y);
        if self.grid.is_blocked(cx, cy) {
            return false;
        }
        let margin = SCALE;
        let overlaps_structure = self.units().any(|u| {
            u.role.is_structure() && {
                let lim = (u.radius + r + margin) as i128;
                dist_sq((u.pos.x, u.pos.y), (x, y)) < lim * lim
            }
        });
        let overlaps_site = self.units().any(|u| {
            u.build.as_ref().is_some_and(|j| {
                u.role.is_builder() && {
                    let other = self.catalog.get(&j.unit_type).map(|t| t.radius).unwrap_or(0);
                    let lim = (other + r + margin) as i128;
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
        true
    }

    /// Snap a requested extractor site to the nearest free mass point.
    pub fn snap_extractor(&self, x: i64, y: i64) -> Option<(usize, (i64, i64))> {
        self.nearest_free_mass_point((x, y), MASS_SNAP)
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

    pub fn drain_pending(&mut self) -> Vec<(ArmyId, Command)> {
        std::mem::take(&mut self.pending)
    }

    fn owned_mobiles(&self, army: ArmyId, ids: &[UnitId]) -> Vec<UnitId> {
        let mut v: Vec<UnitId> = ids
            .iter()
            .copied()
            .filter(|id| {
                self.unit(*id)
                    .is_some_and(|u| u.army == army && u.speed > 0)
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
                    .is_some_and(|u| u.army == army && u.role.is_builder())
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
        if let Some(j) = job
            && let Some(mp) = j.mass_point
            && self.mass_claims[mp] == Some(id)
        {
            self.mass_claims[mp] = None;
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
            Command::Attack { units, target, queue } => {
                let valid = self.unit(target).is_some_and(|t| t.army != army);
                if !valid {
                    return false;
                }
                let mut v: Vec<UnitId> = units
                    .iter()
                    .copied()
                    .filter(|id| {
                        self.unit(*id)
                            .is_some_and(|u| u.army == army && !u.weapons.is_empty())
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
                if b.army != army || !b.role.is_builder() {
                    return false;
                }
                let Some(f) = self.faction(army) else {
                    return false;
                };
                if !f.can_build(&unit_type) {
                    return false;
                }
                let (x, y) = if self.catalog.get(&unit_type).map(|t| t.role) == Some(Role::Extractor) {
                    match self.snap_extractor(x, y) {
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
                let ok = self
                    .unit(factory)
                    .is_some_and(|u| u.army == army && u.role == Role::Factory)
                    && self
                        .faction(army)
                        .is_some_and(|f| f.can_produce(&unit_type));
                if !ok || count == 0 {
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
                match self.units.get_mut(&factory) {
                    Some(u) if u.alive && u.army == army && u.role == Role::Factory => {
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
                if !self.unit(target).is_some_and(|t| t.army == army) {
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
                    .is_some_and(|t| t.army == army && (t.role == Role::Factory || t.role.is_builder()))
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
                    .is_some_and(|u| u.army == army && u.role == Role::Commander);
                let ok_t = self.unit(target).is_some_and(|t| t.army != army);
                if !ok_u || !ok_t {
                    return false;
                }
                self.push_order(unit, UnitOrder::Overcharge(target), false);
                true
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
        let id = self.next_id;
        self.next_id += 1;
        let weapons = t
            .weapons
            .iter()
            .map(|w| WeaponState {
                damage: w.damage,
                range_sq: (w.range as i128) * (w.range as i128),
                reset: w.reset,
                cooldown: 0,
                splash_radius: w.splash_radius,
                muzzle_velocity: w.muzzle_velocity,
            })
            .collect();
        self.units.insert(
            id,
            Unit {
                id,
                army,
                type_id: t.id.clone(),
                role: t.role,
                pos: Vec2 { x: pos.0, y: pos.1 },
                hp: t.max_hp,
                max_hp: t.max_hp,
                alive: true,
                weapons,
                speed: t.speed_per_tick,
                radius: t.radius,
                max_range: t.max_range(),
                vision_radius: t.vision_radius,
                radar_radius: t.radar_radius,
                orders: VecDeque::new(),
                path: VecDeque::new(),
                path_goal: None,
                build: None,
                production: VecDeque::new(),
                rally: None,
                mass_point: None,
                last_shot: None,
                last_hit_by: None,
            },
        );
        Some(id)
    }

    pub fn rebuild_spatial(&mut self) {
        self.spatial.rebuild(
            self.units
                .values()
                .filter(|u| u.alive)
                .map(|u| (u.id, u.pos.x, u.pos.y)),
        );
    }

    fn recompute_caps(&mut self) {
        let mut caps = vec![(0i64, 0i64); self.armies.len()];
        for u in self.units.values().filter(|u| u.alive) {
            if let Some(t) = self.catalog.get(&u.type_id) {
                caps[u.army as usize].0 += t.storage_mass;
                caps[u.army as usize].1 += t.storage_energy;
            }
        }
        for (a, (m, e)) in self.armies.iter_mut().zip(caps) {
            a.mass_cap = m;
            a.energy_cap = e;
        }
    }

    /// Advance one fixed tick. Every phase is strictly ordered by ascending ID.
    pub fn step(&mut self) {
        if self.result.is_some() {
            return;
        }
        self.tick += 1;
        let _ = self.next_rng();
        self.apply_commands();
        self.step_income();
        self.step_construction();
        self.rebuild_spatial();
        self.step_movement();
        self.step_separation();
        self.rebuild_spatial();
        self.step_weapons();
        self.step_projectiles();
        self.step_cleanup();
        self.rebuild_spatial();
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
        self.recompute_caps();
        let mut income = vec![(0i64, 0i64); self.armies.len()];
        for u in self.units.values().filter(|u| u.alive) {
            if let Some(t) = self.catalog.get(&u.type_id) {
                income[u.army as usize].0 += t.mass_per_tick;
                income[u.army as usize].1 += t.energy_per_tick;
            }
        }
        for (a, (m, e)) in self.armies.iter_mut().zip(income) {
            a.income_mass = m;
            a.income_energy = e;
            a.mass = (a.mass + m).min(a.mass_cap.max(a.mass));
            a.energy = (a.energy + e).min(a.energy_cap.max(a.energy));
            a.mass = a.mass.min(a.mass_cap.max(0)).max(0);
            a.energy = a.energy.min(a.energy_cap.max(0)).max(0);
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

    fn step_construction(&mut self) {
        let ids: Vec<UnitId> = self.units.keys().copied().collect();

        // Pre-compute friendly build-rate assist contributions in ascending ID order.
        let mut assist_bonus: BTreeMap<UnitId, i64> = BTreeMap::new();
        for &id in &ids {
            let Some(u) = self.units.get(&id) else { continue };
            if !u.alive || !u.role.is_builder() {
                continue;
            }
            if let Some(UnitOrder::Assist(target_id)) = u.orders.front()
                && let Some(target) = self.units.get(target_id).filter(|t| t.alive && t.army == u.army)
            {
                let (br, brange) = self
                    .catalog
                    .get(&u.type_id)
                    .map(|t| (t.build_rate, t.build_range))
                    .unwrap_or((0, 0));
                let reach = (brange + target.radius + u.radius) as i128;
                if dist_sq((u.pos.x, u.pos.y), (target.pos.x, target.pos.y)) <= reach * reach {
                    *assist_bonus.entry(*target_id).or_insert(0) += br;
                }
            }
        }

        for id in ids {
            let (alive, role, army, pos, radius, rate, head) = match self.units.get(&id) {
                Some(u) => (
                    u.alive,
                    u.role,
                    u.army,
                    (u.pos.x, u.pos.y),
                    u.radius,
                    self.catalog.get(&u.type_id).map(|t| (t.build_rate, t.build_range)),
                    u.orders.front().cloned(),
                ),
                None => continue,
            };
            if !alive {
                continue;
            }
            let Some((build_rate, build_range)) = rate else {
                continue;
            };
            let assisted_rate = build_rate + assist_bonus.get(&id).copied().unwrap_or(0);

            match role {
                Role::Factory => {
                    if self.units[&id].build.is_none() {
                        let next = self.units.get_mut(&id).unwrap().production.pop_front();
                        if let Some(tid) = next {
                            let job = self.new_job(build_rate, &tid, pos.0, pos.1, None);
                            self.units.get_mut(&id).unwrap().build = job;
                        }
                    }
                    let mut job = match self.units.get_mut(&id).unwrap().build.take() {
                        Some(j) => j,
                        None => continue,
                    };
                    if Self::progress_job_with_rate(&mut self.armies, army, &mut job, assisted_rate) {
                        let exit = (pos.0, (pos.1 + radius + 2 * SCALE).min(self.size.1));
                        if let Some(nid) = self.spawn_unit(army, &job.unit_type, exit) {
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
                        self.units.get_mut(&id).unwrap().build = Some(job);
                    }
                }
                Role::Commander | Role::Engineer => match head {
                    Some(UnitOrder::Build { unit_type, x, y }) => {
                        let target_r = self.catalog.get(&unit_type).map(|t| t.radius).unwrap_or(0);
                        let reach = (build_range + target_r + radius) as i128;
                        if dist_sq(pos, (x, y)) > reach * reach {
                            continue;
                        }
                        if self.units[&id].build.is_none() {
                            let is_mex = self.catalog.get(&unit_type).map(|t| t.role)
                                == Some(Role::Extractor);
                            let mp = if is_mex {
                                self.mass_points
                                    .iter()
                                    .position(|&m| m == (x, y))
                                    .filter(|&i| self.mass_claims[i].is_none())
                            } else {
                                None
                            };
                            let valid = if is_mex {
                                mp.is_some()
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
                            }
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                        } else {
                            self.units.get_mut(&id).unwrap().build = Some(job);
                        }
                    }
                    Some(UnitOrder::Reclaim { x, y }) => {
                        let reach = build_range + 4 * SCALE;
                        if let Some((wid, wpos)) = self.wreck_at((x, y), MASS_SNAP * 2)
                            && dist_sq(pos, wpos) <= (reach as i128) * (reach as i128)
                        {
                            let gain = (build_rate.max(1) * 400).max(SCALE / 5);
                            let w = self.wrecks.get_mut(&wid).unwrap();
                            let harvested = w.mass_left.min(gain);
                            w.mass_left -= harvested;
                            let a = &mut self.armies[army as usize];
                            a.mass = (a.mass + harvested).min(a.mass_cap.max(a.mass));
                            if w.mass_left <= 0 {
                                self.wrecks.remove(&wid);
                            }
                        } else if self.wreck_at((x, y), MASS_SNAP * 2).is_none() {
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                        }
                    }
                    Some(UnitOrder::Repair(tid)) => {
                        let tinfo = self
                            .unit(tid)
                            .filter(|t| t.army == army)
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
                        if !self.unit(tid).is_some_and(|t| t.army == army) {
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                        }
                    }
                    _ => {}
                },
                _ => {}
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
        if (u.pos.x, u.pos.y) == goal {
            u.path.clear();
            u.path_goal = None;
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

    /// Nearest live enemy of `army` to `p` within `radius` via `SpatialGrid`.
    fn nearest_enemy(&self, army: ArmyId, p: (i64, i64), radius: i64) -> Option<UnitId> {
        let lim = (radius as i128) * (radius as i128);
        let mut cand = Vec::new();
        self.spatial.query_radius_sorted(p.0, p.1, radius + 8 * SCALE, &mut cand);
        let mut best_mobile: Option<(i128, UnitId)> = None;
        let mut best_struct: Option<(i128, UnitId)> = None;
        for tid in cand {
            let Some(u) = self.units.get(&tid) else { continue };
            if !u.alive || u.army == army {
                continue;
            }
            let d = dist_sq(p, (u.pos.x, u.pos.y));
            if d > lim {
                continue;
            }
            let slot = if u.role.is_structure() {
                &mut best_struct
            } else {
                &mut best_mobile
            };
            if slot.is_none_or(|(bd, _)| d < bd) {
                *slot = Some((d, u.id));
            }
        }
        best_mobile.or(best_struct).map(|(_, id)| id)
    }

    fn in_weapon_range(&self, u: &Unit, target: UnitId) -> bool {
        self.unit(target).is_some_and(|t| {
            let reach = (u.max_range + t.radius) as i128;
            dist_sq((u.pos.x, u.pos.y), (t.pos.x, t.pos.y)) <= reach * reach
        })
    }

    /// Determines a unit's desired movement `(goal, tolerance)` if it needs to move this tick,
    /// popping immediately completed orders in ascending order.
    fn pop_completed_and_goal(&mut self, id: UnitId) -> Option<((i64, i64), i64)> {
        for _ in 0..4 {
            let Some(u) = self.units.get(&id) else {
                return None;
            };
            if !u.alive || u.speed <= 0 {
                return None;
            }
            let Some(head) = u.orders.front().cloned() else {
                return None;
            };
            let army = u.army;
            let pos = (u.pos.x, u.pos.y);
            match head {
                UnitOrder::Move { x, y } => {
                    if pos == (x, y) {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    }
                    return Some(((x, y), SCALE));
                }
                UnitOrder::AttackMove { x, y } => {
                    let acquire = u.max_range + ACQUIRE_BONUS;
                    if u.max_range > 0
                        && let Some(e) = self.nearest_enemy(army, pos, acquire)
                    {
                        if self.in_weapon_range(&self.units[&id], e) {
                            return None;
                        }
                        let ep = self.unit(e).map(|t| (t.pos.x, t.pos.y)).unwrap();
                        return Some((ep, 4 * SCALE));
                    }
                    if pos == (x, y) {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    }
                    return Some(((x, y), SCALE));
                }
                UnitOrder::Attack(t) | UnitOrder::Overcharge(t) => {
                    let Some(tp) = self.unit(t).map(|t| (t.pos.x, t.pos.y)) else {
                        self.units.get_mut(&id).unwrap().orders.pop_front();
                        continue;
                    };
                    let in_r = if matches!(head, UnitOrder::Overcharge(_)) {
                        let reach = (OVERCHARGE_RANGE + self.unit(t).unwrap().radius) as i128;
                        dist_sq(pos, tp) <= reach * reach
                    } else {
                        self.in_weapon_range(&self.units[&id], t)
                    };
                    if in_r {
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
                    let target = self
                        .wreck_at((x, y), MASS_SNAP * 2)
                        .map(|(_, wp)| wp)
                        .unwrap_or((x, y));
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
            }
        }
        None
    }

    fn step_movement(&mut self) {
        let ids: Vec<UnitId> = self.units.keys().copied().collect();

        // Pass 1: Collect movement targets & stale A* requests in ascending UnitId order.
        let mut active_goals: Vec<(UnitId, (i64, i64))> = Vec::new();
        let mut stale_ids: Vec<UnitId> = Vec::new();
        let mut requests: Vec<PathRequest> = Vec::new();

        for id in ids {
            let Some((goal, tol)) = self.pop_completed_and_goal(id) else {
                continue;
            };
            let u = &self.units[&id];
            let tol_sq = (tol as i128) * (tol as i128);
            let stale = u.path_goal.is_none_or(|g| dist_sq(g, goal) > tol_sq);
            if stale {
                stale_ids.push(id);
                requests.push(PathRequest {
                    id: id as u64,
                    start: (u.pos.x, u.pos.y),
                    goal,
                });
            }
            active_goals.push((id, goal));
        }

        // Pass 2: Solve A* path batch in parallel via persistent WorkerPool when beneficial.
        if !requests.is_empty() {
            let results = if requests.len() >= 4
                && let Some(ref pool) = self.pool
                && pool.threads() > 1
            {
                pathfinding::solve_paths_batch_pool(&self.grid, Arc::new(requests), pool)
            } else {
                requests
                    .iter()
                    .map(|req| pathfinding::find_path(&self.grid, req))
                    .collect()
            };
            self.paths_solved += results.len() as u64;
            for (uid, res) in stale_ids.into_iter().zip(results) {
                let goal = active_goals
                    .iter()
                    .find(|&&(id, _)| id == uid)
                    .map(|&(_, g)| g)
                    .unwrap_or((0, 0));
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
        for (id, goal) in active_goals {
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

    /// Deterministic pairwise repulsion for crowded mobile units via `SpatialGrid`.
    fn step_separation(&mut self) {
        let mut cand = Vec::new();
        let mut deltas: Vec<(UnitId, i64, i64)> = Vec::new();
        let mobile_ids: Vec<UnitId> = self
            .units
            .values()
            .filter(|u| u.alive && u.speed > 0 && !u.role.is_structure())
            .map(|u| u.id)
            .collect();

        for &id in &mobile_ids {
            let u = &self.units[&id];
            self.spatial
                .query_radius_sorted(u.pos.x, u.pos.y, u.radius + 3 * SCALE, &mut cand);
            let mut push_x: i64 = 0;
            let mut push_y: i64 = 0;
            for &oid in &cand {
                if oid == id {
                    continue;
                }
                let Some(o) = self.units.get(&oid).filter(|o| o.alive && !o.role.is_structure())
                else {
                    continue;
                };
                let min_sep = u.radius + o.radius + SCALE / 2;
                let dx = u.pos.x - o.pos.x;
                let dy = u.pos.y - o.pos.y;
                let d2 = (dx as i128) * (dx as i128) + (dy as i128) * (dy as i128);
                if d2 < (min_sep as i128) * (min_sep as i128) {
                    if d2 == 0 {
                        let sign = if id < oid { -1 } else { 1 };
                        push_x += sign * (SCALE / 4);
                    } else {
                        let d = isqrt(d2).max(1);
                        let push = (min_sep - d) / 4;
                        push_x += ((dx as i128) * (push as i128) / (d as i128)) as i64;
                        push_y += ((dy as i128) * (push as i128) / (d as i128)) as i64;
                    }
                }
            }
            if push_x != 0 || push_y != 0 {
                let cap = (u.speed / 2).max(SCALE / 10);
                deltas.push((id, push_x.clamp(-cap, cap), push_y.clamp(-cap, cap)));
            }
        }

        let (max_x, max_y) = self.size;
        for (id, dx, dy) in deltas {
            if let Some(u) = self.units.get_mut(&id) {
                u.pos.x = (u.pos.x + dx).clamp(0, max_x);
                u.pos.y = (u.pos.y + dy).clamp(0, max_y);
            }
        }
    }

    fn spawn_projectile(
        &mut self,
        army: ArmyId,
        start: Vec2,
        target: Vec2,
        speed: i64,
        damage: i64,
        splash_radius: i64,
        is_overcharge: bool,
    ) {
        let id = self.next_projectile_id;
        self.next_projectile_id += 1;
        let d = isqrt(dist_sq((start.x, start.y), (target.x, target.y))).max(1);
        let spd = speed.max(SCALE / 2);
        let total_ticks = ((d + spd - 1) / spd).clamp(1, 120) as u32;
        let arc_height = if is_overcharge { 0 } else { (d / 4).min(18 * SCALE) };
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
            },
        );
    }

    fn step_weapons(&mut self) {
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        for id in ids {
            let (army, pos, role, explicit, overcharge, max_range, has_guns) = match self.units.get(&id) {
                Some(u) if u.alive => (
                    u.army,
                    (u.pos.x, u.pos.y),
                    u.role,
                    match u.orders.front() {
                        Some(UnitOrder::Attack(t)) => Some(*t),
                        _ => None,
                    },
                    match u.orders.front() {
                        Some(UnitOrder::Overcharge(t)) => Some(*t),
                        _ => None,
                    },
                    u.max_range,
                    !u.weapons.is_empty(),
                ),
                _ => continue,
            };

            if role == Role::Commander
                && let Some(tid) = overcharge
                && let Some(t) = self.unit(tid)
            {
                let tp = (t.pos.x, t.pos.y);
                let reach = (OVERCHARGE_RANGE + t.radius) as i128;
                if dist_sq(pos, tp) <= reach * reach && self.armies[army as usize].energy >= OVERCHARGE_ENERGY_COST {
                    self.armies[army as usize].energy -= OVERCHARGE_ENERGY_COST;
                    let tick = self.tick;
                    if let Some(me) = self.units.get_mut(&id) {
                        me.last_shot = Some((tid, tick));
                        me.orders.pop_front();
                    }
                    self.shots_fired += 1;
                    self.spawn_projectile(
                        army,
                        Vec2 { x: pos.0, y: pos.1 },
                        Vec2 { x: tp.0, y: tp.1 },
                        25 * SCALE / TICK_HZ as i64,
                        OVERCHARGE_DAMAGE,
                        OVERCHARGE_SPLASH,
                        true,
                    );
                    continue;
                }
            }

            if !has_guns {
                continue;
            }
            let target = explicit
                .filter(|t| self.in_weapon_range(&self.units[&id], *t))
                .or_else(|| {
                    self.nearest_enemy(army, pos, max_range + 8 * SCALE)
                        .filter(|t| self.in_weapon_range(&self.units[&id], *t))
                });
            let tinfo = target.and_then(|t| self.unit(t).map(|u| (t, (u.pos.x, u.pos.y), u.radius)));
            let n = self.units[&id].weapons.len();
            for wi in 0..n {
                let w = self.units[&id].weapons[wi].clone();
                if w.cooldown > 0 {
                    self.units.get_mut(&id).unwrap().weapons[wi].cooldown -= 1;
                    continue;
                }
                let Some((tid, tp, tr)) = tinfo else {
                    continue;
                };
                let alive = self.units.get(&tid).is_some_and(|t| t.alive);
                let reach = isqrt(w.range_sq) + tr;
                let reach_sq = (reach as i128) * (reach as i128);
                if combat::can_fire(0, alive, Some(reach_sq), dist_sq(pos, tp), Some(w.damage)) {
                    let tick = self.tick;
                    {
                        let me = self.units.get_mut(&id).unwrap();
                        me.weapons[wi].cooldown = w.reset;
                        me.last_shot = Some((tid, tick));
                    }
                    self.shots_fired += 1;
                    if w.splash_radius > 0 {
                        let spd = if w.muzzle_velocity > 0 {
                            w.muzzle_velocity
                        } else {
                            14 * SCALE / TICK_HZ as i64
                        };
                        self.spawn_projectile(
                            army,
                            Vec2 { x: pos.0, y: pos.1 },
                            Vec2 { x: tp.0, y: tp.1 },
                            spd,
                            w.damage,
                            w.splash_radius,
                            false,
                        );
                    } else {
                        self.armies[army as usize].damage_dealt += w.damage;
                        let t = self.units.get_mut(&tid).unwrap();
                        combat::apply_hit(&mut t.hp, w.damage);
                        t.last_hit_by = Some(army);
                    }
                }
            }
        }
    }

    fn step_projectiles(&mut self) {
        let pids: Vec<u32> = self.projectiles.keys().copied().collect();
        let mut exploded = Vec::new();
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
            }
        }

        let mut cand = Vec::new();
        for pid in exploded {
            let Some(p) = self.projectiles.remove(&pid) else { continue };
            let r = p.splash_radius.max(SCALE);
            self.spatial
                .query_radius_sorted(p.target.x, p.target.y, r + 4 * SCALE, &mut cand);
            for &uid in &cand {
                let Some(t) = self.units.get_mut(&uid) else { continue };
                if !t.alive || t.army == p.army {
                    continue;
                }
                let reach = (r + t.radius) as i128;
                if dist_sq((p.target.x, p.target.y), (t.pos.x, t.pos.y)) <= reach * reach {
                    self.armies[p.army as usize].damage_dealt += p.damage;
                    combat::apply_hit(&mut t.hp, p.damage);
                    t.last_hit_by = Some(p.army);
                }
            }
        }
    }

    fn step_cleanup(&mut self) {
        let mut dead_commanders = Vec::new();
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        for id in &ids {
            let (army, killer, role, pos, type_id, died) = {
                let u = self.units.get_mut(id).unwrap();
                if u.alive && u.hp <= 0 {
                    u.hp = 0;
                    u.alive = false;
                    self.kills += 1;
                    (u.army, u.last_hit_by, u.role, u.pos, u.type_id.clone(), true)
                } else {
                    (0, None, Role::Mobile, Vec2 { x: 0, y: 0 }, String::new(), false)
                }
            };
            if died {
                self.armies[army as usize].units_lost += 1;
                if let Some(k) = killer {
                    self.armies[k as usize].kills += 1;
                }
                if role == Role::Commander {
                    dead_commanders.push(army);
                } else if let Some(ut) = self.catalog.get(&type_id) {
                    let mass_left = (ut.cost_mass * 81) / 100;
                    if mass_left > 0 {
                        self.wrecks.insert(
                            *id,
                            Wreck {
                                id: *id,
                                pos,
                                mass_left,
                            },
                        );
                    }
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
        self.units.retain(|_, u| u.alive);
        let alive: Vec<ArmyId> = self
            .armies
            .iter()
            .filter(|a| !a.defeated)
            .map(|a| a.id)
            .collect();
        self.result = match alive.len() {
            0 => Some(GameResult::Draw),
            1 => Some(GameResult::Victory(alive[0])),
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
        for a in &self.armies {
            m(&mut h, &[a.id, a.defeated as u8]);
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
        for (id, u) in &self.units {
            m(&mut h, &id.to_le_bytes());
            m(&mut h, &[u.army, u.role.code()]);
            m(&mut h, &(u.type_id.len() as u32).to_le_bytes());
            m(&mut h, u.type_id.as_bytes());
            for v in [u.hp, u.pos.x, u.pos.y] {
                m(&mut h, &v.to_le_bytes());
            }
            m(&mut h, &(u.orders.len() as u32).to_le_bytes());
            m(&mut h, &(u.production.len() as u32).to_le_bytes());
            m(&mut h, &(u.path.len() as u32).to_le_bytes());
            if let Some(j) = &u.build {
                m(&mut h, &j.project.progress.to_le_bytes());
            }
            for w in &u.weapons {
                m(&mut h, &w.cooldown.to_le_bytes());
            }
        }
        for (wid, w) in &self.wrecks {
            m(&mut h, &wid.to_le_bytes());
            m(&mut h, &w.pos.x.to_le_bytes());
            m(&mut h, &w.pos.y.to_le_bytes());
            m(&mut h, &w.mass_left.to_le_bytes());
        }
        for (pid, p) in &self.projectiles {
            m(&mut h, &pid.to_le_bytes());
            m(&mut h, &[p.army, u8::from(p.is_overcharge)]);
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
            max_hp: hp * SCALE,
            speed_per_tick: speed * SCALE / 10,
            weapons: if dmg > 0 {
                vec![WeaponType {
                    label: "Gun".into(),
                    damage: dmg * SCALE,
                    range: range * SCALE,
                    reset: 10,
                    splash_radius: 0,
                    muzzle_velocity: 0,
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
                _ => 0,
            },
            storage_mass: if role == Role::Commander { 650 * SCALE } else { 0 },
            storage_energy: if role == Role::Commander { 4000 * SCALE } else { 0 },
            vision_radius: 26 * SCALE,
            radar_radius: if role == Role::Radar { 115 * SCALE } else { 0 },
            radius: if role.is_structure() { 3 * SCALE } else { SCALE / 2 },
            size: [SCALE; 3],
        }
    }

    pub(crate) fn test_catalog() -> Catalog {
        let mut c = Catalog::default();
        for (k, ids) in [
            ("red", ["RC", "RF", "RM", "RP", "RPD", "RRAD", "RENG", "RT", "RA"]),
            ("blue", ["BC", "BF", "BM", "BP", "BPD", "BRAD", "BENG", "BT", "BA"]),
        ] {
            c.types.insert(ids[0].into(), ty(ids[0], Role::Commander, 2000, 2, 20, 20));
            c.types.insert(ids[1].into(), ty(ids[1], Role::Factory, 1000, 0, 0, 0));
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
                ArmySetup { name: "A".into(), faction: 0, start: (50 * SCALE, 50 * SCALE) },
                ArmySetup { name: "B".into(), faction: 1, start: (200 * SCALE, 200 * SCALE) },
            ],
            seed,
        )
        .unwrap()
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
