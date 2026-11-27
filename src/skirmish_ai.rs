//! Phase 19 skirmish AI II.
//!
//! Deterministic opponent driven purely through [`Command`]s. It runs inside the lockstep
//! sim on every peer, so it only uses integer math, id-ordered iteration, `BTree*`
//! containers and an RNG seeded from the sim seed. Difficulties ([`Difficulty`]) change
//! decision cadence / APM, build-order tightness, micro and target selection — nobody gets
//! extra resources.
//!
//! - **Economy**: per-faction openings ([`crate::ai_build`]), energy planned from the live
//!   construction drain, engineers expanding to every reachable mass point and hydrocarbon
//!   (threat-aware), T1→T2→T3 extractor and factory upgrades, assist / reclaim when idle.
//! - **Intel**: a [`ThreatMap`] fed only by allied vision and radar, with decaying memory.
//! - **Army**: gather → attack (threat-map target scoring) → retreat when locally
//!   outnumbered; base defence, allied base defence and joint pushes with allied AIs;
//!   scouting, raids on enemy eco, composition that answers enemy air (interceptors / AA)
//!   and static defences (artillery); air and naval factories where the map allows.
//! - **Hard micro**: commander Overcharge, focus fire on exposed commanders, damaged-unit
//!   and engineer retreat.
//! - [`Difficulty::Legacy`] runs the original scripted AI ([`crate::ai_legacy`]).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use crate::ai_build::{self, FactoryStep, Step, StepCounts};
use crate::ai_legacy::LegacyAi;
use crate::ai_threat::{self, Power, Reach, ThreatMap, VIS_RADAR, VIS_SIGHT};
use crate::flowfield::NavDomain;
use crate::pathfinding::PathGrid;
use crate::sim::SCALE;
use crate::skirmish::{
    ArmyId, Command, Faction, OVERCHARGE_ENERGY_COST, OVERCHARGE_RANGE, Role, Skirmish, Unit, UnitId, UnitOrder,
    UnitType, isqrt,
};

/// One world unit in milli-units.
const U: i64 = SCALE;
const DEFEND_RADIUS: i64 = 70 * U;
const GATHER_DIST: i64 = 28 * U;
const ACU_LEASH: i64 = 60 * U;
const ACU_MEX_RANGE: i64 = 60 * U;
/// Number of [`Role`] codes (`Role::code` is 0..ROLES).
const ROLES: usize = 22;
/// Raids start after 4 minutes of game time.
const RAID_START: u64 = 2_400;
/// Air units at least this fast (milli per tick) are used for strikes, otherwise kept home.
const AIR_STRIKE_SPEED: i64 = 150;
/// Pending (not yet applied) build orders are trusted for this many ticks.
const JOB_GRACE: u64 = 30;

// ---------------------------------------------------------------------------
// Difficulty & profiles
// ---------------------------------------------------------------------------

/// AI strength level. Parsed from `easy | normal | hard | adaptive | legacy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Difficulty {
    /// The original Phase 6–12 scripted AI, kept as a baseline.
    Legacy,
    Easy,
    #[default]
    Normal,
    Hard,
    /// Starts as `normal`, then plays harder while losing and softer while winning.
    Adaptive,
    /// Phase 52: the Phase 19 `hard` profile, frozen as an arena benchmark (`hard19`).
    Hard19,
}

impl Difficulty {
    pub const ALL: [Difficulty; 5] = [
        Difficulty::Easy,
        Difficulty::Normal,
        Difficulty::Hard,
        Difficulty::Adaptive,
        Difficulty::Legacy,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Difficulty::Legacy => "legacy",
            Difficulty::Easy => "easy",
            Difficulty::Normal => "normal",
            Difficulty::Hard => "hard",
            Difficulty::Adaptive => "adaptive",
            Difficulty::Hard19 => "hard19",
        }
    }

    /// One-line summary for menus / `--help`.
    pub fn describe(self) -> &'static str {
        match self {
            Difficulty::Legacy => "original Phase 6-12 scripted AI",
            Difficulty::Easy => "slow decisions, loose opening, fixed waves, no micro",
            Difficulty::Normal => "faction openings, threat-map targeting, scouting, retreats",
            Difficulty::Hard => "fast APM, tight queued openings, raids, overcharge & focus-fire micro",
            Difficulty::Adaptive => "normal that plays harder while losing and softer while winning",
            Difficulty::Hard19 => "the Phase 19 hard AI, kept as a benchmark",
        }
    }
}

impl fmt::Display for Difficulty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Difficulty {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "legacy" | "old" => Ok(Difficulty::Legacy),
            "hard19" => Ok(Difficulty::Hard19),
            "easy" => Ok(Difficulty::Easy),
            "normal" | "medium" => Ok(Difficulty::Normal),
            "hard" => Ok(Difficulty::Hard),
            "adaptive" | "auto" => Ok(Difficulty::Adaptive),
            other => Err(format!(
                "unknown AI difficulty '{other}' (easy, normal, hard, adaptive, legacy)"
            )),
        }
    }
}

/// Phase 52: Elo ladder over arena results `(left, right, winner side)` (draws = ½).
/// Batch Elo: 40 passes over the games with a shrinking K so the ratings do not depend
/// on game order; everyone starts at 1500.
pub fn elo_ratings(games: &[(Difficulty, Difficulty, Option<u8>)]) -> std::collections::BTreeMap<Difficulty, f64> {
    let mut r: std::collections::BTreeMap<Difficulty, f64> = std::collections::BTreeMap::new();
    for &(a, b, _) in games {
        r.entry(a).or_insert(1500.0);
        r.entry(b).or_insert(1500.0);
    }
    for pass in 0..40 {
        let k = 32.0 / (1.0 + pass as f64 * 0.5);
        for &(a, b, w) in games {
            if a == b {
                continue;
            }
            let (ra, rb) = (r[&a], r[&b]);
            let ea = 1.0 / (1.0 + 10f64.powf((rb - ra) / 400.0));
            let sa = match w {
                Some(0) => 1.0,
                Some(_) => 0.0,
                None => 0.5,
            };
            *r.get_mut(&a).unwrap() += k * (sa - ea);
            *r.get_mut(&b).unwrap() -= k * (sa - ea);
        }
    }
    r
}

/// Parses `hard` (every army) or a per-army list `hard,normal` (cycled to `armies`).
pub fn parse_difficulty_list(s: &str, armies: usize) -> Result<Vec<Difficulty>, String> {
    let list: Vec<Difficulty> = s
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if list.is_empty() {
        return Err("empty --ai list".into());
    }
    Ok((0..armies).map(|i| list[i % list.len()]).collect())
}

/// Tunables that make up a difficulty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    /// Ticks between decisions.
    pub think: u64,
    /// Commands per decision (APM cap).
    pub max_cmds: usize,
    /// Minimum ticks before the same unit is re-ordered (covers lockstep input delay).
    pub reorder: u64,
    pub eng_base: usize,
    pub eng_max: usize,
    /// Engineers produced in parallel.
    pub eng_parallel: usize,
    pub max_land_fac: usize,
    /// Mass income (per second) that justifies one more land factory.
    pub fac_per_income: i64,
    /// Commander build steps queued per order.
    pub queue_opening: usize,
    pub sloppy_opening: bool,
    /// Percent of the current energy drain that income must cover.
    pub energy_margin: i64,
    /// Energy income kept per unit of mass income.
    pub energy_ratio: i64,
    pub max_pgen_batch: i64,
    /// Attack when our power * 100 >= estimated defence * ratio.
    pub attack_ratio: i64,
    /// Retreat when local enemy * 100 > local own * ratio (0 = never).
    pub retreat_ratio: i64,
    pub min_attack_power: i64,
    /// Fixed wave size in units (0 = power based).
    pub wave_units: usize,
    pub micro: bool,
    pub overcharge: bool,
    pub scouting: bool,
    pub raids: usize,
    pub guards: usize,
    pub smart_comp: bool,
    pub threat_targets: bool,
    /// Highest tolerated known threat at an expansion site.
    pub mex_danger: i64,
    /// Furthest expansion from the start (world units).
    pub mex_range: i64,
    /// Mass income (per second) before extractor upgrades start.
    pub mex_upgrade_income: i64,
    pub mex_parallel: usize,
    pub fac_upgrade: bool,
    pub team_coord: bool,
    pub tech: bool,
    pub air: bool,
    pub naval: bool,
    /// Phase 52: reserve a busy factory for its tech upgrade (stop feeding it, upgrade as
    /// soon as it falls idle) instead of waiting for an idle one that never comes.
    pub tech_push: bool,
}

impl Profile {
    pub fn of(d: Difficulty) -> Self {
        let normal = Self {
            think: 20,
            max_cmds: 10,
            reorder: 40,
            eng_base: 3,
            eng_max: 7,
            eng_parallel: 1,
            max_land_fac: 4,
            fac_per_income: 5,
            queue_opening: 1,
            sloppy_opening: false,
            energy_margin: 100,
            energy_ratio: 6,
            max_pgen_batch: 2,
            attack_ratio: 110,
            retreat_ratio: 180,
            min_attack_power: 900,
            wave_units: 0,
            micro: false,
            overcharge: true,
            scouting: true,
            raids: 0,
            guards: 2,
            smart_comp: true,
            threat_targets: true,
            mex_danger: 150,
            mex_range: 100_000,
            mex_upgrade_income: 18,
            mex_parallel: 1,
            fac_upgrade: true,
            team_coord: true,
            tech: true,
            air: true,
            naval: true,
            tech_push: false,
        };
        match d {
            Difficulty::Easy => Self {
                think: 40,
                max_cmds: 3,
                reorder: 80,
                eng_base: 2,
                eng_max: 3,
                max_land_fac: 2,
                fac_per_income: 8,
                sloppy_opening: true,
                energy_margin: 80,
                energy_ratio: 0,
                max_pgen_batch: 1,
                attack_ratio: 0,
                retreat_ratio: 0,
                min_attack_power: 0,
                wave_units: 10,
                overcharge: false,
                scouting: false,
                guards: 0,
                smart_comp: false,
                threat_targets: false,
                mex_danger: i64::MAX,
                mex_range: 160,
                mex_upgrade_income: 35,
                fac_upgrade: false,
                team_coord: false,
                tech: false,
                air: false,
                naval: false,
                ..normal
            },
            // Phase 52: hard keeps hard19's economy / APM but fights like a committed
            // normal (Phase 19 hard over-retreated and bled units into raids).
            Difficulty::Hard => Self {
                attack_ratio: 110,
                retreat_ratio: 180,
                min_attack_power: 900,
                raids: 1,
                mex_danger: 150,
                tech_push: true,
                mex_parallel: 2,
                eng_max: 14,
                max_land_fac: 8,
                eng_parallel: 3,
                energy_ratio: 5,
                max_pgen_batch: 2,
                energy_margin: 100,
                ..Self::of(Difficulty::Hard19)
            },
            Difficulty::Hard19 => Self {
                think: 10,
                max_cmds: 40,
                reorder: 20,
                eng_base: 4,
                eng_max: 10,
                eng_parallel: 2,
                max_land_fac: 6,
                fac_per_income: 4,
                queue_opening: 3,
                energy_margin: 115,
                energy_ratio: 7,
                max_pgen_batch: 4,
                attack_ratio: 130,
                retreat_ratio: 130,
                min_attack_power: 1200,
                micro: true,
                raids: 5,
                guards: 3,
                mex_danger: 30,
                mex_upgrade_income: 14,
                ..normal
            },
            Difficulty::Normal | Difficulty::Adaptive | Difficulty::Legacy => normal,
        }
    }
}

// ---------------------------------------------------------------------------
// Public AI handle
// ---------------------------------------------------------------------------

/// Skirmish AI for one army. `new` gives [`Difficulty::Normal`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptedAi {
    pub army: ArmyId,
    pub difficulty: Difficulty,
    brain: Brain,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Brain {
    Legacy(LegacyAi),
    Smart(Box<Smart>),
}

impl ScriptedAi {
    pub fn new(army: ArmyId) -> Self {
        Self::with_difficulty(army, Difficulty::Normal)
    }

    pub fn with_difficulty(army: ArmyId, difficulty: Difficulty) -> Self {
        let brain = match difficulty {
            Difficulty::Legacy => Brain::Legacy(LegacyAi::new(army)),
            d => Brain::Smart(Box::new(Smart::new(army, d))),
        };
        Self { army, difficulty, brain }
    }

    /// Difficulty currently in effect (differs from `difficulty` only for `adaptive`).
    #[allow(dead_code)]
    pub fn effective_difficulty(&self) -> Difficulty {
        match &self.brain {
            Brain::Legacy(_) => Difficulty::Legacy,
            Brain::Smart(s) => s.eff,
        }
    }

    pub fn think(&mut self, w: &mut Skirmish) {
        let cmds = self.decide(w);
        for c in cmds {
            w.issue(self.army, c);
        }
    }

    /// Commands for this tick (empty on non-decision ticks).
    pub fn decide(&mut self, w: &Skirmish) -> Vec<Command> {
        match &mut self.brain {
            Brain::Legacy(l) => l.decide(w),
            Brain::Smart(s) => s.decide(w),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn d2(a: (i64, i64), b: (i64, i64)) -> i128 {
    let dx = (a.0 - b.0) as i128;
    let dy = (a.1 - b.1) as i128;
    dx * dx + dy * dy
}

fn sq(r: i64) -> i128 {
    (r as i128) * (r as i128)
}

fn dist(a: (i64, i64), b: (i64, i64)) -> i64 {
    isqrt(d2(a, b))
}

fn pos(u: &Unit) -> (i64, i64) {
    (u.pos.x, u.pos.y)
}

/// Point `len` along the segment from `a` toward `b`.
fn toward(a: (i64, i64), b: (i64, i64), len: i64) -> (i64, i64) {
    let d = dist(a, b);
    if d == 0 {
        return a;
    }
    (a.0 + (b.0 - a.0) * len / d, a.1 + (b.1 - a.1) * len / d)
}

fn clamp_map(w: &Skirmish, p: (i64, i64)) -> (i64, i64) {
    let (sx, sy) = w.size();
    (p.0.clamp(U, (sx - U).max(U)), p.1.clamp(U, (sy - U).max(U)))
}

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn has(w: &Skirmish, id: &str) -> bool {
    !id.is_empty() && w.catalog().get(id).is_some()
}

/// One structure want: role, blueprint, wanted count, site anchor, search ring.
type WantRow<'a> = (Role, &'a String, u32, Option<(i64, i64)>, i64);

/// Phase 30: the air staging platform faction `f` can build (data-driven roster).
fn staging_type(w: &Skirmish, f: &Faction) -> Option<String> {
    let cat = w.catalog();
    let fi = cat.factions.iter().position(|x| x.key == f.key)?;
    cat.types
        .values()
        .filter(|t| t.role == Role::AirStaging && cat.roster.playable.get(fi).is_some_and(|p| p.contains(&t.id)))
        .map(|t| t.id.clone())
        .min()
}

/// Whether builder `b` may place `ty` (fixed faction structures, or a roster option of its
/// own blueprint — the same test `Command::Build` applies).
fn builder_may_build(w: &Skirmish, f: &Faction, b: &Unit, ty: &str) -> bool {
    let cat = w.catalog();
    cat.factions.iter().position(|x| x.key == f.key).is_some_and(|fi| cat.faction_can_build(fi, ty))
        || cat.builder_can_build(&b.type_id, ty)
}

fn role_of(w: &Skirmish, id: &str) -> Option<Role> {
    w.catalog().get(id).map(|t| t.role)
}

fn radius_of(w: &Skirmish, id: &str) -> i64 {
    w.catalog().get(id).map(|t| t.radius).unwrap_or(2 * U)
}

fn mass_index(w: &Skirmish, p: (i64, i64)) -> Option<usize> {
    w.mass_points().iter().position(|&m| m == p)
}

fn hydro_index(w: &Skirmish, p: (i64, i64)) -> Option<usize> {
    w.hydro_points().iter().position(|&h| h == p)
}

/// Catalog key for `id`, matched case-insensitively.
fn resolve_type<'a>(w: &'a Skirmish, id: &str) -> Option<&'a str> {
    let types = &w.catalog().types;
    if let Some((k, _)) = types.get_key_value(id) {
        return Some(k.as_str());
    }
    types.keys().find(|k| k.eq_ignore_ascii_case(id)).map(String::as_str)
}

/// The type a unit upgrades into, if it exists in the catalog.
fn upgrade_target(w: &Skirmish, u: &Unit) -> Option<String> {
    let up = w.catalog().get(&u.type_id)?.upgrades_to.as_deref()?;
    resolve_type(w, up).map(str::to_string)
}

fn is_engineer_type(f: &Faction, id: &str) -> bool {
    id == f.engineer || (!f.t2_engineer.is_empty() && id == f.t2_engineer) || (!f.t3_engineer.is_empty() && id == f.t3_engineer)
}

fn is_idle(u: &Unit) -> bool {
    u.orders.is_empty() && u.build.is_none()
}

fn upgrading(u: &Unit) -> bool {
    matches!(u.orders.front(), Some(UnitOrder::Upgrade(_)))
}

fn centroid(units: &[&Unit]) -> Option<(i64, i64)> {
    if units.is_empty() {
        return None;
    }
    let n = units.len() as i64;
    let (sx, sy) = units.iter().fold((0i64, 0i64), |(x, y), u| (x + u.pos.x, y + u.pos.y));
    Some((sx / n, sy / n))
}

/// Unit class used by composition choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Line,
    Fast,
    Artillery,
    AntiAir,
}

fn kind_of(t: &UnitType) -> Kind {
    let aa = t.weapons.iter().any(|w| w.anti_air);
    let surf = t.weapons.iter().any(|w| w.anti_surface);
    if aa && !surf {
        Kind::AntiAir
    } else if t.tier == 1 && t.max_range() >= 28 * U {
        Kind::Artillery
    } else if t.tier == 1 && t.speed_per_tick >= 380 && t.max_hp < 150 * U {
        Kind::Fast
    } else {
        Kind::Line
    }
}

fn eco_value(w: &Skirmish, u: &Unit) -> i32 {
    let cost = w.catalog().get(&u.type_id).map(|t| t.cost_mass / SCALE).unwrap_or(0) as i32;
    match u.role {
        Role::Extractor => 150 + cost / 4,
        Role::Engineer => 80,
        Role::Generator | Role::Hydro => 60 + cost / 8,
        Role::Factory | Role::AirFactory | Role::NavalFactory => 100 + cost / 10,
        Role::MassStorage | Role::EnergyStorage | Role::Radar => 30,
        _ => 0,
    }
}

/// Deterministic focus enemy shared by a whole team: the alive enemy army whose start is
/// closest (summed) to the team's alive starts, ties by lower army id.
fn focus_enemy(w: &Skirmish, me: ArmyId) -> Option<(ArmyId, (i64, i64))> {
    let team = w.army(me)?.team;
    let mates: Vec<(i64, i64)> = w
        .armies()
        .iter()
        .filter(|a| a.team == team && !a.defeated)
        .map(|a| a.start)
        .collect();
    let mut best: Option<(i64, ArmyId, (i64, i64))> = None;
    for a in w.armies() {
        if a.defeated || a.team == team {
            continue;
        }
        let s: i64 = mates.iter().map(|&m| dist(m, a.start)).sum();
        if best.is_none_or(|(bs, bid, _)| s < bs || (s == bs && a.id < bid)) {
            best = Some((s, a.id, a.start));
        }
    }
    best.map(|(_, id, p)| (id, p))
}

/// Nearest unblocked cell centre of `grid` within `max_r` of `p` (optionally inside `reach`).
fn nearest_open_cell(grid: &PathGrid, p: (i64, i64), max_r: i64, reach: Option<&Reach>) -> Option<(i64, i64)> {
    let cs = grid.cell_size_milli.max(1);
    let r = max_r / cs + 1;
    let (cx, cz) = grid.world_to_cell(p.0, p.1);
    let mut best: Option<(i128, (i64, i64))> = None;
    for z in (cz as i64 - r).max(0)..=(cz as i64 + r).min(grid.height as i64 - 1) {
        for x in (cx as i64 - r).max(0)..=(cx as i64 + r).min(grid.width as i64 - 1) {
            if grid.is_blocked(x as u32, z as u32) {
                continue;
            }
            if reach.is_some_and(|rc| !rc.get(x as u32, z as u32)) {
                continue;
            }
            let c = grid.cell_to_world(x as u32, z as u32);
            let d = d2(c, p);
            if d <= sq(max_r) && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, c));
            }
        }
    }
    best.map(|(_, c)| c)
}

// ---------------------------------------------------------------------------
// Per-think state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
    Army,
    Guard,
    Scout,
    Raid,
    AirDef,
    AirStrike,
    Naval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Gather,
    Attack,
    Retreat,
    Defend,
}

/// Build orders issued to a builder (counted as planned until the sim applies them).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Job {
    tick: u64,
    items: Vec<(Role, (i64, i64))>,
}

#[derive(Debug, Clone, Copy)]
struct Contact {
    id: UnitId,
    #[allow(dead_code)]
    army: ArmyId,
    pos: (i64, i64),
    role: Role,
    air: bool,
    structure: bool,
    power: Power,
    eco: i32,
}

#[derive(Debug, Clone, Copy, Default)]
struct BaseThreat {
    power: i64,
    pos: Option<(i64, i64)>,
    air_power: i64,
    air_pos: Option<(i64, i64)>,
    ally_power: i64,
    ally_pos: Option<(i64, i64)>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Wants {
    pgens: i64,
    land: u32,
    air: u32,
    naval: u32,
    pd: u32,
    aa: u32,
    radar: u32,
    estor: u32,
    shield: u32,
    tml: u32,
    exp: u32,
    /// Phase 30: air staging platforms.
    staging: u32,
    eng: usize,
}

enum Work {
    Build(Vec<(String, (i64, i64))>),
    Reclaim((i64, i64)),
}

struct View<'a> {
    w: &'a Skirmish,
    f: &'a Faction,
    p: Profile,
    tick: u64,
    start: (i64, i64),
    focus: Option<(ArmyId, (i64, i64))>,
    enemy_starts: Vec<(ArmyId, (i64, i64))>,
    ally_starts: Vec<(ArmyId, (i64, i64))>,
    gather: (i64, i64),
    mass: i64,
    energy: i64,
    mass_cap: i64,
    energy_cap: i64,
    inc_m: i64,
    inc_e: i64,
    dem_m: i64,
    dem_e: i64,
    acu: Option<&'a Unit>,
    engineers: Vec<&'a Unit>,
    factories: Vec<&'a Unit>,
    mexes: Vec<&'a Unit>,
    combat: Vec<&'a Unit>,
    silos: Vec<&'a Unit>,
    have: [u32; ROLES],
    planned: [u32; ROLES],
    contacts: Vec<Contact>,
    reserved: Vec<(i64, i64, i64)>,
    tgt_mass: BTreeSet<usize>,
    tgt_hydro: BTreeSet<usize>,
    eng_queued: usize,
    want: Wants,
    site_budget: usize,
}

impl View<'_> {
    fn count(&self, r: Role) -> u32 {
        self.have[r.code() as usize] + self.planned[r.code() as usize]
    }
    /// Mass income per second.
    fn inc_m_s(&self) -> i64 {
        self.inc_m / 100
    }
    fn mass_float(&self) -> bool {
        self.mass >= 200 * U && self.mass * 10 >= self.mass_cap * 6
    }
    fn mass_stall(&self) -> bool {
        self.mass < 30 * U && self.dem_m > self.inc_m
    }
}

struct Out {
    cmds: Vec<Command>,
    budget: usize,
}

impl Out {
    fn room(&self) -> bool {
        self.cmds.len() < self.budget
    }
    fn push(&mut self, c: Command) -> bool {
        if !self.room() {
            return false;
        }
        self.cmds.push(c);
        true
    }
}

// ---------------------------------------------------------------------------
// The Phase 19 brain
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct Smart {
    army: ArmyId,
    level: Difficulty,
    eff: Difficulty,
    init: bool,
    rng: u64,
    threat: Option<ThreatMap>,
    amph: Option<Reach>,
    land: Option<Reach>,
    naval_home: Option<(i64, i64)>,
    naval_enemy: Vec<(ArmyId, (i64, i64))>,
    tasks: BTreeMap<UnitId, Task>,
    last_cmd: BTreeMap<UnitId, u64>,
    jobs: BTreeMap<UnitId, Job>,
    fac_open: usize,
    phase: Phase,
    phase_tick: u64,
    target: Option<(i64, i64)>,
    attack_power: i64,
    waves: u32,
    raid_target: Option<(i64, i64)>,
    raid_cursor: usize,
    scout_route: Vec<(i64, i64)>,
    scout_idx: usize,
    scout_lost_tick: u64,
    enemy_acu: BTreeMap<ArmyId, ((i64, i64), u64)>,
    enemy_air_seen: i32,
    enemy_def_seen: i32,
    base_hit_tick: u64,
    base_hit_peak: i32,
    last_adapt: u64,
    /// Phase 52 tech push: factory held back for its upgrade.
    upgrade_reserve: Option<UnitId>,
}

impl Smart {
    fn new(army: ArmyId, level: Difficulty) -> Self {
        Self {
            army,
            level,
            eff: if level == Difficulty::Adaptive { Difficulty::Normal } else { level },
            init: false,
            rng: 0,
            threat: None,
            amph: None,
            land: None,
            naval_home: None,
            naval_enemy: Vec::new(),
            tasks: BTreeMap::new(),
            last_cmd: BTreeMap::new(),
            jobs: BTreeMap::new(),
            fac_open: 0,
            phase: Phase::Gather,
            phase_tick: 0,
            target: None,
            attack_power: 0,
            waves: 0,
            raid_target: None,
            raid_cursor: 0,
            scout_route: Vec::new(),
            scout_idx: 0,
            scout_lost_tick: 0,
            enemy_acu: BTreeMap::new(),
            enemy_air_seen: 0,
            enemy_def_seen: 0,
            base_hit_tick: 0,
            base_hit_peak: 0,
            last_adapt: 0,
            upgrade_reserve: None,
        }
    }

    fn rand(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix(self.rng)
    }

    fn recent(&self, id: UnitId, tick: u64, window: u64) -> bool {
        self.last_cmd.get(&id).is_some_and(|&t| tick < t + window)
    }

    fn mark(&mut self, ids: &[UnitId], tick: u64) {
        for id in ids {
            self.last_cmd.insert(*id, tick);
        }
    }

    fn tm(&self) -> &ThreatMap {
        self.threat.as_ref().expect("threat map initialised")
    }

    fn set_phase(&mut self, p: Phase, tick: u64) {
        if self.phase != p {
            self.phase = p;
            self.phase_tick = tick;
        }
    }

    fn initialize(&mut self, w: &Skirmish) {
        self.init = true;
        self.rng = splitmix(w.seed() ^ ((self.army as u64 + 1) << 40) ^ 0xA15E_ED00);
        self.threat = Some(ThreatMap::new(w.size()));
        let start = w.army(self.army).map(|a| a.start).unwrap_or((0, 0));
        self.amph = Some(Reach::flood(w.domain_grid(NavDomain::Amphibious), start));
        self.land = Some(Reach::flood(w.domain_grid(NavDomain::Land), start));
        let ng = w.domain_grid(NavDomain::Naval);
        if ng.blocked_count() < (ng.width as usize) * (ng.height as usize)
            && let Some(home) = nearest_open_cell(ng, start, 90 * U, None)
        {
            let water = Reach::flood(ng, home);
            if water.count >= 150 {
                self.naval_home = Some(home);
                for a in w.armies() {
                    if w.are_allies(self.army, a.id) {
                        continue;
                    }
                    if let Some(p) = nearest_open_cell(ng, a.start, 160 * U, Some(&water)) {
                        self.naval_enemy.push((a.id, p));
                    }
                }
            }
        }
    }

    /// `adaptive`: re-evaluate every minute from the public kill/loss score.
    fn adapt(&mut self, w: &Skirmish) {
        if self.level != Difficulty::Adaptive {
            return;
        }
        let tick = w.tick();
        if tick < 1_800 || tick < self.last_adapt + 600 {
            return;
        }
        self.last_adapt = tick;
        let Some(a) = w.army(self.army) else { return };
        let ratio = (a.kills as i64 + 3) * 100 / (a.units_lost as i64 + 3);
        self.eff = if ratio < 70 {
            Difficulty::Hard
        } else if ratio > 170 {
            Difficulty::Easy
        } else {
            Difficulty::Normal
        };
    }

    fn decide(&mut self, w: &Skirmish) -> Vec<Command> {
        if w.result().is_some() {
            return Vec::new();
        }
        let tick = w.tick();
        let think = Profile::of(self.eff).think;
        if (tick + self.army as u64 * 3) % think != 0 {
            return Vec::new();
        }
        let Some(me) = w.army(self.army) else {
            return Vec::new();
        };
        if me.defeated {
            return Vec::new();
        }
        let Some(f) = w.faction(self.army) else {
            return Vec::new();
        };
        if !self.init {
            self.initialize(w);
        }
        self.adapt(w);
        let p = Profile::of(self.eff);
        self.prune(w, tick);
        let mut v = self.view(w, f, p);
        self.update_intel(&mut v);
        self.compute_wants(&mut v);
        self.assign_tasks(&v);
        let bt = self.base_threat(&v);
        let mut out = Out { cmds: Vec::new(), budget: p.max_cmds };

        self.plan_acu(&mut v, &bt, &mut out);
        self.plan_army(&v, &bt, &mut out);
        self.plan_factories(&mut v, &mut out);
        self.plan_engineers(&mut v, &mut out);
        self.plan_mex_upgrades(&v, &mut out);
        self.plan_air_naval(&v, &bt, &mut out);
        self.plan_raids(&v, &mut out);
        self.plan_scouts(&v, &mut out);
        self.plan_silos(&v, &mut out);
        out.cmds
    }

    fn prune(&mut self, w: &Skirmish, tick: u64) {
        let me = self.army;
        let alive = |id: &UnitId| w.unit(*id).is_some_and(|u| u.army == me);
        let scout_died = self
            .tasks
            .iter()
            .any(|(id, t)| *t == Task::Scout && !alive(id));
        if scout_died {
            self.scout_lost_tick = tick;
        }
        self.tasks.retain(|id, _| alive(id));
        self.last_cmd.retain(|id, _| alive(id));
        self.jobs.retain(|id, j| alive(id) && tick < j.tick + 600);
    }

    // ----- perception ---------------------------------------------------------

    fn view<'a>(&self, w: &'a Skirmish, f: &'a Faction, p: Profile) -> View<'a> {
        let me = self.army;
        let a = w.army(me).expect("army checked");
        let start = a.start;
        let mut enemy_starts = Vec::new();
        let mut ally_starts = Vec::new();
        for x in w.armies() {
            if x.defeated || x.id == me {
                continue;
            }
            if w.are_allies(me, x.id) {
                ally_starts.push((x.id, x.start));
            } else {
                enemy_starts.push((x.id, x.start));
            }
        }
        let focus = focus_enemy(w, me);
        let gather = match focus {
            Some((_, fs)) => {
                let g = clamp_map(w, toward(start, fs, GATHER_DIST));
                if self.land.as_ref().is_none_or(|r| r.near(g, 1)) { g } else { start }
            }
            None => start,
        };
        let mut v = View {
            w,
            f,
            p,
            tick: w.tick(),
            start,
            focus,
            enemy_starts,
            ally_starts,
            gather,
            mass: a.mass,
            energy: a.energy,
            mass_cap: a.mass_cap,
            energy_cap: a.energy_cap,
            inc_m: a.income_mass,
            inc_e: a.income_energy,
            dem_m: 0,
            dem_e: 0,
            acu: None,
            engineers: Vec::new(),
            factories: Vec::new(),
            mexes: Vec::new(),
            combat: Vec::new(),
            silos: Vec::new(),
            have: [0; ROLES],
            planned: [0; ROLES],
            contacts: Vec::new(),
            reserved: Vec::new(),
            tgt_mass: BTreeSet::new(),
            tgt_hydro: BTreeSet::new(),
            eng_queued: 0,
            want: Wants::default(),
            site_budget: 900,
        };
        let cat = w.catalog();
        for u in w.units() {
            // Phase 31: pod drones are driven by their engineering station, not the AI.
            if u.army == me && u.drone_of.is_none() {
                v.have[u.role.code() as usize] += 1;
                match u.role {
                    Role::Commander => v.acu = Some(u),
                    Role::Engineer => v.engineers.push(u),
                    Role::Factory | Role::AirFactory | Role::NavalFactory => v.factories.push(u),
                    Role::Extractor => v.mexes.push(u),
                    Role::MissileSilo => v.silos.push(u),
                    Role::Mobile | Role::Experimental if !u.weapons.is_empty() => v.combat.push(u),
                    _ => {}
                }
                if let Some(j) = &u.build {
                    let (m, e) = j.project.cost_per_step();
                    v.dem_m += m;
                    v.dem_e += e;
                    if u.role.is_builder()
                        && let Some(r) = role_of(w, &j.unit_type)
                    {
                        v.planned[r.code() as usize] += 1;
                    }
                    if u.role.is_factory() && is_engineer_type(f, &j.unit_type) {
                        v.eng_queued += 1;
                    }
                }
                if u.role.is_factory() {
                    v.eng_queued += u.production.iter().filter(|t| is_engineer_type(f, t)).count();
                }
                if u.shield_on && u.shield_max_hp > 0 {
                    v.dem_e += cat.get(&u.type_id).map(|t| t.shield_energy_upkeep).unwrap_or(0);
                }
            }
            if u.role.is_builder() && w.are_allies(me, u.army) {
                for (k, o) in u.orders.iter().enumerate() {
                    if let UnitOrder::Build { unit_type, x, y } = o {
                        let role = role_of(w, unit_type);
                        v.reserved.push((*x, *y, radius_of(w, unit_type)));
                        match role {
                            Some(Role::Extractor) => {
                                if let Some(i) = mass_index(w, (*x, *y)) {
                                    v.tgt_mass.insert(i);
                                }
                            }
                            Some(Role::Hydro) => {
                                if let Some(i) = hydro_index(w, (*x, *y)) {
                                    v.tgt_hydro.insert(i);
                                }
                            }
                            _ => {}
                        }
                        if u.army == me
                            && !(k == 0 && u.build.is_some())
                            && let Some(r) = role
                        {
                            v.planned[r.code() as usize] += 1;
                        }
                    }
                }
            }
        }
        // Orders issued but not yet applied (lockstep input delay).
        for (id, job) in &self.jobs {
            if v.tick > job.tick + JOB_GRACE {
                continue;
            }
            let Some(u) = w.unit(*id) else { continue };
            if !is_idle(u) {
                continue;
            }
            for &(role, site) in &job.items {
                v.planned[role.code() as usize] += 1;
                v.reserved.push((site.0, site.1, 4 * U));
                if role == Role::Extractor
                    && let Some(i) = mass_index(w, site)
                {
                    v.tgt_mass.insert(i);
                }
                if role == Role::Hydro
                    && let Some(i) = hydro_index(w, site)
                {
                    v.tgt_hydro.insert(i);
                }
            }
        }
        v
    }

    fn update_intel(&mut self, v: &mut View) {
        let w = v.w;
        let me = self.army;
        let tick = v.tick;
        let Some(tm) = self.threat.as_mut() else { return };
        let dt = tick.saturating_sub(tm.last_update);
        tm.begin(tick);
        for u in w.units() {
            if w.are_allies(me, u.army) {
                tm.mark_eye(pos(u), u.vision_radius, u.radar_radius);
                if !u.weapons.is_empty() {
                    tm.add_ally(pos(u), ai_threat::unit_power(u));
                }
            }
        }
        tm.clear_observed();
        let (mut air_now, mut def_now) = (0i32, 0i32);
        for u in w.units() {
            if w.are_allies(me, u.army) {
                continue;
            }
            let p = pos(u);
            match tm.visibility(p) {
                VIS_SIGHT => {
                    let c = Contact {
                        id: u.id,
                        army: u.army,
                        pos: p,
                        role: u.role,
                        air: u.domain == NavDomain::Air,
                        structure: u.role.is_structure(),
                        power: ai_threat::unit_power(u),
                        eco: eco_value(w, u),
                    };
                    tm.add_enemy(p, c.power, c.air, c.structure, c.eco);
                    if c.air {
                        air_now = air_now.saturating_add(c.power.total().max(20));
                    }
                    if c.structure {
                        def_now = def_now.saturating_add(c.power.surface);
                    }
                    if u.role == Role::Commander {
                        self.enemy_acu.insert(u.army, (p, tick));
                    }
                    v.contacts.push(c);
                }
                VIS_RADAR => tm.add_blip(p),
                _ => {}
            }
        }
        self.enemy_air_seen = ai_threat::decay(self.enemy_air_seen, dt, 6_000).max(air_now);
        self.enemy_def_seen = ai_threat::decay(self.enemy_def_seen, dt, 6_000).max(def_now);
        self.enemy_acu
            .retain(|a, (_, t)| tick < *t + 3_000 && w.army(*a).is_some_and(|x| !x.defeated));
    }

    fn base_threat(&mut self, v: &View) -> BaseThreat {
        let mut bt = BaseThreat::default();
        let r2 = sq(DEFEND_RADIUS);
        let mut best: Option<(i128, (i64, i64))> = None;
        let mut air_best: Option<(i128, (i64, i64))> = None;
        for c in &v.contacts {
            let d = d2(c.pos, v.start);
            if d > r2 {
                continue;
            }
            if c.air {
                bt.air_power += c.power.total().max(20) as i64;
                if air_best.is_none_or(|(bd, _)| d < bd) {
                    air_best = Some((d, c.pos));
                }
            } else if c.power.surface > 0 || matches!(c.role, Role::Engineer | Role::Commander) {
                bt.power += (c.power.surface as i64).max(10);
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, c.pos));
                }
            }
        }
        bt.pos = best.map(|b| b.1);
        bt.air_pos = air_best.map(|b| b.1);
        if bt.power > 0 {
            self.base_hit_tick = v.tick;
            self.base_hit_peak = self.base_hit_peak.max(bt.power.min(i32::MAX as i64) as i32);
        } else if v.tick > self.base_hit_tick + 1_800 {
            self.base_hit_peak = 0;
        }
        if v.p.team_coord {
            for &(_, astart) in &v.ally_starts {
                let mut pw = 0i64;
                let mut near: Option<(i128, (i64, i64))> = None;
                for c in &v.contacts {
                    let d = d2(c.pos, astart);
                    if c.air || c.power.surface <= 0 || d > r2 {
                        continue;
                    }
                    pw += c.power.surface as i64;
                    if near.is_none_or(|(bd, _)| d < bd) {
                        near = Some((d, c.pos));
                    }
                }
                if pw > bt.ally_power {
                    bt.ally_power = pw;
                    bt.ally_pos = near.map(|n| n.1);
                }
            }
        }
        bt
    }

    fn free_mex_count(&self, v: &View) -> usize {
        let w = v.w;
        let r2 = sq(v.p.mex_range.saturating_mul(U));
        w.mass_points()
            .iter()
            .enumerate()
            .filter(|(i, m)| {
                w.mass_claim(*i).is_none()
                    && !v.tgt_mass.contains(i)
                    && d2(**m, v.start) <= r2
                    && self.amph.as_ref().is_none_or(|r| r.near(**m, 2))
            })
            .count()
    }

    fn compute_wants(&self, v: &mut View) {
        let w = v.w;
        let f = v.f;
        let p = v.p;
        let cat = w.catalog();
        let inc = v.inc_m_s();
        let mut want = Wants::default();

        // Energy: cover the live drain (with margin) and a mass-proportional target.
        let pgen_out = cat.get(&f.generator).map(|t| t.energy_per_tick).unwrap_or(2_000).max(1);
        let hydro_out = cat.get(&f.hydro).map(|t| t.energy_per_tick).unwrap_or(0);
        let planned_out = v.planned[Role::Generator.code() as usize] as i64 * pgen_out
            + v.planned[Role::Hydro.code() as usize] as i64 * hydro_out;
        let target = (v.dem_e * p.energy_margin / 100).max(v.inc_m * p.energy_ratio);
        let deficit = target - v.inc_e - planned_out;
        let full = v.energy_cap > 0 && v.energy * 10 >= v.energy_cap * 9;
        let mut pg = if deficit > 0 { (deficit + pgen_out - 1) / pgen_out } else { 0 };
        if full && v.inc_e >= v.dem_e {
            pg = 0;
        }
        if pg == 0 && v.energy * 10 < v.energy_cap && v.inc_e < v.dem_e && planned_out == 0 {
            pg = 1;
        }
        want.pgens = pg.min(p.max_pgen_batch);

        let lands = v.count(Role::Factory);
        want.land = ((1 + inc / p.fac_per_income.max(1)) as u32).clamp(1, p.max_land_fac as u32);
        if v.mass_float() && lands >= want.land {
            want.land = (lands + 1).min(p.max_land_fac as u32);
        }
        if p.air && has(w, &f.air_factory) && lands >= 1 && (self.enemy_air_seen > 0 || inc >= 18) {
            want.air = 1 + u32::from(p.micro && inc >= 45);
        }
        if p.naval && self.naval_home.is_some() && has(w, &f.naval_factory) && inc >= 10 && lands >= 2 {
            want.naval = 1 + u32::from(p.micro && inc >= 30);
        }
        // Phase 52 map awareness: when no enemy base is reachable by land (island maps),
        // scale air / naval production with income.
        let island = p.tech_push
            && !v.enemy_starts.is_empty()
            && self.land.as_ref().is_some_and(|r| v.enemy_starts.iter().all(|(_, s)| !r.near(*s, 2)));
        if island {
            if p.air && has(w, &f.air_factory) && lands >= 1 {
                want.air = (2 + (inc / 30) as u32).min(5);
            }
            if p.naval && self.naval_home.is_some() && has(w, &f.naval_factory) && lands >= 1 {
                want.naval = (2 + (inc / 30) as u32).min(5);
            }
        }
        if has(w, &f.point_defense) {
            let attacked = self.base_hit_tick > 0 && v.tick < self.base_hit_tick + 1_500;
            let base_inc = if p.micro { 10 } else if p.tech { 8 } else { 5 };
            let mut pd = u32::from(inc >= base_inc);
            if attacked {
                pd += 1 + (self.base_hit_peak / 300) as u32;
            }
            want.pd = pd.min(if p.tech { 5 } else { 2 });
        }
        if has(w, &f.anti_air) && self.enemy_air_seen > 0 && inc >= 3 {
            want.aa = (1 + self.enemy_air_seen as u32 / 250).min(if p.tech { 5 } else { 1 });
        }
        if p.tech && inc >= 6 && has(w, &f.radar) {
            want.radar = 1;
        }
        if p.overcharge && v.inc_e >= 6_000 && has(w, &f.energy_storage) {
            want.estor = 1;
        }
        if p.tech && inc >= 30 && has(w, &f.shield) {
            want.shield = 1 + u32::from(inc >= 60);
        }
        if p.micro && p.tech && inc >= 35 && has(w, &f.tac_missile) {
            want.tml = 1;
        }
        // Phase 30: one staging platform (two with a big air force) once aircraft fly.
        let aircraft = w
            .units()
            .filter(|u| u.army == self.army && u.alive && u.domain == NavDomain::Air && u.speed > 0)
            .count() as u32;
        if aircraft >= 4 && inc >= 8 && staging_type(w, v.f).is_some() {
            want.staging = 1 + u32::from(aircraft >= 20);
        }
        if p.tech && has(w, &f.experimental) && inc >= if p.micro { 70 } else { 90 } && v.mass_float() {
            want.exp = v.have[Role::Experimental.code() as usize] + 1;
        }
        let free = self.free_mex_count(v);
        want.eng = (p.eng_base + free.div_ceil(3) + (inc / 10) as usize).clamp(p.eng_base, p.eng_max);
        v.want = want;
    }

    fn assign_tasks(&mut self, v: &View) {
        let p = v.p;
        let mut n = [0usize; 7];
        let slot = |t: Task| t as usize;
        for t in self.tasks.values() {
            n[slot(*t)] += 1;
        }
        let attacked = self.base_hit_tick > 0 && v.tick < self.base_hit_tick + 1_200;
        let guard_target = if p.guards == 0 { 0 } else { p.guards + if attacked { 2 } else { 0 } };
        let naval_types = &v.f.naval_units;
        for u in &v.combat {
            if self.tasks.contains_key(&u.id) {
                continue;
            }
            let fast_land = u.tier == 1 && u.speed >= 380 && u.max_hp < 150 * U;
            let t = if u.domain == NavDomain::Air {
                if u.can_fire_surface && p.raids > 0 && u.speed >= AIR_STRIKE_SPEED {
                    Task::AirStrike
                } else {
                    Task::AirDef
                }
            } else if u.domain == NavDomain::Naval || naval_types.iter().any(|t| *t == u.type_id) {
                Task::Naval
            } else if u.role == Role::Experimental {
                Task::Army
            } else if p.scouting
                && n[slot(Task::Scout)] == 0
                && fast_land
                && v.tick >= self.scout_lost_tick + if p.micro { 900 } else { 1_800 }
            {
                Task::Scout
            } else if p.raids > 0 && fast_land && n[slot(Task::Raid)] < p.raids && v.tick >= RAID_START {
                Task::Raid
            } else if n[slot(Task::Guard)] < guard_target {
                Task::Guard
            } else {
                Task::Army
            };
            n[slot(t)] += 1;
            self.tasks.insert(u.id, t);
        }
    }

    fn task_units<'a>(&self, v: &View<'a>, t: Task) -> Vec<&'a Unit> {
        v.combat
            .iter()
            .copied()
            .filter(|u| self.tasks.get(&u.id) == Some(&t))
            .collect()
    }

    // ----- placement ----------------------------------------------------------

    /// Ring search for a free, reachable, unreserved site around `anchor`.
    fn find_site(&self, v: &mut View, type_id: &str, anchor: (i64, i64), max_ring: i64) -> Option<(i64, i64)> {
        let w = v.w;
        let t = w.catalog().get(type_id)?;
        let naval = t.role == Role::NavalFactory;
        let r = t.radius;
        let step = ((r * 2 + 2 * U).max(4 * U) / U) * U;
        for ring in 0..=max_ring {
            for dy in -ring..=ring {
                for dx in -ring..=ring {
                    if dx.abs() != ring && dy.abs() != ring {
                        continue;
                    }
                    let (x, y) = (anchor.0 + dx * step, anchor.1 + dy * step);
                    if v.reserved.iter().any(|&(rx, ry, rr)| d2((rx, ry), (x, y)) < sq(rr + r + U)) {
                        continue;
                    }
                    if !naval && self.amph.as_ref().is_some_and(|a| !a.near((x, y), 1)) {
                        continue;
                    }
                    if v.site_budget == 0 {
                        return None;
                    }
                    v.site_budget -= 1;
                    if w.site_is_free(type_id, x, y) {
                        return Some((x, y));
                    }
                }
            }
        }
        None
    }

    fn pgen_anchor(&self, v: &View) -> (i64, i64) {
        let lands: Vec<&&Unit> = v.factories.iter().filter(|f| f.role != Role::NavalFactory).collect();
        if lands.is_empty() {
            return v.start;
        }
        let i = v.have[Role::Generator.code() as usize] as usize / 3 % lands.len();
        pos(lands[i])
    }

    fn forward(&self, v: &View, len: i64) -> (i64, i64) {
        match v.focus {
            Some((_, fs)) => clamp_map(v.w, toward(v.start, fs, len)),
            None => v.start,
        }
    }

    fn best_mex(&self, v: &View, from: (i64, i64), is_acu: bool, near: Option<((i64, i64), i64)>) -> Option<(usize, (i64, i64))> {
        let w = v.w;
        let tm = self.tm();
        let max_r = if is_acu { ACU_MEX_RANGE } else { v.p.mex_range.saturating_mul(U) };
        let enemy = v.focus.map(|f| f.1);
        let mut best: Option<(i64, usize, (i64, i64))> = None;
        for (i, &m) in w.mass_points().iter().enumerate() {
            if w.mass_claim(i).is_some() || v.tgt_mass.contains(&i) || d2(m, v.start) > sq(max_r) {
                continue;
            }
            if let Some((c, r)) = near
                && d2(m, c) > sq(r)
            {
                continue;
            }
            if self.amph.as_ref().is_some_and(|r| !r.near(m, 2)) {
                continue;
            }
            let danger = tm.surface_threat(m, U);
            if danger > v.p.mex_danger {
                continue;
            }
            let mut score = dist(from, m) + dist(v.start, m) / 2 + danger * U;
            if let Some(e) = enemy
                && d2(m, e) < d2(m, v.start)
            {
                score += if v.p.micro { 60 * U } else { 150 * U };
            }
            if best.is_none_or(|(bs, bi, _)| score < bs || (score == bs && i < bi)) {
                best = Some((score, i, m));
            }
        }
        best.map(|(_, i, m)| (i, m))
    }

    fn best_hydro(&self, v: &View, from: (i64, i64), max_r: i64) -> Option<(usize, (i64, i64))> {
        let w = v.w;
        let tm = self.tm();
        let mut best: Option<(i128, usize, (i64, i64))> = None;
        for (i, &h) in w.hydro_points().iter().enumerate() {
            if w.hydro_claim(i).is_some() || v.tgt_hydro.contains(&i) || d2(h, v.start) > sq(max_r) {
                continue;
            }
            if self.amph.as_ref().is_some_and(|r| !r.near(h, 2)) || tm.surface_threat(h, U) > v.p.mex_danger {
                continue;
            }
            let d = d2(from, h);
            if best.is_none_or(|(bd, bi, _)| d < bd || (d == bd && i < bi)) {
                best = Some((d, i, h));
            }
        }
        best.map(|(_, i, h)| (i, h))
    }

    fn step_counts(v: &View) -> StepCounts {
        let c = |r: Role| v.count(r);
        StepCounts([
            c(Role::Extractor),
            c(Role::Generator),
            c(Role::Hydro),
            c(Role::Factory),
            c(Role::AirFactory),
            c(Role::NavalFactory),
            c(Role::EnergyStorage),
            c(Role::Radar),
            c(Role::PointDefense),
        ])
    }

    fn step_type<'a>(f: &'a Faction, s: Step) -> &'a str {
        match s {
            Step::Mex => &f.extractor,
            Step::Pgen => &f.generator,
            Step::Hydro => &f.hydro,
            Step::LandFactory => &f.factory,
            Step::AirFactory => &f.air_factory,
            Step::NavalFactory => &f.naval_factory,
            Step::EnergyStorage => &f.energy_storage,
            Step::Radar => &f.radar,
            Step::PointDefense => &f.point_defense,
        }
    }

    fn step_available(&self, v: &View, s: Step, from: (i64, i64)) -> bool {
        if !has(v.w, Self::step_type(v.f, s)) {
            return false;
        }
        match s {
            Step::Mex => self.best_mex(v, from, true, None).is_some(),
            Step::Hydro => self.best_hydro(v, from, 90 * U).is_some(),
            Step::NavalFactory => self.naval_home.is_some(),
            _ => true,
        }
    }

    /// Concrete type and site for a build step (reserving it for this think).
    fn step_site(&self, v: &mut View, s: Step, from: (i64, i64)) -> Option<(String, (i64, i64))> {
        let ty = Self::step_type(v.f, s).to_string();
        let site = match s {
            Step::Mex => {
                let (i, m) = self.best_mex(v, from, true, None)?;
                v.tgt_mass.insert(i);
                m
            }
            Step::Hydro => {
                let (i, h) = self.best_hydro(v, from, 90 * U)?;
                v.tgt_hydro.insert(i);
                h
            }
            Step::Pgen | Step::EnergyStorage => {
                let a = self.pgen_anchor(v);
                self.find_site(v, &ty, a, 10)?
            }
            Step::NavalFactory => self.find_site(v, &ty, self.naval_home?, 16)?,
            Step::PointDefense => {
                let a = self.forward(v, 22 * U);
                self.find_site(v, &ty, a, 8)?
            }
            _ => self.find_site(v, &ty, v.start, 14)?,
        };
        v.reserved.push((site.0, site.1, radius_of(v.w, &ty)));
        Some((ty, site))
    }

    /// Dynamic (post-opening) builder job.
    fn pick_job(&self, v: &mut View, b: &Unit, is_acu: bool) -> Option<Work> {
        let w = v.w;
        let f = v.f;
        let bp = pos(b);
        // 1. Energy.
        if v.want.pgens > 0 {
            if has(w, &f.hydro)
                && let Some((i, h)) = self.best_hydro(v, bp, if is_acu { 50 * U } else { 160 * U })
            {
                v.tgt_hydro.insert(i);
                v.want.pgens -= 5;
                return Some(Work::Build(vec![(f.hydro.clone(), h)]));
            }
            let t2 = b.tier >= 2 && has(w, &f.t2_pgen) && v.inc_m >= 1_500;
            let ty = if t2 { f.t2_pgen.clone() } else { f.generator.clone() };
            if has(w, &ty) {
                let a = self.pgen_anchor(v);
                if let Some(s) = self.find_site(v, &ty, a, 10) {
                    v.want.pgens -= if t2 { 20 } else { 1 };
                    return Some(Work::Build(vec![(ty, s)]));
                }
            }
        }
        // 2. Mass.
        if has(w, &f.extractor)
            && let Some((i, m)) = self.best_mex(v, bp, is_acu, None)
        {
            v.tgt_mass.insert(i);
            let mut items = vec![(f.extractor.clone(), m)];
            if v.p.queue_opening > 1
                && !is_acu
                && let Some((j, m2)) = self.best_mex(v, m, false, Some((m, 40 * U)))
            {
                v.tgt_mass.insert(j);
                items.push((f.extractor.clone(), m2));
            }
            return Some(Work::Build(items));
        }
        // 3. Structures by priority.
        let pd_anchor = self.forward(v, 22 * U);
        let pg_anchor = self.pgen_anchor(v);
        let staging_ty = staging_type(w, f).unwrap_or_default();
        let wants: [WantRow; 11] = [
            (Role::Factory, &f.factory, v.want.land, Some(v.start), 14),
            (Role::AntiAir, &f.anti_air, v.want.aa, Some(v.start), 10),
            (Role::PointDefense, &f.point_defense, v.want.pd, Some(pd_anchor), 8),
            (Role::AirFactory, &f.air_factory, v.want.air, Some(v.start), 14),
            (Role::NavalFactory, &f.naval_factory, v.want.naval, self.naval_home, 16),
            (Role::Radar, &f.radar, v.want.radar, Some(v.start), 10),
            (Role::EnergyStorage, &f.energy_storage, v.want.estor, Some(pg_anchor), 10),
            (Role::Shield, &f.shield, v.want.shield, Some(v.start), 10),
            (Role::MissileSilo, &f.tac_missile, v.want.tml, Some(v.start), 12),
            (Role::Experimental, &f.experimental, v.want.exp, Some(v.start), 14),
            (Role::AirStaging, &staging_ty, v.want.staging, Some(v.start), 14),
        ];
        for (role, ty, want, anchor, ring) in wants {
            let Some(anchor) = anchor else { continue };
            if want == 0 || v.count(role) >= want || !has(w, ty) {
                continue;
            }
            let heavy = matches!(
                role,
                Role::Factory | Role::AirFactory | Role::NavalFactory | Role::Experimental | Role::Shield | Role::MissileSilo
            );
            if heavy && v.mass_stall() {
                continue;
            }
            if is_acu && matches!(role, Role::Experimental | Role::NavalFactory) {
                continue;
            }
            // Roster-derived types (e.g. air staging) need a builder of the right tech tier.
            if !builder_may_build(w, f, b, ty) {
                continue;
            }
            if let Some(s) = self.find_site(v, ty, anchor, ring) {
                return Some(Work::Build(vec![(ty.clone(), s)]));
            }
        }
        // 4. Reclaim battlefield wrecks.
        if !is_acu
            && v.mass * 10 < v.mass_cap * 6
            && let Some(wp) = w.reclaimable_mass_near(bp, 45 * U)
            && self.tm().surface_threat(wp, U) <= v.p.mex_danger
        {
            return Some(Work::Reclaim(wp));
        }
        None
    }

    fn issue_builds(&mut self, v: &mut View, builder: UnitId, items: Vec<(String, (i64, i64))>, out: &mut Out) {
        let w = v.w;
        let mut job = Job { tick: v.tick, items: Vec::new() };
        for (k, (ty, (x, y))) in items.into_iter().enumerate() {
            let Some(role) = role_of(w, &ty) else { break };
            let r = radius_of(w, &ty);
            if !out.push(Command::Build {
                builder,
                unit_type: ty,
                x,
                y,
                queue: k > 0,
            }) {
                break;
            }
            v.planned[role.code() as usize] += 1;
            v.reserved.push((x, y, r));
            job.items.push((role, (x, y)));
        }
        if !job.items.is_empty() {
            self.jobs.insert(builder, job);
            self.mark(&[builder], v.tick);
        }
    }

    /// Something worth assisting near `from`: big builds, factory upgrades, busy factories.
    fn assist_target(&self, v: &View, from: (i64, i64)) -> Option<UnitId> {
        if v.mass_stall() {
            return None;
        }
        let w = v.w;
        let mut best: Option<(u8, i128, UnitId)> = None;
        let mut consider = |prio: u8, u: &Unit| {
            let d = d2(from, pos(u));
            if best.is_none_or(|(bp, bd, bid)| (prio, d, u.id) < (bp, bd, bid)) {
                best = Some((prio, d, u.id));
            }
        };
        for b in v.engineers.iter().chain(v.acu.iter()) {
            if let Some(j) = &b.build
                && w.catalog().get(&j.unit_type).is_some_and(|t| t.build_work >= 15_000)
            {
                consider(0, b);
            }
        }
        for f in &v.factories {
            if upgrading(f) {
                consider(1, f);
            } else if f.build.is_some() || !f.production.is_empty() {
                consider(2, f);
            }
        }
        best.map(|b| b.2)
    }

    // ----- commander ----------------------------------------------------------

    fn plan_acu(&mut self, v: &mut View, bt: &BaseThreat, out: &mut Out) {
        let Some(acu) = v.acu else { return };
        let p = v.p;
        let tick = v.tick;
        let ap = pos(acu);
        let busy_oc = matches!(acu.orders.front(), Some(UnitOrder::Overcharge(_)));
        if busy_oc {
            return;
        }
        // Overcharge the most valuable enemy in reach.
        let reserve = if p.micro { 0 } else { 1_000 * U };
        if p.overcharge && (p.micro || bt.power > 0) && v.energy >= OVERCHARGE_ENERGY_COST + reserve {
            let reach = OVERCHARGE_RANGE + 8 * U;
            let mut best: Option<(i64, UnitId)> = None;
            for c in &v.contacts {
                if c.air || d2(c.pos, ap) > sq(reach) {
                    continue;
                }
                let val = if c.role == Role::Commander {
                    100_000
                } else {
                    c.power.surface as i64 + c.eco as i64 / 2
                };
                if val < 60 {
                    continue;
                }
                if best.is_none_or(|(bv, bid)| val > bv || (val == bv && c.id < bid)) {
                    best = Some((val, c.id));
                }
            }
            if let Some((_, t)) = best {
                if out.push(Command::Overcharge { unit: acu.id, target: t }) {
                    self.mark(&[acu.id], tick);
                }
                return;
            }
        }
        let away = d2(ap, v.start) > sq(15 * U);
        // Retreat when hurt and outgunned.
        if p.threat_targets && away && acu.hp * 100 < acu.max_hp * 45 {
            let tm = self.tm();
            let danger = tm.surface_threat(ap, 30 * U);
            let ally = tm.sum(&tm.ally, ap, 30 * U);
            let heading_home = matches!(acu.orders.front(), Some(UnitOrder::Move { x, y }) if d2((*x, *y), v.start) < sq(10 * U));
            if danger > 0 && danger * 100 > ally * 70 && !heading_home {
                if out.push(Command::Move { units: vec![acu.id], x: v.start.0, y: v.start.1, queue: false }) {
                    self.mark(&[acu.id], tick);
                }
                return;
            }
        }
        // Fight intruders close to the base.
        if let Some(ip) = bt.pos
            && d2(ip, v.start) <= sq(45 * U)
            && acu.build.is_none()
            && !self.recent(acu.id, tick, p.reorder)
            && !matches!(acu.orders.front(), Some(UnitOrder::AttackMove { .. }))
        {
            if out.push(Command::AttackMove { units: vec![acu.id], x: ip.0, y: ip.1, queue: false }) {
                self.mark(&[acu.id], tick);
            }
            return;
        }
        if !is_idle(acu) || self.recent(acu.id, tick, p.reorder) {
            return;
        }
        if d2(ap, v.start) > sq(ACU_LEASH) {
            if out.push(Command::Move { units: vec![acu.id], x: v.start.0, y: v.start.1, queue: false }) {
                self.mark(&[acu.id], tick);
            }
            return;
        }
        // Opening build order, then dynamic eco.
        let op = ai_build::opening(&v.f.key, p.sloppy_opening);
        let counts = Self::step_counts(v);
        let steps = {
            let vr: &View = v;
            ai_build::plan_steps(op.acu, &counts, |s| self.step_available(vr, s, ap), p.queue_opening)
        };
        let mut items = Vec::new();
        for s in steps {
            match self.step_site(v, s, ap) {
                Some(it) => items.push(it),
                None => break,
            }
        }
        if items.is_empty()
            && let Some(Work::Build(it)) = self.pick_job(v, acu, true)
        {
            items = it;
        }
        if !items.is_empty() {
            self.issue_builds(v, acu.id, items, out);
            return;
        }
        if let Some(t) = self.assist_target(v, ap)
            && out.push(Command::Assist { units: vec![acu.id], target: t, queue: false })
        {
            self.mark(&[acu.id], tick);
        }
    }

    // ----- production ---------------------------------------------------------

    fn engineer_for(v: &View, tier: u8) -> Option<String> {
        let f = v.f;
        if tier >= 3 && has(v.w, &f.t3_engineer) {
            Some(f.t3_engineer.clone())
        } else if tier >= 2 && has(v.w, &f.t2_engineer) {
            Some(f.t2_engineer.clone())
        } else if has(v.w, &f.engineer) {
            Some(f.engineer.clone())
        } else {
            None
        }
    }

    fn weighted(&mut self, items: &[(String, u32)]) -> Option<String> {
        let total: u64 = items.iter().map(|(_, w)| *w as u64).sum();
        if total == 0 {
            return None;
        }
        let mut r = self.rand() % total;
        for (id, w) in items {
            if r < *w as u64 {
                return Some(id.clone());
            }
            r -= *w as u64;
        }
        None
    }

    fn scout_type(v: &View) -> Option<String> {
        let cat = v.w.catalog();
        v.f.land_units
            .iter()
            .filter_map(|id| cat.get(id).map(|t| (id, t)))
            .max_by_key(|(id, t)| (t.speed_per_tick, std::cmp::Reverse((*id).clone())))
            .map(|(id, _)| id.clone())
    }

    fn land_choice(&mut self, v: &View, fac: &Unit) -> Option<String> {
        let f = v.f;
        let cat = v.w.catalog();
        let mut cands: Vec<(&String, &UnitType)> = Vec::new();
        let tiers: [(&Vec<String>, u8); 3] = [(&f.land_units, 1), (&f.t2_land_units, 2), (&f.t3_land_units, 3)];
        for (list, tier) in tiers {
            if fac.tier < tier {
                continue;
            }
            for id in list {
                if let Some(t) = cat.get(id)
                    && t.domain != NavDomain::Air
                    && t.domain != NavDomain::Naval
                {
                    cands.push((id, t));
                }
            }
        }
        if cands.is_empty() {
            return None;
        }
        if !v.p.smart_comp {
            let i = (self.rand() % cands.len() as u64) as usize;
            return Some(cands[i].0.clone());
        }
        let top = cands.iter().map(|(_, t)| t.tier).max().unwrap_or(1);
        let raiders = self.tasks.values().filter(|t| **t == Task::Raid).count();
        let raid_short = v.p.raids > 0 && raiders < v.p.raids && v.tick >= RAID_START;
        let items: Vec<(String, u32)> = cands
            .iter()
            .map(|(id, t)| {
                let k = kind_of(t);
                let mut wgt: u32 = match k {
                    Kind::Line if t.tier == top => 45,
                    Kind::Line => 20,
                    Kind::Artillery => 16,
                    Kind::Fast => 8,
                    Kind::AntiAir => 0,
                };
                if k == Kind::Artillery && self.enemy_def_seen > 250 {
                    wgt += 30;
                }
                if k == Kind::Fast && raid_short {
                    wgt += 40;
                }
                if self.enemy_air_seen > 0 && t.weapons.iter().any(|w| w.anti_air) {
                    wgt += 40;
                }
                ((*id).clone(), wgt)
            })
            .collect();
        self.weighted(&items)
    }

    fn air_choice(&mut self, v: &View, fac: &Unit) -> Option<String> {
        let cat = v.w.catalog();
        let mut interceptors: Vec<(&String, &UnitType)> = Vec::new();
        let mut strikers: Vec<(&String, &UnitType)> = Vec::new();
        for id in &v.f.air_units {
            let Some(t) = cat.get(id) else { continue };
            if t.tier > fac.tier || t.tier > 3 {
                continue;
            }
            let aa = t.weapons.iter().any(|w| w.anti_air);
            let surf = t.weapons.iter().any(|w| w.anti_surface);
            if aa && !surf {
                interceptors.push((id, t));
            } else if surf {
                strikers.push((id, t));
            }
        }
        let best_of = |l: &[(&String, &UnitType)]| {
            l.iter()
                .max_by_key(|(id, t)| (t.tier, std::cmp::Reverse((*id).clone())))
                .map(|(id, _)| (*id).clone())
        };
        let airdef = self.task_units(v, Task::AirDef);
        let my_aa: i64 = airdef.iter().map(|u| ai_threat::unit_power(u).aa as i64).sum();
        if self.enemy_air_seen > 0 && my_aa * 3 < self.enemy_air_seen as i64 * 4 {
            return best_of(&interceptors).or_else(|| best_of(&strikers));
        }
        let strikes = self.tasks.values().filter(|t| **t == Task::AirStrike).count();
        if v.p.raids > 0 && strikes < 8 {
            let fast: Vec<(&String, &UnitType)> =
                strikers.iter().copied().filter(|(_, t)| t.speed_per_tick >= AIR_STRIKE_SPEED).collect();
            if let Some(s) = best_of(&fast) {
                return Some(s);
            }
        }
        if airdef.len() < if v.p.micro { 4 } else { 3 } {
            return best_of(&interceptors).or_else(|| best_of(&strikers));
        }
        None
    }

    fn naval_choice(&mut self, v: &View, fac: &Unit) -> Option<String> {
        let navy = self.tasks.values().filter(|t| **t == Task::Naval).count();
        if navy >= 16 {
            return None;
        }
        let cat = v.w.catalog();
        let items: Vec<(String, u32)> = v
            .f
            .naval_units
            .iter()
            .filter_map(|id| cat.get(id).map(|t| (id, t)))
            .filter(|(_, t)| t.tier <= fac.tier && t.tier <= 3)
            .map(|(id, t)| (id.clone(), if t.tier == fac.tier { 60 } else { 40 }))
            .collect();
        self.weighted(&items)
    }

    fn pick_factory_upgrade(&self, v: &View) -> Option<(UnitId, String)> {
        let p = v.p;
        if !p.fac_upgrade || v.factories.iter().any(|f| upgrading(f)) || (v.want.pgens > 0 && !p.tech_push) {
            return None;
        }
        let inc = v.inc_m_s();
        let lands: Vec<&&Unit> = v.factories.iter().filter(|f| f.role == Role::Factory).collect();
        if lands.len() < 2 {
            return None;
        }
        let maxed = v.count(Role::Factory) >= v.want.land;
        if !(maxed || v.mass_float()) {
            return None;
        }
        let mut cands: Vec<&&Unit> = v
            .factories
            .iter()
            .filter(|f| p.tech_push || (is_idle(f) && f.production.is_empty()))
            .filter(|f| match (f.role, f.tier) {
                (Role::Factory, 1) => inc >= 15,
                (Role::Factory, 2) => inc >= 35,
                (Role::AirFactory, 1) => p.micro && inc >= 30,
                _ => false,
            })
            .collect();
        cands.sort_by_key(|f| (f.tier, f.role != Role::Factory, f.id));
        cands
            .into_iter()
            .find_map(|f| upgrade_target(v.w, f).map(|t| (f.id, t)))
    }

    fn plan_factories(&mut self, v: &mut View, out: &mut Out) {
        let tick = v.tick;
        let p = v.p;
        let op = ai_build::opening(&v.f.key, p.sloppy_opening);
        let mut upgrade = self.pick_factory_upgrade(v);
        if p.tech_push {
            // Keep one reserved factory until it has upgraded (or died).
            let still = self.upgrade_reserve.and_then(|id| v.factories.iter().find(|f| f.id == id && !upgrading(f)).map(|f| f.id));
            match (still, &upgrade) {
                (Some(id), _) => {
                    let f = v.factories.iter().find(|f| f.id == id).unwrap();
                    upgrade = upgrade_target(v.w, f).map(|t| (id, t));
                    if upgrade.is_none() {
                        self.upgrade_reserve = None;
                    }
                }
                (None, Some((id, _))) => self.upgrade_reserve = Some(*id),
                (None, None) => self.upgrade_reserve = None,
            }
        }
        let first_land = v.factories.iter().filter(|f| f.role == Role::Factory).map(|f| f.id).min();
        let mut eng_total = v.have[Role::Engineer.code() as usize] as usize + v.eng_queued;
        let mut eng_now = v.eng_queued;
        let facs = v.factories.clone();
        for fac in facs {
            if !out.room() {
                break;
            }
            let upg = upgrading(fac);
            if fac.rally.is_none() && !upg {
                let r = match fac.role {
                    Role::NavalFactory => self.naval_home.unwrap_or(pos(fac)),
                    _ => v.gather,
                };
                out.push(Command::Rally { factory: fac.id, x: r.0, y: r.1 });
            }
            if upg || !fac.production.is_empty() || fac.build.is_some() || !fac.orders.is_empty() {
                continue;
            }
            if self.recent(fac.id, tick, p.reorder) {
                continue;
            }
            if let Some((uid, ty)) = &upgrade
                && *uid == fac.id
            {
                if out.push(Command::Upgrade { unit: fac.id, target_type: ty.clone() }) {
                    self.mark(&[fac.id], tick);
                    self.upgrade_reserve = None;
                }
                continue;
            }
            let mut from_opening = false;
            let mut is_eng = false;
            let choice = match fac.role {
                Role::Factory => {
                    if Some(fac.id) == first_land && self.fac_open < op.factory.len() {
                        from_opening = true;
                        match op.factory[self.fac_open] {
                            FactoryStep::Engineer => {
                                is_eng = true;
                                Self::engineer_for(v, fac.tier)
                            }
                            FactoryStep::Scout => Self::scout_type(v),
                            FactoryStep::Line => self.land_choice(v, fac),
                        }
                    } else if eng_total < v.want.eng && eng_now < p.eng_parallel {
                        is_eng = true;
                        Self::engineer_for(v, fac.tier)
                    } else {
                        self.land_choice(v, fac)
                    }
                }
                Role::AirFactory => self.air_choice(v, fac),
                Role::NavalFactory => self.naval_choice(v, fac),
                _ => None,
            };
            let Some(ty) = choice else {
                if from_opening {
                    self.fac_open += 1;
                }
                continue;
            };
            if out.push(Command::Produce { factory: fac.id, unit_type: ty, count: 1 }) {
                self.mark(&[fac.id], tick);
                if from_opening {
                    self.fac_open += 1;
                }
                if is_eng {
                    eng_total += 1;
                    eng_now += 1;
                }
            }
        }
    }

    fn plan_mex_upgrades(&mut self, v: &View, out: &mut Out) {
        let p = v.p;
        let inc = v.inc_m_s();
        if inc < p.mex_upgrade_income || !out.room() {
            return;
        }
        if v.want.pgens > 1 && v.energy * 4 < v.energy_cap {
            return;
        }
        if v.mass < 60 * U && v.dem_m > v.inc_m {
            return;
        }
        let busy = v.mexes.iter().filter(|m| upgrading(m)).count();
        let parallel = p.mex_parallel + if p.micro { (inc / 25) as usize } else { 0 };
        if busy >= parallel {
            return;
        }
        let tm = self.tm();
        let mut best: Option<((u8, i128, UnitId), String)> = None;
        for m in &v.mexes {
            if !is_idle(m) || m.tier >= 3 || (m.tier == 2 && inc < 40) {
                continue;
            }
            if tm.surface_threat(pos(m), U) > 0 {
                continue;
            }
            let key = (m.tier, d2(pos(m), v.start), m.id);
            if best.as_ref().is_some_and(|(bk, _)| *bk <= key) {
                continue;
            }
            if let Some(t) = upgrade_target(v.w, m) {
                best = Some((key, t));
            }
        }
        if let Some(((_, _, id), t)) = best
            && out.push(Command::Upgrade { unit: id, target_type: t })
        {
            self.mark(&[id], v.tick);
        }
    }

    fn plan_engineers(&mut self, v: &mut View, out: &mut Out) {
        let tick = v.tick;
        let p = v.p;
        let engs = v.engineers.clone();
        if p.micro {
            let tm = self.tm();
            let flee: Vec<UnitId> = engs
                .iter()
                .filter(|e| {
                    let ep = pos(e);
                    d2(ep, v.start) > sq(40 * U)
                        && !self.recent(e.id, tick, p.reorder)
                        && tm.surface_threat(ep, 16 * U) > tm.sum(&tm.ally, ep, 16 * U) + 40
                })
                .map(|e| e.id)
                .collect();
            if !flee.is_empty() && out.push(Command::Move { units: flee.clone(), x: v.start.0, y: v.start.1, queue: false }) {
                self.mark(&flee, tick);
            }
        }
        let mut assist: BTreeMap<UnitId, Vec<UnitId>> = BTreeMap::new();
        for e in engs {
            if !out.room() {
                break;
            }
            let idle = is_idle(e);
            let assisting = matches!(e.orders.front(), Some(UnitOrder::Assist(_)));
            if !(idle || assisting) || self.recent(e.id, tick, if idle { p.reorder } else { 300 }) {
                continue;
            }
            match self.pick_job(v, e, false) {
                Some(Work::Build(items)) => self.issue_builds(v, e.id, items, out),
                Some(Work::Reclaim(wp)) => {
                    if out.push(Command::Reclaim { units: vec![e.id], x: wp.0, y: wp.1, queue: false }) {
                        self.mark(&[e.id], tick);
                    }
                }
                None => {
                    if idle && let Some(t) = self.assist_target(v, pos(e)) {
                        assist.entry(t).or_default().push(e.id);
                    }
                }
            }
        }
        for (t, ids) in assist {
            if out.push(Command::Assist { units: ids.clone(), target: t, queue: false }) {
                self.mark(&ids, tick);
            }
        }
    }

    // ----- army ---------------------------------------------------------------

    fn needs(&self, u: &Unit, t: (i64, i64), tol: i64, tick: u64, window: u64) -> bool {
        if self.recent(u.id, tick, window) {
            return false;
        }
        match u.orders.front() {
            None => d2(pos(u), t) > sq(tol),
            Some(UnitOrder::AttackMove { x, y }) | Some(UnitOrder::Move { x, y }) | Some(UnitOrder::Patrol { x, y }) => {
                d2((*x, *y), t) > sq(tol)
            }
            Some(UnitOrder::Attack(_)) => false,
            _ => true,
        }
    }

    fn push_group(&mut self, out: &mut Out, ids: Vec<UnitId>, t: (i64, i64), mode: u8, tick: u64) {
        if ids.is_empty() {
            return;
        }
        let cmd = match mode {
            0 => Command::AttackMove { units: ids.clone(), x: t.0, y: t.1, queue: false },
            1 => Command::Move { units: ids.clone(), x: t.0, y: t.1, queue: false },
            _ => Command::Patrol { units: ids.clone(), x: t.0, y: t.1, queue: false },
        };
        if out.push(cmd) {
            self.mark(&ids, tick);
        }
    }

    /// Best attack target `(position, estimated defence)` seen from `from`.
    fn choose_target(&self, v: &View, from: (i64, i64), power: i64) -> Option<((i64, i64), i64)> {
        let (focus_id, focus_start) = v.focus?;
        if !v.p.threat_targets {
            let t = self.enemy_acu.get(&focus_id).map(|x| x.0).unwrap_or(focus_start);
            return Some((t, 0));
        }
        let tm = self.tm();
        let hard = v.p.micro;
        let mut cands: Vec<((i64, i64), i64)> = Vec::new();
        for (_, &(p, t)) in &self.enemy_acu {
            if v.tick < t + 1_200 {
                cands.push((p, if hard { 3_000 } else { 1_500 }));
            }
        }
        for i in 0..tm.len() {
            let (e, d) = (tm.eco[i] as i64, tm.defense[i] as i64);
            if e > 0 || d > 0 {
                cands.push((tm.center(i), if hard { e + d / 4 + 50 } else { 100 }));
            }
        }
        for &(a, s) in &v.enemy_starts {
            cands.push((s, if a == focus_id { 400 } else { 150 }));
        }
        let mut best_ok: Option<(i64, (i64, i64), i64)> = None;
        let mut safest: Option<(i64, (i64, i64))> = None;
        for (pt, mut val) in cands {
            if d2(pt, focus_start) < sq(200 * U) {
                val += val / 2;
            }
            let risk = tm.surface_threat(pt, 48 * U);
            let score = val * 1_000 / (dist(from, pt) / U + 64);
            if risk * v.p.attack_ratio <= power * 100
                && best_ok.is_none_or(|(bs, bp, _)| score > bs || (score == bs && pt < bp))
            {
                best_ok = Some((score, pt, risk));
            }
            if safest.is_none_or(|(sr, sp)| risk < sr || (risk == sr && pt < sp)) {
                safest = Some((risk, pt));
            }
        }
        best_ok.map(|(_, p, r)| (p, r)).or(safest.map(|(r, p)| (p, r)))
    }

    /// An allied army's current push (≥ 6 units attack-moving to the same far area).
    fn ally_push(&self, v: &View) -> Option<(i64, i64)> {
        let tm = self.tm();
        for &(aid, astart) in &v.ally_starts {
            let mut cells: BTreeMap<usize, (u32, (i64, i64))> = BTreeMap::new();
            for u in v.w.units() {
                if u.army != aid || u.weapons.is_empty() || !matches!(u.role, Role::Mobile | Role::Experimental) {
                    continue;
                }
                if let Some(UnitOrder::AttackMove { x, y }) = u.orders.front()
                    && d2((*x, *y), astart) > sq(120 * U)
                {
                    let e = cells.entry(tm.index((*x, *y))).or_insert((0, (*x, *y)));
                    e.0 += 1;
                }
            }
            let best = cells
                .iter()
                .max_by_key(|(i, (n, _))| (*n, std::cmp::Reverse(**i)))
                .map(|(_, v)| *v);
            if let Some((n, p)) = best
                && n >= 6
            {
                return Some(p);
            }
        }
        None
    }

    fn plan_army(&mut self, v: &View, bt: &BaseThreat, out: &mut Out) {
        let tick = v.tick;
        let p = v.p;
        let army = self.task_units(v, Task::Army);
        let guards = self.task_units(v, Task::Guard);
        let power: i64 = army.iter().map(|u| ai_threat::unit_power(u).surface as i64).sum();

        // Hard: pull badly damaged units out of the fight; they guard the base afterwards.
        if p.micro {
            let hurt: Vec<UnitId> = army
                .iter()
                .filter(|u| {
                    u.role != Role::Experimental
                        && u.hp * 4 < u.max_hp
                        && d2(pos(u), v.start) > sq(80 * U)
                        && !self.recent(u.id, tick, p.reorder)
                })
                .map(|u| u.id)
                .collect();
            if !hurt.is_empty()
                && out.push(Command::Move { units: hurt.clone(), x: v.start.0, y: v.start.1, queue: false })
            {
                for id in &hurt {
                    self.tasks.insert(*id, Task::Guard);
                }
                self.mark(&hurt, tick);
            }
        }

        // Guards.
        let guard_spot = self.forward(v, 10 * U);
        if let Some(ip) = bt.pos {
            let ids: Vec<UnitId> = guards.iter().filter(|u| self.needs(u, ip, 20 * U, tick, p.reorder)).map(|u| u.id).collect();
            self.push_group(out, ids, ip, 0, tick);
        } else {
            let ids: Vec<UnitId> = guards
                .iter()
                .filter(|u| u.orders.is_empty() && d2(pos(u), guard_spot) > sq(25 * U) && !self.recent(u.id, tick, p.reorder))
                .map(|u| u.id)
                .collect();
            self.push_group(out, ids, guard_spot, 0, tick);
        }

        // Base / ally defence switch.
        let home_def = {
            let tm = self.tm();
            tm.sum(&tm.ally, v.start, DEFEND_RADIUS) - power.min(tm.sum(&tm.ally, v.start, DEFEND_RADIUS))
        };
        let defend_at = if let Some(ip) = bt.pos {
            let urgent = bt.power * 100 > home_def * 80;
            (self.phase != Phase::Attack || urgent).then_some(ip)
        } else if p.team_coord && bt.ally_power >= 100 && self.phase != Phase::Attack {
            bt.ally_pos.filter(|ap| d2(*ap, v.start) <= sq(450 * U))
        } else {
            None
        };
        if let Some(dp) = defend_at {
            self.set_phase(Phase::Defend, tick);
            self.target = Some(dp);
        } else if self.phase == Phase::Defend {
            self.set_phase(Phase::Gather, tick);
            self.target = None;
        }

        match self.phase {
            Phase::Defend => {
                let t = self.target.unwrap_or(v.start);
                let ids: Vec<UnitId> = army.iter().filter(|u| self.needs(u, t, 25 * U, tick, p.reorder)).map(|u| u.id).collect();
                self.push_group(out, ids, t, 0, tick);
            }
            Phase::Gather => {
                let ids: Vec<UnitId> = army
                    .iter()
                    .filter(|u| u.orders.is_empty() && d2(pos(u), v.gather) > sq(30 * U) && !self.recent(u.id, tick, p.reorder))
                    .map(|u| u.id)
                    .collect();
                self.push_group(out, ids, v.gather, 0, tick);
                if army.is_empty() {
                    return;
                }
                let target = self.choose_target(v, v.gather, power);
                let ready = if p.wave_units > 0 {
                    army.len() >= (p.wave_units + 4 * self.waves as usize).min(30)
                } else {
                    let risk = target.map(|t| t.1).unwrap_or(0);
                    let need = (risk * p.attack_ratio / 100).max(p.min_attack_power);
                    power >= need || (tick > self.phase_tick + 3_000 && army.len() >= 24)
                };
                let ally = if p.team_coord { self.ally_push(v) } else { None };
                let join = ally.is_some() && power * 2 >= p.min_attack_power;
                let goal = if join { ally } else if ready { target.map(|t| t.0) } else { None };
                if let Some(t) = goal {
                    self.set_phase(Phase::Attack, tick);
                    self.target = Some(t);
                    self.attack_power = power.max(1);
                    self.waves += 1;
                    let ids: Vec<UnitId> = army.iter().map(|u| u.id).collect();
                    self.push_group(out, ids, t, 0, tick);
                }
            }
            Phase::Attack => {
                let Some(mut t) = self.target else {
                    self.set_phase(Phase::Gather, tick);
                    return;
                };
                if army.is_empty() {
                    self.set_phase(Phase::Gather, tick);
                    self.target = None;
                    return;
                }
                let front: Vec<&Unit> = army.iter().copied().filter(|u| d2(pos(u), v.gather) > sq(50 * U)).collect();
                let c = centroid(&front).unwrap_or(v.gather);
                // Chase an exposed commander near the push.
                if p.threat_targets
                    && let Some(acu) = v
                        .contacts
                        .iter()
                        .filter(|k| k.role == Role::Commander && (d2(k.pos, t) <= sq(120 * U) || d2(k.pos, c) <= sq(80 * U)))
                        .min_by_key(|k| (d2(k.pos, c), k.id))
                {
                    t = acu.pos;
                    self.target = Some(t);
                }
                if p.retreat_ratio > 0 && !front.is_empty() {
                    let tm = self.tm();
                    let enemy_local = tm.surface_threat(c, 40 * U);
                    let own_local: i64 = front
                        .iter()
                        .filter(|u| d2(pos(u), c) <= sq(45 * U))
                        .map(|u| ai_threat::unit_power(u).surface as i64)
                        .sum();
                    let outgunned = enemy_local > 150 && enemy_local * 100 > own_local * p.retreat_ratio;
                    let depleted = power * 100 < self.attack_power * 30;
                    if outgunned || depleted {
                        self.set_phase(Phase::Retreat, tick);
                        let ids: Vec<UnitId> = army.iter().map(|u| u.id).collect();
                        self.push_group(out, ids, v.gather, 1, tick);
                        return;
                    }
                }
                let arrived = front.iter().any(|u| d2(pos(u), t) <= sq(20 * U));
                let enemies_left = v.contacts.iter().any(|k| !k.air && d2(k.pos, t) <= sq(35 * U));
                if arrived && !enemies_left {
                    match self.choose_target(v, c, power) {
                        Some((nt, risk)) if p.wave_units > 0 || risk * p.attack_ratio <= power * 100 => {
                            t = nt;
                            self.target = Some(t);
                        }
                        _ => {
                            self.set_phase(Phase::Gather, tick);
                            self.target = None;
                            let ids: Vec<UnitId> = army.iter().map(|u| u.id).collect();
                            self.push_group(out, ids, v.gather, 0, tick);
                            return;
                        }
                    }
                }
                // Hard: focus fire on an enemy commander in reach of the push.
                let mut focused: BTreeSet<UnitId> = BTreeSet::new();
                if p.micro
                    && let Some(acu) = v
                        .contacts
                        .iter()
                        .filter(|k| k.role == Role::Commander && d2(k.pos, c) <= sq(60 * U))
                        .min_by_key(|k| (d2(k.pos, c), k.id))
                {
                    let ids: Vec<UnitId> = army
                        .iter()
                        .filter(|u| {
                            d2(pos(u), acu.pos) <= sq(50 * U)
                                && !matches!(u.orders.front(), Some(UnitOrder::Attack(x)) if *x == acu.id)
                        })
                        .map(|u| u.id)
                        .collect();
                    if !ids.is_empty() && out.push(Command::Attack { units: ids.clone(), target: acu.id, queue: false }) {
                        focused.extend(ids.iter().copied());
                        self.mark(&ids, tick);
                    }
                }
                // Reinforcements leave home in groups; units at the front keep pushing.
                let home: Vec<&Unit> = army
                    .iter()
                    .copied()
                    .filter(|u| d2(pos(u), v.gather) <= sq(50 * U) && !focused.contains(&u.id))
                    .collect();
                let min_group = if p.micro { 3 } else { 5 };
                let mut ids: Vec<UnitId> = front
                    .iter()
                    .filter(|u| !focused.contains(&u.id) && self.needs(u, t, 40 * U, tick, p.reorder))
                    .map(|u| u.id)
                    .collect();
                let idle_home: Vec<UnitId> = home.iter().filter(|u| self.needs(u, t, 40 * U, tick, p.reorder)).map(|u| u.id).collect();
                if idle_home.len() >= min_group {
                    ids.extend(idle_home);
                }
                self.push_group(out, ids, t, 0, tick);
            }
            Phase::Retreat => {
                let ids: Vec<UnitId> = army
                    .iter()
                    .filter(|u| {
                        !self.recent(u.id, tick, p.reorder)
                            && match u.orders.front() {
                                None => d2(pos(u), v.gather) > sq(30 * U),
                                Some(UnitOrder::Move { x, y }) => d2((*x, *y), v.gather) > sq(30 * U),
                                _ => true,
                            }
                    })
                    .map(|u| u.id)
                    .collect();
                self.push_group(out, ids, v.gather, 1, tick);
                let home = army.iter().filter(|u| d2(pos(u), v.gather) <= sq(45 * U)).count();
                if home * 10 >= army.len() * 7 || tick > self.phase_tick + 900 {
                    self.set_phase(Phase::Gather, tick);
                    self.target = None;
                }
            }
        }
    }

    fn plan_raids(&mut self, v: &View, out: &mut Out) {
        let tick = v.tick;
        let p = v.p;
        let raiders = self.task_units(v, Task::Raid);
        if raiders.is_empty() {
            self.raid_target = None;
            return;
        }
        let power: i64 = raiders.iter().map(|u| ai_threat::unit_power(u).surface as i64).sum();
        let c = centroid(&raiders).unwrap_or(v.gather);
        if let Some(t) = self.raid_target {
            let danger = self.tm().surface_threat(c, 24 * U);
            if danger * 100 > power * 120 {
                self.raid_target = None;
                let ids: Vec<UnitId> = raiders.iter().map(|u| u.id).collect();
                self.push_group(out, ids, v.gather, 1, tick);
                return;
            }
            let arrived = d2(c, t) <= sq(18 * U);
            let left = v.contacts.iter().any(|k| !k.air && k.eco > 0 && d2(k.pos, t) <= sq(30 * U));
            if !(arrived && !left) {
                let ids: Vec<UnitId> = raiders.iter().filter(|u| self.needs(u, t, 15 * U, tick, p.reorder)).map(|u| u.id).collect();
                self.push_group(out, ids, t, 0, tick);
                return;
            }
            self.raid_target = None;
        }
        let at_home = raiders.iter().filter(|u| d2(pos(u), v.gather) <= sq(60 * U)).count();
        if at_home < (p.raids / 2 + 1).max(2) && at_home < raiders.len() {
            let ids: Vec<UnitId> = raiders
                .iter()
                .filter(|u| u.orders.is_empty() && d2(pos(u), v.gather) > sq(30 * U) && !self.recent(u.id, tick, p.reorder))
                .map(|u| u.id)
                .collect();
            self.push_group(out, ids, v.gather, 1, tick);
            return;
        }
        if raiders.len() < (p.raids / 2 + 1).max(2) {
            return;
        }
        let tm = self.tm();
        let enemy = v.focus.map(|f| f.1);
        let mut best: Option<(i64, (i64, i64))> = None;
        for i in 0..tm.len() {
            let eco = tm.eco[i] as i64;
            if eco <= 0 {
                continue;
            }
            let pt = tm.center(i);
            if tm.surface_threat(pt, 32 * U) * 100 > power * 70 {
                continue;
            }
            let score = eco * 1_000 / (dist(c, pt) / U + 64) + enemy.map(|e| dist(pt, e) / U).unwrap_or(0);
            if best.is_none_or(|(bs, bp)| score > bs || (score == bs && pt < bp)) {
                best = Some((score, pt));
            }
        }
        let target = best.map(|b| b.1).or_else(|| {
            let e = enemy?;
            let pts: Vec<(i64, i64)> = v
                .w
                .mass_points()
                .iter()
                .copied()
                .filter(|&m| d2(m, e) < d2(m, v.start) && d2(m, e) > sq(60 * U) && tm.surface_threat(m, 24 * U) * 100 <= power * 70)
                .collect();
            if pts.is_empty() {
                return None;
            }
            let pick = pts[self.raid_cursor % pts.len()];
            Some(pick)
        });
        if let Some(t) = target {
            self.raid_cursor += 1;
            self.raid_target = Some(t);
            let ids: Vec<UnitId> = raiders.iter().map(|u| u.id).collect();
            self.push_group(out, ids, t, 0, tick);
        }
    }

    fn plan_scouts(&mut self, v: &View, out: &mut Out) {
        if !v.p.scouting {
            return;
        }
        let scouts = self.task_units(v, Task::Scout);
        if scouts.is_empty() {
            return;
        }
        if self.scout_route.is_empty() {
            let mut route: Vec<(i64, i64)> = Vec::new();
            if let Some((_, fs)) = v.focus {
                route.push(fs);
            }
            for &(_, s) in &v.enemy_starts {
                if !route.contains(&s) {
                    route.push(s);
                }
            }
            let anchor = v.focus.map(|f| f.1).unwrap_or(v.start);
            let mut pts: Vec<(i64, i64)> = v
                .w
                .mass_points()
                .iter()
                .copied()
                .filter(|&m| d2(m, v.start) > sq(120 * U) && self.land.as_ref().is_none_or(|r| r.near(m, 2)))
                .collect();
            pts.sort_by_key(|&m| (d2(m, anchor), m));
            route.extend(pts.into_iter().step_by(3).take(12));
            if route.is_empty() {
                route.push(v.start);
            }
            self.scout_route = route;
        }
        for s in scouts {
            if !out.room() {
                break;
            }
            if !s.orders.is_empty() || self.recent(s.id, v.tick, 30) {
                continue;
            }
            let wp = self.scout_route[self.scout_idx % self.scout_route.len()];
            self.scout_idx += 1;
            self.push_group(out, vec![s.id], wp, 1, v.tick);
        }
    }

    fn plan_air_naval(&mut self, v: &View, bt: &BaseThreat, out: &mut Out) {
        let tick = v.tick;
        let p = v.p;
        // Air cover: chase air intruders, otherwise patrol over the base.
        let airdef = self.task_units(v, Task::AirDef);
        if !airdef.is_empty() {
            let (dx, dy) = match v.focus {
                Some((_, fs)) => (fs.0 - v.start.0, fs.1 - v.start.1),
                None => (U, 0),
            };
            let len = isqrt((dx as i128) * (dx as i128) + (dy as i128) * (dy as i128)).max(1);
            let patrol = clamp_map(v.w, (v.start.0 - dy * 30 * U / len, v.start.1 + dx * 30 * U / len));
            if let Some(ap) = bt.air_pos {
                let ids: Vec<UnitId> = airdef.iter().filter(|u| self.needs(u, ap, 20 * U, tick, p.reorder)).map(|u| u.id).collect();
                self.push_group(out, ids, ap, 0, tick);
            } else {
                let ids: Vec<UnitId> = airdef.iter().filter(|u| u.orders.is_empty() && !self.recent(u.id, tick, p.reorder)).map(|u| u.id).collect();
                self.push_group(out, ids, patrol, 2, tick);
            }
        }
        // Air strikes against enemy eco (only with fast aircraft).
        let strikers = self.task_units(v, Task::AirStrike);
        if strikers.len() >= 3 {
            let tm = self.tm();
            let power: i64 = strikers.iter().map(|u| ai_threat::unit_power(u).total() as i64).sum();
            let mut best: Option<(i64, (i64, i64))> = None;
            for i in 0..tm.len() {
                let eco = tm.eco[i] as i64;
                if eco <= 0 {
                    continue;
                }
                let pt = tm.center(i);
                if tm.sum(&tm.anti_air, pt, 32 * U) * 100 > power * 60 {
                    continue;
                }
                let score = eco * 1_000 / (dist(v.start, pt) / U + 64);
                if best.is_none_or(|(bs, bp)| score > bs || (score == bs && pt < bp)) {
                    best = Some((score, pt));
                }
            }
            let goal = best.map(|b| b.1).unwrap_or(v.start);
            let ids: Vec<UnitId> = strikers.iter().filter(|u| self.needs(u, goal, 20 * U, tick, p.reorder)).map(|u| u.id).collect();
            self.push_group(out, ids, goal, 0, tick);
        }
        // Navy: push toward water near the focus enemy once a squadron exists.
        let navy = self.task_units(v, Task::Naval);
        if navy.is_empty() {
            return;
        }
        let home = self.naval_home.unwrap_or(v.start);
        let power: i64 = navy.iter().map(|u| ai_threat::unit_power(u).surface as i64).sum();
        let c = centroid(&navy).unwrap_or(home);
        let danger = self.tm().surface_threat(c, 40 * U);
        if danger > 150 && danger * 100 > power * 150 {
            let ids: Vec<UnitId> = navy.iter().filter(|u| self.needs(u, home, 30 * U, tick, p.reorder)).map(|u| u.id).collect();
            self.push_group(out, ids, home, 1, tick);
            return;
        }
        let goal = v
            .focus
            .and_then(|(fid, _)| self.naval_enemy.iter().find(|(a, _)| *a == fid).map(|x| x.1));
        match goal {
            Some(g) if navy.len() >= 3 => {
                let ids: Vec<UnitId> = navy.iter().filter(|u| self.needs(u, g, 30 * U, tick, p.reorder)).map(|u| u.id).collect();
                self.push_group(out, ids, g, 0, tick);
            }
            _ => {
                let ids: Vec<UnitId> = navy
                    .iter()
                    .filter(|u| u.orders.is_empty() && d2(pos(u), home) > sq(30 * U) && !self.recent(u.id, tick, p.reorder))
                    .map(|u| u.id)
                    .collect();
                self.push_group(out, ids, home, 1, tick);
            }
        }
    }

    fn plan_silos(&mut self, v: &View, out: &mut Out) {
        let tm = self.tm();
        let mut cmds: Vec<(UnitId, (i64, i64))> = Vec::new();
        for s in &v.silos {
            if s.silo_missiles == 0 || !s.orders.is_empty() || self.recent(s.id, v.tick, 50) {
                continue;
            }
            let range = v
                .w
                .catalog()
                .get(&s.type_id)
                .map(|t| t.missile_range)
                .filter(|r| *r > 0)
                .unwrap_or(200 * U);
            let sp = pos(s);
            let acu = self
                .enemy_acu
                .values()
                .filter(|(p, t)| v.tick < *t + 100 && d2(*p, sp) <= sq(range))
                .map(|x| x.0)
                .min_by_key(|p| (d2(*p, sp), *p));
            let target = acu.or_else(|| {
                let mut best: Option<(i64, (i64, i64))> = None;
                for i in 0..tm.len() {
                    let val = tm.eco[i] as i64 + tm.defense[i] as i64;
                    let pt = tm.center(i);
                    if val <= 0 || d2(pt, sp) > sq(range) {
                        continue;
                    }
                    if best.is_none_or(|(bv, bp)| val > bv || (val == bv && pt < bp)) {
                        best = Some((val, pt));
                    }
                }
                best.map(|b| b.1)
            });
            if let Some(t) = target {
                cmds.push((s.id, t));
            }
        }
        for (id, t) in cmds {
            if out.push(Command::LaunchMissile { unit: id, x: t.0, y: t.1 }) {
                self.mark(&[id], v.tick);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skirmish::tests::test_world;

    fn play(seed: u64, diffs: [Difficulty; 2], ticks: u64) -> Skirmish {
        let mut w = test_world(seed);
        let mut ais = [
            ScriptedAi::with_difficulty(0, diffs[0]),
            ScriptedAi::with_difficulty(1, diffs[1]),
        ];
        for _ in 0..ticks {
            for ai in &mut ais {
                ai.think(&mut w);
            }
            w.step();
            if w.result().is_some() {
                break;
            }
        }
        w
    }

    #[test]
    fn difficulty_parses_displays_and_lists() {
        for d in Difficulty::ALL {
            assert_eq!(d.to_string().parse::<Difficulty>(), Ok(d));
            assert!(!d.describe().is_empty());
        }
        assert_eq!(" HARD ".parse::<Difficulty>(), Ok(Difficulty::Hard));
        assert_eq!("medium".parse::<Difficulty>(), Ok(Difficulty::Normal));
        assert_eq!("old".parse::<Difficulty>(), Ok(Difficulty::Legacy));
        assert!("brutal".parse::<Difficulty>().unwrap_err().contains("unknown AI difficulty"));
        assert_eq!(Difficulty::default(), Difficulty::Normal);
        assert_eq!(ScriptedAi::new(3).difficulty, Difficulty::Normal);
        assert_eq!(
            parse_difficulty_list("hard,easy", 3).unwrap(),
            vec![Difficulty::Hard, Difficulty::Easy, Difficulty::Hard]
        );
        assert_eq!(parse_difficulty_list("legacy", 2).unwrap(), vec![Difficulty::Legacy; 2]);
        assert!(parse_difficulty_list("", 2).is_err());
        assert!(parse_difficulty_list("hard,nope", 2).is_err());
    }

    #[test]
    fn elo_ladder_ranks_the_stronger_side_higher() {
        let mut games = Vec::new();
        for i in 0..100u32 {
            let hard_wins = i % 10 < 7;
            let (l, r) = if i % 2 == 0 { (Difficulty::Hard, Difficulty::Hard19) } else { (Difficulty::Hard19, Difficulty::Hard) };
            let hard_side = if l == Difficulty::Hard { 0 } else { 1 };
            games.push((l, r, Some(if hard_wins { hard_side } else { 1 - hard_side })));
        }
        let e = elo_ratings(&games);
        let gap = e[&Difficulty::Hard] - e[&Difficulty::Hard19];
        assert!((100.0..200.0).contains(&gap), "70% wins ≈ +147 Elo, got {gap}");
        assert!((e[&Difficulty::Hard] + e[&Difficulty::Hard19] - 3000.0).abs() < 1e-6, "zero-sum");
    }

    #[test]
    fn profiles_scale_with_difficulty() {
        let (e, n, h) = (
            Profile::of(Difficulty::Easy),
            Profile::of(Difficulty::Normal),
            Profile::of(Difficulty::Hard),
        );
        assert!(e.think > n.think && n.think > h.think, "decision cadence");
        assert!(e.max_cmds < n.max_cmds && n.max_cmds < h.max_cmds, "APM");
        assert!(e.eng_max < n.eng_max && n.eng_max < h.eng_max);
        assert!(e.queue_opening <= n.queue_opening && n.queue_opening < h.queue_opening);
        assert!(!e.micro && !n.micro && h.micro);
        assert!(e.wave_units > 0 && n.wave_units == 0);
        assert!(h.raids > 0 && n.raids == 0);
        assert!(e.sloppy_opening && !n.sloppy_opening);
        assert_eq!(Profile::of(Difficulty::Adaptive), n);
    }

    #[test]
    fn smart_ai_is_deterministic_builds_and_fights() {
        for diffs in [
            [Difficulty::Normal, Difficulty::Normal],
            [Difficulty::Hard, Difficulty::Easy],
            [Difficulty::Adaptive, Difficulty::Legacy],
        ] {
            let a = play(11, diffs, 9_000);
            let b = play(11, diffs, 9_000);
            assert_eq!(a.hash_hex(), b.hash_hex(), "{diffs:?}");
            assert!(a.armies().iter().all(|x| x.units_built >= 3), "{diffs:?} {:?}", a.armies());
            assert!(a.shots_fired() > 0, "{diffs:?}");
            assert!(a.kills() > 0, "{diffs:?}");
            assert_eq!(a.commands_rejected(), b.commands_rejected());
        }
    }

    #[test]
    fn ai_state_round_trips_through_clone() {
        let mut w = test_world(5);
        let mut ai = ScriptedAi::with_difficulty(0, Difficulty::Hard);
        for _ in 0..600 {
            ai.think(&mut w);
            w.step();
        }
        let mut w2 = w.clone();
        let mut ai2 = ai.clone();
        assert_eq!(ai, ai2);
        for _ in 0..600 {
            ai.think(&mut w);
            w.step();
            ai2.think(&mut w2);
            w2.step();
        }
        assert_eq!(w.state_hash(), w2.state_hash());
        assert_eq!(ai, ai2);
    }

    #[test]
    fn hard_ai_expands_and_queues_its_opening() {
        let w = play(3, [Difficulty::Hard, Difficulty::Easy], 1_500);
        let mex = |a: ArmyId| w.units().filter(|u| u.army == a && u.role == Role::Extractor).count();
        let fac = |a: ArmyId| w.units().filter(|u| u.army == a && u.role == Role::Factory).count();
        assert!(fac(0) >= 1 && mex(0) >= 1, "hard built eco: fac={} mex={}", fac(0), mex(0));
        assert!(w.units().any(|u| u.army == 0 && u.role == Role::Engineer));
    }
}
