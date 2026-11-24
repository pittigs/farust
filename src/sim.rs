//! Phase 4 deterministic sim core.
//!
//! Fixed-tick (10 Hz) integer-only world: spawn, move/order queue, economy
//! ticks, hitscan weapon fire and damage, plus a headless N-tick run with a
//! stable FNV-1a state hash. No `f32`/`f64` in tick state, no `HashMap`
//! iteration, no threads, no wall-clock: iteration is always in ascending
//! unit id order so every run with the same seed hashes identically.
//!
//! Blueprint `f64` values are converted once at spawn (`SCALE = 1000`,
//! milli-units, rounding) and never inside the tick loop. Regen, AoE damage
//! radius and overcharge energy gates are NOT modelled in Phase 4 (see
//! [`crate::blueprint::UnitSummary`]); every weapon is a plain single-target
//! hitscan weapon.

use std::collections::{BTreeMap, VecDeque};

use crate::blueprint::UnitSummary;
use crate::combat;
use crate::economy::ConstructionProject;

/// Ticks per second (Forged Alliance runs at 10 Hz).
pub const TICK_HZ: u64 = 10;
/// Fixed step in microseconds (0.1 s).
pub const DT_MICROS: u64 = 100_000;
/// Milli-unit scale: 1 game unit = 1000 sim steps.
pub const SCALE: i64 = 1000;
/// Magic header mixed into the state hash (format changes invalidate goldens).
pub const HASH_MAGIC: &[u8] = b"FARUSTSIM1";
/// Sim format version, also hashed.
pub const SIM_VERSION: u64 = 1;
/// Default seed for `sim-run` (`0xFA57`).
pub const DEFAULT_SEED: u64 = 0xFA57;
/// Hard cap for a single headless run.
pub const MAX_TICKS: u64 = 100_000;

pub type Tick = u64;
pub type UnitId = u32;

/// Milli-unit 2D position (x, y). The map plane is x/y here; scenario z maps
/// to y.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vec2 {
    pub x: i64,
    pub y: i64,
}

/// Orders a unit can hold. The queue head is the active order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Order {
    Move { x: i64, y: i64 },
    Attack(UnitId),
}

/// Spawn request. Blueprint-derived numbers arrive as `f64` and are quantized
/// once; tests can bypass blueprints with milli-unit overrides.
#[derive(Debug, Clone)]
pub struct UnitSpawn {
    pub blueprint_id: String,
    pub x: i64,
    pub y: i64,
    pub hp_override: Option<i64>,
    pub weapon_damage_override: Option<i64>,
}

impl UnitSpawn {
    pub fn new(blueprint_id: &str, x: i64, y: i64) -> Self {
        Self {
            blueprint_id: blueprint_id.to_string(),
            x,
            y,
            hp_override: None,
            weapon_damage_override: None,
        }
    }
}

/// One weapon on a unit, fully quantized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimWeapon {
    pub damage: i64,
    pub range_sq: i128,
    pub cooldown: u32,
    pub reset: u32,
}

/// Per-unit quantized parameters derived once from a [`UnitSummary`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct UnitParams {
    pub max_hp: i64,
    pub speed_per_tick: i64,
    pub weapons: Vec<SimWeapon>,
    pub cost_mass: i64,
    pub cost_energy: i64,
    pub build_ticks: u64,
}

/// Convert once, at the edge: `(v * SCALE).round() as i64`.
#[allow(dead_code)]
pub fn quantize(v: f64) -> i64 {
    (v * SCALE as f64).round() as i64
}

#[allow(dead_code)]
impl UnitParams {
    /// Derive params from a blueprint summary. Missing health falls back to 1
    /// hp (never 0/div-by-zero); missing speed is stationary; weapons without
    /// damage/range/rof never fire.
    pub fn from_summary(s: &UnitSummary) -> Self {
        let max_hp = s.health.map(quantize).unwrap_or(SCALE).max(1);
        let speed_per_tick = s
            .max_speed
            .map(|v| quantize(v) / TICK_HZ as i64)
            .unwrap_or(0)
            .max(0);
        let weapons = s
            .weapons
            .iter()
            .map(|w| {
                let damage = w.damage.map(quantize).unwrap_or(0);
                let range_sq = w
                    .max_radius
                    .map(|r| {
                        let q = quantize(r) as i128;
                        q * q
                    })
                    .unwrap_or(-1);
                let rof_milli = w.rate_of_fire.map(quantize).unwrap_or(0);
                SimWeapon {
                    damage,
                    range_sq,
                    cooldown: 0,
                    reset: combat::cooldown_reset_ticks(TICK_HZ, rof_milli),
                }
            })
            .collect();
        Self {
            max_hp,
            speed_per_tick,
            weapons,
            cost_mass: s.build_cost_mass.map(quantize).unwrap_or(1).max(1),
            cost_energy: s.build_cost_energy.map(quantize).unwrap_or(1).max(1),
            build_ticks: s
                .build_time
                .map(|t| t.max(1.0).round() as u64)
                .unwrap_or(1)
                .max(1),
        }
    }
}

#[derive(Debug)]
struct Unit {
    id: UnitId,
    bp: String,
    pos: Vec2,
    hp: i64,
    max_hp: i64,
    alive: bool,
    weapons: Vec<SimWeapon>,
    speed_per_tick: i64,
    orders: VecDeque<Order>,
}

/// Read-only snapshot of one live unit for parallel render preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct SimUnitSnapshot {
    pub id: UnitId,
    pub blueprint_id: String,
    pub x: i64,
    pub y: i64,
    pub hp: i64,
    pub max_hp: i64,
}

/// Fixed-tick deterministic world. Single-threaded; `BTreeMap` keeps unit
/// iteration in ascending id order.
#[derive(Debug)]
#[allow(dead_code)]
pub struct World {
    tick: Tick,
    seed: u64,
    rng: u64,
    next_id: UnitId,
    next_project: u64,
    units: BTreeMap<UnitId, Unit>,
    pub mass: i64,
    pub energy: i64,
    projects: Vec<ConstructionProject>,
    shots_fired: u64,
    kills: u64,
}

#[allow(dead_code)]
impl World {
    pub fn new(seed: u64) -> Self {
        Self {
            tick: 0,
            seed,
            rng: seed ^ 0x9E3779B97F4A7C15,
            next_id: 1,
            next_project: 0,
            units: BTreeMap::new(),
            mass: 0,
            energy: 0,
            projects: Vec::new(),
            shots_fired: 0,
            kills: 0,
        }
    }

    pub fn with_stockpiles(seed: u64, mass: i64, energy: i64) -> Self {
        let mut w = Self::new(seed);
        w.mass = mass;
        w.energy = energy;
        w
    }

    pub fn tick(&self) -> Tick {
        self.tick
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn units_alive(&self) -> usize {
        self.units.values().filter(|u| u.alive).count()
    }

    pub fn shots_fired(&self) -> u64 {
        self.shots_fired
    }

    pub fn kills(&self) -> u64 {
        self.kills
    }

    pub fn unit_pos(&self, id: UnitId) -> Option<(i64, i64)> {
        self.units.get(&id).map(|u| (u.pos.x, u.pos.y))
    }

    pub fn unit_hp(&self, id: UnitId) -> Option<i64> {
        self.units.get(&id).map(|u| u.hp)
    }

    pub fn is_alive(&self, id: UnitId) -> bool {
        self.units.get(&id).is_some_and(|u| u.alive)
    }

    /// SplitMix64; only consumed inside [`World::step`] in id order.
    /// Phase 4 combat uses no RNG (hits always land); the stream exists for
    /// future phases and is part of the hashed state.
    fn next_rng(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Spawn a unit from pre-quantized params. Returns the new unit id.
    pub fn spawn(&mut self, req: &UnitSpawn, params: &UnitParams) -> UnitId {
        let id = self.next_id;
        self.next_id += 1;
        let mut weapons = params.weapons.clone();
        if let Some(d) = req.weapon_damage_override {
            for w in &mut weapons {
                w.damage = d;
            }
        }
        for w in &mut weapons {
            w.cooldown = 0;
        }
        let initial_hp = req.hp_override.unwrap_or(params.max_hp).max(1);
        self.units.insert(
            id,
            Unit {
                id,
                bp: req.blueprint_id.clone(),
                pos: Vec2 { x: req.x, y: req.y },
                hp: initial_hp,
                max_hp: initial_hp,
                alive: true,
                weapons,
                speed_per_tick: params.speed_per_tick,
                orders: VecDeque::new(),
            },
        );
        id
    }

    /// Snapshot all alive units in ascending `UnitId` order for read-only render preparation.
    pub fn snapshot_units(&self) -> Vec<SimUnitSnapshot> {
        self.units
            .values()
            .filter(|u| u.alive)
            .map(|u| SimUnitSnapshot {
                id: u.id,
                blueprint_id: u.bp.clone(),
                x: u.pos.x,
                y: u.pos.y,
                hp: u.hp,
                max_hp: u.max_hp,
            })
            .collect()
    }

    /// Replace a unit's order queue with waypoints from a solved path.
    pub fn apply_path_waypoints(&mut self, id: UnitId, waypoints: &[(i64, i64)]) {
        if let Some(u) = self.units.get_mut(&id) {
            u.orders.clear();
            for &(x, y) in waypoints {
                u.orders.push_back(Order::Move { x, y });
            }
        }
    }

    /// Replace a unit's order queue with a single order. Unknown ids are
    /// ignored (no error branch inside the tick).
    pub fn order(&mut self, id: UnitId, order: Order) {
        if let Some(u) = self.units.get_mut(&id) {
            u.orders.clear();
            u.orders.push_back(order);
        }
    }

    /// Append an order to a unit's queue. Unknown ids are ignored.
    pub fn queue_order(&mut self, id: UnitId, order: Order) {
        if let Some(u) = self.units.get_mut(&id) {
            u.orders.push_back(order);
        }
    }

    /// Queue a construction project paid from this world's stockpiles.
    /// Returns the project id. Projects run in ascending id order.
    pub fn build(&mut self, total_mass: i64, total_energy: i64, work_total: i64, build_rate: i64) -> u64 {
        let id = self.next_project;
        self.next_project += 1;
        self.projects.push(ConstructionProject::new(
            id,
            total_mass,
            total_energy,
            work_total,
            build_rate,
        ));
        id
    }

    /// Advance exactly one fixed tick: orders/move, then economy, then
    /// weapons, then death cleanup. All unit loops run in ascending id order.
    pub fn step(&mut self) {
        self.tick += 1;
        let _ = self.next_rng();
        self.step_move();
        self.step_economy();
        self.step_weapons();
        self.step_cleanup();
    }

    pub fn run(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Integer step toward `(tx, ty)` capped at `speed` (no overshoot).
    /// Applied x then y; kept as a helper so move/attack share one rounding.
    fn approach(pos: &mut Vec2, tx: i64, ty: i64, speed: i64) {
        let dx = tx - pos.x;
        let dy = ty - pos.y;
        let dist_sq = (dx as i128) * (dx as i128) + (dy as i128) * (dy as i128);
        if dist_sq <= 0 {
            return;
        }
        let dist = (dist_sq as u64 as f64).sqrt() as i64;
        if dist <= 0 {
            return;
        }
        let step = speed.min(dist);
        pos.x += dx * step / dist;
        pos.y += dy * step / dist;
    }

    fn step_move(&mut self) {
        // Ascending id order. Each iteration snapshots immutable state first,
        // then applies one mutation, so no borrow is held across the map.
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        for id in ids {
            loop {
                let (alive, pos, speed, head, ranges) = match self.units.get(&id) {
                    Some(u) => (
                        u.alive,
                        u.pos,
                        u.speed_per_tick,
                        u.orders.front().cloned(),
                        u.weapons.iter().map(|w| w.range_sq).collect::<Vec<_>>(),
                    ),
                    None => break,
                };
                if !alive {
                    break;
                }
                match head {
                    None => break,
                    Some(Order::Move { x, y }) => {
                        if pos.x == x && pos.y == y {
                            self.units.get_mut(&id).unwrap().orders.pop_front();
                            continue;
                        }
                        if speed <= 0 {
                            break;
                        }
                        let u = self.units.get_mut(&id).unwrap();
                        Self::approach(&mut u.pos, x, y, speed);
                        if u.pos.x == x && u.pos.y == y {
                            u.orders.pop_front();
                        }
                        break;
                    }
                    Some(Order::Attack(target)) => {
                        let attack_pos = match self.units.get(&target) {
                            Some(t) if t.alive => Some((t.pos.x, t.pos.y)),
                            _ => None,
                        };
                        match attack_pos {
                            None => {
                                self.units.get_mut(&id).unwrap().orders.pop_front();
                                continue;
                            }
                            Some((tx, ty)) => {
                                // Close to weapon range, then hold and let the
                                // weapon step do the damage.
                                let in_range = ranges.iter().any(|r| {
                                    *r >= 0 && combat::dist_sq(pos.x, pos.y, tx, ty) <= *r
                                });
                                if in_range || speed <= 0 {
                                    break;
                                }
                                let u = self.units.get_mut(&id).unwrap();
                                Self::approach(&mut u.pos, tx, ty, speed);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    fn step_economy(&mut self) {
        self.projects.sort_by_key(|p| p.id);
        let mut stock = (self.mass, self.energy);
        let mut done = Vec::new();
        for p in &mut self.projects {
            if p.step(&mut stock) {
                done.push(p.id);
            }
        }
        self.mass = stock.0.max(0);
        self.energy = stock.1.max(0);
        self.projects.retain(|p| !done.contains(&p.id));
    }

    fn step_weapons(&mut self) {
        // Ascending attacker id, then weapon index. Damage accumulates in that
        // order; kills are resolved in the cleanup pass.
        let ids: Vec<UnitId> = self.units.keys().copied().collect();
        // Snapshot of positions/aliveness for targeting (effects apply below).
        let targets: Vec<(UnitId, i64, i64, bool)> = ids
            .iter()
            .map(|id| {
                let u = &self.units[id];
                (u.id, u.pos.x, u.pos.y, u.alive)
            })
            .collect();
        for id in ids {
            let (ax, ay) = match self.units.get(&id) {
                Some(u) if u.alive => (u.pos.x, u.pos.y),
                _ => continue,
            };
            // Resolve this attacker's target: explicit Attack order wins,
            // otherwise nearest live enemy (any other live unit).
            let explicit = self.units.get(&id).and_then(|u| match u.orders.front() {
                Some(Order::Attack(t)) => Some(*t),
                _ => None,
            });
            let pick_target = |ax: i64, ay: i64| -> Option<UnitId> {
                if let Some(t) = explicit {
                    if targets.iter().any(|(tid, _, _, alive)| *tid == t && *alive) {
                        return Some(t);
                    }
                    return None;
                }
                let mut best: Option<(i128, UnitId)> = None;
                for (tid, tx, ty, alive) in &targets {
                    if *tid == id || !alive {
                        continue;
                    }
                    let d = combat::dist_sq(ax, ay, *tx, *ty);
                    if best.is_none_or(|(bd, _)| d < bd) {
                        best = Some((d, *tid));
                    }
                }
                best.map(|(_, tid)| tid)
            };
            let Some(target_id) = pick_target(ax, ay) else {
                // No target: still tick cooldowns down.
                if let Some(u) = self.units.get_mut(&id) {
                    for w in &mut u.weapons {
                        if w.cooldown > 0 {
                            w.cooldown -= 1;
                        }
                    }
                }
                continue;
            };
            let (tx, ty) = {
                let t = &self.units[&target_id];
                (t.pos.x, t.pos.y)
            };
            let d2 = combat::dist_sq(ax, ay, tx, ty);
            // Snapshot this attacker's weapons so the fire check holds no
            // borrow; effects apply below in (attacker, weapon) order.
            let guns: Vec<SimWeapon> = self.units.get(&id).unwrap().weapons.clone();
            for (wi, g) in guns.iter().enumerate() {
                if g.cooldown > 0 {
                    self.units.get_mut(&id).unwrap().weapons[wi].cooldown -= 1;
                    continue;
                }
                let alive = self.units.get(&target_id).is_some_and(|t| t.alive);
                if combat::can_fire(
                    0,
                    alive,
                    (g.range_sq >= 0).then_some(g.range_sq),
                    d2,
                    Some(g.damage),
                ) {
                    self.units.get_mut(&id).unwrap().weapons[wi].cooldown = g.reset;
                    self.shots_fired += 1;
                    let t = self.units.get_mut(&target_id).unwrap();
                    combat::apply_hit(&mut t.hp, g.damage);
                }
            }
        }
    }

    fn step_cleanup(&mut self) {
        // Ascending id: hp <= 0 -> dead, hp clamped at 0. Corpses stay in the
        // map as dead records (no wreck sim in Phase 4) so ids stay stable.
        for u in self.units.values_mut() {
            if u.alive && u.hp <= 0 {
                u.hp = 0;
                u.alive = false;
                self.kills += 1;
            }
        }
    }

    fn fnv_mix(hash: &mut u64, bytes: &[u8]) {
        for b in bytes {
            *hash ^= *b as u64;
            *hash = hash.wrapping_mul(0x100000001B3);
        }
    }

    /// FNV-1a 64 over header + tick/rng/stockpiles/counters + per-unit
    /// records in ascending id order. Integers only, little-endian
    /// fixed-width; strings length-prefixed; floats never hashed raw.
    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xCBF29CE484222325;
        Self::fnv_mix(&mut h, HASH_MAGIC);
        Self::fnv_mix(&mut h, &SIM_VERSION.to_le_bytes());
        Self::fnv_mix(&mut h, &self.seed.to_le_bytes());
        Self::fnv_mix(&mut h, &self.tick.to_le_bytes());
        Self::fnv_mix(&mut h, &self.rng.to_le_bytes());
        Self::fnv_mix(&mut h, &self.mass.to_le_bytes());
        Self::fnv_mix(&mut h, &self.energy.to_le_bytes());
        Self::fnv_mix(&mut h, &self.shots_fired.to_le_bytes());
        Self::fnv_mix(&mut h, &self.kills.to_le_bytes());
        for (id, u) in &self.units {
            Self::fnv_mix(&mut h, &id.to_le_bytes());
            Self::fnv_mix(&mut h, &(u.bp.len() as u32).to_le_bytes());
            Self::fnv_mix(&mut h, u.bp.as_bytes());
            Self::fnv_mix(&mut h, &[u.alive as u8]);
            Self::fnv_mix(&mut h, &u.hp.to_le_bytes());
            Self::fnv_mix(&mut h, &u.pos.x.to_le_bytes());
            Self::fnv_mix(&mut h, &u.pos.y.to_le_bytes());
            for w in &u.weapons {
                Self::fnv_mix(&mut h, &w.damage.to_le_bytes());
                Self::fnv_mix(&mut h, &w.cooldown.to_le_bytes());
            }
        }
        h
    }

    /// 16-char lowercase hex of [`World::state_hash`].
    pub fn hash_hex(&self) -> String {
        format!("{:016x}", self.state_hash())
    }
}

/// Summary of a headless self-test run for CLI/tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfTestReport {
    pub ticks: u64,
    pub seed: u64,
    pub hash: u64,
    pub units_alive: usize,
    pub shots_fired: u64,
    pub kills: u64,
}

/// Build the canonical 10,000-tick scenario: shooter A spawns at x=0 with a
/// 22-range / 100-damage / 1-rof gun and an attack order on passive target B
/// (1100 hp) at x=30 units. A closes in, fires on cooldown, B dies before the
/// run ends. Covers move + shoot + kill.
pub fn run_selftest(ticks: u64, seed: u64) -> (World, SelfTestReport) {
    let mut world = World::new(seed);
    let shooter = UnitParams {
        max_hp: 11_000 * SCALE,
        speed_per_tick: 5 * SCALE / TICK_HZ as i64,
        weapons: vec![SimWeapon {
            damage: 100 * SCALE,
            range_sq: (22 * SCALE as i64) as i128 * (22 * SCALE as i64) as i128,
            cooldown: 0,
            reset: combat::cooldown_reset_ticks(TICK_HZ, SCALE),
        }],
        cost_mass: 1,
        cost_energy: 1,
        build_ticks: 1,
    };
    let target = UnitParams {
        max_hp: 1100 * SCALE,
        speed_per_tick: 0,
        weapons: vec![],
        cost_mass: 1,
        cost_energy: 1,
        build_ticks: 1,
    };
    let a = world.spawn(&UnitSpawn::new("SELFTEST_A", 0, 0), &shooter);
    let b = world.spawn(&UnitSpawn::new("SELFTEST_B", 30 * SCALE, 0), &target);
    world.order(a, Order::Attack(b));
    world.run(ticks);
    let report = SelfTestReport {
        ticks,
        seed,
        hash: world.state_hash(),
        units_alive: world.units_alive(),
        shots_fired: world.shots_fired(),
        kills: world.kills(),
    };
    (world, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shooter_params() -> UnitParams {
        UnitParams {
            max_hp: 11_000 * SCALE,
            speed_per_tick: 5 * SCALE / TICK_HZ as i64,
            weapons: vec![SimWeapon {
                damage: 100 * SCALE,
                range_sq: (22 * SCALE) as i128 * (22 * SCALE) as i128,
                cooldown: 0,
                reset: combat::cooldown_reset_ticks(TICK_HZ, SCALE),
            }],
            cost_mass: 1,
            cost_energy: 1,
            build_ticks: 1,
        }
    }

    #[allow(dead_code)]
    fn target_params() -> UnitParams {
        UnitParams {
            max_hp: 1100 * SCALE,
            speed_per_tick: 0,
            weapons: vec![],
            cost_mass: 1,
            cost_energy: 1,
            build_ticks: 1,
        }
    }

    #[test]
    fn hash_stable_across_runs() {
        let (_, r1) = run_selftest(10_000, DEFAULT_SEED);
        let (_, r2) = run_selftest(10_000, DEFAULT_SEED);
        assert_eq!(r1.hash, r2.hash);
    }

    #[test]
    fn selftest_moves_shoots_kills() {
        let (world, report) = run_selftest(10_000, DEFAULT_SEED);
        assert!(report.shots_fired > 0, "no shots fired");
        assert_eq!(report.kills, 1, "target was not killed");
        assert!(!world.is_alive(2), "target still alive");
        assert!(world.is_alive(1), "shooter died unexpectedly");
        let (ax, _) = world.unit_pos(1).unwrap();
        assert!(ax > 0, "shooter never moved");
        assert_eq!(report.units_alive, 1);
    }

    #[test]
    fn move_order_reaches_waypoint_without_overshoot() {
        let mut w = World::new(1);
        let p = UnitParams {
            max_hp: 100 * SCALE,
            speed_per_tick: 7 * SCALE,
            weapons: vec![],
            cost_mass: 1,
            cost_energy: 1,
            build_ticks: 1,
        };
        let id = w.spawn(&UnitSpawn::new("M", 0, 0), &p);
        w.order(id, Order::Move { x: 10 * SCALE, y: 0 });
        w.run(100);
        assert_eq!(w.unit_pos(id), Some((10 * SCALE, 0)));
    }

    #[test]
    fn unknown_order_target_is_ignored() {
        let mut w = World::new(1);
        let id = w.spawn(&UnitSpawn::new("M", 0, 0), &shooter_params());
        w.order(9999, Order::Move { x: 5, y: 5 });
        w.order(id, Order::Attack(9999));
        w.run(10);
        assert!(w.is_alive(id));
    }

    #[test]
    fn economy_project_drains_stockpiles() {
        let mut w = World::with_stockpiles(1, 18_000 * SCALE, 5_000_000 * SCALE);
        w.build(18_000 * SCALE, 5_000_000 * SCALE, 60_000, 10);
        w.run(6000);
        assert!(w.projects.is_empty(), "project should complete in 6000 ticks");
        assert!(w.mass >= 0 && w.energy >= 0);
    }

    #[test]
    fn from_summary_quantizes_once() {
        let s = UnitSummary {
            id: "UAL0001".to_string(),
            path: "units/UAL0001/UAL0001_unit.bp".to_string(),
            description: None,
            categories: vec![],
            health: Some(11000.0),
            build_cost_mass: Some(18000.0),
            build_cost_energy: Some(5000000.0),
            build_time: Some(60000.0),
            max_speed: Some(5.0),
            motion_type: None,
            weapons: vec![crate::blueprint::WeaponSummary {
                label: Some("RightDisruptor".to_string()),
                damage: Some(100.0),
                max_radius: Some(22.0),
                rate_of_fire: Some(1.0),
            }],
        };
        let p = UnitParams::from_summary(&s);
        assert_eq!(p.max_hp, 11_000_000);
        assert_eq!(p.weapons[0].damage, 100_000);
        assert_eq!(p.weapons[0].reset, 10);
    }

    #[test]
    fn different_seeds_still_deterministic() {
        let (_, r1) = run_selftest(100, 42);
        let (_, r2) = run_selftest(100, 42);
        assert_eq!(r1.hash, r2.hash);
    }
}
