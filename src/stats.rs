//! Phase 21 end-of-game statistics, sampled UI-side.
//!
//! [`ScoreSampler::observe`] is called after every simulation step with the per-army command
//! counts of that step. It only *reads* the world: economy totals are integrated every tick
//! (gross income = collected; `min(prev + income, cap) - now` = spent, the clamp is waste) and
//! a time-series point (mass/energy income, army value, unit count) is taken every
//! [`SAMPLE_INTERVAL`] ticks. Nothing here can feed back into the deterministic simulation.

use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{ArmyId, Role, Skirmish};

/// Ticks between graph samples (5 s of game time).
pub const SAMPLE_INTERVAL: u64 = 50;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArmyTotals {
    /// Gross mass/energy produced (milli-units).
    pub mass_collected: i64,
    pub mass_spent: i64,
    pub mass_wasted: i64,
    pub energy_collected: i64,
    pub energy_spent: i64,
    pub energy_wasted: i64,
    /// Commands issued by this army (human, AI or network).
    pub actions: u64,
    prev: Option<(i64, i64)>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ScoreSample {
    pub tick: u64,
    /// Per second.
    pub mass_income: f32,
    pub energy_income: f32,
    /// Mass cost of alive mobile, non-commander units.
    pub army_value: f32,
    pub units: u32,
}

/// One row of the score table.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreRow {
    pub army: ArmyId,
    pub name: String,
    pub defeated: bool,
    pub built: u32,
    pub lost: u32,
    pub kills: u32,
    pub damage: i64,
    pub mass_collected: i64,
    pub mass_spent: i64,
    pub energy_collected: i64,
    pub energy_spent: i64,
    pub apm: f64,
}

/// Graph selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Series {
    MassIncome,
    EnergyIncome,
    ArmyValue,
    UnitCount,
}

impl Series {
    pub const ALL: [Series; 4] = [Series::MassIncome, Series::EnergyIncome, Series::ArmyValue, Series::UnitCount];

    pub fn title(self) -> &'static str {
        match self {
            Series::MassIncome => "MASS INCOME /S",
            Series::EnergyIncome => "ENERGY INCOME /S",
            Series::ArmyValue => "ARMY VALUE (MASS)",
            Series::UnitCount => "UNIT COUNT",
        }
    }

    pub fn value(self, s: &ScoreSample) -> f32 {
        match self {
            Series::MassIncome => s.mass_income,
            Series::EnergyIncome => s.energy_income,
            Series::ArmyValue => s.army_value,
            Series::UnitCount => s.units as f32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoreSampler {
    pub interval: u64,
    pub totals: Vec<ArmyTotals>,
    /// Per army, oldest first.
    pub series: Vec<Vec<ScoreSample>>,
    last_tick: Option<u64>,
}

impl Default for ScoreSampler {
    fn default() -> Self {
        Self::new(SAMPLE_INTERVAL)
    }
}

impl ScoreSampler {
    pub fn new(interval: u64) -> Self {
        Self {
            interval: interval.max(1),
            totals: Vec::new(),
            series: Vec::new(),
            last_tick: None,
        }
    }

    fn ensure(&mut self, n: usize) {
        if self.totals.len() < n {
            self.totals.resize(n, ArmyTotals::default());
            self.series.resize(n, Vec::new());
        }
    }

    /// Baseline at the current tick (storage levels + first graph point).
    pub fn prime(&mut self, world: &Skirmish) {
        self.ensure(world.armies().len());
        for (t, a) in self.totals.iter_mut().zip(world.armies()) {
            t.prev = Some((a.mass, a.energy));
        }
        self.sample(world);
        self.last_tick = Some(world.tick());
    }

    /// Integrates one stepped tick. `actions` = commands applied this step per army.
    pub fn observe(&mut self, world: &Skirmish, actions: &[(ArmyId, u32)]) {
        let tick = world.tick();
        if self.last_tick == Some(tick) {
            return;
        }
        self.last_tick = Some(tick);
        self.ensure(world.armies().len());
        for &(army, n) in actions {
            if let Some(t) = self.totals.get_mut(army as usize) {
                t.actions += u64::from(n);
            }
        }
        for (t, a) in self.totals.iter_mut().zip(world.armies()) {
            let (pm, pe) = t.prev.unwrap_or((a.mass, a.energy));
            let (sm, wm) = spent_and_waste(pm, a.income_mass, a.mass_cap, a.mass);
            let (se, we) = spent_and_waste(pe, a.income_energy, a.energy_cap, a.energy);
            t.mass_collected += a.income_mass;
            t.energy_collected += a.income_energy;
            t.mass_spent += sm;
            t.mass_wasted += wm;
            t.energy_spent += se;
            t.energy_wasted += we;
            t.prev = Some((a.mass, a.energy));
        }
        if tick % self.interval == 0 {
            self.sample(world);
        }
    }

    fn sample(&mut self, world: &Skirmish) {
        let n = world.armies().len();
        self.ensure(n);
        let mut value = vec![0i64; n];
        let mut units = vec![0u32; n];
        for u in world.units() {
            let i = u.army as usize;
            if i >= n {
                continue;
            }
            units[i] += 1;
            if !u.role.is_structure() && u.role != Role::Commander {
                value[i] += world.catalog().get(&u.type_id).map(|t| t.cost_mass).unwrap_or(0);
            }
        }
        let per_sec = TICK_HZ as f32 / SCALE as f32;
        for (i, a) in world.armies().iter().enumerate() {
            let s = ScoreSample {
                tick: world.tick(),
                mass_income: a.income_mass as f32 * per_sec,
                energy_income: a.income_energy as f32 * per_sec,
                army_value: value[i] as f32 / SCALE as f32,
                units: units[i],
            };
            let ser = &mut self.series[i];
            if ser.last().is_some_and(|l| l.tick == s.tick) {
                ser.pop();
            }
            ser.push(s);
        }
    }

    /// Final score table rows (one per army) at the world's current state.
    pub fn rows(&self, world: &Skirmish) -> Vec<ScoreRow> {
        let minutes = (world.tick() as f64 / (60 * TICK_HZ) as f64).max(1.0 / 60.0);
        world
            .armies()
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let t = self.totals.get(i).cloned().unwrap_or_default();
                ScoreRow {
                    army: a.id,
                    name: a.name.clone(),
                    defeated: a.defeated,
                    built: a.units_built,
                    lost: a.units_lost,
                    kills: a.kills,
                    damage: a.damage_dealt / SCALE,
                    mass_collected: t.mass_collected / SCALE,
                    mass_spent: t.mass_spent / SCALE,
                    energy_collected: t.energy_collected / SCALE,
                    energy_spent: t.energy_spent / SCALE,
                    apm: t.actions as f64 / minutes,
                }
            })
            .collect()
    }

    /// Largest value of `which` over all armies (graph scaling).
    pub fn max_of(&self, which: Series) -> f32 {
        self.series
            .iter()
            .flat_map(|s| s.iter())
            .map(|s| which.value(s))
            .fold(0.0, f32::max)
    }

    pub fn last_sample_tick(&self) -> u64 {
        self.series.iter().filter_map(|s| s.last()).map(|s| s.tick).max().unwrap_or(0)
    }
}

/// `(spent, wasted)` for one tick: income is added first and clamped at the cap (waste);
/// whatever is missing at the end of the tick was spent.
pub fn spent_and_waste(prev: i64, income: i64, cap: i64, now: i64) -> (i64, i64) {
    let gross = prev + income;
    let after_income = gross.min(cap.max(prev)).min(cap.max(0)).max(0);
    let wasted = (gross - after_income).max(0);
    let spent = (after_income - now).max(0);
    (spent, wasted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::play::PlaySession;
    use crate::skirmish::tests::test_world;

    #[test]
    fn spent_and_waste_accounting() {
        assert_eq!(spent_and_waste(100, 10, 1000, 80), (30, 0));
        assert_eq!(spent_and_waste(995, 10, 1000, 1000), (0, 5));
        assert_eq!(spent_and_waste(995, 10, 1000, 990), (10, 5));
        assert_eq!(spent_and_waste(0, 0, 1000, 0), (0, 0));
    }

    #[test]
    fn sampler_aggregates_totals_series_and_apm() {
        let mut s = PlaySession::new(test_world(31), 0, true, 1);
        let n = s.world.armies().len();
        let mut income = vec![(0i64, 0i64); n];
        let mut actions = vec![0u64; n];
        let mut prev: Vec<i64> = s.world.armies().iter().map(|a| a.mass).collect();
        let mut spent = vec![0i64; n];
        for _ in 0..600 {
            // Independent bookkeeping mirroring what the sampler must compute.
            for ai in &mut s.ais {
                ai.think(&mut s.world);
            }
            for (a, _) in s.world.pending_view() {
                actions[*a as usize] += 1;
            }
            let counts = s.recorder.as_mut().unwrap().record_pending(&s.world);
            s.world.step();
            s.stats.observe(&s.world, &counts);
            for (i, a) in s.world.armies().iter().enumerate() {
                income[i].0 += a.income_mass;
                income[i].1 += a.income_energy;
                spent[i] += spent_and_waste(prev[i], a.income_mass, a.mass_cap, a.mass).0;
                prev[i] = a.mass;
            }
            // Observing the same tick twice is a no-op.
            s.stats.observe(&s.world, &counts);
        }
        assert_eq!(s.world.tick(), 600);
        for i in 0..n {
            let t = &s.stats.totals[i];
            assert_eq!(t.mass_collected, income[i].0);
            assert_eq!(t.energy_collected, income[i].1);
            assert_eq!(t.mass_spent, spent[i]);
            assert_eq!(t.actions, actions[i]);
            assert!(t.actions > 0, "AI issued commands");
            // tick 0 + every 50 ticks up to 600.
            assert_eq!(s.stats.series[i].len(), 1 + 600 / SAMPLE_INTERVAL as usize);
            assert!(s.stats.series[i].windows(2).all(|w| w[1].tick - w[0].tick == SAMPLE_INTERVAL));
        }
        let rows = s.stats.rows(&s.world);
        assert_eq!(rows.len(), n);
        // 600 ticks = 1 minute: APM equals the action count.
        assert!((rows[0].apm - actions[0] as f64).abs() < 1e-9);
        assert_eq!(rows[1].mass_collected, income[1].0 / SCALE);
        let last = s.stats.series[0].last().unwrap();
        let units0 = s.world.units().filter(|u| u.army == 0).count() as u32;
        assert_eq!(last.units, units0);
        assert!(s.stats.max_of(Series::UnitCount) >= units0 as f32);
        assert_eq!(s.stats.last_sample_tick(), 600);
    }
}
