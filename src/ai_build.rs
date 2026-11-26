//! Phase 19 faction opening build orders and pure build-step selection.
//!
//! An opening is a flat list of structure [`Step`]s for the commander plus a list of
//! [`FactoryStep`]s for the first land factory. Selection is count-based and therefore
//! self-correcting: the next step is the first one whose cumulative requirement for its
//! kind exceeds what already exists (built, under construction or reserved), so steps
//! completed by engineers, or structures lost to raids, are handled automatically.

/// Structure kinds a build order can ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(dead_code)]
pub enum Step {
    Mex,
    Pgen,
    Hydro,
    LandFactory,
    AirFactory,
    NavalFactory,
    EnergyStorage,
    Radar,
    PointDefense,
}

pub const STEP_KINDS: usize = 9;

impl Step {
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Opening production of the first land factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactoryStep {
    Engineer,
    Scout,
    Line,
}

/// Count per [`Step`] kind (existing + under construction + reserved).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepCounts(pub [u32; STEP_KINDS]);

impl StepCounts {
    pub fn get(&self, s: Step) -> u32 {
        self.0[s.index()]
    }
    pub fn add(&mut self, s: Step, n: u32) {
        self.0[s.index()] += n;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Opening {
    pub acu: &'static [Step],
    pub factory: &'static [FactoryStep],
}

use FactoryStep::{Engineer, Line, Scout};
use Step::{AirFactory, Hydro, LandFactory, Mex, Pgen};

/// UEF: classic land opener, double factory before the first air.
const UEF_ACU: &[Step] = &[LandFactory, Mex, Mex, Pgen, Pgen, Mex, Mex, Pgen, LandFactory, Pgen, Pgen];
/// Cybran: extra early power for an air factory after the second land factory.
const CYBRAN_ACU: &[Step] = &[
    LandFactory, Mex, Mex, Pgen, Pgen, Pgen, Mex, Mex, LandFactory, Pgen, AirFactory, Pgen,
];
/// Aeon: grabs a hydrocarbon early (cheap, large energy) when one is near.
const AEON_ACU: &[Step] = &[LandFactory, Mex, Mex, Pgen, Hydro, Pgen, Mex, Mex, LandFactory, Pgen, Pgen];
/// Seraphim: power first to keep the energy-hungry roster flowing.
const SERAPHIM_ACU: &[Step] = &[LandFactory, Pgen, Mex, Mex, Pgen, Mex, Mex, Pgen, LandFactory, Pgen, Pgen];
/// Easy: short, loose opener (under-builds power, no second factory).
const EASY_ACU: &[Step] = &[LandFactory, Mex, Pgen, Mex, Pgen];

const UEF_FAC: &[FactoryStep] = &[Engineer, Engineer, Scout, Engineer, Line, Engineer, Line];
const CYBRAN_FAC: &[FactoryStep] = &[Engineer, Scout, Engineer, Engineer, Line, Line, Engineer];
const AEON_FAC: &[FactoryStep] = &[Engineer, Engineer, Engineer, Scout, Line, Engineer];
const SERAPHIM_FAC: &[FactoryStep] = &[Engineer, Engineer, Scout, Engineer, Engineer, Line];
const EASY_FAC: &[FactoryStep] = &[Engineer, Line, Line, Engineer];

/// Opening for a faction key (`uef`, `cybran`, `aeon`, `seraphim`; unknown keys get UEF's).
pub fn opening(faction_key: &str, sloppy: bool) -> Opening {
    if sloppy {
        return Opening { acu: EASY_ACU, factory: EASY_FAC };
    }
    match faction_key.to_ascii_lowercase().as_str() {
        "cybran" => Opening { acu: CYBRAN_ACU, factory: CYBRAN_FAC },
        "aeon" => Opening { acu: AEON_ACU, factory: AEON_FAC },
        "seraphim" => Opening { acu: SERAPHIM_ACU, factory: SERAPHIM_FAC },
        _ => Opening { acu: UEF_ACU, factory: UEF_FAC },
    }
}

/// First unsatisfied step of `order` given `have`, skipping kinds `available` rejects.
pub fn next_step(order: &[Step], have: &StepCounts, available: impl Fn(Step) -> bool) -> Option<(usize, Step)> {
    let mut need = [0u32; STEP_KINDS];
    for (i, &s) in order.iter().enumerate() {
        need[s.index()] += 1;
        if need[s.index()] > have.get(s) && available(s) {
            return Some((i, s));
        }
    }
    None
}

/// Up to `max` upcoming steps, assuming each picked step gets started.
pub fn plan_steps(order: &[Step], have: &StepCounts, available: impl Fn(Step) -> bool, max: usize) -> Vec<Step> {
    let mut h = *have;
    let mut out = Vec::new();
    while out.len() < max {
        let Some((_, s)) = next_step(order, &h, &available) else {
            break;
        };
        h.add(s, 1);
        out.push(s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(pairs: &[(Step, u32)]) -> StepCounts {
        let mut c = StepCounts::default();
        for &(s, n) in pairs {
            c.add(s, n);
        }
        c
    }

    #[test]
    fn next_step_walks_order_by_cumulative_counts() {
        let o = opening("uef", false).acu;
        assert_eq!(next_step(o, &StepCounts::default(), |_| true), Some((0, LandFactory)));
        let c = counts(&[(LandFactory, 1), (Mex, 1)]);
        assert_eq!(next_step(o, &c, |_| true), Some((2, Mex)));
        // Engineers already grabbed four mexes: the ACU moves straight on to power.
        let c = counts(&[(LandFactory, 1), (Mex, 4)]);
        assert_eq!(next_step(o, &c, |_| true), Some((3, Pgen)));
        let c = counts(&[(LandFactory, 2), (Mex, 4), (Pgen, 5)]);
        assert_eq!(next_step(o, &c, |_| true), None, "opening complete");
        // Losing a generator re-opens the matching step.
        let c = counts(&[(LandFactory, 2), (Mex, 4), (Pgen, 4)]);
        assert_eq!(next_step(o, &c, |_| true), Some((10, Pgen)));
    }

    #[test]
    fn unavailable_steps_are_skipped() {
        let o = opening("aeon", false).acu;
        let c = counts(&[(LandFactory, 1), (Mex, 2), (Pgen, 1)]);
        assert_eq!(next_step(o, &c, |_| true), Some((4, Hydro)));
        assert_eq!(next_step(o, &c, |s| s != Hydro), Some((5, Pgen)));
        assert_eq!(next_step(o, &c, |_| false), None);
    }

    #[test]
    fn plan_steps_queues_distinct_upcoming_steps() {
        let o = opening("cybran", false).acu;
        let c = counts(&[(LandFactory, 1)]);
        assert_eq!(plan_steps(o, &c, |_| true, 3), vec![Mex, Mex, Pgen]);
        assert_eq!(plan_steps(o, &c, |s| s != Mex, 2), vec![Pgen, Pgen]);
        let all = plan_steps(o, &StepCounts::default(), |_| true, 100);
        assert_eq!(all.len(), o.len());
    }

    #[test]
    fn factions_have_distinct_openings_and_easy_is_short() {
        let keys = ["uef", "cybran", "aeon", "seraphim"];
        for (i, a) in keys.iter().enumerate() {
            for b in &keys[i + 1..] {
                assert_ne!(opening(a, false).acu, opening(b, false).acu, "{a} vs {b}");
            }
            assert_eq!(opening(a, false).acu[0], LandFactory);
            assert!(opening(a, true).acu.len() < opening(a, false).acu.len());
        }
        assert_eq!(opening("UEF", false), opening("nomads", false));
    }
}
