//! Phase 4 economy: construction projects paid from mass/energy stockpiles.
//!
//! All quantities are integers. Mass/energy costs are milli-units
//! (`SCALE = 1000` in [`crate::sim`]); work is in abstract work-units where
//! one builder with `BuildRate = r` contributes `r` work-units per tick.
//! Blueprint `f64` values are converted once at spawn time, never here.

/// One unit under construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructionProject {
    /// Stable id; projects iterate in ascending id order.
    pub id: u64,
    /// Total cost over the whole project, milli-mass / milli-energy.
    pub total_mass: i64,
    /// Total cost over the whole project, milli-energy.
    pub total_energy: i64,
    /// Total work required (`build_time`, minimum 1). Same units as `build_rate`.
    pub work_total: i64,
    /// Work done so far.
    pub progress: i64,
    /// Builder rate (`BuildRate`, units of work per tick, minimum 1).
    pub build_rate: i64,
}

#[allow(dead_code)]
impl ConstructionProject {
    /// New project. Zero/negative totals are clamped to the spec minimum of 1
    /// so per-step division can never divide by zero.
    pub fn new(id: u64, total_mass: i64, total_energy: i64, work_total: i64, build_rate: i64) -> Self {
        Self {
            id,
            total_mass: total_mass.max(1),
            total_energy: total_energy.max(1),
            work_total: work_total.max(1),
            progress: 0,
            build_rate: build_rate.max(1),
        }
    }

    /// Cost of one full tick of work, milli-mass and milli-energy.
    /// Uses `i128` intermediates; truncates like the blueprint edge conversion.
    pub fn cost_per_step(&self) -> (i64, i64) {
        let m = (self.total_mass as i128 * self.build_rate as i128) / self.work_total as i128;
        let e = (self.total_energy as i128 * self.build_rate as i128) / self.work_total as i128;
        (m.max(0) as i64, e.max(0) as i64)
    }

    /// Run one tick against `stock` (`(mass, energy)` milli-units, updated in
    /// place, clamped at zero, never debt). Returns `true` when the project
    /// completes on this tick.
    ///
    /// Affordable fraction `f = min(1, mass_have/cost_m, energy_have/cost_e)`:
    /// full step when affordable, otherwise proportional partial progress and
    /// the stockpile is drained to zero on the limiting resource.
    pub fn step(&mut self, stock: &mut (i64, i64)) -> bool {
        let (cost_m, cost_e) = self.cost_per_step();
        // Zero-cost steps (tiny project vs. huge work_total) still advance.
        if cost_m <= 0 && cost_e <= 0 {
            self.progress += self.build_rate;
            return self.progress >= self.work_total;
        }
        let pay_m = stock.0.min(cost_m);
        let pay_e = stock.1.min(cost_e);
        // Fraction of a full step we can afford, scaled to 1_000_000 for
        // integer math. If a cost leg is zero it never limits.
        let mut frac_ppm: i128 = 1_000_000;
        if cost_m > 0 {
            frac_ppm = frac_ppm.min(pay_m as i128 * 1_000_000 / cost_m as i128);
        }
        if cost_e > 0 {
            frac_ppm = frac_ppm.min(pay_e as i128 * 1_000_000 / cost_e as i128);
        }
        stock.0 -= pay_m;
        stock.1 -= pay_e;
        debug_assert!(stock.0 >= 0 && stock.1 >= 0);
        self.progress += (self.build_rate as i128 * frac_ppm / 1_000_000) as i64;
        self.progress >= self.work_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_steps_complete() {
        let mut p = ConstructionProject::new(0, 18_000_000, 5_000_000_000, 60_000, 10);
        let mut stock = (1_000_000_000, 9_999_000_000_000);
        let mut ticks = 0;
        while !p.step(&mut stock) {
            ticks += 1;
            assert!(ticks < 100_000, "project stalled");
        }
        // 60000 work / 10 per tick = 6000 ticks.
        assert_eq!(ticks + 1, 6000);
        assert!(stock.0 >= 0 && stock.1 >= 0);
    }

    #[test]
    fn poor_stockpiles_stall_without_debt() {
        let mut p = ConstructionProject::new(0, 1_000_000, 1_000_000, 100, 10);
        let mut stock = (0, 0);
        for _ in 0..50 {
            assert!(!p.step(&mut stock));
        }
        assert_eq!(p.progress, 0);
        assert_eq!(stock, (0, 0));
    }

    #[test]
    fn partial_payment_makes_partial_progress() {
        let mut p = ConstructionProject::new(0, 1_000, 1_000, 10, 10);
        // cost/step = 1000*10/10 = 1000 each; pay half of each.
        let mut stock = (500, 500);
        assert!(!p.step(&mut stock));
        assert_eq!(p.progress, 5);
        assert_eq!(stock, (0, 0));
    }

    #[test]
    fn zero_totals_cannot_divide_by_zero() {
        // Clamped to minimums: stalls on empty stock, completes when funded.
        let mut p = ConstructionProject::new(0, 0, 0, 0, 0);
        let mut stock = (0, 0);
        assert!(!p.step(&mut stock));
        let mut funded = (100, 100);
        assert!(p.step(&mut funded));
    }
}
