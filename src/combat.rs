//! Phase 4 combat: hitscan weapon fire and damage.
//!
//! Integer-only. Positions, ranges and hit points are milli-units
//! (`SCALE = 1000` in [`crate::sim`]); conversions from blueprint `f64`
//! happen once at spawn time. A hit always lands (no RNG in Phase 4);
//! damage is applied immediately to the target's hp.

/// Ceiling division `tick_hz * 1000 / rate_of_fire_milli`, minimum 1 tick.
/// `rate_of_fire` is shots per second in milli-units (e.g. rof 1.0 -> 1000).
/// Missing/zero rof maps to `u32::MAX` (effectively never refires) so the
/// caller can skip the weapon instead of dividing by zero.
pub fn cooldown_reset_ticks(tick_hz: u64, rate_of_fire_milli: i64) -> u32 {
    if rate_of_fire_milli <= 0 {
        return u32::MAX;
    }
    ((tick_hz as i128 * 1000 + rate_of_fire_milli as i128 - 1) / rate_of_fire_milli as i128)
        .max(1)
        .min(u32::MAX as i128) as u32
}

/// Squared distance between two milli-unit positions. `i128` intermediate so
/// large maps cannot overflow.
pub fn dist_sq(ax: i64, ay: i64, bx: i64, by: i64) -> i128 {
    let dx = ax as i128 - bx as i128;
    let dy = ay as i128 - by as i128;
    dx * dx + dy * dy
}

/// Whether a weapon may fire this tick: cooldown elapsed, target alive, and
/// target within `max_radius` (compared as squares, no sqrt). A missing
/// range (`None`) or missing damage means the weapon never fires.
pub fn can_fire(
    cooldown: u32,
    target_alive: bool,
    range_sq: Option<i128>,
    dist_sq: i128,
    damage: Option<i64>,
) -> bool {
    if cooldown != 0 || !target_alive {
        return false;
    }
    match (range_sq, damage) {
        (Some(r), Some(d)) if d > 0 => dist_sq <= r,
        _ => false,
    }
}

/// Apply a hit: subtract damage, clamp at zero, report whether this hit
/// killed the target. Overkill is discarded.
pub fn apply_hit(target_hp: &mut i64, damage: i64) -> bool {
    *target_hp -= damage;
    if *target_hp <= 0 {
        *target_hp = 0;
        true
    } else {
        false
    }
}

/// Phase 29: one full turn in milliradians (rounded `2π`).
pub const TWO_PI_MRAD: i64 = 6_283;

/// `atan(2^-i)` in nanoradians (CORDIC vectoring table).
const CORDIC_ATAN_NRAD: [i128; 32] = [
    785_398_163, 463_647_609, 244_978_663, 124_354_995, 62_418_810, 31_239_833, 15_623_729, 7_812_341,
    3_906_230, 1_953_123, 976_562, 488_281, 244_141, 122_070, 61_035, 30_518, 15_259, 7_629, 3_815,
    1_907, 954, 477, 238, 119, 60, 30, 15, 7, 4, 2, 1, 0,
];
const PI_NRAD: i128 = 3_141_592_654;

/// Phase 29: integer-only `atan2(y, x)` in milliradians, range `[-π, π]`, bit-identical on
/// every platform (CORDIC vectoring, no floats). `(0, 0)` yields 0.
pub fn atan2_mrad(y: i64, x: i64) -> i64 {
    if x == 0 && y == 0 {
        return 0;
    }
    let (mut x, mut y) = (x as i128, y as i128);
    let mut ang: i128 = 0;
    if x < 0 {
        ang = if y >= 0 { PI_NRAD } else { -PI_NRAD };
        x = -x;
        y = -y;
    }
    x <<= 20;
    y <<= 20;
    for (i, &a) in CORDIC_ATAN_NRAD.iter().enumerate() {
        let (dx, dy) = (y >> i, x >> i);
        if y > 0 {
            x += dx;
            y -= dy;
            ang += a;
        } else {
            x -= dx;
            y += dy;
            ang -= a;
        }
    }
    (ang + 500_000).div_euclid(1_000_000) as i64
}

/// Phase 29: wraps an angle difference into `[-π, π]` milliradians.
pub fn wrap_mrad(a: i64) -> i64 {
    (a + TWO_PI_MRAD / 2).rem_euclid(TWO_PI_MRAD) - TWO_PI_MRAD / 2
}

#[cfg(test)]
mod tests {
    #[test]
    fn integer_atan2_matches_float_within_a_milliradian() {
        let samples = [
            (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (3, -4), (-7, 2), (1_000_000, 3),
            (-3, -2_000_000), (123_456, -654_321), (5, 5_000_000), (-1, 1), (1, -1), (0, 7),
        ];
        for (y, x) in samples {
            let want = ((y as f64).atan2(x as f64) * 1000.0).round() as i64;
            let got = atan2_mrad(y, x);
            assert!((got - want).abs() <= 1, "atan2({y}, {x}) = {got}, want {want}");
        }
        assert_eq!(atan2_mrad(0, 0), 0);
        assert_eq!(wrap_mrad(3_200), 3_200 - TWO_PI_MRAD);
        assert_eq!(wrap_mrad(-3_200), -3_200 + TWO_PI_MRAD);
        assert_eq!(wrap_mrad(1_000), 1_000);
    }

    use super::*;

    #[test]
    fn rof_one_per_second_is_ten_ticks() {
        assert_eq!(cooldown_reset_ticks(10, 1000), 10);
    }

    #[test]
    fn fractional_rof_rounds_up() {
        // UAL0001 overcharge rof 0.3 -> ceil(10/0.3) = 34.
        assert_eq!(cooldown_reset_ticks(10, 300), 34);
    }

    #[test]
    fn zero_rof_never_refires() {
        assert_eq!(cooldown_reset_ticks(10, 0), u32::MAX);
    }

    #[test]
    fn range_compared_as_squares() {
        // Range 22 units = 22000 milli; target at exactly 22 -> in range.
        let r: i128 = 22_000i128 * 22_000i128;
        assert!(can_fire(0, true, Some(r), dist_sq(0, 0, 22_000, 0), Some(100_000)));
        assert!(!can_fire(0, true, Some(r), dist_sq(0, 0, 22_001, 0), Some(100_000)));
    }

    #[test]
    fn cooldown_and_dead_block_fire() {
        let r: i128 = 100;
        assert!(!can_fire(1, true, Some(r), 0, Some(1)));
        assert!(!can_fire(0, false, Some(r), 0, Some(1)));
        assert!(!can_fire(0, true, None, 0, Some(1)));
        assert!(!can_fire(0, true, Some(r), 0, None));
    }

    #[test]
    fn kill_clamps_and_reports() {
        let mut hp = 50_000;
        assert!(!apply_hit(&mut hp, 30_000));
        assert_eq!(hp, 20_000);
        assert!(apply_hit(&mut hp, 100_000));
        assert_eq!(hp, 0);
    }
}
