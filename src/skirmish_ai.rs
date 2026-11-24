//! Phase 6 scripted opponent.
//!
//! A deterministic, rule-based AI that drives one army purely through
//! [`Command`]s (exactly the interface a human player uses). It reads the
//! world read-only once per second of game time and never touches sim state
//! directly, so AI armies replay bit-identically.
//!
//! Behaviour:
//! - Commander: fixed opening (2 mex, factory, power), then keeps adding
//!   mex/power/factories up to a cap, placing structures on a deterministic
//!   lattice around its start position.
//! - Factories: rally toward the enemy and loop production (2:1 mix of the
//!   faction's two land units).
//! - Army: idle units gather at home; once a wave is big enough everything
//!   attack-moves at the enemy commander; waves grow over time. Units near
//!   home react to intruders.

use crate::skirmish::{ArmyId, Command, Role, Skirmish, UnitId};
use crate::sim::SCALE;

/// Ticks between decisions (1 s of game time).
pub const THINK_INTERVAL: u64 = 10;
const HOME_RADIUS: i64 = 80 * SCALE;
const DEFEND_RADIUS: i64 = 70 * SCALE;
const MAX_FACTORIES: usize = 4;
const MAX_GENERATORS: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptedAi {
    pub army: ArmyId,
    pub wave_size: usize,
    pub waves_sent: u32,
    produced: u64,
}

fn dist_sq(a: (i64, i64), b: (i64, i64)) -> i128 {
    let dx = (a.0 - b.0) as i128;
    let dy = (a.1 - b.1) as i128;
    dx * dx + dy * dy
}

impl ScriptedAi {
    pub fn new(army: ArmyId) -> Self {
        Self {
            army,
            wave_size: 8,
            waves_sent: 0,
            produced: 0,
        }
    }

    /// The first undefeated enemy army (lowest id).
    fn enemy(&self, w: &Skirmish) -> Option<ArmyId> {
        w.armies()
            .iter()
            .find(|a| a.id != self.army && !a.defeated)
            .map(|a| a.id)
    }

    /// Issue this tick's commands into `w`. Cheap no-op between decisions.
    pub fn think(&mut self, w: &mut Skirmish) {
        let cmds = self.decide(w);
        for c in cmds {
            w.issue(self.army, c);
        }
    }

    /// Pure decision function: world snapshot in, commands out.
    pub fn decide(&mut self, w: &Skirmish) -> Vec<Command> {
        let mut out = Vec::new();
        // Stagger armies so they do not all think on the same tick.
        if w.result().is_some()
            || (w.tick() + self.army as u64) % THINK_INTERVAL != 0
        {
            return out;
        }
        let Some(me) = w.army(self.army) else {
            return out;
        };
        if me.defeated {
            return out;
        }
        let Some(faction) = w.faction(self.army).cloned() else {
            return out;
        };
        let start = me.start;
        let enemy = self.enemy(w);
        let enemy_start = enemy.and_then(|e| w.army(e)).map(|a| a.start);
        let enemy_target = enemy
            .and_then(|e| w.commander_of(e))
            .map(|u| (u.pos.x, u.pos.y))
            .or(enemy_start);

        self.plan_commander(w, &faction, start, &mut out);

        // Factories: rally + production loop.
        for f in w.units().filter(|u| u.army == self.army && u.role == Role::Factory) {
            if f.rally.is_none()
                && let Some(t) = enemy_start
            {
                let dx = t.0 - f.pos.x;
                let dy = t.1 - f.pos.y;
                let d = crate::skirmish::isqrt((dx as i128) * (dx as i128) + (dy as i128) * (dy as i128)).max(1);
                let step = 20 * SCALE;
                out.push(Command::Rally {
                    factory: f.id,
                    x: f.pos.x + dx * step / d,
                    y: f.pos.y + dy * step / d,
                });
            }
            if f.production.is_empty() && f.build.is_none() && !faction.land_units.is_empty() {
                let pick = if faction.land_units.len() > 1 && self.produced % 3 == 2 { 1 } else { 0 };
                self.produced += 1;
                out.push(Command::Produce {
                    factory: f.id,
                    unit_type: faction.land_units[pick].clone(),
                    count: 1,
                });
            }
        }

        // Army control.
        let mobiles: Vec<(UnitId, (i64, i64), bool)> = w
            .units()
            .filter(|u| u.army == self.army && u.role == Role::Mobile && !u.weapons.is_empty())
            .map(|u| (u.id, (u.pos.x, u.pos.y), u.orders.is_empty()))
            .collect();
        let home_sq = (HOME_RADIUS as i128) * (HOME_RADIUS as i128);
        let idle_home: Vec<UnitId> = mobiles
            .iter()
            .filter(|(_, p, idle)| *idle && dist_sq(*p, start) <= home_sq)
            .map(|(id, _, _)| *id)
            .collect();
        let idle_away: Vec<UnitId> = mobiles
            .iter()
            .filter(|(_, p, idle)| *idle && dist_sq(*p, start) > home_sq)
            .map(|(id, _, _)| *id)
            .collect();

        // Defend against intruders first.
        let def_sq = (DEFEND_RADIUS as i128) * (DEFEND_RADIUS as i128);
        let intruder = w
            .units()
            .filter(|u| u.army != self.army && !u.role.is_structure())
            .map(|u| (dist_sq((u.pos.x, u.pos.y), start), u.id, (u.pos.x, u.pos.y)))
            .filter(|(d, _, _)| *d <= def_sq)
            .min();
        if let Some((_, _, p)) = intruder {
            if !idle_home.is_empty() {
                out.push(Command::AttackMove { units: idle_home, x: p.0, y: p.1, queue: false });
            }
        } else if idle_home.len() >= self.wave_size
            && let Some(t) = enemy_target
        {
            out.push(Command::AttackMove { units: idle_home, x: t.0, y: t.1, queue: false });
            self.waves_sent += 1;
            self.wave_size = (self.wave_size + 4).min(40);
        }
        // Survivors of a wave keep pushing toward the enemy commander.
        if !idle_away.is_empty()
            && let Some(t) = enemy_target
        {
            out.push(Command::AttackMove { units: idle_away, x: t.0, y: t.1, queue: false });
        }
        out
    }

    fn plan_commander(
        &self,
        w: &Skirmish,
        faction: &crate::skirmish::Faction,
        start: (i64, i64),
        out: &mut Vec<Command>,
    ) {
        let Some(acu) = w.commander_of(self.army) else {
            return;
        };
        if !acu.orders.is_empty() || acu.build.is_some() {
            return;
        }
        let count = |role: Role| {
            w.units()
                .filter(|u| u.army == self.army && u.role == role)
                .count()
        };
        let (mex, fac, pgens) = (count(Role::Extractor), count(Role::Factory), count(Role::Generator));
        let near_mex = |r: i64| w.nearest_free_mass_point(start, r * SCALE).map(|(_, p)| p);
        let pick = if mex < 2 && near_mex(120).is_some() {
            Some((faction.extractor.clone(), near_mex(120)))
        } else if fac < 1 {
            Some((faction.factory.clone(), None))
        } else if pgens < fac + 1 && pgens < MAX_GENERATORS {
            Some((faction.generator.clone(), None))
        } else if mex < 4 + fac && near_mex(160).is_some() {
            Some((faction.extractor.clone(), near_mex(160)))
        } else if fac < MAX_FACTORIES {
            Some((faction.factory.clone(), None))
        } else if pgens < MAX_GENERATORS {
            Some((faction.generator.clone(), None))
        } else {
            near_mex(260).map(|p| (faction.extractor.clone(), Some(p)))
        };
        let Some((unit_type, at)) = pick else {
            return;
        };
        let site = at.or_else(|| Self::find_site(w, &unit_type, start));
        if let Some((x, y)) = site {
            out.push(Command::Build { builder: acu.id, unit_type, x, y, queue: false });
        }
    }

    /// First free lattice point around `start`, ring by ring, in a fixed
    /// order (deterministic placement).
    fn find_site(w: &Skirmish, unit_type: &str, start: (i64, i64)) -> Option<(i64, i64)> {
        let spacing = if w.catalog().get(unit_type).map(|t| t.role) == Some(Role::Factory) {
            12 * SCALE
        } else {
            6 * SCALE
        };
        for ring in 1..=14i64 {
            for dy in -ring..=ring {
                for dx in -ring..=ring {
                    if dx.abs() != ring && dy.abs() != ring {
                        continue;
                    }
                    let (x, y) = (start.0 + dx * spacing, start.1 + dy * spacing);
                    if w.site_is_free(unit_type, x, y) {
                        return Some((x, y));
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skirmish::tests::test_world;

    #[test]
    fn ai_vs_ai_builds_fights_and_is_deterministic() {
        let play = || {
            let mut w = test_world(11);
            let mut ais = [ScriptedAi::new(0), ScriptedAi::new(1)];
            for _ in 0..12_000 {
                for ai in &mut ais {
                    ai.think(&mut w);
                }
                w.step();
                if w.result().is_some() {
                    break;
                }
            }
            w
        };
        let a = play();
        let b = play();
        assert_eq!(a.hash_hex(), b.hash_hex());
        assert!(a.armies().iter().all(|x| x.units_built >= 3), "{:?}", a.armies());
        assert!(a.shots_fired() > 0);
        assert!(a.kills() > 0);
    }
}
