//! Phase 19: the original (Phases 6, 8 & 12) scripted opponent, preserved verbatim as the
//! `legacy` difficulty so `farust ai-arena` can measure the new AI against it.
//!
//! Deterministic rule-based AI operating strictly through [`Command`]s:
//! - Commander: builds extractors, Land/Air/Naval Factories, Hydro, Power Generators,
//!   and adjacent Mass Storage.
//! - Factories: produce Engineers (up to 3), T1/T2/T3 Land, Air, and Naval units
//!   plus factory upgrades while setting forward rally points.
//! - Engineers: expand to outer mass points, construct Point Defense, Radar,
//!   Bubble Shields, Anti-Air, Tactical Missile Launchers, late-game Experimentals,
//!   and reclaim nearby wrecks.
//! - Silos: fire loaded missiles at high-value enemy targets.
//! - Army: stages waves at home, defends intruders, and attack-moves enemy commander.

use crate::flowfield::NavDomain;
use crate::sim::SCALE;
use crate::skirmish::{ArmyId, Command, Faction, Role, Skirmish, UnitId, UnitOrder};

pub const LEGACY_THINK_INTERVAL: u64 = 10;
const HOME_RADIUS: i64 = 80 * SCALE;
const DEFEND_RADIUS: i64 = 75 * SCALE;
const MAX_FACTORIES: usize = 4;
const MAX_GENERATORS: usize = 10;
const MAX_ENGINEERS: usize = 3;
const MAX_POINT_DEFENSE: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyAi {
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

impl LegacyAi {
    pub fn new(army: ArmyId) -> Self {
        Self {
            army,
            wave_size: 8,
            waves_sent: 0,
            produced: 0,
        }
    }

    fn enemy(&self, w: &Skirmish) -> Option<ArmyId> {
        w.armies()
            .iter()
            .find(|a| !w.are_allies(self.army, a.id) && !a.defeated)
            .map(|a| a.id)
    }

    #[allow(dead_code)]
    pub fn think(&mut self, w: &mut Skirmish) {
        let cmds = self.decide(w);
        for c in cmds {
            w.issue(self.army, c);
        }
    }

    pub fn decide(&mut self, w: &Skirmish) -> Vec<Command> {
        let mut out = Vec::new();
        if w.result().is_some() || (w.tick() + self.army as u64) % LEGACY_THINK_INTERVAL != 0 {
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
        self.plan_engineers(w, &faction, start, enemy_start, &mut out);

        // Factories: rally + Engineers / Land (T1-T3) / Air / Naval production & upgrades.
        let n_eng = w
            .units()
            .filter(|u| u.army == self.army && u.role == Role::Engineer)
            .count();
        let queued_eng: usize = w
            .units()
            .filter(|u| u.army == self.army && u.role == Role::Factory)
            .map(|f| {
                f.production
                    .iter()
                    .filter(|id| **id == faction.engineer)
                    .count()
                    + usize::from(
                        f.build
                            .as_ref()
                            .is_some_and(|j| j.unit_type == faction.engineer),
                    )
            })
            .sum();
        let mut eng_needed = MAX_ENGINEERS.saturating_sub(n_eng + queued_eng);

        let land_fac_count = w
            .units()
            .filter(|u| u.army == self.army && u.role == Role::Factory)
            .count();
        let mut upgrading_factories = w
            .units()
            .filter(|u| {
                u.army == self.army
                    && u.role.is_factory()
                    && matches!(u.orders.front(), Some(UnitOrder::Upgrade(_)))
            })
            .count();
        let combat_mobiles = w
            .units()
            .filter(|u| {
                u.army == self.army
                    && (u.role == Role::Mobile || u.role == Role::Experimental)
                    && !u.weapons.is_empty()
            })
            .count();

        for f in w
            .units()
            .filter(|u| u.army == self.army && u.role.is_factory())
        {
            if f.rally.is_none()
                && let Some(t) = enemy_start
            {
                let dx = t.0 - f.pos.x;
                let dy = t.1 - f.pos.y;
                let d = crate::skirmish::isqrt(
                    (dx as i128) * (dx as i128) + (dy as i128) * (dy as i128),
                )
                .max(1);
                let step = 20 * SCALE;
                out.push(Command::Rally {
                    factory: f.id,
                    x: f.pos.x + dx * step / d,
                    y: f.pos.y + dy * step / d,
                });
            }
            if !f.production.is_empty() || f.build.is_some() || !f.orders.is_empty() {
                continue;
            }
            match f.role {
                Role::Factory => {
                    if upgrading_factories == 0
                        && f.tier == 1
                        && !faction.t2_land_factory.is_empty()
                        && w.catalog().get(&faction.t2_land_factory).is_some()
                        && (land_fac_count >= 2
                            || (me.mass >= 240 * SCALE
                                && me.energy >= 900 * SCALE
                                && combat_mobiles >= 4))
                    {
                        upgrading_factories += 1;
                        out.push(Command::Upgrade {
                            unit: f.id,
                            target_type: faction.t2_land_factory.clone(),
                        });
                        continue;
                    }
                    if upgrading_factories == 0
                        && f.tier == 2
                        && !faction.t3_land_factory.is_empty()
                        && w.catalog().get(&faction.t3_land_factory).is_some()
                        && combat_mobiles >= 12
                        && me.mass >= 450 * SCALE
                        && me.energy >= 2000 * SCALE
                    {
                        upgrading_factories += 1;
                        out.push(Command::Upgrade {
                            unit: f.id,
                            target_type: faction.t3_land_factory.clone(),
                        });
                        continue;
                    }

                    if eng_needed > 0 && w.catalog().get(&faction.engineer).is_some() {
                        eng_needed -= 1;
                        out.push(Command::Produce {
                            factory: f.id,
                            unit_type: faction.engineer.clone(),
                            count: 1,
                        });
                    } else if f.tier >= 3 && !faction.t3_land_units.is_empty() && self.produced % 3 != 0
                    {
                        let idx = (self.produced as usize) % faction.t3_land_units.len();
                        self.produced += 1;
                        out.push(Command::Produce {
                            factory: f.id,
                            unit_type: faction.t3_land_units[idx].clone(),
                            count: 1,
                        });
                    } else if f.tier >= 2 && !faction.t2_land_units.is_empty() && self.produced % 2 == 1
                    {
                        let idx = (self.produced as usize) % faction.t2_land_units.len();
                        self.produced += 1;
                        out.push(Command::Produce {
                            factory: f.id,
                            unit_type: faction.t2_land_units[idx].clone(),
                            count: 1,
                        });
                    } else if !faction.land_units.is_empty() {
                        let n = faction.land_units.len();
                        let pick = match self.produced % 5 {
                            3 if n > 1 => 1,
                            4 if n > 2 => 2,
                            _ => 0,
                        };
                        self.produced += 1;
                        out.push(Command::Produce {
                            factory: f.id,
                            unit_type: faction.land_units[pick].clone(),
                            count: 1,
                        });
                    }
                }
                Role::AirFactory => {
                    if !faction.air_units.is_empty() {
                        let idx = (self.produced as usize) % faction.air_units.len();
                        self.produced += 1;
                        out.push(Command::Produce {
                            factory: f.id,
                            unit_type: faction.air_units[idx].clone(),
                            count: 1,
                        });
                    }
                }
                Role::NavalFactory => {
                    if !faction.naval_units.is_empty() {
                        let idx = (self.produced as usize) % faction.naval_units.len();
                        self.produced += 1;
                        out.push(Command::Produce {
                            factory: f.id,
                            unit_type: faction.naval_units[idx].clone(),
                            count: 1,
                        });
                    }
                }
                _ => {}
            }
        }

        // Fire loaded missile silos at enemy target.
        if let Some(t) = enemy_target {
            for silo in w
                .units()
                .filter(|u| u.army == self.army && u.role == Role::MissileSilo && u.silo_missiles > 0)
            {
                if silo.orders.is_empty() {
                    out.push(Command::LaunchMissile {
                        unit: silo.id,
                        x: t.0,
                        y: t.1,
                    });
                }
            }
        }

        // Army control (Land + Air + Naval + Experimentals).
        let mobiles: Vec<(UnitId, (i64, i64), bool)> = w
            .units()
            .filter(|u| {
                u.army == self.army
                    && (u.role == Role::Mobile || u.role == Role::Experimental)
                    && !u.weapons.is_empty()
            })
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

        let def_sq = (DEFEND_RADIUS as i128) * (DEFEND_RADIUS as i128);
        let intruder = w
            .units()
            .filter(|u| !w.are_allies(self.army, u.army) && !u.role.is_structure())
            .map(|u| (dist_sq((u.pos.x, u.pos.y), start), u.id, (u.pos.x, u.pos.y)))
            .filter(|(d, _, _)| *d <= def_sq)
            .min();
        if let Some((_, _, p)) = intruder {
            if !idle_home.is_empty() {
                out.push(Command::AttackMove {
                    units: idle_home,
                    x: p.0,
                    y: p.1,
                    queue: false,
                });
            }
        } else if idle_home.len() >= self.wave_size
            && let Some(t) = enemy_target
        {
            out.push(Command::AttackMove {
                units: idle_home,
                x: t.0,
                y: t.1,
                queue: false,
            });
            self.waves_sent += 1;
            self.wave_size = (self.wave_size + 4).min(40);
        }
        if !idle_away.is_empty()
            && let Some(t) = enemy_target
        {
            out.push(Command::AttackMove {
                units: idle_away,
                x: t.0,
                y: t.1,
                queue: false,
            });
        }
        out
    }

    fn plan_commander(
        &self,
        w: &Skirmish,
        faction: &Faction,
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
        let mex = count(Role::Extractor);
        let fac = count(Role::Factory);
        let air_fac = count(Role::AirFactory);
        let naval_fac = count(Role::NavalFactory);
        let pgens = count(Role::Generator);
        let hydros = count(Role::Hydro);
        let storages = count(Role::MassStorage);

        let has_cat = |id: &str| !id.is_empty() && w.catalog().get(id).is_some();
        let near_mex = |r: i64| w.nearest_free_mass_point(start, r * SCALE).map(|(_, p)| p);
        let near_hydro = |r: i64| w.nearest_free_hydro_point(start, r * SCALE).map(|(_, p)| p);
        let naval_grid = w.domain_grid(NavDomain::Naval);
        let has_naval_water =
            naval_grid.blocked_count() < (naval_grid.width as usize) * (naval_grid.height as usize);

        let pick = if mex < 2 && near_mex(120).is_some() {
            Some((faction.extractor.clone(), near_mex(120)))
        } else if fac < 1 {
            Some((faction.factory.clone(), None))
        } else if hydros == 0 && has_cat(&faction.hydro) && near_hydro(140).is_some() {
            Some((faction.hydro.clone(), near_hydro(140)))
        } else if pgens < fac + 1 && pgens < MAX_GENERATORS {
            Some((faction.generator.clone(), None))
        } else if mex < 4 + fac && near_mex(160).is_some() {
            Some((faction.extractor.clone(), near_mex(160)))
        } else if air_fac == 0 && fac >= 1 && has_cat(&faction.air_factory) {
            Some((faction.air_factory.clone(), None))
        } else if storages == 0 && mex >= 3 && has_cat(&faction.mass_storage) {
            let anchor = w
                .units()
                .find(|u| u.army == self.army && u.role == Role::Extractor)
                .map(|m| (m.pos.x, m.pos.y))
                .unwrap_or(start);
            Some((faction.mass_storage.clone(), Self::find_site(w, &faction.mass_storage, anchor)))
        } else if naval_fac == 0
            && fac >= 2
            && has_naval_water
            && has_cat(&faction.naval_factory)
            && Self::find_site(w, &faction.naval_factory, start).is_some()
        {
            Some((
                faction.naval_factory.clone(),
                Self::find_site(w, &faction.naval_factory, start),
            ))
        } else if fac < MAX_FACTORIES {
            Some((faction.factory.clone(), None))
        } else if pgens < MAX_GENERATORS {
            Some((faction.generator.clone(), None))
        } else {
            // Upgrade one T1 extractor when economy stabilizes, otherwise grab outer mexes.
            let any_mex_upgrading = w.units().any(|u| {
                u.army == self.army
                    && u.role == Role::Extractor
                    && matches!(u.orders.front(), Some(UnitOrder::Upgrade(_)))
            });
            if !any_mex_upgrading
                && has_cat(&faction.t2_mex)
                && let Some(t1_mex) = w
                    .units()
                    .find(|u| u.army == self.army && u.role == Role::Extractor && u.tier == 1 && u.orders.is_empty())
                && w.army(self.army).is_some_and(|a| a.mass >= 250 * SCALE)
            {
                out.push(Command::Upgrade {
                    unit: t1_mex.id,
                    target_type: faction.t2_mex.clone(),
                });
                return;
            }
            near_mex(260).map(|p| (faction.extractor.clone(), Some(p)))
        };
        let Some((unit_type, at)) = pick else {
            return;
        };
        let site = at.or_else(|| Self::find_site(w, &unit_type, start));
        if let Some((x, y)) = site {
            out.push(Command::Build {
                builder: acu.id,
                unit_type,
                x,
                y,
                queue: false,
            });
        }
    }

    fn plan_engineers(
        &self,
        w: &Skirmish,
        faction: &Faction,
        start: (i64, i64),
        enemy_start: Option<(i64, i64)>,
        out: &mut Vec<Command>,
    ) {
        let count = |role: Role| {
            w.units()
                .filter(|u| u.army == self.army && u.role == role)
                .count()
        };
        let n_pd = count(Role::PointDefense);
        let n_rad = count(Role::Radar);
        let n_sh = count(Role::Shield);
        let n_aa = count(Role::AntiAir);
        let n_tml = count(Role::MissileSilo);
        let n_exp = count(Role::Experimental);

        let mut pd_planned = 0usize;
        let mut rad_planned = 0usize;
        let mut sh_planned = 0usize;
        let mut aa_planned = 0usize;
        let mut tml_planned = 0usize;
        let mut exp_planned = 0usize;

        let has_cat = |id: &str| !id.is_empty() && w.catalog().get(id).is_some();
        let me_mass = w.army(self.army).map(|a| a.mass).unwrap_or(0);
        let me_energy = w.army(self.army).map(|a| a.energy).unwrap_or(0);

        let forward = if let Some(es) = enemy_start {
            let dx = es.0 - start.0;
            let dy = es.1 - start.1;
            let d = crate::skirmish::isqrt(
                (dx as i128) * (dx as i128) + (dy as i128) * (dy as i128),
            )
            .max(1);
            (
                start.0 + dx * 36 * SCALE / d,
                start.1 + dy * 36 * SCALE / d,
            )
        } else {
            start
        };

        for eng in w.units().filter(|u| {
            u.army == self.army
                && u.role == Role::Engineer
                && u.orders.is_empty()
                && u.build.is_none()
        }) {
            if n_rad + rad_planned == 0
                && let Some((rx, ry)) = Self::find_site(w, &faction.radar, start)
            {
                rad_planned += 1;
                out.push(Command::Build {
                    builder: eng.id,
                    unit_type: faction.radar.clone(),
                    x: rx,
                    y: ry,
                    queue: false,
                });
                continue;
            }
            if n_pd + pd_planned < MAX_POINT_DEFENSE
                && let Some((px, py)) = Self::find_site(w, &faction.point_defense, forward)
            {
                pd_planned += 1;
                out.push(Command::Build {
                    builder: eng.id,
                    unit_type: faction.point_defense.clone(),
                    x: px,
                    y: py,
                    queue: false,
                });
                continue;
            }
            if n_sh + sh_planned == 0
                && has_cat(&faction.shield)
                && count(Role::Extractor) >= 2
                && let Some((sx, sy)) = Self::find_site(w, &faction.shield, start)
            {
                sh_planned += 1;
                out.push(Command::Build {
                    builder: eng.id,
                    unit_type: faction.shield.clone(),
                    x: sx,
                    y: sy,
                    queue: false,
                });
                continue;
            }
            if n_aa + aa_planned < 2
                && has_cat(&faction.anti_air)
                && count(Role::Extractor) >= 2
                && let Some((ax, ay)) = Self::find_site(w, &faction.anti_air, start)
            {
                aa_planned += 1;
                out.push(Command::Build {
                    builder: eng.id,
                    unit_type: faction.anti_air.clone(),
                    x: ax,
                    y: ay,
                    queue: false,
                });
                continue;
            }
            if n_tml + tml_planned == 0
                && has_cat(&faction.tac_missile)
                && me_mass >= 350 * SCALE
                && let Some((tx, ty)) = Self::find_site(w, &faction.tac_missile, start)
            {
                tml_planned += 1;
                out.push(Command::Build {
                    builder: eng.id,
                    unit_type: faction.tac_missile.clone(),
                    x: tx,
                    y: ty,
                    queue: false,
                });
                continue;
            }
            if n_exp + exp_planned == 0
                && has_cat(&faction.experimental)
                && me_mass >= 900 * SCALE
                && me_energy >= 2500 * SCALE
                && let Some((ex, ey)) = Self::find_site(w, &faction.experimental, start)
            {
                exp_planned += 1;
                out.push(Command::Build {
                    builder: eng.id,
                    unit_type: faction.experimental.clone(),
                    x: ex,
                    y: ey,
                    queue: false,
                });
                continue;
            }
            if let Some((_, wp)) = w.wreck_at((eng.pos.x, eng.pos.y), 90 * SCALE) {
                out.push(Command::Reclaim {
                    units: vec![eng.id],
                    x: wp.0,
                    y: wp.1,
                    queue: false,
                });
                continue;
            }
            if let Some((_, mp)) =
                w.nearest_free_mass_point((eng.pos.x, eng.pos.y), 240 * SCALE)
            {
                out.push(Command::Build {
                    builder: eng.id,
                    unit_type: faction.extractor.clone(),
                    x: mp.0,
                    y: mp.1,
                    queue: false,
                });
            }
        }
    }

    fn find_site(w: &Skirmish, unit_type: &str, start: (i64, i64)) -> Option<(i64, i64)> {
        let role = w.catalog().get(unit_type).map(|t| t.role);
        let is_naval = role == Some(Role::NavalFactory);
        let spacing = if is_naval {
            8 * SCALE
        } else if role.is_some_and(|r| r.is_factory() || r == Role::Experimental) {
            12 * SCALE
        } else {
            6 * SCALE
        };
        let max_ring: i64 = if is_naval { 32 } else { 14 };
        for ring in 1..=max_ring {
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
    fn legacy_ai_vs_ai_is_deterministic_and_fights() {
        let play = || {
            let mut w = test_world(11);
            let mut ais = [LegacyAi::new(0), LegacyAi::new(1)];
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
        assert!(a.kills() > 0);
    }
}
