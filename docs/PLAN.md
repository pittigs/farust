# FARust Plan

Goal: Rust engine that loads original Forged Alliance data from the user's own
install and runs a deterministic, multicore-capable RTS. No game assets in git.

Game dir (default, override with `FA_PATH`):
`/home/bas_pit/.local/share/Steam/steamapps/common/Supreme Commander Forged Alliance`

## Principles

1. Data first: parse original formats before writing gameplay.
2. Deterministic sim from day one: fixed tick, seeded RNG, no float-order
   dependence in sim state. Rendering/audio never feed back into sim.
3. Multicore where it is safe: rendering, asset loading, pathfinding jobs,
   audio. Parallel sim is last and only behind determinism tests.
4. Read-only against the game install. Unpacking, if needed, goes to
   `fa-unpack/` (git-ignored), never into `gamedata/`.

## Phases

### Phase 0 — Inspector (done)
- `.scd` (= ZIP) listing and `cat`. Commands: `inventory`, `list`, `cat`.
- Exit: lists all 17 archives, reads `lua.scd` (395 entries). Done.

### Phase 1 — Blueprints (done)
- Parse `*_unit.bp` (`UnitBlueprint { ... }`, pure data, no functions) into a
  typed value tree, then a summary: id, description, categories, health,
  build costs/time, speed/motion, weapons.
- Commands: `farust units [--limit N] [--search TEXT]`, `farust unit <ID>`.
- Exit: all 568 unit blueprints in `units.scd` parse without error; spot
  values match hand checks (e.g. UAL0001 ACU: MaxHealth 11000). Done.

### Phase 2 — Other data formats (done)
- Maps: `.scmap` header/heightmap, `_scenario.lua`/`_save.lua` start spots and resource markers.
- Meshes: `.scm` header, bones, vertex/index counts, bounds.
- Textures: `.dds` header (size, format, mip count, cubemap/volume flags).
- Exit: `farust maps --check`, `farust map <NAME>`, `farust meshes --check`, `farust mesh <PATH>`, `farust textures --check`, `farust dds <PATH>` pass on all 61 maps, 1,252 meshes, and 11,506 textures. Done.

### Phase 3 — Viewer (done)
- Window + GPU (`wgpu` + `winit`), pure-Rust BC1/BC2/BC3/uncompressed DDS RGBA8
  decoder with faction team-color + specular compositing, orbit/pan/strategic-zoom
  camera, and heightmap terrain + water + spawn/mass beacons.
- Commands: `farust view-unit <ID> [--snapshot OUT.png]`,
  `farust view-map <NAME> [--snapshot OUT.png]`.
- Exit: `UAL0001` renders textured; `SCMP_001` terrain renders at strategic zoom. Done.

### Phase 4 — Deterministic sim core (done)
- Fixed-tick world (`src/sim.rs`, TICK_HZ=10, SCALE=1000 int-only): spawn/move/
  order queue, seeded RNG, FNV-1a64 state hash. Economy (`src/economy.rs`):
  affordable-fraction ticks, stall-without-debt. Combat (`src/combat.rs`):
  ceil cooldowns, dist-sq range, hitscan kill/clamp.
- Headless: `farust sim-run --ticks N [--seed S] [--json]`. Game-data-
  independent synthetic selftest, BTreeMap/id-ordered ticks.
- Exit: 10,000-tick hash `35db8f2bd7a6faf2` stable across runs; alive 1,
  shots 11, kills 1. `cargo test`: 71 passed. Done.

### Phase 5 — Multicore (done)
- Scoped order-preserving job pool (`src/jobs.rs`, `std::thread::scope`):
  parallel asset batch decode (`.bp` + `.scm` + `.dds` RGBA8), integer-only A*
  pathfinding batch solver (`src/pathfinding.rs`), and parallel render-frame
  preparation (`src/render_prep.rs`, frustum culling + LOD + instance transforms).
- Command: `farust bench-multicore [GAME_DIR] [--threads N] [--limit N] [--json]`.
- Exit: load time, pathfinding batch time, and frame prep time measured 1-thread
  vs N-threads with identical output digests; 10,000-tick sim hash unchanged
  (`35db8f2bd7a6faf2`). Done.

### Phase 6 — Playable slice (done)
- Multi-army deterministic skirmish (`src/skirmish.rs`) + scripted opponent
  (`src/skirmish_ai.rs`) + raycast picking, `.scm` unit cache, and interactive/
  snapshot GPU skirmish viewer (`src/play.rs`).
- Selection & box-selection (`LMB`), move/attack/rally orders (`RMB`),
  commander construction (`B`/`M`/`P` with mass-point snapping), factory
  production (`T`/`Y`), A* routing, health/build bars, and assassination victory.
  Lua unit scripts are *not* reimplemented; behaviour is re-created in Rust
  from blueprint data.
- Commands: `farust skirmish [MAP]`, `farust play [MAP] [--snapshot OUT.png]`.
- Exit: start to finish on `SCMP_001` with built units fighting, deterministic
  replay hash, and GPU battle snapshot rendered. Done.

### Phase 7 — Textured instancing, terrain splatting, water, skybox, props & HUD (done)
- `.scmap` v2.60 full lighting, water parameters, sky cubemap, normal-map &
  prop placement parsing (`src/scmap.rs`) + parallel asset loaders & 64×64 FA
  unit icon atlas (`src/scene.rs`, `src/dds_decode.rs`).
- GPU multi-stratum terrain splatting (10 albedo + 9 tangent-normal `2d_array`),
  skybox cubemap, transparent depth-faded rippled water with sky reflections,
  `2048×2048` PCF shadow map, alpha-cutout instanced `.scmap` trees/rocks/logs,
  `_LOD0` units + charred `.scm` wrecks, clickable minimap, and real FA icon HUD
  (`src/viewer.rs`, `src/hud.rs`, `src/play.rs`).

### Phase 8 — Deeper FA gameplay mechanics (done)
- T1 Engineers (`build`, `repair`, `assist`, `reclaim` wrecks), Point Defense,
  Radar (`Vision`/`RadarBlip`/`Hidden` intel & `F2` Fog-of-War toggle), ballistic
  AoE artillery shells, ACU Overcharge, and deterministic unit separation (`src/skirmish.rs`,
  `src/skirmish_ai.rs`).

### Phase 9 — Binary replays & TCP lockstep multiplayer (done)
- Compact binary `.farust-replay` (`FARPL1`) codec covering all `Command` variants
  with periodic checkpoint hashes (`src/replay.rs`: `skirmish --record`, `replay verify`,
  `replay play`).
- Length-prefixed `TcpStream` 2-player lockstep (`src/net.rs`: `host`, `join`) with
  canonical `ArmyId` command ordering and per-turn hash desync verification.

### Phase 10 — Persistent `WorkerPool`, `SpatialGrid` & 2,000+ unit scaling (done)
- Zero-`unsafe` persistent `WorkerPool` (`src/jobs.rs`) driving batched A* routing and
  parallel `step_separation` in `Skirmish` with exact `1t == Nt` parity and parallel
  `render_prep` (`src/pathfinding.rs`, `src/render_prep.rs`), plus tight 6m
  zero-alloc `SpatialGrid` (`src/spatial.rs`) and dense `UnitSpatialSnap`.
- Verified 2,050-unit active combat benchmark via `farust bench-scale SCMP_001 --units 2048 --ticks 200`:
  210 ticks/s (195→210 ticks/s), 6,654 paths, 14,094 shots, 541 kills, exact `1t == 20t`
  hash `f1eebe99299d22c7`. Done.

### Phase 11 — Living battlefield (done)
- `.sca` v5 parser + SLERP sampler (`src/sca.rs`); per-type rigs drive GPU skinning
  (walk while moving at `MaxSpeed / AnimationWalkRate`, idle/rest otherwise) with
  turret yaw tracking composed on top (`src/render_prep.rs` `LivingBattlefield`).
- Army-tinted FA strategic icons at strategic zoom, LOD0→LOD1 by blueprint
  `LODCutoff`, 128×128 FoW mask in the terrain shader, VFX (`src/vfx.rs`) derived
  from frame-to-frame sim diffs. Render-only: sim hash unchanged.
- `farust view-unit <ID> --anim walk --anim-time 0.4 --snapshot OUT.png`.

### Phase 12 — Combined arms (done)
- Land/naval/amphibious/air domains (`src/flowfield.rs`), T1–T3 upgrades, hydro,
  adjacency, shields, tactical/strategic missiles & silos, veterancy, experimentals,
  AI tech-up (`src/skirmish.rs`, `src/skirmish_ai.rs`).

### Phase 13 — RTS controls & UI (done)
- `src/controls.rs`: control groups (`Ctrl+1..9` / `1..9`), double-click/Ctrl+click
  same-type select, idle-engineer cycle (`.`), weapon/vision/radar/shield range
  rings, RMB-drag formation lines, LMB-drag structure rows, Ctrl+RMB area
  reclaim/repair, unit stats card. Quick-selects moved to `F`/`H`/`J`.
- Replay timeline: seek (backward = deterministic re-sim from tick 0), 1x–16x,
  `[`/`]` ±30s, observer/army FoW perspective (`V`).

### Phase 14 — Audio (done)
- Clean-room XACT `.xwb`/`.xsb` parser (`src/xact.rs`): 3,043/3,043 waves decode
  (all PCM16 in this install; MS-ADPCM tested synthetically; xWMA/XMA unsupported).
- `src/audio_play.rs`: sim-event → cue mapping (per-unit weapon cues, impacts,
  construction, deaths, selection voices, music playlists Base_Building/Battle),
  screen-pan + distance attenuation, 40-voice limit, cooldowns, limiter; output via
  `pw-cat`/`pacat`/`aplay`, silent fallback. `F3` mute, `F4`/`F5` volume, `--mute`.
- `farust audio <list|coverage|cues|play-cue|render-mix>`; audio never touches the sim.

### Phase 15 — Multiplayer & lobby (done)
- Protocol v2 star lockstep for 2–8 players (`src/net.rs`): canonical command
  order, 2-turn input delay, per-turn hash verification with desync reports;
  extra `--armies` become deterministic AI on every peer.
- `farust host ADDR [MAP] --players N [--play]`, `farust join ADDR [--play]`.
- In-window lobby (`src/lobby.rs`, `farust play --lobby`): 54 skirmish maps with
  previews, slot/faction/team/seed setup, launches on the same event loop.
- Exit: 3-peer headless game ends on identical hash `e9f83360aa6bc86d`.

### Phase 16 — Massive-battle scale (done)
- Shared group flow fields with bounded LRU cache, parallel deterministic target
  acquisition via `WorkerPool`, prop chunk culling and LOD1 switching.

### Phase 17 — Visual fidelity II (done)
- Turret pitch + barrel recoil (`AimPose` in `src/sca.rs`), death/open/build/upgrade clips with crossfade blending, Fresnel shield domes + radial hit ripples (`src/viewer.rs`), `.scd`/blueprint projectile meshes (`projectiles.scd`) + smoke trails (`src/vfx.rs`), aircraft banking & contrails, ship wakes & sinking naval wrecks, and construction build-up (`src/render_prep.rs`, `src/play.rs`). Render-only: sim hash unchanged.

### Phase 18 — Gameplay fidelity II (done)
- `GameOptions` (victory conditions: `assassination`/`supremacy`/`annihilation`/`sandbox`; configurable `--unit-cap`) persisted in v3 replay headers (`src/replay.rs`); allied shared vision & resource overflow; stealth fields, personal stealth, cloak, radar jamming & omni intel; air transports (`Command::Load`/`Unload`, hotkey `L` + RMB board); ACU/SCU enhancements (`Command::Enhance`, hotkey `K`/`Shift+K`); factory repeat queue (`Command::SetRepeat`, hotkey `Z`/`Ctrl+R`); shield toggle (`X`); and extended stats card info (`src/skirmish.rs`, `src/controls.rs`, `src/hud.rs`).

### Phase 19 — Skirmish AI II (done)
- Difficulty levels (`easy | normal | hard | adaptive | legacy`), faction opening build orders (`src/ai_build.rs`), decaying `ThreatMap` (`src/ai_threat.rs`), scouting, raids, counter-composition, air/naval factories, allied coordination, Overcharge/retreat micro, original baseline preserved (`src/ai_legacy.rs`), `--ai <diff>` flag, and `farust ai-arena` parallel AI-vs-AI tournament runner (`src/skirmish_ai.rs`, `src/main.rs`).

### Phase 20 — Front end (done)
- Graphical front-end shell on a single `winit` event loop hosting `PlayApp` embedded (`src/menu.rs`: Skirmish / Multiplayer / Replays / Options / Quit; bare `farust` or `farust menu [--screen ...] [--snapshot ...]`), TrueType glyph atlas (`src/ttf.rs`) + UI toolkit (`src/ui_kit.rs`), `$XDG_CONFIG_HOME/farust/settings.ini` persistence (`src/settings.rs`), replay browser + verifier, and TCP v3 multiplayer lobby (`src/net_lobby.rs`, `src/net.rs`) with peer list, slot editing, ready checks, lobby chat, and lockstep handover.

### Phase 21 — Match session (done)
- Pause modal menu (`Esc`), `.farust-save` (`FARSAV1`) binary save/load with time-sliced deterministic re-sim & shadow AI resume (`src/savegame.rs`, `F6` quicksave, `F9` quickload, `farust play --load FILE`), automatic match replay recording (`--no-record` disables), end-of-game score screen & time-series line graphs (`src/stats.rs`, `Tab`, `src/hud.rs`), synchronized network pause/speed (`NetControl`), in-game `[All]`/`[Allies]` chat (`src/chat.rs`, `Enter`/`Shift+Enter`), automatic AI takeover on peer disconnect, and post-desync state dumps (`src/net.rs`).

### Phase 22 — Per-shadow-quality shadow maps (done)
- `settings::shadow_quality()` hook (`Medium` = stock 2048px when no settings are
  active); `GpuSceneResources::new` sizes the shadow target from it (1024 / 2048 /
  4096, 1px placeholder + skipped shadow pass for `Off`).
- PCF texel + enabled flag travel in `Uniforms::shadow_params`; both shadow
  shaders (terrain, instanced) use the live texel size and early-out when off.
- `set_shadow_quality()` rebuilds the target live; called on window-config
  resize in `play.rs` and `viewer.rs`. Render-only: sim hash unchanged.
- Exit: `cargo test` 213 passed, `UAL0001` + `SCMP_001` snapshots render,
  `sim-run --ticks 10000` hash `35db8f2bd7a6faf2` unchanged. Done.

### Phase 23 — FA command interface (done)
- Blueprint/roster-driven build & command card (`src/command_card.rs`, `src/hud.rs`:
  tech tabs T1–T4, category chips, Quantum Gate SACU filtering & mod-unit text
  fallback when icons are absent, factory queue/cancel + upgrade panel,
  ACU/SCU enhancement panel, order buttons with targeting cursors), remappable
  key bindings in Options (`src/keymap.rs`), camera bookmarks, ally map pings.

### Phase 24 — Content & mods (done)
- Full data-driven unit roster across all factions (`src/roster.rs`: 568
  blueprints → 356 playable, `farust roster [--verify MAP]` builds 352/352
  verified units on `SCMP_009`), user-installed maps (`src/scenario.rs`:
  stock + FA user folders, loose + `.scd`/`.zip` with extraction cache),
  blueprint mods directory (`src/mods.rs`: `$XDG_DATA_HOME/farust/mods`,
  `--mod`/`FARUST_MODS`, deep-merge/replace/add, identities checked in
  replay v4 / save / lockstep v4 headers), layered XACT cues + looping
  unit sounds (`src/xact.rs` tracks/layers, `src/audio_play.rs`
  construction/movement `*_Loop` bookkeeping). Sim hashes unchanged.

### Phase 25 — Performance & robustness (done)
- Remappable frame profiler overlay (`src/profiler.rs`, `F12` via `keymap.rs`
  since `F10` is camera bookmark 3), `farust bench-render` GPU/CPU frame timing
  at 4k–8k units, and `farust fuzz-parsers` never-panic parser hardening (`src/fuzz.rs`).
- Parallel `.scd` unit mesh/texture loading (`UnitMeshCache::load`: 6,376 ms →
  453 ms), zero-`BTreeMap` `U32Map`/indexed batch prep at 8,192 units (`cpu prep`
  p50 6.24 ms → 3.52 ms; `frame` p50 9.18 ms → 6.05 ms), and shadow-pass prop
  chunk frustum culling (584 → 252 draws, shadow pass 0.496 ms → 0.272 ms).
- Sim determinism hashes unchanged (`35db8f2bd7a6faf2` / `192e89b083b978cb`);
  252/252 tests pass (including 30/30 on `net_lobby` loopback test). Done.

### Phase 26 — Online play with friends (done)
- **Invite codes & zero-dependency TCP rendezvous + relay** (`src/rendezvous.rs`):
  unambiguous 8-character codes (`XXXX-XXXX`), `RendezvousServer` (`farust rendezvous [BIND_ADDR]`),
  host control channel expiring codes automatically on match start or close, and
  client resolution trying direct addresses first before falling back cleanly to a
  server-spliced TCP reverse tunnel when behind NAT.
- **Lockstep & lobby integration** (`src/net_lobby.rs`, `src/net.rs`, `src/menu.rs`, `src/main.rs`):
  `farust host --code` / `farust join --code` / `--relay` / `+connect_lobby <CODE>`,
  Multiplayer Hub (`HOST WITH INVITE CODE` toggle, rendezvous server field, `HOST ADDRESS OR INVITE CODE`),
  and Multiplayer Lobby `[COPY CODE]` header + `[R]`/`[D]` relay status indicators.

### Phase 27 — Old-school FA layout & visual theme (done)
- **Persistent theme toggle** (`src/settings.rs`, `src/menu.rs`): `ui_theme=modern|classic`
  (plus `FARUST_UI_THEME` env override and Options `UI Theme` selector; defaults to `modern`).
- **Authentic Classic FA HUD & Faction Skins** (`src/ui_kit.rs`, `src/hud.rs`):
  4 faction palettes (`UEF` steel-blue, `Aeon` emerald/gold, `Cybran` crimson/charcoal, `Seraphim`
  obsidian/amber), classic FA `.scd` menu backdrops (`background-colossus_bmp.dds` /
  `background-paint01_bmp.dds`), top-left split Mass/Energy console + open top-center sky +
  top-right tactical clock pod, bottom-left minimap + `[Orders | Command Card | Queue | Enhancements]`
  bottom console, and bottom-right unit portrait/vitals panel.

## Roadmap — Phases 28–57 (planned)

Ordered so each arc builds on the previous one. Every sim-touching phase
bumps `SIM_VERSION`, re-baselines golden hashes deliberately, and keeps
`1t == Nt` parity; render/audio phases must leave sim hashes unchanged.

### Arc A — Close the simulation gaps (28–35)

### Phase 28 — Walls & structure footprints (done)
- `Role::Wall` (`src/skirmish.rs`, `src/roster.rs`): blueprint `WALL` structures are
  playable for all four factions (roster excluded 18 → 14; `roster --verify SCMP_009`
  builds 356/356). Walls sit flush against each other and other structures.
- Wall overlay: live wall footprints are stamped onto terrain-only base grids
  (`sync_walls`, land + amphibious; naval and air unaffected). On change, flow fields
  and cached paths are dropped so movers re-route, ground units on a new wall cell are
  pushed to the nearest open cell, separation never shoves units into walls, and goals
  inside a wall are approached from the nearest open cell. Dead walls re-open the cells.
- Walls are last-resort auto-targets and do not keep an army alive under
  supremacy/annihilation (FA's `ALLUNITS - WALL`).
- LMB-drag wall lines (`controls::drag_wall_sites`): snapped to the 1 m grid, gap-free
  8-connected chains, up to 96 sections per drag.
- Lockstep protocol v5 (older peers would reject wall `Build` commands).
- Exit: walled-corridor detour, mid-route re-route for flow-field groups, eviction,
  target priority, defeat and placement tests pass (263/263). Games without walls are
  bit-identical: `sim-run` `35db8f2bd7a6faf2`, `skirmish SCMP_001` `192e89b083b978cb`,
  `bench-scale --units 2048` `69975978586139a0` (`1t == 20t`). Done.
- Deferred: reclaiming live walls (moves to Phase 34 with prop reclaim) and
  structures other than walls blocking pathing (moves to Phase 29).

### Phase 29 — Ballistics & line of fire (done)
- Integer terrain heightfield in the sim (`src/terrain.rs`, quantized once from the
  `.scmap` heightmap; bilinear, `segment_hit` line-of-fire sampling) attached by the new
  `Skirmish::attach_map` at every map-based world construction.
- Surface direct fire is a 3-D shell (`MuzzleVelocity`) aimed at the target's centre and
  led by its last-tick velocity (`LeadTarget`); per tick it hits the first thing on its
  path — terrain, or any non-allied surface unit/wall it passes through — or misses at its
  aim point. Instant/beam weapons (`BeamLifetime`, `ContinuousBeam`, no muzzle velocity)
  are terrain-checked; anti-air stays hitscan, artillery keeps its ballistic AoE arc.
- Turrets slew at `TurretYawSpeed` and fire only within `FiringTolerance`; yaw uses a new
  integer CORDIC `combat::atan2_mrad` (no floats in sim state). Real data: 189 projectile
  vs 27 beam/instant surface weapons, 156 turreted.
- Every structure footprint blocks pathing (Phase 28 overlay; cells whose centre lies in
  the footprint, walls keep full-cover), naval grid included; only paths near newly
  blocked cells are re-routed; factory-built units roll off onto open ground.
- `RULES_VERSION` 2 recorded in replay v5 headers; replays/saves from other rules are
  refused with a clear error. Lockstep protocol v6.
- Exit: ridge soaks shells and beams, turret slew delay, wall soaks shots aimed past it,
  leading beats no-leading, structure detour + roll-off and replay-rules tests pass
  (271/271). `bench-scale --units 2048`: 181 ticks/s 1t, 346 ticks/s 20t, exact `1t == 20t`
  (`3325c3259e0e6e74`); 46% of shells miss (half overkill on dead targets, half dodged),
  4% hit terrain. `skirmish SCMP_009 --record` → `replay verify` OK; 12-game `ai-arena`
  all decisive. Done.
- Deferred: turret yaw *range* limits and hull-facing fixed weapons (needs hull facing,
  Phase 35); render still draws direct shells at ground + 1.2 m, not their sim height.

### Phase 30 — Air fuel, staging & repair (done)
- Aircraft burn `Physics.FuelUseTime` while airborne and fly at half speed when empty.
- `Role::AirStaging` (all four platforms playable; roster excluded 14 → 10; `roster
  --verify SCMP_009` 360/360): `Transport.DockingSlots`, refuel at `FuelRechargeRate ×
  AI.RefuelingMultiplier`, repair at `Transport.RepairRate` paid per second
  (`AI.RepairConsume*`), slot limits, aircraft released when the platform dies, undock
  and resume queued orders when full (`step_staging`, docked aircraft reuse `carried_by`).
- Orders: `Command::Dock` (RMB an own platform with aircraft selected), `UnitOrder::Dock`,
  `Command::SetAutoRefuel`; auto-refuel (default on) queues a dock at the nearest platform
  with a free slot within `AI.StagingPlatformScanRadius` below 20% fuel. Replay tags 21/22,
  fuel hashed, rules v3.
- AI: wants a staging platform once it fields ≥ 4 aircraft; AI build jobs now respect the
  builder's tech tier for roster-derived structures (previously T1 engineers retried
  rejected T2 builds forever). Fixed the AI role-count array (`ROLES` 18 → 20), which
  would have panicked on an AI-owned wall (Phase 28) or platform.
- Exit: fuel burn/half speed, auto-dock → refuel → repair (energy cost) → resume, slot
  limit, enemy-dock refusal and platform-death release tests pass (283/283);
  `bench-scale --units 2048` 180 / 330 ticks/s, exact `1t == 20t`. Done.
- Limitation: today's AI rarely has a T2 engineer before ~40 min, so it seldom builds
  staging in practice (tech pacing is Phase 52); staging HUD (fuel bar, auto-refuel
  toggle button) is not on the command card yet.

### Phase 31 — Engineering stations & drones (done)
- `Role::EngStation` (UEF Kennels `XEB0104`/`XEB0204`, Cybran Hives `XRB0104`→`0204`→`0304`;
  roster excluded 10 → 5, `roster --verify SCMP_009` 365/365): stationary, never-moving
  workers that every second pick the nearest allied construction (builder site, factory,
  in-place upgrade) in their radius (`MaxBuildDistance` / `AI.GuardScanRadius`), else the
  nearest damaged surface ally, and assist / repair it with their `BuildRate`.
- Pod drones: stations with `Economy.EngineeringPods` spawn their `PodUnitID` drones
  (loaded into the catalog even though nothing builds them); pods fly out to the work,
  return home when idle, die with their station, are rebuilt at the station's rate
  (`BuildTime / BuildRate`, paid in full) and upgrades grow the pod count. Pods burn no
  fuel and are invisible to the AIs and the idle-engineer cycle. Rules v4.
- Exit: Hive speeds up a factory by > 20 % and repairs only inside its radius; Kennel pod
  flies out, repairs, returns, is rebuilt (paid), upgrade → two pods, pods die with the
  station (308/308). `bench-scale --units 2048` 176 / 318 ticks/s, exact `1t == 20t`. Done.
- Not modelled: ACU/SCU shoulder-pod enhancements (no `PodUnitID` in their blueprint data),
  station reclaim/capture, AI building stations.

### Phase 32 — Mobile factories, carriers & eggs (done)
- Roster: mobile `FACTORY` units (Fatboy, Atlantis, CZAR, Tempest, T3 carriers) get
  `Produce` options for any domain; mobile `CONSTRUCTION` non-engineers (Megalith) get
  `Build` options. The sim derives `mobile_factory` / `mobile_builder` from those options
  (`Skirmish::is_producer` / `is_constructor`), so `Produce`/`Rally`/`SetRepeat`/
  `CancelProduce`/`Build` and the HUD accept them; units roll off behind the moving factory.
- Crab eggs: structures with `Economy.BuildUnit` hatch into it the moment they finish
  (hatchlings loaded into the catalog). Roster excluded 5 → 1 (only the Novax satellite);
  `roster --verify SCMP_009` 369/369 (hatchlings count as completions).
- `AIRSTAGINGPLATFORM` mobiles (Fatboy, Atlantis, CZAR) are air staging platforms; docked
  aircraft ride along. Rules v5.
- Exit: produce-while-moving + mobile docking and egg-laying → hatch tests pass (310/310);
  `bench-scale --units 2048` 176 / 315 ticks/s, exact `1t == 20t`. Done.
- Not modelled: carrier aircraft storage (`StorageSlots`) and launch, AI use of mobile
  factories / Megalith eggs.

### Phase 33 — Experimental & special-ability completeness (done)
- Novax: `Role::Orbital` launch centres (`ORBITALSYSTEM`) put their faction's `SATELLITE`
  blueprint (`XEA0002`) into orbit on completion; `UNTARGETABLE` units are never acquired
  or attackable; the satellite falls with its centre. Roster excluded 1 → 0: all 374 units
  of the FA skirmish tree are playable (`roster --verify SCMP_009` 370/370).
- EMP stun: weapon `Buffs` STUN entries, unit `Buffs.Stun` (on EMP impact / on death) and
  silo missiles (Cybran nukes) stun enemy surface mobiles — no moving, firing or building.
- Death weapons (`DeathWeapon`/`DeathImpact`/`Death`/`FireOnDeath`, incl. ACU and Paragon
  death nukes) damage every surface unit in radius, friend or foe; chain deaths resolve the
  next tick. Kamikaze weapons (Fire Beetle) detonate once and kill their carrier (they
  previously re-fired a 4,500-damage shot every reload).
- `farust roster --abilities [--missing]`: every `Display.Abilities` entry of the playable
  roster with unit count, status and data coverage: 26 modelled (data 39/39 death, 5/5
  stun, 2/2 kamikaze, 8/8 factory, 6/6 staging), 7 partial (sonar, torpedo, depth charge,
  submersible, carrier storage, massive, missile deflection), 3 not modelled (anti-torpedo,
  deploy, sacrifice), 1 n/a (not capturable). Rules v6.
- Exit: EMP stun, friend-and-foe death blast, single kamikaze detonation and untargetable
  satellite tests pass (313/313); `bench-scale --units 2048` 173 / 317 ticks/s, exact
  `1t == 20t`; 12-game `ai-arena` all decisive. Done — the "no unmodelled flags" goal is
  not met: the 3 not-modelled and 7 partial abilities above remain.

### Phase 34 — Map prop reclaim & live-unit reclaim (done)
- Reclaimable props: `MapContext::load_props` reads every `.scmap` prop's blueprint from
  `env.scd` (`Economy.ReclaimMassMax` / `ReclaimEnergyMax`, cached per blueprint) — e.g.
  `SCMP_009` 5,182 props / 32,597 mass / 283,280 energy, `SCMP_001` 10,939 / 53,532 /
  293,932 (`farust map NAME` prints the totals). Props are sim state (`Skirmish::props`,
  hashed); reclaim orders take a wreck (16 m snap), else a prop (3 m), else a live unit.
- Reclaim drains props at the wreck rate (energy ×10) and removes them; live units (walls,
  structures, land units; never commanders or aircraft) drain back into their mass cost
  and vanish without wreck or kill credit, re-opening the path grid. Area reclaim
  (Ctrl+RMB) and the AI's reclaim job include props. Rules v7.
- Render: reclaimed props disappear (instances filtered by their `.scmap` position bits,
  re-uploaded at most once a second; a backward replay seek restores them).
- Exit: exact tree/rock yields vs. a control run + determinism, wall reclaim refund and
  render-filter tests pass (316/316); an AI game on `SCMP_009` (which now harvests props)
  records and `replay verify`s. Done.
- Not modelled: falling / burning tree animation, prop destruction by weapons, reclaim
  time from `ReclaimTime` (props use the wreck rate).

### Phase 35 — Fire states & unit behaviours (done)
- `FireState` (fire at will / return fire / hold fire, `Command::SetFireState`): return
  fire only engages the unit's last attacker; hold fire never auto-engages; units not
  firing at will no longer stop to fight on attack-move / patrol.
- Ground fire (`Command::AttackGround` → `UnitOrder::AttackGround`): ready weapons in range
  fire at the map position — artillery arcs, direct shells into the ground (hitting
  whatever is in the way), until another order; works under hold fire.
- Guard (`Command::Guard` → `UnitOrder::Guard`): follow a friendly unit and engage threats
  around it; the command card's Guard on a mobile now issues it (was a plain move).
- Build pause (`Command::SetBuildPaused`): builders / factories stop progressing and
  spending until resumed.
- Command card: GROUND (targeted), FIRE (cycles, lit unless firing at will), HOLD BLD
  (toggle) with EN/DE tooltips. Replay tags 23–26, rules v8.
- Exit: every new command round-trips in the replay codec test; fire-state, ground-fire,
  guard and pause tests pass (319/319); `skirmish SCMP_009 --record` → `replay verify` OK.
  `bench-scale --units 2048` 170 / 305 ticks/s (gradual drift from 181 / 346 since Phase 29
  as each sim phase adds per-tick work — target of Phase 54). Done.
- Not modelled: blueprint `TargetPriorities`, air dive/hover, stop-and-shoot.

### Arc B — Controls & UX (36–39)

### Phase 36 — Build templates, hotbuild, waypoint editing & ferries (done)
- Build templates (`src/templates.rs`): `Alt+S` saves the selected own structures as a
  layout (blueprint + offset from the lowest-id anchor) to
  `$XDG_CONFIG_HOME/farust/templates.txt`; `Alt+T` enters / cycles placement; LMB queues
  every free site as ordinary `Command::Build`s on the best builder, so a placed template
  replays exactly. Malformed lines and non-id blueprints are rejected on load.
- Hotbuild: `Alt+F` / `Alt+D` / `Alt+E` / `Alt+R` cycle the selected builder's factory /
  defense / economy / intel roster options into build mode (rebindable in Options).
- Waypoint editing: an LMB drag that starts on a queued Move / AttackMove / Patrol point of
  the selection moves it (`Command::MoveWaypoint`) instead of box-selecting.
- Ferries: FERRY order (pick-up click, drop-off click) → `UnitOrder::Ferry` loops: load idle
  own land units within 12 m of the pick-up point, fly, unload in formation, return.
  Replay tags 27/28, rules v9. EN/DE tooltips.
- Exit: template layout → identical queued builds, waypoint drag, ferry delivery and the
  replay codec (every command) tests pass (322/322); keymap defaults stay conflict-free;
  `skirmish SCMP_009 --record` → `replay verify` OK. Done.
- Not done: ghost preview of a template before placing, cross-faction template mapping.

### Phase 37 — Localization & tooltips (done)
- Never-panic FA `loc` parser (`src/loc.rs`, fuzzed via `Kind::Loc` in `src/fuzz.rs`):
  loads `gamedata/loc_<LANG>.scd` (`strings_db.lua`/`usdb.lua`), loose `loc/<LANG>/`
  and `$XDG_DATA_HOME/farust/loc/<LANG>/`, plus `lua.scd` `unitdescription.lua`
  (448 entries) & `tooltips.lua` (463 entries) and blueprint enhancement metadata
  (150 entries), with fallback chain active lang → built-in UI/DE fallback → US →
  inline `<LOC key>Fallback`. CLI: `farust loc --check`, `farust loc <KEY> [--lang]`,
  `farust loc --unit <ID> [--lang]`.
- `language=<code>` in `settings.ini` (`src/settings.rs`), `FARUST_LANG` env override,
  and live-previewing `Language` / `Sprache` selector in Options (`src/menu.rs`).
- Latin-1 Supplement (`0xA0..=0xFF`, incl. German `äöüÄÖÜß`), typographic punctuation,
  and Cyrillic (`0x0401`, `0x0410..=0x0451`) glyphs baked into the TrueType atlas
  (`src/ttf.rs`) and rendered via a dedicated `hud_ttf` pass (`src/viewer.rs`,
  `src/ui_kit.rs`, `src/hud.rs`) for crisp tooltips and localized unit stats cards
  (`src/controls.rs`), with tooltips on every command-card slot, tab, category chip,
  page, queue, upgrade, enhancement, and order button (`src/command_card.rs`).
- Exit: `farust loc --check` verifies 6,413 `loc_US.scd` entries + German lookups;
  German Options & in-match tooltip snapshots (`FARUST_LANG=de`) render localized
  unit names, descriptions, and umlauts cleanly; 273/273 tests pass (incl. `Loc`
  fuzzing); sim hashes unchanged (`sim-run` `35db8f2bd7a6faf2`, `skirmish SCMP_001`
  `192e89b083b978cb`). Done.

### Phase 38 — Observer & caster tools (planned)
- Live spectator slots in the lobby, observer HUD (all-army economy, army
  value graph, APM), follow-unit camera, picture-in-picture minimap.
- Exit: a 2-player match with 1 live observer completes without desync.

### Phase 39 — Onboarding & accessibility (done)
- Tutorial: `missions/tutorial.farust-mission` (built into the binary, main menu → TUTORIAL)
  walks a UEF opening one objective at a time — move the Commander, mass extractor, power
  generator, land factory, three Strikers, destroy scripted Cybran scouts. Mission
  conditions gained `COND and COND …` (`Cond::All`) to chain steps.
- Accessibility (`settings.ini` + Options checkboxes): `colorblind` (Okabe–Ito palette per
  army slot instead of faction colours, also `FARUST_COLORBLIND=1`), `mouse_swap` (left /
  right buttons swapped in matches), `subtitles` (default on: every announcer voice line
  is captioned in the status banner, EN/DE caption texts).
- Exit: `farust mission missions/tutorial.farust-mission --human-ai normal` completes all
  six objectives → `Victory(0)` at 3:21 and its replay verifies; main menu shows TUTORIAL,
  Options shows the three toggles; 325/325 tests. Done.
- Not done: UI scale factor, controller input, captions while audio is disabled.

### Arc C — Visual & audio fidelity III (40–43)

### Phase 40 — FA effect blueprints (done)
- `src/effects.rs` + `src/blueprint.rs`: Lua long-bracket string (`[[...]]`) and nesting-depth
  parser support; parses all 2,827 `.bp` files in `effects.scd` (2,437 `EmitterBlueprint`,
  103 `BeamBlueprint`, 184 `TrailEmitterBlueprint`, 103 other) with 0 failures (`farust effects --check`)
  and `Kind::EffectBp` mutation fuzzing (`src/fuzz.rs`).
- Data-driven `EffectTemplates.lua` + weapon/projectile/collision-beam Lua mapping (`farust effects --coverage`):
  404/404 (100.0%) weapons mapped to FA emitters (368 exact Lua chain, 2 name match, 34 faction/category
  FA fallback; 789 templates, 2,724 valid effect blueprints).
- `src/vfx.rs` + `src/viewer.rs` + `src/render_prep.rs`: evaluates FA emitter curves (emit rate,
  lifetime, size, velocity, acceleration, rotation, sprite-sheet strips/frames, 4-stop DDS color ramps,
  additive/alpha premultiplied blending) into a `128x128` GPU `texture_2d_array` (up to 256 layers)
  with a hard 4,096-particle cap driving muzzle flashes, impacts, explosions, ACU nukes, shield hits,
  projectile poly-trails, and faction build beams. Render-only.
- Exit: 404/404 (100.0%) weapons use FA emitters (≥ 90% target); `cargo test --release` 271/271 passed;
  sim hashes unchanged (`sim-run` `35db8f2bd7a6faf2`, `skirmish SCMP_001` `192e89b083b978cb`);
  `bench-render` at 8,192 units stays at ~5.7 ms p50 frame / 252 draws. Done.

### Phase 41 — Dynamic decals & terrain marks (done)
- `src/decals.rs` `DecalTracker` (render-only, observes the world while frames are
  prepared and during snapshot warm-up): craters under new wrecks, scorch marks where
  splash shells land, tank treads every 2.5 m behind moving land units, footprints under
  live structures. FA textures (`env/Common/decals/Scorch_001` / `crater01`,
  `textures/tilesets/_global/tank_treads` / `scorch_mark`), terrain-draped 3×3 quads.
- Fixed budgets per kind (256 scorch / 128 crater / 384 tread / 512 footprint, oldest
  recycled) and fade-out lifetimes (3 min / 5 min / 15 s / while the structure stands);
  a backward replay seek clears the marks. GPU geometry is rebuilt only when the tracker
  revision changes; the decal shader now honours vertex alpha (map decals use 1.0).
- Exit: budget/expiry and live-battle tracker tests pass (327/327); a 12-minute `SCMP_001`
  battle holds 28 scorch / 68 crater / 384 tread / 87 footprint marks. `bench-render
  --no-sim` 8,192 units: frame p50 7.36 ms (7.33 ms before), decals 0.03 ms/frame avg.
  Bisecting the frame time also found and fixed a Phase 39 regression (per-unit settings
  clone: 21.6 ms → 7.4 ms). Done.

### Phase 42 — HDR, bloom & glow (done)
- `Rgba16Float` HDR 3-D scene target (`src/viewer.rs`, `src/play.rs`, `src/perf_cmds.rs`,
  `src/map_author.rs`) with a 5-level half-res bright-pass (Karis anti-firefly + soft knee) →
  Jimenez 13-tap downsample → additive 3×3 tent upsample bloom mip chain driven by `.scmap`
  `Bloom` (skipped when off or 0), Narkowicz ACES filmic tonemap with exposure from `.scmap`
  `LightingMultiplier`, relative gamma (`0.8..1.4`), optional FXAA 3.11 or MSAA 4× (auto-falls
  back to FXAA if `Rgba16Float` 4× MSAA is unsupported), and LDR 2-D HUD composited on top
  after tonemapping/AA.
- Data-driven unit glow & specular maps (`SpecGlowMap` in `src/dds_decode.rs`, `src/scene.rs`,
  `src/play.rs`, `src/render_prep.rs`): extracts FA's `*_SpecTeam.dds` `G` (specular) and `B`
  (emissive/glow mask per `effects/mesh.fx`) into a linear `Rg8Unorm` texture for units
  (both `view-unit` and instanced LOD0/LOD1 skirmish meshes; disabled on wrecks) so glowing
  visors/cores/vents and additive VFX/shield hits bloom naturally.
- Settings & Options UI (`src/settings.rs`, `src/menu.rs`, `src/loc.rs`): `bloom=on|off`,
  `aa=off|fxaa|msaa4`, `gamma=0.80..1.40` (plus `FARUST_BLOOM` / `FARUST_AA` / `FARUST_GAMMA`
  overrides) backed by a lock-free `AtomicU32` cache (`settings::render_post`) so per-frame
  reads never clone `Settings`. Render-only.
- Exit: 331/331 tests pass (incl. tonemap/gamma curve, bloom mip chain, AA fallback, settings
  round-trip & atomic cache); determinism hashes unchanged (`sim-run --ticks 10000`
  `35db8f2bd7a6faf2`, `skirmish SCMP_001` `0fe934990406d829`); `bench-render --no-sim` at
  8,192 units: frame p50 `8.00 ms` MSAA4 / `7.97 ms` FXAA (`gpu` `1.93 ms` / `1.66 ms`,
  `gpu.post_pass` `0.077–0.097 ms`, within +15% of baseline); battle, `UEL0001`/`UAL0001`,
  `view-map`, and Options snapshots verified visually with bloom on/off and FXAA. Done.

### Phase 43 — Announcer, ambience & adaptive music (done)
- Recursive SoundBank/WaveBank scan & direct `"Bank#Index"` wave playback
  (`src/xact.rs`, `src/main.rs`): indexes nested `sounds/Voice/<lang>/XGG.xsb`
  (89 `.xwb` wavebanks, 3,163 cues, 3,043/3,043 waves decoded) and raises
  `MAX_STREAM_VOICES` to 6 so music crossfades and dual-layer ambience never
  evict each other. `farust audio cues --all` dumps every indexed cue.
- Render-side announcer (`AnnouncerEvent`, `AnnouncerQueue`, `EconWatch` in
  `src/audio_play.rs`): 16 tactical/economy events derived from sim events,
  army economy, unit damage/shields/transports/cap and shared allied intel for
  the focus army, with priority ordering, single-voice playback (0.35 s breath
  gap, 12 s staleness expiry), and 10–60 s per-event cooldowns. Maps 14/16 to
  authentic `XGG.xsb` cues (including 12 per-experimental `XGG_HQ_GD1_04179..04190`,
  15 domain/tier/compass inbound `04142..04172`, and 8 structure-lost `04213..04221`
  variants); `mass_storage_full` / `energy_storage_full` target SupCom 1's
  `UEFComputer_Resources_01313`/`01312` and honestly report `MISSING` on
  standalone FA installs where `COMPUTER_*_VO` is absent.
- Per-map biome ambience (`MapBiome`, `MapAmbience` in `src/audio_play.rs`,
  `src/play.rs`): classifies `.scmap` stratum paths, skybox and heightmap water
  fraction into `Temperate` / `Oceanic` / `Volcanic` / `Arid` / `Tundra` dual
  looping streams (`AMB_Planet_Rumble_zoom` + surface water/fire/tundra/arid
  detail) cross-scaled by camera zoom.
- Adaptive music (`MusicDirector` in `src/audio_play.rs`): camera- and
  focus-army-weighted combat impulse with asymmetric EMA smoothing (2.5 s
  attack / 8.0 s release), Schmitt-trigger hysteresis (`0.45` enter / `0.15`
  exit, 10 s min battle dwell, 18 s calm hold), crossfades `Base_Building` ↔
  `Battle`, and ducks to `0.62×` under announcer VO.
- Exit: `farust audio coverage` reports all 16 announcer events (`14/16` mapped,
  `2 MISSING` on standalone FA) plus biome ambience & music playlists;
  `audio render-mix` on `SCMP_001` (`--from 11800 --ticks 1800`) voices 13
  announcer callouts, oceanic dual-layer ambience and `Base_Building` → `Battle`
  transition with `identical` state hash (`6c2f941aef13f77b`). 268/268 unit
  tests pass; `sim-run --ticks 10000` `35db8f2bd7a6faf2`, `skirmish SCMP_001`
  `192e89b083b978cb`. Done.

### Arc D — Netcode & online (44–47)

### Phase 44 — UDP lockstep & NAT hole punching (done)
- Reliable-ordered UDP transport (`src/udp_transport.rs`, `std::net::UdpSocket` only):
  shared `UdpMux` + `UdpChannel` with sequence numbers, cumulative + 64-bit SACK,
  fast retransmit, Karn/Jacobson SRTT/RTTVAR/RTO (with Karn Part 2 RTO backoff),
  duplicate suppression, MTU-safe fragmentation (`1180` B chunks, up to 2 MiB frames),
  keepalive/idle timeout, graceful FIN drain, and a seeded in-process `NetemSocket`
  impairment shim (`loss`, `dup`, `reorder`, `delay`, `jitter`, `seed`).
- Transport abstraction & adaptive input delay (`src/net.rs`, protocol v14): `PeerStream`
  runs `LockstepHost` / `LockstepClient` over TCP or UDP identically; `AdaptiveDelayController`
  samples worst-peer `(srtt, rttvar)` and emits `NetControl::InputDelay` (`SYSTEM_SENDER`,
  clamped `2..=8` turns with down-step hysteresis) so all peers switch `input_delay` on the
  exact same turn boundary. CLI: `--udp` and `--netem loss=5,dup=2,reorder=3` on `host`/`join`.
- NAT hole punching (`src/rendezvous.rs`): `RendezvousServer` binds TCP+UDP on the same port,
  observes peer public UDP endpoints (`PKT_RV_REG`/`PKT_RV_ACK`), and coordinates simultaneous
  punches (`ClientPunch`/`HostPunch`/`PunchInfo`, backward-compatible v1 extensions); punch
  failure or `--relay` falls back seamlessly to direct TCP / TCP relay.
- Exit: 336/336 release tests pass (2- and 3-peer UDP loopback with 5% loss / 2% dup / reorder 3
  matches TCP hash, high-RTT adaptive delay syncs identically across peers, rendezvous UDP punch
  and TCP fallback pass); localhost 2-peer `SCMP_001` 600-tick runs over TCP, clean UDP, lossy
  `--netem` UDP (46 drops / 17 dups / 282 reorders), and rendezvous `--code --udp --netem`
  (`udp-punch`, 48 drops / 13 dups / 291 reorders) all end on identical hash `4f868f25ff409113`;
  `sim-run --ticks 10000` `35db8f2bd7a6faf2` and `skirmish SCMP_001` `0fe934990406d829` unchanged. Done.
- Not done: menu lobby (`net_lobby.rs`) still runs its pre-game handshake over TCP; symmetric-NAT
  port prediction (falls back to TCP relay).

### Phase 45 — Reconnect & late join (planned)
- Rejoin after a drop: savegame snapshot transfer + catch-up re-sim, AI
  hands control back, observers can join a live match.
- Exit: killing and restarting one peer mid-match ends on the same final
  hash as an uninterrupted run.

### Phase 46 — Steamworks lobby & friend invites (planned, blocked)
- Formerly the Phase 28 candidate. Needs a dedicated Steam Direct App ID;
  `CreateLobby` stores the Phase 26 invite code in lobby metadata,
  `GameLobbyJoinRequested_t` and `+connect_lobby` join directly. Behind a
  `steam` cargo feature so the default build stays dependency-light.
- Exit: a friend invite from the Steam overlay lands in the FARust lobby.

### Phase 47 — Replay vault & sharing (planned)
- Replay index with metadata, thumbnails and search, optional upload to and
  download from the rendezvous server, event bookmarks (kills, experimentals).
- Exit: a replay uploaded by one client plays back verified on another.

### Arc E — Content & authoring (48–51)

### Phase 48 — Scenario scripting engine (done)
- `src/mission.rs`: line-based `*.farust-mission` scripts — `mission`, `map`, `victory`,
  `army N faction F [team T] human|ai DIFF|ai none`, `objective ID primary|secondary
  "TEXT"`, `when COND do ACTION; ACTION…`. Conditions: `at m:ss`, `every m:ss from m:ss
  [until m:ss]`, `dead commander A`, `no units A`, `count A TYPE|any in X,Z rR OP N`,
  `objective ID complete|failed`. Actions: `say`, `spawn A TYPE xN at X,Z [attack|move
  X,Z]`, `objective ID complete|fail`, `win A`, `lose A`, `camera X,Z [zoom D]`. Parse
  errors carry line numbers; unknown objectives and gappy army indices are rejected.
- The runtime is deterministic world state: evaluated after every tick's cleanup in file
  order, hashed, cloned with the world (seek / saves), and its source travels in replay
  v6 headers (`ReplayHeader::apply_mission` on verify, playback and save loading).
- `farust mission FILE [--human-ai DIFF] [--record OUT]` (headless) and
  `farust play --mission FILE` (messages and objective changes in the status banner,
  scripted camera moves). Example: `missions/hold_the_line.farust-mission` (SCMP_009,
  10-minute defence against scripted Cybran waves).
- Exit: with `--human-ai normal` the example ends `Victory(0)` at 10:00 with both primary
  objectives complete and its recorded replay verifies; with an idle player the ACU falls
  at ~8:19 → defeat. Parser + runtime/determinism tests pass (324/324). Done.
- Not done: mission editor, dialogue portraits / voice, multiplayer missions.

### Phase 49 — First campaign operation (planned)
- Reimplement one FA campaign operation (e.g. `X1CA_001`) on the scripting
  engine using the user's map data, voice cues and objectives.
- Exit: the operation is beatable start to finish with all primary objectives.

### Phase 50 — Map authoring tools (done)
- Byte-exact `.scmap` v2.60 writer (`src/scmap.rs`): retains header unknowns, raw
  C-strings, and post-water tail verbatim (`write(parse(b)) == b`), with world-space
  raise/lower/smooth heightmap brush (`apply_brush`), bilinear `sample_height`, and
  prop-slice re-serialization (`add_prop` / `remove_prop_near`) when props change.
  Embedded DXT5 normal-map DDS in the tail is preserved as-is; geometric terrain
  normals recompute from the edited heightmap.
- Scenario & save writers (`src/scenario.rs`): case-insensitive `rewrite_map_id`,
  `rewrite_scenario_for_clone` (`map`/`save`/`script` VFS paths + title), and
  comment/string-aware `rewrite_save_markers` regenerating start spots (`ARMY_n`),
  `Mass`, and `Hydrocarbon` markers while keeping all other markers and Lua outside
  `Markers` verbatim.
- CLI & interactive GPU map editor (`src/map_author.rs`, `src/viewer.rs`,
  `src/ui_kit.rs`, `src/scene.rs`): `farust map-roundtrip [--all | MAP]`,
  `farust map-edit <SRC> <NEW_ID> [--raise|--lower|--smooth|--add-mass|--del-mass|--add-hydro|--del-hydro|--move-start|--add-prop|--del-prop]`,
  and `farust map-editor <MAP> [--out NEW_ID] [--snapshot OUT.png]` with a `ui_kit`
  toolbar, heightmap/smooth brushes, mass/hydro/start/prop editing, and safe saves
  to `$XDG_DATA_HOME/farust/maps/<NEW_ID>/` (never writing to the game install).
- Exit: `map-roundtrip --all` passes 61/61 stock maps byte-identically (825.55 MiB,
  61/61 reparse ok); `map-edit SCMP_001 FARUST_TEST01` produces a map that plays to
  completion via `skirmish FARUST_TEST01` and renders via `view-map` and `map-editor`;
  272/272 tests pass; golden hashes unchanged (`sim-run` `35db8f2bd7a6faf2`,
  `skirmish SCMP_001` `192e89b083b978cb`). Done.

### Phase 51 — Mod manager & lobby mod sync (planned)
- In-game mod manager (enable/disable, conflicts), automatic mod transfer
  between lobby peers with hash verification, mod-defined weapons/abilities
  as data.
- Exit: a client missing the host's mod receives it and the match starts.

### Arc F — AI III (52–53)

### Phase 52 — Strategic AI planner (done)
- New `hard` built on the Phase 19 one, which is kept frozen as the `hard19` benchmark
  difficulty (`--ai hard19`, not in the menu):
  - tech push: a busy land factory is reserved for its T1→T2→T3 upgrade (no new
    production, upgrade the moment it idles) — Phase 19 only upgraded idle factories,
    which never happened because every factory was refilled each think;
  - map awareness: when no enemy base is land-reachable (island maps such as
    `SCMP_005`), air and naval factories scale with income;
  - economy: leaner power (energy ratio 5, batches of 2) — hard19 overbuilt generators
    while starving mass — plus 2 parallel extractor jobs, 3 parallel engineers, up to 14
    engineers / 8 land factories;
  - committed fighting: attack at 110 %, retreat at 180 %, a single raid group.
- `farust ai-arena` standings print a batch-Elo ladder (`skirmish_ai::elo_ratings`).
- Exit: 100-game arena (`SCMP_001/002/005/009/014` × seeds 1–10 × both sides,
  60 min limit): new hard 70 – 22 – 8 vs hard19; held-out seeds 11–20: 69 – 23 – 8
  (≈ 75 % of decisive games). Ladder on a 144-game mixed arena: hard 1572, normal 1480,
  hard19 1448. Normal AI unchanged (`skirmish SCMP_001` hash identical). Done.
- Tried and rejected (no gain or worse): earlier tech thresholds, earlier extractor
  upgrades, harder commitment (100 / 220 %), capping land factories on island maps.

### Phase 53 — AI personalities & co-op commands (done)
- Pings are sim commands (`Command::Ping`, replay tag 29, rules v10, protocol v15) as
  well as chat lines, so every lockstep peer's allied AI sees the same pings. AIs that
  coordinate with their team answer the newest ally ping within 30 s: attack → push the
  army there, alert → defend there, move → lend two engineers that assist the ally's
  nearest builder (released after 2 min).
- Personalities over any difficulty (`--personality rush,turtle` per army on `skirmish` /
  `play`, `personality X` on mission `army` lines): rush (early waves, no tech), turtle
  (extra point defenses / shields, home guard, attacks only at 170 %), air (air
  factories scale with income), tech (tech push, experimentals from 45 mass/s).
- Exit: a 2v1 test where the human pings and the allied AI's army closes in on an attack
  ping, lends two engineers on a move ping and gathers at an alert ping, all within 300
  ticks (339/339 tests). Default skirmish hash unchanged. Done.
- Not done: difficulty-scaled resource cheats (need a per-army replay-header option),
  personalities in `ai-arena` matchups.

### Arc G — Scale, platforms & release (54–57)

### Phase 54 — Sim throughput at 10k units (done)
- Per-unit economy cached on the unit (`eco` = mass/energy per tick, storage; refreshed on
  upgrade) so income and storage caps are one catalog-free pass; shields only visit
  shielded units; construction skips plain mobile units with nothing to build. All pure
  optimisations: every hash is unchanged.
- Exit: `bench-scale SCMP_001 --units 10000 --ticks 100 --threads 16`: 67 ticks/s
  (60 before) with exact `1t == 16t` (`15575a588a785eda`); construction 1.68 → 0.75 ms,
  economy 1.26 → 0.47 ms per tick. 2,048 units: 179 / 338 ticks/s (was 170 / 305). Done.
- Next candidates (not done): parallel weapon fire decisions (they do not depend on
  same-tick damage, so a parallel decide + ordered apply is exact; 2.9 ms at 10k units),
  flow-field reuse across goals (4.3 ms).

### Phase 55 — 10k-unit rendering (done, without GPU-driven culling)
- Measured first: with Phase 42's HDR pipeline, 10,000 units render at frame p50 9.4 ms
  (≈ 106 fps; GPU 2.0 ms, CPU prep 6.3 ms) at the bench camera, so the 60 fps target holds
  without moving culling to compute shaders.
- Fixed the real stutter: new projectiles found their shooter (for the shell / missile
  look) by scanning every unit; shells now carry their shooter id (artillery included —
  not hashed, no rules change), cutting `render.begin.proj` from 3.47 to 0.42 ms average
  and the worst prep frame from 110 ms to 10 ms at 10k units.
- Exit: `bench-render SCMP_001 --units 10000 --frames 30 --no-sim`: frame p50 9.37 ms,
  p95 9.94 ms. Done for the target; compute-shader culling / indirect draws (the original
  plan) are not implemented — CPU instance preparation (≈ 6 ms) is the next lever.

### Phase 56 — Windows & macOS ports (planned)
- Native audio output (WASAPI / CoreAudio) behind the existing silent
  fallback, FA install discovery per OS, cross-architecture determinism test
  (x86_64 vs aarch64 hashes must match).
- Exit: a Linux–Windows lockstep match ends on an identical hash.

### Phase 57 — CI, packaging & 1.0 (planned)
- CI running tests, fuzzers and golden-hash suites; crash reporter with
  desync dumps; AppImage/Flatpak + Windows zip; 1.0 release checklist.
- Exit: tagged 1.0 builds produced by CI for all three platforms.

### Other known gaps
- FAF network/replay compatibility stays out of scope (see Risks).

## Risks / non-goals

- Not a drop-in `ForgedAlliance.exe` replacement; no FAF network/replay
  compatibility in these phases.
- Lua: full Moho Lua compatibility is out of scope. Blueprints are data and
  are parsed directly; gameplay scripts are reimplemented.
- `.scm`/`.sca`/`.scmap` are binary and partly undocumented; REA/Ghidra are
  fallback tools only if header parsing stalls.
