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

### Phase 29 — Ballistics & line of fire (planned)
- Every structure footprint becomes a pathing blocker (reusing the Phase 28 wall
  overlay), with factory roll-off points so produced units exit cleanly.
- Replace direct-fire hitscan with simulated integer projectiles
  (`MuzzleVelocity`, `FiringTolerance`, turret yaw/pitch limits from the
  blueprint), terrain and wall occlusion, misses vs. moving targets, beam
  weapons kept as instant but terrain-checked.
- Exit: shots fired over a ridge hit the ridge; `bench-scale --units 2048`
  stays ≥ 180 ticks/s.

### Phase 30 — Air fuel, staging & repair (planned)
- `Fuel` / `FuelUseTime` / `FuelRechargeRate`, out-of-fuel speed penalty,
  air staging docking slots, refuel + repair, auto-return-to-staging toggle.
- Exit: the 4 air staging blueprints are playable; fighters on patrol return
  to staging when fuel runs low.

### Phase 31 — Engineering stations & drones (planned)
- Kennel / Hive builder stations, UEF pod drones and Cybran build drones as
  sim units owned by a parent, upgrades T2→T3, auto-assist radius.
- Exit: the 5 engineering-station blueprints leave the exclusion list.

### Phase 32 — Mobile factories, carriers & eggs (planned)
- Production from mobile units: Fatboy, Atlantis, Czar, Tempest, Megalith
  crab eggs (`XRL0002–0005`) with hatching, internal storage and launch.
- Exit: all four crab eggs are buildable; the exclusion list is down to the
  orbital satellite.

### Phase 33 — Experimental & special-ability completeness (planned)
- Novax orbital satellite (`XEB2402`), resource generators (Paragon),
  teleport, artillery experimentals, Yolona Oss, Salvation, Eye of Rhianne /
  Soothsayer intel. New `farust roster --abilities` audit lists every
  blueprint ability flag and whether it is modelled.
- Exit: `farust roster` reports 0 excluded units; the ability audit has no
  unmodelled flags on playable units.

### Phase 34 — Map prop reclaim & destructible trees (planned)
- Reclaiming live units (walls first, carried over from Phase 28).
- `.scmap` prop reclaim values (mass/energy/time), tree falling and burning
  (render), reclaim beams on props, area reclaim covering props, AI
  reclaim fields.
- Exit: an engineer area-reclaim on a forest on `SCMP_009` yields mass, and
  replays of it hash identically.

### Phase 35 — Fire states & unit behaviours (planned)
- Hold fire / return fire / ground fire (`Command::SetFireState`,
  `Command::AttackGround`), guard, pause construction, target priorities from
  blueprint `TargetPriorities`, air dive/hover, dock and stop-and-shoot.
- Exit: a replay covering every new command round-trips through replay v5.

### Arc B — Controls & UX (36–39)

### Phase 36 — Build templates, hotbuild & waypoint editing (planned)
- Saved build templates (base layouts), hotbuild key cycling, dragging queued
  waypoints, ferry routes for transports, shift-queue visual editing.
- Exit: a template placed in-game builds identically in a replay.

### Phase 37 — Localization & tooltips (planned)
- Parse FA `loc` string tables, unit/enhancement descriptions and help text,
  tooltips on every command-card button, language selection in Options.
- Exit: switching to German shows localized unit names and tooltips.

### Phase 38 — Observer & caster tools (planned)
- Live spectator slots in the lobby, observer HUD (all-army economy, army
  value graph, APM), follow-unit camera, picture-in-picture minimap.
- Exit: a 2-player match with 1 live observer completes without desync.

### Phase 39 — Onboarding & accessibility (planned)
- Interactive tutorial scenario, colour-blind team palettes, UI scale,
  rebindable mouse buttons, subtitle captions for voice cues.
- Exit: the tutorial completes start to finish from `farust menu`.

### Arc C — Visual & audio fidelity III (40–43)

### Phase 40 — FA effect blueprints (planned)
- Parse emitter/beam/trail blueprints from `effects.scd` and drive the
  `vfx.rs` GPU particle pool with them (muzzle flashes, impacts, explosions,
  shield hits, build beams). Render-only.
- Exit: weapon effects use FA emitters for ≥ 90% of weapons; sim hash
  unchanged.

### Phase 41 — Dynamic decals & terrain marks (planned)
- Runtime scorch, crater and tread decals with a bounded decal atlas and
  fade-out, footprint decals under structures, wreck shadows.
- Exit: a 20-minute battle stays under a fixed decal budget without frame
  time regressions.

### Phase 42 — HDR, bloom & glow (planned)
- HDR render target, FA-like tonemap and bloom from `.scmap` lighting values,
  emissive/glow maps on units, MSAA/FXAA option, gamma setting.
- Exit: `bench-render` at 8k units keeps frame p50 within +15% of Phase 25.

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

### Phase 44 — UDP lockstep & NAT hole punching (planned)
- UDP transport with sequence/ack/resend, NAT hole punching through the
  Phase 26 rendezvous server, TCP relay kept as fallback, adaptive input
  delay from measured RTT.
- Exit: a match survives 5% simulated packet loss with no desync; the netem
  loopback test passes.

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

### Phase 48 — Scenario scripting engine (planned)
- Rust-native, data-driven mission scripts (triggers, objectives, timers,
  spawn waves, dialogue, scripted camera). Not Moho Lua; deterministic and
  included in replays.
- Exit: a custom "defend the base for 10 minutes" mission is playable and
  replayable.

### Phase 49 — First campaign operation (planned)
- Reimplement one FA campaign operation (e.g. `X1CA_001`) on the scripting
  engine using the user's map data, voice cues and objectives.
- Exit: the operation is beatable start to finish with all primary objectives.

### Phase 50 — Map authoring tools (planned)
- `.scmap` v2.60 writer, in-game marker/prop editor, heightmap brush,
  round-trip `parse → write → parse` identity on all 61 stock maps.
- Exit: a map edited in FARust loads and plays as a skirmish map.

### Phase 51 — Mod manager & lobby mod sync (planned)
- In-game mod manager (enable/disable, conflicts), automatic mod transfer
  between lobby peers with hash verification, mod-defined weapons/abilities
  as data.
- Exit: a client missing the host's mod receives it and the match starts.

### Arc F — AI III (52–53)

### Phase 52 — Strategic AI planner (planned)
- Map-aware expansion and mex control, multi-front army management,
  experimental timing, naval/air dominance goals, built on `ai_threat.rs`.
  Elo ladder added to `farust ai-arena`.
- Exit: the new `hard` AI beats Phase 19 `hard` in ≥ 70% of 100 arena games.

### Phase 53 — AI personalities & co-op commands (planned)
- Personalities (turtle, rush, air, tech), allied AI that responds to pings
  ("attack here", "send engineers"), difficulty-scaled cheats as an option.
- Exit: pinged requests are acted on within 30 s in co-op matches.

### Arc G — Scale, platforms & release (54–57)

### Phase 54 — Parallel sim II, 10k units (planned)
- Deterministic parallel phases for projectiles, damage application and
  economy via per-thread buffers merged in canonical order.
- Exit: `bench-scale --units 10000` ≥ 60 ticks/s with exact `1t == Nt` hash.

### Phase 55 — GPU-driven rendering (planned)
- Compute-shader culling + LOD selection, indirect draws, GPU instance
  buffers updated incrementally.
- Exit: 10k units at ≥ 60 fps at strategic zoom on the reference machine.

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
