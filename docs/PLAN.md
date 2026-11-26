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

### Known gaps / next candidates
- 18 roster units excluded for unmodelled mechanics (walls, air staging,
  engineering stations, orbital satellite, crab eggs — see `farust roster`).
- FAF network/replay compatibility (see Risks).

## Risks / non-goals

- Not a drop-in `ForgedAlliance.exe` replacement; no FAF network/replay
  compatibility in these phases.
- Lua: full Moho Lua compatibility is out of scope. Blueprints are data and
  are parsed directly; gameplay scripts are reimplemented.
- `.scm`/`.sca`/`.scmap` are binary and partly undocumented; REA/Ghidra are
  fallback tools only if header parsing stalls.
