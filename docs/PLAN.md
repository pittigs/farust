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

## Risks / non-goals

- Not a drop-in `ForgedAlliance.exe` replacement; no FAF network/replay
  compatibility in these phases.
- Lua: full Moho Lua compatibility is out of scope. Blueprints are data and
  are parsed directly; gameplay scripts are reimplemented.
- `.scm`/`.sca`/`.scmap` are binary and partly undocumented; REA/Ghidra are
  fallback tools only if header parsing stalls.
