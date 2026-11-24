<img src="assets/icon.svg" width="64" height="64" alt="">

# FARust

Rust rebuild around original Forged Alliance data. Game files stay outside git.

## Game path

Default: `/home/bas_pit/.local/share/Steam/steamapps/common/Supreme Commander Forged Alliance`

Override: `FA_PATH=/path/to/game cargo run -- inventory`

## Step 1 — done: read-only .scd inspector

`.scd` files are ZIP archives. Commands:

```sh
cargo run -- inventory
cargo run -- list gamedata/lua.scd --limit 20
cargo run -- cat gamedata/lua.scd lua/abilitydefinition.lua
```

## Step 2 — done: unit blueprints

Full phased plan: [docs/PLAN.md](/home/bas_pit/.t3/projects/farust/docs/PLAN.md)

```sh
cargo run -- units --limit 20
cargo run -- units --search bomber --limit 5
cargo run -- unit UAL0001
```

All 568 `*_unit.bp` in `units.scd` parse.

## Step 3 — done: maps, meshes, and DDS headers

```sh
cargo run -- maps --check
cargo run -- map SCMP_001
cargo run -- meshes --check
cargo run -- mesh units/UAL0001/UAL0001_LOD0.scm
cargo run -- textures --check
cargo run -- dds units/UAL0001/UAL0001_Albedo.dds
```

Next: Phase 3 `wgpu` viewer.

## Step 4 — done: GPU (`wgpu`) unit & map viewer

Interactive window (`LMB` orbit, `RMB`/`WASD` pan, scroll wheel strategic zoom, `R` reset, `Esc` quit) and offscreen PNG snapshot mode:

```sh
cargo run -- view-unit UAL0001
cargo run -- view-map SCMP_001
cargo run -- view-unit UAL0001 --snapshot fa-unpack/renders/ual0001.png
cargo run -- view-map SCMP_001 --snapshot fa-unpack/renders/scmp_001.png
```

Next: Phase 4 deterministic simulation core.

## Step 5 — done: deterministic sim core & multicore job pool

```sh
cargo run -- sim-run --ticks 10000
cargo run -- bench-multicore
```

## Step 6 — done: playable skirmish slice

Headless deterministic skirmish (UEF vs. Cybran AI to assassination victory on `SCMP_001`) and interactive/snapshot 3D skirmish viewer:

```sh
cargo run --release -- skirmish SCMP_001
cargo run --release -- play SCMP_001
cargo run --release -- play SCMP_001 --snapshot fa-unpack/renders/scmp_001_skirmish.png --snapshot-ticks 5400
```

Controls in `farust play`:
- `LMB` click/drag: select / box-select (`Shift` adds); `1` Commander, `2` Factories, `3` Combat Army
- `RMB`: move / attack / factory rally (`Shift` queues), `RMB` drag: pan, `MMB` drag or `Q`/`E`: orbit, Wheel: zoom
- `B` Land Factory, `M` Mass Extractor (auto-snaps to mass points), `P` Power Generator, `T`/`Y` produce units (`Shift` x5)
- `A` Attack-Move, `S` Stop, `Space` Pause, `+`/`-` Speed (`1x..8x`), `C` Center camera, `R` Full-map camera, `F1` Toggle AI control

## Steps 7–10 — done: FA rendering/HUD, deeper gameplay, replays & TCP lockstep, WorkerPool + SpatialGrid scale

- **Phase 7 — Textured Instancing, Terrain Strata, Strategic Zoom & HUD**:
  - `.scmap` terrain strata (`10` albedo + `9` tangent-normal layers + `texturemap0`/`texturemap1`), map lighting/fog, water params, sky cubemap (`SkyCube_*.dds`), and placed props (`src/scmap.rs`, `src/scene.rs`).
  - GPU multi-stratum terrain splatting, 2048×2048 PCF shadow mapping, transparent depth-faded rippled water with sky reflections, alpha-cutout instanced `.scmap` props (`Pine06`, rocks, logs), `_LOD0.scm` team-colored units + charred `.scm` wrecks, clickable minimap, and real FA 64×64 DDS unit icons in the HUD (`src/viewer.rs`, `src/hud.rs`).
- **Phase 8 — Deeper FA Gameplay Mechanics**:
  - T1 Engineers (`G` build, repair, assist, reclaim wrecks), Point Defense (`D`), Radar (`N`), ballistic AoE artillery (`U`), ACU Overcharge (`O`), unit separation, and Fog-of-War / Radar intel toggle (`F2`).
- **Phase 9 — Replays & 2-Player TCP Lockstep**:
  - Binary `.farust-replay` (`FARPL1`) recording/verification/playback (`src/replay.rs`) and `TcpStream` 2-player lockstep networking with per-turn desync checking (`src/net.rs`).
- **Phase 10 — Persistent `WorkerPool`, `SpatialGrid` & 2,000+ Unit Scaling**:
  - Zero-`unsafe` persistent `WorkerPool` (`src/jobs.rs`) for batched A* pathfinding + deterministic parallel `step_separation` (`src/skirmish.rs`) with `1t == Nt` bit-identical hashes, dense zero-BTreeMap `UnitSpatialSnap`, parallel `render_prep`, and tight `SpatialGrid` (`src/spatial.rs`) sustaining 2,050 combat units at ~210 ticks/s.

```sh
cargo run --release -- skirmish SCMP_001 --record fa-unpack/replays/scmp_001.farust-replay
cargo run --release -- replay verify fa-unpack/replays/scmp_001.farust-replay
cargo run --release -- replay play fa-unpack/replays/scmp_001.farust-replay
cargo run --release -- host 127.0.0.1:15000 SCMP_001 --ticks 600
cargo run --release -- join 127.0.0.1:15000
cargo run --release -- bench-scale SCMP_001 --units 2048 --ticks 200
```
