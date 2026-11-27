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
- `LMB` click/drag: select / box-select (`Shift` adds); `F` Commander, `H` Factories, `J` Combat Army, `Ctrl+1..9`/`1..9` control groups, `.` idle engineer
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
cargo run --release -- host 127.0.0.1:15000 SCMP_001 --headless-ticks 600
cargo run --release -- join 127.0.0.1:15000
cargo run --release -- bench-scale SCMP_001 --units 2048 --ticks 200
```

## Steps 11–16 — done: animation, combined arms, RTS controls, audio, multiplayer & lobby, scale

Details in [docs/PLAN.md](docs/PLAN.md).

```sh
cargo run --release -- play --lobby                       # in-window lobby -> skirmish
cargo run --release -- play SCMP_001 --mute               # F3 mute, F4/F5 volume
cargo run --release -- view-unit URL0402 --anim walk --anim-time 0.7
cargo run --release -- host 127.0.0.1:15000 SCMP_001 --players 2 --play   # + join 127.0.0.1:15000 --play
cargo run --release -- host 127.0.0.1:15000 SCMP_001 --players 3 --headless-ticks 600
cargo run --release -- audio render-mix fa-unpack/audio/mix.wav SCMP_001 --from 7200 --ticks 1200
```

New controls: RMB-drag = formation line, LMB-drag in build mode = structure row,
Ctrl+RMB-drag = area reclaim/repair, double-click = select same type, range rings and
stats card on selection. Replays: timeline bar (click to seek), `[`/`]` ±30s, `V` perspective.

## Steps 17–21 — done: visual fidelity II, gameplay fidelity II, skirmish AI II, front end & match session

- **Phase 17 — Visual Fidelity II**: Turret pitch & barrel recoil (`src/sca.rs`), animation crossfading (walk/idle/death/open/build), Fresnel shield domes + hit ripples (`src/viewer.rs`), projectile meshes from `projectiles.scd` + smoke trails (`src/vfx.rs`), aircraft banking & contrails, naval wakes & sinking wrecks (`src/render_prep.rs`).
- **Phase 18 — Gameplay Fidelity II**: Victory conditions (`assassination`, `supremacy`, `annihilation`, `sandbox`) and configurable `--unit-cap` stored in v3 replay headers (`src/replay.rs`), allied shared vision & overflow, stealth/cloak/omni/jammer intel, air transports (`L` unload, `RMB` board), ACU/SCU enhancements (`K` / `Shift+K`), factory repeat queue (`Z` / `Ctrl+R`), shield toggle (`X`), and extended unit stats card (`src/skirmish.rs`, `src/controls.rs`, `src/hud.rs`).
- **Phase 19 — Skirmish AI II**: Difficulties (`easy`, `normal`, `hard`, `adaptive`, `legacy`), per-faction openings (`src/ai_build.rs`), decaying threat map (`src/ai_threat.rs`), scouting, raids, counter-composition, air/naval production, allied coordination, micro (`src/skirmish_ai.rs`), `--ai <diff>`, and parallel `farust ai-arena` tournaments.
- **Phase 20 — Front-End Shell & Settings**: Graphical menu on a single event loop (`src/menu.rs`, `src/ui_kit.rs`, `src/ttf.rs`), `settings.ini` (`src/settings.rs`), skirmish lobby, replay browser + verifier, options screen, and v3 TCP multiplayer lobby with peer list, ready flags, and lobby chat (`src/net_lobby.rs`).
- **Phase 21 — Match Session**: Pause menu (`Esc`), `.farust-save` save/load via fast re-sim with shadow AI restoration (`src/savegame.rs`, `F6`/`F9`, `--load`), automatic replay recording (`--no-record` disables), score screen with line graphs (`src/stats.rs`, `Tab`), synchronized net pause/speed, in-game chat (`src/chat.rs`, `Enter`/`Shift+Enter`), AI takeover on disconnect, and desync state dumps (`src/net.rs`).

```sh
cargo run --release                                       # open front-end main menu
cargo run --release -- menu --screen skirmish             # open skirmish setup
cargo run --release -- play SCMP_001 --ai hard,normal --victory assassination --unit-cap 500
cargo run --release -- play --load fa-unpack/saves/quicksave.farust-save
cargo run --release -- ai-arena --maps SCMP_001 --seeds 2 --matchups hard:normal,normal:easy,hard:legacy
```

## Step 22 — done: per-shadow-quality shadow maps

- `settings::shadow_quality()` hook (`Medium` = stock 2048px with no active settings);
  `GpuSceneResources::new` sizes the shadow target from it (1024 / 2048 / 4096,
  1px placeholder + skipped pass for `Off`).
- PCF texel + enabled flag in `Uniforms::shadow_params`; terrain and instanced
  shaders use the live texel size and render fully lit when off.
- `set_shadow_quality()` rebuilds the target live, applied on window-config
  resize in `play.rs` and `viewer.rs`. Render-only: sim hash unchanged
  (`sim-run --ticks 10000` → `35db8f2bd7a6faf2`).

## Steps 23 & 25 — done: FA command interface, profiler & 8k-unit render performance

- **Phase 23 — FA Command Interface**: Blueprint/roster-driven command card (`src/command_card.rs`, `src/hud.rs`) with T1–T4/EXP tabs, category chips, factory queue & upgrade panels (`Command::CancelProduce`), ACU/SCU enhancement slots, targeting-mode order buttons, camera bookmarks (`F7`/`F8`/`F10`/`F11`), ally map pings, and 87 remappable actions in Options → Keys (`src/keymap.rs`).
- **Phase 25 — Performance & Robustness**: Remappable `F12` profiler overlay (`src/profiler.rs`), `farust bench-render` GPU/CPU timing (`453 ms` parallel asset load, `3.52 ms` CPU prep / `6.05 ms` total frame p50 at 8,192 units with shadow-pass prop frustum culling), and `farust fuzz-parsers` mutation tests (`src/fuzz.rs`).

```sh
cargo run --release -- bench-render SCMP_001 --units 4096,8192 --frames 30 --no-sim
cargo run --release -- fuzz-parsers --iters 100000
```

## Step 24 — done: content & mods

- **Data-driven roster** (`src/roster.rs`): all 568 blueprints across 4 factions →
  356 playable (was 157 hand-listed); 18 excluded for unmodelled mechanics.
  `farust roster --verify SCMP_009` builds 352/352 verified units in-sim.
- **User maps** (`src/scenario.rs`): stock maps plus FA user folders
  (loose folders and `.scd`/`.zip` archives, extraction cache), shown in lobby/menu.
- **Blueprint mods** (`src/mods.rs`): directories under
  `$XDG_DATA_HOME/farust/mods` with `mod_info.lua` + `units/**/*.bp`
  (deep-merge / replace / add-new-unit), enabled via `--mod` / `FARUST_MODS`;
  mod identities are recorded in replay/save/lockstep headers and mismatches rejected.
- **Layered audio** (`src/xact.rs`, `src/audio_play.rs`): XACT tracks play as
  simultaneous layers with per-layer variations; construction/movement
  `*_Loop` cues follow their units. Audio never touches the sim hash.

```sh
cargo run --release -- roster
cargo run --release -- roster --verify SCMP_009
cargo run --release -- mods
cargo run --release -- --mod MyMod skirmish SCMP_001
```

## Steps 26–27 — done: online invite codes + relay & classic FA UI theme

- **Invite Codes & TCP Relay** (`src/rendezvous.rs`, `src/net_lobby.rs`, `src/net.rs`):
  Host and join using `XXXX-XXXX` invite codes via a lightweight self-hosted
  `farust rendezvous` server (tries direct TCP first, falls back to bi-directional
  TCP relay through the rendezvous server without port-forwarding). Supports
  Steam-compatible `+connect_lobby <CODE>` CLI launch args.
- **Classic FA Layout & Faction Skins** (`src/settings.rs`, `src/ui_kit.rs`, `src/hud.rs`):
  Toggle `UI Theme` in Options (`modern` / `classic`, or `FARUST_UI_THEME=classic`)
  for the old-school top-left split Mass/Energy console, open sky, bottom-left minimap +
  orders + build card + bottom-right unit portrait dock, and faction-colored beveled chrome.

```sh
cargo run --release -- rendezvous 0.0.0.0:15050
cargo run --release -- host --code SCMP_001 --rendezvous 127.0.0.1:15050 --play
cargo run --release -- join --code XXXX-XXXX --rendezvous 127.0.0.1:15050 --play
FARUST_UI_THEME=classic cargo run --release -- play SCMP_001
```

## Step 43 — done: announcer, map ambience & adaptive music

- **Tactical & Economy Announcer** (`src/xact.rs`, `src/audio_play.rs`): Recursive `sounds/` scan indexes `Voice/US/XGG.xsb` (89 wavebanks, 3,163 cues, plus direct `"Bank#Index"` wave playback). Render-side detection drives a single-voice priority queue with 10–60 s per-event cooldowns for the focus army & shared allied intel (nuke launch/inbound/intercepted, commander under attack, 12 unit-specific experimental callouts, 15 directional/domain/tier base-under-attack callouts, 8 structure-lost callouts, shield down, mass/energy low/empty, unit cap, transport full; SupCom 1 storage-full cues reported `MISSING` on standalone FA).
- **Per-Map Biome Ambience & Adaptive Music** (`src/audio_play.rs`, `src/play.rs`): `.scmap` terrain strata + water fraction select `Temperate` / `Oceanic` / `Volcanic` / `Arid` / `Tundra` dual-layer ambient loops scaled by strategic zoom; smoothed camera-weighted combat intensity crossfades `Base_Building` ↔ `Battle` playlists with hysteresis (`0.45` / `0.15`, 18 s calm hold) and ducks under announcer VO without touching sim hashes.

```sh
cargo run --release -- audio coverage
cargo run --release -- audio cues --all
cargo run --release -- audio render-mix fa-unpack/audio/phase43_mix.wav SCMP_001 --from 11800 --ticks 1800
```
