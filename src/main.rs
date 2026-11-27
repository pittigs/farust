mod ai_build;
mod ai_legacy;
mod ai_threat;
mod audio_play;
mod blueprint;
mod camera;
mod chat;
mod combat;
mod command_card;
mod controls;
mod dds;
mod dds_decode;
mod economy;
mod flowfield;
mod fuzz;
mod hud;
mod jobs;
mod keymap;
mod lobby;
mod menu;
mod mods;
mod net;
mod net_lobby;
mod pathfinding;
mod perf_cmds;
mod play;
mod profiler;
mod render_prep;
mod rendezvous;
mod replay;
mod roster;
mod savegame;
mod sca;
mod scenario;
mod scd;
mod scene;
mod scm;
mod scmap;
mod settings;
mod sim;
mod skirmish;
mod skirmish_ai;
mod spatial;
mod stats;
mod ttf;
mod ui_kit;
mod vfx;
mod viewer;
mod xact;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn print_usage() {
    eprintln!("farust - Forged Alliance Rust tooling");
    eprintln!();
    eprintln!("usage:");
    eprintln!("  farust [GAME_DIR]                              (launch graphical front-end menu)");
    eprintln!("  farust menu [GAME_DIR] [--screen NAME] [--map MAP] [--seed S] [--size WxH] [--snapshot OUT.png]");
    eprintln!("  farust inventory [GAME_DIR]");
    eprintln!("  farust list <FILE.scd> [--limit N]");
    eprintln!("  farust cat <FILE.scd> <INNER_PATH>");
    eprintln!("  farust units [FILE.scd] [--limit N] [--search TEXT]");
    eprintln!("  farust unit <ID> [FILE.scd]");
    eprintln!("  farust maps [GAME_DIR] [--limit N] [--check]      (stock + user maps, with source)");
    eprintln!("  farust roster [GAME_DIR] [--faction F] [--json]   (data-driven unit roster & coverage)");
    eprintln!("  farust mods                                       (installed blueprint mods)");
    eprintln!("  global: --mod <DIR|NAME> (repeatable) enables a blueprint mod for any subcommand");
    eprintln!("  farust map <NAME> [GAME_DIR]");
    eprintln!("  farust meshes [FILE.scd] [--limit N] [--check]");
    eprintln!("  farust mesh <PATH> [FILE.scd]");
    eprintln!("  farust textures [FILE.scd] [--limit N] [--check]");
    eprintln!("  farust dds <PATH> [FILE.scd]");
    eprintln!("  farust anim <list|UNIT_ID> [GAME_DIR] [--limit N] [--check]");
    eprintln!("  farust audio <list|info|export> [TARGET] [GAME_DIR] [--check] [--all] [--out DIR]");
    eprintln!("  farust audio <coverage|cues> [GAME_DIR] | audio play-cue <CUE> [GAME_DIR] [--mute] | audio render-mix <OUT.wav> [MAP] [GAME_DIR] [--from TICK] [--ticks N] [--seed S] [--factions a,b]");
    eprintln!("  farust view-unit <ID> [GAME_DIR] [--size WxH] [--snapshot OUT.png] [--anim walk|idle|NAME] [--anim-time SECS]");
    eprintln!("  farust view-map <NAME> [GAME_DIR] [--size WxH] [--snapshot OUT.png]");
    eprintln!("  farust sim-run --ticks N [--seed S] [--scenario NAME] [--dt SEC] [--hash] [--json]");
    eprintln!("  farust bench-multicore [GAME_DIR] [--threads N] [--limit N] [--json]");
    eprintln!("  farust bench-scale [MAP] [GAME_DIR] [--units N] [--ticks T] [--threads K] [--json]");
    eprintln!("  farust bench-render [MAP] [GAME_DIR] [--units N[,N..]] [--frames F] [--warmup-ticks T] [--size WxH] [--camera-dist D] [--no-sim] [--snapshot OUT.png] [--json]");
    eprintln!("  farust fuzz-parsers [GAME_DIR] [--iters N] [--seed S] [--no-real]");
    eprintln!("  farust skirmish [MAP] [GAME_DIR] [--armies 1,2] [--factions uef,cybran] [--ai D] [--victory V] [--unit-cap N] [--seed S] [--max-ticks N] [--record FILE] [--quiet] [--json]");
    eprintln!("  farust ai-arena [GAME_DIR] [--maps M1,M2] [--seeds N|S1,S2] [--matchups A:B,C:D] [--factions F0,F1] [--max-ticks N] [--victory V] [--unit-cap N] [--threads K] [--json [FILE]]");
    eprintln!("  farust play [MAP] [GAME_DIR] [--armies 1,2] [--factions uef,cybran] [--ai D] [--victory V] [--unit-cap N] [--seed S] [--speed N] [--load FILE] [--spectate] [--lobby [--lobby-autostart]] [--fow] [--mute] [--no-record] [--record FILE] [--camera-dist F] [--size WxH] [--snapshot OUT.png] [--snapshot-ticks N]");
    eprintln!("  farust replay <verify|play> <FILE> [GAME_DIR] [--speed N] [--mute] [--size WxH] [--snapshot OUT.png] [--snapshot-ticks N]");
    eprintln!("  farust host [<ADDR:PORT>] [MAP] [GAME_DIR] [--code] [--rendezvous ADDR] [--players N] [--armies 1,2] [--factions uef,cybran] [--seed S] [--play] [--mute] [--size WxH] [--headless-ticks N]");
    eprintln!("  farust join <ADDR:PORT | CODE> [GAME_DIR] [--code CODE] [--rendezvous ADDR] [--relay] [--play] [--mute] [--size WxH] [--headless-ticks N]");
    eprintln!("  farust rendezvous [BIND_ADDR]                     (run lightweight invite-code + TCP relay server)");
    eprintln!("  global: +connect_lobby <CODE|ADDR:PORT>           (Steam-compatible lobby auto-join arg)");
    eprintln!();
    eprintln!("env:");
    eprintln!("  FA_PATH             override for the game install dir");
    eprintln!("  FARUST_RENDEZVOUS   override rendezvous server host:port (default 127.0.0.1:15050)");
    eprintln!("  FARUST_FORCE_RELAY  =1 forces client relay fallback through rendezvous server");
    eprintln!("  FARUST_UI_THEME     modern | classic (old-school FA console layout + faction skins)");
    eprintln!("  FARUST_PROFILE      enable the scoped profiler (=overlay also shows the F12 overlay)");
    eprintln!("  FARUST_PROFILE_EXIT print the profiler span table on exit");
}

fn main() {
    let raw: Vec<String> = std::env::args().collect();
    // Phase 24/26: `--mod <DIR|NAME>` and `+connect_lobby <CODE>` are global; strip before dispatch.
    let mut args: Vec<String> = Vec::with_capacity(raw.len());
    let mut mod_specs = Vec::new();
    let mut connect_lobby: Option<String> = None;
    let mut it = raw.into_iter();
    while let Some(a) = it.next() {
        if a == "--mod" {
            match it.next() {
                Some(m) => mod_specs.push(m),
                None => {
                    eprintln!("error: --mod needs a directory or mod name");
                    std::process::exit(1);
                }
            }
        } else if a == "+connect_lobby" {
            match it.next() {
                Some(t) => connect_lobby = Some(t),
                None => {
                    eprintln!("error: +connect_lobby needs an invite code or HOST:PORT");
                    std::process::exit(1);
                }
            }
        } else {
            args.push(a);
        }
    }
    mods::set_enabled(mod_specs);
    let result = run(&args, connect_lobby.as_deref());
    if std::env::var_os("FARUST_PROFILE_EXIT").is_some() {
        profiler::end_frame(None);
        eprintln!("{}", profiler::report_table(None));
    }
    if let Err(e) = result {
        eprintln!("error: {e}");
        print_usage();
        std::process::exit(1);
    }
}

fn run(args: &[String], connect_lobby: Option<&str>) -> Result<(), String> {
    profiler::init_from_env();
    if args.len() < 2 {
        return cmd_menu(args, connect_lobby);
    }
    match args[1].as_str() {
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        "menu" => cmd_menu(args, connect_lobby),
        "rendezvous" => cmd_rendezvous(args),
        "ai-arena" => cmd_ai_arena(args),
        "inventory" => {
            let game_dir = positional_after(args, 2).unwrap_or_else(scd::default_game_dir);
            let gamedata = scd::gamedata_dir(&game_dir);
            println!("game: {}", game_dir.display());
            println!("gamedata: {}", gamedata.display());
            let entries = std::fs::read_dir(&gamedata)
                .map_err(|e| format!("cannot read {}: {e}", gamedata.display()))?;
            let mut scds: Vec<_> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "scd"))
                .collect();
            scds.sort();
            if scds.is_empty() {
                return Err(format!("no .scd files in {}", gamedata.display()));
            }
            for path in scds {
                let size = path.metadata().map(|m| m.len()).unwrap_or(0);
                println!(
                    "  {}  {:>12} bytes",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    size
                );
            }
            Ok(())
        }
        "list" => {
            if args.len() < 3 {
                return Err("missing FILE.scd".to_string());
            }
            let limit = flag_usize(args, "--limit").unwrap_or(50);
            let path = PathBuf::from(&args[2]);
            let entries =
                scd::list_scd(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            println!("{}: {} entries", path.display(), entries.len());
            for entry in entries.iter().take(limit) {
                let kind = if entry.is_dir { "dir " } else { "file" };
                println!("  {kind} {:>10}  {}", entry.size, entry.name);
            }
            if entries.len() > limit {
                println!("  ... ({} more, use --limit N)", entries.len() - limit);
            }
            Ok(())
        }
        "cat" => {
            if args.len() < 4 {
                return Err("usage: farust cat <FILE.scd> <INNER_PATH>".to_string());
            }
            let path = PathBuf::from(&args[2]);
            let data = scd::read_scd_file(&path, &args[3])
                .map_err(|e| format!("{}: {e}", path.display()))?;
            match String::from_utf8(data) {
                Ok(text) => print!("{text}"),
                Err(e) => {
                    return Err(format!("{} is not UTF-8 ({} bytes)", args[3], e.as_bytes().len()));
                }
            }
            Ok(())
        }
        "units" => {
            let scd_path = positional_after(args, 2).unwrap_or_else(default_units_scd);
            let limit = flag_usize(args, "--limit").unwrap_or(20);
            let search = flag_str(args, "--search").map(|s| s.to_lowercase());
            let units = load_units(&scd_path)?;
            let shown: Vec<_> = units
                .iter()
                .filter(|u| {
                    search.as_ref().is_none_or(|q| {
                        u.id.to_lowercase().contains(q)
                            || u.description
                                .as_ref()
                                .is_some_and(|d| d.to_lowercase().contains(q))
                    })
                })
                .collect();
            println!(
                "{}: {} units (showing {})",
                scd_path.display(),
                units.len(),
                shown.len().min(limit)
            );
            for u in shown.iter().take(limit) {
                println!(
                    "  {:<10} hp={:<8} mass={:<8} time={:<8} {}",
                    u.id,
                    fmt_opt(u.health),
                    fmt_opt(u.build_cost_mass),
                    fmt_opt(u.build_time),
                    u.description.as_deref().unwrap_or("")
                );
            }
            Ok(())
        }
        "unit" => {
            if args.len() < 3 {
                return Err("usage: farust unit <ID> [FILE.scd]".to_string());
            }
            let want = args[2].to_lowercase();
            let scd_path = positional_after(args, 3).unwrap_or_else(default_units_scd);
            let units = load_units(&scd_path)?;
            let unit = units
                .iter()
                .find(|u| u.id.to_lowercase() == want)
                .ok_or_else(|| format!("unit '{want}' not found in {}", scd_path.display()))?;
            println!("id:          {}", unit.id);
            println!("path:        {}", unit.path);
            println!("description: {}", unit.description.as_deref().unwrap_or("-"));
            println!("categories:  {}", unit.categories.join(", "));
            println!("health:      {}", fmt_opt(unit.health));
            println!(
                "build:       mass={} energy={} time={}",
                fmt_opt(unit.build_cost_mass),
                fmt_opt(unit.build_cost_energy),
                fmt_opt(unit.build_time)
            );
            println!(
                "speed:       {} ({})",
                fmt_opt(unit.max_speed),
                unit.motion_type.as_deref().unwrap_or("-")
            );
            println!("weapons:     {}", unit.weapons.len());
            for w in &unit.weapons {
                println!(
                    "  - {:<24} damage={} radius={} rof={}",
                    w.label.as_deref().unwrap_or("-"),
                    fmt_opt(w.damage),
                    fmt_opt(w.max_radius),
                    fmt_opt(w.rate_of_fire)
                );
            }
            Ok(())
        }
        "maps" => cmd_maps(args),
        "roster" => roster::cmd_roster(args),
        "mods" => mods::cmd_mods(args),
        "map" => cmd_map(args),
        "meshes" => cmd_meshes(args),
        "mesh" => cmd_mesh(args),
        "textures" => cmd_textures(args),
        "dds" => cmd_dds(args),
        "anim" => cmd_anim(args),
        "audio" => cmd_audio(args),
        "view-unit" => cmd_view_unit(args),
        "view-map" => cmd_view_map(args),
        "sim-run" => cmd_sim_run(args),
        "bench-multicore" => cmd_bench_multicore(args),
        "bench-scale" => cmd_bench_scale(args),
        "bench-render" => perf_cmds::cmd_bench_render(args),
        "fuzz-parsers" => perf_cmds::cmd_fuzz_parsers(args),
        "skirmish" => cmd_skirmish(args),
        "play" => cmd_play(args),
        "replay" => cmd_replay(args),
        "host" => cmd_host(args),
        "join" => cmd_join(args, connect_lobby),
        cmd => Err(format!("unknown command: {cmd}")),
    }
}

fn cmd_maps(args: &[String]) -> Result<(), String> {
    let game_dir = positional_after(args, 2).unwrap_or_else(scd::default_game_dir);
    let check = has_flag(args, "--check");
    let limit = flag_usize(args, "--limit");
    let maps = scenario::list_maps(&game_dir)?;

    if check {
        let mut failures = Vec::new();
        let mut scenarios_ok = 0usize;
        let mut saves_ok = 0usize;
        let mut scmaps_ok = 0usize;
        let mut strata_ok = 0usize;
        let mut decals_total = 0usize;

        for m in &maps {
            let meta = match scenario::parse_scenario_file(&m.scenario_path, &m.id) {
                Ok(meta) => {
                    scenarios_ok += 1;
                    meta
                }
                Err(e) => {
                    failures.push(format!("{}: scenario: {e}", m.id));
                    continue;
                }
            };
            match m.save_path.as_deref() {
                Some(save_path) => match scenario::parse_save_markers_file(save_path, &meta.armies) {
                    Ok(_) => saves_ok += 1,
                    Err(e) => failures.push(format!("{}: save: {e}", m.id)),
                },
                None => failures.push(format!("{}: missing _save.lua", m.id)),
            }
            match m.scmap_path.as_deref() {
                Some(scmap_path) => match scmap::ScMap::parse_file(scmap_path) {
                    Ok(map_data) => match dds::parse_dds(&map_data.preview_dds) {
                        Ok(_) => {
                            scmaps_ok += 1;
                            if let Some(ref tt) = map_data.terrain_textures {
                                strata_ok += 1;
                                decals_total += tt.decals.len();
                            }
                        }
                        Err(e) => failures.push(format!("{}: preview dds: {e}", m.id)),
                    },
                    Err(e) => failures.push(format!("{}: scmap: {e}", m.id)),
                },
                None => failures.push(format!("{}: missing .scmap", m.id)),
            }
        }

        if !failures.is_empty() {
            return Err(format!(
                "{} map check failure(s):\n{}",
                failures.len(),
                failures.join("\n")
            ));
        }
        println!(
            "{}: checked {} maps: {} scenarios, {} saves, {} scmaps ({} with terrain strata, {} decals, 0 errors)",
            scd::maps_dir(&game_dir).display(),
            maps.len(),
            scenarios_ok,
            saves_ok,
            scmaps_ok,
            strata_ok,
            decals_total
        );
        return Ok(());
    }

    let shown = limit.unwrap_or(maps.len());
    println!(
        "{}: {} maps (showing {})",
        scd::maps_dir(&game_dir).display(),
        maps.len(),
        shown.min(maps.len())
    );
    for m in maps.iter().take(shown) {
        let meta = scenario::parse_scenario_file(&m.scenario_path, &m.id)?;
        println!(
            "  {:<12} {:>4}x{:<4}  {:<10} armies={:<2}  {:<7}  {}",
            meta.id,
            meta.size.0,
            meta.size.1,
            meta.scenario_type,
            meta.armies.len(),
            m.source.label(),
            meta.name
        );
    }
    Ok(())
}

fn cmd_map(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust map <NAME> [GAME_DIR]".to_string());
    }
    let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
    let m = scenario::find_map(&game_dir, &args[2])?;
    let meta = scenario::parse_scenario_file(&m.scenario_path, &m.id)?;

    println!("id:          {}", meta.id);
    println!("name:        {}", meta.name);
    println!("type:        {}", meta.scenario_type);
    println!("size:        {}x{}", meta.size.0, meta.size.1);
    println!("armies:      {}", if meta.armies.is_empty() { "-".to_string() } else { meta.armies.join(", ") });
    println!("map_vfs:     {}", meta.map_vfs_path);
    println!("save_vfs:    {}", meta.save_vfs_path);

    if let Some(save_path) = &m.save_path {
        let markers = scenario::parse_save_markers_file(save_path, &meta.armies)?;
        println!(
            "save:        {} ({} markers: {} start_spots, {} mass, {} hydro)",
            save_path.display(),
            markers.total_markers,
            markers.start_spots.len(),
            markers.mass_points.len(),
            markers.hydro_points.len()
        );
        for spot in markers.start_spots.iter().take(16) {
            println!(
                "  - {:<12} [{:.1}, {:.1}, {:.1}]",
                spot.name, spot.position[0], spot.position[1], spot.position[2]
            );
        }
    } else {
        println!("save:        (missing)");
    }

    if let Some(scmap_path) = &m.scmap_path {
        let map_data = scmap::ScMap::parse_file(scmap_path)?;
        let (min_h, max_h, avg_h) = map_data.height_stats();
        println!(
            "scmap:       {} (v{}.{}, map {}x{}, heightmap {}x{}, scale={}, min/max/avg={:.2}/{:.2}/{:.2})",
            scmap_path.display(),
            map_data.version_major,
            map_data.version_minor,
            map_data.map_width,
            map_data.map_height,
            map_data.heightmap_width,
            map_data.heightmap_height,
            map_data.height_scale,
            min_h,
            max_h,
            avg_h
        );
        println!(
            "shader:      {} (cubemaps: {})",
            map_data.terrain_shader,
            map_data.cubemaps.len()
        );
        println!(
            "water:       {} (elev={:.2}, deep={:.2}, abyss={:.2})",
            map_data.water.has_water,
            map_data.water.elevation,
            map_data.water.elevation_deep,
            map_data.water.elevation_abyss
        );
        let preview = dds::parse_dds(&map_data.preview_dds)?;
        println!(
            "preview_dds: {}x{} {} mips={}",
            preview.width, preview.height, preview.format, preview.mip_count
        );
    } else {
        println!("scmap:       (missing)");
    }
    Ok(())
}

fn cmd_meshes(args: &[String]) -> Result<(), String> {
    let scd_path = positional_after(args, 2).unwrap_or_else(default_units_scd);
    let check = has_flag(args, "--check");
    let limit = flag_usize(args, "--limit").unwrap_or(20);

    if check {
        let files = scd::read_scd_matching(&scd_path, |n| ends_with_ci(n, ".scm"))
            .map_err(|e| format!("{}: {e}", scd_path.display()))?;
        let mut total_verts = 0usize;
        let mut total_tris = 0usize;
        let mut failures = Vec::new();
        for (path, bytes) in &files {
            match scm::ScmMesh::parse(bytes) {
                Ok(mesh) => {
                    total_verts += mesh.vertices.len();
                    total_tris += mesh.triangle_count();
                }
                Err(e) => failures.push(format!("{path}: {e}")),
            }
        }
        if !failures.is_empty() {
            return Err(format!(
                "{} mesh(es) failed to parse:\n{}",
                failures.len(),
                failures.join("\n")
            ));
        }
        println!(
            "{}: checked {} .scm meshes (verts={}, tris={}, 0 errors)",
            scd_path.display(),
            files.len(),
            total_verts,
            total_tris
        );
        return Ok(());
    }

    let entries = scd::list_scd(&scd_path).map_err(|e| format!("{}: {e}", scd_path.display()))?;
    let meshes: Vec<_> = entries
        .into_iter()
        .filter(|e| !e.is_dir && ends_with_ci(&e.name, ".scm"))
        .collect();
    println!(
        "{}: {} .scm meshes (showing {})",
        scd_path.display(),
        meshes.len(),
        meshes.len().min(limit)
    );
    for entry in meshes.iter().take(limit) {
        println!("  {:>10}  {}", entry.size, entry.name);
    }
    Ok(())
}

fn cmd_mesh(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust mesh <PATH> [FILE.scd]".to_string());
    }
    let (actual, bytes) = read_asset(&args[2], positional_after(args, 3))?;
    let mesh = scm::ScmMesh::parse(&bytes)?;

    println!("path:        {actual}");
    println!("version:     {}", mesh.version);
    println!("vertices:    {}", mesh.vertices.len());
    println!("triangles:   {}", mesh.triangle_count());
    println!(
        "bones:       {}/{} (weighted/total)",
        mesh.weighted_bone_count,
        mesh.bones.len()
    );
    for bone in mesh.bones.iter().take(8) {
        println!("  - {:<24} parent={}", bone.name, bone.parent_index);
    }
    if mesh.bones.len() > 8 {
        println!("  ... ({} more bones)", mesh.bones.len() - 8);
    }
    match mesh.bounds {
        Some(b) => println!(
            "bounds:      min=[{:.3}, {:.3}, {:.3}] max=[{:.3}, {:.3}, {:.3}]",
            b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]
        ),
        None => println!("bounds:      (empty)"),
    }
    if !mesh.info.is_empty() {
        println!("info:        {}", mesh.info.join(" | "));
    }
    Ok(())
}

fn cmd_textures(args: &[String]) -> Result<(), String> {
    let scd_path = positional_after(args, 2).unwrap_or_else(default_units_scd);
    let check = has_flag(args, "--check");
    let limit = flag_usize(args, "--limit").unwrap_or(20);

    if check {
        let files = scd::read_scd_matching(&scd_path, |n| ends_with_ci(n, ".dds"))
            .map_err(|e| format!("{}: {e}", scd_path.display()))?;
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut failures = Vec::new();
        for (path, bytes) in &files {
            match dds::parse_dds(bytes) {
                Ok(hdr) => {
                    *counts.entry(hdr.format.to_string()).or_insert(0) += 1;
                }
                Err(e) => failures.push(format!("{path}: {e}")),
            }
        }
        if !failures.is_empty() {
            return Err(format!(
                "{} texture(s) failed to parse:\n{}",
                failures.len(),
                failures.join("\n")
            ));
        }
        let summary: Vec<String> = counts
            .into_iter()
            .map(|(fmt, n)| format!("{fmt}={n}"))
            .collect();
        println!(
            "{}: checked {} .dds textures ({}, 0 errors)",
            scd_path.display(),
            files.len(),
            summary.join(", ")
        );
        return Ok(());
    }

    let entries = scd::list_scd(&scd_path).map_err(|e| format!("{}: {e}", scd_path.display()))?;
    let textures: Vec<_> = entries
        .into_iter()
        .filter(|e| !e.is_dir && ends_with_ci(&e.name, ".dds"))
        .collect();
    println!(
        "{}: {} .dds textures (showing {})",
        scd_path.display(),
        textures.len(),
        textures.len().min(limit)
    );
    for entry in textures.iter().take(limit) {
        println!("  {:>10}  {}", entry.size, entry.name);
    }
    Ok(())
}

fn cmd_dds(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust dds <PATH> [FILE.scd]".to_string());
    }
    let (actual, bytes) = read_asset(&args[2], positional_after(args, 3))?;
    let hdr = dds::parse_dds(&bytes)?;

    println!("path:        {actual}");
    println!("dimensions:  {}x{}", hdr.width, hdr.height);
    println!("format:      {}", hdr.format);
    println!("mips:        {}", hdr.mip_count);
    println!("depth:       {}", hdr.depth);
    println!(
        "flags:       cubemap={} volume={}",
        hdr.is_cubemap, hdr.is_volume
    );
    println!(
        "data_bytes:  {} (offset {})",
        hdr.data_len, hdr.data_offset
    );
    Ok(())
}

fn cmd_anim(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust anim <list|UNIT_ID> [GAME_DIR] [--limit N] [--check]".to_string());
    }
    let target = args[2].as_str();
    let check = has_flag(args, "--check");
    let limit = flag_usize(args, "--limit").unwrap_or(25);
    let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
    let units_scd = scd::gamedata_dir(&game_dir).join("units.scd");

    let scas = scd::read_scd_matching(&units_scd, |n| ends_with_ci(n, ".sca"))
        .map_err(|e| format!("{}: {e}", units_scd.display()))?;

    if target.eq_ignore_ascii_case("list") {
        if check {
            let mut failures = Vec::new();
            let mut total_frames = 0usize;
            let mut total_bones = 0usize;
            for (path, bytes) in &scas {
                match sca::ScaAnimation::parse(bytes) {
                    Ok(anim) => {
                        total_frames += anim.frame_count();
                        total_bones += anim.bone_count();
                    }
                    Err(e) => failures.push(format!("{path}: {e}")),
                }
            }
            if !failures.is_empty() {
                return Err(format!(
                    "{} .sca animation failure(s):\n{}",
                    failures.len(),
                    failures.join("\n")
                ));
            }
            println!(
                "{}: checked {} .sca animations ({} frames, {} bone tracks, 0 errors)",
                units_scd.display(),
                scas.len(),
                total_frames,
                total_bones
            );
            return Ok(());
        }
        println!(
            "{}: {} .sca animations (showing {})",
            units_scd.display(),
            scas.len(),
            scas.len().min(limit)
        );
        for (path, bytes) in scas.iter().take(limit) {
            if let Ok(a) = sca::ScaAnimation::parse(bytes) {
                println!(
                    "  {:<54} frames={:<4} bones={:<3} dur={:.2}s",
                    path,
                    a.frame_count(),
                    a.bone_count(),
                    a.duration
                );
            }
        }
        return Ok(());
    }

    let needle = target.to_ascii_lowercase();
    let matched: Vec<_> = scas
        .into_iter()
        .filter(|(p, _)| p.to_ascii_lowercase().contains(&needle))
        .collect();
    if matched.is_empty() {
        return Err(format!("no .sca animations matching '{target}' in {}", units_scd.display()));
    }
    println!("animations matching '{target}': {}", matched.len());
    for (path, bytes) in matched.iter().take(limit) {
        let a = sca::ScaAnimation::parse(bytes)?;
        println!(
            "  {} | frames={} bones={} duration={:.2}s | sample_bones={}",
            path,
            a.frame_count(),
            a.bone_count(),
            a.duration,
            a.bone_names.iter().take(6).cloned().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}

fn collect_sound_files(sounds_dir: &Path, ext: &str) -> Result<Vec<PathBuf>, String> {
    let rd = std::fs::read_dir(sounds_dir)
        .map_err(|e| format!("cannot read {}: {e}", sounds_dir.display()))?;
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case(ext))
        })
        .collect();
    out.sort();
    Ok(out)
}

fn cmd_audio(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust audio <list|info|export> [TARGET] [GAME_DIR] [--check] [--out DIR]".to_string());
    }
    let sub = args[2].as_str();
    if matches!(sub, "render-mix" | "coverage" | "cues" | "play-cue") {
        return cmd_audio_ext(args);
    }
    let has_target = !matches!(sub, "list")
        && args.get(3).is_some_and(|a| !a.starts_with("--"));
    let game_dir = positional_after(args, if has_target { 4 } else { 3 })
        .unwrap_or_else(scd::default_game_dir);
    let sounds_dir = game_dir.join("sounds");

    match sub {
        "list" => {
            let check = has_flag(args, "--check");
            let xwbs = collect_sound_files(&sounds_dir, "xwb")?;
            let xsbs = collect_sound_files(&sounds_dir, "xsb")?;
            let mut total_waves = 0usize;
            let mut total_cues = 0usize;
            let mut failures = Vec::new();
            for p in &xwbs {
                match std::fs::read(p).map_err(|e| format!("{e}")).and_then(xact::WaveBank::parse) {
                    Ok(wb) => {
                        total_waves += wb.entries.len();
                        if check {
                            for (i, e) in wb.entries.iter().enumerate() {
                                if e.format_tag == 0 && wb.decode_mono_f32(i).is_none() {
                                    failures.push(format!("{}#{i}: PCM decode failed", p.display()));
                                }
                            }
                        } else {
                            println!(
                                "  XWB {:<24} waves={:<4} v{}",
                                p.file_name().unwrap_or_default().to_string_lossy(),
                                wb.entries.len(),
                                wb.version
                            );
                        }
                    }
                    Err(e) => failures.push(format!("{}: {e}", p.display())),
                }
            }
            for p in &xsbs {
                match std::fs::read(p)
                    .map_err(|e| format!("{e}"))
                    .and_then(|b| xact::SoundBank::parse(&b))
                {
                    Ok(sb) => {
                        total_cues += sb.cues.len();
                        if !check {
                            println!(
                                "  XSB {:<24} cues={:<4} wbs={}",
                                p.file_name().unwrap_or_default().to_string_lossy(),
                                sb.cues.len(),
                                sb.wavebank_names.join(",")
                            );
                        }
                    }
                    Err(e) => failures.push(format!("{}: {e}", p.display())),
                }
            }
            if !failures.is_empty() {
                return Err(format!(
                    "{} audio bank failure(s):\n{}",
                    failures.len(),
                    failures.join("\n")
                ));
            }
            println!(
                "{}: checked {} .xwb wavebanks ({total_waves} waves) and {} .xsb soundbanks ({total_cues} cues), 0 errors",
                sounds_dir.display(),
                xwbs.len(),
                xsbs.len()
            );
            Ok(())
        }
        "info" => {
            let target = args
                .get(3)
                .ok_or_else(|| "usage: farust audio info <BANK_FILE> [GAME_DIR]".to_string())?;
            let path = sounds_dir.join(target);
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            if ends_with_ci(target, ".xsb") {
                let sb = xact::SoundBank::parse(&bytes)?;
                println!("soundbank: {} | cues={} | wavebanks={:?}", sb.name, sb.cues.len(), sb.wavebank_names);
                for cue in sb.cues.iter().take(if has_flag(args, "--all") { usize::MAX } else { 25 }) {
                    println!("  {:<32} {:?}", cue.name, cue.waves);
                }
            } else {
                let wb = xact::WaveBank::parse(bytes)?;
                println!("wavebank: {} (v{}, compact={}) | waves={}", wb.name, wb.version, wb.compact, wb.entries.len());
                for e in wb.entries.iter().take(if has_flag(args, "--all") { usize::MAX } else { 25 }) {
                    println!(
                        "  #{:<3} {:<24} {}Hz {}ch {}bit bytes={}",
                        e.index,
                        e.name.as_deref().unwrap_or("-"),
                        e.sample_rate,
                        e.channels,
                        e.bits_per_sample,
                        e.data_length
                    );
                }
            }
            Ok(())
        }
        "export" => {
            let target = args
                .get(3)
                .ok_or_else(|| "usage: farust audio export <BANK.xwb> [GAME_DIR] [--out DIR]".to_string())?;
            let out_dir = PathBuf::from(flag_str(args, "--out").unwrap_or("/tmp/farust_wav"));
            std::fs::create_dir_all(&out_dir)
                .map_err(|e| format!("create_dir {}: {e}", out_dir.display()))?;
            let path = sounds_dir.join(target);
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            let wb = xact::WaveBank::parse(bytes)?;
            let mut exported = 0usize;
            for i in 0..wb.entries.len() {
                if let Some(wav) = wb.to_wav_bytes(i) {
                    let raw_name = wb.entries[i]
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("wave_{i:04}"));
                    let safe: String = raw_name
                        .chars()
                        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
                        .collect();
                    let out_file = out_dir.join(format!("{i:04}_{safe}.wav"));
                    std::fs::write(&out_file, wav)
                        .map_err(|e| format!("write {}: {e}", out_file.display()))?;
                    exported += 1;
                }
            }
            println!(
                "exported {exported}/{} WAV file(s) from {} -> {}",
                wb.entries.len(),
                path.display(),
                out_dir.display()
            );
            Ok(())
        }
        other => Err(format!(
            "unknown audio subcommand '{other}' (expected list|info|export|coverage|cues|render-mix)"
        )),
    }
}

/// Phase 14 audio tooling: decode coverage, cue mapping, headless skirmish mixdown.
fn cmd_audio_ext(args: &[String]) -> Result<(), String> {
    match args[2].as_str() {
        "coverage" => {
            let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
            audio_play::print_decode_coverage(&game_dir.join("sounds"))
        }
        "play-cue" => {
            let cue = args.get(3).ok_or_else(|| "usage: farust audio play-cue <CUE> [GAME_DIR] [--mute]".to_string())?;
            let game_dir = positional_after(args, 4).unwrap_or_else(scd::default_game_dir);
            audio_play::play_cue_realtime(&game_dir, cue, has_flag(args, "--mute"))
        }
        "cues" => {
            let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
            let catalog = skirmish::Catalog::load(&game_dir).ok();
            audio_play::print_cue_mapping(&game_dir, catalog.as_ref());
            Ok(())
        }
        _ => {
            let out = args
                .get(3)
                .filter(|a| !a.starts_with("--"))
                .map(PathBuf::from)
                .ok_or_else(|| "usage: farust audio render-mix <OUT.wav> [MAP] [GAME_DIR] [--ticks N]".to_string())?;
            let map_name = args
                .get(4)
                .filter(|a| !a.starts_with("--"))
                .map(String::as_str)
                .unwrap_or("SCMP_001");
            // `--from N` is local to render-mix; hide it from the shared positional scan.
            let pos_args: Vec<String> = args
                .iter()
                .enumerate()
                .filter(|(i, a)| *a != "--from" && (*i == 0 || args[i - 1] != "--from"))
                .map(|(_, a)| a.clone())
                .collect();
            let game_dir = positional_after(&pos_args, 5).unwrap_or_else(scd::default_game_dir);
            let seed = flag_str(args, "--seed")
                .map(|s| s.parse::<u64>().map_err(|_| format!("invalid --seed '{s}'")))
                .transpose()?
                .unwrap_or(sim::DEFAULT_SEED);
            let ticks = flag_usize(args, "--ticks").unwrap_or(3000) as u64;
            let from = flag_usize(args, "--from").unwrap_or(0) as u64;
            let catalog = skirmish::Catalog::load(&game_dir)?;
            let map = skirmish::MapContext::load(&game_dir, map_name)?;
            let setup = parse_skirmish_setup(args, &catalog, map.starts.len())?;
            let world = skirmish::Skirmish::on_map(catalog, &map, &setup, seed)?;
            let session = play::PlaySession::new(world, 0, true, 1);
            audio_play::render_mix(&game_dir, session, &map.scmap, from, ticks, &out)
        }
    }
}

fn cmd_view_unit(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust view-unit <ID> [GAME_DIR] [--size WxH] [--snapshot OUT.png]".to_string());
    }
    let unit_id = &args[2];
    // `--anim walk|idle|<sca-substring>` + `--anim-time SECS` pose the mesh (Phase 11).
    let anim = flag_str(args, "--anim").map(str::to_owned);
    let anim_time = flag_str(args, "--anim-time")
        .map(|s| s.parse::<f32>().map_err(|_| format!("invalid --anim-time '{s}'")))
        .transpose()?
        .unwrap_or(0.0);
    let rest: Vec<String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| {
            let prev = i.checked_sub(1).map(|p| args[p].as_str());
            *a != "--anim" && *a != "--anim-time" && prev != Some("--anim") && prev != Some("--anim-time")
        })
        .map(|(_, a)| a.clone())
        .collect();
    let game_dir = positional_after(&rest, 3).unwrap_or_else(scd::default_game_dir);
    let (width, height) = parse_size_flag(args).unwrap_or((1280, 800));
    let aspect = width as f32 / height as f32;
    let scene = scene::load_unit_scene_posed(&game_dir, unit_id, aspect, anim.as_deref(), anim_time)?;
    run_or_snapshot(scene, width, height, args)
}

fn cmd_view_map(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust view-map <NAME> [GAME_DIR] [--size WxH] [--snapshot OUT.png]".to_string());
    }
    let map_name = &args[2];
    let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
    let (width, height) = parse_size_flag(args).unwrap_or((1280, 800));
    let aspect = width as f32 / height as f32;
    let scene = scene::load_map_scene(&game_dir, map_name, aspect)?;
    run_or_snapshot(scene, width, height, args)
}

fn cmd_sim_run(args: &[String]) -> Result<(), String> {
    let ticks = flag_usize(args, "--ticks")
        .ok_or_else(|| "usage: farust sim-run --ticks N [--seed S] [--scenario NAME] [--dt SEC] [--hash] [--json]".to_string())?;
    if ticks == 0 {
        return Err("usage: farust sim-run --ticks N [--seed S] [--scenario NAME] [--dt SEC] [--hash] [--json]".to_string());
    }
    if ticks as u64 > sim::MAX_TICKS {
        return Err(format!(
            "--ticks {ticks} exceeds cap {} (fixed-step headless run)",
            sim::MAX_TICKS
        ));
    }
    let seed = flag_str(args, "--seed")
        .map(|s| {
            s.parse::<u64>()
                .map_err(|_| format!("invalid --seed '{s}'"))
        })
        .transpose()?
        .unwrap_or(sim::DEFAULT_SEED);
    let scenario = flag_str(args, "--scenario").unwrap_or("selftest");
    if scenario != "selftest" {
        return Err(format!(
            "unknown scenario '{scenario}' (only 'selftest' is supported)"
        ));
    }
    if let Some(dt) = flag_str(args, "--dt") {
        let dt_secs: f64 = dt
            .parse()
            .map_err(|_| format!("invalid --dt '{dt}'"))?;
        let dt_micros = (dt_secs * 1_000_000.0).round() as u64;
        if dt_micros != sim::DT_MICROS {
            return Err(format!(
                "unsupported --dt {dt} (sim is fixed-step at 0.1s)"
            ));
        }
    }
    let (_world, report) = sim::run_selftest(ticks as u64, seed);
    let hash_hex = format!("{:016x}", report.hash);
    if has_flag(args, "--json") {
        println!(
            "{{\"ticks\":{},\"seed\":{},\"hash\":\"{hash_hex}\",\"units_alive\":{},\"shots_fired\":{},\"kills\":{}}}",
            report.ticks, report.seed, report.units_alive, report.shots_fired, report.kills
        );
        return Ok(());
    }
    if has_flag(args, "--hash") {
        println!("{hash_hex}");
        return Ok(());
    }
    println!("scenario:    selftest");
    println!("ticks:       {}", report.ticks);
    println!("seed:        {seed}");
    println!("hash:        {hash_hex}");
    println!("units_alive: {}", report.units_alive);
    println!("shots_fired: {}", report.shots_fired);
    println!("kills:       {}", report.kills);
    Ok(())
}

fn cmd_bench_multicore(args: &[String]) -> Result<(), String> {
    let game_dir = positional_after(args, 2).unwrap_or_else(scd::default_game_dir);
    let threads = flag_usize(args, "--threads")
        .unwrap_or_else(jobs::default_thread_count)
        .max(2);
    let asset_limit = flag_usize(args, "--limit").unwrap_or(96);

    // 1. Pre-load raw asset buffers from units.scd (so we measure CPU decode, not disk seek)
    let units_scd = scd::gamedata_dir(&game_dir).join("units.scd");
    let all_assets = scd::read_scd_matching(&units_scd, |n| {
        ends_with_ci(n, "_unit.bp") || ends_with_ci(n, "_lod0.scm") || ends_with_ci(n, "_albedo.dds")
    })
    .map_err(|e| format!("{}: {e}", units_scd.display()))?;
    let asset_slice: Vec<(String, Vec<u8>)> = all_assets.into_iter().take(asset_limit).collect();

    let t0 = Instant::now();
    let asset_1t = jobs::decode_assets_batch(&asset_slice, 1)?;
    let asset_1t_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let asset_nt = jobs::decode_assets_batch(&asset_slice, threads)?;
    let asset_nt_ms = t1.elapsed().as_secs_f64() * 1000.0;

    if asset_1t.digest_fnv1a != asset_nt.digest_fnv1a {
        return Err(format!(
            "asset decode digest mismatch: 1t={:016x} vs {threads}t={:016x}",
            asset_1t.digest_fnv1a, asset_nt.digest_fnv1a
        ));
    }

    // 2. Pathfinding batch benchmark on SCMP_001 heightmap grid
    let map_files = scenario::find_map(&game_dir, "SCMP_001")?;
    let scmap_path = map_files
        .scmap_path
        .as_deref()
        .ok_or_else(|| "SCMP_001 missing .scmap".to_string())?;
    let scmap = scmap::ScMap::parse_file(scmap_path)?;
    let grid = pathfinding::PathGrid::from_scmap(&scmap, 2, 2500, true);
    let map_w_milli = (scmap.map_width as i64) * sim::SCALE;
    let map_h_milli = (scmap.map_height as i64) * sim::SCALE;

    let mut path_requests = Vec::with_capacity(64);
    for i in 0..64u64 {
        let sx = 64 * sim::SCALE + ((i as i64 * 13) % 200) * sim::SCALE;
        let sz = 64 * sim::SCALE + ((i as i64 * 29) % 880) * sim::SCALE;
        let gx = (map_w_milli - 64 * sim::SCALE) - ((i as i64 * 17) % 200) * sim::SCALE;
        let gz = (map_h_milli - 64 * sim::SCALE) - ((i as i64 * 31) % 880) * sim::SCALE;
        path_requests.push(pathfinding::PathRequest {
            id: i,
            start: (sx, sz),
            goal: (gx, gz),
        });
    }

    let t2 = Instant::now();
    let paths_1t = pathfinding::solve_paths_batch(&grid, &path_requests, 1);
    let path_1t_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let path_hash_1t = pathfinding::path_batch_hash(&paths_1t);

    let t3 = Instant::now();
    let paths_nt = pathfinding::solve_paths_batch(&grid, &path_requests, threads);
    let path_nt_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let path_hash_nt = pathfinding::path_batch_hash(&paths_nt);

    if path_hash_1t != path_hash_nt {
        return Err(format!(
            "pathfinding batch hash mismatch: 1t={path_hash_1t:016x} vs {threads}t={path_hash_nt:016x}"
        ));
    }

    // 3. Render preparation benchmark (16,384 live sim units over 25 frames, tactical camera)
    let mut bench_world = sim::World::new(sim::DEFAULT_SEED);
    let dummy_params = sim::UnitParams {
        max_hp: 1000 * sim::SCALE,
        speed_per_tick: 200,
        weapons: Vec::new(),
        cost_mass: 100,
        cost_energy: 100,
        build_ticks: 10,
    };
    for i in 0..16_384u32 {
        let x = ((i % 128) as i64 * 8 + 8) * sim::SCALE;
        let y = ((i / 128) as i64 * 8 + 8) * sim::SCALE;
        let uid = bench_world.spawn(&sim::UnitSpawn::new("UAL0001", x, y), &dummy_params);
        if (i as usize) < paths_nt.len() && paths_nt[i as usize].found {
            bench_world.apply_path_waypoints(uid, &paths_nt[i as usize].waypoints);
        }
    }
    let snapshots = std::sync::Arc::new(bench_world.snapshot_units());
    let scmap_arc = std::sync::Arc::new(scmap);
    let mut cam = camera::OrbitCamera::for_map(scmap_arc.map_width, scmap_arc.map_height, 17.5, 1.6);
    cam.distance = 220.0;
    cam.pitch = 0.72;
    let vp = cam.view_proj_matrix();
    let eye = cam.eye_position();
    let frames = 25usize;

    let t4 = Instant::now();
    let mut frame_1t = render_prep::prepare_render_frame(&snapshots, Some(&scmap_arc), vp, eye, 1);
    for _ in 1..frames {
        frame_1t = render_prep::prepare_render_frame(&snapshots, Some(&scmap_arc), vp, eye, 1);
    }
    let render_1t_ms = t4.elapsed().as_secs_f64() * 1000.0 / frames as f64;

    let pool = jobs::WorkerPool::new(threads);
    let t5 = Instant::now();
    let mut frame_nt =
        render_prep::prepare_render_frame_pool(&snapshots, Some(&scmap_arc), vp, eye, &pool);
    for _ in 1..frames {
        frame_nt =
            render_prep::prepare_render_frame_pool(&snapshots, Some(&scmap_arc), vp, eye, &pool);
    }
    let render_nt_ms = t5.elapsed().as_secs_f64() * 1000.0 / frames as f64;

    if frame_1t.digest_fnv1a != frame_nt.digest_fnv1a {
        return Err(format!(
            "render prep digest mismatch: 1t={:016x} vs {threads}t={:016x}",
            frame_1t.digest_fnv1a, frame_nt.digest_fnv1a
        ));
    }

    // 4. Single-threaded deterministic sim gate (must remain identical to Phase 4 golden hash)
    let (_, sim_report) = sim::run_selftest(10_000, sim::DEFAULT_SEED);
    let sim_hash_hex = format!("{:016x}", sim_report.hash);
    let expected_golden: u64 = 0x35db8f2bd7a6faf2;
    if sim_report.hash != expected_golden {
        return Err(format!(
            "sim hash drifted: got {sim_hash_hex}, expected {expected_golden:016x}"
        ));
    }

    let asset_speedup = asset_1t_ms / asset_nt_ms.max(0.001);
    let path_speedup = path_1t_ms / path_nt_ms.max(0.001);
    let render_speedup = render_1t_ms / render_nt_ms.max(0.001);

    if has_flag(args, "--json") {
        println!(
            "{{\"threads\":{threads},\"assets\":{{\"items\":{},\"ms_1t\":{asset_1t_ms:.2},\"ms_nt\":{asset_nt_ms:.2},\"speedup\":{asset_speedup:.2},\"digest\":\"{:016x}\"}},\"pathfinding\":{{\"requests\":{},\"ms_1t\":{path_1t_ms:.2},\"ms_nt\":{path_nt_ms:.2},\"speedup\":{path_speedup:.2},\"hash\":\"{path_hash_nt:016x}\"}},\"render_prep\":{{\"units\":{},\"frame_ms_1t\":{render_1t_ms:.3},\"frame_ms_nt\":{render_nt_ms:.3},\"speedup\":{render_speedup:.2},\"digest\":\"{:016x}\"}},\"sim_hash\":\"{sim_hash_hex}\"}}",
            asset_nt.total_items,
            asset_nt.digest_fnv1a,
            path_requests.len(),
            snapshots.len(),
            frame_nt.digest_fnv1a
        );
        return Ok(());
    }

    println!("threads:        1 vs {threads}");
    println!(
        "asset_decode:   {} items (bp={}, scm={}, dds={}, texels={}) | 1t={asset_1t_ms:.2}ms -> {threads}t={asset_nt_ms:.2}ms ({asset_speedup:.2}x) | digest={:016x}",
        asset_nt.total_items,
        asset_nt.blueprints_parsed,
        asset_nt.meshes_parsed,
        asset_nt.textures_decoded,
        asset_nt.total_texels,
        asset_nt.digest_fnv1a
    );
    println!(
        "pathfinding:    {} requests on {}x{} grid (blocked={}) | 1t={path_1t_ms:.2}ms -> {threads}t={path_nt_ms:.2}ms ({path_speedup:.2}x) | hash={path_hash_nt:016x}",
        path_requests.len(),
        grid.width,
        grid.height,
        grid.blocked_count()
    );
    println!(
        "render_prep:    {} units (visible={}, culled={}, lod={:?}) | 1t={render_1t_ms:.3}ms/frame -> {threads}t={render_nt_ms:.3}ms/frame ({render_speedup:.2}x) | digest={:016x}",
        snapshots.len(),
        frame_nt.instances.len(),
        frame_nt.culled_count,
        frame_nt.lod_counts,
        frame_nt.digest_fnv1a
    );
    println!(
        "sim_10k_hash:   {sim_hash_hex} (matches Phase 4 golden 35db8f2bd7a6faf2, alive={}, shots={}, kills={})",
        sim_report.units_alive, sim_report.shots_fired, sim_report.kills
    );
    Ok(())
}

fn run_or_snapshot(
    scene: scene::LoadedScene,
    width: u32,
    height: u32,
    args: &[String],
) -> Result<(), String> {
    if let Some(out) = flag_str(args, "--snapshot") {
        let out_path = PathBuf::from(out);
        let gpu = viewer::render_scene_to_png(&scene, width, height, &out_path)?;
        println!(
            "rendered '{}' ({} verts, {} tris, tex {}x{}) on {} -> {}",
            scene.title,
            scene.vertices.len(),
            scene.indices.len() / 3,
            scene.albedo.width,
            scene.albedo.height,
            gpu,
            out_path.display()
        );
        Ok(())
    } else {
        println!(
            "opening viewer '{}' ({} verts, {} tris) — LMB orbit, RMB/WASD pan, wheel zoom, R reset, Esc quit",
            scene.title,
            scene.vertices.len(),
            scene.indices.len() / 3
        );
        viewer::run_window(scene, width, height)
    }
}

fn parse_size_flag(args: &[String]) -> Option<(u32, u32)> {
    let raw = flag_str(args, "--size")?;
    let (w, h) = raw.split_once(['x', 'X'])?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

fn read_asset(path_arg: &str, scd_override: Option<PathBuf>) -> Result<(String, Vec<u8>), String> {
    let disk_path = Path::new(path_arg);
    if disk_path.is_file() {
        let bytes = std::fs::read(disk_path)
            .map_err(|e| format!("failed to read {}: {e}", disk_path.display()))?;
        return Ok((path_arg.to_string(), bytes));
    }
    let scd_path = scd_override.unwrap_or_else(default_units_scd);
    scd::read_scd_file_ci(&scd_path, path_arg)
        .map_err(|e| format!("{}: {e}", scd_path.display()))
}

fn positional_after(args: &[String], start: usize) -> Option<PathBuf> {
    let mut i = start;
    while i < args.len() {
        let a = &args[i];
        if a == "--limit"
            || a == "--search"
            || a == "--size"
            || a == "--snapshot"
            || a == "--ticks"
            || a == "--seed"
            || a == "--seeds"
            || a == "--scenario"
            || a == "--dt"
            || a == "--threads"
            || a == "--armies"
            || a == "--factions"
            || a == "--ai"
            || a == "--victory"
            || a == "--unit-cap"
            || a == "--max-ticks"
            || a == "--speed"
            || a == "--snapshot-ticks"
            || a == "--record"
            || a == "--load"
            || a == "--screen"
            || a == "--map"
            || a == "--maps"
            || a == "--matchups"
            || a == "--units"
            || a == "--headless-ticks"
            || a == "--players"
            || a == "--camera-dist"
            || a == "--out"
            || a == "--rendezvous"
        {
            i += 2;
            continue;
        }
        if a.starts_with("--") {
            i += 1;
            continue;
        }
        return Some(PathBuf::from(a));
    }
    None
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn flag_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn flag_usize(args: &[String], flag: &str) -> Option<usize> {
    flag_str(args, flag).and_then(|s| s.parse::<usize>().ok())
}

fn ends_with_ci(s: &str, suffix: &str) -> bool {
    let sb = s.as_bytes();
    let tb = suffix.as_bytes();
    sb.len() >= tb.len() && sb[sb.len() - tb.len()..].eq_ignore_ascii_case(tb)
}

fn default_units_scd() -> PathBuf {
    scd::gamedata_dir(&scd::default_game_dir()).join("units.scd")
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(n) if n.fract() == 0.0 => format!("{}", n as i64),
        Some(n) => format!("{n}"),
        None => "-".to_string(),
    }
}

fn load_units(scd_path: &PathBuf) -> Result<Vec<blueprint::UnitSummary>, String> {
    let files = scd::read_scd_matching(scd_path, |n| n.ends_with("_unit.bp"))
        .map_err(|e| format!("{}: {e}", scd_path.display()))?;
    let mut out = Vec::with_capacity(files.len());
    let mut failures = Vec::new();
    for (path, bytes) in &files {
        let text = String::from_utf8_lossy(bytes);
        match blueprint::parse_blueprint(&text) {
            Ok(root) => out.push(blueprint::summarize(
                &blueprint::unit_id_from_path(path),
                path,
                &root,
            )),
            Err(e) => failures.push(format!("{path}: {e}")),
        }
    }
    if !failures.is_empty() {
        return Err(format!(
            "{} blueprint(s) failed to parse:\n{}",
            failures.len(),
            failures.join("\n")
        ));
    }
    Ok(out)
}

/// Parse `--armies 1,2` (1-based start spots) and `--factions uef,cybran`.
fn parse_skirmish_setup(
    args: &[String],
    catalog: &skirmish::Catalog,
    spots: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let armies: Vec<usize> = flag_str(args, "--armies")
        .unwrap_or("1,2")
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .ok()
                .filter(|n| *n >= 1 && *n <= spots)
                .map(|n| n - 1)
                .ok_or_else(|| format!("invalid army spot '{s}' (1..={spots})"))
        })
        .collect::<Result<_, _>>()?;
    let factions: Vec<&str> = flag_str(args, "--factions")
        .unwrap_or("uef,cybran")
        .split(',')
        .map(str::trim)
        .collect();
    armies
        .iter()
        .enumerate()
        .map(|(i, &spot)| {
            let key = factions.get(i).copied().unwrap_or(factions[i % factions.len()]);
            let f = catalog
                .faction_index(key)
                .ok_or_else(|| format!("unknown faction '{key}' (uef, cybran, aeon, seraphim)"))?;
            Ok((spot, f))
        })
        .collect()
}

fn parse_game_options(args: &[String]) -> Result<skirmish::GameOptions, String> {
    let mut opts = skirmish::GameOptions::default();
    if let Some(v) = flag_str(args, "--victory") {
        opts.victory = skirmish::VictoryCondition::parse(v).ok_or_else(|| {
            format!("unknown victory condition '{v}' (assassination, supremacy, annihilation, sandbox)")
        })?;
    }
    if let Some(cap_s) = flag_str(args, "--unit-cap") {
        opts.unit_cap = cap_s
            .parse::<u32>()
            .map_err(|_| format!("invalid --unit-cap '{cap_s}'"))?;
    }
    Ok(opts)
}

fn parse_ai_diffs(args: &[String], armies: usize) -> Result<Vec<skirmish_ai::Difficulty>, String> {
    match flag_str(args, "--ai") {
        Some(s) => skirmish_ai::parse_difficulty_list(s, armies),
        None => Ok(vec![skirmish_ai::Difficulty::Normal; armies]),
    }
}

fn cmd_skirmish(args: &[String]) -> Result<(), String> {
    let map_arg = args.get(2).filter(|a| !a.starts_with("--"));
    let map_name = map_arg.map(String::as_str).unwrap_or("SCMP_001");
    let game_dir = positional_after(args, if map_arg.is_some() { 3 } else { 2 }).unwrap_or_else(scd::default_game_dir);
    let seed = flag_str(args, "--seed")
        .map(|s| s.parse::<u64>().map_err(|_| format!("invalid --seed '{s}'")))
        .transpose()?
        .unwrap_or(sim::DEFAULT_SEED);
    let max_ticks = flag_usize(args, "--max-ticks").map(|n| n as u64).unwrap_or(skirmish::DEFAULT_MAX_TICKS);
    let quiet = has_flag(args, "--quiet") || has_flag(args, "--json");
    let options = parse_game_options(args)?;

    let t0 = Instant::now();
    let catalog = skirmish::Catalog::load(&game_dir)?;
    let map = skirmish::MapContext::load(&game_dir, map_name)?;
    let setup = parse_skirmish_setup(args, &catalog, map.starts.len())?;
    let diffs = parse_ai_diffs(args, setup.len())?;
    let mut world = skirmish::Skirmish::on_map(catalog, &map, &setup, seed)?.with_options(options);
    let record_out = flag_str(args, "--record").map(PathBuf::from);
    let mut recorder = record_out.as_ref().map(|_| {
        let setups: Vec<skirmish::ArmySetup> = world
            .armies()
            .iter()
            .map(|a| skirmish::ArmySetup {
                name: a.name.clone(),
                faction: a.faction,
                start: a.start,
                team: a.team,
            })
            .collect();
        let mut rec = replay::ReplayRecorder::new(&map.map_id, seed, setups);
        rec.set_options(options);
        rec
    });
    let mut ais: Vec<skirmish_ai::ScriptedAi> = (0..setup.len())
        .map(|i| skirmish_ai::ScriptedAi::with_difficulty(i as skirmish::ArmyId, diffs[i]))
        .collect();
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if !quiet {
        println!(
            "map:      {} ({}) {}x{} | {} land mass points | grid {}x{} blocked={}",
            map.map_name,
            map.map_id,
            map.size.0 / sim::SCALE,
            map.size.1 / sim::SCALE,
            map.mass_points.len(),
            map.grid.width,
            map.grid.height,
            map.grid.blocked_count()
        );
        for a in world.armies() {
            let f = &world.catalog().factions[a.faction];
            let acu = world.catalog().get(&f.commander).unwrap();
            println!(
                "army {}:   {} {:<8} start=({}, {}) commander={} hp={} | factory={} units={}",
                a.id,
                a.name,
                f.key,
                a.start.0 / sim::SCALE,
                a.start.1 / sim::SCALE,
                acu.id,
                acu.max_hp / sim::SCALE,
                f.factory,
                f.land_units.join("/")
            );
        }
    }

    let t1 = Instant::now();
    // Phase 24: units built per army and tech level (sampled every 10 ticks).
    let mut tier_built: Vec<[u32; 4]> = vec![[0; 4]; world.armies().len()];
    let mut top_built: BTreeMap<String, u32> = BTreeMap::new();
    let mut last_seen: skirmish::UnitId = world.units().map(|u| u.id).max().unwrap_or(0);
    while world.tick() < max_ticks && world.result().is_none() {
        let tick = world.tick();
        if tick % 10 == 0 {
            let mut hi = last_seen;
            for u in world.units().filter(|u| u.id > last_seen) {
                hi = hi.max(u.id);
                tier_built[u.army as usize][(u.tier.clamp(1, 4) - 1) as usize] += 1;
                if u.tier >= 3 {
                    *top_built.entry(u.type_id.clone()).or_insert(0) += 1;
                }
            }
            last_seen = hi;
        }
        for ai in &mut ais {
            for cmd in ai.decide(&world) {
                if let Some(ref mut rec) = recorder {
                    rec.record(tick, ai.army, &cmd);
                }
                world.issue(ai.army, cmd);
            }
        }
        world.step();
        if let Some(ref mut rec) = recorder {
            rec.maybe_checkpoint(world.tick(), world.state_hash());
        }
        if !quiet && world.tick() % 1200 == 0 {
            print_skirmish_status(&world);
        }
    }
    if let (Some(rec), Some(ref path)) = (recorder, record_out) {
        let rep = rec.finish(world.tick(), world.state_hash());
        rep.save(path)?;
        if !quiet {
            println!(
                "replay:   saved {} ({} cmds, {} checkpoints)",
                path.display(),
                rep.commands.len(),
                rep.checkpoints.len()
            );
        }
    }
    let sim_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let result = match world.result() {
        Some(skirmish::GameResult::Victory(a)) => format!("victory: army {} ({})", a, world.armies()[a as usize].name),
        Some(skirmish::GameResult::Draw) => "draw".to_string(),
        None => "time limit".to_string(),
    };
    let built: u32 = world.armies().iter().map(|a| a.units_built).sum();
    if has_flag(args, "--json") {
        println!(
            "{{\"map\":\"{}\",\"seed\":{},\"ticks\":{},\"result\":\"{}\",\"hash\":\"{}\",\"units_built\":{},\"shots_fired\":{},\"kills\":{},\"paths\":{},\"commands\":{}}}",
            map.map_id, seed, world.tick(), result, world.hash_hex(), built,
            world.shots_fired(), world.kills(), world.paths_solved(), world.commands_applied()
        );
        return Ok(());
    }
    if !quiet && world.tick() % 1200 != 0 {
        print_skirmish_status(&world);
    }
    println!("result:   {result} after {} ticks ({})", world.tick(), fmt_game_time(world.tick()));
    for a in world.armies() {
        println!(
            "  army {} {:<7} built={:<4} lost={:<4} kills={:<4} damage={}{}",
            a.id,
            world.catalog().factions[a.faction].key,
            a.units_built,
            a.units_lost,
            a.kills,
            a.damage_dealt / sim::SCALE,
            if a.defeated { "  [defeated]" } else { "" }
        );
    }
    for (i, t) in tier_built.iter().enumerate() {
        println!("  army {i} built by tier: T1={} T2={} T3={} T4={}", t[0], t[1], t[2], t[3]);
    }
    if !top_built.is_empty() {
        let list: Vec<String> = top_built.iter().map(|(k, v)| format!("{k}x{v}")).collect();
        println!("  T3/T4 types: {}", list.join(" "));
    }
    println!(
        "totals:   built={built} shots={} kills={} paths={} commands={} (rejected {})",
        world.shots_fired(),
        world.kills(),
        world.paths_solved(),
        world.commands_applied(),
        world.commands_rejected()
    );
    println!("hash:     {}", world.hash_hex());
    println!("timing:   load {load_ms:.0}ms, sim {sim_ms:.0}ms ({:.0} ticks/s)", world.tick() as f64 / (sim_ms / 1000.0).max(1e-9));
    Ok(())
}

fn fmt_game_time(ticks: u64) -> String {
    let s = ticks / sim::TICK_HZ;
    format!("{:02}:{:02}", s / 60, s % 60)
}

fn print_skirmish_status(world: &skirmish::Skirmish) {
    let mut line = format!("[{}]", fmt_game_time(world.tick()));
    for a in world.armies() {
        let n = |r: skirmish::Role| world.units().filter(|u| u.army == a.id && u.role == r).count();
        line.push_str(&format!(
            " | army {}: M {:>4}(+{:.1}) E {:>5}(+{:.0}) mex={} pgen={} fac={} units={} acu={}",
            a.id,
            a.mass / sim::SCALE,
            a.income_mass as f64 * sim::TICK_HZ as f64 / sim::SCALE as f64,
            a.energy / sim::SCALE,
            a.income_energy as f64 * sim::TICK_HZ as f64 / sim::SCALE as f64,
            n(skirmish::Role::Extractor),
            n(skirmish::Role::Generator),
            n(skirmish::Role::Factory),
            n(skirmish::Role::Mobile),
            world
                .commander_of(a.id)
                .map(|u| format!("{}%", u.hp * 100 / u.max_hp.max(1)))
                .unwrap_or_else(|| "dead".into())
        ));
    }
    println!("{line}");
}

fn cmd_play(args: &[String]) -> Result<(), String> {
    let load_file = flag_str(args, "--load").map(PathBuf::from);
    let map_arg = args.get(2).filter(|a| !a.starts_with("--"));
    let map_name = map_arg.map(String::as_str).unwrap_or("SCMP_001");
    let game_dir = positional_after(args, if map_arg.is_some() { 3 } else { 2 }).unwrap_or_else(scd::default_game_dir);
    let seed = flag_str(args, "--seed")
        .map(|s| s.parse::<u64>().map_err(|_| format!("invalid --seed '{s}'")))
        .transpose()?
        .unwrap_or(sim::DEFAULT_SEED);
    let speed = flag_usize(args, "--speed").unwrap_or(1).clamp(1, 8) as u32;
    let spectate = has_flag(args, "--spectate");
    let no_record = has_flag(args, "--no-record");
    let (width, height) = parse_size_flag(args).unwrap_or((1280, 800));
    let snapshot_out = flag_str(args, "--snapshot").map(PathBuf::from);
    let snapshot_ticks = flag_usize(args, "--snapshot-ticks").unwrap_or(5400) as u64;
    if has_flag(args, "--lobby") {
        return lobby::run_lobby(&game_dir, map_name, seed, width, height, snapshot_out.as_deref(), has_flag(args, "--lobby-autostart"));
    }

    let catalog = skirmish::Catalog::load(&game_dir)?;
    let meshes = play::UnitMeshCache::load(Some(&game_dir), &catalog);

    let (map, mut session) = if let Some(ref save_path) = load_file {
        let save = savegame::SaveGame::read(save_path)?;
        let map = skirmish::MapContext::load(&game_dir, &save.replay.header.map_id)?;
        let mut origin = savegame::origin_world(catalog, &map, &save.replay)?;
        origin.set_options(save.replay.header.options);
        let sess = save.into_session(origin)?;
        (map, sess)
    } else {
        let options = parse_game_options(args)?;
        let map = skirmish::MapContext::load(&game_dir, map_name)?;
        let setup = parse_skirmish_setup(args, &catalog, map.starts.len())?;
        let diffs = parse_ai_diffs(args, setup.len())?;
        let world = skirmish::Skirmish::on_map(catalog, &map, &setup, seed)?.with_options(options);
        let mut sess = play::PlaySession::new(world, 0, spectate, speed);
        for ai in &mut sess.ais {
            let d = diffs.get(ai.army as usize).copied().unwrap_or_default();
            *ai = skirmish_ai::ScriptedAi::with_difficulty(ai.army, d);
        }
        (map, sess)
    };
    session.muted = has_flag(args, "--mute");
    if no_record {
        session.auto_record = false;
    }

    if let Some(out_path) = snapshot_out {
        if session.loading() {
            session.finish_load_blocking();
        } else {
            session.spectate = true;
            if !session.ais.iter().any(|ai| ai.army == session.human_army) {
                let d = parse_ai_diffs(args, session.world.armies().len())
                    .ok()
                    .and_then(|v| v.first().copied())
                    .unwrap_or_default();
                session.ais.insert(0, skirmish_ai::ScriptedAi::with_difficulty(session.human_army, d));
            }
            for _ in 0..snapshot_ticks {
                if session.world.result().is_some() {
                    break;
                }
                session.step_sim();
                render_prep::snapshot_warm_up(
                    &session,
                    &map.scmap,
                    &meshes,
                    width as f32 / height as f32,
                    snapshot_ticks,
                );
            }
        }
        let cam = play::battle_snapshot_camera(
            &session.world,
            &map.scmap,
            width as f32 / height as f32,
        );
        let adapter = play::render_skirmish_snapshot(
            Some(&game_dir),
            &session,
            &map,
            &meshes,
            width,
            height,
            &out_path,
            Some(cam),
        )?;
        let built: u32 = session.world.armies().iter().map(|a| a.units_built).sum();
        println!("map:      {} ({})", map.map_name, map.map_id);
        println!("ticks:    {} ({})", session.world.tick(), fmt_game_time(session.world.tick()));
        println!("alive:    {} units", session.world.units().count());
        println!("built:    {built} | shots: {} | kills: {}", session.world.shots_fired(), session.world.kills());
        println!("hash:     {}", session.world.hash_hex());
        println!("adapter:  {adapter}");
        println!("snapshot: {} ({width}x{height})", out_path.display());
        return Ok(());
    }

    println!("Starting interactive skirmish on {} ({})", map.map_name, map.map_id);
    println!("Controls: LMB select/box (dbl-click/Ctrl+click same type) | RMB move/attack/repair/assist/reclaim | MMB/Q/E orbit | Wheel zoom");
    println!("          RMB drag: formation line (units selected) / pan (nothing selected or Alt) | Ctrl+RMB drag: area reclaim/repair");
    println!("          Ctrl+1..9 set group (Ctrl+Shift add) | 1..9 recall (double-tap center) | F ACU | H Factories | J Army | . idle engineer");
    println!("          B Fac | M Mex | P PGen | D PD | N Rad (LMB drag = row) | G Eng | T Tank | Y Bot | U Arty | O Overcharge | L Unload | K/Shift+K Enhance | Z Repeat | X Shield");
    println!("          F1 AI | F2 FOW | F6 Quicksave | F9 Quickload | Enter/Shift+Enter Chat | Tab Score | Esc Pause Menu | Space pause | +/- speed | F3 mute | F4/F5 volume");
    println!("          Command card (bottom-left): click = place/queue, Shift x5, RMB remove | ` next tech tab | Ctrl+M/P/A/G/E/C/L/O/N order modes | F7/F8/F10/F11 camera bookmarks (Ctrl saves) | Alt+LMB/Alt+Shift+LMB/Alt+RMB ping | keys: Options -> Keys");
    play::run_play_window(Some(&game_dir), session, map, meshes, width, height)
}

fn cmd_replay(args: &[String]) -> Result<(), String> {
    if args.len() < 4 {
        return Err("usage: farust replay <verify|play> <FILE> [GAME_DIR] [--size WxH] [--snapshot OUT.png] [--snapshot-ticks N]".to_string());
    }
    let mode = args[2].as_str();
    let replay_path = PathBuf::from(&args[3]);
    let game_dir = positional_after(args, 4).unwrap_or_else(scd::default_game_dir);

    let rep = replay::Replay::load(&replay_path)?;
    let catalog = skirmish::Catalog::load(&game_dir)?;
    let map = skirmish::MapContext::load(&game_dir, &rep.header.map_id)?;

    match mode {
        "verify" => {
            let t0 = Instant::now();
            let report = rep.verify(catalog, &map)?;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!(
                "replay verified: {} | ticks={} cmds={} checkpoints={} | hash={:016x} ({ms:.0}ms)",
                report.map_id,
                report.ticks,
                report.commands,
                report.checkpoints_verified,
                report.final_hash
            );
            Ok(())
        }
        "play" => {
            let speed = flag_usize(args, "--speed").unwrap_or(2).clamp(1, 8) as u32;
            let (width, height) = parse_size_flag(args).unwrap_or((1280, 800));
            let meshes = play::UnitMeshCache::load(Some(&game_dir), &catalog);
            let mut world = skirmish::Skirmish::new(
                catalog,
                map.grid.clone(),
                map.size,
                map.mass_points.clone(),
                &rep.header.setups,
                rep.header.seed,
            )?;
            world.set_domain_grids(
                map.naval_grid.clone(),
                map.amphib_grid.clone(),
                map.hydro_points.clone(),
            );
            world.set_options(rep.header.options);
            let final_tick = rep.final_tick;
            let mut session = play::PlaySession::new_replay(world, rep, speed);
            session.muted = has_flag(args, "--mute");
            if let Some(out) = flag_str(args, "--snapshot") {
                let target_ticks = flag_usize(args, "--snapshot-ticks").unwrap_or(3000) as u64;
                let warm_end = target_ticks.min(final_tick);
                for _ in 0..target_ticks {
                    if session.paused || session.world.result().is_some() {
                        break;
                    }
                    session.step_sim();
                    render_prep::snapshot_warm_up(
                        &session,
                        &map.scmap,
                        &meshes,
                        width as f32 / height as f32,
                        warm_end,
                    );
                }
                let cam = play::battle_snapshot_camera(
                    &session.world,
                    &map.scmap,
                    width as f32 / height as f32,
                );
                let gpu = play::render_skirmish_snapshot(
                    Some(&game_dir),
                    &session,
                    &map,
                    &meshes,
                    width,
                    height,
                    Path::new(out),
                    Some(cam),
                )?;
                println!(
                    "replay snapshot @ tick {}: hash={} on {} -> {out}",
                    session.world.tick(),
                    session.world.hash_hex(),
                    gpu
                );
                return Ok(());
            }
            println!("Replay: timeline click/drag seek | Space pause | +/- speed 1x-16x | [ ] seek -/+30s | Home restart | V perspective (observer/army FoW)");
            play::run_play_window(Some(&game_dir), session, map, meshes, width, height)
        }
        other => Err(format!("unknown replay subcommand '{other}' (expected verify|play)")),
    }
}

fn cmd_rendezvous(args: &[String]) -> Result<(), String> {
    let bind_addr = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("0.0.0.0:15050");
    let srv = rendezvous::RendezvousServer::bind(bind_addr)?;
    println!("FARust rendezvous + TCP relay listening on {}", srv.local_addr());
    srv.run_forever();
    Ok(())
}

fn cmd_host(args: &[String]) -> Result<(), String> {
    let use_code = has_flag(args, "--code");
    let rv_addr = flag_str(args, "--rendezvous")
        .map(str::to_string)
        .unwrap_or_else(|| settings::effective_rendezvous(None));

    // Positional args (skipping value flags): optional ADDR:PORT (if contains ':'), MAP, GAME_DIR.
    let mut pos: Vec<&str> = Vec::new();
    let mut i = 2usize;
    while i < args.len() {
        let a = args[i].as_str();
        if matches!(
            a,
            "--seed" | "--headless-ticks" | "--players" | "--armies" | "--factions" | "--size" | "--rendezvous"
        ) {
            i += 2;
            continue;
        }
        if a.starts_with("--") {
            i += 1;
            continue;
        }
        pos.push(a);
        i += 1;
    }

    let (addr, map_idx) = match pos.first().copied() {
        Some(first) if first.contains(':') => (first, 1usize),
        _ if use_code => ("0.0.0.0:0", 0usize),
        _ => {
            return Err("usage: farust host [<ADDR:PORT>] [MAP] [GAME_DIR] [--code] [--rendezvous ADDR] [--players N] [--armies 1,2,3] [--factions uef,cybran] [--seed S] [--play] [--headless-ticks N]".to_string());
        }
    };
    let map_name = pos.get(map_idx).copied().unwrap_or("SCMP_001");
    let game_dir = pos
        .get(map_idx + 1)
        .map(PathBuf::from)
        .unwrap_or_else(scd::default_game_dir);
    let seed = flag_str(args, "--seed")
        .map(|s| s.parse::<u64>().map_err(|_| format!("invalid --seed '{s}'")))
        .transpose()?
        .unwrap_or(sim::DEFAULT_SEED);
    let ticks = flag_usize(args, "--headless-ticks").unwrap_or(1800) as u64;
    let players = flag_usize(args, "--players").unwrap_or(2);
    if !(2..=8).contains(&players) {
        return Err(format!("--players {players} out of range 2..=8"));
    }
    let interactive = has_flag(args, "--play");

    let catalog = skirmish::Catalog::load(&game_dir)?;
    let map = skirmish::MapContext::load(&game_dir, map_name)?;
    if map.starts.len() < players {
        return Err(format!("{} has only {} start spots for {players} players", map.map_id, map.starts.len()));
    }
    // Defaults: spots 1..=players, factions cycling through all four.
    let mut setup_args = args.to_vec();
    if flag_str(args, "--armies").is_none() {
        let spots: Vec<String> = (1..=players).map(|i| i.to_string()).collect();
        setup_args.extend(["--armies".to_string(), spots.join(",")]);
    }
    if flag_str(args, "--factions").is_none() {
        setup_args.extend(["--factions".to_string(), "uef,cybran,aeon,seraphim".to_string()]);
    }
    let setup = parse_skirmish_setup(&setup_args, &catalog, map.starts.len())?;
    if setup.len() < players {
        return Err(format!("--armies lists {} armies but --players is {players}", setup.len()));
    }
    let setups: Vec<skirmish::ArmySetup> = setup
        .iter()
        .enumerate()
        .map(|(i, &(spot, faction))| skirmish::ArmySetup {
            name: map.starts[spot].0.clone(),
            faction,
            start: map.starts[spot].1,
            team: i as u8,
        })
        .collect();
    let hs = net::Handshake {
        map_id: map.map_id.clone(),
        seed,
        setups,
        player_count: players,
        turn_ticks: net::TURN_TICKS,
        input_delay: net::DEFAULT_INPUT_DELAY,
        tick_limit: if interactive { 0 } else { ticks },
        assigned_army: 0,
    };

    println!(
        "Hosting {players}-player lockstep match on {addr} ({}, {} armies, {} AI, seed {seed}); waiting for {} client(s)...",
        map.map_id,
        hs.setups.len(),
        hs.setups.len() - players,
        players - 1
    );
    let (host, bound) = if use_code {
        let (h, b, _code) = net::LockstepHost::bind_players_with_code(
            addr,
            &rv_addr,
            &hs,
            |army, peer, relayed| {
                let via = if relayed { "relay" } else { "direct" };
                println!("  army {army} joined from {peer} ({via})");
            },
        )?;
        (h, b)
    } else {
        net::LockstepHost::bind_players(addr, &hs, |army, peer| {
            println!("  army {army} joined from {peer}");
        })?
    };
    println!("All players connected on {bound}");
    let meshes = interactive.then(|| play::UnitMeshCache::load(Some(&game_dir), &catalog));
    let world = net::world_from_handshake(catalog, &map, &hs)?;
    if let Some(meshes) = meshes {
        let (width, height) = parse_size_flag(args).unwrap_or((1280, 800));
        let mut session = play::PlaySession::new(world, 0, false, 1);
        session.muted = has_flag(args, "--mute");
        net::NetHandle::start(&session.world, &hs, Box::new(host))?.attach(&mut session);
        return play::run_play_window(Some(&game_dir), session, map, meshes, width, height);
    }
    run_headless_peer("host", Box::new(host), world, &hs, ticks)
}

fn cmd_join(args: &[String], connect_lobby: Option<&str>) -> Result<(), String> {
    let code_flag = flag_str(args, "--code")
        .map(str::to_string)
        .or_else(|| connect_lobby.map(str::to_string));
    let rv_addr = flag_str(args, "--rendezvous")
        .map(str::to_string)
        .unwrap_or_else(|| settings::effective_rendezvous(None));
    let force_relay = has_flag(args, "--relay");

    let mut pos: Vec<&str> = Vec::new();
    let mut i = 2usize;
    while i < args.len() {
        let a = args[i].as_str();
        if matches!(a, "--code" | "--rendezvous" | "--headless-ticks" | "--size") {
            i += 2;
            continue;
        }
        if a.starts_with("--") {
            i += 1;
            continue;
        }
        pos.push(a);
        i += 1;
    }

    let (target, dir_idx) = match code_flag {
        Some(c) => (c, 0usize),
        None => {
            let first = pos.first().copied().ok_or_else(|| {
                "usage: farust join <ADDR:PORT | CODE> [GAME_DIR] [--code CODE] [--rendezvous ADDR] [--relay] [--play] [--headless-ticks N]".to_string()
            })?;
            (first.to_string(), 1usize)
        }
    };
    let game_dir = pos
        .get(dir_idx)
        .map(PathBuf::from)
        .unwrap_or_else(scd::default_game_dir);
    let interactive = has_flag(args, "--play");

    let (client, hs) = if rendezvous::looks_like_code(&target) {
        let (c, hs, via) = net::LockstepClient::connect_code_retry(
            &rv_addr,
            &target,
            force_relay,
            std::time::Duration::from_secs(30),
        )?;
        let kind = if via.is_relayed() { "relay" } else { "direct" };
        println!("Connected via code {target} ({kind} @ {})", via.addr());
        (c, hs)
    } else {
        net::LockstepClient::connect_retry(&target, std::time::Duration::from_secs(30))?
    };
    println!(
        "Joined {target} as army {} of {} players ({} armies) on {} seed {}",
        hs.assigned_army,
        hs.player_count,
        hs.setups.len(),
        hs.map_id,
        hs.seed
    );
    let ticks = if hs.tick_limit > 0 {
        if let Some(local) = flag_usize(args, "--headless-ticks")
            && local as u64 != hs.tick_limit
        {
            println!("note: using host tick limit {} (local --headless-ticks {local} ignored)", hs.tick_limit);
        }
        hs.tick_limit
    } else {
        flag_usize(args, "--headless-ticks").unwrap_or(1800) as u64
    };
    let catalog = skirmish::Catalog::load(&game_dir)?;
    let map = skirmish::MapContext::load(&game_dir, &hs.map_id)?;
    let meshes = interactive.then(|| play::UnitMeshCache::load(Some(&game_dir), &catalog));
    let world = net::world_from_handshake(catalog, &map, &hs)?;
    if let Some(meshes) = meshes {
        let (width, height) = parse_size_flag(args).unwrap_or((1280, 800));
        let mut session = play::PlaySession::new(world, hs.assigned_army, false, 1);
        session.muted = has_flag(args, "--mute");
        net::NetHandle::start(&session.world, &hs, Box::new(client))?.attach(&mut session);
        return play::run_play_window(Some(&game_dir), session, map, meshes, width, height);
    }
    run_headless_peer("client", Box::new(client), world, &hs, ticks)
}

fn run_headless_peer(
    role: &str,
    mut transport: Box<dyn net::TurnTransport>,
    mut world: skirmish::Skirmish,
    hs: &net::Handshake,
    ticks: u64,
) -> Result<(), String> {
    let mut core = net::LockstepCore::from_handshake(&world, hs);
    println!(
        "running {ticks} ticks (turn {} ticks, input delay {} turns, AI slots {:?})...",
        hs.turn_ticks,
        hs.input_delay,
        core.shared_ai_armies()
    );
    let t0 = Instant::now();
    let final_hash = net::run_headless(transport.as_mut(), &mut core, &mut world, ticks)?;
    println!(
        "lockstep {role} complete: army={} players={} ticks={} turns={} hash={final_hash:016x} ({:.0}ms)",
        hs.assigned_army,
        hs.player_count,
        world.tick(),
        core.exec_turn(),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn cmd_bench_scale(args: &[String]) -> Result<(), String> {
    let map_arg = args.get(2).filter(|a| !a.starts_with("--"));
    let map_name = map_arg.map(String::as_str).unwrap_or("SCMP_001");
    let game_dir = positional_after(args, if map_arg.is_some() { 3 } else { 2 }).unwrap_or_else(scd::default_game_dir);
    let unit_count = flag_usize(args, "--units").unwrap_or(2048).clamp(64, 16_384);
    let ticks = flag_usize(args, "--ticks").unwrap_or(200).clamp(10, 5000) as u64;
    let threads = flag_usize(args, "--threads")
        .unwrap_or_else(jobs::default_thread_count)
        .max(1);

    let catalog = skirmish::Catalog::load(&game_dir)?;
    let map = skirmish::MapContext::load(&game_dir, map_name)?;
    let setup = vec![(0, 0), (1, 1)];

    let build_army_world = |pool: Option<std::sync::Arc<jobs::WorkerPool>>| -> Result<skirmish::Skirmish, String> {
        let mut w = skirmish::Skirmish::on_map(catalog.clone(), &map, &setup, sim::DEFAULT_SEED)?;
        w.set_worker_pool(pool);
        let half = unit_count / 2;
        let s0 = w.armies()[0].start;
        let s1 = w.armies()[1].start;
        let dx = s1.0 - s0.0;
        let dy = s1.1 - s0.1;
        let dist = skirmish::isqrt((dx as i128) * (dx as i128) + (dy as i128) * (dy as i128)).max(1);
        let p0 = s0;
        let mut p1 = s1;
        for s_dec in (3i64..=11).rev() {
            let step = s_dec * 10;
            let cx = (s0.0 + ((dx as i128 * (step * sim::SCALE) as i128) / dist as i128) as i64)
                .clamp(40 * sim::SCALE, map.size.0 - 40 * sim::SCALE);
            let cy = (s0.1 + ((dy as i128 * (step * sim::SCALE) as i128) / dist as i128) as i64)
                .clamp(40 * sim::SCALE, map.size.1 - 40 * sim::SCALE);
            let (gx, gy) = map.grid.world_to_cell(cx, cy);
            if !map.grid.is_blocked(gx, gy) {
                p1 = (cx, cy);
                break;
            }
        }
        let f0_tank = w.faction(0).unwrap().land_units[0].clone();
        let f1_tank = w.faction(1).unwrap().land_units[0].clone();
        let cols = (half as f64).sqrt().ceil() as i64;
        let mut u0s = Vec::with_capacity(half);
        let mut u1s = Vec::with_capacity(half);
        for i in 0..half as i64 {
            let ox = ((i % cols) - cols / 2) * 5 * sim::SCALE / 2;
            let oy = ((i / cols) - cols / 2) * 5 * sim::SCALE / 2;
            if let Some(id) = w.spawn_unit(
                0,
                &f0_tank,
                (
                    (p0.0 + ox).clamp(20 * sim::SCALE, map.size.0 - 20 * sim::SCALE),
                    (p0.1 + oy).clamp(20 * sim::SCALE, map.size.1 - 20 * sim::SCALE),
                ),
            ) {
                u0s.push(id);
            }
            if let Some(id) = w.spawn_unit(
                1,
                &f1_tank,
                (
                    (p1.0 + ox).clamp(20 * sim::SCALE, map.size.0 - 20 * sim::SCALE),
                    (p1.1 + oy).clamp(20 * sim::SCALE, map.size.1 - 20 * sim::SCALE),
                ),
            ) {
                u1s.push(id);
            }
        }
        w.rebuild_spatial();
        // Split each army across 16 attack-move squads so batched A* routes solve in parallel
        for chunk in u0s.chunks((u0s.len() / 16).max(1)) {
            w.issue(
                0,
                skirmish::Command::AttackMove {
                    units: chunk.to_vec(),
                    x: p1.0,
                    y: p1.1,
                    queue: false,
                },
            );
        }
        for chunk in u1s.chunks((u1s.len() / 16).max(1)) {
            w.issue(
                1,
                skirmish::Command::AttackMove {
                    units: chunk.to_vec(),
                    x: p0.0,
                    y: p0.1,
                    queue: false,
                },
            );
        }
        Ok(w)
    };

    let mut w_1t = build_army_world(None)?;
    let t0 = Instant::now();
    w_1t.run(ticks);
    let sim_1t_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let hash_1t = w_1t.state_hash();

    let pool = std::sync::Arc::new(jobs::WorkerPool::new(threads));
    let mut w_nt = build_army_world(Some(std::sync::Arc::clone(&pool)))?;
    profiler::reset();
    let t1 = Instant::now();
    w_nt.run(ticks);
    let sim_nt_ms = t1.elapsed().as_secs_f64() * 1000.0;
    if profiler::enabled() {
        profiler::end_frame(None);
        println!("profile ({threads}t, per tick):\n{}", profiler::report_table(Some(ticks)));
    }
    let hash_nt = w_nt.state_hash();

    if hash_1t != hash_nt {
        return Err(format!(
            "scale determinism failure: 1t={hash_1t:016x} vs {threads}t={hash_nt:016x}"
        ));
    }

    // Also measure render_prep across all surviving units with persistent WorkerPool
    let snaps: Vec<sim::SimUnitSnapshot> = w_nt
        .units()
        .map(|u| sim::SimUnitSnapshot {
            id: u.id,
            blueprint_id: u.type_id.clone(),
            x: u.pos.x,
            y: u.pos.y,
            hp: u.hp,
            max_hp: u.max_hp,
        })
        .collect();
    let snaps_arc = std::sync::Arc::new(snaps);
    let scmap_arc = std::sync::Arc::new(map.scmap.clone());
    let mut cam = camera::OrbitCamera::for_map(scmap_arc.map_width, scmap_arc.map_height, 17.5, 1.6);
    cam.distance = 220.0;
    cam.pitch = 0.72;
    let vp = cam.view_proj_matrix();
    let eye = cam.eye_position();
    let frames = 25usize;

    let t2 = Instant::now();
    let mut r_1t = render_prep::prepare_render_frame(&snaps_arc, Some(&scmap_arc), vp, eye, 1);
    for _ in 1..frames {
        r_1t = render_prep::prepare_render_frame(&snaps_arc, Some(&scmap_arc), vp, eye, 1);
    }
    let rp_1t_ms = t2.elapsed().as_secs_f64() * 1000.0 / frames as f64;

    let t3 = Instant::now();
    let mut r_nt =
        render_prep::prepare_render_frame_pool(&snaps_arc, Some(&scmap_arc), vp, eye, &pool);
    for _ in 1..frames {
        r_nt = render_prep::prepare_render_frame_pool(&snaps_arc, Some(&scmap_arc), vp, eye, &pool);
    }
    let rp_nt_ms = t3.elapsed().as_secs_f64() * 1000.0 / frames as f64;

    if r_1t.digest_fnv1a != r_nt.digest_fnv1a {
        return Err("scale render_prep digest mismatch".to_string());
    }

    let sim_speedup = sim_1t_ms / sim_nt_ms.max(0.001);
    let rp_speedup = rp_1t_ms / rp_nt_ms.max(0.001);
    if has_flag(args, "--json") {
        println!(
            "{{\"map\":\"{}\",\"units\":{},\"ticks\":{ticks},\"threads\":{threads},\"sim_1t_ms\":{sim_1t_ms:.2},\"sim_nt_ms\":{sim_nt_ms:.2},\"sim_speedup\":{sim_speedup:.2},\"render_1t_ms\":{rp_1t_ms:.3},\"render_nt_ms\":{rp_nt_ms:.3},\"render_speedup\":{rp_speedup:.2},\"paths\":{},\"shots\":{},\"kills\":{},\"hash\":\"{hash_nt:016x}\"}}",
            map.map_id,
            snaps_arc.len(),
            w_nt.paths_solved(),
            w_nt.shots_fired(),
            w_nt.kills()
        );
        return Ok(());
    }

    println!(
        "scale test:  {} | spawned={} alive={} | ticks={ticks} | threads=1 vs {threads}",
        map.map_id,
        unit_count + 2,
        w_nt.units().count()
    );
    println!(
        "sim_tick:    1t={sim_1t_ms:.2}ms ({:.0} ticks/s) -> {threads}t={sim_nt_ms:.2}ms ({:.0} ticks/s) [{sim_speedup:.2}x]",
        ticks as f64 / (sim_1t_ms / 1000.0),
        ticks as f64 / (sim_nt_ms / 1000.0)
    );
    println!(
        "render_prep: 1t={rp_1t_ms:.3}ms/frame -> {threads}t={rp_nt_ms:.3}ms/frame [{rp_speedup:.2}x] | digest={:016x}",
        r_nt.digest_fnv1a
    );
    println!(
        "activity:    paths={} shots={} kills={} | hash={hash_nt:016x} (1t == {threads}t exact)",
        w_nt.paths_solved(),
        w_nt.shots_fired(),
        w_nt.kills()
    );
    Ok(())
}

fn cmd_menu(args: &[String], connect_lobby: Option<&str>) -> Result<(), String> {
    let start_idx = if args.get(1).is_some_and(|s| s == "menu") { 2 } else { 1 };
    let game_dir = positional_after(args, start_idx).unwrap_or_else(scd::default_game_dir);
    let screen = match flag_str(args, "--screen") {
        Some(s) => s.parse::<menu::ScreenId>()?,
        None => menu::ScreenId::Main,
    };
    let seed = flag_str(args, "--seed")
        .map(|s| s.parse::<u64>().map_err(|_| format!("invalid --seed '{s}'")))
        .transpose()?;
    let opts = menu::MenuOptions {
        screen,
        size: parse_size_flag(args),
        snapshot: flag_str(args, "--snapshot").map(PathBuf::from),
        initial_map: flag_str(args, "--map").map(str::to_string),
        seed,
        autostart: has_flag(args, "--lobby-autostart"),
        connect_lobby: connect_lobby.map(str::to_string),
    };
    menu::run(&game_dir, opts)
}

fn cmd_ai_arena(args: &[String]) -> Result<(), String> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut game_dir: Option<PathBuf> = None;
    let mut maps_arg = "SCMP_001".to_string();
    let mut seeds_arg = "2".to_string();
    let mut matchups_arg = "hard:normal,normal:easy,hard:legacy".to_string();
    let mut factions_arg = "uef,cybran".to_string();
    let mut max_ticks: u64 = 12_000;
    let mut threads: usize = jobs::default_thread_count().max(1);
    let mut json_stdout = false;
    let mut json_file: Option<PathBuf> = None;

    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--maps" => {
                i += 1;
                maps_arg = args.get(i).ok_or("--maps needs value")?.clone();
            }
            "--seeds" => {
                i += 1;
                seeds_arg = args.get(i).ok_or("--seeds needs value")?.clone();
            }
            "--matchups" => {
                i += 1;
                matchups_arg = args.get(i).ok_or("--matchups needs value")?.clone();
            }
            "--factions" => {
                i += 1;
                factions_arg = args.get(i).ok_or("--factions needs value")?.clone();
            }
            "--max-ticks" => {
                i += 1;
                let v = args.get(i).ok_or("--max-ticks needs value")?;
                max_ticks = v.parse().map_err(|_| format!("invalid --max-ticks '{v}'"))?;
            }
            "--threads" => {
                i += 1;
                let v = args.get(i).ok_or("--threads needs value")?;
                threads = v.parse::<usize>().map_err(|_| format!("invalid --threads '{v}'"))?.max(1);
            }
            "--victory" | "--unit-cap" => {
                i += 1;
            }
            "--out" => {
                i += 1;
                json_file = Some(PathBuf::from(args.get(i).ok_or("--out needs path")?));
            }
            "--json" => {
                if let Some(next) = args.get(i + 1)
                    && !next.starts_with("--")
                {
                    json_file = Some(PathBuf::from(next));
                    i += 1;
                } else {
                    json_stdout = true;
                }
            }
            a if a.starts_with("--") => return Err(format!("unknown flag '{a}' for ai-arena")),
            a => {
                if game_dir.is_none() {
                    game_dir = Some(PathBuf::from(a));
                }
            }
        }
        i += 1;
    }

    let game_dir = game_dir.unwrap_or_else(scd::default_game_dir);
    let options = parse_game_options(args)?;
    let catalog = skirmish::Catalog::load(&game_dir)?;

    let faction_parts: Vec<&str> = factions_arg.split(',').map(str::trim).collect();
    if faction_parts.len() != 2 {
        return Err(format!("--factions expects 2 entries (got '{factions_arg}')"));
    }
    let f0 = catalog
        .faction_index(faction_parts[0])
        .ok_or_else(|| format!("unknown faction '{}'", faction_parts[0]))?;
    let f1 = catalog
        .faction_index(faction_parts[1])
        .ok_or_else(|| format!("unknown faction '{}'", faction_parts[1]))?;

    let map_ids: Vec<String> = maps_arg
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if map_ids.is_empty() {
        return Err("--maps must list at least one map".to_string());
    }
    let mut maps = Vec::with_capacity(map_ids.len());
    for id in &map_ids {
        maps.push(skirmish::MapContext::load(&game_dir, id)?);
    }

    let seeds: Vec<u64> = if !seeds_arg.contains(',') {
        let count: u64 = seeds_arg
            .trim()
            .parse()
            .map_err(|_| format!("invalid --seeds '{seeds_arg}'"))?;
        if count == 0 {
            return Err("--seeds must be >= 1".to_string());
        }
        (1..=count).collect()
    } else {
        seeds_arg
            .split(',')
            .map(|s| s.trim().parse::<u64>().map_err(|_| format!("invalid seed '{s}'")))
            .collect::<Result<_, _>>()?
    };

    let mut matchups = Vec::new();
    for pair in matchups_arg.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (a, b) = pair
            .split_once(':')
            .or_else(|| pair.split_once("vs"))
            .ok_or_else(|| format!("matchup '{pair}' must be LEFT:RIGHT (e.g. hard:normal)"))?;
        let d0: skirmish_ai::Difficulty = a.parse()?;
        let d1: skirmish_ai::Difficulty = b.parse()?;
        matchups.push((d0, d1));
    }
    if matchups.is_empty() {
        return Err("--matchups must list at least one pair".to_string());
    }

    #[derive(Clone)]
    struct Job {
        idx: usize,
        map_idx: usize,
        seed: u64,
        d0: skirmish_ai::Difficulty,
        d1: skirmish_ai::Difficulty,
    }
    struct Outcome {
        idx: usize,
        map: String,
        seed: u64,
        d0: skirmish_ai::Difficulty,
        d1: skirmish_ai::Difficulty,
        winner: Option<skirmish::ArmyId>,
        ticks: u64,
        hash: String,
        kills0: u32,
        kills1: u32,
    }

    let mut jobs = Vec::new();
    for map_idx in 0..maps.len() {
        for &seed in &seeds {
            for &(d0, d1) in &matchups {
                jobs.push(Job { idx: jobs.len(), map_idx, seed, d0, d1 });
            }
        }
    }

    let t0 = Instant::now();
    let total = jobs.len();
    let next = AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<Result<Outcome, String>>> = std::sync::Mutex::new(Vec::with_capacity(total));

    std::thread::scope(|scope| {
        for _ in 0..threads.min(total).max(1) {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(k) else { break };
                let map = &maps[job.map_idx];
                let res = (|| {
                    let mut w = skirmish::Skirmish::on_map(catalog.clone(), map, &[(0, f0), (1, f1)], job.seed)?
                        .with_options(options);
                    let mut ai0 = skirmish_ai::ScriptedAi::with_difficulty(0, job.d0);
                    let mut ai1 = skirmish_ai::ScriptedAi::with_difficulty(1, job.d1);
                    while w.tick() < max_ticks && w.result().is_none() {
                        ai0.think(&mut w);
                        ai1.think(&mut w);
                        w.step();
                    }
                    let winner = match w.result() {
                        Some(skirmish::GameResult::Victory(a)) => Some(a),
                        _ => None,
                    };
                    Ok(Outcome {
                        idx: job.idx,
                        map: map.map_id.clone(),
                        seed: job.seed,
                        d0: job.d0,
                        d1: job.d1,
                        winner,
                        ticks: w.tick(),
                        hash: w.hash_hex(),
                        kills0: w.armies()[0].kills,
                        kills1: w.armies()[1].kills,
                    })
                })();
                results.lock().unwrap().push(res);
            });
        }
    });

    let mut outcomes = results.into_inner().unwrap().into_iter().collect::<Result<Vec<_>, String>>()?;
    outcomes.sort_by_key(|o| o.idx);
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Aggregate per-difficulty stats
    #[derive(Default)]
    struct Record {
        games: usize,
        wins: usize,
        losses: usize,
        draws: usize,
        ticks: u64,
    }
    let mut table: BTreeMap<skirmish_ai::Difficulty, Record> = BTreeMap::new();
    for o in &outcomes {
        for (side, diff) in [(0u8, o.d0), (1u8, o.d1)] {
            let e = table.entry(diff).or_default();
            e.games += 1;
            e.ticks += o.ticks;
            match o.winner {
                Some(w) if w == side => e.wins += 1,
                Some(_) => e.losses += 1,
                None => e.draws += 1,
            }
        }
    }

    let json_rows: Vec<String> = outcomes
        .iter()
        .map(|o| {
            let w_str = match o.winner {
                Some(w) => w.to_string(),
                None => "null".to_string(),
            };
            format!(
                "{{\"map\":\"{}\",\"seed\":{},\"left\":\"{}\",\"right\":\"{}\",\"winner\":{w_str},\"ticks\":{},\"kills\":[{},{}],\"hash\":\"{}\"}}",
                o.map, o.seed, o.d0.name(), o.d1.name(), o.ticks, o.kills0, o.kills1, o.hash
            )
        })
        .collect();
    let json_body = format!(
        "{{\"games\":{},\"max_ticks\":{max_ticks},\"threads\":{threads},\"wall_ms\":{wall_ms:.1},\"results\":[{}]}}\n",
        outcomes.len(),
        json_rows.join(",")
    );

    if let Some(ref path) = json_file {
        if let Some(p) = path.parent()
            && !p.as_os_str().is_empty()
        {
            std::fs::create_dir_all(p).map_err(|e| format!("create dir {}: {e}", p.display()))?;
        }
        std::fs::write(path, &json_body).map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    if json_stdout {
        print!("{json_body}");
        return Ok(());
    }

    println!(
        "ai-arena: {} games across {} map(s) x {} seed(s) x {} matchup(s) (max {} ticks, {} threads, {wall_ms:.0}ms)",
        outcomes.len(),
        maps.len(),
        seeds.len(),
        matchups.len(),
        max_ticks,
        threads
    );
    for o in &outcomes {
        let res_s = match o.winner {
            Some(0) => format!("{} WINS", o.d0.name()),
            Some(1) => format!("{} WINS", o.d1.name()),
            _ => "DRAW/LIMIT".to_string(),
        };
        println!(
            "  {:<8} seed={:<3} {:<8} vs {:<8} -> {:<14} @ {:>5}t ({}) kills={}/{} hash={}",
            o.map,
            o.seed,
            o.d0.name(),
            o.d1.name(),
            res_s,
            o.ticks,
            fmt_game_time(o.ticks),
            o.kills0,
            o.kills1,
            o.hash
        );
    }
    println!("standings:");
    for (diff, rec) in &table {
        println!(
            "  {:<9} games={:<3} wins={:<3} losses={:<3} draws={:<3} avg_ticks={}",
            diff.name(),
            rec.games,
            rec.wins,
            rec.losses,
            rec.draws,
            rec.ticks / rec.games.max(1) as u64
        );
    }
    if let Some(ref path) = json_file {
        println!("json:     {}", path.display());
    }
    Ok(())
}
