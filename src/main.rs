mod blueprint;
mod camera;
mod combat;
mod dds;
mod dds_decode;
mod economy;
mod jobs;
mod hud;
mod net;
mod pathfinding;
mod play;
mod render_prep;
mod replay;
mod scenario;
mod scd;
mod scene;
mod scm;
mod scmap;
mod sim;
mod skirmish;
mod skirmish_ai;
mod spatial;
mod viewer;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn print_usage() {
    eprintln!("farust - Forged Alliance Rust tooling");
    eprintln!();
    eprintln!("usage:");
    eprintln!("  farust inventory [GAME_DIR]");
    eprintln!("  farust list <FILE.scd> [--limit N]");
    eprintln!("  farust cat <FILE.scd> <INNER_PATH>");
    eprintln!("  farust units [FILE.scd] [--limit N] [--search TEXT]");
    eprintln!("  farust unit <ID> [FILE.scd]");
    eprintln!("  farust maps [GAME_DIR] [--limit N] [--check]");
    eprintln!("  farust map <NAME> [GAME_DIR]");
    eprintln!("  farust meshes [FILE.scd] [--limit N] [--check]");
    eprintln!("  farust mesh <PATH> [FILE.scd]");
    eprintln!("  farust textures [FILE.scd] [--limit N] [--check]");
    eprintln!("  farust dds <PATH> [FILE.scd]");
    eprintln!("  farust view-unit <ID> [GAME_DIR] [--size WxH] [--snapshot OUT.png]");
    eprintln!("  farust view-map <NAME> [GAME_DIR] [--size WxH] [--snapshot OUT.png]");
    eprintln!("  farust sim-run --ticks N [--seed S] [--scenario NAME] [--dt SEC] [--hash] [--json]");
    eprintln!("  farust bench-multicore [GAME_DIR] [--threads N] [--limit N] [--json]");
    eprintln!("  farust bench-scale [MAP] [GAME_DIR] [--units N] [--ticks T] [--threads K] [--json]");
    eprintln!("  farust skirmish [MAP] [GAME_DIR] [--armies 1,2] [--factions uef,cybran] [--seed S] [--max-ticks N] [--record FILE] [--quiet] [--json]");
    eprintln!("  farust play [MAP] [GAME_DIR] [--armies 1,2] [--factions uef,cybran] [--seed S] [--speed N] [--spectate] [--size WxH] [--snapshot OUT.png] [--snapshot-ticks N]");
    eprintln!("  farust replay <verify|play> <FILE> [GAME_DIR] [--size WxH] [--snapshot OUT.png] [--snapshot-ticks N]");
    eprintln!("  farust host <ADDR:PORT> [MAP] [GAME_DIR] [--factions uef,cybran] [--seed S] [--headless-ticks N]");
    eprintln!("  farust join <ADDR:PORT> [GAME_DIR] [--headless-ticks N]");
    eprintln!();
    eprintln!("env:");
    eprintln!("  FA_PATH  override for the game install dir");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = run(&args);
    if let Err(e) = result {
        eprintln!("error: {e}");
        print_usage();
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        print_usage();
        return Ok(());
    }
    match args[1].as_str() {
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
        "map" => cmd_map(args),
        "meshes" => cmd_meshes(args),
        "mesh" => cmd_mesh(args),
        "textures" => cmd_textures(args),
        "dds" => cmd_dds(args),
        "view-unit" => cmd_view_unit(args),
        "view-map" => cmd_view_map(args),
        "sim-run" => cmd_sim_run(args),
        "bench-multicore" => cmd_bench_multicore(args),
        "bench-scale" => cmd_bench_scale(args),
        "skirmish" => cmd_skirmish(args),
        "play" => cmd_play(args),
        "replay" => cmd_replay(args),
        "host" => cmd_host(args),
        "join" => cmd_join(args),
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
                            if map_data.terrain_textures.is_some() {
                                strata_ok += 1;
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
            "{}: checked {} maps: {} scenarios, {} saves, {} scmaps ({} with terrain strata, 0 errors)",
            scd::maps_dir(&game_dir).display(),
            maps.len(),
            scenarios_ok,
            saves_ok,
            scmaps_ok,
            strata_ok
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
            "  {:<12} {:>4}x{:<4}  {:<10} armies={:<2}  {}",
            meta.id,
            meta.size.0,
            meta.size.1,
            meta.scenario_type,
            meta.armies.len(),
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

fn cmd_view_unit(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust view-unit <ID> [GAME_DIR] [--size WxH] [--snapshot OUT.png]".to_string());
    }
    let unit_id = &args[2];
    let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
    let (width, height) = parse_size_flag(args).unwrap_or((1280, 800));
    let aspect = width as f32 / height as f32;
    let scene = scene::load_unit_scene(&game_dir, unit_id, aspect)?;
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
            || a == "--scenario"
            || a == "--dt"
            || a == "--threads"
            || a == "--armies"
            || a == "--factions"
            || a == "--max-ticks"
            || a == "--speed"
            || a == "--snapshot-ticks"
            || a == "--record"
            || a == "--units"
            || a == "--headless-ticks"
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

fn cmd_skirmish(args: &[String]) -> Result<(), String> {
    let map_name = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("SCMP_001");
    let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
    let seed = flag_str(args, "--seed")
        .map(|s| s.parse::<u64>().map_err(|_| format!("invalid --seed '{s}'")))
        .transpose()?
        .unwrap_or(sim::DEFAULT_SEED);
    let max_ticks = flag_usize(args, "--max-ticks").map(|n| n as u64).unwrap_or(skirmish::DEFAULT_MAX_TICKS);
    let quiet = has_flag(args, "--quiet") || has_flag(args, "--json");

    let t0 = Instant::now();
    let catalog = skirmish::Catalog::load(&game_dir)?;
    let map = skirmish::MapContext::load(&game_dir, map_name)?;
    let setup = parse_skirmish_setup(args, &catalog, map.starts.len())?;
    let mut world = skirmish::Skirmish::on_map(catalog, &map, &setup, seed)?;
    let record_out = flag_str(args, "--record").map(PathBuf::from);
    let mut recorder = record_out.as_ref().map(|_| {
        let setups: Vec<skirmish::ArmySetup> = world
            .armies()
            .iter()
            .map(|a| skirmish::ArmySetup {
                name: a.name.clone(),
                faction: a.faction,
                start: a.start,
            })
            .collect();
        replay::ReplayRecorder::new(&map.map_id, seed, setups)
    });
    let mut ais: Vec<skirmish_ai::ScriptedAi> = (0..setup.len())
        .map(|i| skirmish_ai::ScriptedAi::new(i as skirmish::ArmyId))
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
    while world.tick() < max_ticks && world.result().is_none() {
        let tick = world.tick();
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
    let map_name = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("SCMP_001");
    let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
    let seed = flag_str(args, "--seed")
        .map(|s| s.parse::<u64>().map_err(|_| format!("invalid --seed '{s}'")))
        .transpose()?
        .unwrap_or(sim::DEFAULT_SEED);
    let speed = flag_usize(args, "--speed").unwrap_or(1).clamp(1, 8) as u32;
    let spectate = has_flag(args, "--spectate");
    let (width, height) = parse_size_flag(args).unwrap_or((1280, 800));
    let snapshot_out = flag_str(args, "--snapshot").map(PathBuf::from);
    let snapshot_ticks = flag_usize(args, "--snapshot-ticks").unwrap_or(5400) as u64;

    let catalog = skirmish::Catalog::load(&game_dir)?;
    let meshes = play::UnitMeshCache::load(Some(&game_dir), &catalog);
    let map = skirmish::MapContext::load(&game_dir, map_name)?;
    let setup = parse_skirmish_setup(args, &catalog, map.starts.len())?;
    let world = skirmish::Skirmish::on_map(catalog, &map, &setup, seed)?;

    if let Some(out_path) = snapshot_out {
        let mut session = play::PlaySession::new(world, 0, true, speed);
        for _ in 0..snapshot_ticks {
            if session.world.result().is_some() {
                break;
            }
            session.step_sim();
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
    println!("Controls: LMB select/box | RMB move/attack/repair/assist/reclaim | RMB drag pan | MMB/Q/E orbit | Wheel zoom");
    println!("          1 ACU | 2 Factories | 3 Army | B Fac | M Mex | P PGen | D PD | N Rad | G Eng | T Tank | Y Bot | U Arty | O Overcharge | F1 AI | F2 FOW");
    let session = play::PlaySession::new(world, 0, spectate, speed);
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
            let world = skirmish::Skirmish::new(
                catalog,
                map.grid.clone(),
                map.size,
                map.mass_points.clone(),
                &rep.header.setups,
                rep.header.seed,
            )?;
            let mut session = play::PlaySession::new_replay(world, rep, speed);
            if let Some(out) = flag_str(args, "--snapshot") {
                let target_ticks = flag_usize(args, "--snapshot-ticks").unwrap_or(3000) as u64;
                for _ in 0..target_ticks {
                    if session.paused || session.world.result().is_some() {
                        break;
                    }
                    session.step_sim();
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
            play::run_play_window(Some(&game_dir), session, map, meshes, width, height)
        }
        other => Err(format!("unknown replay subcommand '{other}' (expected verify|play)")),
    }
}

fn cmd_host(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust host <ADDR:PORT> [MAP] [GAME_DIR] [--factions uef,cybran] [--seed S] [--headless-ticks N]".to_string());
    }
    let addr = &args[2];
    let map_name = args
        .get(3)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("SCMP_001");
    let game_dir = positional_after(args, 4).unwrap_or_else(scd::default_game_dir);
    let seed = flag_str(args, "--seed")
        .map(|s| s.parse::<u64>().map_err(|_| format!("invalid --seed '{s}'")))
        .transpose()?
        .unwrap_or(sim::DEFAULT_SEED);
    let ticks = flag_usize(args, "--headless-ticks").unwrap_or(1800) as u64;

    let catalog = skirmish::Catalog::load(&game_dir)?;
    let map = skirmish::MapContext::load(&game_dir, map_name)?;
    let setup = parse_skirmish_setup(args, &catalog, map.starts.len())?;
    let setups: Vec<skirmish::ArmySetup> = setup
        .iter()
        .take(2)
        .map(|&(spot, faction)| skirmish::ArmySetup {
            name: map.starts[spot].0.clone(),
            faction,
            start: map.starts[spot].1,
        })
        .collect();
    if setups.len() < 2 {
        return Err("lockstep host requires 2 armies".to_string());
    }

    println!("Hosting TCP lockstep match on {addr} ({})...", map.map_id);
    let (mut peer, bound_addr) = net::LockstepPeer::host_bind(addr, &map.map_id, seed, &setups)?;
    println!("Peer connected on {bound_addr}; running {ticks} ticks...");
    let mut world = skirmish::Skirmish::new(
        catalog,
        map.grid.clone(),
        map.size,
        map.mass_points.clone(),
        &setups,
        seed,
    )?;
    let final_hash = net::run_headless_lockstep(&mut peer, &mut world, ticks)?;
    println!(
        "lockstep host complete: ticks={} hash={final_hash:016x}",
        world.tick()
    );
    Ok(())
}

fn cmd_join(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: farust join <ADDR:PORT> [GAME_DIR] [--headless-ticks N]".to_string());
    }
    let addr = &args[2];
    let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
    let ticks = flag_usize(args, "--headless-ticks").unwrap_or(1800) as u64;

    let (mut peer, hs) = net::LockstepPeer::connect(addr)?;
    let catalog = skirmish::Catalog::load(&game_dir)?;
    let map = skirmish::MapContext::load(&game_dir, &hs.map_id)?;
    let mut world = skirmish::Skirmish::new(
        catalog,
        map.grid.clone(),
        map.size,
        map.mass_points.clone(),
        &hs.setups,
        hs.seed,
    )?;
    let final_hash = net::run_headless_lockstep(&mut peer, &mut world, ticks)?;
    println!(
        "lockstep client complete: army={} ticks={} hash={final_hash:016x}",
        peer.local_army,
        world.tick()
    );
    Ok(())
}

fn cmd_bench_scale(args: &[String]) -> Result<(), String> {
    let map_name = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .unwrap_or("SCMP_001");
    let game_dir = positional_after(args, 3).unwrap_or_else(scd::default_game_dir);
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
        let p0 = w.armies()[0].start;
        let p1 = w.armies()[1].start;
        let f0_tank = w.faction(0).unwrap().land_units[0].clone();
        let f1_tank = w.faction(1).unwrap().land_units[0].clone();
        let cols = (half as f64).sqrt().ceil() as i64;
        let mut u0s = Vec::with_capacity(half);
        let mut u1s = Vec::with_capacity(half);
        for i in 0..half as i64 {
            let ox = ((i % cols) - cols / 2) * 3 * sim::SCALE;
            let oy = ((i / cols) - cols / 2) * 3 * sim::SCALE;
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
    let t1 = Instant::now();
    w_nt.run(ticks);
    let sim_nt_ms = t1.elapsed().as_secs_f64() * 1000.0;
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
