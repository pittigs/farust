//! Phase 25 performance & robustness commands: `bench-render` and `fuzz-parsers`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::camera::OrbitCamera;
use crate::play::{self, PlaySession, UnitMeshCache};
use crate::profiler;
use crate::sim::{self, SCALE};
use crate::skirmish::{self, Catalog, MapContext, Skirmish};
use crate::viewer::{self, GpuSceneResources};
use crate::{jobs, scd};

/// Spawns `unit_count` tanks split over two armies facing each other with attack-move orders
/// (same layout as `bench-scale`).
pub fn spawn_battle(
    catalog: &Catalog,
    map: &MapContext,
    unit_count: usize,
    pool: Option<Arc<jobs::WorkerPool>>,
) -> Result<Skirmish, String> {
    let setup = vec![(0, 0), (1, 1)];
    let mut w = Skirmish::on_map(catalog.clone(), map, &setup, sim::DEFAULT_SEED)?;
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
        let cx = (s0.0 + ((dx as i128 * (step * SCALE) as i128) / dist as i128) as i64)
            .clamp(40 * SCALE, map.size.0 - 40 * SCALE);
        let cy = (s0.1 + ((dy as i128 * (step * SCALE) as i128) / dist as i128) as i64)
            .clamp(40 * SCALE, map.size.1 - 40 * SCALE);
        let (gx, gy) = map.grid.world_to_cell(cx, cy);
        if !map.grid.is_blocked(gx, gy) {
            p1 = (cx, cy);
            break;
        }
    }
    let f0 = w.faction(0).and_then(|f| f.land_units.first().cloned()).ok_or("army 0 has no land units")?;
    let f1 = w.faction(1).and_then(|f| f.land_units.first().cloned()).ok_or("army 1 has no land units")?;
    let cols = (half as f64).sqrt().ceil() as i64;
    let mut u0s = Vec::with_capacity(half);
    let mut u1s = Vec::with_capacity(half);
    let clampx = |v: i64| v.clamp(20 * SCALE, map.size.0 - 20 * SCALE);
    let clampy = |v: i64| v.clamp(20 * SCALE, map.size.1 - 20 * SCALE);
    for i in 0..half as i64 {
        let ox = ((i % cols) - cols / 2) * 5 * SCALE / 2;
        let oy = ((i / cols) - cols / 2) * 5 * SCALE / 2;
        if let Some(id) = w.spawn_unit(0, &f0, (clampx(p0.0 + ox), clampy(p0.1 + oy))) {
            u0s.push(id);
        }
        if let Some(id) = w.spawn_unit(1, &f1, (clampx(p1.0 + ox), clampy(p1.1 + oy))) {
            u1s.push(id);
        }
    }
    w.rebuild_spatial();
    for (army, units, goal) in [(0u8, &u0s, p1), (1u8, &u1s, p0)] {
        for chunk in units.chunks((units.len() / 16).max(1)) {
            w.issue(
                army,
                skirmish::Command::AttackMove {
                    units: chunk.to_vec(),
                    x: goal.0,
                    y: goal.1,
                    queue: false,
                },
            );
        }
    }
    Ok(w)
}

/// Camera over the centroid of all units at `distance` (mid zoom: real meshes, no icons).
fn battle_camera(world: &Skirmish, map: &MapContext, distance: f32, aspect: f32) -> OrbitCamera {
    let scale = SCALE as f32;
    let (mut sx, mut sz, mut n) = (0.0f64, 0.0f64, 0usize);
    for u in world.units() {
        sx += f64::from(u.pos.x as f32 / scale);
        sz += f64::from(u.pos.y as f32 / scale);
        n += 1;
    }
    let (cx, cz) = if n > 0 {
        ((sx / n as f64) as f32, (sz / n as f64) as f32)
    } else {
        (map.scmap.map_width * 0.5, map.scmap.map_height * 0.5)
    };
    let cy = play::sample_terrain_y(&map.scmap, cx, cz);
    let mut cam = OrbitCamera::for_map(map.scmap.map_width, map.scmap.map_height, cy, aspect);
    cam.target = [cx, cy, cz];
    cam.yaw = 0.5;
    cam.pitch = 0.8;
    cam.distance = distance;
    cam
}

#[derive(Debug, Default, Clone)]
struct RenderBenchRow {
    units: usize,
    alive: usize,
    frames: usize,
    prep_ms: Vec<f64>,
    submit_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    total_ms: Vec<f64>,
    stats: viewer::DrawStats,
    gpu_timed: bool,
}

fn pct(v: &[f64]) -> (f64, f64, f64) {
    (profiler::percentile(v, 50.0), profiler::percentile(v, 95.0), profiler::percentile(v, 99.0))
}

/// `farust bench-render [MAP] [GAME_DIR] [--units N[,N..]] [--frames F] [--warmup-ticks T]
/// [--size WxH] [--camera-dist D] [--no-sim] [--snapshot OUT.png] [--json]`
pub fn cmd_bench_render(args: &[String]) -> Result<(), String> {
    let map_arg = args.get(2).filter(|a| !a.starts_with("--"));
    let map_name = map_arg.map(String::as_str).unwrap_or("SCMP_001").to_string();
    let game_dir = bench_positional_dir(args, if map_arg.is_some() { 3 } else { 2 }).unwrap_or_else(scd::default_game_dir);
    let counts: Vec<usize> = crate::flag_str(args, "--units")
        .unwrap_or("2048,4096,8192")
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .map(|n| n.clamp(64, 16_384))
        .collect();
    if counts.is_empty() {
        return Err("--units needs at least one count".into());
    }
    let frames = crate::flag_usize(args, "--frames").unwrap_or(120).clamp(2, 10_000);
    let warmup_ticks = crate::flag_usize(args, "--warmup-ticks").unwrap_or(120).min(5000) as u64;
    let (width, height) = crate::parse_size_flag(args).unwrap_or((1920, 1080));
    let (width, height) = (width.clamp(64, 8192), height.clamp(64, 8192));
    let cam_dist = crate::flag_str(args, "--camera-dist").and_then(|s| s.parse::<f32>().ok()).unwrap_or(150.0);
    let sim_each_frame = !args.iter().any(|a| a == "--no-sim");
    let json = args.iter().any(|a| a == "--json");
    let snapshot = crate::flag_str(args, "--snapshot").map(PathBuf::from);

    let t_load = Instant::now();
    let catalog = Catalog::load(&game_dir)?;
    let map = MapContext::load(&game_dir, &map_name)?;
    let meshes = UnitMeshCache::load(Some(&game_dir), &catalog);
    let pool = Arc::new(jobs::WorkerPool::new(jobs::default_thread_count()));
    if !json {
        println!(
            "bench-render: {map_name} {width}x{height} frames={frames} warmup={warmup_ticks} ticks, sim/frame={sim_each_frame} (load {:.0} ms)",
            t_load.elapsed().as_secs_f64() * 1000.0
        );
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no compatible Vulkan/GPU adapter found".to_string())?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bench_render"),
            required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
            ..wgpu::DeviceDescriptor::default()
        },
        None,
    ))
    .map_err(|e| format!("failed to create GPU device: {e}"))?;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let msaa4_ok = viewer::hdr_msaa4_supported(&adapter);
    let (resolve_tex, resolve_view) = viewer::create_attachment(
        &device,
        width,
        height,
        format,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        "bench_resolve",
    );

    let was_enabled = profiler::enabled();
    profiler::set_enabled(true);
    let mut rows = Vec::new();
    for (ci, &n) in counts.iter().enumerate() {
        let mut world = spawn_battle(&catalog, &map, n, Some(Arc::clone(&pool)))?;
        world.run(warmup_ticks);
        let mut session = PlaySession::new(world, 0, true, 1);
        session.selected.clear();
        let aspect = width as f32 / height as f32;
        let mut camera = battle_camera(&session.world, &map, cam_dist, aspect);
        camera.set_aspect(width, height);

        let (scene, minimap) = play::build_terrain_loaded_scene(Some(&game_dir), &map, camera.clone());
        let mut resources =
            GpuSceneResources::new(&device, &queue, &scene, format, width, height, msaa4_ok, &camera);
        let gpu_timed = resources.enable_gpu_timer(&device, &queue);
        if let Some(ref mm) = minimap {
            resources.set_minimap_texture(&device, &queue, mm);
        }
        play::upload_hud_icons(Some(&game_dir), &device, &queue, &mut resources, &session);
        let templates = play::upload_unit_templates(&device, &queue, &resources, &meshes);

        let mut row = RenderBenchRow {
            units: n,
            gpu_timed,
            ..RenderBenchRow::default()
        };
        profiler::reset();
        let warm = 3usize;
        for f in 0..frames + warm {
            if sim_each_frame {
                let _p = profiler::scope("bench.sim_step");
                session.world.step();
            }
            let t0 = Instant::now();
            play::upload_frame_resources(
                &device, &mut resources, &templates, &session, &map, &meshes, &camera, width, height,
            );
            resources.update_camera(&queue, &camera);
            let t1 = Instant::now();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bench_enc") });
            {
                let _p = profiler::scope("render.encode_submit");
                resources.encode_draw(&mut encoder, &resolve_view, scene.clear_color);
                queue.submit(Some(encoder.finish()));
                resources.gpu_timer_after_submit();
            }
            let t2 = Instant::now();
            {
                let _p = profiler::scope("render.gpu_wait");
                device.poll(wgpu::Maintain::Wait);
            }
            let t3 = Instant::now();
            let gpu = resources.poll_gpu_timer();
            profiler::end_frame(Some((t3 - t0).as_secs_f64() * 1000.0));
            if f < warm {
                continue;
            }
            row.prep_ms.push((t1 - t0).as_secs_f64() * 1000.0);
            row.submit_ms.push((t2 - t1).as_secs_f64() * 1000.0);
            row.total_ms.push((t3 - t0).as_secs_f64() * 1000.0);
            if let Some((s, m)) = gpu {
                row.gpu_ms.push(s + m);
            }
        }
        row.frames = frames;
        row.alive = session.world.units().count();
        row.stats = resources.draw_stats();
        if !json {
            print_row(&row);
            println!("  spans (avg ms/frame):\n{}", indent(&profiler::report_table(None), "    "));
        }
        if let Some(path) = snapshot.as_ref().filter(|_| ci == counts.len() - 1) {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bench_snap") });
            resources.encode_draw(&mut encoder, &resolve_view, scene.clear_color);
            queue.submit(Some(encoder.finish()));
            resources.gpu_timer_after_submit();
            save_texture_png(&device, &queue, &resolve_tex, width, height, path)?;
            if !json {
                println!("  snapshot -> {}", path.display());
            }
        }
        rows.push(row);
    }
    profiler::set_enabled(was_enabled);

    if json {
        let mut out = format!(
            "{{\"map\":\"{}\",\"adapter\":\"{}\",\"width\":{width},\"height\":{height},\"frames\":{frames},\"results\":[",
            map_name,
            info.name.replace('"', "'")
        );
        for (i, r) in rows.iter().enumerate() {
            let (p50, p95, p99) = pct(&r.total_ms);
            let (c50, c95, c99) = pct(&r.prep_ms);
            let (g50, g95, g99) = pct(&r.gpu_ms);
            out.push_str(&format!(
                "{}{{\"units\":{},\"alive\":{},\"cpu_prep_ms\":[{c50:.3},{c95:.3},{c99:.3}],\"gpu_ms\":[{g50:.3},{g95:.3},{g99:.3}],\"frame_ms\":[{p50:.3},{p95:.3},{p99:.3}],\"submit_ms_p50\":{:.3},\"draw_calls\":{},\"instances\":{},\"triangles\":{},\"unit_instances\":{},\"gpu_timed\":{}}}",
                if i > 0 { "," } else { "" },
                r.units,
                r.alive,
                profiler::percentile(&r.submit_ms, 50.0),
                r.stats.draw_calls,
                r.stats.instances,
                r.stats.triangles,
                r.stats.unit_instances,
                r.gpu_timed
            ));
        }
        out.push_str("]}");
        println!("{out}");
    } else {
        println!("adapter: {} ({:?})", info.name, info.backend);
        println!(
            "{:>6} {:>6} | {:>22} | {:>22} | {:>22} | {:>6} {:>8} {:>11}",
            "units", "alive", "cpu prep p50/95/99", "gpu p50/95/99", "frame p50/95/99", "draws", "inst", "tris"
        );
        for r in &rows {
            let f3 = |v: &[f64]| {
                let (a, b, c) = pct(v);
                format!("{a:6.2} {b:6.2} {c:6.2}")
            };
            println!(
                "{:>6} {:>6} | {:>22} | {:>22} | {:>22} | {:>6} {:>8} {:>11}",
                r.units,
                r.alive,
                f3(&r.prep_ms),
                if r.gpu_ms.is_empty() { "n/a".to_string() } else { f3(&r.gpu_ms) },
                f3(&r.total_ms),
                r.stats.draw_calls,
                r.stats.instances,
                r.stats.triangles
            );
        }
    }
    Ok(())
}

fn print_row(r: &RenderBenchRow) {
    let (c50, c95, c99) = pct(&r.prep_ms);
    let (g50, g95, g99) = pct(&r.gpu_ms);
    let (t50, t95, t99) = pct(&r.total_ms);
    println!(
        "units={} alive={} | cpu prep {c50:.2}/{c95:.2}/{c99:.2} ms | gpu {g50:.2}/{g95:.2}/{g99:.2} ms{} | frame {t50:.2}/{t95:.2}/{t99:.2} ms | submit p50 {:.2} ms",
        r.units,
        r.alive,
        if r.gpu_timed { "" } else { " (no TIMESTAMP_QUERY)" },
        profiler::percentile(&r.submit_ms, 50.0)
    );
    println!(
        "  draws={} instances={} triangles={} unit_batches={} unit_instances={}",
        r.stats.draw_calls, r.stats.instances, r.stats.triangles, r.stats.unit_batches, r.stats.unit_instances
    );
}

fn indent(s: &str, pre: &str) -> String {
    s.lines().map(|l| format!("{pre}{l}")).collect::<Vec<_>>().join("\n")
}

/// First positional (non-flag, non-flag-value) argument from `start` (the game dir).
fn bench_positional_dir(args: &[String], start: usize) -> Option<PathBuf> {
    const VALUED: [&str; 8] = ["--units", "--frames", "--warmup-ticks", "--size", "--camera-dist", "--snapshot", "--iters", "--seed"];
    let mut i = start;
    while i < args.len() {
        let a = &args[i];
        if VALUED.contains(&a.as_str()) {
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

fn save_texture_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tex: &wgpu::Texture,
    width: u32,
    height: u32,
    out: &Path,
) -> Result<(), String> {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let unpadded = width * 4;
    let padded = unpadded.div_ceil(align) * align;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench_readback"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bench_copy") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| format!("readback channel failed: {e}"))?
        .map_err(|e| format!("GPU buffer map failed: {e}"))?;
    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height as usize {
        let s = row * padded as usize;
        pixels.extend_from_slice(&mapped[s..s + unpadded as usize]);
    }
    drop(mapped);
    buf.unmap();
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let file = std::fs::File::create(out).map_err(|e| format!("create {}: {e}", out.display()))?;
    let mut png_enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    png_enc.set_color(png::ColorType::Rgba);
    png_enc.set_depth(png::BitDepth::Eight);
    let mut writer = png_enc.write_header().map_err(|e| format!("png header error: {e}"))?;
    writer.write_image_data(&pixels).map_err(|e| format!("png write error: {e}"))?;
    Ok(())
}

/// `farust fuzz-parsers [GAME_DIR] [--iters N] [--seed S] [--no-real]`
pub fn cmd_fuzz_parsers(args: &[String]) -> Result<(), String> {
    let iters = crate::flag_usize(args, "--iters").unwrap_or(100_000) as u64;
    let seed = crate::flag_str(args, "--seed").and_then(|s| s.parse::<u64>().ok()).unwrap_or(0x5EED_F00D);
    let game_dir = bench_positional_dir(args, 2).unwrap_or_else(scd::default_game_dir);
    let mut seeds = crate::fuzz::synthetic_seeds();
    if !args.iter().any(|a| a == "--no-real") {
        let real = crate::fuzz::real_seeds(&game_dir, 3, true);
        println!("fuzz-parsers: {} synthetic + {} real seeds ({})", seeds.len(), real.len(), game_dir.display());
        seeds.extend(real);
    }
    let t0 = Instant::now();
    let report = crate::fuzz::run(&seeds, iters, seed, &mut |_, _| None);
    print!("{}", crate::fuzz::format_report(&report));
    println!(
        "{} iterations in {:.1} s, panics: {}",
        iters,
        t0.elapsed().as_secs_f64(),
        report.panics()
    );
    if report.panics() > 0 {
        return Err(format!("{} parser panics found", report.panics()));
    }
    Ok(())
}
