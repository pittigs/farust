//! Phase 7 in-window 2D HUD, embedded bitmap font, command buttons, and clickable minimap.

use crate::camera::OrbitCamera;
use crate::play::{PlaySession, army_rgb};
use crate::sim::{SCALE, TICK_HZ};
use crate::skirmish::{GameResult, IntelVisibility, MapContext, Role};
use crate::viewer::HudVertex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HudAction {
    BuildFactory,
    BuildExtractor,
    BuildGenerator,
    BuildPointDefense,
    BuildRadar,
    ProduceEngineer,
    ProduceTank,
    ProduceBot,
    ProduceArtillery,
    AttackMove,
    Stop,
    TogglePause,
    ToggleFog,
    ToggleAi,
}

#[derive(Debug, Clone)]
pub struct HudButton {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub label: &'static str,
    pub action: HudAction,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct HudLayout {
    pub width: f32,
    pub height: f32,
    pub buttons: Vec<HudButton>,
    pub minimap: [f32; 4], // [x, y, size, size] in screen pixels
}

impl HudLayout {
    pub fn compute(session: &PlaySession, width: u32, height: u32) -> Self {
        let w = width.max(480) as f32;
        let h = height.max(360) as f32;
        let mm_size = (h * 0.24).clamp(136.0, 220.0);
        let minimap = [w - mm_size - 12.0, h - mm_size - 12.0, mm_size, mm_size];

        let mut buttons = Vec::new();
        let active_build = session.build_mode.as_deref();
        let f = session.world.faction(session.human_army);
        let is_build = |tid: Option<&str>| tid.is_some() && active_build == tid;

        let row1: [(&'static str, HudAction, bool); 7] = [
            ("[B]FAC", HudAction::BuildFactory, is_build(f.map(|x| x.factory.as_str()))),
            ("[M]MEX", HudAction::BuildExtractor, is_build(f.map(|x| x.extractor.as_str()))),
            ("[P]PGEN", HudAction::BuildGenerator, is_build(f.map(|x| x.generator.as_str()))),
            ("[D]PD", HudAction::BuildPointDefense, is_build(f.map(|x| x.point_defense.as_str()))),
            ("[N]RAD", HudAction::BuildRadar, is_build(f.map(|x| x.radar.as_str()))),
            ("[A]ATK", HudAction::AttackMove, session.attack_move_mode),
            ("[S]STOP", HudAction::Stop, false),
        ];
        let row2: [(&'static str, HudAction, bool); 7] = [
            ("[G]ENG", HudAction::ProduceEngineer, false),
            ("[T]TANK", HudAction::ProduceTank, false),
            ("[Y]BOT", HudAction::ProduceBot, false),
            ("[U]ARTY", HudAction::ProduceArtillery, false),
            ("[SPC]", HudAction::TogglePause, session.paused),
            ("[F2]FOW", HudAction::ToggleFog, session.fog_of_war),
            ("[F1]AI", HudAction::ToggleAi, session.spectate),
        ];

        let bw = 66.0;
        let bh = 54.0;
        let gap = 6.0;
        let base_x = 12.0;
        let y2 = h - bh - 10.0;
        let y1 = y2 - bh - gap;

        for (i, (label, action, active)) in row1.into_iter().enumerate() {
            buttons.push(HudButton {
                x: base_x + i as f32 * (bw + gap),
                y: y1,
                w: bw,
                h: bh,
                label,
                action,
                active,
            });
        }
        for (i, (label, action, active)) in row2.into_iter().enumerate() {
            buttons.push(HudButton {
                x: base_x + i as f32 * (bw + gap),
                y: y2,
                w: bw,
                h: bh,
                label,
                action,
                active,
            });
        }

        Self {
            width: w,
            height: h,
            buttons,
            minimap,
        }
    }

    pub fn hit_test_button(&self, px: f32, py: f32) -> Option<HudAction> {
        self.buttons
            .iter()
            .find(|b| px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h)
            .map(|b| b.action)
    }

    /// If `(px, py)` lies inside the minimap, returns normalized map `[u, v]` in `[0, 1]^2`.
    pub fn hit_test_minimap(&self, px: f32, py: f32) -> Option<(f32, f32)> {
        let [mx, my, mw, mh] = self.minimap;
        if px >= mx && px <= mx + mw && py >= my && py <= my + mh {
            Some(((px - mx) / mw, (py - my) / mh))
        } else {
            None
        }
    }
}

/// Returns the 9 unit IDs of `human_army` matching icon atlas slots `0..9`.
pub fn human_faction_icon_ids(session: &PlaySession) -> Vec<String> {
    let Some(f) = session.world.faction(session.human_army) else {
        return Vec::new();
    };
    let tank = f.land_units.first().cloned().unwrap_or_default();
    let bot = f.land_units.get(1).cloned().unwrap_or_else(|| tank.clone());
    let arty = f.land_units.get(2).cloned().unwrap_or_else(|| tank.clone());
    vec![
        f.factory.clone(),
        f.extractor.clone(),
        f.generator.clone(),
        f.point_defense.clone(),
        f.radar.clone(),
        f.engineer.clone(),
        tank,
        bot,
        arty,
    ]
}

fn hud_action_icon_slot(action: HudAction) -> Option<usize> {
    match action {
        HudAction::BuildFactory => Some(0),
        HudAction::BuildExtractor => Some(1),
        HudAction::BuildGenerator => Some(2),
        HudAction::BuildPointDefense => Some(3),
        HudAction::BuildRadar => Some(4),
        HudAction::ProduceEngineer => Some(5),
        HudAction::ProduceTank => Some(6),
        HudAction::ProduceBot => Some(7),
        HudAction::ProduceArtillery => Some(8),
        _ => None,
    }
}

/// Builds `(minimap_v, minimap_i, bg_v, bg_i, icons_v, icons_i, text_v, text_i)` for the 2D HUD pass.
pub fn build_hud_geometry(
    session: &PlaySession,
    map: &MapContext,
    camera: &OrbitCamera,
    width: u32,
    height: u32,
) -> (
    Vec<HudVertex>,
    Vec<u32>,
    Vec<HudVertex>,
    Vec<u32>,
    Vec<HudVertex>,
    Vec<u32>,
    Vec<HudVertex>,
    Vec<u32>,
) {
    let layout = HudLayout::compute(session, width, height);
    let w = layout.width;
    let h = layout.height;

    let mut mm_v = Vec::new();
    let mut mm_i = Vec::new();
    let mut v = Vec::with_capacity(2048);
    let mut i = Vec::with_capacity(3072);
    let mut ic_v = Vec::with_capacity(64);
    let mut ic_i = Vec::with_capacity(96);
    let mut fg_v = Vec::with_capacity(4096);
    let mut fg_i = Vec::with_capacity(6144);

    // 1. Minimap textured backdrop quad
    let [mx, my, mw, mh] = layout.minimap;
    push_quad_uv(&mut mm_v, &mut mm_i, mx, my, mw, mh, [1.0, 1.0, 1.0, 0.96], w, h);

    // 2. Top Economy & Status Bar
    push_rect(&mut v, &mut i, 0.0, 0.0, w, 46.0, [0.06, 0.08, 0.12, 0.84], w, h);
    push_rect(&mut v, &mut i, 0.0, 46.0, w, 2.0, [0.22, 0.45, 0.72, 0.85], w, h);

    if let Some(a) = session.world.army(session.human_army) {
        let m_fill = (a.mass as f32 / a.mass_cap.max(1) as f32).clamp(0.0, 1.0);
        let e_fill = (a.energy as f32 / a.energy_cap.max(1) as f32).clamp(0.0, 1.0);

        // Mass gauge
        push_rect(&mut v, &mut i, 14.0, 10.0, 180.0, 11.0, [0.12, 0.15, 0.18, 0.95], w, h);
        push_rect(&mut v, &mut i, 14.0, 10.0, 180.0 * m_fill, 11.0, [0.24, 0.86, 0.44, 0.95], w, h);
        let m_txt = format!(
            "M {}/{} (+{:.1})",
            a.mass / SCALE,
            a.mass_cap / SCALE,
            a.income_mass as f64 * TICK_HZ as f64 / SCALE as f64
        );
        draw_text(&mut fg_v, &mut fg_i, 14.0, 26.0, 1.6, &m_txt, [0.55, 0.98, 0.65, 1.0], w, h);

        // Energy gauge
        push_rect(&mut v, &mut i, 214.0, 10.0, 180.0, 11.0, [0.12, 0.15, 0.18, 0.95], w, h);
        push_rect(&mut v, &mut i, 214.0, 10.0, 180.0 * e_fill, 11.0, [0.96, 0.80, 0.22, 0.95], w, h);
        let e_txt = format!(
            "E {}/{} (+{:.0})",
            a.energy / SCALE,
            a.energy_cap / SCALE,
            a.income_energy as f64 * TICK_HZ as f64 / SCALE as f64
        );
        draw_text(&mut fg_v, &mut fg_i, 214.0, 26.0, 1.6, &e_txt, [1.0, 0.90, 0.42, 1.0], w, h);
    }

    let secs = session.world.tick() / TICK_HZ;
    let speed_str = if session.paused {
        "PAUSED".to_string()
    } else {
        format!("{}X", session.speed)
    };
    let n_units = session
        .world
        .units()
        .filter(|u| u.army == session.human_army && !u.role.is_structure())
        .count();
    let mid_txt = format!(
        "{:02}:{:02} [{}] UNITS:{} SEL:{}",
        secs / 60,
        secs % 60,
        speed_str,
        n_units,
        session.selected.len()
    );
    draw_text(&mut fg_v, &mut fg_i, 420.0, 16.0, 1.9, &mid_txt, [0.92, 0.95, 1.0, 1.0], w, h);

    if let Some(res) = session.world.result() {
        let banner = match res {
            GameResult::Victory(a) if a == session.human_army => "VICTORY!",
            GameResult::Victory(_) => "DEFEAT",
            GameResult::Draw => "DRAW",
        };
        push_rect(&mut v, &mut i, w * 0.5 - 120.0, 56.0, 240.0, 34.0, [0.06, 0.08, 0.12, 0.90], w, h);
        draw_text(&mut fg_v, &mut fg_i, w * 0.5 - 64.0, 65.0, 2.5, banner, [1.0, 0.85, 0.20, 1.0], w, h);
    } else if !session.status_banner.is_empty() {
        let trunc: String = session.status_banner.chars().take(48).collect();
        draw_text(&mut fg_v, &mut fg_i, 420.0, 32.0, 1.4, &trunc, [0.72, 0.82, 0.95, 0.95], w, h);
    }

    // 3. Bottom-left Command / Build / Production Palette (with FA unit icons + hotkeys)
    let bar_w = 7.0 * (66.0 + 6.0) + 12.0;
    let bar_h = 2.0 * 54.0 + 6.0 + 14.0;
    push_rect(&mut v, &mut i, 6.0, h - bar_h - 4.0, bar_w, bar_h, [0.05, 0.07, 0.11, 0.80], w, h);
    const ICON_SLOTS: f32 = 9.0;
    for btn in &layout.buttons {
        let bg = if btn.active {
            [0.18, 0.48, 0.78, 0.95]
        } else {
            [0.11, 0.15, 0.22, 0.92]
        };
        let border = if btn.active {
            [0.45, 0.92, 1.0, 1.0]
        } else {
            [0.26, 0.34, 0.46, 0.95]
        };
        push_rect(&mut v, &mut i, btn.x, btn.y, btn.w, btn.h, bg, w, h);
        push_rect_outline(&mut fg_v, &mut fg_i, btn.x, btn.y, btn.w, btn.h, 1.5, border, w, h);

        if let Some(slot) = hud_action_icon_slot(btn.action) {
            let icon_sz = 32.0;
            let ix = btn.x + (btn.w - icon_sz) * 0.5;
            let iy = btn.y + 4.0;
            let u0 = (slot as f32 + 0.03) / ICON_SLOTS;
            let u1 = (slot as f32 + 0.97) / ICON_SLOTS;
            let base = ic_v.len() as u32;
            ic_v.extend_from_slice(&[
                HudVertex { pos: to_ndc(ix, iy, w, h), uv: [u0, 0.0], color: [1.0, 1.0, 1.0, 0.98] },
                HudVertex { pos: to_ndc(ix + icon_sz, iy, w, h), uv: [u1, 0.0], color: [1.0, 1.0, 1.0, 0.98] },
                HudVertex { pos: to_ndc(ix + icon_sz, iy + icon_sz, w, h), uv: [u1, 1.0], color: [1.0, 1.0, 1.0, 0.98] },
                HudVertex { pos: to_ndc(ix, iy + icon_sz, w, h), uv: [u0, 1.0], color: [1.0, 1.0, 1.0, 0.98] },
            ]);
            ic_i.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            draw_text(&mut fg_v, &mut fg_i, btn.x + 5.0, btn.y + 39.0, 1.35, btn.label, [0.94, 0.97, 1.0, 1.0], w, h);
        } else {
            draw_text(&mut fg_v, &mut fg_i, btn.x + 6.0, btn.y + 22.0, 1.45, btn.label, [0.94, 0.97, 1.0, 1.0], w, h);
        }
    }

    // 4. Minimap border + live unit dots + camera viewport target
    push_rect_outline(&mut fg_v, &mut fg_i, mx - 2.0, my - 2.0, mw + 4.0, mh + 4.0, 2.0, [0.30, 0.44, 0.64, 0.95], w, h);
    let map_w = (map.size.0 as f32 / SCALE as f32).max(1.0);
    let map_h = (map.size.1 as f32 / SCALE as f32).max(1.0);

    for &(mpx, mpz) in session.world.mass_points() {
        let ux = mx + ((mpx as f32 / SCALE as f32) / map_w) * mw;
        let uy = my + ((mpz as f32 / SCALE as f32) / map_h) * mh;
        push_rect(&mut fg_v, &mut fg_i, ux - 1.5, uy - 1.5, 3.0, 3.0, [0.22, 0.90, 0.42, 0.75], w, h);
    }
    for u in session.world.units() {
        if session.fog_of_war && u.army != session.human_army {
            match session.world.intel_visibility(session.human_army, u) {
                IntelVisibility::Hidden => continue,
                IntelVisibility::RadarBlip => {
                    let ux = mx + ((u.pos.x as f32 / SCALE as f32) / map_w) * mw;
                    let uy = my + ((u.pos.y as f32 / SCALE as f32) / map_h) * mh;
                    push_rect(&mut fg_v, &mut fg_i, ux - 2.0, uy - 2.0, 4.0, 4.0, [0.96, 0.30, 0.25, 0.85], w, h);
                    continue;
                }
                IntelVisibility::Visible => {}
            }
        }
        let ux = mx + ((u.pos.x as f32 / SCALE as f32) / map_w) * mw;
        let uy = my + ((u.pos.y as f32 / SCALE as f32) / map_h) * mh;
        let dot = if u.role == Role::Commander { 5.0 } else if u.role.is_structure() { 3.5 } else { 2.5 };
        let rgb = army_rgb(&session.world, u.army);
        push_rect(
            &mut fg_v,
            &mut fg_i,
            ux - dot * 0.5,
            uy - dot * 0.5,
            dot,
            dot,
            [rgb[0], rgb[1], rgb[2], 1.0],
            w,
            h,
        );
    }
    // Camera focus reticle on minimap
    let cx = mx + (camera.target[0] / map_w).clamp(0.0, 1.0) * mw;
    let cy = my + (camera.target[2] / map_h).clamp(0.0, 1.0) * mh;
    let rw = (camera.distance / map_w * mw * 0.45).clamp(10.0, mw * 0.6);
    let rh = rw / camera.aspect.max(0.5);
    push_rect_outline(
        &mut fg_v,
        &mut fg_i,
        (cx - rw * 0.5).clamp(mx, mx + mw - 4.0),
        (cy - rh * 0.5).clamp(my, my + mh - 4.0),
        rw.min(mw),
        rh.min(mh),
        1.2,
        [1.0, 1.0, 1.0, 0.9],
        w,
        h,
    );

    (mm_v, mm_i, v, i, ic_v, ic_i, fg_v, fg_i)
}

#[inline]
fn to_ndc(px: f32, py: f32, w: f32, h: f32) -> [f32; 2] {
    [(2.0 * px / w) - 1.0, 1.0 - (2.0 * py / h)]
}

fn push_quad_uv(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    qw: f32,
    qh: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    let base = verts.len() as u32;
    verts.extend_from_slice(&[
        HudVertex { pos: to_ndc(x, y, w, h), uv: [0.0, 0.0], color },
        HudVertex { pos: to_ndc(x + qw, y, w, h), uv: [1.0, 0.0], color },
        HudVertex { pos: to_ndc(x + qw, y + qh, w, h), uv: [1.0, 1.0], color },
        HudVertex { pos: to_ndc(x, y + qh, w, h), uv: [0.0, 1.0], color },
    ]);
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn push_rect(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    rw: f32,
    rh: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    push_quad_uv(verts, indices, x, y, rw, rh, color, w, h);
}

fn push_rect_outline(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    rw: f32,
    rh: f32,
    t: f32,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    push_rect(verts, indices, x, y, rw, t, color, w, h);
    push_rect(verts, indices, x, y + rh - t, rw, t, color, w, h);
    push_rect(verts, indices, x, y, t, rh, color, w, h);
    push_rect(verts, indices, x + rw - t, y, t, rh, color, w, h);
}

fn draw_text(
    verts: &mut Vec<HudVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    scale: f32,
    text: &str,
    color: [f32; 4],
    w: f32,
    h: f32,
) {
    let mut cx = x;
    for ch in text.bytes() {
        let g = glyph_3x5(ch.to_ascii_uppercase());
        for row in 0..5usize {
            let bits = g[row];
            for col in 0..3usize {
                if (bits >> (2 - col)) & 1 != 0 {
                    push_rect(
                        verts,
                        indices,
                        cx + col as f32 * scale,
                        y + row as f32 * scale,
                        scale,
                        scale,
                        color,
                        w,
                        h,
                    );
                }
            }
        }
        cx += 4.0 * scale;
    }
}

/// Compact 3x5 bitmap glyphs (3 bits per row x 5 rows) for legible HUD text.
fn glyph_3x5(c: u8) -> [u8; 5] {
    match c {
        b'0' => [7, 5, 5, 5, 7],
        b'1' => [2, 6, 2, 2, 7],
        b'2' => [7, 1, 7, 4, 7],
        b'3' => [7, 1, 7, 1, 7],
        b'4' => [5, 5, 7, 1, 1],
        b'5' => [7, 4, 7, 1, 7],
        b'6' => [7, 4, 7, 5, 7],
        b'7' => [7, 1, 1, 2, 2],
        b'8' => [7, 5, 7, 5, 7],
        b'9' => [7, 5, 7, 1, 7],
        b'A' => [2, 5, 7, 5, 5],
        b'B' => [6, 5, 6, 5, 6],
        b'C' => [7, 4, 4, 4, 7],
        b'D' => [6, 5, 5, 5, 6],
        b'E' => [7, 4, 6, 4, 7],
        b'F' => [7, 4, 6, 4, 4],
        b'G' => [7, 4, 5, 5, 7],
        b'H' => [5, 5, 7, 5, 5],
        b'I' => [7, 2, 2, 2, 7],
        b'J' => [1, 1, 1, 5, 7],
        b'K' => [5, 5, 6, 5, 5],
        b'L' => [4, 4, 4, 4, 7],
        b'M' => [5, 7, 7, 5, 5],
        b'N' => [6, 5, 5, 5, 5],
        b'O' => [7, 5, 5, 5, 7],
        b'P' => [7, 5, 7, 4, 4],
        b'Q' => [7, 5, 5, 7, 3],
        b'R' => [7, 5, 6, 5, 5],
        b'S' => [7, 4, 7, 1, 7],
        b'T' => [7, 2, 2, 2, 2],
        b'U' => [5, 5, 5, 5, 7],
        b'V' => [5, 5, 5, 5, 2],
        b'W' => [5, 5, 7, 7, 5],
        b'X' => [5, 5, 2, 5, 5],
        b'Y' => [5, 5, 2, 2, 2],
        b'Z' => [7, 1, 2, 4, 7],
        b':' => [0, 2, 0, 2, 0],
        b'.' => [0, 0, 0, 0, 2],
        b'+' => [0, 2, 7, 2, 0],
        b'-' => [0, 0, 7, 0, 0],
        b'/' => [1, 1, 2, 4, 4],
        b'(' => [2, 4, 4, 4, 2],
        b')' => [2, 1, 1, 1, 2],
        b'[' => [6, 4, 4, 4, 6],
        b']' => [3, 1, 1, 1, 3],
        b'!' => [2, 2, 2, 0, 2],
        _ => [0, 0, 0, 0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathfinding::PathGrid;
    use crate::scmap::{ScMap, WaterInfo};
    use crate::skirmish::tests::test_world;

    fn dummy_map() -> MapContext {
        MapContext {
            map_id: "TEST".into(),
            map_name: "Test".into(),
            scmap: ScMap {
                version_major: 2,
                version_minor: 60,
                map_width: 256.0,
                map_height: 256.0,
                preview_dds: vec![],
                heightmap_width: 8,
                heightmap_height: 8,
                height_scale: 0.01,
                heightmap: vec![1000; 81],
                terrain_shader: String::new(),
                background_path: String::new(),
                sky_cubemap_path: String::new(),
                cubemaps: vec![],
                lighting: crate::scmap::ScMapLighting::default(),
                water: WaterInfo {
                    has_water: true,
                    elevation: 10.0,
                    elevation_deep: 8.0,
                    elevation_abyss: 2.0,
                },
                terrain_textures: None,
            },
            grid: PathGrid::new(64, 64, 4 * SCALE),
            size: (256 * SCALE, 256 * SCALE),
            starts: vec![],
            mass_points: vec![],
        }
    }

    #[test]
    fn hud_layout_buttons_and_minimap_hit_test() {
        let w = test_world(50);
        let sess = PlaySession::new(w, 0, false, 1);
        let map = dummy_map();
        let cam = OrbitCamera::for_map(256.0, 256.0, 10.0, 1.6);
        let layout = HudLayout::compute(&sess, 1280, 800);

        let b0 = &layout.buttons[0];
        assert_eq!(
            layout.hit_test_button(b0.x + 10.0, b0.y + 10.0),
            Some(HudAction::BuildFactory)
        );
        let [mx, my, mw, mh] = layout.minimap;
        let mm = layout
            .hit_test_minimap(mx + mw * 0.25, my + mh * 0.75)
            .expect("minimap hit");
        assert!((mm.0 - 0.25).abs() < 1e-4 && (mm.1 - 0.75).abs() < 1e-4);

        let (mv, mi, bg_v, bg_i, ic_v, ic_i, fg_v, fg_i) =
            build_hud_geometry(&sess, &map, &cam, 1280, 800);
        assert_eq!(mv.len(), 4);
        assert_eq!(mi.len(), 6);
        assert!(!bg_v.is_empty() && !bg_i.is_empty());
        assert_eq!(ic_v.len(), 9 * 4);
        assert_eq!(ic_i.len(), 9 * 6);
        assert!(!fg_v.is_empty() && !fg_i.is_empty());
        assert!(fg_v.iter().all(|vx| vx.pos.iter().all(|c| c.is_finite())));
    }
}
