//! Minimal TrueType (`glyf` outlines) reader and anti-aliased coverage rasterizer.
//!
//! Used by the front end ([`crate::ui_kit`]) to bake the fonts shipped with the user's
//! Forged Alliance install (`fonts/ARIAL.TTF`, `fonts/zeroes_3.ttf`, ...) into a glyph
//! atlas. Supports `cmap` formats 4 and 0, short/long `loca`, simple and composite
//! glyphs, and quadratic contours flattened into lines; the rasterizer accumulates signed
//! area per pixel (non-zero winding for well-formed fonts) like `font-rs`.

use crate::dds_decode::RgbaImage;

fn u16_at(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*d.get(o)?, *d.get(o + 1)?]))
}

fn i16_at(d: &[u8], o: usize) -> Option<i16> {
    u16_at(d, o).map(|v| v as i16)
}

fn u32_at(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_be_bytes([*d.get(o)?, *d.get(o + 1)?, *d.get(o + 2)?, *d.get(o + 3)?]))
}

#[derive(Debug, Clone, Copy)]
enum Cmap {
    Format4 { offset: usize, symbol: bool },
    Format0 { offset: usize },
}

pub struct Font {
    data: Vec<u8>,
    pub units_per_em: f32,
    pub ascender: f32,
    pub descender: f32,
    #[allow(dead_code)]
    pub line_gap: f32,
    num_glyphs: u16,
    loca_long: bool,
    loca: usize,
    glyf: usize,
    hmtx: usize,
    num_hmetrics: u16,
    cmap: Cmap,
}

impl Font {
    pub fn parse(data: Vec<u8>) -> Result<Self, String> {
        let bad = || "truncated TrueType file".to_string();
        let num_tables = u16_at(&data, 4).ok_or_else(bad)? as usize;
        let table = |tag: &[u8; 4]| -> Option<usize> {
            (0..num_tables).find_map(|i| {
                let rec = 12 + i * 16;
                (data.get(rec..rec + 4)? == tag).then(|| u32_at(&data, rec + 8).map(|o| o as usize))?
            })
        };
        let head = table(b"head").ok_or("missing head table")?;
        let hhea = table(b"hhea").ok_or("missing hhea table")?;
        let maxp = table(b"maxp").ok_or("missing maxp table")?;
        let loca = table(b"loca").ok_or("missing loca table (CFF fonts unsupported)")?;
        let glyf = table(b"glyf").ok_or("missing glyf table")?;
        let hmtx = table(b"hmtx").ok_or("missing hmtx table")?;
        let cmap_off = table(b"cmap").ok_or("missing cmap table")?;
        let units_per_em = u16_at(&data, head + 18).ok_or_else(bad)?.max(16) as f32;
        let loca_long = i16_at(&data, head + 50).ok_or_else(bad)? != 0;
        let ascender = i16_at(&data, hhea + 4).ok_or_else(bad)? as f32;
        let descender = i16_at(&data, hhea + 6).ok_or_else(bad)? as f32;
        let line_gap = i16_at(&data, hhea + 8).ok_or_else(bad)? as f32;
        let num_hmetrics = u16_at(&data, hhea + 34).ok_or_else(bad)?.max(1);
        let num_glyphs = u16_at(&data, maxp + 4).ok_or_else(bad)?;

        let n_sub = u16_at(&data, cmap_off + 2).ok_or_else(bad)? as usize;
        let mut best: Option<(u8, Cmap)> = None;
        for i in 0..n_sub {
            let rec = cmap_off + 4 + i * 8;
            let (Some(pid), Some(eid), Some(off)) =
                (u16_at(&data, rec), u16_at(&data, rec + 2), u32_at(&data, rec + 4))
            else {
                continue;
            };
            let sub = cmap_off + off as usize;
            let cand = match u16_at(&data, sub) {
                Some(4) => {
                    let symbol = pid == 3 && eid == 0;
                    let rank = match (pid, eid) {
                        (3, 1) => 4,
                        (0, _) => 3,
                        (3, 0) => 2,
                        _ => 1,
                    };
                    Some((rank, Cmap::Format4 { offset: sub, symbol }))
                }
                Some(0) => Some((0, Cmap::Format0 { offset: sub })),
                _ => None,
            };
            if let Some((rank, c)) = cand
                && best.as_ref().is_none_or(|(r, _)| rank > *r)
            {
                best = Some((rank, c));
            }
        }
        let cmap = best.map(|(_, c)| c).ok_or("no supported cmap subtable (format 4/0)")?;
        Ok(Self {
            data,
            units_per_em,
            ascender,
            descender,
            line_gap,
            num_glyphs,
            loca_long,
            loca,
            glyf,
            hmtx,
            num_hmetrics,
            cmap,
        })
    }

    pub fn glyph_index(&self, c: char) -> u16 {
        let code = c as u32;
        match self.cmap {
            Cmap::Format0 { offset } => {
                if code < 256 {
                    self.data.get(offset + 6 + code as usize).copied().unwrap_or(0) as u16
                } else {
                    0
                }
            }
            Cmap::Format4 { offset, symbol } => {
                let g = self.format4_lookup(offset, code);
                if g == 0 && symbol { self.format4_lookup(offset, 0xF000 | code) } else { g }
            }
        }
    }

    fn format4_lookup(&self, off: usize, code: u32) -> u16 {
        if code > 0xFFFF {
            return 0;
        }
        let d = &self.data;
        let Some(seg_x2) = u16_at(d, off + 6).map(|v| v as usize) else {
            return 0;
        };
        let ends = off + 14;
        let starts = ends + seg_x2 + 2;
        let deltas = starts + seg_x2;
        let ranges = deltas + seg_x2;
        for i in 0..seg_x2 / 2 {
            let Some(end) = u16_at(d, ends + 2 * i) else {
                return 0;
            };
            if code > end as u32 {
                continue;
            }
            let start = u16_at(d, starts + 2 * i).unwrap_or(0xFFFF) as u32;
            if code < start {
                return 0;
            }
            let delta = u16_at(d, deltas + 2 * i).unwrap_or(0) as u32;
            let ro_addr = ranges + 2 * i;
            let ro = u16_at(d, ro_addr).unwrap_or(0) as usize;
            if ro == 0 {
                return ((code + delta) & 0xFFFF) as u16;
            }
            let addr = ro_addr + ro + 2 * (code - start) as usize;
            let g = u16_at(d, addr).unwrap_or(0) as u32;
            return if g == 0 { 0 } else { ((g + delta) & 0xFFFF) as u16 };
        }
        0
    }

    /// Advance width in font units.
    pub fn advance(&self, gid: u16) -> f32 {
        let i = (gid.min(self.num_hmetrics - 1)) as usize;
        u16_at(&self.data, self.hmtx + i * 4).unwrap_or(0) as f32
    }

    fn glyph_range(&self, gid: u16) -> Option<(usize, usize)> {
        if gid >= self.num_glyphs {
            return None;
        }
        let g = gid as usize;
        let (a, b) = if self.loca_long {
            (u32_at(&self.data, self.loca + g * 4)? as usize, u32_at(&self.data, self.loca + g * 4 + 4)? as usize)
        } else {
            (
                u16_at(&self.data, self.loca + g * 2)? as usize * 2,
                u16_at(&self.data, self.loca + g * 2 + 2)? as usize * 2,
            )
        };
        (b > a).then_some((self.glyf + a, self.glyf + b))
    }

    /// Glyph contours in font units (y up) as `(x, y, on_curve)` points.
    fn contours(&self, gid: u16, depth: u32, out: &mut Vec<Vec<(f32, f32, bool)>>) {
        if depth > 6 {
            return;
        }
        let Some((start, end)) = self.glyph_range(gid) else {
            return;
        };
        let d = &self.data[..end.min(self.data.len())];
        let Some(n) = i16_at(d, start) else {
            return;
        };
        if n >= 0 {
            self.simple_glyph(d, start, n as usize, out);
        } else {
            self.composite_glyph(d, start + 10, depth, out);
        }
    }

    fn simple_glyph(&self, d: &[u8], start: usize, n: usize, out: &mut Vec<Vec<(f32, f32, bool)>>) {
        let mut ends = Vec::with_capacity(n);
        for i in 0..n {
            let Some(e) = u16_at(d, start + 10 + 2 * i) else {
                return;
            };
            ends.push(e as usize);
        }
        let Some(&last) = ends.last() else {
            return;
        };
        let npts = last + 1;
        let Some(ilen) = u16_at(d, start + 10 + 2 * n) else {
            return;
        };
        let mut p = start + 12 + 2 * n + ilen as usize;
        let mut flags = Vec::with_capacity(npts);
        while flags.len() < npts {
            let Some(&f) = d.get(p) else {
                return;
            };
            p += 1;
            flags.push(f);
            if f & 8 != 0 {
                let Some(&rep) = d.get(p) else {
                    return;
                };
                p += 1;
                for _ in 0..rep {
                    flags.push(f);
                }
            }
        }
        flags.truncate(npts);
        let mut read_axis = |short: u8, same: u8| -> Option<Vec<f32>> {
            let mut v = 0i32;
            let mut vals = Vec::with_capacity(npts);
            for &f in &flags {
                if f & short != 0 {
                    let b = *d.get(p)? as i32;
                    p += 1;
                    v += if f & same != 0 { b } else { -b };
                } else if f & same == 0 {
                    v += i16_at(d, p)? as i32;
                    p += 2;
                }
                vals.push(v as f32);
            }
            Some(vals)
        };
        let Some(xs) = read_axis(2, 16) else {
            return;
        };
        let Some(ys) = read_axis(4, 32) else {
            return;
        };
        let mut s = 0;
        for &e in &ends {
            if e < s || e >= npts {
                return;
            }
            out.push((s..=e).map(|i| (xs[i], ys[i], flags[i] & 1 != 0)).collect());
            s = e + 1;
        }
    }

    fn composite_glyph(&self, d: &[u8], mut p: usize, depth: u32, out: &mut Vec<Vec<(f32, f32, bool)>>) {
        loop {
            let (Some(flags), Some(gid)) = (u16_at(d, p), u16_at(d, p + 2)) else {
                return;
            };
            p += 4;
            let (dx, dy) = if flags & 1 != 0 {
                let r = (i16_at(d, p), i16_at(d, p + 2));
                p += 4;
                (r.0.unwrap_or(0) as f32, r.1.unwrap_or(0) as f32)
            } else {
                let r = (d.get(p).map(|&b| b as i8), d.get(p + 1).map(|&b| b as i8));
                p += 2;
                (r.0.unwrap_or(0) as f32, r.1.unwrap_or(0) as f32)
            };
            let f2 = |o: usize| i16_at(d, o).unwrap_or(0) as f32 / 16384.0;
            let (mut a, mut b, mut c, mut dd) = (1.0, 0.0, 0.0, 1.0);
            if flags & 0x08 != 0 {
                a = f2(p);
                dd = a;
                p += 2;
            } else if flags & 0x40 != 0 {
                a = f2(p);
                dd = f2(p + 2);
                p += 4;
            } else if flags & 0x80 != 0 {
                a = f2(p);
                b = f2(p + 2);
                c = f2(p + 4);
                dd = f2(p + 6);
                p += 8;
            }
            let xy = flags & 2 != 0;
            let mut sub = Vec::new();
            self.contours(gid, depth + 1, &mut sub);
            for contour in sub {
                out.push(
                    contour
                        .into_iter()
                        .map(|(x, y, on)| {
                            let (tx, ty) = if xy { (dx, dy) } else { (0.0, 0.0) };
                            (a * x + c * y + tx, b * x + dd * y + ty, on)
                        })
                        .collect(),
                );
            }
            if flags & 0x20 == 0 {
                return;
            }
        }
    }

    /// Glyph outline flattened to closed polylines, scaled by `scale` (pixels per unit),
    /// still y-up.
    pub fn outline(&self, gid: u16, scale: f32) -> Vec<Vec<(f32, f32)>> {
        let mut raw = Vec::new();
        self.contours(gid, 0, &mut raw);
        raw.into_iter()
            .filter(|c| c.len() >= 2)
            .map(|c| flatten_contour(&c, scale))
            .collect()
    }
}

/// Expands implicit on-curve midpoints and flattens quadratic segments.
fn flatten_contour(pts: &[(f32, f32, bool)], scale: f32) -> Vec<(f32, f32)> {
    let p: Vec<(f32, f32, bool)> = pts.iter().map(|&(x, y, on)| (x * scale, y * scale, on)).collect();
    let n = p.len();
    // Start on an on-curve point (or the midpoint of two off-curve points).
    let first_on = p.iter().position(|q| q.2);
    let (start, start_idx) = match first_on {
        Some(i) => ((p[i].0, p[i].1), i),
        None => (((p[0].0 + p[1].0) * 0.5, (p[0].1 + p[1].1) * 0.5), 0),
    };
    let mut out = vec![start];
    let mut ctrl: Option<(f32, f32)> = None;
    let mut cur = start;
    let emit_quad = |out: &mut Vec<(f32, f32)>, from: (f32, f32), c: (f32, f32), to: (f32, f32)| {
        let dev = ((from.0 - 2.0 * c.0 + to.0).powi(2) + (from.1 - 2.0 * c.1 + to.1).powi(2)).sqrt();
        let steps = ((dev * 0.6).sqrt().ceil() as usize).clamp(1, 16);
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            let mt = 1.0 - t;
            out.push((
                mt * mt * from.0 + 2.0 * mt * t * c.0 + t * t * to.0,
                mt * mt * from.1 + 2.0 * mt * t * c.1 + t * t * to.1,
            ));
        }
    };
    let skip_first = first_on.is_some();
    for k in 1..=n {
        let i = (start_idx + k) % n;
        if skip_first && k == n {
            // Close back onto the starting on-curve point.
            match ctrl.take() {
                Some(c) => emit_quad(&mut out, cur, c, start),
                None => out.push(start),
            }
            break;
        }
        let q = (p[i].0, p[i].1);
        if p[i].2 {
            match ctrl.take() {
                Some(c) => emit_quad(&mut out, cur, c, q),
                None => out.push(q),
            }
            cur = q;
        } else if let Some(c) = ctrl {
            let mid = ((c.0 + q.0) * 0.5, (c.1 + q.1) * 0.5);
            emit_quad(&mut out, cur, c, mid);
            cur = mid;
            ctrl = Some(q);
        } else {
            ctrl = Some(q);
        }
    }
    if !skip_first {
        match ctrl.take() {
            Some(c) => emit_quad(&mut out, cur, c, start),
            None => out.push(start),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Coverage rasterizer
// ---------------------------------------------------------------------------

struct Raster {
    w: usize,
    h: usize,
    a: Vec<f32>,
}

impl Raster {
    fn new(w: usize, h: usize) -> Self {
        Self { w, h, a: vec![0.0; w * h + 4] }
    }

    fn add(&mut self, i: isize, v: f32) {
        if i >= 0
            && let Some(c) = self.a.get_mut(i as usize)
        {
            *c += v;
        }
    }

    /// Accumulates the signed area of one line segment (pixel coords, y down).
    fn line(&mut self, p0: (f32, f32), p1: (f32, f32)) {
        if (p0.1 - p1.1).abs() <= f32::EPSILON {
            return;
        }
        let (dir, p0, p1) = if p0.1 < p1.1 { (1.0, p0, p1) } else { (-1.0, p1, p0) };
        let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
        let mut x = p0.0;
        if p0.1 < 0.0 {
            x -= p0.1 * dxdy;
        }
        let y0 = p0.1.max(0.0) as usize;
        let y1 = (p1.1.ceil().max(0.0) as usize).min(self.h);
        for y in y0..y1 {
            let ls = (y * self.w) as isize;
            let dy = ((y + 1) as f32).min(p1.1) - (y as f32).max(p0.1);
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (x0, x1) = if x < xnext { (x, xnext) } else { (xnext, x) };
            let x0f = x0.floor();
            let x0i = x0f as isize;
            let x1c = x1.ceil();
            let x1i = x1c as isize;
            if x1i <= x0i + 1 {
                let xmf = 0.5 * (x + xnext) - x0f;
                self.add(ls + x0i, d - d * xmf);
                self.add(ls + x0i + 1, d * xmf);
            } else {
                let s = (x1 - x0).recip();
                let fr = x0 - x0f;
                let a0 = 0.5 * s * (1.0 - fr) * (1.0 - fr);
                let x1f = x1 - x1c + 1.0;
                let am = 0.5 * s * x1f * x1f;
                self.add(ls + x0i, d * a0);
                if x1i == x0i + 2 {
                    self.add(ls + x0i + 1, d * (1.0 - a0 - am));
                } else {
                    let a1 = s * (1.5 - fr);
                    self.add(ls + x0i + 1, d * (a1 - a0));
                    for xi in x0i + 2..x1i - 1 {
                        self.add(ls + xi, d * s);
                    }
                    let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                    self.add(ls + x1i - 1, d * (1.0 - a2 - am));
                }
                self.add(ls + x1i, d * am);
            }
            x = xnext;
        }
    }

    fn coverage(&self) -> Vec<u8> {
        let mut acc = 0.0f32;
        self.a[..self.w * self.h]
            .iter()
            .map(|v| {
                acc += v;
                (acc.abs().min(1.0) * 255.0 + 0.5) as u8
            })
            .collect()
    }
}

/// One rasterized glyph: coverage bitmap plus placement relative to the pen position on
/// the baseline (`left`, `top` in pixels, y down).
pub struct GlyphBitmap {
    pub width: usize,
    pub height: usize,
    pub left: i32,
    pub top: i32,
    pub advance: f32,
    pub coverage: Vec<u8>,
}

impl Font {
    pub fn rasterize(&self, c: char, px: f32) -> GlyphBitmap {
        let scale = px / self.units_per_em;
        let gid = self.glyph_index(c);
        let advance = self.advance(gid) * scale;
        let contours = self.outline(gid, scale);
        let pts = contours.iter().flatten();
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in pts {
            x0 = x0.min(x);
            x1 = x1.max(x);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
        if contours.is_empty() || x1 <= x0 || y1 <= y0 {
            return GlyphBitmap { width: 0, height: 0, left: 0, top: 0, advance, coverage: Vec::new() };
        }
        let left = x0.floor() as i32 - 1;
        let top = y1.ceil() as i32 + 1;
        let width = (x1.ceil() as i32 + 1 - left).max(1) as usize;
        let height = (top - (y0.floor() as i32 - 1)).max(1) as usize;
        let mut r = Raster::new(width, height);
        for c in &contours {
            for seg in c.windows(2) {
                let a = (seg[0].0 - left as f32, top as f32 - seg[0].1);
                let b = (seg[1].0 - left as f32, top as f32 - seg[1].1);
                r.line(a, b);
            }
        }
        GlyphBitmap { width, height, left, top: -top, advance, coverage: r.coverage() }
    }
}

// ---------------------------------------------------------------------------
// Glyph atlas
// ---------------------------------------------------------------------------

pub const FIRST_CHAR: u8 = 32;
pub const LAST_CHAR: u8 = 126;

#[derive(Debug, Clone, Copy, Default)]
pub struct AtlasGlyph {
    pub advance: f32,
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub uv: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct AtlasFace {
    pub px: f32,
    pub ascent: f32,
    pub descent: f32,
    pub glyphs: Vec<AtlasGlyph>,
}

impl AtlasFace {
    pub fn glyph(&self, c: char) -> &AtlasGlyph {
        let i = c as u32;
        let i = if (FIRST_CHAR as u32..=LAST_CHAR as u32).contains(&i) { i } else { b'?' as u32 };
        &self.glyphs[(i - FIRST_CHAR as u32) as usize]
    }
}

/// Bakes every `(font, pixel size)` into one RGBA atlas (white, coverage in alpha). A
/// 4x4 opaque white block at the origin serves solid rectangles (`white_uv`).
pub fn build_atlas(specs: &[(&Font, f32)], width: u32) -> (RgbaImage, Vec<AtlasFace>, [f32; 2]) {
    let w = width as usize;
    let mut rows: Vec<Vec<u8>> = Vec::new(); // alpha rows, grown on demand
    let ensure = |rows: &mut Vec<Vec<u8>>, h: usize| {
        while rows.len() < h {
            rows.push(vec![0; w]);
        }
    };
    ensure(&mut rows, 4);
    for row in rows.iter_mut().take(4) {
        row[..4].fill(255);
    }
    let (mut x, mut y, mut shelf) = (6usize, 0usize, 4usize);
    let mut placed: Vec<(usize, usize, GlyphBitmap)> = Vec::new();
    let mut faces = Vec::new();
    for (font, px) in specs {
        let scale = px / font.units_per_em;
        let mut glyphs = Vec::new();
        for code in FIRST_CHAR..=LAST_CHAR {
            let g = font.rasterize(code as char, *px);
            if g.width + 2 > w {
                glyphs.push(AtlasGlyph { advance: g.advance, ..Default::default() });
                continue;
            }
            if x + g.width + 1 > w {
                x = 0;
                y += shelf + 1;
                shelf = 0;
            }
            let idx = placed.len();
            glyphs.push(AtlasGlyph {
                advance: g.advance,
                left: g.left as f32,
                top: g.top as f32,
                width: g.width as f32,
                height: g.height as f32,
                uv: [idx as f32, 0.0, 0.0, 0.0], // patched below
            });
            shelf = shelf.max(g.height);
            placed.push((x, y, g));
            x += placed.last().map_or(0, |p| p.2.width) + 1;
        }
        faces.push(AtlasFace {
            px: *px,
            ascent: font.ascender * scale,
            descent: -font.descender * scale,
            glyphs,
        });
    }
    let total_h = (y + shelf + 1).max(4).next_power_of_two();
    ensure(&mut rows, total_h);
    let (fw, fh) = (w as f32, total_h as f32);
    for (gx, gy, g) in &placed {
        for r in 0..g.height {
            rows[gy + r][*gx..gx + g.width].copy_from_slice(&g.coverage[r * g.width..(r + 1) * g.width]);
        }
    }
    for face in &mut faces {
        for gl in &mut face.glyphs {
            if gl.width == 0.0 {
                continue;
            }
            let (gx, gy, _) = &placed[gl.uv[0] as usize];
            gl.uv = [
                *gx as f32 / fw,
                *gy as f32 / fh,
                (*gx as f32 + gl.width) / fw,
                (*gy as f32 + gl.height) / fh,
            ];
        }
    }
    let mut pixels = Vec::with_capacity(w * total_h * 4);
    for row in rows.iter().take(total_h) {
        for &a in row {
            pixels.extend_from_slice(&[255, 255, 255, a]);
        }
    }
    let img = RgbaImage { width: w as u32, height: total_h as u32, pixels };
    (img, faces, [2.0 / fw, 2.0 / fh])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raster_fills_a_square() {
        let mut r = Raster::new(8, 8);
        let sq = [(2.0, 2.0), (6.0, 2.0), (6.0, 6.0), (2.0, 6.0), (2.0, 2.0)];
        for s in sq.windows(2) {
            r.line(s[0], s[1]);
        }
        let c = r.coverage();
        assert_eq!(c[4 * 8 + 4], 255);
        assert_eq!(c[0], 0);
        assert_eq!(c[4 * 8 + 7], 0);
        let total: u32 = c.iter().map(|&v| v as u32).sum();
        assert_eq!(total, 16 * 255);
    }

    #[test]
    fn flatten_handles_implicit_midpoints() {
        // All off-curve points form a closed curve around the origin.
        let pts = [(1.0, 0.0, false), (0.0, 1.0, false), (-1.0, 0.0, false), (0.0, -1.0, false)];
        let poly = flatten_contour(&pts, 10.0);
        assert!(poly.len() > 8);
        assert_eq!(poly.first(), poly.last());
    }

    #[test]
    fn fa_arial_rasterizes_when_installed() {
        let path = crate::scd::default_game_dir().join("fonts").join("ARIAL.TTF");
        let Ok(bytes) = std::fs::read(&path) else {
            return; // no install on this machine
        };
        let f = Font::parse(bytes).expect("parse arial");
        assert_ne!(f.glyph_index('A'), 0);
        let g = f.rasterize('A', 32.0);
        assert!(g.width > 10 && g.height > 15, "{}x{}", g.width, g.height);
        assert!(g.coverage.iter().filter(|&&v| v > 200).count() > 40);
        let (img, faces, _) = build_atlas(&[(&f, 16.0)], 512);
        assert_eq!(faces[0].glyphs.len(), (LAST_CHAR - FIRST_CHAR + 1) as usize);
        assert!(faces[0].glyph('W').advance > faces[0].glyph('i').advance);
        assert_eq!(img.pixels.len(), (img.width * img.height * 4) as usize);
    }
}
