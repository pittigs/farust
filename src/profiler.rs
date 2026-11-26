//! Phase 25 lightweight scoped profiler.
//!
//! * `let _p = profiler::scope("name");` times the enclosing block. When profiling is disabled
//!   (the default) this is a single relaxed atomic load and nothing is recorded.
//! * Spans are thread-aware: each thread buffers into a thread-local vector and flushes into a
//!   shared sink when its outermost span closes, so worker-pool spans cost no lock per span.
//! * [`end_frame`] folds all spans recorded since the previous call into a per-frame record kept
//!   in a fixed-size ring buffer (frame-time graph) and into running totals (CLI tables).
//! * [`record_ms`] injects externally measured durations (GPU timestamp queries).
//! * [`append_overlay`] draws the F12 overlay (frame graph + top spans) with the HUD helpers.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Frames kept in the ring buffer (≈4 s at 60 fps).
pub const RING_FRAMES: usize = 240;

static ENABLED: AtomicBool = AtomicBool::new(false);
static OVERLAY: AtomicBool = AtomicBool::new(false);
static NEXT_THREAD: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy, Debug)]
struct Event {
    name: &'static str,
    dur_ns: u64,
    depth: u16,
    thread: u32,
}

struct ThreadBuf {
    id: u32,
    depth: u16,
    events: Vec<Event>,
}

thread_local! {
    static TL: RefCell<ThreadBuf> = RefCell::new(ThreadBuf {
        id: NEXT_THREAD.fetch_add(1, Ordering::Relaxed),
        depth: 0,
        events: Vec::with_capacity(64),
    });
}

/// Aggregated per-name statistics.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpanStat {
    pub total_ms: f64,
    pub calls: u64,
    pub max_ms: f64,
    /// Bit set of threads (mod 64) that recorded this span.
    pub threads: u64,
    /// Nesting depth of the first occurrence (for indentation).
    pub depth: u16,
}

#[derive(Clone, Debug, Default)]
pub struct FrameRecord {
    pub frame_ms: f32,
    pub spans: Vec<(&'static str, SpanStat)>,
}

#[derive(Default)]
struct Sink {
    pending: Vec<Event>,
    ring: VecDeque<FrameRecord>,
    totals: BTreeMap<&'static str, SpanStat>,
    frames: u64,
    last_frame: Option<Instant>,
}

fn sink() -> &'static Mutex<Sink> {
    static S: OnceLock<Mutex<Sink>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(Sink::default()))
}

fn lock() -> std::sync::MutexGuard<'static, Sink> {
    sink().lock().unwrap_or_else(|p| p.into_inner())
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// F12: toggles the in-game overlay (and enables recording while it is shown).
pub fn toggle_overlay() -> bool {
    let on = !OVERLAY.load(Ordering::Relaxed);
    OVERLAY.store(on, Ordering::Relaxed);
    if on {
        set_enabled(true);
    } else if std::env::var_os("FARUST_PROFILE").is_none() {
        set_enabled(false);
    }
    on
}

pub fn overlay_visible() -> bool {
    OVERLAY.load(Ordering::Relaxed)
}

/// Enables recording when `FARUST_PROFILE` is set (and the overlay when it equals `overlay`).
pub fn init_from_env() {
    if let Some(v) = std::env::var_os("FARUST_PROFILE") {
        set_enabled(true);
        if v == "overlay" {
            OVERLAY.store(true, Ordering::Relaxed);
        }
    }
}

/// RAII span guard returned by [`scope`].
#[must_use = "the span ends when this guard is dropped"]
pub struct Span {
    start: Option<Instant>,
    name: &'static str,
}

#[inline]
pub fn scope(name: &'static str) -> Span {
    if !enabled() {
        return Span { start: None, name };
    }
    TL.with(|t| t.borrow_mut().depth += 1);
    Span {
        start: Some(Instant::now()),
        name,
    }
}

impl Drop for Span {
    #[inline]
    fn drop(&mut self) {
        let Some(start) = self.start else {
            return;
        };
        let dur_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let flush = TL.with(|t| {
            let mut t = t.borrow_mut();
            t.depth = t.depth.saturating_sub(1);
            let ev = Event {
                name: self.name,
                dur_ns,
                depth: t.depth,
                thread: t.id,
            };
            t.events.push(ev);
            (t.depth == 0 || t.events.len() >= 4096).then(|| std::mem::take(&mut t.events))
        });
        if let Some(evs) = flush {
            let mut s = lock();
            // Bounded when nobody closes frames (e.g. a long headless run with profiling on).
            if s.pending.len() < MAX_PENDING {
                s.pending.extend(evs);
            } else {
                fold_into_totals(&mut s, &evs);
            }
        }
    }
}

/// Pending (not yet frame-closed) events kept before spilling straight into the totals.
const MAX_PENDING: usize = 1 << 18;

fn fold_into_totals(s: &mut Sink, events: &[Event]) {
    for (name, st) in fold(events) {
        let t = s.totals.entry(name).or_insert_with(|| SpanStat {
            depth: st.depth,
            ..SpanStat::default()
        });
        t.total_ms += st.total_ms;
        t.calls += st.calls;
        t.max_ms = t.max_ms.max(st.max_ms);
        t.threads |= st.threads;
        t.depth = t.depth.min(st.depth);
    }
}

/// Records an externally measured duration (e.g. GPU timestamps) into the current frame.
pub fn record_ms(name: &'static str, ms: f64) {
    if !enabled() || !ms.is_finite() || ms < 0.0 {
        return;
    }
    let thread = TL.with(|t| t.borrow().id);
    lock().pending.push(Event {
        name,
        dur_ns: (ms * 1e6) as u64,
        depth: 0,
        thread,
    });
}

fn fold(events: &[Event]) -> Vec<(&'static str, SpanStat)> {
    let mut order: Vec<&'static str> = Vec::new();
    let mut map: BTreeMap<&'static str, SpanStat> = BTreeMap::new();
    for e in events {
        let ms = e.dur_ns as f64 / 1e6;
        let s = map.entry(e.name).or_insert_with(|| {
            order.push(e.name);
            SpanStat {
                depth: e.depth,
                ..SpanStat::default()
            }
        });
        s.total_ms += ms;
        s.calls += 1;
        s.max_ms = s.max_ms.max(ms);
        s.threads |= 1u64 << (e.thread % 64);
        s.depth = s.depth.min(e.depth);
    }
    order.into_iter().map(|n| (n, map[n])).collect()
}

/// Closes the current frame. `frame_ms` overrides the measured wall time between calls.
pub fn end_frame(frame_ms: Option<f64>) {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    let mut s = lock();
    let measured = s
        .last_frame
        .map(|t| now.duration_since(t).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    s.last_frame = Some(now);
    let events = std::mem::take(&mut s.pending);
    let spans = fold(&events);
    for (name, st) in &spans {
        let t = s.totals.entry(name).or_insert_with(|| SpanStat {
            depth: st.depth,
            ..SpanStat::default()
        });
        t.total_ms += st.total_ms;
        t.calls += st.calls;
        t.max_ms = t.max_ms.max(st.total_ms);
        t.threads |= st.threads;
        t.depth = t.depth.min(st.depth);
    }
    s.frames += 1;
    if s.ring.len() == RING_FRAMES {
        s.ring.pop_front();
    }
    s.ring.push_back(FrameRecord {
        frame_ms: frame_ms.unwrap_or(measured) as f32,
        spans,
    });
}

/// Clears the ring buffer and the running totals.
pub fn reset() {
    let mut s = lock();
    *s = Sink::default();
}

/// Frame times in the ring buffer, oldest first.
pub fn frame_times() -> Vec<f32> {
    lock().ring.iter().map(|f| f.frame_ms).collect()
}

/// Running totals since the last [`reset`]: (name, stat) in first-seen order plus frame count.
pub fn totals() -> (Vec<(&'static str, SpanStat)>, u64) {
    let s = lock();
    let mut v: Vec<(&'static str, SpanStat)> = s.totals.iter().map(|(k, v)| (*k, *v)).collect();
    v.sort_by(|a, b| b.1.total_ms.total_cmp(&a.1.total_ms));
    (v, s.frames)
}

/// Average per frame over the last `frames` ring entries, heaviest first.
pub fn top_spans(frames: usize, n: usize) -> Vec<(&'static str, f64, f64)> {
    let s = lock();
    let take = s.ring.len().min(frames.max(1));
    let mut acc: BTreeMap<&'static str, (f64, f64)> = BTreeMap::new();
    for f in s.ring.iter().rev().take(take) {
        for (name, st) in &f.spans {
            let e = acc.entry(name).or_default();
            e.0 += st.total_ms;
            e.1 = e.1.max(st.total_ms);
        }
    }
    let mut v: Vec<(&'static str, f64, f64)> = acc
        .into_iter()
        .map(|(k, (sum, mx))| (k, sum / take.max(1) as f64, mx))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(n);
    v
}

/// Formats the running totals as a text table (`avg ms/frame`, calls/frame, max, threads).
/// `per` overrides the divisor (e.g. sim ticks when frames were not closed per tick).
pub fn report_table(per: Option<u64>) -> String {
    let (spans, frames) = totals();
    let frames = per.unwrap_or(frames).max(1) as f64;
    let mut out = format!(
        "{:<34} {:>10} {:>10} {:>10} {:>4}\n",
        "span", "avg ms", "max ms", "calls/f", "thr"
    );
    for (name, st) in spans {
        out.push_str(&format!(
            "{:<34} {:>10.3} {:>10.3} {:>10.1} {:>4}\n",
            format!("{}{}", "  ".repeat(st.depth.min(6) as usize), name),
            st.total_ms / frames,
            st.max_ms,
            st.calls as f64 / frames,
            st.threads.count_ones()
        ));
    }
    out
}

/// Percentile (0..=100) of a sample set (nearest-rank on a sorted copy).
pub fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * (v.len() as f64 - 1.0)).round() as usize;
    v[rank.min(v.len() - 1)]
}

/// Draws the F12 overlay: frame-time graph with 16.7/33.3 ms guides and the top spans.
pub fn append_overlay(
    bg_v: &mut Vec<crate::viewer::HudVertex>,
    bg_i: &mut Vec<u32>,
    fg_v: &mut Vec<crate::viewer::HudVertex>,
    fg_i: &mut Vec<u32>,
    w: f32,
    h: f32,
) {
    use crate::hud::{draw_text, push_rect};
    if !overlay_visible() {
        return;
    }
    let times = frame_times();
    let top = top_spans(60, 12);
    let (x0, y0, pw) = (12.0, 64.0, 360.0);
    let graph_h = 70.0;
    let ph = graph_h + 34.0 + top.len() as f32 * 12.0 + 8.0;
    if h < y0 + ph || w < x0 + pw {
        return;
    }
    push_rect(bg_v, bg_i, x0, y0, pw, ph, [0.02, 0.03, 0.05, 0.82], w, h);
    let gx = x0 + 8.0;
    let gy = y0 + 22.0;
    let gw = pw - 16.0;
    push_rect(bg_v, bg_i, gx, gy, gw, graph_h, [0.07, 0.09, 0.12, 0.95], w, h);
    let scale_ms = 50.0f32;
    for (ms, col) in [(16.7f32, [0.3, 0.8, 0.4, 0.6]), (33.3, [0.9, 0.7, 0.2, 0.6])] {
        let y = gy + graph_h - (ms / scale_ms) * graph_h;
        push_rect(bg_v, bg_i, gx, y, gw, 1.0, col, w, h);
    }
    let n = times.len().max(1);
    let bar_w = gw / RING_FRAMES as f32;
    let off = RING_FRAMES.saturating_sub(n) as f32 * bar_w;
    for (i, &ms) in times.iter().enumerate() {
        let bh = (ms / scale_ms).clamp(0.0, 1.0) * graph_h;
        let col = if ms <= 17.0 {
            [0.30, 0.85, 0.45, 0.95]
        } else if ms <= 34.0 {
            [0.95, 0.78, 0.22, 0.95]
        } else {
            [0.95, 0.28, 0.22, 0.95]
        };
        push_rect(bg_v, bg_i, gx + off + i as f32 * bar_w, gy + graph_h - bh, bar_w.max(1.0), bh, col, w, h);
    }
    let samples: Vec<f64> = times.iter().map(|&t| f64::from(t)).collect();
    let last = times.last().copied().unwrap_or(0.0);
    let head = format!(
        "PROFILER F12  {:.1} MS  P95 {:.1}  {:.0} FPS",
        last,
        percentile(&samples, 95.0),
        if last > 0.0 { 1000.0 / last } else { 0.0 }
    );
    draw_text(fg_v, fg_i, x0 + 8.0, y0 + 6.0, 1.6, &head, [0.85, 0.95, 1.0, 1.0], w, h);
    let mut y = gy + graph_h + 8.0;
    draw_text(fg_v, fg_i, gx, y, 1.4, "SPAN                     AVG MS   MAX MS", [0.6, 0.7, 0.8, 1.0], w, h);
    y += 14.0;
    for (name, avg, mx) in top {
        let line = format!("{:<24} {:>7.2} {:>8.2}", name.to_ascii_uppercase(), avg, mx);
        draw_text(fg_v, fg_i, gx, y, 1.4, &line, [0.92, 0.92, 0.88, 1.0], w, h);
        y += 12.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_fold_into_frames_and_totals() {
        // Tests run in parallel; the global profiler is only exercised here.
        set_enabled(true);
        {
            let _a = scope("test_outer");
            {
                let _b = scope("test_inner");
            }
            let _c = scope("test_inner");
        }
        std::thread::spawn(|| {
            let _w = scope("test_worker");
        })
        .join()
        .unwrap();
        record_ms("test_gpu", 1.5);
        end_frame(Some(10.0));
        let (tot, frames) = totals();
        assert!(frames >= 1);
        let inner = tot.iter().find(|(n, _)| *n == "test_inner").unwrap().1;
        assert_eq!(inner.calls, 2);
        assert_eq!(inner.depth, 1);
        assert!(tot.iter().any(|(n, _)| *n == "test_worker"));
        let gpu = tot.iter().find(|(n, _)| *n == "test_gpu").unwrap().1;
        assert!((gpu.total_ms - 1.5).abs() < 1e-6);
        assert!(frame_times().contains(&10.0));
        assert!(report_table(None).contains("test_outer"));
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 50.0), 3.0);
    }

    #[test]
    fn disabled_scope_records_nothing() {
        let s = Span { start: None, name: "x" };
        drop(s);
    }
}
