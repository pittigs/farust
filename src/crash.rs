//! Phase 57: crash reporter and build identity.
//!
//! [`install`] adds a panic hook (after the default one printed the message) that writes
//! `crashes/<utc>-<pid>/crash.txt` under the data dir: build identity, OS/arch, command
//! line, thread, panic message and location, a forced backtrace, and — when the newest
//! desync dump is less than an hour old — its path and `report.txt`, so a crash right
//! after a desync ships everything needed to replay it. Nothing is uploaded; the path is
//! printed for the player to attach. `FARUST_NO_CRASH_REPORT=1` disables it.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Build identity, e.g. `farust 0.1.0 (rules v10, protocol v17, linux/x86_64, build local)`.
/// CI sets `FARUST_BUILD_ID` at compile time (tag or commit).
pub fn version_line() -> String {
    format!(
        "farust {} (rules v{}, protocol v{}, {}/{}, build {})",
        env!("CARGO_PKG_VERSION"),
        crate::skirmish::RULES_VERSION,
        crate::net::PROTOCOL_VERSION,
        std::env::consts::OS,
        std::env::consts::ARCH,
        option_env!("FARUST_BUILD_ID").unwrap_or("local")
    )
}

/// Installs the crash-report panic hook (no-op with `FARUST_NO_CRASH_REPORT`).
pub fn install() {
    if std::env::var_os("FARUST_NO_CRASH_REPORT").is_some() {
        return;
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default_hook(info);
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "(non-string panic payload)".to_string());
        let location = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()));
        let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
        let args: Vec<String> = std::env::args().collect();
        let desync = recent_desync(&crate::savegame::data_dir("desyncs"), Duration::from_secs(3600));
        let text = render_report(&msg, location.as_deref(), &thread, &args, &backtrace, desync.as_deref());
        match write_report(&crate::savegame::data_dir("crashes"), &text) {
            Ok(path) => eprintln!("farust crashed - crash report written to {}", path.display()),
            Err(e) => eprintln!("farust crashed - could not write a crash report: {e}"),
        }
    }));
}

/// The newest desync dump directory in `dir` modified within `max_age`.
fn recent_desync(dir: &Path, max_age: Duration) -> Option<PathBuf> {
    let now = SystemTime::now();
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let m = e.metadata().ok()?.modified().ok()?;
            (e.path().is_dir() && now.duration_since(m).unwrap_or_default() <= max_age).then(|| (m, e.path()))
        })
        .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
        .map(|(_, p)| p)
}

/// Crash report text (pure, for tests).
pub fn render_report(
    msg: &str,
    location: Option<&str>,
    thread: &str,
    args: &[String],
    backtrace: &str,
    desync: Option<&Path>,
) -> String {
    let mut t = String::new();
    t.push_str("FARust crash report\n===================\n");
    t.push_str(&format!("version:  {}\n", version_line()));
    t.push_str(&format!("time:     {} (UTC stamp)\n", crate::savegame::utc_stamp()));
    t.push_str(&format!("command:  {}\n", args.join(" ")));
    t.push_str(&format!("thread:   {thread}\n"));
    t.push_str(&format!("panic:    {msg}\n"));
    t.push_str(&format!("at:       {}\n", location.unwrap_or("(unknown)")));
    if let Some(d) = desync {
        t.push_str(&format!("\nrecent desync dump: {}\n", d.display()));
        if let Ok(r) = std::fs::read_to_string(d.join("report.txt")) {
            t.push_str("--- report.txt ---\n");
            t.push_str(&r.chars().take(64 * 1024).collect::<String>());
            t.push('\n');
        }
    }
    t.push_str("\nbacktrace:\n");
    t.push_str(backtrace);
    t.push('\n');
    t
}

fn write_report(base: &Path, text: &str) -> Result<PathBuf, String> {
    let dir = base.join(format!("{}-{}", crate::savegame::utc_stamp(), std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("crash.txt");
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

/// `farust crash-test`: panics on purpose to exercise the crash reporter.
pub fn cmd_crash_test() -> Result<(), String> {
    panic!("farust crash-test: deliberate panic to exercise the crash reporter");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_has_identity_panic_site_and_recent_desync() {
        let base = std::env::temp_dir().join(format!("farust-crash-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let desyncs = base.join("desyncs");
        let dump = desyncs.join("20261127-army1");
        std::fs::create_dir_all(&dump).unwrap();
        std::fs::write(dump.join("report.txt"), "LOCKSTEP DESYNC at turn 7").unwrap();
        let found = recent_desync(&desyncs, Duration::from_secs(3600)).expect("recent dump");
        assert_eq!(found, dump);
        assert!(recent_desync(&base.join("none"), Duration::from_secs(3600)).is_none());

        let args = vec!["farust".to_string(), "join".to_string()];
        let text = render_report("boom", Some("src/x.rs:1:2"), "main", &args, "0: frame", Some(&found));
        assert!(text.contains(&version_line()));
        assert!(text.contains("panic:    boom") && text.contains("at:       src/x.rs:1:2"));
        assert!(text.contains("command:  farust join"));
        assert!(text.contains("LOCKSTEP DESYNC at turn 7"), "{text}");
        assert!(text.contains("backtrace:\n0: frame"));
        let path = write_report(&base.join("crashes"), &text).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        assert!(version_line().starts_with(&format!("farust {} (rules v", env!("CARGO_PKG_VERSION"))));
        let _ = std::fs::remove_dir_all(&base);
    }
}
