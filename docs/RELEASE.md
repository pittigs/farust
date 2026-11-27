# FARust 1.0 release checklist

The release pipeline is `.github/workflows/release.yml` (tag `v*` → tests on every
platform → archives → draft GitHub release). Locally, `packaging/package.sh` builds the
same archive for the host (or a given `--target`).

## Decisions only the maintainer can make

- [ ] **License.** The repository has no `LICENSE` yet. Pick one, add `LICENSE`, set
      `license = "…"` in `Cargo.toml` and `<project_license>` in
      `packaging/linux/org.farust.Farust.metainfo.xml`. Packaging copies `LICENSE`
      automatically once it exists.
- [ ] **App id / names.** `org.farust.Farust` is a placeholder id used by the desktop
      file, AppStream metadata and Flatpak manifest; change it if you own a domain or
      GitHub org, and add a `<url type="homepage">` to the metainfo (AppStream warns without one).
- [ ] **Remote + CI.** Push the branch to a GitHub repository; CI (`ci.yml`) runs on push.
- [ ] **Steam (Phase 46)** stays out of 1.0 until a Steam App ID exists.

## Before tagging

- [ ] `Cargo.toml` `version = "1.0.0"`; update the `<release>` entry in the metainfo.
- [ ] CI green on all runners: Linux x86_64 + aarch64, Windows, macOS arm64 + x86_64
      (tests incl. `golden_hashes_pin_cross_platform_determinism`, sim golden
      `35db8f2bd7a6faf2`, parser fuzzers).
- [ ] Cross-platform lockstep: host on Linux, join from Windows and macOS
      (`farust host 0.0.0.0:15000 SCMP_001 --players 3 --headless-ticks 6000`, then
      `farust join HOST:15000` on each): all peers print the same final hash.
- [ ] With game data: `farust skirmish SCMP_001` → `hash: 0fe934990406d829` (or a
      documented new golden after an intentional rules change, with `RULES_VERSION`
      and `PROTOCOL_VERSION` bumped).
- [ ] Campaign: `farust mission missions/x1ca_001.farust-mission --human-ai hard` →
      `Victory(0)`, all six primary objectives complete.
- [ ] Reconnect: kill a client mid-match and `join … --rejoin N`; the final hash equals
      an uninterrupted run (Phase 45).
- [ ] Mod sync: host with `--mod examples/mods/heavy_strikers`, client without it;
      both finish on the same hash (Phase 51).
- [ ] Performance: `farust bench-render SCMP_001 --units 10000` p50 ≤ 16.6 ms on the
      reference GPU; `farust bench-scale` ≥ 60 ticks/s at 10k units.
- [ ] Crash reporter: `farust crash-test` writes `<data>/crashes/…/crash.txt`.
- [ ] Clippy backlog: decide whether 1.0 ships with the pre-existing warnings
      (CI reports them without failing) or clears them and adds `-D warnings`.
- [ ] README "getting started" reviewed; `FIRST-RUN.txt` in the archive is accurate.

## Tag and publish

```sh
git tag -a v1.0.0 -m "FARust 1.0.0"
git push origin v1.0.0          # release.yml builds, tests and drafts the release
```

- [ ] Download every artifact from the draft release, run `farust version` (shows
      `build v1.0.0`) and start the menu on each OS once.
- [ ] Publish the draft.
