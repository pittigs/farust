#!/usr/bin/env bash
# Phase 57: builds a release archive for one target into dist/.
#
#   packaging/package.sh                         # host target
#   packaging/package.sh x86_64-pc-windows-gnu   # cross target (needs its linker)
#
# Output: dist/farust-<version>-<os>-<arch>.tar.gz (.zip for Windows), plus
# dist/farust-<version>-x86_64.AppImage on Linux when `appimagetool` is on PATH.
# Set FARUST_BUILD_ID (tag or commit) to stamp `farust version`.
set -euo pipefail

cd "$(dirname "$0")/.."
target="${1:-}"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n1)"
export FARUST_BUILD_ID="${FARUST_BUILD_ID:-$(git describe --tags --always --dirty 2>/dev/null || echo local)}"

if [[ -n "$target" ]]; then
  cargo build --release --locked --target "$target"
  bin_dir="target/$target/release"
  os_arch="$target"
else
  cargo build --release --locked
  bin_dir="target/release"
  os_arch="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
fi

exe="farust"
case "$os_arch" in *windows*) exe="farust.exe" ;; esac

name="farust-$version-$os_arch"
stage="dist/$name"
rm -rf "$stage"
mkdir -p "$stage/docs" "$stage/missions" "$stage/examples"
cp "$bin_dir/$exe" "$stage/"
cp README.md "$stage/"
cp docs/PLAN.md docs/RELEASE.md "$stage/docs/"
cp missions/*.farust-mission "$stage/missions/"
cp -r examples/mods "$stage/examples/"
[[ -f LICENSE ]] && cp LICENSE "$stage/"
cat > "$stage/FIRST-RUN.txt" <<'TXT'
FARust needs an installed copy of Supreme Commander: Forged Alliance (Steam or GOG).
It is found automatically (run `farust platform` to see where it looks); otherwise
set FA_PATH to the install folder. The game files are only read, never modified.
Start the main menu with `farust` (or `farust.exe`).
TXT

case "$os_arch" in
  *windows*)
    (cd dist && python3 -m zipfile -c "$name.zip" "$name")
    echo "dist/$name.zip"
    ;;
  *)
    tar -C dist -czf "dist/$name.tar.gz" "$name"
    echo "dist/$name.tar.gz"
    ;;
esac

# Optional AppImage (Linux x86_64 host builds).
if [[ -z "$target" && "$(uname -s)" == "Linux" ]] && command -v appimagetool >/dev/null; then
  appdir="dist/AppDir"
  rm -rf "$appdir"
  mkdir -p "$appdir/usr/bin" "$appdir/usr/share/applications" "$appdir/usr/share/metainfo"
  cp "$bin_dir/farust" "$appdir/usr/bin/"
  cp packaging/linux/org.farust.Farust.desktop "$appdir/" "$appdir/usr/share/applications/"
  cp packaging/linux/org.farust.Farust.metainfo.xml "$appdir/usr/share/metainfo/"
  cp assets/icon.svg "$appdir/org.farust.Farust.svg"
  printf '#!/bin/sh\nexec "$(dirname "$(readlink -f "$0")")/usr/bin/farust" "$@"\n' > "$appdir/AppRun"
  chmod +x "$appdir/AppRun"
  ARCH="$(uname -m)" appimagetool "$appdir" "dist/farust-$version-$(uname -m).AppImage"
  echo "dist/farust-$version-$(uname -m).AppImage"
fi
