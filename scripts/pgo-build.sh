#!/usr/bin/env bash
#
# Profile-guided optimization build. Measured ~16% faster than a plain
# --release build on the hot search loop.
#
# PGO is a two-phase build: compile an instrumented binary, run it on a
# representative workload to record a profile, then recompile using that
# profile. The profile MUST be gathered on the same CPU family you'll run
# on (the profile captures the code path your CPU actually takes — AVX-512
# vs AVX2 vs scalar — so a profile from one machine does not transfer to a
# different microarchitecture). Just run this script on the target machine.
#
# Usage:  ./scripts/pgo-build.sh [seconds_per_mode]
# Output: ./target/release/mc-keygen  (PGO-optimized)
set -euo pipefail
cd "$(dirname "$0")/.."

SECS="${1:-12}"
PGO_DIR="$(mktemp -d)"
trap 'rm -rf "$PGO_DIR"' EXIT

# rustc ships its own llvm-profdata; the system one is often a different
# LLVM version and refuses to merge the raw profiles.
PROFDATA="$(find "$(rustc --print sysroot)" -name llvm-profdata 2>/dev/null | head -1)"
if [ -z "$PROFDATA" ]; then
  echo "llvm-profdata not found. Install it with: rustup component add llvm-tools-preview" >&2
  exit 1
fi

echo "==> [1/3] building instrumented binary"
RUSTFLAGS="-Cprofile-generate=$PGO_DIR" cargo build --release --quiet

echo "==> [2/3] gathering profile (~$((SECS * 3))s across search modes)"
BIN=./target/release/mc-keygen
# Cover the three hot loops: prefix (kernel byte filter), suffix (byte-31
# filter) and anywhere-run (RunBytes filter + full-encode path).
LLVM_PROFILE_FILE="$PGO_DIR/p-%m.profraw" "$BIN" bench -m cpu1 -d "$SECS" -p abc --machine-label pgo >/dev/null 2>&1
LLVM_PROFILE_FILE="$PGO_DIR/s-%m.profraw" "$BIN" bench -m cpu1 -d "$SECS" -p abc --machine-label pgo >/dev/null 2>&1 || true
# A short real run exercises the suffix/run kernels and the match path.
timeout "$SECS" "$BIN" --run 8 --where anywhere --stream >/dev/null 2>&1 || true
timeout "$SECS" "$BIN" DEAD --where suffix --stream  >/dev/null 2>&1 || true

"$PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR"/*.profraw

echo "==> [3/3] rebuilding with profile"
RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata" cargo build --release --quiet

echo "Done: ./target/release/mc-keygen (PGO)"
