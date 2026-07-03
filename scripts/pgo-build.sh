#!/usr/bin/env bash
#
# Profile-guided optimization build. The effect is MICROARCHITECTURE-
# SPECIFIC and can go either way: measured +16% on an Ice Lake-class
# AVX-512 machine, but -8% on a Broadwell Xeon D-1541 (AVX2 path).
# The script therefore benchmarks plain vs PGO at the end and tells you
# which one won — only deploy the PGO binary if it actually wins here.
#
# PGO is a two-phase build: compile an instrumented binary, run it on a
# representative workload to record a profile, then recompile using that
# profile. Run this script on the machine you'll search on.
#
# Usage:  ./scripts/pgo-build.sh [seconds_per_mode]
# Output: ./target/release/mc-keygen        (plain --release)
#         ./target/release/mc-keygen-pgo    (PGO build)
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
cp ./target/release/mc-keygen ./target/release/mc-keygen-pgo

# Rebuild plain for an apples-to-apples comparison on THIS machine.
cargo build --release --quiet

rate() { # print keys/s of one bench cpu1 run
  "$1" bench -m cpu1 -d "$SECS" -p abc --machine-label pgo-ab 2>/dev/null \
    | grep -o '"rate_keys_per_sec":[0-9.]*' | cut -d: -f2
}
echo "==> comparing plain vs PGO on this machine (cpu1, ${SECS}s each)"
PLAIN=$(rate ./target/release/mc-keygen)
PGO=$(rate ./target/release/mc-keygen-pgo)
python3 - "$PLAIN" "$PGO" << 'PYEOF'
import sys
plain, pgo = float(sys.argv[1]), float(sys.argv[2])
delta = (pgo / plain - 1) * 100
print(f"    plain: {plain/1e6:.2f} MH/s")
print(f"    pgo:   {pgo/1e6:.2f} MH/s   ({delta:+.1f}%)")
if pgo > plain * 1.02:
    print("==> PGO wins here: use ./target/release/mc-keygen-pgo")
elif pgo < plain * 0.98:
    print("==> PGO LOSES on this CPU: keep the plain ./target/release/mc-keygen")
else:
    print("==> No significant difference: keep the plain binary")
PYEOF
