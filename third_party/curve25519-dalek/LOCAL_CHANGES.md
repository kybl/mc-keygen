# Local fork of curve25519-dalek 5.0.0-pre.6

Vendored and wired in via `[patch.crates-io]` in the repo root `Cargo.toml`.

The **only** local change is two added public methods on `EdwardsPoint`
(`src/edwards.rs`), used by the CPU vanity-search hot loop in `src/search.rs`:

- `compress_batch_y_only<const N>(&[EdwardsPoint; N]) -> [[u8; 32]; N]`
  Like `compress_batch`, but encodes only the y-coordinate and leaves the sign
  bit zero, skipping the `X·Z⁻¹` multiply per point (one field mul saved per
  key). The prefix scan only reads the low bytes; the true signed pubkey is
  recomputed with `compress()` on the rare hit.

- `chain_compress_y_only<const N>(&self, step) -> ([[u8; 32]; N], EdwardsPoint)`
  Fused fixed-step walk + y-only compression. Walks `self + i·step`, derives
  the niels form of `step` once (one field mul saved per link versus repeated
  `EdwardsPoint + EdwardsPoint`), and keeps only each point's `(Y, Z)` so the
  pre-compression working set is half the size of an `[EdwardsPoint; N]`
  staging buffer — friendlier to L1/L2. This is the method the hot loop calls;
  `compress_batch_y_only` is retained as a standalone and for the invariant test.

Both are guarded by the `compress_batch_y_only_matches_canonical` and
`search_handle_scalar_matches_pubkey` tests in the parent crate.

To re-sync with upstream: re-vendor the target version and re-apply these two
methods. Nothing else in the crate is modified.
