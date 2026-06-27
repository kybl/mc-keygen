//! Experimental SIMD-across-keys Ed25519 field arithmetic.
//!
//! The scalar hot loop is throughput-bound on field multiplies, and
//! curve25519-dalek's AVX2 backend only vectorizes *within* one point (its 4
//! coordinates), which does not help here. This module takes the GPU's
//! approach instead: the radix-2^25.5 ref10 field (10 signed limbs) processed
//! **4 independent keys at a time** in AVX2 lanes (one key per 64-bit lane).
//!
//! Stage 1 (this file): the field multiply, validated against the reference
//! field (`EdwardsPoint::field_mul_reference`) and microbenchmarked to measure
//! the throughput ceiling before point arithmetic is built on top. If 4-wide
//! `mul4` is not meaningfully faster than 4 scalar muls, the whole approach is
//! a dead end and nothing further should be built.

#![allow(dead_code)]

/// One field element in radix 2^25.5: limbs 0,2,4,6,8 hold 26 bits and limbs
/// 1,3,5,7,9 hold 25 bits, all signed. This is the ref10 representation.
pub type Fe = [i32; 10];

/// The field element `1`.
pub const FE_ONE: Fe = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];

#[inline]
fn load3(b: &[u8]) -> i64 {
    (b[0] as i64) | ((b[1] as i64) << 8) | ((b[2] as i64) << 16)
}

#[inline]
fn load4(b: &[u8]) -> i64 {
    (b[0] as i64) | ((b[1] as i64) << 8) | ((b[2] as i64) << 16) | ((b[3] as i64) << 24)
}

/// Decode 32 little-endian bytes into a reduced field element (ref10
/// `fe_frombytes`; the top bit of byte 31 is ignored).
pub fn fe_frombytes(s: &[u8; 32]) -> Fe {
    let mut h = [0i64; 10];
    h[0] = load4(&s[0..]);
    h[1] = load3(&s[4..]) << 6;
    h[2] = load3(&s[7..]) << 5;
    h[3] = load3(&s[10..]) << 3;
    h[4] = load3(&s[13..]) << 2;
    h[5] = load4(&s[16..]);
    h[6] = load3(&s[20..]) << 7;
    h[7] = load3(&s[23..]) << 5;
    h[8] = load3(&s[26..]) << 4;
    h[9] = (load3(&s[29..]) & 0x7f_ffff) << 2;
    carry_scalar(&mut h);
    let mut out = [0i32; 10];
    for i in 0..10 {
        out[i] = h[i] as i32;
    }
    out
}

/// Full ref10 carry chain that reduces wide limbs back into the radix bounds.
fn carry_scalar(h: &mut [i64; 10]) {
    macro_rules! c {
        ($i:expr, $shift:expr, $round:expr) => {{
            let carry = (h[$i] + (1i64 << $round)) >> $shift;
            h[($i + 1) % 10] += if $i == 9 { carry * 19 } else { carry };
            h[$i] -= carry << $shift;
        }};
    }
    c!(0, 26, 25);
    c!(4, 26, 25);
    c!(1, 25, 24);
    c!(5, 25, 24);
    c!(2, 26, 25);
    c!(6, 26, 25);
    c!(3, 25, 24);
    c!(7, 25, 24);
    c!(4, 26, 25);
    c!(8, 26, 25);
    c!(9, 25, 24);
    c!(0, 26, 25);
}

/// Reduce a field element to its canonical representative: non-negative limbs
/// in radix-2^25.5, value in `[0, 2^255-19)`. This is the front half of ref10
/// `fe_tobytes`; the back half is [`fe_pack`].
pub fn fe_reduce_canonical(h_in: &Fe) -> Fe {
    let mut h = *h_in;
    let mut q = (19 * h[9] + (1 << 24)) >> 25;
    q = (h[0] + q) >> 26;
    q = (h[1] + q) >> 25;
    q = (h[2] + q) >> 26;
    q = (h[3] + q) >> 25;
    q = (h[4] + q) >> 26;
    q = (h[5] + q) >> 25;
    q = (h[6] + q) >> 26;
    q = (h[7] + q) >> 25;
    q = (h[8] + q) >> 26;
    q = (h[9] + q) >> 25;
    h[0] += 19 * q;

    let shifts = [26, 25, 26, 25, 26, 25, 26, 25, 26, 25];
    for i in 0..9 {
        let carry = h[i] >> shifts[i];
        h[i + 1] += carry;
        h[i] -= carry << shifts[i];
    }
    let carry9 = h[9] >> 25;
    h[9] -= carry9 << 25;
    h
}

/// Pack canonical limbs (output of [`fe_reduce_canonical`]) into 32 bytes.
pub fn fe_pack(h_in: &Fe) -> [u8; 32] {
    let h: [i64; 10] = core::array::from_fn(|i| h_in[i] as i64);
    let mut s = [0u8; 32];
    s[0] = (h[0] >> 0) as u8;
    s[1] = (h[0] >> 8) as u8;
    s[2] = (h[0] >> 16) as u8;
    s[3] = ((h[0] >> 24) | (h[1] << 2)) as u8;
    s[4] = (h[1] >> 6) as u8;
    s[5] = (h[1] >> 14) as u8;
    s[6] = ((h[1] >> 22) | (h[2] << 3)) as u8;
    s[7] = (h[2] >> 5) as u8;
    s[8] = (h[2] >> 13) as u8;
    s[9] = ((h[2] >> 21) | (h[3] << 5)) as u8;
    s[10] = (h[3] >> 3) as u8;
    s[11] = (h[3] >> 11) as u8;
    s[12] = ((h[3] >> 19) | (h[4] << 6)) as u8;
    s[13] = (h[4] >> 2) as u8;
    s[14] = (h[4] >> 10) as u8;
    s[15] = (h[4] >> 18) as u8;
    s[16] = (h[5] >> 0) as u8;
    s[17] = (h[5] >> 8) as u8;
    s[18] = (h[5] >> 16) as u8;
    s[19] = ((h[5] >> 24) | (h[6] << 1)) as u8;
    s[20] = (h[6] >> 7) as u8;
    s[21] = (h[6] >> 15) as u8;
    s[22] = ((h[6] >> 23) | (h[7] << 3)) as u8;
    s[23] = (h[7] >> 5) as u8;
    s[24] = (h[7] >> 13) as u8;
    s[25] = ((h[7] >> 21) | (h[8] << 4)) as u8;
    s[26] = (h[8] >> 4) as u8;
    s[27] = (h[8] >> 12) as u8;
    s[28] = ((h[8] >> 20) | (h[9] << 6)) as u8;
    s[29] = (h[9] >> 2) as u8;
    s[30] = (h[9] >> 10) as u8;
    s[31] = (h[9] >> 18) as u8;
    s
}

/// Encode a field element to 32 canonical little-endian bytes (ref10
/// `fe_tobytes` = reduce then pack).
pub fn fe_tobytes(h: &Fe) -> [u8; 32] {
    fe_pack(&fe_reduce_canonical(h))
}

/// Scalar ref10 field multiply `h = f * g mod (2^255 - 19)`. Reference and
/// non-AVX2 fallback.
pub fn fe_mul(f: &Fe, g: &Fe) -> Fe {
    let f: [i64; 10] = core::array::from_fn(|i| f[i] as i64);
    let g: [i64; 10] = core::array::from_fn(|i| g[i] as i64);

    let g1_19 = 19 * g[1];
    let g2_19 = 19 * g[2];
    let g3_19 = 19 * g[3];
    let g4_19 = 19 * g[4];
    let g5_19 = 19 * g[5];
    let g6_19 = 19 * g[6];
    let g7_19 = 19 * g[7];
    let g8_19 = 19 * g[8];
    let g9_19 = 19 * g[9];
    let f1_2 = 2 * f[1];
    let f3_2 = 2 * f[3];
    let f5_2 = 2 * f[5];
    let f7_2 = 2 * f[7];
    let f9_2 = 2 * f[9];
    let (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9) =
        (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]);
    let (g0, g1, g2, g3, g4, g5, g6, g7, g8, g9) =
        (g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9]);

    let mut h = [0i64; 10];
    h[0] = f0 * g0 + f1_2 * g9_19 + f2 * g8_19 + f3_2 * g7_19 + f4 * g6_19
        + f5_2 * g5_19 + f6 * g4_19 + f7_2 * g3_19 + f8 * g2_19 + f9_2 * g1_19;
    h[1] = f0 * g1 + f1 * g0 + f2 * g9_19 + f3 * g8_19 + f4 * g7_19
        + f5 * g6_19 + f6 * g5_19 + f7 * g4_19 + f8 * g3_19 + f9 * g2_19;
    h[2] = f0 * g2 + f1_2 * g1 + f2 * g0 + f3_2 * g9_19 + f4 * g8_19
        + f5_2 * g7_19 + f6 * g6_19 + f7_2 * g5_19 + f8 * g4_19 + f9_2 * g3_19;
    h[3] = f0 * g3 + f1 * g2 + f2 * g1 + f3 * g0 + f4 * g9_19
        + f5 * g8_19 + f6 * g7_19 + f7 * g6_19 + f8 * g5_19 + f9 * g4_19;
    h[4] = f0 * g4 + f1_2 * g3 + f2 * g2 + f3_2 * g1 + f4 * g0
        + f5_2 * g9_19 + f6 * g8_19 + f7_2 * g7_19 + f8 * g6_19 + f9_2 * g5_19;
    h[5] = f0 * g5 + f1 * g4 + f2 * g3 + f3 * g2 + f4 * g1
        + f5 * g0 + f6 * g9_19 + f7 * g8_19 + f8 * g7_19 + f9 * g6_19;
    h[6] = f0 * g6 + f1_2 * g5 + f2 * g4 + f3_2 * g3 + f4 * g2
        + f5_2 * g1 + f6 * g0 + f7_2 * g9_19 + f8 * g8_19 + f9_2 * g7_19;
    h[7] = f0 * g7 + f1 * g6 + f2 * g5 + f3 * g4 + f4 * g3
        + f5 * g2 + f6 * g1 + f7 * g0 + f8 * g9_19 + f9 * g8_19;
    h[8] = f0 * g8 + f1_2 * g7 + f2 * g6 + f3_2 * g5 + f4 * g4
        + f5_2 * g3 + f6 * g2 + f7_2 * g1 + f8 * g0 + f9_2 * g9_19;
    h[9] = f0 * g9 + f1 * g8 + f2 * g7 + f3 * g6 + f4 * g5
        + f5 * g4 + f6 * g3 + f7 * g2 + f8 * g1 + f9 * g0;

    carry_scalar(&mut h);
    core::array::from_fn(|i| h[i] as i32)
}

/// `h = f + g` (limb-wise; ref10 `fe_add`).
pub fn fe_add(f: &Fe, g: &Fe) -> Fe {
    core::array::from_fn(|i| f[i] + g[i])
}

/// `h = f - g` (limb-wise; ref10 `fe_sub`).
pub fn fe_sub(f: &Fe, g: &Fe) -> Fe {
    core::array::from_fn(|i| f[i] - g[i])
}

/// `h = f^2`.
pub fn fe_sq(f: &Fe) -> Fe {
    fe_mul(f, f)
}

/// `out = z^(p-2) = z^-1` via the ref10 addition chain.
pub fn fe_invert(z: &Fe) -> Fe {
    let mut t0 = fe_sq(z); // z^2
    let mut t1 = fe_sq(&t0);
    t1 = fe_sq(&t1); // z^8
    t1 = fe_mul(z, &t1); // z^9
    t0 = fe_mul(&t0, &t1); // z^11
    let mut t2 = fe_sq(&t0); // z^22
    t1 = fe_mul(&t1, &t2); // z^(2^5-1)
    t2 = fe_sq(&t1);
    for _ in 1..5 {
        t2 = fe_sq(&t2);
    }
    t1 = fe_mul(&t2, &t1); // z^(2^10-1)
    t2 = fe_sq(&t1);
    for _ in 1..10 {
        t2 = fe_sq(&t2);
    }
    t2 = fe_mul(&t2, &t1); // z^(2^20-1)
    let mut t3 = fe_sq(&t2);
    for _ in 1..20 {
        t3 = fe_sq(&t3);
    }
    t2 = fe_mul(&t3, &t2); // z^(2^40-1)
    t2 = fe_sq(&t2);
    for _ in 1..10 {
        t2 = fe_sq(&t2);
    }
    t1 = fe_mul(&t2, &t1); // z^(2^50-1)
    t2 = fe_sq(&t1);
    for _ in 1..50 {
        t2 = fe_sq(&t2);
    }
    t2 = fe_mul(&t2, &t1); // z^(2^100-1)
    t3 = fe_sq(&t2);
    for _ in 1..100 {
        t3 = fe_sq(&t3);
    }
    t2 = fe_mul(&t3, &t2); // z^(2^200-1)
    t2 = fe_sq(&t2);
    for _ in 1..50 {
        t2 = fe_sq(&t2);
    }
    t1 = fe_mul(&t2, &t1); // z^(2^250-1)
    t1 = fe_sq(&t1);
    for _ in 1..5 {
        t1 = fe_sq(&t1);
    }
    fe_mul(&t1, &t0)
}

// =====================================================================
// AVX2 4-way (4 independent keys per call)
// =====================================================================

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use super::Fe;
    use core::arch::x86_64::*;

    /// Four field elements, transposed: `l[i]` holds limb `i` of all four
    /// keys, one key per 64-bit lane. Limbs are kept sign-extended in the full
    /// 64-bit lane so that adds/subs work; `_mm256_mul_epi32` reads only the
    /// low 32 bits (sign-extending them), which is valid because reduced limbs
    /// fit in 32 bits.
    #[derive(Clone, Copy)]
    pub struct Fe4 {
        pub l: [__m256i; 10],
    }

    /// Arithmetic (sign-propagating) right shift of each 64-bit lane by a
    /// compile-time count `C` in `1..=31`. AVX2 has no `srai_epi64`, so combine
    /// a logical 64-bit shift (correct low dword) with a 32-bit arithmetic
    /// shift (sign-correct high dword) and blend.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn srai64<const C: i32>(x: __m256i) -> __m256i {
        let lo = _mm256_srli_epi64(x, C);
        let hi = _mm256_srai_epi32(x, C);
        _mm256_blend_epi32(lo, hi, 0xAA)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn m19(x: __m256i) -> __m256i {
        _mm256_mul_epi32(x, _mm256_set1_epi64x(19))
    }

    /// Carry-reduce ten wide accumulators in place (ref10 carry order).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn carry4(h: &mut [__m256i; 10]) {
        let r25 = _mm256_set1_epi64x(1 << 25);
        let r24 = _mm256_set1_epi64x(1 << 24);

        macro_rules! ce {
            ($i:expr, $j:expr) => {{
                // even limb -> 26-bit, round 1<<25
                let c = srai64::<26>(_mm256_add_epi64(h[$i], r25));
                h[$j] = _mm256_add_epi64(h[$j], c);
                h[$i] = _mm256_sub_epi64(h[$i], _mm256_slli_epi64(c, 26));
            }};
        }
        macro_rules! co {
            ($i:expr, $j:expr) => {{
                // odd limb -> 25-bit, round 1<<24
                let c = srai64::<25>(_mm256_add_epi64(h[$i], r24));
                h[$j] = _mm256_add_epi64(h[$j], c);
                h[$i] = _mm256_sub_epi64(h[$i], _mm256_slli_epi64(c, 25));
            }};
        }

        ce!(0, 1);
        ce!(4, 5);
        co!(1, 2);
        co!(5, 6);
        ce!(2, 3);
        ce!(6, 7);
        co!(3, 4);
        co!(7, 8);
        ce!(4, 5);
        ce!(8, 9);
        // carry9 -> 0 with *19
        let c9 = srai64::<25>(_mm256_add_epi64(h[9], r24));
        h[0] = _mm256_add_epi64(h[0], m19(c9));
        h[9] = _mm256_sub_epi64(h[9], _mm256_slli_epi64(c9, 25));
        ce!(0, 1);
    }

    /// `h = f * g mod (2^255 - 19)` for four independent keys at once.
    #[target_feature(enable = "avx2")]
    pub unsafe fn mul4(f: &Fe4, g: &Fe4) -> Fe4 {
        let fl = &f.l;
        let gl = &g.l;

        let g1_19 = m19(gl[1]);
        let g2_19 = m19(gl[2]);
        let g3_19 = m19(gl[3]);
        let g4_19 = m19(gl[4]);
        let g5_19 = m19(gl[5]);
        let g6_19 = m19(gl[6]);
        let g7_19 = m19(gl[7]);
        let g8_19 = m19(gl[8]);
        let g9_19 = m19(gl[9]);
        let f1_2 = _mm256_add_epi64(fl[1], fl[1]);
        let f3_2 = _mm256_add_epi64(fl[3], fl[3]);
        let f5_2 = _mm256_add_epi64(fl[5], fl[5]);
        let f7_2 = _mm256_add_epi64(fl[7], fl[7]);
        let f9_2 = _mm256_add_epi64(fl[9], fl[9]);

        let p = |a, b| _mm256_mul_epi32(a, b);
        let add = |a, b| _mm256_add_epi64(a, b);
        macro_rules! sum {
            ($($x:expr),+ $(,)?) => {{ let mut acc = _mm256_setzero_si256(); $(acc = add(acc, $x);)+ acc }};
        }

        let (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9) =
            (fl[0], fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7], fl[8], fl[9]);
        let (g0, g1, g2, g3, g4, g5, g6, g7, g8, g9) =
            (gl[0], gl[1], gl[2], gl[3], gl[4], gl[5], gl[6], gl[7], gl[8], gl[9]);

        let mut h = [
            sum!(p(f0, g0), p(f1_2, g9_19), p(f2, g8_19), p(f3_2, g7_19), p(f4, g6_19),
                 p(f5_2, g5_19), p(f6, g4_19), p(f7_2, g3_19), p(f8, g2_19), p(f9_2, g1_19)),
            sum!(p(f0, g1), p(f1, g0), p(f2, g9_19), p(f3, g8_19), p(f4, g7_19),
                 p(f5, g6_19), p(f6, g5_19), p(f7, g4_19), p(f8, g3_19), p(f9, g2_19)),
            sum!(p(f0, g2), p(f1_2, g1), p(f2, g0), p(f3_2, g9_19), p(f4, g8_19),
                 p(f5_2, g7_19), p(f6, g6_19), p(f7_2, g5_19), p(f8, g4_19), p(f9_2, g3_19)),
            sum!(p(f0, g3), p(f1, g2), p(f2, g1), p(f3, g0), p(f4, g9_19),
                 p(f5, g8_19), p(f6, g7_19), p(f7, g6_19), p(f8, g5_19), p(f9, g4_19)),
            sum!(p(f0, g4), p(f1_2, g3), p(f2, g2), p(f3_2, g1), p(f4, g0),
                 p(f5_2, g9_19), p(f6, g8_19), p(f7_2, g7_19), p(f8, g6_19), p(f9_2, g5_19)),
            sum!(p(f0, g5), p(f1, g4), p(f2, g3), p(f3, g2), p(f4, g1),
                 p(f5, g0), p(f6, g9_19), p(f7, g8_19), p(f8, g7_19), p(f9, g6_19)),
            sum!(p(f0, g6), p(f1_2, g5), p(f2, g4), p(f3_2, g3), p(f4, g2),
                 p(f5_2, g1), p(f6, g0), p(f7_2, g9_19), p(f8, g8_19), p(f9_2, g7_19)),
            sum!(p(f0, g7), p(f1, g6), p(f2, g5), p(f3, g4), p(f4, g3),
                 p(f5, g2), p(f6, g1), p(f7, g0), p(f8, g9_19), p(f9, g8_19)),
            sum!(p(f0, g8), p(f1_2, g7), p(f2, g6), p(f3_2, g5), p(f4, g4),
                 p(f5_2, g3), p(f6, g2), p(f7_2, g1), p(f8, g0), p(f9_2, g9_19)),
            sum!(p(f0, g9), p(f1, g8), p(f2, g7), p(f3, g6), p(f4, g5),
                 p(f5, g4), p(f6, g3), p(f7, g2), p(f8, g1), p(f9, g0)),
        ];

        carry4(&mut h);
        Fe4 { l: h }
    }

    /// Transpose four scalar field elements into the lane-packed form.
    #[target_feature(enable = "avx2")]
    pub unsafe fn load4(fes: &[Fe; 4]) -> Fe4 {
        Fe4 {
            l: core::array::from_fn(|i| {
                _mm256_set_epi64x(
                    fes[3][i] as i64,
                    fes[2][i] as i64,
                    fes[1][i] as i64,
                    fes[0][i] as i64,
                )
            }),
        }
    }

    /// Transpose back to four scalar field elements (limbs truncated to i32).
    #[target_feature(enable = "avx2")]
    pub unsafe fn store4(x: &Fe4) -> [Fe; 4] {
        let mut out = [[0i32; 10]; 4];
        let mut tmp = [0i64; 4];
        for i in 0..10 {
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, x.l[i]);
            for k in 0..4 {
                out[k][i] = tmp[k] as i32;
            }
        }
        out
    }

    /// Broadcast one scalar field element into all four lanes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn broadcast(fe: &Fe) -> Fe4 {
        Fe4 {
            l: core::array::from_fn(|i| _mm256_set1_epi64x(fe[i] as i64)),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add4(f: &Fe4, g: &Fe4) -> Fe4 {
        Fe4 {
            l: core::array::from_fn(|i| _mm256_add_epi64(f.l[i], g.l[i])),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn sub4(f: &Fe4, g: &Fe4) -> Fe4 {
        Fe4 {
            l: core::array::from_fn(|i| _mm256_sub_epi64(f.l[i], g.l[i])),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn mc(x: __m256i, c: i64) -> __m256i {
        _mm256_mul_epi32(x, _mm256_set1_epi64x(c))
    }

    /// `h = f^2` (ref10 `fe_sq`, 4-wide). Exploits symmetry to roughly halve
    /// the limb products versus `mul4(f, f)`; used heavily by `invert4`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sq4(f: &Fe4) -> Fe4 {
        let l = &f.l;
        let (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9) =
            (l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9]);
        let p = |a, b| _mm256_mul_epi32(a, b);
        let add = |a, b| _mm256_add_epi64(a, b);
        let dbl = |a| _mm256_add_epi64(a, a);

        let f0_2 = dbl(f0);
        let f1_2 = dbl(f1);
        let f2_2 = dbl(f2);
        let f3_2 = dbl(f3);
        let f4_2 = dbl(f4);
        let f5_2 = dbl(f5);
        let f6_2 = dbl(f6);
        let f7_2 = dbl(f7);
        let f5_38 = mc(f5, 38);
        let f6_19 = mc(f6, 19);
        let f7_38 = mc(f7, 38);
        let f8_19 = mc(f8, 19);
        let f9_38 = mc(f9, 38);

        macro_rules! sum {
            ($($x:expr),+ $(,)?) => {{ let mut a = _mm256_setzero_si256(); $(a = add(a, $x);)+ a }};
        }

        let mut h = [
            sum!(p(f0, f0), p(f1_2, f9_38), p(f2_2, f8_19), p(f3_2, f7_38), p(f4_2, f6_19), p(f5, f5_38)),
            sum!(p(f0_2, f1), p(f2, f9_38), p(f3_2, f8_19), p(f4, f7_38), p(f5_2, f6_19)),
            sum!(p(f0_2, f2), p(f1_2, f1), p(f3_2, f9_38), p(f4_2, f8_19), p(f5_2, f7_38), p(f6, f6_19)),
            sum!(p(f0_2, f3), p(f1_2, f2), p(f4, f9_38), p(f5_2, f8_19), p(f6, f7_38)),
            sum!(p(f0_2, f4), p(f1_2, f3_2), p(f2, f2), p(f5_2, f9_38), p(f6_2, f8_19), p(f7, f7_38)),
            sum!(p(f0_2, f5), p(f1_2, f4), p(f2_2, f3), p(f6, f9_38), p(f7_2, f8_19)),
            sum!(p(f0_2, f6), p(f1_2, f5_2), p(f2_2, f4), p(f3_2, f3), p(f7_2, f9_38), p(f8, f8_19)),
            sum!(p(f0_2, f7), p(f1_2, f6), p(f2_2, f5), p(f3_2, f4), p(f8, f9_38)),
            sum!(p(f0_2, f8), p(f1_2, f7_2), p(f2_2, f6), p(f3_2, f5_2), p(f4, f4), p(f9, f9_38)),
            sum!(p(f0_2, f9), p(f1_2, f8), p(f2_2, f7), p(f3_2, f6), p(f4_2, f5)),
        ];

        carry4(&mut h);
        Fe4 { l: h }
    }

    /// A point in extended coordinates, four keys deep.
    #[derive(Clone, Copy)]
    pub struct Point4 {
        pub x: Fe4,
        pub y: Fe4,
        pub z: Fe4,
        pub t: Fe4,
    }

    /// A fixed `+step` addend in projective-niels form (same for all lanes).
    #[derive(Clone, Copy)]
    pub struct Niels4 {
        pub yp: Fe4,
        pub ym: Fe4,
        pub z: Fe4,
        pub t2d: Fe4,
    }

    /// `p + niels` via the dalek mixed-add formula (p3 + niels -> p1p1 -> p3).
    /// Mirrors ref10/dalek exactly; the balanced (rounded) carry in `mul4`
    /// keeps every intermediate within the `19*g < 2^31` tolerance, so no
    /// extra reduction is needed.
    #[target_feature(enable = "avx2")]
    pub unsafe fn madd4(p: &Point4, n: &Niels4) -> Point4 {
        let yp_x = add4(&p.y, &p.x);
        let ym_x = sub4(&p.y, &p.x);
        let pp = mul4(&yp_x, &n.yp);
        let mm = mul4(&ym_x, &n.ym);
        let tt2d = mul4(&p.t, &n.t2d);
        let zz = mul4(&p.z, &n.z);
        let zz2 = add4(&zz, &zz);
        // completed point
        let cx = sub4(&pp, &mm);
        let cy = add4(&pp, &mm);
        let cz = add4(&zz2, &tt2d);
        let ct = sub4(&zz2, &tt2d);
        // p1p1 -> p3
        Point4 {
            x: mul4(&cx, &ct),
            y: mul4(&cy, &cz),
            z: mul4(&cz, &ct),
            t: mul4(&cx, &cy),
        }
    }

    /// `out = z^-1` for four keys, sharing one addition chain across lanes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn invert4(z: &Fe4) -> Fe4 {
        let mut t0 = sq4(z);
        let mut t1 = sq4(&t0);
        t1 = sq4(&t1);
        t1 = mul4(z, &t1);
        t0 = mul4(&t0, &t1);
        let mut t2 = sq4(&t0);
        t1 = mul4(&t1, &t2);
        t2 = sq4(&t1);
        for _ in 1..5 {
            t2 = sq4(&t2);
        }
        t1 = mul4(&t2, &t1);
        t2 = sq4(&t1);
        for _ in 1..10 {
            t2 = sq4(&t2);
        }
        t2 = mul4(&t2, &t1);
        let mut t3 = sq4(&t2);
        for _ in 1..20 {
            t3 = sq4(&t3);
        }
        t2 = mul4(&t3, &t2);
        t2 = sq4(&t2);
        for _ in 1..10 {
            t2 = sq4(&t2);
        }
        t1 = mul4(&t2, &t1);
        t2 = sq4(&t1);
        for _ in 1..50 {
            t2 = sq4(&t2);
        }
        t2 = mul4(&t2, &t1);
        t3 = sq4(&t2);
        for _ in 1..100 {
            t3 = sq4(&t3);
        }
        t2 = mul4(&t3, &t2);
        t2 = sq4(&t2);
        for _ in 1..50 {
            t2 = sq4(&t2);
        }
        t1 = mul4(&t2, &t1);
        t1 = sq4(&t1);
        for _ in 1..5 {
            t1 = sq4(&t1);
        }
        mul4(&t1, &t0)
    }

    /// Build a `Point4` from four scalar points' extended coordinates.
    #[target_feature(enable = "avx2")]
    pub unsafe fn point4_from_xyzt(
        x: &[Fe; 4],
        y: &[Fe; 4],
        z: &[Fe; 4],
        t: &[Fe; 4],
    ) -> Point4 {
        Point4 {
            x: load4(x),
            y: load4(y),
            z: load4(z),
            t: load4(t),
        }
    }

    /// Build the shared `+step` niels addend (broadcast to all lanes) from its
    /// canonical bytes `(Y+X, Y-X, Z, T·2d)`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn niels4_from_bytes(nb: &[[u8; 32]; 4]) -> Niels4 {
        Niels4 {
            yp: broadcast(&super::fe_frombytes(&nb[0])),
            ym: broadcast(&super::fe_frombytes(&nb[1])),
            z: broadcast(&super::fe_frombytes(&nb[2])),
            t2d: broadcast(&super::fe_frombytes(&nb[3])),
        }
    }

    /// Walk `K` steps of `+step` from `p` across all four lanes, then y-only-
    /// compress every visited point with one shared batch inversion. Returns
    /// `out[s][lane]` = the y-encoding (sign bit zero) of `p + s·step` in lane
    /// `lane`, plus the next base `p + K·step`. This is the SIMD analog of
    /// `EdwardsPoint::chain_compress_y_only`, producing `4·K` keys per call.
    #[target_feature(enable = "avx2")]
    pub unsafe fn chain_y_only<const K: usize>(
        mut p: Point4,
        niels: &Niels4,
        out: &mut [[[u8; 32]; 4]; K],
    ) -> Point4 {
        let mut ys = [p.y; K];
        let mut zs = [p.z; K];
        for s in 0..K {
            ys[s] = p.y;
            zs[s] = p.z;
            p = madd4(&p, niels);
        }

        // Montgomery batch inversion of the K Z-values (4 lanes each).
        let one = broadcast(&super::FE_ONE);
        let mut prefix = [one; K];
        let mut acc = one;
        for s in 0..K {
            prefix[s] = acc;
            acc = mul4(&acc, &zs[s]);
        }
        let mut inv = invert4(&acc);

        for s in (0..K).rev() {
            let zinv = mul4(&inv, &prefix[s]);
            inv = mul4(&inv, &zs[s]);
            let yz = mul4(&ys[s], &zinv);
            let lanes = store4(&yz);
            for lane in 0..4 {
                out[s][lane] = super::fe_tobytes(&lanes[lane]);
            }
        }
        p
    }
}

// =====================================================================
// AVX-512 8-wide (8 independent keys per call). Same radix-2^25.5 math as
// the AVX2 path, doubled lane count; `_mm512_srai_epi64` is native so the
// carry needs no emulation. Kept only if it actually beats AVX2 at runtime.
// =====================================================================

#[cfg(target_arch = "x86_64")]
pub mod avx512 {
    use super::Fe;
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct Fe8 {
        pub l: [__m512i; 10],
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn m19(x: __m512i) -> __m512i {
        _mm512_mul_epi32(x, _mm512_set1_epi64(19))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mc(x: __m512i, c: i64) -> __m512i {
        _mm512_mul_epi32(x, _mm512_set1_epi64(c))
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn carry8(h: &mut [__m512i; 10]) {
        let r25 = _mm512_set1_epi64(1 << 25);
        let r24 = _mm512_set1_epi64(1 << 24);
        macro_rules! ce {
            ($i:expr, $j:expr) => {{
                let c = _mm512_srai_epi64(_mm512_add_epi64(h[$i], r25), 26);
                h[$j] = _mm512_add_epi64(h[$j], c);
                h[$i] = _mm512_sub_epi64(h[$i], _mm512_slli_epi64(c, 26));
            }};
        }
        macro_rules! co {
            ($i:expr, $j:expr) => {{
                let c = _mm512_srai_epi64(_mm512_add_epi64(h[$i], r24), 25);
                h[$j] = _mm512_add_epi64(h[$j], c);
                h[$i] = _mm512_sub_epi64(h[$i], _mm512_slli_epi64(c, 25));
            }};
        }
        ce!(0, 1);
        ce!(4, 5);
        co!(1, 2);
        co!(5, 6);
        ce!(2, 3);
        ce!(6, 7);
        co!(3, 4);
        co!(7, 8);
        ce!(4, 5);
        ce!(8, 9);
        let c9 = _mm512_srai_epi64(_mm512_add_epi64(h[9], r24), 25);
        h[0] = _mm512_add_epi64(h[0], m19(c9));
        h[9] = _mm512_sub_epi64(h[9], _mm512_slli_epi64(c9, 25));
        ce!(0, 1);
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn mul8(f: &Fe8, g: &Fe8) -> Fe8 {
        let fl = &f.l;
        let gl = &g.l;
        let g1_19 = m19(gl[1]);
        let g2_19 = m19(gl[2]);
        let g3_19 = m19(gl[3]);
        let g4_19 = m19(gl[4]);
        let g5_19 = m19(gl[5]);
        let g6_19 = m19(gl[6]);
        let g7_19 = m19(gl[7]);
        let g8_19 = m19(gl[8]);
        let g9_19 = m19(gl[9]);
        let f1_2 = _mm512_add_epi64(fl[1], fl[1]);
        let f3_2 = _mm512_add_epi64(fl[3], fl[3]);
        let f5_2 = _mm512_add_epi64(fl[5], fl[5]);
        let f7_2 = _mm512_add_epi64(fl[7], fl[7]);
        let f9_2 = _mm512_add_epi64(fl[9], fl[9]);
        let p = |a, b| _mm512_mul_epi32(a, b);
        let add = |a, b| _mm512_add_epi64(a, b);
        macro_rules! sum {
            ($($x:expr),+ $(,)?) => {{ let mut a = _mm512_setzero_si512(); $(a = add(a, $x);)+ a }};
        }
        let (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9) =
            (fl[0], fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7], fl[8], fl[9]);
        let (g0, g1, g2, g3, g4, g5, g6, g7, g8, g9) =
            (gl[0], gl[1], gl[2], gl[3], gl[4], gl[5], gl[6], gl[7], gl[8], gl[9]);
        let mut h = [
            sum!(p(f0, g0), p(f1_2, g9_19), p(f2, g8_19), p(f3_2, g7_19), p(f4, g6_19),
                 p(f5_2, g5_19), p(f6, g4_19), p(f7_2, g3_19), p(f8, g2_19), p(f9_2, g1_19)),
            sum!(p(f0, g1), p(f1, g0), p(f2, g9_19), p(f3, g8_19), p(f4, g7_19),
                 p(f5, g6_19), p(f6, g5_19), p(f7, g4_19), p(f8, g3_19), p(f9, g2_19)),
            sum!(p(f0, g2), p(f1_2, g1), p(f2, g0), p(f3_2, g9_19), p(f4, g8_19),
                 p(f5_2, g7_19), p(f6, g6_19), p(f7_2, g5_19), p(f8, g4_19), p(f9_2, g3_19)),
            sum!(p(f0, g3), p(f1, g2), p(f2, g1), p(f3, g0), p(f4, g9_19),
                 p(f5, g8_19), p(f6, g7_19), p(f7, g6_19), p(f8, g5_19), p(f9, g4_19)),
            sum!(p(f0, g4), p(f1_2, g3), p(f2, g2), p(f3_2, g1), p(f4, g0),
                 p(f5_2, g9_19), p(f6, g8_19), p(f7_2, g7_19), p(f8, g6_19), p(f9_2, g5_19)),
            sum!(p(f0, g5), p(f1, g4), p(f2, g3), p(f3, g2), p(f4, g1),
                 p(f5, g0), p(f6, g9_19), p(f7, g8_19), p(f8, g7_19), p(f9, g6_19)),
            sum!(p(f0, g6), p(f1_2, g5), p(f2, g4), p(f3_2, g3), p(f4, g2),
                 p(f5_2, g1), p(f6, g0), p(f7_2, g9_19), p(f8, g8_19), p(f9_2, g7_19)),
            sum!(p(f0, g7), p(f1, g6), p(f2, g5), p(f3, g4), p(f4, g3),
                 p(f5, g2), p(f6, g1), p(f7, g0), p(f8, g9_19), p(f9, g8_19)),
            sum!(p(f0, g8), p(f1_2, g7), p(f2, g6), p(f3_2, g5), p(f4, g4),
                 p(f5_2, g3), p(f6, g2), p(f7_2, g1), p(f8, g0), p(f9_2, g9_19)),
            sum!(p(f0, g9), p(f1, g8), p(f2, g7), p(f3, g6), p(f4, g5),
                 p(f5, g4), p(f6, g3), p(f7, g2), p(f8, g1), p(f9, g0)),
        ];
        carry8(&mut h);
        Fe8 { l: h }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn sq8(f: &Fe8) -> Fe8 {
        let l = &f.l;
        let (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9) =
            (l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], l[9]);
        let p = |a, b| _mm512_mul_epi32(a, b);
        let add = |a, b| _mm512_add_epi64(a, b);
        let dbl = |a| _mm512_add_epi64(a, a);
        let f0_2 = dbl(f0);
        let f1_2 = dbl(f1);
        let f2_2 = dbl(f2);
        let f3_2 = dbl(f3);
        let f4_2 = dbl(f4);
        let f5_2 = dbl(f5);
        let f6_2 = dbl(f6);
        let f7_2 = dbl(f7);
        let f5_38 = mc(f5, 38);
        let f6_19 = mc(f6, 19);
        let f7_38 = mc(f7, 38);
        let f8_19 = mc(f8, 19);
        let f9_38 = mc(f9, 38);
        macro_rules! sum {
            ($($x:expr),+ $(,)?) => {{ let mut a = _mm512_setzero_si512(); $(a = add(a, $x);)+ a }};
        }
        let mut h = [
            sum!(p(f0, f0), p(f1_2, f9_38), p(f2_2, f8_19), p(f3_2, f7_38), p(f4_2, f6_19), p(f5, f5_38)),
            sum!(p(f0_2, f1), p(f2, f9_38), p(f3_2, f8_19), p(f4, f7_38), p(f5_2, f6_19)),
            sum!(p(f0_2, f2), p(f1_2, f1), p(f3_2, f9_38), p(f4_2, f8_19), p(f5_2, f7_38), p(f6, f6_19)),
            sum!(p(f0_2, f3), p(f1_2, f2), p(f4, f9_38), p(f5_2, f8_19), p(f6, f7_38)),
            sum!(p(f0_2, f4), p(f1_2, f3_2), p(f2, f2), p(f5_2, f9_38), p(f6_2, f8_19), p(f7, f7_38)),
            sum!(p(f0_2, f5), p(f1_2, f4), p(f2_2, f3), p(f6, f9_38), p(f7_2, f8_19)),
            sum!(p(f0_2, f6), p(f1_2, f5_2), p(f2_2, f4), p(f3_2, f3), p(f7_2, f9_38), p(f8, f8_19)),
            sum!(p(f0_2, f7), p(f1_2, f6), p(f2_2, f5), p(f3_2, f4), p(f8, f9_38)),
            sum!(p(f0_2, f8), p(f1_2, f7_2), p(f2_2, f6), p(f3_2, f5_2), p(f4, f4), p(f9, f9_38)),
            sum!(p(f0_2, f9), p(f1_2, f8), p(f2_2, f7), p(f3_2, f6), p(f4_2, f5)),
        ];
        carry8(&mut h);
        Fe8 { l: h }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add8(f: &Fe8, g: &Fe8) -> Fe8 {
        Fe8 {
            l: core::array::from_fn(|i| _mm512_add_epi64(f.l[i], g.l[i])),
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub8(f: &Fe8, g: &Fe8) -> Fe8 {
        Fe8 {
            l: core::array::from_fn(|i| _mm512_sub_epi64(f.l[i], g.l[i])),
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn broadcast(fe: &Fe) -> Fe8 {
        Fe8 {
            l: core::array::from_fn(|i| _mm512_set1_epi64(fe[i] as i64)),
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn load8(fes: &[Fe; 8]) -> Fe8 {
        Fe8 {
            l: core::array::from_fn(|i| {
                _mm512_set_epi64(
                    fes[7][i] as i64,
                    fes[6][i] as i64,
                    fes[5][i] as i64,
                    fes[4][i] as i64,
                    fes[3][i] as i64,
                    fes[2][i] as i64,
                    fes[1][i] as i64,
                    fes[0][i] as i64,
                )
            }),
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn store8(x: &Fe8) -> [Fe; 8] {
        let mut out = [[0i32; 10]; 8];
        let mut tmp = [0i64; 8];
        for i in 0..10 {
            _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, x.l[i]);
            for k in 0..8 {
                out[k][i] = tmp[k] as i32;
            }
        }
        out
    }

    #[derive(Clone, Copy)]
    pub struct Point8 {
        pub x: Fe8,
        pub y: Fe8,
        pub z: Fe8,
        pub t: Fe8,
    }

    #[derive(Clone, Copy)]
    pub struct Niels8 {
        pub yp: Fe8,
        pub ym: Fe8,
        pub z: Fe8,
        pub t2d: Fe8,
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn madd8(p: &Point8, n: &Niels8) -> Point8 {
        let yp_x = add8(&p.y, &p.x);
        let ym_x = sub8(&p.y, &p.x);
        let pp = mul8(&yp_x, &n.yp);
        let mm = mul8(&ym_x, &n.ym);
        let tt2d = mul8(&p.t, &n.t2d);
        let zz = mul8(&p.z, &n.z);
        let zz2 = add8(&zz, &zz);
        let cx = sub8(&pp, &mm);
        let cy = add8(&pp, &mm);
        let cz = add8(&zz2, &tt2d);
        let ct = sub8(&zz2, &tt2d);
        Point8 {
            x: mul8(&cx, &ct),
            y: mul8(&cy, &cz),
            z: mul8(&cz, &ct),
            t: mul8(&cx, &cy),
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn invert8(z: &Fe8) -> Fe8 {
        let mut t0 = sq8(z);
        let mut t1 = sq8(&t0);
        t1 = sq8(&t1);
        t1 = mul8(z, &t1);
        t0 = mul8(&t0, &t1);
        let mut t2 = sq8(&t0);
        t1 = mul8(&t1, &t2);
        t2 = sq8(&t1);
        for _ in 1..5 {
            t2 = sq8(&t2);
        }
        t1 = mul8(&t2, &t1);
        t2 = sq8(&t1);
        for _ in 1..10 {
            t2 = sq8(&t2);
        }
        t2 = mul8(&t2, &t1);
        let mut t3 = sq8(&t2);
        for _ in 1..20 {
            t3 = sq8(&t3);
        }
        t2 = mul8(&t3, &t2);
        t2 = sq8(&t2);
        for _ in 1..10 {
            t2 = sq8(&t2);
        }
        t1 = mul8(&t2, &t1);
        t2 = sq8(&t1);
        for _ in 1..50 {
            t2 = sq8(&t2);
        }
        t2 = mul8(&t2, &t1);
        t3 = sq8(&t2);
        for _ in 1..100 {
            t3 = sq8(&t3);
        }
        t2 = mul8(&t3, &t2);
        t2 = sq8(&t2);
        for _ in 1..50 {
            t2 = sq8(&t2);
        }
        t1 = mul8(&t2, &t1);
        t1 = sq8(&t1);
        for _ in 1..5 {
            t1 = sq8(&t1);
        }
        mul8(&t1, &t0)
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn point8_from_xyzt(
        x: &[Fe; 8],
        y: &[Fe; 8],
        z: &[Fe; 8],
        t: &[Fe; 8],
    ) -> Point8 {
        Point8 {
            x: load8(x),
            y: load8(y),
            z: load8(z),
            t: load8(t),
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn niels8_from_bytes(nb: &[[u8; 32]; 4]) -> Niels8 {
        Niels8 {
            yp: broadcast(&super::fe_frombytes(&nb[0])),
            ym: broadcast(&super::fe_frombytes(&nb[1])),
            z: broadcast(&super::fe_frombytes(&nb[2])),
            t2d: broadcast(&super::fe_frombytes(&nb[3])),
        }
    }

    // ===== first-byte prefilter helpers (feature = "prefilter") =====

    /// Vectorized [`super::fe_reduce_canonical`]: reduce 8 field elements to
    /// their canonical non-negative limbs at once.
    #[cfg(feature = "prefilter")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn canon8(yz: &Fe8) -> Fe8 {
        let mut h = yz.l;
        let r24 = _mm512_set1_epi64(1 << 24);
        // q is computed across lanes in lockstep (each lane independent).
        let mut q = _mm512_srai_epi64(_mm512_add_epi64(m19(h[9]), r24), 25);
        macro_rules! qstep {
            ($i:expr, $sh:expr) => {
                q = _mm512_srai_epi64(_mm512_add_epi64(h[$i], q), $sh);
            };
        }
        qstep!(0, 26);
        qstep!(1, 25);
        qstep!(2, 26);
        qstep!(3, 25);
        qstep!(4, 26);
        qstep!(5, 25);
        qstep!(6, 26);
        qstep!(7, 25);
        qstep!(8, 26);
        qstep!(9, 25);
        h[0] = _mm512_add_epi64(h[0], m19(q));
        macro_rules! cstep {
            ($i:expr, $sh:expr) => {{
                let c = _mm512_srai_epi64(h[$i], $sh);
                h[$i + 1] = _mm512_add_epi64(h[$i + 1], c);
                h[$i] = _mm512_sub_epi64(h[$i], _mm512_slli_epi64(c, $sh));
            }};
        }
        cstep!(0, 26);
        cstep!(1, 25);
        cstep!(2, 26);
        cstep!(3, 25);
        cstep!(4, 26);
        cstep!(5, 25);
        cstep!(6, 26);
        cstep!(7, 25);
        cstep!(8, 26);
        let c9 = _mm512_srai_epi64(h[9], 25);
        h[9] = _mm512_sub_epi64(h[9], _mm512_slli_epi64(c9, 25));
        Fe8 { l: h }
    }

    /// 8-bit mask of lanes whose canonical first byte can match some prefix.
    /// `filter[i] = (mask, value)`: lane passes if `byte0 & mask == value`.
    #[cfg(feature = "prefilter")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn firstbyte_accept_mask8(canon: &Fe8, filter: &[(u8, u8)]) -> u8 {
        let b0 = _mm512_and_si512(canon.l[0], _mm512_set1_epi64(0xFF));
        let mut accept: u8 = 0;
        for &(mask, value) in filter {
            let m = _mm512_and_si512(b0, _mm512_set1_epi64(mask as i64));
            accept |= _mm512_cmpeq_epi64_mask(m, _mm512_set1_epi64(value as i64));
        }
        accept
    }

    /// 8-wide analog of [`super::avx2::chain_y_only`]: `8·K` keys per call.
    /// `filter` drives the first-byte prefilter (see the `prefilter` feature);
    /// it is ignored when that feature is off.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn chain_y_only<const K: usize>(
        mut p: Point8,
        niels: &Niels8,
        filter: &[(u8, u8)],
        out: &mut [[[u8; 32]; 8]; K],
    ) -> Point8 {
        let mut ys = [p.y; K];
        let mut zs = [p.z; K];
        for s in 0..K {
            ys[s] = p.y;
            zs[s] = p.z;
            p = madd8(&p, niels);
        }
        let one = broadcast(&super::FE_ONE);
        let mut prefix = [one; K];
        let mut acc = one;
        for s in 0..K {
            prefix[s] = acc;
            acc = mul8(&acc, &zs[s]);
        }
        let mut inv = invert8(&acc);
        for s in (0..K).rev() {
            let zinv = mul8(&inv, &prefix[s]);
            inv = mul8(&inv, &zs[s]);
            let yz = mul8(&ys[s], &zinv);

            // ---- first-byte prefilter (remove this block / the `prefilter`
            //      feature to always fully encode every lane) ----
            #[cfg(feature = "prefilter")]
            {
                // byte0 = limb0 & 0xFF (higher limbs weigh 0 mod 256). This is
                // the canonical first byte except when the value is in
                // [p, 2^255) (~2^-250), so the filter is exact in practice and
                // avoids a full canonical reduction here. Survivors are encoded
                // exactly with fe_tobytes below, so results never have false
                // positives.
                let accept = firstbyte_accept_mask8(&yz, filter);
                if accept != 0 {
                    let lanes = store8(&yz);
                    for lane in 0..8 {
                        if accept & (1 << lane) != 0 {
                            out[s][lane] = super::fe_tobytes(&lanes[lane]);
                        } else {
                            out[s][lane][0] = 0;
                        }
                    }
                } else {
                    for lane in 0..8 {
                        // 0x00 first byte -> should_skip() rejects it.
                        out[s][lane][0] = 0;
                    }
                }
            }
            #[cfg(not(feature = "prefilter"))]
            {
                let _ = filter;
                let lanes = store8(&yz);
                for lane in 0..8 {
                    out[s][lane] = super::fe_tobytes(&lanes[lane]);
                }
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use curve25519_dalek::EdwardsPoint;

    // Small deterministic PRNG so tests don't pull in rand here.
    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state
    }
    fn rand_fe_bytes(state: &mut u64) -> [u8; 32] {
        let mut b = [0u8; 32];
        for chunk in b.chunks_mut(8) {
            chunk.copy_from_slice(&lcg(state).to_le_bytes());
        }
        b[31] &= 0x7f; // clear the ignored high bit for unambiguous comparison
        b
    }

    #[test]
    fn scalar_fe_mul_matches_dalek() {
        let mut st = 0x1234_5678_9abc_def0;
        for _ in 0..2000 {
            let a = rand_fe_bytes(&mut st);
            let b = rand_fe_bytes(&mut st);
            let got = fe_tobytes(&fe_mul(&fe_frombytes(&a), &fe_frombytes(&b)));
            let want = EdwardsPoint::field_mul_reference(&a, &b);
            assert_eq!(got, want, "scalar ref10 mul disagrees with dalek");
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_mul4_matches_dalek() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available, skipping");
            return;
        }
        let mut st = 0x0fed_cba9_8765_4321;
        for _ in 0..500 {
            let ab: [[u8; 32]; 4] = core::array::from_fn(|_| rand_fe_bytes(&mut st));
            let bb: [[u8; 32]; 4] = core::array::from_fn(|_| rand_fe_bytes(&mut st));
            let fa: [Fe; 4] = core::array::from_fn(|k| fe_frombytes(&ab[k]));
            let fb: [Fe; 4] = core::array::from_fn(|k| fe_frombytes(&bb[k]));
            let prod = unsafe {
                let r = avx2::mul4(&avx2::load4(&fa), &avx2::load4(&fb));
                avx2::store4(&r)
            };
            for k in 0..4 {
                let got = fe_tobytes(&prod[k]);
                let want = EdwardsPoint::field_mul_reference(&ab[k], &bb[k]);
                assert_eq!(got, want, "mul4 lane {k} disagrees with dalek");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_sq4_matches_mul4() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut st = 0x2468_ace0_1357_bdf9;
        for _ in 0..500 {
            let xb: [[u8; 32]; 4] = core::array::from_fn(|_| rand_fe_bytes(&mut st));
            let xf: [Fe; 4] = core::array::from_fn(|k| fe_frombytes(&xb[k]));
            let (sq, mm) = unsafe {
                let x = avx2::load4(&xf);
                (avx2::store4(&avx2::sq4(&x)), avx2::store4(&avx2::mul4(&x, &x)))
            };
            for k in 0..4 {
                assert_eq!(fe_tobytes(&sq[k]), fe_tobytes(&mm[k]), "sq4 != mul4(x,x) lane {k}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_invert4_matches_dalek() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let one = {
            let mut o = [0u8; 32];
            o[0] = 1;
            o
        };
        let mut st = 0xa1b2_c3d4_e5f6_0789;
        for _ in 0..200 {
            let xb: [[u8; 32]; 4] = core::array::from_fn(|_| rand_fe_bytes(&mut st));
            let xf: [Fe; 4] = core::array::from_fn(|k| fe_frombytes(&xb[k]));
            let prod = unsafe {
                let x = avx2::load4(&xf);
                let xi = avx2::invert4(&x);
                avx2::store4(&avx2::mul4(&x, &xi))
            };
            for k in 0..4 {
                assert_eq!(fe_tobytes(&prod[k]), one, "x * x^-1 != 1 in lane {k}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_madd4_matches_dalek() {
        use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
        use curve25519_dalek::scalar::Scalar;
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let step = ED25519_BASEPOINT_TABLE * &Scalar::from(8u64);
        let nb = step.niels_bytes();
        let niels = unsafe {
            avx2::Niels4 {
                yp: avx2::broadcast(&fe_frombytes(&nb[0])),
                ym: avx2::broadcast(&fe_frombytes(&nb[1])),
                z: avx2::broadcast(&fe_frombytes(&nb[2])),
                t2d: avx2::broadcast(&fe_frombytes(&nb[3])),
            }
        };

        let mut st = 0x5151_2323_9797_0001u64;
        for _ in 0..200 {
            let pts: [_; 4] = core::array::from_fn(|_| {
                EdwardsPoint::mul_base_clamped({
                    let mut s = rand_fe_bytes(&mut st);
                    s[0] &= 248;
                    s[31] = (s[31] & 63) | 64;
                    s
                })
            });
            let sums: [_; 4] = core::array::from_fn(|k| pts[k] + step);

            let xyzt: [_; 4] = core::array::from_fn(|k| pts[k].xyzt_bytes());
            let p4 = unsafe {
                avx2::Point4 {
                    x: avx2::load4(&core::array::from_fn(|k| fe_frombytes(&xyzt[k][0]))),
                    y: avx2::load4(&core::array::from_fn(|k| fe_frombytes(&xyzt[k][1]))),
                    z: avx2::load4(&core::array::from_fn(|k| fe_frombytes(&xyzt[k][2]))),
                    t: avx2::load4(&core::array::from_fn(|k| fe_frombytes(&xyzt[k][3]))),
                }
            };
            let (rx, ry, rz) = unsafe {
                let r = avx2::madd4(&p4, &niels);
                (avx2::store4(&r.x), avx2::store4(&r.y), avx2::store4(&r.z))
            };

            for k in 0..4 {
                let exp = sums[k].xyzt_bytes();
                let (ex, ey, ez) = (
                    fe_frombytes(&exp[0]),
                    fe_frombytes(&exp[1]),
                    fe_frombytes(&exp[2]),
                );
                // Projective coords are unique only up to scale: cross-multiply.
                assert_eq!(
                    fe_tobytes(&fe_mul(&rx[k], &ez)),
                    fe_tobytes(&fe_mul(&ex, &rz[k])),
                    "madd4 X mismatch lane {k}"
                );
                assert_eq!(
                    fe_tobytes(&fe_mul(&ry[k], &ez)),
                    fe_tobytes(&fe_mul(&ey, &rz[k])),
                    "madd4 Y mismatch lane {k}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_chain_y_only_matches_dalek() {
        use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
        use curve25519_dalek::scalar::Scalar;
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        const K: usize = 8;
        let step = ED25519_BASEPOINT_TABLE * &Scalar::from(8u64);
        let niels = unsafe { avx2::niels4_from_bytes(&step.niels_bytes()) };

        let mut st = 0x7777_3333_dddd_0001u64;
        let starts: [_; 4] = core::array::from_fn(|_| {
            EdwardsPoint::mul_base_clamped({
                let mut s = rand_fe_bytes(&mut st);
                s[0] &= 248;
                s[31] = (s[31] & 63) | 64;
                s
            })
        });
        let xyzt: [_; 4] = core::array::from_fn(|k| starts[k].xyzt_bytes());
        let p4 = unsafe {
            avx2::point4_from_xyzt(
                &core::array::from_fn(|k| fe_frombytes(&xyzt[k][0])),
                &core::array::from_fn(|k| fe_frombytes(&xyzt[k][1])),
                &core::array::from_fn(|k| fe_frombytes(&xyzt[k][2])),
                &core::array::from_fn(|k| fe_frombytes(&xyzt[k][3])),
            )
        };

        let mut out = [[[0u8; 32]; 4]; K];
        unsafe { avx2::chain_y_only::<K>(p4, &niels, &mut out) };

        for lane in 0..4 {
            let mut cur = starts[lane];
            for s in 0..K {
                let exp = cur.compress().to_bytes();
                // y-only encoding: low 31 bytes match; sign bit (byte 31 high)
                // is zeroed in the SIMD output.
                assert_eq!(
                    out[s][lane][..31],
                    exp[..31],
                    "chain_y_only mismatch lane {lane} step {s}"
                );
                assert_eq!(out[s][lane][31] & 0x80, 0);
                cur += step;
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "prefilter"))]
    fn simd_prefilter_canon_and_mask() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut st = 0xfeed_face_dead_0001u64;
        for _ in 0..200 {
            let xb: [[u8; 32]; 8] = core::array::from_fn(|_| rand_fe_bytes(&mut st));
            let xf: [Fe; 8] = core::array::from_fn(|k| fe_frombytes(&xb[k]));
            let canon = unsafe { avx512::canon8(&avx512::load8(&xf)) };
            let canon_lanes = unsafe { avx512::store8(&canon) };
            // canon8 + pack must reproduce the scalar canonical encoding.
            for k in 0..8 {
                assert_eq!(fe_pack(&canon_lanes[k]), fe_tobytes(&xf[k]), "canon8 lane {k}");
            }
            // Mask must accept exactly the lanes whose byte0 matches the filter.
            let target = fe_tobytes(&xf[3])[0];
            let mask = unsafe { avx512::firstbyte_accept_mask8(&canon, &[(0xFFu8, target)]) };
            for k in 0..8 {
                let expect = fe_tobytes(&xf[k])[0] == target;
                assert_eq!((mask >> k) & 1 == 1, expect, "mask lane {k}");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_mul8_matches_dalek() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512F not available, skipping");
            return;
        }
        let mut st = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..300 {
            let ab: [[u8; 32]; 8] = core::array::from_fn(|_| rand_fe_bytes(&mut st));
            let bb: [[u8; 32]; 8] = core::array::from_fn(|_| rand_fe_bytes(&mut st));
            let fa: [Fe; 8] = core::array::from_fn(|k| fe_frombytes(&ab[k]));
            let fb: [Fe; 8] = core::array::from_fn(|k| fe_frombytes(&bb[k]));
            let prod = unsafe {
                let r = avx512::mul8(&avx512::load8(&fa), &avx512::load8(&fb));
                avx512::store8(&r)
            };
            for k in 0..8 {
                assert_eq!(
                    fe_tobytes(&prod[k]),
                    EdwardsPoint::field_mul_reference(&ab[k], &bb[k]),
                    "mul8 lane {k}"
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_chain_y_only8_matches_dalek() {
        use curve25519_dalek::constants::ED25519_BASEPOINT_TABLE;
        use curve25519_dalek::scalar::Scalar;
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        const K: usize = 8;
        let step = ED25519_BASEPOINT_TABLE * &Scalar::from(8u64);
        let niels = unsafe { avx512::niels8_from_bytes(&step.niels_bytes()) };
        let mut stt = 0x1357_9bdf_0246_8aceu64;
        let starts: [_; 8] = core::array::from_fn(|_| {
            EdwardsPoint::mul_base_clamped({
                let mut s = rand_fe_bytes(&mut stt);
                s[0] &= 248;
                s[31] = (s[31] & 63) | 64;
                s
            })
        });
        let xyzt: [_; 8] = core::array::from_fn(|k| starts[k].xyzt_bytes());
        let p8 = unsafe {
            avx512::point8_from_xyzt(
                &core::array::from_fn(|k| fe_frombytes(&xyzt[k][0])),
                &core::array::from_fn(|k| fe_frombytes(&xyzt[k][1])),
                &core::array::from_fn(|k| fe_frombytes(&xyzt[k][2])),
                &core::array::from_fn(|k| fe_frombytes(&xyzt[k][3])),
            )
        };
        let mut out = [[[0u8; 32]; 8]; K];
        // (0, 0) accepts every byte0, so all lanes are fully encoded.
        unsafe { avx512::chain_y_only::<K>(p8, &niels, &[(0u8, 0u8)], &mut out) };
        for lane in 0..8 {
            let mut cur = starts[lane];
            for s in 0..K {
                let exp = cur.compress().to_bytes();
                assert_eq!(out[s][lane][..31], exp[..31], "chain8 lane {lane} step {s}");
                cur += step;
            }
        }
    }

    /// Throughput ceiling: run with
    /// `cargo test --release --lib simd4 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn bench_field_mul_ceiling() {
        use std::hint::black_box;
        use std::time::Instant;
        if !is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not available");
            return;
        }
        let mut st = 0xdead_beef_cafe_0001;
        let seed = rand_fe_bytes(&mut st);
        let f0 = fe_frombytes(&seed);

        const ITERS: u64 = 20_000_000;

        // Scalar: ITERS field muls.
        let mut a = f0;
        let mut b = f0;
        let t = Instant::now();
        for _ in 0..ITERS {
            a = fe_mul(black_box(&a), black_box(&b));
            b = fe_mul(black_box(&b), black_box(&a));
        }
        let scalar_secs = t.elapsed().as_secs_f64();
        black_box(a);
        let scalar_rate = (2 * ITERS) as f64 / scalar_secs / 1e6;

        // SIMD: ITERS mul4 calls = 4*ITERS field muls.
        let fa = unsafe { avx2::load4(&[f0; 4]) };
        let mut x = fa;
        let mut y = fa;
        let t = Instant::now();
        unsafe {
            for _ in 0..ITERS {
                x = avx2::mul4(black_box(&x), black_box(&y));
                y = avx2::mul4(black_box(&y), black_box(&x));
            }
        }
        let simd_secs = t.elapsed().as_secs_f64();
        black_box(unsafe { avx2::store4(&x) });
        let simd_rate = 4.0 * (2 * ITERS) as f64 / simd_secs / 1e6;

        eprintln!("scalar fe_mul : {scalar_rate:8.1} M muls/s");
        eprintln!("avx2  mul4    : {simd_rate:8.1} M muls/s  ({:.2}x)", simd_rate / scalar_rate);
    }
}
