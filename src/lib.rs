//! # simd-popcnt
//!
//! Count the number of 1 bits (bit population count, a.k.a. Hamming weight) in
//! a byte slice as quickly as possible using specialized CPU instructions:
//! POPCNT, AVX2 and AVX512 on x86/x86-64, and NEON and SVE on AArch64.
//!
//! This is a Rust port of the C/C++ [`libpopcnt.h`](https://github.com/kimwalisch/libpopcnt)
//! header-only library by Kim Walisch.
//!
//! ## Usage
//!
//! ```
//! let bytes = [0xFFu8; 16];
//! assert_eq!(simd_popcnt::popcnt(&bytes), 128);
//! ```
//!
//! ## Performance
//!
//! For the fastest possible code, compile with `RUSTFLAGS="-C target-cpu=native"`.
//! This lets the crate select the best SIMD path at compile time with zero
//! runtime dispatch overhead. Otherwise the best available instruction set is
//! detected once at runtime and cached.

// Enable SVE intrinsics only when build.rs confirmed they compile on this rustc.
#![cfg_attr(simd_popcnt_have_sve, feature(stdarch_aarch64_sve))]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Counts the number of one bits (population count) in `bytes`.
///
/// Dispatches to the fastest implementation for the running CPU: SIMD where
/// available, a scalar fallback otherwise.
///
/// To count the bits in a slice of a wider integer type (`&[u64]`, `&[u32]`, …),
/// use the [`PopcntExt::popcnt`] method rather than converting to bytes by hand.
///
/// # Examples
///
/// ```
/// assert_eq!(simd_popcnt::popcnt(&[]), 0);
/// assert_eq!(simd_popcnt::popcnt(&[0xFF, 0x0F]), 12);
/// ```
#[must_use]
#[inline]
pub fn popcnt(bytes: &[u8]) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        popcnt_x86(bytes)
    }

    #[cfg(target_arch = "aarch64")]
    {
        popcnt_aarch64(bytes)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        popcnt_scalar(bytes)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Ergonomic extension trait for integer slices
// ────────────────────────────────────────────────────────────────────────────

/// Adds a [`popcnt`](PopcntExt::popcnt) method to slices of the built-in integer
/// types, counting their bits without a manual byte cast. Implemented for slices,
/// arrays and `Vec`s of `u8`/`u16`/`u32`/`u64`/`u128`/`usize` and their signed
/// counterparts; bring it into scope with `use simd_popcnt::PopcntExt;`.
///
/// ```
/// use simd_popcnt::PopcntExt;
///
/// let words: &[u64] = &[u64::MAX, 0x0F0F_0F0F_0F0F_0F0F];
/// assert_eq!(words.popcnt(), 64 + 32);
/// assert_eq!(vec![1u32, 2, 3].popcnt(), 4);
/// ```
pub trait PopcntExt {
    /// Count the total number of 1 bits across all elements of the slice.
    #[must_use]
    fn popcnt(&self) -> u64;
}

/// Implement [`PopcntExt`] for `[$t]` by viewing the slice as bytes and
/// delegating to [`popcnt`]. Population count is byte-order independent, so this
/// is correct on both little- and big-endian targets.
macro_rules! impl_popcnt_ext {
    ($($t:ty),+ $(,)?) => {$(
        impl PopcntExt for [$t] {
            #[inline]
            fn popcnt(&self) -> u64 {
                // SAFETY: `$t` is a plain integer (no padding, every bit pattern
                // valid) and `u8` is always 1-aligned, so the slice is a valid
                // `&[u8]` of `size_of_val` bytes.
                let bytes = unsafe {
                    core::slice::from_raw_parts(
                        self.as_ptr().cast::<u8>(),
                        core::mem::size_of_val(self),
                    )
                };
                popcnt(bytes)
            }
        }
    )+};
}

impl_popcnt_ext!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

// ────────────────────────────────────────────────────────────────────────────
// Portable scalar fallbacks (available on every architecture)
// ────────────────────────────────────────────────────────────────────────────

/// Packs the trailing `rem.len()` (0..=7) bytes into a zero-padded `u64` for
/// counting. Uses native byte order (the popcount is order-independent), which
/// avoids a byte swap on big-endian targets.
#[inline]
fn tail_u64(rem: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf[..rem.len()].copy_from_slice(rem);
    u64::from_ne_bytes(buf)
}

/// Scalar population count loop, summing `count_ones()` over 8-byte chunks.
/// `count_ones()` is inlined in release and lowers to the target's hardware
/// popcount where available (x86 POPCNT, PowerPC popcntd, WebAssembly
/// `i64.popcnt`, …), otherwise to an inline bit-twiddling sequence — never a
/// library call. Shared with the POPCNT-`target_feature` variant below.
macro_rules! popcnt_scalar_loop {
    ($bytes:expr) => {{
        let mut cnt = 0u64;
        let (chunks, rem) = $bytes.as_chunks::<8>();
        for chunk in chunks {
            cnt += u64::from_ne_bytes(*chunk).count_ones() as u64;
        }
        if !rem.is_empty() {
            cnt += tail_u64(rem).count_ones() as u64;
        }
        cnt
    }};
}

/// Portable scalar population count via [`u64::count_ones`].
#[allow(dead_code)]
fn popcnt_scalar(bytes: &[u8]) -> u64 {
    popcnt_scalar_loop!(bytes)
}

// ════════════════════════════════════════════════════════════════════════════
// x86 / x86-64
// ════════════════════════════════════════════════════════════════════════════

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn popcnt_x86(bytes: &[u8]) -> u64 {
    // Compile-time AVX512 path (e.g. with `-C target-cpu=native`).
    #[cfg(target_feature = "avx512vpopcntdq")]
    {
        // AVX512 isn't worth its setup cost for tiny arrays.
        if bytes.len() >= 40 {
            unsafe { popcnt_avx512(bytes) }
        } else {
            popcnt_scalar_static(bytes)
        }
    }

    // Compile-time AVX2 path.
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512vpopcntdq")))]
    {
        let mut cnt = 0u64;
        let mut rest = bytes;
        // AVX2 only wins for arrays >= 512 bytes.
        if bytes.len() >= 512 {
            let n = bytes.len() / 32 * 32;
            cnt += unsafe { popcnt_avx2(&bytes[..n]) };
            rest = &bytes[n..];
        }
        cnt + popcnt_scalar_static(rest)
    }

    // No SIMD enabled at compile time: detect at runtime.
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512vpopcntdq")))]
    {
        popcnt_x86_runtime(bytes)
    }
}

/// Scalar count for the compile-time SIMD paths' small arrays and tails:
/// hardware POPCNT when statically enabled, otherwise the integer fallback.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(target_feature = "avx2", target_feature = "avx512vpopcntdq")
))]
#[inline]
fn popcnt_scalar_static(bytes: &[u8]) -> u64 {
    #[cfg(target_feature = "popcnt")]
    {
        // SAFETY: `popcnt` is statically enabled for the whole crate.
        unsafe { popcnt_scalar_hw(bytes) }
    }
    #[cfg(not(target_feature = "popcnt"))]
    {
        popcnt_scalar(bytes)
    }
}

/// Runtime dispatch using cached CPU feature detection. Only compiled when no
/// SIMD feature is statically enabled (otherwise the compile-time paths run).
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(any(target_feature = "avx2", target_feature = "avx512vpopcntdq"))
))]
fn popcnt_x86_runtime(bytes: &[u8]) -> u64 {
    // AVX512: not worth its setup cost below ~40 bytes, handles any length.
    if bytes.len() >= 40
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vpopcntdq")
    {
        return unsafe { popcnt_avx512(bytes) };
    }

    let mut cnt = 0u64;
    let mut rest = bytes;

    // AVX2 only wins for arrays >= 512 bytes.
    if bytes.len() >= 512 && is_x86_feature_detected!("avx2") {
        let n = bytes.len() / 32 * 32;
        cnt += unsafe { popcnt_avx2(&bytes[..n]) };
        rest = &bytes[n..];
    }

    // Scalar tail (or the whole array if AVX2 didn't fire). Dispatching on
    // POPCNT is essential: outside a `#[target_feature(enable = "popcnt")]`
    // function, `count_ones()` compiles to a software fallback even on
    // POPCNT-capable CPUs.
    cnt += if is_x86_feature_detected!("popcnt") {
        unsafe { popcnt_scalar_hw(rest) }
    } else {
        popcnt_scalar(rest)
    };

    cnt
}

/// Scalar population count via the hardware POPCNT instruction. The
/// `#[target_feature(enable = "popcnt")]` attribute is what lets `count_ones()`
/// lower to a single `popcnt`; only call it once POPCNT support is confirmed.
#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "popcnt")]
fn popcnt_scalar_hw(bytes: &[u8]) -> u64 {
    popcnt_scalar_loop!(bytes) // count_ones() lowers to popcntq here
}

// ── AVX2 ────────────────────────────────────────────────────────────────────

/// Carry-save adder: returns the `(carry, sum)` bit-planes of `a + b + c`,
/// computed across all lanes in parallel.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx512vpopcntdq")
))]
#[target_feature(enable = "avx2")]
#[inline]
fn csa256(a: __m256i, b: __m256i, c: __m256i) -> (__m256i, __m256i) {
    let u = _mm256_xor_si256(a, b);
    let h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    let l = _mm256_xor_si256(u, c);
    (h, l)
}

/// Per-byte population count of a 256-bit vector using the nibble lookup, then
/// horizontal sum of each 8-byte lane via `_mm256_sad_epu8` (result in 4 u64s).
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx512vpopcntdq")
))]
#[target_feature(enable = "avx2")]
#[inline]
fn popcnt256(v: __m256i) -> __m256i {
    let lookup1 = _mm256_setr_epi8(
        4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7,
        7, 8,
    );
    let lookup2 = _mm256_setr_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1,
        1, 0,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    let popcnt1 = _mm256_shuffle_epi8(lookup1, lo);
    let popcnt2 = _mm256_shuffle_epi8(lookup2, hi);
    _mm256_sad_epu8(popcnt1, popcnt2)
}

/// AVX2 Harley-Seal population count (4th iteration), from "Faster Population
/// Counts using AVX2 Instructions" by Lemire, Kurz and Muła (2016),
/// <https://arxiv.org/abs/1611.07612>.
///
/// `bytes.len()` must be a multiple of 32.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx512vpopcntdq")
))]
#[target_feature(enable = "avx2")]
// Hand-aligned: keep the 16-way CSA tree readable.
#[rustfmt::skip]
fn popcnt_avx2(bytes: &[u8]) -> u64 {
    let zero = _mm256_setzero_si256();
    let mut cnt = zero;
    let mut ones = zero;
    let mut twos = zero;
    let mut fours = zero;
    let mut eights = zero;
    let mut twos_a;
    let mut twos_b;
    let mut fours_a;
    let mut fours_b;
    let mut eights_a;
    let mut eights_b;
    let mut sixteens;

    // 16 vectors (512 bytes) per iteration.
    let (blocks, tail) = bytes.as_chunks::<512>();
    for chunk in blocks {
        let p = chunk.as_ptr() as *const __m256i;
        // SAFETY: `chunk` is 512 bytes, so all 16 loads (32 bytes each) are in bounds.
        unsafe {
            (twos_a, ones) = csa256(ones, _mm256_loadu_si256(p.add(0)), _mm256_loadu_si256(p.add(1)));
            (twos_b, ones) = csa256(ones, _mm256_loadu_si256(p.add(2)), _mm256_loadu_si256(p.add(3)));
            (fours_a, twos) = csa256(twos, twos_a, twos_b);
            (twos_a, ones) = csa256(ones, _mm256_loadu_si256(p.add(4)), _mm256_loadu_si256(p.add(5)));
            (twos_b, ones) = csa256(ones, _mm256_loadu_si256(p.add(6)), _mm256_loadu_si256(p.add(7)));
            (fours_b, twos) = csa256(twos, twos_a, twos_b);
            (eights_a, fours) = csa256(fours, fours_a, fours_b);
            (twos_a, ones) = csa256(ones, _mm256_loadu_si256(p.add(8)), _mm256_loadu_si256(p.add(9)));
            (twos_b, ones) = csa256(ones, _mm256_loadu_si256(p.add(10)), _mm256_loadu_si256(p.add(11)));
            (fours_a, twos) = csa256(twos, twos_a, twos_b);
            (twos_a, ones) = csa256(ones, _mm256_loadu_si256(p.add(12)), _mm256_loadu_si256(p.add(13)));
            (twos_b, ones) = csa256(ones, _mm256_loadu_si256(p.add(14)), _mm256_loadu_si256(p.add(15)));
            (fours_b, twos) = csa256(twos, twos_a, twos_b);
            (eights_b, fours) = csa256(fours, fours_a, fours_b);
            (sixteens, eights) = csa256(eights, eights_a, eights_b);
            cnt = _mm256_add_epi64(cnt, popcnt256(sixteens));
        }
    }

    cnt = _mm256_slli_epi64(cnt, 4);
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(popcnt256(eights), 3));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(popcnt256(fours), 2));
    cnt = _mm256_add_epi64(cnt, _mm256_slli_epi64(popcnt256(twos), 1));
    cnt = _mm256_add_epi64(cnt, popcnt256(ones));

    // Remaining whole 32-byte vectors.
    let (vecs, _) = tail.as_chunks::<32>();
    for chunk in vecs {
        let v = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
        cnt = _mm256_add_epi64(cnt, popcnt256(v));
    }

    // Sum the four 64-bit lanes.
    // SAFETY: `__m256i` and `[u64; 4]` are both 32 bytes with no invalid bit patterns.
    let lanes: [u64; 4] = unsafe { core::mem::transmute(cnt) };
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

// ── AVX512 ──────────────────────────────────────────────────────────────────

/// AVX512-VPOPCNTDQ population count, handling any length: a 4×-unrolled
/// 256-byte loop, then a 64-byte loop, then a masked load for the final
/// 1..=63 bytes.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(not(target_feature = "avx2"), target_feature = "avx512vpopcntdq")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
fn popcnt_avx512(bytes: &[u8]) -> u64 {
    let mut cnt = _mm512_setzero_si512();

    // 4× unrolled 64-byte loop (256 bytes per iteration).
    let (blocks, tail256) = bytes.as_chunks::<256>();
    for chunk in blocks {
        let p = chunk.as_ptr();
        // SAFETY: `chunk` is 256 bytes, so the four 64-byte loads are in bounds.
        unsafe {
            let v0 = _mm512_loadu_si512(p.add(0) as *const _);
            let v1 = _mm512_loadu_si512(p.add(64) as *const _);
            let v2 = _mm512_loadu_si512(p.add(128) as *const _);
            let v3 = _mm512_loadu_si512(p.add(192) as *const _);
            cnt = _mm512_add_epi64(cnt, _mm512_popcnt_epi64(v0));
            cnt = _mm512_add_epi64(cnt, _mm512_popcnt_epi64(v1));
            cnt = _mm512_add_epi64(cnt, _mm512_popcnt_epi64(v2));
            cnt = _mm512_add_epi64(cnt, _mm512_popcnt_epi64(v3));
        }
    }

    // Remaining complete 64-byte blocks.
    let (vecs, tail64) = tail256.as_chunks::<64>();
    for chunk in vecs {
        let v = unsafe { _mm512_loadu_si512(chunk.as_ptr() as *const _) };
        cnt = _mm512_add_epi64(cnt, _mm512_popcnt_epi64(v));
    }

    // Masked load for the final 1..=63 bytes.
    if !tail64.is_empty() {
        let len = tail64.len();
        // Mask covering the final `len` bytes.
        let mask = (u64::MAX >> (64 - len)) as __mmask64;
        // SAFETY: the mask selects only the `len` valid bytes; masked-off lanes
        // are not accessed.
        unsafe {
            let v = _mm512_maskz_loadu_epi8(mask, tail64.as_ptr() as *const _);
            cnt = _mm512_add_epi64(cnt, _mm512_popcnt_epi64(v));
        }
    }

    _mm512_reduce_add_epi64(cnt) as u64
}

// ════════════════════════════════════════════════════════════════════════════
// AArch64
// ════════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
fn popcnt_aarch64(bytes: &[u8]) -> u64 {
    // Compile-time SVE path.
    #[cfg(all(target_feature = "sve", simd_popcnt_have_sve))]
    {
        unsafe { popcnt_arm_sve(bytes) }
    }

    // NEON baseline; `popcnt_neon` dispatches to SVE at runtime when available.
    #[cfg(not(all(target_feature = "sve", simd_popcnt_have_sve)))]
    {
        popcnt_neon(bytes)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn vpadalq(sum: uint64x2_t, t: uint8x16_t) -> uint64x2_t {
    unsafe { vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(t))) }
}

#[cfg(target_arch = "aarch64")]
fn popcnt_neon(bytes: &[u8]) -> u64 {
    // Runtime SVE dispatch (present only when the build probe enabled SVE).
    #[cfg(simd_popcnt_have_sve)]
    {
        // Cached SVE support (-1 = unknown). Relaxed ordering is fine: the
        // value is idempotent and guards no other state.
        use core::sync::atomic::{AtomicI32, Ordering};
        static SVE_SUPPORT: AtomicI32 = AtomicI32::new(-1);
        let cached = SVE_SUPPORT.load(Ordering::Relaxed);
        let has_sve = if cached == -1 {
            let v = has_arm_sve() as i32;
            SVE_SUPPORT.store(v, Ordering::Relaxed);
            v
        } else {
            cached
        };
        if has_sve != 0 {
            return unsafe { popcnt_arm_sve(bytes) };
        }
    }

    // NEON path.
    const CHUNK: usize = 64;
    let mut cnt = 0u64;
    let iters = bytes.len() / CHUNK;
    let ptr = bytes.as_ptr();

    if iters > 0 {
        // SAFETY: `iters = len / 64`, so every `vld4q_u8` at `i * 64` (i < iters)
        // reads 64 in-bounds bytes; the final store targets a local array.
        unsafe {
            let mut sum = vdupq_n_u64(0);
            let zero = vdupq_n_u8(0);
            let mut i = 0usize;

            while i < iters {
                let mut t0 = zero;
                let mut t1 = zero;
                let mut t2 = zero;
                let mut t3 = zero;

                // Accumulate at most 31 chunks before draining into `sum`:
                // 31 × 8 bits = 248 ≤ 255 guarantees no u8 lane overflow.
                let limit = (i + 31).min(iters);
                while i < limit {
                    let input = vld4q_u8(ptr.add(i * CHUNK));
                    t0 = vaddq_u8(t0, vcntq_u8(input.0));
                    t1 = vaddq_u8(t1, vcntq_u8(input.1));
                    t2 = vaddq_u8(t2, vcntq_u8(input.2));
                    t3 = vaddq_u8(t3, vcntq_u8(input.3));
                    i += 1;
                }

                sum = vpadalq(sum, t0);
                sum = vpadalq(sum, t1);
                sum = vpadalq(sum, t2);
                sum = vpadalq(sum, t3);
            }

            let mut tmp = [0u64; 2];
            vst1q_u64(tmp.as_mut_ptr(), sum);
            cnt += tmp[0] + tmp[1];
        }
    }

    // Scalar tail. On AArch64 `count_ones()` always lowers to NEON `cnt`, so no
    // POPCNT runtime check is needed here.
    let rest = &bytes[iters * CHUNK..];
    cnt += popcnt_scalar_loop!(rest);
    cnt
}

// ── ARM SVE ─────────────────────────────────────────────────────────────────

#[cfg(all(target_arch = "aarch64", simd_popcnt_have_sve))]
fn has_arm_sve() -> bool {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let hwcaps = unsafe { libc::getauxval(libc::AT_HWCAP) };
        hwcaps & libc::HWCAP_SVE != 0
    }
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::Threading::IsProcessorFeaturePresent;
        // windows-sys 0.59 doesn't define PF_ARM_SVE_INSTRUCTIONS_AVAILABLE (39)
        // yet, so use the literal value.
        const PF_ARM_SVE_INSTRUCTIONS_AVAILABLE: u32 = 39;
        unsafe { IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE) != 0 }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows")))]
    {
        false
    }
}

/// SVE population count: a 4×-unrolled main loop over full vectors, then a
/// predicated tail loop that needs no separate scalar remainder.
#[cfg(all(target_arch = "aarch64", simd_popcnt_have_sve))]
#[target_feature(enable = "sve")]
fn popcnt_arm_sve(bytes: &[u8]) -> u64 {
    // SAFETY: the loop bound keeps each full load within `len`; the tail loop's
    // predicate masks off any lanes past the end.
    unsafe {
        let mut i = 0usize;
        let mut vcnt = svdup_n_u64(0);
        let vl = svcntb() as usize; // SVE vector length in bytes (hardware-defined)
        let ptr = bytes.as_ptr();
        let len = bytes.len();

        // 4× unrolled full-predicate loop.
        while i + vl * 4 <= len {
            let v0 = svreinterpret_u64_u8(svld1_u8(svptrue_b8(), ptr.add(i)));
            let v1 = svreinterpret_u64_u8(svld1_u8(svptrue_b8(), ptr.add(i + vl)));
            let v2 = svreinterpret_u64_u8(svld1_u8(svptrue_b8(), ptr.add(i + vl * 2)));
            let v3 = svreinterpret_u64_u8(svld1_u8(svptrue_b8(), ptr.add(i + vl * 3)));
            vcnt = svadd_u64_x(svptrue_b64(), vcnt, svcnt_u64_x(svptrue_b64(), v0));
            vcnt = svadd_u64_x(svptrue_b64(), vcnt, svcnt_u64_x(svptrue_b64(), v1));
            vcnt = svadd_u64_x(svptrue_b64(), vcnt, svcnt_u64_x(svptrue_b64(), v2));
            vcnt = svadd_u64_x(svptrue_b64(), vcnt, svcnt_u64_x(svptrue_b64(), v3));
            i += vl * 4;
        }

        // Predicated tail: the load zero-fills inactive lanes, so no separate
        // scalar remainder is needed.
        let mut pg = svwhilelt_b8_u64(i as u64, len as u64);
        while svptest_any(svptrue_b8(), pg) {
            let v = svreinterpret_u64_u8(svld1_u8(pg, ptr.add(i)));
            vcnt = svadd_u64_x(svptrue_b64(), vcnt, svcnt_u64_x(svptrue_b64(), v));
            i += vl;
            pg = svwhilelt_b8_u64(i as u64, len as u64);
        }

        svaddv_u64(svptrue_b64(), vcnt)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: count bits one byte at a time.
    fn reference(bytes: &[u8]) -> u64 {
        bytes.iter().map(|b| b.count_ones() as u64).sum()
    }

    /// Independent integer-only popcount oracle (does not use `count_ones`),
    /// so the sweep cross-checks the crate against a different algorithm.
    fn popcnt64_bitwise(x: u64) -> u64 {
        const M1: u64 = 0x5555555555555555;
        const M2: u64 = 0x3333333333333333;
        const M4: u64 = 0x0F0F0F0F0F0F0F0F;
        const H01: u64 = 0x0101010101010101;
        let x = x - ((x >> 1) & M1);
        let x = (x & M2) + ((x >> 2) & M2);
        let x = (x + (x >> 4)) & M4;
        x.wrapping_mul(H01) >> 56
    }

    #[test]
    fn empty() {
        assert_eq!(popcnt(&[]), 0);
    }

    #[test]
    fn all_ones() {
        for &size in &[
            0, 1, 7, 8, 31, 32, 39, 40, 63, 64, 255, 256, 511, 512, 4095, 4096, 65537,
        ] {
            let bytes = vec![0xFFu8; size];
            assert_eq!(popcnt(&bytes), size as u64 * 8, "size={size}");
        }
    }

    #[test]
    fn all_zeros() {
        let bytes = vec![0u8; 65536];
        assert_eq!(popcnt(&bytes), 0);
    }

    #[test]
    fn single_bits() {
        for bit in 0u64..64 {
            let val = 1u64 << bit;
            assert_eq!(popcnt(&val.to_le_bytes()), 1, "bit={bit}");
        }
    }

    /// `PopcntExt::popcnt` on each integer width must equal the per-element
    /// `count_ones()` sum (an oracle independent of the byte reinterpretation).
    #[test]
    fn ext_trait_widths() {
        let u8s: &[u8] = &[0xFF, 0x0F, 0x00, 0xAB, 0x01];
        assert_eq!(
            u8s.popcnt(),
            u8s.iter().map(|x| x.count_ones() as u64).sum()
        );

        let u16s: &[u16] = &[0xFFFF, 0x0F0F, 0x1234, 0];
        assert_eq!(
            u16s.popcnt(),
            u16s.iter().map(|x| x.count_ones() as u64).sum()
        );

        let u32s: &[u32] = &[u32::MAX, 0, 0x8000_0001];
        assert_eq!(
            u32s.popcnt(),
            u32s.iter().map(|x| x.count_ones() as u64).sum()
        );

        let u64s: &[u64] = &[u64::MAX, 0x0F0F_0F0F_0F0F_0F0F, 0];
        assert_eq!(
            u64s.popcnt(),
            u64s.iter().map(|x| x.count_ones() as u64).sum()
        );

        // Signed types and arrays resolve through the same impls (the doc
        // example covers `Vec`).
        let i32s = [-1i32, 0, 1, i32::MIN];
        assert_eq!(
            i32s.popcnt(),
            i32s.iter().map(|x| x.count_ones() as u64).sum()
        );
        assert_eq!([u128::MAX, 0].popcnt(), 128);
    }

    /// Sweep every boundary-relevant size against the byte-wise reference using
    /// a deterministic pseudo-random fill (xorshift). Covers tail handling,
    /// the AVX2/AVX512 thresholds and multiple Harley-Seal outer iterations.
    #[test]
    fn pseudorandom_all_sizes() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // Largest size + largest offset exercised below, plus margin. 4695
        // bytes spans several 512-byte Harley-Seal iterations.
        const MAX_SIZE: usize = 4695;
        const MAX_OFF: usize = 7;
        let mut bytes = vec![0u8; MAX_SIZE + MAX_OFF + 1];
        for b in bytes.iter_mut() {
            *b = (next() & 0xFF) as u8;
        }

        // Every size from 0 up through the AVX2/AVX512 active range, plus a few
        // larger ones, exercised at multiple start offsets so alignment varies.
        let sizes =
            (0usize..=600).chain([1023, 1024, 1025, 2048, 4095, 4096, 4097, 4608, MAX_SIZE]);
        for size in sizes {
            for &off in &[0usize, 1, 3, MAX_OFF] {
                let slice = &bytes[off..off + size];
                assert_eq!(popcnt(slice), reference(slice), "size={size} off={off}");
            }
        }
    }

    /// Verify `popcnt()` of every suffix `bytes[i..]` against an independent
    /// byte-wise reference, covering every length and a range of start
    /// alignments in one sweep.
    ///
    /// Size defaults to 20_000 to keep `cargo test` fast — the sweep is O(n²) in
    /// the work `popcnt` performs. Override with `SIMD_POPCNT_TEST_SIZE` for a
    /// heavier run, e.g. `SIMD_POPCNT_TEST_SIZE=100000 cargo test --release suffix_sweep`.
    #[test]
    fn suffix_sweep() {
        let size = std::env::var("SIMD_POPCNT_TEST_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(20_000);

        // All-ones array.
        let ones = vec![0xFFu8; size];
        check_all_suffixes(&ones);

        // Deterministic pseudo-random array (fixed seed → reproducible failures).
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut bytes = vec![0u8; size];
        for b in bytes.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = state as u8;
        }
        check_all_suffixes(&bytes);
    }

    /// Assert `popcnt(&bytes[i..])` for every `i` against an O(1) prefix-sum
    /// reference, so only `popcnt` itself does O(n) work per suffix.
    fn check_all_suffixes(bytes: &[u8]) {
        let total: u64 = bytes.iter().map(|&b| popcnt64_bitwise(b as u64)).sum();
        let mut prefix = 0u64; // popcount of bytes[..i]
        for (i, &byte) in bytes.iter().enumerate() {
            assert_eq!(popcnt(&bytes[i..]), total - prefix, "suffix at offset {i}");
            prefix += popcnt64_bitwise(byte as u64);
        }
        // Empty suffix.
        assert_eq!(popcnt(&bytes[bytes.len()..]), 0);
    }
}
