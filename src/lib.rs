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
//! let data = [0xFFu8; 16];
//! assert_eq!(simd_popcnt::popcnt(&data), 128);
//! ```
//!
//! ## Performance
//!
//! For the fastest possible code, compile with `RUSTFLAGS="-C target-cpu=native"`.
//! This lets the crate select the best SIMD path at compile time with zero
//! runtime dispatch overhead. Without it, the best available instruction set is
//! detected at runtime (cached after the first call), exactly like the C library.

// Enable SVE intrinsics only when build.rs confirmed they compile on this rustc.
#![cfg_attr(simd_popcnt_have_sve, feature(stdarch_aarch64_sve))]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Count the number of 1 bits (set bits) in `data`.
///
/// This is the single public entry point. It dispatches to the fastest
/// implementation available for the target architecture and CPU.
///
/// # Examples
///
/// ```
/// assert_eq!(simd_popcnt::popcnt(&[]), 0);
/// assert_eq!(simd_popcnt::popcnt(&[0xFF]), 8);
/// assert_eq!(simd_popcnt::popcnt(&[0b1010_1010, 0b0000_0001]), 5);
/// ```
#[inline]
pub fn popcnt(data: &[u8]) -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        popcnt_x86(data)
    }

    #[cfg(target_arch = "aarch64")]
    {
        popcnt_aarch64(data)
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        popcnt_scalar_bitwise(data)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Portable scalar fallbacks (available on every architecture)
// ────────────────────────────────────────────────────────────────────────────

/// Pure-integer Hamming weight of a single `u64`.
///
/// Uses 12 arithmetic operations (one of which is a multiply) and emits no
/// POPCNT instruction, so it is correct on CPUs that lack hardware POPCNT.
/// <http://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation>
///
/// Part of the portable fallback chain; unused in fully-optimized builds (e.g.
/// `target-cpu=native` with hardware POPCNT) and on AArch64.
#[allow(dead_code)]
#[inline]
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

/// Read up to 8 trailing bytes of `rem` (length 0..=7) into the low bytes of a
/// `u64`, little-endian, zero-extending the rest. Mirrors the C tail loop
/// `val |= ((uint64_t) ptr[i + j]) << (j * 8)`.
#[inline]
fn tail_u64(rem: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf[..rem.len()].copy_from_slice(rem);
    u64::from_le_bytes(buf)
}

/// Scalar bit count using the pure-integer algorithm. Used as the universal
/// fallback and on CPUs without hardware POPCNT; unused in fully-optimized
/// x86 builds with hardware POPCNT and on AArch64.
#[allow(dead_code)]
fn popcnt_scalar_bitwise(data: &[u8]) -> u64 {
    let mut cnt = 0u64;
    let (chunks, rem) = data.as_chunks::<8>();
    for chunk in chunks {
        cnt += popcnt64_bitwise(u64::from_le_bytes(*chunk));
    }
    if !rem.is_empty() {
        cnt += popcnt64_bitwise(tail_u64(rem));
    }
    cnt
}

// ════════════════════════════════════════════════════════════════════════════
// x86 / x86-64
// ════════════════════════════════════════════════════════════════════════════

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn popcnt_x86(data: &[u8]) -> u64 {
    // ── Compile-time fast path: AVX512 statically enabled ───────────────────
    // (e.g. `-C target-cpu=native` on an AVX512 machine, or
    // `-C target-feature=+avx512vpopcntdq`). `popcnt_avx512` handles arrays of
    // ANY length including the masked tail, so a complete return path exists for
    // every size — no runtime detection and no dead code in the binary.
    #[cfg(target_feature = "avx512vpopcntdq")]
    {
        // For tiny arrays the AVX512 setup is not worth it.
        if data.len() >= 40 {
            unsafe { popcnt_avx512(data) }
        } else {
            popcnt_scalar_static(data)
        }
    }

    // ── Compile-time fast path: AVX2 (but not AVX512) statically enabled ─────
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512vpopcntdq")))]
    {
        let mut cnt = 0u64;
        let mut rest = data;
        // AVX2 is only faster for arrays >= 512 bytes.
        if data.len() >= 512 {
            let n = data.len() / 32 * 32;
            cnt += unsafe { popcnt_avx2(&data[..n]) };
            rest = &data[n..];
        }
        cnt + popcnt_scalar_static(rest)
    }

    // ── No compile-time SIMD: cached runtime detection ──────────────────────
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512vpopcntdq")))]
    {
        popcnt_x86_runtime(data)
    }
}

/// Scalar bit count chosen at COMPILE TIME based on the static feature set.
/// Used by the compile-time SIMD fast paths for small arrays and tails, so it
/// only exists when one of those paths is compiled in.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(target_feature = "avx2", target_feature = "avx512vpopcntdq")
))]
#[inline]
fn popcnt_scalar_static(data: &[u8]) -> u64 {
    // With `+popcnt` in the static feature set (always true under
    // `target-cpu=native`), `count_ones()` lowers to a single POPCNT.
    #[cfg(target_feature = "popcnt")]
    {
        // SAFETY: `popcnt` is statically enabled for the whole crate.
        unsafe { popcnt_scalar_hw(data) }
    }
    #[cfg(not(target_feature = "popcnt"))]
    {
        popcnt_scalar_bitwise(data)
    }
}

/// Cached runtime dispatch. Replaces the entire CPUID-guarded block of the C
/// `popcnt()`. `is_x86_feature_detected!` performs CPUID once and caches the
/// result internally, so we do not reimplement the C `libpopcnt_cpuid` global.
///
/// Only compiled when no SIMD feature is statically enabled — a `target-cpu`
/// that already guarantees AVX2/AVX512 takes the zero-overhead compile-time path
/// instead and never needs runtime detection.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(any(target_feature = "avx2", target_feature = "avx512vpopcntdq"))
))]
fn popcnt_x86_runtime(data: &[u8]) -> u64 {
    // AVX512 handles everything, including the masked tail, in one call.
    if data.len() >= 40
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vpopcntdq")
    {
        return unsafe { popcnt_avx512(data) };
    }

    let mut cnt = 0u64;
    let mut rest = data;

    // AVX2 is only faster for arrays >= 512 bytes.
    if data.len() >= 512 && is_x86_feature_detected!("avx2") {
        let n = data.len() / 32 * 32;
        cnt += unsafe { popcnt_avx2(&data[..n]) };
        rest = &data[n..];
    }

    // Scalar tail (or the whole array if AVX2 did not fire).
    // IMPORTANT: dispatch on POPCNT here. Without a
    // `#[target_feature(enable = "popcnt")]` function, `count_ones()` emits a
    // software fallback even on POPCNT-capable hardware.
    cnt += if is_x86_feature_detected!("popcnt") {
        unsafe { popcnt_scalar_hw(rest) }
    } else {
        popcnt_scalar_bitwise(rest)
    };

    cnt
}

/// Scalar bit count using the hardware POPCNT instruction.
///
/// The `#[target_feature(enable = "popcnt")]` attribute is what makes
/// `count_ones()` compile to a single `popcntq`. This function must only be
/// called after POPCNT support has been confirmed (statically via
/// `cfg(target_feature = "popcnt")`, or at runtime via
/// `is_x86_feature_detected!("popcnt")`).
///
/// Unused only in the unusual case of a SIMD feature enabled without `popcnt`
/// (e.g. `-C target-feature=+avx2` alone, without `+popcnt`).
#[allow(dead_code)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "popcnt")]
fn popcnt_scalar_hw(data: &[u8]) -> u64 {
    let mut cnt = 0u64;
    let (chunks, rem) = data.as_chunks::<8>();
    for chunk in chunks {
        cnt += u64::from_le_bytes(*chunk).count_ones() as u64; // emits: popcntq
    }
    if !rem.is_empty() {
        cnt += tail_u64(rem).count_ones() as u64;
    }
    cnt
}

// ── AVX2 ────────────────────────────────────────────────────────────────────

/// Carry-save adder for two 256-bit lanes (one full-adder bit-slice step).
/// Returns `(high, low)`. Returning a tuple (rather than C's out-pointers)
/// avoids borrow conflicts when the same variable is both an input and an
/// output, e.g. `(twos_a, ones) = csa256(ones, x, y)`.
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

/// AVX2 Harley-Seal popcount (4th iteration). Based on "Faster Population
/// Counts using AVX2 Instructions" by Lemire, Kurz and Muła (2016),
/// <https://arxiv.org/abs/1611.07612>.
///
/// `data.len()` must be a multiple of 32 (guaranteed by the caller).
///
/// Referenced by the AVX2 compile-time path and by runtime dispatch; both imply
/// AVX512 is not statically enabled, hence the `not(avx512vpopcntdq)` gate.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx512vpopcntdq")
))]
#[target_feature(enable = "avx2")]
// Keep the 16-vector CSA tree line-for-line with the C source for verifiability.
#[rustfmt::skip]
fn popcnt_avx2(data: &[u8]) -> u64 {
    let zero = _mm256_setzero_si256();
    let mut cnt = zero;
    let mut ones = zero;
    let mut twos = zero;
    let mut fours = zero;
    let mut eights = zero;
    // Per-iteration temporaries (deferred init; always written before read).
    let mut twos_a;
    let mut twos_b;
    let mut fours_a;
    let mut fours_b;
    let mut eights_a;
    let mut eights_b;
    let mut sixteens;

    // 16-vector (512-byte) Harley-Seal loop. `as_chunks::<512>()` yields
    // `&[u8; 512]` blocks, so `p.add(0)..=p.add(15)` are provably in bounds.
    let (blocks, tail) = data.as_chunks::<512>();
    for chunk in blocks {
        let p = chunk.as_ptr() as *const __m256i;
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

    // Single-vector tail: the 0..=15 complete 32-byte vectors not covered by the
    // 512-byte loop. `tail` is a multiple of 32 (the caller passed a
    // multiple-of-32 length), so nothing is left over.
    let (vecs, _) = tail.as_chunks::<32>();
    for chunk in vecs {
        let v = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
        cnt = _mm256_add_epi64(cnt, popcnt256(v));
    }

    // Horizontal sum of the four u64 lanes.
    let lanes: [u64; 4] = unsafe { core::mem::transmute(cnt) };
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

// ── AVX512 ──────────────────────────────────────────────────────────────────

/// AVX512-VPOPCNTDQ popcount. Handles arrays of any length: a 4×-unrolled
/// 256-byte loop, then a 64-byte loop, then a masked load for the final 1..=63
/// bytes.
///
/// Referenced by the AVX512 compile-time path and by runtime dispatch. Runtime
/// dispatch only exists when AVX2 is not statically enabled, so the gate is
/// "AVX512 statically enabled, or AVX2 not statically enabled".
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    any(not(target_feature = "avx2"), target_feature = "avx512vpopcntdq")
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
fn popcnt_avx512(data: &[u8]) -> u64 {
    let mut cnt = _mm512_setzero_si512();

    // 4× unrolled 64-byte loop (256 bytes per iteration).
    let (blocks, tail256) = data.as_chunks::<256>();
    for chunk in blocks {
        let p = chunk.as_ptr();
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
        // Mask with `len` low bits set. Equivalent to the C
        // `(__mmask64)(0xffffffffffffffff >> (i + 64 - size))`.
        let mask = (u64::MAX >> (64 - len)) as __mmask64;
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
fn popcnt_aarch64(data: &[u8]) -> u64 {
    // Compile-time: SVE statically enabled (e.g. `-C target-feature=+sve`) and
    // the toolchain provides SVE intrinsics.
    #[cfg(all(target_feature = "sve", simd_popcnt_have_sve))]
    {
        unsafe { popcnt_arm_sve(data) }
    }

    // NEON is mandatory on all AArch64 CPUs. `popcnt_neon` performs SVE runtime
    // dispatch internally when `simd_popcnt_have_sve` is set.
    #[cfg(not(all(target_feature = "sve", simd_popcnt_have_sve)))]
    {
        popcnt_neon(data)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn vpadalq(
    sum: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe { vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(t))) }
}

#[cfg(target_arch = "aarch64")]
fn popcnt_neon(data: &[u8]) -> u64 {
    use core::arch::aarch64::*;

    // ── ARM SVE runtime dispatch (compiled only when build.rs probe succeeded) ─
    #[cfg(simd_popcnt_have_sve)]
    {
        // Cached SVE detection, the equivalent of the C `libpopcnt_arm_sve`
        // global. Unlike C, no `simd_popcnt_` name prefix is needed: this
        // static is private to the function and cannot collide with user or
        // third-party symbols. Relaxed ordering is correct: the value is
        // idempotent (all threads compute the same result) and guards no
        // other data.
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
            return unsafe { popcnt_arm_sve(data) };
        }
    }

    // ── NEON path ───────────────────────────────────────────────────────────
    const CHUNK: usize = 64;
    let mut cnt = 0u64;
    let iters = data.len() / CHUNK;
    let ptr = data.as_ptr();

    if iters > 0 {
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

    // Scalar tail. On AArch64 `count_ones()` always lowers to the NEON `cnt`
    // sequence, so no separate POPCNT runtime check is needed.
    let rest = &data[iters * CHUNK..];
    let (chunks, rem) = rest.as_chunks::<8>();
    for chunk in chunks {
        cnt += u64::from_le_bytes(*chunk).count_ones() as u64;
    }
    if !rem.is_empty() {
        cnt += tail_u64(rem).count_ones() as u64;
    }
    cnt
}

// ── ARM SVE ─────────────────────────────────────────────────────────────────

#[cfg(all(target_arch = "aarch64", simd_popcnt_have_sve))]
fn has_arm_sve() -> bool {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        // HWCAP_SVE, defined literally to avoid needing <asm/hwcap.h>, which is
        // not installed by default on some Linux distros.
        const HWCAP_SVE: u64 = 1 << 22;
        let hwcaps = unsafe { libc::getauxval(libc::AT_HWCAP) };
        hwcaps & HWCAP_SVE != 0
    }
    #[cfg(target_os = "windows")]
    {
        // PF_ARM_SVE_INSTRUCTIONS_AVAILABLE = 39, defined literally so this also
        // builds with older Windows SDKs that predate the constant.
        const PF_ARM_SVE_INSTRUCTIONS_AVAILABLE: u32 = 39;
        unsafe {
            windows_sys::Win32::System::Threading::IsProcessorFeaturePresent(
                PF_ARM_SVE_INSTRUCTIONS_AVAILABLE,
            ) != 0
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows")))]
    {
        false
    }
}

/// ARM SVE popcount. Gated by both `simd_popcnt_have_sve` (intrinsics compile on
/// this rustc) and `#[target_feature(enable = "sve")]` (SVE instructions are
/// generated even without `-C target-feature=+sve`).
#[cfg(all(target_arch = "aarch64", simd_popcnt_have_sve))]
#[target_feature(enable = "sve")]
fn popcnt_arm_sve(data: &[u8]) -> u64 {
    use core::arch::aarch64::*;

    unsafe {
        let mut i = 0usize;
        let mut vcnt = svdup_n_u64(0);
        let vl = svcntb() as usize; // SVE vector length in bytes (hardware-defined)
        let ptr = data.as_ptr();
        let len = data.len();

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

        // Predicated tail loop. `svld1_u8` with a tail predicate zero-fills the
        // inactive lanes, so no separate byte-by-byte tail is needed.
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
    fn reference(data: &[u8]) -> u64 {
        data.iter().map(|b| b.count_ones() as u64).sum()
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
            let data = vec![0xFFu8; size];
            assert_eq!(popcnt(&data), size as u64 * 8, "size={size}");
        }
    }

    #[test]
    fn all_zeros() {
        let data = vec![0u8; 65536];
        assert_eq!(popcnt(&data), 0);
    }

    #[test]
    fn single_bits() {
        for bit in 0u64..64 {
            let val = 1u64 << bit;
            assert_eq!(popcnt(&val.to_le_bytes()), 1, "bit={bit}");
        }
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
        let mut data = vec![0u8; MAX_SIZE + MAX_OFF + 1];
        for b in data.iter_mut() {
            *b = (next() & 0xFF) as u8;
        }

        // Every size from 0 up through the AVX2/AVX512 active range, plus a few
        // larger ones, exercised at multiple start offsets so alignment varies.
        let sizes =
            (0usize..=600).chain([1023, 1024, 1025, 2048, 4095, 4096, 4097, 4608, MAX_SIZE]);
        for size in sizes {
            for &off in &[0usize, 1, 3, MAX_OFF] {
                let slice = &data[off..off + size];
                assert_eq!(popcnt(slice), reference(slice), "size={size} off={off}");
            }
        }
    }

    /// Faithful port of `libpopcnt/test/test1.cpp` (and the identical `test2.c`):
    /// count `popcnt()` of every suffix `data[i..]` and verify against an
    /// independent byte-wise reference. The sweep covers every length from 0 up
    /// to the array size, each at a shifting start offset, so length and
    /// alignment both vary across the run.
    ///
    /// Size defaults to 20_000 to keep `cargo test` fast — the sweep is O(n²) in
    /// the work `popcnt` performs. Override with the `SIMD_POPCNT_TEST_SIZE`
    /// environment variable for a heavier run matching the C test's default,
    /// e.g. `SIMD_POPCNT_TEST_SIZE=100000 cargo test --release suffix_sweep`.
    #[test]
    fn suffix_sweep() {
        let size = std::env::var("SIMD_POPCNT_TEST_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(20_000);

        // All-ones array (matches test1.cpp's initial check).
        let ones = vec![0xFFu8; size];
        check_all_suffixes(&ones);

        // Deterministic pseudo-random array. The C test seeds with time(0);
        // a fixed xorshift seed instead keeps any failure reproducible.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut data = vec![0u8; size];
        for b in data.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = state as u8;
        }
        check_all_suffixes(&data);
    }

    /// Assert `popcnt(&data[i..])` for every `i`, against an O(1) prefix-sum
    /// reference built from `popcnt64_bitwise` per byte — the same independent
    /// oracle `test1.cpp` uses. Only `popcnt` itself does O(n) work per suffix.
    fn check_all_suffixes(data: &[u8]) {
        let total: u64 = data.iter().map(|&b| popcnt64_bitwise(b as u64)).sum();
        let mut prefix = 0u64; // popcount of data[..i]
        for (i, &byte) in data.iter().enumerate() {
            assert_eq!(popcnt(&data[i..]), total - prefix, "suffix at offset {i}");
            prefix += popcnt64_bitwise(byte as u64);
        }
        // Empty suffix.
        assert_eq!(popcnt(&data[data.len()..]), 0);
    }
}
