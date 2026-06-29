# simd-popcnt

A fast Rust library for counting the number of 1 bits (bit population count,
a.k.a. Hamming weight) in a byte slice, using specialized CPU instructions:

| Architecture | Instruction sets used |
|--------------|-----------------------|
| x86 / x86-64 | POPCNT, AVX2 (Harley-Seal), AVX512-VPOPCNTDQ |
| AArch64      | NEON (mandatory), SVE (runtime-detected) |
| other        | portable pure-integer fallback |

This is a Rust port of the C/C++ header-only library
[libpopcnt](https://github.com/kimwalisch/libpopcnt) by Kim Walisch.

## API

The entire public API is a single function:

```rust
pub fn popcnt(data: &[u8]) -> u64
```

```rust
use simd_popcnt::popcnt;

assert_eq!(popcnt(&[]), 0);
assert_eq!(popcnt(&[0xFF]), 8);
assert_eq!(popcnt(&[0b1010_1010, 0b0000_0001]), 5);
```

## Performance

By default the crate detects the best available instruction set **at runtime**
(via `is_x86_feature_detected!` / `is_aarch64_feature_detected!`), caching the
result after the first call — exactly like the C library.

For the fastest possible code, build with:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

This selects the best SIMD path **at compile time** with zero runtime-dispatch
overhead, mirroring `-march=native` in the C library.

## Benchmarking

A command-line benchmark ships as an example (a developer tool — it is not built
when you depend on the crate). Measure throughput on your CPU with:

```sh
RUSTFLAGS="-C target-cpu=native" cargo run --release --example benchmark [array_bytes] [iters]
```

It prints the selected kernel and GB/s. Defaults: 16 KiB array, 10,000,000
iterations.

## How it works

* **Compile-time dispatch** — when a feature such as `avx2` or
  `avx512vpopcntdq` is in the static target feature set, `popcnt` calls the
  matching SIMD kernel directly with no runtime check (`#[cfg(target_feature)]`).
* **Runtime dispatch** — otherwise it probes the CPU once and caches the result.
* **Scalar POPCNT** — the scalar hot loop lives in a
  `#[target_feature(enable = "popcnt")]` function so that `u64::count_ones()`
  lowers to a hardware `popcnt` instruction (without it, LLVM emits a slow
  software sequence on runtime-detected CPUs — the same pitfall
  `__builtin_popcountll` has in C without `-mpopcnt`).
* **Bounds-check elimination** — hot loops use `slice::as_chunks::<N>()` so the
  compiler can prove SIMD loads are in bounds without `unsafe` indexing.

## ARM SVE on stable Rust

SVE intrinsics in `std::arch` are still nightly-gated
(`feature(stdarch_aarch64_sve)`). [`build.rs`](build.rs) probes whether the
active compiler accepts SVE intrinsics and, only if so, sets the
`simd_popcnt_have_sve` cfg. On stable Rust the SVE code is compiled out and the
NEON path is used; the rest of the crate builds on stable.

## Minimum supported Rust version

Rust **1.89** (the release that stabilized the AVX512 intrinsics in `std::arch`).
SVE support additionally requires a nightly toolchain.

## License

Licensed under either of

* MIT license ([LICENSE-MIT](LICENSE-MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
