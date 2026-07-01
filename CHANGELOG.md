# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-01

### Added
- `no_std` support via a new default-on `std` feature. The crate needs the Rust
  standard library only for runtime SIMD dispatch (CPU feature detection);
  build with `default-features = false` for a `no_std` build, which selects its
  SIMD path at compile time and otherwise falls back to the portable integer
  algorithm. The crate also compiles as `no_std` *automatically* whenever a
  build has no runtime dispatch left to do — e.g. under `-C target-cpu=native`
  on x86, or on any non-x86/AArch64 target — even with the `std` feature on.
- Native (`-C target-cpu=native`) CI jobs on Linux, Windows, macOS and ARM64
  Linux that exercise the compile-time SIMD paths, plus `no_std` CI jobs that
  build for the bare-metal `x86_64-unknown-none` and `aarch64-unknown-none`
  targets.
- Valgrind memcheck CI jobs (x86-64 and AArch64) that run the test suite and
  benchmark under memcheck to check the SIMD code for memory-safety undefined
  behavior.
- README badges: crates.io version, docs.rs, CI status, and minimum supported
  Rust version.

### Changed
- Annotated the dispatch glue and SIMD kernels with `#[inline]` so the runtime
  dispatch ladder collapses into the caller, and native builds can inline the
  whole path — reducing per-call overhead, most noticeably for small arrays.
- Refreshed the crate-level (`lib.rs`) and README documentation: the portable
  `u64::count_ones()` fallback, thread-safety, dependency / `no_std` status, a
  `PopcntExt` usage example, and a note inviting a bug report if another crate
  is faster for your use case.

## [0.1.0] - 2026-07-01

Initial release.

- `popcnt(&[u8]) -> u64`: counts the 1 bits in a byte slice, dispatching at
  runtime to the fastest instruction set the running CPU supports.
- `PopcntExt` trait: adds a `.popcnt()` method to slices, arrays and `Vec`s of
  every built-in integer type (`u8`–`u128`, `i8`–`i128`, `usize`, `isize`).
- SIMD acceleration: POPCNT / AVX2 / AVX512 on x86 and x86-64, NEON / SVE on
  AArch64, and a portable `u64::count_ones()` fallback on all other targets.

[0.2.0]: https://github.com/kimwalisch/simd-popcnt/releases/tag/v0.2.0
[0.1.0]: https://github.com/kimwalisch/simd-popcnt/releases/tag/v0.1.0
