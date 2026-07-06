# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-07-06

### Added
- Automated benchmark CI (Linux, macOS and Windows) that compares the current
  code against the previous release at several array sizes and fails on a more
  than 2% throughput regression, plus a benchmark status badge in the README.

### Changed
- Much faster small-array counting on the x86-64 runtime-dispatch path (up to
  ~3.5x for tiny arrays, ~2x around 100 bytes): the scalar POPCNT loop now emits
  the `popcnt` instruction through inline assembly instead of a
  `#[target_feature(enable = "popcnt")]` helper. A target-feature function
  cannot be inlined into the feature-less dispatcher, so it forced a real call
  on every count; the inline-asm helper carries no such barrier and folds
  directly into the dispatcher (as libpopcnt's `__asm__("popcnt …")` and MSVC's
  `__popcnt64` do). The runtime POPCNT check is unchanged.
- The 1..=7 byte scalar tail no longer copies through a `memcpy` libcall; it is
  packed into a `u64` with an inlinable shift-or loop.

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
- Improved ARM NEON performance by up to 2x.
- Improved ARM SVE performance by up to 2x (four independent accumulators for
  higher instruction-level parallelism).
- Improved AVX512 performance by ~10% (four independent accumulators for higher
  instruction-level parallelism).
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

[Unreleased]: https://github.com/kimwalisch/simd-popcnt/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/kimwalisch/simd-popcnt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kimwalisch/simd-popcnt/releases/tag/v0.2.0
[0.1.0]: https://github.com/kimwalisch/simd-popcnt/releases/tag/v0.1.0
