# gpurs

[![crates.io](https://shields.io/crates/v/gpurs)](https://crates.io/crates/gpurs)
[![docs.io](https://docs.rs/gpurs/badge.svg)](https://docs.rs/gpurs/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Welcome to gpurs (pronounced "Jeepers!"), a general-purpose linear algebra toolkit for scientific computing including GPU acceleration through OpenCL.

## Notices

- **This crate is currently very much in development, and every feature and piece of syntax is subject to change. I'm trying to slow down with the major syntactic overhauls, but no promises.**

- **This crate loads GPU acelerator modules and thus requires the OpenCL SDK by default. To opt out of this (for instance if you have no GPU) set the flag `default-features = false` when adding gpurs to your Cargo.toml file.**

    - For NVIDIA GPUs, the OpenCL SDK is available through the CUDA installer.

## Documentation

- [GitHub Wiki](https://github.com/i-saac/gpurs/wiki) (Deep dives into specific functionality with examples)
- [RustDocs](https://docs.rs/gpurs/) (Has descriptions of most functions with example code)