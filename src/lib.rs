//! # gpurs
//! 
//! A general-purpose linear algebra toolkit for scientific computing
//! including GPU acceleration through OpenCL
//! 
//! ## Notices
//!
//! - **This crate is currently very much in development, and every feature and piece of syntax is subject to change.
//! I'm trying to slow down with the major syntactic overhauls, but no promises.**
//!
//! - **This crate loads GPU acelerator modules and thus requires the OpenCL SDK by default.
//! To opt out of this (for instance if you have no GPU) set the flag `default-features = false` when adding gpurs to your Cargo.toml file.**
//!
//!     - For NVIDIA GPUs, the OpenCL SDK is available through the CUDA installer.
//! 
//! ## Further Reading
//! 
//! - [GitHub Wiki](https://github.com/i-saac/gpurs/wiki)
//! 
//! ## Tips and Tricks
//!
//! - Declare a type at the top of your file that is equivalent to your desired floating point precision,
//! and use that as the generic parameter for your initializations. 
//! This means you only have to change one value to change your entire file's precision instead of changing each of them individually!
//! If you really wanted you could make a global type and import that to change it in one place for your entire project!
//!
//! ```
//! # use gpurs::linalg::Matrix;
//! # let size: usize = 3;
//! type P = f64; // Replace f64 with f32 at any point to change the precision of the whole file!
//!
//! let identity: Matrix<P> = Matrix::<P>::identity(size);
//! ```

use std::result;

mod error;
pub mod geo;
pub mod linalg;

#[cfg(feature = "gpu_accel")]
pub mod gpu;

#[cfg(feature = "gpu_accel")]
pub mod geogpu;

pub use error::Jeeperr;

/// Shorthand result type for gpurs.
pub type Result<T> = result::Result<T, Jeeperr>;

/// Trait allowing objects to be implemented for single or double precision floats
pub trait IsFloat {}
impl IsFloat for f32 {}
impl IsFloat for f64 {}