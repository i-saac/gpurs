use std::result;

mod error;
pub mod linalg;
pub mod gpu;

pub use error::Jeeperr;

pub type Result<T> = result::Result<T, Jeeperr>;