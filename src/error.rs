use std::fmt;
use std::error::Error;

#[cfg(feature = "gpu_accel")]
use opencl3::error_codes::ClError;

/// Error enum for gpurs crate
#[derive(Debug)]
pub enum Jeeperr {
    ArgumentError,
    DimensionError,
    IndexError,
    MemoryError,
    OutputError,
    #[cfg(feature = "gpu_accel")] ClError(ClError),
    Error(String)
}

impl fmt::Display for Jeeperr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Jeeperr::ArgumentError =>
                write!(f, "Too many or too few arguments provided"),
            Jeeperr::DimensionError =>
                write!(f, "Invalid matrix dimensions for requested operation"),
            Jeeperr::IndexError =>
                write!(f, "Invalid index for requested operation"),
            Jeeperr::MemoryError =>
                write!(f, "Memory Calculator and Handler have inconsistent memory"),
            Jeeperr::OutputError =>
                write!(f, "Invalid output type"),
            #[cfg(feature = "gpu_accel")] Jeeperr::ClError(error) =>
                write!(f, "{}", error),
            Jeeperr::Error(error) =>
                write!(f, "{}", error)
        }
    }
}

impl Error for Jeeperr {}

#[cfg(feature = "gpu_accel")]
impl From<ClError> for Jeeperr {
    fn from(err: ClError) -> Self {
        Jeeperr::ClError(err)
    }
}