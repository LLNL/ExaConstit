extern crate anyhow;
extern crate rand;

#[cfg(not(feature = "polars"))]
extern crate data_reader;
#[cfg(feature = "polars")]
extern crate polars;

#[cfg(feature = "python")]
extern crate numpy;
#[cfg(feature = "python")]
extern crate pyo3;

pub mod coarsen;
#[cfg(feature = "python")]
pub mod pycoarsen;