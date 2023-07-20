extern crate anyhow;
extern crate rand;

#[cfg(not(feature = "polars"))]
extern crate data_reader;
#[cfg(feature = "polars")]
extern crate polars;

extern crate numpy;
extern crate pyo3;

pub mod coarsen;
pub mod pycoarsen;
