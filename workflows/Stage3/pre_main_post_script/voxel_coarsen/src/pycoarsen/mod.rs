use numpy::PyArray1;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use crate::coarsen::voxel_coarsen;

#[pymodule]
fn rust_voxel_coarsen(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "voxel_coarsen")]
    fn voxel_coarsen_py<'py>(
        py: Python<'py>,
        file: &str,
        coarsen_size: usize,
    ) -> anyhow::Result<((usize, usize, usize), &'py PyArray1<i32>)> {
        let result = voxel_coarsen(file, coarsen_size)?;
        Ok((result.0, PyArray1::from_vec(py, result.1)))
    }

    Ok(())
}
