// pip install --upgrade pyOpenSSL cryptography

use pyo3::prelude::*;
pub mod litdata_core;

#[pyfunction]
fn hello_from_bin() -> String {
    "RUST: Hello from LitData!".to_string()
}

/// A Python module implemented in Rust. The name of this function (`_core`) must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
    m.add_class::<litdata_core::LitDataLoaderCore>()?;
    // m.add_class::<rust_impl::fs::s3::S3Storage>()?;
    // m.add_class::<rust_impl::streaming_data_provider::StreamingDataProvider>()?;
    Ok(())
}
