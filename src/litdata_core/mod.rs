use pyo3::prelude::*;
pub mod downloader;

#[pyclass]
pub struct LitDataLoaderCore {
    index: usize,
    worker_chunks: Vec<u32>,
    worker_intervals: Vec<Vec<u32>>,
    batch_size: u32,       // number of chunks to be processed in a batch
    pre_download: u32,     // number of chunks to pre-download ahead of current chunk
    prefetch_workers: u32, // number of workers to be used for download & decompressing chunk files
    prefetch_factor: u32,  // number of batches to prefetch ahead of current batch
}

#[pymethods]
impl LitDataLoaderCore {
    #[new]
    fn new(
        worker_chunks: Vec<u32>,
        worker_intervals: Vec<Vec<u32>>,
        batch_size: u32,
        pre_download: u32,
        prefetch_workers: u32,
        prefetch_factor: u32,
    ) -> Self {
        LitDataLoaderCore {
            index: 0,
            worker_chunks,
            worker_intervals,
            batch_size,
            pre_download,
            prefetch_workers,
            prefetch_factor,
        }
    }

    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<u32> {
        if slf.index < slf.worker_chunks.len() {
            let item = slf.worker_chunks[slf.index];
            slf.index += 1;
            Some(item)
        } else {
            None // signals StopIteration in Python
        }
    }
}
