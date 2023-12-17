use ndarray::prelude::*;
use numpy::{convert::IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::cmp;

fn process(results: &mut Vec<bool>, arr: ArrayView1<f32>, shift: i32, len: usize) {
    for i in 0..len {
        if !results[i] {
            continue;
        }
        let plus = arr[cmp::min(i as i32 + shift, (len - 1) as i32) as usize];
        let minus = arr[cmp::max(i as i32 - shift, 0) as usize];
        let data = arr[i];
        results[i] = (data > plus) && (data > minus);
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn argrelmax<'py>(py: Python<'py>, arr: PyReadonlyArray1<f32>, order: i32) -> &'py PyArray1<u32> {
    let arr = arr.as_array();
    let len = arr.len();

    let mut results = vec![true; len];
    for shift in 1..=order {
        process(&mut results, arr, shift, len);
    }

    let mut nonzero_array: Vec<u32> = Vec::with_capacity(10000);
    for i in 0..len {
        if results[i] {
            nonzero_array.push(i as u32);
        }
    }

    let output = nonzero_array.to_owned();
    output.into_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn python_binding_pyo3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(argrelmax, m)?)?;
    Ok(())
}
