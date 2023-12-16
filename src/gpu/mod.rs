//! Module for GPU Acceleration of Linear Algebra operations.
//! 
//! gpurs provides two main objects for GPU acceleration: MemoryCalculator and QuickCalculator.
//! 
//! - MemoryCalculator requires all inputs to be stored in memory ahead of time, and keeps all outputs stored for future use.
//! This approach is less memory efficient, but can be faster if you plan to chain together a lot of operations.
//! 
//! - QuickCalculator allows inputs to be pre-stored in memory, but also accepts matrices directly for certain function calls.
//! QuickCalculator does not keep outputs stored for future use. This approach is more memory efficient and is much better if you
//! don't necessarily want to keep all of your inputs/outputs stored in GPU memory for the entire lifetime of the calculator.
//! 
//! These objects also allow custom kernels to be added to their own internal memory, so you can take advantage of gpurs' built-in
//! kernel compilation and memory management for your own kernels.

mod memory;
mod memory_calculator;
mod quick_calculator;

mod sparse_memory;
mod sparse_calculator;

pub use memory_calculator::MemoryCalculator;
pub use memory_calculator::MemoryParameterFunction;
pub use quick_calculator::QuickCalculator;
pub use quick_calculator::QuickParameterFunction;

pub use sparse_calculator::SparseCalculator;

// Default amount of memory slots for matrices in MemoryCalculator and QuickCalculator
const INIT_MEMORY_CAPACITY: usize = 3;

// List of default kernel names
const PROGRAM_LIST_FLOAT: [&str; 1] = [
    "mat_mul"
];
const PROGRAM_LIST_DOUBLE: [&str; 1] = [
    "mat_mul"
];
const SPARSE_LIST_DOUBLE: [&str; 1] = [
    "mat_mul"
];

// Source code for default kernels
const PROGRAM_SOURCE_FLOAT: &str = r#"
kernel void mat_mul (
    global float* c,
    const int N,
    const int K,
    const global float* a,
    const global float* b
) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float interm = 0.0f;
    for (int k = 0; k < K; k++) {
        interm += a[globalRow * K + k] * b[k * N + globalCol];
    }

    c[globalRow * N + globalCol] = interm;
}
"#;
const PROGRAM_SOURCE_DOUBLE: &str = r#"
kernel void mat_mul (
    global double* c,
    const int N,
    const int K,
    const global double* a,
    const global double* b
) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    double interm = 0.0;
    for (int k = 0; k < K; k++) {
        interm += a[globalRow * K + k] * b[k * N + globalCol];
    }

    c[globalRow * N + globalCol] = interm;
}
"#;

const SPARSE_SOURCE_DOUBLE: &str = r#"
kernel void mat_mul (
    global double* c,
    const int N,
    const global int* a_row,
    const global int* a_col,
    const global double* a_val,
    const global double* b
) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    double interm = 0.0;
    for (int k = a_row[globalRow]; k < a_row[globalRow + 1]; k++) {
        interm += a_val[k] * b[a_col[k] * N + globalCol];
    }

    c[globalRow * N + globalCol] = interm;
}
"#;