mod memory;
mod memory_calculator;
mod quick_calculator;

pub use memory_calculator::MemoryCalculator;
pub use memory_calculator::MemoryParameterFunction;
pub use quick_calculator::QuickCalculator;
pub use quick_calculator::QuickParameterFunction;

/// Default amount of memory slots for matrices in MemoryHandler and Calculator
const INIT_MEMORY_CAPACITY: usize = 3;

/// List of default kernel names
const PROGRAM_LIST: [&str; 1] = [
    "mat_mul"
];

/// Source code for default kernels
const PROGRAM_SOURCE: &str = r#"
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