mod gpu_transformer;

pub use gpu_transformer::GPUTransformer;

/// Default amount of memory slots for matrices in MemoryHandler and Calculator
const INIT_MEMORY_CAPACITY: usize = 3;

/// List of default kernel names
const PROGRAM_LIST_FLOAT: [&str; 1] = [
    "bulk_transform"
];
const PROGRAM_LIST_DOUBLE: [&str; 1] = [
    "bulk_transform"
];

/// Source code for default kernels
const PROGRAM_SOURCE_FLOAT: &str = r#"
kernel void bulk_transform (
    global float* output,
    const int N,
    const global float* transform,
    const global float* vectors
) {
    const int globalCol = get_global_id(0);

    for (int k = 0; k < 4; k++) {
        float interm = transform[k * 4 + 0] * vectors[globalCol * 4 + 0];
        interm += transform[k * 4 + 1] * vectors[globalCol * 4 + 1];
        interm += transform[k * 4 + 2] * vectors[globalCol * 4 + 2];
        interm += transform[k * 4 + 3] * vectors[globalCol * 4 + 3];
        output[globalCol * 4 + k] = interm;
    }
}
"#;
const PROGRAM_SOURCE_DOUBLE: &str = r#"
kernel void bulk_transform (
    global double* output,
    const int N,
    const global double* transform,
    const global double* vectors
) {
    const int globalCol = get_global_id(0);

    for (int k = 0; k < 4; k++) {
        double interm = transform[k * 4 + 0] * vectors[globalCol * 4 + 0];
        interm += transform[k * 4 + 1] * vectors[globalCol * 4 + 1];
        interm += transform[k * 4 + 2] * vectors[globalCol * 4 + 2];
        interm += transform[k * 4 + 3] * vectors[globalCol * 4 + 3];
        output[globalCol * 4 + k] = interm;
    }
}
"#;