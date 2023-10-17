//! Module for GPU-based mass vector transformation.

mod gpu_transformer;

pub use gpu_transformer::GPUTransformer;

// Default amount of memory slots for transformations in GPUTransformer.
const INIT_MEMORY_CAPACITY: usize = 3;

// List of default kernel names.
const PROGRAM_LIST_FLOAT: [&str; 1] = [
    "bulk_transform",
];
const PROGRAM_LIST_DOUBLE: [&str; 1] = [
    "bulk_transform",
];

// Source code for default kernels.
const PROGRAM_SOURCE_FLOAT: &str = r#"
typedef struct _vec3h {
    float x;
    float y;
    float z;
    float w;
}vec3h;

kernel void bulk_transform (
    global vec3h* output,
    const int N,
    const global float* transform,
    const global vec3h* vectors
) {
    const int globalCol = get_global_id(0);

    float interm_x = transform[0] * vectors[globalCol].x;
    interm_x += transform[1] * vectors[globalCol].y;
    interm_x += transform[2] * vectors[globalCol].z;
    interm_x += transform[3] * vectors[globalCol].w;
    output[globalCol].x = interm_x;

    float interm_y = transform[4] * vectors[globalCol].x;
    interm_y += transform[5] * vectors[globalCol].y;
    interm_y += transform[6] * vectors[globalCol].z;
    interm_y += transform[7] * vectors[globalCol].w;
    output[globalCol].y = interm_y;

    float interm_z = transform[8] * vectors[globalCol].x;
    interm_z += transform[9] * vectors[globalCol].y;
    interm_z += transform[10] * vectors[globalCol].z;
    interm_z += transform[11] * vectors[globalCol].w;
    output[globalCol].z = interm_z;

    float interm_w = transform[12] * vectors[globalCol].x;
    interm_w += transform[13] * vectors[globalCol].y;
    interm_w += transform[14] * vectors[globalCol].z;
    interm_w += transform[15] * vectors[globalCol].w;
    output[globalCol].w = interm_w;
}
"#;
const PROGRAM_SOURCE_DOUBLE: &str = r#"
typedef struct _vec3h {
    double x;
    double y;
    double z;
    double w;
}vec3h;

kernel void bulk_transform (
    global vec3h* output,
    const int N,
    const global double* transform,
    const global vec3h* vectors
) {
    const int globalCol = get_global_id(0);

    double interm_x = transform[0] * vectors[globalCol].x;
    interm_x += transform[1] * vectors[globalCol].y;
    interm_x += transform[2] * vectors[globalCol].z;
    interm_x += transform[3] * vectors[globalCol].w;
    output[globalCol].x = interm_x;

    double interm_y = transform[4] * vectors[globalCol].x;
    interm_y += transform[5] * vectors[globalCol].y;
    interm_y += transform[6] * vectors[globalCol].z;
    interm_y += transform[7] * vectors[globalCol].w;
    output[globalCol].y = interm_y;

    double interm_z = transform[8] * vectors[globalCol].x;
    interm_z += transform[9] * vectors[globalCol].y;
    interm_z += transform[10] * vectors[globalCol].z;
    interm_z += transform[11] * vectors[globalCol].w;
    output[globalCol].z = interm_z;

    double interm_w = transform[12] * vectors[globalCol].x;
    interm_w += transform[13] * vectors[globalCol].y;
    interm_w += transform[14] * vectors[globalCol].z;
    interm_w += transform[15] * vectors[globalCol].w;
    output[globalCol].w = interm_w;
}
"#;