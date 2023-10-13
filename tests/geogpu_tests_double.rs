#![cfg(feature = "gpu_accel")]

use std::time::Instant;

use gpurs::geo::{
    Vec3h,
    Transform3h
};
use gpurs::geogpu::GPUTransformer;

#[test]
fn stress_test() {
    let test_vectors: Vec<Vec3h<f64>> = vec![Vec3h::<f64>::i(); 1000000];

    let rotate: Transform3h<f64> = Transform3h::<f64>::rotation_z(std::f64::consts::PI / 2.0);
    let scale: Transform3h<f64> = Transform3h::<f64>::scale(2.0);
    let translate: Transform3h<f64> = Transform3h::<f64>::translate(-1.0, 0.0, 1.0);

    let transform: Transform3h<f64> = translate * (scale * rotate);

    let mut transformer: GPUTransformer<f64> = GPUTransformer::<f64>::init().expect("Failed to initialize GPUTransformer");

    let transform_idx: usize = transformer.store_transform3h(transform).expect("Failed to store transform");

    let transform_start = Instant::now();
    let _ = transformer.mass_transform_3h(transform_idx, &test_vectors).expect("Failed mass transformation");
    println!("Transformation Time: {:?}", transform_start.elapsed());
}