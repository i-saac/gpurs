#![cfg(feature = "gpu_accel")]

use std::time::Instant;

mod test_parameters;
use test_parameters::P;

use gpurs::Result;
use gpurs::geo::{
    Vec3h,
    Transform3h
};
use gpurs::geogpu::GPUTransformer;

#[test]
fn stress_test() -> Result<()> {
    let test_vectors: Vec<Vec3h<P>> = vec![Vec3h::<P>::i(); 1000000];

    let rotate: Transform3h<P> = Transform3h::<P>::rotation_z(std::f64::consts::PI as P / 2.0);
    let scale: Transform3h<P> = Transform3h::<P>::scale(2.0);
    let translate: Transform3h<P> = Transform3h::<P>::translate(-1.0, 0.0, 1.0);

    let transform: Transform3h<P> = translate * (scale * rotate);

    let mut transformer: GPUTransformer<P> = GPUTransformer::<P>::init()?;

    let transform_idx: usize = transformer.store_transform3h(transform)?;

    let transform_start = Instant::now();
    let _ = transformer.mass_transform_3h(transform_idx, &test_vectors)?;
    println!("Transformation Time: {:?}", transform_start.elapsed());

    Ok(())
}