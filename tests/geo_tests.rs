use gpurs::geo::{
    Vec3h,
    Transform3h
};

#[test]
fn transform_test() {
    let i_vec: Vec3h<f32> = Vec3h::<f32>::i();

    let rotate: Transform3h<f32> = Transform3h::<f32>::rotation_z(std::f32::consts::PI / 2.0);
    let scale: Transform3h<f32> = Transform3h::<f32>::scale(2.0);
    let translate: Transform3h<f32> = Transform3h::<f32>::translate(-1.0, 0.0, 1.0);

    let transform: Transform3h<f32> = translate * (scale * rotate);

    println!("{:?}", &transform * &i_vec);
}