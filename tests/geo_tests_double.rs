use gpurs::geo::{
    Vec3h,
    Transform3h
};

#[test]
fn transform_test() {
    let i_vec: Vec3h<f64> = Vec3h::<f64>::i();

    let rotate: Transform3h<f64> = Transform3h::<f64>::rotation_z(std::f64::consts::PI / 2.0);
    let scale: Transform3h<f64> = Transform3h::<f64>::scale(2.0);
    let translate: Transform3h<f64> = Transform3h::<f64>::translate(-1.0, 0.0, 1.0);

    let transform: Transform3h<f64> = translate * (scale * rotate);

    println!("{:?}", &transform * &i_vec);
}