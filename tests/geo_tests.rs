use gpurs::geo::{
    Vec3h,
    Transform3h
};

type P = f32;

#[test]
fn transform_test() {
    let i_vec: Vec3h<P> = Vec3h::<P>::i();

    let rotate: Transform3h<P> = Transform3h::<P>::rotation_z(std::f64::consts::PI as P / 2.0);
    let scale: Transform3h<P> = Transform3h::<P>::scale(2.0);
    let translate: Transform3h<P> = Transform3h::<P>::translate(-1.0, 0.0, 1.0);

    let transform: Transform3h<P> = translate * (scale * rotate);

    println!("{:?}", &transform * &i_vec);
}