use std::ops;

use crate::IsFloat;
use crate::geo::Vec3h;

#[derive(Debug, Clone)]
pub struct Transform3h<T: IsFloat> {
    data: [T; 16]
}

impl<T: IsFloat> Transform3h<T> {
    pub fn get_data(&self) -> &[T] {
        return &self.data
    }
}

impl Transform3h<f32> {
    pub fn identity() -> Transform3h<f32> {
        Transform3h { data: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn scale(factor: f32) -> Transform3h<f32> {
        Transform3h { data: [
            factor, 0.0, 0.0, 0.0,
            0.0, factor, 0.0, 0.0,
            0.0, 0.0, factor, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn translate(delta_x: f32, delta_y: f32, delta_z: f32) -> Transform3h<f32> {
        Transform3h { data: [
            1.0, 0.0, 0.0, delta_x,
            0.0, 1.0, 0.0, delta_y,
            0.0, 0.0, 1.0, delta_z,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn rotation_x(theta_x: f32) -> Transform3h<f32> {
        Transform3h { data: [
            1.0, 0.0, 0.0, 0.0,
            0.0, theta_x.cos(), -theta_x.sin(), 0.0,
            0.0, theta_x.sin(), theta_x.cos(), 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn rotation_y(theta_y: f32) -> Transform3h<f32> {
        Transform3h { data: [
            theta_y.cos(), 0.0, theta_y.sin(), 0.0,
            0.0, 1.0, 0.0, 0.0,
            -theta_y.sin(), 0.0, theta_y.cos(), 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn rotation_z(theta_z: f32) -> Transform3h<f32> {
        Transform3h { data: [
            theta_z.cos(), -theta_z.sin(), 0.0, 0.0,
            theta_z.sin(), theta_z.cos(), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn rotation_arbitrary(axis_x: f32, axis_y: f32, axis_z: f32, theta: f32) -> Transform3h<f32> {
        let ct: f32 = theta.cos();
        let st: f32 = theta.sin();

        Transform3h { data: [
            ct + (1.0 - ct) * axis_x * axis_x,
            (1.0 - ct) * axis_x * axis_y - st * axis_z,
            (1.0 - ct) * axis_x * axis_z + st * axis_y,
            0.0,
            (1.0 - ct) * axis_y * axis_x + st * axis_z,
            ct + (1.0 - ct) * axis_y * axis_y,
            (1.0 - ct) * axis_y * axis_z - st * axis_x,
            0.0,
            (1.0 - ct) * axis_z * axis_x - st * axis_y,
            (1.0 - ct) * axis_z * axis_y + st * axis_x,
            ct + (1.0 - ct) * axis_z * axis_z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0
        ] }
    }
}

impl Transform3h<f64> {
    pub fn identity() -> Transform3h<f64> {
        Transform3h { data: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn scale(factor: f64) -> Transform3h<f64> {
        Transform3h { data: [
            factor, 0.0, 0.0, 0.0,
            0.0, factor, 0.0, 0.0,
            0.0, 0.0, factor, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn translate(delta_x: f64, delta_y: f64, delta_z: f64) -> Transform3h<f64> {
        Transform3h { data: [
            1.0, 0.0, 0.0, delta_x,
            0.0, 1.0, 0.0, delta_y,
            0.0, 0.0, 1.0, delta_z,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn rotation_x(theta_x: f64) -> Transform3h<f64> {
        Transform3h { data: [
            1.0, 0.0, 0.0, 0.0,
            0.0, theta_x.cos(), -theta_x.sin(), 0.0,
            0.0, theta_x.sin(), theta_x.cos(), 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn rotation_y(theta_y: f64) -> Transform3h<f64> {
        Transform3h { data: [
            theta_y.cos(), 0.0, theta_y.sin(), 0.0,
            0.0, 1.0, 0.0, 0.0,
            -theta_y.sin(), 0.0, theta_y.cos(), 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn rotation_z(theta_z: f64) -> Transform3h<f64> {
        Transform3h { data: [
            theta_z.cos(), -theta_z.sin(), 0.0, 0.0,
            theta_z.sin(), theta_z.cos(), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ] }
    }

    pub fn rotation_arbitrary(axis_x: f64, axis_y: f64, axis_z: f64, theta: f64) -> Transform3h<f64> {
        let ct: f64 = theta.cos();
        let st: f64 = theta.sin();

        Transform3h { data: [
            ct + (1.0 - ct) * axis_x * axis_x,
            (1.0 - ct) * axis_x * axis_y - st * axis_z,
            (1.0 - ct) * axis_x * axis_z + st * axis_y,
            0.0,
            (1.0 - ct) * axis_y * axis_x + st * axis_z,
            ct + (1.0 - ct) * axis_y * axis_y,
            (1.0 - ct) * axis_y * axis_z - st * axis_x,
            0.0,
            (1.0 - ct) * axis_z * axis_x - st * axis_y,
            (1.0 - ct) * axis_z * axis_y + st * axis_x,
            ct + (1.0 - ct) * axis_z * axis_z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0
        ] }
    }
}

impl ops::Mul<Transform3h<f32>> for Transform3h<f32> {
    type Output = Transform3h<f32>;

    fn mul(self, rhs: Transform3h<f32>) -> Transform3h<f32> {
        let mut data: [f32; 16] = [0.0; 16];

        for lhs_row in 0..4 {
            for rhs_col in 0..4 {
                data[lhs_row * 4 + rhs_col] = (0..4).into_iter()
                    .map(|idx| self.data[lhs_row * 4 + idx] * rhs.data[idx * 4 + rhs_col])
                    .sum();
            }
        }

        Transform3h { data }
    }
}

impl ops::Mul<Transform3h<f64>> for Transform3h<f64> {
    type Output = Transform3h<f64>;

    fn mul(self, rhs: Transform3h<f64>) -> Transform3h<f64> {
        let mut data: [f64; 16] = [0.0; 16];

        for lhs_row in 0..4 {
            for rhs_col in 0..4 {
                data[lhs_row * 4 + rhs_col] = (0..4).into_iter()
                    .map(|idx| self.data[lhs_row * 4 + idx] * rhs.data[idx * 4 + rhs_col])
                    .sum();
            }
        }

        Transform3h { data }
    }
}

impl ops::Mul<&Vec3h<f32>> for &Transform3h<f32> {
    type Output = Vec3h<f32>;

    fn mul(self, rhs: &Vec3h<f32>) -> Vec3h<f32> {
        Vec3h {
            x: rhs.x * self.data[0] + rhs.y * self.data[1] + rhs.z * self.data[2] + rhs.w * self.data[3],
            y: rhs.x * self.data[4] + rhs.y * self.data[5] + rhs.z * self.data[6] + rhs.w * self.data[7],
            z: rhs.x * self.data[8] + rhs.y * self.data[9] + rhs.z * self.data[10] + rhs.w * self.data[11],
            w: rhs.x * self.data[12] + rhs.y * self.data[13] + rhs.z * self.data[14] + rhs.w * self.data[15],
        }
    }
}

impl ops::Mul<&Vec3h<f64>> for &Transform3h<f64> {
    type Output = Vec3h<f64>;

    fn mul(self, rhs: &Vec3h<f64>) -> Vec3h<f64> {
        Vec3h {
            x: rhs.x * self.data[0] + rhs.y * self.data[1] + rhs.z * self.data[2] + rhs.w * self.data[3],
            y: rhs.x * self.data[4] + rhs.y * self.data[5] + rhs.z * self.data[6] + rhs.w * self.data[7],
            z: rhs.x * self.data[8] + rhs.y * self.data[9] + rhs.z * self.data[10] + rhs.w * self.data[11],
            w: rhs.x * self.data[12] + rhs.y * self.data[13] + rhs.z * self.data[14] + rhs.w * self.data[15],
        }
    }
}