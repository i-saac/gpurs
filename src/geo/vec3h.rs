use std::ops;

use crate::IsFloat;

#[derive(Debug, Clone, Copy)]
pub struct Vec3h<T: IsFloat> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T
}

impl Vec3h<f32> {
    pub fn i() -> Vec3h<f32> {
        Vec3h { x: 1.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    pub fn j() -> Vec3h<f32> {
        Vec3h { x: 0.0, y: 1.0, z: 0.0, w: 1.0 }
    }

    pub fn k() -> Vec3h<f32> {
        Vec3h { x: 0.0, y: 0.0, z: 1.0, w: 1.0 }
    }
}

impl Vec3h<f64> {
    pub fn i() -> Vec3h<f64> {
        Vec3h { x: 1.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    pub fn j() -> Vec3h<f64> {
        Vec3h { x: 0.0, y: 1.0, z: 0.0, w: 1.0 }
    }

    pub fn k() -> Vec3h<f64> {
        Vec3h { x: 0.0, y: 0.0, z: 1.0, w: 1.0 }
    }
}

impl ops::Add<Vec3h<f32>> for Vec3h<f32> {
    type Output = Vec3h<f32>;

    fn add(self, rhs: Vec3h<f32>) -> Vec3h<f32> {
        Vec3h { 
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: 1.0
        }
    }
}

impl ops::Add<Vec3h<f64>> for Vec3h<f64> {
    type Output = Vec3h<f64>;

    fn add(self, rhs: Vec3h<f64>) -> Vec3h<f64> {
        Vec3h { 
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: 1.0
        }
    }
}

impl ops::Neg for Vec3h<f32> {
    type Output = Vec3h<f32>;

    fn neg(self) -> Vec3h<f32> {
        Vec3h { x: -self.x, y: -self.y, z: -self.z, w: 1.0 }
    }
}

impl ops::Neg for Vec3h<f64> {
    type Output = Vec3h<f64>;

    fn neg(self) -> Vec3h<f64> {
        Vec3h { x: -self.x, y: -self.y, z: -self.z, w: 1.0 }
    }
}

impl ops::Sub<Vec3h<f32>> for Vec3h<f32> {
    type Output = Vec3h<f32>;

    fn sub(self, rhs: Vec3h<f32>) -> Vec3h<f32> {
        Vec3h { 
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: 1.0
        }
    }
}

impl ops::Sub<Vec3h<f64>> for Vec3h<f64> {
    type Output = Vec3h<f64>;

    fn sub(self, rhs: Vec3h<f64>) -> Vec3h<f64> {
        Vec3h { 
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: 1.0
        }
    }
}