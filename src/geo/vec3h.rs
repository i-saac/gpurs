use std::ops;

#[derive(Debug, Clone, Copy)]
pub struct Vec3h {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32
}

impl Vec3h {
    pub fn i() -> Vec3h {
        Vec3h { x: 1.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    pub fn j() -> Vec3h {
        Vec3h { x: 0.0, y: 1.0, z: 0.0, w: 1.0 }
    }

    pub fn k() -> Vec3h {
        Vec3h { x: 0.0, y: 0.0, z: 1.0, w: 1.0 }
    }
}

impl ops::Add<Vec3h> for Vec3h {
    type Output = Vec3h;

    fn add(self, rhs: Vec3h) -> Vec3h {
        Vec3h { 
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: 1.0
        }
    }
}

impl ops::Neg for Vec3h {
    type Output = Vec3h;

    fn neg(self) -> Vec3h {
        Vec3h { x: -self.x, y: -self.y, z: -self.z, w: 1.0 }
    }
}

impl ops::Sub<Vec3h> for Vec3h {
    type Output = Vec3h;

    fn sub(self, rhs: Vec3h) -> Vec3h {
        Vec3h { 
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: 1.0
        }
    }
}