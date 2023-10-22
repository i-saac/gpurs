use std::ops;

use crate::IsFloat;

/// Vector for 3D Homogeneous Coordinates.
/// 
/// Make sure to set your w coordinate to 1.0 to make translations work properly.
#[derive(Debug, Clone, Copy)]
pub struct Vec3h<T: IsFloat> {
    /// X component of vector
    pub x: T,
    /// Y component of vector
    pub y: T,
    /// Z component of vector
    pub z: T,
    /// W component (usually 1, primarily used for translation)
    pub w: T,
}

impl Vec3h<f32> {
    /// Unit vector along the x axis.
    pub fn i() -> Vec3h<f32> {
        Vec3h { x: 1.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    /// Unit vector along the y axis.
    pub fn j() -> Vec3h<f32> {
        Vec3h { x: 0.0, y: 1.0, z: 0.0, w: 1.0 }
    }

    /// Unit vector along the z axis.
    pub fn k() -> Vec3h<f32> {
        Vec3h { x: 0.0, y: 0.0, z: 1.0, w: 1.0 }
    }

    /// Return length (magnitude) of vector.
    /// 
    /// ```
    /// # use gpurs::geo::Vec3h;
    /// let vector: Vec3h<f32> = Vec3h { x: 1.0, y: 2.0, z: 2.0, w: 2.0 };
    /// let magnitude: f32 = vector.len();
    /// 
    /// assert!((magnitude - 3.0).abs() < 0.00001); // Check answer (ignoring floating point error)
    /// ```
    pub fn len(&self) -> f32 {
        let length: f32 = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();

        return length
    }

    /// Normalize vector.
    /// 
    /// ```
    /// # use gpurs::geo::Vec3h;
    /// let vector: Vec3h<f32> = Vec3h { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
    /// let unit_vector: Vec3h<f32> = vector.norm();
    /// 
    /// assert!((unit_vector.len() - 1.0).abs() < 0.00001); // Check answer (ignoring floating point error)
    /// ```
    pub fn norm(&self) -> Vec3h<f32> {
        let length: f32 = self.len();
        
        Vec3h { x: self.x / length, y: self.y / length, z: self.z / length, w: 1.0 }
    }
}

impl Vec3h<f64> {
    /// Unit vector along the x axis.
    pub fn i() -> Vec3h<f64> {
        Vec3h { x: 1.0, y: 0.0, z: 0.0, w: 1.0 }
    }

    /// Unit vector along the y axis.
    pub fn j() -> Vec3h<f64> {
        Vec3h { x: 0.0, y: 1.0, z: 0.0, w: 1.0 }
    }

    /// Unit vector along the z axis.
    pub fn k() -> Vec3h<f64> {
        Vec3h { x: 0.0, y: 0.0, z: 1.0, w: 1.0 }
    }

    /// Return length (magnitude) of vector.
    /// 
    /// ```
    /// # use gpurs::geo::Vec3h;
    /// let vector: Vec3h<f64> = Vec3h { x: 1.0, y: 2.0, z: 2.0, w: 2.0 };
    /// let magnitude: f64 = vector.len();
    /// 
    /// assert!((magnitude - 3.0).abs() < 0.00001); // Check answer (ignoring floating point error)
    /// ```
    pub fn len(&self) -> f64 {
        let length: f64 = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();

        return length
    }

    /// Normalize vector
    /// 
    /// ```
    /// # use gpurs::geo::Vec3h;
    /// let vector: Vec3h<f64> = Vec3h { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
    /// let unit_vector: Vec3h<f64> = vector.norm();
    /// 
    /// assert!((unit_vector.len() - 1.0).abs() < 0.00001); // Check answer (ignoring floating point error)
    /// ```
    pub fn norm(&self) -> Vec3h<f64> {
        let length: f64 = self.len();
        
        Vec3h { x: self.x / length, y: self.y / length, z: self.z / length, w: 1.0 }
    }
}

/// Add vector to vector.
/// Keeps w component as 1.0 to maintain ability to translate.
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

/// Add vector to vector.
/// Keeps w component as 1.0 to maintain ability to translate.
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

/// Negate vector.
/// Keeps w component as 1.0 to maintain ability to translate.
impl ops::Neg for Vec3h<f32> {
    type Output = Vec3h<f32>;

    fn neg(self) -> Vec3h<f32> {
        Vec3h { x: -self.x, y: -self.y, z: -self.z, w: 1.0 }
    }
}

/// Negate vector.
/// Keeps w component as 1.0 to maintain ability to translate.
impl ops::Neg for Vec3h<f64> {
    type Output = Vec3h<f64>;

    fn neg(self) -> Vec3h<f64> {
        Vec3h { x: -self.x, y: -self.y, z: -self.z, w: 1.0 }
    }
}

/// Subtract vector from vector.
/// Keeps w component as 1.0 to maintain ability to translate.
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

/// Subtract vector from vector.
/// Keeps w component as 1.0 to maintain ability to translate.
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