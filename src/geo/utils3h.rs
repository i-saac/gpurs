use crate::IsFloat;
use crate::geo::Vec3h;

/// Trait containing utility functions for Vec3h struct.
pub trait Vec3hUtilities<T: IsFloat> {
    /// Perform dot product between vectors a and b.
    /// Ignores w component.
    fn dot(a_vec: &Vec3h<T>, b_vec: &Vec3h<T>) -> T;
    /// Perform cross product between left and right vectors.
    /// Ignores w component.
    fn cross(left: &Vec3h<T>, right: &Vec3h<T>) -> Vec3h<T>;
}

impl Vec3hUtilities<f32> for Vec3h<f32> {
    /// Perform dot product between vectors a and b.
    /// Ignores w component.
    /// 
    /// ```
    /// # use gpurs::geo::{Vec3h, Vec3hUtilities};
    /// let vector_a: Vec3h<f32> = Vec3h { x: 2.0, y: 1.0, z: 3.0, w: 1.0 };
    /// let vector_b: Vec3h<f32> = Vec3h { x: 1.0, y: 2.0, z: 1.0, w: 1.0 };
    /// 
    /// assert_eq!(Vec3h::dot(&vector_a, &vector_b), 7.0);
    /// ```
    fn dot(a_vec: &Vec3h<f32>, b_vec: &Vec3h<f32>) -> f32 {
        let dot_prod: f32 = a_vec.x * b_vec.x + a_vec.y * b_vec.y + a_vec.z * b_vec.z;

        return dot_prod
    }

    /// Perform cross product between left and right vectors.
    /// Ignores w component.
    /// 
    /// ```
    /// # use gpurs::geo::{Vec3h, Vec3hUtilities};
    /// let vector_a: Vec3h<f32> = Vec3h::<f32>::i();
    /// let vector_b: Vec3h<f32> = Vec3h::<f32>::j();
    /// let cross_prod: Vec3h<f32> = Vec3h::cross(&vector_a, &vector_b);
    /// 
    /// assert_eq!(cross_prod.x, 0.0);
    /// assert_eq!(cross_prod.y, 0.0);
    /// assert_eq!(cross_prod.z, 1.0);
    /// ```
    fn cross(left: &Vec3h<f32>, right: &Vec3h<f32>) -> Vec3h<f32> {
        let cross_x: f32 = left.y * right.z - left.z * right.y;
        let cross_y: f32 = left.z * right.x - left.x * right.z;
        let cross_z: f32 = left.x * right.y - left.y * right.x;

        return Vec3h { x: cross_x, y: cross_y, z: cross_z, w: 1.0 }
    }
}

impl Vec3hUtilities<f64> for Vec3h<f64> {
    /// Perform dot product between vectors a and b.
    /// Ignores w component.
    /// 
    /// ```
    /// # use gpurs::geo::{Vec3h, Vec3hUtilities};
    /// let vector_a: Vec3h<f64> = Vec3h { x: 2.0, y: 1.0, z: 3.0, w: 1.0 };
    /// let vector_b: Vec3h<f64> = Vec3h { x: 1.0, y: 2.0, z: 1.0, w: 1.0 };
    /// 
    /// assert_eq!(Vec3h::dot(&vector_a, &vector_b), 7.0);
    /// ```
    fn dot(a_vec: &Vec3h<f64>, b_vec: &Vec3h<f64>) -> f64 {
        let dot_prod: f64 = a_vec.x * b_vec.x + a_vec.y * b_vec.y + a_vec.z * b_vec.z;

        return dot_prod
    }

    /// Perform cross product between left and right vectors.
    /// Ignores w component.
    /// 
    /// ```
    /// # use gpurs::geo::{Vec3h, Vec3hUtilities};
    /// let vector_a: Vec3h<f64> = Vec3h::<f64>::i();
    /// let vector_b: Vec3h<f64> = Vec3h::<f64>::j();
    /// let cross_prod: Vec3h<f64> = Vec3h::cross(&vector_a, &vector_b);
    /// 
    /// assert_eq!(cross_prod.x, 0.0);
    /// assert_eq!(cross_prod.y, 0.0);
    /// assert_eq!(cross_prod.z, 1.0);
    /// ```
    fn cross(left: &Vec3h<f64>, right: &Vec3h<f64>) -> Vec3h<f64> {
        let cross_x: f64 = left.y * right.z - left.z * right.y;
        let cross_y: f64 = left.z * right.x - left.x * right.z;
        let cross_z: f64 = left.x * right.y - left.y * right.x;

        return Vec3h { x: cross_x, y: cross_y, z: cross_z, w: 1.0 }
    }
}