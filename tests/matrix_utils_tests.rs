use gpurs::Matrix;
use gpurs::utils;

#[test]
fn dot_test() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![3.0, 4.0, 5.0];

    assert_eq!(utils::dot(a, b).unwrap(), 26.0, "Dot product not as expected");
}

#[test]
fn max_min_test() {
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a_mat: Matrix = Matrix::new(a_data, 2, 2)
        .expect("Failed to create Matrix A");

    let max_val: f32 = utils::max(&a_mat);
    assert_eq!(max_val, 4.0, "Max value not as expected");

    let row_max: Matrix = utils::axis_max(&a_mat, utils::Axis::Row);
    assert_eq!(row_max.get_data(), &[2.0, 4.0], "Max values along row axis not as expected");

    let col_max: Matrix = utils::axis_max(&a_mat, utils::Axis::Col);
    assert_eq!(col_max.get_data(), &[3.0, 4.0], "Max values along col axis not as expected");

    let min_val: f32 = utils::min(&a_mat);
    assert_eq!(min_val, 1.0, "Min value not as expected");

    let row_min: Matrix = utils::axis_min(&a_mat, utils::Axis::Row);
    assert_eq!(row_min.get_data(), &[1.0, 3.0], "Min values along row axis not as expected");

    let col_min: Matrix = utils::axis_min(&a_mat, utils::Axis::Col);
    assert_eq!(col_min.get_data(), &[1.0, 2.0], "Min values along col axis not as expected");
}

#[test]
fn concatenation_test() {
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a_mat: Matrix = Matrix::new(a_data, 2, 2)
        .expect("Failed to create Matrix A");

    println!("{}", a_mat);

    let row_concat: Matrix = utils::concatenate(&[a_mat.clone(), a_mat.clone(), a_mat.clone()], utils::Axis::Row)
        .expect("Failed to concatenate along rows");
    assert_eq!(row_concat.get_data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], "Row concatenation data not as expected");
    assert_eq!(row_concat.get_rows(), 6, "Row concatenation row dimension not as expected");
    assert_eq!(row_concat.get_cols(), 2, "Row concatenation col dimension not as expected");

    println!("{}", row_concat);

    let col_concat: Matrix = utils::concatenate(&[a_mat.clone(), a_mat.clone(), a_mat.clone()], utils::Axis::Col)
        .expect("Failed to concatenate along cols");
    assert_eq!(col_concat.get_data(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0], "Col concatenation data not as expected");
    assert_eq!(col_concat.get_rows(), 2, "Col concatenation row dimension not as expected");
    assert_eq!(col_concat.get_cols(), 6, "Col concatenation col dimension not as expected");

    println!("{}", col_concat);
}
