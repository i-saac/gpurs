use gpurs::linalg::{
    Matrix,
    MatrixUtilities,
    Axis
};

#[test]
fn linear_solve_test() {
    let a_mat: Matrix<f64> = Matrix::new(vec![1.0, 1.0, 1.0, 3.0, 1.0, -3.0, 1.0, -2.0, -5.0], 3, 3)
        .expect("Failed to create a square matrix");
    let b_vec: Matrix<f64> = Matrix::new(vec![1.0, 5.0, 10.0], 3, 1)
        .expect("Failed to create b column matrix");

    let x_vec: Matrix<f64> = Matrix::linear_solve_matrix(&a_mat, &b_vec)
        .expect("Failed to linearly solve a\"b");

    println!("{}", x_vec);
}

#[test]
fn max_min_test() {
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a_mat: Matrix<f32> = Matrix::new(a_data, 2, 2)
        .expect("Failed to create Matrix A");

    let max_val: f32 = Matrix::max(&a_mat);
    assert_eq!(max_val, 4.0, "Max value not as expected");

    let row_max: Matrix<f32> = Matrix::axis_max(&a_mat, Axis::Row);
    assert_eq!(row_max.get_data(), &[2.0, 4.0], "Max values along row axis not as expected");

    let col_max: Matrix<f32> = Matrix::axis_max(&a_mat, Axis::Col);
    assert_eq!(col_max.get_data(), &[3.0, 4.0], "Max values along col axis not as expected");

    let min_val: f32 = Matrix::min(&a_mat);
    assert_eq!(min_val, 1.0, "Min value not as expected");

    let row_min: Matrix<f32> = Matrix::axis_min(&a_mat, Axis::Row);
    assert_eq!(row_min.get_data(), &[1.0, 3.0], "Min values along row axis not as expected");

    let col_min: Matrix<f32> = Matrix::axis_min(&a_mat, Axis::Col);
    assert_eq!(col_min.get_data(), &[1.0, 2.0], "Min values along col axis not as expected");
}

#[test]
fn concatenation_test() {
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let a_mat: Matrix<f32> = Matrix::new(a_data, 2, 2)
        .expect("Failed to create Matrix A");

    println!("{}", a_mat);

    let row_concat: Matrix<f32> = Matrix::concatenate(&[a_mat.clone(), a_mat.clone(), a_mat.clone()], Axis::Row)
        .expect("Failed to concatenate along rows");
    assert_eq!(row_concat.get_data(), &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], "Row concatenation data not as expected");
    assert_eq!(row_concat.get_rows(), 6, "Row concatenation row dimension not as expected");
    assert_eq!(row_concat.get_cols(), 2, "Row concatenation col dimension not as expected");

    println!("{}", row_concat);

    let col_concat: Matrix<f32> = Matrix::concatenate(&[a_mat.clone(), a_mat.clone(), a_mat.clone()], Axis::Col)
        .expect("Failed to concatenate along cols");
    assert_eq!(col_concat.get_data(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0], "Col concatenation data not as expected");
    assert_eq!(col_concat.get_rows(), 2, "Col concatenation row dimension not as expected");
    assert_eq!(col_concat.get_cols(), 6, "Col concatenation col dimension not as expected");

    println!("{}", col_concat);
}
