use gpurs::linalg::Matrix;

#[test]
fn indexing_tests() {
    let mat_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut mat: Matrix<f32> = Matrix::new(mat_data, 3, 4).expect("Failed to create Matrix");

    assert_eq!(mat[[1, 2]], 7.0, "Index value not as expected");
    assert_eq!(mat.col_vec(2).expect("Failed to index col"), vec![3.0, 7.0, 11.0], "Column vector not as expected");
    assert_eq!(mat.row_vec(0).expect("Failed to index row"), vec![1.0, 2.0, 3.0, 4.0], "Row vector not as expected");
    assert_eq!(mat.slice_index(&[0, 2], &[2, 3]).get_data(), vec![3.0, 4.0, 11.0, 12.0], "Slice index not as expected");

    mat[[0, 3]] = 5.0;
    assert_eq!(mat.col_vec(3).expect("Failed to index col"), vec![5.0, 8.0, 12.0], "Mutable index not working");
}

#[test]
fn arithmetic_test() {
    let a_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<f32> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];

    let a_mat: Matrix<f32> = Matrix::new(a_vec, 2, 3)
        .expect("Failed to create Matrix A");
    let b_mat: Matrix<f32> = Matrix::new(b_vec, 2, 3)
        .expect("Failed to create Matrix B");

    let b_transpose_vec: Vec<f32> = vec![2.0, 3.0, 1.0, 2.0, 2.0, 1.0];
    let b_transpose: Matrix<f32> = b_mat.clone().transpose();
    assert_eq!(b_transpose.get_data(), b_transpose_vec, "Transpose data not as expected");
    assert_eq!(b_transpose.get_rows(), 3, "Transpose row dimension not as expected");
    assert_eq!(b_transpose.get_cols(), 2, "Transpose col dimension not as expected");

    let value: f32 = 2.0;

    let fladd_vec: Vec<f32> = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let fladd_mat: Matrix<f32> = a_mat.clone() + value;
    assert_eq!(fladd_mat.get_data(), fladd_vec, "Matrix-Float addition not as expected");

    let add_vec: Vec<f32> = vec![3.0, 3.0, 5.0, 7.0, 7.0, 7.0];
    let add_mat: Matrix<f32> = (a_mat.clone() + b_mat.clone()).expect("Failed to add Matrix A and Matrix B");
    assert_eq!(add_mat.get_data(), add_vec, "Matrix-Matrix addition not as expected");

    let neg_vec: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    let neg_mat: Matrix<f32> = -a_mat.clone();
    assert_eq!(neg_mat.get_data(), neg_vec, "Matrix negation not as expected");

    let flsub_vec: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let flsub_mat: Matrix<f32> = a_mat.clone() - value;
    assert_eq!(flsub_mat.get_data(), flsub_vec, "Matrix-Float subtraction not as expected");

    let sub_vec: Vec<f32> = vec![-1.0, 1.0, 1.0, 1.0, 3.0, 5.0];
    let sub_mat: Matrix<f32> = (a_mat.clone() - b_mat.clone()).expect("Failed to subtract Matrix B from Matrix A");
    assert_eq!(sub_mat.get_data(), sub_vec, "Matrix-Matrix subtraction not as expected");

    let flmul_vec: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let flmul_mat: Matrix<f32> = value * a_mat.clone();
    assert_eq!(flmul_mat.get_data(), flmul_vec, "Matrix-Float multiplication not as expected");

    let mul_vec: Vec<f32> = vec![10.0, 10.0, 25.0, 28.0];
    let mul_mat: Matrix<f32> = (a_mat.clone() * b_transpose).expect("Failed to multiply Matrix A by Matrix B");
    assert_eq!(mul_mat.get_data(), mul_vec, "Matrix-Matrix multiplication data not as expected");
    assert_eq!(mul_mat.get_rows(), 2, "Matrix-Matrix multiplication row dimension not as expected");
    assert_eq!(mul_mat.get_cols(), 2, "Matrix-Matrix multiplication col dimension not as expected");

    let elementwise_vec: Vec<f32> = vec![2.0, 2.0, 6.0, 12.0, 10.0, 6.0];
    let elementwise_mat: Matrix<f32> = a_mat.elementwise(b_mat).expect("Failed to elementwise multiply Matrix A by Matrix B");
    assert_eq!(elementwise_mat.get_data(), elementwise_vec, "Elementwise multiplication data not as expected");
    assert_eq!(elementwise_mat.get_rows(), 2, "Elementwise multiplication row dimension not as expected");
    assert_eq!(elementwise_mat.get_cols(), 3, "Elementwise multiplication col dimension not as expected");
}