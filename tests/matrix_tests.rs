use gpurs::Matrix;
use gpurs::Calculator;

#[test]
fn indexing_tests() {
    let mat_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mat: Matrix = Matrix::new(mat_data, 3, 4).expect("Failed to create Matrix");

    assert_eq!(mat[[1, 2]], 7.0, "Index value not as expected");
    assert_eq!(mat.col_vec(2).expect("Failed to index row"), vec![3.0, 7.0, 11.0], "Column vector not as expected");
    assert_eq!(mat.row_vec(0).expect("Failed to index col"), vec![1.0, 2.0, 3.0, 4.0], "Row vector not as expected");
}

#[test]
fn arithmetic_test() {
    let a_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<f32> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];

    let a_mat: Matrix = Matrix::new(a_vec, 2, 3)
        .expect("Failed to create Matrix A");
    let b_mat: Matrix = Matrix::new(b_vec, 2, 3)
        .expect("Failed to create Matrix B");

    let b_transpose_vec: Vec<f32> = vec![2.0, 3.0, 1.0, 2.0, 2.0, 1.0];
    let b_transpose: Matrix = b_mat.clone().transpose();
    assert_eq!(b_transpose.get_data(), b_transpose_vec, "Transpose data not as expected");
    assert_eq!(b_transpose.get_rows(), 3, "Transpose row dimension not as expected");
    assert_eq!(b_transpose.get_cols(), 2, "Transpose col dimension not as expected");

    let value: f32 = 2.0;

    let fladd_vec: Vec<f32> = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let fladd_mat: Matrix = a_mat.clone() + value;
    assert_eq!(fladd_mat.get_data(), fladd_vec, "Matrix-Float addition not as expected");

    let add_vec: Vec<f32> = vec![3.0, 3.0, 5.0, 7.0, 7.0, 7.0];
    let add_mat: Matrix = (a_mat.clone() + b_mat.clone()).expect("Failed to add Matrix A and Matrix B");
    assert_eq!(add_mat.get_data(), add_vec, "Matrix-Matrix addition not as expected");

    let neg_vec: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0];
    let neg_mat: Matrix = -a_mat.clone();
    assert_eq!(neg_mat.get_data(), neg_vec, "Matrix negation not as expected");

    let flsub_vec: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let flsub_mat: Matrix = a_mat.clone() - value;
    assert_eq!(flsub_mat.get_data(), flsub_vec, "Matrix-Float subtraction not as expected");

    let sub_vec: Vec<f32> = vec![-1.0, 1.0, 1.0, 1.0, 3.0, 5.0];
    let sub_mat: Matrix = (a_mat.clone() - b_mat.clone()).expect("Failed to subtract Matrix B from Matrix A");
    assert_eq!(sub_mat.get_data(), sub_vec, "Matrix-Matrix subtraction not as expected");

    let flmul_vec: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let flmul_mat: Matrix = value * a_mat.clone();
    assert_eq!(flmul_mat.get_data(), flmul_vec, "Matrix-Float multiplication not as expected");

    let mul_vec: Vec<f32> = vec![10.0, 10.0, 25.0, 28.0];
    let mul_mat: Matrix = (a_mat.clone() * b_transpose).expect("Failed to multiply Matrix A by Matrix B");
    assert_eq!(mul_mat.get_data(), mul_vec, "Matrix-Matrix multiplication data not as expected");
    assert_eq!(mul_mat.get_rows(), 2, "Matrix-Matrix multiplication row dimension not as expected");
    assert_eq!(mul_mat.get_cols(), 2, "Matrix-Matrix multiplication col dimension not as expected");

    let ewmult_vec: Vec<f32> = vec![2.0, 2.0, 6.0, 12.0, 10.0, 6.0];
    let ewmult_mat: Matrix = a_mat.ewmult(b_mat).expect("Failed to elementwise multiply Matrix A by Matrix B");
    assert_eq!(ewmult_mat.get_data(), ewmult_vec, "Elementwise multiplication data not as expected");
    assert_eq!(ewmult_mat.get_rows(), 2, "Elementwise multiplication row dimension not as expected");
    assert_eq!(ewmult_mat.get_cols(), 3, "Elementwise multiplication col dimension not as expected");
}

#[test]
fn matrix_multiplication() {
    let mut calc: Calculator = Calculator::init()
        .expect("Failed to initialize calculator");

    let a_vec: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_vec: Vec<f32> = vec![2.0, 1.0, 2.0, 3.0, 2.0, 1.0];

    let c_vec: Vec<f32> = vec![12.0, 10.0, 30.0, 25.0];
    let d_vec: Vec<f32> = vec![52.0, 74.0, 96.0, 130.0, 185.0, 240.0];

    let e_vec: Vec<f32> = vec![6.0, 9.0, 12.0, 14.0, 19.0, 24.0, 6.0, 9.0, 12.0];
    let f_vec: Vec<f32> = vec![54.0, 45.0, 114.0, 95.0, 54.0, 45.0];

    let a_mat: Matrix = Matrix::new(a_vec, 2, 3)
        .expect("Failed to create Matrix A");
    let b_mat: Matrix = Matrix::new(b_vec, 3, 2)
        .expect("Failed to create Matrix B");

    let a_idx: usize = calc.store_matrix(a_mat)
        .expect("Failed to store Matrix A in calculator memory");
    let b_idx: usize = calc.store_matrix(b_mat)
        .expect("Failed to store Matrix B in calculator memory");

    let (c_mat, c_idx) = calc.mat_mul(a_idx, b_idx)
        .expect("Failed to mulitply Matrix A and Matrix B");

    assert_eq!(c_mat.get_data(), c_vec, "Matrix C data not as expected");
    assert_eq!(c_mat.get_rows(), 2, "Matrix C row dimension not as expected");
    assert_eq!(c_mat.get_cols(), 2, "Matrix C col dimension not as expected");

    let (d_mat, _) = calc.mat_mul(c_idx, a_idx)
        .expect("Failed to multiply Matrix C and Matrix A");

    assert_eq!(d_mat.get_data(), d_vec, "Matrix D data not as expected");
    assert_eq!(d_mat.get_rows(), 2, "Matrix D row dimension not as expected");
    assert_eq!(d_mat.get_cols(), 3, "Matrix D col dimension not as expected");

    let (e_mat, e_idx) = calc.mat_mul(b_idx, a_idx)
        .expect("Failed to mulitply Matrix B and Matrix A");

    assert_eq!(e_mat.get_data(), e_vec, "Matrix E data not as expected");
    assert_eq!(e_mat.get_rows(), 3, "Matrix E row dimension not as expected");
    assert_eq!(e_mat.get_cols(), 3, "Matrix E col dimension not as expected");

    let (f_mat, _) = calc.mat_mul(e_idx, b_idx)
        .expect("Failed to multiply Matrix E and Matrix B");

    assert_eq!(f_mat.get_data(), f_vec, "Matrix F data not as expected");
    assert_eq!(f_mat.get_rows(), 3, "Matrix F row dimension not as expected");
    assert_eq!(f_mat.get_cols(), 2, "Matrix F col dimension not as expected");
}