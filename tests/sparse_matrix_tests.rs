mod test_parameters;
use test_parameters::P;

use gpurs::Result;

use gpurs::linalg::Matrix;
use gpurs::linalg::SparseMatrix;

#[test]
fn indexing_tests() -> Result<()> {
    let mat_data: Vec<P> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut mat: SparseMatrix<P> = SparseMatrix::<P>::from_dense(Matrix::new(mat_data, 3, 4)?);

    assert_eq!(mat[[1, 2]], 7.0, "Index value not as expected");
    mat.reassign_at([1, 2], -1.0)?;
    assert_eq!(mat[[1, 2]], -1.0, "Reassigned value not as expected");

    Ok(())
}

#[test]
fn csr_test() -> Result<()> {
    let mat_data: Vec<P> = vec![
        10.0, 20.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 30.0, 0.0, 40.0, 0.0, 0.0,
        0.0, 0.0, 50.0, 60.0, 70.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 80.0,
    ];
    let mat: SparseMatrix<P> = SparseMatrix::<P>::from_dense(Matrix::new(mat_data, 4, 6)?);

    println!("{:?}", mat.to_csr());

    Ok(())
}

#[test]
fn mul_tests() -> Result<()> {
    let mat_data: Vec<P> = vec![1.0, 2.0, 3.0, 4.0];
    let d_mat: Matrix<P> = Matrix::new(mat_data, 2, 2)?;
    let s_mat: SparseMatrix<P> = SparseMatrix::<P>::from_dense(d_mat.clone());

    println!("{}", (s_mat * d_mat)?);

    Ok(())
}