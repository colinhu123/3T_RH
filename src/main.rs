mod state;
mod weno;
mod linalg;
mod utils;
use ndarray::{array, Array2, Array1};

fn check_eigenvectors(
    a: &Array2<f64>,
    lambda: &Array1<f64>,
    r: &Array2<f64>,
) {

    for i in 0..lambda.len() {

        // extract eigenvector column
        let v = r.column(i).to_owned();

        // A*v
        let av = a.dot(&v);

        // lambda*v
        let lv = &v * lambda[i];

        // error
        let error = &av - &lv;

        let norm = error.dot(&error).sqrt();


        println!(
            "Eigenvalue {} = {}, error = {}",
            i,
            lambda[i],
            norm
        );
    }
}

fn main() {

    let a: Array2<f64> = array![
        [4.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 3.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 4.0],
    ];


    let k = array![[1.0],[1.0],[1.0],[1.0],[1.0]];

    let ans = a.dot(&k);

    println!{"{:?}",ans};
}


