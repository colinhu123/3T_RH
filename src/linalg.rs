use ndarray::{Array2, Array1};

pub fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x,y)| x*y)
        .sum()
}


pub fn norm(a: &Array1<f64>) -> f64 {
    dot(a,a).sqrt()
}

pub fn qr(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {

    let (m,n)=a.dim();

    let mut q = Array2::<f64>::zeros((m,n));

    let mut r = Array2::<f64>::zeros((n,n));


    for j in 0..n {

        // take column j of A
        let mut v = a.column(j).to_owned();


        // subtract projections
        for i in 0..j {

            let qi = q.column(i);

            let rij = qi.dot(&a.column(j));

            r[[i,j]] = rij;


            for k in 0..m {
                v[k] -= rij*q[[k,i]];
            }
        }


        // normalize
        let rjj = norm(&v);

        r[[j,j]]=rjj;


        for k in 0..m {
            q[[k,j]]=v[k]/rjj;
        }
    }


    (q,r)
}

pub fn eigen_qr(
    a:&Array2<f64>,
    iterations:usize
)->(Array1<f64>,Array2<f64>)
{

    let n=a.nrows();

    let mut ak=a.clone();


    // accumulated eigenvectors
    let mut v=Array2::<f64>::eye(n);


    for _ in 0..iterations {

        let (q,r)=qr(&ak);


        // A(k+1)=RQ
        ak=r.dot(&q);


        // accumulate Q
        v=v.dot(&q);
    }


    let mut lambda=Array1::<f64>::zeros(n);

    for i in 0..n {
        lambda[i]=ak[[i,i]];
    }


    (lambda,v)
}

pub fn inverse(a: &Array2<f64>) -> Array2<f64> {

    let n = a.nrows();

    assert_eq!(n, a.ncols());

    // augmented matrix [A | I]
    let mut aug = Array2::<f64>::zeros((n, 2*n));


    for i in 0..n {
        for j in 0..n {
            aug[[i,j]] = a[[i,j]];
        }

        aug[[i,n+i]] = 1.0;
    }


    // Gauss-Jordan elimination
    for i in 0..n {

        // Find pivot
        let mut pivot = i;

        for k in i+1..n {
            if aug[[k,i]].abs() > aug[[pivot,i]].abs() {
                pivot = k;
            }
        }


        // Swap rows
        if pivot != i {
            for j in 0..2*n {
                let tmp = aug[[i,j]];
                aug[[i,j]] = aug[[pivot,j]];
                aug[[pivot,j]] = tmp;
            }
        }


        // Normalize pivot row
        let diag = aug[[i,i]];

        assert!(
            diag.abs() > 1e-14,
            "Matrix is singular"
        );


        for j in 0..2*n {
            aug[[i,j]] /= diag;
        }


        // Eliminate other rows
        for k in 0..n {

            if k != i {

                let factor = aug[[k,i]];


                for j in 0..2*n {
                    aug[[k,j]] -= factor * aug[[i,j]];
                }
            }
        }
    }


    // Extract right half
    let mut inv = Array2::<f64>::zeros((n,n));

    for i in 0..n {
        for j in 0..n {
            inv[[i,j]] = aug[[i,n+j]];
        }
    }


    inv
}