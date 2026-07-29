mod state;
mod weno;
mod linalg;
mod utils;
use ndarray::{array, Array2, Array1};
mod noncon;

fn constant_stencil(s: state::State) -> [state::State; 7] {
        [s; 7]
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

    let s = state::State{
            rho: 1.0,
            mom: 0.5,
            ee: 2.0,
            ei: 2.0,
            er: 2.0,
    };

        let stencil = constant_stencil(s);

        let result = noncon::nonconservative(
            &stencil,
            1.0,
        );

        let result = noncon::dNdx(&stencil,1.0);
        let result = noncon::upwind_jump_left(&stencil);

        println!("{:?}",result);

        let state =
    state::State{
        rho:1.0,
        mom:0.5,
        ee:3.0,
        ei:2.0,
        er:1.0,
    };


    let stencil =
    weno::Stencil6{
        points:[
            state,
            state,
            state,
            state,
            state,
            state,
        ]
    };

    let ans = stencil.reconstruction();
    println!("{:?}",ans);
}


