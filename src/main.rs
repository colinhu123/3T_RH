mod state;
mod weno;
mod utils;
mod noncon;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::Write;

use ndarray::Array2;

const CFL: f64 = 0.3;

fn noncon_stencil_extractor(u: &Vec<state::State>,i: usize) -> [state::State; 9] {
    let nx = u.len();
    let mut stencil = [state::State::new(); 9];
    for j in 0..9 {
        let n = (i + nx - 4 + j) % nx;
        stencil[j] = u[n];
    }
    stencil
}

fn weno_stencil_extractor(u: &Vec<state::State>, i: usize) -> weno::Stencil6 {
    //here i range is 0..(nx+1)
    let nx = u.len();
    let mut points = [state::State::new(); 6];
    for j in 0..6 {
        let n = (i + nx - 3 + j) % nx;
        points[j] = u[n];
    }

    weno::Stencil6 {
        points: points,
    }
}


fn l(
    u: &Vec<state::State>,
    global_alpha: f64,
    dx: f64
) -> Vec<state::State> {

    let nx = u.len();

    let mut flux = vec![
        state::State::new();
        nx + 1
    ];
    let mut qp1 = vec![
        state::State::new();
        nx
    ];

    for i in 0..(nx+1) {

        let stencil =
            weno_stencil_extractor(&u, i);
            //println!("{:?}",stencil);

        flux[i] =
            stencil.reconstruction(global_alpha, true);
    }
    //println!("{:?}",flux);
    for i in 0..nx {
        let k1 = state::update(flux[i], flux[i+1]);
        let k1 = k1.scalar_prod(1.0/dx);
        let sten = noncon_stencil_extractor(&u, i);
        let k2 = noncon::nonconservative(&sten, dx);
        qp1[i] = k1.add(k2);
    }
    qp1
}


fn rk3_ssp(u: &Vec<state::State>, global_alpha: f64, dx: f64, dt: f64)->Vec<state::State> {
    let nx = u.len();

    let l1 = l(u,global_alpha,dx);
    let mut u1 = vec![
        state::State::new();
        nx
    ];
    for i in 0..nx {
        let k1 = l1[i].scalar_prod(dt);
        u1[i] = k1.add(u[i]);
    }

    let mut u2 = vec![state::State::new(); nx];
    let l2 = l(&u1, global_alpha, dx);
    for i in 0..nx {
        let state_term = u1[i].scalar_prod(0.25);
        let flux_term = l2[i].scalar_prod(dt/4.0);
        u2[i] = u[i].scalar_prod(0.75).add(state_term).add(flux_term);
    }

    let mut u3 = vec![state::State::new(); nx];
    let l3 = l(&u2, global_alpha, dx);
    for i in 0..nx {
        let state_term = u2[i].scalar_prod(2.0/3.0);
        let flux_term = l3[i].scalar_prod(2.0*dt/3.0);
        u3[i] = u[i].scalar_prod(1.0/3.0).add(state_term).add(flux_term);
    }

    u3
}

fn init() -> (Vec<state::State>,usize){
    let nx = 400;
    let mut u = vec![state::State::new(); nx];

    let s1 = state::State {
        rho: 0.445,
        mom: 0.31061,
        ee: 1.8,
        ei: 1.8,
        er: 3.564,
    };

    let s2 = state::State {
        rho:0.5,
        mom: 0.0,
        ee: 0.285,
        ei: 0.285,
        er: 0.571,
    };

    for i in 0..nx {
        if i < (0.25*nx as f64) as usize {
            u[i] = s1;
        }
        else if i > (0.75*nx as f64) as usize {
            u[i] = s1;
        }
        else {
            u[i] = s2;
        }
    }

    (u, nx)

}

fn calc_global_alpha(u: &Vec<state::State>) ->f64 {
    let nx = u.len();

    let mut alpha_max = 0.0;

    for i in 0..nx {

        let state1 = u[i];
        let u = state1.mom/state1.rho;

        let ee = state1.ee/state1.rho - u*u/6.0;
        let ei = state1.ei/state1.rho - u*u/6.0;
        let er = state1.er/state1.rho - u*u/6.0;
        let gi = state::GAMMA_I - 1.0;
        let ge = state::GAMMA_E - 1.0;
        let gr = state::GAMMA_R - 1.0;
        let cs = (state::GAMMA_E*ge*ee + state::GAMMA_I*gi*ei + state::GAMMA_R*gr*er).sqrt();

        let alpha = cs + (state1.mom/state1.rho).abs();

        if alpha > alpha_max {
            alpha_max = alpha;
        }
        else {
            continue;
        }
    }
    alpha_max
}

fn save_data(
    u: &Vec<state::State>,
    filename: &str
) {

    // create data folder if it does not exist
    create_dir_all("data")
        .expect("Cannot create data directory");


    let path = format!("data/{}", filename);


    let mut file =
        File::create(path)
        .expect("Cannot create output file");


    // header
    writeln!(
        file,
        "x,rho,mom,ee,ei,er"
    ).unwrap();


    let nx = u.len();


    for i in 0..nx {

        let x =
            i as f64 / (nx-1) as f64;


        writeln!(
            file,
            "{},{},{},{},{},{}",
            x,
            u[i].rho,
            u[i].mom,
            u[i].ee,
            u[i].ei,
            u[i].er
        ).unwrap();
    }

}

fn clear_data_folder() {

    let path = "data";

    if std::path::Path::new(path).exists() {

        remove_dir_all(path)
            .expect("Failed to remove old data folder");

    }

    create_dir_all(path)
        .expect("Failed to create data folder");
}

fn main() {
    
    clear_data_folder();

    let  (mut u,nx) = init();


    save_data(
        &u,
        "solution_0000.dat"
    );


    let dx = 1.0/nx as f64;

    let mut t = 0.0;

    let t_f = 0.05;

    for n in 0..800 {

        let alpha = 2.0*calc_global_alpha(&u);

        let dt = dx/alpha*CFL;
        println!("{:?}", dt);

        u =
            rk3_ssp(
                &u,
                alpha,
                dx,
                dt
            );


        let filename =
            format!("solution_{:04}.dat", n+1);


        save_data(
            &u,
            &filename
        );

        t += dt;

        if t >= t_f {
            println!("t_final has reached");
            break;
        }
    }
}



//fn main() {
//    let s1 = state::State {
//        rho: 0.445,
//        mom: 0.31061,
//        ee: 1.8,
//        ei: 1.8,
//        er: 3.564,
//    };
//
//    let stencil = weno::Stencil6 {
//        points: [s1; 6],
//    };
//
//    let (lambda,r) = stencil.build_r();
//    let gamma = Array2::from_diag(&lambda);
//    let a = stencil.build_a();
//    println!("{:?}",a.dot(&r));
//    println!("{:?}",r.dot(&gamma));
//}