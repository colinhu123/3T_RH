mod state;
mod weno;
mod dt;
mod noncon;
mod source;
mod diffusion;
mod constant;
mod io;

use rayon::prelude::*;

type Grid = Vec<Vec<state::State>>;

fn noncon_stencil_extractor(
    u: &Grid,
    i: usize,
    j: usize,
    dir: state::Direction,
) -> [state::State; 9] {
    let nx = u.len();
    let ny = u[0].len();

    let mut stencil = [state::State::new(); 9];

    for k in 0..9 {
        match dir {
            state::Direction::X => {
                let ii = (i + nx - 4 + k) % nx;
                stencil[k] = u[ii][j];
            }

            state::Direction::Y => {
                let jj = (j + ny - 4 + k) % ny;
                stencil[k] = u[i][jj];
            }
        }
    }

    stencil
}

fn weno_stencil_extractor(
    u: &Grid,
    i: usize,
    j: usize,
    dir: state::Direction,
) -> weno::Stencil6 {
    let nx = u.len();
    let ny = u[0].len();

    let mut points = [state::State::new(); 6];

    for k in 0..6 {
        match dir {
            state::Direction::X => {
                let ii = (i + nx - 3 + k) % nx;
                points[k] = u[ii][j];
            }

            state::Direction::Y => {
                let jj = (j + ny - 3 + k) % ny;
                points[k] = u[i][jj];
            }
        }
    }

    weno::Stencil6 {
        points,
        dir,
    }
}

fn diffusion_stencil_extractor(
    u: &Grid,
    i: usize,
    j: usize,
    dir: state::Direction,
) -> diffusion::DiffusionStencil {
    let nx = u.len();
    let ny = u[0].len();

    let mut points = [state::State::new(); 6];

    for k in 0..6 {
        match dir {
            state::Direction::X => {
                let ii = (i + nx - 3 + k) % nx;
                points[k] = u[ii][j];
            }

            state::Direction::Y => {
                let jj = (j + ny - 3 + k) % ny;
                points[k] = u[i][jj];
            }
        }
    }

    diffusion::DiffusionStencil {
        points,
        dir,
    }
}


fn l(
    u: &Grid,
    dx: f64,
    dy: f64,
) -> Grid {
    let nx = u.len();
    let ny = u[0].len();

    // ============================================================
    // X interface flux + diffusion
    // ============================================================

    let flux_x: Grid =
        (0..=nx)
            .into_par_iter()
            .map(|i| {
                let ii = i % nx;

                (0..ny)
                    .map(|j| {
                        let stencil =
                            weno_stencil_extractor(
                                u,
                                ii,
                                j,
                                state::Direction::X,
                            );

                        stencil.reconstruction(true)
                    })
                    .collect()
            })
            .collect();

    let diff_x: Grid =
        (0..=nx)
            .into_par_iter()
            .map(|i| {
                let ii = i % nx;

                (0..ny)
                    .map(|j| {
                        let stencil =
                            diffusion_stencil_extractor(
                                u,
                                ii,
                                j,
                                state::Direction::X,
                            );

                        stencil.build_diffusion()
                    })
                    .collect()
            })
            .collect();

    // ============================================================
    // Y interface flux + diffusion
    // ============================================================

    let flux_y: Grid =
        (0..nx)
            .into_par_iter()
            .map(|i| {
                (0..=ny)
                    .map(|j| {
                        let jj = j % ny;

                        let stencil =
                            weno_stencil_extractor(
                                u,
                                i,
                                jj,
                                state::Direction::Y,
                            );

                        stencil.reconstruction(true)
                    })
                    .collect()
            })
            .collect();

    let diff_y: Grid =
        (0..nx)
            .into_par_iter()
            .map(|i| {
                (0..=ny)
                    .map(|j| {
                        let jj = j % ny;

                        let stencil =
                            diffusion_stencil_extractor(
                                u,
                                i,
                                jj,
                                state::Direction::Y,
                            );

                        stencil.build_diffusion()
                    })
                    .collect()
            })
            .collect();

    // ============================================================
    // Cell-centered RHS
    // ============================================================

    (0..nx)
        .into_par_iter()
        .map(|i| {
            (0..ny)
                .map(|j| {
                    // --------------------------------------------
                    // Conservative flux divergence
                    // --------------------------------------------

                    let fx =
                        state::update(
                            flux_x[i][j],
                            flux_x[i + 1][j],
                        )
                        .scalar_prod(1.0 / dx);

                    let fy =
                        state::update(
                            flux_y[i][j],
                            flux_y[i][j + 1],
                        )
                        .scalar_prod(1.0 / dy);

                    // --------------------------------------------
                    // Nonconservative X
                    // --------------------------------------------

                    let stencil_x =
                        noncon_stencil_extractor(
                            u,
                            i,
                            j,
                            state::Direction::X,
                        );

                    let nc_x =
                        noncon::nonconservative_x(
                            &stencil_x,
                            dx,
                        );

                    // --------------------------------------------
                    // Nonconservative Y
                    // --------------------------------------------

                    let stencil_y =
                        noncon_stencil_extractor(
                            u,
                            i,
                            j,
                            state::Direction::Y,
                        );

                    let nc_y =
                        noncon::nonconservative_y(
                            &stencil_y,
                            dy,
                        );

                    // --------------------------------------------
                    // Source
                    // --------------------------------------------

                    let s =
                        source::source(u[i][j]);

                    // --------------------------------------------
                    // Diffusion
                    // --------------------------------------------

                    let dif_x =
                        state::update(
                            diff_x[i][j],
                            diff_x[i + 1][j],
                        )
                        .scalar_prod(-1.0 / (dx * dx));

                    let dif_y =
                        state::update(
                            diff_y[i][j],
                            diff_y[i][j + 1],
                        )
                        .scalar_prod(-1.0 / (dy * dy));

                    // --------------------------------------------
                    // Total
                    // --------------------------------------------

                    fx
                        .add(fy)
                        .add(nc_x)
                        .add(nc_y)
                        .add(s)
                        .add(dif_x)
                        .add(dif_y)
                })
                .collect()
        })
        .collect()
}


fn rk3_ssp(
    u: &Grid,
    dx: f64,
    dy: f64,
    dt: f64,
) -> Grid {
    let nx = u.len();
    let ny = u[0].len();

    // ============================================================
    // Stage 1
    // ============================================================

    let l1 = l(u, dx, dy);

    let u1: Grid =
        (0..nx)
            .into_par_iter()
            .map(|i| {
                (0..ny)
                    .map(|j| {
                        u[i][j]
                            .add(
                                l1[i][j]
                                    .scalar_prod(dt)
                            )
                    })
                    .collect()
            })
            .collect();

    // ============================================================
    // Stage 2
    // ============================================================

    let l2 = l(&u1, dx, dy);

    let u2: Grid =
        (0..nx)
            .into_par_iter()
            .map(|i| {
                (0..ny)
                    .map(|j| {
                        u[i][j]
                            .scalar_prod(0.75)
                            .add(
                                u1[i][j]
                                    .scalar_prod(0.25)
                            )
                            .add(
                                l2[i][j]
                                    .scalar_prod(dt / 4.0)
                            )
                    })
                    .collect()
            })
            .collect();

    // ============================================================
    // Stage 3
    // ============================================================

    let l3 = l(&u2, dx, dy);

    (0..nx)
        .into_par_iter()
        .map(|i| {
            (0..ny)
                .map(|j| {
                    u[i][j]
                        .scalar_prod(1.0 / 3.0)
                        .add(
                            u2[i][j]
                                .scalar_prod(2.0 / 3.0)
                        )
                        .add(
                            l3[i][j]
                                .scalar_prod(
                                    2.0 * dt / 3.0
                                )
                        )
                })
                .collect()
        })
        .collect()
}

fn init() -> (Grid, usize, usize) {
    let nx = 400;
    let ny = 100;

    let mut u =
        vec![
            vec![state::State::new(); ny];
            nx
        ];

    let s1 = state::State {
        rho: 0.445,
        mom_x: 0.31061,
        mom_y: 0.0,
        ee: 1.8,
        ei: 1.8,
        er: 3.564,
    };

    let s2 = state::State {
        rho: 0.5,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: 0.285,
        ei: 0.285,
        er: 0.571,
    };

    for i in 0..nx {
        for j in 0..ny {
            if i < (0.25 * nx as f64) as usize {
                u[i][j] = s1;
            } else if i >
                (0.75 * nx as f64) as usize
            {
                u[i][j] = s1;
            } else {
                u[i][j] = s2;
            }
        }
    }

    (u, nx, ny)
}



fn calc_global_dt(
    u: &Grid,
    dx: f64,
    dy: f64,
) -> f64 {
    let nx = u.len();
    let ny = u[0].len();

    let mut global_dt = f64::INFINITY;

    for i in 0..nx {
        for j in 0..ny {
            let dt_x =
                dt::get_local_dt(
                    u[i][j],
                    dx,
                    dy,
                );
            global_dt =
                global_dt
                .min(dt_x)
        }
    }

    global_dt
}


fn main() {
    io::clear_data_folder();

    let (mut u, nx, ny) = init();

    let lx = 40.0;
    let ly = 10.0;

    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    io::save_data(
        &u,
        "solution_0000.dat",
        lx,
        ly,
    );

    let mut t = 0.0;
    let t_final = 1.0;

    let mut n = 0usize;

    while t < t_final {
        let mut dt =
            calc_global_dt(
                &u,
                dx,
                dy,
            );

        // Do not step beyond final time
        if t + dt > t_final {
            dt = t_final - t;
        }

        u = rk3_ssp(
            &u,
            dx,
            dy,
            dt,
        );

        t += dt;
        n += 1;

        println!(
            "step = {}, t = {:.8e}, dt = {:.8e}",
            n,
            t,
            dt
        );

        let filename =
            format!(
                "solution_{:04}.dat",
                n
            );

        io::save_data(
            &u,
            &filename,
            lx,
            ly,
        );
    }

    println!(
        "Finished: t = {}, steps = {}",
        t,
        n
    );
}