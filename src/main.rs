mod state;
mod weno;
mod dt;
mod noncon;
mod source;
mod diffusion;
mod constant;
mod io;
mod field;
mod bc;

use bc::BCType;
use field::{Field, GridInfo};
use state::{Direction, State};

#[inline]
fn noncon_stencil_extractor(
    u: &Field,
    i: isize,
    j: isize,
    dir: Direction,
) -> [State; 9] {
    match dir {
        Direction::X => [
            u.get((i - 4, j)),
            u.get((i - 3, j)),
            u.get((i - 2, j)),
            u.get((i - 1, j)),
            u.get((i,     j)),
            u.get((i + 1, j)),
            u.get((i + 2, j)),
            u.get((i + 3, j)),
            u.get((i + 4, j)),
        ],

        Direction::Y => [
            u.get((i, j - 4)),
            u.get((i, j - 3)),
            u.get((i, j - 2)),
            u.get((i, j - 1)),
            u.get((i, j)),
            u.get((i, j + 1)),
            u.get((i, j + 2)),
            u.get((i, j + 3)),
            u.get((i, j + 4)),
        ],
    }
}

#[inline]
fn weno_stencil_extractor(
    u: &Field,
    i: isize,
    j: isize,
    dir: Direction,
) -> weno::Stencil6 {
    let points = match dir {
        Direction::X => [
            u.get((i - 3, j)),
            u.get((i - 2, j)),
            u.get((i - 1, j)),
            u.get((i,     j)),
            u.get((i + 1, j)),
            u.get((i + 2, j)),
        ],

        Direction::Y => [
            u.get((i, j - 3)),
            u.get((i, j - 2)),
            u.get((i, j - 1)),
            u.get((i, j)),
            u.get((i, j + 1)),
            u.get((i, j + 2)),
        ],
    };

    weno::Stencil6 {
        points,
        dir,
    }
}

#[inline]
fn diffusion_stencil_extractor(
    u: &Field,
    i: isize,
    j: isize,
    dir: Direction,
) -> diffusion::DiffusionStencil {
    let points = match dir {
        Direction::X => [
            u.get((i - 3, j)),
            u.get((i - 2, j)),
            u.get((i - 1, j)),
            u.get((i,     j)),
            u.get((i + 1, j)),
            u.get((i + 2, j)),
        ],

        Direction::Y => [
            u.get((i, j - 3)),
            u.get((i, j - 2)),
            u.get((i, j - 1)),
            u.get((i, j)),
            u.get((i, j + 1)),
            u.get((i, j + 2)),
        ],
    };

    diffusion::DiffusionStencil {
        points,
        dir,
    }
}


fn l(
    u: &Field,
    dx: f64,
    dy: f64,
) -> Field {
    let nx = u.nx();
    let ny = u.ny();

    let zero = State::new();

    // RHS has the same grid and BC as U.
    //
    // BC actually won't normally be queried on RHS during this
    // routine, but using the same grid keeps the RK interface simple.
    let mut rhs = Field::new(
        *u.grid(),
        [
            BCType::Periodic,
            BCType::Periodic,
            BCType::Periodic,
            BCType::Periodic,
        ],
    );

    let mut flux_x =
        vec![vec![zero; ny]; nx + 1];

    let mut flux_y =
        vec![vec![zero; ny + 1]; nx];

    let mut diff_x =
        vec![vec![zero; ny]; nx + 1];

    let mut diff_y =
        vec![vec![zero; ny + 1]; nx];

    // ============================================================
    // X interfaces
    // ============================================================

    for i in 0..=nx {
        for j in 0..ny {
            let ii = i as isize;
            let jj = j as isize;

            let stencil =
                weno_stencil_extractor(
                    u,
                    ii,
                    jj,
                    Direction::X,
                );

            flux_x[i][j] =
                stencil.reconstruction(true);

            let stencil =
                diffusion_stencil_extractor(
                    u,
                    ii,
                    jj,
                    Direction::X,
                );

            diff_x[i][j] =
                stencil.build_diffusion();
        }
    }

    // ============================================================
    // Y interfaces
    // ============================================================

    for i in 0..nx {
        for j in 0..=ny {
            let ii = i as isize;
            let jj = j as isize;

            let stencil =
                weno_stencil_extractor(
                    u,
                    ii,
                    jj,
                    Direction::Y,
                );

            flux_y[i][j] =
                stencil.reconstruction(true);

            let stencil =
                diffusion_stencil_extractor(
                    u,
                    ii,
                    jj,
                    Direction::Y,
                );

            diff_y[i][j] =
                stencil.build_diffusion();
        }
    }

    // ============================================================
    // Cell-centered RHS
    // ============================================================

    for i in 0..nx {
        for j in 0..ny {
            let ii = i as isize;
            let jj = j as isize;

            // ----------------------------------------------------
            // Conservative flux
            // ----------------------------------------------------

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

            // ----------------------------------------------------
            // Nonconservative X
            // ----------------------------------------------------

            let stencil_x =
                noncon_stencil_extractor(
                    u,
                    ii,
                    jj,
                    Direction::X,
                );

            let nc_x =
                noncon::nonconservative_x(
                    &stencil_x,
                    dx,
                );

            // ----------------------------------------------------
            // Nonconservative Y
            // ----------------------------------------------------

            let stencil_y =
                noncon_stencil_extractor(
                    u,
                    ii,
                    jj,
                    Direction::Y,
                );

            let nc_y =
                noncon::nonconservative_y(
                    &stencil_y,
                    dy,
                );

            // ----------------------------------------------------
            // Source
            // ----------------------------------------------------

            let s =
                source::source(
                    u.get((ii, jj))
                );

            // ----------------------------------------------------
            // Diffusion
            // ----------------------------------------------------

            let dif_x =
                state::update(
                    diff_x[i][j],
                    diff_x[i + 1][j],
                )
                .scalar_prod(-1.0)
                .scalar_prod(
                    1.0 / (dx * dx)
                );

            let dif_y =
                state::update(
                    diff_y[i][j],
                    diff_y[i][j + 1],
                )
                .scalar_prod(-1.0)
                .scalar_prod(
                    1.0 / (dy * dy)
                );

            // ----------------------------------------------------
            // Total RHS
            // ----------------------------------------------------

            let value =
                fx
                    .add(fy)
                    .add(nc_x)
                    .add(nc_y)
                    .add(s)
                    .add(dif_x)
                    .add(dif_y);

            rhs.set(
                (ii, jj),
                value,
            );
        }
    }

    rhs
}


fn rk3_ssp(
    u: &Field,
    dx: f64,
    dy: f64,
    dt: f64,
) -> Field {
    let nx = u.nx();
    let ny = u.ny();

    // ============================================================
    // Stage 1
    // ============================================================

    let l1 = l(u, dx, dy);

    let mut u1 =
        u.empty_like();

    for i in 0..nx {
        for j in 0..ny {
            let idx =
                (i as isize, j as isize);

            let value =
                u.get(idx)
                .add(
                    l1.get(idx)
                    .scalar_prod(dt)
                );

            u1.set(idx, value);
        }
    }

    // ============================================================
    // Stage 2
    // ============================================================

    let l2 = l(&u1, dx, dy);

    let mut u2 =
        u.empty_like();

    for i in 0..nx {
        for j in 0..ny {
            let idx =
                (i as isize, j as isize);

            let value =
                u.get(idx)
                .scalar_prod(0.75)
                .add(
                    u1.get(idx)
                    .scalar_prod(0.25)
                )
                .add(
                    l2.get(idx)
                    .scalar_prod(dt / 4.0)
                );

            u2.set(idx, value);
        }
    }

    // ============================================================
    // Stage 3
    // ============================================================

    let l3 = l(&u2, dx, dy);

    let mut u3 =
        u.empty_like();

    for i in 0..nx {
        for j in 0..ny {
            let idx =
                (i as isize, j as isize);

            let value =
                u.get(idx)
                .scalar_prod(1.0 / 3.0)
                .add(
                    u2.get(idx)
                    .scalar_prod(2.0 / 3.0)
                )
                .add(
                    l3.get(idx)
                    .scalar_prod(
                        2.0 * dt / 3.0
                    )
                );

            u3.set(idx, value);
        }
    }

    u3
}

fn init() -> Field {
    let nx = 400;
    let ny = 100;

    let lx = 40.0;
    let ly = 10.0;

    let dx =
        lx / nx as f64;

    let dy =
        ly / ny as f64;

    let grid =
        GridInfo::new(
            nx,
            ny,
            dx,
            dy,
            0.0,
            0.0,
        );

    // Current test case:
    // periodic in both X and Y.
    let bc = [
        BCType::Periodic, // Bottom
        BCType::Periodic, // Right
        BCType::Periodic, // Top
        BCType::Periodic, // Left
    ];

    let mut u =
        Field::new(
            grid,
            bc,
        );

    let s1 = State {
        rho: 0.445,
        mom_x: 0.31061,
        mom_y: 0.0,
        ee: 1.8,
        ei: 1.8,
        er: 3.564,
    };

    let s2 = State {
        rho: 0.5,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: 0.285,
        ei: 0.285,
        er: 0.571,
    };

    for i in 0..nx {
        for j in 0..ny {
            let state =
                if i <
                    (0.25 * nx as f64)
                        as usize
                {
                    s1
                } else if i >
                    (0.75 * nx as f64)
                        as usize
                {
                    s1
                } else {
                    s2
                };

            u.set(
                (
                    i as isize,
                    j as isize,
                ),
                state,
            );
        }
    }

    u
}

fn init_bubble() -> Field {
    // ============================================================
    // Computational domain
    // ============================================================

    let nx = 800;
    let ny = 268;

    let x0 = 0.0;
    let y0 = 0.0;

    let lx = 6.5;
    let ly = 0.89;

    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    let grid = GridInfo::new(
        nx,
        ny,
        dx,
        dy,
        x0,
        y0,
    );

    // ============================================================
    // Initial states
    // ============================================================

    // Left/background state:
    //
    // rho = 1
    // u = v = 0
    // pe = pi = pr = 0.238095
    let left_state = State::primi2con(
        1.0,
        0.0,
        0.0,
        0.238095,
        0.238095,
        0.238095,
    );

    // Post-shock/right state:
    //
    // rho = 1.3764
    // u = -0.3336
    // v = 0
    // pe = pi = pr = 0.373762
    let right_state = State::primi2con(
        1.3764,
        -0.3336,
        0.0,
        0.373762,
        0.373762,
        0.373762,
    );

    // Bubble:
    //
    // rho = 0.1819
    // u = v = 0
    // pe = pi = pr = 0.146972
    let bubble_state = State::primi2con(
        0.1819,
        0.0,
        0.0,
        0.146972,
        0.146972,
        0.146972,
    );

    // ============================================================
    // Boundary conditions
    //
    // Field BC ordering:
    //
    // [Bottom, Right, Top, Left]
    // ============================================================

    let bc = [
        BCType::Wall,                  // Bottom: reflective
        BCType::Constant(right_state), // Right:  Dirichlet
        BCType::Wall,                  // Top:    reflective
        BCType::Constant(left_state),  // Left:   Dirichlet
    ];

    let mut u = Field::new(
        grid,
        bc,
    );

    // ============================================================
    // Bubble geometry
    // ============================================================

    let xc = 3.5;
    let yc = 0.0;
    let radius = 0.5;

    let r2 = radius * radius;

    // Shock/interface initially at x = 4.5
    let shock_x = 4.5;

    // ============================================================
    // Fill physical cells
    // ============================================================

    for i in 0..nx {
        for j in 0..ny {
            let ii = i as isize;
            let jj = j as isize;

            // Cell-center coordinates
            let x = grid.x(ii);
            let y = grid.y(jj);

            // Distance from bubble center
            let bubble_distance2 =
                (x - xc).powi(2)
                + (y - yc).powi(2);

            let state =
                if bubble_distance2 <= r2 {
                    // Bubble takes precedence over background state.
                    bubble_state
                } else if x < shock_x {
                    // Pre-shock/background region
                    left_state
                } else {
                    // Post-shock region
                    right_state
                };

            u.set(
                (ii, jj),
                state,
            );
        }
    }

    u
}


fn calc_global_dt(
    u: &Field,
) -> f64 {
    let nx = u.nx();
    let ny = u.ny();

    let dx = u.grid().dx;
    let dy = u.grid().dy;

    let mut global_dt =
        f64::INFINITY;

    for i in 0..nx {
        for j in 0..ny {
            let state =
                u.get((
                    i as isize,
                    j as isize,
                ));

            let local_dt =
                dt::get_local_dt(
                    state,
                    dx,
                    dy,
                );

            global_dt =
                global_dt.min(local_dt);
        }
    }

    global_dt
}


fn main() {
    io::clear_data_folder();

    let mut u = init_bubble();

    let dx = u.grid().dx;
    let dy = u.grid().dy;
    let lx = dx * u.nx() as f64;
    let ly = dy * u.ny() as f64;


    let mut t = 0.0;

    let t_final = 1.1099;

    // Physical-time interval between stored solutions.
    let t_store_interval = 0.05;
    let mut next_store_time = t_store_interval;
    let mut n = 0usize;
    let mut store_id = 0usize;

    io::save_data(
        &u,
        "solution_0000.dat",
        lx,
        ly,
    );

    println!(
        "stored solution_0000.dat at t = {:.8e}",
        t
    );

    while t < t_final {
        // CFL-limited time step
        let dt_cfl = calc_global_dt(&u);
        let mut dt = dt_cfl;

        if next_store_time <= t_final
            && t + dt > next_store_time
        {
            dt =
                next_store_time - t;
        }
        if t + dt > t_final {
            dt =
                t_final - t;
        }

        u =
            rk3_ssp(
                &u,
                dx,
                dy,
                dt,
            );

        t += dt;
        n += 1;

        println!(
            "step = {}, t = {:.8e}, dt = {:.8e}, dt_cfl = {:.8e}",
            n,
            t,
            dt,
            dt_cfl,
        );

        let reached_store_time =
            next_store_time <= t_final
            && t >= next_store_time - 1e-12;

        if reached_store_time {
            store_id += 1;

            let filename =
                format!(
                    "solution_{:04}.dat",
                    store_id
                );

            io::save_data(
                &u,
                &filename,
                lx,
                ly,
            );

            println!(
                "stored {} at t = {:.8e}",
                filename,
                t
            );
            next_store_time =
                (store_id + 1) as f64
                * t_store_interval;
        }
    }

    let last_regular_store_time =
        store_id as f64
        * t_store_interval;

    if (t - last_regular_store_time).abs() > 1e-12 {
        store_id += 1;

        let filename =
            format!(
                "solution_{:04}.dat",
                store_id
            );

        io::save_data(
            &u,
            &filename,
            lx,
            ly,
        );

        println!(
            "stored final solution {} at t = {:.8e}",
            filename,
            t
        );
    }

    println!(
        "Finished: t = {:.8e}, steps = {}, stored = {}",
        t,
        n,
        store_id + 1,
    );
}