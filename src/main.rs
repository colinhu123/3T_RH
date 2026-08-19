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
mod geometry;
mod bc1;
mod field1;
mod ghost;
use bc1::BCType;
use field1::{Field, GridInfo};
use geometry::{FluidSide, Point, Polygon};
use state::{Direction, State};
use rayon::prelude::*;
use ghost::GhostGrid;


use std::sync::atomic::{AtomicBool, Ordering};
static REPORTED_BAD_STATE: AtomicBool = AtomicBool::new(false);
#[inline(always)] fn state_is_finite(s: State)
->bool{s.rho.is_finite()&&s.mom_x.is_finite()&&s.mom_y.is_finite()&&s.ee.is_finite()&&s.ei.is_finite()&&s.er.is_finite()}
#[inline(always)] fn internal_energies(s: State)
->Option<(f64,f64,f64)>{
    if !state_is_finite(s)||s.rho<=0.0{return None;} 
    let ux=s.mom_x/s.rho; let uy=s.mom_y/s.rho; 
    let k=(ux*ux+uy*uy)/6.0; 
    Some((s.ee/s.rho-k,s.ei/s.rho-k,s.er/s.rho-k))}
#[inline(always)] fn assert_admissible(s:State,idx:(isize,isize),where_:&str){
    let e=internal_energies(s);
    let bad=!state_is_finite(s)||s.rho<=0.0||e.map_or(true,|q|q.0<=0.0||q.1<=0.0||q.2<=0.0);
    if bad{if !REPORTED_BAD_STATE.swap(true,Ordering::SeqCst){eprintln!("\nFIRST NON-PHYSICAL STATE\nwhere = {}\nidx = {:?}\nstate = {:?}\ninternal energies = {:?}\n",where_,idx,s,e);} panic!("non-physical state at {:?} in {}",idx,where_);}}

#[inline]
fn noncon_stencil_extractor(
    u: &Field,
    ghosts: &GhostGrid,
    i: isize,
    j: isize,
    dir: Direction,
) -> [State; 9] {
    match dir {
        Direction::X => [
            ghosts.get_with_field(u, (i - 4, j)),
            ghosts.get_with_field(u, (i - 3, j)),
            ghosts.get_with_field(u, (i - 2, j)),
            ghosts.get_with_field(u, (i - 1, j)),
            ghosts.get_with_field(u, (i,     j)),
            ghosts.get_with_field(u, (i + 1, j)),
            ghosts.get_with_field(u, (i + 2, j)),
            ghosts.get_with_field(u, (i + 3, j)),
            ghosts.get_with_field(u, (i + 4, j)),
        ],

        Direction::Y => [
            ghosts.get_with_field(u, (i, j - 4)),
            ghosts.get_with_field(u, (i, j - 3)),
            ghosts.get_with_field(u, (i, j - 2)),
            ghosts.get_with_field(u, (i, j - 1)),
            ghosts.get_with_field(u, (i, j)),
            ghosts.get_with_field(u, (i, j + 1)),
            ghosts.get_with_field(u, (i, j + 2)),
            ghosts.get_with_field(u, (i, j + 3)),
            ghosts.get_with_field(u, (i, j + 4)),
        ],
    }
}

#[inline]
fn weno_stencil_extractor(
    u: &Field,
    ghosts: &GhostGrid,
    i: isize,
    j: isize,
    dir: Direction,
) -> weno::Stencil6 {
    let points = match dir {
        Direction::X => [
            ghosts.get_with_field(u, (i - 3, j)),
            ghosts.get_with_field(u, (i - 2, j)),
            ghosts.get_with_field(u, (i - 1, j)),
            ghosts.get_with_field(u, (i,     j)),
            ghosts.get_with_field(u, (i + 1, j)),
            ghosts.get_with_field(u, (i + 2, j)),
        ],

        Direction::Y => [
            ghosts.get_with_field(u, (i, j - 3)),
            ghosts.get_with_field(u, (i, j - 2)),
            ghosts.get_with_field(u, (i, j - 1)),
            ghosts.get_with_field(u, (i, j)),
            ghosts.get_with_field(u, (i, j + 1)),
            ghosts.get_with_field(u, (i, j + 2)),
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
    ghosts: &GhostGrid,
    i: isize,
    j: isize,
    dir: Direction,
) -> diffusion::DiffusionStencil {
    let points = match dir {
        Direction::X => [
            ghosts.get_with_field(u, (i - 3, j)),
            ghosts.get_with_field(u, (i - 2, j)),
            ghosts.get_with_field(u, (i - 1, j)),
            ghosts.get_with_field(u, (i,     j)),
            ghosts.get_with_field(u, (i + 1, j)),
            ghosts.get_with_field(u, (i + 2, j)),
        ],

        Direction::Y => [
            ghosts.get_with_field(u, (i, j - 3)),
            ghosts.get_with_field(u, (i, j - 2)),
            ghosts.get_with_field(u, (i, j - 1)),
            ghosts.get_with_field(u, (i, j)),
            ghosts.get_with_field(u, (i, j + 1)),
            ghosts.get_with_field(u, (i, j + 2)),
        ],
    };

    diffusion::DiffusionStencil {
        points,
        dir,
    }
}


fn l(
    u: &Field,
    ghosts: &mut GhostGrid,
    dx: f64,
    dy: f64,
) -> Vec<State> {
    let nx = u.grid.nx;
    let ny = u.grid.ny;

    // Every unique ghost is reconstructed exactly once for this RK stage.
    // Start with the serial version for correctness. After validating the
    // cached solver, this can be changed to update_values_parallel(u).
    let ghost_start = std::time::Instant::now();
    ghosts.update_values_parallel(u);
    eprintln!(
        "DEBUG L: updated {} unique ghosts in {:?}",
        ghosts.len(),
        ghost_start.elapsed()
    );

    let zero = State::new();
    let mut rhs = vec![zero; nx * ny];

    rhs.par_iter_mut()
        .enumerate()
        .for_each(|(linear, out)| {
            let i = linear / ny;
            let j = linear % ny;
            let ii = i as isize;
            let jj = j as isize;
            let idx = (ii, jj);

            if !u.is_in_domain(idx) {
                *out = State::new();
                return;
            }

            let flux_l = weno_stencil_extractor(
                u, ghosts, ii, jj, Direction::X,
            ).reconstruction(true);

            let flux_r = weno_stencil_extractor(
                u, ghosts, ii + 1, jj, Direction::X,
            ).reconstruction(true);

            let flux_b = weno_stencil_extractor(
                u, ghosts, ii, jj, Direction::Y,
            ).reconstruction(true);

            let flux_t = weno_stencil_extractor(
                u, ghosts, ii, jj + 1, Direction::Y,
            ).reconstruction(true);

            let fx = state::update(flux_l, flux_r)
                .scalar_prod(1.0 / dx);
            let fy = state::update(flux_b, flux_t)
                .scalar_prod(1.0 / dy);

            let stencil_x = noncon_stencil_extractor(
                u, ghosts, ii, jj, Direction::X,
            );
            let stencil_y = noncon_stencil_extractor(
                u, ghosts, ii, jj, Direction::Y,
            );

            let nc_x = noncon::nonconservative_x(&stencil_x, dx);
            let nc_y = noncon::nonconservative_y(&stencil_y, dy);

            // idx is guaranteed fluid here, so no ghost lookup is needed.
            let source_term = source::source(
                u.value[u.linear_index(idx)]
            );

            let diff_l = diffusion_stencil_extractor(
                u, ghosts, ii, jj, Direction::X,
            ).build_diffusion();

            let diff_r = diffusion_stencil_extractor(
                u, ghosts, ii + 1, jj, Direction::X,
            ).build_diffusion();

            let diff_b = diffusion_stencil_extractor(
                u, ghosts, ii, jj, Direction::Y,
            ).build_diffusion();

            let diff_t = diffusion_stencil_extractor(
                u, ghosts, ii, jj + 1, Direction::Y,
            ).build_diffusion();

            let dif_x = state::update(diff_l, diff_r)
                .scalar_prod(-1.0 / (dx * dx));
            let dif_y = state::update(diff_b, diff_t)
                .scalar_prod(-1.0 / (dy * dy));

            *out = fx
                .add(fy)
                .add(nc_x)
                .add(nc_y)
                .add(source_term)
                .add(dif_x)
                .add(dif_y);
        });

    rhs
}

fn empty_like(u: &Field) -> Field {
    let outer = Polygon::new(u.outer_bound.points.clone(), FluidSide::Inside);
    let inner = Polygon::new(u.inner_bound.points.clone(), FluidSide::Outside);

    Field::new(
        u.grid,
        u.bc_inner.clone(),
        u.bc_outer.clone(),
        State::new(),
        outer,
        inner,
    )
}


fn rk3_ssp(
    u: &Field,
    ghosts: &mut GhostGrid,
    dx: f64,
    dy: f64,
    dt: f64,
) -> Field {
    let nx = u.grid.nx;
    let ny = u.grid.ny;

    let l1 = l(u, ghosts, dx, dy);
    let mut u1 = empty_like(u);
    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);
            if !u.is_in_domain(idx) { continue; }
            let linear = i * ny + j;
            let value = u.get(idx).add(l1[linear].scalar_prod(dt));
            assert_admissible(value, idx, "RK1 state");
            u1.set(idx, u.get(idx).add(l1[linear].scalar_prod(dt)));
        }
    }

    let l2 = l(&u1, ghosts, dx, dy);
    let mut u2 = empty_like(u);
    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);
            if !u.is_in_domain(idx) { continue; }
            let linear = i * ny + j;
            let value = u.get(idx)
                .scalar_prod(0.75)
                .add(u1.get(idx).scalar_prod(0.25))
                .add(l2[linear].scalar_prod(dt / 4.0));
            assert_admissible(value, idx, "RK2 state");
            u2.set(idx, value);
        }
    }

    let l3 = l(&u2, ghosts, dx, dy);
    let mut u3 = empty_like(u);
    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);
            if !u.is_in_domain(idx) { continue; }
            let linear = i * ny + j;
            let value = u.get(idx)
                .scalar_prod(1.0 / 3.0)
                .add(u2.get(idx).scalar_prod(2.0 / 3.0))
                .add(l3[linear].scalar_prod(2.0 * dt / 3.0));
            assert_admissible(value, idx, "RK3 state");
            u3.set(idx, value);
        }
    }

    u3
}


fn init_bubble() -> Field {
    // ============================================================
    // Computational domain
    // ============================================================

    let nx = 400;
    let ny = 134;

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

    // Polygon edge ordering is [Bottom, Right, Top, Left].
    let outer_bound = Polygon::new(
        vec![
            Point { x: x0,      y: y0 },
            Point { x: x0 + lx, y: y0 },
            Point { x: x0 + lx, y: y0 + ly },
            Point { x: x0,      y: y0 + ly },
        ],
        FluidSide::Inside,
    );
    let bc_outer = vec![
        BCType::Wall,
        BCType::Constant(right_state),
        BCType::Wall,
        BCType::Constant(left_state),
    ];

    // This test has no solid embedded obstacle: keep the mandatory inner
    // polygon far outside the computational domain so it excludes no cells.
    let inner_bound = Polygon::new(
        vec![
            Point { x: -1002.0, y: -1002.0 },
            Point { x: -1001.0, y: -1002.0 },
            Point { x: -1001.0, y: -1001.0 },
            Point { x: -1002.0, y: -1001.0 },
        ],
        FluidSide::Outside,
    );
    let bc_inner = vec![BCType::Wall; 4];

    let mut u = Field::new(
        grid,
        bc_inner,
        bc_outer,
        State::new(),
        outer_bound,
        inner_bound,
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

            if u.is_in_domain((ii, jj)) {
                u.set((ii, jj), state);
            }
        }
    }

    u
}


fn calc_global_dt(
    u: &Field,
) -> f64 {
    let nx = u.grid.nx;
    let ny = u.grid.ny;

    let dx = u.grid.dx;
    let dy = u.grid.dy;

    let mut global_dt =
        f64::INFINITY;

    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);
            if !u.is_in_domain(idx) {
                continue;
            }
            let state = u.get(idx);

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

    let dx = u.grid.dx;
    let dy = u.grid.dy;
    let lx = dx * u.grid.nx as f64;
    let ly = dy * u.grid.ny as f64;

    let offsets =
        ghost::default_stencil_offsets();

    let mut ghosts =
        ghost::GhostGrid::build(
            &u,
            &offsets,
        );

    ghosts.print_summary();

    let mut t = 0.0;

    let t_final = 7.1571;

    // Physical-time interval between stored solutions.
    let t_store_interval = 0.02;
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
        eprintln!("DEBUG: entering step {}, t={}", n + 1, t);

    eprintln!("DEBUG: calculating dt...");
    let dt_cfl = calc_global_dt(&u);
    eprintln!("DEBUG: dt done: {}", dt_cfl);

    let mut dt = dt_cfl;

    if next_store_time <= t_final
        && t + dt > next_store_time
    {
        dt = next_store_time - t;
    }

    if t + dt > t_final {
        dt = t_final - t;
    }

    eprintln!("DEBUG: entering RK3, dt={}", dt);

    u = rk3_ssp(
        &u,
        &mut ghosts,
        dx,
        dy,
        dt,
    );

    eprintln!("DEBUG: RK3 finished");

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

    
