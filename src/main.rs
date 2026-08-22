mod state;
mod weno;
mod dt;
mod noncon;
mod source;
mod diffusion;
mod constant;
mod io;
mod geometry;
mod bc1;
mod field1;
mod ghost;
mod init;

use field1::Field;
use state::{Derived, Direction, State};
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

// ============================================================================
// Per-step scratch buffers, allocated once in main().
//
// fx[i][j] : x-interface flux at i-1/2          (i in 0..=nx)
// fy[j][i] : y-interface flux at j-1/2          (j in 0..=ny)
// dfx/dfy  : diffusion interface fluxes (same layout)
// rhs      : semi-discrete operator output
// derived  : per-stage derived quantities for fluid cells
// ============================================================================

struct Scratch {
    fx: Vec<State>,
    fy: Vec<State>,
    dfx: Vec<State>,
    dfy: Vec<State>,
    rhs: Vec<State>,
    derived: Vec<Derived>,
}

impl Scratch {
    fn new(u: &Field) -> Self {
        let nx = u.grid.nx;
        let ny = u.grid.ny;
        Self {
            fx: vec![State::new(); (nx + 1) * ny],
            fy: vec![State::new(); nx * (ny + 1)],
            dfx: vec![State::new(); (nx + 1) * ny],
            dfy: vec![State::new(); nx * (ny + 1)],
            rhs: vec![State::new(); nx * ny],
            derived: vec![Derived::new(); nx * ny],
        }
    }
}

#[inline(always)]
fn v_at(u: &Field, g: &GhostGrid, t: u32) -> State {
    let t = t as usize;
    if t < u.grid.len() { u.value[t] } else { g.values[t - u.grid.len()] }
}

#[inline(always)]
fn d_at(g: &GhostGrid, derived: &[Derived], t: u32) -> Derived {
    let t = t as usize;
    if t < derived.len() { derived[t] } else { g.derived[t - derived.len()] }
}

/// Gather a 6-point interface stencil from a fluid anchor cell.
///
/// `di0` is the offset (relative to the anchor) of the first stencil point
/// along `dir`. For an interface centered exactly on the anchor, di0 = -3;
/// for an interface centered one cell to the right/top, di0 = -2.
#[inline(always)]
fn gather6(
    u: &Field,
    g: &GhostGrid,
    derived: &[Derived],
    anchor: usize,
    di0: isize,
    dir: Direction,
) -> ([State; 6], [Derived; 6]) {
    let mut st = [State::new(); 6];
    let mut dd = [Derived::new(); 6];
    for q in 0..6isize {
        let d = di0 + q;
        let k = match dir {
            Direction::X => g.k_for(d, 0),
            Direction::Y => g.k_for(0, d),
        };
        let t = g.target(anchor, k);
        st[q as usize] = v_at(u, g, t);
        dd[q as usize] = d_at(g, derived, t);
    }
    (st, dd)
}

/// Gather a 9-point derived stencil (non-conservative term).
#[inline(always)]
fn gather9d(
    g: &GhostGrid,
    derived: &[Derived],
    anchor: usize,
    di0: isize,
    dir: Direction,
) -> [Derived; 9] {
    let mut dd = [Derived::new(); 9];
    for q in 0..9isize {
        let d = di0 + q;
        let k = match dir {
            Direction::X => g.k_for(d, 0),
            Direction::Y => g.k_for(0, d),
        };
        let t = g.target(anchor, k);
        dd[q as usize] = d_at(g, derived, t);
    }
    dd
}

/// Compute the interface fluxes (WENO + diffusion) once per interface, and
/// then assemble the cell-centered semi-discrete operator from the stored
/// interface values plus the non-conservative / source terms.
fn l(
    u: &Field,
    ghosts: &mut GhostGrid,
    s: &mut Scratch,
) {
    let nx = u.grid.nx;
    let ny = u.grid.ny;
    let dx = u.grid.dx;
    let dy = u.grid.dy;

    // Every unique ghost is reconstructed exactly once for this RK stage.
    ghosts.update_values_parallel(u);

    // Per-stage derived quantities for fluid cells.
    s.derived.par_iter_mut().enumerate().for_each(|(l, o)| {
        if u.fluid[l] {
            *o = Derived::from_state(u.value[l]);
        }
    });

    let g = &*ghosts;

    // ------------------------------------------------------------------
    // X-direction interface fluxes.
    //
    // Interface i-1/2 is centered on cell (i, j):
    //   * if (i, j)     is fluid, anchor = (i, j),   offsets -3..=2
    //   * else if (i-1,j) is fluid, anchor = (i-1,j), offsets -2..=3
    //   * else the interface is unused (both neighbors solid).
    // ------------------------------------------------------------------
    {
        let fx = &mut s.fx;
        let derived = &s.derived;
        fx.par_iter_mut().enumerate().for_each(|(lin, out)| {
            let i = lin / ny;
            let anchor = if i < nx && u.fluid[lin] {
                (lin, -3isize)
            } else if i >= 1 && u.fluid[lin - ny] {
                (lin - ny, -2isize)
            } else {
                *out = State::new();
                return;
            };
            let (st, dd) = gather6(u, g, derived, anchor.0, anchor.1, Direction::X);
            *out = weno::Stencil6::reconstruction_fast(&st, &dd, Direction::X, true);
        });
    }

    // ------------------------------------------------------------------
    // Y-direction interface fluxes (layout: [j][i] = j*nx + i).
    // ------------------------------------------------------------------
    {
        let fy = &mut s.fy;
        let derived = &s.derived;
        fy.par_iter_mut().enumerate().for_each(|(lin, out)| {
            let j = lin / nx;
            let i = lin % nx;
            let cell = i * ny + j;
            let anchor = if j < ny && u.fluid[cell] {
                (cell, -3isize)
            } else if j >= 1 && u.fluid[cell - 1] {
                (cell - 1, -2isize)
            } else {
                *out = State::new();
                return;
            };
            let (st, dd) = gather6(u, g, derived, anchor.0, anchor.1, Direction::Y);
            *out = weno::Stencil6::reconstruction_fast(&st, &dd, Direction::Y, true);
        });
    }

    // ------------------------------------------------------------------
    // Diffusion interface fluxes (only when diffusion is enabled).
    // ------------------------------------------------------------------
    if constant::DIFFUSION_ACTIVE {
        {
            let dfx = &mut s.dfx;
            let derived = &s.derived;
            dfx.par_iter_mut().enumerate().for_each(|(lin, out)| {
                let i = lin / ny;
                let anchor = if i < nx && u.fluid[lin] {
                    (lin, -3isize)
                } else if i >= 1 && u.fluid[lin - ny] {
                    (lin - ny, -2isize)
                } else {
                    *out = State::new();
                    return;
                };
                let (_st, dd) = gather6(u, g, derived, anchor.0, anchor.1, Direction::X);
                *out = diffusion::build_diffusion_from_derived(&dd);
            });
        }
        {
            let dfy = &mut s.dfy;
            let derived = &s.derived;
            dfy.par_iter_mut().enumerate().for_each(|(lin, out)| {
                let j = lin / nx;
                let i = lin % nx;
                let cell = i * ny + j;
                let anchor = if j < ny && u.fluid[cell] {
                    (cell, -3isize)
                } else if j >= 1 && u.fluid[cell - 1] {
                    (cell - 1, -2isize)
                } else {
                    *out = State::new();
                    return;
                };
                let (_st, dd) = gather6(u, g, derived, anchor.0, anchor.1, Direction::Y);
                *out = diffusion::build_diffusion_from_derived(&dd);
            });
        }
    }

    // ------------------------------------------------------------------
    // Cell-centered right-hand side.
    // ------------------------------------------------------------------
    {
        let rhs = &mut s.rhs;
        let fx = &s.fx;
        let fy = &s.fy;
        let dfx = &s.dfx;
        let dfy = &s.dfy;
        let derived = &s.derived;

        rhs.par_iter_mut().enumerate().for_each(|(lin, out)| {
            if !u.fluid[lin] {
                *out = State::new();
                return;
            }

            let i = lin / ny;
            let j = lin % ny;

            let flux_l = fx[lin];
            let flux_r = fx[lin + ny];
            let flux_b = fy[j * nx + i];
            let flux_t = fy[(j + 1) * nx + i];

            let fx_term = state::update(flux_l, flux_r).scalar_prod(1.0 / dx);
            let fy_term = state::update(flux_b, flux_t).scalar_prod(1.0 / dy);

            let dx9 = gather9d(g, derived, lin, -4, Direction::X);
            let dy9 = gather9d(g, derived, lin, -4, Direction::Y);

            let nc_x = noncon::nonconservative_x_pre(&dx9, dx);
            let nc_y = noncon::nonconservative_y_pre(&dy9, dy);

            let source_term = if constant::SOURCE_ACTIVE {
                source::source(u.value[lin])
            } else {
                State::new()
            };

            let mut dif_term = State::new();
            if constant::DIFFUSION_ACTIVE {
                let dif_x = state::update(dfx[lin], dfx[lin + ny])
                    .scalar_prod(-1.0 / (dx * dx));
                let dif_y = state::update(dfy[j * nx + i], dfy[(j + 1) * nx + i])
                    .scalar_prod(-1.0 / (dy * dy));
                dif_term = dif_x.add(dif_y);
            }

            *out = fx_term
                .add(fy_term)
                .add(nc_x)
                .add(nc_y)
                .add(source_term)
                .add(dif_term);
        });
    }
}

#[inline]
fn stage_update_rhs(
    base: &Field,
    dst: &mut Field,
    rhs: &[State],
    coef: f64,
    label: &str,
) {
    let ny = base.grid.ny;
    dst.value.par_iter_mut().enumerate().for_each(|(l, o)| {
        if !base.fluid[l] {
            return;
        }
        let value = base.value[l].add(rhs[l].scalar_prod(coef));
        assert_admissible(value, ((l / ny) as isize, (l % ny) as isize), label);
        *o = value;
    });
}

#[inline]
fn stage_update_comb(
    base: &Field,
    add: &Field,
    dst: &mut Field,
    rhs: &[State],
    coef: f64,
    w_base: f64,
    w_add: f64,
    label: &str,
) {
    let ny = base.grid.ny;
    dst.value.par_iter_mut().enumerate().for_each(|(l, o)| {
        if !base.fluid[l] {
            return;
        }
        let value = base.value[l]
            .scalar_prod(w_base)
            .add(add.value[l].scalar_prod(w_add))
            .add(rhs[l].scalar_prod(coef));
        assert_admissible(value, ((l / ny) as isize, (l % ny) as isize), label);
        *o = value;
    });
}

fn rk3_ssp(
    u: &mut Field,
    ghosts: &mut GhostGrid,
    dt: f64,
    u1: &mut Field,
    u2: &mut Field,
    u3: &mut Field,
    s: &mut Scratch,
) {
    l(&*u, ghosts, s);
    stage_update_rhs(u, u1, &s.rhs, dt, "RK1 state");
    u1.time = u.time + dt;

    l(&*u1, ghosts, s);
    stage_update_comb(u, u1, u2, &s.rhs, dt / 4.0, 0.75, 0.25, "RK2 state");
    u2.time = u.time + 0.5 * dt;

    l(&*u2, ghosts, s);
    stage_update_comb(u, u2, u3, &s.rhs, 2.0 * dt / 3.0, 1.0 / 3.0, 2.0 / 3.0, "RK3 state");
    u3.time = u.time + dt;

    std::mem::swap(u, u3);
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
    // Fresh: cargo run --release
    // Restart from solution_0012.bin: cargo run --release -- 12
    let restart_id=std::env::args().nth(1).map(|s|s.parse::<usize>().expect("restart id must be integer"));
    if restart_id.is_none() {
        io::clear_data_folder();
    }

    let mut u = init::init_cylinder();

    let t_store_interval = 0.001_f64;

    // ---------------------------------------------------------
    // Load restart file first
    // ---------------------------------------------------------
    if let Some(id) = restart_id {
    let path = format!("data_new/solution_{:04}.bin", id);
    io::load_data(&mut u, &path);
    }

    let mut store_id = if restart_id.is_some() {
            (u.time / t_store_interval).round() as usize
        } else {
            0
    };

    let dx=u.grid.dx; let dy=u.grid.dy;
    let lx=dx*u.grid.nx as f64; let ly=dy*u.grid.ny as f64;
    let offsets=ghost::default_stencil_offsets();
    let mut ghosts=ghost::GhostGrid::build(&u,&offsets);
    ghosts.print_summary();

    let mut scratch = Scratch::new(&u);
    let mut u1 = u.empty_like();
    let mut u2 = u.empty_like();
    let mut u3 = u.empty_like();

    let mut t=u.time;
    let t_final=0.6_f64;
    let mut next_store_time=(store_id+1) as f64*t_store_interval;
    let mut n=0usize;

    if restart_id.is_none() {
        io::save_data(&u,"solution_0000.bin",lx,ly);
        println!("stored solution_0000.bin at t = {:.8e}",t);
    } else {
        println!("Restarting from id={}, t={:.8e}; next output t={:.8e}",store_id,t,next_store_time);
    }

    while t < t_final-1e-14 {
        let dt_cfl=calc_global_dt(&u);
        let mut dt=0.05*dt_cfl;
        if next_store_time<=t_final && t+dt>next_store_time { dt=next_store_time-t; }
        if t+dt>t_final { dt=t_final-t; }
        assert!(dt>0.0,"non-positive dt at t={}",t);

        rk3_ssp(&mut u,&mut ghosts,dt,&mut u1,&mut u2,&mut u3,&mut scratch);
        t=u.time; n+=1;
        println!("step={}, t={:.8e}, dt={:.8e}, dt_cfl={:.8e}",n,t,dt,dt_cfl);

        if next_store_time<=t_final && t>=next_store_time-1e-12 {
            store_id+=1;
            let filename=format!("solution_{:04}.bin",store_id);
            io::save_data(&u,&filename,lx,ly);
            println!("stored {} at t={:.8e}",filename,t);
            next_store_time=(store_id+1) as f64*t_store_interval;
        }
    }

    let last_regular=store_id as f64*t_store_interval;
    if (t-last_regular).abs()>1e-12 {
        store_id+=1;
        let filename=format!("solution_{:04}.bin",store_id);
        io::save_data(&u,&filename,lx,ly);
        println!("stored final {} at t={:.8e}",filename,t);
    }
    println!("Finished: t={:.8e}, restart-local steps={}, last id={}",t,n,store_id);
}

#[cfg(test)]
mod parity {
    use super::*;

    fn rng(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn state_close(a: &State, b: &State, tol: f64) {
        let d = [
            (a.rho - b.rho).abs(),
            (a.mom_x - b.mom_x).abs(),
            (a.mom_y - b.mom_y).abs(),
            (a.ee - b.ee).abs(),
            (a.ei - b.ei).abs(),
            (a.er - b.er).abs(),
        ];
        let maxd = d.into_iter().fold(0.0f64, f64::max);
        assert!(maxd < tol, "state mismatch {} >= {}", maxd, tol);
    }

    #[test]
    fn reconstruction_fast_matches_reconstruction() {
        let mut seed = 12345u64;
        for _trial in 0..100 {
            let mut pts = [State::new(); 6];
            for p in pts.iter_mut() {
                *p = State {
                    rho: 0.5 + rng(&mut seed),
                    mom_x: rng(&mut seed) - 0.5,
                    mom_y: rng(&mut seed) - 0.5,
                    ee: 1.0 + rng(&mut seed),
                    ei: 1.0 + rng(&mut seed),
                    er: 1.0 + rng(&mut seed),
                };
            }
            let mut d = [Derived::new(); 6];
            for i in 0..6 {
                d[i] = Derived::from_state(pts[i]);
            }

            for recon in [true, false] {
                let st = weno::Stencil6 { points: pts, dir: Direction::X };
                let a = st.reconstruction(recon);
                let b = weno::Stencil6::reconstruction_fast(&pts, &d, Direction::X, recon);
                state_close(&a, &b, 1e-12);

                let st = weno::Stencil6 { points: pts, dir: Direction::Y };
                let a = st.reconstruction(recon);
                let b = weno::Stencil6::reconstruction_fast(&pts, &d, Direction::Y, recon);
                state_close(&a, &b, 1e-12);
            }
        }
    }

    #[test]
    fn ghost_bc_fast_matches_slow() {
        let field = init::init_cylinder();
        let h = (field.grid.dx * field.grid.dy).sqrt();
        let beta = bc1::beta_quadratic_forms(h);

        for &idx in &[
            (-1isize, 100isize),
            (-1isize, 240isize),
            (-2isize, 240isize),
            (-3isize, 240isize),
            (0isize, 0isize),
            (2isize, 240isize),
            (3isize, 240isize),
            (1isize, 100isize),
        ] {
            let p = geometry::Point {
                x: field.grid.x(idx.0),
                y: field.grid.y(idx.1),
            };
            let poly = &field.outer_bound;
            let project = geometry::project(poly, p);
            let side = ghost::select_boundary_side(project.point, poly, &field.bc_outer);
            let pre = bc1::precompute_ghost_bc(&project, &field, &beta);

            let a = bc1::set_ghost_point_value(
                idx,
                project,
                ghost::BoundaryKind::Outer,
                side,
                &field,
                None,
            );
            let b = bc1::set_ghost_point_value(
                idx,
                project,
                ghost::BoundaryKind::Outer,
                side,
                &field,
                Some(&pre),
            );
            state_close(&a, &b, 1e-10);
        }
    }
}
