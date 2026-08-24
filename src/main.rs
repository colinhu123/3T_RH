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
    let path = format!("data/solution_{:04}.bin", id);
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
    let t_final=1.0_f64;
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
        let mut dt=0.1*dt_cfl;
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

            // Same analytic BoundaryElement lookup as the real solver:
            // nearest element wins, BC priority breaks junction ties.
            let (boundary_id, project) =
                bc1::find_boundary_element(p, &field.outer_boundary);

            let q = match &field.outer_boundary[boundary_id].bc {
                bc1::BCType::PrimitiveWall => bc1::PRIMITIVE_WALL_WENO_Q,
                _ => constant::WENO_Q,
            };
            let pre = bc1::precompute_ghost_bc(&project, &field, &beta, q);

            let a = bc1::set_ghost_point_value(
                idx,
                project,
                ghost::BoundaryKind::Outer,
                boundary_id,
                &field,
                None,
            );
            let b = bc1::set_ghost_point_value(
                idx,
                project,
                ghost::BoundaryKind::Outer,
                boundary_id,
                &field,
                Some(&pre),
            );
            state_close(&a, &b, 1e-10);
        }
    }

    #[test]
    fn cylinder_has_one_physical_arc() {
        let u = init::init_cylinder();

        // Six physical elements: 5 FarField lines + 1 cylinder arc.
        assert_eq!(u.outer_boundary.len(), 6, "cylinder must have 6 physical elements");

        let arcs: Vec<_> = u
            .outer_boundary
            .iter()
            .filter(|e| matches!(e.geometry, geometry::BoundaryGeometry::Arc(_)))
            .collect();
        assert_eq!(arcs.len(), 1, "cylinder must have exactly ONE physical arc");

        assert!(
            matches!(&arcs[0].bc, bc1::BCType::PrimitiveWall),
            "cylinder arc must use the PrimitiveWall BC"
        );

        // The 360 Polygon arc segments remain only as domain-classifier
        // detail: the physical boundary must NOT scale with them.
        assert_eq!(u.bc_outer.len(), 365);
    }

    #[test]
    fn cylinder_physical_geometry_ignores_polygon_segments() {
        let field = init::init_cylinder();

        // Stagnation-line ghost coordinate, projected through the analytic
        // boundary lookup. The result must be the exact analytic arc
        // projection, independent of any of the 360 Polygon segments.
        let p = geometry::Point { x: -1.2, y: 0.0 };
        let (id, project) = bc1::find_boundary_element(p, &field.outer_boundary);

        let arc = match &field.outer_boundary[id].geometry {
            geometry::BoundaryGeometry::Arc(a) => a,
            _ => panic!("stagnation point must resolve to the analytic arc"),
        };

        let direct = arc.project(p);

        assert!((project.point.x - direct.point.x).abs() < 1e-14);
        assert!((project.point.y - direct.point.y).abs() < 1e-14);
        assert!((project.normal.x - direct.normal.x).abs() < 1e-14);
        assert!((project.normal.y - direct.normal.y).abs() < 1e-14);
        assert!((project.distance - direct.distance).abs() < 1e-14);
    }

    #[test]
    fn ghostgrid_uses_analytic_arc_projection() {
        let field = init::init_cylinder();
        let offsets = ghost::default_stencil_offsets();
        let ghosts = ghost::GhostGrid::build(&field, &offsets);

        // The ONE analytic arc element.
        let arc = field
            .outer_boundary
            .iter()
            .find_map(|e| match &e.geometry {
                geometry::BoundaryGeometry::Arc(a) => Some(a),
                _ => None,
            })
            .expect("cylinder must have one analytic arc");

        let cx = arc.center.x;
        let cy = arc.center.y;

        // 1. EVERY cylinder-arc ghost must have its cached Projection
        //    equal (to roundoff) the direct analytic arc projection of
        //    its Cartesian coordinate. This proves GhostGrid::build
        //    resolved physical geometry through the analytic arc, with
        //    zero Polygon-segment contamination.
        let mut arc_ghosts = 0usize;
        for g in &ghosts.info {
            let is_arc = matches!(
                field.outer_boundary[g.boundary_id].geometry,
                geometry::BoundaryGeometry::Arc(_)
            );
            if !is_arc {
                continue;
            }
            arc_ghosts += 1;

            let p = geometry::Point {
                x: field.grid.x(g.idx.0),
                y: field.grid.y(g.idx.1),
            };
            let direct = arc.project(p);

            assert!((g.project.point.x - direct.point.x).abs() < 1e-12);
            assert!((g.project.point.y - direct.point.y).abs() < 1e-12);
            assert!((g.project.normal.x - direct.normal.x).abs() < 1e-12);
            assert!((g.project.normal.y - direct.normal.y).abs() < 1e-12);
            assert!((g.project.distance - direct.distance).abs() < 1e-12);
        }
        assert!(arc_ghosts > 0, "no cylinder-arc ghosts found");

        // 2. Stagnation-line query: exact analytic P0 and normal.
        let p = geometry::Point {
            x: cx - arc.radius - 0.2,
            y: cy,
        };
        let (id, proj) = bc1::find_boundary_element(p, &field.outer_boundary);
        assert!(matches!(
            field.outer_boundary[id].geometry,
            geometry::BoundaryGeometry::Arc(_)
        ));
        assert!((proj.point.x - (cx - arc.radius)).abs() < 1e-12);
        assert!((proj.point.y - cy).abs() < 1e-12);
        assert!((proj.normal.x - 1.0).abs() < 1e-12);
        assert!(proj.normal.y.abs() < 1e-12);

        // 3. Upper/lower mirror symmetry about the arc center line.
        let (_, up) = bc1::find_boundary_element(
            geometry::Point {
                x: cx - arc.radius - 0.1,
                y: cy + 0.2,
            },
            &field.outer_boundary,
        );
        let (_, dn) = bc1::find_boundary_element(
            geometry::Point {
                x: cx - arc.radius - 0.1,
                y: cy - 0.2,
            },
            &field.outer_boundary,
        );

        assert!((up.point.x - dn.point.x).abs() < 1e-12);
        assert!((up.point.y + dn.point.y - 2.0 * cy).abs() < 1e-12);
        assert!((up.normal.x - dn.normal.x).abs() < 1e-12);
        assert!((up.normal.y + dn.normal.y).abs() < 1e-12);
        assert!((up.distance - dn.distance).abs() < 1e-12);
    }

    // ============================================================
    // PrimitiveWall (order zero): Phase-D validation.
    //
    // The conservative Wall produces non-physical states for the
    // Mach-3 cylinder at t=0; the primitive order-zero wall must not.
    // ============================================================

    #[test]
    fn primitive_wall_initial_ghosts_admissible() {
        let field = init::init_cylinder();
        let offsets = ghost::default_stencil_offsets();
        let mut ghosts = ghost::GhostGrid::build(&field, &offsets);

        ghosts.update_values_parallel(&field);

        let mut count = 0usize;
        let mut min_rho = f64::INFINITY;
        let mut max_rho = f64::NEG_INFINITY;
        let mut min_p = f64::INFINITY;
        let mut max_p = f64::NEG_INFINITY;

        for (i, g) in ghosts.info.iter().enumerate() {
            if !matches!(
                &field.outer_boundary[g.boundary_id].bc,
                bc1::BCType::PrimitiveWall
            ) {
                continue;
            }
            count += 1;

            let s = ghosts.values[i];
            assert!(
                s.rho.is_finite()
                    && s.mom_x.is_finite()
                    && s.mom_y.is_finite()
                    && s.ee.is_finite()
                    && s.ei.is_finite()
                    && s.er.is_finite(),
                "non-finite PrimitiveWall ghost at {:?}",
                g.idx
            );
            assert!(s.rho > 0.0, "rho <= 0 at {:?}", g.idx);

            let d = Derived::from_state(s);
            let p = d.pe + d.pi + d.pr;
            assert!(p > 0.0, "p <= 0 at {:?}", g.idx);
            assert!(d.e_e > 0.0, "e_e <= 0 at {:?}", g.idx);
            assert!(d.e_i > 0.0, "e_i <= 0 at {:?}", g.idx);
            assert!(d.e_r > 0.0, "e_r <= 0 at {:?}", g.idx);

            min_rho = min_rho.min(s.rho);
            max_rho = max_rho.max(s.rho);
            min_p = min_p.min(p);
            max_p = max_p.max(p);
        }

        assert!(count > 0, "no PrimitiveWall ghosts found");
        println!(
            "PrimitiveWall ghosts: count={}, rho=[{:.6e},{:.6e}], p=[{:.6e},{:.6e}]",
            count, min_rho, max_rho, min_p, max_p
        );
    }

    #[test]
    fn primitive_wall_first_rk1_stays_admissible() {
        let field = init::init_cylinder();
        let offsets = ghost::default_stencil_offsets();
        let mut ghosts = ghost::GhostGrid::build(&field, &offsets);

        let mut scratch = Scratch::new(&field);
        let mut u1 = field.empty_like();

        let dt_cfl = calc_global_dt(&field);
        let dt = 0.05 * dt_cfl;
        assert!(dt > 0.0);

        // Initial ghost update + first spatial RHS + first RK1 update.
        // assert_admissible() inside stage_update_rhs panics on any
        // non-physical state, so reaching the end of this test means the
        // first RK1 step produced a fully admissible state.
        l(&field, &mut ghosts, &mut scratch);
        stage_update_rhs(&field, &mut u1, &scratch.rhs, dt, "test RK1 state");
    }
}
