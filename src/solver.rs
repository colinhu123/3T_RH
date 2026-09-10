//! Semi-discrete spatial operator, RK3-SSP time integration, and the
//! per-run buffers that drive a simulation.
//!
//! This module is pure numerics + time stepping; it knows nothing about
//! CLI parsing, file output, or the monitor. The `main` driver builds a
//! [`Solver`] and calls [`Solver::step`] inside the time loop.

use rayon::prelude::*;

use crate::constant;
use crate::diffusion;
use crate::dt;
use crate::field1::Field;
use crate::ghost::{self, GhostGrid};
use crate::noncon;
use crate::source;
use crate::state::{self, Derived, Direction, State};
use crate::weno;

use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// State admissibility validation.
//
// The RK stage updates reject any non-physical intermediate (non-finite
// state, rho <= 0, or any non-positive internal energy). The first bad
// state is reported once with full diagnostics; any bad state aborts.
// ============================================================================

static REPORTED_BAD_STATE: AtomicBool = AtomicBool::new(false);

#[inline(always)]
fn state_is_finite(s: State) -> bool {
    s.rho.is_finite()
        && s.mom_x.is_finite()
        && s.mom_y.is_finite()
        && s.ee.is_finite()
        && s.ei.is_finite()
        && s.er.is_finite()
}

#[inline(always)]
fn internal_energies(s: State) -> Option<(f64, f64, f64)> {
    if !state_is_finite(s) || s.rho <= 0.0 {
        return None;
    }
    let ux = s.mom_x / s.rho;
    let uy = s.mom_y / s.rho;
    let k = (ux * ux + uy * uy) / 6.0;
    Some((s.ee / s.rho - k, s.ei / s.rho - k, s.er / s.rho - k))
}

#[inline(always)]
fn assert_admissible(s: State, idx: (isize, isize), where_: &str) {
    let e = internal_energies(s);
    let bad = !state_is_finite(s)
        || s.rho <= 0.0
        || e.map_or(true, |q| q.0 <= 0.0 || q.1 <= 0.0 || q.2 <= 0.0);
    if bad {
        if !REPORTED_BAD_STATE.swap(true, Ordering::SeqCst) {
            eprintln!(
                "\nFIRST NON-PHYSICAL STATE\nwhere = {}\nidx = {:?}\nstate = {:?}\ninternal energies = {:?}\n",
                where_, idx, s, e
            );
        }
        panic!("non-physical state at {:?} in {}", idx, where_);
    }
}

// ============================================================================
// Per-step scratch buffers, allocated once per run.
//
// fx[i][j] : x-interface flux at i-1/2          (i in 0..=nx)
// fy[j][i] : y-interface flux at j-1/2          (j in 0..=ny)
// dfx/dfy  : diffusion interface fluxes (same layout)
// rhs      : semi-discrete operator output
// derived  : per-stage derived quantities for fluid cells
// ============================================================================

pub(crate) struct Scratch {
    fx: Vec<State>,
    fy: Vec<State>,
    dfx: Vec<State>,
    dfy: Vec<State>,
    /// Semi-discrete operator output; read by the crate-internal parity
    /// tests to drive a single explicit RK1 stage.
    pub(crate) rhs: Vec<State>,
    derived: Vec<Derived>,
    /// Copy of the semi-discrete RHS dU/dt at the start state of the most
    /// recent RK3-SSP step (the stage-1 RHS). Used ONLY by the residual
    /// monitor; it never feeds back into the time integration.
    residual: Vec<State>,
}

impl Scratch {
    pub(crate) fn new(u: &Field) -> Self {
        let nx = u.grid.nx;
        let ny = u.grid.ny;
        Self {
            fx: vec![State::new(); (nx + 1) * ny],
            fy: vec![State::new(); nx * (ny + 1)],
            dfx: vec![State::new(); (nx + 1) * ny],
            dfy: vec![State::new(); nx * (ny + 1)],
            rhs: vec![State::new(); nx * ny],
            derived: vec![Derived::new(); nx * ny],
            residual: vec![State::new(); nx * ny],
        }
    }
}

// ============================================================================
// Stencil gathers from the fluid field + cached ghost grid.
// ============================================================================

#[inline(always)]
fn v_at(u: &Field, g: &GhostGrid, t: u32) -> State {
    let t = t as usize;
    if t < u.grid.len() {
        u.value[t]
    } else {
        g.values[t - u.grid.len()]
    }
}

#[inline(always)]
fn d_at(g: &GhostGrid, derived: &[Derived], t: u32) -> Derived {
    let t = t as usize;
    if t < derived.len() {
        derived[t]
    } else {
        g.derived[t - derived.len()]
    }
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

// ============================================================================
// Semi-discrete spatial operator.
//
// Compute the interface fluxes (WENO + diffusion) once per interface, and
// then assemble the cell-centered semi-discrete operator from the stored
// interface values plus the non-conservative / source terms.
// ============================================================================

pub(crate) fn l(u: &Field, ghosts: &mut GhostGrid, s: &mut Scratch) {
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

            let x = u.grid.x(i as isize);
            let y = u.grid.y(j as isize);

            let mms_term = source::wall_mms_source(x, y, u.time);

            let mut dif_term = State::new();
            if constant::DIFFUSION_ACTIVE {
                let dif_x = state::update(dfx[lin], dfx[lin + ny]).scalar_prod(-1.0 / (dx * dx));
                let dif_y = state::update(dfy[j * nx + i], dfy[(j + 1) * nx + i])
                    .scalar_prod(-1.0 / (dy * dy));
                dif_term = dif_x.add(dif_y);
            }

            *out = fx_term
                .add(fy_term)
                .add(nc_x)
                .add(nc_y)
                .add(source_term)
                .add(dif_term)
                .add(mms_term);
        });
    }
}

// ============================================================================
// Explicit SSP-RK3 stage updates.
// ============================================================================

#[inline]
pub(crate) fn stage_update_rhs(base: &Field, dst: &mut Field, rhs: &[State], coef: f64, label: &str) {
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
pub(crate) fn stage_update_comb(
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

pub(crate) fn rk3_ssp(
    u: &mut Field,
    ghosts: &mut GhostGrid,
    dt: f64,
    u1: &mut Field,
    u2: &mut Field,
    u3: &mut Field,
    s: &mut Scratch,
) {
    l(&*u, ghosts, s);

    // Monitoring hook (no influence on the integration): the first l() call
    // of an RK3-SSP step evaluates the semi-discrete operator exactly at the
    // step-start state, i.e. RHS(U^n) = dU/dt(U^n). Keep a copy so the
    // residual monitor can report the true RHS residual without a full extra
    // operator evaluation. Subsequent l() calls overwrite s.rhs, hence the
    // copy here.
    s.residual.copy_from_slice(&s.rhs);

    stage_update_rhs(u, u1, &s.rhs, dt, "RK1 state");
    u1.time = u.time + dt;

    l(&*u1, ghosts, s);
    stage_update_comb(u, u1, u2, &s.rhs, dt / 4.0, 0.75, 0.25, "RK2 state");
    u2.time = u.time + 0.5 * dt;

    l(&*u2, ghosts, s);
    stage_update_comb(
        u,
        u2,
        u3,
        &s.rhs,
        2.0 * dt / 3.0,
        1.0 / 3.0,
        2.0 / 3.0,
        "RK3 state",
    );
    u3.time = u.time + dt;

    std::mem::swap(u, u3);
}

pub(crate) fn calc_global_dt(u: &Field) -> f64 {
    let nx = u.grid.nx;
    let ny = u.grid.ny;

    let dx = u.grid.dx;
    let dy = u.grid.dy;

    let mut global_dt = f64::INFINITY;

    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);
            if !u.is_in_domain(idx) {
                continue;
            }
            let state = u.get(idx);

            let local_dt = dt::get_local_dt(state, dx, dy);

            global_dt = global_dt.min(local_dt);
        }
    }

    global_dt
}

// ============================================================================
// Solver: per-run buffers (ghost layout, scratch, RK3 stage fields) plus the
// high-level step / query API used by the driver in `main`.
// ============================================================================

pub struct Solver {
    ghosts: GhostGrid,
    scratch: Scratch,
    u1: Field,
    u2: Field,
    u3: Field,
}

impl Solver {
    pub fn new(u: &Field) -> Self {
        let offsets = ghost::default_stencil_offsets();
        Self {
            ghosts: GhostGrid::build(u, &offsets),
            scratch: Scratch::new(u),
            u1: u.empty_like(),
            u2: u.empty_like(),
            u3: u.empty_like(),
        }
    }

    pub fn ghosts(&self) -> &GhostGrid {
        &self.ghosts
    }

    /// Min over all fluid cells of the local CFL-limited time step.
    pub fn global_dt(&self, u: &Field) -> f64 {
        calc_global_dt(u)
    }

    /// Advance `u` by one SSP-RK3 substep of size `dt`.
    ///
    /// After the step `u` holds U^{n+1}; [`Solver::previous_state`] returns
    /// the pre-step field U^n and [`Solver::residual`] the stage-1 RHS
    /// dU/dt(U^n) (monitoring only).
    pub fn step(&mut self, u: &mut Field, dt: f64) {
        rk3_ssp(
            u,
            &mut self.ghosts,
            dt,
            &mut self.u1,
            &mut self.u2,
            &mut self.u3,
            &mut self.scratch,
        );
    }

    /// Pre-step field U^n left by the final mem::swap of the last RK3 step.
    pub fn previous_state(&self) -> &Field {
        &self.u3
    }

    /// Semi-discrete RHS dU/dt at the most recent step-start state.
    pub fn residual(&self) -> &[State] {
        &self.scratch.residual
    }
}

// ============================================================================
// Run configuration.
// ============================================================================

pub struct Config {
    pub t_final: f64,
    pub dt_factor: f64,
    pub store_interval: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            t_final: 10.0,
            dt_factor: 0.8,
            store_interval: 0.05,
        }
    }
}
