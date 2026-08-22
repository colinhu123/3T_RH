# 2D Three-Temperature Radiation Hydrodynamics WENO Solver

A Rust implementation of a two-dimensional, high-order finite-difference solver for the three-temperature radiation hydrodynamics (3-T RH) system on Cartesian grids with arbitrary polygon-embedded geometries.

The implementation follows the 2D algorithm described by Cheng and Shu in *High order conservative finite difference WENO scheme for three-temperature radiation hydrodynamics* (Journal of Computational Physics, 2024). The boundary treatment follows the high-order inverse Lax–Wendroff (ILW) / WENO extrapolation approach of Tan, Wang and Shu (*Accurate numerical boundary conditions for computational fluid dynamics*, 2012).

## What this code solves

The model contains six evolved quantities in two spatial dimensions:

- density `rho`
- x-momentum `mom_x`
- y-momentum `mom_y`
- electron energy variable `ee`
- ion energy variable `ei`
- radiation energy variable `er`

The paper writes the 2D system in the form

```text
U_t + dF1/dx + dF2/dy
    - (u/3) dN/dx - (v/3) dN/dy
  = dG1/dx + dG2/dy + S
```

where `F1` and `F2` are the conservative convection fluxes, `N` contains the non-conservative pressure combinations, `G1` and `G2` are the electron/ion/radiation diffusion fluxes, and `S` contains the electron-ion and electron-radiation energy exchange terms.

This is the same operator structure assembled in `l()` in `main.rs`: conservative x/y flux divergences, non-conservative x/y contributions, source terms, and diffusion in both directions are combined to form the semi-discrete right-hand side.

## Numerical method

The spatial discretization is a fifth-order finite-difference WENO scheme with local characteristic decomposition (Roe-averaged 6x6 eigen-decomposition per interface, `weno.rs`), following the 2D construction in Section 4 of the reference paper.

The main algorithm is:

1. Recompute all ghost-cell values for the current RK stage (`GhostGrid`).
2. Build x-direction WENO interface fluxes (characteristic space).
3. Build y-direction WENO interface fluxes.
4. Add the non-conservative x/y terms (6th-order central derivative + upwind jump terms, `noncon.rs`).
5. Add the source term (`source.rs`).
6. Add the diffusion contribution in x/y (`diffusion.rs`).
7. Advance the solution with third-order SSP Runge-Kutta time integration.

The paper explicitly notes that the 2D finite-difference method can reuse the 1D algorithm independently in each coordinate direction. This project follows that structure: the WENO reconstruction is called once for x-directed stencils and once for y-directed stencils.

## Time integration

Time advancement uses the three-stage, third-order SSP Runge-Kutta method from the paper:

```text
u1 = u^n + dt L(u^n)

u2 = 3/4 u^n + 1/4 u1 + dt/4 L(u1)

u^(n+1) = 1/3 u^n + 2/3 u2 + 2 dt/3 L(u2)
```

In the code this is implemented in `rk3_ssp()`.

The global time step is `dt = 0.05 * dt_cfl`, where `dt_cfl` is the minimum over all fluid cells of `dt::get_local_dt()` (which includes advection, diffusion and exchange-term eigenvalue estimates, scaled by `LAMBDA = 0.5`). The step is clipped so the simulation lands exactly on output times and `t_final`.

## Geometry and boundary conditions

The fluid domain is defined by polygons (`geometry.rs`):

- `outer_bound`: polygon with fluid inside (`FluidSide::Inside`). The cylinder surface is part of this polygon.
- `inner_bound`: polygon with fluid outside (`FluidSide::Outside`). Currently a dummy placed far outside the domain.

Every side of each polygon carries a `BCType`. On a Cartesian point that lies outside the fluid domain, `Field::is_in_domain` returns false and the point is treated as a ghost cell.

### Ghost grid

`ghost.rs` discovers, once at startup, every ghost index referenced by the solver stencils (a radius-4 cross, `default_stencil_offsets()`, covering WENO, non-conservative and diffusion stencils), and caches for each ghost:

- the closest boundary point `P0` and outward fluid normal,
- the boundary side (at polygon vertices the side is selected by BC priority: `Wall > ReflectiveWall > Constant/TimeDependent > FarField > Outflow > ZerothOrder > Periodic`),
- the signed normal distance from `P0` to the ghost point.

Ghost values are recomputed in parallel (Rayon) once per RK stage in `GhostGrid::update_values_parallel()`. All ghosts are first-stage-independent, so the update is data-parallel.

### Boundary conditions (`bc1.rs`)

| BCType | Description |
|---|---|
| `Wall` | High-order ILW: no-penetration constraint on the momentum row, characteristic WENO extrapolation for the other rows, 4th-order Taylor expansion to the ghost point. |
| `ReflectiveWall` | Geometric reflection of the nearest interior state with normal momentum flipped. |
| `FarField(state)` | Characteristic BC: outgoing characteristics from WENO extrapolation, incoming characteristics from the freestream state. Becomes supersonic inflow/outflow automatically. |
| `Outflow { p_inf, sigma, l_domain }` | LODI pressure relaxation for the incoming acoustic wave. |
| `Constant(state)` / `TimeDependent(f)` | Prescribed boundary state. |
| `ZerothOrder` | Ghost value copied from the mirrored interior cell. |
| `Periodic` | y-periodic wrap (used by the translating-shock test). |

The high-order machinery (`weno_extrapolation()`) follows Tan et al. (2012), Sec. 2.4: for each order r = 0..4 a 2D polynomial of degree r is least-squares fitted to the (r+1)^2-point stencil `E_r` of characteristic variables in boundary-normal coordinates; smoothness indicators and nonlinear weights select the WENO combination of the k-th normal derivatives, which are Taylor-expanded to the ghost point.

## Current test problem in `main.rs`

The current executable is the **Mach-3 flow past the front half of a circular cylinder** (`init::init_cylinder()`):

```text
domain      : x in [-3, 0], y in [-6, 6]
obstacle    : half-disk x^2 + y^2 < 1, x <= 0  (part of the outer polygon)
grid        : nx = 121, ny = 481, dx = dy = 1/40
freestream  : rho = 1, p = 1, M = 3  (splits: ee = ei = er)
t_final     : 0.4
output      : every dt_store = 0.001
```

The cylinder arc is approximated by 360 polygon segments with `ReflectiveWall`; all outer rectangular sides use `FarField` (the left side `x = -3` is supersonic inflow).

`init::init_double_mach()` is also available (the paper's double-Mach-reflection benchmark with a polygonal domain and a moving shock).

## Project layout

```text
.
├── main.rs          grid/driver setup, spatial operator l(), SSP-RK3, time loop, output
├── state.rs         State representation, pressure splits, physical fluxes, arithmetic
├── weno.rs          WENO stencil, Roe-average eigen-decomposition, characteristic WENO5 reconstruction
├── noncon.rs        non-conservative pressure term (6th-order derivative + upwind jumps)
├── diffusion.rs     diffusion flux (6-point derivative of temperature)
├── source.rs        electron-ion / electron-radiation energy exchange
├── dt.rs            local time-step estimate
├── constant.rs      physical and numerical constants
├── geometry.rs      points, vectors, projections, polygons, normals
├── field1.rs        GridInfo + Field with polygon-defined fluid region
├── ghost.rs         GhostGrid: static ghost layout, parallel per-stage updates
├── bc1.rs           boundary conditions (ILW wall, WENO extrapolation, far-field, LODI, ...)
├── init.rs          initial conditions (cylinder, double Mach reflection)
└── io.rs            binary output/restart writer and reader
```

`bc.rs` and `field.rs` are legacy rectangular-grid variants that are no longer wired into the build (`main.rs` declares `bc1`/`field1` instead).

## Build and run

```bash
cargo build --release
```

Fresh run (clears `data_new/`):

```bash
cargo run --release
```

Restart from an existing snapshot, e.g. `data_new/solution_0012.bin`:

```bash
cargo run --release -- 12
```

The run prints the grid summary, per-step time/time-step info, and output-file writes.

## Tests

```bash
cargo test
```

Test coverage includes:

- WENO: eigen-decomposition (`L * R = I` for x/y), characteristic round-trip, constant-state flux preservation, WENO5 convergence order on a smooth profile.
- noncon: constant-state and zero-velocity vanishing, direction sensitivity, wrapper consistency.
- bc1: polynomial least-squares reproduction, derivative extraction, paper stencil `E_r` cardinality/structure on vertical and horizontal walls, constant-state characteristic extrapolation and final ghost reconstruction.
- ghost: stencil-offset bookkeeping.
- state: primitive-to-conservative conversion.

## Output

Binary files `data_new/solution_NNNN.bin` (little endian):

```text
header:
  [8]u8  magic = "RH3TBIN1"
  u32    version = 1
  u32    nvar = 8
  u64    nx
  u64    ny
  f64    time

payload, j-major (x fastest):
  repeated nx*ny times: f64 x, y, rho, mom_x, mom_y, ee, ei, er
```

Points outside the fluid polygon are stored as NaN so post-processing can mask them. Files are written to a `.tmp` name and atomically renamed, so a live visualizer never sees a partial file.

### Visualization

Interactive density viewer (arrow keys step through frames, `q` quits; safe to run while the solver is still writing):

```bash
python visualize_sol.py
```

Additional scripts in `py_utils/`:

- `contour_gen.py` — contour plots of a single snapshot (edit the gamma values to match `constant.rs`).
- `energy_split.py` — history of the maximum electron/ion/radiation energy splits across snapshots; writes `energy_split_history.csv`.
- `conservation_check.py` — legacy text-format (`*.dat`) conservation check.

## Current physical parameters (`constant.rs`)

```text
KAPPA_E = KAPPA_I = KAPPA_R = 0   (no diffusion)
OMEGA_EI = OMEGA_ER = 0           (no energy exchange)
CVE = CVI = 1, A = 1
GAMMA_E = GAMMA_I = GAMMA_R = 1.4
LAMBDA = 0.5, WENO_Q = 2.0
```

With these settings the code reduces to the 3-T Euler equations; diffusion and exchange terms are in place but inactive.

## Accuracy and verification

The reference paper reports fifth-order spatial accuracy and third-order SSP Runge-Kutta time discretization for its finite-difference WENO construction.

Recommended verification workflow:

1. Verify constant-state preservation (`cargo test` covers this at the component level).
2. Reproduce the paper's 2D manufactured-solution test to measure `L1`/`Linf` convergence.
3. Check conservation of mass, momentum, and total energy for compatible boundary conditions.
4. Test the discontinuous shock-tube configuration.
5. Enable diffusion and energy exchange only after the non-diffusive case is verified.
6. Compare density and the three temperatures against a reference solution.

## Reference

> J. Cheng and C.-W. Shu, "High order conservative finite difference WENO scheme for three-temperature radiation hydrodynamics," *Journal of Computational Physics* 517 (2024), 113304.

The boundary treatment follows:

> S. Tan, C. Wang, C.-W. Shu, "Accurate numerical boundary conditions for compressible flow problems," *Journal of Computational Physics* 231 (2012), 2510–2527.
