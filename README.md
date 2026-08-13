# 2D Three-Temperature Radiation Hydrodynamics WENO Solver

A Rust implementation of a two-dimensional, high-order finite-difference solver for the three-temperature radiation hydrodynamics (3-T RH) system.

The implementation follows the 2D algorithm described by Cheng and Shu in *High order conservative finite difference WENO scheme for three-temperature radiation hydrodynamics* (Journal of Computational Physics, 2024). In particular, the solver follows the paper's direction-by-direction construction: the 2D spatial operator is assembled from 1D WENO flux evaluations in the x- and y-directions, together with the non-conservative terms, diffusion, and energy-exchange source terms.

## What this code solves

The model contains six evolved quantities in two spatial dimensions:

- density
- x-momentum
- y-momentum
- electron energy variable
- ion energy variable
- radiation energy variable

The paper writes the 2D system in the form

```text
U_t + dF1/dx + dF2/dy
    - (u/3) dN/dx - (v/3) dN/dy
  = dG1/dx + dG2/dy + S
```

where `F1` and `F2` are the conservative convection fluxes, `N` contains the non-conservative pressure combinations, `G1` and `G2` are the electron/ion/radiation diffusion fluxes, and `S` contains the electron-ion and electron-radiation energy exchange terms.

This is the same operator structure assembled in `main.rs`: conservative x/y flux divergences, non-conservative x/y contributions, source terms, and diffusion in both directions are combined to form the semi-discrete right-hand side.

## Numerical method

The spatial discretization is based on a fifth-order WENO finite-difference formulation, following the 2D construction in Section 4 of the reference paper.

The main algorithm is:

1. Build x-direction WENO interface fluxes.
2. Build x-direction diffusion fluxes.
3. Build y-direction WENO interface fluxes.
4. Build y-direction diffusion fluxes.
5. Compute the cell-centered conservative flux divergence.
6. Add the non-conservative x/y terms.
7. Add the source term.
8. Add the diffusion contribution in x/y.
9. Advance the solution with third-order SSP Runge-Kutta time integration.

The paper explicitly notes that the 2D finite-difference method can reuse the 1D algorithm independently in each coordinate direction. This project follows that structure: the WENO reconstruction is called once for x-directed stencils and once for y-directed stencils.

## Time integration

Time advancement uses the three-stage, third-order SSP Runge-Kutta method from the paper:

```text
u1 = u^n + dt L(u^n)

u2 = 3/4 u^n + 1/4 u1 + dt/4 L(u1)

u^(n+1) = 1/3 u^n + 2/3 u2 + 2 dt/3 L(u2)
```

In the code this is implemented in `rk3_ssp()`.

## Spatial operator in `main.rs`

The main file contains the top-level numerical workflow.

### Stencil construction

Three stencil extractors are provided:

- `weno_stencil_extractor()` builds the six-point stencil passed to `weno::Stencil6`.
- `noncon_stencil_extractor()` builds the nine-point stencil used by the non-conservative discretization.
- `diffusion_stencil_extractor()` builds the six-point stencil used for diffusion.

The x/y indexing uses modulo arithmetic, so the currently implemented stencil handling is periodic in both coordinate directions.

### Semi-discrete operator

The function `l(u, dx, dy)` assembles the complete spatial operator. Its structure mirrors the paper's 2D semi-discrete equation:

```text
L(u) = conservative flux divergence
     + non-conservative x/y terms
     + source
     + diffusion x/y terms
```

The conservative contribution is formed from differences of neighboring interface fluxes. The diffusion contribution is likewise obtained from neighboring diffusion fluxes.

### Parallel execution

The code uses Rayon (`rayon::prelude::*`) to parallelize grid loops. The x/y interface flux calculations and cell-wise RHS construction use parallel iterators, which is useful for the large uniform grids typically used by high-order finite-difference solvers.

## Current test problem in `main.rs`

The current executable is configured as a 2D double-wave / radiation shock-tube style initial condition based on the constant states used in the reference problem family.

The grid and physical domain are currently hard-coded as:

```text
Nx = 400
Ny = 100
Lx = 40
Ly = 10
Tfinal = 1
```

so that

```text
dx = Lx / Nx
dy = Ly / Ny
```

The initial state is piecewise constant in x:

```text
left quarter:    State 1
middle region:   State 2
right quarter:   State 1
```

with the states currently defined in `init()` as

```text
State 1:
    rho   = 0.445
    mom_x = 0.31061
    mom_y = 0.0
    ee    = 1.8
    ei    = 1.8
    er    = 3.564

State 2:
    rho   = 0.5
    mom_x = 0.0
    mom_y = 0.0
    ee    = 0.285
    ei    = 0.285
    er    = 0.571
```

The main loop advances the solution until `t = 1.0` and writes the initial state plus every subsequent time step to the data directory through the I/O module.

## Project layout

The uploaded `main.rs` expects the following Rust modules:

```text
.
├── main.rs
├── state.rs
├── weno.rs
├── dt.rs
├── noncon.rs
├── source.rs
├── diffusion.rs
├── constant.rs
└── io.rs
```

The responsibilities are approximately:

| Module | Role |
|---|---|
| `main.rs` | Grid setup, spatial operator, SSP-RK3 stepping, global time loop, output |
| `state.rs` | State representation and state arithmetic |
| `weno.rs` | WENO stencil representation and interface reconstruction |
| `noncon.rs` | Non-conservative term discretization |
| `diffusion.rs` | Diffusion stencil and diffusion flux construction |
| `dt.rs` | Local time-step estimate |
| `source.rs` | Energy-exchange / additional source contribution |
| `constant.rs` | Physical/numerical constants |
| `io.rs` | Output-file management and solution writing |

The exact formulas and implementation details of these components are contained in their respective modules; `main.rs` acts as the orchestration layer.

## Build and run

This project is intended to be built as a normal Rust/Cargo application.

```bash
cargo build --release
```

Run the solver with:

```bash
cargo run --release
```

The program clears the previous output directory, initializes the solution, writes `solution_0000.dat`, advances the solution, and then writes files named like:

```text
solution_0001.dat
solution_0002.dat
...
```

The run also prints the current step, simulation time, and time step to the terminal.

## Output

The output is generated through `io::save_data()`. The exact file format is therefore controlled by `io.rs` rather than `main.rs`.

A typical output sequence is:

```text
solution_0000.dat
solution_0001.dat
solution_0002.dat
...
```

These files contain the numerical solution at successive time levels and can be post-processed or visualized with an external analysis/plotting script.

## Relation to the reference 2D algorithm

The reference paper's 2D method is built from a one-dimensional high-order finite-difference WENO algorithm applied independently along each coordinate direction. The paper's Algorithm 4.1 computes the x-direction interface fluxes by applying the 1D procedure along each row, and the y-direction fluxes by applying the same procedure along each column.

The project follows the same high-level decomposition:

```text
                    2D problem
                       |
          +------------+------------+
          |                         |
       x direction              y direction
          |                         |
   WENO interface flux      WENO interface flux
          |                         |
   diffusion contribution    diffusion contribution
          +------------+------------+
                       |
              cell-centered RHS
                       |
                  SSP-RK3 step
                       |
                 updated state
```

The implementation also keeps the separate non-conservative, diffusion, and source contributions rather than folding them into a single conservative flux.

## Important implementation details

### Periodic indexing in the current driver

The stencil extractors use modulo indexing in both x and y. This is an explicit implementation choice in `main.rs` and means the current driver is set up around periodic stencil access.

Changing to reflective, Dirichlet, outflow, or mixed boundaries will require boundary treatment outside the current modulo-based extraction logic.

### Adaptive global time step

`calc_global_dt()` scans all cells and takes the minimum local time step returned by `dt::get_local_dt()`.

The main loop then clips the final time step so that the simulation does not advance beyond `t_final`.

### Parallelism

Rayon is used for grid-level parallelism. The current implementation therefore targets a CPU-parallel workflow rather than GPU execution.

## Accuracy and verification

The reference paper reports fifth-order spatial accuracy and third-order SSP Runge-Kutta time discretization for its finite-difference WENO construction. In the paper's 2D manufactured-solution test, the reported scheme reaches approximately fifth-order convergence in both `L1` and `Linf` norms for the directly evolved variables when the grid is refined.

For this project, a good next verification step is to reproduce the paper's 2D manufactured-solution test before relying on shock/interface problems as validation. This separates implementation errors in order-of-accuracy-sensitive pieces from physical-model behavior near discontinuities.

## Recommended verification workflow

1. Verify constant-state preservation.
2. Verify the 2D manufactured solution and measure `L1`/`Linf` convergence.
3. Check conservation of mass, momentum, and total energy for periodic or otherwise compatible boundary conditions.
4. Test the discontinuous shock-tube configuration.
5. Add diffusion and energy exchange only after the non-diffusive case is verified.
6. Compare density and the three temperatures against a reference solution.

## Limitations of the current driver

The README describes the implementation visible in the uploaded `main.rs`; it does not assume capabilities that are not shown there.

In particular, the current driver has:

- a fixed `400 x 100` grid;
- a fixed domain `40 x 10`;
- a fixed final time `1.0`;
- a hard-coded initial condition;
- modulo-based periodic stencil access;
- no command-line configuration layer visible in `main.rs`;
- no AMR or embedded-boundary machinery in the uploaded driver.

These can be refactored later into configuration files, command-line parameters, or dedicated problem-definition modules.

## Reference

The main numerical reference for this implementation is:

> J. Cheng and C.-W. Shu, “High order conservative finite difference WENO scheme for three-temperature radiation hydrodynamics,” *Journal of Computational Physics* 517 (2024), 113304.

The paper describes the 3-T radiation-hydrodynamics equations, the conservative reformulation using three new energy variables, the fifth-order finite-difference WENO discretization, the direction-by-direction 2D extension, and the third-order SSP Runge-Kutta time discretization.

## Source mapping

For readers comparing the code directly with the paper:

- `l()` corresponds to the semi-discrete spatial operator.
- `weno_stencil_extractor()` supplies the directional stencil used by the WENO reconstruction.
- `noncon_stencil_extractor()` supplies the stencil for the non-conservative contribution.
- `diffusion_stencil_extractor()` supplies the diffusion stencil.
- `rk3_ssp()` implements the three-stage SSP-RK3 update.
- `calc_global_dt()` computes the global step from the per-cell time-step estimate.
- `init()` defines the current numerical experiment.
- `main()` controls initialization, output, time stepping, and data writing.

The uploaded main file shows these components explicitly, including the x/y flux evaluation, the assembled RHS, the SSP-RK3 stages, the `400 x 100` initialization, and the output sequence. 

## Citation

If this solver is used in academic work, cite the Cheng and Shu paper above and describe any modifications made to the original algorithm.
