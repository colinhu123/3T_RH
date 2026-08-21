use crate::state::*;
use crate::geometry::*;
use crate::field1::*;
use crate::bc1::*;
use std::sync::Arc;

fn euler_to_three_energy(
    rho: f64,
    ux: f64,
    uy: f64,
    p: f64,
    gamma: f64,
) -> State {
    let kinetic = 0.5 * rho * (ux * ux + uy * uy);
    let e_total = p / (gamma - 1.0) + kinetic;
    State {
        rho,
        mom_x: rho * ux,
        mom_y: rho * uy,
        ee: e_total / 3.0,
        ei: e_total / 3.0,
        er: e_total / 3.0,
    }
}

pub fn init_double_mach() -> Field {
    // Fig. 3.2(a): polygonal computational domain.
    let sqrt3 = 3.0_f64.sqrt();
    let a = Point { x: -0.5 - sqrt3 / 12.0, y: 0.0 };
    let b = Point { x: 0.0, y: 0.0 };
    let c = Point { x: 23.0 * sqrt3 / 12.0, y: 23.0 / 12.0 };
    let d = Point { x: c.x, y: 23.0 / 12.0 + sqrt3 / 2.0 };
    let e = Point { x: a.x, y: d.y };

    // Paper benchmark: uniform square mesh h = 1/320.
    let h = 1.0 / 320.0;
    let x0 = a.x;
    let y0 = 0.0;
    let lx = c.x - a.x;
    let ly = d.y;
    let nx = (lx / h).ceil() as usize + 1;
    let ny = (ly / h).ceil() as usize + 1;
    let grid = GridInfo::new(nx, ny, h, h, x0, y0);

    // Standard Mach-10 normal-shock states for gamma=1.4.
    // The shock moves horizontally to +x and is initially at x=0.
    let gamma = 1.4;
    let pre = euler_to_three_energy(1.4, 0.0, 0.0, 1.0, gamma);
    let post = euler_to_three_energy(8.0, 8.25, 0.0, 116.5, gamma);

    // CCW edge ordering:
    // A->B: y=0 exact post-shock
    // B->C: 30-degree solid wall
    // C->D: supersonic outflow
    // D->E: exact moving Mach-10 shock
    // E->A: supersonic inflow
    let outer_bound = Polygon::new(vec![a, b, c, d, e], FluidSide::Inside);

    let top_pre = pre;
    let top_post = post;
    let top_bc = BCType::TimeDependent(Arc::new(move |p: Point, _n, t: f64| {
        // Upstream sound speed is 1, so Mach 10 shock speed is 10.
        let x_shock = 10.0 * t;
        if p.x <= x_shock { top_post } else { top_pre }
    }));

    let bc_outer = vec![
        BCType::Constant(post), // A -> B
        BCType::Wall,           // B -> C, inclined solid wall
        BCType::Constant(pre),  // C -> D, supersonic outflow: all chars leave
        BCType::ZerothOrder,                 // D -> E, exact moving shock
        BCType::Constant(post), // E -> A, supersonic inflow
    ];

    // No inner obstacle.
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
        0.0,
    );

    // Initial vertical shock at x=0: post-shock on the left, pre-shock right.
    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);
            if !u.is_in_domain(idx) { continue; }
            let x = grid.x(idx.0);
            u.set(idx, if x <= 0.0 { post } else { pre });
        }
    }

    println!(
        "Double Mach grid: nx={}, ny={}, h={:.8e}, bbox=({:.6},{:.6})x({:.6},{:.6})",
        nx, ny, h, x0, x0 + lx, y0, y0 + ly
    );

    u
}


pub fn init_cylinder() -> Field {
    // ============================================================
    // Mach-3 flow past the FRONT HALF of a circular cylinder
    //
    // Fluid domain:
    //
    //     -3 <= x <= 0
    //     -6 <= y <= 6
    //
    // with the solid half-disk
    //
    //     x^2 + y^2 < 1,  x <= 0
    //
    // removed from the domain.
    //
    // There is NO inner boundary.
    //
    // The cylinder arc is directly part of the outer polygon.
    // ============================================================

    let gamma = 1.4_f64;

    // ------------------------------------------------------------
    // Freestream
    // ------------------------------------------------------------

    let rho_inf = 1.0_f64;
    let p_inf = 1.0_f64;
    let mach_inf = 3.0_f64;

    let a_inf =
        (gamma * p_inf / rho_inf).sqrt();

    let ux_inf = mach_inf * a_inf;
    let uy_inf = 0.0;

    let u_inf =
        euler_to_three_energy(
            rho_inf,
            ux_inf,
            uy_inf,
            p_inf,
            gamma,
        );

    println!(
        "Cylinder freestream: rho={}, p={}, a={:.8e}, u={:.8e}, M={:.8e}",
        rho_inf,
        p_inf,
        a_inf,
        ux_inf,
        ux_inf / a_inf,
    );

    // ------------------------------------------------------------
    // Cartesian grid
    // ------------------------------------------------------------

    let h = 1.0 / 40.0;

    let x0 = -3.0_f64;
    let y0 = -6.0_f64;

    let lx = 3.0_f64;
    let ly = 12.0_f64;

    let nx =
        (lx / h).round() as usize + 1;

    let ny =
        (ly / h).round() as usize + 1;

    let grid =
        GridInfo::new(
            nx,
            ny,
            h,
            h,
            x0,
            y0,
        );

    // ============================================================
    // Build ONE outer polygon.
    //
    // Traverse the FLUID boundary counter-clockwise:
    //
    //   A = (-3,-6)
    //   B = ( 0,-6)
    //   C = ( 0,-1)
    //
    //   then along the LEFT semicircle:
    //
    //       (0,-1) -> (-1,0) -> (0,1)
    //
    //   then
    //
    //   D = (0,6)
    //   E = (-3,6)
    //
    //
    // The semicircle points are
    //
    //       x = cos(theta)
    //       y = sin(theta)
    //
    // theta : -pi/2 -> pi/2 THROUGH pi
    //
    // i.e. we need the LEFT half:
    //
    //       theta = -pi/2 -> -3pi/2
    //
    // when following the polygon CCW.
    // ============================================================

    let mut outer_points = Vec::<Point>::new();
    let mut bc_outer = Vec::<BCType>::new();

    // ------------------------------------------------------------
    // A = (-3,-6)
    // ------------------------------------------------------------

    outer_points.push(Point {
        x: -3.0,
        y: -6.0,
    });

    // ------------------------------------------------------------
    // A -> B
    //
    // bottom far field
    // ------------------------------------------------------------

    outer_points.push(Point {
        x: 0.0,
        y: -6.0,
    });

    bc_outer.push(
        BCType::FarField(u_inf)
    );

    // ------------------------------------------------------------
    // B -> C
    //
    // x = 0, -6 <= y <= -1
    //
    // This is an open far-field/outflow boundary.
    // ------------------------------------------------------------

    outer_points.push(Point {
        x: 0.0,
        y: -1.0,
    });

    bc_outer.push(
        BCType::FarField(u_inf)
    );

    // ============================================================
    // C -> ... -> D
    //
    // LEFT semicircle:
    //
    //       (0,-1)
    //          \
    //           \
    //          (-1,0)
    //           /
    //          /
    //       (0,1)
    //
    // Each segment gets Wall BC.
    // ============================================================

    let n_arc = 360_usize;

    for k in 1..=n_arc {
        let s =
            k as f64 / n_arc as f64;

        // Start:
        //     theta = -pi/2
        //
        // End:
        //     theta = -3pi/2
        //
        // This traces the LEFT semicircle:
        //
        //     (0,-1) -> (-1,0) -> (0,1)
        //
        let theta =
            -0.5 * std::f64::consts::PI
            - std::f64::consts::PI * s;

        outer_points.push(
            Point {
                x: theta.cos(),
                y: theta.sin(),
            }
        );

        bc_outer.push(
            BCType::ReflectiveWall
        );
    }

    // At this point the last arc point should be approximately:
    //
    //     (0,1)

    // ------------------------------------------------------------
    // (0,1) -> (0,6)
    //
    // right-side open boundary
    // ------------------------------------------------------------

    outer_points.push(Point {
        x: 0.0,
        y: 6.0,
    });

    bc_outer.push(
        BCType::FarField(u_inf)
    );

    // ------------------------------------------------------------
    // (0,6) -> (-3,6)
    //
    // top far field
    // ------------------------------------------------------------

    outer_points.push(Point {
        x: -3.0,
        y: 6.0,
    });

    bc_outer.push(
        BCType::FarField(u_inf)
    );

    // ------------------------------------------------------------
    // (-3,6) -> (-3,-6)
    //
    // left Mach-3 inflow.
    //
    // FarField automatically becomes supersonic inflow here.
    // ------------------------------------------------------------

    bc_outer.push(
        BCType::FarField(u_inf)
    );

    // Number of BCs MUST equal number of polygon sides.
    assert_eq!(
        bc_outer.len(),
        outer_points.len()
    );

    let outer_bound =
        Polygon::new(
            outer_points,
            FluidSide::Inside,
        );

    // ============================================================
    // NO physical inner boundary.
    //
    // Field currently requires an inner polygon, so leave a dummy
    // polygon far outside the computational domain.
    //
    // It will never participate in the cylinder BC.
    // ============================================================

    let inner_bound =
        Polygon::new(
            vec![
                Point {
                    x: -1002.0,
                    y: -1002.0,
                },
                Point {
                    x: -1001.0,
                    y: -1002.0,
                },
                Point {
                    x: -1001.0,
                    y: -1001.0,
                },
                Point {
                    x: -1002.0,
                    y: -1001.0,
                },
            ],
            FluidSide::Outside,
        );

    let bc_inner =
        vec![BCType::Wall; 4];

    // ============================================================
    // Construct Field
    // ============================================================

    let mut u =
        Field::new(
            grid,
            bc_inner,
            bc_outer,
            State::new(),
            outer_bound,
            inner_bound,
            0.0,
        );

    // ============================================================
    // Initial condition
    //
    // Uniform Mach-3 freestream on every FLUID grid point.
    //
    // Points inside the half cylinder are now automatically
    // excluded by outer_bound.is_fluid().
    // ============================================================

    for i in 0..nx {
        for j in 0..ny {
            let idx =
                (i as isize, j as isize);

            if !u.is_in_domain(idx) {
                continue;
            }

            u.set(idx, u_inf);
        }
    }

    println!(
        "Half-cylinder grid: nx={}, ny={}, h={:.8e}, \
         bbox=({:.6},{:.6})x({:.6},{:.6})",
        nx,
        ny,
        h,
        x0,
        x0 + lx,
        y0,
        y0 + ly,
    );

    println!(
        "Half-cylinder: R=1, arc segments={}, outer sides={}",
        n_arc,
        u.bc_outer.len(),
    );

    u
}