use crate::bc1::*;
use crate::field1::*;
use crate::geometry::*;
use crate::state::*;
use std::sync::Arc;

/// Analytic straight boundary element.
///
/// The outward FLUID-domain normal is derived from the segment
/// direction using the polygon convention of this project:
///
///     CCW polygon with fluid INSIDE  =>  outward normal = (dy, -dx)/len
fn line_element(start: Point, end: Point, bc: BCType) -> BoundaryElement {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let len = dx.hypot(dy);
    assert!(len > 1e-14, "degenerate boundary line");

    BoundaryElement {
        geometry: BoundaryGeometry::Line(LineSegment::new(
            start,
            end,
            Vec2 {
                x: dy / len,
                y: -dx / len,
            },
        )),
        bc,
    }
}

fn euler_to_three_energy(rho: f64, ux: f64, uy: f64, p: f64, gamma: f64) -> State {
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
    let a = Point {
        x: -0.5 - sqrt3 / 12.0,
        y: 0.0,
    };
    let b = Point { x: 0.0, y: 0.0 };
    let c = Point {
        x: 23.0 * sqrt3 / 12.0,
        y: 23.0 / 12.0,
    };
    let d = Point {
        x: c.x,
        y: 23.0 / 12.0 + sqrt3 / 2.0,
    };
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
        BCType::ZerothOrder,    // D -> E, exact moving shock
        BCType::Constant(post), // E -> A, supersonic inflow
    ];

    // No inner obstacle.
    let inner_bound = Polygon::new(
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

    // Analytic physical outer boundary (CCW polygon, fluid inside):
    //  A->B bottom (post-shock), B->C wedge Wall, C->D supersonic
    //  outflow, D->E moving shock, E->A supersonic inflow.
    u.outer_boundary = vec![
        line_element(a, b, BCType::Constant(post)),
        line_element(b, c, BCType::Wall),
        line_element(c, d, BCType::Constant(pre)),
        line_element(d, e, BCType::ZerothOrder),
        line_element(e, a, BCType::Constant(post)),
    ];

    // Initial vertical shock at x=0: post-shock on the left, pre-shock right.
    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);
            if !u.is_in_domain(idx) {
                continue;
            }
            let x = grid.x(idx.0);
            u.set(idx, if x <= 0.0 { post } else { pre });
        }
    }

    println!(
        "Double Mach grid: nx={}, ny={}, h={:.8e}, bbox=({:.6},{:.6})x({:.6},{:.6})",
        nx,
        ny,
        h,
        x0,
        x0 + lx,
        y0,
        y0 + ly
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

    let a_inf = (gamma * p_inf / rho_inf).sqrt();

    let ux_inf = mach_inf * a_inf;
    let uy_inf = 0.0;

    let u_inf = euler_to_three_energy(rho_inf, ux_inf, uy_inf, p_inf, gamma);

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

    let nx = (lx / h).round() as usize + 1;

    let ny = (ly / h).round() as usize + 1;

    let grid = GridInfo::new(nx, ny, h, h, x0, y0);

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

    outer_points.push(Point { x: -3.0, y: -6.0 });

    // ------------------------------------------------------------
    // A -> B
    //
    // bottom far field
    // ------------------------------------------------------------

    outer_points.push(Point { x: 0.0, y: -6.0 });

    bc_outer.push(BCType::FarField(u_inf));

    // ------------------------------------------------------------
    // B -> C
    //
    // x = 0, -6 <= y <= -1
    //
    // This is an open far-field/outflow boundary.
    // ------------------------------------------------------------

    outer_points.push(Point {
        x: 0.0,
        y: -1.0 + 0.0125,
    });

    bc_outer.push(BCType::FarField(u_inf));

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
        let s = k as f64 / n_arc as f64;

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
        let theta = -0.5 * std::f64::consts::PI - std::f64::consts::PI * s;

        outer_points.push(Point {
            x: theta.cos(),
            y: theta.sin() + 0.0125,
        });

        bc_outer.push(BCType::Wall);
    }

    // Analytic circular arc defining the physical cylinder boundary.
    // The three points select the LEFT arc passing through `mid`.
    let cylinder_arc = crate::geometry::CircularArc::from_three_points(
        Point {
            x: 0.0,
            y: -1.0 + 0.0125,
        },
        Point {
            x: -1.0,
            y: 0.0 + 0.0125,
        },
        Point {
            x: 0.0,
            y: 1.0 + 0.0125,
        },
        FluidSide::Outside,
    );

    // At this point the last arc point should be approximately:
    //
    //     (0,1)

    // ------------------------------------------------------------
    // (0,1) -> (0,6)
    //
    // right-side open boundary
    // ------------------------------------------------------------

    outer_points.push(Point { x: 0.0, y: 6.0 });

    bc_outer.push(BCType::FarField(u_inf));

    // ------------------------------------------------------------
    // (0,6) -> (-3,6)
    //
    // top far field
    // ------------------------------------------------------------

    outer_points.push(Point { x: -3.0, y: 6.0 });

    bc_outer.push(BCType::FarField(u_inf));

    // ------------------------------------------------------------
    // (-3,6) -> (-3,-6)
    //
    // left Mach-3 inflow.
    //
    // FarField automatically becomes supersonic inflow here.
    // ------------------------------------------------------------

    bc_outer.push(BCType::FarField(u_inf));

    // Number of BCs MUST equal number of polygon sides.
    assert_eq!(bc_outer.len(), outer_points.len());

    let outer_bound = Polygon::new(outer_points, FluidSide::Inside);

    // ============================================================
    // NO physical inner boundary.
    //
    // Field currently requires an inner polygon, so leave a dummy
    // polygon far outside the computational domain.
    //
    // It will never participate in the cylinder BC.
    // ============================================================

    let inner_bound = Polygon::new(
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

    let bc_inner = vec![BCType::Wall; 4];

    // ============================================================
    // Analytic physical outer boundary: SIX BoundaryElements.
    //
    // The 360 Polygon arc segments above remain ONLY as the domain
    // classifier / fluid-mask approximation. All physical ghost
    // boundary geometry (P0, normal, D) and BC ownership now come
    // from these analytic elements:
    //
    //   0. bottom line  (-3,-6) -> (0,-6)        FarField
    //   1. lower right  (0,-6)  -> (0,-1)        FarField
    //   2. LEFT semicircle (0,-1) -> (-1,0) -> (0,+1)  cylinder BC
    //   3. upper right  (0,+1)  -> (0,+6)        FarField
    //   4. top line     (0,+6)  -> (-3,+6)       FarField
    //   5. left line    (-3,+6) -> (-3,-6)       FarField
    // ============================================================

    let mut u = Field::new(
        grid,
        bc_inner,
        bc_outer,
        State::new(),
        outer_bound,
        inner_bound,
        0.0,
    );

    u.outer_boundary = vec![
        line_element(
            Point { x: -3.0, y: -6.0 },
            Point { x: 0.0, y: -6.0 },
            BCType::FarField(u_inf),
        ),
        line_element(
            Point { x: 0.0, y: -6.0 },
            Point { x: 0.0, y: -1.0 },
            BCType::FarField(u_inf),
        ),
        crate::bc1::BoundaryElement {
            geometry: crate::geometry::BoundaryGeometry::Arc(cylinder_arc),
            bc: BCType::Wall,
        },
        line_element(
            Point { x: 0.0, y: 1.0 },
            Point { x: 0.0, y: 6.0 },
            BCType::FarField(u_inf),
        ),
        line_element(
            Point { x: 0.0, y: 6.0 },
            Point { x: -3.0, y: 6.0 },
            BCType::FarField(u_inf),
        ),
        line_element(
            Point { x: -3.0, y: 6.0 },
            Point { x: -3.0, y: -6.0 },
            BCType::FarField(u_inf),
        ),
    ];

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
            let idx = (i as isize, j as isize);

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

// ============================================================================
// Full interior cylinder inside a rectangular computational domain.
//
// Domain separation:
//
//     outer Polygon      -> rectangular domain / fluid classifier
//     inner Polygon      -> circle classifier (ONLY for fluid masking)
//     outer_boundary     -> four analytic LineSegment elements (FarField)
//     inner_boundary     -> ONE analytic Circle element (cylinder wall BC)
//
// The physical cylinder wall is a single complete Circle BoundaryElement.
// The inner Polygon is only used to answer "is this Cartesian point in the
// fluid domain?"; it NEVER provides the cylinder normal / P0 / distance.
// ============================================================================

/// Wall treatment for the interior cylinder obstacle.
///
/// `Reflective` gives a safe low-order startup; after evolving with
/// `Reflective` (e.g. to t ~= 0.1) one may restart the SAME geometry with
/// `HighOrder` or `Primitive` and a rebuilt GhostGrid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CylinderWallMode {
    /// Low-order reflective wall (mirror the state, no-penetration).
    Reflective,
    /// High-order conservative ILW wall.
    HighOrder,
    /// Primitive-variable wall (Euler-equivalent benchmark specialization).
    Primitive,
}

impl CylinderWallMode {
    fn bc(self) -> BCType {
        match self {
            CylinderWallMode::Reflective => BCType::ReflectiveWall,
            CylinderWallMode::HighOrder => BCType::Wall,
            CylinderWallMode::Primitive => BCType::PrimitiveWall,
        }
    }
}

/// Full interior cylinder in a rectangular box with the recommended
/// defaults:
///
///     [-3,+3] x [-6,+6]  (h = 1/40, nx = 241, ny = 481)
///     cylinder center (0,0), radius 1
///     n_circle = 360 classifier segments

pub fn init_shock_cylinder_in_box(wall: CylinderWallMode) -> Field {
    init_shock_cylinder_in_box_with(
        wall,
        1.0 / 40.0, // h
        360,        // inner Polygon classifier resolution
        3.0,        // shock Mach number
        -1.10,      // initial shock position
    )
}

pub fn init_shock_cylinder_in_box_with(
    wall: CylinderWallMode,
    h: f64,
    n_circle: usize,
    shock_mach: f64,
    shock_x0: f64,
) -> Field {
    assert!(h > 0.0, "grid spacing must be positive");

    assert!(
        n_circle >= 3,
        "circle classifier polygon needs at least 3 points"
    );

    assert!(shock_mach > 1.0, "shock Mach number must be > 1");

    let gamma = 1.4_f64;

    let rho_pre = 1.0_f64;
    let p_pre = 1.0_f64;

    let ux_pre = 0.0_f64;
    let uy_pre = 0.0_f64;

    let a_pre = (gamma * p_pre / rho_pre).sqrt();

    let shock_speed = shock_mach * a_pre;
    let ms2 = shock_mach * shock_mach;

    let density_ratio = ((gamma + 1.0) * ms2) / ((gamma - 1.0) * ms2 + 2.0);

    let pressure_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (ms2 - 1.0);

    let rho_post = rho_pre * density_ratio;

    let p_post = p_pre * pressure_ratio;

    let ux_post = shock_speed * (1.0 - rho_pre / rho_post);

    let uy_post = 0.0_f64;

    let pre = euler_to_three_energy(rho_pre, ux_pre, uy_pre, p_pre, gamma);

    let post = euler_to_three_energy(rho_post, ux_post, uy_post, p_post, gamma);

    // ============================================================
    // Diagnostics
    // ============================================================

    println!("Shock-cylinder case:");

    println!("  gamma       = {:.8e}", gamma,);

    println!("  shock Mach  = {:.8e}", shock_mach,);

    println!("  shock x0    = {:.8e}", shock_x0,);

    println!("  shock speed = {:.8e}", shock_speed,);

    println!(
        "  pre-shock : rho={:.8e}, p={:.8e}, u={:.8e}, a={:.8e}",
        rho_pre, p_pre, ux_pre, a_pre,
    );

    println!(
        "  post-shock: rho={:.8e}, p={:.8e}, u={:.8e}",
        rho_post, p_post, ux_post,
    );

    let xmin = -4.0_f64;
    let xmax = 20.0_f64;

    let ymin = -5.0_f64;
    let ymax = 5.0_f64;

    let cx = 0.0_f64;
    let cy = 0.0125_f64;

    let radius = 1.0_f64;

    let x0 = xmin;
    let y0 = ymin;

    let lx = xmax - xmin;
    let ly = ymax - ymin;

    let nx = (lx / h).round() as usize + 1;

    let ny = (ly / h).round() as usize + 1;

    let grid = GridInfo::new(nx, ny, h, h, x0, y0);

    let outer_bound = Polygon::new(
        vec![
            Point { x: xmin, y: ymin },
            Point { x: xmax, y: ymin },
            Point { x: xmax, y: ymax },
            Point { x: xmin, y: ymax },
        ],
        FluidSide::Inside,
    );

    let mut inner_points = Vec::with_capacity(n_circle);

    for k in 0..n_circle {
        let theta = 2.0 * std::f64::consts::PI * k as f64 / n_circle as f64;

        inner_points.push(Point {
            x: cx + radius * theta.cos(),

            y: cy + radius * theta.sin(),
        });
    }

    let inner_bound = Polygon::new(inner_points, FluidSide::Outside);

    let shock_bc =
        BCType::TimeDependent(Arc::new(move |p: Point, _normal: Vec2, t: f64| -> State {
            let x_shock = shock_x0 + shock_speed * t;

            if p.x <= x_shock { post } else { pre }
        }));

    // ============================================================
    // Legacy Polygon-side BC arrays.
    //
    // Rectangle side ordering:
    //
    //      0 bottom
    //      1 right
    //      2 top
    //      3 left
    //
    // The analytic BoundaryElements below are authoritative for
    // physical ghost geometry/BC lookup.
    // ============================================================

    let bc_outer = vec![
        BCType::FarField(post),
        BCType::ZerothOrder,
        BCType::FarField(post),
        BCType::Constant(post),
    ];

    // Inner Polygon has n_circle sides.
    //
    // This is legacy/domain compatibility data only.
    // Physical cylinder BC comes from ONE Circle element.

    let bc_inner = vec![wall.bc(); n_circle];

    // ============================================================
    // Construct Field
    // ============================================================

    let mut u = Field::new(
        grid,
        bc_inner,
        bc_outer,
        State::new(),
        outer_bound,
        inner_bound,
        0.0,
    );

    // ============================================================
    // ANALYTIC OUTER PHYSICAL BOUNDARY
    //
    // Side ordering:
    //
    //      bottom
    //      right
    //      top
    //      left
    //
    // ------------------------------------------------------------
    //
    // bottom:
    //
    //      exact moving shock BC
    //
    // right:
    //
    //      zeroth-order outflow
    //
    // top:
    //
    //      exact moving shock BC
    //
    // left:
    //
    //      constant post-shock inflow
    //
    // ============================================================

    u.outer_boundary = vec![
        // ----------------------------------------------------
        // Bottom
        // ----------------------------------------------------
        line_element(
            Point { x: xmin, y: ymin },
            Point { x: xmax, y: ymin },
            BCType::ReflectiveWall,
        ),
        // ----------------------------------------------------
        // Right
        // ----------------------------------------------------
        line_element(
            Point { x: xmax, y: ymin },
            Point { x: xmax, y: ymax },
            BCType::ZerothOrder,
        ),
        // ----------------------------------------------------
        // Top
        // ----------------------------------------------------
        line_element(
            Point { x: xmax, y: ymax },
            Point { x: xmin, y: ymax },
            BCType::ReflectiveWall,
        ),
        // ----------------------------------------------------
        // Left
        // ----------------------------------------------------
        line_element(
            Point { x: xmin, y: ymax },
            Point { x: xmin, y: ymin },
            BCType::Constant(post),
        ),
    ];

    // ============================================================
    // ANALYTIC INNER PHYSICAL BOUNDARY
    //
    // ONE complete Circle.
    //
    // Fluid is OUTSIDE the cylinder.
    //
    // Therefore FluidSide::Outside makes Projection.normal point
    // from fluid into the solid cylinder.
    // ============================================================

    u.inner_boundary = vec![BoundaryElement {
        geometry: BoundaryGeometry::Circle(Circle::new(
            Point { x: cx, y: cy },
            radius,
            FluidSide::Outside,
        )),

        bc: wall.bc(),
    }];

    // ============================================================
    // INITIAL CONDITION
    //
    // At t=0:
    //
    //      x <= shock_x0
    //
    //          POST-shock state
    //
    //      x > shock_x0
    //
    //          PRE-shock stationary state
    //
    // Because the cylinder is around x=[-1,+1] and the shock
    // starts at x=-2, the entire cylinder is initially surrounded
    // by stationary pre-shock gas.
    //
    // Therefore the initial condition is compatible with the
    // no-penetration cylinder wall:
    //
    //      u_n = 0
    //
    // ============================================================

    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);

            if !u.is_in_domain(idx) {
                continue;
            }

            let x = grid.x(idx.0);

            let state = if x <= shock_x0 { post } else { pre };

            u.set(idx, state);
        }
    }

    // ============================================================
    // Diagnostics
    // ============================================================

    println!(
        "Shock-cylinder grid: nx={}, ny={}, h={:.8e}, \
         bbox=({:.6},{:.6})x({:.6},{:.6})",
        nx, ny, h, xmin, xmax, ymin, ymax,
    );

    println!(
        "Cylinder: center=({:.6},{:.6}), R={}, \
         classifier segments={}, wall mode={:?}",
        cx, cy, radius, n_circle, wall,
    );

    println!(
        "Physical boundaries: inner={} element(s), outer={} element(s)",
        u.inner_boundary.len(),
        u.outer_boundary.len(),
    );

    // ============================================================
    // Useful physical time estimates
    // ============================================================

    let cylinder_front_x = cx - radius;

    let impact_time = (cylinder_front_x - shock_x0) / shock_speed;

    println!("Shock-cylinder impact estimate:");

    println!("  cylinder front x = {:.8e}", cylinder_front_x,);

    println!("  initial distance = {:.8e}", cylinder_front_x - shock_x0,);

    println!("  impact time      = {:.8e}", impact_time,);

    u
}

pub fn init_rotated_shock_cylinder(wall: CylinderWallMode) -> Field {
    init_rotated_shock_cylinder_with(wall, 1.0 / 40.0, 360, 3.0, -1.10, 30.0)
}

pub fn init_rotated_shock_cylinder_with(
    wall: CylinderWallMode,
    h: f64,
    n_circle: usize,
    shock_mach: f64,
    shock_xi0: f64,
    angle_deg: f64,
) -> Field {
    assert!(h > 0.0);
    assert!(n_circle >= 3);
    assert!(shock_mach > 1.0);

    let gamma = 1.4_f64;

    // ============================================================
    // Rotation
    // ============================================================

    let theta = angle_deg.to_radians();

    let ct = theta.cos();
    let st = theta.sin();

    // Channel streamwise direction:
    //
    //     e_xi = (cos(theta), sin(theta))
    //
    // Channel transverse direction:
    //
    //     e_eta = (-sin(theta), cos(theta))

    // ============================================================
    // Normal shock
    // ============================================================

    let rho_pre = 1.0_f64;
    let p_pre = 1.0_f64;

    let a_pre = (gamma * p_pre / rho_pre).sqrt();

    let shock_speed = shock_mach * a_pre;

    let ms2 = shock_mach * shock_mach;

    let density_ratio = ((gamma + 1.0) * ms2) / ((gamma - 1.0) * ms2 + 2.0);

    let pressure_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (ms2 - 1.0);

    let rho_post = rho_pre * density_ratio;

    let p_post = p_pre * pressure_ratio;

    // Post-shock velocity magnitude in lab frame.
    let u_post = shock_speed * (1.0 - rho_pre / rho_post);

    // ------------------------------------------------------------
    // PRE state:
    // stationary gas
    // ------------------------------------------------------------

    let pre = euler_to_three_energy(rho_pre, 0.0, 0.0, p_pre, gamma);

    // ------------------------------------------------------------
    // POST state:
    // velocity follows rotated channel direction
    // ------------------------------------------------------------

    let ux_post = u_post * ct;

    let uy_post = u_post * st;

    let post = euler_to_three_energy(rho_post, ux_post, uy_post, p_post, gamma);

    // ============================================================
    // Channel geometry in LOCAL coordinates
    //
    // xi  = streamwise
    // eta = transverse
    // ============================================================

    let xi_min = -6.0_f64;
    let xi_max = 20.0_f64;

    let eta_min = -6.0_f64;
    let eta_max = 6.0_f64;

    // Rotate local point into global Cartesian coordinates.
    let rotate = |xi: f64, eta: f64| -> Point {
        Point {
            x: ct * xi - st * eta,
            y: st * xi + ct * eta,
        }
    };

    // ============================================================
    // Four physical corners
    //
    // IMPORTANT:
    // Keep CCW ordering.
    // ============================================================

    let p00 = rotate(xi_min, eta_min);

    let p10 = rotate(xi_max, eta_min);

    let p11 = rotate(xi_max, eta_max);

    let p01 = rotate(xi_min, eta_max);

    // ============================================================
    // Cartesian bounding box containing the rotated channel
    //
    // The computational GridInfo remains Cartesian.
    // ============================================================

    let xmin = p00.x.min(p10.x).min(p11.x).min(p01.x);

    let xmax = p00.x.max(p10.x).max(p11.x).max(p01.x);

    let ymin = p00.y.min(p10.y).min(p11.y).min(p01.y);

    let ymax = p00.y.max(p10.y).max(p11.y).max(p01.y);

    // Give the Cartesian bbox a small margin.
    //
    // This is optional but avoids floating-point clipping of
    // polygon vertices at the grid edge.

    let pad = 2.0 * h;

    let x0 = xmin - pad;

    let y0 = ymin - pad;

    let lx = (xmax + pad) - x0;

    let ly = (ymax + pad) - y0;

    let nx = (lx / h).ceil() as usize + 1;

    let ny = (ly / h).ceil() as usize + 1;

    let grid = GridInfo::new(nx, ny, h, h, x0, y0);

    // ============================================================
    // OUTER DOMAIN CLASSIFIER
    //
    // Rotated rectangle.
    //
    // Fluid is inside.
    // ============================================================

    let outer_bound = Polygon::new(vec![p00, p10, p11, p01], FluidSide::Inside);

    // ============================================================
    // INNER CIRCLE CLASSIFIER
    //
    // Circle center remains at global (0,0).
    // ============================================================

    let cx = 0.0 + h / 3.0;
    let cy = 0.0 + h / 3.0;
    let radius = 1.0_f64;

    let mut inner_points = Vec::with_capacity(n_circle);

    for k in 0..n_circle {
        let phi = 2.0 * std::f64::consts::PI * k as f64 / n_circle as f64;

        inner_points.push(Point {
            x: cx + radius * phi.cos(),
            y: cy + radius * phi.sin(),
        });
    }

    let inner_bound = Polygon::new(inner_points, FluidSide::Outside);

    // ============================================================
    // Polygon-side BC compatibility arrays
    //
    // CCW sides:
    //
    // p00 -> p10 : lower wall
    // p10 -> p11 : downstream outflow
    // p11 -> p01 : upper wall
    // p01 -> p00 : upstream inflow
    // ============================================================

    let bc_outer = vec![
        BCType::Wall,
        BCType::ZerothOrder,
        BCType::Wall,
        BCType::Constant(post),
    ];

    let bc_inner = vec![wall.bc(); n_circle];

    // ============================================================
    // Field
    // ============================================================

    let mut u = Field::new(
        grid,
        bc_inner,
        bc_outer,
        State::new(),
        outer_bound,
        inner_bound,
        0.0,
    );

    // ============================================================
    // ANALYTIC OUTER BOUNDARIES
    //
    // line_element derives the fluid-domain outward normal from
    // the CCW segment orientation, so the rotated geometry works
    // automatically.
    // ============================================================

    u.outer_boundary = vec![
        // lower reflective wall
        line_element(p00, p10, BCType::Wall),
        // downstream outflow
        line_element(p10, p11, BCType::ZerothOrder),
        // upper reflective wall
        line_element(p11, p01, BCType::Wall),
        // upstream post-shock inflow
        line_element(p01, p00, BCType::Constant(post)),
    ];

    // ============================================================
    // ANALYTIC CYLINDER
    // ============================================================

    u.inner_boundary = vec![BoundaryElement {
        geometry: BoundaryGeometry::Circle(Circle::new(
            Point { x: cx, y: cy },
            radius,
            FluidSide::Outside,
        )),

        bc: wall.bc(),
    }];

    // ============================================================
    // INITIAL CONDITION
    //
    // In local channel coordinates:
    //
    //     xi = x cos(theta) + y sin(theta)
    //
    // Initial shock:
    //
    //     xi = shock_xi0
    //
    // Behind shock:
    //
    //     xi <= shock_xi0
    //
    // Ahead:
    //
    //     xi > shock_xi0
    //
    // ============================================================

    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);

            if !u.is_in_domain(idx) {
                continue;
            }

            let x = grid.x(idx.0);

            let y = grid.y(idx.1);

            // inverse rotation:
            //
            // xi = x cos(theta) + y sin(theta)

            let xi = x * ct + y * st;

            let state = if xi <= shock_xi0 { post } else { pre };

            u.set(idx, state);
        }
    }

    // ============================================================
    // Diagnostics
    // ============================================================

    println!("Rotated shock-cylinder channel:");

    println!("  angle       = {:.8} deg", angle_deg,);

    println!("  shock Mach  = {:.8}", shock_mach,);

    println!("  shock xi0   = {:.8}", shock_xi0,);

    println!("  shock speed = {:.8e}", shock_speed,);

    println!(
        "  post state: rho={:.8e}, p={:.8e}, \
         ux={:.8e}, uy={:.8e}",
        rho_post, p_post, ux_post, uy_post,
    );

    println!(
        "  channel local domain: \
         xi=[{:.4},{:.4}], eta=[{:.4},{:.4}]",
        xi_min, xi_max, eta_min, eta_max,
    );

    println!("  Cartesian grid: nx={}, ny={}, h={:.8e}", nx, ny, h,);

    println!(
        "  Cartesian bbox: \
         [{:.6},{:.6}] x [{:.6},{:.6}]",
        x0,
        x0 + (nx - 1) as f64 * h,
        y0,
        y0 + (ny - 1) as f64 * h,
    );

    println!("  cylinder: center=(0,0), R={}, wall={:?}", radius, wall,);

    u
}

pub fn init_planar_shock_channel() -> Field {
    init_planar_shock_channel_with(
        1.0 / 40.0, // h
        20.0,       // shock Mach number
        -2.0,       // initial shock position
    )
}

pub fn init_planar_shock_channel_with(h: f64, shock_mach: f64, shock_x0: f64) -> Field {
    assert!(h > 0.0);
    assert!(shock_mach > 1.0);

    // ============================================================
    // Gas
    // ============================================================

    let gamma = 1.4_f64;

    // ============================================================
    // Pre-shock state
    //
    // Stationary gas.
    // ============================================================

    let rho_pre = 1.0_f64;
    let p_pre = 1.0_f64;

    let ux_pre = 0.0_f64;
    let uy_pre = 0.0_f64;

    let a_pre = (gamma * p_pre / rho_pre).sqrt();

    // ============================================================
    // Shock speed
    // ============================================================

    let shock_speed = shock_mach * a_pre;

    // ============================================================
    // Normal-shock Rankine-Hugoniot relations
    // ============================================================

    let ms2 = shock_mach * shock_mach;

    let density_ratio = ((gamma + 1.0) * ms2) / ((gamma - 1.0) * ms2 + 2.0);

    let pressure_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (ms2 - 1.0);

    let rho_post = rho_pre * density_ratio;

    let p_post = p_pre * pressure_ratio;

    // ============================================================
    // Post-shock lab-frame velocity
    //
    // rho1 * D
    //      =
    // rho2 * (D - u2)
    //
    // therefore
    //
    // u2 =
    // D * (1 - rho1/rho2)
    // ============================================================

    let ux_post = shock_speed * (1.0 - rho_pre / rho_post);

    let uy_post = 0.0_f64;

    // ============================================================
    // Convert to three-energy state
    // ============================================================

    let pre = euler_to_three_energy(rho_pre, ux_pre, uy_pre, p_pre, gamma);

    let post = euler_to_three_energy(rho_post, ux_post, uy_post, p_post, gamma);

    // ============================================================
    // Computational domain
    //
    // Long enough that the shock can propagate for a while before
    // reaching the outlet.
    // ============================================================

    let xmin = -4.0_f64;
    let xmax = 12.0_f64;

    let ymin = -2.0_f64;
    let ymax = 2.0_f64;

    let lx = xmax - xmin;
    let ly = ymax - ymin;

    let nx = (lx / h).round() as usize + 1;

    let ny = (ly / h).round() as usize + 1;

    let grid = GridInfo::new(nx, ny, h, h, xmin, ymin);

    // ============================================================
    // Outer-domain classifier
    //
    // Counter-clockwise rectangle.
    // ============================================================

    let p_bottom_left = Point { x: xmin, y: ymin };

    let p_bottom_right = Point { x: xmax, y: ymin };

    let p_top_right = Point { x: xmax, y: ymax };

    let p_top_left = Point { x: xmin, y: ymax };

    let outer_bound = Polygon::new(
        vec![p_bottom_left, p_bottom_right, p_top_right, p_top_left],
        FluidSide::Inside,
    );

    // ============================================================
    // No physical inner boundary.
    //
    // Field currently expects an inner Polygon, so put a tiny
    // dummy polygon far outside the computational domain.
    //
    // If your Field now supports "no inner boundary", replace this
    // with your corresponding empty/no-inner-boundary constructor.
    // ============================================================

    let dummy_x = xmin - 100.0;
    let dummy_y = ymin - 100.0;

    let inner_bound = Polygon::new(
        vec![
            Point {
                x: dummy_x,
                y: dummy_y,
            },
            Point {
                x: dummy_x + 1.0,
                y: dummy_y,
            },
            Point {
                x: dummy_x,
                y: dummy_y + 1.0,
            },
        ],
        FluidSide::Outside,
    );

    // ============================================================
    // OUTER BC
    //
    // Polygon side order:
    //
    //      0: bottom
    //      1: right
    //      2: top
    //      3: left
    //
    // Start with ReflectiveWall because it gives us a robust
    // baseline.
    //
    // Once this works, replace top/bottom with BCType::Wall.
    // ============================================================

    let p_inf = p_pre;

    let sigma = 0.25_f64;

    let l_domain = xmax - xmin;

    let outflow_bc = BCType::Outflow {
        p_inf,
        sigma,
        l_domain,
    };
    let outflow_bc = BCType::ZerothOrder;

    let bc_outer = vec![
        // bottom
        BCType::ReflectiveWall,
        // right
        outflow_bc.clone(),
        // top
        BCType::ReflectiveWall,
        // left
        BCType::Constant(post),
    ];

    // Dummy inner BC.
    let bc_inner = vec![BCType::ZerothOrder; 3];

    // ============================================================
    // Construct Field
    // ============================================================

    let mut u = Field::new(
        grid,
        bc_inner,
        bc_outer,
        State::new(),
        outer_bound,
        inner_bound,
        0.0,
    );

    // ============================================================
    // Analytic physical outer boundaries
    // ============================================================

    u.outer_boundary = vec![
        // ----------------------------------------------------
        // Bottom wall
        //
        // CCW:
        //
        // bottom-left -> bottom-right
        // ----------------------------------------------------
        line_element(p_bottom_left, p_bottom_right, BCType::ReflectiveWall),
        // ----------------------------------------------------
        // Right outlet
        // ----------------------------------------------------
        line_element(p_bottom_right, p_top_right, outflow_bc),
        // ----------------------------------------------------
        // Top wall
        // ----------------------------------------------------
        line_element(p_top_right, p_top_left, BCType::ReflectiveWall),
        // ----------------------------------------------------
        // Left post-shock inflow
        // ----------------------------------------------------
        line_element(p_top_left, p_bottom_left, BCType::Constant(post)),
    ];

    // No physical inner BoundaryElement.
    u.inner_boundary = Vec::new();

    // ============================================================
    // Initial condition
    //
    //          POST      PRE
    //
    //      -------->|-----------
    //                 shock_x0
    //
    // ============================================================

    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);

            if !u.is_in_domain(idx) {
                continue;
            }

            let x = grid.x(idx.0);

            let state = if x <= shock_x0 { post } else { pre };

            u.set(idx, state);
        }
    }

    // ============================================================
    // Diagnostics
    // ============================================================

    println!();
    println!("==============================================");
    println!("PLANAR SHOCK CHANNEL");
    println!("==============================================");

    println!("grid: nx={}, ny={}, h={:.8e}", nx, ny, h,);

    println!(
        "domain: [{:.4},{:.4}] x [{:.4},{:.4}]",
        xmin, xmax, ymin, ymax,
    );

    println!("shock Mach = {:.8}", shock_mach,);

    println!("shock x0   = {:.8}", shock_x0,);

    println!("shock speed = {:.8e}", shock_speed,);

    println!(
        "pre : rho={:.8e}, p={:.8e}, ux={:.8e}",
        rho_pre, p_pre, ux_pre,
    );

    println!(
        "post: rho={:.8e}, p={:.8e}, ux={:.8e}",
        rho_post, p_post, ux_post,
    );

    // Time at which shock reaches outlet:
    //
    // x_s(t) = shock_x0 + D_s t

    let outlet_hit_time = (xmax - shock_x0) / shock_speed;

    println!("expected shock/outlet time = {:.8e}", outlet_hit_time,);

    println!(
        "BC: left=Constant(post), \
         right=Outflow, \
         top/bottom=ReflectiveWall"
    );

    println!("==============================================");
    println!();

    u
}

pub fn init_forward_facing_step_rotated() -> Field {
    init_forward_facing_step_rotated_with(1.0 / 80.0, 8.0, 0.2, 0.0)
}

pub fn init_forward_facing_step_rotated_with(
    h: f64,
    shock_mach: f64,
    shock_xi0: f64,
    angle_deg: f64,
) -> Field {
    assert!(h > 0.0);
    assert!(shock_mach > 1.0);

    let gamma = 1.4_f64;

    // ============================================================
    // Rotation
    // ============================================================

    let theta = angle_deg.to_radians();
    let ct = theta.cos();
    let st = theta.sin();

    // Local -> global Cartesian
    let rotate = |xi: f64, eta: f64| -> Point {
        Point {
            x: ct * xi - st * eta,
            y: st * xi + ct * eta,
        }
    };

    // ============================================================
    // Pre-shock state: stationary gas
    // ============================================================

    let rho_pre = 1.0_f64;
    let p_pre = 1.0_f64;

    let a_pre = (gamma * p_pre / rho_pre).sqrt();

    let shock_speed = shock_mach * a_pre;

    // ============================================================
    // Normal-shock RH relations
    // ============================================================

    let ms2 = shock_mach * shock_mach;

    let density_ratio = ((gamma + 1.0) * ms2) / ((gamma - 1.0) * ms2 + 2.0);

    let pressure_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (ms2 - 1.0);

    let rho_post = rho_pre * density_ratio;

    let p_post = p_pre * pressure_ratio;

    // Post-shock velocity magnitude along +xi.
    let u_post = shock_speed * (1.0 - rho_pre / rho_post);

    // Rotate velocity into Cartesian coordinates.
    let ux_post = u_post * ct;

    let uy_post = u_post * st;

    let gamma = 1.4_f64;

    let rho_inf = 1.0_f64;
    let p_inf = 1.0_f64;
    let mach_inf = 3.0_f64;

    let a_inf = (gamma * p_inf / rho_inf).sqrt();
    let v_inf = mach_inf * a_inf;

    let ux_inf = v_inf * ct;
    let uy_inf = v_inf * st;

    let inflow = euler_to_three_energy(rho_inf, ux_inf, uy_inf, p_inf, gamma);

    // ============================================================
    // States
    // ============================================================

    let pre = euler_to_three_energy(rho_pre, 0.0, 0.0, p_pre, gamma);

    let post = euler_to_three_energy(rho_post, ux_post, uy_post, p_post, gamma);

    // ============================================================
    // Physical geometry in LOCAL coordinates (xi, eta)
    // ============================================================

    let xi_min = -1.0_f64;
    let xi_max = 5.0_f64;

    let eta_min = 0.0_f64;
    let eta_max = 1.5_f64;

    let step_xi = 0.6_f64;

    // Keep your current slightly offset step height.
    let step_eta = 0.2 + 0.3 * h;

    assert!(shock_xi0 > xi_min && shock_xi0 < step_xi);

    // ============================================================
    // Forward-step vertices in LOCAL coordinates
    //
    // p5 ----------------------------- p4
    // |                                  |
    // |                                  |
    // |        p2 -------------------- p3
    // |        |
    // |        |
    // p0 ----- p1
    //
    // ============================================================

    let p0 = rotate(xi_min, eta_min);

    let p1 = rotate(step_xi, eta_min);

    let p2 = rotate(step_xi, step_eta);

    let p3 = rotate(xi_max, step_eta);

    let p4 = rotate(xi_max, eta_max);

    let p5 = rotate(xi_min, eta_max);

    // ============================================================
    // Cartesian bounding box containing the rotated polygon
    // ============================================================

    let points = [p0, p1, p2, p3, p4, p5];

    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;

    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;

    for p in points {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);

        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }

    // Padding gives the embedded outer boundary enough ghost space.
    let pad = 4.0 * h;

    xmin -= pad;
    xmax += pad;

    ymin -= pad;
    ymax += pad;

    let nx = ((xmax - xmin) / h).ceil() as usize + 1;

    let ny = ((ymax - ymin) / h).ceil() as usize + 1;

    let grid = GridInfo::new(nx, ny, h, h, xmin, ymin);

    // ============================================================
    // Rotated forward-step polygon
    // ============================================================

    let outer_bound = Polygon::new(vec![p0, p1, p2, p3, p4, p5], FluidSide::Inside);

    // ============================================================
    // Dummy inner boundary
    // ============================================================

    let inner_bound = Polygon::new(
        vec![
            Point {
                x: -100.0,
                y: -100.0,
            },
            Point {
                x: -99.0,
                y: -100.0,
            },
            Point {
                x: -100.0,
                y: -99.0,
            },
        ],
        FluidSide::Outside,
    );

    // ============================================================
    // Outflow
    //
    // IMPORTANT:
    // This outlet is now oblique relative to the Cartesian grid.
    //
    // Since you already found that the high-order Outflow stencil
    // can fail on oblique outer boundaries, I recommend routing
    // this BC through robust_outflow_copy for this experiment.
    // ============================================================

    let outflow_bc = BCType::ZerothOrder;
    BCType::Outflow {
        p_inf: p_pre,
        sigma: 0.25,
        l_domain: xi_max - xi_min,
    };

    // ============================================================
    // Polygon BCs
    //
    // p0 -> p1 : bottom Wall
    // p1 -> p2 : step front Wall
    // p2 -> p3 : step top Wall
    // p3 -> p4 : outlet
    // p4 -> p5 : top Wall
    // p5 -> p0 : inflow
    // ============================================================

    let bc_outer = vec![
        // Wall: high-order ILW with a LOCAL ReflectiveWall fallback
        // for ghosts where the ILW reconstruction is untrustworthy.
        BCType::Wall,
        BCType::Wall,
        BCType::Wall,
        outflow_bc.clone(),
        BCType::Wall,
        BCType::Constant(post),
    ];

    let bc_inner = vec![BCType::ZerothOrder; 3];

    // ============================================================
    // Field
    // ============================================================

    let mut u = Field::new(
        grid,
        bc_inner,
        bc_outer,
        State::new(),
        outer_bound,
        inner_bound,
        0.0,
    );

    // ============================================================
    // Analytic rotated boundaries
    // ============================================================

    u.outer_boundary = vec![
        // bottom
        line_element(p0, p1, BCType::Wall),
        // step vertical face
        line_element(p1, p2, BCType::Wall),
        // step top
        line_element(p2, p3, BCType::Wall),
        // downstream outlet
        line_element(p3, p4, outflow_bc),
        // upper wall
        line_element(p4, p5, BCType::Wall),
        // upstream boundary
        line_element(p5, p0, BCType::Constant(inflow)),
    ];

    u.inner_boundary = Vec::new();

    // ============================================================
    // Initial condition
    //
    // Convert every Cartesian grid point back into local
    // forward-step coordinates:
    //
    //     xi  =  x cos(theta) + y sin(theta)
    //     eta = -x sin(theta) + y cos(theta)
    //
    // Shock:
    //
    //     xi = shock_xi0
    //
    // ============================================================

    for i in 0..nx {
        for j in 0..ny {
            let idx = (i as isize, j as isize);

            if !u.is_in_domain(idx) {
                continue;
            }

            let x = grid.x(idx.0);

            let y = grid.y(idx.1);

            // Inverse rotation.
            let xi = x * ct + y * st;

            let state = if xi <= shock_xi0 { post } else { post };

            u.set(idx, inflow);
        }
    }

    // ============================================================
    // Diagnostics
    // ============================================================

    let step_hit_time = (step_xi - shock_xi0) / shock_speed;

    println!();
    println!("============================================================");

    println!("ROTATED SHOCK-DRIVEN FORWARD-FACING STEP");

    println!("============================================================");

    println!("rotation angle = {:.8} deg", angle_deg,);

    println!("grid: nx={}, ny={}, h={:.8e}", nx, ny, h,);

    println!(
        "Cartesian bbox: [{:.8},{:.8}] x [{:.8},{:.8}]",
        xmin, xmax, ymin, ymax,
    );

    println!("local step: xi={:.8}, eta={:.8}", step_xi, step_eta,);

    println!("initial shock: xi={:.8}", shock_xi0,);

    println!("shock equation:");

    println!("  x*cos(theta) + y*sin(theta) = {:.8}", shock_xi0,);

    println!("shock speed = {:.8e}", shock_speed,);

    println!("post velocity magnitude = {:.8e}", u_post,);

    println!(
        "post Cartesian velocity = ({:.8e}, {:.8e})",
        ux_post, uy_post,
    );

    println!("expected shock-step impact time = {:.8e}", step_hit_time,);

    println!("BC:");

    println!("  bottom     = Wall");

    println!("  step front = Wall");

    println!("  step top   = Wall");

    println!("  top        = Wall");

    println!("  left       = Constant(post)");

    println!("  right      = Outflow");

    println!("============================================================");

    println!();

    u
}
