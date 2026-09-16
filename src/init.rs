use crate::bc::*;
use crate::field::*;
use crate::geometry::*;
use crate::state::*;

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

    // Analytic physical outer boundary (single source of boundary
    // definition): each element is ONE analytic segment + ONE BC. The
    // classifier polygon and the fluid mask are derived inside Field.
    let outer_elements = vec![
        line_element(a, b, BCType::Constant(post)), // A -> B
        line_element(b, c, BCType::Wall),           // B -> C, inclined wall
        line_element(c, d, BCType::Constant(pre)),  // C -> D, supersonic outflow
        line_element(d, e, BCType::ZerothOrder),    // D -> E, moving shock
        line_element(e, a, BCType::Constant(post)), // E -> A, supersonic inflow
    ];

    let mut u = Field::from_boundaries(grid, outer_elements, Vec::new(), State::new(), 0.0);

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

    // Half a Cartesian cell: keeps the wall half a cell away from the
    // nearest grid center so no center sits exactly on the boundary.
    let delta = 0.0125;

    // Analytic circular arc defining the physical cylinder boundary.
    // The three points select the LEFT arc passing through `mid`.
    let cylinder_arc = crate::geometry::CircularArc::from_three_points(
        Point {
            x: 0.0,
            y: -1.0 + delta,
        },
        Point {
            x: -1.0,
            y: delta,
        },
        Point {
            x: 0.0,
            y: 1.0 + delta,
        },
        FluidSide::Outside,
    );

    // ============================================================
    // Analytic physical outer boundary (single source): SIX elements.
    //
    // The classifier polygon and the fluid mask are derived inside
    // Field::from_boundaries. The vertical far-field lines end exactly at
    // the arc endpoints so the element chain closes:
    //
    //   0. bottom line  (-3,-6) -> (0,-6)              FarField
    //   1. lower right  (0,-6)  -> (0,-1+delta)         FarField
    //   2. LEFT semicircle (0,-1+delta) -> (0,1+delta)  Wall (cylinder)
    //   3. upper right  (0,1+delta) -> (0,6)            FarField
    //   4. top line     (0,6)    -> (-3,6)              FarField
    //   5. left line    (-3,6)   -> (-3,-6)             FarField
    // ============================================================

    let outer_elements = vec![
        line_element(
            Point { x: -3.0, y: -6.0 },
            Point { x: 0.0, y: -6.0 },
            BCType::FarField(u_inf),
        ),
        line_element(
            Point { x: 0.0, y: -6.0 },
            Point { x: 0.0, y: -1.0 + delta },
            BCType::FarField(u_inf),
        ),
        crate::bc::BoundaryElement {
            geometry: crate::geometry::BoundaryGeometry::Arc(cylinder_arc),
            bc: BCType::Wall,
        },
        line_element(
            Point { x: 0.0, y: 1.0 + delta },
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

    // No physical inner boundary.
    let mut u = Field::from_boundaries(grid, outer_elements, Vec::new(), State::new(), 0.0);

    // ============================================================
    // Initial condition
    //
    // Uniform Mach-3 freestream on every FLUID grid point.
    //
    // Points inside the half cylinder are excluded automatically by
    // the derived fluid mask.
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
        "Half-cylinder: R=1, analytic arc, outer elements={}",
        u.outer_boundary.len(),
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

    // ============================================================
    // ANALYTIC OUTER / INNER PHYSICAL BOUNDARIES (single source)
    //
    // Outer rectangle side ordering:
    //
    //      bottom (moving-shock region wall)
    //      right  (zeroth-order outflow)
    //      top    (moving-shock region wall)
    //      left   (constant post-shock inflow)
    //
    // Inner: ONE complete Circle, fluid OUTSIDE the cylinder
    // (FluidSide::Outside makes Projection.normal point from the fluid
    // into the solid cylinder).
    //
    // The classifier polygons and the fluid mask are derived inside
    // Field::from_boundaries.
    // ============================================================

    let outer_elements = vec![
        line_element(
            Point { x: xmin, y: ymin },
            Point { x: xmax, y: ymin },
            BCType::ReflectiveWall,
        ),
        line_element(
            Point { x: xmax, y: ymin },
            Point { x: xmax, y: ymax },
            BCType::ZerothOrder,
        ),
        line_element(
            Point { x: xmax, y: ymax },
            Point { x: xmin, y: ymax },
            BCType::ReflectiveWall,
        ),
        line_element(
            Point { x: xmin, y: ymax },
            Point { x: xmin, y: ymin },
            BCType::Constant(post),
        ),
    ];

    let inner_elements = vec![BoundaryElement {
        geometry: BoundaryGeometry::Circle(Circle::new(
            Point { x: cx, y: cy },
            radius,
            FluidSide::Outside,
        )),
        bc: wall.bc(),
    }];

    let mut u = Field::from_boundaries(grid, outer_elements, inner_elements, State::new(), 0.0);

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
    init_rotated_shock_cylinder_with(wall, 1.0 / 40.0, 360, 5.0, -1.10, 5.0)
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
    // ANALYTIC OUTER / INNER BOUNDARIES (single source)
    //
    // The 4 rotated LineSegments (CCW: p00->p10 lower wall, p10->p11
    // downstream outflow, p11->p01 upper wall, p01->p00 upstream inflow)
    // and the ONE analytic Circle for the cylinder are the only boundary
    // definition. The classifier polygons and the fluid mask are derived
    // inside Field::from_boundaries.
    // ============================================================

    let cx = 0.0 + h / 3.0;
    let cy = 0.0 + h / 3.0;
    let radius = 1.0_f64;

    let outer_elements = vec![
        // lower wall
        line_element(p00, p10, BCType::NonReflectiveOutflow),
        // downstream outflow
        line_element(p10, p11, BCType::ZerothOrder),
        // upper wall
        line_element(p11, p01, BCType::NonReflectiveOutflow),
        // upstream post-shock inflow
        line_element(p01, p00, BCType::Constant(post)),
    ];

    let inner_elements = vec![BoundaryElement {
        geometry: BoundaryGeometry::Circle(Circle::new(
            Point { x: cx, y: cy },
            radius,
            FluidSide::Outside,
        )),
        bc: wall.bc(),
    }];

    let mut u = Field::from_boundaries(grid, outer_elements, inner_elements, State::new(), 0.0);

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
            u.set(idx,post);
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

    // ============================================================
    // Analytic physical outer boundaries (single source of boundary
    // definition). The classifier polygons and fluid mask are derived
    // inside Field::from_boundaries.
    // ============================================================

    let outflow_bc = BCType::ZerothOrder;

    let outer_elements = vec![
        // Bottom wall
        line_element(p_bottom_left, p_bottom_right, BCType::ReflectiveWall),
        // Right outlet
        line_element(p_bottom_right, p_top_right, outflow_bc),
        // Top wall
        line_element(p_top_right, p_top_left, BCType::ReflectiveWall),
        // Left post-shock inflow
        line_element(p_top_left, p_bottom_left, BCType::Constant(post)),
    ];

    // No physical inner boundary.
    let mut u = Field::from_boundaries(grid, outer_elements, Vec::new(), State::new(), 0.0);

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
    init_forward_facing_step_rotated_with(1.0 / 80.0, 3.0, 0.2, 5.0)
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

    let xi_min = 0.0_f64;
    let xi_max = 3.0_f64;

    let eta_min = 0.0_f64;
    let eta_max = 1.0_f64;

    let step_xi = 0.6_f64;

    // Keep your current slightly offset step height.
    let step_eta = 0.2;

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
    // Analytic rotated boundaries (single source of boundary
    // definition): each side is ONE analytic segment + ONE BC. The
    // classifier polygon and the fluid mask are derived in Field.
    // ============================================================

    // Outflow:
    //
    // This outlet is oblique relative to the Cartesian grid. The
    // high-order Outflow stencil can fail on oblique outer
    // boundaries, so route this BC through ZerothOrder for now.
    let outflow_bc = BCType::ZerothOrder;

    let outer_elements = vec![
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

    let mut u = Field::from_boundaries(grid, outer_elements, Vec::new(), State::new(), 0.0);

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



#[inline(always)]
pub fn mms_63_exact_state(x: f64, y: f64, t: f64) -> State {
    let xi = x + y - 2.0 * t;

    let s = xi.sin();
    let c = xi.cos();

    // ------------------------------------------------------------------------
    // Exact primitive / internal-energy-density fields, Eq. (6.9)
    // ------------------------------------------------------------------------

    let rho = 1.0 + 0.5 * s;

    let ux = 1.0;
    let uy = 1.0;

    let rho_ee = 3.0 * (1.0 + 0.2 * s);
    let rho_ei = 3.0 * (1.0 + 0.2 * c);
    let rho_er = 2.0 * (1.0 + 0.1 * s);

    // ------------------------------------------------------------------------
    // Modified energy variables used by this solver:
    //
    //     E_alpha
    //       =
    //     rho e_alpha + rho (u^2 + v^2)/6.
    //
    // Since u=v=1 here,
    //
    //     kinetic_share = rho/3.
    // ------------------------------------------------------------------------

    let kinetic_share = rho * (ux * ux + uy * uy) / 6.0;

    State {
        rho,

        mom_x: rho * ux,
        mom_y: rho * uy,

        ee: rho_ee + kinetic_share,
        ei: rho_ei + kinetic_share,
        er: rho_er + kinetic_share,
    }
}


/// Convenience version using the number of cells N directly.
///
/// For the paper:
///
///     N = 40, 80, 120, 160, 200
///
pub fn init_mms_63(n: usize) -> Field {
    assert!(n >= 8, "MMS grid is too small");

    let pi = std::f64::consts::PI;

    // ========================================================================
    // Computational domain
    // ========================================================================

    let xmin = 0.0_f64;
    let xmax = 2.0 * pi;

    let ymin = 0.0_f64;
    let ymax = 2.0 * pi;

    let lx = xmax - xmin;
    let ly = ymax - ymin;

    // N uniform intervals in each direction.
    let dx = lx / n as f64;
    let dy = ly / n as f64;

    // ------------------------------------------------------------------------
    // Periodic grid size.
    //
    // The periodic cell count is the grid extent along the axis (see
    // bc::periodic_value, which wraps with rem_euclid(nx)), so the domain
    // is one full period of exactly n cells per axis:
    //
    //     nx = ny = n,     dx = dy = 2*pi/n
    //
    // The stored points span [0, 2*pi - dx]; the point at x = 2*pi (or
    // y = 2*pi) is the periodic image of the point at x = 0 (or y = 0)
    // and is obtained by wrapping, never stored twice.
    // ------------------------------------------------------------------------

    let nx = n;
    let ny = n;

    let grid = GridInfo::new(
        nx,
        ny,
        dx,
        dy,
        xmin,
        ymin,
    );

    // ========================================================================
    // Rectangular domain
    //
    // Counter-clockwise:
    //
    //       p01 ---------------- p11
    //        |                    |
    //        |                    |
    //        |                    |
    //       p00 ---------------- p10
    //
    // Polygon side ordering:
    //
    //     0 : bottom     p00 -> p10
    //     1 : right      p10 -> p11
    //     2 : top        p11 -> p01
    //     3 : left       p01 -> p00
    //
    // All four sides are periodic.
    // ========================================================================

    let p00 = Point {
        x: xmin,
        y: ymin,
    };

    let p10 = Point {
        x: xmax,
        y: ymin,
    };

    let p11 = Point {
        x: xmax,
        y: ymax,
    };

    let p01 = Point {
        x: xmin,
        y: ymax,
    };

    // Analytic physical outer boundaries (single source of boundary
    // definition). The classifier polygon and the fluid mask are derived
    // inside Field::from_boundaries.
    let outer_elements = vec![
        line_element(p00, p10, BCType::Periodic), // bottom
        line_element(p10, p11, BCType::Periodic), // right
        line_element(p11, p01, BCType::Periodic), // top
        line_element(p01, p00, BCType::Periodic), // left
    ];

    // No physical inner boundary.
    let inner_elements = Vec::new();

    // ========================================================================
    // Construct Field
    // ========================================================================

    let mut u = Field::from_boundaries(
        grid,
        outer_elements,
        inner_elements,
        State::new(),
        0.0,
    );

    // ========================================================================
    // Initial condition
    //
    // U(x,y,0) = exact manufactured solution.
    // ========================================================================

    for i in 0..nx {
        for j in 0..ny {
            let idx = (
                i as isize,
                j as isize,
            );

            if !u.is_in_domain(idx) {
                continue;
            }

            let x = grid.x(idx.0);
            let y = grid.y(idx.1);

            let state = mms_63_exact_state(
                x,
                y,
                0.0,
            );

            u.set(idx, state);
        }
    }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    println!();
    println!("============================================================");
    println!("CHENG-LEI-SHU SEC. 6.3 MMS ACCURACY TEST");
    println!("============================================================");

    println!(
        "N = {}, nx = {}, ny = {}",
        n, nx, ny
    );

    println!(
        "dx = {:.16e}, dy = {:.16e}",
        dx, dy
    );

    println!(
        "domain = [{:.8},{:.8}] x [{:.8},{:.8}]",
        xmin,
        xmax,
        ymin,
        ymax
    );

    println!();
    println!("Exact solution:");
    println!("  xi        = x + y - 2 t");
    println!("  rho       = 1 + 0.5 sin(xi)");
    println!("  u = v     = 1");
    println!("  rho e_e   = 3 [1 + 0.2 sin(xi)]");
    println!("  rho e_i   = 3 [1 + 0.2 cos(xi)]");
    println!("  rho e_r   = 2 [1 + 0.1 sin(xi)]");

    println!();
    println!("Parameters:");
    println!("  gamma_e = gamma_i = 5/3");
    println!("  gamma_r = 4/3");
    println!("  omega_ei = omega_er = 0");
    println!("  kappa_e = kappa_i = kappa_r = 0");

    println!();
    println!("BC:");
    println!("  x = 0    <-> x = 2*pi : Periodic");
    println!("  y = 0    <-> y = 2*pi : Periodic");

    println!();
    println!("Recommended final time:");
    println!("  t_final = 0.1");

    println!();
    println!("IMPORTANT:");
    println!("  mms_63_source(x,y,t) must be included in RHS.");
    println!("  project_equal_energies() must be disabled.");
    println!("============================================================");
    println!();

    u
}

#[inline(always)]
pub fn wall_mms_exact_state(x: f64, y: f64, t: f64) -> State {
    let sx = x.sin();
    let cx = x.cos();

    let sy = y.sin();
    let cy = y.cos();

    let st = t.sin();
    let ct = t.cos();

    // ------------------------------------------------------------------------
    // Density
    // ------------------------------------------------------------------------

    let rho =
        1.0
        + 0.1 * sx * cy * ct;

    // ------------------------------------------------------------------------
    // Velocity
    //
    // Notice:
    //
    //     v(x,0,t)     = 0
    //     v(x,2*pi,t)  = 0
    //
    // so the exact solution satisfies the top/bottom slip walls.
    // ------------------------------------------------------------------------

    let ux =
        1.0
        + 0.2 * cx * cy * ct;

    let uy =
        0.2 * sx * sy * ct;

    // ------------------------------------------------------------------------
    // Physical internal-energy densities:
    //
    //     q_alpha = rho e_alpha
    // ------------------------------------------------------------------------

    let rho_ee =
        3.0
        + 0.2 * cx * cy * st;

    let rho_ei =
        3.0
        + 0.15 * sx * cy * ct;

    let rho_er =
        2.0
        + 0.1 * (2.0 * x).cos() * cy * st;

    // ------------------------------------------------------------------------
    // Modified energy variables used by the solver
    // ------------------------------------------------------------------------

    let kinetic_share =
        rho * (ux * ux + uy * uy) / 6.0;

    State {
        rho,

        mom_x: rho * ux,
        mom_y: rho * uy,

        ee: rho_ee + kinetic_share,
        ei: rho_ei + kinetic_share,
        er: rho_er + kinetic_share,
    }
}

pub fn init_mms_wall(n: usize) -> Field {
    assert!(
        n >= 8,
        "Wall MMS grid is too small"
    );

    let pi = std::f64::consts::PI;

    let xmin = 0.0_f64;
    let xmax = 2.0 * pi;

    let ymin = 0.0_f64;
    let ymax = 2.0 * pi;

    let lx = xmax - xmin;
    let ly = ymax - ymin;

    let dx = lx / n as f64;
    let dy = ly / n as f64;

    let nx = n;
    let ny = n;

    let x0 = xmin + 0.5 * dx;
    let y0 = ymin + 0.5 * dy;

    let grid = GridInfo::new(
        nx,
        ny,
        dx,
        dy,
        x0,
        y0,
    );


    let p00 = Point {
        x: xmin,
        y: ymin,
    };

    let p10 = Point {
        x: xmax,
        y: ymin,
    };

    let p11 = Point {
        x: xmax,
        y: ymax,
    };

    let p01 = Point {
        x: xmin,
        y: ymax,
    };

    let outer_elements = vec![


        line_element(
            p00,
            p10,
            BCType::Wall,
        ),


        line_element(
            p10,
            p11,
            BCType::Periodic,
        ),

        // --------------------------------------------------------------------
        // Top:
        //
        //     y = 2*pi
        //
        // Exact normal velocity:
        //
        //     uy = 0.2 sin(x) sin(2*pi) cos(t) = 0
        // --------------------------------------------------------------------

        line_element(
            p11,
            p01,
            BCType::Wall,
        ),

        // --------------------------------------------------------------------
        // Left:
        //
        //     x = 0
        //
        // Periodic partner: x = 2*pi
        // --------------------------------------------------------------------

        line_element(
            p01,
            p00,
            BCType::Periodic,
        ),
    ];

    // No physical inner boundary.
    let inner_elements = Vec::new();

    // ========================================================================
    // Construct Field
    // ========================================================================

    let mut u = Field::from_boundaries(
        grid,
        outer_elements,
        inner_elements,
        State::new(),
        0.0,
    );

    // ========================================================================
    // Initial condition
    //
    //     U_ij(t=0) = U_exact(x_i,y_j,0)
    //
    // ========================================================================

    for i in 0..nx {
        for j in 0..ny {
            let idx = (
                i as isize,
                j as isize,
            );

            if !u.is_in_domain(idx) {
                continue;
            }

            let x = grid.x(idx.0);
            let y = grid.y(idx.1);

            let state =
                wall_mms_exact_state(
                    x,
                    y,
                    0.0,
                );

            u.set(
                idx,
                state,
            );
        }
    }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    println!();
    println!("============================================================");
    println!("SMOOTH WALL MMS ACCURACY TEST");
    println!("============================================================");

    println!(
        "N = {}, nx = {}, ny = {}",
        n,
        nx,
        ny,
    );

    println!(
        "dx = {:.16e}, dy = {:.16e}",
        dx,
        dy,
    );

    println!(
        "domain = [{:.8},{:.8}] x [{:.8},{:.8}]",
        xmin,
        xmax,
        ymin,
        ymax,
    );

    println!();

    println!("Exact solution:");

    println!(
        "  rho       = 1 + 0.1 sin(x) cos(y) cos(t)"
    );

    println!(
        "  u         = 1 + 0.2 cos(x) cos(y) cos(t)"
    );

    println!(
        "  v         = 0.2 sin(x) sin(y) cos(t)"
    );

    println!(
        "  rho e_e   = 3 + 0.2 cos(x) cos(y) sin(t)"
    );

    println!(
        "  rho e_i   = 3 + 0.15 sin(x) cos(y) cos(t)"
    );

    println!(
        "  rho e_r   = 2 + 0.1 cos(2x) cos(y) sin(t)"
    );

    println!();

    println!("BC:");

    println!(
        "  x = 0    <-> x = 2*pi : Periodic"
    );

    println!(
        "  y = 0                 : Wall"
    );

    println!(
        "  y = 2*pi              : Wall"
    );

    println!();

    println!("Wall compatibility:");

    println!(
        "  v(x,0,t)    = 0"
    );

    println!(
        "  v(x,2*pi,t) = 0"
    );

    println!(
        "  velocity dot normal = 0 exactly"
    );

    println!();

    println!("Parameters:");

    println!(
        "  gamma_e = gamma_i = 5/3"
    );

    println!(
        "  gamma_r = 4/3"
    );

    println!(
        "  omega_ei = omega_er = 0"
    );

    println!(
        "  kappa_e = kappa_i = kappa_r = 0"
    );

    println!();

    println!("Recommended convergence study:");

    println!(
        "  N = 40, 80, 120, 160, 200"
    );

    println!(
        "  t_final = 0.1"
    );

    println!(
        "  dt = 1e-5"
    );

    println!();

    println!("IMPORTANT:");

    println!(
        "  wall_mms_source(x,y,t) must be included in RHS."
    );

    println!(
        "  project_equal_energies() must be disabled."
    );

    println!("============================================================");
    println!();

    u
}

// ============================================================================
// Example 5.8
// 3-T radiation-hydrodynamic Rayleigh-Taylor instability
//
// Computational domain:
//
//     [0, 0.25] x [0, 1]
//
// Initial interface:
//
//     y = 0.5
//
// Heavy fluid:
//     rho = 2,  0 <= y < 0.5
//
// Light fluid:
//     rho = 1,  0.5 <= y <= 1
//
// Initial perturbation:
//
//     u = 0
//     v = -0.025 * c_s * cos(8*pi*x)
//
// Three-temperature pressures:
//
//     p_e = p_i = p_r
//
//     lower:
//         p = (2y + 1)/3
//
//     upper:
//         p = (2y + 3)/6
//
// Boundary conditions:
//
//     left/right : reflective
//
//     bottom:
//         rho = 2
//         u = v = 0
//         pe = pi = pr = 1/3
//
//     top:
//         rho = 1
//         u = v = 0
//         pe = pi = pr = 5/6
//
// Paper:
//     gamma_e = gamma_i = 5/3
//     omega_ei = omega_er = 0
//     kappa_e = kappa_i = kappa_r = 0
//
//     grid = 200 x 1200
//     T_final = 1.95
//
// IMPORTANT:
//
// The acceleration is in the +y direction.
//
// The governing equations therefore also require the gravitational/
// acceleration source:
//
//     S_(rho v) = rho
//
// and the corresponding energy source:
//
//     S_E = rho * v
//
// That source belongs in the RHS/time-evolution code, NOT here.
// ============================================================================

pub fn init_rayleigh_taylor_3t() -> Field {
    // ========================================================================
    // Domain
    // ========================================================================

    let xmin = 0.0_f64;
    let xmax = 0.25_f64;

    let ymin = 0.0_f64;
    let ymax = 1.0_f64;

    // ------------------------------------------------------------------------
    // Paper grid:
    //
    //     200 x 1200
    //
    // Here these are interpreted as the number of Cartesian grid points,
    // consistent with GridInfo::new(nx, ny, ...).
    // ------------------------------------------------------------------------

    let nx = 200_usize;
    let ny = 1200_usize;

    let dx = (xmax - xmin) / ((nx - 1) as f64);
    let dy = (ymax - ymin) / ((ny - 1) as f64);

    let grid = GridInfo::new(
        nx,
        ny,
        dx,
        dy,
        xmin,
        ymin,
    );

    // ========================================================================
    // Boundary states
    // ========================================================================

    // Bottom:
    //
    //     rho = 2
    //     u = v = 0
    //     pe = pi = pr = 1/3
    //
    // Use the solver's own primitive -> conservative conversion.
    let bottom_state = State::primi2con(
        2.0,
        0.0,
        0.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    );

    // Top:
    //
    //     rho = 1
    //     u = v = 0
    //     pe = pi = pr = 5/6
    let top_state = State::primi2con(
        1.0,
        0.0,
        0.0,
        5.0 / 6.0,
        5.0 / 6.0,
        5.0 / 6.0,
    );

    // ========================================================================
    // Rectangle
    //
    // Counter-clockwise ordering:
    //
    //        p01 ---------------- p11
    //         |                    |
    //         |                    |
    //         |                    |
    //        p00 ---------------- p10
    //
    // Boundary ordering:
    //
    //     p00 -> p10 : bottom
    //     p10 -> p11 : right
    //     p11 -> p01 : top
    //     p01 -> p00 : left
    //
    // This is the same CCW convention used by line_element().
    // ========================================================================

    let p00 = Point {
        x: xmin,
        y: ymin,
    };

    let p10 = Point {
        x: xmax,
        y: ymin,
    };

    let p11 = Point {
        x: xmax,
        y: ymax,
    };

    let p01 = Point {
        x: xmin,
        y: ymax,
    };

    // ========================================================================
    // Physical outer boundary
    // ========================================================================
    //
    // Paper:
    //
    //     left/right = reflective
    //     bottom     = prescribed constant state
    //     top        = prescribed constant state
    //
    // For the two vertical sides use ReflectiveWall, since the paper calls
    // these "reflective boundary conditions".
    // ========================================================================

    let outer_elements = vec![
        // --------------------------------------------------------------------
        // Bottom:
        //
        // rho = 2
        // u = v = 0
        // pe = pi = pr = 1/3
        // --------------------------------------------------------------------
        line_element(
            p00,
            p10,
            BCType::Constant(bottom_state),
        ),

        // --------------------------------------------------------------------
        // Right:
        //
        // reflective
        // --------------------------------------------------------------------
        line_element(
            p10,
            p11,
            BCType::ReflectiveWall,
        ),

        // --------------------------------------------------------------------
        // Top:
        //
        // rho = 1
        // u = v = 0
        // pe = pi = pr = 5/6
        // --------------------------------------------------------------------
        line_element(
            p11,
            p01,
            BCType::Constant(top_state),
        ),

        // --------------------------------------------------------------------
        // Left:
        //
        // reflective
        // --------------------------------------------------------------------
        line_element(
            p01,
            p00,
            BCType::ReflectiveWall,
        ),
    ];

    // No inner boundary.
    let inner_elements = Vec::new();

    // ========================================================================
    // Construct field
    // ========================================================================

    let mut u = Field::from_boundaries(
        grid,
        outer_elements,
        inner_elements,
        State::new(),
        0.0,
    );

    // ========================================================================
    // Initial condition
    //
    // Eq. (5.8)
    //
    // lower:
    //
    //     rho = 2
    //     u   = 0
    //     v   = -0.025 c_s cos(8*pi*x)
    //
    //     pe = pi = pr = (2y+1)/3
    //
    // upper:
    //
    //     rho = 1
    //     u   = 0
    //     v   = -0.025 c_s cos(8*pi*x)
    //
    //     pe = pi = pr = (2y+3)/6
    //
    // IMPORTANT:
    //
    // c_s is obtained from State::cs().
    //
    // We first construct a zero-velocity thermodynamic state, evaluate
    // State::cs(), then construct the actual perturbed state.
    //
    // This guarantees that the initialization uses exactly the same
    // sound-speed definition as the rest of the solver.
    // ========================================================================

    let pi_const = std::f64::consts::PI;

    for i in 0..nx {
        for j in 0..ny {
            let idx = (
                i as isize,
                j as isize,
            );

            if !u.is_in_domain(idx) {
                continue;
            }

            let x = grid.x(idx.0);
            let y = grid.y(idx.1);

            // ================================================================
            // Density and hydrostatic three-temperature pressure
            // ================================================================

            let (rho, p) = if y < 0.5 {
                // ------------------------------------------------------------
                // Heavy fluid
                //
                //     rho = 2
                //     p_e = p_i = p_r = (2y+1)/3
                // ------------------------------------------------------------

                (
                    2.0_f64,
                    (2.0 * y + 1.0) / 3.0,
                )
            } else {
                // ------------------------------------------------------------
                // Light fluid
                //
                //     rho = 1
                //     p_e = p_i = p_r = (2y+3)/6
                // ------------------------------------------------------------

                (
                    1.0_f64,
                    (2.0 * y + 3.0) / 6.0,
                )
            };
            let base_state = State::primi2con(
                rho,
                0.0,
                0.0,
                p,
                p,
                p,
            );

            let cs = base_state.cs();


            let ux = 0.0_f64;

            let uy =
                -0.025
                * cs
                * (8.0 * pi_const * x).cos();

            // ================================================================
            // Final conservative state
            // ================================================================

            let state = State::primi2con(
                rho,
                ux,
                uy,
                p,
                p,
                p,
            );

            u.set(
                idx,
                state,
            );
        }
    }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    println!();
    println!("============================================================");
    println!("3-T RAYLEIGH-TAYLOR INSTABILITY");
    println!("Example 5.8");
    println!("============================================================");

    println!(
        "grid: nx={}, ny={}",
        nx,
        ny,
    );

    println!(
        "dx = {:.16e}",
        dx,
    );

    println!(
        "dy = {:.16e}",
        dy,
    );

    println!(
        "domain = [{:.8},{:.8}] x [{:.8},{:.8}]",
        xmin,
        xmax,
        ymin,
        ymax,
    );

    println!();

    println!("Initial interface:");
    println!("  y = 0.5");

    println!();

    println!("Lower/heavy fluid:");
    println!("  rho = 2");
    println!("  pe = pi = pr = (2*y + 1)/3");

    println!();

    println!("Upper/light fluid:");
    println!("  rho = 1");
    println!("  pe = pi = pr = (2*y + 3)/6");

    println!();

    println!("Initial velocity:");
    println!("  u = 0");
    println!("  v = -0.025 * cs * cos(8*pi*x)");
    println!("  cs is evaluated using State::cs()");

    println!();

    println!("Boundary conditions:");
    println!("  left   = ReflectiveWall");
    println!("  right  = ReflectiveWall");
    println!("  bottom = Constant(rho=2, u=v=0, pe=pi=pr=1/3)");
    println!("  top    = Constant(rho=1, u=v=0, pe=pi=pr=5/6)");

    println!();

    println!("Paper parameters:");
    println!("  gamma_e = gamma_i = 5/3");
    println!("  omega_ei = omega_er = 0");
    println!("  kappa_e = kappa_i = kappa_r = 0");

    println!();

    println!("Acceleration:");
    println!("  direction = +y");
    println!("  magnitude = 1");
    println!("  RHS must include momentum/energy source terms.");

    println!();

    println!("Recommended final time:");
    println!("  T_final = 1.95");

    println!("============================================================");
    println!();

    u
}