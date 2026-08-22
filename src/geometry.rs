
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

fn cross(a: Point, b: Point) -> f64 {
    a.x * b.y - a.y * b.x
}

pub fn point_on_segment(p: Point, a: Point, b: Point, eps: f64) -> bool {
    let ab = Point {
        x: b.x - a.x,
        y: b.y - a.y,
    };

    let ap = Point {
        x: p.x - a.x,
        y: p.y - a.y,
    };

    // Collinearity
    if cross(ab, ap).abs() > eps {
        return false;
    }

    // Check that p lies within the bounding box of the segment
    p.x >= a.x.min(b.x) - eps
        && p.x <= a.x.max(b.x) + eps
        && p.y >= a.y.min(b.y) - eps
        && p.y <= a.y.max(b.y) + eps
}

pub fn find_boundary_sides(
    p: Point,
    polygon: &Polygon,
    eps: f64,
) -> Vec<usize> {
    let n = polygon.points.len();
    let pts = &polygon.points;

    let mut sides = Vec::new();

    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];

        if point_on_segment(p, a, b, eps) {
            sides.push(i);
        }
    }

    sides
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    #[inline]
    pub fn norm(self) -> f64 {
        self.x.hypot(self.y)
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();

        assert!(n > 1e-14, "Cannot normalize a zero vector");

        Self {
            x: self.x / n,
            y: self.y / n,
        }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    pub point: Point,    // P0
    pub normal: Vec2,    // fluid-domain outward normal
    pub distance: f64,  // D = (P - P0) dot n
}

impl Projection {
    

    pub fn gloabl2local_coord(&self,p: Point) -> (f64, f64) {
        let p0 = self.point;
        let normal = self.normal;
        let dx = p.x - p0.x;
        let dy = p.y - p0.y;

        let nor = normal.x*dx + normal.y*dy;
        let tan = -dx*normal.y + dy*normal.x;

        (nor,tan)
    }
}

pub trait Geometry {
    /// True if the point lies inside the solid geometry.
    fn is_fluid(&self, p: Point) -> bool;

    /// Closest point on the boundary to p.
    fn closest_point(&self, p: Point) -> Point;

    /// Outward normal of the FLUID domain at a boundary point.
    fn normal(&self, p0: Point) -> Vec2;
}

pub fn project<G: Geometry + ?Sized>(
    geom: &G,
    p: Point,
) -> Projection {
    let p0 = geom.closest_point(p);
    let n = geom.normal(p0);

    let dx = p.x - p0.x;
    let dy = p.y - p0.y;

    let distance = dx * n.x + dy * n.y;

    Projection {
        point: p0,
        normal: n,
        distance,
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FluidSide {
    Inside,
    Outside,
}

#[derive(Debug)]
pub struct Polygon {
    pub points: Vec<Point>,
    pub fluid: FluidSide,
}

impl Polygon {
    pub fn new(points: Vec<Point>,fluid_side: FluidSide) -> Self {
        Self {
            points: points,
            fluid: fluid_side,
        }
    }
}

impl Polygon {
    /// Twice the signed area of the polygon.
    ///
    /// > 0 => counter-clockwise
    /// < 0 => clockwise
    fn signed_area2(&self) -> f64 {
        let n = self.points.len();

        let mut area2 = 0.0;

        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];

            area2 += a.x * b.y - b.x * a.y;
        }

        area2
    }

    #[inline]
    fn closest_point_on_segment(p: Point, a: Point, b: Point) -> Point {
        let abx = b.x - a.x;
        let aby = b.y - a.y;

        let apx = p.x - a.x;
        let apy = p.y - a.y;

        let ab2 = abx * abx + aby * aby;

        // Degenerate edge
        if ab2 <= 1e-30 {
            return a;
        }

        let t = (apx * abx + apy * aby) / ab2;
        let t = t.clamp(0.0, 1.0);

        Point {
            x: a.x + t * abx,
            y: a.y + t * aby,
        }
    }

    #[inline]
    fn distance2(a: Point, b: Point) -> f64 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;

        dx * dx + dy * dy
    }

    pub fn outward_normal_of_side(
        &self,
        side: usize,
    ) -> Vec2 {
        let n = self.points.len();

        let a = self.points[side];
        let b = self.points[(side + 1) % n];

        let tx = b.x - a.x;
        let ty = b.y - a.y;

        let len = tx.hypot(ty);

        assert!(len > 1.0e-14);

        let left = Vec2 {
            x: -ty / len,
            y:  tx / len,
        };

        let right = Vec2 {
            x:  ty / len,
            y: -tx / len,
        };

        let ccw =
            self.signed_area2() > 0.0;

        match self.fluid {
            FluidSide::Inside => {
                if ccw { right } else { left }
            }

            FluidSide::Outside => {
                if ccw { left } else { right }
            }
        }
    }
}

impl Geometry for Polygon {
    fn is_fluid(&self, p: Point)-> bool{
        let n = self.points.len();
        if n< 3 {
            return false
        }
        let mut inside = false;
        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i+1)%n];
            let crosses = (a.y > p.y) != (b.y > p.y);

            if crosses {
                let x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);

                if p.x < x_intersect {
                    inside = !inside;
                }
            }
        }

        match self.fluid {
            FluidSide::Inside => inside,
            FluidSide::Outside => !inside,
        }
    }

    fn closest_point(&self, p: Point) -> Point {
        let n = self.points.len();

        assert!(n >= 2, "Polygon must have at least two points");

        let mut closest = self.points[0];
        let mut min_dist2 = f64::INFINITY;

        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];

            let candidate = Self::closest_point_on_segment(p, a, b);
            let dist2 = Self::distance2(p, candidate);

            if dist2 < min_dist2 {
                min_dist2 = dist2;
                closest = candidate;
            }
        }

        closest
    }

    fn normal(&self, p0: Point) -> Vec2 {
        let n = self.points.len();

        assert!(n >= 2, "Polygon must have at least two points");

        /*
         * Determine polygon orientation.
         *
         * CCW:
         *
         *       interior
         *          ^
         *          |
         *      a ------> b
         *
         * The left-hand normal points INTO the polygon.
         * The right-hand normal points OUT of the polygon.
         *
         * For a clockwise polygon, this is reversed.
         */
        let area2 = self.signed_area2();
        assert!(
            area2.abs() > 1e-14,
            "Degenerate polygon has zero area"
        );

        let ccw = area2 > 0.0;

        // Find the edge closest to p0.
        let mut best_edge = 0;
        let mut min_dist2 = f64::INFINITY;

        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];

            let candidate = Self::closest_point_on_segment(p0, a, b);
            let dist2 = Self::distance2(p0, candidate);

            if dist2 < min_dist2 {
                min_dist2 = dist2;
                best_edge = i;
            }
        }

        let a = self.points[best_edge];
        let b = self.points[(best_edge + 1) % n];

        let tx = b.x - a.x;
        let ty = b.y - a.y;

        let tangent_norm = tx.hypot(ty);

        assert!(
            tangent_norm > 1e-14,
            "Degenerate polygon edge"
        );

        /*
         * Left-hand normal:
         *
         *     (-ty, tx)
         *
         * Right-hand normal:
         *
         *     (ty, -tx)
         */
        let left = Vec2 {
            x: -ty / tangent_norm,
            y:  tx / tangent_norm,
        };

        let right = Vec2 {
            x: ty / tangent_norm,
            y: -tx / tangent_norm,
        };

        match self.fluid {
            /*
             * Polygon interior is fluid.
             *
             * For CCW polygon:
             *     interior = left
             *     outward fluid = right
             *
             * For CW polygon:
             *     interior = right
             *     outward fluid = left
             */
            FluidSide::Inside => {
                if ccw {
                    right
                } else {
                    left
                }
            }

            /*
             * Polygon exterior is fluid.
             *
             * For CCW polygon:
             *     exterior = right
             *     outward fluid = left
             *
             * For CW polygon:
             *     exterior = left
             *     outward fluid = right
             */
            FluidSide::Outside => {
                if ccw {
                    left
                } else {
                    right
                }
            }
        }
    }
}

// ============================================================================
// Analytic circular arc boundary geometry.
//
// CircularArc is a GEOMETRIC OVERRIDE for selected Polygon side ranges.
// It does NOT define a full 2D domain (no is_fluid), so it does not
// implement the Geometry trait. Polygon remains the authoritative domain
// representation and the fluid-mask source.
// ============================================================================

const ARC_ANGLE_TOL: f64 = 1e-12;

#[inline]
fn normalize_angle(theta: f64) -> f64 {
    theta.rem_euclid(2.0 * std::f64::consts::PI)
}

#[derive(Clone, Copy, Debug)]
pub struct CircularArc {
    pub center: Point,
    pub radius: f64,

    /// Starting polar angle in radians (canonical internal form).
    pub theta_start: f64,

    /// Signed angular sweep in radians.
    ///
    /// sweep > 0 : counter-clockwise
    /// sweep < 0 : clockwise
    pub sweep: f64,

    /// Determines the FLUID-DOMAIN outward normal direction.
    pub fluid: FluidSide,
}

impl CircularArc {
    /// Canonical low-level constructor.
    pub fn new(
        center: Point,
        radius: f64,
        theta_start: f64,
        sweep: f64,
        fluid: FluidSide,
    ) -> Self {
        assert!(radius > 0.0, "CircularArc radius must be positive");
        assert!(
            sweep.abs() > ARC_ANGLE_TOL,
            "CircularArc sweep must be nonzero"
        );
        assert!(
            sweep.abs() <= 2.0 * std::f64::consts::PI + ARC_ANGLE_TOL,
            "CircularArc sweep must not exceed 2*pi"
        );

        Self {
            center,
            radius,
            theta_start,
            sweep,
            fluid,
        }
    }

    /// Preferred high-level constructor from three arc points.
    ///
    /// start/mid/end uniquely determine center and radius (when not
    /// collinear); mid selects which of the two start->end arcs to keep.
    /// The three points are used ONLY for construction: afterwards the
    /// arc is fully described by the canonical (center, radius,
    /// theta_start, sweep, fluid) representation.
    pub fn from_three_points(
        start: Point,
        mid: Point,
        end: Point,
        fluid: FluidSide,
    ) -> Self {
        let (x1, y1) = (start.x, start.y);
        let (x2, y2) = (mid.x, mid.y);
        let (x3, y3) = (end.x, end.y);

        // Twice the signed triangle area = 2 * det, in length^2 units.
        let d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));

        let chord_scale = [
            (x1 - x2).hypot(y1 - y2),
            (x2 - x3).hypot(y2 - y3),
            (x1 - x3).hypot(y1 - y3),
        ]
        .into_iter()
        .fold(0.0f64, f64::max)
        .max(1.0);

        assert!(
            d.abs() > 1e-12 * chord_scale * chord_scale,
            "CircularArc::from_three_points: points are (nearly) collinear"
        );

        // Circumcenter.
        let r1 = x1 * x1 + y1 * y1;
        let r2 = x2 * x2 + y2 * y2;
        let r3 = x3 * x3 + y3 * y3;

        let ux = (r1 * (y2 - y3) + r2 * (y3 - y1) + r3 * (y1 - y2)) / d;
        let uy = (r1 * (x3 - x2) + r2 * (x1 - x3) + r3 * (x2 - x1)) / d;

        let center = Point { x: ux, y: uy };
        let radius = (x1 - ux).hypot(y1 - uy);

        assert!(
            radius > 0.0,
            "CircularArc::from_three_points: zero-radius arc"
        );

        let theta_start = (y1 - uy).atan2(x1 - ux);
        let theta_mid = (y2 - uy).atan2(x2 - ux);
        let theta_end = (y3 - uy).atan2(x3 - ux);

        let ccw_sweep = normalize_angle(theta_end - theta_start);
        let ccw_to_mid = normalize_angle(theta_mid - theta_start);

        let sweep = if ccw_to_mid <= ccw_sweep + ARC_ANGLE_TOL {
            ccw_sweep
        } else {
            -normalize_angle(theta_start - theta_end)
        };

        Self::new(center, radius, theta_start, sweep, fluid)
    }

    /// True if polar angle `theta` lies on the finite arc.
    pub fn contains_angle(&self, theta: f64) -> bool {
        if self.sweep > 0.0 {
            normalize_angle(theta - self.theta_start)
                <= self.sweep + ARC_ANGLE_TOL
        } else {
            normalize_angle(self.theta_start - theta)
                <= -self.sweep + ARC_ANGLE_TOL
        }
    }

    pub fn start_point(&self) -> Point {
        Point {
            x: self.center.x + self.radius * self.theta_start.cos(),
            y: self.center.y + self.radius * self.theta_start.sin(),
        }
    }

    pub fn end_point(&self) -> Point {
        let theta_end = self.theta_start + self.sweep;
        Point {
            x: self.center.x + self.radius * theta_end.cos(),
            y: self.center.y + self.radius * theta_end.sin(),
        }
    }

    /// Exact closest point on the finite arc.
    pub fn closest_point(&self, p: Point) -> Point {
        let rx = p.x - self.center.x;
        let ry = p.y - self.center.y;
        let rnorm = rx.hypot(ry);

        if rnorm > ARC_ANGLE_TOL {
            let theta = ry.atan2(rx);
            if self.contains_angle(theta) {
                let s = self.radius / rnorm;
                return Point {
                    x: self.center.x + s * rx,
                    y: self.center.y + s * ry,
                };
            }
        }

        // Radial projection falls outside the finite arc (or p is at the
        // center): fall back to the nearest endpoint.
        let start = self.start_point();
        let end = self.end_point();

        let d2s = (p.x - start.x).powi(2) + (p.y - start.y).powi(2);
        let d2e = (p.x - end.x).powi(2) + (p.y - end.y).powi(2);

        if d2s <= d2e {
            start
        } else {
            end
        }
    }

    /// FLUID-DOMAIN outward normal at a point on the arc.
    ///
    /// FluidSide::Inside  -> normal = radial
    /// FluidSide::Outside -> normal = -radial
    pub fn outward_normal(&self, p0: Point) -> Vec2 {
        let dx = p0.x - self.center.x;
        let dy = p0.y - self.center.y;
        let nrm = dx.hypot(dy);

        assert!(
            nrm > 1e-14,
            "CircularArc::outward_normal: point coincides with arc center"
        );

        let radial = Vec2 {
            x: dx / nrm,
            y: dy / nrm,
        };

        match self.fluid {
            FluidSide::Inside => radial,
            FluidSide::Outside => Vec2 {
                x: -radial.x,
                y: -radial.y,
            },
        }
    }

    /// Exact projection: P0 = closest point, n = fluid outward normal,
    /// D = (p - P0) . n, matching the existing Projection convention.
    pub fn project(&self, p: Point) -> Projection {
        let p0 = self.closest_point(p);
        let n = self.outward_normal(p0);

        let dx = p.x - p0.x;
        let dy = p.y - p0.y;

        let distance = dx * n.x + dy * n.y;

        Projection {
            point: p0,
            normal: n,
            distance,
        }
    }
}

/// Marks an inclusive range of Polygon sides as an analytic circular arc.
///
/// The Polygon sides keep their role for the domain/mask and BC ownership;
/// the arc only overrides the boundary geometry (P0, n, D).
#[derive(Clone, Debug)]
pub struct ArcOverride {
    pub side_start: usize,
    pub side_end: usize,
    pub arc: CircularArc,
}

impl ArcOverride {
    /// Inclusive side-range check; ranges do not wrap.
    #[inline(always)]
    pub fn contains_side(&self, side: usize) -> bool {
        side >= self.side_start && side <= self.side_end
    }
}

#[cfg(test)]
mod arc_tests {
    use super::*;
    use std::f64::consts::PI;

    fn cylinder_arc() -> CircularArc {
        CircularArc::from_three_points(
            Point { x: 0.0, y: -1.0 },
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
            FluidSide::Outside,
        )
    }

    #[test]
    fn three_point_construction_left_semicircle() {
        let arc = cylinder_arc();

        assert!((arc.center.x - 0.0).abs() < 1e-12);
        assert!((arc.center.y - 0.0).abs() < 1e-12);
        assert!((arc.radius - 1.0).abs() < 1e-12);

        // (-1,0) lies on the selected finite arc.
        assert!(arc.contains_angle(PI));

        // (+1,0) does NOT lie on the selected finite arc.
        assert!(!arc.contains_angle(0.0));
        assert!(!arc.contains_angle(2.0 * PI));
    }

    #[test]
    fn stagnation_projection() {
        let arc = cylinder_arc();

        let p = Point { x: -1.2, y: 0.0 };
        let proj = arc.project(p);

        assert!((proj.point.x + 1.0).abs() < 1e-12);
        assert!(proj.point.y.abs() < 1e-12);

        // Fluid is OUTSIDE the cylinder => outward fluid normal = +x.
        assert!((proj.normal.x - 1.0).abs() < 1e-12);
        assert!(proj.normal.y.abs() < 1e-12);

        // Ghost/query point is on the fluid side: D = (p-P0).n = -0.2.
        assert!((proj.distance + 0.2).abs() < 1e-12);
    }

    #[test]
    fn upper_lower_mirror_symmetry() {
        let arc = cylinder_arc();

        let p_upper = Point { x: -1.1, y: 0.2 };
        let p_lower = Point { x: -1.1, y: -0.2 };

        let a = arc.project(p_upper);
        let b = arc.project(p_lower);

        assert!((a.point.x - b.point.x).abs() < 1e-12);
        assert!((a.point.y + b.point.y).abs() < 1e-12);

        assert!((a.normal.x - b.normal.x).abs() < 1e-12);
        assert!((a.normal.y + b.normal.y).abs() < 1e-12);

        assert!((a.distance - b.distance).abs() < 1e-12);
    }

    #[test]
    fn finite_arc_endpoint_selection() {
        let arc = cylinder_arc();

        // Radial projection of p points at theta=0, which is NOT on the
        // LEFT semicircle; nearest endpoint must be (0,1) = end.
        let p = Point { x: 0.3, y: 0.7 };
        let p0 = arc.closest_point(p);

        assert!((p0.x - 0.0).abs() < 1e-12);
        assert!((p0.y - 1.0).abs() < 1e-12);

        // Mirror case selects the start endpoint (0,-1).
        let p = Point { x: 0.3, y: -0.7 };
        let p0 = arc.closest_point(p);
        assert!((p0.x - 0.0).abs() < 1e-12);
        assert!((p0.y + 1.0).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "collinear")]
    fn collinear_points_rejected() {
        let _ = CircularArc::from_three_points(
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 2.0 },
            FluidSide::Outside,
        );
    }

    #[test]
    fn arc_override_inclusive_side_range() {
        let arc = CircularArc::from_three_points(
            Point { x: 0.0, y: -1.0 },
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
            FluidSide::Outside,
        );

        let ov = ArcOverride {
            side_start: 5,
            side_end: 9,
            arc,
        };

        assert!(ov.contains_side(5));
        assert!(ov.contains_side(9));
        assert!(ov.contains_side(7));
        assert!(!ov.contains_side(4));
        assert!(!ov.contains_side(10));
    }
}
