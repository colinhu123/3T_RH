
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
