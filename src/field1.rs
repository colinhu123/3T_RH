use crate::bc1;
use crate::geometry::{self, FluidSide, Geometry, Point, Polygon};
use crate::state::State;

#[derive(Clone, Copy, Debug)]
pub struct GridInfo {
    pub nx: usize,
    pub ny: usize,

    pub dx: f64,
    pub dy: f64,

    pub x0: f64,
    pub y0: f64,
}

impl GridInfo {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, x0: f64, y0: f64) -> Self {
        assert!(nx > 0, "nx must be greater than zero");
        assert!(ny > 0, "ny must be greater than zero");
        assert!(dx > 0.0, "dx must be greater than zero");
        assert!(dy > 0.0, "dy must be greater than zero");

        Self {
            nx,
            ny,
            dx,
            dy,
            x0,
            y0,
        }
    }

    #[inline(always)]
    pub fn is_in_domain(&self, idx: (isize, isize)) -> bool {
        let (i, j) = idx;

        i >= 0 && i < self.nx as isize && j >= 0 && j < self.ny as isize
    }

    /// Physical x coordinate of cell center i.
    #[inline(always)]
    pub fn x(&self, i: isize) -> f64 {
        self.x0 + (i as f64) * self.dx
    }

    /// Physical y coordinate of cell center j.
    #[inline(always)]
    pub fn y(&self, j: isize) -> f64 {
        self.y0 + (j as f64) * self.dy
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    pub fn coord2idx(&self, p: Point) -> (isize, isize) {
        let i = ((p.x - self.x0) / self.dx).round() as isize;
        let j = ((p.y - self.y0) / self.dy).round() as isize;
        (i, j)
    }
}

/// Derive the fluid-domain classifier polygon from a list of analytic
/// boundary elements (the single source of boundary definition).
///
/// `polygonize_boundary` samples lines / arcs / circles so that the derived
/// polygon reproduces the fluid mask; the analytic elements remain the
/// authoritative geometry for ghost reconstruction.
fn classifier_polygon(elements: &[bc1::BoundaryElement], chord_max: f64, fluid: FluidSide) -> Polygon {
    let geoms: Vec<geometry::BoundaryGeometry> = elements.iter().map(|e| e.geometry).collect();
    geometry::polygonize_boundary(&geoms, chord_max, fluid)
}

pub struct Field {
    pub grid: GridInfo,
    pub value: Vec<State>,

    /// Derived outer-domain classifier polygon (fluid inside). Generated
    /// from `outer_boundary`; kept only for mask / Outer-vs-Inner
    /// classification, never for ghost geometry.
    pub outer_bound: Polygon,

    /// Derived obstacle classifier polygon (fluid outside), if any.
    /// `None` when there is no inner boundary (no manual dummy needed).
    pub inner_bound: Option<Polygon>,

    pub time: f64,
    /// Fluid mask over the Cartesian grid (linear index = i*ny + j),
    /// computed once at construction. `is_in_domain` reads this mask.
    pub fluid: Vec<bool>,

    /// Analytic physical boundary elements (ONE geometry + ONE BC each).
    /// These are the authoritative physical boundary geometry for ghosts.
    pub outer_boundary: Vec<bc1::BoundaryElement>,
    pub inner_boundary: Vec<bc1::BoundaryElement>,
}

impl Field {
    /// Construct a Field from the authoritative analytic boundaries.
    ///
    /// The classifier polygons and the fluid mask are derived internally:
    /// this is the only boundary description callers must provide.
    pub fn from_boundaries(
        grid: GridInfo,
        outer: Vec<bc1::BoundaryElement>,
        inner: Vec<bc1::BoundaryElement>,
        value: State,
        time: f64,
    ) -> Self {
        // Chord target keeps the sampled classifier within a quarter cell
        // of the exact analytic geometry, so boundary-adjacent cell centers
        // keep the same fluid classification.
        let chord_max = 0.25 * grid.dx.min(grid.dy);

        let outer_bound = classifier_polygon(&outer, chord_max, FluidSide::Inside);
        let inner_bound = if inner.is_empty() {
            None
        } else {
            Some(classifier_polygon(&inner, chord_max, FluidSide::Outside))
        };

        Self::from_parts(grid, outer_bound, inner_bound, outer, inner, value, time)
    }

    /// Construct a Field from explicitly supplied classifier polygons and
    /// analytic boundary elements.
    ///
    /// Used internally by [`Field::from_boundaries`] and by unit tests that
    /// need a hand-built classifier polygon (thin strips, corner boxes,
    /// oblique walls). The analytic elements remain the authoritative
    /// boundary geometry for ghost reconstruction.
    pub(crate) fn from_parts(
        grid: GridInfo,
        outer_bound: Polygon,
        inner_bound: Option<Polygon>,
        outer_elements: Vec<bc1::BoundaryElement>,
        inner_elements: Vec<bc1::BoundaryElement>,
        value: State,
        time: f64,
    ) -> Self {
        assert!(
            outer_bound.fluid == FluidSide::Inside,
            "Wrong outer classifier setting"
        );
        if let Some(inner) = &inner_bound {
            assert!(
                inner.fluid == FluidSide::Outside,
                "Wrong inner classifier setting"
            );
        }

        let nx = grid.nx;
        let ny = grid.ny;

        let fluid = (0..nx * ny)
            .map(|linear| {
                let i = linear / ny;
                let j = linear % ny;
                let p = Point {
                    x: grid.x(i as isize),
                    y: grid.y(j as isize),
                };
                outer_bound.is_fluid(p)
                    && inner_bound
                        .as_ref()
                        .map_or(true, |poly| poly.is_fluid(p))
            })
            .collect();

        Self {
            grid,
            value: vec![value; grid.len()],
            outer_bound,
            inner_bound,
            time,
            fluid,
            outer_boundary: outer_elements,
            inner_boundary: inner_elements,
        }
    }

    /// Scratch field with the same geometry and fluid mask, but zeroed
    /// values. Does not re-run the mask construction.
    pub fn empty_like(&self) -> Self {
        Self {
            grid: self.grid,
            value: vec![State::new(); self.grid.len()],
            outer_bound: self.outer_bound.clone(),
            inner_bound: self.inner_bound.clone(),
            time: self.time,
            fluid: self.fluid.clone(),
            outer_boundary: self.outer_boundary.clone(),
            inner_boundary: self.inner_boundary.clone(),
        }
    }

    /// True if `p` is fluid with respect to the outer domain classifier.
    #[inline(always)]
    pub fn outer_contains(&self, p: Point) -> bool {
        self.outer_bound.is_fluid(p)
    }

    /// True if `p` is not cut out by an inner obstacle (always true when
    /// there is no inner boundary).
    #[inline(always)]
    pub fn inner_contains(&self, p: Point) -> bool {
        self.inner_bound
            .as_ref()
            .map_or(true, |poly| poly.is_fluid(p))
    }

    /// Fluid if inside the outer domain AND not inside an obstacle.
    #[inline(always)]
    pub fn is_fluid_point(&self, p: Point) -> bool {
        self.outer_contains(p) && self.inner_contains(p)
    }

    pub fn is_in_domain(&self, idx: (isize, isize)) -> bool {
        if !self.grid.is_in_domain(idx) {
            let p = Point {
                x: self.grid.x(idx.0),
                y: self.grid.y(idx.1),
            };
            return self.is_fluid_point(p);
        }

        let i = idx.0 as usize;
        let j = idx.1 as usize;
        self.fluid[i * self.grid.ny + j]
    }

    #[inline(always)]
    fn get_inside(&self, idx: (isize, isize)) -> State {
        debug_assert!(self.is_in_domain(idx));
        let i = idx.0 as usize;
        let j = idx.1 as usize;
        self.value[i * self.grid.ny + j]
    }
    #[inline(always)]
    pub fn linear_index(&self, idx: (isize, isize)) -> usize {
        debug_assert!(self.is_in_domain(idx));

        let (i, j) = idx;

        i as usize * self.grid.ny + j as usize
    }

    #[inline]
    pub fn set(&mut self, idx: (isize, isize), value: State) {
        assert!(self.is_in_domain(idx), "cannot write ghost cell {:?}", idx);

        let linear = self.linear_index(idx);

        self.value[linear] = value;
    }

    #[inline(always)]
    pub(crate) fn _as_mut_slice(&mut self) -> &mut [State] {
        &mut self.value
    }

    pub fn get(&self, idx: (isize, isize)) -> State {
        // ============================================================
        // Interior fluid point
        // ============================================================

        if self.is_in_domain(idx) {
            return self.get_inside(idx);
        }

        // ============================================================
        // Ghost point
        // ============================================================

        let p = Point {
            x: self.grid.x(idx.0),
            y: self.grid.y(idx.1),
        };

        // ------------------------------------------------------------
        // Determine outer / inner boundary.
        //
        // The derived Polygon only classifies the domain here; the
        // physical boundary geometry comes from the analytic
        // BoundaryElements.
        //
        //     outside the outer domain   -> outer boundary
        //     inside the interior cutout -> inner boundary
        //
        // A ghost point must satisfy exactly one of the two, so the
        // remaining case (fluid by both) is a bug.
        // ------------------------------------------------------------

        let outer_fluid = self.outer_contains(p);

        let inner_fluid = self.inner_contains(p);

        let (boundary, elements) = if !outer_fluid {
            (crate::ghost::BoundaryKind::Outer, &self.outer_boundary)
        } else if !inner_fluid {
            (crate::ghost::BoundaryKind::Inner, &self.inner_boundary)
        } else {
            panic!(
                "Field::get: ghost point p=({:.6e},{:.6e}) is classified fluid \
             by BOTH outer and inner domains",
                p.x, p.y,
            );
        };

        // Analytic boundary lookup: nearest BoundaryElement wins; BC
        // priority (Wall > ... > FarField) breaks geometric ties at
        // junctions. Returns the exact analytic Projection {P0, n, D}.
        let (boundary_id, project) = bc1::find_boundary_element(p, elements);

        // ------------------------------------------------------------
        // Reconstruct ghost using the selected analytic element's BC.
        // ------------------------------------------------------------

        bc1::set_ghost_point_value(idx, project, boundary, boundary_id, self, None)
    }
}

