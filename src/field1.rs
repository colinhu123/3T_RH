use crate::bc1::{self, BCType};
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

pub struct Field {
    pub grid: GridInfo,
    pub value: Vec<State>,
    pub outer_bound: Polygon,
    pub inner_bound: Polygon,
    // LEGACY / domain-compat: BC lists keyed by Polygon side. After the
    // analytic-boundary refactor these are no longer authoritative for
    // ghost BCs (use outer_boundary / inner_boundary instead). Kept
    // because Field::new() and existing initializers still build them.
    pub bc_inner: Vec<BCType>,
    pub bc_outer: Vec<BCType>,
    pub time: f64,
    /// Fluid mask over the Cartesian grid (linear index = i*ny + j),
    /// computed once at construction. `is_in_domain` reads this mask.
    pub fluid: Vec<bool>,
    /// Analytic physical boundary elements (ONE geometry + ONE BC each).
    /// These are the authoritative physical boundary geometry for ghosts;
    /// the Polygon above is only the domain classifier / fluid-mask source.
    pub outer_boundary: Vec<bc1::BoundaryElement>,
    pub inner_boundary: Vec<bc1::BoundaryElement>,
}

impl Field {
    pub fn new(
        grid: GridInfo,
        bc_inner: Vec<BCType>,
        bc_outer: Vec<BCType>,
        value: State,
        outer_bound: Polygon,
        inner_bound: Polygon,
        time: f64,
    ) -> Self {
        assert!(
            outer_bound.fluid == FluidSide::Inside,
            "Wrong outer boundary setting"
        );
        assert!(
            inner_bound.fluid == FluidSide::Outside,
            "Wrong inner boundary setting"
        );
        assert!(bc_inner.len() == inner_bound.points.len());
        assert!(bc_outer.len() == outer_bound.points.len());

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
                outer_bound.is_fluid(p) && inner_bound.is_fluid(p)
            })
            .collect();

        Self {
            grid: grid,
            value: vec![value; grid.len()],
            outer_bound: outer_bound,
            inner_bound: inner_bound,
            bc_inner: bc_inner,
            bc_outer: bc_outer,
            time: time,
            fluid: fluid,
            outer_boundary: Vec::new(),
            inner_boundary: Vec::new(),
        }
    }

    /// Scratch field with the same geometry, BC lists and fluid mask,
    /// but zeroed values. Does not re-run the polygon point-in-polygon
    /// mask construction.
    pub fn empty_like(&self) -> Self {
        Self {
            grid: self.grid,
            value: vec![State::new(); self.grid.len()],
            outer_bound: Polygon::new(self.outer_bound.points.clone(), self.outer_bound.fluid),
            inner_bound: Polygon::new(self.inner_bound.points.clone(), self.inner_bound.fluid),
            bc_inner: self.bc_inner.clone(),
            bc_outer: self.bc_outer.clone(),
            time: self.time,
            fluid: self.fluid.clone(),
            outer_boundary: self.outer_boundary.clone(),
            inner_boundary: self.inner_boundary.clone(),
        }
    }

    pub fn is_in_domain(&self, idx: (isize, isize)) -> bool {
        if !self.grid.is_in_domain(idx) {
            let x = self.grid.x(idx.0);
            let y = self.grid.y(idx.1);
            let p = Point { x: x, y: y };
            let con1 = self.outer_bound.is_fluid(p);
            let con2 = self.inner_bound.is_fluid(p);
            return con1 && con2;
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
        // The Polygon only classifies the domain here; the physical
        // boundary geometry comes from the analytic BoundaryElements.
        //
        // Both classifications are explicit:
        //
        //     outside the outer rectangle   -> outer boundary
        //     inside the interior obstacle  -> inner boundary
        //
        // A ghost point must satisfy exactly one of the two, so the
        // remaining case (fluid by both polygons) is a bug.
        // ------------------------------------------------------------

        let outer_fluid = self.outer_bound.is_fluid(p);

        let inner_fluid = self.inner_bound.is_fluid(p);

        let (boundary, elements) = if !outer_fluid {
            (crate::ghost::BoundaryKind::Outer, &self.outer_boundary)
        } else if !inner_fluid {
            (crate::ghost::BoundaryKind::Inner, &self.inner_boundary)
        } else {
            panic!(
                "Field::get: ghost point p=({:.6e},{:.6e}) is classified fluid \
             by BOTH polygons (outer and inner)",
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
