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
    pub fn new(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        x0: f64,
        y0: f64,
    ) -> Self {
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

        i >= 0
            && i < self.nx as isize
            && j >= 0
            && j < self.ny as isize
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

    pub fn coord2idx(&self,p: Point)-> (isize, isize) {
        let i = ((p.x - self.x0)/self.dx).round() as isize;
        let j = ((p.y - self.y0)/self.dy).round() as isize;
        (i,j)
    }
}

pub struct Field {
    pub grid: GridInfo,
    pub value: Vec<State>,
    pub outer_bound: Polygon,
    pub inner_bound: Polygon,
    pub bc_inner: Vec<BCType>,
    pub bc_outer: Vec<BCType>,
    pub time: f64,
    /// Fluid mask over the Cartesian grid (linear index = i*ny + j),
    /// computed once at construction. `is_in_domain` reads this mask.
    pub fluid: Vec<bool>,
    /// Analytic circular-arc overrides for selected outer/inner polygon
    /// side ranges. The Polygon remains authoritative for the domain and
    /// the fluid mask; the arcs only override boundary geometry
    /// (P0, normal, distance) for ghosts attached to those sides.
    pub outer_arcs: Vec<geometry::ArcOverride>,
    pub inner_arcs: Vec<geometry::ArcOverride>,
}

impl Field {
    pub fn new(
        grid:GridInfo,
        bc_inner: Vec<BCType>,
        bc_outer: Vec<BCType>,
        value: State,
        outer_bound: Polygon,
        inner_bound: Polygon,
        time: f64,
    ) -> Self {
        assert!(outer_bound.fluid == FluidSide::Inside, "Wrong outer boundary setting");
        assert!(inner_bound.fluid == FluidSide::Outside, "Wrong inner boundary setting");
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
            grid:grid,
            value: vec![value; grid.len()],
            outer_bound: outer_bound,
            inner_bound: inner_bound,
            bc_inner: bc_inner,
            bc_outer: bc_outer,
            time: time,
            fluid: fluid,
            outer_arcs: Vec::new(),
            inner_arcs: Vec::new(),
        }
    }

    /// Scratch field with the same geometry, BC lists and fluid mask,
    /// but zeroed values. Does not re-run the polygon point-in-polygon
    /// mask construction.
    pub fn empty_like(&self) -> Self {
        Self {
            grid: self.grid,
            value: vec![State::new(); self.grid.len()],
            outer_bound: Polygon::new(
                self.outer_bound.points.clone(),
                self.outer_bound.fluid,
            ),
            inner_bound: Polygon::new(
                self.inner_bound.points.clone(),
                self.inner_bound.fluid,
            ),
            bc_inner: self.bc_inner.clone(),
            bc_outer: self.bc_outer.clone(),
            time: self.time,
            fluid: self.fluid.clone(),
            outer_arcs: self.outer_arcs.clone(),
            inner_arcs: self.inner_arcs.clone(),
        }
    }

    /// Find the analytic CircularArc overriding the given polygon side,
    /// if any. Linear search: this is a cold/build-time operation.
    pub fn arc_for_side(
        &self,
        boundary: crate::ghost::BoundaryKind,
        side_id: usize,
    ) -> Option<&geometry::CircularArc> {
        let arcs = match boundary {
            crate::ghost::BoundaryKind::Outer => &self.outer_arcs,
            crate::ghost::BoundaryKind::Inner => &self.inner_arcs,
        };

        arcs.iter()
            .find(|ov| ov.contains_side(side_id))
            .map(|ov| &ov.arc)
    }

    pub fn is_in_domain(&self, idx: (isize, isize))-> bool {
        if !self.grid.is_in_domain(idx) {
            let x= self.grid.x(idx.0);
            let y = self.grid.y(idx.1);
            let p = Point {x: x, y: y};
            let con1 = self.outer_bound.is_fluid(p);
            let con2 = self.inner_bound.is_fluid(p);
            return con1 && con2;
        }

        let i = idx.0 as usize;
        let j = idx.1 as usize;
        self.fluid[i*self.grid.ny + j]
    }

    #[inline(always)]
    fn get_inside(
        &self,
        idx: (isize, isize),
    )-> State {
        debug_assert!(self.is_in_domain(idx));
        let i = idx.0 as usize;
        let j = idx.1 as usize;
        self.value[i*self.grid.ny + j]
    }
    #[inline(always)]
    pub fn linear_index(
        &self,
        idx: (isize, isize),
    ) -> usize {
        debug_assert!(self.is_in_domain(idx));

        let (i, j) = idx;

        i as usize * self.grid.ny
            + j as usize
    }

    #[inline]
    pub fn set(
        &mut self,
        idx: (isize, isize),
        value: State,
    ) {
        assert!(
            self.is_in_domain(idx),
            "cannot write ghost cell {:?}",
            idx
        );

        let linear =
            self.linear_index(idx);

        self.value[linear] = value;
    }

    #[inline(always)]
    pub(crate) fn _as_mut_slice(&mut self) -> &mut [State] {
        &mut self.value
    }

    pub fn get(
    &self,
    idx: (isize, isize),
) -> State {
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
    // ------------------------------------------------------------

    let outer_fluid =
        self.outer_bound.is_fluid(p);

    let (polygon, bc_list) =
        if !outer_fluid {
            (
                &self.outer_bound,
                &self.bc_outer,
            )
        } else {
            (
                &self.inner_bound,
                &self.bc_inner,
            )
        };

    let boundary =
    if !outer_fluid {
        crate::ghost::BoundaryKind::Outer
    } else {
        crate::ghost::BoundaryKind::Inner
    };

    // ------------------------------------------------------------
    // First obtain the closest boundary point P0.
    //
    // Do not trust raw_project.normal at a polygon vertex yet.
    // ------------------------------------------------------------

    let raw_project =
        geometry::project(
            polygon,
            p,
        );

    // ------------------------------------------------------------
    // Find ALL sides containing P0 and select according to BC
    // priority.
    //
    // In particular:
    //
    //       Wall > FarField
    //
    // so the two cylinder junctions behave identically.
    // ------------------------------------------------------------

    let side_id =
        crate::ghost::select_boundary_side(
            raw_project.point,
            polygon,
            bc_list,
        );

    // ------------------------------------------------------------
    // IMPORTANT:
    //
    // Final projection: if the selected polygon side belongs to an
    // analytic CircularArc override, use the exact arc geometry
    // (closest point, continuous normal, signed distance).
    //
    // Otherwise recompute the normal from the SELECTED side.
    //
    // Otherwise at a polygon vertex:
    //
    //       BC may come from Wall side
    //       normal may come from FarField side
    //
    // which is inconsistent.
    // ------------------------------------------------------------

    let project =
    if let Some(arc) =
        self.arc_for_side(boundary, side_id)
    {
        arc.project(p)
    }
    else
    {
        let normal =
            polygon.outward_normal_of_side(
                side_id,
            );

        let dx =
            p.x - raw_project.point.x;

        let dy =
            p.y - raw_project.point.y;

        let distance =
            dx * normal.x
            + dy * normal.y;

        geometry::Projection {
            point: raw_project.point,
            normal,
            distance,
        }
    };

    // ------------------------------------------------------------
    // Reconstruct ghost using the BC side that WE selected.
    // ------------------------------------------------------------

    bc1::set_ghost_point_value(
        idx,
        project,
        boundary,
        side_id,
        self,
        None,
    )
}

}