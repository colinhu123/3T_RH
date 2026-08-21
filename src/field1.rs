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
        Self {
            grid:grid,
            value: vec![value; grid.len()],
            outer_bound: outer_bound,
            inner_bound: inner_bound,
            bc_inner: bc_inner,
            bc_outer: bc_outer,
            time: time,
        }
    }

    pub fn is_in_domain(&self, idx: (isize, isize))-> bool {
        let x= self.grid.x(idx.0);
        let y = self.grid.y(idx.1);
        let p = Point {x: x, y: y};
        let con1 = self.outer_bound.is_fluid(p);
        let con2 = self.inner_bound.is_fluid(p);
        if con1 && con2 {
            true
        }
        else {
            false
        }
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
    )-> State {
        if self.is_in_domain(idx) {
            return self.get_inside(idx)

        } else {
            let x= self.grid.x(idx.0);
            let y = self.grid.y(idx.1);
            let p = Point {x: x, y: y};
            let con1 = self.outer_bound.is_fluid(p);
            //let con2 = self.inner_bound.is_fluid(p);

            let project = if con1 == false {geometry::project(&self.outer_bound, p)} else {
                geometry::project(&self.inner_bound, p)
            };

            bc1::set_ghost_point_value(idx, project, self)
            //Space prepared for WENO extrapolation and ILW
            //State::new()
        }
    }

}