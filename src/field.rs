use crate::bc::{BCType, BoundLoc, BoundaryCondition};
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
        self.x0 + (i as f64 + 0.5) * self.dx
    }

    /// Physical y coordinate of cell center j.
    #[inline(always)]
    pub fn y(&self, j: isize) -> f64 {
        self.y0 + (j as f64 + 0.5) * self.dy
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    grid: GridInfo,

    /// Flattened physical cells only.
    ///
    /// Layout:
    ///     linear(i,j) = i * ny + j
    ///
    /// i is the x-direction index.
    /// j is the y-direction index.
    value: Vec<State>,

    /// Boundary-condition order:
    ///
    /// [Bottom, Right, Top, Left]
    bc: [BCType; 4],
}

impl Field {
    pub fn new(
        grid: GridInfo,
        bc: [BCType; 4],
    ) -> Self {
        Self {
            value: vec![State::new(); grid.len()],
            grid,
            bc,
        }
    }

    pub fn filled(
        grid: GridInfo,
        bc: [BCType; 4],
        value: State,
    ) -> Self {
        Self {
            value: vec![value; grid.len()],
            grid,
            bc,
        }
    }

    pub fn empty_like(&self) -> Self {
        Self {
            grid: self.grid,
            value: vec![
                State::new();
                self.grid.len()
            ],
            bc: self.bc,
        }
    }

    #[inline(always)]
    pub fn nx(&self) -> usize {
        self.grid.nx
    }

    #[inline(always)]
    pub fn ny(&self) -> usize {
        self.grid.ny
    }

    #[inline(always)]
    pub fn grid(&self) -> &GridInfo {
        &self.grid
    }

    #[inline(always)]
    pub fn is_in_domain(
        &self,
        idx: (isize, isize),
    ) -> bool {
        self.grid.is_in_domain(idx)
    }

    #[inline(always)]
    fn linear_index(
        &self,
        idx: (isize, isize),
    ) -> usize {
        debug_assert!(self.is_in_domain(idx));

        let (i, j) = idx;

        i as usize * self.grid.ny
            + j as usize
    }

    #[inline(always)]
    fn get_inside(
        &self,
        idx: (isize, isize),
    ) -> State {
        self.value[self.linear_index(idx)]
    }

    #[inline(always)]
    fn boundary_location(
        &self,
        idx: (isize, isize),
    ) -> BoundLoc {
        debug_assert!(!self.is_in_domain(idx));

        let (i, j) = idx;

        if i < 0 {
            BoundLoc::Left
        } else if i >= self.grid.nx as isize {
            BoundLoc::Right
        } else if j < 0 {
            BoundLoc::Bottom
        } else {
            BoundLoc::Top
        }
    }

    /// Unified read interface.
    ///
    /// Interior indices return the physical cell directly.
    /// Out-of-domain indices are interpreted as ghost cells and
    /// delegated to the boundary condition on that side.
    ///
    /// Convention:
    ///     i -> x direction
    ///     j -> y direction
    #[inline]
    pub fn get(
        &self,
        idx: (isize, isize),
    ) -> State {
        if self.is_in_domain(idx) {
            return self.get_inside(idx);
        }

        let boundary =
            self.boundary_location(idx);

        self.bc[boundary.boundloc2idx()]
            .get_ghost(
                idx,
                &self.grid,
                &self.value,
            )
    }

    /// Write a physical-domain cell.
    ///
    /// Ghost cells are virtual and therefore cannot be written.
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

    /// Contiguous storage for solver infrastructure such as Rayon.
    ///
    /// Numerical operators (WENO/noncon/diffusion) should normally
    /// use Field::get instead.
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[State] {
        &self.value
    }

    /// Mutable contiguous storage for RK/Rayon updates.
    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [State] {
        &mut self.value
    }

    /// Convert a flattened physical-cell index back to (i,j).
    #[inline(always)]
    pub fn coords(&self, linear: usize) -> (isize, isize) {
        assert!(linear < self.value.len());

        (
            (linear / self.grid.ny) as isize,
            (linear % self.grid.ny) as isize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(value: f64) -> State {
        State {
            rho: value,
            mom_x: 2.0 * value,
            mom_y: 3.0 * value,
            ee: 4.0 * value,
            ei: 5.0 * value,
            er: 6.0 * value,
        }
    }

    fn periodic_field(nx: usize, ny: usize) -> Field {
        let grid =
            GridInfo::new(
                nx,
                ny,
                0.1,
                0.2,
                0.0,
                0.0,
            );

        Field::new(
            grid,
            [
                BCType::Periodic,
                BCType::Periodic,
                BCType::Periodic,
                BCType::Periodic,
            ],
        )
    }

    fn assert_state_close(a: State, b: State) {
        const TOL: f64 = 1e-12;

        assert!((a.rho - b.rho).abs() < TOL);
        assert!((a.mom_x - b.mom_x).abs() < TOL);
        assert!((a.mom_y - b.mom_y).abs() < TOL);
        assert!((a.ee - b.ee).abs() < TOL);
        assert!((a.ei - b.ei).abs() < TOL);
        assert!((a.er - b.er).abs() < TOL);
    }

    #[test]
    fn set_and_get_interior() {
        let mut field = periodic_field(4, 3);

        let state = make_state(2.0);

        field.set((2, 1), state);

        assert_state_close(
            field.get((2, 1)),
            state,
        );
    }

    #[test]
    fn flattened_layout_is_i_times_ny_plus_j() {
        let mut field = periodic_field(4, 3);

        for i in 0..4 {
            for j in 0..3 {
                let value =
                    (100 * i + j) as f64;

                field.set(
                    (i as isize, j as isize),
                    make_state(value),
                );
            }
        }

        for i in 0..4 {
            for j in 0..3 {
                let linear = i * 3 + j;

                assert!(
                    (
                        field.as_slice()[linear].rho
                        - (100 * i + j) as f64
                    )
                    .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn coords_is_inverse_of_flattening() {
        let field = periodic_field(5, 7);

        for i in 0..5 {
            for j in 0..7 {
                let linear = i * 7 + j;

                assert_eq!(
                    field.coords(linear),
                    (i as isize, j as isize),
                );
            }
        }
    }

    #[test]
    fn periodic_get_handles_negative_indices() {
        let mut field = periodic_field(4, 4);

        let target = make_state(7.0);
        field.set((3, 2), target);

        assert_state_close(
            field.get((-1, 2)),
            target,
        );
    }

    #[test]
    #[should_panic(expected = "cannot write ghost cell")]
    fn set_rejects_ghost_cell() {
        let mut field = periodic_field(4, 4);

        field.set(
            (-1, 0),
            make_state(1.0),
        );
    }

    #[test]
    fn cell_center_coordinates() {
        let field = periodic_field(4, 4);

        assert!(
            (field.grid().x(0) - 0.05).abs()
                < 1e-12
        );

        assert!(
            (field.grid().y(0) - 0.10).abs()
                < 1e-12
        );
    }
}
