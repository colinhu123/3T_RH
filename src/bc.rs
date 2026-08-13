use crate::field::GridInfo;
use crate::state::State;

pub trait BoundaryCondition {
    fn get_ghost(
        &self,
        idx: (isize, isize),
        grid: &GridInfo,
        field: &[State],
    ) -> State;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoundLoc {
    Bottom,
    Top,
    Left,
    Right,
}

impl BoundLoc {
    /// Boundary-condition array layout:
    /// [Bottom, Right, Top, Left]
    #[inline(always)]
    pub const fn boundloc2idx(self) -> usize {
        match self {
            BoundLoc::Bottom => 0,
            BoundLoc::Right => 1,
            BoundLoc::Top => 2,
            BoundLoc::Left => 3,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BCType {
    /// Mirror the state without changing momentum.
    Symmetry,
    Wall,
    Periodic,
    Constant(State),
}

#[inline(always)]
fn linear_index(grid: &GridInfo, idx: (isize, isize)) -> usize {
    debug_assert!(grid.is_in_domain(idx));

    let (i, j) = idx;
    i as usize * grid.ny + j as usize
}

#[inline(always)]
fn get_inside(
    field: &[State],
    grid: &GridInfo,
    idx: (isize, isize),
) -> State {
    field[linear_index(grid, idx)]
}

#[inline(always)]
fn boundary_location(idx: (isize, isize), grid: &GridInfo) -> BoundLoc {
    if idx.0 < 0 {
        BoundLoc::Left
    } else if idx.0 >= grid.nx as isize {
        BoundLoc::Right
    } else if idx.1 < 0 {
        BoundLoc::Bottom
    } else {
        BoundLoc::Top
    }
}

#[inline(always)]
fn mirrored_index(
    idx: (isize, isize),
    grid: &GridInfo,
    boundary: BoundLoc,
) -> (isize, isize) {
    match boundary {
        BoundLoc::Left => (-idx.0 - 1, idx.1),

        BoundLoc::Right => (
            2 * grid.nx as isize - 1 - idx.0,
            idx.1,
        ),

        BoundLoc::Bottom => (
            idx.0,
            -idx.1 - 1,
        ),

        BoundLoc::Top => (
            idx.0,
            2 * grid.ny as isize - 1 - idx.1,
        ),
    }
}

impl BoundaryCondition for BCType {
    fn get_ghost(
        &self,
        idx: (isize, isize),
        grid: &GridInfo,
        field: &[State],
    ) -> State {
        debug_assert!(!grid.is_in_domain(idx));

        let boundary = boundary_location(idx, grid);

        match self {
            BCType::Periodic => {
                let periodic_idx = (
                    idx.0.rem_euclid(grid.nx as isize),
                    idx.1.rem_euclid(grid.ny as isize),
                );

                get_inside(field, grid, periodic_idx)
            }

            BCType::Constant(value) => *value,

            BCType::Symmetry => {
                let mirrored = mirrored_index(idx, grid, boundary);
                get_inside(field, grid, mirrored)
            }

            BCType::Wall => {
                let mirrored = mirrored_index(idx, grid, boundary);
                let mut state = get_inside(field, grid, mirrored);

                match boundary {
                    BoundLoc::Left | BoundLoc::Right => {
                        state.mom_x = -state.mom_x;
                    }

                    BoundLoc::Bottom | BoundLoc::Top => {
                        state.mom_y = -state.mom_y;
                    }
                }

                state
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid() -> GridInfo {
        GridInfo::new(4, 4, 0.1, 0.1, 0.0, 0.0)
    }

    fn make_state(i: usize, j: usize) -> State {
        State {
            rho: (100 * i + j) as f64,
            mom_x: i as f64 + 1.0,
            mom_y: j as f64 + 1.0,
            ee: 1.0,
            ei: 2.0,
            er: 3.0,
        }
    }

    fn make_field(grid: &GridInfo) -> Vec<State> {
        let mut field = vec![State::new(); grid.nx * grid.ny];

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                field[i * grid.ny + j] = make_state(i, j);
            }
        }

        field
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
    fn periodic_left_wraps_to_right() {
        let grid = make_grid();
        let field = make_field(&grid);

        let ghost = BCType::Periodic.get_ghost((-1, 1), &grid, &field);
        let expected = get_inside(&field, &grid, (3, 1));

        assert_state_close(ghost, expected);
    }

    #[test]
    fn periodic_corner_wraps_both_axes() {
        let grid = make_grid();
        let field = make_field(&grid);

        let ghost = BCType::Periodic.get_ghost((-1, -1), &grid, &field);
        let expected = get_inside(&field, &grid, (3, 3));

        assert_state_close(ghost, expected);
    }

    #[test]
    fn symmetry_left_mirrors_state() {
        let grid = make_grid();
        let field = make_field(&grid);

        let ghost = BCType::Symmetry.get_ghost((-2, 2), &grid, &field);
        let expected = get_inside(&field, &grid, (1, 2));

        assert_state_close(ghost, expected);
    }

    #[test]
    fn wall_left_reverses_x_momentum() {
        let grid = make_grid();
        let field = make_field(&grid);

        let ghost = BCType::Wall.get_ghost((-1, 2), &grid, &field);
        let interior = get_inside(&field, &grid, (0, 2));

        assert!((ghost.rho - interior.rho).abs() < 1e-12);
        assert!((ghost.mom_x + interior.mom_x).abs() < 1e-12);
        assert!((ghost.mom_y - interior.mom_y).abs() < 1e-12);
        assert!((ghost.ee - interior.ee).abs() < 1e-12);
        assert!((ghost.ei - interior.ei).abs() < 1e-12);
        assert!((ghost.er - interior.er).abs() < 1e-12);
    }

    #[test]
    fn wall_bottom_reverses_y_momentum() {
        let grid = make_grid();
        let field = make_field(&grid);

        let ghost = BCType::Wall.get_ghost((2, -1), &grid, &field);
        let interior = get_inside(&field, &grid, (2, 0));

        assert!((ghost.rho - interior.rho).abs() < 1e-12);
        assert!((ghost.mom_x - interior.mom_x).abs() < 1e-12);
        assert!((ghost.mom_y + interior.mom_y).abs() < 1e-12);
    }

    #[test]
    fn constant_returns_fixed_state() {
        let grid = make_grid();
        let field = make_field(&grid);

        let target = State {
            rho: 2.5,
            mom_x: 0.5,
            mom_y: -1.0,
            ee: 7.0,
            ei: 8.0,
            er: 9.0,
        };

        let ghost = BCType::Constant(target).get_ghost((-3, 0), &grid, &field);

        assert_state_close(ghost, target);
    }
}
