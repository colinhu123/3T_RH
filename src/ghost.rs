use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use crate::bc1;
use crate::field1::Field;
use crate::geometry::{self, Geometry, Projection};
use crate::state::State;

pub type Offset = (isize, isize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    Outer,
    Inner,
}

#[derive(Clone, Debug)]
pub struct GhostInfo {
    pub idx: (isize, isize),
    pub project: Projection,
    pub nearest_idx: (isize, isize),
    pub boundary: BoundaryKind,
    pub side_id: Option<usize>,
}

#[derive(Debug)]
pub struct GhostGrid {
    pub info: Vec<GhostInfo>,
    pub values: Vec<State>,
    lookup: HashMap<(isize, isize), usize>,
}

impl GhostGrid {
    /// Build the static ghost layout once from the actual solver stencil.
    pub fn build(field: &Field, offsets: &[Offset]) -> Self {
        let indices = discover_ghost_indices(field, offsets);
        let mut info = Vec::with_capacity(indices.len());
        let mut lookup = HashMap::with_capacity(indices.len());

        for idx in indices {
            let id = info.len();
            let g = build_ghost_info(field, idx);
            assert!(lookup.insert(idx, id).is_none());
            info.push(g);
        }

        let values = vec![State::new(); info.len()];
        Self { info, values, lookup }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.info.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.info.is_empty()
    }

    #[inline(always)]
    pub fn contains(&self, idx: (isize, isize)) -> bool {
        self.lookup.contains_key(&idx)
    }

    #[inline(always)]
    pub fn id(&self, idx: (isize, isize)) -> Option<usize> {
        self.lookup.get(&idx).copied()
    }

    /// Read a value that has already been computed for this RK stage.
    #[inline(always)]
    pub fn get(&self, idx: (isize, isize)) -> State {
        let id = self.lookup.get(&idx).copied().unwrap_or_else(|| {
            panic!(
                "ghost {:?} is not cached; register all solver stencil offsets",
                idx
            )
        });
        self.values[id]
    }

    /// Unified O(1)-ish access for stencil extraction.
    #[inline(always)]
    pub fn get_with_field(&self, field: &Field, idx: (isize, isize)) -> State {
        if field.is_in_domain(idx) {
            field.value[field.linear_index(idx)]
        } else {
            self.get(idx)
        }
    }

    /// Correctness-first implementation: calculate every unique ghost once.
    pub fn update_values(&mut self, field: &Field, t: f64) {
        for id in 0..self.info.len() {
            let g = &self.info[id];
            self.values[id] =
                bc1::set_ghost_point_value(g.idx, g.project, field);
        }
    }

    /// Parallel ghost update.
///
/// Each ghost is reconstructed once for the current RK stage.
/// Fail immediately if bc1 produces a non-finite ghost state.
pub fn update_values_parallel(&mut self, field: &Field, t: f64) {
    let info = &self.info;

    self.values
        .par_iter_mut()
        .enumerate()
        .for_each(|(id, value)| {
            let g = &info[id];

            let result =
                bc1::set_ghost_point_value(
                    g.idx,
                    g.project,
                    field,
                );

            // ----------------------------------------------------
            // Diagnostic: bc1 must NEVER return NaN / Inf.
            // ----------------------------------------------------
            if !result.rho.is_finite()
                || !result.mom_x.is_finite()
                || !result.mom_y.is_finite()
                || !result.ee.is_finite()
                || !result.ei.is_finite()
                || !result.er.is_finite()
            {
                panic!(
                    "\n\
                     ========================================\n\
                     NON-FINITE GHOST STATE\n\
                     ========================================\n\
                     ghost id   = {}\n\
                     ghost idx  = {:?}\n\
                     P0         = ({:.16e}, {:.16e})\n\
                     normal     = ({:.16e}, {:.16e})\n\
                     distance   = {:.16e}\n\
                     nearest    = {:?}\n\
                     state      = {:?}\n\
                     ========================================\n",
                    id,
                    g.idx,
                    g.project.point.x,
                    g.project.point.y,
                    g.project.normal.x,
                    g.project.normal.y,
                    g.project.distance,
                    g.nearest_idx,
                    result,
                );
            }

            *value = result;
        });
}

    pub fn print_summary(&self) {
        let outer = self.info.iter()
            .filter(|g| g.boundary == BoundaryKind::Outer)
            .count();
        println!(
            "GhostGrid: total={}, outer={}, inner={}",
            self.info.len(),
            outer,
            self.info.len() - outer
        );
    }
}

/// A point is cached iff a real fluid cell's registered stencil references it
/// and the referenced point is outside the fluid domain.
pub fn discover_ghost_indices(field: &Field, offsets: &[Offset])
    -> Vec<(isize, isize)>
{
    let mut set = HashSet::<(isize, isize)>::new();

    for i in 0..field.grid.nx as isize {
        for j in 0..field.grid.ny as isize {
            let center = (i, j);
            if !field.is_in_domain(center) {
                continue;
            }

            for &(di, dj) in offsets {
                let idx = (i + di, j + dj);
                if !field.is_in_domain(idx) {
                    set.insert(idx);
                }
            }
        }
    }

    let mut result: Vec<_> = set.into_iter().collect();
    result.sort_unstable();
    result
}

fn build_ghost_info(field: &Field, idx: (isize, isize)) -> GhostInfo {
    let p = geometry::Point {
        x: field.grid.x(idx.0),
        y: field.grid.y(idx.1),
    };

    // Compatible with the current Field layout:
    // outside outer polygon -> outer boundary;
    // otherwise             -> inner boundary.
    let outer_fluid = field.outer_bound.is_fluid(p);
    let (boundary, polygon) = if !outer_fluid {
        (BoundaryKind::Outer, &field.outer_bound)
    } else {
        (BoundaryKind::Inner, &field.inner_bound)
    };

    let project = geometry::project(polygon, p);
    let nearest_idx = bc1::find_nearest_grid_point(project, field);
    let side_id = geometry::find_boundary_side(
        project.point,
        polygon,
        crate::constant::DEFAULT_EPS,
    );

    GhostInfo {
        idx,
        project,
        nearest_idx,
        boundary,
        side_id,
    }
}

/// Union of the current dimension-by-dimension stencils.
/// noncon reaches +/-4, so this cross covers WENO, diffusion and noncon.
pub fn default_stencil_offsets() -> Vec<Offset> {
    let mut offsets = Vec::new();

    for d in -4isize..=4isize {
        offsets.push((d, 0));
        offsets.push((0, d));
    }

    offsets.sort_unstable();
    offsets.dedup();
    offsets
}

/// Utility for future operators with additional stencil shapes.
pub fn union_offsets(groups: &[&[Offset]]) -> Vec<Offset> {
    let mut set = HashSet::new();
    for group in groups {
        for &offset in *group {
            set.insert(offset);
        }
    }

    let mut result: Vec<_> = set.into_iter().collect();
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_offsets_are_radius_four_cross() {
        let offsets = default_stencil_offsets();
        assert_eq!(offsets.len(), 17);
        assert!(offsets.contains(&(-4, 0)));
        assert!(offsets.contains(&(4, 0)));
        assert!(offsets.contains(&(0, -4)));
        assert!(offsets.contains(&(0, 4)));
        assert!(!offsets.contains(&(4, 4)));
    }
}