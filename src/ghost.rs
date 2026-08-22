use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use crate::bc1;
use crate::field1::Field;
use crate::geometry::{self, Geometry, Projection};
use crate::state::{Derived, State};

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
    pub side_id: usize,
    /// Precomputed WENO-extrapolation data (only for Wall / Outflow /
    /// FarField ghosts).
    pub bc: Option<Box<bc1::GhostBC>>,
}

#[derive(Debug)]
pub struct GhostGrid {
    pub info: Vec<GhostInfo>,
    pub values: Vec<State>,
    /// Per-stage derived quantities of the cached ghost values,
    /// computed in the same parallel pass as the ghost update.
    pub derived: Vec<Derived>,
    lookup: HashMap<(isize, isize), usize>,
    /// Per-fluid-cell stencil access table.
    ///
    /// For a fluid cell with linear index `l` (i*ny + j) and stencil slot
    /// `k` (position in the offset list), `access[l*n_offsets + k]` is:
    ///   * the linear index of the target cell, if it is fluid;
    ///   * `ghost_id + nx*ny`, if it is a cached ghost.
    pub access: Vec<u32>,
    pub n_offsets: usize,
    off_map: [[u8; 9]; 9],
}

impl GhostGrid {
    /// Build the static ghost layout once from the actual solver stencil.
    pub fn build(field: &Field, offsets: &[Offset]) -> Self {
        let indices = discover_ghost_indices(field, offsets);
        let mut info = Vec::with_capacity(indices.len());
        let mut lookup = HashMap::with_capacity(indices.len());

        let h = (field.grid.dx * field.grid.dy).sqrt();
        let beta_forms = bc1::beta_quadratic_forms(h);

        for idx in indices {
            let id = info.len();
            let g = build_ghost_info(field, idx, &beta_forms);
            assert!(lookup.insert(idx, id).is_none());
            info.push(g);
        }

        let values = vec![State::new(); info.len()];
        let derived = vec![Derived::new(); info.len()];

        // ------------------------------------------------------------
        // Precomputed per-fluid-cell access table.
        // ------------------------------------------------------------
        let nx = field.grid.nx;
        let ny = field.grid.ny;
        let n_off = offsets.len();
        let mut access = vec![0u32; nx * ny * n_off];

        let mut off_map = [[u8::MAX; 9]; 9];
        for (k, &(di, dj)) in offsets.iter().enumerate() {
            assert!((-4..=4).contains(&di) && (-4..=4).contains(&dj));
            off_map[(di + 4) as usize][(dj + 4) as usize] = k as u8;
        }

        for i in 0..nx as isize {
            for j in 0..ny as isize {
                let l = (i as usize) * ny + j as usize;
                if !field.fluid[l] {
                    continue;
                }
                for (k, &(di, dj)) in offsets.iter().enumerate() {
                    let target = (i + di, j + dj);
                    let t = if field.grid.is_in_domain(target)
                        && field.fluid[target.0 as usize * ny + target.1 as usize]
                    {
                        target.0 as usize * ny + target.1 as usize
                    } else {
                        let id = lookup.get(&target).copied().unwrap_or_else(|| {
                            panic!(
                                "ghost {:?} referenced by fluid cell {:?} is not cached",
                                target,
                                (i, j)
                            )
                        });
                        id + nx * ny
                    };
                    access[l * n_off + k] = t as u32;
                }
            }
        }

        Self {
            info,
            values,
            derived,
            lookup,
            access,
            n_offsets: n_off,
            off_map,
        }
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

    /// Slot index of stencil offset (di, dj) in the registered offset list.
    #[inline(always)]
    pub fn k_for(&self, di: isize, dj: isize) -> usize {
        let k = self.off_map[(di + 4) as usize][(dj + 4) as usize];
        assert!(k != u8::MAX, "stencil offset ({}, {}) is not registered", di, dj);
        k as usize
    }

    /// Access-table target for a fluid anchor cell and a stencil slot.
    #[inline(always)]
    pub fn target(&self, anchor: usize, k: usize) -> u32 {
        self.access[anchor * self.n_offsets + k]
    }

    /// O(1)-ish access for API compatibility (cold path).
    #[inline(always)]
    pub fn get_with_field(&self, field: &Field, idx: (isize, isize)) -> State {
        if field.is_in_domain(idx) {
            field.value[field.linear_index(idx)]
        } else {
            self.get(idx)
        }
    }

    /// Correctness-first implementation: calculate every unique ghost once.
    pub fn update_values(&mut self, field: &Field, _t: f64) {
        for id in 0..self.info.len() {
            let g = &self.info[id];
            let result =
                bc1::set_ghost_point_value(
                    g.idx,
                    g.project,
                    g.boundary,
                    g.side_id,
                    field,
                    g.bc.as_deref(),
                );
            self.values[id] = result;
            self.derived[id] = Derived::from_state(result);
        }
    }

    /// Parallel ghost update.
    ///
    /// Each ghost is reconstructed once for the current RK stage, using
    /// its precomputed BC data. Derived quantities are filled in the same
    /// pass. Fail immediately if bc1 produces a non-finite ghost state.
    pub fn update_values_parallel(
        &mut self,
        field: &Field,
    ) {
        let info = &self.info;

        let values = &mut self.values;
        let derived = &mut self.derived;

        values
            .par_iter_mut()
            .zip(derived.par_iter_mut())
            .enumerate()
            .for_each(|(id, (value, dvalue))| {
                let g = &info[id];

                let result =
                    bc1::set_ghost_point_value(
                        g.idx,
                        g.project,
                        g.boundary,
                        g.side_id,
                        field,
                        g.bc.as_deref(),
                    );

                // ----------------------------------------------------
                // bc1 must NEVER return NaN / Inf.
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
                         boundary   = {:?}\n\
                         side id    = {}\n\
                         P0         = ({:.16e}, {:.16e})\n\
                         normal     = ({:.16e}, {:.16e})\n\
                         distance   = {:.16e}\n\
                         nearest    = {:?}\n\
                         state      = {:?}\n\
                         ========================================\n",
                        id,
                        g.idx,
                        g.boundary,
                        g.side_id,
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
                *dvalue = Derived::from_state(result);
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


#[inline]
fn bc_priority(bc: &bc1::BCType) -> usize {
    match bc {
        // Solid/symmetry constraints have highest priority.
        bc1::BCType::Wall => 100,
        bc1::BCType::ReflectiveWall => 90,

        // Prescribed states.
        bc1::BCType::Constant(_) => 70,
        bc1::BCType::TimeDependent(_) => 70,

        // Open boundaries.
        bc1::BCType::FarField(_) => 20,
        bc1::BCType::Outflow { .. } => 10,

        // Adapt these if needed.
        bc1::BCType::ZerothOrder => 5,
        bc1::BCType::Periodic => 0,

        _ => 0,
    }
}

pub fn select_boundary_side(
    p0: geometry::Point,
    polygon: &geometry::Polygon,
    bc_list: &[bc1::BCType],
) -> usize {
    let candidates =
        geometry::find_boundary_sides(
            p0,
            polygon,
            crate::constant::DEFAULT_EPS,
        );

    assert!(
        !candidates.is_empty(),
        "projection point ({:.16e},{:.16e}) \
         does not lie on any polygon side",
        p0.x,
        p0.y,
    );

    assert_eq!(
        polygon.points.len(),
        bc_list.len(),
        "polygon side count and BC count disagree"
    );

    let mut best_side = candidates[0];
    let mut best_priority =
        bc_priority(&bc_list[best_side]);

    for &side in &candidates[1..] {
        let priority =
            bc_priority(&bc_list[side]);

        if priority > best_priority {
            best_side = side;
            best_priority = priority;
        }
    }

    best_side
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

fn build_ghost_info(
    field: &Field,
    idx: (isize, isize),
    beta_forms: &[nalgebra::DMatrix<f64>; 5],
) -> GhostInfo {
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
    let bc_list = match boundary {
    BoundaryKind::Outer => {
        &field.bc_outer
    }

    BoundaryKind::Inner => {
        &field.bc_inner
    }
    };

    let side_id =
    select_boundary_side(
        project.point,
        polygon,
        bc_list,
    );

    let normal =
    polygon.outward_normal_of_side(
        side_id
    );

    let dx =
    p.x - project.point.x;

    let dy =
    p.y - project.point.y;

    let distance =
    dx * normal.x
    + dy * normal.y;

    let project =
    Projection {
        point: project.point,
        normal,
        distance,
    };

    // Precompute the heavy per-stage extrapolation data for BC types that
    // use it. Cheap BCs (Constant/ReflectiveWall/ZerothOrder/Periodic/...)
    // skip it entirely.
    let bc_pre = match &bc_list[side_id] {
        bc1::BCType::Wall
        | bc1::BCType::Outflow { .. }
        | bc1::BCType::FarField(_) => {
            Some(Box::new(bc1::precompute_ghost_bc(&project, field, beta_forms)))
        }
        _ => None,
    };

    GhostInfo {
        idx,
        project,
        nearest_idx,
        boundary,
        side_id,
        bc: bc_pre,
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