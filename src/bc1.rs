use ndarray::{Array1, Array2, array};
use nalgebra::{DMatrix, DVector};
use crate::field1::{self, GridInfo};
use crate::geometry::Geometry;
use ndarray_linalg::Solve;
use crate::state::{self, State};
use crate::{geometry, weno};
use crate::constant;
use std::sync::Arc;

const WALL_TAYLOR_ORDER: usize = 4;


pub type TimeBCFn = Arc<dyn Fn(geometry::Point, geometry::Vec2, f64) -> state::State + Send + Sync>;

#[derive(Clone)]
pub enum BCType {
    /// Mirror the state without changing momentum.
    ReflectiveWall,
    Wall,
    Periodic,
    Constant(State),
    /// Prescribed boundary state U(P0,n,t).
    TimeDependent(TimeBCFn),
    Outflow { p_inf: f64, sigma: f64, l_domain: f64 },
    ZerothOrder,
    FarField(State),
}

// ============================================================================
// Precomputed per-ghost WENO-extrapolation data (Tan et al. 2012, Sec. 2.4).
//
// The E_r stencils and the local-coordinate regression matrices depend ONLY
// on the geometry (projection + grid), never on field values. All the heavy
// least-squares work is therefore done once at GhostGrid build time:
//
//   * `stencil[r]`: (r+1)^2 fluid-cell targets (linear indices), r-major
//   * `s[r]`:       symmetric matrix with beta_r = data^T S_r data
//   * `w[r][k]`:    weight vector with V^(k) = w . data (k <= r)
//
// At each RK stage a ghost update is then just gathers + small dot products,
// instead of ~30 SVD least-squares fits.
// ============================================================================

pub const BC_STENCIL_OFF: [usize; 6] = [0, 1, 5, 14, 30, 55];
pub const BC_S_OFF: [usize; 6] = [0, 1, 17, 98, 354, 979];
pub const BC_W_OFF: [usize; 6] = [0, 1, 9, 36, 100, 225];

#[derive(Clone, Debug)]
pub struct GhostBC {
    pub nearest: (isize, isize),
    pub normal: geometry::Vec2,
    pub d: [f64; 5],
    pub stencil: [u32; 55],
    pub s: [f64; 979],
    pub w: [f64; 225],
}

/// Coefficient-space quadratic form of the smoothness indicator,
/// beta_r(c) = c^T M_r c. Depends only on r and h, not on the stencil
/// geometry (the monomial integrals are taken over the fixed cell K).
pub fn beta_quadratic_form(r: usize, h: f64) -> DMatrix<f64> {
    let n = (r + 1) * (r + 2) / 2;
    let mut m = DMatrix::<f64>::zeros(n, n);
    for a in 0..n {
        for b in a..n {
            let mut ea = DVector::<f64>::zeros(n);
            let mut eb = DVector::<f64>::zeros(n);
            ea[a] = 1.0;
            eb[b] = 1.0;
            let eab = &ea + &eb;
            let v = 0.5
                * (smooth_indicator(&eab, r, h)
                    - smooth_indicator(&ea, r, h)
                    - smooth_indicator(&eb, r, h));
            m[(a, b)] = v;
            m[(b, a)] = v;
        }
    }
    m
}

pub fn beta_quadratic_forms(h: f64) -> [DMatrix<f64>; 5] {
    std::array::from_fn(|r| beta_quadratic_form(r, h))
}

/// Column index of the x^k monomial in the `fit_polynomial_surface`
/// coefficient ordering.
fn xk_coefficient_column(k: usize) -> usize {
    let mut col = 0;
    for t in 0..=k {
        for i2 in 0..=t {
            let j2 = t - i2;
            if i2 == k && j2 == 0 {
                return col;
            }
            col += 1;
        }
    }
    unreachable!()
}

/// Build all per-ghost extrapolation data once.
///
/// The pseudoinverse is computed with the same truncated SVD (tol 1e-12)
/// used by the per-stage `fit_polynomial_surface` path, so the two paths
/// agree to roundoff.
pub fn precompute_ghost_bc(
    project: &geometry::Projection,
    field: &field1::Field,
    beta_forms: &[DMatrix<f64>; 5],
) -> GhostBC {
    let nearest = find_nearest_grid_point(*project, field);
    let h = (field.grid.dx * field.grid.dy).sqrt();

    let d0 = 2.0 * h.powi(4);
    let d1 = 2.0 * h.powi(3);
    let d2 = 2.0 * h.powi(2);
    let d3 = 2.0 * h;
    let d = [d0, d1, d2, d3, 1.0 - d0 - d1 - d2 - d3];

    let mut bc = GhostBC {
        nearest,
        normal: project.normal,
        d,
        stencil: [0; 55],
        s: [0.0; 979],
        w: [0.0; 225],
    };

    for r in 0..=4usize {
        let m = (r + 1) * (r + 1);
        let n_terms = (r + 1) * (r + 2) / 2;

        let idxs = weno_stencil_extractor_indices(*project, field, r);

        let mut xy = [[0.0; 2]; 25];
        for a in 0..m {
            let (gi, gj) = idxs[a];
            bc.stencil[BC_STENCIL_OFF[r] + a] = field.linear_index((gi, gj)) as u32;
            let p = geometry::Point {
                x: field.grid.x(gi),
                y: field.grid.y(gj),
            };
            let (nor, tan) = project.gloabl2local_coord(p);
            xy[a] = [nor, tan];
        }

        let mut a_mat = DMatrix::<f64>::zeros(m, n_terms);
        for row in 0..m {
            let mut col = 0;
            for t in 0..=r {
                for i2 in 0..=t {
                    let j2 = t - i2;
                    a_mat[(row, col)] =
                        xy[row][0].powi(i2 as i32) * xy[row][1].powi(j2 as i32);
                    col += 1;
                }
            }
        }

        let pinv = a_mat
            .svd(true, true)
            .pseudo_inverse(1.0e-12)
            .expect("pinv failed");

        if r == 0 {
            bc.s[0] = 2.0 * h * h;
        } else {
            let s_r = pinv.transpose() * &beta_forms[r] * &pinv;
            for a in 0..m {
                for b in 0..m {
                    bc.s[BC_S_OFF[r] + a * m + b] = s_r[(a, b)];
                }
            }
        }

        for k in 0..=r {
            let col = xk_coefficient_column(k);
            let kf = factorial(k);
            for a in 0..m {
                bc.w[BC_W_OFF[r] + k * m + a] = kf * pinv[(col, a)];
            }
        }
    }

    bc
}

/// Fast per-stage characteristic WENO extrapolation using precomputed data.
///
/// Numerically identical to `weno_extrapolation` for the same stencil.
pub fn weno_extrapolation_pre(
    left: &[[f64; 6]; 6],
    bc: &GhostBC,
    field: &field1::Field,
    k: usize,
) -> [f64; 6] {
    let n = bc.normal;
    let mut alpha = [[0.0; 5]; 6];
    let mut vks = [[0.0; 5]; 6];
    let mut alpha_sum = [0.0; 6];

    for r in 0..=4usize {
        let m = (r + 1) * (r + 1);

        let mut data = [[0.0f64; 25]; 6];

        for a in 0..m {
            let s = field.value[bc.stencil[BC_STENCIL_OFF[r] + a] as usize];
            let mn = s.mom_x * n.x + s.mom_y * n.y;
            let mt = -s.mom_x * n.y + s.mom_y * n.x;
            let u6 = [s.rho, mn, mt, s.ee, s.ei, s.er];
            for c in 0..6 {
                let row = &left[c];
                data[c][a] = row[0]*u6[0] + row[1]*u6[1] + row[2]*u6[2]
                    + row[3]*u6[3] + row[4]*u6[4] + row[5]*u6[5];
            }
        }

        for c in 0..6 {
            let soff = BC_S_OFF[r];
            let mut beta = 0.0;
            for a in 0..m {
                let base = soff + a * m;
                let mut acc = 0.0;
                for b in 0..m {
                    acc += bc.s[base + b] * data[c][b];
                }
                beta += data[c][a] * acc;
            }

            let mut vk = 0.0;
            if k <= r {
                let woff = BC_W_OFF[r] + k * m;
                for a in 0..m {
                    vk += bc.w[woff + a] * data[c][a];
                }
            }

            alpha[c][r] = bc.d[r] / (constant::DEFAULT_EPS + beta).powf(constant::WENO_Q);
            vks[c][r] = vk;
            alpha_sum[c] += alpha[c][r];
        }
    }

    let mut result = [0.0; 6];
    for c in 0..6 {
        for r in 0..=4usize {
            result[c] += (alpha[c][r] / alpha_sum[c]) * vks[c][r];
        }
    }

    result
}

pub fn find_nearest_grid_point(project: geometry::Projection, field: &field1::Field) -> (isize, isize) {
    let (i,j) = field.grid.coord2idx(project.point);

    let idx_list = [(i,j),(i-1,j),(i+1,j),(i,j-1),(i,j+1),(i+1,j+1),(i+1,j-1),(i-1,j+1),(i-1,j-1)];
    let mut target_idx = (0,0);
    let mut target_distance = 1e50;
    for k in 0..9 {
        if field.is_in_domain(idx_list[k]) {
            let dx = field.grid.x(idx_list[k].0) - project.point.x;
            let dy = field.grid.y(idx_list[k].1) - project.point.y;
            let dis = (dx.powi(2) + dy.powi(2)).sqrt();
            if dis < target_distance {
                target_idx = idx_list[k];
                target_distance = dis;
            }
            else {
                continue;
            }
        }
        else {
            continue;
        }
    }
    target_idx
}
/// build_local returns (L, R, lambda) in sequence
/// 
pub fn build_local(project: geometry::Projection,idx: (isize, isize), field: &field1::Field)->(Array2<f64>,Array2<f64>,Array1<f64>) {
    let (l, r, lambda) = build_local_plain(project.normal, idx, field);
    (
        Array2::from_shape_fn((6, 6), |(i, j)| l[i][j]),
        Array2::from_shape_fn((6, 6), |(i, j)| r[i][j]),
        Array1::from_vec(lambda.to_vec()),
    )
}

/// Allocation-free variant of `build_local`.
pub fn build_local_plain(normal: geometry::Vec2, idx: (isize, isize), field: &field1::Field)
    -> ([[f64; 6]; 6], [[f64; 6]; 6], [f64; 6]) {
    let val = field.value[field.linear_index(idx)];
    let mom_n = val.mom_x*normal.x + val.mom_y*normal.y;
    let mom_t = -val.mom_x*normal.y + val.mom_y*normal.x;
    let s1 = state::State {
        rho: val.rho,
        mom_x: mom_n,
        mom_y: mom_t,
        ee: val.ee,
        ei: val.ei,
        er: val.er,
    };
    let d1 = state::Derived::from_state(s1);

    let l = weno::Stencil6::build_l_plain(&s1, &s1, &d1, &d1, state::Direction::X);
    let (lambda, r) = weno::Stencil6::build_r_plain(&s1, &s1, &d1, &d1, state::Direction::X);
    (l, r, lambda)
}

fn stencil2arr(stencil: Vec<(f64, f64, state::State)>) 
-> [Vec<(f64,f64,f64)>;6] {
    let mut rho_list = vec![];
    let mut momx_list = vec![];
    let mut momy_list = vec![];
    let mut ee_list = vec![];
    let mut ei_list = vec![];
    let mut er_list = vec![];

    for i in 0..stencil.len() {
        rho_list.push((stencil[i].0,stencil[i].1,stencil[i].2.rho));
        momx_list.push((stencil[i].0,stencil[i].1,stencil[i].2.mom_x));
        momy_list.push((stencil[i].0,stencil[i].1,stencil[i].2.mom_y));
        ee_list.push((stencil[i].0,stencil[i].1,stencil[i].2.ee));
        ei_list.push((stencil[i].0,stencil[i].1,stencil[i].2.ei));
        er_list.push((stencil[i].0,stencil[i].1,stencil[i].2.er));
    }
    [rho_list,momx_list,momy_list,ee_list,ei_list,er_list]

}


/*
This part is gonna be WENO extrpolation code, stencil extractor and ILW.
A bridge between common boundary condition and characteristic wave should be built.
*/


///k indicate the derivative requested from this extrapolation,
/// 
/// return should be \[V^(k); 6\], or `state::State`
pub fn weno_extrapolation(
    project: geometry::Projection,
    field: &field1::Field,
    k: usize,
) -> [f64; 6] {
    // Tan et al. (2012), Sec. 2.4: for each r=0..4 construct E_r
    // independently, with |E_r|=(r+1)^2, and fit p_r in P_r.
    let mut vk = Array2::from_elem((5, 6), 0.0);
    let mut beta = Array2::from_elem((5, 6), 0.0);

    for r in 0..5 {
        let stencil_idx = weno_stencil_extractor(project, field, r);
        let stencil = weno_data_preprocess(project, &stencil_idx, field);

        debug_assert_eq!(stencil.len(), (r + 1) * (r + 1));

        let mut rho_list = vec![];
        let mut momx_list = vec![];
        let mut momy_list = vec![];
        let mut ee_list = vec![];
        let mut ei_list = vec![];
        let mut er_list = vec![];

        for &(x, y, state) in &stencil {
            rho_list.push((x, y, state.rho));
            momx_list.push((x, y, state.mom_x));
            momy_list.push((x, y, state.mom_y));
            ee_list.push((x, y, state.ee));
            ei_list.push((x, y, state.ei));
            er_list.push((x, y, state.er));
        }

        let (b, v) = poly_regression(project, &rho_list, k, r, field);
        beta[[r, 0]] = b; vk[[r, 0]] = v;
        let (b, v) = poly_regression(project, &momx_list, k, r, field);
        beta[[r, 1]] = b; vk[[r, 1]] = v;
        let (b, v) = poly_regression(project, &momy_list, k, r, field);
        beta[[r, 2]] = b; vk[[r, 2]] = v;
        let (b, v) = poly_regression(project, &ee_list, k, r, field);
        beta[[r, 3]] = b; vk[[r, 3]] = v;
        let (b, v) = poly_regression(project, &ei_list, k, r, field);
        beta[[r, 4]] = b; vk[[r, 4]] = v;
        let (b, v) = poly_regression(project, &er_list, k, r, field);
        beta[[r, 5]] = b; vk[[r, 5]] = v;
    }

    // The paper assumes a uniform Cartesian mesh with dx=dy=h.
    let dx = field.grid.dx;
    let dy = field.grid.dy;
    //let scale = dx.abs().max(dy.abs()).max(1.0);
    let h = (dx * dy).sqrt();

    let d0 = 2.0 * h.powi(4);
    let d1 = 2.0 * h.powi(3);
    let d2 = 2.0 * h.powi(2);
    let d3 = 2.0 * h;
    let d4 = 1.0 - d0 - d1 - d2 - d3;
    let d = [d0, d1, d2, d3, d4];

    let epsilon = constant::DEFAULT_EPS;
    let q = constant::WENO_Q; // Double Mach reflection benchmark: paper uses q=20 for boundary stabilization
    let mut result = [0.0; 6];

    for m in 0..6 {
        let mut alpha = [0.0; 5];
        let mut alpha_sum = 0.0;
        for r in 0..5 {
            alpha[r] = d[r] / (epsilon + beta[[r, m]]).powf(q);
            alpha_sum += alpha[r];
        }
        for r in 0..5 {
            result[m] += (alpha[r] / alpha_sum) * vk[[r, m]];
        }
    }

    result
}


fn local_pressure_and_a2(rho: f64, u: f64, v: f64, ee: f64, ei: f64, er: f64) -> (f64, f64) {
    let w2 = u * u + v * v;
    let ee1 = ee / rho - w2 / 6.0;
    let ei1 = ei / rho - w2 / 6.0;
    let er1 = er / rho - w2 / 6.0;

    let ge = constant::GAMMA_E - 1.0;
    let gi = constant::GAMMA_I - 1.0;
    let gr = constant::GAMMA_R - 1.0;

    let p = rho * (ge * ee1 + gi * ei1 + gr * er1);
    let a2 = constant::GAMMA_E * ge * ee1
        + constant::GAMMA_I * gi * ei1
        + constant::GAMMA_R * gr * er1;

    (p, a2)
}
// ============================================================
// ILW row-1 substitution for the no-penetration momentum PDE,
// R -> infinity (flat polygon walls => curvature RHS term is 0).
// Coefficients derived from the rotated momentum equation; see
// conversation derivation. u0 = [rho, mom_n, mom_t, ee, ei, er]
// i.e. the already-solved U^(0) at the boundary point.
// ============================================================
fn build_ilw_row1(u0: &Array1<f64>) -> Array1<f64> {
    let u1 = u0[0]; // rho
    let u3 = u0[2]; // mom_t (tangential momentum)

    let gi = constant::GAMMA_I - 1.0;
    let ge = constant::GAMMA_E - 1.0;
    let gr = constant::GAMMA_R - 1.0;
    let gt = gi + ge + gr; // == gamma_t - 3

    array![
        -gt * u3.powi(2) / (6.0 * u1.powi(2)),
        0.0,
        gt * u3 / (6.0 * u1),
        -ge,
        -gi,
        -gr,
    ]
}

fn solve6(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    a.solve(b).expect("singular ILW/extrapolation system at boundary")
}

fn zeroth_order_value(
    idx: (isize, isize),
    project: geometry::Projection,
    field: &field1::Field,
) -> state::State {
    // Ghost point Pg.
    let pg = geometry::Point {
        x: field.grid.x(idx.0),
        y: field.grid.y(idx.1),
    };

    // Reflect Pg geometrically across the boundary:
    //
    //     Pi = 2 P0 - Pg
    //
    // Pi lies on the interior side and gives a natural
    // interior sampling location for this ghost point.
    let pi = geometry::Point {
        x: 2.0 * project.point.x - pg.x,
        y: 2.0 * project.point.y - pg.y,
    };

    // Cartesian index nearest to Pi.
    let (ic, jc) = field.grid.coord2idx(pi);

    // Search nearby cells because the nearest Cartesian point
    // can occasionally lie outside the fluid for an oblique boundary.
    let candidates = [
        (ic, jc),
        (ic - 1, jc),
        (ic + 1, jc),
        (ic, jc - 1),
        (ic, jc + 1),
        (ic - 1, jc - 1),
        (ic - 1, jc + 1),
        (ic + 1, jc - 1),
        (ic + 1, jc + 1),
    ];

    let mut best = None;
    let mut best_d2 = f64::INFINITY;

    for q in candidates {
        if !field.is_in_domain(q) {
            continue;
        }

        let xq = field.grid.x(q.0);
        let yq = field.grid.y(q.1);

        let d2 =
            (xq - pi.x).powi(2)
            + (yq - pi.y).powi(2);

        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(q);
        }
    }

    let interior_idx = best.unwrap_or_else(|| {
        panic!(
            "cannot find interior point for zeroth-order ghost {:?}",
            idx
        )
    });

    // Zeroth-order extrapolation:
    //
    //     U_g = U_i
    //
    // IMPORTANT:
    // Unlike ReflectiveWall, do NOT modify the normal momentum.
    field.value[field.linear_index(interior_idx)]
}

fn periodic_value(
    idx: (isize, isize),
    project: geometry::Projection,
    field: &field1::Field,
) -> state::State {
    let (i, j) = idx;

    // This implementation is for the translating-shock test,
    // where periodicity is only applied in the y direction.
    //
    // Physical periodic interval:
    //
    //     y in [y0, y0 + Ly)
    //
    // The polygon upper boundary itself is not a fluid grid point.
    // Therefore the valid periodic y indices are
    //
    //     j = 0, ..., Np - 1.
    //
    // Find Np from the projected horizontal boundary.
    let dy = field.grid.dy;

    let y_bottom = field.grid.y0;

    // For a top/bottom periodic boundary, project.point.y is
    // either y_bottom or y_top.
    //
    // Determine Ly from the rectangular domain.
    //
    // Here the translating-shock case uses Ly = 0.5.
    let ly = 0.5_f64;

    let np =
        (ly / dy).round() as isize;

    debug_assert!(np > 0);

    // Euclidean modulo:
    //
    //   -1   -> np-1
    //   -2   -> np-2
    //   np   -> 0
    //   np+1 -> 1
    let j_wrap = j.rem_euclid(np);

    let wrapped_idx = (i, j_wrap);

    if !field.is_in_domain(wrapped_idx) {
        panic!(
            "periodic wrapped point is not fluid: \
             ghost={:?}, wrapped={:?}, np={}, P0=({:.8e},{:.8e})",
            idx,
            wrapped_idx,
            np,
            project.point.x,
            project.point.y,
        );
    }

    field.get(wrapped_idx)
}

pub fn set_ghost_point_value(
    idx:(isize,isize),
    project: geometry::Projection,
    boundary: crate::ghost::BoundaryKind,
    side_id: usize,
    field: &field1::Field,
    bc_pre: Option<&GhostBC>,
)
-> state::State {

    let bc = match boundary {
    crate::ghost::BoundaryKind::Outer => {
        &field.bc_outer[side_id]
    }

    crate::ghost::BoundaryKind::Inner => {
        &field.bc_inner[side_id]
    }
    };

    let t= field.time;

    if let BCType::Constant(value) = bc {
        return *value;
    }
    if let BCType::TimeDependent(f) = bc {
        return f(project.point, project.normal, t);
    }

    if let BCType::ReflectiveWall = bc {
        return reflective_wall_value(
            idx,
            project,
            field,
        );
    }

    if let BCType::ZerothOrder = bc {
    return zeroth_order_value(
        idx,
        project,
        field,
    );
    }

    if let BCType::Periodic = bc {
    return periodic_value(
        idx,
        project,
        field,
    );
    }   

    let mut v = [[0.0; 6]; 5];
    // The characteristic matrix is evaluated at U_0, the interior grid
    // point nearest to the boundary foot P_0 (Tan et al., Sec. 2.3),
    // never at the ghost-cell index itself.
    //
    // Fast path: per-ghost stencil/regression data were precomputed at
    // GhostGrid build time. Slow path (tests / standalone use): rebuild
    // everything per call.
    let (left, right, lambda) = if let Some(pre) = bc_pre {
        let (l_plain, r_plain, lam) =
            build_local_plain(project.normal, pre.nearest, field);
        for k in 0..5 {
            v[k] = weno_extrapolation_pre(&l_plain, pre, field, k);
        }
        (
            Array2::from_shape_fn((6, 6), |(i, j)| l_plain[i][j]),
            Array2::from_shape_fn((6, 6), |(i, j)| r_plain[i][j]),
            Array1::from_vec(lam.to_vec()),
        )
    } else {
        let idx_near = find_nearest_grid_point(project, field);
        let (l, r, lam) = build_local(project, idx_near, field);
        for k in 0..5 {
            v[k] = weno_extrapolation(project, field, k);
        }
        (l, r, lam)
    };

    match bc {
        BCType::Wall => {
            // ---------------------------------------------------------
            // k = 0: enforce mom_n = 0 (no-penetration), and extrapolate
            //        the other five outgoing characteristics — Eq. (2.19).
            // ---------------------------------------------------------
            let mut left0 = left.clone();
            for c in 0..6 {
                left0[[0, c]] = if c == 1 { 1.0 } else { 0.0 };
            }
            let mut rhs0 = Array1::from_vec(v[0].to_vec());
            rhs0[0] = 0.0;
            let u0 = solve6(&left0, &rhs0);

            // ---------------------------------------------------------
            // k = 1: ILW momentum row + extrapolation rows — Eq. (2.20)/(2.22)
            //        analogue. R -> infinity for flat polygon walls,
            //        so the curvature RHS term is 0.
            // ---------------------------------------------------------
            let mut left1 = left.clone();
            let row1 = build_ilw_row1(&u0);
            for c in 0..6 {
                left1[[0, c]] = row1[c];
            }
            let mut rhs1 = Array1::from_vec(v[1].to_vec());
            rhs1[0] = 0.0; // TODO: nonzero once curved geometries are supported
            let u1 = solve6(&left1, &rhs1);

            // ---------------------------------------------------------
            // k = 2,3,4: pure WENO extrapolation, Eq. (2.21).
            //            No ILW / PDE substitution needed here.
            // ---------------------------------------------------------
            let mut u = vec![u0, u1, Array1::zeros(6), Array1::zeros(6), Array1::zeros(6)];
            for k in 2..=4 {
                let rhs_k = Array1::from_vec(v[k].to_vec());
                u[k] = right.dot(&rhs_k);
            }

            // ---------------------------------------------------------
            // Taylor expansion to the ghost point, Eq. (2.17).
            // project.distance is the signed normal offset D of the
            // ghost point relative to the boundary foot point x0.
            // ---------------------------------------------------------
            let d = project.distance;
            let mut u_ghost = u[0].clone();
            let mut coef = 1.0;
            for k in 1..=WALL_TAYLOR_ORDER {
                coef *= d / (k as f64);
                u_ghost = u_ghost + coef * &u[k];
            }

            // rotate momentum back: local (normal, tangential) -> global (x, y)
            let n = project.normal;
            let mom_n = u_ghost[1];
            let mom_t = u_ghost[2];
            let mom_x = mom_n * n.x - mom_t * n.y;
            let mom_y = mom_n * n.y + mom_t * n.x;

            state::State {
                rho: u_ghost[0],
                mom_x,
                mom_y,
                ee: u_ghost[3],
                ei: u_ghost[4],
                er: u_ghost[5],
            }
        }

        // Symmetry / Periodic / Constant(_) still need their own row-0
        // substitution (different g(t), different BC-row structure).
        BCType::Outflow { p_inf, sigma, l_domain } => {
            // ---------------------------------------------------------
            // k = 0: no algebraic constraint on the state itself at this
            //        order — LODI relaxation only touches the wave
            //        amplitude (k=1). Pure extrapolation, same as k=2..4.
            // ---------------------------------------------------------
            let rhs0 = Array1::from_vec(v[0].to_vec());
            let u0 = right.dot(&rhs0);

            let rho0 = u0[0];
            let un0 = u0[1] / rho0; // normal velocity
            let ut0 = u0[2] / rho0;
            let (p0, a2) = local_pressure_and_a2(rho0, un0, ut0, u0[3], u0[4], u0[5]);
            let a = a2.sqrt();
            let lambda1 = un0 - a; // incoming iff < 0 (subsonic outflow: 0 < un0 < a)

            // ---------------------------------------------------------
            // k = 1: rows 1..5 (outgoing) stay WENO-extrapolated as-is.
            //        row 0 (incoming acoustic, lambda1 < 0) is replaced
            //        by the LODI relaxation value V_1^(1) = L_1 / lambda_1.
            //        `left` itself is untouched -> reuse `right` directly.
            // ---------------------------------------------------------
            let mut rhs1 = Array1::from_vec(v[1].to_vec());

            if lambda1 < 0.0 {
                let m_local = (un0 / a).abs(); // TODO: swap for a stored global M_max if you track one
                let l1 = *sigma * (1.0 - m_local.powi(2)) * a / *l_domain * (p0 - *p_inf);
                rhs1[0] = l1 / lambda1;
            }
            // else: this boundary point is locally supersonic-out; row 0 is
            // also outgoing there, so leave the WENO-extrapolated value alone.
        
            let u1 = right.dot(&rhs1);
        
            // ---------------------------------------------------------
            // k = 2,3,4: pure extrapolation, unchanged left/right.
            // ---------------------------------------------------------
            let mut u = vec![u0, u1, Array1::zeros(6), Array1::zeros(6), Array1::zeros(6)];
            for k in 2..=4 {
                let rhs_k = Array1::from_vec(v[k].to_vec());
                u[k] = right.dot(&rhs_k);
            }
        
            // ---------------------------------------------------------
            // Taylor expansion + rotate back to global frame (same as Wall).
            // ---------------------------------------------------------
            let d = project.distance;
            let mut u_ghost = u[0].clone();
            let mut coef = 1.0;
            for k in 1..=WALL_TAYLOR_ORDER {
                coef *= d / (k as f64);
                u_ghost = u_ghost + coef * &u[k];
            }
        
            let n = project.normal;
            let mom_n = u_ghost[1];
            let mom_t = u_ghost[2];
            let mom_x = mom_n * n.x - mom_t * n.y;
            let mom_y = mom_n * n.y + mom_t * n.x;
        
            state::State {
                rho: u_ghost[0],
                mom_x,
                mom_y,
                ee: u_ghost[3],
                ei: u_ghost[4],
                er: u_ghost[5],
            }
        },
        BCType::FarField(u_inf) => {
    // Characteristic matrices are evaluated from the nearest
    // interior state, exactly as in the existing BC machinery
    // (already computed above with the correct idx_near).

    // ------------------------------------------------------------
    // Convert freestream state to the LOCAL (normal,tangential)
    // coordinate system before applying L.
    // ------------------------------------------------------------
    let n = project.normal;

    let uinf_local = rotate_state_to_local(*u_inf, n);
    let uinf_arr =
        Array1::from_vec(uinf_local.state2arr().to_vec());

    let v_inf = left.dot(&uinf_arr);

    // u[k] = kth normal derivative of conservative state
    let mut u = vec![
        Array1::zeros(6),
        Array1::zeros(6),
        Array1::zeros(6),
        Array1::zeros(6),
        Array1::zeros(6),
    ];

    // ------------------------------------------------------------
    // k = 0
    //
    // outgoing characteristic : WENO extrapolation
    // incoming characteristic : freestream
    // ------------------------------------------------------------
    let mut rhs0 =
        Array1::from_vec(v[0].to_vec());

    for m in 0..6 {
        if lambda[m] < 0.0 {
            rhs0[m] = v_inf[m];
        }
    }

    u[0] = right.dot(&rhs0);

    // ------------------------------------------------------------
    // k = 1..4
    //
    // Far-field state is spatially constant, therefore incoming
    // characteristic derivatives are zero.
    // ------------------------------------------------------------
    for k in 1..=4 {
        let mut rhs_k =
            Array1::from_vec(v[k].to_vec());

        for m in 0..6 {
            if lambda[m] < 0.0 {
                rhs_k[m] = 0.0;
            }
        }

        u[k] = right.dot(&rhs_k);
    }

    // ------------------------------------------------------------
    // Taylor expansion from boundary foot P0 to ghost point
    // ------------------------------------------------------------
    let d = project.distance;

    let mut u_ghost = u[0].clone();
    let mut coef = 1.0;

    for k in 1..=WALL_TAYLOR_ORDER {
        coef *= d / (k as f64);
        u_ghost = u_ghost + coef * &u[k];
    }

    // ------------------------------------------------------------
    // local momentum -> global momentum
    // ------------------------------------------------------------
    let mom_n = u_ghost[1];
    let mom_t = u_ghost[2];

    let mom_x =
        mom_n * n.x - mom_t * n.y;

    let mom_y =
        mom_n * n.y + mom_t * n.x;

    state::State {
        rho: u_ghost[0],
        mom_x,
        mom_y,
        ee: u_ghost[3],
        ei: u_ghost[4],
        er: u_ghost[5],
    }
},
        BCType::Constant(_) | BCType::TimeDependent(_) | BCType::ReflectiveWall => unreachable!(),
        _ => state::State::new(),
    }
}

fn reflective_wall_value(
    idx: (isize, isize),
    project: geometry::Projection,
    field: &field1::Field,
) -> state::State {
    let n = project.normal;

    // ------------------------------------------------------------
    // Mirror the ghost point geometrically across the wall.
    //
    // project.distance = signed normal distance from ghost to P0.
    //
    // If Pg is the ghost point and P0 the wall foot,
    //
    //     Pi = 2 P0 - Pg
    //
    // ------------------------------------------------------------

    let pg = geometry::Point {
        x: field.grid.x(idx.0),
        y: field.grid.y(idx.1),
    };

    let pi = geometry::Point {
        x: 2.0 * project.point.x - pg.x,
        y: 2.0 * project.point.y - pg.y,
    };

    // Nearest Cartesian grid point to the reflected position.
    let (ic, jc) =
        field.grid.coord2idx(pi);

    // Search a small neighborhood in case coord2idx lands on a
    // non-fluid point near an oblique polygon boundary.
    let candidates = [
        (ic, jc),
        (ic - 1, jc),
        (ic + 1, jc),
        (ic, jc - 1),
        (ic, jc + 1),
        (ic - 1, jc - 1),
        (ic - 1, jc + 1),
        (ic + 1, jc - 1),
        (ic + 1, jc + 1),
    ];

    let mut best = None;
    let mut best_d2 = f64::INFINITY;

    for q in candidates {
        if !field.is_in_domain(q) {
            continue;
        }

        let xq = field.grid.x(q.0);
        let yq = field.grid.y(q.1);

        let d2 =
            (xq - pi.x).powi(2)
            + (yq - pi.y).powi(2);

        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(q);
        }
    }

    let interior_idx =
        best.unwrap_or_else(|| {
            panic!(
                "cannot find reflected interior point for ghost {:?}",
                idx
            )
        });

    // IMPORTANT:
    // interior_idx has already been checked as fluid.
    let s =
        field.value[
            field.linear_index(interior_idx)
        ];

    // ------------------------------------------------------------
    // Reflect momentum relative to arbitrary polygon normal.
    // ------------------------------------------------------------

    let mom_n =
        s.mom_x * n.x
        + s.mom_y * n.y;

    // m_g = m_i - 2 (m_i . n) n
    let mom_x =
        s.mom_x
        - 2.0 * mom_n * n.x;

    let mom_y =
        s.mom_y
        - 2.0 * mom_n * n.y;

    state::State {
        rho: s.rho,
        mom_x,
        mom_y,
        ee: s.ee,
        ei: s.ei,
        er: s.er,
    }
}

fn poly_regression(project: geometry::Projection,
    stencil: &Vec<(f64, f64, f64)>,
    k:usize,r: usize, 
    field: &field1::Field) 
-> (f64, f64) {
    let h  = (field.grid.dx*field.grid.dy).sqrt();
    let poly  = fit_polynomial_surface(&stencil, r);
    let smooth_indicator = smooth_indicator(&poly, r, h);
    let v_kp = x_derivative_at_origin(&poly, r, k);
    (smooth_indicator,v_kp)
}

/// Coefficient ordering:
///
/// r = 0:
/// [a00]
///
/// r = 1:
/// [a00, a10, a01]
///
/// r = 2:
/// [a00, a10, a01, a20, a11, a02]
///
/// r = 3:
/// [a00, a10, a01, a20, a11, a02,
///  a30, a21, a12, a03]
///
/// r = 4:
/// [a00, a10, a01, a20, a11, a02,
///  a30, a21, a12, a03,
///  a40, a31, a22, a13, a04]
pub fn smooth_indicator(
    coef: &DVector<f64>,
    r: usize,
    h: f64,
) -> f64 {
    assert!(r <= 4, "Only r = 0..4 is supported.");

    let expected_len = (r + 1) * (r + 2) / 2;
    assert_eq!(
        coef.len(),
        expected_len,
        "Coefficient vector has wrong length."
    );

    // Paper definition:
    // beta_0 = 2 h^2
    if r == 0 {
        return 2.0 * h * h;
    }

    let mut beta = 0.0;

    // alpha = (ax, ay)
    for ax in 0..=r {
        for ay in 0..=(r - ax) {
            let order = ax + ay;

            // Exclude alpha = (0,0)
            if order == 0 {
                continue;
            }

            // Construct D^(ax,ay) p_r
            let derivative = derivative_coeffs(coef, r, ax, ay);

            // Integral over K of (D^alpha p)^2
            let integral = integrate_square(&derivative, r - order, h);

            // |K|^(|alpha|-1)
            // |K| = h^2
            let scale = h.powi((2 * (order - 1)) as i32);

            beta += scale * integral;
        }
    }

    beta
}

fn derivative_coeffs(
    coef: &DVector<f64>,
    degree: usize,
    ax: usize,
    ay: usize,
) -> Vec<((usize, usize), f64)> {
    let mut result = Vec::new();

    let mut k = 0;

    for total in 0..=degree {
        for i in 0..=total {
            let j = total - i;

            let aij = coef[k];
            k += 1;

            if i < ax || j < ay {
                continue;
            }

            let mut factor = 1.0;

            // i! / (i-ax)!
            for m in 0..ax {
                factor *= (i - m) as f64;
            }

            // j! / (j-ay)!
            for m in 0..ay {
                factor *= (j - m) as f64;
            }

            result.push((
                (i - ax, j - ay),
                aij * factor,
            ));
        }
    }

    result
}

fn integrate_square(
    poly: &[((usize, usize), f64)],
    _degree: usize,
    h: f64,
) -> f64 {
    let mut result = 0.0;

    for &((i1, j1), c1) in poly {
        for &((i2, j2), c2) in poly {
            let px = i1 + i2;
            let py = j1 + j2;

            let ix = integrate_1d_monomial(px, h);
            let iy = integrate_1d_monomial(py, h);

            result += c1 * c2 * ix * iy;
        }
    }

    result
}

#[inline]
fn integrate_1d_monomial(power: usize, h: f64) -> f64 {
    // Integral of x^power from -h/2 to h/2

    if power % 2 == 1 {
        return 0.0;
    }

    let half = 0.5 * h;

    2.0 * half.powi((power + 1) as i32)
        / (power + 1) as f64
}

/// Fit
///
/// p(x,y) = sum_{i+j <= degree} a_{ij} x^i y^j
///
/// Coefficients are ordered as:
///
/// degree 0:
///     1
///
/// degree 1:
///     1, y, x
///
/// degree 2:
///     1, y, x, y^2, xy, x^2
///
/// degree 3:
///     1, y, x, y^2, xy, x^2, y^3, xy^2, x^2y, x^3
pub fn fit_polynomial_surface(
    data: &[(f64, f64, f64)],
    degree: usize,
) -> DVector<f64> {
    let n_points = data.len();

    let n_terms = (degree + 1) * (degree + 2) / 2;

    assert!(
        n_points >= n_terms,
        "Not enough data points for polynomial degree {}",
        degree
    );

    let mut a = DMatrix::<f64>::zeros(n_points, n_terms);
    let mut b = DVector::<f64>::zeros(n_points);

    for (row, &(x, y, value)) in data.iter().enumerate() {
        let mut col = 0;

        for total_degree in 0..=degree {
            for i in 0..=total_degree {
                let j = total_degree - i;

                a[(row, col)] =
                    x.powi(i as i32) * y.powi(j as i32);

                col += 1;
            }
        }

        b[row] = value;
    }

    // Eq. (2.24) is a rectangular least-squares problem for r>0:
    // |E_r|=(r+1)^2 while dim(P_r)=(r+1)(r+2)/2.
    let svd = a.svd(true, true);
    svd.solve(&b, 1.0e-12)
        .expect("SVD least-squares solve failed")
}

/// Compute factorial(n) as f64.
fn factorial(n: usize) -> f64 {
    (1..=n)
        .map(|k| k as f64)
        .product()
}

/// Returns the k-th derivative with respect to x at (0, 0):
///
///     d^k p / dx^k (0,0)
///
/// If k > degree, returns 0.0.
pub fn x_derivative_at_origin(
    coeffs: &DVector<f64>,
    degree: usize,
    k: usize,
) -> f64 {
    // A polynomial of degree `degree` has zero
    // derivatives of order greater than `degree`.
    if k > degree {
        return 0.0;
    }

    // The x^k term is the last term of total degree k
    // in the coefficient ordering:
    //
    // degree 0: 1
    // degree 1: y, x
    // degree 2: y^2, xy, x^2
    // degree 3: y^3, xy^2, x^2y, x^3
    //
    // Find coefficient of x^k.
    let mut col = 0;

    for total_degree in 0..=k {
        for i in 0..=total_degree {
            let j = total_degree - i;

            if i == k && j == 0 {
                return coeffs[col] * factorial(k);
            }

            col += 1;
        }
    }

    // Should never reach here.
    0.0
}

fn first_interior_j(
    project: geometry::Projection,
    field: &field1::Field,
) -> isize {
    let q =
        (project.point.y - field.grid.y0)
        / field.grid.dy;

    let mut j = if project.normal.y > 0.0 {
        q.ceil() as isize - 1
    } else {
        q.floor() as isize + 1
    };

    j = j.clamp(
        0,
        field.grid.ny as isize - 1,
    );

    j
}

fn first_interior_i(
    project: geometry::Projection,
    field: &field1::Field,
) -> isize {
    let q =
        (project.point.x - field.grid.x0)
        / field.grid.dx;

    let mut i = if project.normal.x > 0.0 {
        // outward → +x, interior → -x
        q.ceil() as isize - 1
    } else {
        // outward → -x, interior → +x
        q.floor() as isize + 1
    };

    i = i.clamp(
        0,
        field.grid.nx as isize - 1,
    );

    i
}

/// Build the paper stencil E_r for the 2D WENO extrapolation in
/// Tan et al. (2012), Sec. 2.4.
///
/// E_r is made of r+1 one-dimensional substencils S_l.  Each S_l
/// contains r+1 consecutive Cartesian grid points, hence
/// |E_r|=(r+1)^2.  The grid lines are selected successively in the
/// inward-normal direction.  If a candidate S_l is not wholly inside
/// the fluid domain, the whole S_l is shifted along that grid line;
/// individual points are never dropped.
pub fn weno_stencil_extractor(
    project: geometry::Projection,
    field: &field1::Field,
    r: usize,
) -> Vec<(isize, isize, state::State)> {
    weno_stencil_extractor_indices(project, field, r)
        .into_iter()
        .map(|(gi, gj)| (gi, gj, field.get((gi, gj))))
        .collect()
}

/// Index-only variant of `weno_stencil_extractor` (geometry-only, used by
/// the precompute path; all returned points are fluid cells).
pub fn weno_stencil_extractor_indices(
    project: geometry::Projection,
    field: &field1::Field,
    r: usize,
) -> Vec<(isize, isize)> {
    assert!(r <= 4, "paper WENO extrapolation only uses r=0..4");

    let n = project.normal;
    let p0 = project.point;
    let dx = field.grid.dx;
    let dy = field.grid.dy;
    let x0 = field.grid.x0;
    let y0 = field.grid.y0;
    let width = (r + 1) as isize;

    // Use the family of grid lines for which the normal component is
    // largest.  This avoids a nearly parallel line/normal intersection.
    let dir = if n.x.abs() >= n.y.abs() {
        state::Direction::X
    } else {
        state::Direction::Y
    };

    let mut e_r = Vec::with_capacity((r + 1) * (r + 1));

    match dir {
        state::Direction::X => {
            assert!(n.x.abs() > 1.0e-14);

            // Vertical grid lines x=x_i, ordered from the boundary into Omega.
            let i0 = first_interior_i(project, field);

            let step_i =
                if project.normal.x > 0.0 {
                    -1
                } else {
                    1
                };

            for l in 0..=r {
                let i_line = i0 + l as isize * step_i;
                let x_line = field.grid.x(i_line);

                // p(t)=p0-n*t, t>0 points into Omega.
                let t = (p0.x - x_line) / n.x;
                let y_star = p0.y - n.y * t;

                // Paper's r=3 example uses m-1,m,m+1,m+2, where m is
                // immediately below/left of the intersection.  The formula
                // below is its centered r+1-point generalization.
                let m = ((y_star - y0) / dy).floor() as isize;
                let base_start = m - (r as isize / 2);

                let s_l = shifted_vertical_substencil_indices(
                    field, i_line, base_start, width,
                ).unwrap_or_else(|| {
                    panic!(
                        "cannot fit paper substencil S_{} (r={}) on vertical grid line i={} inside fluid domain",
                        l, r, i_line
                    )
                });

                e_r.extend(s_l);
            }
        }

        state::Direction::Y => {
            assert!(n.y.abs() > 1.0e-14);

            // Horizontal grid lines y=y_j, ordered from the boundary into Omega.
            let q = (p0.y - y0) / dy;
            let j0 = first_interior_j(project, field);

            let interior_step_j =
                if project.normal.y > 0.0 {
                    -1
                } else {
                     1
                };
            
            for l in 0..=r {
                let j_line = j0 + (l as isize) * interior_step_j;
                let y_line = field.grid.y(j_line);

                let t = (p0.y - y_line) / n.y;
                let x_star = p0.x - n.x * t;

                let m = ((x_star - x0) / dx).floor() as isize;
                let base_start = m - (r as isize / 2);

                let s_l = shifted_horizontal_substencil_indices(
                    field, j_line, base_start, width,
                ).unwrap_or_else(|| {
                    panic!(
                        "cannot fit paper substencil S_{} (r={}) on horizontal grid line j={} inside fluid domain",
                        l, r, j_line
                    )
                });

                e_r.extend(s_l);
            }
        }
    }

    assert_eq!(
        e_r.len(),
        (r + 1) * (r + 1),
        "paper stencil E_r must contain exactly (r+1)^2 points"
    );

    e_r
}

/// Shift a vertical S_l as a whole along y until all `width` consecutive
/// points are in Omega.  Search nearest shifts first; never discard points.
fn shifted_vertical_substencil_indices(
    field: &field1::Field,
    i: isize,
    base_start_j: isize,
    width: isize,
) -> Option<Vec<(isize, isize)>> {
    let max_shift = field.grid.ny as isize + width;
    for mag in 0..=max_shift {
        let shifts: [isize; 2] = [-mag, mag];
        let count = if mag == 0 { 1 } else { 2 };
        for &shift in &shifts[..count] {
            let start_j = base_start_j + shift;
            let mut indices = Vec::with_capacity(width as usize);
            let mut ok = true;
            for a in 0..width {
                let idx = (i, start_j + a);
                if !field.grid.is_in_domain(idx) || !field.is_in_domain(idx) {
                    ok = false;
                    break;
                }
                indices.push(idx);
            }
            if ok {
                return Some(indices);
            }
        }
    }
    None
}

/// Shift a horizontal S_l as a whole along x until all `width` consecutive
/// points are in Omega.  Search nearest shifts first; never discard points.
fn shifted_horizontal_substencil_indices(
    field: &field1::Field,
    j: isize,
    base_start_i: isize,
    width: isize,
) -> Option<Vec<(isize, isize)>> {
    let max_shift = field.grid.nx as isize + width;
    for mag in 0..=max_shift {
        let shifts: [isize; 2] = [-mag, mag];
        let count = if mag == 0 { 1 } else { 2 };
        for &shift in &shifts[..count] {
            let start_i = base_start_i + shift;
            let mut indices = Vec::with_capacity(width as usize);
            let mut ok = true;
            for a in 0..width {
                let idx = (start_i + a, j);
                if !field.grid.is_in_domain(idx) || !field.is_in_domain(idx) {
                    ok = false;
                    break;
                }
                indices.push(idx);
            }
            if ok {
                return Some(indices);
            }
        }
    }
    None
}

fn rotate_state_to_local(s: state::State, n: geometry::Vec2) -> state::State {
    let mom_n =  s.mom_x * n.x + s.mom_y * n.y;
    let mom_t = -s.mom_x * n.y + s.mom_y * n.x;
    state::State { mom_x: mom_n, mom_y: mom_t, ..s }
}

fn weno_data_preprocess(
    project: geometry::Projection,
    stencil: &[(isize, isize, state::State)],
    field: &field1::Field,
) -> Vec<(f64, f64, state::State)> {
    let idx_near = find_nearest_grid_point(project, field);
    let (left, _right, _lambda) = build_local(project, idx_near, field);

    let mut result = Vec::with_capacity(stencil.len());
    for &(gi, gj, s) in stencil {
        let p = geometry::Point {
            x: field.grid.x(gi),
            y: field.grid.y(gj),
        };

        let s_local = rotate_state_to_local(s, project.normal);
        let tmp_k = Array1::from_vec(s_local.state2arr().to_vec());
        let (nor, tan) = project.gloabl2local_coord(p);
        let v_m = left.dot(&tmp_k);

        result.push((nor, tan, state::State::arr2state(v_m)));
    }
    result
}
















// ============================================================================
// Verification tests for bc1.rs
//
// Append this block to the END of src/bc1.rs.
//
// Run:
//     cargo test bc1::tests -- --nocapture
//
// These tests focus on the parts that can be verified independently:
//   1. polynomial least-squares reproduction;
//   2. normal-x derivative extraction;
//   3. paper E_r cardinality and Cartesian substencil structure;
//   4. constant-state characteristic WENO extrapolation;
//   5. constant-state final ghost reconstruction;
//   6. oblique-wall E_r construction.
//
// Tan et al. Sec. 2.4 requires |E_r|=(r+1)^2 and p_r in P_r.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{FluidSide, Point, Polygon};

    const TOL_POLY: f64 = 5.0e-10;
    const TOL_CONST: f64 = 5.0e-10;

    fn max_abs(xs: &[f64]) -> f64 {
        xs.iter().copied().map(f64::abs).fold(0.0_f64, f64::max)
    }

    fn state_max_error(a: State, b: State) -> f64 {
        [
            (a.rho - b.rho).abs(),
            (a.mom_x - b.mom_x).abs(),
            (a.mom_y - b.mom_y).abs(),
            (a.ee - b.ee).abs(),
            (a.ei - b.ei).abs(),
            (a.er - b.er).abs(),
        ]
        .into_iter()
        .fold(0.0_f64, f64::max)
    }

    fn make_rect_field(
        nx: usize,
        ny: usize,
        lx: f64,
        ly: f64,
        value: State,
    ) -> field1::Field {
        let grid = GridInfo::new(
            nx,
            ny,
            lx / nx as f64,
            ly / ny as f64,
            0.0,
            0.0,
        );

        // CCW: bottom -> right -> top -> left.
        let outer = Polygon::new(
            vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: lx, y: 0.0 },
                Point { x: lx, y: ly },
                Point { x: 0.0, y: ly },
            ],
            FluidSide::Inside,
        );

        // Dummy obstacle outside the computational domain.
        let inner = Polygon::new(
            vec![
                Point { x: -1002.0, y: -1002.0 },
                Point { x: -1001.0, y: -1002.0 },
                Point { x: -1001.0, y: -1001.0 },
                Point { x: -1002.0, y: -1001.0 },
            ],
            FluidSide::Outside,
        );

        let outer_bc = vec![
            BCType::Wall,
            BCType::Wall,
            BCType::Wall,
            BCType::Wall,
        ];

        let inner_bc = vec![
            BCType::Wall,
            BCType::Wall,
            BCType::Wall,
            BCType::Wall,
        ];

        field1::Field::new(
            grid,
            inner_bc,
            outer_bc,
            value,
            outer,
            inner,
            0.0,
        )
    }

    // ------------------------------------------------------------------------
    // TEST 1
    //
    // fit_polynomial_surface must exactly reproduce a polynomial belonging
    // to P_4 even though the system is rectangular (25 data vs 15 DOF).
    //
    // The polynomial follows the SAME monomial ordering used by the code:
    // x^i y^j, total degree increasing.
    // ------------------------------------------------------------------------
    #[test]
    fn polynomial_least_squares_reproduces_degree4() {
        fn f(x: f64, y: f64) -> f64 {
            1.0
                + 2.0 * x
                + 3.0 * y
                + 4.0 * x * x
                + 5.0 * x * y
                + 6.0 * y * y
                + 7.0 * x.powi(3)
                + 8.0 * x.powi(2) * y
                + 9.0 * x * y.powi(2)
                + 10.0 * y.powi(3)
                + 11.0 * x.powi(4)
                + 12.0 * x.powi(3) * y
                + 13.0 * x.powi(2) * y.powi(2)
                + 14.0 * x * y.powi(3)
                + 15.0 * y.powi(4)
        }

        let h = 0.1;
        let mut data = Vec::new();

        for i in -2..=2 {
            for j in -2..=2 {
                let x = i as f64 * h;
                let y = j as f64 * h;
                data.push((x, y, f(x, y)));
            }
        }

        assert_eq!(data.len(), 25);

        let c = fit_polynomial_surface(&data, 4);
        assert_eq!(c.len(), 15);

        let mut max_err = 0.0_f64;
        for &(x, y, exact) in &data {
            let mut col = 0usize;
            let mut reconstructed = 0.0;

            for total in 0..=4 {
                for i in 0..=total {
                    let j = total - i;
                    reconstructed += c[col]
                        * x.powi(i as i32)
                        * y.powi(j as i32);
                    col += 1;
                }
            }

            max_err = max_err.max((reconstructed - exact).abs());
        }

        println!("degree-4 LS reproduction max error = {:.16e}", max_err);
        assert!(max_err < TOL_POLY);
    }

    // ------------------------------------------------------------------------
    // TEST 2
    //
    // Directly verify x_derivative_at_origin().
    //
    // For
    // f = 1 + 2x + ... + 4x^2 + ... + 7x^3 + ... + 11x^4 + ...
    //
    // at (0,0):
    // f       = 1
    // f_x     = 2
    // f_xx    = 2!*4  = 8
    // f_xxx   = 3!*7  = 42
    // f_xxxx  = 4!*11 = 264
    // ------------------------------------------------------------------------
    #[test]
    fn x_derivatives_at_origin_are_correct() {
        // Coefficient ordering produced by fit_polynomial_surface:
        //
        // total 0: 1
        // total 1: y, x
        // total 2: y^2, xy, x^2
        // total 3: y^3, xy^2, x^2y, x^3
        // total 4: y^4, xy^3, x^2y^2, x^3y, x^4
        let c = DVector::from_vec(vec![
            1.0,
            3.0, 2.0,
            6.0, 5.0, 4.0,
            10.0, 9.0, 8.0, 7.0,
            15.0, 14.0, 13.0, 12.0, 11.0,
        ]);

        let expected = [1.0, 2.0, 8.0, 42.0, 264.0];

        for k in 0..=4 {
            let got = x_derivative_at_origin(&c, 4, k);
            let err = (got - expected[k]).abs();

            println!(
                "k={} derivative: got={:.16e}, exact={:.16e}, err={:.3e}",
                k, got, expected[k], err
            );

            assert!(err < 1.0e-13);
        }
    }

    // ------------------------------------------------------------------------
    // TEST 3
    //
    // Paper Sec. 2.4:
    //     |E_r| = (r+1)^2
    //
    // On a vertical wall the chosen S_l should lie on successive vertical
    // grid lines and each S_l should contain r+1 consecutive y-indices.
    // ------------------------------------------------------------------------
    #[test]
    fn paper_stencil_cardinality_and_structure_vertical_wall() {
        let value = State::primi2con(
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        );

        let field = make_rect_field(80, 80, 1.0, 1.0, value);

        // Ghost immediately to the left, away from corners.
        let ghost_idx = (-1isize, 40isize);
        let p = Point {
            x: field.grid.x(ghost_idx.0),
            y: field.grid.y(ghost_idx.1),
        };
        let project = geometry::project(&field.outer_bound, p);

        println!(
            "P0=({:.8e},{:.8e}), n=({:.8e},{:.8e})",
            project.point.x,
            project.point.y,
            project.normal.x,
            project.normal.y
        );

        for r in 0..=4usize {
            let e = weno_stencil_extractor(project, &field, r);
            let width = r + 1;

            println!("r={}, |E_r|={}, points={:?}",
                r,
                e.len(),
                e.iter().map(|q| (q.0, q.1)).collect::<Vec<_>>()
            );

            assert_eq!(e.len(), width * width);

            // Extractor appends one complete S_l at a time.
            for l in 0..width {
                let s = &e[l * width..(l + 1) * width];

                // Same vertical grid line.
                let i0 = s[0].0;
                assert!(s.iter().all(|q| q.0 == i0));

                // Consecutive y indices.
                for a in 1..width {
                    assert_eq!(s[a].1, s[a - 1].1 + 1);
                }

                // Every point must be a true interior/fluid point.
                for q in s {
                    assert!(field.grid.is_in_domain((q.0, q.1)));
                    assert!(field.is_in_domain((q.0, q.1)));
                }
            }

            // Successive S_l must move into the fluid.
            for l in 1..width {
                let previous_i = e[(l - 1) * width].0;
                let current_i = e[l * width].0;
                assert_eq!(current_i, previous_i + 1);
            }
        }
    }

    // ------------------------------------------------------------------------
    // TEST 4
    //
    // Same structure test for a horizontal wall.
    // ------------------------------------------------------------------------
    #[test]
    fn paper_stencil_cardinality_and_structure_horizontal_wall() {
        let value = State::primi2con(
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        );

        let field = make_rect_field(80, 80, 1.0, 1.0, value);

        let ghost_idx = (40isize, -1isize);
        let p = Point {
            x: field.grid.x(ghost_idx.0),
            y: field.grid.y(ghost_idx.1),
        };
        let project = geometry::project(&field.outer_bound, p);

        for r in 0..=4usize {
            let e = weno_stencil_extractor(project, &field, r);
            let width = r + 1;

            println!("r={}, |E_r|={}, points={:?}",
                r,
                e.len(),
                e.iter().map(|q| (q.0, q.1)).collect::<Vec<_>>()
            );

            assert_eq!(e.len(), width * width);

            for l in 0..width {
                let s = &e[l * width..(l + 1) * width];

                // Same horizontal grid line.
                let j0 = s[0].1;
                assert!(s.iter().all(|q| q.1 == j0));

                // Consecutive x indices.
                for a in 1..width {
                    assert_eq!(s[a].0, s[a - 1].0 + 1);
                }

                for q in s {
                    assert!(field.grid.is_in_domain((q.0, q.1)));
                    assert!(field.is_in_domain((q.0, q.1)));
                }
            }

            // Bottom outward normal is -y, so inward is +y.
            for l in 1..width {
                let previous_j = e[(l - 1) * width].1;
                let current_j = e[l * width].1;
                assert_eq!(current_j, previous_j + 1);
            }
        }
    }

    // ------------------------------------------------------------------------
    // TEST 5
    //
    // For a constant state, all characteristic variables are constant.
    // Therefore WENO extrapolation must give:
    //
    //     k=0 : finite constant characteristic value
    //     k>=1: zero derivative (to roundoff)
    //
    // This directly exercises E_r -> preprocessing -> LS -> beta -> weights.
    // ------------------------------------------------------------------------
    #[test]
    fn characteristic_weno_constant_state_has_zero_higher_derivatives() {
        let value = State::primi2con(
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        );

        // Square grid keeps this test within the paper's original mesh setting.
        let field = make_rect_field(80, 80, 1.0, 1.0, value);

        let ghost_idx = (-1isize, 40isize);
        let p = Point {
            x: field.grid.x(ghost_idx.0),
            y: field.grid.y(ghost_idx.1),
        };
        let project = geometry::project(&field.outer_bound, p);

        let v0 = weno_extrapolation(project, &field, 0);
        println!("V*(0) = {:?}", v0);
        assert!(v0.iter().all(|x| x.is_finite()));

        for k in 1..=4usize {
            let vk = weno_extrapolation(project, &field, k);
            let err = max_abs(&vk);

            println!("max |V*({})| = {:.16e}, V={:?}", k, err, vk);

            assert!(
                err < TOL_CONST,
                "constant characteristic field produced nonzero derivative k={}, max={}",
                k,
                err
            );
        }
    }

    // ------------------------------------------------------------------------
    // TEST 6
    //
    // Full final ghost reconstruction for a stationary constant state.
    // This exercises the wall row substitution + ILW + Taylor expansion.
    // ------------------------------------------------------------------------
    #[test]
    fn final_wall_ghost_preserves_stationary_constant_state() {
        let exact = State::primi2con(
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        );

        let field = make_rect_field(80, 80, 1.0, 1.0, exact);

        let indices = [
            (-1, 40), (-2, 40), (-3, 40),
            (80, 40), (81, 40), (82, 40),
            (40, -1), (40, -2), (40, -3),
            (40, 80), (40, 81), (40, 82),
        ];

        let mut worst = 0.0_f64;

        for idx in indices {
            let got = field.get(idx);
            let err = state_max_error(got, exact);
            worst = worst.max(err);
            println!("ghost {:?}: err={:.16e}", idx, err);
        }

        println!("worst final ghost constant-state error={:.16e}", worst);
        assert!(worst < TOL_CONST);
    }

    // ------------------------------------------------------------------------
    // TEST 7
    //
    // Oblique boundary: verify that E_r still has the paper cardinality,
    // consists only of fluid points, and each S_l is a consecutive Cartesian
    // row/column.  This is the important geometry test beyond axis-aligned
    // rectangles.
    //
    // Triangle:
    //     (0,0) -- (1,0)
    //        \       /
    //          (0,1)
    //
    // Fluid is inside.
    // ------------------------------------------------------------------------
    #[test]
    fn oblique_boundary_paper_stencil_is_valid() {
        let nx = 100usize;
        let ny = 100usize;
        let grid = GridInfo::new(
            nx, ny,
            1.0 / nx as f64,
            1.0 / ny as f64,
            0.0, 0.0,
        );

        let value = State::primi2con(
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        );

        let outer = Polygon::new(
            vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 0.0, y: 1.0 },
            ],
            FluidSide::Inside,
        );

        let inner = Polygon::new(
            vec![
                Point { x: -1002.0, y: -1002.0 },
                Point { x: -1001.0, y: -1002.0 },
                Point { x: -1001.0, y: -1001.0 },
                Point { x: -1002.0, y: -1001.0 },
            ],
            FluidSide::Outside,
        );

        let field = field1::Field::new(
            grid,
            vec![BCType::Wall; 4],
            vec![BCType::Wall; 3],
            value,
            outer,
            inner,
            0.0
        );

        // Point outside the hypotenuse x+y=1, away from vertices.
        let p = Point { x: 0.62, y: 0.42 };
        assert!(!field.outer_bound.is_fluid(p));

        let project = geometry::project(&field.outer_bound, p);

        println!(
            "oblique P=({:.6},{:.6}), P0=({:.6},{:.6}), n=({:.6},{:.6})",
            p.x, p.y,
            project.point.x, project.point.y,
            project.normal.x, project.normal.y
        );

        for r in 0..=4usize {
            let e = weno_stencil_extractor(project, &field, r);
            let width = r + 1;

            assert_eq!(e.len(), width * width);

            println!(
                "oblique r={} points={:?}",
                r,
                e.iter().map(|q| (q.0, q.1)).collect::<Vec<_>>()
            );

            for q in &e {
                assert!(field.grid.is_in_domain((q.0, q.1)));
                assert!(field.is_in_domain((q.0, q.1)));
            }

            // Determine orientation from the first S_l and ensure every S_l
            // is a consecutive Cartesian row/column.
            for l in 0..width {
                let s = &e[l * width..(l + 1) * width];

                if width == 1 {
                    continue;
                }

                let vertical = s[0].0 == s[1].0;

                if vertical {
                    assert!(s.iter().all(|q| q.0 == s[0].0));
                    for a in 1..width {
                        assert_eq!(s[a].1, s[a - 1].1 + 1);
                    }
                } else {
                    assert!(s.iter().all(|q| q.1 == s[0].1));
                    for a in 1..width {
                        assert_eq!(s[a].0, s[a - 1].0 + 1);
                    }
                }
            }
        }
    }

    #[test]
fn constant_polynomial_fit_has_zero_high_order_coefficients() {
    let h = 0.0125;

    let mut data = Vec::new();

    for i in -2..=2 {
        for j in -2..=2 {
            let x = i as f64 * h;
            let y = j as f64 * h;

            data.push((x, y, 1.0));
        }
    }

    let c = fit_polynomial_surface(&data, 4);

    println!("coefficients = {:?}", c);

    println!("constant = {:.16e}", c[0]);

    for i in 1..c.len() {
        println!("c[{}] = {:.16e}", i, c[i]);
    }

    assert!((c[0] - 1.0).abs() < 1e-12);
}
}