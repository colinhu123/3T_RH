use crate::state;

/// N(u): the pointwise pressure-split vector.
///     N_ee = 2*p_e - p_i - p_r
///     N_ei = 2*p_i - p_e - p_r
///     N_er = 2*p_r - p_e - p_i
/// Matches N() in the reference C++ implementation.
pub fn n_vector(u: &state::State) -> state::State {
    let (pe, pi, pr) = u.pressure_spilit();

    n_vector_from_pressures(pe, pi, pr)
}

#[inline]
fn n_vector_from_pressures(pe: f64, pi: f64, pr: f64) -> state::State {
    state::State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: 2.0*pe - pi - pr,
        ei: 2.0*pi - pe - pr,
        er: 2.0*pr - pe - pi,
    }
}

fn sixth_order_derivative(
    qm3: f64,
    qm2: f64,
    qm1: f64,
    qp1: f64,
    qp2: f64,
    qp3: f64,
    dx: f64,
) -> f64 {
    (
        qp3
        - 9.0*qp2
        + 45.0*qp1
        - 45.0*qm1
        + 9.0*qm2
        - qm3
    )
    / (60.0*dx)
}

/// Fixed-coefficient 7-point interpolation used to reconstruct a cell-edge
/// value from 7 point values centered on the interpolating cell.
/// Matches q_pulse() in the reference C++ implementation exactly.
fn q_pulse(b: &[f64; 7]) -> f64 {
    -0.00714285714285723*b[0]
    + 0.0595238095238118*b[1]
    - 0.240476190476197*b[2]
    + 0.759523809523819*b[3]
    + 0.509523809523803*b[4]
    - 0.0904761904761891*b[5]
    + 0.00952380952380977*b[6]
}

/// (∂N/∂x)_i : the sixth-order central-difference derivative of the
/// pressure-split components, computed from a 7-point stencil
/// [i-3, i-2, i-1, i, i+1, i+2, i+3] (index 3 = i is unused, matching
/// the reference deN()).
pub fn dN(stencil: &[state::State; 7], dx: f64) -> state::State {
    let mut pe = [0.0; 7];
    let mut pi = [0.0; 7];
    let mut pr = [0.0; 7];
    for i in 0..7 {
        let (p1, p2, p3) = stencil[i].pressure_spilit();
        pe[i] = p1;
        pi[i] = p2;
        pr[i] = p3;
    }

    let dpe = sixth_order_derivative(pe[0], pe[1], pe[2], pe[4], pe[5], pe[6], dx);
    let dpi = sixth_order_derivative(pi[0], pi[1], pi[2], pi[4], pi[5], pi[6], dx);
    let dpr = sixth_order_derivative(pr[0], pr[1], pr[2], pr[4], pr[5], pr[6], dx);

    state::State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: 2.0*dpe - dpi - dpr,
        ei: 2.0*dpi - dpe - dpr,
        er: 2.0*dpr - dpe - dpi,
    }
}

/// [N]_{base+1/2}: the jump of N across the interface immediately to the
/// right of `base`, using an 8-point stencil [base-3, ..., base+4].
///
/// B2 = points[0..7]        = [base-3 .. base+3], q_pulse(B2) = q_base(x_{base+1/2})
/// B1 = reverse(points[1..8]) = [base+4 .. base-2], q_pulse(B1) = q_{base+1}(x_{base+1/2})
///
/// (B1 works because q_pulse's coefficients are symmetric under reflection,
/// so reversing the mirror-image stencil and reapplying q_pulse reconstructs
/// the left-edge value of the neighboring cell's own interpolant.)
///
/// Matches N0() in the reference C++ implementation.
fn n_jump(stencil8: &[state::State; 8]) -> state::State {
    let mut n_ee = [0.0; 8];
    let mut n_ei = [0.0; 8];
    let mut n_er = [0.0; 8];
    for k in 0..8 {
        let n = n_vector(&stencil8[k]);
        n_ee[k] = n.ee;
        n_ei[k] = n.ei;
        n_er[k] = n.er;
    }

    let jump_component = |arr: &[f64; 8]| -> f64 {
        let b2: [f64; 7] = [arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]];
        let mut b1: [f64; 7] = [arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]];
        b1.reverse();
        q_pulse(&b1) - q_pulse(&b2)
    };

    state::State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: jump_component(&n_ee),
        ei: jump_component(&n_ei),
        er: jump_component(&n_er),
    }
}

/// N1 at cell i: the full nonconservative convection term, Eq. (3.5)-(3.7).
///
/// `stencil` is a 9-point window [i-4, i-3, ..., i, ..., i+3, i+4]
/// (local index 4 = cell i). This is wide enough to supply:
///   - deN's 7-point window [i-3..i+3]           (local indices 1..8)
///   - N0(i-1)'s 8-point window [i-4..i+3]       (local indices 0..8)
///   - N0(i)'s 8-point window   [i-3..i+4]       (local indices 1..9)
///
/// Matches N1_cal() in the reference C++ implementation, including the
/// division of the upwind jump terms by dx.
pub fn nonconservative_direction(
    stencil: &[state::State; 9],
    ds: f64,
    dir: state::Direction,
) -> state::State {
    let center = &stencil[4];
    let v = dir.velocity(center);

    let deriv_stencil: [state::State; 7] = [
        stencil[1],
        stencil[2],
        stencil[3],
        stencil[4],
        stencil[5],
        stencil[6],
        stencil[7],
    ];

    let derivative = dN(&deriv_stencil, ds);

    let stencil_m: [state::State; 8] = [
        stencil[0],
        stencil[1],
        stencil[2],
        stencil[3],
        stencil[4],
        stencil[5],
        stencil[6],
        stencil[7],
    ];

    let stencil_p: [state::State; 8] = [
        stencil[1],
        stencil[2],
        stencil[3],
        stencil[4],
        stencil[5],
        stencil[6],
        stencil[7],
        stencil[8],
    ];

    let jump_m = n_jump(&stencil_m);
    let jump_p = n_jump(&stencil_p);

    let vm = dir.velocity(&stencil[3]);
    let vp = dir.velocity(&stencil[5]);

    let coef_left = v.max(vm).max(0.0) / 3.0;
    let coef_right = v.min(vp).min(0.0) / 3.0;

    state::State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,

        ee: v/3.0 * derivative.ee
            + (coef_left*jump_m.ee + coef_right*jump_p.ee) / ds,

        ei: v/3.0 * derivative.ei
            + (coef_left*jump_m.ei + coef_right*jump_p.ei) / ds,

        er: v/3.0 * derivative.er
            + (coef_left*jump_m.er + coef_right*jump_p.er) / ds,
    }
}

pub fn nonconservative_x(
    stencil: &[state::State; 9],
    dx: f64,
) -> state::State {
    nonconservative_direction(stencil, dx, state::Direction::X)
}

pub fn nonconservative_y(
    stencil: &[state::State; 9],
    dy: f64,
) -> state::State {
    nonconservative_direction(stencil, dy, state::Direction::Y)
}

/// Hot-path variant: identical numerics, but reads pressures/velocities
/// from per-stage precomputed `Derived` quantities instead of recomputing
/// `pressure_spilit` / momentum divisions pointwise.
pub fn nonconservative_x_pre(
    d: &[state::Derived; 9],
    dx: f64,
) -> state::State {
    nonconservative_direction_pre(d, dx, state::Direction::X)
}

pub fn nonconservative_y_pre(
    d: &[state::Derived; 9],
    dy: f64,
) -> state::State {
    nonconservative_direction_pre(d, dy, state::Direction::Y)
}

fn dN_pre(stencil: &[state::Derived; 7], dx: f64) -> state::State {
    let mut pe = [0.0; 7];
    let mut pi = [0.0; 7];
    let mut pr = [0.0; 7];
    for i in 0..7 {
        pe[i] = stencil[i].pe;
        pi[i] = stencil[i].pi;
        pr[i] = stencil[i].pr;
    }

    let dpe = sixth_order_derivative(pe[0], pe[1], pe[2], pe[4], pe[5], pe[6], dx);
    let dpi = sixth_order_derivative(pi[0], pi[1], pi[2], pi[4], pi[5], pi[6], dx);
    let dpr = sixth_order_derivative(pr[0], pr[1], pr[2], pr[4], pr[5], pr[6], dx);

    state::State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: 2.0*dpe - dpi - dpr,
        ei: 2.0*dpi - dpe - dpr,
        er: 2.0*dpr - dpe - dpi,
    }
}

fn n_jump_pre(stencil8: &[state::Derived; 8]) -> state::State {
    let mut n_ee = [0.0; 8];
    let mut n_ei = [0.0; 8];
    let mut n_er = [0.0; 8];
    for k in 0..8 {
        let n = n_vector_from_pressures(stencil8[k].pe, stencil8[k].pi, stencil8[k].pr);
        n_ee[k] = n.ee;
        n_ei[k] = n.ei;
        n_er[k] = n.er;
    }

    let jump_component = |arr: &[f64; 8]| -> f64 {
        let b2: [f64; 7] = [arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]];
        let mut b1: [f64; 7] = [arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]];
        b1.reverse();
        q_pulse(&b1) - q_pulse(&b2)
    };

    state::State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: jump_component(&n_ee),
        ei: jump_component(&n_ei),
        er: jump_component(&n_er),
    }
}

/// Hot-path variant of `nonconservative_direction`, numerically identical.
pub fn nonconservative_direction_pre(
    d: &[state::Derived; 9],
    ds: f64,
    dir: state::Direction,
) -> state::State {
    let v = if matches!(dir, state::Direction::X) { d[4].u } else { d[4].v };

    let deriv_stencil: [state::Derived; 7] = [
        d[1], d[2], d[3], d[4], d[5], d[6], d[7],
    ];

    let derivative = dN_pre(&deriv_stencil, ds);

    let stencil_m: [state::Derived; 8] = [
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
    ];

    let stencil_p: [state::Derived; 8] = [
        d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8],
    ];

    let jump_m = n_jump_pre(&stencil_m);
    let jump_p = n_jump_pre(&stencil_p);

    let vm = if matches!(dir, state::Direction::X) { d[3].u } else { d[3].v };
    let vp = if matches!(dir, state::Direction::X) { d[5].u } else { d[5].v };

    let coef_left = v.max(vm).max(0.0) / 3.0;
    let coef_right = v.min(vp).min(0.0) / 3.0;

    state::State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,

        ee: v/3.0 * derivative.ee
            + (coef_left*jump_m.ee + coef_right*jump_p.ee) / ds,

        ei: v/3.0 * derivative.ei
            + (coef_left*jump_m.ei + coef_right*jump_p.ei) / ds,

        er: v/3.0 * derivative.er
            + (coef_left*jump_m.er + coef_right*jump_p.er) / ds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;

    fn make_state(
        rho: f64,
        mom_x: f64,
        mom_y: f64,
        ee: f64,
        ei: f64,
        er: f64,
    ) -> State {
        State {
            rho,
            mom_x,
            mom_y,
            ee,
            ei,
            er,
        }
    }

    fn constant_stencil7(s: State) -> [State; 7] {
        [s; 7]
    }

    fn constant_stencil8(s: State) -> [State; 8] {
        [s; 8]
    }

    fn constant_stencil9(s: State) -> [State; 9] {
        [s; 9]
    }

    // ============================================================
    // n_vector
    // ============================================================

    #[test]
    fn test_n_vector_zero_pressure() {
        let s = make_state(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );

        let n = n_vector(&s);

        assert_eq!(n.rho, 0.0);
        assert_eq!(n.mom_x, 0.0);
        assert_eq!(n.mom_y, 0.0);
        assert_eq!(n.ee, 0.0);
        assert_eq!(n.ei, 0.0);
        assert_eq!(n.er, 0.0);
    }

    #[test]
    fn test_n_vector_pressure_split_inverse() {
        /*
        The transformation is:

        Ne = 2Pe - Pi - Pr
        Ni = 2Pi - Pe - Pr
        Nr = 2Pr - Pe - Pi
        */

        let s = make_state(
            1.0,
            0.0,
            0.0,
            5.0,
            5.0,
            5.0,
        );

        let n = n_vector(&s);

        assert!((n.ee - 5.0 / 3.0).abs() < 1e-12);
        assert!((n.ei - 5.0 / 3.0).abs() < 1e-12);
        assert!((n.er + 10.0 / 3.0).abs() < 1e-12);
    }

    // ============================================================
    // sixth_order_derivative
    // ============================================================

    #[test]
    fn test_sixth_order_derivative_constant_function() {
        let result = sixth_order_derivative(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.1,
        );

        assert!(result.abs() < 1e-14);
    }

    #[test]
    fn test_sixth_order_derivative_linear_function() {
        /*
        For f(s) = s:

        qm3 = -3
        qm2 = -2
        qm1 = -1
        qp1 =  1
        qp2 =  2
        qp3 =  3

        derivative should be 1.
        */

        let result = sixth_order_derivative(
            -3.0,
            -2.0,
            -1.0,
            1.0,
            2.0,
            3.0,
            1.0,
        );

        assert!((result - 1.0).abs() < 1e-12);
    }

    // ============================================================
    // dN
    // ============================================================

    #[test]
    fn test_dN_constant_state() {
        let s = make_state(
            1.0,
            0.0,
            0.0,
            3.0,
            3.0,
            3.0,
        );

        let stencil = constant_stencil7(s);

        let result = dN(&stencil, 1.0);

        assert_eq!(result.rho, 0.0);
        assert_eq!(result.mom_x, 0.0);
        assert_eq!(result.mom_y, 0.0);

        assert!(result.ee.abs() < 1e-14);
        assert!(result.ei.abs() < 1e-14);
        assert!(result.er.abs() < 1e-14);
    }

    // ============================================================
    // n_jump
    // ============================================================

    #[test]
    fn test_n_jump_constant_state() {
        let s = make_state(
            1.0,
            0.5,
            0.7,
            2.0,
            2.0,
            2.0,
        );

        let stencil = constant_stencil8(s);

        let jump = n_jump(&stencil);

        assert_eq!(jump.rho, 0.0);
        assert_eq!(jump.mom_x, 0.0);
        assert_eq!(jump.mom_y, 0.0);

        assert!(jump.ee.abs() < 1e-10);
        assert!(jump.ei.abs() < 1e-10);
        assert!(jump.er.abs() < 1e-10);
    }

    // ============================================================
    // Direction
    // ============================================================

    #[test]
    fn test_direction_velocity_x() {
        let s = make_state(
            2.0,
            6.0,
            10.0,
            1.0,
            1.0,
            1.0,
        );

        let velocity = state::Direction::X.velocity(&s);

        assert!((velocity - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_direction_velocity_y() {
        let s = make_state(
            2.0,
            6.0,
            10.0,
            1.0,
            1.0,
            1.0,
        );

        let velocity = state::Direction::Y.velocity(&s);

        assert!((velocity - 5.0).abs() < 1e-14);
    }

    // ============================================================
    // nonconservative: constant state
    // ============================================================

    #[test]
    fn test_nonconservative_x_constant_state() {
        let s = make_state(
            1.0,
            0.5,
            0.8,
            2.0,
            2.0,
            2.0,
        );

        let stencil = constant_stencil9(s);

        let result = nonconservative_x(&stencil, 1.0);

        /*
        Uniform state:

            dN/dx = 0
            jumps = 0

        Therefore the x-direction nonconservative
        contribution vanishes.
        */

        assert_eq!(result.rho, 0.0);
        assert_eq!(result.mom_x, 0.0);
        assert_eq!(result.mom_y, 0.0);

        assert!(result.ee.abs() < 1e-10);
        assert!(result.ei.abs() < 1e-10);
        assert!(result.er.abs() < 1e-10);
    }

    #[test]
    fn test_nonconservative_y_constant_state() {
        let s = make_state(
            1.0,
            0.5,
            0.8,
            2.0,
            2.0,
            2.0,
        );

        let stencil = constant_stencil9(s);

        let result = nonconservative_y(&stencil, 1.0);

        /*
        Uniform state:

            dN/dy = 0
            jumps = 0

        Therefore the y-direction nonconservative
        contribution vanishes.
        */

        assert_eq!(result.rho, 0.0);
        assert_eq!(result.mom_x, 0.0);
        assert_eq!(result.mom_y, 0.0);

        assert!(result.ee.abs() < 1e-10);
        assert!(result.ei.abs() < 1e-10);
        assert!(result.er.abs() < 1e-10);
    }

    // ============================================================
    // zero velocity
    // ============================================================

    #[test]
    fn test_nonconservative_x_zero_velocity() {
        let stencil = [
            make_state(1.0, 0.0, 1.0, 0.9, 1.9, 2.9),
            make_state(1.0, 0.0, 1.0, 1.0, 2.0, 3.0),
            make_state(1.0, 0.0, 1.0, 1.1, 2.1, 3.1),
            make_state(1.0, 0.0, 1.0, 1.2, 2.2, 3.2),
            make_state(1.0, 0.0, 1.0, 1.3, 2.3, 3.3),
            make_state(1.0, 0.0, 1.0, 1.4, 2.4, 3.4),
            make_state(1.0, 0.0, 1.0, 1.5, 2.5, 3.5),
            make_state(1.0, 0.0, 1.0, 1.6, 2.6, 3.6),
            make_state(1.0, 0.0, 1.0, 1.7, 2.7, 3.7),
        ];

        /*
        mom_x = 0 everywhere.

        mom_y is deliberately nonzero.

        Therefore this checks that Direction::X really
        uses mom_x rather than mom_y.
        */

        let result = nonconservative_x(&stencil, 1.0);

        assert_eq!(result.rho, 0.0);
        assert_eq!(result.mom_x, 0.0);
        assert_eq!(result.mom_y, 0.0);

        assert!(result.ee.abs() < 1e-10);
        assert!(result.ei.abs() < 1e-10);
        assert!(result.er.abs() < 1e-10);
    }

    #[test]
    fn test_nonconservative_y_zero_velocity() {
        let stencil = [
            make_state(1.0, 1.0, 0.0, 0.9, 1.9, 2.9),
            make_state(1.0, 1.0, 0.0, 1.0, 2.0, 3.0),
            make_state(1.0, 1.0, 0.0, 1.1, 2.1, 3.1),
            make_state(1.0, 1.0, 0.0, 1.2, 2.2, 3.2),
            make_state(1.0, 1.0, 0.0, 1.3, 2.3, 3.3),
            make_state(1.0, 1.0, 0.0, 1.4, 2.4, 3.4),
            make_state(1.0, 1.0, 0.0, 1.5, 2.5, 3.5),
            make_state(1.0, 1.0, 0.0, 1.6, 2.6, 3.6),
            make_state(1.0, 1.0, 0.0, 1.7, 2.7, 3.7),
        ];

        /*
        mom_y = 0 everywhere.

        mom_x is deliberately nonzero.

        Therefore this checks that Direction::Y really
        uses mom_y rather than mom_x.
        */

        let result = nonconservative_y(&stencil, 1.0);

        assert_eq!(result.rho, 0.0);
        assert_eq!(result.mom_x, 0.0);
        assert_eq!(result.mom_y, 0.0);

        assert!(result.ee.abs() < 1e-10);
        assert!(result.ei.abs() < 1e-10);
        assert!(result.er.abs() < 1e-10);
    }

    // ============================================================
    // wrapper consistency
    // ============================================================

    #[test]
    fn test_x_wrapper_matches_direction_function() {
        let stencil = [
            make_state(1.0, 0.5, 0.8, 0.9, 1.9, 2.9),
            make_state(1.0, 0.5, 0.8, 1.0, 2.0, 3.0),
            make_state(1.0, 0.5, 0.8, 1.1, 2.1, 3.1),
            make_state(1.0, 0.5, 0.8, 1.2, 2.2, 3.2),
            make_state(1.0, 0.5, 0.8, 1.3, 2.3, 3.3),
            make_state(1.0, 0.5, 0.8, 1.4, 2.4, 3.4),
            make_state(1.0, 0.5, 0.8, 1.5, 2.5, 3.5),
            make_state(1.0, 0.5, 0.8, 1.6, 2.6, 3.6),
            make_state(1.0, 0.5, 0.8, 1.7, 2.7, 3.7),
        ];

        let a = nonconservative_x(&stencil, 0.5);
        let b = nonconservative_direction(
            &stencil,
            0.5,
            state::Direction::X,
        );

        assert!((a.ee - b.ee).abs() < 1e-14);
        assert!((a.ei - b.ei).abs() < 1e-14);
        assert!((a.er - b.er).abs() < 1e-14);
    }

    #[test]
    fn test_y_wrapper_matches_direction_function() {
        let stencil = [
            make_state(1.0, 0.5, 0.8, 0.9, 1.9, 2.9),
            make_state(1.0, 0.5, 0.8, 1.0, 2.0, 3.0),
            make_state(1.0, 0.5, 0.8, 1.1, 2.1, 3.1),
            make_state(1.0, 0.5, 0.8, 1.2, 2.2, 3.2),
            make_state(1.0, 0.5, 0.8, 1.3, 2.3, 3.3),
            make_state(1.0, 0.5, 0.8, 1.4, 2.4, 3.4),
            make_state(1.0, 0.5, 0.8, 1.5, 2.5, 3.5),
            make_state(1.0, 0.5, 0.8, 1.6, 2.6, 3.6),
            make_state(1.0, 0.5, 0.8, 1.7, 2.7, 3.7),
        ];

        let a = nonconservative_y(&stencil, 0.25);
        let b = nonconservative_direction(
            &stencil,
            0.25,
            state::Direction::Y,
        );

        assert!((a.ee - b.ee).abs() < 1e-14);
        assert!((a.ei - b.ei).abs() < 1e-14);
        assert!((a.er - b.er).abs() < 1e-14);
    }
}