use crate::constant;
use crate::state::State;

pub fn source(s: State) -> State {
    let te = s.te();
    let tr = s.tr();
    let ti = s.ti();

    State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: -constant::OMEGA_EI * (te - ti) - constant::OMEGA_ER * (te.powi(4) - tr.powi(4)),
        ei: constant::OMEGA_EI * (te - ti),
        er: constant::OMEGA_ER * (te.powi(4) - tr.powi(4)),
    }
}

#[inline(always)]
pub fn rt_source(s: State)-> State {
    State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: s.rho,
        ee: 0.0,
        ei: s.mom_y,
        er: 0.0,
    }
}


#[inline(always)]
pub fn mms_63_source(x: f64, y: f64, t: f64) -> State {
    let xi = x + y - 2.0 * t;

    let s = xi.sin();
    let c = xi.cos();

    // dp/dxi
    //
    // p = 14/3 + (7/15) sin(xi) + (2/5) cos(xi)
    //
    // therefore
    //
    // dp/dxi = (7/15) cos(xi) - (2/5) sin(xi)

    let dp = (7.0 / 15.0) * c - (2.0 / 5.0) * s;

    State {
        // Exact continuity equation already satisfied.
        rho: 0.0,

        // x/y momentum:
        //
        // S_mx = dp/dx = dp/dxi
        // S_my = dp/dy = dp/dxi
        mom_x: dp,
        mom_y: dp,

        // Modified three-energy equations.
        //
        // Each source reduces to (2/3) dp/dxi.
        ee: (2.0 / 3.0) * dp,
        ei: (2.0 / 3.0) * dp,
        er: (2.0 / 3.0) * dp,
    }
}

#[inline(always)]
pub fn wall_mms_source(x: f64, y: f64, t: f64) -> State {
    let sx = x.sin();
    let cx = x.cos();

    let sy = y.sin();
    let cy = y.cos();

    let st = t.sin();
    let ct = t.cos();

    // ============================================================
    // rho and derivatives
    // ============================================================

    let rho = 1.0 + 0.1 * sx * cy * ct;

    let rho_t = -0.1 * sx * cy * st;
    let rho_x =  0.1 * cx * cy * ct;
    let rho_y = -0.1 * sx * sy * ct;

    // ============================================================
    // velocity and derivatives
    // ============================================================

    let u = 1.0 + 0.2 * cx * cy * ct;
    let v =       0.2 * sx * sy * ct;

    let u_t = -0.2 * cx * cy * st;
    let u_x = -0.2 * sx * cy * ct;
    let u_y = -0.2 * cx * sy * ct;

    let v_t = -0.2 * sx * sy * st;
    let v_x =  0.2 * cx * sy * ct;
    let v_y =  0.2 * sx * cy * ct;

    // ============================================================
    // internal-energy densities q_alpha = rho e_alpha
    // ============================================================

    let qe = 3.0 + 0.2 * cx * cy * st;

    let qe_t =  0.2 * cx * cy * ct;
    let qe_x = -0.2 * sx * cy * st;
    let qe_y = -0.2 * cx * sy * st;


    let qi = 3.0 + 0.15 * sx * cy * ct;

    let qi_t = -0.15 * sx * cy * st;
    let qi_x =  0.15 * cx * cy * ct;
    let qi_y = -0.15 * sx * sy * ct;


    let qr = 2.0 + 0.1 * (2.0 * x).cos() * cy * st;

    let qr_t =  0.1 * (2.0 * x).cos() * cy * ct;
    let qr_x = -0.2 * (2.0 * x).sin() * cy * st;
    let qr_y = -0.1 * (2.0 * x).cos() * sy * st;

    // ============================================================
    // pressure
    //
    // gamma_e = gamma_i = 5/3
    // gamma_r = 4/3
    // ============================================================

    let pe = (2.0 / 3.0) * qe;
    let pi = (2.0 / 3.0) * qi;
    let pr = (1.0 / 3.0) * qr;

    let pe_x = (2.0 / 3.0) * qe_x;
    let pe_y = (2.0 / 3.0) * qe_y;

    let pi_x = (2.0 / 3.0) * qi_x;
    let pi_y = (2.0 / 3.0) * qi_y;

    let pr_x = (1.0 / 3.0) * qr_x;
    let pr_y = (1.0 / 3.0) * qr_y;

    let p_x = pe_x + pi_x + pr_x;
    let p_y = pe_y + pi_y + pr_y;

    // ============================================================
    // Modified-energy kinetic share
    // ============================================================

    let w2 = u * u + v * v;

    let ks = rho * w2 / 6.0;

    let ks_t =
        (rho_t * w2
            + 2.0 * rho * (u * u_t + v * v_t))
            / 6.0;

    let ks_x =
        (rho_x * w2
            + 2.0 * rho * (u * u_x + v * v_x))
            / 6.0;

    let ks_y =
        (rho_y * w2
            + 2.0 * rho * (u * u_y + v * v_y))
            / 6.0;

    let ee = qe + ks;
    let ei = qi + ks;
    let er = qr + ks;

    let ee_t = qe_t + ks_t;
    let ee_x = qe_x + ks_x;
    let ee_y = qe_y + ks_y;

    let ei_t = qi_t + ks_t;
    let ei_x = qi_x + ks_x;
    let ei_y = qi_y + ks_y;

    let er_t = qr_t + ks_t;
    let er_x = qr_x + ks_x;
    let er_y = qr_y + ks_y;

    // ============================================================
    // Nonconservative pressure combinations
    // ============================================================

    let ne_x = 2.0 * pe_x - pi_x - pr_x;
    let ne_y = 2.0 * pe_y - pi_y - pr_y;

    let ni_x = 2.0 * pi_x - pe_x - pr_x;
    let ni_y = 2.0 * pi_y - pe_y - pr_y;

    let nr_x = 2.0 * pr_x - pe_x - pi_x;
    let nr_y = 2.0 * pr_y - pe_y - pi_y;

    // ============================================================
    // Mass
    // ============================================================

    let s_rho =
        rho_t
        + u * rho_x
        + v * rho_y
        + rho * (u_x + v_y);

    // ============================================================
    // x momentum
    // ============================================================

    let s_mx =
        rho_t * u
        + rho * u_t

        + rho_x * u * u
        + 2.0 * rho * u * u_x
        + p_x

        + rho_y * u * v
        + rho * u_y * v
        + rho * u * v_y;

    // ============================================================
    // y momentum
    // ============================================================

    let s_my =
        rho_t * v
        + rho * v_t

        + rho_x * u * v
        + rho * u_x * v
        + rho * u * v_x

        + rho_y * v * v
        + 2.0 * rho * v * v_y
        + p_y;

    // ============================================================
    // Energy equations
    // ============================================================

    let s_ee =
        ee_t
        + u * (ee_x + pe_x)
        + (ee + pe) * u_x
        + v * (ee_y + pe_y)
        + (ee + pe) * v_y
        - (u * ne_x + v * ne_y) / 3.0;

    let s_ei =
        ei_t
        + u * (ei_x + pi_x)
        + (ei + pi) * u_x
        + v * (ei_y + pi_y)
        + (ei + pi) * v_y
        - (u * ni_x + v * ni_y) / 3.0;

    let s_er =
        er_t
        + u * (er_x + pr_x)
        + (er + pr) * u_x
        + v * (er_y + pr_y)
        + (er + pr) * v_y
        - (u * nr_x + v * nr_y) / 3.0;

    State {
        rho: s_rho,
        mom_x: s_mx,
        mom_y: s_my,
        ee: s_ee,
        ei: s_ei,
        er: s_er,
    }
}