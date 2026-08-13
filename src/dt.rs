use crate::state::State;
use crate::constant::*;


pub fn get_local_dt(state1: State, dx: f64, dy: f64) -> f64 {
    let u = state1.mom_x/state1.rho;
    let v = state1.mom_y/state1.rho;

    let ee = state1.ee/state1.rho - (u.powi(2) + v.powi(2))/6.0;
    let ei = state1.ei/state1.rho - (u.powi(2) + v.powi(2))/6.0;
    let er = state1.er/state1.rho - (u.powi(2) + v.powi(2))/6.0;
    let gi = GAMMA_I - 1.0;
    let ge = GAMMA_E - 1.0;
    let gr = GAMMA_R - 1.0;
    let cs = (GAMMA_E*ge*ee + GAMMA_I*gi*ei + GAMMA_R*gr*er).sqrt();

    let s1 = ((A*OMEGA_EI*CVE.powi(4) + 
            A*OMEGA_EI*CVE.powi(3)*CVI + 
            4.0*A*OMEGA_ER*CVI*ee.powi(3) + 
            OMEGA_ER*CVE.powi(4)*CVI*state1.rho).powi(2) - 
            4.0*A*OMEGA_EI*OMEGA_ER*CVE.powi(4)*CVI*(4.0*A*ee.powi(3)+CVE.powi(4)*state1.rho+CVE.powi(3)*CVI*state1.rho)).sqrt();

    let s2 = -(A*OMEGA_EI*CVE.powi(4) + A*OMEGA_EI*CVE.powi(3)*CVI + 4.0*A*OMEGA_ER*CVI*ee.powi(3) + OMEGA_ER*CVE.powi(4)*CVI*state1.rho);

    let s = 2.0*A*CVE.powi(4)*CVI*state1.rho;

    let alpha1 = ((-s1 + s2)/s).abs();
    let alpha2 = ((s1 + s2)/s).abs();

    let sj = alpha1.max(alpha2);
    let dj = (KAPPA_E/(CVE*state1.rho)).max(KAPPA_I/(CVI*state1.rho)).max(KAPPA_R/A);

    let vj = (u.abs()+cs)/dx + 2.0*dj/(dx.powi(2)+dy.powi(2)) + sj + (v.abs() + cs)/dy;
    LAMBDA/vj
}