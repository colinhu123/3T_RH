pub const DEFAULT_EPS: f64 = 1e-12;

pub const KAPPA_E: f64 = 0.0;
pub const KAPPA_I: f64 = 0.0;
pub const KAPPA_R: f64 = 0.0;

pub const CVE: f64 = 1.0;
pub const CVI: f64 = 1.0;
pub const A  : f64 = 1.0;

pub const GAMMA_I: f64 = 1.4;
pub const GAMMA_E: f64 = 1.4;
pub const GAMMA_R: f64 = 1.4;

pub const OMEGA_EI: f64 = 0.0;
pub const OMEGA_ER: f64 = 0.0;

pub const LAMBDA: f64 = 0.5;

pub const WENO_Q: f64 = 2.0;

pub const DIFFUSION_ACTIVE: bool =
    KAPPA_E != 0.0 || KAPPA_I != 0.0 || KAPPA_R != 0.0;

pub const SOURCE_ACTIVE: bool = OMEGA_EI != 0.0 || OMEGA_ER != 0.0;