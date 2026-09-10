pub const DEFAULT_EPS: f64 = 1e-12;

pub const KAPPA_E: f64 = 0.0;
pub const KAPPA_I: f64 = 0.0;
pub const KAPPA_R: f64 = 0.0;

pub const CVE: f64 = 1.0;
pub const CVI: f64 = 1.0;
pub const A: f64 = 1.0;

pub const GAMMA_I: f64 = 5.0/3.0;
pub const GAMMA_E: f64 = 5.0/3.0;
pub const GAMMA_R: f64 = 4.0/3.0;

pub const OMEGA_EI: f64 = 0.0;
pub const OMEGA_ER: f64 = 0.0;

pub const LAMBDA: f64 = 0.5;

// WENO nonlinear-weight exponent for the BOUNDARY extrapolation only (the
// interior flux reconstruction uses its own hardcoded q=2 in `weno.rs`).
// q=2 is the standard WENO-JS value and is required for the boundary to
// retain the design order on smooth fields: at a wall the characteristic
// modes have critical points, and a large q makes the smoothness
// indicators (which are not comparable across the different-degree
// substencils) hijack the weights to a low-degree substencil.  Use a
// larger value only if a strong-shock boundary case needs the extra
// stabilization.
pub const WENO_Q: f64 = 2.0;

pub const DIFFUSION_ACTIVE: bool = KAPPA_E != 0.0 || KAPPA_I != 0.0 || KAPPA_R != 0.0;

pub const SOURCE_ACTIVE: bool = OMEGA_EI != 0.0 || OMEGA_ER != 0.0;

pub const PHI: f64 = 5.0; //this is used in modified LF to reduce carbuncle effect

pub const EPS_LIMITER: f64 = 1e-13; //constant used in positivity limiter
pub const PP_W: f64 = 1.0/12.0;
pub const MAX_ITER: usize = 60;

pub const PP_SWITCH: bool = false;