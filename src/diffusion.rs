use crate::state::{self, State};
use crate::constant;


pub struct DiffusionStencil {
    pub points: [State; 6],
    pub dir: state::Direction,
}

impl DiffusionStencil {
    pub fn build_diffusion(&self) -> State {
        build_diffusion_from_derived(
            &[
                state::Derived::from_state(self.points[0]),
                state::Derived::from_state(self.points[1]),
                state::Derived::from_state(self.points[2]),
                state::Derived::from_state(self.points[3]),
                state::Derived::from_state(self.points[4]),
                state::Derived::from_state(self.points[5]),
            ],
        )
    }
}

/// Hot-path variant: identical numerics, temperatures from precomputed
/// per-stage `Derived` quantities.
pub fn build_diffusion_from_derived(
    d: &[state::Derived; 6],
) -> State {
    let sjp3 = &d[5];
    let sjp2 = &d[4];
    let sjp1 = &d[3];
    let sj = &d[2];
    let sjm1 = &d[1];
    let sjm2 = &d[0];
    let ee_term = constant::KAPPA_E*(2.0*sjp3.te-25.0*sjp2.te + 245.0*sjp1.te -
                            245.0*sj.te + 25.0*sjm1.te - 2.0*sjm2.te);
    let ei_term = constant::KAPPA_I*((2.0*sjp3.ti-25.0*sjp2.ti + 245.0*sjp1.ti -
                            245.0*sj.ti + 25.0*sjm1.ti - 2.0*sjm2.ti));
    let er_term = constant::KAPPA_R*((2.0*sjp3.tr.powi(4)-25.0*sjp2.tr.powi(4) + 245.0*sjp1.tr.powi(4) -
                            245.0*sj.tr.powi(4) + 25.0*sjm1.tr.powi(4) - 2.0*sjm2.tr.powi(4)));

    State {
        rho: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
        ee: ee_term/180.0,
        ei: ei_term/180.0,
        er: er_term/180.0,
    }
}