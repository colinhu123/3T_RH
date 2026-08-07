use crate::state::State;
use crate::constant;


pub struct DiffusionStencil {
    pub points: [State; 6],
}

impl DiffusionStencil {
    pub fn build_diffusion(&self) -> State {
        let sjp3 = self.points[5];
        let sjp2 = self.points[4];
        let sjp1 = self.points[3];
        let sj = self.points[2];
        let sjm1 = self.points[1];
        let sjm2 = self.points[0];
        let ee_term = constant::KAPPA_E*(2.0*sjp3.te()-25.0*sjp2.te() + 245.0*sjp1.te() - 
                                245.0*sj.te() + 25.0*sjm1.te() - 2.0*sjm2.te());
        let ei_term = constant::KAPPA_I*((2.0*sjp3.ti()-25.0*sjp2.ti() + 245.0*sjp1.ti() - 
                                245.0*sj.ti() + 25.0*sjm1.ti() - 2.0*sjm2.ti()));
        let er_term = constant::KAPPA_R*((2.0*sjp3.tr().powi(4)-25.0*sjp2.tr().powi(4) + 245.0*sjp1.tr().powi(4) - 
                                245.0*sj.tr().powi(4) + 25.0*sjm1.tr().powi(4) - 2.0*sjm2.tr().powi(4)));

        State {
            rho: 0.0,
            mom: 0.0,
            ee: ee_term/180.0,
            ei: ei_term/180.0,
            er: er_term/180.0,
        }
    }
}