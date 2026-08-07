use crate::state::State;
use crate::constant;


pub struct DiffusionStencil {
    pub points: [State; 5],
}

impl DiffusionStencil {
    pub fn build_diffusion(&self) -> State {
        let sjp3 = self.points[4];
        let sjp2 = self.points[3];
        let sjp1 = self.points[2];
        let sj = self.points[1];
        let sjm1 = self.points[0];
        let ee_term = constant::KAPPA_E*(2.0*sjp3.te()-25.0*sjp2.te() + 245.0*sjp1.te() - 
                                245.0*sj.te() + 25.0*sjm1.te() + 2.0*sj.te());
        let ei_term = constant::KAPPA_I*((2.0*sjp3.ti()-25.0*sjp2.ti() + 245.0*sjp1.ti() - 
                                245.0*sj.ti() + 25.0*sjm1.ti() + 2.0*sj.ti()));
        let er_term = constant::KAPPA_R*((2.0*sjp3.tr()-25.0*sjp2.tr() + 245.0*sjp1.tr() - 
                                245.0*sj.tr() + 25.0*sjm1.tr() + 2.0*sj.tr()));

        State {
            rho: 0.0,
            mom: 0.0,
            ee: ee_term/180.0,
            ei: ei_term/180.0,
            er: er_term/180.0,
        }
    }
}