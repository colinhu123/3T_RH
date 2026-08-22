use ndarray::{Array1};
use crate::constant;
#[derive(Clone,Copy,Debug)]
pub enum Direction {
    X,
    Y,
}

impl Direction {
    pub fn velocity(&self, s: &State) -> f64 {
        match self {
            Direction::X => s.mom_x / s.rho,
            Direction::Y => s.mom_y / s.rho,
        }
    }
}
#[derive(Clone,Copy,Debug)]
pub struct State {
    pub rho: f64,
    pub mom_x: f64,
    pub mom_y: f64,
    pub ee: f64,
    pub ei: f64,
    pub er: f64,
}

impl State {
    pub fn new() -> Self {
        Self {
            rho: 0.0,
            mom_x: 0.0,
            mom_y: 0.0,
            ee: 0.0,
            ei: 0.0,
            er: 0.0,
        }
    }
    pub fn pressure_tot(&self) -> f64 {
        let u = self.mom_x/self.rho;
        let v = self.mom_y/self.rho;
        let pe = (constant::GAMMA_E - 1.0) * (self.ee - self.rho * (u.powi(2) + v.powi(2))/6.0);
        let pi = (constant::GAMMA_I - 1.0) * (self.ei - self.rho * (u.powi(2) + v.powi(2))/6.0);
        let pr = (constant::GAMMA_R - 1.0) * (self.er - self.rho * (u.powi(2) + v.powi(2))/6.0);
        pe+pi+pr
    }

    pub fn pressure_spilit(&self) -> (f64,f64,f64) {
        let u = self.mom_x/self.rho;
        let v = self.mom_y/self.rho;
        let pe = (constant::GAMMA_E - 1.0) * (self.ee - self.rho * (u.powi(2) + v.powi(2))/6.0);
        let pi = (constant::GAMMA_I - 1.0) * (self.ei - self.rho * (u.powi(2) + v.powi(2))/6.0);
        let pr = (constant::GAMMA_R - 1.0) * (self.er - self.rho * (u.powi(2) + v.powi(2))/6.0);
        (pe, pi, pr)
    }

    pub fn flux_from_derived(&self, d: &Derived, dir: Direction) -> Self {
        let u = d.u;
        let v = d.v;
        let p = d.pe + d.pi + d.pr;
        let (pe, pi, pr) = (d.pe, d.pi, d.pr);
        let rho = self.rho;

        match dir {
            Direction::X => {
                return Self {
                rho: rho*u,
                mom_x: rho*u*u + p,
                mom_y: rho*u*v,
                ee: (self.ee + pe) * u,
                ei: (self.ei + pi) * u,
                er: (self.er + pr) * u,
            }
            }
            Direction::Y => {
                return Self {
                    rho: rho * v,
                    mom_x: rho* u * v,
                    mom_y: rho * v * v + p,
                    ee: (self.ee + pe) * v,
                    ei: (self.ei + pi) * v,
                    er: (self.er + pr) * v,
                }
            }
        }
    }

    pub fn flux(&self, dir: Direction) -> Self {
        let u = self.mom_x/self.rho;
        let v = self.mom_y/self.rho;
        let p = self.pressure_tot();
        let (pe,pi, pr) = self.pressure_spilit();
        let rho = self.rho;

        match dir {
            Direction::X => {
                return Self {
                rho: rho*u,
                mom_x: rho*u*u + p,
                mom_y: rho*u*v,
                ee: (self.ee + pe) * u,
                ei: (self.ei + pi) * u,
                er: (self.er + pr) * u,
            }
            }
            Direction::Y => {
                return Self {
                    rho: rho * v,
                    mom_x: rho* u * v,
                    mom_y: rho * v * v + p,
                    ee: (self.ee + pe) * v,
                    ei: (self.ei + pi) * v,
                    er: (self.er + pr) * v,
                }
            }
        }
    }

    pub fn te(&self) -> f64 {//this three method has physics problem
        let u = self.mom_x/self.rho;
        let v= self.mom_y/self.rho;
        self.ee/(self.rho*constant::CVE) - (u.powi(2) + v.powi(2)) / (6.0 * constant::CVE)
    }

    pub fn ti(&self) -> f64 {
        let u = self.mom_x / self.rho;
        let v = self.mom_y/self.rho;
        self.ei/(self.rho * constant::CVI) - (u.powi(2) + v.powi(2))/(6.0 *constant::CVI)
    }

    pub fn tr(&self) -> f64 {
        let u = self.mom_x/self.rho;
        let v = self.mom_y/self.rho;
        let er = self.er/self.rho - (u.powi(2) + v.powi(2))/(6.0);
        return (er*self.rho/constant::A).powf(0.25);
    }

    pub fn _roe_ave(&self, state: State) -> Self {
        let rho1 = self.rho;
        let rho2 = state.rho;
        let denom = rho1 + rho2;
        Self {
            rho: (self.rho*rho1 +state.rho*rho2)/denom,
            mom_x: (self.mom_x*rho1 + state.mom_x*rho2)/denom,
            mom_y: (self.mom_y*rho1 + state.mom_y*rho2)/denom,
            ee: (self.ee*rho1 + state.ee*rho2)/denom,
            ei: (self.ei* rho1 + state.ei*rho2)/denom,
            er: (self.er * rho1 + state.er * rho2) / denom,
        }
    }

    pub fn primi2con(rho: f64, u: f64,v: f64, pe: f64, pi: f64, pr: f64) -> Self {
        let ee1 = pe/((constant::GAMMA_E-1.0)*rho);
        let ei1 = pi/((constant::GAMMA_I-1.0)*rho);
        let er1 = pr/((constant::GAMMA_R-1.0)*rho);

        let kinetic_share = rho*(u*u + v*v)/66.0;
        
        Self {
            rho: rho,
            mom_x: rho*u,
            mom_y: rho*v,
            ee: rho*ee1 + kinetic_share,
            ei: rho*ei1 + kinetic_share,
            er: rho*er1 + kinetic_share,
        }
    }

    pub fn state2arr(&self) -> [f64; 6] {
        [self.rho, self.mom_x, self.mom_y, self.ee, self.ei, self.er]
    }

    pub fn arr2state(arr: Array1<f64>) -> Self {
        Self {
            rho:arr[0],
            mom_x: arr[1],
            mom_y: arr[2],
            ee: arr[3],
            ei: arr[4],
            er: arr[5],
        }
    }

    pub fn add(&self, s2: State) -> Self {
        Self {
            rho: self.rho + s2.rho,
            mom_x: self.mom_x + s2.mom_x,
            mom_y: self.mom_y + s2.mom_y,
            ee:self.ee + s2.ee,
            ei:self.ei + s2.ei,
            er:self.er + s2.er,
        }
    }

    pub fn scalar_prod(&self, scalar: f64) -> Self {
        Self {
            rho: self.rho*scalar,
            mom_x: self.mom_x*scalar,
            mom_y: self.mom_y*scalar,
            ee: self.ee*scalar,
            ei: self.ei * scalar,
            er: self.er * scalar,
        }
    }
}



/// Per-stage derived quantities cached once per cell / ghost cell.
///
/// Bit-identical to the pointwise expressions used by `State::flux`,
/// `State::pressure_spilit`, `State::te/ti/tr` and the Roe averages in
/// `weno.rs` (u and v use the same division form as `State`).
#[derive(Clone, Copy, Debug)]
pub struct Derived {
    pub u: f64,
    pub v: f64,
    pub e_e: f64,
    pub e_i: f64,
    pub e_r: f64,
    pub pe: f64,
    pub pi: f64,
    pub pr: f64,
    pub te: f64,
    pub ti: f64,
    pub tr: f64,
}

impl Derived {
    pub fn new() -> Self {
        Self {
            u: 0.0,
            v: 0.0,
            e_e: 0.0,
            e_i: 0.0,
            e_r: 0.0,
            pe: 0.0,
            pi: 0.0,
            pr: 0.0,
            te: 0.0,
            ti: 0.0,
            tr: 0.0,
        }
    }

    pub fn from_state(s: State) -> Self {
        let u = s.mom_x / s.rho;
        let v = s.mom_y / s.rho;
        let w2 = u.powi(2) + v.powi(2);

        let e_e = s.ee / s.rho - w2 / 6.0;
        let e_i = s.ei / s.rho - w2 / 6.0;
        let e_r = s.er / s.rho - w2 / 6.0;

        let pe = (constant::GAMMA_E - 1.0) * (s.ee - s.rho * w2 / 6.0);
        let pi = (constant::GAMMA_I - 1.0) * (s.ei - s.rho * w2 / 6.0);
        let pr = (constant::GAMMA_R - 1.0) * (s.er - s.rho * w2 / 6.0);

        let te = s.ee / (s.rho * constant::CVE) - w2 / (6.0 * constant::CVE);
        let ti = s.ei / (s.rho * constant::CVI) - w2 / (6.0 * constant::CVI);
        let er_term = s.er / s.rho - w2 / 6.0;
        let tr = (er_term * s.rho / constant::A).powf(0.25);

        Self {
            u,
            v,
            e_e,
            e_i,
            e_r,
            pe,
            pi,
            pr,
            te,
            ti,
            tr,
        }
    }
}

///flux1: value at i - 1/2
/// flux2: value at i + 1/2
pub fn update(flux1: State, flux2: State) -> State {
    State {
        rho: - (flux2.rho - flux1.rho),
        mom_x: - (flux2.mom_x - flux1.mom_x),
        mom_y: - (flux2.mom_y - flux1.mom_y),
        ee: - (flux2.ee - flux1.ee),
        ei: - (flux2.ei - flux1.ei),
        er: - (flux2.er - flux1.er),
    }
}




#[cfg(test)]
mod test {
    use crate::state::State;

    fn close(a: f64, b: f64) -> bool{
        if (a-b).abs() < 1e-4 {
            true
        }
        else {
            false
        }
    }

    #[test]
    fn test_state_primi2con() {
        let s1 = State::primi2con(0.445, 0.4935605, 0.4935605, 1.176, 1.176, 1.176);

        assert!(close(s1.rho,0.445));
        assert!(close(s1.mom_x, 0.2196344373));
        assert!(close(s1.ee, 1.800134));
        assert!(close(s1.er, 3.56413429));
    }
}
