use ndarray::{Array1};

pub const CVE: f64 = 1.0;
pub const CVI: f64 = 1.0;
pub const A  : f64 = 1.0;

pub const GAMMA_I: f64 = 5.0/3.0;
pub const GAMMA_E: f64 = 5.0/3.0;
pub const GAMMA_R: f64 = 4.0/3.0;
#[derive(Clone,Copy,Debug)]
pub struct State {
    pub rho: f64,
    pub mom: f64,
    pub ee: f64,
    pub ei: f64,
    pub er: f64,
}

impl State {
    pub fn new() -> Self {
        Self {
            rho: 0.0,
            mom: 0.0,
            ee: 0.0,
            ei: 0.0,
            er: 0.0,
        }
    }
    pub fn pressure_tot(&self) -> f64 {
        let pe = 2.0/3.0 * self.rho * self.ee;
        let pi = 2.0/3.0 * self.rho * self.ei;
        let pr = 1.0/3.0 * self.rho * self.er;
        pe+pi+pr
    }

    pub fn pressure_spilit(&self) -> (f64,f64,f64) {
        let pe = 2.0/3.0 * self.rho * self.ee;
        let pi = 2.0/3.0 * self.rho * self.ei;
        let pr = 1.0/3.0 * self.rho * self.er;
        (pe, pi, pr)
    }

    pub fn flux(&self) -> Self {
        let rho_u = self.mom;
        let u = self.mom/self.rho;
        let p = self.pressure_tot();
        let (pe,pi, pr) = self.pressure_spilit();

        Self {
            rho: rho_u,
            mom: self.rho*u*u + p,
            ee: (self.ee + pe) * u,
            ei: (self.ei + pi) * u,
            er: (self.er + pr) * u,
        }
    }

    pub fn te(&self) -> f64 {
        let u = self.mom/self.rho;
        self.ee/(self.rho*CVE) - u * u / (6.0 * CVE)
    }

    pub fn ti(&self) -> f64 {
        let u = self.mom / self.rho;
        self.ei/(self.rho * CVI) - u*u/(6.0 *CVI)
    }

    pub fn tr(&self) -> f64 {
        let u = self.mom/self.rho;
        let er = self.er/self.rho - u*u/(6.0*self.rho);
        return (er*self.rho/A).powf(0.25);
    }

    pub fn roe_ave(&self, state: State) -> Self {
        let rho1 = self.rho;
        let rho2 = state.rho;
        let denom = rho1 + rho2;
        Self {
            rho: (self.rho*rho1 +state.rho*rho2)/denom,
            mom: (self.mom*rho1 + state.mom*rho2)/denom,
            ee: (self.ee*rho1 + state.ee*rho2)/denom,
            ei: (self.ei* rho1 + state.ei*rho2)/denom,
            er: (self.er * rho1 + state.er * rho2) / denom,
        }
    }

    pub fn state2arr(&self) -> [f64; 5] {
        [self.rho, self.mom, self.ee, self.ei, self.er]
    }

    pub fn arr2state(arr: Array1<f64>) -> Self {
        Self {
            rho:arr[0],
            mom: arr[1],
            ee: arr[2],
            ei: arr[3],
            er: arr[4],
        }
    }

    pub fn add(&self, s2: State) -> Self {
        Self {
            rho: self.rho + s2.rho,
            mom: self.mom + s2.mom,
            ee:self.ee + s2.ee,
            ei:self.ei + s2.ei,
            er:self.er + s2.er,
        }
    }

    pub fn scalar_prod(&self, scalar: f64) -> Self {
        Self {
            rho: self.rho*scalar,
            mom: self.mom*scalar,
            ee: self.ee*scalar,
            ei: self.ei * scalar,
            er: self.er * scalar,
        }
    }
}



///flux1: value at i - 1/2
/// flux2: value at i + 1/2
pub fn update(flux1: State, flux2: State) -> State {
    State {
        rho: - (flux2.rho - flux1.rho),
        mom: - (flux2.mom - flux1.mom),
        ee: - (flux2.ee - flux1.ee),
        ei: - (flux2.ee - flux1.ee),
        er: - (flux2.er - flux1.er),
    }
}


