pub const CVE: f64 = 1.0;
pub const CVI: f64 = 1.0;
pub const A  : f64 = 1.0;

#[derive(Clone,Copy,Debug)]
pub struct State {
    pub rho: f64,
    pub mom: f64,
    pub ee: f64,
    pub ei: f64,
    pub er: f64,
}

impl State {
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
}





