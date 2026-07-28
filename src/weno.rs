use crate::state;
use ndarray::{Array1,Array2,array};
use crate::linalg;
use crate::utils;

pub struct Stencil6 {
    points: [state::State; 6],

}


impl Stencil6 {

    pub fn build_a(&self)->Array2<f64> {
        let state1 = self.points[3];
        let state2 = self.points[2];
        let state_roe_ave = state2.roe_ave(state1);
        let rho = state_roe_ave.rho;
        let u = state_roe_ave.mom/rho;
        let ee = state_roe_ave.ee/rho - rho*u*u/6.0;
        let ei = state_roe_ave.ei/rho - rho*u*u/6.0;
        let er = state_roe_ave.er/rho - rho*u*u/6.0;

        let gamma_sum = state::GAMMA_E + state::GAMMA_I + state::GAMMA_R;

        let a: Array2<f64> = array![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [(gamma_sum-9.0)*u*u/6.0, -(gamma_sum - 9.0)*u/3.0, state::GAMMA_E-1.0, state::GAMMA_I - 1.0, state::GAMMA_R - 1.0],
            [-state::GAMMA_E*ee*u+(gamma_sum-6.0)/18.0*u.powf(3.0), state::GAMMA_E*ee-(2.0*gamma_sum-9.0)/18.0*u.powf(2.0), (state::GAMMA_E + 2.0)/3.0*u, (state::GAMMA_I-1.0)/3.0*u,(state::GAMMA_R - 1.0)/3.0*u],
            [-state::GAMMA_I*ei*u + (gamma_sum-6.0)/18.0*u.powf(3.0), state::GAMMA_I*ei - (2.0*gamma_sum-9.0)/18.0*u.powf(2.0),(state::GAMMA_E - 1.0)/3.0*u, (state::GAMMA_I+2.0)/3.0*u,(state::GAMMA_R - 1.0)/3.0*u],
            [-state::GAMMA_R*er*u + (gamma_sum-6.0)/18.0*u.powf(3.0), state::GAMMA_R*er - (2.0*gamma_sum-9.0)/18.0*u.powf(2.0),(state::GAMMA_E-1.0)/3.0*u,(state::GAMMA_I-1.0)/3.0*u,(state::GAMMA_R+2.0)/3.0*u],
        ];
        a
    }
    

    pub fn build_l(&self) -> Array2<f64> {
        let (_lambda,r) = self.build_r();
        let l = linalg::inverse(&r);
        l
    }

    pub fn build_r(&self)-> (Array1<f64>,Array2<f64>) {
        let (lambda, r) = linalg::eigen_qr(&self.build_a(), utils::MAX_ITER);
        (lambda,r)
    }

    pub fn con2char(&self) -> Self {
        let mut new_stencil: [state::State; 6] = [state::State::new(); 6];
        let l = self.build_l();
        for i in 0..6 {
        let tmp_k: Array1<f64> = Array1::from_vec(
            self.points[i].state2arr().to_vec()
            );
        

        new_stencil[i] = state::State::arr2state(l.dot(&tmp_k));
        }

        Self {
            points: new_stencil
        }
    }

    pub fn state2flux(&self) ->Self {
        let mut new_stencil = [state::State::new(); 6];
        for i in 0..6{
            new_stencil[i] = self.points[i].flux();
        }

        Self {
            points: new_stencil
        }
    }

    pub fn stencil2arr(&self) -> [[f64; 6]; 5] {
        let mut rho_list = [0.0; 6];
        let mut mom_list = [0.0; 6];
        let mut ee_list = [0.0; 6];
        let mut ei_list = [0.0; 6];
        let mut er_list = [0.0; 6];

        for i in 0..6 {
            rho_list[i] = self.points[i].rho;
            mom_list[i] = self.points[i].mom;
            ee_list[i] = self.points[i].ee;
            ei_list[i] = self.points[i].ei;
            er_list[i] = self.points[i].er;
        }

        [rho_list, mom_list,ee_list,ei_list,er_list]
    }

    pub fn reconstruction(&self) -> state::State {
        let flux_l = self.state2flux().con2char().points;
        let state_l = self.con2char().points;
        let (lambda,r) = self.build_r();
        let mut f_plus_stencil = [state::State::new(); 6];
        let mut f_minus_stencil = [state::State::new(); 6];
        //build F plus stencil
        for i in 0..6 {
            f_plus_stencil[i] = state::State {rho:0.5*(flux_l[i].rho +lambda[0]*state_l[i].rho),
                                                mom: 0.5*(flux_l[i].mom + lambda[1]*state_l[i].mom),
                                                ee: 0.5*(flux_l[i].ee + lambda[2]*state_l[i].ee),
                                                ei: 0.5*(flux_l[i].ei + lambda[3]*state_l[i].ei),
                                                er: 0.5*(flux_l[i].er + lambda[4]*state_l[i].er)};

            f_minus_stencil[i] = state::State {rho:0.5*(flux_l[i].rho - lambda[0]*state_l[i].rho),
                                                mom: 0.5*(flux_l[i].mom - lambda[1]*state_l[i].mom),
                                                ee: 0.5*(flux_l[i].ee - lambda[2]*state_l[i].ee),
                                                ei: 0.5*(flux_l[i].ei - lambda[3]*state_l[i].ei),
                                                er: 0.5*(flux_l[i].er - lambda[4]*state_l[i].er)};

        }

        
        let f_plus_stencil = Self {points: f_plus_stencil};
        let f_minus_stencil = Self {points: f_minus_stencil};
        let tmp = f_plus_stencil.stencil2arr();
        let tmp1 = f_minus_stencil.stencil2arr();
        let mut flux_plus = [0.0; 5];
        let mut flux_minus = [0.0; 5];
        for i in  0..5 {
            let stencil = [
                tmp[i][0],
                tmp[i][1],
                tmp[i][2],
                tmp[i][3],
                tmp[i][4],
            ];
            flux_plus[i] = weno(&stencil);
            let stencil = [
                tmp1[i][5],
                tmp1[i][4],
                tmp1[i][3],
                tmp1[i][2],
                tmp1[i][1]
            ];
            flux_minus[i] = weno(&stencil);
        }

        let flux = state::State {rho: flux_plus[0]+flux_minus[0], 
                                        mom: flux_plus[1]+flux_minus[1], 
                                        ee: flux_plus[2]+flux_minus[2], 
                                        ei: flux_plus[3]+flux_minus[3], 
                                        er: flux_plus[4]+flux_minus[4]};
        let tmp_k: Array1<f64> = Array1::from_vec(
            flux.state2arr().to_vec()
            );
        

        state::State::arr2state(r.dot(&tmp_k))

    }


}


#[inline]
pub fn weno(stencil: &[f64; 5]) -> f64 {

    let u_im2 = stencil[0];
    let u_im1 = stencil[1];
    let u_i   = stencil[2];
    let u_ip1 = stencil[3];
    let u_ip2 = stencil[4];


    // candidate reconstruction
    let u0 = 3.0/8.0*u_im2 
           - 5.0/4.0*u_im1 
           + 15.0/8.0*u_i;

    let u1 = -1.0/8.0*u_im1
           + 3.0/4.0*u_i
           + 3.0/8.0*u_ip1;

    let u2 = 3.0/8.0*u_i
           + 3.0/4.0*u_ip1
           - 1.0/8.0*u_ip2;


    // smoothness indicators
    let beta0 = (1.0/3.0) * (
          4.0*u_im2.powi(2)
        -19.0*u_im2*u_im1
        +25.0*u_im1.powi(2)
        +11.0*u_im2*u_i
        -31.0*u_im1*u_i
        +10.0*u_i.powi(2)
    );


    let beta1 = (1.0/3.0) * (
          4.0*u_im1.powi(2)
        -13.0*u_im1*u_i
        +13.0*u_i.powi(2)
        +5.0*u_im1*u_ip1
        -13.0*u_i*u_ip1
        +4.0*u_ip1.powi(2)
    );


    let beta2 = (1.0/3.0) * (
          10.0*u_i.powi(2)
        -31.0*u_i*u_ip1
        +25.0*u_ip1.powi(2)
        +11.0*u_i*u_ip2
        -19.0*u_ip1*u_ip2
        +4.0*u_ip2.powi(2)
    );


    // nonlinear weights
    let eps = 1e-6;

    let alpha0 = 0.0625 / (eps + beta0).powi(2);
    let alpha1 = 0.625 / (eps + beta1).powi(2);
    let alpha2 = 0.3125 / (eps + beta2).powi(2);


    let alpha_sum = alpha0 + alpha1 + alpha2;


    let w0 = alpha0 / alpha_sum;
    let w1 = alpha1 / alpha_sum;
    let w2 = alpha2 / alpha_sum;


    w0*u0 + w1*u1 + w2*u2
}



#[cfg(test)]
mod tests {

use super::*;


#[test]
    fn test_weno5_accuracy()
    {

        let grids = [
            40usize,
            80usize,
            160usize,
            320usize
        ];


        let mut errors: Vec<f64> = Vec::new();


        for &nx in grids.iter()
        {

            let dx =
                2.0*std::f64::consts::PI
                /(nx as f64);


            let mut error = 0.0;

            let mut count = 0;



            for j in 3..nx-3
            {

                let x =
                    j as f64 * dx;


                let stencil = [
                    (x-2.0*dx).sin(),
                    (x-dx).sin(),
                    x.sin(),
                    (x+dx).sin(),
                    (x+2.0*dx).sin(),
                ];


                let numerical =
                    weno(&stencil);


                let exact =
                    (x+0.5*dx).sin();



                error +=
                    (numerical-exact).abs();


                count +=1;

            }


            errors.push(
                error/(count as f64)
            );

        }



        println!("errors = {:?}", errors);



        // calculate convergence orders

        let mut orders = Vec::new();


        for i in 1..errors.len()
        {

            let order =
                (errors[i-1]/errors[i])
                .log2();


            orders.push(order);

        }


        println!("orders = {:?}", orders);



        // Ignore the first order because coarse grid
        // may not be in asymptotic region

        for order in orders.iter().skip(1)
        {

            assert!(
                *order > 4.5,
                "WENO order too low: {}",
                order
            );

        }

    }



#[test]
fn test_weno_constant()
{

let stencil=[
    5.0,
    5.0,
    5.0,
    5.0,
    5.0
];


let result=weno(&stencil);


assert!(
    (result-5.0).abs()<1e-12
);


}

#[test]
fn test_eigen_inverse()
{

    let state =
    state::State{
        rho:1.0,
        mom:0.5,
        ee:3.0,
        ei:2.0,
        er:1.0,
    };


    let stencil =
    Stencil6{
        points:[
            state,
            state,
            state,
            state,
            state,
            state,
        ]
    };

    let l = stencil.build_l();
    let (_,r) = stencil.build_r();
    let identity = l.dot(&r);

    for i in 0..5 {
        for j in 0..5 {
            if i==j {
                assert!((identity[[i,j]]-1.0).abs()<1e-10);
            }
            else {
                assert!(identity[[i,j]].abs()<1e-10);
            }

        }

    }

}
}

