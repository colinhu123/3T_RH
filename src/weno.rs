use crate::{constant, state};
use ndarray::{Array1,Array2,array};
use ndarray_linalg::Inverse;

#[derive(Clone,Copy,Debug)]
pub struct Stencil6 {
    pub points: [state::State; 6],

}


impl Stencil6 {

    pub fn _build_a(&self)->Array2<f64> {
        let state1 = self.points[3];
        let state2 = self.points[2];
        let u1 = state1.mom/state1.rho;
        let u2 = state2.mom/state2.rho;
        let ee1 = state1.ee/state1.rho - u1*u1/6.0;
        let ei1 = state1.ei/state1.rho - u1*u1/6.0;
        let er1 = state1.er/state1.rho - u1*u1/6.0;
        let ee2 = state2.ee/state2.rho - u2*u2/6.0;
        let ei2 = state2.ei/state2.rho - u2*u2/6.0;
        let er2 = state2.er/state2.rho - u2*u2/6.0;
        let rho1 = state1.rho;
        let rho2 = state2.rho;

        let u = (u1*rho1 + u2*rho2)/(rho1+rho2);
        let ee = (ee1*rho1 + ee2*rho2)/(rho1 + rho2);
        let ei = (ei1*rho1 + ei2*rho2)/(rho1+rho2);
        let er = (er1*rho1 + er2*rho2)/ (rho1+rho2);

        let gamma_sum = constant::GAMMA_E + constant::GAMMA_I + constant::GAMMA_R;

        let a: Array2<f64> = array![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [(gamma_sum-9.0)*u*u/6.0, -(gamma_sum - 9.0)*u/3.0, constant::GAMMA_E-1.0, constant::GAMMA_I - 1.0, constant::GAMMA_R - 1.0],
            [-constant::GAMMA_E*ee*u+(gamma_sum-6.0)/18.0*u.powf(3.0), constant::GAMMA_E*ee-(2.0*gamma_sum-9.0)/18.0*u.powf(2.0), (constant::GAMMA_E + 2.0)/3.0*u, (constant::GAMMA_I-1.0)/3.0*u,(constant::GAMMA_R - 1.0)/3.0*u],
            [-constant::GAMMA_I*ei*u + (gamma_sum-6.0)/18.0*u.powf(3.0), constant::GAMMA_I*ei - (2.0*gamma_sum-9.0)/18.0*u.powf(2.0),(constant::GAMMA_E - 1.0)/3.0*u, (constant::GAMMA_I+2.0)/3.0*u,(constant::GAMMA_R - 1.0)/3.0*u],
            [-constant::GAMMA_R*er*u + (gamma_sum-6.0)/18.0*u.powf(3.0), constant::GAMMA_R*er - (2.0*gamma_sum-9.0)/18.0*u.powf(2.0),(constant::GAMMA_E-1.0)/3.0*u,(constant::GAMMA_I-1.0)/3.0*u,(constant::GAMMA_R+2.0)/3.0*u],
        ];
        a
    }
    

    pub fn build_l(&self) -> Array2<f64> {
        let (_lambda,r) = self.build_r_roe_ave();
        let l = r.inv().unwrap();
        l
    }

    pub fn build_r_roe_ave(&self)-> (Array1<f64>,Array2<f64>) {
        let state1 = self.points[3];
        let state2 = self.points[2];
        let u1 = state1.mom/state1.rho;
        let u2 = state2.mom/state2.rho;
        let ee1 = state1.ee/state1.rho - u1*u1/6.0;
        let ei1 = state1.ei/state1.rho - u1*u1/6.0;
        let er1 = state1.er/state1.rho - u1*u1/6.0;
        let ee2 = state2.ee/state2.rho - u2*u2/6.0;
        let ei2 = state2.ei/state2.rho - u2*u2/6.0;
        let er2 = state2.er/state2.rho - u2*u2/6.0;
        let rho1 = state1.rho.sqrt();
        let rho2 = state2.rho.sqrt();

        let u = (u1*rho1 + u2*rho2)/(rho1+rho2);
        let ee = (ee1*rho1 + ee2*rho2)/(rho1 + rho2);
        let ei = (ei1*rho1 + ei2*rho2)/(rho1+rho2);
        let er = (er1*rho1 + er2*rho2)/ (rho1+rho2);

        let gi = constant::GAMMA_I - 1.0;
        let ge = constant::GAMMA_E - 1.0;
        let gr = constant::GAMMA_R - 1.0;

        let cs = (constant::GAMMA_E*ge*ee + constant::GAMMA_I*gi*ei + constant::GAMMA_R*gr*er).sqrt();

        let gt = gi + ge + gr;
        let r = array![
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [u-cs, u, u, u, u+cs ],
        [constant::GAMMA_E*ee+u.powi(2)/6.0-u*cs/3.0,gt*u.powi(2)/(6.0*ge),-gr,gi,constant::GAMMA_E*ee+u.powi(2)/6.0+u*cs/3.0],
        [constant::GAMMA_I*ei+u.powi(2)/6.0-u*cs/3.0,gr,gt*u.powi(2)/(6.0*gi),-ge,constant::GAMMA_I*ei+u.powi(2)/6.0+u*cs/3.0],
        [constant::GAMMA_R*er+u.powi(2)/6.0-u*cs/3.0,-gi,ge,gt*u.powi(2)/(6.0*gr),constant::GAMMA_R*er+u.powi(2)/6.0+u*cs/3.0],
        ];
        let lambda = array![u-cs,u,u,u,u+cs];

       (lambda,r)
    }

    pub fn con2char(&self,l:&Array2<f64>) -> Self {
        let mut new_stencil: [state::State; 6] = [state::State::new(); 6];
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

    pub fn reconstruction(&self,recon_type: bool) -> state::State {
        let l = self.build_l();
        let (flux_l, state_l): ([state::State; 6], [state::State; 6]) = if recon_type {
        (self.state2flux().con2char(&l).points, self.con2char(&l).points)
    } else {
        (self.state2flux().points, self.points)
    };

        //let flux_stencil = self.state2flux();
        let (lambda, r) = self.build_r_roe_ave();

        let mut f_plus_stencil = [state::State::new(); 6];
        let mut f_minus_stencil = [state::State::new(); 6];


        let a0 = lambda[0].abs();
        let a1 = lambda[1].abs();
        let a2 = lambda[2].abs();
        let a3 = lambda[3].abs();
        let a4 = lambda[4].abs();

        for i in 0..6 {
            f_plus_stencil[i] = state::State {
                rho: 0.5*(flux_l[i].rho + a0*state_l[i].rho),
                mom: 0.5*(flux_l[i].mom + a1*state_l[i].mom),
                ee:  0.5*(flux_l[i].ee  + a2*state_l[i].ee),
                ei:  0.5*(flux_l[i].ei  + a3*state_l[i].ei),
                er:  0.5*(flux_l[i].er  + a4*state_l[i].er),
            };
            f_minus_stencil[i] = state::State {
                rho: 0.5*(flux_l[i].rho - a0*state_l[i].rho),
                mom: 0.5*(flux_l[i].mom - a1*state_l[i].mom),
                ee:  0.5*(flux_l[i].ee  - a2*state_l[i].ee),
                ei:  0.5*(flux_l[i].ei  - a3*state_l[i].ei),
                er:  0.5*(flux_l[i].er  - a4*state_l[i].er),
            };
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
            flux_plus[i] = weno5(&stencil);
            let stencil = [
                tmp1[i][5],
                tmp1[i][4],
                tmp1[i][3],
                tmp1[i][2],
                tmp1[i][1]
            ];
            flux_minus[i] = weno5(&stencil);
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
pub fn weno5(stencil: &[f64; 5]) -> f64 {

    let u0 = stencil[0];
    let u1 = stencil[1];
    let u2 = stencil[2];
    let u3 = stencil[3];
    let u4 = stencil[4];

    let beta2 = 13.0/12.0 * (u2 - 2.0*u3 + u4).powi(2)
                                + 0.25*(3.0*u2 - 4.0*u3 + u4).powi(2);
    let beta1 = 13.0/12.0*(u1 - 2.0*u2 + u3).powi(2)
                    + 0.25*(u1 - u3).powi(2);
    let beta0 = 13.0/12.0*(u0 - 2.0*u1 + u2).powi(2)
                    + 0.25*(u0 - 4.0*u1 + 3.0*u2).powi(2);
    let d0 = 0.1;
    let d1 = 0.6;
    let d2 = 0.3;

    let a0 = d0/(constant::DEFAULT_EPS + beta0).powi(2);
    let a1 = d1/(constant::DEFAULT_EPS + beta1).powi(2);
    let a2 = d2/(constant::DEFAULT_EPS + beta2).powi(2);

    //let tau5 = (beta0-beta2).abs();

    //let a0 = d0*(1.0 + (tau5/(beta0+utils::DEFAULT_EPS)).powi(2));
    //let a1 = d1*(1.0 + (tau5/(beta1+utils::DEFAULT_EPS)).powi(2));
    //let a2 = d2*(1.0 + (tau5/(beta2+utils::DEFAULT_EPS)).powi(2));

    let sum_o = a0 + a1 + a2;

    let w0 = a0/sum_o;
    let w1 = a1/sum_o;
    let w2 = a2/sum_o;

    let p0 = u0/3.0 - 7.0/6.0*u1 + 11.0/6.0*u2;
    let p1 = - u1/6.0 + 5.0/6.0*u2 + u3/3.0;
    let p2 = u2/3.0 + 5.0/6.0*u3 - u4/6.0;

    w0*p0 + w1*p1 + w2*p2
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

            // h_{i+1/2}: stencil centered at x
            let stencil_r = [
                (x-2.0*dx).sin(),
                (x-dx).sin(),
                x.sin(),
                (x+dx).sin(),
                (x+2.0*dx).sin(),
            ];

            // h_{i-1/2}: same stencil shape, shifted back by dx
            let xm = x - dx;
            let stencil_l = [
                (xm-2.0*dx).sin(),
                (xm-dx).sin(),
                xm.sin(),
                (xm+dx).sin(),
                (xm+2.0*dx).sin(),
            ];

            let h_iphalf =
                weno5(&stencil_r);

            let h_imhalf =
                weno5(&stencil_l);

            let numerical =
                (h_iphalf - h_imhalf) / dx;

            let exact =
                x.cos();



            error +=
                (numerical-exact).abs();

            count +=1;

        }


        errors.push(
            error/(count as f64)
        );

    }



    // calculate convergence orders

    let mut orders = Vec::new();


    for i in 1..errors.len()
    {

        let order =
            (errors[i-1]/errors[i])
            .log2();


        orders.push(order);

    }


    println!("errors = {:?}", errors);
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


let result=weno5(&stencil);


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
    let (_,r) = stencil.build_r_roe_ave();
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

#[test]
fn test_constant_flux_preserving(){

    let state =
    state::State{
        rho:1.0,
        mom:0.5,
        ee:3.0,
        ei:2.0,
        er:1.0,
    };


    let stencil=Stencil6{
        points:[state;6]
    };


    let ans=stencil.reconstruction(true);
    let flux=state.flux();


    assert!((ans.rho-flux.rho).abs()<1e-10);
    assert!((ans.mom-flux.mom).abs()<1e-10);
    assert!((ans.ee-flux.ee).abs()<1e-10);
    assert!((ans.ei-flux.ei).abs()<1e-10);
    assert!((ans.er-flux.er).abs()<1e-10);
}

#[test]
fn test_characteristic_projection()
{
    let state =
    state::State{
        rho:1.0,
        mom:0.5,
        ee:3.0,
        ei:2.0,
        er:1.0,
    };


    let stencil=Stencil6{
        points:[state;6]
    };


    let l=stencil.build_l();
    let (_,r)=stencil.build_r_roe_ave();


    let id=l.dot(&r);


    println!("{:?}",id);
}
}

