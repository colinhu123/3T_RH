use crate::state;
use crate::weno;

pub fn n_vector(u:&state::State)->state::State
{

    let (pe, pi, pr) = u.pressure_spilit();


    state::State{
        rho: 0.0,
        mom: 0.0,
        ee: 2.0*pe-pi-pr,
        ei: 2.0*pi-pe-pr,
        er: 2.0*pr-pe-pi,

    }

}

fn sixth_order_derivative(
    qm3:f64,
    qm2:f64,
    qm1:f64,
    qp1:f64,
    qp2:f64,
    qp3:f64,
    dx:f64
)->f64
{

    (
        qp3
        -9.0*qp2
        +45.0*qp1
        -45.0*qm1
        +9.0*qm2
        -qm3
    )
    /(60.0*dx)

}

pub fn dNdx(stencil:&[state::State;7],dx:f64 )->state::State{
    
    let mut pe = [0.0; 7];
    let mut pi = [0.0; 7];
    let mut pr = [0.0; 7];
    for i in 0..7 {
        let (p1,p2,p3) = stencil[i].pressure_spilit();
        pe[i] = p1;
        pi[i] = p2;
        pr[i] = p3;
    }

    let dpe =sixth_order_derivative(pe[0],pe[1],pe[2],
                                    pe[4],pe[5],pe[6],
                                    dx);


    let dpi =sixth_order_derivative(pi[0],pi[1],pi[2],
                                    pi[4],pi[5],pi[6],
                                    dx);


    let dpr =sixth_order_derivative(pr[0],pr[1],pr[2],
                                    pr[4],pr[5],pr[6],
                                    dx);

    state::State{rho:0.0,
                mom:0.0,
                ee:2.0*dpe-dpi-dpr,
                ei:2.0*dpi-dpe-dpr,
                er:2.0*dpr-dpe-dpi,
            }

}


/// Compute the left-biased WENO reconstruction of the conservative state at the
/// interface between stencil[3] and stencil[4], then compute the N-vector jump.
///
/// The 6-point conservative stencil [1,2,3,4,5,6] is used with Stencil6,
/// whose Roe-averaged interface lies between points[2]=stencil[3] and
/// points[3]=stencil[4].
///
/// Returns n_vector(reconstructed_state) - n_vector(stencil[3]).
pub fn upwind_jump_left(stencil: &[state::State; 7]) -> state::State {
    // Build a 6-point conservative stencil for the left-biased WENO reconstruction.
    // Stencil6 uses points[2] and points[3] for the Roe average, corresponding
    // to the interface between original indices 3 and 4.
    let stencil6 = weno::Stencil6 {
        points: [
            stencil[1],
            stencil[2],
            stencil[3],
            stencil[4],
            stencil[5],
            stencil[6],
        ],
    };

    // Reconstruct the conservative state at the interface from the left side
    let state_recon = stencil6.reconstruction();

    // N-vector from reconstructed state minus cell-centered N-vector
    let n_recon = n_vector(&state_recon);
    //let n_center = n_vector(&stencil[3]);
    let statej = stencil[3];
    let statejm = stencil[2];
    let uj = statej.mom/statej.rho;
    let ujm = statejm.mom/statejm.rho;
    let coef = uj.max(ujm).max(0.0)/3.0;

    state::State {
        rho: coef*n_recon.rho,
        mom: coef*n_recon.mom,
        ee: coef*n_recon.ee,
        ei: coef*n_recon.ei ,
        er: coef*n_recon.er,
    }
}

/// Compute the right-biased WENO reconstruction of the conservative state at the
/// interface between stencil[3] and stencil[4], then compute the N-vector jump.
///
/// The 6-point conservative stencil [5,4,3,2,1,0] (reversed) is used so that
/// the same interface is approached from the right side.
///
/// Returns n_vector(reconstructed_state) - n_vector(stencil[3]).
pub fn upwind_jump_right(stencil: &[state::State; 7]) -> state::State {
    // Build a 6-point conservative stencil (reversed) for right-biased
    // WENO reconstruction at the same interface.
    let stencil6 = weno::Stencil6 {
        points: [
            stencil[5],
            stencil[4],
            stencil[3],
            stencil[2],
            stencil[1],
            stencil[0],
        ],
    };

    // Reconstruct the conservative state at the interface from the right side
    let state_recon = stencil6.reconstruction();

    // N-vector from reconstructed state minus cell-centered N-vector
    let n_recon = n_vector(&state_recon);
    //let n_center = n_vector(&stencil[3]);
    let statej = stencil[3];
    let statejp = stencil[4];
    let uj = statej.mom/statej.rho;
    let ujp = statejp.mom/statejp.rho;
    let coef = uj.max(ujp).max(0.0)/3.0;

    state::State {
        rho: coef*n_recon.rho,
        mom: coef*n_recon.mom,
        ee: coef*n_recon.ee,
        ei: coef*n_recon.ei ,
        er: coef*n_recon.er,
    }
}

pub fn nonconservative(
    stencil:&[state::State;7],
    dx:f64
    )->state::State
{
    let center=&stencil[3];
    let u = center.mom/center.rho;
    let derivative = dNdx(stencil,dx);
    let n_left = upwind_jump_left(&stencil);
    let n_right = upwind_jump_right(&stencil);

    state::State{
        rho:0.0,
        mom:0.0,
        ee: u/3.0*derivative.ee +n_left.ee +n_right.ee,
        ei: u/3.0*derivative.ei +n_left.ei +n_right.ei,
        er: u/3.0*derivative.er +n_left.er +n_right.er,
}

}









#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::State;

    fn make_state(rho: f64, mom: f64, ee: f64, ei: f64, er: f64) -> State {
        State {
            rho,
            mom,
            ee,
            ei,
            er,
        }
    }

    fn constant_stencil(s: State) -> [State; 7] {
        [s; 7]
    }


    #[test]
    fn test_n_vector_zero_pressure() {
        let s = make_state(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );

        let n = n_vector(&s);

        assert_eq!(n.rho, 0.0);
        assert_eq!(n.mom, 0.0);
        assert_eq!(n.ee, 0.0);
        assert_eq!(n.ei, 0.0);
        assert_eq!(n.er, 0.0);
    }


    #[test]
    fn test_n_vector_pressure_split_inverse() {
        /*
         The transformation is:

         Ne = 2Pe - Pi - Pr
         Ni = 2Pi - Pe - Pr
         Nr = 2Pr - Pe - Pi

         For equal pressures the result should equal the pressure.
        */

        let s = make_state(
            1.0,
            0.0,
            5.0,
            5.0,
            5.0,
        );

        let n = n_vector(&s);

        assert!((n.ee - 5.0/3.0).abs() < 1e-12);
        assert!((n.ei - 5.0/3.0).abs() < 1e-12);
        assert!((n.er + 10.0/3.0).abs() < 1e-12);
    }


    #[test]
    fn test_sixth_order_derivative_constant_function() {
        let result = sixth_order_derivative(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.1,
        );

        assert!(result.abs() < 1e-14);
    }


    #[test]
    fn test_sixth_order_derivative_linear_function() {
        /*
        For f(x)=x:

        points:
        qm3=-3
        qm2=-2
        qm1=-1
        qp1=1
        qp2=2
        qp3=3

        derivative should be 1.
        */

        let result = sixth_order_derivative(
            -3.0,
            -2.0,
            -1.0,
            1.0,
            2.0,
            3.0,
            1.0,
        );

        assert!((result - 1.0).abs() < 1e-12);
    }


    #[test]
    fn test_dNdx_constant_state() {
        let s = make_state(
            1.0,
            0.0,
            3.0,
            3.0,
            3.0,
        );

        let stencil = constant_stencil(s);

        let result = dNdx(&stencil, 1.0);

        assert!(result.ee.abs() < 1e-14);
        assert!(result.ei.abs() < 1e-14);
        assert!(result.er.abs() < 1e-14);
    }


    #[test]
    fn test_upwind_jump_left_constant_state() {
        let s = make_state(
            1.0,
            0.5,
            2.0,
            2.0,
            2.0,
        );

        let stencil = constant_stencil(s);

        let jump = upwind_jump_left(&stencil);

        assert!(jump.ee.abs() < 1e-12);
        assert!(jump.ei.abs() < 1e-12);
        assert!(jump.er.abs() < 1e-12);
    }


    #[test]
    fn test_upwind_jump_right_constant_state() {
        let s = make_state(
            1.0,
            0.5,
            2.0,
            2.0,
            2.0,
        );

        let stencil = constant_stencil(s);

        let jump = upwind_jump_right(&stencil);

        assert!(jump.ee.abs() < 1e-12);
        assert!(jump.ei.abs() < 1e-12);
        assert!(jump.er.abs() < 1e-12);
    }


    #[test]
    fn test_nonconservative_constant_state() {
        let s = make_state(
            1.0,
            0.5,
            2.0,
            2.0,
            2.0,
        );

        let stencil = constant_stencil(s);

        let result = nonconservative(
            &stencil,
            1.0,
        );

        /*
        A uniform state has:

        dNdx = 0
        WENO jumps = 0

        Therefore the non-conservative contribution vanishes.
        */

        assert!(result.ee.abs() < 1e-12);
        assert!(result.ei.abs() < 1e-12);
        assert!(result.er.abs() < 1e-12);
    }


    #[test]
    fn test_nonconservative_zero_velocity() {
        let stencil = [
            make_state(1.0, 0.1, 1.0, 2.0, 3.0),
            make_state(1.0, 0.1, 1.1, 2.1, 3.1),
            make_state(1.0, 0.1, 1.2, 2.2, 3.2),
            make_state(1.0, 0.1, 1.3, 2.3, 3.3),
            make_state(1.0, 0.1, 1.4, 2.4, 3.4),
            make_state(1.0, 0.1, 1.5, 2.5, 3.5),
            make_state(1.0, 0.1, 1.6, 2.6, 3.6),
        ];

        let result = nonconservative(&stencil, 1.0);

        /*
        Since u=0:

        noncon = jumps only
        */

        assert!(result.rho == 0.0);
        assert!(result.mom == 0.0);
        assert!(result.ee.is_finite());
        assert!(result.ei.is_finite());
        assert!(result.er.is_finite());
    }
}