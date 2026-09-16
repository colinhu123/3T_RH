use crate::{constant, state,state::State};
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;


#[derive(Clone, Copy, Debug)]
pub struct Stencil6 {
    pub points: [state::State; 6],
    pub dir: state::Direction,
}

impl Stencil6 {
    pub fn build_l(&self) -> Array2<f64> {
        let d3 = state::Derived::from_state(self.points[3]);
        let d2 = state::Derived::from_state(self.points[2]);
        let m = Self::build_l_plain(&self.points[3], &self.points[2], &d3, &d2, self.dir);
        Array2::from_shape_fn((6, 6), |(i, j)| m[i][j])
    }

    pub fn build_l_plain(
        state1: &state::State,
        state2: &state::State,
        d3: &state::Derived,
        d2: &state::Derived,
        dir: state::Direction,
    ) -> [[f64; 6]; 6] {
        let u1 = d3.u;
        let u2 = d2.u;
        let v1 = d3.v;
        let v2 = d2.v;

        let ee1 = d3.e_e;
        let ei1 = d3.e_i;
        let er1 = d3.e_r;
        let ee2 = d2.e_e;
        let ei2 = d2.e_i;
        let er2 = d2.e_r;

        let rho1 = state1.rho.sqrt();
        let rho2 = state2.rho.sqrt();
        let rho_sum = rho1 + rho2;

        let u = (u1 * rho1 + u2 * rho2) / rho_sum;
        let v = (v1 * rho1 + v2 * rho2) / rho_sum;
        let w2 = u.powi(2) + v.powi(2);
        let ee = (ee1 * rho1 + ee2 * rho2) / rho_sum;
        let ei = (ei1 * rho1 + ei2 * rho2) / rho_sum;
        let er = (er1 * rho1 + er2 * rho2) / rho_sum;

        let gi = constant::GAMMA_I - 1.0;
        let ge = constant::GAMMA_E - 1.0;
        let gr = constant::GAMMA_R - 1.0;

        let gt = gi + ge + gr;
        let cs2 =
            constant::GAMMA_E * ge * ee + constant::GAMMA_I * gi * ei + constant::GAMMA_R * gr * er;
        let cs = cs2.sqrt();

        let he = 6.0 * gi * gr * (constant::GAMMA_I * ei - constant::GAMMA_R * er)
            + constant::GAMMA_E * gt * ee * w2;

        let hi = 6.0 * ge * gr * (constant::GAMMA_R * er - constant::GAMMA_E * ee)
            + constant::GAMMA_I * gt * ei * w2;

        let hr = 6.0 * ge * gi * (constant::GAMMA_E * ee - constant::GAMMA_I * ei)
            + constant::GAMMA_R * gt * er * w2;

        let b = gt * (36.0 * ge * gi * gr + gt * w2.powi(2)) * cs2;

        let acoustic_den = 12.0 * cs2;

        match dir {
            // ========================================================
            // X direction
            // ========================================================
            state::Direction::X => {
                [
                    // ------------------------------------------------
                    // L_A^(1)
                    // acoustic: u - cs
                    // ------------------------------------------------
                    [
                        (gt * w2 + 6.0 * u * cs) / acoustic_den,
                        (-2.0 * gt * u - 6.0 * cs) / acoustic_den,
                        (-2.0 * gt * v) / acoustic_den,
                        6.0 * ge / acoustic_den,
                        6.0 * gi / acoustic_den,
                        6.0 * gr / acoustic_den,
                    ],
                    // ------------------------------------------------
                    // L_A^(2)
                    // transverse velocity mode
                    // ------------------------------------------------

                    // ------------------------------------------------
                    // L_A^(3)
                    // electron-energy mode
                    // ------------------------------------------------
                    [
                        ge / gt - ge * w2 / (6.0 * cs2) - ge * gt * w2 / b * (he - cs2 * w2),
                        ge * u / (3.0 * cs2) + 2.0 * ge * gt * u / b * (he - cs2 * w2),
                        // TYPO FIX:
                        // this must use v in BOTH terms.
                        ge * v / (3.0 * cs2) + 2.0 * ge * gt * v / b * (he - cs2 * w2),
                        -ge.powi(2) / (gt * cs2) - 6.0 * ge / b * (ge * he - gt * cs2 * w2),
                        -ge * gi / (gt * cs2) - 6.0 * ge * gi / b * (he - 6.0 * gr * cs2),
                        -ge * gr / (gt * cs2) - 6.0 * ge * gr / b * (he + 6.0 * gi * cs2),
                    ],
                    // ------------------------------------------------
                    // L_A^(4)
                    // ion-energy mode
                    // ------------------------------------------------
                    [
                        gi / gt - gi * w2 / (6.0 * cs2) - gi * gt * w2 / b * (hi - cs2 * w2),
                        gi * u / (3.0 * cs2) + 2.0 * gi * gt * u / b * (hi - cs2 * w2),
                        gi * v / (3.0 * cs2) + 2.0 * gi * gt * v / b * (hi - cs2 * w2),
                        -gi * ge / (gt * cs2) - 6.0 * gi * ge / b * (hi + 6.0 * gr * cs2),
                        -gi.powi(2) / (gt * cs2) - 6.0 * gi / b * (gi * hi - gt * cs2 * w2),
                        -gi * gr / (gt * cs2) - 6.0 * gi * gr / b * (hi - 6.0 * ge * cs2),
                    ],
                    // ------------------------------------------------
                    // L_A^(5)
                    // radiation-energy mode
                    // ------------------------------------------------
                    [
                        gr / gt - gr * w2 / (6.0 * cs2) - gr * gt * w2 / b * (hr - cs2 * w2),
                        gr * u / (3.0 * cs2) + 2.0 * gr * gt * u / b * (hr - cs2 * w2),
                        gr * v / (3.0 * cs2) + 2.0 * gr * gt * v / b * (hr - cs2 * w2),
                        -gr * ge / (gt * cs2) - 6.0 * gr * ge / b * (hr - 6.0 * gi * cs2),
                        -gr * gi / (gt * cs2) - 6.0 * gr * gi / b * (hr + 6.0 * ge * cs2),
                        -gr.powi(2) / (gt * cs2) - 6.0 * gr / b * (gr * hr - gt * cs2 * w2),
                    ],
                    [-v, 0.0, 1.0, 0.0, 0.0, 0.0],
                    // ------------------------------------------------
                    // L_A^(6)
                    // acoustic: u + cs
                    // ------------------------------------------------
                    [
                        (gt * w2 - 6.0 * u * cs) / acoustic_den,
                        (-2.0 * gt * u + 6.0 * cs) / acoustic_den,
                        (-2.0 * gt * v) / acoustic_den,
                        6.0 * ge / acoustic_den,
                        6.0 * gi / acoustic_den,
                        6.0 * gr / acoustic_den,
                    ],
                ]
            }

            // ========================================================
            // Y direction
            // ========================================================
            state::Direction::Y => {
                [
                    // ------------------------------------------------
                    // L_B^(1)
                    // acoustic: v - cs
                    // ------------------------------------------------
                    [
                        (gt * w2 + 6.0 * v * cs) / acoustic_den,
                        (-2.0 * gt * u) / acoustic_den,
                        (-2.0 * gt * v - 6.0 * cs) / acoustic_den,
                        6.0 * ge / acoustic_den,
                        6.0 * gi / acoustic_den,
                        6.0 * gr / acoustic_den,
                    ],
                    // ------------------------------------------------
                    // L_B^(2)
                    // transverse velocity mode
                    // ------------------------------------------------
                    [-u, 1.0, 0.0, 0.0, 0.0, 0.0],
                    // ------------------------------------------------
                    // L_B^(3)
                    // electron-energy mode
                    // ------------------------------------------------
                    [
                        ge / gt - ge * w2 / (6.0 * cs2) - ge * gt * w2 / b * (he - cs2 * w2),
                        ge * u / (3.0 * cs2) + 2.0 * ge * gt * u / b * (he - cs2 * w2),
                        ge * v / (3.0 * cs2) + 2.0 * ge * gt * v / b * (he - cs2 * w2),
                        -ge.powi(2) / (gt * cs2) - 6.0 * ge / b * (ge * he - gt * cs2 * w2),
                        -ge * gi / (gt * cs2) - 6.0 * ge * gi / b * (he - 6.0 * gr * cs2),
                        -ge * gr / (gt * cs2) - 6.0 * ge * gr / b * (he + 6.0 * gi * cs2),
                    ],
                    // ------------------------------------------------
                    // L_B^(4)
                    // ion-energy mode
                    // ------------------------------------------------
                    [
                        gi / gt - gi * w2 / (6.0 * cs2) - gi * gt * w2 / b * (hi - cs2 * w2),
                        gi * u / (3.0 * cs2) + 2.0 * gi * gt * u / b * (hi - cs2 * w2),
                        gi * v / (3.0 * cs2) + 2.0 * gi * gt * v / b * (hi - cs2 * w2),
                        -gi * ge / (gt * cs2) - 6.0 * gi * ge / b * (hi + 6.0 * gr * cs2),
                        -gi.powi(2) / (gt * cs2) - 6.0 * gi / b * (gi * hi - gt * cs2 * w2),
                        -gi * gr / (gt * cs2) - 6.0 * gi * gr / b * (hi - 6.0 * ge * cs2),
                    ],
                    // ------------------------------------------------
                    // L_B^(5)
                    // radiation-energy mode
                    // ------------------------------------------------
                    [
                        gr / gt - gr * w2 / (6.0 * cs2) - gr * gt * w2 / b * (hr - cs2 * w2),
                        gr * u / (3.0 * cs2) + 2.0 * gr * gt * u / b * (hr - cs2 * w2),
                        gr * v / (3.0 * cs2) + 2.0 * gr * gt * v / b * (hr - cs2 * w2),
                        -gr * ge / (gt * cs2) - 6.0 * gr * ge / b * (hr - 6.0 * gi * cs2),
                        -gr * gi / (gt * cs2) - 6.0 * gr * gi / b * (hr + 6.0 * ge * cs2),
                        -gr.powi(2) / (gt * cs2) - 6.0 * gr / b * (gr * hr - gt * cs2 * w2),
                    ],
                    // ------------------------------------------------
                    // L_B^(6)
                    // acoustic: v + cs
                    // ------------------------------------------------
                    [
                        (gt * w2 - 6.0 * v * cs) / acoustic_den,
                        (-2.0 * gt * u) / acoustic_den,
                        (-2.0 * gt * v + 6.0 * cs) / acoustic_den,
                        6.0 * ge / acoustic_den,
                        6.0 * gi / acoustic_den,
                        6.0 * gr / acoustic_den,
                    ],
                ]
            }
        }
    }

    pub fn _build_l_raw(&self) -> Array2<f64> {
        let (_lambda, r) = self.build_r_roe_ave();
        let l = r.inv().unwrap();
        l
    }

    pub fn build_r_roe_ave(&self) -> (Array1<f64>, Array2<f64>) {
        let d3 = state::Derived::from_state(self.points[3]);
        let d2 = state::Derived::from_state(self.points[2]);
        let (lambda, r) = Self::build_r_plain(&self.points[3], &self.points[2], &d3, &d2, self.dir);
        (
            Array1::from_vec(lambda.to_vec()),
            Array2::from_shape_fn((6, 6), |(i, j)| r[i][j]),
        )
    }

    pub fn build_r_plain(
        state1: &state::State,
        state2: &state::State,
        d3: &state::Derived,
        d2: &state::Derived,
        dir: state::Direction,
    ) -> ([f64; 6], [[f64; 6]; 6]) {
        let u1 = d3.u;
        let u2 = d2.u;
        let v1 = d3.v;
        let v2 = d2.v;
        let ee1 = d3.e_e;
        let ei1 = d3.e_i;
        let er1 = d3.e_r;
        let ee2 = d2.e_e;
        let ei2 = d2.e_i;
        let er2 = d2.e_r;
        let rho1 = state1.rho.sqrt();
        let rho2 = state2.rho.sqrt();

        let u = (u1 * rho1 + u2 * rho2) / (rho1 + rho2);
        let v = (v1 * rho1 + v2 * rho2) / (rho1 + rho2);
        let w2 = u.powi(2) + v.powi(2);
        let ee = (ee1 * rho1 + ee2 * rho2) / (rho1 + rho2);
        let ei = (ei1 * rho1 + ei2 * rho2) / (rho1 + rho2);
        let er = (er1 * rho1 + er2 * rho2) / (rho1 + rho2);

        let gi = constant::GAMMA_I - 1.0;
        let ge = constant::GAMMA_E - 1.0;
        let gr = constant::GAMMA_R - 1.0;

        let cs = (constant::GAMMA_E * ge * ee
            + constant::GAMMA_I * gi * ei
            + constant::GAMMA_R * gr * er)
            .sqrt();
        let gt = gi + ge + gr;

        match dir {
            state::Direction::X => {
                let r = [
                    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                    [u - cs, u, u, u, 0.0, u + cs],
                    [v, v, v, v, 1.0, v],
                    [
                        constant::GAMMA_E * ee + w2 / 6.0 - u * cs / 3.0,
                        gt * w2 / (6.0 * ge),
                        -gr,
                        gi,
                        v / 3.0,
                        constant::GAMMA_E * ee + w2 / 6.0 + u * cs / 3.0,
                    ],
                    [
                        constant::GAMMA_I * ei + w2 / 6.0 - u * cs / 3.0,
                        gr,
                        gt * w2 / (6.0 * gi),
                        -ge,
                        v / 3.0,
                        constant::GAMMA_I * ei + w2 / 6.0 + u * cs / 3.0,
                    ],
                    [
                        constant::GAMMA_R * er + w2 / 6.0 - u * cs / 3.0,
                        -gi,
                        ge,
                        gt * w2 / (6.0 * gr),
                        v / 3.0,
                        constant::GAMMA_R * er + w2 / 6.0 + u * cs / 3.0,
                    ],
                ];
                let lambda = [u - cs, u, u, u, u, u + cs];

                return (lambda, r);
            }
            state::Direction::Y => {
                let r = [
                    [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    [u, 1.0, u, u, u, u],
                    [v - cs, 0.0, v, v, v, v + cs],
                    [
                        constant::GAMMA_E * ee + w2 / 6.0 - v * cs / 3.0,
                        u / 3.0,
                        gt * w2 / (6.0 * ge),
                        -gr,
                        gi,
                        constant::GAMMA_E * ee + w2 / 6.0 + v * cs / 3.0,
                    ],
                    [
                        constant::GAMMA_I * ei + w2 / 6.0 - v * cs / 3.0,
                        u / 3.0,
                        gr,
                        gt * w2 / (6.0 * gi),
                        -ge,
                        constant::GAMMA_I * ei + w2 / 6.0 + v * cs / 3.0,
                    ],
                    [
                        constant::GAMMA_R * er + w2 / 6.0 - v * cs / 3.0,
                        u / 3.0,
                        -gi,
                        ge,
                        gt * w2 / (6.0 * gr),
                        constant::GAMMA_R * er + w2 / 6.0 + v * cs / 3.0,
                    ],
                ];
                let lambda = [v - cs, v, v, v, v, v + cs];
                return (lambda, r);
            }
        }
    }

    pub fn con2char(&self, l: &Array2<f64>) -> Self {
        let mut new_stencil: [state::State; 6] = [state::State::new(); 6];
        for i in 0..6 {
            let tmp_k: Array1<f64> = Array1::from_vec(self.points[i].state2arr().to_vec());

            new_stencil[i] = state::State::arr2state(l.dot(&tmp_k));
        }

        Self {
            points: new_stencil,
            dir: self.dir,
        }
    }

    pub fn state2flux(&self) -> Self {
        let mut new_stencil = [state::State::new(); 6];
        for i in 0..6 {
            new_stencil[i] = self.points[i].flux(self.dir);
        }

        Self {
            points: new_stencil,
            dir: self.dir,
        }
    }

    pub fn stencil2arr(&self) -> [[f64; 6]; 6] {
        let mut rho_list = [0.0; 6];
        let mut momx_list = [0.0; 6];
        let mut momy_list = [0.0; 6];
        let mut ee_list = [0.0; 6];
        let mut ei_list = [0.0; 6];
        let mut er_list = [0.0; 6];

        for i in 0..6 {
            rho_list[i] = self.points[i].rho;
            momx_list[i] = self.points[i].mom_x;
            momy_list[i] = self.points[i].mom_y;
            ee_list[i] = self.points[i].ee;
            ei_list[i] = self.points[i].ei;
            er_list[i] = self.points[i].er;
        }

        [rho_list, momx_list, momy_list, ee_list, ei_list, er_list]
    }

    pub fn reconstruction(&self, recon_type: bool) -> state::State {
        let l = self.build_l();
        let (flux_l, state_l): ([state::State; 6], [state::State; 6]) = if recon_type {
            (
                self.state2flux().con2char(&l).points,
                self.con2char(&l).points,
            )
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
        let a5 = lambda[5].abs();

        for i in 0..6 {
            f_plus_stencil[i] = state::State {
                rho: 0.5 * (flux_l[i].rho + a0 * state_l[i].rho),
                mom_x: 0.5 * (flux_l[i].mom_x + a1 * state_l[i].mom_x),
                mom_y: 0.5 * (flux_l[i].mom_y + a2 * state_l[i].mom_y),
                ee: 0.5 * (flux_l[i].ee + a3 * state_l[i].ee),
                ei: 0.5 * (flux_l[i].ei + a4 * state_l[i].ei),
                er: 0.5 * (flux_l[i].er + a5 * state_l[i].er),
            };
            f_minus_stencil[i] = state::State {
                rho: 0.5 * (flux_l[i].rho - a0 * state_l[i].rho),
                mom_x: 0.5 * (flux_l[i].mom_x - a1 * state_l[i].mom_x),
                mom_y: 0.5 * (flux_l[i].mom_y - a2 * state_l[i].mom_y),
                ee: 0.5 * (flux_l[i].ee - a3 * state_l[i].ee),
                ei: 0.5 * (flux_l[i].ei - a4 * state_l[i].ei),
                er: 0.5 * (flux_l[i].er - a5 * state_l[i].er),
            };
        }

        let f_plus_stencil = Self {
            points: f_plus_stencil,
            dir: self.dir,
        };
        let f_minus_stencil = Self {
            points: f_minus_stencil,
            dir: self.dir,
        };
        let tmp = f_plus_stencil.stencil2arr();
        let tmp1 = f_minus_stencil.stencil2arr();
        let mut flux_plus = [0.0; 6];
        let mut flux_minus = [0.0; 6];
        for i in 0..6 {
            let stencil = [tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4]];
            flux_plus[i] = weno5(&stencil);
            let stencil = [tmp1[i][5], tmp1[i][4], tmp1[i][3], tmp1[i][2], tmp1[i][1]];
            flux_minus[i] = weno5(&stencil);
        }

        let flux = state::State {
            rho: flux_plus[0] + flux_minus[0],
            mom_x: flux_plus[1] + flux_minus[1],
            mom_y: flux_plus[2] + flux_minus[2],
            ee: flux_plus[3] + flux_minus[3],
            ei: flux_plus[4] + flux_minus[4],
            er: flux_plus[5] + flux_minus[5],
        };
        let tmp_k: Array1<f64> = Array1::from_vec(flux.state2arr().to_vec());

        state::State::arr2state(r.dot(&tmp_k))
    }

    /// Allocation-free reconstruction for the hot path.
    ///
    /// Takes the 6-point stencil of states plus per-point precomputed
    /// `Derived` quantities. Numerically identical to `reconstruction()`.
    pub fn reconstruction_fast(
        points: &[state::State; 6],
        d: &[state::Derived; 6],
        dir: state::Direction,
        recon_type: bool,
    ) -> state::State {
        let l = Self::build_l_plain(&points[3], &points[2], &d[3], &d[2], dir);
        let (lambda, r) = Self::build_r_plain(&points[3], &points[2], &d[3], &d[2], dir);

        // ====================================================================
        // Positivity-preserving characteristic FD-WENO5 branch
        // (the PP_SWITCH == true "else" path of the original split).
        //
        // Input stencil for interface i+1/2:
        //
        //     points[0] = U_{i-2}
        //     points[1] = U_{i-1}
        //     points[2] = U_i
        //     points[3] = U_{i+1}
        //     points[4] = U_{i+2}
        //     points[5] = U_{i+3}
        //
        // Zhang-Shu normalized LF splitting:
        //
        //     f+(U) = 1/2 [ U + F(U)/alpha ]
        //     f-(U) = 1/2 [ U - F(U)/alpha ]
        //
        // with a single scalar local Lax-Friedrichs speed alpha, followed by
        //
        //     Fhat = alpha [ (h+)^-_PP - (h-)^+_PP ].
        //
        // NOTE: this branch ignores `recon_type`; it always works in the
        // Roe characteristic space.
        // ====================================================================
        if constant::PP_SWITCH {
            // ----------------------------------------------------------------
            // Scalar local Lax-Friedrichs alpha over the whole stencil:
            //
            //     X:  alpha = max_j ( |u_j| + c_j )
            //     Y:  alpha = max_j ( |v_j| + c_j )
            //
            // We need a single scalar LF speed, so no PHI-based c reduction.
            // ----------------------------------------------------------------
            let mut alpha = 0.0_f64;

            for j in 0..6 {
                let velocity = match dir {
                    state::Direction::X => d[j].u,
                    state::Direction::Y => d[j].v,
                };

                let cs = points[j].cs();

                if !velocity.is_finite() || !cs.is_finite() {
                    panic!(
                        "PP-WENO: non-finite velocity/sound speed at stencil point {}. \
                         velocity={}, cs={}, state={:?}",
                        j, velocity, cs, points[j]
                    );
                }

                if cs < 0.0 {
                    panic!(
                        "PP-WENO: negative sound speed at stencil point {}: cs={}",
                        j, cs
                    );
                }

                alpha = alpha.max(velocity.abs() + cs);
            }

            if !alpha.is_finite() || alpha <= 0.0 {
                panic!(
                    "PP-WENO: invalid local LF alpha={} for stencil {:?}",
                    alpha, points
                );
            }

            // ----------------------------------------------------------------
            // Physical-space normalized LF split states:
            //
            //     fp = 1/2 ( U + F/alpha )
            //     fm = 1/2 ( U - F/alpha )
            // ----------------------------------------------------------------
            let mut fp_phys = [State::new(); 6];
            let mut fm_phys = [State::new(); 6];

            for j in 0..6 {
                let flux = points[j].flux_from_derived(&d[j], dir);

                fp_phys[j] = State {
                    rho: 0.5 * (points[j].rho + flux.rho / alpha),
                    mom_x: 0.5 * (points[j].mom_x + flux.mom_x / alpha),
                    mom_y: 0.5 * (points[j].mom_y + flux.mom_y / alpha),
                    ee: 0.5 * (points[j].ee + flux.ee / alpha),
                    ei: 0.5 * (points[j].ei + flux.ei / alpha),
                    er: 0.5 * (points[j].er + flux.er / alpha),
                };

                fm_phys[j] = State {
                    rho: 0.5 * (points[j].rho - flux.rho / alpha),
                    mom_x: 0.5 * (points[j].mom_x - flux.mom_x / alpha),
                    mom_y: 0.5 * (points[j].mom_y - flux.mom_y / alpha),
                    ee: 0.5 * (points[j].ee - flux.ee / alpha),
                    ei: 0.5 * (points[j].ei - flux.ei / alpha),
                    er: 0.5 * (points[j].er - flux.er / alpha),
                };
            }

            // ----------------------------------------------------------------
            // Transform the split states into the Roe characteristic space:
            //
            //     cp_j = L fp_j
            //     cm_j = L fm_j
            // ----------------------------------------------------------------
            let mut cp = [[0.0_f64; 6]; 6];
            let mut cm = [[0.0_f64; 6]; 6];

            for j in 0..6 {
                cp[j] = l_dot(&l, &fp_phys[j].state2arr());
                cm[j] = l_dot(&l, &fm_phys[j].state2arr());
            }

            // ----------------------------------------------------------------
            // Positive split: positive waves run left -> right, so reconstruct
            // from the LEFT of interface i+1/2 using points 0..4 (i-2 .. i+2).
            //
            //     hp_char = L (h+)^-_{i+1/2}
            // ----------------------------------------------------------------
            let mut hp_char = [0.0_f64; 6];

            for k in 0..6 {
                let stencil = [cp[0][k], cp[1][k], cp[2][k], cp[3][k], cp[4][k]];
                hp_char[k] = weno5(&stencil);
            }

            // ----------------------------------------------------------------
            // Negative split: negative waves run right -> left, so mirror the
            // stencil to points 5..1 (i+3 .. i-1) and reuse the same
            // left-biased weno5().
            //
            //     hm_char = L (h-)^+_{i+1/2}
            // ----------------------------------------------------------------
            let mut hm_char = [0.0_f64; 6];

            for k in 0..6 {
                let stencil = [cm[5][k], cm[4][k], cm[3][k], cm[2][k], cm[1][k]];
                hm_char[k] = weno5(&stencil);
            }

            // ----------------------------------------------------------------
            // Transform the reconstructed split states back to physical space.
            // ----------------------------------------------------------------
            let hp = r_dot(&r, &hp_char);
            let hm = r_dot(&r, &hm_char);

            let h_plus_high = State {
                rho: hp[0],
                mom_x: hp[1],
                mom_y: hp[2],
                ee: hp[3],
                ei: hp[4],
                er: hp[5],
            };

            let h_minus_high = State {
                rho: hm[0],
                mom_x: hm[1],
                mom_y: hm[2],
                ee: hm[3],
                ei: hm[4],
                er: hm[5],
            };

            // ----------------------------------------------------------------
            // Safe states:
            //
            //     h_plus_high  is reconstructed on cell i      -> safe_plus  = f+(U_i)    = fp_phys[2]
            //     h_minus_high is reconstructed on cell i+1    -> safe_minus = f-(U_{i+1}) = fm_phys[3]
            // ----------------------------------------------------------------
            let safe_plus = fp_phys[2];
            let safe_minus = fm_phys[3];

            // ----------------------------------------------------------------
            // Apply the positivity/admissibility limiter to each split state.
            // ----------------------------------------------------------------
            let h_plus_pp = positivity_limiter(safe_plus, h_plus_high);
            let h_minus_pp = positivity_limiter(safe_minus, h_minus_high);

            // ----------------------------------------------------------------
            // Final numerical flux:
            //
            //     Fhat = alpha [ (h+)^-_PP - (h-)^+_PP ]
            // ----------------------------------------------------------------
            return State {
                rho: alpha * (h_plus_pp.rho - h_minus_pp.rho),
                mom_x: alpha * (h_plus_pp.mom_x - h_minus_pp.mom_x),
                mom_y: alpha * (h_plus_pp.mom_y - h_minus_pp.mom_y),
                ee: alpha * (h_plus_pp.ee - h_minus_pp.ee),
                ei: alpha * (h_plus_pp.ei - h_minus_pp.ei),
                er: alpha * (h_plus_pp.er - h_minus_pp.er),
            };
        }

        let mut char_flux = [state::State::new(); 6];
        let mut char_state = [state::State::new(); 6];

        for i in 0..6 {
            let fl = points[i].flux_from_derived(&d[i], dir);
            if recon_type {
                let c = l_dot(&l, &[fl.rho, fl.mom_x, fl.mom_y, fl.ee, fl.ei, fl.er]);
                char_flux[i] = state::State {
                    rho: c[0],
                    mom_x: c[1],
                    mom_y: c[2],
                    ee: c[3],
                    ei: c[4],
                    er: c[5],
                };
                let p = &points[i];
                let c = l_dot(&l, &[p.rho, p.mom_x, p.mom_y, p.ee, p.ei, p.er]);
                char_state[i] = state::State {
                    rho: c[0],
                    mom_x: c[1],
                    mom_y: c[2],
                    ee: c[3],
                    ei: c[4],
                    er: c[5],
                };
            } else {
                char_flux[i] = fl;
                char_state[i] = points[i];
            }
        }

        /*
        1 define cs value
        2 modify wave speed of each component
        */

        let u1 = match dir {
            state::Direction::X => points[2].mom_x / points[2].rho,
            state::Direction::Y => points[2].mom_y / points[2].rho,
        };
        let u2 = match dir {
            state::Direction::X => points[3].mom_x / points[3].rho,
            state::Direction::Y => points[3].mom_y / points[3].rho,
        };

        let cs1 = points[2].cs();
        let cs1 = (constant::PHI * u1.abs()).min(cs1);
        let cs2 = points[3].cs();
        let cs2 = (constant::PHI * u2.abs()).min(cs2);

        let a0 = ((u1 - cs1).abs()).max((u2 - cs2).abs());
        let a1 = (u1.abs()).max(u2.abs());
        let a2 = (u1.abs()).max(u2.abs());
        let a3 = (u1.abs()).max(u2.abs());
        let a4 = (u1.abs()).max(u2.abs());
        let a5 = ((u1 + cs1).abs()).max((u2 + cs2).abs());
        
        let mut f_plus_stencil = [state::State::new(); 6];
        let mut f_minus_stencil = [state::State::new(); 6];

        for i in 0..6 {
            f_plus_stencil[i] = state::State {
                rho: 0.5 * (char_flux[i].rho + a0 * char_state[i].rho),
                mom_x: 0.5 * (char_flux[i].mom_x + a1 * char_state[i].mom_x),
                mom_y: 0.5 * (char_flux[i].mom_y + a2 * char_state[i].mom_y),
                ee: 0.5 * (char_flux[i].ee + a3 * char_state[i].ee),
                ei: 0.5 * (char_flux[i].ei + a4 * char_state[i].ei),
                er: 0.5 * (char_flux[i].er + a5 * char_state[i].er),
            };
            f_minus_stencil[i] = state::State {
                rho: 0.5 * (char_flux[i].rho - a0 * char_state[i].rho),
                mom_x: 0.5 * (char_flux[i].mom_x - a1 * char_state[i].mom_x),
                mom_y: 0.5 * (char_flux[i].mom_y - a2 * char_state[i].mom_y),
                ee: 0.5 * (char_flux[i].ee - a3 * char_state[i].ee),
                ei: 0.5 * (char_flux[i].ei - a4 * char_state[i].ei),
                er: 0.5 * (char_flux[i].er - a5 * char_state[i].er),
            };
        }

        let tmp = stencil_arr(&f_plus_stencil);
        let tmp1 = stencil_arr(&f_minus_stencil);

        let mut flux_plus = [0.0; 6];
        let mut flux_minus = [0.0; 6];
        for i in 0..6 {
            let stencil = [tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], tmp[i][4]];
            flux_plus[i] = weno5(&stencil);
            let stencil = [tmp1[i][5], tmp1[i][4], tmp1[i][3], tmp1[i][2], tmp1[i][1]];
            flux_minus[i] = weno5(&stencil);
        }

        // The PP_SWITCH path was already handled above by an early return,
        // so this legacy reconstruction is only reached when PP is disabled.
        let flux = state::State {
            rho: flux_plus[0] + flux_minus[0],
            mom_x: flux_plus[1] + flux_minus[1],
            mom_y: flux_plus[2] + flux_minus[2],
            ee: flux_plus[3] + flux_minus[3],
            ei: flux_plus[4] + flux_minus[4],
            er: flux_plus[5] + flux_minus[5],
        };

        let c = r_dot(
            &r,
            &[flux.rho, flux.mom_x, flux.mom_y, flux.ee, flux.ei, flux.er],
        );
        state::State {
            rho: c[0],
            mom_x: c[1],
            mom_y: c[2],
            ee: c[3],
            ei: c[4],
            er: c[5],
        
        }
    }
}


#[inline(always)]
fn q_star(safe: State, high: State) -> State {
    let inv = 1.0 / (1.0 - constant::PP_W);

    State {
        rho: (safe.rho - constant::PP_W * high.rho) * inv,
        mom_x: (safe.mom_x - constant::PP_W * high.mom_x) * inv,
        mom_y: (safe.mom_y - constant::PP_W * high.mom_y) * inv,
        ee: (safe.ee - constant::PP_W * high.ee) * inv,
        ei: (safe.ei - constant::PP_W * high.ei) * inv,
        er: (safe.er - constant::PP_W * high.er) * inv,
    }
}

#[derive(Clone, Copy)]
enum EnergyComponent {
    Electron,
    Ion,
    Radiation,
}
#[inline(always)]
fn state_lerp(safe: State, q: State, theta: f64) -> State {
    State {
        rho: safe.rho + theta * (q.rho - safe.rho),
        mom_x: safe.mom_x + theta * (q.mom_x - safe.mom_x),
        mom_y: safe.mom_y + theta * (q.mom_y - safe.mom_y),
        ee: safe.ee + theta * (q.ee - safe.ee),
        ei: safe.ei + theta * (q.ei - safe.ei),
        er: safe.er + theta * (q.er - safe.er),
    }
}

///
/// Your admissibility constraint is
///
///     e_s = E_s/rho - (u^2 + v^2)/6 > 0
///
/// which, for rho > 0, is equivalent to
///
///     G_s(U)
///       = E_s - (mx^2 + my^2)/(6 rho)
///       > 0.
///
/// We return G_s here.
///
#[inline(always)]
fn internal_constraint(s: State, component: EnergyComponent) -> f64 {
    if !s.rho.is_finite() || s.rho <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let energy = match component {
        EnergyComponent::Electron => s.ee,
        EnergyComponent::Ion => s.ei,
        EnergyComponent::Radiation => s.er,
    };

    energy - (s.mom_x * s.mom_x + s.mom_y * s.mom_y) / (6.0 * s.rho)
}


#[inline(always)]
fn state_is_finite(s: State) -> bool {
    s.rho.is_finite()
        && s.mom_x.is_finite()
        && s.mom_y.is_finite()
        && s.ee.is_finite()
        && s.ei.is_finite()
        && s.er.is_finite()
}


#[inline(always)]
fn is_admissible(s: State) -> bool {
    state_is_finite(s)
        && s.rho >= constant::EPS_LIMITER
        && internal_constraint(s, EnergyComponent::Electron) >= constant::EPS_LIMITER
        && internal_constraint(s, EnergyComponent::Ion) >= constant::EPS_LIMITER
        && internal_constraint(s, EnergyComponent::Radiation) >= constant::EPS_LIMITER
}

#[inline]
fn density_limiter(safe: State, high: State) -> State {
    let qs = q_star(safe, high);

    let rho_min = high.rho.min(qs.rho);

    if rho_min >= constant::EPS_LIMITER {
        return high;
    }

    let denominator = safe.rho - rho_min;

    if !denominator.is_finite() || denominator <= 0.0 {
        panic!(
            "PP density limiter: invalid denominator. \
             safe.rho={}, high.rho={}, qstar.rho={}",
            safe.rho, high.rho, qs.rho
        );
    }

    let theta =
        ((safe.rho - constant::EPS_LIMITER) / denominator)
            .clamp(0.0, 1.0);

    if !theta.is_finite() {
        panic!(
            "PP density limiter produced non-finite theta: \
             safe.rho={}, rho_min={}",
            safe.rho, rho_min
        );
    }

    let mut limited = high;

    limited.rho =
        safe.rho + theta * (high.rho - safe.rho);

    limited
}

#[inline]
fn theta_for_component(
    safe: State,
    candidate: State,
    component: EnergyComponent,
) -> f64 {
    let g_candidate = internal_constraint(candidate, component);

    if g_candidate >= constant::EPS_LIMITER {
        return 1.0;
    }

    let g_safe = internal_constraint(safe, component);

    if !g_safe.is_finite() || g_safe <= constant::EPS_LIMITER {
        panic!(
            "PP limiter: safe state is not safely admissible. \
             G_safe={}",
            g_safe
        );
    }

    let mut lo = 0.0;
    let mut hi = 1.0;

    for _ in 0..constant::MAX_ITER {
        let mid = 0.5 * (lo + hi);

        let s = state_lerp(safe, candidate, mid);

        let g = internal_constraint(s, component);

        if g.is_finite() && g >= constant::EPS_LIMITER {
            // Still inside the admissible region.
            // Try to retain more of the high-order state.
            lo = mid;
        } else {
            // Outside the admissible region.
            hi = mid;
        }
    }

    lo
}

#[inline]
fn theta_admissible(safe: State, candidate: State) -> f64 {
    // Fast path: overwhelmingly common in smooth regions.
    if internal_constraint(candidate, EnergyComponent::Electron) >= constant::EPS_LIMITER
        && internal_constraint(candidate, EnergyComponent::Ion) >= constant::EPS_LIMITER
        && internal_constraint(candidate, EnergyComponent::Radiation) >= constant::EPS_LIMITER
    {
        return 1.0;
    }

    let theta_e =
        theta_for_component(
            safe,
            candidate,
            EnergyComponent::Electron,
        );

    let theta_i =
        theta_for_component(
            safe,
            candidate,
            EnergyComponent::Ion,
        );

    let theta_r =
        theta_for_component(
            safe,
            candidate,
            EnergyComponent::Radiation,
        );

    theta_e.min(theta_i).min(theta_r)
}

pub fn positivity_limiter(safe: State, high: State) -> State {
    // ------------------------------------------------------------------
    // 0. Validate safe state.
    // ------------------------------------------------------------------

    if !state_is_finite(safe) {
        panic!(
            "PP limiter received non-finite safe state: {:?}",
            safe
        );
    }

    if !is_admissible(safe) {
        panic!(
            "PP limiter safe state is not admissible.\n\
             safe = {:?}\n\
             G_e = {:e}\n\
             G_i = {:e}\n\
             G_r = {:e}",
            safe,
            internal_constraint(safe, EnergyComponent::Electron),
            internal_constraint(safe, EnergyComponent::Ion),
            internal_constraint(safe, EnergyComponent::Radiation),
        );
    }

    if !state_is_finite(high) {
        panic!(
            "PP limiter received non-finite high-order state: {:?}",
            high
        );
    }


    // ------------------------------------------------------------------
    // 1. Density limiter.
    //
    // This modifies only high.rho.
    // ------------------------------------------------------------------

    let high_hat = density_limiter(safe, high);


    // ------------------------------------------------------------------
    // 2. Reconstruct q* AFTER density limiting.
    // ------------------------------------------------------------------

    let qstar_hat = q_star(safe, high_hat);


    // ------------------------------------------------------------------
    // 3. Density must now be positive for BOTH states.
    // ------------------------------------------------------------------

    let rho_tol = 100.0 * f64::EPSILON * safe.rho.abs().max(1.0);

    if high_hat.rho < constant::EPS_LIMITER - rho_tol
        || qstar_hat.rho < constant::EPS_LIMITER - rho_tol
    {
        panic!(
            "PP density limiter failed.\n\
             safe.rho     = {:e}\n\
             high.rho     = {:e}\n\
             high_hat.rho = {:e}\n\
             qstar.rho    = {:e}",
            safe.rho,
            high.rho,
            high_hat.rho,
            qstar_hat.rho,
        );
    }


    // ------------------------------------------------------------------
    // 4. Generalized pressure/internal-energy limiter.
    //
    // We must check BOTH:
    //
    //     high_hat
    //     qstar_hat
    //
    // against the same safe state.
    // ------------------------------------------------------------------

    let theta_high =
        theta_admissible(safe, high_hat);

    let theta_star =
        theta_admissible(safe, qstar_hat);

    let theta =
        theta_high.min(theta_star);


    if !theta.is_finite() || theta < 0.0 || theta > 1.0 {
        panic!(
            "PP limiter produced invalid theta: {}",
            theta
        );
    }


    // ------------------------------------------------------------------
    // 5. Full conservative-state convex scaling.
    // ------------------------------------------------------------------

    let limited =
        state_lerp(safe, high_hat, theta);


    // ------------------------------------------------------------------
    // 6. q* is NOT independently limited.
    //
    // Reconstruct it from the final interface state so that:
    //
    //     safe = (1-w) q* + w limited
    //
    // remains exactly satisfied.
    // ------------------------------------------------------------------

    let qstar_final =
        q_star(safe, limited);


    // ------------------------------------------------------------------
    // 7. Final sanity check.
    // ------------------------------------------------------------------

    if !is_admissible(limited)
        || !is_admissible(qstar_final)
    {
        panic!(
            "PP limiter final state is not admissible.\n\
             safe        = {:?}\n\
             high        = {:?}\n\
             high_hat    = {:?}\n\
             limited     = {:?}\n\
             qstar_final = {:?}\n\
             theta_high  = {:e}\n\
             theta_star  = {:e}\n\
             theta       = {:e}",
            safe,
            high,
            high_hat,
            limited,
            qstar_final,
            theta_high,
            theta_star,
            theta,
        );
    }

    limited
}

/// component-major: arr[component][point] for a 6-point stencil
#[inline]
fn stencil_arr(st: &[state::State; 6]) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        let s = &st[i];
        out[0][i] = s.rho;
        out[1][i] = s.mom_x;
        out[2][i] = s.mom_y;
        out[3][i] = s.ee;
        out[4][i] = s.ei;
        out[5][i] = s.er;
    }
    out
}

#[inline]
fn l_dot(l: &[[f64; 6]; 6], u: &[f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for i in 0..6 {
        let row = &l[i];
        out[i] = row[0] * u[0]
            + row[1] * u[1]
            + row[2] * u[2]
            + row[3] * u[3]
            + row[4] * u[4]
            + row[5] * u[5];
    }
    out
}

#[inline]
fn r_dot(r: &[[f64; 6]; 6], u: &[f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for i in 0..6 {
        let row = &r[i];
        out[i] = row[0] * u[0]
            + row[1] * u[1]
            + row[2] * u[2]
            + row[3] * u[3]
            + row[4] * u[4]
            + row[5] * u[5];
    }
    out
}

#[inline]
pub fn weno5(stencil: &[f64; 5]) -> f64 {
    let u0 = stencil[0];
    let u1 = stencil[1];
    let u2 = stencil[2];
    let u3 = stencil[3];
    let u4 = stencil[4];

    let beta2 =
        13.0 / 12.0 * (u2 - 2.0 * u3 + u4).powi(2) + 0.25 * (3.0 * u2 - 4.0 * u3 + u4).powi(2);
    let beta1 = 13.0 / 12.0 * (u1 - 2.0 * u2 + u3).powi(2) + 0.25 * (u1 - u3).powi(2);
    let beta0 =
        13.0 / 12.0 * (u0 - 2.0 * u1 + u2).powi(2) + 0.25 * (u0 - 4.0 * u1 + 3.0 * u2).powi(2);
    let d0 = 0.1;
    let d1 = 0.6;
    let d2 = 0.3;

    let a0 = d0 / (constant::DEFAULT_EPS + beta0).powi(2);
    let a1 = d1 / (constant::DEFAULT_EPS + beta1).powi(2);
    let a2 = d2 / (constant::DEFAULT_EPS + beta2).powi(2);

    //let tau5 = (beta0-beta2).abs();

    //let a0 = d0*(1.0 + (tau5/(beta0+utils::DEFAULT_EPS)).powi(2));
    //let a1 = d1*(1.0 + (tau5/(beta1+utils::DEFAULT_EPS)).powi(2));
    //let a2 = d2*(1.0 + (tau5/(beta2+utils::DEFAULT_EPS)).powi(2));

    let sum_o = a0 + a1 + a2;

    let w0 = a0 / sum_o;
    let w1 = a1 / sum_o;
    let w2 = a2 / sum_o;

    let p0 = u0 / 3.0 - 7.0 / 6.0 * u1 + 11.0 / 6.0 * u2;
    let p1 = -u1 / 6.0 + 5.0 / 6.0 * u2 + u3 / 3.0;
    let p2 = u2 / 3.0 + 5.0 / 6.0 * u3 - u4 / 6.0;

    w0 * p0 + w1 * p1 + w2 * p2
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Direction, State};

    const TOL: f64 = 1e-10;

    fn make_state(rho: f64, mom_x: f64, mom_y: f64, ee: f64, ei: f64, er: f64) -> State {
        State {
            rho,
            mom_x,
            mom_y,
            ee,
            ei,
            er,
        }
    }

    fn test_state() -> State {
        make_state(1.0, 0.5, 0.3, 3.0, 2.0, 1.0)
    }

    fn constant_stencil(state: State, dir: Direction) -> Stencil6 {
        Stencil6 {
            points: [state; 6],
            dir,
        }
    }

    fn assert_state_close(a: &State, b: &State, tol: f64) {
        assert!((a.rho - b.rho).abs() < tol, "rho: {} != {}", a.rho, b.rho);

        assert!(
            (a.mom_x - b.mom_x).abs() < tol,
            "mom_x: {} != {}",
            a.mom_x,
            b.mom_x
        );

        assert!(
            (a.mom_y - b.mom_y).abs() < tol,
            "mom_y: {} != {}",
            a.mom_y,
            b.mom_y
        );

        assert!((a.ee - b.ee).abs() < tol, "ee: {} != {}", a.ee, b.ee);

        assert!((a.ei - b.ei).abs() < tol, "ei: {} != {}", a.ei, b.ei);

        assert!((a.er - b.er).abs() < tol, "er: {} != {}", a.er, b.er);
    }

    // ============================================================
    // Scalar WENO5
    // ============================================================

    #[test]
    fn test_weno_constant() {
        let stencil = [5.0, 5.0, 5.0, 5.0, 5.0];

        let result = weno5(&stencil);

        assert!((result - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_weno5_accuracy() {
        let grids = [40usize, 80usize, 160usize, 320usize];

        let mut errors = Vec::new();

        for &nx in &grids {
            let dx = 2.0 * std::f64::consts::PI / nx as f64;

            let mut error = 0.0;
            let mut count = 0;

            for j in 3..nx - 3 {
                let x = j as f64 * dx;

                let stencil_r = [
                    (x - 2.0 * dx).sin(),
                    (x - dx).sin(),
                    x.sin(),
                    (x + dx).sin(),
                    (x + 2.0 * dx).sin(),
                ];

                let xm = x - dx;

                let stencil_l = [
                    (xm - 2.0 * dx).sin(),
                    (xm - dx).sin(),
                    xm.sin(),
                    (xm + dx).sin(),
                    (xm + 2.0 * dx).sin(),
                ];

                let h_iphalf = weno5(&stencil_r);
                let h_imhalf = weno5(&stencil_l);

                let numerical = (h_iphalf - h_imhalf) / dx;

                let exact = x.cos();

                error += (numerical - exact).abs();
                count += 1;
            }

            errors.push(error / count as f64);
        }

        let mut orders = Vec::new();

        for i in 1..errors.len() {
            orders.push((errors[i - 1] / errors[i]).log2());
        }

        println!("errors = {:?}", errors);
        println!("orders = {:?}", orders);

        for order in orders.iter().skip(1) {
            assert!(*order > 4.5, "WENO order too low: {}", order);
        }
    }

    // ============================================================
    // Eigenvalues
    // ============================================================

    #[test]
    fn test_x_has_six_eigenvalues() {
        let stencil = constant_stencil(test_state(), Direction::X);

        let (lambda, _) = stencil.build_r_roe_ave();

        assert_eq!(lambda.len(), 6);
    }

    #[test]
    fn test_y_has_six_eigenvalues() {
        let stencil = constant_stencil(test_state(), Direction::Y);

        let (lambda, _) = stencil.build_r_roe_ave();

        assert_eq!(lambda.len(), 6);
    }

    // ============================================================
    // Eigenvector matrices
    // ============================================================

    fn check_eigen_inverse(dir: Direction) {
        let stencil = constant_stencil(test_state(), dir);

        let l = stencil.build_l();
        let (_, r) = stencil.build_r_roe_ave();

        assert_eq!(l.shape(), &[6, 6]);
        assert_eq!(r.shape(), &[6, 6]);

        let identity = l.dot(&r);

        for i in 0..6 {
            for j in 0..6 {
                if i == j {
                    assert!(
                        (identity[[i, j]] - 1.0).abs() < TOL,
                        "dir={:?}, ({},{}): {}",
                        dir,
                        i,
                        j,
                        identity[[i, j]]
                    );
                } else {
                    assert!(
                        identity[[i, j]].abs() < TOL,
                        "dir={:?}, ({},{}): {}",
                        dir,
                        i,
                        j,
                        identity[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_eigen_inverse_x() {
        check_eigen_inverse(Direction::X);
    }

    #[test]
    fn test_eigen_inverse_y() {
        check_eigen_inverse(Direction::Y);
    }

    // ============================================================
    // Characteristic transformation
    // ============================================================

    fn check_characteristic_round_trip(dir: Direction) {
        let state = test_state();

        let stencil = constant_stencil(state, dir);

        let l = stencil.build_l();
        let (_, r) = stencil.build_r_roe_ave();

        let characteristic = stencil.con2char(&l);

        for q in characteristic.points.iter() {
            let q_char = Array1::from_vec(q.state2arr().to_vec());

            let q_back = State::arr2state(r.dot(&q_char));

            assert_state_close(&q_back, &state, TOL);
        }
    }

    #[test]
    fn test_characteristic_round_trip_x() {
        check_characteristic_round_trip(Direction::X);
    }

    #[test]
    fn test_characteristic_round_trip_y() {
        check_characteristic_round_trip(Direction::Y);
    }

    // ============================================================
    // Constant-state reconstruction
    // ============================================================

    fn check_constant_flux_preserving(dir: Direction) {
        let state = test_state();

        let stencil = constant_stencil(state, dir);

        let reconstructed = stencil.reconstruction(true);

        let exact_flux = state.flux(dir);

        assert_state_close(&reconstructed, &exact_flux, TOL);
    }

    #[test]
    fn test_constant_flux_preserving_x() {
        check_constant_flux_preserving(Direction::X);
    }

    #[test]
    fn test_constant_flux_preserving_y() {
        check_constant_flux_preserving(Direction::Y);
    }

    // ============================================================
    // Conservative-space reconstruction
    // ============================================================

    fn check_constant_flux_preserving_conservative(dir: Direction) {
        let state = test_state();

        let stencil = constant_stencil(state, dir);

        let reconstructed = stencil.reconstruction(false);

        let exact_flux = state.flux(dir);

        assert_state_close(&reconstructed, &exact_flux, TOL);
    }

    #[test]
    fn test_constant_flux_preserving_conservative_x() {
        check_constant_flux_preserving_conservative(Direction::X);
    }

    #[test]
    fn test_constant_flux_preserving_conservative_y() {
        check_constant_flux_preserving_conservative(Direction::Y);
    }

    // ============================================================
    // Direction-sensitive flux
    // ============================================================

    #[test]
    fn test_x_and_y_flux_are_direction_sensitive() {
        /*
        Choose mom_x != mom_y so an accidental X/Y swap
        cannot pass unnoticed.
        */

        let state = make_state(1.0, 0.7, 0.2, 3.0, 2.0, 1.0);

        let flux_x = state.flux(Direction::X);
        let flux_y = state.flux(Direction::Y);

        assert!((flux_x.rho - flux_y.rho).abs() > 1e-12);

        assert!((flux_x.rho - state.mom_x).abs() < TOL);

        assert!((flux_y.rho - state.mom_y).abs() < TOL);
    }
}
