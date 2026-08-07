use crate::state::State;
use crate::constant;




pub fn source(s: State) -> State {

    let te = s.te();
    let tr = s.tr();
    let ti = s.ti();

    State {
        rho: 0.0,
        mom: 0.0,
        ee: -constant::OMEGA_EI*(te-ti) - constant::OMEGA_ER*(te.powi(4) - tr.powi(4)),
        ei: constant::OMEGA_EI*(te - ti),
        er: constant::OMEGA_ER*(te.powi(4) - tr.powi(4)),
    }

}