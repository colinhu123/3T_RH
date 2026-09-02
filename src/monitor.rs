use crate::constant;
use crate::field1::Field;
use crate::state::{Derived, State};

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

// ============================================================
// Per-step numerical-health diagnostics for an explicit
// transient compressible flow solver.
//
// This is NOT a steady-state residual monitor.
//
// For an explicit transient solver the per-step data of interest
// is:
//   - time stepping           (dt / dt_cfl ratio)
//   - physical admissibility  (rho / pressure / internal-energy
//                              extrema, useful to catch negative
//                              density or pressure long before a
//                              NaN crash)
//   - global integral quantities (mass, momentum, total energy)
//   - solution temporal activity  ||(U^{n+1}-U^n)/dt||_RMS
//   - spatial oscillation     (TV, second-difference S2)
//
// Only physical fluid grid points (field.fluid[l] == true) are
// included. Embedded solid cells are never counted, and spatial
// differences never cross a fluid/solid interface.
// ============================================================

pub struct Monitor {
    writer: BufWriter<File>,
    flush_interval: usize,
}

#[derive(Clone, Copy, Debug)]
struct Stats {
    n_fluid: usize,

    // extrema
    rho_min: f64,
    rho_max: f64,

    p_min: f64,
    p_max: f64,

    pe_min: f64,
    pi_min: f64,
    pr_min: f64,

    ee_int_min: f64,
    ei_int_min: f64,
    er_int_min: f64,

    speed_max: f64,
    mach_max: f64,

    // global integrals
    mass: f64,
    mom_x: f64,
    mom_y: f64,
    energy: f64,

    // temporal change
    drho_l2: f64,
    dmom_x_l2: f64,
    dmom_y_l2: f64,
    dee_l2: f64,
    dei_l2: f64,
    der_l2: f64,

    // spatial variation
    tv_rho: f64,
    tv_p: f64,

    // second-difference indicator
    s2_rho: f64,
    s2_p: f64,

    // --------------------------------------------------------------
    // Semi-discrete RHS residual norms, per conservative component in the
    // fixed order [rho, mom_x, mom_y, ee, ei, er] (see RES_COMPONENTS).
    //
    // For a valid cell set {i} with N members and per-cell RHS r_k(i):
    //   R1_k   = (1/N) * sum_i |r_k(i)|          (mean absolute)
    //   R2_k   = sqrt( (1/N) * sum_i r_k(i)^2 )  (RMS)
    //   Rinf_k = max_i |r_k(i)|                  (max absolute)
    // --------------------------------------------------------------
    res_r1: [f64; 6],
    res_r2: [f64; 6],
    res_rinf: [f64; 6],
}

/// Conservative-variable order shared by the residual-norm arrays and the
/// CSV column names written by Monitor. The entries are the exact column
/// prefixes used in the header (e_e / e_i / e_r, matching the internal
/// energy compartments ee / ei / er).
const RES_COMPONENTS: [&str; 6] = ["rho", "mom_x", "mom_y", "e_e", "e_i", "e_r"];

impl Monitor {
    /// CSV header written by the current build. New residual columns are
    /// appended at the END so the position of every pre-existing column is
    /// unchanged (readers that index by name or position keep working).
    fn header_string() -> String {
        concat!(
            "step,time,dt,dt_cfl,dt_over_dt_cfl,",
            "n_fluid,",
            "rho_min,rho_max,",
            "p_min,p_max,",
            "pe_min,pi_min,pr_min,",
            "ee_int_min,ei_int_min,er_int_min,",
            "speed_max,mach_max,",
            "mass,mom_x,mom_y,total_energy,",
            "drho_dt_l2,dmom_x_dt_l2,dmom_y_dt_l2,",
            "dee_dt_l2,dei_dt_l2,der_dt_l2,",
            "tv_rho,tv_p,",
            "s2_rho,s2_p,",
            "rho_R1,rho_R2,rho_Rinf,",
            "mom_x_R1,mom_x_R2,mom_x_Rinf,",
            "mom_y_R1,mom_y_R2,mom_y_Rinf,",
            "e_e_R1,e_e_R2,e_e_Rinf,",
            "e_i_R1,e_i_R2,e_i_Rinf,",
            "e_r_R1,e_r_R2,e_r_Rinf"
        )
        .to_string()
    }

    pub fn new(path: &str, append: bool) -> Self {
        let header = Self::header_string();
        let exists = Path::new(path).exists();

        // Make sure the parent directory exists (data/ may be deleted by
        // io::clear_data_folder() between runs; a restart must still be able
        // to recreate the monitor file).
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).unwrap_or_else(|e| {
                    panic!("cannot create monitor dir {}: {}", parent.display(), e)
                });
            }
        }

        // Appending to a file written by an OLDER build (whose header lacks
        // the residual columns) would produce a row/column count mismatch and
        // corrupt the CSV. In that case fall back to a fresh file instead of
        // silently mislabelling data.
        let mut append = append;
        if append && exists {
            if let Ok(mut f) = File::open(path) {
                let mut first = String::new();
                if BufReader::new(&mut f).read_line(&mut first).is_ok() {
                    let existing = first.trim_end_matches(['\r', '\n']);
                    if existing != header {
                        eprintln!(
                            "monitor: existing {} has a stale header (schema changed); \
                             starting a fresh monitor file",
                            path
                        );
                        append = false;
                    }
                }
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .unwrap_or_else(|e| panic!("cannot open monitor file {}: {}", path, e));

        let mut writer = BufWriter::new(file);

        if !append || !exists {
            writeln!(writer, "{}", header).expect("failed writing monitor header");
        }

        Self {
            writer,
            flush_interval: 20,
        }
    }

    pub fn write_step(
        &mut self,
        step: usize,
        u: &Field,
        u_old: Option<&Field>,
        dt: f64,
        dt_cfl: f64,
        rhs: Option<&[State]>,
    ) {
        let s = compute_stats(u, u_old, dt, rhs);

        let dt_ratio = if dt_cfl > 0.0 { dt / dt_cfl } else { f64::NAN };

        // The 18 residual columns are appended after the legacy fields; they
        // are formatted separately so the original format string stays intact.
        // The main format string already supplies the separating comma before
        // `{}`, so the tail carries the 18 values WITHOUT a leading comma.
        let mut tail_parts: Vec<String> = Vec::with_capacity(6);
        for k in 0..6 {
            tail_parts.push(format!(
                "{:.16e},{:.16e},{:.16e}",
                s.res_r1[k], s.res_r2[k], s.res_rinf[k]
            ));
        }
        let tail = tail_parts.join(",");

        writeln!(
            self.writer,
            concat!(
                "{},{:.16e},{:.16e},{:.16e},{:.16e},",
                "{},",
                "{:.16e},{:.16e},",
                "{:.16e},{:.16e},",
                "{:.16e},{:.16e},{:.16e},",
                "{:.16e},{:.16e},{:.16e},",
                "{:.16e},{:.16e},",
                "{:.16e},{:.16e},{:.16e},{:.16e},",
                "{:.16e},{:.16e},{:.16e},",
                "{:.16e},{:.16e},{:.16e},",
                "{:.16e},{:.16e},",
                "{:.16e},{:.16e},",
                "{}"
            ),
            step,
            u.time,
            dt,
            dt_cfl,
            dt_ratio,
            s.n_fluid,
            s.rho_min,
            s.rho_max,
            s.p_min,
            s.p_max,
            s.pe_min,
            s.pi_min,
            s.pr_min,
            s.ee_int_min,
            s.ei_int_min,
            s.er_int_min,
            s.speed_max,
            s.mach_max,
            s.mass,
            s.mom_x,
            s.mom_y,
            s.energy,
            s.drho_l2,
            s.dmom_x_l2,
            s.dmom_y_l2,
            s.dee_l2,
            s.dei_l2,
            s.der_l2,
            s.tv_rho,
            s.tv_p,
            s.s2_rho,
            s.s2_p,
            tail,
        )
        .expect("failed writing monitor data");

        if step % self.flush_interval == 0 {
            self.writer.flush().expect("failed flushing monitor file");
        }
    }
}

// ============================================================
// Primitive diagnostic helpers
// ============================================================

/// Total thermodynamic pressure, i.e. State::pressure_tot().
///
/// This is exactly the definition used by the solver EOS (each of the
/// three energy compartments contributes (gamma-1) * internal energy,
/// after removing its 1/6 share of kinetic energy).
#[inline(always)]
fn pressure(s: State) -> f64 {
    s.pressure_tot()
}

// ============================================================
// Semi-discrete RHS residual norms
// ============================================================

/// Reduce a per-cell RHS buffer (dU/dt) to the three residual norms of each
/// conservative component.
///
/// `rhs` holds the solver's semi-discrete operator output RHS(U), one `State`
/// per grid cell. Only fluid cells whose RHS is fully finite enter the
/// statistics; cells containing NaN/Inf (which would otherwise pollute the
/// norms) are skipped, mirroring the "valid computational cells" rule of the
/// monitor. Cell count is therefore the number of cells actually summed, not
/// necessarily `n_fluid`.
///
/// Component order is RES_COMPONENTS = [rho, mom_x, mom_y, ee, ei, er].
/// Returns (R1, R2, Rinf) arrays; all NaN when no RHS is supplied.
fn residual_norms(rhs: Option<&[State]>, fluid: &[bool]) -> ([f64; 6], [f64; 6], [f64; 6]) {
    let Some(rhs) = rhs else {
        return ([f64::NAN; 6], [f64::NAN; 6], [f64::NAN; 6]);
    };

    let mut r1 = [0.0f64; 6];
    let mut r2 = [0.0f64; 6];
    let mut rinf = [0.0f64; 6];
    let mut n = 0usize;

    let ncell = rhs.len().min(fluid.len());
    for l in 0..ncell {
        if !fluid[l] {
            continue;
        }
        let s = rhs[l];
        let vals = [s.rho, s.mom_x, s.mom_y, s.ee, s.ei, s.er];
        if !vals.iter().all(|v| v.is_finite()) {
            continue; // never let NaN / inf pollute a residual norm
        }
        n += 1;
        for (k, &v) in vals.iter().enumerate() {
            let a = v.abs();
            r1[k] += a;
            r2[k] += a * a;
            rinf[k] = rinf[k].max(a);
        }
    }

    let inv_n = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    for k in 0..6 {
        r1[k] *= inv_n;
        r2[k] = (r2[k] * inv_n).sqrt();
    }

    (r1, r2, rinf)
}

// ============================================================
// Main diagnostic calculation
// ============================================================

fn compute_stats(field: &Field, old: Option<&Field>, dt: f64, rhs: Option<&[State]>) -> Stats {
    let nx = field.grid.nx;
    let ny = field.grid.ny;
    let dx = field.grid.dx;
    let dy = field.grid.dy;
    let dv = dx * dy;

    // Sound-speed coefficients, identical to State::cs() and dt.rs.
    let ge = constant::GAMMA_E - 1.0;
    let gi = constant::GAMMA_I - 1.0;
    let gr = constant::GAMMA_R - 1.0;

    let mut rho_min = f64::INFINITY;
    let mut rho_max = f64::NEG_INFINITY;

    let mut p_min = f64::INFINITY;
    let mut p_max = f64::NEG_INFINITY;

    let mut pe_min = f64::INFINITY;
    let mut pi_min = f64::INFINITY;
    let mut pr_min = f64::INFINITY;

    let mut ee_int_min = f64::INFINITY;
    let mut ei_int_min = f64::INFINITY;
    let mut er_int_min = f64::INFINITY;

    let mut speed_max: f64 = 0.0;
    let mut mach_max: f64 = 0.0;

    let mut mass = 0.0;
    let mut mom_x = 0.0;
    let mut mom_y = 0.0;
    let mut energy = 0.0;

    // squared temporal derivative norms
    let mut drho2 = 0.0;
    let mut dmx2 = 0.0;
    let mut dmy2 = 0.0;
    let mut dee2 = 0.0;
    let mut dei2 = 0.0;
    let mut der2 = 0.0;

    let mut n_fluid = 0usize;

    // A previous-state buffer is needed for the temporal norm.
    // The SSP-RK3 swap leaves U^n in the u3 buffer, so no extra
    // full-field clone is required by the caller.
    let prev = if old.is_some() && dt.is_finite() && dt > 0.0 {
        old
    } else {
        None
    };

    // --------------------------------------------------------
    // Pointwise / global diagnostics.
    //
    // Single grid traversal: extrema, integrals and the temporal
    // norm are all accumulated here.
    // --------------------------------------------------------

    for l in 0..field.value.len() {
        if !field.fluid[l] {
            continue;
        }

        n_fluid += 1;

        let q = field.value[l];

        let d = Derived::from_state(q);

        let p = d.pe + d.pi + d.pr;

        let speed = (d.u * d.u + d.v * d.v).sqrt();

        let cs = (constant::GAMMA_E * ge * d.e_e
            + constant::GAMMA_I * gi * d.e_i
            + constant::GAMMA_R * gr * d.e_r)
            .sqrt();

        let mach = if cs > 0.0 {
            speed / cs
        } else {
            // cs <= 0 means a non-physical state; report it loudly.
            f64::INFINITY
        };

        rho_min = rho_min.min(q.rho);
        rho_max = rho_max.max(q.rho);

        p_min = p_min.min(p);
        p_max = p_max.max(p);

        pe_min = pe_min.min(d.pe);
        pi_min = pi_min.min(d.pi);
        pr_min = pr_min.min(d.pr);

        ee_int_min = ee_int_min.min(d.e_e);
        ei_int_min = ei_int_min.min(d.e_i);
        er_int_min = er_int_min.min(d.e_r);

        speed_max = speed_max.max(speed);
        mach_max = mach_max.max(mach);

        mass += q.rho * dv;
        mom_x += q.mom_x * dv;
        mom_y += q.mom_y * dv;

        // ee + ei + er contains the full kinetic energy:
        //
        // each component contains rho*(u^2+v^2)/6,
        // so their sum contains rho*(u^2+v^2)/2.
        //
        // Total energy density = internal + kinetic, with each of the
        // three compartments counted exactly once.
        energy += (q.ee + q.ei + q.er) * dv;

        // ----------------------------------------------------
        // Temporal activity
        //
        //   || (U^{n+1} - U^n) / dt ||_RMS
        //
        // Only fluid points enter the RMS. This is a measure of how
        // strongly the solution is evolving, NOT a steady residual.
        // ----------------------------------------------------

        if let Some(prev) = prev {
            let qo = prev.value[l];

            let inv_dt = 1.0 / dt;

            let a = (q.rho - qo.rho) * inv_dt;
            let b = (q.mom_x - qo.mom_x) * inv_dt;
            let c = (q.mom_y - qo.mom_y) * inv_dt;
            let d = (q.ee - qo.ee) * inv_dt;
            let e = (q.ei - qo.ei) * inv_dt;
            let f = (q.er - qo.er) * inv_dt;

            drho2 += a * a;
            dmx2 += b * b;
            dmy2 += c * c;
            dee2 += d * d;
            dei2 += e * e;
            der2 += f * f;
        }
    }

    let inv_n = if n_fluid > 0 {
        1.0 / n_fluid as f64
    } else {
        0.0
    };

    // Without a previous state the norms are NaN, so a reader never
    // mistakes a missing value for a real "zero temporal change".
    let (drho_l2, dmom_x_l2, dmom_y_l2, dee_l2, dei_l2, der_l2) = match prev {
        Some(_) => (
            (drho2 * inv_n).sqrt(),
            (dmx2 * inv_n).sqrt(),
            (dmy2 * inv_n).sqrt(),
            (dee2 * inv_n).sqrt(),
            (dei2 * inv_n).sqrt(),
            (der2 * inv_n).sqrt(),
        ),
        None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
    };

    // ========================================================
    // 2D total variation
    //
    // Integral approximation:
    //
    // TV(q) ~ sum_x |q(i+1,j)-q(i,j)| dy
    //       + sum_y |q(i,j+1)-q(i,j)| dx
    //
    // IMPORTANT:
    // only fluid-fluid edges are counted.
    //
    // This prevents the embedded solid boundary itself from
    // appearing as a huge artificial jump.
    // ========================================================

    let mut tv_rho = 0.0;
    let mut tv_p = 0.0;

    for i in 0..nx {
        for j in 0..ny {
            let l = i * ny + j;

            if !field.fluid[l] {
                continue;
            }

            let q = field.value[l];
            let p0 = pressure(q);

            // x neighbor
            if i + 1 < nx {
                let r = (i + 1) * ny + j;

                if field.fluid[r] {
                    let qr = field.value[r];

                    tv_rho += (qr.rho - q.rho).abs() * dy;
                    tv_p += (pressure(qr) - p0).abs() * dy;
                }
            }

            // y neighbor
            if j + 1 < ny {
                let r = i * ny + (j + 1);

                if field.fluid[r] {
                    let qr = field.value[r];

                    tv_rho += (qr.rho - q.rho).abs() * dx;
                    tv_p += (pressure(qr) - p0).abs() * dx;
                }
            }
        }
    }

    // ========================================================
    // Second-difference oscillation indicator
    //
    // S2 = sum |q_{i+1} - 2 q_i + q_{i-1}|
    //
    // Computed only when all three Cartesian points are fluid.
    //
    // This is intentionally NOT divided by dx^2.
    // We want a grid-scale wiggle indicator rather than an
    // approximation to the physical second derivative.
    // ========================================================

    let mut s2_rho = 0.0;
    let mut s2_p = 0.0;

    // x direction
    for i in 1..nx - 1 {
        for j in 0..ny {
            let lm = (i - 1) * ny + j;
            let l0 = i * ny + j;
            let lp = (i + 1) * ny + j;

            if !(field.fluid[lm] && field.fluid[l0] && field.fluid[lp]) {
                continue;
            }

            let qm = field.value[lm];
            let q0 = field.value[l0];
            let qp = field.value[lp];

            s2_rho += (qp.rho - 2.0 * q0.rho + qm.rho).abs();

            s2_p += (pressure(qp) - 2.0 * pressure(q0) + pressure(qm)).abs();
        }
    }

    // y direction
    for i in 0..nx {
        for j in 1..ny - 1 {
            let lm = i * ny + (j - 1);
            let l0 = i * ny + j;
            let lp = i * ny + (j + 1);

            if !(field.fluid[lm] && field.fluid[l0] && field.fluid[lp]) {
                continue;
            }

            let qm = field.value[lm];
            let q0 = field.value[l0];
            let qp = field.value[lp];

            s2_rho += (qp.rho - 2.0 * q0.rho + qm.rho).abs();

            s2_p += (pressure(qp) - 2.0 * pressure(q0) + pressure(qm)).abs();
        }
    }

    // Semi-discrete RHS residual norms over the valid (fluid, finite) cells.
    let (res_r1, res_r2, res_rinf) = residual_norms(rhs, &field.fluid);

    Stats {
        n_fluid,

        rho_min,
        rho_max,

        p_min,
        p_max,

        pe_min,
        pi_min,
        pr_min,

        ee_int_min,
        ei_int_min,
        er_int_min,

        speed_max,
        mach_max,

        mass,
        mom_x,
        mom_y,
        energy,

        drho_l2,
        dmom_x_l2,
        dmom_y_l2,
        dee_l2,
        dei_l2,
        der_l2,

        tv_rho,
        tv_p,

        s2_rho,
        s2_p,

        res_r1,
        res_r2,
        res_rinf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_norms_formulas() {
        // Two valid fluid cells; a third cell is solid and must be excluded.
        let fluid = vec![true, true, false];
        let mut rhs = vec![State::new(); 3];
        rhs[0].rho = 2.0;
        rhs[0].mom_x = 1.0;
        rhs[1].rho = -4.0;
        rhs[1].mom_x = 2.0;

        let (r1, r2, rinf) = residual_norms(Some(&rhs), &fluid);

        // rho: values 2, -4  -> |r|: 2, 4
        let tol = 1e-12;
        assert!((r1[0] - 3.0).abs() < tol, "R1 rho = {}", r1[0]);
        assert!((r2[0] - (10.0f64).sqrt()).abs() < tol, "R2 rho = {}", r2[0]);
        assert!((rinf[0] - 4.0).abs() < tol, "Rinf rho = {}", rinf[0]);

        // mom_x: values 1, 2
        assert!((r1[1] - 1.5).abs() < tol, "R1 mom_x = {}", r1[1]);
        assert!((r2[1] - 2.5f64.sqrt()).abs() < tol, "R2 mom_x = {}", r2[1]);
        assert!((rinf[1] - 2.0).abs() < tol, "Rinf mom_x = {}", rinf[1]);

        // Cells never written keep zero norms.
        assert!(r1[2] == 0.0 && r2[2] == 0.0 && rinf[2] == 0.0);
    }

    #[test]
    fn residual_norms_skip_nonfinite_cells() {
        // A NaN RHS in an otherwise fluid cell must not poison the norms; the
        // offending cell is simply not counted.
        let fluid = vec![true, true, true];
        let mut rhs = vec![State::new(); 3];
        rhs[0].rho = 1.0;
        rhs[1].rho = f64::NAN;
        rhs[2].rho = 3.0;

        let (r1, r2, rinf) = residual_norms(Some(&rhs), &fluid);

        let tol = 1e-12;
        assert!((r1[0] - 2.0).abs() < tol, "R1 rho = {}", r1[0]);
        assert!((r2[0] - 5.0f64.sqrt()).abs() < tol, "R2 rho = {}", r2[0]);
        assert!((rinf[0] - 3.0).abs() < tol, "Rinf rho = {}", rinf[0]);
        assert!(r1[0].is_finite() && r2[0].is_finite() && rinf[0].is_finite());
    }

    #[test]
    fn no_rhs_yields_nan_norms() {
        let fluid = vec![true];
        let (r1, r2, rinf) = residual_norms(None, &fluid);
        assert!(r1.iter().all(|v| v.is_nan()));
        assert!(r2.iter().all(|v| v.is_nan()));
        assert!(rinf.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn header_has_residual_columns_in_component_major_order() {
        let header = Monitor::header_string();

        let tail: Vec<String> = RES_COMPONENTS
            .iter()
            .flat_map(|c| {
                ["R1", "R2", "Rinf"]
                    .iter()
                    .map(move |n| format!("{}_{}", c, n))
            })
            .collect();
        let tail_csv = format!(",{}", tail.join(","));

        assert!(
            header.ends_with(&tail_csv),
            "header must end with the residual columns, got ...{}",
            &header[header.len().saturating_sub(120)..]
        );
    }
}
