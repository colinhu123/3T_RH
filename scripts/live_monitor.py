#!/usr/bin/env python3
"""
scripts/live_monitor.py — 3T-RH Simulation Dashboard

Single-window dashboard that combines the Rust CFD solver's

  * solution snapshots  (data/solution_*.bin, binary format from src/io.rs)
  * live diagnostics    (data/monitor.csv, written by src/monitor.rs)

into ONE matplotlib window with page switching:

  1 Solution | 2 Health | 3 Activity | 4 Oscillation | 5 Global

Pages are switched with the keys 1..5 (the active page is shown in the
compact header title; there are no large Button widgets). Solution snapshot
navigation (Left/Right/Up/Down/Home/End), live-follow (Space) and the
current-snapshot time marker on the monitor pages are all handled here.

Architecture (Python and Rust remain fully decoupled):
  - pure readers only: never locks/truncates/renames the CSV or the *.bin
  - binary reading reuses py_utils/solution_io.py (same reader as the
    standalone visualize_sol.py)
  - a single figure; page axes are created on page entry and removed on
    switch, so no unbounded growth of axes/colorbars/artists

Keys:
  1-5     switch page
  Left/Right  previous/next snapshot
  Up/Down     +/-10 snapshots
  Home        first snapshot          (disables live follow)
  End         latest snapshot         (re-enables live follow)
  Space       toggle LIVE FOLLOW
  R           force refresh
  Q           quit
"""

import argparse
import io
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reliable repo-root based defaults (see "working directory" notes): the
# dashboard lives in scripts/ but the data lives in the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from py_utils import solution_io  # noqa: E402

# EOS constants (must match src/constant.rs) used to derive pressure and Mach.
GAMMA_E = 1.4
GAMMA_I = 1.4
GAMMA_R = 1.4

PAGE_TITLES = [
    "Solution",
    "Physical Health",
    "Activity / Residual",
    "Oscillation",
    "Global Quantities",
]

SOLUTION_VARIABLES = [
    "rho",
    "u",
    "v",
    "speed",
    "mom_x",
    "mom_y",
    "ee",
    "ei",
    "er",
    "pressure",
    "mach",
    "vorticity",
]

# Display labels for the solution page title / colorbar.
VARIABLE_TITLES = {
    "vorticity": r"Vorticity $\omega_z$",
}
VARIABLE_COLORBAR = {
    "vorticity": r"$\omega_z$",
}

# ============================================================
# RHS residual configuration
#
# The Rust solver (src/monitor.rs) writes, for each conservative component,
# three residual norms of the semi-discrete RHS dU/dt = RHS(U):
#   R1_k   = (1/N) sum_i |r_k(i)|          (mean absolute)
#   R2_k   = sqrt((1/N) sum_i r_k(i)^2)    (RMS)
#   Rinf_k = max_i |r_k(i)|                (max absolute)
# Column layout:  <component>_R1/_R2/_Rinf  (e.g. rho_R1, mom_x_R2, e_r_Rinf).
# Normalized columns <component>_R<n>_norm = R / R_ref are derived at read
# time with R_ref = the first valid value of that column (see _derived_frame).
# ============================================================

RESIDUAL_COMPONENTS = ["rho", "mom_x", "mom_y", "e_e", "e_i", "e_r"]
RESIDUAL_NORMS = ["R1", "R2", "Rinf"]
RESIDUAL_NORM_DISPLAY = {"R1": "R1", "R2": "R2 (RMS)", "Rinf": "R_inf"}

# Reference-residual guard: if the reference residual is <= this the
# normalized residual is meaningless (division by ~0) and is pinned to 0.
# 1e-12 mirrors constant::DEFAULT_EPS, the tolerance used throughout the Rust
# solver for double-precision smoothness/round-off checks.
RESIDUAL_NORM_EPS = 1e-12


def residual_col(component, norm):
    """Raw CSV column holding residual norm ``norm`` of ``component``."""
    return "{}_{}".format(component, norm)


def residual_norm_col(component, norm):
    """Derived column holding the normalized residual (R / R_ref)."""
    return "{}_{}_norm".format(component, norm)


# (radio label, norm, normalized?) choices for the Activity page residual view.
RESIDUAL_MODES = [
    ("R2  normalized", "R2", True),
    ("R2  raw", "R2", False),
    ("R1  normalized", "R1", True),
    ("R1  raw", "R1", False),
    ("R_inf  normalized", "Rinf", True),
    ("R_inf  raw", "Rinf", False),
]

DEFAULT_RESIDUAL_MODE = 0


def _log(msg):
    """Print a status line, always flushed (survives kill / long idle runs)."""
    print(msg, flush=True)


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "3T-RH Simulation Dashboard: single-window solution + diagnostics "
            "monitor (read-only, does not touch the solver)."
        )
    )
    p.add_argument(
        "--solution-dir",
        default=str(REPO_ROOT / "data"),
        help=f"directory with solution_*.bin files (default: {REPO_ROOT / 'data'})",
    )
    p.add_argument(
        "--file",
        default=str(REPO_ROOT / "data" / "monitor.csv"),
        help="path to the solver diagnostics CSV (default: <repo>/data/monitor.csv)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="refresh interval in seconds (default: 2.0)",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="maximum points per monitor line after stride downsampling "
             "(default: 5000)",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        help="directory to periodically write a dashboard.png snapshot "
             "(overwritten each refresh)",
    )
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="headless mode: no GUI window; uses the Agg backend and writes "
             "dashboard.png to --save-dir (auto-provided if omitted)",
    )
    p.add_argument("--gamma-e", type=float, default=GAMMA_E,
                   help="electron gamma, matches src/constant.rs (default 1.4)")
    p.add_argument("--gamma-i", type=float, default=GAMMA_I,
                   help="ion gamma, matches src/constant.rs (default 1.4)")
    p.add_argument("--gamma-r", type=float, default=GAMMA_R,
                   help="radiation gamma, matches src/constant.rs (default 1.4)")
    args = p.parse_args()

    if args.interval <= 0.0:
        p.error("--interval must be positive")
    if args.max_points < 100:
        p.error("--max-points must be at least 100")

    if args.no_gui and args.save_dir is None:
        args.save_dir = str(REPO_ROOT / "data" / "monitor_plots")
        _log("--no-gui without --save-dir: snapshots go to "
             f"{REPO_ROOT / 'data' / 'monitor_plots'}")

    return args


# ============================================================
# Robust CSV reading (same robustness as the previous live_monitor.py)
# ============================================================

def _row_fully_numeric(row):
    """True if every comma-separated field of ``row`` parses as a float."""
    for tok in row.split(","):
        try:
            float(tok)
        except ValueError:
            return False
    return True


def _clean_lines(lines):
    """Tolerate a CSV file that is being written right now.

    Returns cleaned CSV text or None. Incomplete trailing rows (mid-write by
    the Rust solver) are dropped; a header-only file yields just the header.
    """
    rows = [ln.rstrip("\r\n") for ln in lines]
    rows = [ln for ln in rows if ln.strip()]

    if not rows:
        return None

    header = rows[0]
    nfields = header.count(",") + 1
    if nfields < 2:
        return None

    body = [ln for ln in rows[1:] if ln.count(",") + 1 == nfields]

    if body and not _row_fully_numeric(body[-1]):
        body = body[:-1]

    if not body:
        return header + "\n"

    return "\n".join([header] + body)


def safe_read_csv(path):
    """Read the CSV into a pandas DataFrame (None when not ready yet).

    Returns None for missing/empty/mid-write files, an empty DataFrame when
    only a header is present. Never raises for malformed/incomplete input.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    if st.st_size == 0:
        return None

    lines = None
    for _ in range(3):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            break
        except OSError:
            time.sleep(0.05)

    if not lines:
        return None

    cleaned = _clean_lines(lines)
    if cleaned is None:
        return None

    try:
        return pd.read_csv(io.StringIO(cleaned))
    except Exception:
        return None


def sanitize_series(values):
    """Return a float array where +Inf / -Inf are replaced by NaN."""
    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def downsample_dataframe(df, max_points):
    """Stride-downsample the whole history; the latest point is always kept."""
    n = len(df)
    if n <= max_points:
        return df
    stride = math.ceil(n / max_points)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return df.iloc[idx]


# ============================================================
# Vorticity (embedded-boundary safe derivatives)
# ============================================================

def _masked_gradient(f, valid, coord, axis):
    """Masked first derivative of ``f`` along ``axis`` using physical coords.

    Parameters
    ----------
    f : (ny, nx) float array
        Field values; non-fluid entries may be anything (they are masked out
        by ``valid`` and never used in the stencil).
    valid : (ny, nx) bool array
        True for fluid points whose value is usable.
    coord : 1-D float array
        Physical coordinate vector along ``axis`` (x for axis=1, y for axis=0).
    axis : int
        0 -> d/dy, 1 -> d/dx.

    Strategy (conservative, embedded-geometry safe):
      * centered 2nd-order difference where both neighbours are valid fluid;
      * one-sided 1st-order difference where exactly one adjacent neighbour
        is valid and the centre point is valid;
      * NaN where no derivative can be formed from valid fluid data.

    Solid / NaN cells are never replaced by 0, so an embedded obstacle never
    manufactures a fake vortex layer at its boundary.
    """
    n = f.shape[axis]          # number of points along the derivative axis
    m = f.shape[1 - axis]      # number of points along the other axis
    out = np.full(f.shape, np.nan)

    for k in range(n):
        # slice views of the centre point and its two neighbours
        if axis == 0:                       # d/dy along rows
            cur, cvalid = f[k, :], valid[k, :]
            km1 = f[k - 1, :] if k - 1 >= 0 else None
            vkm1 = valid[k - 1, :] if k - 1 >= 0 else None
            kp1 = f[k + 1, :] if k + 1 < n else None
            vkp1 = valid[k + 1, :] if k + 1 < n else None
        else:                               # d/dx along columns
            cur, cvalid = f[:, k], valid[:, k]
            km1 = f[:, k - 1] if k - 1 >= 0 else None
            vkm1 = valid[:, k - 1] if k - 1 >= 0 else None
            kp1 = f[:, k + 1] if k + 1 < n else None
            vkp1 = valid[:, k + 1] if k + 1 < n else None

        outk = np.full(m, np.nan)

        if km1 is not None and kp1 is not None:
            cent = vkm1 & vkp1
            outk[cent] = (kp1[cent] - km1[cent]) / (coord[k + 1] - coord[k - 1])
            # one-sided fallback where centred is impossible
            if not cent.all():
                back = vkm1 & cvalid & ~cent
                outk[back] = (cur[back] - km1[back]) / (coord[k] - coord[k - 1])
                fwd = vkp1 & cvalid & ~cent
                outk[fwd] = (kp1[fwd] - cur[fwd]) / (coord[k + 1] - coord[k])
        elif km1 is not None:
            back = vkm1 & cvalid
            outk[back] = (cur[back] - km1[back]) / (coord[k] - coord[k - 1])
        elif kp1 is not None:
            fwd = vkp1 & cvalid
            outk[fwd] = (kp1[fwd] - cur[fwd]) / (coord[k + 1] - coord[k])

        if axis == 0:
            out[k, :] = outk
        else:
            out[:, k] = outk

    return out


def compute_vorticity(x, y, rho, mom_x, mom_y):
    """2D out-of-plane vorticity omega_z = dv/dx - du/dy.

    Inputs are the raw masked solution fields (masked outside the fluid). Only
    fluid points with a valid derivative stencil contribute; solid / NaN cells
    are never treated as zero. The result is a masked array with the same
    shape as the input fields.
    """
    rho_a = np.ma.getdata(rho)
    mx_a = np.ma.getdata(mom_x)
    my_a = np.ma.getdata(mom_y)

    with np.errstate(divide="ignore", invalid="ignore"):
        u = mx_a / rho_a
        v = my_a / rho_a

    valid = (
        (~np.ma.getmaskarray(rho))
        & (~np.ma.getmaskarray(mom_x))
        & (~np.ma.getmaskarray(mom_y))
        & np.isfinite(u)
        & np.isfinite(v)
        & (rho_a > 0.0)
    )
    u = np.where(valid, u, np.nan)
    v = np.where(valid, v, np.nan)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dv_dx = _masked_gradient(v, valid, x, axis=1)
    du_dy = _masked_gradient(u, valid, y, axis=0)

    return np.ma.masked_invalid(dv_dx - du_dy)


def _symmetric_vorticity_limits(vals):
    """Symmetric color limits (-L, +L) centered on 0 for signed vorticity.

    Uses the 99th percentile of |omega| over valid fluid values so a few
    extreme numerical spikes do not destroy the contrast; falls back to the
    max |value| when the percentile is unusable. Returns None when there is no
    usable data.
    """
    data = np.ma.getdata(vals)
    mask = np.ma.getmaskarray(vals)
    finite = data[(~mask) & np.isfinite(data)]
    if finite.size == 0:
        return None
    lim = float(np.nanpercentile(np.abs(finite), 99.0))
    if not np.isfinite(lim) or lim <= 0.0:
        lim = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(lim) or lim <= 0.0:
        return None
    return (-lim, lim)


# ============================================================
# Simulation data (I/O, no plotting)
# ============================================================

class SimulationData:
    """Owns solution-file discovery/reading and monitor.csv reading."""

    def __init__(self, solution_dir, monitor_path, max_points, gammas):
        self.solution_dir = Path(solution_dir)
        self.monitor_path = Path(monitor_path)
        self.max_points = max_points
        self.gamma_e, self.gamma_i, self.gamma_r = gammas

        # solution state
        self.solution_files = []
        self.current_frame = 0
        self.read_frame = None       # index of the last successfully read snapshot
        self.sol_time = None
        self.x_grid = None
        self.y_grid = None
        self.field = None            # raw conservative field (masked arrays)
        self.variables = {}          # name -> 2D masked array (incl. derived)
        self.sol_read_error = False

        # monitor state
        self.monitor_df = None
        self.plot_df = None
        self.xcol = None
        self.csv_columns = None
        self.csv_state = "NONE"      # NONE / HEADER / OK / RETRY
        self.csv_reset_detected = False
        self.prev_row_count = 0
        self.prev_max_x = None
        self._last_csv_stat = None
        self.warned_cols = set()

    # ---------------- solution ----------------

    def refresh_solution_files(self):
        self.solution_files = solution_io.refresh_solution_files(self.solution_dir)

    def clamp_frame(self):
        n = len(self.solution_files)
        if n == 0:
            self.current_frame = 0
            return
        self.current_frame = max(0, min(self.current_frame, n - 1))

    def read_solution(self, index):
        """Read snapshot ``index``; returns True on success, False on error."""
        try:
            x, y, time, field = solution_io.read_solution_file(
                self.solution_files[index]
            )
        except Exception as exc:
            if not self.sol_read_error:
                _log(f"warning: cannot read {self.solution_files[index].name}: {exc}")
                self.sol_read_error = True
            return False

        self.sol_read_error = False
        self.read_frame = index
        self.sol_time = time
        self.x_grid = x
        self.y_grid = y
        self.field = field
        self.variables = self._build_variables(field, x, y)
        return True

    def _build_variables(self, field, x, y):
        """Derived solution fields.

        u = mom_x/rho, v = mom_y/rho, |u| = sqrt(u^2+v^2).
        pressure and Mach follow the solver EOS exactly (src/state.rs):
          e_k = ee/rho - (u^2+v^2)/6, ...
          p   = (GAMMA_E-1)*(ee - rho*(u^2+v^2)/6) + ... + ...
          cs  = sqrt(GAMMA_E*(GAMMA_E-1)*e_e + GAMMA_I*(GAMMA_I-1)*e_i
                     + GAMMA_R*(GAMMA_R-1)*e_r)
        Zero/NaN/outside-domain cells are masked and never displayed.
        """
        ge, gi, gr = self.gamma_e, self.gamma_i, self.gamma_r
        rho = field["rho"]

        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.ma.masked_invalid(field["mom_x"] / rho)
            v = np.ma.masked_invalid(field["mom_y"] / rho)
            w2 = u ** 2 + v ** 2
            speed = np.ma.masked_invalid(np.sqrt(w2))

            e_e = field["ee"] / rho - w2 / 6.0
            e_i = field["ei"] / rho - w2 / 6.0
            e_r = field["er"] / rho - w2 / 6.0

            pe = (ge - 1.0) * (field["ee"] - rho * w2 / 6.0)
            pi = (gi - 1.0) * (field["ei"] - rho * w2 / 6.0)
            pr = (gr - 1.0) * (field["er"] - rho * w2 / 6.0)
            p = np.ma.masked_invalid(pe + pi + pr)

            cs = np.sqrt(ge * (ge - 1.0) * e_e
                         + gi * (gi - 1.0) * e_i
                         + gr * (gr - 1.0) * e_r)
            mach = np.ma.masked_invalid(speed / cs)

        return {
            "rho": rho,
            "mom_x": field["mom_x"],
            "mom_y": field["mom_y"],
            "ee": field["ee"],
            "ei": field["ei"],
            "er": field["er"],
            "u": u,
            "v": v,
            "speed": speed,
            "pressure": p,
            "mach": mach,
            # omega_z = dv/dx - du/dy, recomputed per snapshot (read_solution
            # only runs when the frame changes, so monitor-only updates never
            # re-trigger this O(Nx*Ny) calculation).
            "vorticity": compute_vorticity(
                x, y, field["rho"], field["mom_x"], field["mom_y"],
            ),
        }

    # ---------------- monitor ----------------

    def _read_csv(self):
        """Return ``(df, changed)``.

        ``df`` is None when there is no usable data this cycle. When the file
        is unchanged since the last successful read, the cached DataFrame is
        returned with ``changed=False`` (state must not be degraded just
        because nothing new arrived).
        """
        try:
            st = os.stat(self.monitor_path)
        except OSError:
            return None, True
        key = (st.st_size, st.st_mtime_ns)
        if self._last_csv_stat == key:
            return self.monitor_df, False
        df = safe_read_csv(self.monitor_path)
        if df is not None:
            self._last_csv_stat = key
            return df, True
        return None, True

    def _choose_xcol(self, df):
        if "time" in df.columns:
            return "time"
        if "step" in df.columns:
            return "step"
        return df.columns[0]

    def _detect_reset(self, df, cols_changed):
        if self.csv_columns is None:
            return False
        if cols_changed:
            return True
        n = len(df)
        if self.prev_row_count and n < self.prev_row_count * 0.5:
            return True
        if self.prev_max_x is not None and self.xcol is not None and self.xcol in df.columns:
            xs = pd.to_numeric(df[self.xcol], errors="coerce")
            if not xs.dropna().empty:
                mx = float(xs.max())
                if mx < self.prev_max_x * 0.5:
                    return True
        return False

    def _derived_frame(self, df):
        """Add derived columns to a monitor history frame.

        * ``<base>_rel`` relative change of the global integrals, baseline Q0 =
          first valid value (guarded like the old monitor).
        * ``<comp>_R<n>_norm`` normalized RHS residual
          ``R(t) / R(t_ref)`` with ``t_ref`` = the first valid data point of
          that column, exactly the task definition. Guarding: if the reference
          is 0 / non-finite (within RESIDUAL_NORM_EPS) the ratio is
          meaningless, so the normalized value is pinned to 0 instead of
          dividing by zero.
        """
        out = df.copy()
        for base in ("mass", "mom_x", "total_energy"):
            if base not in out.columns:
                continue
            series = pd.to_numeric(out[base], errors="coerce")
            finite = series.dropna()
            if finite.empty:
                continue
            q0 = float(finite.iloc[0])
            if not np.isfinite(q0):
                continue
            if abs(q0) > 1e-30:
                out[base + "_rel"] = (series - q0) / abs(q0)
            else:
                out[base + "_rel"] = series - q0

        # Per-component, per-norm normalized RHS residual: each component uses
        # its OWN reference value (rho / momentum / energy differ in scale, so
        # no single global reference is used).
        for comp in RESIDUAL_COMPONENTS:
            for norm in RESIDUAL_NORMS:
                col = residual_col(comp, norm)
                if col not in out.columns:
                    continue
                series = pd.to_numeric(out[col], errors="coerce")
                finite = series.dropna()
                if finite.empty:
                    continue
                ref = float(finite.iloc[0])
                if np.isfinite(ref) and ref > RESIDUAL_NORM_EPS:
                    out[col + "_norm"] = series / ref
                else:
                    # Zero / near-zero reference: division by R_ref is
                    # undefined, so report a flat 0 (never inf/NaN pollution).
                    out[col + "_norm"] = 0.0
        return out

    def refresh_monitor(self):
        """Re-read monitor.csv if changed; update state. Returns changed flag."""
        df, changed = self._read_csv()
        if df is None:
            if changed:
                self.csv_state = (
                    "RETRY" if os.path.exists(self.monitor_path) else "NONE"
                )
            return False

        if len(df) == 0:
            self.csv_state = "HEADER"
            self.csv_columns = list(df.columns)
            self.monitor_df = df
            self.plot_df = None
            return False

        if not changed:
            # identical to what we already processed; keep the current state
            return False

        cols_changed = self.csv_columns is not None and list(df.columns) != self.csv_columns
        if self._detect_reset(df, cols_changed):
            self.csv_reset_detected = True

        self.csv_columns = list(df.columns)
        self.csv_state = "OK"
        self.monitor_df = df
        if self.xcol is None:
            self.xcol = self._choose_xcol(df)
        self.prev_row_count = len(df)
        if self.xcol in df.columns:
            xs = pd.to_numeric(df[self.xcol], errors="coerce")
            if not xs.dropna().empty:
                self.prev_max_x = float(xs.max())
        self.plot_df = downsample_dataframe(
            self._derived_frame(df), self.max_points
        )
        return True


# ============================================================
# Plotting helpers
# ============================================================

class Panel:
    """One axes with a set of lines, autoscaling, optional y=0 line and an
    optional current-snapshot vertical marker. Columns absent from the CSV are
    skipped (one-time warning).

    ``log_floor`` (residual pages): before plotting/autoscaling, values <= 0
    are turned into NaN so the logarithmic scale is never forced to linear by
    a handful of exactly-zero points (e.g. a fully converged residual).
    """

    def __init__(self, ax, columns, present, warn, ylabel, xlabel,
                 log_ok=False, ref_zero=False, marker=False, log_floor=False,
                 legend_title=None):
        self.ax = ax
        self.lines = []
        self.log_ok = log_ok
        self.ref_zero = ref_zero
        self.log_floor = log_floor
        # full set of columns that may ever be plotted on this panel; used by
        # reconfigure() so it never re-asks for missing-column warnings.
        self.available = set(present)
        self.columns = list(columns)
        self.legend_title = legend_title

        for col, label in columns:
            if col not in self.available:
                if warn is not None:
                    warn(col)
                continue
            line, = ax.plot([], [], label=label, lw=1.2)
            self.lines.append((col, line))

        if ref_zero:
            ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5, zorder=0)

        if self.lines:
            ax.legend(loc="best", fontsize="small", ncol=2,
                      title=legend_title, title_fontsize="small")

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

        self.marker = None
        if marker:
            self.marker = ax.axvline(0.0, color="tab:red", lw=1.2, ls="--",
                                     alpha=0.85, zorder=5)

    def _plottable(self, values):
        """Apply the log-floor mask when the panel plots on a log scale."""
        v = np.asarray(values, dtype=float)
        if self.log_floor:
            v = np.where(np.isfinite(v) & (v > 0.0), v, np.nan)
        return v

    def reconfigure(self, columns, ylabel, title=None, legend_title=None):
        """Swap the plotted series without rebuilding the axes.

        Used by the Activity-page residual radio to switch norm / raw /
        normalized on the fly. Old artists and legend are removed first.
        """
        for _, line in self.lines:
            try:
                line.remove()
            except Exception:
                pass
        self.lines = []
        self.columns = list(columns)
        self.legend_title = legend_title

        leg = self.ax.get_legend()
        if leg is not None:
            leg.remove()

        for col, label in columns:
            if col in self.available:
                line, = self.ax.plot([], [], label=label, lw=1.2)
                self.lines.append((col, line))

        if self.lines:
            self.ax.legend(loc="best", fontsize="small", ncol=2,
                           title=legend_title, title_fontsize="small")

        self.ax.set_ylabel(ylabel)
        if title is not None:
            self.ax.set_title(title, fontsize=10)

    def update(self, df, xcol, marker_time=None):
        if df is None or len(df) == 0:
            for _, line in self.lines:
                line.set_data([], [])
            if self.marker is not None:
                self.marker.set_visible(False)
            return

        if xcol in df.columns:
            xs = sanitize_series(df[xcol])
        else:
            xs = np.arange(len(df), dtype=float)

        for col, line in self.lines:
            if col in df.columns:
                line.set_data(xs, self._plottable(sanitize_series(df[col])))
            else:
                line.set_data([], [])

        self._autoscale(df, xs, marker_time)

    def _autoscale(self, df, xs, marker_time):
        xs = np.asarray(xs, dtype=float)
        xf = xs[np.isfinite(xs)]
        if xf.size == 0:
            return

        xmin, xmax = float(xf[0]), float(xf[-1])
        if self.marker is not None and marker_time is not None:
            self.marker.set_xdata([marker_time, marker_time])
            self.marker.set_visible(True)
            xmin = min(xmin, marker_time)
            xmax = max(xmax, marker_time)
        elif self.marker is not None:
            self.marker.set_visible(False)

        if not (xmax > xmin):
            # single data point (e.g. just after a CSV reset): matplotlib warns
            # on identical x-limits, so give the panel a small extent.
            xmin -= 0.5
            xmax += 0.5
        self.ax.set_xlim(xmin, xmax)

        yvals = []
        for col, _ in self.lines:
            if col not in df.columns:
                continue
            yvals.append(self._plottable(sanitize_series(df[col])))
        if not yvals:
            return
        ys = np.concatenate([v for v in yvals if v.size])
        ys = ys[np.isfinite(ys)]
        if ys.size == 0:
            return

        ymin, ymax = float(np.min(ys)), float(np.max(ys))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return

        if self.log_ok and ymin > 0.0:
            self.ax.set_yscale("log")
            if ymax > ymin:
                self.ax.set_ylim(ymin * 0.8, ymax * 1.25)
            else:
                self.ax.set_ylim(ymin * 0.5, ymin * 2.0)
        else:
            self.ax.set_yscale("linear")
            if not (ymax > ymin):
                pad = max(abs(ymax) * 0.05, 1e-300)
                ymin -= pad
                ymax += pad
            span = ymax - ymin
            pad = max(span * 0.1, abs(ymax) * 1e-3, 1e-300)
            if self.ref_zero:
                ymin = min(ymin, 0.0)
            self.ax.set_ylim(ymin - pad, ymax + pad)


# ============================================================
# Dashboard (UI + rendering)
# ============================================================

class Dashboard:
    """One figure, page switching, live refresh, keyboard/button control."""

    def __init__(self, args, plt):
        self.args = args
        self.plt = plt

        # Imported after the backend is selected (matplotlib.widgets imports
        # pyplot internally, which would otherwise pick the wrong backend).
        from matplotlib.widgets import RadioButtons
        self.RadioButtons = RadioButtons

        self.data = SimulationData(
            args.solution_dir, args.file, args.max_points,
            (args.gamma_e, args.gamma_i, args.gamma_r),
        )

        self.current_page = 0
        self.live_follow = True
        self.selected_variable = "rho"

        # Activity-page residual view: which norm (R1/R2/R_inf) and whether
        # the raw or normalized residual is displayed.
        self._res_mode_index = DEFAULT_RESIDUAL_MODE
        self._res_panel = None

        self.fig = None
        self.page_area = None
        self.title_text = None
        self.nav_text = None
        self.status_text = None

        # per-page artists / widgets (cleared on page switch)
        self._page_axes = []
        self._panels = []
        self._radio = None
        self._cbar = None
        self._mesh = None
        self._sol_ax = None
        self._drawn_sol_key = None
        self._sol_failed = False
        self._built_csv_schema = None

        self._timer = None
        self._closed = False
        self.last_save_key = None

        self._build_ui()

    # --------------------------------------------------------
    # figure construction
    # --------------------------------------------------------

    def _build_ui(self):
        plt = self.plt
        # Larger default window; the page plotting area is the dominant part
        # of the figure (no large Button widgets anymore).
        fig = plt.figure(figsize=(13, 8))

        gs = fig.add_gridspec(
            3, 1,
            height_ratios=[0.16, 1.0, 0.17],
            hspace=0.06,
            left=0.045, right=0.995, top=0.985, bottom=0.015,
        )

        # top: compact title + page-navigation hint (plain text, no Buttons)
        head_ax = fig.add_subplot(gs[0, 0])
        head_ax.axis("off")
        self.title_text = head_ax.text(
            0.0, 0.72, "", va="center", ha="left", fontsize=13,
            fontweight="bold", transform=head_ax.transAxes,
        )
        self.nav_text = head_ax.text(
            0.0, 0.15,
            "1 Solution  |  2 Health  |  3 Activity  |  4 Oscillation  |  5 Global",
            va="center", ha="left", fontsize=10, transform=head_ax.transAxes,
        )

        # middle: page plotting area (dominates the window)
        self.page_area = gs[1]

        # bottom: status bar + keyboard help (kept thin)
        sgs = gs[2].subgridspec(2, 1, hspace=0.0)
        status_ax = fig.add_subplot(sgs[0, 0])
        status_ax.axis("off")
        self.status_text = status_ax.text(
            0.01, 0.5, "", va="center", ha="left", fontsize=10,
            transform=status_ax.transAxes,
        )
        help_ax = fig.add_subplot(sgs[1, 0])
        help_ax.axis("off")
        help_ax.text(
            0.01, 0.5,
            "1-5 page | \u2190\u2192 frame | \u2191\u2193 \u00b110 | End live | "
            "Space pause/live | R refresh | Q quit",
            va="center", ha="left", fontsize=9, transform=help_ax.transAxes,
        )

        self.fig = fig
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        fig.canvas.mpl_connect("close_event", self._on_close)

        self.show_page(0)
        self._update_status()
        fig.canvas.draw_idle()

    # --------------------------------------------------------
    # page switching
    # --------------------------------------------------------

    def _clear_page(self):
        for ax in self._page_axes:
            try:
                ax.remove()
            except Exception:
                pass
        self._page_axes = []
        self._panels = []

        if self._radio is not None:
            try:
                self._radio.disconnect_events()
            except Exception:
                pass
            self._radio = None

        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self._mesh = None
        self._sol_ax = None
        self._res_panel = None

    def _set_title(self, title):
        self.title_text.set_text(f"3T-RH Simulation Dashboard  \u2014  {title}")

    def show_page(self, page):
        if not (0 <= page < len(PAGE_TITLES)):
            return
        if page == self.current_page and self._page_axes:
            return
        self.current_page = page
        self._build_page(page)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _build_page(self, page):
        self._clear_page()
        self._set_title(PAGE_TITLES[page])
        if page == 0:
            self._build_solution_page()
        else:
            self._build_monitor_page(page)
        self._built_csv_schema = (
            tuple(self.data.csv_columns) if self.data.csv_columns else None
        )

    # --------------------------------------------------------
    # solution page
    # --------------------------------------------------------

    def _build_solution_page(self):
        fig = self.fig
        # Main field dominates; compact variable selector on the right
        # (roughly 12-15% of the page width).
        sg = self.page_area.subgridspec(
            1, 2, wspace=0.03, width_ratios=[0.87, 0.13]
        )
        main_ax = fig.add_subplot(sg[0, 0])
        radio_ax = fig.add_subplot(sg[0, 1])

        self._sol_ax = main_ax
        self._page_axes.extend([main_ax, radio_ax])

        names = list(SOLUTION_VARIABLES)
        try:
            active = names.index(self.selected_variable)
        except ValueError:
            active = 0
        radio = self.RadioButtons(radio_ax, names, active=active)
        # compact selector: the radio-button circles scale with the label
        # fontsize, so a small label font keeps the whole widget compact.
        radio.set_label_props({"fontsize": [7.5]})
        radio_ax.set_title("Variable", fontsize=9, pad=2)
        radio.on_clicked(self._on_variable_change)
        self._radio = radio

        main_ax.set_aspect("equal")
        main_ax.set_xlabel("x")
        main_ax.set_ylabel("y")

        self._mesh = None
        self._cbar = None
        self._drawn_sol_key = None
        self._sol_failed = False
        self._draw_solution()

    def _on_variable_change(self, label):
        self.selected_variable = label
        self._draw_solution()
        self.fig.canvas.draw_idle()

    def _redraw_mesh(self):
        data = self.data
        var = self.selected_variable
        vals = data.variables[var]

        if self._mesh is not None:
            self._mesh.remove()
            self._mesh = None

        if var == "vorticity":
            # signed out-of-plane vorticity: diverging colormap, color scale
            # centred on 0 (robust 99th-percentile limit).
            cmap = "RdBu_r"
            limits = _symmetric_vorticity_limits(vals)
            cmap_kw = dict(vmin=limits[0], vmax=limits[1]) if limits else {}
            cbar_label = VARIABLE_COLORBAR[var]
        else:
            cmap = "viridis"
            cmap_kw = {}
            cbar_label = var

        self._mesh = self._sol_ax.pcolormesh(
            data.x_grid, data.y_grid, vals,
            shading="auto", cmap=cmap, **cmap_kw,
        )

        label = VARIABLE_TITLES.get(var, var)
        tstr = f"t = {data.sol_time:.4e}" if data.sol_time is not None else "t = ?"
        self._sol_ax.set_title(
            f"{label}   |   Frame {data.current_frame + 1}/{len(data.solution_files)}"
            f"   |   {tstr}",
            fontsize=10,
        )

        if self._cbar is None:
            self._cbar = self.fig.colorbar(self._mesh, ax=self._sol_ax,
                                           label=cbar_label)
        else:
            self._cbar.update_normal(self._mesh)
            self._cbar.set_label(cbar_label)

    def _draw_solution(self):
        if self._sol_ax is None:
            # Not on the Solution page: keep the variable selection and the
            # current frame in state; the mesh is (re)built on page entry.
            return
        data = self.data
        if not data.solution_files:
            return

        if data.read_frame != data.current_frame:
            ok = data.read_solution(data.current_frame)
        else:
            ok = True

        key = (data.current_frame, self.selected_variable)
        if ok and key != self._drawn_sol_key:
            self._redraw_mesh()
            self._drawn_sol_key = key
            self._sol_failed = False
        elif not ok and not self._sol_failed:
            self._sol_failed = True

    # --------------------------------------------------------
    # monitor pages
    # --------------------------------------------------------

    def _monitor_specs(self, page, present):
        """Return per-page panel specs: (columns, ylabel, log_ok, ref_zero, extra)."""
        if page == 1:      # Physical Health
            return [
                ([("rho_min", "rho_min"), ("p_min", "p_min"),
                  ("ee_int_min", "ee_int_min"), ("ei_int_min", "ei_int_min"),
                  ("er_int_min", "er_int_min")],
                 "minimum quantity", False, True, []),
                ([("mach_max", "mach_max")], "Mach", False, False, []),
                ([("dt", "dt"), ("dt_cfl", "dt_cfl")], "dt", True, False, []),
                ([("dt_over_dt_cfl", "dt / dt_cfl")], "dt / dt_cfl", False, True, []),
            ]
        if page == 2:      # Temporal Activity
            return [
                ([("drho_dt_l2", "d(rho)/dt"), ("dmom_x_dt_l2", "d(mom_x)/dt"),
                  ("dmom_y_dt_l2", "d(mom_y)/dt"), ("dee_dt_l2", "d(ee)/dt"),
                  ("dei_dt_l2", "d(ei)/dt"), ("der_dt_l2", "d(er)/dt")],
                 "RMS ||(U^{n+1} - U^n)/dt||", True, False, []),
            ]
        if page == 3:      # Oscillation
            return [
                ([("tv_rho", "TV(rho)"), ("tv_p", "TV(p)")],
                 "Total Variation", True, False, []),
                ([("s2_rho", "S2(rho)"), ("s2_p", "S2(p)")],
                 "Second Difference", True, False, []),
            ]
        # page == 4       # Global Quantities
        rel_cols = []
        for base in ("mass", "mom_x", "total_energy"):
            if base in present:
                rel_cols.append((base + "_rel", base + " rel"))
        return [
            ([("mass", "mass")], "mass", False, False, []),
            ([("total_energy", "total_energy")], "total energy", False, False, []),
            ([("mom_x", "mom_x")], "mom_x", False, False, []),
            ([("mom_y", "mom_y")], "mom_y", False, False, []),
            (rel_cols, "relative change (Q-Q0)/|Q0|", False, True,
             [c for c, _ in rel_cols]),
        ]

    def _build_monitor_page(self, page):
        fig = self.fig
        cols = self.data.csv_columns

        if not cols:
            ax = fig.add_subplot(self.page_area)
            ax.axis("off")
            ax.text(0.5, 0.5, "Waiting for monitor.csv ...",
                    ha="center", va="center", fontsize=14,
                    transform=ax.transAxes)
            self._page_axes.append(ax)
            return

        xlabel = "Time" if self.data.xcol == "time" else "Step"
        marker = (self.data.xcol == "time")
        present = set(cols)

        # Page 3 (Activity): when the CSV carries the RHS residual columns,
        # show the residual view (normalized by default) plus the legacy
        # temporal-activity panel. Without residual columns the page falls
        # back to the old temporal-activity layout, so historical CSVs are
        # still fully supported.
        if page == 2 and self._residual_data_ready(present):
            self._build_residual_activity_page(cols, xlabel, marker, present)
            return

        specs = self._monitor_specs(page, present)

        sg = self.page_area.subgridspec(len(specs), 1, hspace=0.45)
        axes = [fig.add_subplot(sg[k, 0]) for k in range(len(specs))]
        self._page_axes.extend(axes)

        self._panels = []
        for ax, (columns, ylabel, log_ok, ref_zero, extra) in zip(axes, specs):
            pan = Panel(
                ax, columns, present | set(extra), self._warn_missing,
                ylabel, xlabel, log_ok=log_ok, ref_zero=ref_zero, marker=marker,
            )
            self._panels.append(pan)

        self._update_monitor_panels()

    def _residual_data_ready(self, present):
        """True when every raw RHS residual column expected by the Activity
        page is present in the CSV header."""
        return all(
            residual_col(comp, norm) in present
            for comp in RESIDUAL_COMPONENTS
            for norm in RESIDUAL_NORMS
        )

    def _build_residual_activity_page(self, cols, xlabel, marker, present):
        """Page 3 layout when residual columns are available.

        Left: two stacked panels
            [0] semi-discrete RHS residual (norm / raw / normalized switchable
                through the radio on the right; default = normalized R2)
            [1] legacy temporal activity  ||(U^{n+1}-U^n)/dt||_RMS  (kept)
        Right: vertical radio to pick R1 / R2 / R_inf x raw / normalized.
        """
        fig = self.fig

        # *_norm derived columns exist only once the file has data rows; they
        # are still declared up-front so the normalized view works as soon as
        # the first row arrives (reconfigure filters by what df actually has).
        avail = set(cols)
        for comp in RESIDUAL_COMPONENTS:
            for norm in RESIDUAL_NORMS:
                if residual_col(comp, norm) in avail:
                    avail.add(residual_norm_col(comp, norm))
        if self.data.plot_df is not None:
            avail |= set(self.data.plot_df.columns)

        has_temporal = any(
            c in avail for c in (
                "drho_dt_l2", "dmom_x_dt_l2", "dmom_y_dt_l2",
                "dee_dt_l2", "dei_dt_l2", "der_dt_l2",
            )
        )
        nrows = 2 if has_temporal else 1

        sg = self.page_area.subgridspec(
            1, 2, width_ratios=[0.80, 0.20], wspace=0.06
        )
        main_sg = sg[0, 0].subgridspec(nrows, 1, hspace=0.5)
        radio_ax = fig.add_subplot(sg[0, 1])
        self._page_axes.append(radio_ax)

        self._panels = []

        res_ax = fig.add_subplot(main_sg[0, 0])
        self._page_axes.append(res_ax)
        self._res_panel = None

        labels = [m[0] for m in RESIDUAL_MODES]
        idx = max(0, min(self._res_mode_index, len(labels) - 1))
        radio = self.RadioButtons(radio_ax, labels, active=idx)
        radio.set_label_props({"fontsize": [7.5] * len(labels)})
        radio_ax.set_title("RHS residual", fontsize=9, pad=3)
        radio.on_clicked(self._on_residual_mode)
        self._radio = radio

        res_panel = Panel(
            res_ax, [], avail, None, "", xlabel,
            log_ok=True, marker=marker, log_floor=True,
        )
        self._panels.append(res_panel)
        self._res_panel = res_panel

        # Apply the currently selected norm / raw / normalized mode; this also
        # draws the six component lines with the correct labels/scale.
        self._apply_residual_mode()

        if has_temporal:
            tact_ax = fig.add_subplot(main_sg[1, 0])
            self._page_axes.append(tact_ax)
            tact = Panel(
                tact_ax,
                [("drho_dt_l2", "d(rho)/dt"), ("dmom_x_dt_l2", "d(mom_x)/dt"),
                 ("dmom_y_dt_l2", "d(mom_y)/dt"), ("dee_dt_l2", "d(ee)/dt"),
                 ("dei_dt_l2", "d(ei)/dt"), ("der_dt_l2", "d(er)/dt")],
                avail, self._warn_missing,
                "RMS ||(U^{n+1} - U^n)/dt||", xlabel,
                log_ok=True, marker=marker, log_floor=True,
            )
            self._panels.append(tact)

        self._update_monitor_panels()

    def _residual_mode(self):
        """(norm, normalized) selected by the Activity-page radio."""
        labels = [m[0] for m in RESIDUAL_MODES]
        idx = max(0, min(self._res_mode_index, len(labels) - 1))
        return RESIDUAL_MODES[idx][1], RESIDUAL_MODES[idx][2]

    def _on_residual_mode(self, label):
        labels = [m[0] for m in RESIDUAL_MODES]
        if label not in labels:
            return
        self._res_mode_index = labels.index(label)
        self._apply_residual_mode()
        self._update_monitor_panels()
        self.fig.canvas.draw_idle()

    def _apply_residual_mode(self):
        """(Re)configure the residual panel for the selected norm / mode.

        Six component lines; the selected norm and raw/normalized state are
        spelled out in the title, y-label and legend title so a user always
        knows which residual definition is shown.
        """
        if self._res_panel is None:
            return
        norm, normalized = self._residual_mode()
        norm_disp = RESIDUAL_NORM_DISPLAY.get(norm, norm)

        columns = []
        for comp in RESIDUAL_COMPONENTS:
            if normalized:
                col = residual_norm_col(comp, norm)
            else:
                col = residual_col(comp, norm)
            columns.append((col, comp))

        if normalized:
            mode_word = "normalized  R/R_ref  (t_ref = first row)"
            ylabel = "normalized RHS residual  R / R_ref"
        else:
            mode_word = "raw"
            ylabel = "raw RHS residual magnitude"

        title = f"Semi-discrete RHS residual dU/dt=RHS(U)  |  {norm_disp}  {mode_word}"
        legend_title = f"{norm_disp}  ({'normalized' if normalized else 'raw'})"

        self._res_panel.reconfigure(
            columns, ylabel, title=title, legend_title=legend_title,
        )

    def _warn_missing(self, col):
        if col not in self.data.warned_cols:
            self.data.warned_cols.add(col)
            _log(f"warning: column '{col}' missing from CSV -- skipping its plot")

    def _update_monitor_panels(self):
        data = self.data
        plot_df = data.plot_df
        if data.xcol is None:
            for pan in self._panels:
                pan.update(None, "time")
            return
        marker_time = data.sol_time if data.xcol == "time" else None
        for pan in self._panels:
            pan.update(plot_df, data.xcol, marker_time=marker_time)

    # --------------------------------------------------------
    # navigation / live follow
    # --------------------------------------------------------

    def _goto_frame(self, index, live=None):
        data = self.data
        if not data.solution_files:
            return
        if live is not None:
            self.live_follow = live
        data.current_frame = max(0, min(index, len(data.solution_files) - 1))
        data.read_solution(data.current_frame)   # updates sol_time for markers
        self._refresh_current()

    def _refresh_current(self):
        if self.current_page == 0:
            self._draw_solution()
        else:
            self._update_monitor_panels()
        self._update_status()
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        k = event.key
        if k in ("1", "2", "3", "4", "5"):
            self.show_page(int(k) - 1)
        elif k == "right":
            self._goto_frame(self.data.current_frame + 1, live=False)
        elif k == "left":
            self._goto_frame(self.data.current_frame - 1, live=False)
        elif k == "up":
            self._goto_frame(self.data.current_frame + 10, live=False)
        elif k == "down":
            self._goto_frame(self.data.current_frame - 10, live=False)
        elif k == "home":
            self._goto_frame(0, live=False)
        elif k == "end":
            self._goto_frame(len(self.data.solution_files) - 1, live=True)
        elif k == " ":
            self.live_follow = not self.live_follow
            _log("LIVE FOLLOW " + ("ON" if self.live_follow else "OFF"))
            self._update_status()
            self.fig.canvas.draw_idle()
        elif k == "r":
            self._refresh_all()
        elif k == "q":
            self.plt.close(self.fig)

    # --------------------------------------------------------
    # live refresh
    # --------------------------------------------------------

    def _refresh_all(self):
        if self._closed:
            return
        data = self.data

        monitor_changed = data.refresh_monitor()
        data.refresh_solution_files()
        data.clamp_frame()

        if data.solution_files and self.live_follow:
            data.current_frame = len(data.solution_files) - 1

        # Keep the selected snapshot's physical time fresh for the monitor-page
        # markers (live follow may have moved the frame index here without
        # going through _goto_frame).
        if data.solution_files and data.read_frame != data.current_frame:
            data.read_solution(data.current_frame)

        if data.csv_reset_detected:
            _log("CSV reset detected -- new simulation data")
            data.csv_reset_detected = False

        # If the CSV schema changed (first data / reset / new columns) while a
        # monitor page is shown, rebuild the page so lines stay consistent.
        if self.current_page > 0:
            schema = tuple(data.csv_columns) if data.csv_columns else None
            if schema != self._built_csv_schema:
                self._build_page(self.current_page)

        if self.current_page == 0:
            self._draw_solution()
        else:
            self._update_monitor_panels()

        self._update_status()
        self._maybe_save()

    def _update_status(self):
        data = self.data
        parts = []
        n = len(data.solution_files)
        if n == 0:
            parts.append("Waiting for solution...")
        else:
            parts.append(f"Frame {data.current_frame + 1}/{n}")
            parts.append(f"t={data.sol_time:.6e}" if data.sol_time is not None else "t=?")

        if data.csv_state == "OK":
            parts.append(f"Monitor rows={len(data.monitor_df)}")
        elif data.csv_state == "HEADER":
            parts.append("Monitor: header only")
        elif data.csv_state == "RETRY":
            parts.append("CSV RETRY")
        else:
            parts.append("Waiting for monitor.csv...")

        parts.append("LIVE" if self.live_follow else "PAUSED")
        csv_tok = {"OK": "CSV OK", "RETRY": "CSV RETRY",
                   "HEADER": "CSV WAIT", "NONE": "CSV WAIT"}[data.csv_state]
        parts.append(csv_tok)
        self.status_text.set_text(" | ".join(parts))

    def _maybe_save(self):
        if not self.args.save_dir:
            return
        data = self.data
        key = (
            len(data.solution_files),
            data.current_frame if data.solution_files else 0,
            len(data.monitor_df) if data.monitor_df is not None else 0,
            data.csv_state,
        )
        if key == self.last_save_key:
            return
        self.last_save_key = key
        try:
            os.makedirs(self.args.save_dir, exist_ok=True)
            self.fig.savefig(os.path.join(self.args.save_dir, "dashboard.png"),
                             dpi=110)
        except Exception as exc:
            _log(f"warning: failed to save snapshot: {exc}")

    # --------------------------------------------------------
    # event loop
    # --------------------------------------------------------

    def _on_timer(self):
        if self._closed:
            return
        try:
            self._refresh_all()
            if not self._closed:
                self.fig.canvas.draw_idle()
        except Exception as exc:
            _log(f"warning: refresh error: {exc}")
            if not self.plt.fignum_exists(self.fig.number):
                self._closed = True
                if self._timer is not None:
                    self._timer.stop()

    def _on_close(self, _event):
        self._closed = True
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass

    def run_gui(self):
        interval_ms = max(100, int(self.args.interval * 1000))
        self._timer = self.fig.canvas.new_timer(interval=interval_ms)
        self._timer.add_callback(self._on_timer)
        self._timer.start()

        # Ctrl+C: Tk's mainloop may otherwise swallow KeyboardInterrupt while
        # blocked inside plt.show(). Stop the event loop so show() returns and
        # main() can clean up without a traceback. (No plt.close here: widget
        # teardown during close can race the closing canvas and print a benign
        # matplotlib traceback; the process exit tears the window down anyway.)
        def _sigint(_sig, _frame):
            _log("\nStopping live monitor.")
            try:
                self.fig.canvas.stop_event_loop()
            except Exception:
                pass

        import signal
        signal.signal(signal.SIGINT, _sigint)

        self.plt.show(block=True)

    def run_headless(self):
        try:
            while not self._closed:
                t0 = time.time()
                self._on_timer()
                remaining = self.args.interval - (time.time() - t0)
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            raise


def main():
    args = parse_args()

    # Select the backend before pyplot is imported anywhere.
    if args.no_gui:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dashboard = Dashboard(args, plt)
    try:
        if args.no_gui:
            dashboard.run_headless()
        else:
            dashboard.run_gui()
    except KeyboardInterrupt:
        _log("\nStopping live monitor.")
        try:
            plt.close("all")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
