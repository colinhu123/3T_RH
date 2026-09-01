#!/usr/bin/env python3
"""
scripts/live_monitor.py

Standalone Python live monitor for the Rust CFD solver's diagnostics file
data/monitor.csv.

Architecture (Python and Rust are fully decoupled):

    Rust solver
        |
        | continuously appends rows
        v
    data/monitor.csv
        ^
        |
        | periodically reads (polling)
        |
    scripts/live_monitor.py
        |
        v
    matplotlib live figures (+ optional PNG snapshots)

The monitor is a *pure reader*:
  * it never locks / truncates / renames / rewrites / deletes the CSV
  * it never touches the Rust process or its memory (no FFI, no bindings)
  * if the monitor crashes, the Rust simulation is completely unaffected

Robustness handled here (this is the main design point):
  * CSV not created yet               -> print "Waiting for ...", keep waiting
  * CSV mid-write / partial row       -> the trailing incomplete line is dropped;
                                          previously plotted data/figures are kept
  * header-only CSV                   -> wait for data rows, no crash
  * NaN / +/-Inf values               -> sanitized to NaN before plotting
  * missing columns                   -> the column is skipped with a one-time
                                          warning; the rest of the figures keep
                                          working (future-proof against schema changes)
  * CSV truncate / fresh run          -> detected as a reset; figures are rebuilt
                                          from the new data (old/new are never mixed)

Plotted quantities (same names as the CSV columns):

  time stepping : dt, dt_cfl, dt_over_dt_cfl
  physical      : rho_min, p_min, ee_int_min, ei_int_min, er_int_min, mach_max
  temporal      : drho_dt_l2, dmom_x_dt_l2, dmom_y_dt_l2, dee_dt_l2, dei_dt_l2, der_dt_l2
                  || (U^{n+1} - U^n) / dt ||_RMS  -> a *temporal activity* measure,
                  NOT a steady-state residual
  oscillation   : tv_rho, tv_p, s2_rho, s2_p
  global        : mass, mom_x, mom_y, total_energy (+ relative change vs first value)
"""

import argparse
import io
import math
import os
import time

import numpy as np
import pandas as pd


def _log(msg):
    """Print a status line, always flushed (survives kill / long idle runs)."""
    print(msg, flush=True)


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Live diagnostic monitor for data/monitor.csv written by the "
            "Rust CFD solver (read-only, does not touch the solver)."
        )
    )
    p.add_argument(
        "--file",
        default="data/monitor.csv",
        help="path to the solver diagnostics CSV (default: data/monitor.csv)",
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
        help="maximum number of points plotted per line after stride downsampling "
             "(default: 5000)",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        help="directory to periodically write PNG snapshots (files are overwritten "
             "on every refresh, no unbounded file growth)",
    )
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="headless mode: no GUI window; PNG snapshots are written to --save-dir "
             "(auto-provided as data/monitor_plots if --save-dir is omitted)",
    )
    args = p.parse_args()

    if args.interval <= 0.0:
        p.error("--interval must be positive")
    if args.max_points < 100:
        p.error("--max-points must be at least 100")

    if args.no_gui and args.save_dir is None:
        args.save_dir = "data/monitor_plots"
        _log(
            "--no-gui without --save-dir: snapshots will be written to "
            "data/monitor_plots"
        )

    return args


# ============================================================
# Robust CSV reading
# ============================================================

def _row_fully_numeric(row):
    """True if every comma-separated field of `row` parses as a float.

    Used to detect a trailing row the Rust solver is still writing: a
    partially-written number (e.g. "1.23e" or "1.2e-") fails float().
    Plain 'nan' / 'inf' / '-inf' parse fine and are intentionally accepted.
    """
    for tok in row.split(","):
        try:
            float(tok)
        except ValueError:
            return False
    return True


def _clean_lines(lines):
    """Tolerate a CSV file that is being written right now.

    Returns a cleaned CSV text, or None when nothing usable is there yet.

    * blank lines are removed
    * rows whose field count differs from the header are dropped (this is
      typically the incomplete trailing row)
    * if the final remaining row does not parse as pure floats it is dropped
      too (a number can have the right field count yet still be truncated)
    * a header-only file yields just the header (an empty DataFrame later)
    """
    rows = [ln.rstrip("\r\n") for ln in lines]
    rows = [ln for ln in rows if ln.strip()]

    if not rows:
        return None

    header = rows[0]
    nfields = header.count(",") + 1
    if nfields < 2:
        # malformed header (possibly mid-write): treat as not ready yet
        return None

    body = [ln for ln in rows[1:] if ln.count(",") + 1 == nfields]

    # A trailing row that is mid-write can still have the right field count
    # (only its last number is truncated). Only the last row is examined,
    # because the solver writes one row at a time.
    if body and not _row_fully_numeric(body[-1]):
        body = body[:-1]

    if not body:
        return header + "\n"

    return "\n".join([header] + body)


def safe_read_csv(path):
    """Read the CSV into a pandas DataFrame.

    Returns None when the file is missing / empty / mid-write, and an empty
    DataFrame (columns only) when only a header is present. This function
    never raises for malformed, incomplete or changing input.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    if st.st_size == 0:
        return None

    lines = None
    # A very short read (file being appended/truncated at the same moment)
    # is retried briefly before giving up for this cycle.
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
        # e.g. the header was captured mid-write; retry next cycle
        return None


# ============================================================
# Plotting helpers
# ============================================================

def sanitize_series(values):
    """Return a float array where +Inf / -Inf are replaced by NaN.

    NaN is simply not drawn by matplotlib and does not corrupt autoscaling,
    while +Inf / -Inf would. The original CSV is never modified.
    """
    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def downsample_dataframe(df, max_points):
    """Stride-downsample so the whole history stays visible.

    With N <= max_points all points are drawn. Otherwise ~max_points points
    are kept using a uniform stride over the *entire* history (not just the
    tail), and the very last (latest) point is always included.
    """
    n = len(df)
    if n <= max_points:
        return df
    stride = math.ceil(n / max_points)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return df.iloc[idx]


class Panel:
    """One axes with a set of lines, per-refresh autoscaling and a y=0 line.

    Columns that are absent from the CSV are skipped (one-time warning) so a
    schema change in monitor.csv never crashes the monitor.
    """

    def __init__(self, ax, columns, present, warn, ylabel, xlabel,
                 log_ok=False, ref_zero=False):
        self.ax = ax
        self.lines = []
        self.log_ok = log_ok
        self.ref_zero = ref_zero

        for col, label in columns:
            if col not in present:
                if warn is not None:
                    warn(col)
                continue
            line, = ax.plot([], [], label=label, lw=1.2)
            self.lines.append((col, line))

        if ref_zero:
            ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5, zorder=0)

        if self.lines:
            ax.legend(loc="best", fontsize="small", ncol=2)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    def update(self, df, xcol):
        if xcol in df.columns:
            xs = sanitize_series(df[xcol])
        else:
            xs = np.arange(len(df), dtype=float)

        for col, line in self.lines:
            if col in df.columns:
                line.set_data(xs, sanitize_series(df[col]))
            else:
                line.set_data([], [])

        self._autoscale(df, xs)

    def _autoscale(self, df, xs):
        xs = np.asarray(xs, dtype=float)
        xf = xs[np.isfinite(xs)]
        if xf.size == 0:
            return
        self.ax.set_xlim(float(xf[0]), float(xf[-1]))

        yvals = [sanitize_series(df[col]) for col, _ in self.lines
                 if col in df.columns]
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
            # log scale only on strictly-positive data; otherwise we switch
            # back to linear so 0 / negative values never raise warnings.
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
# Live monitor application
# ============================================================

class LiveMonitor:
    """Owns the figures and the polling loop."""

    FIG_NAMES = {
        "time_stepping": "time_stepping.png",
        "physical_health": "physical_health.png",
        "temporal_activity": "temporal_activity.png",
        "oscillation_indicators": "oscillation_indicators.png",
        "global_quantities": "global_quantities.png",
    }

    def __init__(self, args, plt):
        self.args = args
        self.plt = plt

        self.xcol = None
        self.figs = {}
        self.panels = []
        self.header_cols = None
        self.present_cols = set()
        self.warned_cols = set()

        self.last_stat = None
        self.prev_row_count = 0
        self.prev_max_x = None

        self.last_saved_count = 0
        self.last_status_time = 0.0
        self.last_status_count = 0
        self.header_only = False

    # --------------------------------------------------------
    # main loop
    # --------------------------------------------------------

    def run(self):
        waiting_printed = False
        while True:
            df = self._read_csv()
            if df is None:
                # Only keep the "Waiting" message while the file does not
                # exist yet. If it exists but is mid-write / header-only /
                # unchanged, stay quiet (we already reported that state).
                if os.path.exists(self.args.file):
                    waiting_printed = False
                elif not waiting_printed:
                    _log(f"Waiting for {self.args.file} ...")
                    waiting_printed = True
                self._sleep()
                continue
            waiting_printed = False

            cols_changed = (
                self.header_cols is None
                or list(df.columns) != self.header_cols
            )
            reset = self._detect_reset(df, cols_changed)

            if cols_changed or reset:
                if reset:
                    _log("CSV reset detected -- figures rebuilt from new simulation")
                elif self.header_cols is not None:
                    _log("CSV columns changed -- figures rebuilt")
                self._build_figures(df)

            if len(df) == 0:
                if not self.header_only:
                    _log(f"{self.args.file} present (header only) -- waiting for data rows")
                    self.header_only = True
                self._sleep()
                continue
            self.header_only = False

            self._update(df)
            self._maybe_save(df)
            self._maybe_status(df)
            self._sleep()

    # --------------------------------------------------------
    # file access
    # --------------------------------------------------------

    def _read_csv(self):
        """Skip re-parsing when the file is unchanged since the last read."""
        try:
            st = os.stat(self.args.file)
        except OSError:
            return None
        key = (st.st_size, st.st_mtime_ns)
        if self.last_stat == key:
            return None
        df = safe_read_csv(self.args.file)
        if df is not None:
            self.last_stat = key
        return df

    def _sleep(self):
        if self.args.no_gui:
            time.sleep(self.args.interval)
        else:
            self.plt.pause(self.args.interval)

    # --------------------------------------------------------
    # reset detection
    # --------------------------------------------------------

    def _detect_reset(self, df, cols_changed):
        if self.header_cols is None:
            return False
        if cols_changed:
            return True  # a different header implies a fresh simulation

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

    # --------------------------------------------------------
    # figure construction (called once, or again only on reset)
    # --------------------------------------------------------

    def _build_figures(self, df):
        self._close_figures()
        self.header_cols = list(df.columns)
        self.present_cols = set(self.header_cols)
        self.warned_cols = set()
        # A rebuilt figure set belongs to a (new) simulation: forget the old
        # snapshot/status counters so PNGs regenerate immediately.
        self.last_saved_count = 0
        self.last_status_count = 0
        self.last_status_time = 0.0

        self.xcol = self._choose_xcol(df)
        xlabel = "Time" if self.xcol == "time" else "Step"

        self.figs["time_stepping"] = self._build_time_stepping(xlabel)
        self.figs["physical_health"] = self._build_physical_health(xlabel)
        self.figs["temporal_activity"] = self._build_temporal_activity(xlabel)
        self.figs["oscillation_indicators"] = self._build_oscillation(xlabel)
        self.figs["global_quantities"] = self._build_global(xlabel)

        if not self.args.no_gui:
            self.plt.show(block=False)

    def _close_figures(self):
        self.panels = []
        for fig in list(self.figs.values()):
            try:
                self.plt.close(fig)
            except Exception:
                pass
        self.figs = {}

    def _choose_xcol(self, df):
        if "time" in df.columns:
            return "time"
        if "step" in df.columns:
            return "step"
        return df.columns[0]

    def _subplots(self, name, title, nrows, figsize=(9.0, 4.5), sharex=False):
        fig, axes = self.plt.subplots(
            nrows, 1, figsize=figsize, sharex=sharex, num=name
        )
        fig.suptitle(title, fontsize=12)
        if nrows == 1:
            axes = [axes]
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig, axes

    def _add_panel(self, ax, columns, ylabel, xlabel, log_ok=False, ref_zero=False,
                   extra_present=()):
        present = self.present_cols | set(extra_present)
        panel = Panel(
            ax,
            columns,
            present,
            self._warn_missing,
            ylabel,
            xlabel,
            log_ok=log_ok,
            ref_zero=ref_zero,
        )
        self.panels.append(panel)
        return panel

    def _warn_missing(self, col):
        if col not in self.warned_cols:
            self.warned_cols.add(col)
            _log(f"warning: column '{col}' missing from CSV -- skipping its plot")

    # --------------------------------------------------------
    # individual figures
    # --------------------------------------------------------

    def _build_time_stepping(self, xlabel):
        fig, axes = self._subplots(
            "time_stepping", "Time Stepping", 2, figsize=(9.0, 6.0), sharex=True
        )
        self._add_panel(
            axes[0],
            [("dt", "dt"), ("dt_cfl", "dt_cfl")],
            "dt",
            xlabel,
            log_ok=True,
        )
        self._add_panel(
            axes[1],
            [("dt_over_dt_cfl", "dt / dt_cfl")],
            "dt / dt_cfl",
            xlabel,
            log_ok=False,
            ref_zero=True,
        )
        return fig

    def _build_physical_health(self, xlabel):
        fig, axes = self._subplots(
            "physical_health", "Physical Health", 2, figsize=(9.0, 6.0), sharex=True
        )
        self._add_panel(
            axes[0],
            [
                ("rho_min", "rho_min"),
                ("p_min", "p_min"),
                ("ee_int_min", "ee_int_min"),
                ("ei_int_min", "ei_int_min"),
                ("er_int_min", "er_int_min"),
            ],
            "minimum quantity",
            xlabel,
            log_ok=False,   # may legitimately approach or cross zero
            ref_zero=True,  # y=0 is the key numerical-health signal
        )
        self._add_panel(
            axes[1],
            [("mach_max", "mach_max")],
            "Mach",
            xlabel,
            log_ok=False,
        )
        return fig

    def _build_temporal_activity(self, xlabel):
        fig, axes = self._subplots(
            "temporal_activity", "Temporal Activity (RMS time derivative, not a residual)",
            1, figsize=(9.0, 4.5),
        )
        self._add_panel(
            axes[0],
            [
                ("drho_dt_l2", "d(rho)/dt"),
                ("dmom_x_dt_l2", "d(mom_x)/dt"),
                ("dmom_y_dt_l2", "d(mom_y)/dt"),
                ("dee_dt_l2", "d(ee)/dt"),
                ("dei_dt_l2", "d(ei)/dt"),
                ("der_dt_l2", "d(er)/dt"),
            ],
            "RMS ||(U^{n+1} - U^n)/dt||",
            xlabel,
            log_ok=True,  # positive; log keeps the dynamic range readable
        )
        return fig

    def _build_oscillation(self, xlabel):
        fig, axes = self._subplots(
            "oscillation_indicators", "Oscillation Indicators", 2,
            figsize=(9.0, 6.0), sharex=True,
        )
        self._add_panel(
            axes[0],
            [("tv_rho", "TV(rho)"), ("tv_p", "TV(p)")],
            "Total Variation",
            xlabel,
            log_ok=True,
        )
        self._add_panel(
            axes[1],
            [("s2_rho", "S2(rho)"), ("s2_p", "S2(p)")],
            "Second Difference",
            xlabel,
            log_ok=True,
        )
        return fig

    def _build_global(self, xlabel):
        fig, axes = self._subplots(
            "global_quantities", "Global Quantities", 5,
            figsize=(9.0, 11.0), sharex=True,
        )
        self._add_panel(axes[0], [("mass", "mass")], "mass", xlabel, log_ok=False)
        self._add_panel(
            axes[1], [("total_energy", "total_energy")], "total energy",
            xlabel, log_ok=False,
        )
        self._add_panel(axes[2], [("mom_x", "mom_x")], "mom_x", xlabel, log_ok=False)
        self._add_panel(
            axes[3], [("mom_y", "mom_y")], "mom_y", xlabel, log_ok=False, ref_zero=True
        )

        # Relative change vs the first valid value: (Q - Q0) / |Q0|.
        #
        # NOTE: for open / inflow-outflow / embedded-boundary simulations
        # mass(t) - mass(0) must NOT be interpreted as a conservation error;
        # this figure is intentionally called "Global Quantities" (relative
        # change), never "Conservation Error".
        rel_cols = []
        for base in ("mass", "mom_x", "total_energy"):
            if base in self.present_cols:
                rel_cols.append((base + "_rel", base + " rel"))
        self._add_panel(
            axes[4],
            rel_cols,
            "relative change (Q-Q0)/|Q0|",
            xlabel,
            log_ok=False,
            ref_zero=True,
            extra_present=[c for c, _ in rel_cols],
        )
        return fig

    # --------------------------------------------------------
    # derived columns + refresh
    # --------------------------------------------------------

    def _derived_frame(self, df):
        """Add *_rel columns; guard against a near-zero baseline Q0."""
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
                # Q0 essentially zero (e.g. mom_y): plot absolute change instead
                out[base + "_rel"] = series - q0
        return out

    def _update(self, df):
        plot_df = downsample_dataframe(
            self._derived_frame(df), self.args.max_points
        )

        for panel in self.panels:
            fig = panel.ax.figure
            if not self.plt.fignum_exists(fig.number):
                continue  # user closed this window; keep the rest running
            panel.update(plot_df, self.xcol)

        self.prev_row_count = len(df)
        if self.xcol in df.columns:
            xs = pd.to_numeric(df[self.xcol], errors="coerce")
            if not xs.dropna().empty:
                self.prev_max_x = float(xs.max())

        try:
            self.plt.draw()
        except Exception:
            pass

    # --------------------------------------------------------
    # snapshots + status
    # --------------------------------------------------------

    def _maybe_save(self, df):
        if not self.args.save_dir:
            return
        if len(df) == 0 or len(df) <= self.last_saved_count:
            return  # nothing new since the last snapshot
        try:
            os.makedirs(self.args.save_dir, exist_ok=True)
            for name, fname in self.FIG_NAMES.items():
                fig = self.figs.get(name)
                if fig is not None and self.plt.fignum_exists(fig.number):
                    fig.savefig(
                        os.path.join(self.args.save_dir, fname), dpi=110
                    )
            self.last_saved_count = len(df)
        except Exception as exc:
            _log(f"warning: failed to save snapshots: {exc}")

    def _maybe_status(self, df):
        now = time.time()
        n = len(df)
        if n == self.last_status_count:
            return
        if now - self.last_status_time < 60.0:
            return
        self.last_status_time = now
        self.last_status_count = n
        latest = "?"
        if self.xcol in df.columns:
            xs = pd.to_numeric(df[self.xcol], errors="coerce").dropna()
            if not xs.empty:
                latest = f"{float(xs.iloc[-1]):.6e}"
        _log(f"Monitoring {n} steps, latest {self.xcol} = {latest}")


def main():
    args = parse_args()

    # Select the backend before pyplot is imported anywhere.
    if args.no_gui:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monitor = LiveMonitor(args, plt)
    try:
        monitor.run()
    except KeyboardInterrupt:
        _log("\nStopping live monitor.")
        try:
            plt.close("all")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
