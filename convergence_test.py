import sys
from pathlib import Path

import numpy as np


# ============================================================
# Import binary reader
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from py_utils import solution_io  # noqa: E402


# ============================================================
# MMS convergence cases
#
# init_mms_wall now uses a CELL-CENTERED grid:
#
#     x_i = dx/2 + i*dx,   y_j = dy/2 + j*dy
#
# so coarse and fine cell centers do NOT coincide and a pointwise
# self-convergence restriction is no longer valid.  We therefore
# measure the error of each solution against the analytic
# manufactured solution (the standard MMS convergence test).
# ============================================================

CASES = [
    (40,  "n40.bin"),
    (80,  "n80.bin"),
    (160, "n160.bin"),
    (320, "n320.bin"),
    #(640, "n640.bin"),
]


VARIABLES = [
    "rho",
    "mom_x",
    "mom_y",
    "ee",
    "ei",
    "er",
]


# Number of grid layers near bottom/top wall for the wall-band Linf
WALL_LAYERS = 10


# ============================================================
# Analytic manufactured solution (init.rs::wall_mms_exact_state)
# ============================================================

def wall_mms_exact(x, y, t):
    sx = np.sin(x)
    cx = np.cos(x)

    sy = np.sin(y)
    cy = np.cos(y)

    st = np.sin(t)
    ct = np.cos(t)

    rho = 1.0 + 0.1 * sx * cy * ct

    ux = 1.0 + 0.2 * cx * cy * ct
    uy = 0.2 * sx * sy * ct

    rho_ee = 3.0 + 0.2 * cx * cy * st
    rho_ei = 3.0 + 0.15 * sx * cy * ct
    rho_er = 2.0 + 0.1 * np.cos(2.0 * x) * cy * st

    kinetic_share = rho * (ux * ux + uy * uy) / 6.0

    return {
        "rho": rho,
        "mom_x": rho * ux,
        "mom_y": rho * uy,
        "ee": rho_ee + kinetic_share,
        "ei": rho_ei + kinetic_share,
        "er": rho_er + kinetic_share,
    }


# ============================================================
# Read solution
# ============================================================

def read_case(n, filename):
    path = SCRIPT_DIR / filename

    if not path.exists():
        path = SCRIPT_DIR / "data" / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {filename}"
        )

    x, y, time, field = solution_io.read_solution_file(path)

    return {
        "N": n,
        "path": path,
        "x": np.asarray(x),
        "y": np.asarray(y),
        "time": time,
        "field": {
            key: np.asarray(value)
            for key, value in field.items()
        },
    }


# ============================================================
# Norms of difference between two arrays
# ============================================================

def difference_norms(a, b):
    diff = np.asarray(a) - np.asarray(b)

    l1 = np.mean(np.abs(diff))
    l2 = np.sqrt(np.mean(diff * diff))
    linf = np.max(np.abs(diff))

    return {
        "L1": l1,
        "L2": l2,
        "Linf": linf,
    }


# ============================================================
# Wall-band Linf difference
#
# We use the first/last WALL_LAYERS rows as the wall band.
# ============================================================

def wall_linf_difference(a, b):
    diff = np.abs(
        np.asarray(a)
        - np.asarray(b)
    )

    layers = min(
        WALL_LAYERS,
        diff.shape[0] // 2,
    )

    bottom = diff[:layers, :]
    top = diff[-layers:, :]

    return max(
        np.max(bottom),
        np.max(top),
    )


# ============================================================
# Observed convergence order
#
# For refinement ratio r = N_prev / N_curr:
#
#     p = log( e_prev / e_curr ) / log( r )
# ============================================================

def order_from_errors(e_prev, e_curr, n_prev, n_curr):
    if e_prev <= 0.0 or e_curr <= 0.0:
        return np.nan

    r = float(n_curr) / float(n_prev)

    return np.log(e_prev / e_curr) / np.log(r)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 100)
    print("MMS CONVERGENCE TEST  (error vs analytic solution)")
    print("=" * 100)

    # --------------------------------------------------------
    # Read solutions
    # --------------------------------------------------------

    solutions = [
        read_case(n, filename)
        for n, filename in CASES
    ]

    # --------------------------------------------------------
    # Verify final times
    # --------------------------------------------------------

    t0 = solutions[0]["time"]

    for sol in solutions:
        print(
            f"N={sol['N']:4d}  "
            f"t={sol['time']:.16e}  "
            f"shape={sol['field']['rho'].shape}  "
            f"file={sol['path'].name}"
        )

        if abs(sol["time"] - t0) > 1.0e-10:
            raise RuntimeError(
                "Convergence requires all solutions "
                "at the same physical time."
            )

    print()
    print("=" * 120)
    print("MMS ERRORS AND OBSERVED ORDERS")
    print("=" * 120)

    for var in VARIABLES:

        print()
        print("-" * 120)
        print(f"Variable: {var}")
        print("-" * 120)

        print(
            f"{'N':>6}  "
            f"{'L1 error':>14}  {'order':>7}  "
            f"{'L2 error':>14}  {'order':>7}  "
            f"{'Linf error':>14}  {'order':>7}  "
            f"{'wall Linf':>14}  {'order':>7}"
        )

        prev = None

        for sol in solutions:
            n = sol["N"]
            t = sol["time"]

            x = sol["x"]
            y = sol["y"]

            X, Y = np.meshgrid(x, y)

            exact = wall_mms_exact(X, Y, t)[var]
            num = sol["field"][var]

            err = difference_norms(num, exact)
            wall = wall_linf_difference(num, exact)

            if prev is None:
                p_l1 = p_l2 = p_linf = p_wall = np.nan
            else:
                p_l1 = order_from_errors(
                    prev["err"]["L1"], err["L1"], prev["N"], n
                )
                p_l2 = order_from_errors(
                    prev["err"]["L2"], err["L2"], prev["N"], n
                )
                p_linf = order_from_errors(
                    prev["err"]["Linf"], err["Linf"], prev["N"], n
                )
                p_wall = order_from_errors(
                    prev["wall"], wall, prev["N"], n
                )

            print(
                f"{n:6d}  "
                f"{err['L1']:14.6e}  {p_l1:7.3f}  "
                f"{err['L2']:14.6e}  {p_l2:7.3f}  "
                f"{err['Linf']:14.6e}  {p_linf:7.3f}  "
                f"{wall:14.6e}  {p_wall:7.3f}"
            )

            prev = {
                "N": n,
                "err": err,
                "wall": wall,
            }

    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
