#!/usr/bin/env python3

import argparse
import os
import re
import struct

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Binary format -- identical to visualize_sol.py
# ============================================================

MAGIC = b"RH3TBIN1"
HEADER = struct.Struct("<8sIIQQd")
NVAR = 8


# ============================================================
# Read solution
# ============================================================

def read_file(filename):
    with open(filename, "rb") as f:
        raw = f.read(HEADER.size)

    if len(raw) != HEADER.size:
        raise RuntimeError(
            f"Incomplete binary header in {filename}"
        )

    magic, version, nvar, nx, ny, time = HEADER.unpack(raw)

    if magic != MAGIC:
        raise RuntimeError(
            f"Wrong file magic: {magic!r}; "
            f"expected {MAGIC!r}"
        )

    if version != 1:
        raise RuntimeError(
            f"Unsupported binary version {version}"
        )

    if nvar != NVAR:
        raise RuntimeError(
            f"Unexpected nvar={nvar}; "
            f"expected {NVAR}"
        )

    expected_bytes = (
        HEADER.size
        + nx * ny * nvar * 8
    )

    actual_bytes = os.path.getsize(filename)

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Binary size mismatch:\n"
            f"expected = {expected_bytes}\n"
            f"actual   = {actual_bytes}"
        )

    data = np.memmap(
        filename,
        dtype="<f8",
        mode="r",
        offset=HEADER.size,
        shape=(ny, nx, nvar),
        order="C",
    )

    x = np.asarray(data[0, :, 0])
    y = np.asarray(data[:, 0, 1])

    return data, x, y, time


# ============================================================
# Components
# ============================================================

def get_component(data, name):
    """
    Payload:
        0 = x
        1 = y
        2 = rho
        3 = mom_x
        4 = mom_y
        5 = ee
        6 = ei
        7 = er

    Additional derived quantities:
        ux
        uy
        speed
    """

    direct = {
        "rho": 2,
        "mom_x": 3,
        "mom_y": 4,
        "ee": 5,
        "ei": 6,
        "er": 7,
    }

    if name in direct:
        return np.asarray(
            data[:, :, direct[name]]
        )

    rho = np.asarray(data[:, :, 2])
    mom_x = np.asarray(data[:, :, 3])
    mom_y = np.asarray(data[:, :, 4])

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        ux = mom_x / rho
        uy = mom_y / rho

    if name == "ux":
        return ux

    if name == "uy":
        return uy

    if name == "speed":
        return np.sqrt(
            ux * ux + uy * uy
        )

    raise ValueError(
        f"Unknown component: {name}"
    )


# ============================================================
# Parse line
# ============================================================

def parse_line(expr):
    """
    Supported:

        y=0
        y=0.025
        x=-1.4
        y=0.5*x+0.2
        y=-2*x+1

    Internal representation:

        A*x + B*y + C = 0
    """

    expr = expr.replace(" ", "")

    number = (
        r"[+-]?"
        r"(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?"
    )

    # x = c
    m = re.fullmatch(
        rf"x=({number})",
        expr,
    )

    if m:
        c = float(m.group(1))

        return 1.0, 0.0, -c

    # y = c
    m = re.fullmatch(
        rf"y=({number})",
        expr,
    )

    if m:
        c = float(m.group(1))

        return 0.0, 1.0, -c

    # y = a*x + b
    #
    # Examples:
    # y=x
    # y=-x
    # y=0.5*x
    # y=0.5*x+0.2

    m = re.fullmatch(
        r"y="
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)?)"
        r"\*?x"
        r"("
        r"[+-]"
        r"(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?"
        r")?",
        expr,
    )

    if m:
        a_str = m.group(1)

        if a_str in ("", "+"):
            a = 1.0
        elif a_str == "-":
            a = -1.0
        else:
            a = float(a_str)

        b = (
            float(m.group(2))
            if m.group(2)
            else 0.0
        )

        # y = ax+b
        #
        # ax - y + b = 0

        return a, -1.0, b

    raise ValueError(
        f"Cannot parse line: {expr}\n\n"
        f"Examples:\n"
        f'  --line "y=0"\n'
        f'  --line "x=-1.4"\n'
        f'  --line "y=0.5*x+0.2"'
    )


# ============================================================
# Intersect infinite line with bounding box
# ============================================================

def line_box_intersection(
    A,
    B,
    C,
    xmin,
    xmax,
    ymin,
    ymax,
):
    """
    Find intersections between

        A*x + B*y + C = 0

    and the rectangular grid bounding box.

    Returns two endpoints.
    """

    points = []

    eps = 1.0e-12

    # x = xmin / xmax
    if abs(B) > eps:
        for x in (xmin, xmax):
            y = -(A * x + C) / B

            if ymin - eps <= y <= ymax + eps:
                points.append((x, y))

    # y = ymin / ymax
    if abs(A) > eps:
        for y in (ymin, ymax):
            x = -(B * y + C) / A

            if xmin - eps <= x <= xmax + eps:
                points.append((x, y))

    # Remove duplicate corner intersections
    unique = []

    for p in points:
        duplicate = False

        for q in unique:
            if (
                abs(p[0] - q[0]) < eps
                and abs(p[1] - q[1]) < eps
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(p)

    if len(unique) < 2:
        raise ValueError(
            "The requested line does not cross "
            "the computational domain."
        )

    # If numerical corner handling gives >2,
    # choose the farthest pair.

    best = None
    best_d2 = -1.0

    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dx = unique[i][0] - unique[j][0]
            dy = unique[i][1] - unique[j][1]

            d2 = dx * dx + dy * dy

            if d2 > best_d2:
                best_d2 = d2
                best = (
                    unique[i],
                    unique[j],
                )

    return best


# ============================================================
# Bilinear interpolation
# ============================================================

def bilinear_sample(
    field,
    x_grid,
    y_grid,
    xp,
    yp,
):
    """
    Bilinear interpolation on a uniform Cartesian grid.

    Returns NaN if the interpolation cell contains
    invalid/solid data.
    """

    nx = len(x_grid)
    ny = len(y_grid)

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    fx = (xp - x_grid[0]) / dx
    fy = (yp - y_grid[0]) / dy

    i = int(np.floor(fx))
    j = int(np.floor(fy))

    # Deal with points exactly on xmax/ymax.
    if i == nx - 1:
        i = nx - 2
        tx = 1.0
    else:
        tx = fx - i

    if j == ny - 1:
        j = ny - 2
        ty = 1.0
    else:
        ty = fy - j

    if (
        i < 0
        or i >= nx - 1
        or j < 0
        or j >= ny - 1
    ):
        return np.nan

    q00 = field[j, i]
    q10 = field[j, i + 1]
    q01 = field[j + 1, i]
    q11 = field[j + 1, i + 1]

    values = np.array(
        [q00, q10, q01, q11]
    )

    # Important for embedded boundaries:
    #
    # Do NOT interpolate across solid/NaN cells.

    if not np.all(np.isfinite(values)):
        return np.nan

    return (
        (1.0 - tx)
        * (1.0 - ty)
        * q00

        + tx
        * (1.0 - ty)
        * q10

        + (1.0 - tx)
        * ty
        * q01

        + tx
        * ty
        * q11
    )


# ============================================================
# Sample line
# ============================================================

def sample_line(
    field,
    x_grid,
    y_grid,
    line,
    n_samples,
):
    A, B, C = line

    xmin = x_grid[0]
    xmax = x_grid[-1]

    ymin = y_grid[0]
    ymax = y_grid[-1]

    p0, p1 = line_box_intersection(
        A,
        B,
        C,
        xmin,
        xmax,
        ymin,
        ymax,
    )

    x0, y0 = p0
    x1, y1 = p1

    lam = np.linspace(
        0.0,
        1.0,
        n_samples,
    )

    xs = x0 + lam * (x1 - x0)
    ys = y0 + lam * (y1 - y0)

    length = np.hypot(
        x1 - x0,
        y1 - y0,
    )

    s = lam * length

    values = np.empty(n_samples)

    for k in range(n_samples):
        values[k] = bilinear_sample(
            field,
            x_grid,
            y_grid,
            xs[k],
            ys[k],
        )

    return xs, ys, s, values


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a CFD solution component "
            "along a straight line."
        )
    )

    parser.add_argument(
        "file",
        help="Binary solution file",
    )

    parser.add_argument(
        "--line",
        required=True,
        help=(
            'Line, e.g. "y=0", "x=-1.4", '
            '"y=0.5*x+0.2"'
        ),
    )

    parser.add_argument(
        "--component",
        required=True,
        choices=[
            "rho",
            "mom_x",
            "mom_y",
            "ee",
            "ei",
            "er",
            "ux",
            "uy",
            "speed",
        ],
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of points along line",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV filename",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Read solution
    # --------------------------------------------------------

    data, x, y, time = read_file(
        args.file
    )

    field = get_component(
        data,
        args.component,
    )

    line = parse_line(
        args.line
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    xs, ys, s, values = sample_line(
        field,
        x,
        y,
        line,
        args.samples,
    )

    # --------------------------------------------------------
    # Save ALL samples.
    #
    # Solid / invalid locations remain NaN.
    # This preserves geometric position.
    # --------------------------------------------------------

    if args.output is None:
        basename = os.path.splitext(
            os.path.basename(args.file)
        )[0]

        safe_line = (
            args.line
            .replace("=", "_")
            .replace("*", "")
            .replace("+", "p")
            .replace("-", "m")
        )

        args.output = (
            f"{basename}_"
            f"{args.component}_"
            f"{safe_line}.csv"
        )

    output = np.column_stack(
        [
            s,
            xs,
            ys,
            values,
        ]
    )

    np.savetxt(
        args.output,
        output,
        delimiter=",",
        header=(
            f"s,x,y,{args.component}"
        ),
        comments="",
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    valid = np.isfinite(values)

    print(
        f"file      = {args.file}"
    )

    print(
        f"time      = {time:.8e}"
    )

    print(
        f"grid      = {len(x)} x {len(y)}"
    )

    print(
        f"x range   = "
        f"[{x[0]:.8e}, {x[-1]:.8e}]"
    )

    print(
        f"y range   = "
        f"[{y[0]:.8e}, {y[-1]:.8e}]"
    )

    print(
        f"line      = {args.line}"
    )

    print(
        f"component = {args.component}"
    )

    print(
        f"samples   = {len(values)}"
    )

    print(
        f"valid     = {np.count_nonzero(valid)}"
    )

    if np.any(valid):
        print(
            f"min       = "
            f"{np.nanmin(values):.12e}"
        )

        print(
            f"max       = "
            f"{np.nanmax(values):.12e}"
        )

    print(
        f"CSV       = {args.output}"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    if not args.no_plot:
        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            s,
            values,
            linewidth=1.3,
        )

        ax.set_xlabel(
            "distance along line $s$"
        )

        ax.set_ylabel(
            args.component
        )

        ax.set_title(
            f"{args.component} along "
            f"{args.line}\n"
            f"t={time:.8e}  "
            f"{os.path.basename(args.file)}"
        )

        ax.grid(
            alpha=0.25
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()