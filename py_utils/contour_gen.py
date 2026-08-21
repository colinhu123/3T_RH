import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================

FILE = "data_new/solution_0006.bin"

# !!! 改成 constant.rs 中对应的数值 !!!
GAMMA_E = 5.0 / 3.0
GAMMA_I = 5.0 / 3.0
GAMMA_R = 4.0 / 3.0

# contour 等值线数量
LEVELS = 100


# ============================================================
# Read binary file
# ============================================================

def read_data(filename):

    filename = Path(filename)

    with open(filename, "rb") as f:

        # ---------------- Header ----------------

        magic = f.read(8)

        if magic != b"RH3TBIN1":
            raise ValueError(
                f"Invalid magic number: {magic!r}"
            )

        version = struct.unpack("<I", f.read(4))[0]
        nvar = struct.unpack("<I", f.read(4))[0]

        nx = struct.unpack("<Q", f.read(8))[0]
        ny = struct.unpack("<Q", f.read(8))[0]

        time = struct.unpack("<d", f.read(8))[0]

        if nvar != 8:
            raise ValueError(
                f"Expected nvar = 8, got {nvar}"
            )

        # ---------------- Payload ----------------

        data = np.fromfile(
            f,
            dtype="<f8"
        )

    expected = nx * ny * nvar

    if data.size != expected:
        raise ValueError(
            f"Wrong file size:\n"
            f"expected {expected} doubles\n"
            f"got      {data.size}"
        )

    # Rust:
    #
    # for j in 0..ny
    #     for i in 0..nx
    #
    # 所以 shape = (ny, nx, nvar)

    data = data.reshape(
        ny,
        nx,
        nvar
    )

    result = {
        "x":     data[:, :, 0],
        "y":     data[:, :, 1],
        "rho":   data[:, :, 2],
        "mom_x": data[:, :, 3],
        "mom_y": data[:, :, 4],
        "ee":    data[:, :, 5],
        "ei":    data[:, :, 6],
        "er":    data[:, :, 7],
        "time":  time,
        "nx":    nx,
        "ny":    ny,
    }

    print("=" * 40)
    print(f"File    : {filename}")
    print(f"Version : {version}")
    print(f"Grid    : {nx} x {ny}")
    print(f"Time    : {time:.10e}")
    print("=" * 40)

    return result


# ============================================================
# Primitive quantities
# ============================================================

def compute_primitive(data):

    rho = data["rho"]

    mx = data["mom_x"]
    my = data["mom_y"]

    ee = data["ee"]
    ei = data["ei"]
    er = data["er"]

    # --------------------------------------------------------
    # velocity
    #
    # Rust:
    # u = mom_x / rho
    # v = mom_y / rho
    # --------------------------------------------------------

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):
        u = mx / rho
        v = my / rho

    velocity2 = u**2 + v**2

    # --------------------------------------------------------
    # Exactly reproduce state.rs:
    #
    # pe = (GAMMA_E - 1)
    #      * (ee - rho*(u^2+v^2)/6)
    #
    # pi = (GAMMA_I - 1)
    #      * (ei - rho*(u^2+v^2)/6)
    #
    # pr = (GAMMA_R - 1)
    #      * (er - rho*(u^2+v^2)/6)
    # --------------------------------------------------------

    kinetic_share = (
        rho * velocity2 / 6.0
    )

    pe = (
        (GAMMA_E - 1.0)
        * (ee - kinetic_share)
    )

    pi = (
        (GAMMA_I - 1.0)
        * (ei - kinetic_share)
    )

    pr = (
        (GAMMA_R - 1.0)
        * (er - kinetic_share)
    )

    # total pressure
    p = pe + pi + pr

    return {
        "u": u,
        "v": v,
        "pe": pe,
        "pi": pi,
        "pr": pr,
        "p": p,
    }


# ============================================================
# Plot contour
# ============================================================

def plot_line_contour(
    x,
    y,
    q,
    title,
    filename,
    levels=30
):
    """
    Plot line contours similar to CFD paper figures.
    """

    # 屏蔽 Rust 输出中的非物理区域 NaN
    q = np.ma.masked_invalid(q)

    fig, ax = plt.subplots(figsize=(8, 5))

    # --------------------------------------------------------
    # Line contour
    # --------------------------------------------------------
    cs = ax.contour(
        x,
        y,
        q,
        levels=levels,       # 等值线数量
        colors="red",        # 红色线条
        linewidths=0.7       # 细线
    )

    ax.set_xlabel("X", fontsize=16, fontweight="bold")
    ax.set_ylabel("Y", fontsize=16, fontweight="bold")

    ax.set_title(title, fontsize=14)

    # 保持物理坐标比例
    ax.set_aspect("equal", adjustable="box")

    # 刻度样式
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
        direction="out",
        length=5,
        width=1.0
    )

    # 不显示上、右边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=600,
        bbox_inches="tight"
    )

    print(f"Saved: {filename}")

    plt.show()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    data = read_data(FILE)

    primitive = compute_primitive(data)

    x = data["x"]
    y = data["y"]

    rho = data["rho"]

    pressure = primitive["p"]

    time = data["time"]

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()
    print("Solution range")
    print("-" * 40)

    print(
        f"rho : "
        f"{np.nanmin(rho):.8e} "
        f"-> {np.nanmax(rho):.8e}"
    )

    print(
        f"p   : "
        f"{np.nanmin(pressure):.8e} "
        f"-> {np.nanmax(pressure):.8e}"
    )

    print("-" * 40)

    # --------------------------------------------------------
    # Density contour
    # --------------------------------------------------------

    plot_line_contour(
    data["x"],
    data["y"],
    data["rho"],
    title=f"Density, t = {data['time']:.6f}",
    filename="density_contour.png",
    levels=30
)


    # --------------------------------------------------------
    # Pressure contour
    # --------------------------------------------------------

    plot_line_contour(
    data["x"],
    data["y"],
    primitive["p"],
    title=f"Pressure, t = {data['time']:.6f}",
    filename="pressure_contour.png",
    levels=30
)