import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import struct


# ============================================================
# Binary solution format
# ============================================================

MAGIC = b"RH3TBIN1"
HEADER = struct.Struct("<8sIIQQd")
NVAR = 8


# ============================================================
# User settings
# ============================================================

DATA_PATTERN = "data/solution_*.bin"

# Your current uniform-grid spacing.
# Change this if necessary.
H = 1.0 / 40.0


# ============================================================
# Binary reader
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
            f"Wrong file magic in {filename}: "
            f"{magic!r}; expected {MAGIC!r}"
        )

    if version != 1:
        raise RuntimeError(
            f"Unsupported binary version {version} "
            f"in {filename}"
        )

    if nvar != NVAR:
        raise RuntimeError(
            f"Unexpected variable count {nvar}; "
            f"expected {NVAR}"
        )

    expected_bytes = (
        HEADER.size
        + nx * ny * nvar * 8
    )

    actual_bytes = os.path.getsize(filename)

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Binary size mismatch in {filename}: "
            f"expected={expected_bytes}, "
            f"actual={actual_bytes}"
        )

    data = np.memmap(
        filename,
        dtype="<f8",
        mode="r",
        offset=HEADER.size,
        shape=(ny, nx, nvar),
        order="C",
    )

    # --------------------------------------------------------
    # Binary layout:
    #
    # 0 = x
    # 1 = y
    # 2 = rho
    # 3 = mom_x
    # 4 = mom_y
    # 5 = ee
    # 6 = ei
    # 7 = er
    # --------------------------------------------------------

    x = np.asarray(data[:, :, 0])
    y = np.asarray(data[:, :, 1])

    rho = np.asarray(data[:, :, 2])

    mom_x = np.asarray(data[:, :, 3])
    mom_y = np.asarray(data[:, :, 4])

    ee = np.asarray(data[:, :, 5])
    ei = np.asarray(data[:, :, 6])
    er = np.asarray(data[:, :, 7])

    return (
        x,
        y,
        rho,
        mom_x,
        mom_y,
        ee,
        ei,
        er,
        time,
    )


# ============================================================
# Find files
# ============================================================

files = sorted(
    glob.glob(DATA_PATTERN)
)

if len(files) == 0:
    raise RuntimeError(
        f"No binary solution files found: "
        f"{DATA_PATTERN}"
    )

print(f"Found {len(files)} solution files")


# ============================================================
# Storage
# ============================================================

times = []

Ee_int = []
Ei_int = []
Er_int = []

Etot_int = []

# Global integral differences

Dei_int = []
Der_int = []
Dir_int = []

# Maximum local differences

Dei_max = []
Der_max = []
Dir_max = []


# ============================================================
# Loop over solution files
# ============================================================

cell_area = H * H


for k, filename in enumerate(files):

    (
        x,
        y,
        rho,
        mom_x,
        mom_y,
        ee,
        ei,
        er,
        time,
    ) = read_file(filename)

    # --------------------------------------------------------
    # Fluid mask
    #
    # Rust writes cells outside the physical domain as NaN.
    # --------------------------------------------------------

    valid = (
        np.isfinite(rho)
        & np.isfinite(ee)
        & np.isfinite(ei)
        & np.isfinite(er)
    )

    if not np.any(valid):
        print(
            f"WARNING: no valid cells in {filename}"
        )
        continue

    ee_v = ee[valid]
    ei_v = ei[valid]
    er_v = er[valid]

    # --------------------------------------------------------
    # Domain integrals
    # --------------------------------------------------------

    ee_sum = (
        np.sum(ee_v)
        * cell_area
    )

    ei_sum = (
        np.sum(ei_v)
        * cell_area
    )

    er_sum = (
        np.sum(er_v)
        * cell_area
    )

    etot_sum = (
        ee_sum
        + ei_sum
        + er_sum
    )

    # --------------------------------------------------------
    # Integral difference modes
    #
    # These detect global bias between the three energies.
    # --------------------------------------------------------

    dei_sum = (
        np.sum(ee_v - ei_v)
        * cell_area
    )

    der_sum = (
        np.sum(ee_v - er_v)
        * cell_area
    )

    dir_sum = (
        np.sum(ei_v - er_v)
        * cell_area
    )

    # --------------------------------------------------------
    # Maximum local energy splitting
    # --------------------------------------------------------

    dei_max = np.max(
        np.abs(ee_v - ei_v)
    )

    der_max = np.max(
        np.abs(ee_v - er_v)
    )

    dir_max = np.max(
        np.abs(ei_v - er_v)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    times.append(time)

    Ee_int.append(ee_sum)
    Ei_int.append(ei_sum)
    Er_int.append(er_sum)

    Etot_int.append(etot_sum)

    Dei_int.append(dei_sum)
    Der_int.append(der_sum)
    Dir_int.append(dir_sum)

    Dei_max.append(dei_max)
    Der_max.append(der_max)
    Dir_max.append(dir_max)

    print(
        f"{k:5d}  "
        f"t={time:12.6e}  "
        f"Ee={ee_sum:14.7e}  "
        f"Ei={ei_sum:14.7e}  "
        f"Er={er_sum:14.7e}  "
        f"max_split={max(dei_max, der_max, dir_max):12.5e}"
    )


# ============================================================
# Convert to numpy arrays
# ============================================================

times = np.asarray(times)

Ee_int = np.asarray(Ee_int)
Ei_int = np.asarray(Ei_int)
Er_int = np.asarray(Er_int)

Etot_int = np.asarray(Etot_int)

Dei_int = np.asarray(Dei_int)
Der_int = np.asarray(Der_int)
Dir_int = np.asarray(Dir_int)

Dei_max = np.asarray(Dei_max)
Der_max = np.asarray(Der_max)
Dir_max = np.asarray(Dir_max)


# ============================================================
# Figure 1:
# Integral of each energy variable
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    times,
    Ee_int,
    label=r"$\int E_e\,d\Omega$",
)

plt.plot(
    times,
    Ei_int,
    label=r"$\int E_i\,d\Omega$",
)

plt.plot(
    times,
    Er_int,
    label=r"$\int E_r\,d\Omega$",
)

plt.xlabel("time")
plt.ylabel("domain integral")

plt.title(
    "Domain integrals of the three energy variables"
)

plt.grid(True)
plt.legend()
plt.tight_layout()


# ============================================================
# Figure 2:
# Sum of three energy variables
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    times,
    Etot_int,
)

plt.xlabel("time")

plt.ylabel(
    r"$\int (E_e+E_i+E_r)\,d\Omega$"
)

plt.title(
    "Integral of total three-energy sum"
)

plt.grid(True)
plt.tight_layout()


# ============================================================
# Figure 3:
# GLOBAL integral energy splitting
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    times,
    np.abs(Dei_int),
    label=r"$|\int(E_e-E_i)d\Omega|$",
)

plt.plot(
    times,
    np.abs(Der_int),
    label=r"$|\int(E_e-E_r)d\Omega|$",
)

plt.plot(
    times,
    np.abs(Dir_int),
    label=r"$|\int(E_i-E_r)d\Omega|$",
)

plt.yscale("log")

plt.xlabel("time")
plt.ylabel("absolute integral energy split")

plt.title(
    "Global three-energy symmetry diagnostic"
)

plt.grid(True)
plt.legend()
plt.tight_layout()


# ============================================================
# Figure 4:
# LOCAL maximum energy splitting
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    times,
    Dei_max,
    label=r"$\max|E_e-E_i|$",
)

plt.plot(
    times,
    Der_max,
    label=r"$\max|E_e-E_r|$",
)

plt.plot(
    times,
    Dir_max,
    label=r"$\max|E_i-E_r|$",
)

plt.yscale("log")

plt.xlabel("time")
plt.ylabel("maximum absolute energy split")

plt.title(
    "Local three-energy symmetry diagnostic"
)

plt.grid(True)
plt.legend()
plt.tight_layout()


plt.show()