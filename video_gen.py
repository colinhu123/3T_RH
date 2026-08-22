import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import glob
import os
import struct


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data_new"
FILE_PATTERN = os.path.join(DATA_DIR, "solution_*.bin")

OUTPUT_VIDEO = "density.mp4"

# Video settings
FPS = 20
DPI = 150

# Plot settings
CMAP = "viridis"

# If None, determine global limits from ALL frames.
# This is strongly recommended for a scientific animation.
VMIN = None
VMAX = None

# Plot range.
# None means use the complete computational domain.
X_LIM = None
Y_LIM = None

# Example for cylinder front region:
#
# X_LIM = (-2.5, 0.0)
# Y_LIM = (-2.5, 2.5)


# ============================================================
# Binary format
# ============================================================

MAGIC = b"RH3TBIN1"
HEADER = struct.Struct("<8sIIQQd")
NVAR = 8


# ============================================================
# Files
# ============================================================

files = sorted(glob.glob(FILE_PATTERN))

if not files:
    raise RuntimeError(
        f"No binary solution files found: {FILE_PATTERN}"
    )

print(f"Found {len(files)} solution files.")


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
            f"Unsupported binary version "
            f"{version} in {filename}"
        )

    if nvar != NVAR:
        raise RuntimeError(
            f"Unexpected variable count {nvar} "
            f"in {filename}; expected {NVAR}"
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

    # Payload:
    #
    # 0 = x
    # 1 = y
    # 2 = rho
    # 3 = mom_x
    # 4 = mom_y
    # 5 = ee
    # 6 = ei
    # 7 = er

    x = np.asarray(data[0, :, 0])
    y = np.asarray(data[:, 0, 1])

    rho = np.ma.masked_invalid(
        np.asarray(data[:, :, 2])
    )

    return x, y, rho, time


# ============================================================
# Determine global density range
#
# IMPORTANT:
#
# Do NOT rescale the colorbar independently for every frame.
# Otherwise a constant physical state can appear to change color.
#
# Use one global [vmin, vmax] for the entire movie.
# ============================================================

if VMIN is None or VMAX is None:
    print("Scanning all frames for global density range...")

    global_min = np.inf
    global_max = -np.inf

    for k, filename in enumerate(files):
        _, _, rho, _ = read_file(filename)

        if rho.count() > 0:
            local_min = rho.min()
            local_max = rho.max()

            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)

        print(
            f"\rScanning {k + 1}/{len(files)}",
            end="",
            flush=True,
        )

    print()

    if VMIN is None:
        VMIN = float(global_min)

    if VMAX is None:
        VMAX = float(global_max)


print(
    f"Density color range: "
    f"[{VMIN:.8e}, {VMAX:.8e}]"
)


# ============================================================
# Read first frame
# ============================================================

x, y, rho, time = read_file(files[0])


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 8)
)

mesh = ax.pcolormesh(
    x,
    y,
    rho,
    shading="auto",
    cmap=CMAP,
    vmin=VMIN,
    vmax=VMAX,
)

colorbar = fig.colorbar(
    mesh,
    ax=ax,
    label=r"$\rho$",
)

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_aspect(
    "equal",
    adjustable="box",
)

if X_LIM is not None:
    ax.set_xlim(*X_LIM)

if Y_LIM is not None:
    ax.set_ylim(*Y_LIM)

title = ax.set_title(
    f"Density\n"
    f"Frame 0/{len(files)-1}  "
    f"t={time:.8e}"
)

plt.tight_layout()


# ============================================================
# Animation update
# ============================================================

def update(frame):
    filename = files[frame]

    _, _, rho, time = read_file(filename)

    # pcolormesh stores the cell values as one flattened array.
    mesh.set_array(
        np.ma.asarray(rho).ravel()
    )

    title.set_text(
        f"Density\n"
        f"Frame {frame}/{len(files)-1}  "
        f"t={time:.8e}  "
        f"{os.path.basename(filename)}"
    )

    print(
        f"\rRendering frame "
        f"{frame + 1}/{len(files)} "
        f"t={time:.8e}",
        end="",
        flush=True,
    )

    return mesh, title


# ============================================================
# Create animation
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=len(files),
    interval=1000 / FPS,
    blit=False,
)


# ============================================================
# MP4 writer
# ============================================================

writer = FFMpegWriter(
    fps=FPS,
    codec="h264",
    bitrate=5000,
    metadata={
        "title": "Cylinder Density",
        "artist": "CFD Solver",
    },
)


# ============================================================
# Save
# ============================================================

print()
print(f"Writing video: {OUTPUT_VIDEO}")

animation.save(
    OUTPUT_VIDEO,
    writer=writer,
    dpi=DPI,
)

print()
print(f"Done: {OUTPUT_VIDEO}")

plt.close(fig)