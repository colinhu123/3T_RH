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

# Only completed *.bin files are visible. Rust writes *.bin.tmp first
# and atomically renames it after flush/close.
def refresh_files():
    return sorted(glob.glob("data_new/solution_*.bin"))

files = refresh_files()
if len(files) == 0:
    raise RuntimeError("No binary solution files found: data/solution_*.bin")

step = 0

# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
colorbar = None

# ============================================================
# Fast binary reader
# ============================================================

def read_file(filename):
    with open(filename, "rb") as f:
        raw = f.read(HEADER.size)

    if len(raw) != HEADER.size:
        raise RuntimeError(f"Incomplete binary header in {filename}")

    magic, version, nvar, nx, ny, time = HEADER.unpack(raw)

    if magic != MAGIC:
        raise RuntimeError(
            f"Wrong file magic in {filename}: {magic!r}; expected {MAGIC!r}"
        )
    if version != 1:
        raise RuntimeError(f"Unsupported binary version {version} in {filename}")
    if nvar != NVAR:
        raise RuntimeError(
            f"Unexpected variable count {nvar} in {filename}; expected {NVAR}"
        )

    expected_bytes = HEADER.size + nx * ny * nvar * 8
    actual_bytes = os.path.getsize(filename)

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"Binary size mismatch in {filename}: "
            f"expected={expected_bytes} bytes, actual={actual_bytes} bytes"
        )

    # np.memmap avoids parsing text and avoids copying the entire file.
    data = np.memmap(
        filename,
        dtype="<f8",
        mode="r",
        offset=HEADER.size,
        shape=(ny, nx, nvar),
        order="C",
    )

    # Payload columns:
    # 0=x, 1=y, 2=rho, 3=mom_x, 4=mom_y, 5=ee, 6=ei, 7=er
    x = np.asarray(data[0, :, 0])
    y = np.asarray(data[:, 0, 1])

    # Mask outside-polygon cells written as NaN by Rust.
    rho = np.ma.masked_invalid(data[:, :, 2])

    return x, y, rho, time


# ============================================================
# Update plot
# ============================================================

def update():
    global step, colorbar, files

    # Allow the visualizer to stay open while the solver is running.
    files = refresh_files()

    if not files:
        return

    step = min(step, len(files) - 1)

    x, y, rho, time = read_file(files[step])

    ax.clear()

    mesh = ax.pcolormesh(
        x,
        y,
        rho,
        shading="auto",
        cmap="viridis",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Density\n"
        f"Frame {step}/{len(files)-1}  "
        f"t={time:.8e}  "
        f"{os.path.basename(files[step])}"
    )
    ax.set_aspect("equal")

    if colorbar is None:
        colorbar = fig.colorbar(mesh, ax=ax, label=r"$\rho$")
    else:
        colorbar.update_normal(mesh)

    fig.canvas.draw_idle()


# ============================================================
# Keyboard control
# ============================================================

def keyboard(event):
    global step, files

    files = refresh_files()
    if not files:
        return

    if event.key == "right":
        step = min(step + 1, len(files) - 1)
        update()

    elif event.key == "left":
        step = max(step - 1, 0)
        update()

    elif event.key == "up":
        step = min(step + 10, len(files) - 1)
        update()

    elif event.key == "down":
        step = max(step - 10, 0)
        update()

    elif event.key == "end":
        step = len(files) - 1
        update()

    elif event.key == "home":
        step = 0
        update()

    elif event.key == "q":
        plt.close()


fig.canvas.mpl_connect("key_press_event", keyboard)

update()
plt.tight_layout()
plt.show()