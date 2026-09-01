import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Make the shared reader importable when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from py_utils import solution_io  # noqa: E402

# ============================================================
# Only completed *.bin files are visible. Rust writes
# *.bin.tmp first and atomically renames it after flush/close.
# ============================================================

def refresh_files():
    return solution_io.refresh_solution_files("data")


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
# Fast binary reader (shared with scripts/live_monitor.py)
# ============================================================

def read_file(filename):
    x, y, time, field = solution_io.read_solution_file(filename)
    return x, y, field["rho"], time


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
        f"{Path(files[step]).name}"
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
