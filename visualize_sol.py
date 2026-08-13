import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ============================================================
# Find all solution files
# ============================================================

files = sorted(
    glob.glob("data/solution_*.dat")
)

if len(files) == 0:
    raise RuntimeError("No solution files found")

step = 0

# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
colorbar = None

# ============================================================
# Read 2D solution
# ============================================================

def read_file(filename):

    data = pd.read_csv(filename)

    # Remove possible empty rows
    data = data.dropna()

    x = data["x"].values
    y = data["y"].values
    rho = data["rho"].values

    # Unique grid coordinates
    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    nx = len(x_unique)
    ny = len(y_unique)

    if nx * ny != len(rho):
        raise RuntimeError(
            f"Grid size mismatch in {filename}: "
            f"nx={nx}, ny={ny}, "
            f"nx*ny={nx*ny}, rows={len(rho)}"
        )

    # Your Rust save_data loops:
    #
    # for j in 0..ny {
    #     for i in 0..nx {
    #
    # so x changes fastest.
    #
    # Therefore reshape as (ny, nx).

    rho = rho.reshape(ny, nx)

    return x_unique, y_unique, rho


# ============================================================
# Update plot
# ============================================================

def update():
    global step, colorbar

    x, y, rho = read_file(files[step])

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
        f"Step {step}/{len(files)-1}  "
        f"{os.path.basename(files[step])}"
    )

    ax.set_aspect("equal")

    # Create colorbar only once.
    # For later time steps, connect the existing colorbar
    # to the new mesh.
    if colorbar is None:
        colorbar = fig.colorbar(
            mesh,
            ax=ax,
            label=r"$\rho$",
        )
    else:
        colorbar.update_normal(mesh)

    fig.canvas.draw_idle()


# ============================================================
# Keyboard control
# ============================================================

def keyboard(event):

    global step

    # Next time step
    if event.key == "right":

        if step < len(files) - 1:
            step += 1

        update()

    # Previous time step
    elif event.key == "left":

        if step > 0:
            step -= 1

        update()

    # Jump forward 10 steps
    elif event.key == "up":

        step = min(
            step + 10,
            len(files) - 1
        )

        update()

    # Jump backward 10 steps
    elif event.key == "down":

        step = max(
            step - 10,
            0
        )

        update()

    # Quit
    elif event.key == "q":

        plt.close()


# ============================================================
# Register keyboard
# ============================================================

fig.canvas.mpl_connect(
    "key_press_event",
    keyboard
)


# ============================================================
# Initial plot
# ============================================================

update()

plt.tight_layout()
plt.show()