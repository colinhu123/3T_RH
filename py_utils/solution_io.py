"""Shared reader for the Rust solver's binary solution files.

This logic is validated by the standalone viewer ``visualize_sol.py`` and is
now shared between that tool and ``scripts/live_monitor.py`` so both stay in
sync with the binary layout written by ``src/io.rs``.

Binary layout (little endian), written by ``io::save_data``:

  header:
    [8]u8  magic = "RH3TBIN1"
    u32    version (= 1)
    u32    nvar   (= 8)
    u64    nx
    u64    ny
    f64    time

  payload, j-major / i-fastest (i changes fastest), repeated nx*ny times:
    f64 x, y, rho, mom_x, mom_y, ee, ei, er

Cells outside the physical (fluid) domain are stored as NaN by the writer,
so consumers mask them with ``np.ma.masked_invalid``.
"""

import os
import struct
from pathlib import Path

import numpy as np

MAGIC = b"RH3TBIN1"
HEADER = struct.Struct("<8sIIQQd")
NVAR = 8

COL_X = 0
COL_Y = 1
COL_RHO = 2
COL_MOM_X = 3
COL_MOM_Y = 4
COL_EE = 5
COL_EI = 6
COL_ER = 7

# variable name -> payload column index
COLUMNS = {
    "rho": COL_RHO,
    "mom_x": COL_MOM_X,
    "mom_y": COL_MOM_Y,
    "ee": COL_EE,
    "ei": COL_EI,
    "er": COL_ER,
}


class SolutionReadError(Exception):
    """Raised when a binary solution file cannot be validated or read."""


def refresh_solution_files(solution_dir):
    """Return sorted ``Path`` objects for completed ``solution_*.bin`` files.

    Only fully-renamed ``*.bin`` files are matched; the writer's in-progress
    ``*.bin.tmp`` files are never considered.
    """
    return sorted(Path(solution_dir).glob("solution_*.bin"))


def read_solution_file(filename):
    """Read one snapshot.

    Returns ``(x, y, time, field)`` where ``x``/``y`` are the 1-D Cartesian
    coordinates and ``field`` is a dict ``name -> (ny, nx) masked array``
    (masked outside the fluid domain). Raises :class:`SolutionReadError` on
    any validation failure.
    """
    filename = Path(filename)

    with open(filename, "rb") as f:
        raw = f.read(HEADER.size)

    if len(raw) != HEADER.size:
        raise SolutionReadError(f"incomplete header in {filename}")

    magic, version, nvar, nx, ny, time = HEADER.unpack(raw)

    if magic != MAGIC:
        raise SolutionReadError(
            f"bad magic in {filename}: {magic!r}; expected {MAGIC!r}"
        )
    if version != 1:
        raise SolutionReadError(f"unsupported version {version} in {filename}")
    if nvar != NVAR:
        raise SolutionReadError(
            f"unexpected variable count {nvar} in {filename}; expected {NVAR}"
        )

    expected_bytes = HEADER.size + nx * ny * nvar * 8
    actual_bytes = os.path.getsize(filename)
    if actual_bytes != expected_bytes:
        raise SolutionReadError(
            f"size mismatch in {filename}: expected {expected_bytes} bytes, "
            f"got {actual_bytes} bytes"
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

    # x varies along i (row j=0), y along j (column i=0): valid because the
    # grid is Cartesian and uniform.
    x = np.asarray(data[0, :, COL_X])
    y = np.asarray(data[:, 0, COL_Y])

    field = {
        name: np.ma.masked_invalid(data[:, :, col])
        for name, col in COLUMNS.items()
    }

    return x, y, time, field
