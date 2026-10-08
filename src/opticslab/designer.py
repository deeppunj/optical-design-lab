"""
designer.py  -  STAGE 6b: GUI-independent logic for the lens designer
=====================================================================
Everything the PySide6 app computes lives here (no Qt imports), so it
can be unit-tested. Rows -> surfaces -> rays -> numbers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .glass import CATALOG, LAMBDA_C, LAMBDA_D, LAMBDA_F
from .raytrace import (TraceSurface, at_image_plane, last_vertex_z,
                       rms_spot_radius, trace)

CUSTOM = "Custom"
GLASS_CHOICES: List[str] = list(CATALOG.keys()) + [CUSTOM]
Z_START = -10.0
FILE_VERSION = 1


@dataclass
class SurfaceRow:
    """One line of the prescription table."""
    radius: float = 0.0            # mm, 0 or inf = flat
    thickness: float = 5.0         # mm to next surface (last = to image)
    glass: str = "AIR"             # medium AFTER the surface (catalog name or Custom)
    n_custom: float = 1.0          # used only when glass == Custom

    def index(self, wavelength_nm: float) -> float:
        if self.glass == CUSTOM:
            return float(self.n_custom)
        return CATALOG[self.glass].n(wavelength_nm)


# ------------------------------------------------------------ validation
def validate(rows: Sequence[SurfaceRow]) -> None:
    if not rows:
        raise ValueError("Add at least one surface")
    for i, r in enumerate(rows, 1):
        if r.glass != CUSTOM and r.glass not in CATALOG:
            raise ValueError(f"Row {i}: unknown glass '{r.glass}'")
        if not np.isfinite(r.thickness):
            raise ValueError(f"Row {i}: thickness must be finite")
        if r.radius != r.radius:
            raise ValueError(f"Row {i}: radius is not a number")
        if r.glass == CUSTOM and not (np.isfinite(r.n_custom) and r.n_custom >= 1.0):
            raise ValueError(f"Row {i}: custom index must be >= 1")


def build_surfaces(rows: Sequence[SurfaceRow], wavelength_nm: float) -> List[TraceSurface]:
    out = []
    for r in rows:
        R = np.inf if (r.radius == 0 or not np.isfinite(r.radius)) else r.radius
        out.append(TraceSurface(R, r.index(wavelength_nm), r.thickness))
    return out


# --------------------------------------------------------------- paraxial
def paraxial(rows: Sequence[SurfaceRow], wavelength_nm: float,
             n_start: float = 1.0) -> Tuple[float, float]:
    """Paraxial (EFL, BFL) in mm from a parallel ray with y1 = 1."""
    y, u, n_prev = 1.0, 0.0, n_start
    for i, r in enumerate(rows):
        c = 0.0 if (r.radius == 0 or not np.isfinite(r.radius)) else 1.0 / r.radius
        n = r.index(wavelength_nm)
        u = (n_prev * u - y * (n - n_prev) * c) / n
        n_prev = n
        if i < len(rows) - 1:
            y += r.thickness * u
    if abs(u) < 1e-12:
        return float("inf"), float("inf")
    return -1.0 / u, -y / u


def chromatic_table(rows: Sequence[SurfaceRow]) -> List[Dict[str, float]]:
    """EFL/BFL at the F, d and C lines + axial colour (BFL_F - BFL_C)."""
    out = []
    for name, wl in (("F", LAMBDA_F), ("d", LAMBDA_D), ("C", LAMBDA_C)):
        efl, bfl = paraxial(rows, wl)
        out.append({"line": name, "wavelength_nm": wl, "efl": efl, "bfl": bfl})
    return out


def axial_colour(rows: Sequence[SurfaceRow]) -> float:
    t = chromatic_table(rows)
    return t[0]["bfl"] - t[2]["bfl"]


# ------------------------------------------------------------------ rays
def hex_pupil(epd: float, rings: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    px, py = [0.0], [0.0]
    for r in range(1, rings + 1):
        a = np.linspace(0, 2 * np.pi, 6 * r, endpoint=False)
        px += list(r / rings * epd / 2 * np.cos(a))
        py += list(r / rings * epd / 2 * np.sin(a))
    return np.array(px), np.array(py)


def make_beam(px, py, field_deg: float, z_start: float = Z_START):
    """Parallel beam tilted by field_deg in y-z; crosses z=0 at (px, py)."""
    th = np.deg2rad(field_deg)
    d = np.array([0.0, np.sin(th), np.cos(th)])
    n = len(px)
    dirs = np.tile(d, (n, 1))
    back = -z_start / d[2]
    origins = np.stack([np.asarray(px) - d[0] * back,
                        np.asarray(py) - d[1] * back,
                        np.full(n, z_start)], axis=1)
    return origins, dirs


def trace_bundle(rows, wavelength_nm, px, py, field_deg):
    surfs = build_surfaces(rows, wavelength_nm)
    o, d = make_beam(px, py, field_deg)
    pts, dd = trace(o, d, surfs)
    return surfs, o, pts, dd


def image_plane_z(rows) -> float:
    return float(sum(r.thickness for r in rows))


def best_focus_z(pts, dd, z_lo: float, z_hi: float) -> float:
    """Image-plane z with minimum RMS spot (coarse scan + fine scan)."""
    zs = np.linspace(z_lo, z_hi, 600)
    vals = np.array([rms_spot_radius(at_image_plane(pts, dd, z)[:, :2]) for z in zs])
    if np.all(np.isnan(vals)):
        return float(z_lo)
    i = int(np.nanargmin(vals))
    zs2 = np.linspace(zs[max(i - 1, 0)], zs[min(i + 1, len(zs) - 1)], 200)
    v2 = np.array([rms_spot_radius(at_image_plane(pts, dd, z)[:, :2]) for z in zs2])
    return float(zs2[int(np.nanargmin(v2))])


def focus_search_range(rows, bfl: float) -> Tuple[float, float]:
    z_last = last_vertex_z([TraceSurface(1.0, 1.0, r.thickness) for r in rows])
    span = max(5 * abs(bfl), 50.0) if np.isfinite(bfl) else 200.0
    return z_last + 0.05, z_last + span


def sag_profile(R: float, h: np.ndarray) -> np.ndarray:
    if R == 0 or not np.isfinite(R):
        return np.zeros_like(h)
    return R - np.sign(R) * np.sqrt(np.maximum(R * R - h * h, 0.0))


# ------------------------------------------------------------- file I/O
def save_design(path, rows, epd, field_deg, wavelength_nm) -> None:
    data = {"version": FILE_VERSION, "epd": epd, "field_deg": field_deg,
            "wavelength_nm": wavelength_nm,
            "rows": [asdict(r) for r in rows]}
    Path(path).write_text(json.dumps(data, indent=2))


def load_design(path):
    data = json.loads(Path(path).read_text())
    if data.get("version") != FILE_VERSION:
        raise ValueError(f"Unsupported file version: {data.get('version')}")
    rows = [SurfaceRow(**r) for r in data["rows"]]
    validate(rows)
    return rows, float(data["epd"]), float(data["field_deg"]), float(data["wavelength_nm"])


# -------------------------------------------------------------- examples
EXAMPLES: Dict[str, List[SurfaceRow]] = {
    "Singlet N-BK7 (50/-50)": [
        SurfaceRow(50.0, 5.0, "N-BK7"), SurfaceRow(-50.0, 45.0, "AIR")],
    # R2, R3 solved so that EFL(d) = 100 mm and BFL(F) = BFL(C)
    "Achromat N-BK7/N-F2": [
        SurfaceRow(62.5, 4.0, "N-BK7"), SurfaceRow(-33.9323, 2.5, "N-F2"),
        SurfaceRow(-127.7612, 97.0, "AIR")],
    "Plano-convex fused silica": [
        SurfaceRow(0.0, 4.0, "FUSED-SILICA"), SurfaceRow(-30.0, 55.0, "AIR")],
}
