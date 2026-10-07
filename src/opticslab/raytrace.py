"""
raytrace.py  -  LESSON 11: Real (exact) ray tracing in 3D
=====================================================================
THEORY
------
The matrix method (Lessons 1-10) is PARAXIAL: it assumes small angles,
so it can never show aberrations. Here we trace EXACT rays with Snell's
law, so a lens finally shows its real faults.

A ray is  point o = (x, y, z)  +  unit direction d = (dx, dy, dz).
Optical axis = z. For each surface (vertex on the axis at z_v):

  1. INTERSECT. Sphere centre c = (0, 0, z_v + R).
        |o + t d - c|^2 = R^2   ->  t^2 + 2 B t + C = 0
        B = d.(o-c),  C = |o-c|^2 - R^2,  disc = B^2 - C
        t = -B - sign(R) * sqrt(disc)      (picks the cap near the vertex)
     Flat surface (R = inf):  t = (z_v - o_z) / d_z.
  2. NORMAL  N = (p - c)/R, flipped so that d.N < 0.
  3. REFRACT (vector Snell's law), mu = n1/n2, cos_i = -d.N:
        d' = mu d + (mu cos_i - sqrt(1 - mu^2 (1 - cos_i^2))) N
     If the square-root argument < 0: total internal reflection -> NaN.
  4. Move on to the next vertex: z_v += thickness.

ABERRATIONS you will see
  - SPHERICAL: rays at the edge focus closer than rays near the axis.
  - SPOT DIAGRAM: where rays land on the image plane (ideal = one dot).
  - RAY FAN: transverse error (y at image - chief y) vs pupil height.
  - COMA / ASTIGMATISM: appear when you tilt the beam (field angle).
=====================================================================
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize_scalar

from .glass import Glass


@dataclass(frozen=True)
class TraceSurface:
    """One surface. `n_after` = index of the medium AFTER it.
    `thickness` = distance along z to the NEXT surface (mm)."""
    radius: float
    n_after: float
    thickness: float


def doublet_prescription(r1, r2, r3, t1, t2, crown: Glass, flint: Glass,
                         wavelength_nm: float) -> List[TraceSurface]:
    n1, n2 = crown.n(wavelength_nm), flint.n(wavelength_nm)
    return [TraceSurface(r1, n1, t1),
            TraceSurface(r2, n2, t2),
            TraceSurface(r3, 1.0, 0.0)]


def singlet_prescription(r1, r2, t, glass: Glass,
                         wavelength_nm: float) -> List[TraceSurface]:
    n = glass.n(wavelength_nm)
    return [TraceSurface(r1, n, t), TraceSurface(r2, 1.0, 0.0)]


def collimated_beam(diameter, n_rays=41, field_deg=0.0, z_start=-5.0,
                    pattern="line") -> Tuple[np.ndarray, np.ndarray]:
    """Parallel rays, tilted by field_deg in the y-z plane.
    pattern 'line' = ray fan in y (x = 0);  'grid' = filled circle.
    The beam crosses the first vertex plane (z = 0) at the pupil coordinates."""
    r = diameter / 2.0
    if pattern == "line":
        ys = np.linspace(-r, r, n_rays)
        xs = np.zeros_like(ys)
    elif pattern == "grid":
        g = np.linspace(-r, r, n_rays)
        X, Y = np.meshgrid(g, g)
        mask = X**2 + Y**2 <= r**2
        xs, ys = X[mask], Y[mask]
    else:
        raise ValueError("pattern must be 'line' or 'grid'")
    th = np.deg2rad(field_deg)
    d = np.array([0.0, np.sin(th), np.cos(th)])
    dirs = np.tile(d, (len(xs), 1))
    back = (0.0 - z_start) / d[2]
    origins = np.stack([xs - d[0] * back,
                        ys - d[1] * back,
                        np.full_like(xs, z_start)], axis=1)
    return origins, dirs


def _intersect(o, d, z_v, R):
    if np.isinf(R):
        t = (z_v - o[:, 2]) / d[:, 2]
        N = np.zeros_like(o)
        N[:, 2] = 1.0
        return o + t[:, None] * d, N
    c = np.array([0.0, 0.0, z_v + R])
    oc = o - c
    B = np.sum(d * oc, axis=1)
    C = np.sum(oc * oc, axis=1) - R * R
    disc = B * B - C
    with np.errstate(invalid="ignore"):
        t = -B - np.sign(R) * np.sqrt(disc)       # NaN if the ray misses
    p = o + t[:, None] * d
    N = (p - c) / abs(R)
    return p, N


def trace(origins, dirs, surfaces: List[TraceSurface], n_start=1.0):
    """Trace rays through all surfaces.
    Returns (points, final_directions); points has shape (n_surf, N, 3).
    Rays that miss a surface or suffer TIR become NaN."""
    o, d = origins.copy(), dirs.copy()
    n_prev, z_v = n_start, 0.0
    pts = []
    for s in surfaces:
        p, N = _intersect(o, d, z_v, s.radius)
        flip = np.sum(d * N, axis=1) > 0
        N[flip] *= -1.0
        cos_i = -np.sum(d * N, axis=1)
        mu = n_prev / s.n_after
        k = 1.0 - mu * mu * (1.0 - cos_i**2)
        with np.errstate(invalid="ignore"):
            root = np.where(k >= 0, np.sqrt(np.where(k >= 0, k, 0.0)), np.nan)
        d = mu * d + (mu * cos_i - root)[:, None] * N
        o = p
        pts.append(p)
        n_prev = s.n_after
        z_v += s.thickness
    return np.array(pts), d


def last_vertex_z(surfaces: List[TraceSurface]) -> float:
    return float(sum(s.thickness for s in surfaces[:-1]))


def at_image_plane(points, dirs, z_image):
    """Propagate from the last surface to the plane z = z_image."""
    p = points[-1]
    t = (z_image - p[:, 2]) / dirs[:, 2]
    return p + t[:, None] * dirs


def rms_spot_radius(xy: np.ndarray) -> float:
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) == 0:
        return float("nan")
    c = xy.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((xy - c) ** 2, axis=1))))


def marginal_axis_crossing(surfaces, height, n_start=1.0):
    """z (measured from the last vertex) where a collimated ray at `height`
    crosses the axis. A tiny height gives the paraxial focus."""
    o, d = collimated_beam(2 * height, n_rays=2)
    o, d = o[-1:], d[-1:]
    pts, dd = trace(o, d, surfaces, n_start)
    p = pts[-1][0]
    t = -p[1] / dd[0][1]
    return float(p[2] + t * dd[0][2] - last_vertex_z(surfaces))


def best_focus(origins, dirs, surfaces, span=0.15):
    """Image-plane z (mm from the last vertex) with the smallest RMS spot.
    Searches +-span * paraxial focus distance around the paraxial focus."""
    pts, d = trace(origins, dirs, surfaces)
    z0 = last_vertex_z(surfaces)
    z_par = marginal_axis_crossing(surfaces, 1e-3)

    def f(dz):
        return rms_spot_radius(at_image_plane(pts, d, z0 + dz)[:, :2])

    grid = np.linspace(z_par * (1 - span), z_par * (1 + span), 301)
    vals = [f(g) for g in grid]
    i = int(np.nanargmin(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-9})
    return float(res.x), float(res.fun)


def plot_spot_and_fan(surfaces, diameter=10.0, field_deg=0.0, title=""):
    """Spot diagram (left) and y ray fan (right) at best focus."""
    import matplotlib.pyplot as plt
    o, d = collimated_beam(diameter, 41, field_deg, pattern="grid")
    dz, rms = best_focus(o, d, surfaces)
    z_img = last_vertex_z(surfaces) + dz
    pts, dd = trace(o, d, surfaces)
    xy = at_image_plane(pts, dd, z_img)[:, :2]

    of, df = collimated_beam(diameter, 81, field_deg, pattern="line")
    pf, ddf = trace(of, df, surfaces)
    yimg = at_image_plane(pf, ddf, z_img)[:, 1]
    yp = np.linspace(-diameter / 2, diameter / 2, len(yimg))
    chief = yimg[len(yimg) // 2]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].scatter(xy[:, 0] * 1000, xy[:, 1] * 1000, s=6)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (um)")
    ax[0].set_ylabel("y (um)")
    ax[0].set_title(f"Spot, RMS = {rms * 1000:.2f} um")
    ax[1].plot(yp, (yimg - chief) * 1000)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("Pupil y (mm)")
    ax[1].set_ylabel("Ray error (um)")
    ax[1].set_title("Ray fan")
    fig.suptitle(title or f"Field {field_deg} deg, best focus at +{dz:.3f} mm")
    fig.tight_layout()
    return fig
