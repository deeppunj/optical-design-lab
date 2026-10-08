"""
optimizer.py  -  LESSON 12 / STAGE 7: Automatic lens optimisation
=====================================================================
GUI-independent (no Qt imports) so that it can be unit-tested.
It re-uses the code you already have:
    designer.SurfaceRow / trace_bundle / paraxial / axial_colour / hex_pupil
    raytrace.at_image_plane

THEORY
------
Optimising a lens = moving a few numbers (the VARIABLES: radii, thicknesses)
so that one number (the MERIT FUNCTION) becomes as small as possible.

Zemax-style merit functions are SUMS OF SQUARES:

        MF^2  =  sum_k  ( w_k * e_k )^2

where every e_k is one ERROR (an "operand"): how far something is from what
we want.  Because it is a sum of squares, we do not need a general
minimiser.  We hand the whole list of errors e_k to a LEAST-SQUARES solver
(scipy.optimize.least_squares).  It linearises the errors with a Jacobian
J = d e / d x  and repeatedly solves the DAMPED normal equations

        (J^T J + lambda * I) dx = - J^T e          <- 'damped least squares'

which is exactly what Zemax calls the DLS optimiser.

OUR ERRORS (operands)
  1. SPOT: for every field, every wavelength (F, d, C) and every ray in the
     pupil, the distance of the ray from the d-line centroid on the image plane
     (micrometres).  Squared and summed this is the RMS spot size, and since
     the F and C rays are measured from the *d* centroid, colour errors (both
     axial and lateral) are punished automatically.
  2. EFL: effective focal length minus its target (optional).
  3. AXIAL COLOUR: BFL(F) - BFL(C) (paraxial; target = 0, optional).
  4. CONSTRAINTS: glass centre thickness >= min_center, glass edge thickness
     >= min_edge.  They are 'hinge' errors: zero when satisfied, growing
     linearly when violated.  Without them the optimiser happily makes lenses
     with negative thickness, which cannot be manufactured.
  5. BOUNDS on every variable (e.g. |R| >= semi-aperture) are handled by the
     solver itself ('trf' method) - they can never be violated.

PARAMETRISATION TRICK: radii are optimised as CURVATURE c = 1/R.  A flat
surface is c = 0 (a smooth point), whereas R = infinity cannot be
differentiated.  We also scale every variable to order 1
(curvature x 100 mm, thickness / 10 mm) so the solver treats them equally.

LIMITS (be honest!)
  * LOCAL optimiser: it finds the nearest minimum, so it needs a sensible
    starting design.  Zemax has a 'global search' for this; we do not.
  * Spherical surfaces only; no glass substitution; no vignetting.
=====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from . import designer as dz
from .glass import LAMBDA_C, LAMBDA_D, LAMBDA_F
from .raytrace import at_image_plane

CURV_SCALE = 0.01      # 1/mm   -> curvature variables are divided by this
THICK_SCALE = 10.0     # mm     -> thickness variables are divided by this
FLAT_EPS = 1e-9        # |curvature| below this is written back as a flat (0)


# ===================================================================
#  Variables
# ===================================================================
@dataclass(frozen=True)
class Variable:
    """One number the optimiser may change.

    row   : index into the list of SurfaceRow (0 = first surface)
    kind  : 'curvature' (bounds in 1/mm) or 'thickness' (bounds in mm)
    lo,hi : bounds.  The solver can never leave [lo, hi].
    """
    row: int
    kind: str
    lo: float
    hi: float

    def __post_init__(self):
        if self.kind not in ("curvature", "thickness"):
            raise ValueError("kind must be 'curvature' or 'thickness'")
        if not self.lo < self.hi:
            raise ValueError("lower bound must be smaller than upper bound")

    @property
    def scale(self) -> float:
        return CURV_SCALE if self.kind == "curvature" else THICK_SCALE

    def get(self, rows: Sequence[dz.SurfaceRow]) -> float:
        r = rows[self.row]
        if self.kind == "thickness":
            return float(r.thickness)
        return 0.0 if (r.radius == 0 or not np.isfinite(r.radius)) else 1.0 / r.radius

    def put(self, row: dz.SurfaceRow, value: float) -> dz.SurfaceRow:
        if self.kind == "thickness":
            return replace(row, thickness=float(value))
        if abs(value) < FLAT_EPS:
            return replace(row, radius=0.0)
        return replace(row, radius=float(1.0 / value))


def make_variables(rows: Sequence[dz.SurfaceRow], radius_rows: Sequence[int] = (),
                   thickness_rows: Sequence[int] = (), r_min: float = 10.0,
                   t_min: float = 0.5, t_max: float = 300.0) -> List[Variable]:
    """Convenience: build variables from lists of row indices
    (this is what the GUI check-boxes will call).

    r_min : smallest allowed |radius| in mm  ->  |curvature| <= 1/r_min
    t_min, t_max : thickness bounds in mm
    """
    out: List[Variable] = []
    for i in radius_rows:
        _check_row(rows, i)
        out.append(Variable(i, "curvature", -1.0 / r_min, 1.0 / r_min))
    for i in thickness_rows:
        _check_row(rows, i)
        out.append(Variable(i, "thickness", t_min, t_max))
    return out


def _check_row(rows, i):
    if not 0 <= i < len(rows):
        raise IndexError(f"row {i} does not exist (design has {len(rows)} rows)")


def apply_vector(rows: Sequence[dz.SurfaceRow], variables: Sequence[Variable],
                 u: np.ndarray) -> List[dz.SurfaceRow]:
    """Return a NEW list of rows with scaled solver vector u written in.
    The input rows are never modified."""
    new = list(rows)
    for v, ui in zip(variables, u):
        new[v.row] = v.put(new[v.row], ui * v.scale)
    return new


# ===================================================================
#  Merit function
# ===================================================================
@dataclass
class MeritConfig:
    """What 'good' means.  All lengths in mm unless stated."""
    epd: float = 10.0                      # entrance pupil diameter
    fields_deg: Tuple[float, ...] = (0.0,)
    field_weights: Optional[Tuple[float, ...]] = None
    wavelengths_nm: Tuple[float, ...] = (LAMBDA_D, LAMBDA_F, LAMBDA_C)  # first = primary
    wavelength_weights: Optional[Tuple[float, ...]] = None
    rings: int = 4                         # pupil sampling: 1 + 3*r*(r+1) rays
    spot_weight: float = 1.0               # errors are in micrometres
    efl_target: Optional[float] = None     # mm
    efl_weight: float = 1000.0             # 0.01 mm EFL error ~ 10 um of spot
    axial_colour_weight: float = 0.0       # error = BFL(F)-BFL(C) in mm, target 0
    min_center: float = 1.0                # glass centre thickness
    min_edge: float = 1.0                  # glass edge thickness at the full aperture
    constraint_weight: float = 1e5         # hinge errors; large = (almost) hard limit.
                                           # 0.001 mm violation ~ 100 um of spot error
    fail_penalty_um: float = 1000.0        # error of a ray that misses / suffers TIR

    def validate(self, n_rows: int) -> None:
        if self.epd <= 0:
            raise ValueError("epd must be > 0")
        if self.rings < 1:
            raise ValueError("rings must be >= 1")
        if not self.fields_deg or not self.wavelengths_nm:
            raise ValueError("need at least one field and one wavelength")
        for name, w, n in (("field_weights", self.field_weights, len(self.fields_deg)),
                           ("wavelength_weights", self.wavelength_weights,
                            len(self.wavelengths_nm))):
            if w is not None and len(w) != n:
                raise ValueError(f"{name} must have {n} entries")


def _spot_errors(rows, cfg: MeritConfig) -> np.ndarray:
    """Per-ray transverse errors in micrometres (see THEORY, item 1)."""
    px, py = dz.hex_pupil(cfg.epd, cfg.rings)
    n_rays = len(px)
    fw = cfg.field_weights or (1.0,) * len(cfg.fields_deg)
    ww = cfg.wavelength_weights or (1.0,) * len(cfg.wavelengths_nm)
    z_img = dz.image_plane_z(rows)
    errs = []
    for fdeg, wf in zip(cfg.fields_deg, fw):
        ref = None
        for wl, wwl in zip(cfg.wavelengths_nm, ww):
            _, _, pts, dd = dz.trace_bundle(rows, wl, px, py, fdeg)
            xy = at_image_plane(pts, dd, z_img)[:, :2]
            if ref is None:                       # primary wavelength defines the reference
                good = ~np.isnan(xy).any(axis=1)
                ref = xy[good].mean(axis=0) if good.any() else np.zeros(2)
            d_um = (xy - ref) * 1000.0
            d_um = np.where(np.isnan(d_um), cfg.fail_penalty_um, d_um)
            # 1/sqrt(N): sum of squares of these errors = (RMS radius)^2
            errs.append((cfg.spot_weight * wf * wwl / np.sqrt(n_rays)) * d_um.ravel())
    return np.concatenate(errs)


def _constraint_errors(rows, cfg: MeritConfig) -> np.ndarray:
    """Hinge errors for glass centre / edge thickness (see THEORY, item 4)."""
    h = np.array([cfg.epd / 2.0])
    out = []
    for i, r in enumerate(rows[:-1]):
        if r.glass == "AIR":
            continue
        nxt = rows[i + 1]
        edge = r.thickness + dz.sag_profile(nxt.radius, h)[0] - dz.sag_profile(r.radius, h)[0]
        out.append(max(0.0, cfg.min_center - r.thickness))
        out.append(max(0.0, cfg.min_edge - edge))
        # a surface cannot be sampled beyond its own radius of curvature
        for rr in (r, nxt):
            if rr.radius not in (0.0,) and np.isfinite(rr.radius) and abs(rr.radius) < h[0]:
                out.append(h[0] - abs(rr.radius))
    return cfg.constraint_weight * np.array(out, dtype=float)


def residuals(rows: Sequence[dz.SurfaceRow], cfg: MeritConfig) -> np.ndarray:
    """The full error vector e_k.  merit = sqrt(sum(e**2))."""
    rows = list(rows)
    parts = [_spot_errors(rows, cfg)]
    primary = cfg.wavelengths_nm[0]
    if cfg.efl_target is not None:
        efl, _ = dz.paraxial(rows, primary)
        err = cfg.efl_weight * (efl - cfg.efl_target) if np.isfinite(efl) else 1e6
        parts.append(np.array([err]))
    if cfg.axial_colour_weight:
        ac = dz.axial_colour(rows)
        parts.append(np.array([cfg.axial_colour_weight * ac if np.isfinite(ac) else 1e6]))
    parts.append(_constraint_errors(rows, cfg))
    e = np.concatenate(parts)
    return np.nan_to_num(e, nan=1e6, posinf=1e6, neginf=-1e6)


def merit(rows, cfg: MeritConfig) -> float:
    return float(np.sqrt(np.sum(residuals(rows, cfg) ** 2)))


def evaluate(rows: Sequence[dz.SurfaceRow], cfg: MeritConfig) -> Dict[str, object]:
    """Human-readable report of a design (what the GUI will print)."""
    rows = list(rows)
    efl, bfl = dz.paraxial(rows, cfg.wavelengths_nm[0])
    px, py = dz.hex_pupil(cfg.epd, cfg.rings)
    z_img = dz.image_plane_z(rows)
    spots = {}
    for fdeg in cfg.fields_deg:
        ref = None
        for wl in cfg.wavelengths_nm:
            _, _, pts, dd = dz.trace_bundle(rows, wl, px, py, fdeg)
            xy = at_image_plane(pts, dd, z_img)[:, :2]
            if ref is None:
                good = ~np.isnan(xy).any(axis=1)
                ref = xy[good].mean(axis=0) if good.any() else np.zeros(2)
            d = xy - ref
            ok = ~np.isnan(d).any(axis=1)
            spots[(fdeg, wl)] = (float(np.sqrt(np.mean(np.sum(d[ok] ** 2, axis=1)))) * 1000.0
                                 if ok.any() else float("nan"))
    return {"efl": efl, "bfl": bfl, "axial_colour": dz.axial_colour(rows),
            "image_distance": rows[-1].thickness, "rms_spot_um": spots,
            "merit": merit(rows, cfg)}


# ===================================================================
#  The optimiser
# ===================================================================
@dataclass
class OptResult:
    rows: List[dz.SurfaceRow]              # optimised design
    start_rows: List[dz.SurfaceRow]        # the design we started from
    merit_start: float
    merit_end: float
    merit_history: List[float]             # merit after every function evaluation
    n_evals: int
    success: bool
    message: str
    report_start: Dict[str, object] = field(default_factory=dict)
    report_end: Dict[str, object] = field(default_factory=dict)


def optimize(rows: Sequence[dz.SurfaceRow], variables: Sequence[Variable],
             cfg: MeritConfig, max_nfev: int = 200,
             callback: Optional[Callable[[int, float], None]] = None) -> OptResult:
    """Run damped least squares.  The input rows are NOT modified.

    callback(n_evals, merit) is called after every merit evaluation; the GUI
    uses it for the live merit plot.  Raise an exception inside it to abort.
    """
    rows = list(rows)
    dz.validate(rows)
    cfg.validate(len(rows))
    if not variables:
        raise ValueError("Select at least one variable")
    seen = {(v.row, v.kind) for v in variables}
    if len(seen) != len(variables):
        raise ValueError("The same variable was listed twice")
    for v in variables:
        _check_row(rows, v.row)

    lo = np.array([v.lo / v.scale for v in variables])
    hi = np.array([v.hi / v.scale for v in variables])
    u0 = np.array([v.get(rows) / v.scale for v in variables])
    u0 = np.clip(u0, lo, hi)      # a start outside the bounds is moved onto them

    history: List[float] = []

    def fun(u):
        e = residuals(apply_vector(rows, variables, u), cfg)
        m = float(np.sqrt(np.sum(e ** 2)))
        history.append(m)
        if callback is not None:
            callback(len(history), m)
        return e

    sol = least_squares(fun, u0, bounds=(lo, hi), method="trf",
                        diff_step=1e-6, max_nfev=max_nfev)
    best = apply_vector(rows, variables, sol.x)
    return OptResult(
        rows=best, start_rows=rows,
        merit_start=merit(rows, cfg), merit_end=merit(best, cfg),
        merit_history=history, n_evals=len(history),
        success=bool(sol.success), message=str(sol.message),
        report_start=evaluate(rows, cfg), report_end=evaluate(best, cfg))


# ===================================================================
#  Undo / redo
# ===================================================================
class DesignHistory:
    """Stack of designs for the Undo / Redo buttons.

    push(rows)   - remember the design BEFORE you change it
    undo(current) -> previous design (or None)
    redo(current) -> next design (or None)
    """
    def __init__(self, limit: int = 100):
        self._undo: List[List[dz.SurfaceRow]] = []
        self._redo: List[List[dz.SurfaceRow]] = []
        self.limit = limit

    @staticmethod
    def _copy(rows):
        return [replace(r) for r in rows]

    def push(self, rows) -> None:
        self._undo.append(self._copy(rows))
        self._undo = self._undo[-self.limit:]
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self, current):
        if not self._undo:
            return None
        self._redo.append(self._copy(current))
        return self._undo.pop()

    def redo(self, current):
        if not self._redo:
            return None
        self._undo.append(self._copy(current))
        return self._redo.pop()
