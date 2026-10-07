"""
solver.py  -  LESSON 10: Designing an achromat (solving for radii)
=====================================================================
THEORY
------
Unknowns: R2 and R3.  Conditions (two equations):
    (1) EFL(d-line)           = target f
    (2) BFL(F-line) - BFL(C-line) = 0     (blue and red focus together)

STARTING GUESS (thin-lens achromat theory, V = Abbe number):
    phi   = 1/f
    phi1  =  phi * V1 / (V1 - V2)        crown power
    phi2  = -phi * V2 / (V1 - V2)        flint power  (negative)
    phi1 = (n1-1)*(1/R1 - 1/R2)  ->  R2
    phi2 = (n2-1)*(1/R2 - 1/R3)  ->  R3
Thickness makes this slightly wrong, so least_squares polishes it.

NOTE: an achromat only matches TWO wavelengths. The remaining colour
error at other wavelengths is the SECONDARY SPECTRUM (see the demo).
=====================================================================
"""

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from scipy.optimize import least_squares

from .glass import Glass, LAMBDA_D, LAMBDA_F, LAMBDA_C
from .lenses import cemented_doublet, back_focal_length
from .system import OpticalSystem


def system_efl(system: OpticalSystem) -> float:
    """Effective focal length = -1/C of a system that starts and ends in air."""
    C = system.matrix()[1, 0]
    return float("inf") if abs(C) < 1e-15 else float(-1.0 / C)


@dataclass(frozen=True)
class AchromatResult:
    r1: float
    r2: float
    r3: float
    efl: float             # at the d line (mm)
    bfl: float             # at the d line (mm)
    color_shift: float     # BFL(F) - BFL(C) (mm), ~0 when solved
    success: bool


def design_achromat(f, r1, t1, t2, crown: Glass, flint: Glass,
                    lambda_d=LAMBDA_D, lambda_f=LAMBDA_F,
                    lambda_c=LAMBDA_C) -> AchromatResult:
    """Solve R2, R3 of a crown+flint cemented doublet for focal length f (mm)."""
    if f <= 0:
        raise ValueError("f must be positive.")
    v1, v2 = crown.abbe_number(), flint.abbe_number()
    if abs(v1 - v2) < 1e-6:
        raise ValueError("Crown and flint have the same Abbe number.")

    n1, n2 = crown.n(lambda_d), flint.n(lambda_d)
    inv_r1 = 0.0 if np.isinf(r1) else 1.0 / r1
    phi = 1.0 / f
    phi1 = phi * v1 / (v1 - v2)
    phi2 = -phi * v2 / (v1 - v2)
    inv_r2 = inv_r1 - phi1 / (n1 - 1.0)
    inv_r3 = inv_r2 - phi2 / (n2 - 1.0)
    if abs(inv_r2) < 1e-12 or abs(inv_r3) < 1e-12:
        raise ValueError("Starting guess gave a flat surface; change R1.")
    guess = [1.0 / inv_r2, 1.0 / inv_r3]

    def build(r2, r3, wl):
        return cemented_doublet(r1, r2, r3, t1, t2, crown, flint, wl)

    def residuals(x):
        r2, r3 = x
        efl_d = system_efl(build(r2, r3, lambda_d))
        bfl_f = back_focal_length(build(r2, r3, lambda_f))
        bfl_c = back_focal_length(build(r2, r3, lambda_c))
        return [(efl_d - f) / f, (bfl_f - bfl_c) / f]

    sol = least_squares(residuals, guess, xtol=1e-14, ftol=1e-14)
    r2, r3 = sol.x
    d_sys = build(r2, r3, lambda_d)
    shift = (back_focal_length(build(r2, r3, lambda_f))
             - back_focal_length(build(r2, r3, lambda_c)))
    return AchromatResult(r1, float(r2), float(r3), system_efl(d_sys),
                          back_focal_length(d_sys), float(shift),
                          bool(sol.success))


def focal_shift_curve(build: Callable[[float], OpticalSystem],
                      wavelengths_nm: Sequence[float]) -> np.ndarray:
    """BFL (mm) at each wavelength. `build(wl)` returns the system for that wl."""
    return np.array([back_focal_length(build(w)) for w in wavelengths_nm])
