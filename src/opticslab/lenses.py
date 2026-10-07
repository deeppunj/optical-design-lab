"""
lenses.py  -  LESSON 9: Thick lenses and the achromatic doublet
=====================================================================
THEORY
------
THICK LENS  = surface 1 (air->glass), travel t inside glass, surface 2:

    M = S2 @ Space(t) @ S1        (last element on the LEFT)

It reproduces the LENSMAKER'S equation (n = index, t = thickness):

    1/f = (n-1) * [ 1/R1 - 1/R2 + (n-1)*t / (n*R1*R2) ]

EFL = -1/C and BACK FOCAL LENGTH BFL = -A/C   (from the system matrix).
BFL is measured from the LAST surface, EFL from the principal plane.

CHROMATIC ABERRATION: n depends on wavelength, so f does too. Blue
focuses CLOSER than red. A single lens cannot avoid this.

CEMENTED DOUBLET (achromat): crown + flint glued together.
Surfaces: air->crown (R1), crown->flint (R2), flint->air (R3).
We pick R1, R2 and SOLVE for R3 so that BFL(blue) = BFL(red).
That is your first real optimisation (a 1D root find with scipy).
=====================================================================
"""

import numpy as np
from scipy.optimize import brentq
from .elements import OpticalElement, Space
from .surfaces import RefractingSurface
from .system import OpticalSystem
from .glass import Glass, LAMBDA_D, LAMBDA_F, LAMBDA_C


class ThickLens(OpticalElement):
    """One glass lens in air, defined by two radii and a centre thickness (mm)."""

    def __init__(self, r1, r2, thickness, glass: Glass, wavelength_nm=LAMBDA_D):
        self.r1, self.r2 = float(r1), float(r2)
        self.thickness = float(thickness)
        self.glass = glass
        self.wavelength_nm = float(wavelength_nm)
        self.n = glass.n(self.wavelength_nm)
        self.length = self.thickness
        self.name = (f"ThickLens({glass.name}, R1={self.r1:g}, "
                     f"R2={self.r2:g}, t={self.thickness:g})")

    def matrix(self) -> np.ndarray:
        n = self.n
        s1 = RefractingSurface(self.r1, 1.0, n).matrix()
        s2 = RefractingSurface(self.r2, n, 1.0).matrix()
        return s2 @ Space(self.thickness).matrix() @ s1

    def efl(self) -> float:
        """Effective focal length = -1/C."""
        C = self.matrix()[1, 0]
        return float("inf") if abs(C) < 1e-15 else -1.0 / C

    def bfl(self) -> float:
        """Back focal length = -A/C (from the last surface to the focus)."""
        M = self.matrix()
        return float("inf") if abs(M[1, 0]) < 1e-15 else float(-M[0, 0] / M[1, 0])


def cemented_doublet(r1, r2, r3, t1, t2, crown: Glass, flint: Glass,
                     wavelength_nm=LAMBDA_D) -> OpticalSystem:
    """Crown (t1) cemented to flint (t2). Radii in mm, light travels left->right."""
    n1, n2 = crown.n(wavelength_nm), flint.n(wavelength_nm)
    return OpticalSystem([
        RefractingSurface(r1, 1.0, n1), Space(t1),
        RefractingSurface(r2, n1, n2), Space(t2),
        RefractingSurface(r3, n2, 1.0),
    ])


def back_focal_length(system: OpticalSystem) -> float:
    """BFL = -A/C of any system that starts and ends in air."""
    M = system.matrix()
    return float("inf") if abs(M[1, 0]) < 1e-15 else float(-M[0, 0] / M[1, 0])


def solve_achromat_r3(r1, r2, t1, t2, crown, flint, r3_bracket=(-400.0, -30.0),
                      wl_short=LAMBDA_F, wl_long=LAMBDA_C) -> float:
    """Find R3 so the blue (F) and red (C) back focal lengths are equal."""
    def mismatch(r3):
        bs = back_focal_length(cemented_doublet(r1, r2, r3, t1, t2, crown, flint, wl_short))
        bl = back_focal_length(cemented_doublet(r1, r2, r3, t1, t2, crown, flint, wl_long))
        return bs - bl

    lo, hi = r3_bracket
    if mismatch(lo) * mismatch(hi) > 0:
        raise ValueError("No sign change in the bracket; try another r3_bracket.")
    return float(brentq(mismatch, lo, hi))
