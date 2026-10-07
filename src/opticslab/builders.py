"""
builders.py  -  LESSON 5: Beam expanders (ready-made afocal systems)
=====================================================================
THEORY
------
A BEAM EXPANDER makes a collimated (parallel) beam wider or narrower
while keeping it collimated. It is two lenses sharing one focal point:

        separation  d = f1 + f2

AFOCAL means "no focus": parallel rays in -> parallel rays out.
In matrix language, C = 0 (no bending power for the system as a whole).
Prove it yourself with M = L2 @ Space(d) @ L1 and d = f1 + f2:

        A = -f2/f1        (output height / input height)
        C = 0             (afocal)
        D = -f1/f2

So the BEAM MAGNIFICATION is  M = A = -f2/f1.

KEPLER:   f1 > 0, f2 > 0. Beam crosses a real focus in the middle.
          M is NEGATIVE -> the beam is inverted. A pinhole placed at
          the middle focus can clean the beam (spatial filter).
GALILEAN: f1 < 0, f2 > 0. NO internal focus (nice for high-power
          lasers, no air breakdown). M is POSITIVE -> upright.
          The negative lens comes FIRST, otherwise the beam shrinks.
=====================================================================
"""

from .elements import Space, ThinLens
from .system import OpticalSystem


def kepler_expander(f1: float, f2: float) -> OpticalSystem:
    """Two positive lenses, separated by f1 + f2 (all in mm)."""
    if f1 <= 0 or f2 <= 0:
        raise ValueError("Kepler expander needs f1 > 0 and f2 > 0.")
    return OpticalSystem([ThinLens(f1), Space(f1 + f2), ThinLens(f2)])


def galilean_expander(f1: float, f2: float) -> OpticalSystem:
    """Negative lens f1 first, then positive lens f2, separated by f1 + f2."""
    if f1 >= 0 or f2 <= 0:
        raise ValueError("Galilean expander needs f1 < 0 (diverging first) and f2 > 0.")
    gap = f1 + f2                      # f1 is negative, so gap = f2 - |f1|
    if gap <= 0:
        raise ValueError("Need f2 > |f1|, otherwise the lenses would overlap.")
    return OpticalSystem([ThinLens(f1), Space(gap), ThinLens(f2)])


def beam_magnification(system: OpticalSystem) -> float:
    """Beam magnification = element A of the system matrix (= -f2/f1)."""
    return float(system.matrix()[0, 0])
