# Focal length, image position (later)
"""
analysis.py  -  LESSON 4: Reading focal length, image & magnification
=====================================================================
THEORY (all from the system matrix  | A B ; C D |)
------
* EFFECTIVE FOCAL LENGTH:  EFL = -1 / C
  C is the "bending power" of the whole system. A single thin lens has
  C = -1/f, so EFL = f.

* IMAGE POSITION. An object point sends rays in many directions; an
  IMAGE forms where they all meet again. "All meet again" means the
  output height does NOT depend on the starting slope u, i.e. matrix
  element B (the number multiplying u) must be ZERO.
  With object distance s before the system and image distance d after:
      Total = Space(d) @ M @ Space(s) = | A' B' ; C' D' |
      B_total = B' + d*D' = 0   ->   d = -B'/D'
  For one thin lens this reproduces the school formula 1/s + 1/d = 1/f.

* MAGNIFICATION  m = A_total  (image height / object height when B=0).
  m < 0 -> image UPSIDE-DOWN; |m| < 1 -> smaller than the object.
=====================================================================
"""

from .system import OpticalSystem
from .elements import Space


def effective_focal_length(system: OpticalSystem) -> float:
    """EFL = -1/C of the system matrix (inf if C = 0, i.e. afocal)."""
    C = system.matrix()[1, 0]
    return float("inf") if abs(C) < 1e-15 else -1.0 / C


def image_distance_and_magnification(system: OpticalSystem, object_distance: float):
    """
    For an object `object_distance` mm in front of `system`, return
    (image distance behind the system, magnification).
    `system` must contain ONLY the optics (not object/image spaces).
    """
    M = system.matrix() @ Space(object_distance).matrix()   # M' = M @ Space(s)
    Ap, Bp, Cp, Dp = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    d = -Bp / Dp                    # imaging condition: B_total = 0
    m = Ap + d * Cp                 # A_total = magnification
    return float(d), float(m)
