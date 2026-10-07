# Space, Thinlens, Aperture
"""
elements.py  -  LESSON 2: Optical elements as MATRICES (ABCD matrices)
=====================================================================
THEORY
------
Everything light meets in a simple system does one of three things:
  (1) travels through empty space/glass  -> Space
  (2) gets bent by a lens                -> ThinLens
  (3) gets blocked by a hole/stop        -> Aperture

In paraxial optics every element is a 2x2 table of numbers:

        | A  B |        new_y = A*y + B*u
    M = |      |        new_u = C*y + D*u
        | C  D |

This is the ABCD MATRIX (a.k.a. "ray transfer matrix"). You do not
need matrix theory: it is a compact way to write two simple formulas.

--- FREE SPACE of length d ---------------------------------------
Light flies straight. Height grows by slope*distance; slope is kept:
        new_y = y + d*u        new_u = u
    -> A=1, B=d, C=0, D=1

--- THIN LENS of focal length f ----------------------------------
Height does not change (lens is infinitely thin) but the SLOPE changes
in proportion to how far from the axis the ray hits (outer rays are
bent more):
        new_y = y              new_u = u - y/f
    -> A=1, B=0, C=-1/f, D=1
f > 0 : converging (convex) lens - pulls rays together.
f < 0 : diverging (concave) lens - spreads rays apart.
1/f is the OPTICAL POWER (1/mm; in dioptres if f is in metres).
Check: a ray parallel to the axis (u=0) at height y leaves with slope
-y/f, so after travelling f its height is y + f*(-y/f) = 0. It crosses
the axis exactly one focal length behind the lens. That IS the
definition of the focal point.

--- APERTURE of radius R -----------------------------------------
Does not bend light (identity matrix) but any ray hitting it at
|y| > R is stopped. Zemax calls the most restrictive aperture the
"STOP": it decides how much light the system accepts.
=====================================================================
"""

import numpy as np
from .ray import Ray


class OpticalElement:
    """
    Parent ('base') class for all elements. Children provide:
      - matrix(): the 2x2 ABCD matrix
      - length  : distance along z that the element occupies
    """

    name = "element"   # label used in printouts / plots
    length = 0.0       # mm of travel along the axis (0 for lenses & apertures)

    def matrix(self) -> np.ndarray:
        raise NotImplementedError("Child classes must define their ABCD matrix")

    def apply(self, ray: Ray) -> Ray:
        """Send a ray through: new_vector = matrix @ old_vector ('@' = matrix product)."""
        new_vec = self.matrix() @ ray.as_vector()
        return Ray.from_vector(new_vec, blocked=ray.blocked)


class Space(OpticalElement):
    """Free propagation over a distance d (mm)."""

    def __init__(self, d: float):
        self.d = float(d)
        self.length = self.d          # this element DOES advance z
        self.name = f"Space({self.d:g} mm)"

    def matrix(self) -> np.ndarray:
        return np.array([[1.0, self.d],     # A=1, B=d
                         [0.0, 1.0]])       # C=0, D=1


class ThinLens(OpticalElement):
    """Ideal thin lens of focal length f (mm). Positive f = converging."""

    def __init__(self, f: float):
        if f == 0:
            raise ValueError("Focal length cannot be zero (infinite power).")
        self.f = float(f)
        self.name = f"ThinLens(f={self.f:g} mm)"

    def matrix(self) -> np.ndarray:
        return np.array([[1.0, 0.0],             # A=1, B=0
                         [-1.0 / self.f, 1.0]])  # C=-1/f, D=1


class Aperture(OpticalElement):
    """Circular hole of radius R (mm). Rays outside it are blocked."""

    def __init__(self, radius: float):
        self.radius = float(radius)
        self.name = f"Aperture(R={self.radius:g} mm)"

    def matrix(self) -> np.ndarray:
        return np.identity(2)         # does not change the ray

    def apply(self, ray: Ray) -> Ray:
        out = super().apply(ray)
        if abs(out.y) > self.radius:  # missed the hole -> stopped
            out.blocked = True
        return out
