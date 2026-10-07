# Ray (height, angle)
"""
ray.py  -  LESSON 1: What is a "ray"?
=====================================================================
THEORY FOR A COMPLETE BEGINNER
------------------------------
Light travels in straight lines inside air or glass. A "ray" is just a
thin imaginary arrow showing where one little piece of light is going.
Ray tracing = following those arrows through lenses.

Imagine a straight horizontal line through the middle of your optical
system, like the rail of an optical bench. This is the OPTICAL AXIS.
We call its direction "z" (light travels left -> right along z).

At any position z, a ray is fully described by only TWO numbers:

    y  : HEIGHT - how far the ray is above (+) or below (-) the axis.
         Unit: millimetres (mm) in this project.
    u  : SLOPE  - how steeply the ray is tilted:
         u = (change in y) / (change in z).
         Example: u = 0.1 -> the ray rises 0.1 mm for every 1 mm it
         travels forward. u = 0 -> parallel to the axis.

PARAXIAL APPROXIMATION ("para-axial" = "near the axis")
-------------------------------------------------------
If rays stay close to the axis and are only slightly tilted, the messy
trigonometry (sin, tan) of real optics simplifies to plain
multiplication and addition. This "first-order optics" is what Zemax
calls the paraxial layer. It is the skeleton every lens design starts
from (focal length, image position, magnification).
=====================================================================
"""

from dataclasses import dataclass  # shortcut for simple "data holder" classes
import numpy as np                 # numpy = fast numbers & matrices


@dataclass
class Ray:
    """A single paraxial ray = (height y, slope u) plus a 'blocked' flag."""

    y: float                 # height above the optical axis, in mm
    u: float                 # slope (dimensionless, ~ angle in radians if small)
    blocked: bool = False    # becomes True if an aperture (hole) stops this ray

    def as_vector(self) -> np.ndarray:
        """
        Return the ray as the pair [y, u].

        WHY? Each optical element will be a 2x2 table of numbers (a
        MATRIX). Matrix x vector gives the new ray in one step.
        """
        return np.array([self.y, self.u], dtype=float)

    @staticmethod
    def from_vector(vec: np.ndarray, blocked: bool = False) -> "Ray":
        """Build a Ray back from a [y, u] vector (reverse of as_vector)."""
        return Ray(y=float(vec[0]), u=float(vec[1]), blocked=blocked)
