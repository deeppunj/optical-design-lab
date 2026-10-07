"""
surfaces.py  -  LESSON 8: Refraction at one spherical surface
=====================================================================
THEORY
------
Light goes from medium n1 into medium n2 through a spherical surface of
radius R (curvature c = 1/R, R > 0 if the centre lies to the RIGHT).

Paraxial Snell's law:   n2 * u' = n1 * u - y * (n2 - n1) * c

Our ray vector is [y, u] with u the TRUE slope in the CURRENT medium, so

    y' = y
    u' = (n1 - n2)*c/n2 * y  +  (n1/n2) * u

    matrix = | 1                  0     |
             | (n1-n2)*c/n2     n1/n2   |

The determinant is n1/n2, not 1 (unlike lens in air). That is correct:
the slope u changes meaning when the medium changes.
A flat surface (R = infinity, c = 0) bends only tilted rays.
=====================================================================
"""

import numpy as np
from .elements import OpticalElement


class RefractingSurface(OpticalElement):
    def __init__(self, radius: float, n1: float, n2: float):
        if radius == 0:
            raise ValueError("radius = 0 is not allowed (use float('inf') for flat).")
        self.radius = float(radius)
        self.n1, self.n2 = float(n1), float(n2)
        self.curvature = 0.0 if np.isinf(self.radius) else 1.0 / self.radius
        self.length = 0.0                      # a surface has no thickness
        self.name = f"Surface(R={self.radius:g}, {self.n1:.4f}->{self.n2:.4f})"

    def matrix(self) -> np.ndarray:
        n1, n2, c = self.n1, self.n2, self.curvature
        return np.array([[1.0, 0.0],
                         [(n1 - n2) * c / n2, n1 / n2]])
