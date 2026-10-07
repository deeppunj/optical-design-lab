# Optical System
"""
system.py  -  LESSON 3: Chaining elements into an optical SYSTEM
=====================================================================
THEORY
------
A real instrument is a sequence: space -> lens -> space -> lens ...
A ray passes through them one after another, so we apply the elements
in order. Mathematically:

    ray_out = M_n ... M_2 M_1 ray_in     (LAST element on the LEFT!)

Multiplying all matrices gives ONE matrix for the whole system - the
SYSTEM MATRIX. From its numbers we read focal length etc. (analysis.py).

The trace also records WHERE (z) and HOW HIGH (y) the ray is after each
element, so we can draw the classic ray-diagram picture.
=====================================================================
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .ray import Ray
from .elements import OpticalElement


@dataclass
class TracePoint:
    """Snapshot of a ray after one element."""
    z: float        # position along the optical axis (mm)
    y: float        # height (mm)
    u: float        # slope
    label: str      # which element we just passed
    blocked: bool   # was the ray stopped here?


class OpticalSystem:
    """An ordered list of optical elements, light travelling left -> right."""

    def __init__(self, elements: Optional[List[OpticalElement]] = None):
        self.elements: List[OpticalElement] = list(elements) if elements else []

    def add(self, element: OpticalElement) -> "OpticalSystem":
        """Append an element at the end (returns self so calls can be chained)."""
        self.elements.append(element)
        return self

    def matrix(self) -> np.ndarray:
        """Multiply all element matrices into ONE system matrix."""
        total = np.identity(2)               # 'do nothing' matrix to start
        for el in self.elements:             # first element first...
            total = el.matrix() @ total      # ...but new matrices go on the LEFT
        return total

    def trace(self, ray: Ray, z_start: float = 0.0) -> List[TracePoint]:
        """
        Follow one ray through the system. Returns TracePoints (start +
        one per element). Stops early if an aperture blocks the ray.
        """
        z = z_start
        history = [TracePoint(z, ray.y, ray.u, "start", False)]
        for el in self.elements:
            ray = el.apply(ray)              # element acts on the ray
            z += el.length                   # advance along the axis
            history.append(TracePoint(z, ray.y, ray.u, el.name, ray.blocked))
            if ray.blocked:
                break                        # blocked light goes no further
        return history
