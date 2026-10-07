"""
stops.py  -  LESSON 6: Aperture stop, marginal ray and chief ray
=====================================================================
THEORY
------
Several apertures (lens rims, irises) can limit the light. The one
that limits it MOST is the APERTURE STOP. Zemax calls it the "stop".

How to find it: send a ray from the object's axis point with a small
slope (u = 1, a unit test ray). It reaches each aperture at some
height y_k. The aperture with the largest ratio  |y_k| / R_k  would
clip first if we opened the cone wider -> that is the STOP.

Two rays describe the whole system:
 * MARGINAL RAY: starts on the axis at the object, touches the
   EDGE of the stop. It sets the cone of light (brightness, NA).
 * CHIEF RAY: starts at the object's EDGE, passes through the CENTRE
   of the stop. It sets the field of view.

Both are linear in the starting slope, so we scale the unit ray:
   marginal:  u0 = R_stop / y_stop(unit)
   chief:     y_stop = a*h + b*u0 = 0   ->   u0 = -a*h / b
where a, b = stop height for a start ray (y=1,u=0) and (y=0,u=1).

LIMITATION: the object is at a FINITE distance (not infinity).
=====================================================================
"""

import numpy as np
from .ray import Ray
from .elements import Space, Aperture
from .system import OpticalSystem


def _height_at_apertures(system, start_vec, object_distance):
    """Push [y, u] through the system; return [(aperture, y_there), ...]."""
    v = Space(object_distance).matrix() @ np.asarray(start_vec, dtype=float)
    found = []
    for el in system.elements:
        v = el.matrix() @ v          # no blocking here, we only want heights
        if isinstance(el, Aperture):
            found.append((el, float(v[0])))
    return found


def find_stop(system: OpticalSystem, object_distance: float) -> Aperture:
    """Return the Aperture with the largest |y|/R for the unit axial ray."""
    found = _height_at_apertures(system, [0.0, 1.0], object_distance)
    if not found:
        raise ValueError("The system contains no Aperture.")
    return max(found, key=lambda pair: abs(pair[1]) / pair[0].radius)[0]


def _unit_response(system, stop, start_vec, object_distance):
    """Height at the stop for a given start vector."""
    for el, y in _height_at_apertures(system, start_vec, object_distance):
        if el is stop:
            return y
    raise ValueError("stop not in system")


def marginal_ray(system: OpticalSystem, object_distance: float) -> Ray:
    """Ray from the object axis point that just touches the stop edge."""
    stop = find_stop(system, object_distance)
    b = _unit_response(system, stop, [0.0, 1.0], object_distance)
    return Ray(y=0.0, u=stop.radius / b)


def chief_ray(system: OpticalSystem, object_distance: float,
              object_height: float) -> Ray:
    """Ray from an off-axis object point that crosses the stop centre."""
    stop = find_stop(system, object_distance)
    a = _unit_response(system, stop, [1.0, 0.0], object_distance)
    b = _unit_response(system, stop, [0.0, 1.0], object_distance)
    return Ray(y=float(object_height), u=-a * object_height / b)


def with_object_space(system: OpticalSystem, object_distance: float) -> OpticalSystem:
    """Copy of the system with the object-to-first-element gap in front.
    Trace with z_start = -object_distance so z = 0 is the first element."""
    return OpticalSystem([Space(object_distance)] + list(system.elements))
