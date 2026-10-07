"""Cross-check opticslab's ray tracer against Optiland (BK7/F2 doublet, f ~ 100 mm)."""
import numpy as np
import pytest

pytest.importorskip("optiland")

from optiland.materials import IdealMaterial
from optiland.optic import Optic

from opticslab.glass import LAMBDA_D, N_BK7, N_F2
from opticslab.raytrace import (
    at_image_plane,
    collimated_beam,
    doublet_prescription,
    last_vertex_z,
    trace,
)

RADII = (60.0, -34.61924097, -139.861202)
T1, T2, BFD = 4.0, 2.0, 97.1
EPD = 10.0


def build_optiland_doublet() -> Optic:
    n1, n2 = N_BK7.n(LAMBDA_D), N_F2.n(LAMBDA_D)
    o = Optic()
    o.surfaces.add(index=0, thickness=np.inf)
    o.surfaces.add(index=1, radius=RADII[0], thickness=T1,
                   material=IdealMaterial(n=n1), is_stop=True)
    o.surfaces.add(index=2, radius=RADII[1], thickness=T2, material=IdealMaterial(n=n2))
    o.surfaces.add(index=3, radius=RADII[2], thickness=BFD)
    o.surfaces.add(index=4)
    o.set_aperture(aperture_type="EPD", value=EPD)
    o.fields.set_type("angle")
    o.fields.add(y=0)
    o.wavelengths.add(value=LAMBDA_D / 1000, is_primary=True)
    return o


def optiland_image_y(o: Optic, py: float) -> float:
    r = o.trace_generic(Hx=0, Hy=0, Px=0, Py=py, wavelength=LAMBDA_D / 1000)
    return float(np.ravel(r.y)[0])


def our_image_y(height_mm: float) -> float:
    prescription = doublet_prescription(*RADII, T1, T2, N_BK7, N_F2, LAMBDA_D)
    origins, dirs = collimated_beam(2 * height_mm, 3)
    pts, dirs_out = trace(origins, dirs, prescription)
    return float(at_image_plane(pts, dirs_out, last_vertex_z(prescription) + BFD)[-1, 1])


@pytest.mark.parametrize("py", [0.5, 1.0])
def test_rays_match_optiland(py):
    ref = optiland_image_y(build_optiland_doublet(), py)
    assert our_image_y(py * EPD / 2) == pytest.approx(ref, abs=1e-6)


if __name__ == "__main__":
    o = build_optiland_doublet()
    print(f"{'Py':>5} {'h [mm]':>7} {'Optiland y':>14} {'opticslab y':>14} {'diff [mm]':>10}")
    for py in (0.5, 1.0):
        ref, h = optiland_image_y(o, py), py * EPD / 2
        ours = our_image_y(h)
        print(f"{py:5.2f} {h:7.2f} {ref:14.8f} {ours:14.8f} {abs(ours - ref):10.2e}")
