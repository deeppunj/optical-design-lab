"""Cross-check paraxial EFL/BFL and the C/F chromatic focal shift against Optiland."""
import numpy as np
import pytest

pytest.importorskip("optiland")

from optiland.materials import IdealMaterial
from optiland.optic import Optic

from opticslab.glass import LAMBDA_C, LAMBDA_D, LAMBDA_F, N_BK7, N_F2
from opticslab.raytrace import (
    collimated_beam,
    doublet_prescription,
    marginal_axis_crossing,
    trace,
)

RADII = (60.0, -34.61924097, -139.861202)
T1, T2 = 4.0, 2.0
EPD = 10.0
H_PARAXIAL = 1e-3  # mm, tiny ray height ~ paraxial


def build_optiland_doublet(wavelength_nm: float) -> Optic:
    o = Optic()
    o.surfaces.add(index=0, thickness=np.inf)
    o.surfaces.add(index=1, radius=RADII[0], thickness=T1,
                   material=IdealMaterial(n=N_BK7.n(wavelength_nm)), is_stop=True)
    o.surfaces.add(index=2, radius=RADII[1], thickness=T2,
                   material=IdealMaterial(n=N_F2.n(wavelength_nm)))
    o.surfaces.add(index=3, radius=RADII[2], thickness=0.0)
    o.surfaces.add(index=4)
    o.set_aperture(aperture_type="EPD", value=EPD)
    o.fields.set_type("angle")
    o.fields.add(y=0)
    o.wavelengths.add(value=wavelength_nm / 1000.0, is_primary=True)
    return o


def optiland_efl(wl_nm: float) -> float:
    return float(np.ravel(build_optiland_doublet(wl_nm).paraxial.f2())[0])


def optiland_bfl(wl_nm: float) -> float:
    """Axis crossing of a near-axis real ray, measured from the last vertex."""
    o = build_optiland_doublet(wl_nm)
    py = H_PARAXIAL / (EPD / 2)
    r = o.trace_generic(Hx=0, Hy=0, Px=0, Py=py, wavelength=wl_nm / 1000.0)
    y, n = float(np.ravel(r.y)[0]), float(np.ravel(r.N)[0])
    m = float(np.ravel(r.M)[0])
    return -y * n / m


def our_prescription(wl_nm: float):
    return doublet_prescription(*RADII, T1, T2, N_BK7, N_F2, wl_nm)


def our_bfl(wl_nm: float) -> float:
    return marginal_axis_crossing(our_prescription(wl_nm), height=H_PARAXIAL)


def our_efl(wl_nm: float) -> float:
    """EFL = h / |slope| of the emerging near-axis ray for a collimated input."""
    o, d = collimated_beam(2 * H_PARAXIAL, n_rays=2)
    o, d = o[-1:], d[-1:]
    _, dd = trace(o, d, our_prescription(wl_nm))
    slope = dd[0][1] / dd[0][2]
    return float(-H_PARAXIAL / slope)


@pytest.mark.parametrize("wl", [LAMBDA_C, LAMBDA_D, LAMBDA_F])
def test_efl_matches_optiland(wl):
    assert our_efl(wl) == pytest.approx(optiland_efl(wl), abs=1e-4)


@pytest.mark.parametrize("wl", [LAMBDA_C, LAMBDA_D, LAMBDA_F])
def test_bfl_matches_optiland(wl):
    assert our_bfl(wl) == pytest.approx(optiland_bfl(wl), abs=1e-4)


def test_d_line_efl_is_100_mm():
    assert our_efl(LAMBDA_D) == pytest.approx(100.0, abs=0.01)


def test_chromatic_focal_shift_matches_optiland():
    ours = our_bfl(LAMBDA_C) - our_bfl(LAMBDA_F)
    ref = optiland_bfl(LAMBDA_C) - optiland_bfl(LAMBDA_F)
    assert ours == pytest.approx(ref, abs=2e-5)
    assert abs(ours) < 0.01, "achromat: C/F focal shift should be only a few um"

def test_secondary_spectrum_matches_optiland():
    ours = our_bfl(LAMBDA_C) - our_bfl(LAMBDA_D)
    ref = optiland_bfl(LAMBDA_C) - optiland_bfl(LAMBDA_D)
    assert ours == pytest.approx(ref, abs=2e-5)
    assert 0.03 < abs(ours) < 0.07   # about 50 um for a 100 mm BK7/F2 doublet


if __name__ == "__main__":
    print(f"{'line':>5} {'wl [nm]':>9} {'EFL ours':>11} {'EFL Optiland':>13} "
          f"{'BFL ours':>11} {'BFL Optiland':>13}")
    for name, wl in ("C", LAMBDA_C), ("d", LAMBDA_D), ("F", LAMBDA_F):
        print(f"{name:>5} {wl:9.4f} {our_efl(wl):11.5f} {optiland_efl(wl):13.5f} "
              f"{our_bfl(wl):11.5f} {optiland_bfl(wl):13.5f}")
    shift = our_bfl(LAMBDA_C) - our_bfl(LAMBDA_F)
    print(f"C-F focal shift: ours {shift * 1000:.3f} um, "
          f"Optiland {(optiland_bfl(LAMBDA_C) - optiland_bfl(LAMBDA_F)) * 1000:.3f} um")
