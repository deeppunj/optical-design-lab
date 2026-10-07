"""Stage 3 checks: real glass, refracting surfaces, thick lenses, achromat."""
import numpy as np
import pytest
from opticslab import Ray, Space
from opticslab.glass import (N_BK7, FUSED_SILICA, N_F2, N_SF11, AIR,
                             LAMBDA_D, LAMBDA_F, LAMBDA_C, get_glass)
from opticslab.surfaces import RefractingSurface
from opticslab.lenses import (ThickLens, cemented_doublet, back_focal_length,
                              solve_achromat_r3)


@pytest.mark.parametrize("glass, nd", [(N_BK7, 1.5168), (FUSED_SILICA, 1.4585),
                                       (N_F2, 1.6200), (N_SF11, 1.7847)])
def test_refractive_index_at_d_line(glass, nd):
    assert glass.n(LAMBDA_D) == pytest.approx(nd, abs=2e-4)


def test_abbe_numbers():
    assert N_BK7.abbe_number() == pytest.approx(64.17, abs=0.1)
    assert N_F2.abbe_number() == pytest.approx(36.4, abs=0.2)


def test_index_decreases_with_wavelength():
    assert N_BK7.n(486.1) > N_BK7.n(587.6) > N_BK7.n(656.3)


def test_air_and_lookup():
    assert AIR.n(550.0) == pytest.approx(1.0)
    assert get_glass("n-bk7") is N_BK7
    with pytest.raises(KeyError):
        get_glass("unobtainium")


def test_surface_determinant_is_n1_over_n2():
    s = RefractingSurface(50.0, 1.0, 1.5)
    assert np.linalg.det(s.matrix()) == pytest.approx(1.0 / 1.5)


def test_flat_surface_does_not_bend_axis_parallel_ray():
    s = RefractingSurface(float("inf"), 1.0, 1.5)
    out = s.apply(Ray(2.0, 0.0))
    assert out.y == pytest.approx(2.0) and out.u == pytest.approx(0.0)


def test_thick_lens_matches_lensmaker():
    r1, r2, t = 100.0, -100.0, 5.0
    lens = ThickLens(r1, r2, t, N_BK7)
    n = lens.n
    inv_f = (n - 1) * (1/r1 - 1/r2 + (n - 1) * t / (n * r1 * r2))
    assert lens.efl() == pytest.approx(1.0 / inv_f)


def test_thin_limit_equals_simple_formula():
    lens = ThickLens(100.0, -100.0, 1e-9, N_BK7)
    n = lens.n
    assert lens.efl() == pytest.approx(1.0 / ((n - 1) * 0.02), rel=1e-6)


def test_plano_convex_bfl_formula():
    r1, t = 50.0, 5.0
    lens = ThickLens(r1, float("inf"), t, N_BK7)
    n, f = lens.n, lens.efl()
    assert lens.bfl() == pytest.approx(f * (1 - (n - 1) * t / (n * r1)))


def test_parallel_ray_hits_axis_at_bfl():
    lens = ThickLens(80.0, -80.0, 6.0, N_BK7)
    out = lens.apply(Ray(1.0, 0.0))
    y_at_focus = out.y + out.u * lens.bfl()
    assert y_at_focus == pytest.approx(0.0, abs=1e-9)


def test_singlet_has_chromatic_focal_shift():
    bfl_blue = ThickLens(100, -100, 5, N_BK7, LAMBDA_F).bfl()
    bfl_red = ThickLens(100, -100, 5, N_BK7, LAMBDA_C).bfl()
    assert bfl_red - bfl_blue > 0.5      # blue focuses closer than red


def test_achromat_cancels_chromatic_shift():
    r1, r2, t1, t2 = 62.0, -44.0, 4.0, 2.5
    r3 = solve_achromat_r3(r1, r2, t1, t2, N_BK7, N_F2)
    f = lambda wl, r: back_focal_length(cemented_doublet(r1, r2, r, t1, t2, N_BK7, N_F2, wl))
    assert f(LAMBDA_F, r3) == pytest.approx(f(LAMBDA_C, r3), abs=1e-6)
    singlet_shift = (ThickLens(100, -100, 5, N_BK7, LAMBDA_F).bfl()
                     - ThickLens(100, -100, 5, N_BK7, LAMBDA_C).bfl())
    assert abs(f(LAMBDA_F, r3) - f(LAMBDA_C, r3)) < abs(singlet_shift) / 100


def test_solver_rejects_bad_bracket():
    with pytest.raises(ValueError):
        solve_achromat_r3(62, -44, 4, 2.5, N_BK7, N_F2, r3_bracket=(-1000, -500))
