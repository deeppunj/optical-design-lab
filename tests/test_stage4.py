import numpy as np
import pytest
from opticslab.glass import N_BK7, N_F2, LAMBDA_F, LAMBDA_C
from opticslab.lenses import ThickLens, cemented_doublet, back_focal_length
from opticslab.solver import design_achromat, system_efl, focal_shift_curve


@pytest.fixture(scope="module")
def result():
    return design_achromat(100.0, 60.0, 4.0, 2.0, N_BK7, N_F2)


def test_solver_succeeds(result):
    assert result.success


def test_efl_hits_target(result):
    assert result.efl == pytest.approx(100.0, rel=1e-4)


def test_colors_focus_together(result):
    assert abs(result.color_shift) < 1e-3          # mm


def test_radii_have_expected_signs(result):
    assert result.r1 > 0 and result.r2 < 0 and result.r3 < 0


def test_singlet_has_large_color_shift(result):
    def bfl(wl):
        return ThickLens(60.0, -60.0, 4.0, N_BK7, wl).bfl()
    singlet_shift = bfl(LAMBDA_F) - bfl(LAMBDA_C)
    assert abs(singlet_shift) > 0.5
    assert abs(result.color_shift) < abs(singlet_shift) / 100


def test_secondary_spectrum_is_small_but_nonzero(result):
    def build(wl):
        return cemented_doublet(result.r1, result.r2, result.r3, 4.0, 2.0,
                                N_BK7, N_F2, wl)
    bfls = focal_shift_curve(build, [450, 500, 550, 600, 650, 700])
    spread = bfls.max() - bfls.min()
    assert 0.01 < spread < 0.5


def test_bad_inputs():
    with pytest.raises(ValueError):
        design_achromat(-5.0, 60.0, 4.0, 2.0, N_BK7, N_F2)
    with pytest.raises(ValueError):
        design_achromat(100.0, 60.0, 4.0, 2.0, N_BK7, N_BK7)
