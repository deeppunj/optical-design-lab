# Physics sanity checks
"""Physics sanity checks: if these pass, the mathematical functions work correctly."""
import pytest
from opticslab import (Ray, Space, ThinLens, Aperture, OpticalSystem,
                       effective_focal_length, image_distance_and_magnification)

def test_efl_of_single_lens():
    assert effective_focal_length(OpticalSystem([ThinLens(100)])) == pytest.approx(100)

def test_lens_formula_and_magnification():
    d, m = image_distance_and_magnification(OpticalSystem([ThinLens(100)]), 300)
    assert d == pytest.approx(150)           # 1/300 + 1/150 = 1/100
    assert m == pytest.approx(-0.5)

def test_all_rays_meet_at_image():
    s = OpticalSystem([Space(300), ThinLens(100), Space(150)])
    heights = [s.trace(Ray(5, (yl - 5) / 300))[-1].y for yl in (-20, 0, 20)]
    assert heights == pytest.approx([-2.5] * 3)   # all land at y = m*5

def test_parallel_ray_crosses_axis_at_focus():
    out = OpticalSystem([ThinLens(100), Space(100)]).trace(Ray(10, 0))[-1]
    assert out.y == pytest.approx(0)

def test_aperture_blocks():
    out = OpticalSystem([Aperture(5)]).trace(Ray(10, 0))[-1]
    assert out.blocked
