"""Stage 2 physics checks. Run with: pytest -v"""
import pytest
from opticslab import Ray, Space, ThinLens, Aperture, OpticalSystem
from opticslab.builders import kepler_expander, galilean_expander, beam_magnification
from opticslab.stops import find_stop, marginal_ray, chief_ray, with_object_space


def test_kepler_is_afocal_and_inverts():
    s = kepler_expander(50, 150)
    assert s.matrix()[1, 0] == pytest.approx(0.0, abs=1e-12)   # C = 0
    assert beam_magnification(s) == pytest.approx(-3.0)        # -f2/f1


def test_galilean_is_afocal_and_upright():
    s = galilean_expander(-50, 150)
    assert s.matrix()[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert beam_magnification(s) == pytest.approx(3.0)
    assert s.elements[1].d == pytest.approx(100.0)             # f1 + f2


def test_parallel_ray_stays_parallel():
    for s, m in [(kepler_expander(50, 150), -3.0),
                 (galilean_expander(-50, 150), 3.0)]:
        end = s.trace(Ray(1.0, 0.0))[-1]
        assert end.y == pytest.approx(m)
        assert end.u == pytest.approx(0.0, abs=1e-12)


def test_bad_inputs():
    with pytest.raises(ValueError):
        kepler_expander(50, -150)
    with pytest.raises(ValueError):
        galilean_expander(-80, 50)       # lenses would overlap


def _sys():
    # small stop (R=10) at the lens, large aperture (R=50) later
    return OpticalSystem([Aperture(10), ThinLens(100), Space(50), Aperture(50)])


def test_find_stop():
    assert find_stop(_sys(), 300).radius == 10


def test_marginal_ray_touches_stop_edge():
    s = _sys()
    m = marginal_ray(s, 300)
    tr = with_object_space(s, 300).trace(m, z_start=-300)
    assert tr[2].y == pytest.approx(10.0)      # tr[2] = after Aperture(10)


def test_chief_ray_crosses_stop_centre():
    s = _sys()
    c = chief_ray(s, 300, 5.0)
    tr = with_object_space(s, 300).trace(c, z_start=-300)
    assert tr[2].y == pytest.approx(0.0, abs=1e-12)
