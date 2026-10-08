"""Stage 7 tests: the optimiser must recover a known-good design, respect
bounds / constraints, never modify its input, and support undo/redo."""
from dataclasses import replace

import numpy as np
import pytest

from opticslab import designer as dz
from opticslab import optimizer as op
from opticslab.glass import LAMBDA_D


def good_achromat():
    return [replace(r) for r in dz.EXAMPLES["Achromat N-BK7/N-F2"]]


def perturbed():
    """The known achromat with all three radii and the image distance wrong."""
    r = good_achromat()
    r[0] = replace(r[0], radius=r[0].radius * 1.12)
    r[1] = replace(r[1], radius=r[1].radius * 0.90)
    r[2] = replace(r[2], radius=r[2].radius * 1.15, thickness=r[2].thickness - 6.0)
    return r


def cfg():
    return op.MeritConfig(epd=10.0, efl_target=100.0, axial_colour_weight=500.0)


def vars_():
    return op.make_variables(good_achromat(), radius_rows=[0, 1, 2], thickness_rows=[2])


def test_optimizer_recovers_efl_and_colour_from_bad_start():
    start = perturbed()
    res = op.optimize(start, vars_(), cfg())
    efl, _ = dz.paraxial(res.rows, LAMBDA_D)
    assert res.merit_end < 0.1 * res.merit_start
    assert efl == pytest.approx(100.0, abs=0.05)
    assert abs(dz.axial_colour(res.rows)) < 0.05
    assert abs(dz.axial_colour(res.rows)) < abs(dz.axial_colour(start))


def test_spot_size_gets_smaller():
    res = op.optimize(perturbed(), vars_(), cfg())
    before = res.report_start["rms_spot_um"][(0.0, LAMBDA_D)]
    after = res.report_end["rms_spot_um"][(0.0, LAMBDA_D)]
    assert after < 0.2 * before


def test_input_rows_are_not_modified():
    start = perturbed()
    snapshot = [replace(r) for r in start]
    op.optimize(start, vars_(), cfg(), max_nfev=10)
    assert start == snapshot


def test_fixed_rows_stay_fixed():
    start = perturbed()
    res = op.optimize(start, vars_(), cfg(), max_nfev=30)
    for i in (0, 1):
        assert res.rows[i].thickness == start[i].thickness
        assert res.rows[i].glass == start[i].glass
    assert res.rows[2].glass == start[2].glass


def test_bounds_are_respected():
    # allow only |R| >= 80 mm: the doublet cannot reach f = 100 inside that box
    v = op.make_variables(good_achromat(), radius_rows=[0, 1, 2], r_min=80.0)
    res = op.optimize(perturbed(), v, cfg(), max_nfev=60)
    for i in (0, 1, 2):
        r = res.rows[i].radius
        assert r == 0.0 or abs(r) >= 80.0 - 1e-6


def test_thickness_bounds_are_respected():
    v = [op.Variable(2, "thickness", 40.0, 90.0)]       # true answer needs ~97
    res = op.optimize(good_achromat(), v, cfg(), max_nfev=30)
    assert 40.0 - 1e-9 <= res.rows[2].thickness <= 90.0 + 1e-9


def test_edge_thickness_constraint_is_active():
    # a strongly curved biconvex lens with a thin centre has a NEGATIVE edge
    thin = [dz.SurfaceRow(20.0, 0.2, "N-BK7"), dz.SurfaceRow(-20.0, 30.0, "AIR")]
    c = op.MeritConfig(epd=20.0, min_center=1.0, min_edge=1.0)
    assert np.any(op._constraint_errors(thin, c) > 0)
    ok = [dz.SurfaceRow(20.0, 8.0, "N-BK7"), dz.SurfaceRow(-20.0, 30.0, "AIR")]
    assert np.all(op._constraint_errors(ok, c) == 0)


def test_constraint_makes_optimizer_thicken_the_lens():
    start = [dz.SurfaceRow(20.0, 0.3, "N-BK7"), dz.SurfaceRow(-20.0, 30.0, "AIR")]
    c = op.MeritConfig(epd=20.0, min_center=1.0, min_edge=1.0, rings=3)
    res = op.optimize(start, [op.Variable(0, "thickness", 0.1, 20.0)], c, max_nfev=50)
    nxt = res.rows[1]
    h = np.array([10.0])
    edge = (res.rows[0].thickness + dz.sag_profile(nxt.radius, h)[0]
            - dz.sag_profile(res.rows[0].radius, h)[0])
    assert res.rows[0].thickness >= 1.0 - 1e-3 and edge >= 1.0 - 1e-3


def test_flat_surface_can_be_a_variable():
    start = [dz.SurfaceRow(0.0, 4.0, "FUSED-SILICA"), dz.SurfaceRow(-30.0, 55.0, "AIR")]
    c = op.MeritConfig(epd=6.0, efl_target=60.0, wavelengths_nm=(LAMBDA_D,))
    v = op.make_variables(start, radius_rows=[0, 1], thickness_rows=[1])
    res = op.optimize(start, v, c, max_nfev=80)
    efl, _ = dz.paraxial(res.rows, LAMBDA_D)
    assert efl == pytest.approx(60.0, abs=0.05)


def test_callback_receives_every_evaluation():
    seen = []
    res = op.optimize(perturbed(), vars_(), cfg(), max_nfev=10,
                      callback=lambda n, m: seen.append((n, m)))
    assert len(seen) == res.n_evals == len(res.merit_history)
    assert [n for n, _ in seen] == list(range(1, len(seen) + 1))


def test_input_validation():
    with pytest.raises(ValueError):
        op.optimize(good_achromat(), [], cfg())
    with pytest.raises(IndexError):
        op.make_variables(good_achromat(), radius_rows=[7])
    with pytest.raises(ValueError):
        op.Variable(0, "curvature", 1.0, 0.5)
    with pytest.raises(ValueError):
        op.Variable(0, "colour", 0.0, 1.0)
    v = op.Variable(0, "thickness", 1.0, 9.0)
    with pytest.raises(ValueError):
        op.optimize(good_achromat(), [v, v], cfg())


def test_apply_vector_roundtrip():
    rows = good_achromat()
    v = vars_()
    u = np.array([x.get(rows) / x.scale for x in v])
    again = op.apply_vector(rows, v, u)
    for a, b in zip(rows, again):
        assert a.radius == pytest.approx(b.radius, rel=1e-9)
        assert a.thickness == pytest.approx(b.thickness, rel=1e-9)


def test_history_undo_redo():
    h = op.DesignHistory()
    a, b = good_achromat(), perturbed()
    assert h.undo(a) is None and not h.can_undo()
    h.push(a)
    back = h.undo(b)
    assert back == a and h.can_redo()
    fwd = h.redo(back)
    assert fwd == b
    h.push(a); h.push(b)
    h.undo(b)
    h.push(a)                       # a new edit clears the redo stack
    assert not h.can_redo()
