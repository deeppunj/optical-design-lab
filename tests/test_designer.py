import numpy as np
import pytest
from opticslab import designer as dz
from opticslab.glass import LAMBDA_D, N_BK7
from opticslab.raytrace import at_image_plane, last_vertex_z

def singlet():
    return [dz.SurfaceRow(50.0, 5.0, "N-BK7"), dz.SurfaceRow(-50.0, 45.0, "AIR")]

def test_efl_matches_thick_lens_formula():
    n, R1, R2, t = N_BK7.n(LAMBDA_D), 50.0, -50.0, 5.0
    inv_f = (n - 1) * (1 / R1 - 1 / R2 + (n - 1) * t / (n * R1 * R2))
    efl, bfl = dz.paraxial(singlet(), LAMBDA_D)
    assert efl == pytest.approx(1 / inv_f, rel=1e-9)
    assert bfl < efl + 1e-9 and bfl > 0

def test_flat_radius_zero_is_flat():
    rows = [dz.SurfaceRow(0.0, 4.0, "N-BK7"), dz.SurfaceRow(-30.0, 50.0, "AIR")]
    efl, _ = dz.paraxial(rows, LAMBDA_D)
    n = N_BK7.n(LAMBDA_D)
    assert efl == pytest.approx(30.0 / (n - 1), rel=1e-6)

def test_best_focus_matches_paraxial_for_tiny_beam():
    rows = singlet()
    px, py = dz.hex_pupil(0.2, 4)
    surfs, o, pts, dd = dz.trace_bundle(rows, LAMBDA_D, px, py, 0.0)
    _, bfl = dz.paraxial(rows, LAMBDA_D)
    z_last = last_vertex_z(surfs)
    z = dz.best_focus_z(pts, dd, z_last + 0.05, z_last + 200)
    assert z - z_last == pytest.approx(bfl, abs=0.05)

def test_spherical_aberration_pulls_focus_in():
    rows = singlet()
    _, bfl = dz.paraxial(rows, LAMBDA_D)
    px, py = dz.hex_pupil(20.0, 6)
    surfs, o, pts, dd = dz.trace_bundle(rows, LAMBDA_D, px, py, 0.0)
    z_last = last_vertex_z(surfs)
    z = dz.best_focus_z(pts, dd, z_last + 0.05, z_last + 200)
    assert z - z_last < bfl

def test_chromatic_ordering_and_doublet_correction():
    t = dz.chromatic_table(singlet())
    assert t[0]["efl"] < t[1]["efl"] < t[2]["efl"]
    s_single = abs(dz.axial_colour(singlet()))
    s_doublet = abs(dz.axial_colour(dz.EXAMPLES["Achromat N-BK7/N-F2"]))
    assert s_doublet < 0.02 * s_single

def test_custom_index_used():
    rows = [dz.SurfaceRow(50.0, 5.0, dz.CUSTOM, 1.5), dz.SurfaceRow(-50.0, 40.0, "AIR")]
    assert dz.build_surfaces(rows, 550.0)[0].n_after == 1.5

def test_validate_rejects_bad_input():
    with pytest.raises(ValueError):
        dz.validate([])
    with pytest.raises(ValueError):
        dz.validate([dz.SurfaceRow(10, 1, dz.CUSTOM, 0.5)])

def test_json_roundtrip(tmp_path):
    p = tmp_path / "lens.json"
    rows = dz.EXAMPLES["Achromat N-BK7/N-F2"]
    dz.save_design(p, rows, 12.0, 3.0, 532.0)
    r2, epd, fld, wl = dz.load_design(p)
    assert r2 == rows and (epd, fld, wl) == (12.0, 3.0, 532.0)

def test_sag_sign():
    h = np.array([0.0, 5.0])
    assert dz.sag_profile(50.0, h)[1] > 0 > dz.sag_profile(-50.0, h)[1]
    assert np.all(dz.sag_profile(0.0, h) == 0)

def test_achromat_efl_is_100mm():
    efl, _ = dz.paraxial(dz.EXAMPLES["Achromat N-BK7/N-F2"], LAMBDA_D)
    assert efl == pytest.approx(100.0, abs=1e-3)
