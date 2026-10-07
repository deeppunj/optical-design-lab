import numpy as np
import pytest
from opticslab.glass import N_BK7, N_F2, LAMBDA_D, LAMBDA_F, LAMBDA_C
from opticslab.lenses import cemented_doublet, back_focal_length
from opticslab.raytrace import (TraceSurface, doublet_prescription,
                                collimated_beam, trace, best_focus,
                                marginal_axis_crossing, rms_spot_radius,
                                at_image_plane, last_vertex_z)

R1, R2, R3, T1, T2 = 60.0, -34.61924097, -139.861202, 4.0, 2.0


def presc(wl=LAMBDA_D):
    return doublet_prescription(R1, R2, R3, T1, T2, N_BK7, N_F2, wl)


def focus(diam, field=0.0, wl=LAMBDA_D):
    o, d = collimated_beam(diam, 41, field, pattern="grid")
    return best_focus(o, d, presc(wl))


def test_paraxial_limit_matches_matrix_method():
    for wl in (LAMBDA_F, LAMBDA_C):
        sys_ = cemented_doublet(R1, R2, R3, T1, T2, N_BK7, N_F2, wl)
        assert marginal_axis_crossing(presc(wl), 1e-3) == pytest.approx(
            back_focal_length(sys_), abs=1e-4)


def test_plane_parallel_plate_does_not_deviate_rays():
    plate = [TraceSurface(float("inf"), 1.5, 5.0),
             TraceSurface(float("inf"), 1.0, 0.0)]
    o, d = collimated_beam(4.0, 5, field_deg=10.0)
    _, d_out = trace(o, d, plate)
    assert np.abs(d_out - d).max() < 1e-12


def test_edge_rays_focus_closer_spherical_aberration():
    paraxial = marginal_axis_crossing(presc(), 1e-3)
    edge = marginal_axis_crossing(presc(), 10.0)
    assert 0.1 < edge - paraxial < 2.0
    assert edge > paraxial        # this doublet is over-corrected at the edge


def test_spot_grows_faster_than_linearly_with_aperture():
    r5, r10, r20 = focus(5)[1], focus(10)[1], focus(20)[1]
    assert r10 > 4 * r5
    assert r20 > 4 * r10


def test_off_axis_spot_is_worse_than_on_axis():
    assert focus(10, field=2.0)[1] > focus(10, field=0.0)[1]


def test_f_and_c_best_foci_are_close_for_achromat():
    zf, zc = focus(5, wl=LAMBDA_F)[0], focus(5, wl=LAMBDA_C)[0]
    assert abs(zf - zc) < 0.05


def test_ray_missing_a_surface_gives_nan():
    tiny = [TraceSurface(5.0, 1.5, 0.0)]
    o, d = collimated_beam(20.0, 3)
    p, _ = trace(o, d, tiny)
    assert np.isnan(p[0][0]).any() or np.isnan(p[0][-1]).any()


def test_total_internal_reflection_gives_nan():
    flat = [TraceSurface(float("inf"), 1.0, 0.0)]
    o, d = collimated_beam(2.0, 3, field_deg=50.0)
    _, d_out = trace(o, d, flat, n_start=1.5)
    assert np.isnan(d_out).all()
