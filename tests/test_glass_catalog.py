"""Validate opticslab's Sellmeier glass data against published catalog values.

Two independent checks:
1. Schott datasheet values (nd, nF, nC, Abbe number), rounded as published.
2. Optiland's bundled Schott catalog, compared over 400-700 nm.
"""
import numpy as np
import pytest

from opticslab.glass import LAMBDA_C, LAMBDA_D, LAMBDA_F, N_BK7, N_F2

# name: (glass, nd, nF, nC, Abbe vd, Abbe tolerance)
PUBLISHED = {
    "N-BK7": (N_BK7, 1.51680, 1.52238, 1.51432, 64.17, 0.05),
    "N-F2": (N_F2, 1.62004, 1.63208, 1.61503, 36.40, 0.10),
}


@pytest.mark.parametrize("name", PUBLISHED)
def test_indices_match_datasheet(name):
    glass, nd, nF, nC, _, _ = PUBLISHED[name]
    assert glass.n(LAMBDA_D) == pytest.approx(nd, abs=5e-5)
    assert glass.n(LAMBDA_F) == pytest.approx(nF, abs=5e-5)
    assert glass.n(LAMBDA_C) == pytest.approx(nC, abs=5e-5)


@pytest.mark.parametrize("name", PUBLISHED)
def test_abbe_number_matches_datasheet(name):
    glass, *_, vd, tol = PUBLISHED[name]
    assert glass.abbe_number() == pytest.approx(vd, abs=tol)


@pytest.mark.parametrize("name", PUBLISHED)
def test_dispersion_is_normal(name):
    glass = PUBLISHED[name][0]
    wl = np.linspace(400, 700, 31)
    n = np.array([glass.n(w) for w in wl])
    assert np.all(np.diff(n) < 0), "n must fall monotonically with wavelength"


@pytest.mark.parametrize("name", PUBLISHED)
def test_matches_optiland_catalog_400_700nm(name):
    pytest.importorskip("optiland")
    from optiland.materials import Material

    ref = Material(name, reference="SCHOTT")
    for wl_nm in np.linspace(400, 700, 13):
        ref_n = float(np.ravel(ref.n(wl_nm / 1000.0))[0])
        assert PUBLISHED[name][0].n(wl_nm) == pytest.approx(ref_n, abs=2e-5)


if __name__ == "__main__":
    from optiland.materials import Material
    print(f"{'Glass':<7} {'nd':>9} {'Abbe':>7} {'max |dn| vs Optiland':>22}")
    for name, (g, *_rest) in PUBLISHED.items():
        ref = Material(name, reference="SCHOTT")
        dn = max(abs(g.n(w) - float(np.ravel(ref.n(w / 1000))[0]))
                 for w in np.linspace(400, 700, 13))
        print(f"{name:<7} {g.n(LAMBDA_D):9.6f} {g.abbe_number():7.3f} {dn:22.2e}")
