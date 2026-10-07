"""
glass.py  -  LESSON 7: Real glass (refractive index depends on colour)
=====================================================================
THEORY
------
Glass bends blue light more than red light. That is DISPERSION, and it
is why a simple lens has colour fringes (chromatic aberration).

Manufacturers describe the index n(lambda) with the SELLMEIER equation:

    n^2(lambda) = 1 + sum_i  B_i * lambda^2 / (lambda^2 - C_i)

lambda is in MICROMETRES inside the formula; our API takes NANOMETRES.

ABBE NUMBER  V = (n_d - 1) / (n_F - n_C)   -- how weak the dispersion is.
  d = 587.6 nm (yellow), F = 486.1 nm (blue), C = 656.3 nm (red).
  High V (crown glass, e.g. BK7, V = 64): little dispersion.
  Low  V (flint glass, e.g. F2,  V = 36): strong dispersion.
Pairing a crown with a flint lets us cancel colour error (ACHROMAT).
=====================================================================
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

LAMBDA_D, LAMBDA_F, LAMBDA_C = 587.5618, 486.1327, 656.2725   # nm


@dataclass(frozen=True)
class Glass:
    name: str
    B: Tuple[float, float, float]
    C: Tuple[float, float, float]

    def n(self, wavelength_nm: float) -> float:
        """Refractive index at the given wavelength in nanometres."""
        lam2 = (wavelength_nm / 1000.0) ** 2        # micrometres squared
        total = 0.0
        for b, c in zip(self.B, self.C):
            if b:                                    # skip unused terms (air)
                total += b * lam2 / (lam2 - c)
        return float(np.sqrt(1.0 + total))

    def abbe_number(self) -> float:
        nd, nF, nC = self.n(LAMBDA_D), self.n(LAMBDA_F), self.n(LAMBDA_C)
        return (nd - 1.0) / (nF - nC)


AIR = Glass("AIR", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))             # n = 1
N_BK7 = Glass("N-BK7", (1.03961212, 0.231792344, 1.01046945),
              (0.00600069867, 0.0200179144, 103.560653))
FUSED_SILICA = Glass("FUSED-SILICA", (0.6961663, 0.4079426, 0.8974794),
                     (0.00467914826, 0.0135120631, 97.9340025))
N_F2 = Glass("N-F2", (1.39757037, 0.159201403, 1.2686543),
             (0.00995906143, 0.0546931752, 119.248346))
N_SF11 = Glass("N-SF11", (1.73759695, 0.313747346, 1.89878101),
               (0.013188707, 0.0623068142, 155.23629))

CATALOG = {g.name: g for g in (AIR, N_BK7, FUSED_SILICA, N_F2, N_SF11)}


def get_glass(name: str) -> Glass:
    """Look a glass up by name (case-insensitive)."""
    key = name.upper()
    if key not in CATALOG:
        raise KeyError(f"Unknown glass '{name}'. Available: {sorted(CATALOG)}")
    return CATALOG[key]
