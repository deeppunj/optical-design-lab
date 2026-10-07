"""Plots BFL vs wavelength for a BK7 singlet and the solved achromat."""
import numpy as np
import matplotlib.pyplot as plt
from opticslab.glass import N_BK7, N_F2
from opticslab.lenses import ThickLens, cemented_doublet
from opticslab.solver import design_achromat, focal_shift_curve

res = design_achromat(100.0, 60.0, 4.0, 2.0, N_BK7, N_F2)
print(res)

wl = np.linspace(450, 700, 51)
singlet = np.array([ThickLens(60, -60, 4, N_BK7, w).bfl() for w in wl])
doublet = focal_shift_curve(
    lambda w: cemented_doublet(res.r1, res.r2, res.r3, 4.0, 2.0, N_BK7, N_F2, w), wl)

plt.plot(wl, singlet - singlet.mean(), label="BK7 singlet")
plt.plot(wl, doublet - doublet.mean(), label="BK7/F2 achromat")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Wavelength (nm)"); plt.ylabel("Focal shift (mm)")
plt.title("Longitudinal chromatic aberration"); plt.legend(); plt.show()
