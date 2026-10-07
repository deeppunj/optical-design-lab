"""
demo_chromatic.py - focus position vs wavelength: singlet vs achromat.
Run:  python examples/demo_chromatic.py
Look for: the BK7 singlet focus drifts strongly with colour; the
BK7/F2 achromat is much flatter (the curve is a shallow U-shape).
"""
import numpy as np
import matplotlib.pyplot as plt
from opticslab import (N_BK7, N_F2, ThickLens, cemented_doublet,
                       back_focal_length, solve_achromat_r3)

wavelengths = np.linspace(450, 700, 51)                 # nm
r1, r2, t1, t2 = 62.0, -44.0, 4.0, 2.5
r3 = solve_achromat_r3(r1, r2, t1, t2, N_BK7, N_F2)
print(f"Solved third radius R3 = {r3:.2f} mm")

singlet = [ThickLens(100, -100, 5, N_BK7, w).bfl() for w in wavelengths]
doublet = [back_focal_length(cemented_doublet(r1, r2, r3, t1, t2, N_BK7, N_F2, w))
           for w in wavelengths]

print(f"Singlet focus drift 450-700 nm: {max(singlet) - min(singlet):.2f} mm")
print(f"Doublet focus drift 450-700 nm: {max(doublet) - min(doublet):.2f} mm")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, data, title in [(axes[0], singlet, "BK7 singlet (R=100/-100, t=5)"),
                        (axes[1], doublet, f"BK7/F2 achromat (R3={r3:.1f})")]:
    ax.plot(wavelengths, data)
    ax.set_title(title)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("back focal length (mm)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("demo_chromatic.png", dpi=150)
plt.show()
