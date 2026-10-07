# runnable lesson scripts
"""
demo_single_lens.py - your FIRST optical simulation.
Object arrow 5 mm tall, 300 mm in front of a lens with f = 100 mm.
School optics predicts: 1/300 + 1/d = 1/100 -> d = 150 mm,
magnification m = -d/s = -0.5 (half size, upside-down).
Run it and see whether the program agrees!
"""
import matplotlib.pyplot as plt
import numpy as np
from opticslab import (Ray, Space, ThinLens, OpticalSystem,
                       effective_focal_length, image_distance_and_magnification, Aperture)

S, F, H = 200.0, 100.0, 5.0       # S = object distance, F = focal length, H = object height (mm)

lens_only = OpticalSystem([ThinLens(F)])
d, m = image_distance_and_magnification(lens_only, S)
print(f"EFL            = {effective_focal_length(lens_only):.1f} mm")
print(f"Image distance = {d:.1f} mm behind the lens")
print(f"Magnification  = {m:.3f}  (negative = upside-down)")

# Full system for tracing: object space -> lens -> image space
system = OpticalSystem([Space(S), ThinLens(F), Space(d)])

# Full system for tracing: object space -> Aperture -> lens -> image space
#system = OpticalSystem([Space(S), Aperture(10), ThinLens(F), Space(d)])

fig, ax = plt.subplots(figsize=(10, 5))
for y_lens in np.linspace(-20, 20, 9):          # aim 9 rays at different lens heights
    u = (y_lens - H) / S                        # slope needed to reach that height
    pts = system.trace(Ray(y=H, u=u))
    ax.plot([p.z for p in pts], [p.y for p in pts], lw=1)
ax.axvline(S, color="k", lw=3)                  # the lens
ax.annotate("", xy=(0, H), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="g", lw=2))
ax.annotate("", xy=(S + d, m * H), xytext=(S + d, 0), arrowprops=dict(arrowstyle="->", color="r", lw=2))
ax.axhline(0, color="gray", lw=0.5)
ax.set(xlabel="z along optical axis (mm)", ylabel="height y (mm)",
       title=f"Single lens f={F:g} mm: image at {d:.0f} mm, m={m:.2f}")
plt.savefig("demo_single_lens.png", dpi=150)
plt.show()
