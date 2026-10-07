"""
demo_beam_expanders.py - draws a Kepler and a Galilean beam expander.
Run:  python examples/demo_beam_expanders.py
Look for: parallel rays in, parallel (wider) rays out; Kepler crosses
a real focus inside, Galilean does not.
"""
import numpy as np
import matplotlib.pyplot as plt
from opticslab import Ray, ThinLens, kepler_expander, galilean_expander

systems = [("Kepler  f1=50, f2=150  (M = -3)", kepler_expander(50, 150)),
           ("Galilean  f1=-50, f2=150  (M = +3)", galilean_expander(-50, 150))]

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
for ax, (title, system) in zip(axes, systems):
    for y0 in np.linspace(-3, 3, 7):                 # 7 parallel input rays
        trace = system.trace(Ray(y=y0, u=0.0), z_start=0.0)
        z = [p.z for p in trace]
        y = [p.y for p in trace]
        z.append(z[-1] + 80)                          # extend past last lens
        y.append(y[-1] + trace[-1].u * 80)
        ax.plot(z, y, lw=1)
    z_now = 0.0                                       # draw lens positions
    for el in system.elements:
        z_now += el.length
        if isinstance(el, ThinLens):
            ax.axvline(z_now, color="k", ls="--", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("demo_beam_expanders.png", dpi=150)
plt.show()
