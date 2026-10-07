import matplotlib.pyplot as plt
from opticslab.glass import N_BK7, N_F2, LAMBDA_D
from opticslab.raytrace import doublet_prescription, plot_spot_and_fan

presc = doublet_prescription(60.0, -34.61924097, -139.861202, 4.0, 2.0,
                             N_BK7, N_F2, LAMBDA_D)
plot_spot_and_fan(presc, diameter=10.0, field_deg=0.0)
plot_spot_and_fan(presc, diameter=10.0, field_deg=2.0)
plt.show()
