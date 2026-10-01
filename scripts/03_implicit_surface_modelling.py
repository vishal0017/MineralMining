"""
03_implicit_surface_modelling.py

A lightweight, dependency-light stand-in for full implicit 3-D geological
modelling packages (e.g. GemPy), which typically need heavy compiled
backends (theano/aesara/pytensor) that are not always installable in a
constrained environment. This script demonstrates the same core idea used
by implicit modellers -- interpolating a scalar field from control data and
extracting a geological contact as an iso-surface of that field -- using
only SciPy's radial basis function interpolator.

We generate a small set of synthetic "borehole" contact picks (top of the
mineralised horizon used in 05_multibench_floating_cone.py, i.e. 20 m depth
plus a smooth structural undulation, sampled with realistic pick error) and
fit an RBF surface to them, then compare it against the true synthetic
surface to quantify how well a sparse borehole programme would recover the
contact geometry.

Standalone: does not depend on the outputs of any other script.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

from _paths import OUT_DATA, OUT_FIGS

rng = np.random.default_rng(7)

# ---- "true" synthetic contact surface: top of the ore horizon at ~20 m,
# with a smooth NE-plunging undulation, unknown to the interpolation step
X = np.linspace(0, 260, 80); Y = np.linspace(0, 300, 90)
XX, YY = np.meshgrid(X, Y)
Z_true = 20.0 + 4.0*np.sin(XX/90.0) + 3.0*np.cos(YY/120.0) + 0.02*(XX - 130)

# ---- synthetic borehole programme: 24 boreholes on an irregular grid,
# each with a contact pick carrying +/-0.8 m realistic logging error
n_holes = 24
bx = rng.uniform(10, 250, n_holes)
by = rng.uniform(10, 290, n_holes)
bz_true = 20.0 + 4.0*np.sin(bx/90.0) + 3.0*np.cos(by/120.0) + 0.02*(bx - 130)
bz_obs = bz_true + rng.normal(0, 0.8, n_holes)

rbf = RBFInterpolator(np.column_stack([bx, by]), bz_obs, kernel="thin_plate_spline", smoothing=0.5)
Z_pred = rbf(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

rmse = np.sqrt(np.mean((Z_pred - Z_true) ** 2))
print(f"Implicit surface (RBF) recovery RMSE vs true synthetic contact: {rmse:.2f} m "
      f"from {n_holes} boreholes")

with open(OUT_DATA / "implicit_surface_stats.txt", "w") as f:
    f.write(f"n_holes={n_holes}\nrmse_m={rmse:.3f}\n")

# ---- figure: true surface, borehole picks, interpolated surface ----------
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(XX, YY, Z_true, cmap="terrain", alpha=0.55, linewidth=0, antialiased=True)
ax1.scatter(bx, by, bz_obs, color="red", s=25, depthshade=False, label="Borehole picks")
ax1.invert_zaxis()
ax1.set_xlabel("Easting (m)"); ax1.set_ylabel("Northing (m)"); ax1.set_zlabel("Depth (m)")
ax1.set_title("True synthetic contact surface\n+ borehole contact picks")
ax1.legend(loc="upper left", fontsize=8)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(XX, YY, Z_pred, cmap="terrain", alpha=0.85, linewidth=0, antialiased=True)
ax2.scatter(bx, by, bz_obs, color="red", s=25, depthshade=False)
ax2.invert_zaxis()
ax2.set_xlabel("Easting (m)"); ax2.set_ylabel("Northing (m)"); ax2.set_zlabel("Depth (m)")
ax2.set_title(f"RBF-interpolated contact surface\n(RMSE = {rmse:.2f} m vs. true)")

plt.tight_layout()
plt.savefig(OUT_FIGS / "fig05_implicit_surface.png", dpi=160)
plt.close()
print("Saved fig05_implicit_surface.png")
