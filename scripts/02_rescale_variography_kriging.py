"""
02_rescale_variography_kriging.py

Rescale Cu to a realistic porphyry-style range (a linear rescale preserves
all spatial statistics -- variogram shape, correlation, kriging weights --
only the magnitude changes), then run the full variography / ordinary
kriging / cross-validation / IDW-comparison pipeline on the rescaled data.

Overwrites outputs/figs/fig01-04 and outputs/data/walker_lake_clean.csv /
cu_kriged_grid.csv written by 01_load_and_qc.py.

Requires: 01_load_and_qc.py has already been run.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skgstat as skg
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import KFold

from _paths import OUT_DATA, OUT_FIGS

DATA = OUT_DATA / "walker_lake_clean.csv"

df = pd.read_csv(DATA)

# ---- rescale: linear map so the sample mean lands at a realistic porphyry
# Cu grade (~0.45%). Linear rescaling changes only magnitude; correlation,
# variogram shape (range), and relative spatial pattern are unchanged.
K = 0.10
df["Cu_pct_raw"] = df["Cu_pct"]
df["Cu_pct"] = df["Cu_pct_raw"] * K
print(f"Rescale factor K={K}")
print(df["Cu_pct"].describe())

df.to_csv(DATA, index=False)  # overwrite with rescaled Cu

# ---- QC figures (overwrite fig01, fig02) -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["Cu_pct"].dropna(), bins=30, color="#4C78A8", edgecolor="white")
axes[0].set_xlabel("Cu grade (%)"); axes[0].set_ylabel("Frequency")
axes[0].set_title(f"Cu grade distribution (n={df['Cu_pct'].notna().sum()})")
axes[1].hist(df["Au_ppm"].dropna(), bins=30, color="#F58518", edgecolor="white")
axes[1].set_xlabel("Au grade analogue (ppm)")
axes[1].set_title(f"Au grade distribution (n={df['Au_ppm'].notna().sum()})")
plt.tight_layout(); plt.savefig(OUT_FIGS / "fig01_histograms.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(df["X"], df["Y"], c=df["Cu_pct"], cmap="viridis", s=18, edgecolor="k", linewidth=0.2)
ax.set_xlabel("Easting (local grid, m)"); ax.set_ylabel("Northing (local grid, m)")
ax.set_title("Sample location map coloured by Cu grade (%)")
ax.set_aspect("equal"); plt.colorbar(sc, ax=ax, label="Cu (%)")
plt.tight_layout(); plt.savefig(OUT_FIGS / "fig02_locmap.png", dpi=160); plt.close()

# ---- variography ---------------------------------------------------------
coords = df[["X", "Y"]].to_numpy()
values = df["Cu_pct"].to_numpy()
V = skg.Variogram(coords, values, model="spherical", n_lags=20, maxlag=150, estimator="matheron")
rng, sill, nugget = V.parameters
print(f"\nrange={rng:.2f} m, sill={sill:.4f}, nugget={nugget:.4f}")

fig, ax = plt.subplots(figsize=(6, 4.2))
V.plot(axes=ax, show=False)
ax.set_title("Experimental variogram and fitted spherical model (Cu %)")
plt.tight_layout(); plt.savefig(OUT_FIGS / "fig03_variogram.png", dpi=160); plt.close()

# ---- ordinary kriging -----------------------------------------------------
gridx = np.arange(0, 265, 5.0); gridy = np.arange(0, 305, 5.0)
OK = OrdinaryKriging(df.X, df.Y, values, variogram_model="spherical",
                      variogram_parameters={"sill": sill, "range": rng, "nugget": nugget})
z, ss = OK.execute("grid", gridx, gridy)
z = np.clip(z, 0, None)

gx, gy = np.meshgrid(gridx, gridy)
grid_df = pd.DataFrame({"X": gx.ravel(), "Y": gy.ravel(),
                         "Cu_ok": np.asarray(z).ravel(), "Cu_ok_var": np.asarray(ss).ravel()})
grid_df.to_csv(OUT_DATA / "cu_kriged_grid.csv", index=False)
print(f"\nKriged grid: min={grid_df.Cu_ok.min():.3f} mean={grid_df.Cu_ok.mean():.3f} max={grid_df.Cu_ok.max():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
im0 = axes[0].pcolormesh(gridx, gridy, z, cmap="viridis", shading="auto")
axes[0].scatter(df.X, df.Y, s=4, c="white", edgecolor="k", linewidth=0.2)
axes[0].set_title("Ordinary kriging estimate: Cu (%)")
axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)"); axes[0].set_aspect("equal")
plt.colorbar(im0, ax=axes[0], label="Cu (%)")
im1 = axes[1].pcolormesh(gridx, gridy, ss, cmap="magma", shading="auto")
axes[1].scatter(df.X, df.Y, s=4, c="white", edgecolor="k", linewidth=0.2)
axes[1].set_title("Kriging variance (Cu %)")
axes[1].set_xlabel("Easting (m)"); axes[1].set_ylabel("Northing (m)"); axes[1].set_aspect("equal")
plt.colorbar(im1, ax=axes[1], label="Variance")
plt.tight_layout(); plt.savefig(OUT_FIGS / "fig04_kriging_maps.png", dpi=160); plt.close()

# ---- cross-validation: OK vs IDW ------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_all, Y_all = df.X.to_numpy(), df.Y.to_numpy()

def idw_predict(train_xy, train_v, query_xy, power=2.0):
    d = np.sqrt(((query_xy[:, None, :] - train_xy[None, :, :]) ** 2).sum(-1))
    d = np.where(d == 0, 1e-6, d)
    w = 1.0 / d ** power
    return (w * train_v[None, :]).sum(1) / w.sum(1)

ok_errs, idw_errs = [], []
for tr, te in kf.split(coords):
    ok_cv = OrdinaryKriging(X_all[tr], Y_all[tr], values[tr], variogram_model="spherical",
                             variogram_parameters={"sill": sill, "range": rng, "nugget": nugget})
    zhat, _ = ok_cv.execute("points", X_all[te], Y_all[te])
    ok_errs.append(values[te] - np.asarray(zhat))
    idw_pred = idw_predict(coords[tr], values[tr], coords[te])
    idw_errs.append(values[te] - idw_pred)
ok_errs = np.concatenate(ok_errs); idw_errs = np.concatenate(idw_errs)
ok_rmse = np.sqrt((ok_errs ** 2).mean()); idw_rmse = np.sqrt((idw_errs ** 2).mean())
print(f"\nOK RMSE={ok_rmse:.4f}   IDW RMSE={idw_rmse:.4f}   data std={values.std():.4f}")
print(f"OK improvement over IDW: {100*(1-ok_rmse/idw_rmse):.1f}%")

with open(OUT_DATA / "rescaled_stats.txt", "w") as f:
    f.write(f"K={K}\nrange={rng:.3f}\nsill={sill:.5f}\nnugget={nugget:.5f}\n")
    f.write(f"grid_min={grid_df.Cu_ok.min():.4f}\ngrid_mean={grid_df.Cu_ok.mean():.4f}\ngrid_max={grid_df.Cu_ok.max():.4f}\n")
    f.write(f"ok_rmse={ok_rmse:.5f}\nidw_rmse={idw_rmse:.5f}\ndata_std={values.std():.5f}\n")
    f.write(f"cu_mean={df.Cu_pct.mean():.4f}\ncu_std={df.Cu_pct.std():.4f}\ncu_min={df.Cu_pct.min():.4f}\ncu_max={df.Cu_pct.max():.4f}\n")
print("\nSaved rescaled_stats.txt")
