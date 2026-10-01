"""
01_load_and_qc.py

Load and QC the Walker Lake case-study dataset (Isaaks & Srivastava, 1989),
distributed with Cu(%) / Au(ppm) grade-analogue labels via the open-source
pyLPM library (robertorolo/pyLPM, MIT licence).

Source file: data/raw/Walker_Lake.txt (GSLIB-style text format, copied
verbatim from pyLPM/pyLPM/datasets/Walker_Lake.txt).

Output:
  outputs/data/walker_lake_clean.csv
  outputs/figs/fig01_histograms.png
  outputs/figs/fig02_locmap.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _paths import RAW_DIR, OUT_DATA, OUT_FIGS

RAW = RAW_DIR / "Walker_Lake.txt"

# --- parse GSLIB-style file: line1 title, line2 ncols, next ncols lines = names, then data ---
with open(RAW) as f:
    lines = [l.rstrip("\n") for l in f]

title = lines[0]
ncols = int(lines[1].strip())
colnames = [lines[2 + i].strip() for i in range(ncols)]
data_lines = lines[2 + ncols:]

rows = []
for l in data_lines:
    if not l.strip():
        continue
    parts = l.split()
    rows.append([float(p) for p in parts[:ncols]])

df = pd.DataFrame(rows, columns=colnames)
df.columns = ["X", "Y", "Cu_pct", "Au_ppm"]

# missing value code -999 -> NaN (Au was destructively assayed on a subset only,
# mimicking a common secondary-variable sampling pattern in real campaigns)
df["Au_ppm"] = df["Au_ppm"].replace(-999, np.nan)
df["Au_ppm"] = df["Au_ppm"].where(df["Au_ppm"] >= 0, np.nan)

df.insert(0, "SampleID", np.arange(1, len(df) + 1))
df.to_csv(OUT_DATA / "walker_lake_clean.csv", index=False)

print("Title in source file:", title)
print("n samples:", len(df))
print(df.describe(include="all"))
print("\nMissing Au (%):", round(100 * df["Au_ppm"].isna().mean(), 1))

# quick QC figures (overwritten with a realistic Cu rescale by 02_*) -------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["Cu_pct"].dropna(), bins=30, color="#4C78A8", edgecolor="white")
axes[0].set_xlabel("Cu grade analogue (%)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Cu grade distribution (n={})".format(df["Cu_pct"].notna().sum()))

axes[1].hist(df["Au_ppm"].dropna(), bins=30, color="#F58518", edgecolor="white")
axes[1].set_xlabel("Au grade analogue (ppm)")
axes[1].set_title("Au grade distribution (n={})".format(df["Au_ppm"].notna().sum()))
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig01_histograms.png", dpi=160)
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(df["X"], df["Y"], c=df["Cu_pct"], cmap="viridis", s=18, edgecolor="k", linewidth=0.2)
ax.set_xlabel("Easting (local grid, m)")
ax.set_ylabel("Northing (local grid, m)")
ax.set_title("Sample location map coloured by Cu grade analogue (%)")
ax.set_aspect("equal")
plt.colorbar(sc, ax=ax, label="Cu (%)")
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig02_locmap.png", dpi=160)
plt.close()

print("\nSaved:", OUT_DATA / "walker_lake_clean.csv")
print("Saved figs: fig01_histograms.png, fig02_locmap.png")
