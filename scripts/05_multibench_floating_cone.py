"""
05_multibench_floating_cone.py

Build a genuine multi-bench (3-D) block model by vertically extending the
2-D kriged Cu grid into a 3-bench (30 m) column: the kriged grid becomes
the grade of a single buried mineralised horizon (bench 2, 10-20 m depth)
-- a simple, explicitly synthetic stand-in for a sediment-hosted /
stratabound tabular copper deposit under barren cover, built specifically
so that pit-limit optimisation has real overburden to strip. Bench 1
(cover) and bench 3 (footwall) carry a low uniform background grade
representing barren host rock.

A real (heuristic, not globally optimal) floating-cone algorithm is then
run on the resulting 3-D block model to derive an actual pit outline.

Requires: 02_rescale_variography_kriging.py has already been run.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _paths import OUT_DATA, OUT_FIGS

GRID = OUT_DATA / "cu_kriged_grid.csv"

grid = pd.read_csv(GRID)
gx = np.sort(grid.X.unique()); gy = np.sort(grid.Y.unique())
nX, nY = len(gx), len(gy)
Z_ore = grid.pivot(index="Y", columns="X", values="Cu_ok").values  # (nY, nX), bench-2 grade

N_BENCH = 3
BENCH_HEIGHT = 10.0   # m
CELL = 5.0            # m (grid spacing)
DENSITY = 2.70
TONNES_PER_BLOCK = CELL * CELL * BENCH_HEIGHT * DENSITY
BACKGROUND_GRADE = 0.01   # % Cu, barren cover/footwall host rock
ORE_BENCH = 2             # 1-indexed, top = 1 (bench 1 = cover, bench 2 = ore horizon, bench 3 = footwall)

# grade(bench, y, x): bench 2 = kriged grid, others = background
grade = np.full((N_BENCH, nY, nX), BACKGROUND_GRADE)
grade[ORE_BENCH - 1] = Z_ore

# ---- economics (same parameters as the single-bench version) -------------
PRICE, RECOVERY = 14800.0, 0.85
MINING_COST, PROCESS_COST = 3.5, 12.0
PIT_COG = 100 * (MINING_COST + PROCESS_COST) / (PRICE * RECOVERY)   # 0.123 %
print(f"Pit-limit cut-off = {PIT_COG:.3f}% Cu, background grade = {BACKGROUND_GRADE}% Cu")

tonnes = np.full_like(grade, TONNES_PER_BLOCK)
metal_t = tonnes * grade / 100.0
is_ore = grade >= PIT_COG
revenue = np.where(is_ore, metal_t * RECOVERY * PRICE, 0.0)
cost = np.where(is_ore, tonnes * (MINING_COST + PROCESS_COST), tonnes * MINING_COST)
block_value = revenue - cost   # (N_BENCH, nY, nX)

print(f"Ore blocks (of {grade.size}): {is_ore.sum()}  "
      f"({100*is_ore.sum()/grade.size:.1f}% of all blocks)")
for b in range(N_BENCH):
    print(f"  bench {b+1}: {is_ore[b].sum()} ore blocks / {nY*nX}")

# ---- floating cone heuristic ----------------------------------------------
SLOPE_DEG = 45.0
tan_slope = np.tan(np.radians(SLOPE_DEG))

def disk_offsets(radius_cells):
    r = int(np.ceil(radius_cells))
    if r == 0:
        return np.array([[0, 0]])
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    mask = (yy**2 + xx**2) <= radius_cells**2 + 1e-9
    return np.column_stack([yy[mask], xx[mask]])

# precompute cone offsets: for an apex in bench b (1-indexed), for each
# shallower bench s < b, the disk of cells (relative row/col) removed.
max_diff = N_BENCH - 1
cone_offsets = {}
for diff in range(1, max_diff + 1):
    h = diff * BENCH_HEIGHT
    radius_cells = (h / tan_slope) / CELL
    cone_offsets[diff] = disk_offsets(radius_cells)
    print(f"  bench-diff {diff} (h={h:.0f} m): cone radius = {radius_cells:.2f} cells, "
          f"{len(cone_offsets[diff])} cells/level")

removed = np.zeros_like(is_ore, dtype=bool)
apex_candidates = [(block_value[b, r, c], b, r, c)
                    for b in range(N_BENCH) for r in range(nY) for c in range(nX)
                    if is_ore[b, r, c] and block_value[b, r, c] > 0]
apex_candidates.sort(key=lambda t: -t[0])
print(f"\n{len(apex_candidates)} candidate ore apexes to test (floating cone)")

extracted_value = 0.0
n_pits_formed = 0
for val, b, r, c in apex_candidates:
    if removed[b, r, c]:
        continue
    # gather cone cells across all shallower benches (0-indexed bench < b)
    cells = [(b, r, c)]
    cone_val = block_value[b, r, c]
    for diff in range(1, b + 1):  # b is 0-indexed depth; shallower benches 0..b-1
        s = b - diff
        offs = cone_offsets[diff]
        rr = r + offs[:, 0]; cc = c + offs[:, 1]
        valid = (rr >= 0) & (rr < nY) & (cc >= 0) & (cc < nX)
        rr, cc = rr[valid], cc[valid]
        for ri, ci in zip(rr, cc):
            if not removed[s, ri, ci]:
                cone_val += block_value[s, ri, ci]
                cells.append((s, ri, ci))
    if cone_val > 0:
        for (bb, rr2, cc2) in cells:
            removed[bb, rr2, cc2] = True
        extracted_value += cone_val
        n_pits_formed += 1

pit_ore_t = tonnes[removed & is_ore].sum()
pit_waste_t = tonnes[removed & ~is_ore].sum()
pit_metal_t = metal_t[removed & is_ore].sum()
print(f"\nFloating-cone result: {n_pits_formed} cones accepted")
print(f"  Ore tonnes in pit   : {pit_ore_t:,.0f} t")
print(f"  Waste tonnes in pit : {pit_waste_t:,.0f} t")
print(f"  Strip ratio         : {pit_waste_t/max(pit_ore_t,1):.3f}")
print(f"  Avg grade in pit    : {100*pit_metal_t/max(pit_ore_t,1):.3f}% Cu")
print(f"  Contained metal     : {pit_metal_t:,.1f} t Cu")
print(f"  Net extracted value : US${extracted_value:,.0f}")

with open(OUT_DATA / "floatingcone_summary.txt", "w") as f:
    f.write(f"pit_ore_t={pit_ore_t:.1f}\npit_waste_t={pit_waste_t:.1f}\n")
    f.write(f"strip_ratio={pit_waste_t/max(pit_ore_t,1):.4f}\n")
    f.write(f"avg_grade={100*pit_metal_t/max(pit_ore_t,1):.4f}\n")
    f.write(f"contained_metal_t={pit_metal_t:.2f}\nnet_value={extracted_value:.2f}\n")
    f.write(f"n_cones={n_pits_formed}\nn_candidates={len(apex_candidates)}\n")

np.save(OUT_DATA / "removed_mask.npy", removed)
np.save(OUT_DATA / "block_value_3d.npy", block_value)
np.save(OUT_DATA / "grade_3d.npy", grade)

# ---- figure 1: bench-by-bench extraction footprint (plan view) ----------
fig, axes = plt.subplots(1, N_BENCH, figsize=(4*N_BENCH, 4.2), sharey=True)
for b in range(N_BENCH):
    depth_top = b * BENCH_HEIGHT
    ax = axes[b]
    im = ax.pcolormesh(gx, gy, grade[b], cmap="viridis", vmin=0, vmax=np.nanmax(Z_ore), shading="auto")
    ax.contour(gx, gy, removed[b].astype(float), levels=[0.5], colors="red", linewidths=1.8)
    frac = 100 * removed[b].mean()
    ax.text(0.03, 0.03, f"{frac:.0f}% of bench mined", transform=ax.transAxes,
            fontsize=8, color="white", ha="left", va="bottom",
            bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))
    ax.set_title(f"Bench {b+1} ({depth_top:.0f}-{depth_top+BENCH_HEIGHT:.0f} m depth)")
    ax.set_xlabel("Easting (m)"); ax.set_aspect("equal")
    if b == 0:
        ax.set_ylabel("Northing (m)")
fig.suptitle("Floating-cone pit outline (red) by bench, over block grade (%Cu)", y=1.03)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig09_pit_benches.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- figure 2: vertical cross-section through the high-grade zone --------
# pick a row (Y) with a moderate, visually informative extraction fraction
# (near-0% or near-100% rows don't show the pit's tapered shape clearly)
row_extracted_frac = removed[ORE_BENCH - 1].sum(axis=1) / nX
candidate_rows = np.where((row_extracted_frac > 0.30) & (row_extracted_frac < 0.55))[0]
y_idx = candidate_rows[len(candidate_rows) // 2] if len(candidate_rows) else int(np.argmax(row_extracted_frac))
y_val = gy[y_idx]
print(f"\nCross-section row: Y={y_val:.0f} m ({100*row_extracted_frac[y_idx]:.0f}% of ore bench extracted)")

fig, ax = plt.subplots(figsize=(9, 4.5))
depths_top = np.array([b*BENCH_HEIGHT for b in range(N_BENCH)])
depths_center = depths_top + BENCH_HEIGHT/2
G = grade[:, y_idx, :]
R = removed[:, y_idx, :]
im = ax.pcolormesh(gx, depths_center, G, cmap="viridis", vmin=0, vmax=np.nanmax(Z_ore), shading="auto")
# shade the mined-out (extracted) region translucent white/orange so the pit
# outline against un-mined ground is unambiguous, not just a thin contour line
ax.contourf(gx, depths_center, R.astype(float), levels=[0.5, 1.5], colors=["none"], hatches=["///"], alpha=0)
ax.contour(gx, depths_center, R.astype(float), levels=[0.5], colors="red", linewidths=2.2)
ax.invert_yaxis()
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Depth below surface (m)")
ax.set_title(f"Cross-section at Y={y_val:.0f} m: grade (%Cu), hatched = mined by the floating cone, red = pit limit")
plt.colorbar(im, ax=ax, label="Cu (%)")
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig10_cross_section.png", dpi=160)
plt.close()

print(f"\nSaved figs: fig09_pit_benches.png, fig10_cross_section.png (section at Y={y_val:.0f} m)")
