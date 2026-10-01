"""
04_gis_mapping_and_cutoff.py

GIS-based mapping workflow on top of the kriged Cu grid: build vector
layers (sample points, grade-domain polygons, a simple infrastructure
buffer), assign a projected CRS, export standard GIS files (GeoJSON), and
run single-bench cut-off grade theory / grade-tonnage / in-situ block
economic value using the rescaled, realistic Cu grid (mean ~0.435% Cu)
produced by 02_rescale_variography_kriging.py.

NOTE ON UNITS: the Walker Lake case-study dataset is a classic pedagogical
geostatistics dataset (Isaaks & Srivastava, 1989) whose "Cu / Au" labels are
a widely used didactic relabelling of an underlying elevation surface: the
*spatial* statistics (clustering, continuity, anisotropy) are realistic,
but the absolute grade magnitudes are illustrative, not an actual assay
campaign. Cost/price parameters below are illustrative, order-of-magnitude
figures for demonstrating the calculation, not a real feasibility study.

Requires: 02_rescale_variography_kriging.py has already been run.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from matplotlib.patches import Patch

from _paths import OUT_DATA, OUT_FIGS

DATA = OUT_DATA / "walker_lake_clean.csv"
GRID = OUT_DATA / "cu_kriged_grid.csv"

df = pd.read_csv(DATA)
grid = pd.read_csv(GRID)

# ---------------------------------------------------------------- GIS ----
# Treat the local mine grid as tied to an arbitrary UTM false-origin (a very
# common real-world situation: exploration/mine grids are often a local
# system offset from, but parallel to, a real projected CRS). We anchor the
# (0,0) local-grid corner at an arbitrary point inside UTM zone 11N
# (EPSG:32611) purely to demonstrate a realistic reprojection workflow.
UTM_E0, UTM_N0 = 430000.0, 4287000.0
samples = gpd.GeoDataFrame(df, geometry=[Point(UTM_E0+x, UTM_N0+y) for x, y in zip(df.X, df.Y)],
                            crs="EPSG:32611")
samples.to_file(OUT_DATA / "samples_utm.geojson", driver="GeoJSON")

samples_wgs84 = samples.to_crs("EPSG:4326")
print("Sample extent in WGS84 (lon/lat):")
print(samples_wgs84.total_bounds)

gx = np.sort(grid.X.unique()); gy = np.sort(grid.Y.unique())
Z = grid.pivot(index="Y", columns="X", values="Cu_ok").values

ENV_THRESH, HG_THRESH = 0.20, 0.60   # % Cu -- mineralised envelope / high-grade core

fig_tmp, ax_tmp = plt.subplots()
cs = ax_tmp.contourf(gx, gy, Z, levels=[ENV_THRESH, 1e9])
cs_hg = ax_tmp.contourf(gx, gy, Z, levels=[HG_THRESH, 1e9])
plt.close(fig_tmp)

def polys_from_contourset(cset, origin):
    return [Polygon([(origin[0]+x, origin[1]+y) for x, y in seg])
            for lvl in cset.allsegs for seg in lvl if len(seg) >= 3]

lg = polys_from_contourset(cs, (UTM_E0, UTM_N0))
hg = polys_from_contourset(cs_hg, (UTM_E0, UTM_N0))
domains = gpd.GeoDataFrame({"domain": ["envelope"]*len(lg) + ["high_grade"]*len(hg)},
                            geometry=lg+hg, crs="EPSG:32611")
domains["area_m2"] = domains.geometry.area
domains.to_file(OUT_DATA / "grade_domains_utm.geojson", driver="GeoJSON")

# A simple planning/infrastructure layer: haul-road centreline + a 15 m
# safety/exclusion buffer, illustrating vector overlay logic used to flag
# which part of the high-grade domain falls inside a no-extraction
# corridor (e.g. beneath planned infrastructure).
road = LineString([(UTM_E0, UTM_N0+260), (UTM_E0+260, UTM_N0+260)])
road_buffer = road.buffer(15.0)
hg_union = unary_union(hg) if hg else Polygon()
conflict_area = hg_union.intersection(road_buffer).area
print(f"Envelope >= {ENV_THRESH}% Cu: {len(lg)} polys, total area {sum(p.area for p in lg):,.0f} m^2")
print(f"High-grade >= {HG_THRESH}% Cu: {len(hg)} polys, total area {hg_union.area:,.0f} m^2")
print(f"High-grade area under road buffer: {conflict_area:.1f} m^2 "
      f"({100*conflict_area/max(hg_union.area,1):.1f}%)")

fig, ax = plt.subplots(figsize=(7, 7))
domains[domains.domain == "envelope"].plot(ax=ax, color="#c7e9c0", edgecolor="none", alpha=0.8)
domains[domains.domain == "high_grade"].plot(ax=ax, color="#fdae6b", edgecolor="none", alpha=0.9)
road_gdf = gpd.GeoDataFrame({"f": ["road"]}, geometry=[road_buffer], crs="EPSG:32611")
road_gdf.plot(ax=ax, color="none", edgecolor="crimson", hatch="//", linewidth=1.2)
samples.plot(ax=ax, markersize=6, color="black", alpha=0.5)
ax.set_title(f"Grade-domain map with haul-road exclusion corridor\n"
             f"Envelope >= {ENV_THRESH}% Cu, high-grade core >= {HG_THRESH}% Cu (illustrative UTM anchor)")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal")
legend_elems = [Patch(facecolor="#c7e9c0", label=f"Mineralised envelope (Cu >= {ENV_THRESH}%)"),
                Patch(facecolor="#fdae6b", label=f"High-grade core (Cu >= {HG_THRESH}%)"),
                Patch(facecolor="none", edgecolor="crimson", hatch="//", label="Haul-road exclusion buffer")]
ax.legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.9)
plt.tight_layout(); plt.savefig(OUT_FIGS / "fig06_gis_domain_map.png", dpi=160); plt.close()

# ------------------------------------------------------- single-bench ----
# economic parameters (illustrative; representative order-of-magnitude
# open-pit copper price/cost figures used for teaching, not site-specific)
TONNES_PER_BLOCK = 5.0*5.0*10.0*2.70
PRICE, RECOVERY = 14800.0, 0.85
MINING_COST, PROCESS_COST = 3.5, 12.0
mill_cog = 100*PROCESS_COST/(PRICE*RECOVERY)
pit_cog = 100*(MINING_COST+PROCESS_COST)/(PRICE*RECOVERY)
print(f"\nmill_cog={mill_cog:.4f}%  pit_cog={pit_cog:.4f}%")

bm = grid.rename(columns={"Cu_ok": "grade_pct"}).copy()
bm["tonnes"] = TONNES_PER_BLOCK
bm["contained_metal_t"] = bm.tonnes*bm.grade_pct/100.0

cutoffs = np.arange(0, 1.61, 0.02)
rows = []
for c in cutoffs:
    sel = bm[bm.grade_pct >= c]
    t = sel.tonnes.sum(); m = sel.contained_metal_t.sum()
    rows.append((c, t, 100*m/t if t else np.nan))
gt = pd.DataFrame(rows, columns=["cutoff_pct", "tonnes", "avg_grade_pct"])
gt.to_csv(OUT_DATA / "grade_tonnage_curve.csv", index=False)

fig, ax1 = plt.subplots(figsize=(7, 4.6))
ax1.plot(gt.cutoff_pct, gt.tonnes/1000, color="#4C78A8", lw=2)
ax1.set_xlabel("Cut-off grade (% Cu)"); ax1.set_ylabel("Tonnage above cut-off (kt)", color="#4C78A8")
ax1.tick_params(axis="y", labelcolor="#4C78A8")
ax1.axvline(mill_cog, color="grey", ls=":", lw=1); ax1.axvline(pit_cog, color="grey", ls="--", lw=1)
ax2 = ax1.twinx()
ax2.plot(gt.cutoff_pct, gt.avg_grade_pct, color="#F58518", lw=2)
ax2.set_ylabel("Average grade above cut-off (% Cu)", color="#F58518")
ax2.tick_params(axis="y", labelcolor="#F58518")
ax1.set_title("Grade-tonnage curve, single-bench in-situ resource (bench 2 / ore horizon)")
fig.tight_layout(); plt.savefig(OUT_FIGS / "fig07_grade_tonnage.png", dpi=160); plt.close()

bm["is_ore"] = bm.grade_pct >= pit_cog
bm["revenue"] = np.where(bm.is_ore, bm.contained_metal_t*RECOVERY*PRICE, 0.0)
bm["cost"] = np.where(bm.is_ore, bm.tonnes*(MINING_COST+PROCESS_COST), bm.tonnes*MINING_COST)
bm["block_value"] = bm.revenue - bm.cost
bm.to_csv(OUT_DATA / "block_model.csv", index=False)

ore = bm[bm.is_ore]
summary = {
    "cutoff_grade_pct": round(pit_cog, 4),
    "ore_tonnes": ore.tonnes.sum(),
    "waste_tonnes": bm.loc[~bm.is_ore, "tonnes"].sum(),
    "avg_grade_pct": 100*ore.contained_metal_t.sum()/max(ore.tonnes.sum(),1),
    "contained_metal_t": ore.contained_metal_t.sum(),
    "in_situ_ore_fraction": bm.is_ore.mean(),
    "undiscounted_value_no_stripping_usd": bm.block_value.sum(),
}
for k, v in summary.items():
    print(f"{k:35s}: {v:,.4f}" if isinstance(v, float) else f"{k:35s}: {v}")
pd.Series(summary).to_csv(OUT_DATA / "reserve_summary_singlebench.csv")

BV = bm.pivot(index="Y", columns="X", values="block_value").values
fig, ax = plt.subplots(figsize=(6.5, 6))
vmax = np.nanmax(np.abs(BV))
im = ax.pcolormesh(gx, gy, BV, cmap="RdYlGn", vmin=-vmax, vmax=vmax, shading="auto")
ax.contour(gx, gy, BV, levels=[0], colors="black", linewidths=1.3)
ax.set_title(f"In-situ block economic value (US$), ore-horizon bench\nBreak-even contour at {pit_cog:.3f}% Cu cut-off (no stripping cost)")
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal")
plt.colorbar(im, ax=ax, label="Block value (US$)")
plt.tight_layout(); plt.savefig(OUT_FIGS / "fig08_block_value.png", dpi=160); plt.close()

print("\nSaved fig06, fig07, fig08 and updated CSVs.")
