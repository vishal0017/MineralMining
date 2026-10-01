"""
06_price_sensitivity.py

Re-run the floating-cone pit optimisation of 05_multibench_floating_cone.py
at Cu prices +/-20% to get a rigorous (3-D, not just single-bench)
sensitivity table.

Requires: 05_multibench_floating_cone.py has already been run (reads
outputs/data/grade_3d.npy).
"""
import numpy as np
import pandas as pd

from _paths import OUT_DATA

grade = np.load(OUT_DATA / "grade_3d.npy")
N_BENCH, nY, nX = grade.shape
TONNES_PER_BLOCK = 5.0*5.0*10.0*2.70
RECOVERY = 0.85
MINING_COST, PROCESS_COST = 3.5, 12.0
CELL, BENCH_HEIGHT, SLOPE_DEG = 5.0, 10.0, 45.0
tan_slope = np.tan(np.radians(SLOPE_DEG))

def disk_offsets(radius_cells):
    r = int(np.ceil(radius_cells))
    if r == 0:
        return np.array([[0, 0]])
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    mask = (yy**2 + xx**2) <= radius_cells**2 + 1e-9
    return np.column_stack([yy[mask], xx[mask]])

cone_offsets = {}
for diff in range(1, N_BENCH):
    h = diff*BENCH_HEIGHT
    cone_offsets[diff] = disk_offsets((h/tan_slope)/CELL)

def run_floating_cone(price):
    pit_cog = 100*(MINING_COST+PROCESS_COST)/(price*RECOVERY)
    tonnes = np.full_like(grade, TONNES_PER_BLOCK)
    metal_t = tonnes*grade/100.0
    is_ore = grade >= pit_cog
    revenue = np.where(is_ore, metal_t*RECOVERY*price, 0.0)
    cost = np.where(is_ore, tonnes*(MINING_COST+PROCESS_COST), tonnes*MINING_COST)
    block_value = revenue - cost
    removed = np.zeros_like(is_ore, dtype=bool)
    cands = [(block_value[b,r,c], b, r, c) for b in range(N_BENCH) for r in range(nY) for c in range(nX)
              if is_ore[b,r,c] and block_value[b,r,c] > 0]
    cands.sort(key=lambda t: -t[0])
    for val, b, r, c in cands:
        if removed[b, r, c]:
            continue
        cells = [(b, r, c)]; cone_val = block_value[b, r, c]
        for diff in range(1, b+1):
            s = b - diff
            offs = cone_offsets[diff]
            rr = r + offs[:,0]; cc = c + offs[:,1]
            valid = (rr>=0)&(rr<nY)&(cc>=0)&(cc<nX)
            rr, cc = rr[valid], cc[valid]
            for ri, ci in zip(rr, cc):
                if not removed[s, ri, ci]:
                    cone_val += block_value[s, ri, ci]
                    cells.append((s, ri, ci))
        if cone_val > 0:
            for bb, rr2, cc2 in cells:
                removed[bb, rr2, cc2] = True
    pit_ore_t = tonnes[removed & is_ore].sum()
    pit_waste_t = tonnes[removed & ~is_ore].sum()
    pit_metal = metal_t[removed & is_ore].sum()
    net_val = revenue[removed].sum() - cost[removed].sum()
    return dict(price=price, pit_cog=pit_cog, ore_t=pit_ore_t, waste_t=pit_waste_t,
                strip=pit_waste_t/max(pit_ore_t,1), grade=100*pit_metal/max(pit_ore_t,1),
                metal_t=pit_metal, value=net_val)

rows = []
for mult, label in [(0.8, "-20%"), (1.0, "base"), (1.2, "+20%")]:
    price = 14800.0*mult
    res = run_floating_cone(price)
    res["label"] = label
    rows.append(res)
    print(f"{label:5s} price=${price:,.0f}/t  pit_cog={res['pit_cog']:.3f}%  ore_t={res['ore_t']:,.0f}  "
          f"strip={res['strip']:.3f}  grade={res['grade']:.3f}%  metal={res['metal_t']:,.1f}t  "
          f"value=${res['value']:,.0f}")

pd.DataFrame(rows).to_csv(OUT_DATA / "floatingcone_sensitivity.csv", index=False)
print(f"\nSaved {OUT_DATA / 'floatingcone_sensitivity.csv'}")
