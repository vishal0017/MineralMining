"""
run_all.py

Runs the full pipeline end to end, in the correct dependency order, from
the repository root:

    python run_all.py

Equivalent to running each script in scripts/ by hand in numeric order.
Each step prints its own console output; figures land in outputs/figs/ and
processed data/tables in outputs/data/.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))  # so each script's `from _paths import ...` resolves

PIPELINE = [
    "01_load_and_qc.py",
    "02_rescale_variography_kriging.py",
    "03_implicit_surface_modelling.py",
    "04_gis_mapping_and_cutoff.py",
    "05_multibench_floating_cone.py",
    "06_price_sensitivity.py",
]

if __name__ == "__main__":
    for script in PIPELINE:
        path = SCRIPTS_DIR / script
        print(f"\n{'='*70}\nRunning {script}\n{'='*70}")
        runpy.run_path(str(path), run_name="__main__")
    print("\nPipeline complete. See outputs/figs/ and outputs/data/.")
