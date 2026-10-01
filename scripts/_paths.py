"""
Shared path helpers so every script in scripts/ resolves data/outputs
relative to the repository root, regardless of the current working
directory the script is launched from.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DATA = ROOT / "outputs" / "data"
OUT_FIGS = ROOT / "outputs" / "figs"

OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)
