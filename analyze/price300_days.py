"""Per-day learned price and time lost of plan "price300": learned bids (learner A) in the
reference setting for 300 learning days plus a 7-day evaluation week, 10 seeds, on the single
intersection and the 5x5 grid with 1-3 lanes (after the U-turn and driver fixes).

Writes results/analysis/price300/price300_days.csv (one row per chain-day: cell, seed, phase,
day, price, contested, time_lost, wait), which analyze/update_figures.py draws.

    python analyze/price300_days.py [workers]
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from salvage_archive import scan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "params", "price300")
OUT = os.path.join(ROOT, "results", "analysis", "price300")

if __name__ == "__main__":
    folders = sorted(f for f in glob.glob(os.path.join(RUNS, "*_c80")) if os.path.isdir(f))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        rows = [r for rs in pool.map(scan, folders) for r in rs]
    os.makedirs(OUT, exist_ok=True)
    d = pd.DataFrame(rows).sort_values(["cell", "seed", "phase", "day"])
    d.to_csv(os.path.join(OUT, "price300_days.csv"), index=False)
    print(len(d), "chain-days from", d.cell.nunique(), "cells")
