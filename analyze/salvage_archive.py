"""Learning curves of the archived first `main` run (3x3 with U-turns, SUMO default drivers):
price and time lost per day of every finished learner chain, for deciding how long learning
must be and what of that run can be reused.  Writes results/analysis/salvage/archive_days.csv."""
import csv, glob, os, re, sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "results", "params", "archive_uturn", "main")
OUT = os.path.join(ROOT, "results", "analysis", "salvage")
DAY = re.compile(r"^learn_s(\d+)_(day|eval)(\d+)_(prices|summary)\.csv$")


def scan(folder):
    rows = {}
    for name in os.listdir(folder):
        m = DAY.match(name)
        if not m:
            continue
        key = (int(m[1]), m[2], int(m[3]))
        row = rows.setdefault(key, {"cell": os.path.basename(folder), "seed": key[0],
                                    "phase": key[1], "day": key[2]})
        with open(os.path.join(folder, name)) as f:
            data = list(csv.DictReader(f))
        if m[4] == "summary":
            if data:
                row["time_lost"] = float(data[0]["mean_time_loss"])
                row["wait"] = float(data[0]["mean_wait_time"])
        else:
            day = [r for r in data if r.get("scope") == "day" and r.get("win_rel_q50") not in ("", None)]
            if day:
                row["price"] = 100 * sum(float(r["win_rel_q50"]) for r in day) / len(day)
                row["contested"] = sum(float(r["share_contested"]) for r in day) / len(day)
    return list(rows.values())


if __name__ == "__main__":
    import pandas as pd
    folders = sorted(f for f in glob.glob(os.path.join(ARCHIVE, "grid3_*")) if os.path.isdir(f))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        rows = [r for rs in pool.map(scan, folders) for r in rs]
    os.makedirs(OUT, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "archive_days.csv"), index=False)
    print(len(d), "chain-days from", d.cell.nunique(), "cells")
