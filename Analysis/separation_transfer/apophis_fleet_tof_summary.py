#!/usr/bin/env python3
"""
apophis_fleet_tof_summary.py
============================

Purpose
-------
Scan Apophis rendezvous NPZ bundles and print, for each deputy:

- Deployment time (UTC),
- Arrival time (UTC),
- Total time of flight (TOF) from deployment to arrival.

The script is designed to work with fleet NPZ archives produced by the
Apophis rendezvous drivers (e.g. `apophis_rendezvous_full_D{k}.npz`), and it
handles ISO-8601 timestamps with or without fractional seconds.

Expected NPZ format
-------------------
Each NPZ is assumed to be a “full” rendezvous bundle such as:

    apophis_rendezvous_full_D0.npz
    apophis_rendezvous_full_D1.npz
    ...

where the file contains at least:

- `t0`   : 1-element array with the deploy time offset, in seconds, measured
           from the departure epoch stored in `meta`.
- `meta` : 2-element array of strings:
              meta[0] = departure UTC ISO string,
              meta[1] = arrival  UTC ISO string.

Typical `meta` contents look like:

    meta = [
        "2029-04-13T05:41:45Z",    # departure epoch (UTC)
        "2029-04-13T18:25:15Z",    # arrival   epoch (UTC)
    ]

where the strings may include fractional seconds (e.g. “…T05:41:45.123Z”).
The script’s parser (`parse_iso_z`) accepts both integer and fractional
seconds as long as they end with “Z”.

Computation
-----------
For each NPZ bundle:

1. Determine the deputy index:

       deputy_idx = int(npz_path.stem.split("_D")[-1])

2. Load deploy offset and epochs:

       t0_seconds = float(data["t0"][0])
       dep_iso, arr_iso = data["meta"]  # strings

3. Parse departure and arrival times:

       dt_depart = parse_iso_z(dep_iso)
       dt_arrive = parse_iso_z(arr_iso)

   where `parse_iso_z` converts "…Z" strings into RFC-3339-ish form and calls
   `datetime.fromisoformat`, so both:

       "2029-04-13T05:41:45Z"
       "2029-04-13T05:41:45.123Z"

   are handled correctly.

4. Compute deployment time:

       dt_deploy = dt_depart + timedelta(seconds=t0_seconds)

5. Compute time of flight:

       tof = dt_arrive - dt_deploy

Output format
-------------
The script prints a human-readable table to stdout:

- Header:

    # Deployment, arrival, and TOF for Apophis deputies (UTC)

- One line per deputy, sorted by deputy index:

    Deputy <idx> | deploy = <deploy_iso_Z> | arrival = <arrival_iso_Z> | TOF = <D HH:MM:SS>

Where:

- `<deploy_iso_Z>` and `<arrival_iso_Z>` are ISO-like strings with “Z” suffix
  and seconds precision (no fractional seconds):

    e.g., 2029-04-13T18:05:45Z

- TOF is formatted using `fmt_timedelta(td)`:

    * Shows days only if non-zero:
          "1d 03:15:42"
      otherwise:
          "03:15:42"

CLI usage
---------
Basic invocation:

    python apophis_fleet_tof_summary.py

By default, it scans all matching bundles:

    apophis_rendezvous_full_D*.npz

You can also specify explicit files:

    python apophis_fleet_tof_summary.py \
        -f apophis_rendezvous_full_D0.npz apophis_rendezvous_full_D1.npz

Or use globs:

    python apophis_fleet_tof_summary.py \
        -f "apophis_rendezvous_full_D*.npz"

If no files are found, the script exits with:

    No matching .npz files found – nothing to do.

Notes
-----
- The script does not modify any NPZ files; it only reads metadata and t0.
- Any NPZ that fails to parse cleanly is skipped with a warning, and the
  rest of the fleet is processed as usual.
"""


from __future__ import annotations
import argparse, glob, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np


def parse_iso_z(ts: str) -> datetime:
    """Parse an ISO-8601 string that ends in 'Z' (with optional micro-seconds)."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"            # RFC-3339 → fromisoformat friendly
    return datetime.fromisoformat(ts)


def load_times(npz_path: Path) -> tuple[int, datetime, datetime]:
    data = np.load(npz_path, allow_pickle=True)

    deputy_idx = int(npz_path.stem.split("_D")[-1])
    t0_seconds = float(data["t0"][0])

    dep_iso, arr_iso = map(str, data["meta"])
    dt_depart = parse_iso_z(dep_iso)
    dt_arrive = parse_iso_z(arr_iso)
    dt_deploy = dt_depart + timedelta(seconds=t0_seconds)

    return deputy_idx, dt_deploy, dt_arrive


def fmt_timedelta(td: timedelta) -> str:
    """Return D HH:MM:SS representation (days shown only if non-zero)."""
    total = int(td.total_seconds())
    days, rem = divmod(total, 86_400)
    h, rem = divmod(rem, 3_600)
    m, s = divmod(rem, 60)
    day_part = f"{days}d " if days else ""
    return f"{day_part}{h:02}:{m:02}:{s:02}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Print fleet timeline and TOF.")
    ap.add_argument(
        "-f", "--files",
        nargs="+",
        default=glob.glob("apophis_rendezvous_full_D*.npz"),
        help="npz bundle(s) to read (default: all apophis_rendezvous_full_D*.npz)",
    )
    args = ap.parse_args()

    if not args.files:
        sys.exit("No matching .npz files found – nothing to do.")

    rows = []
    for fp in sorted(args.files, key=lambda p: int(Path(p).stem.split("_D")[-1])):
        try:
            rows.append(load_times(Path(fp)))
        except Exception as e:
            print(f"⚠️  Skipped {fp} ({e})")

    print("# Deployment, arrival, and TOF for Apophis deputies (UTC)")
    for idx, dt_deploy, dt_arrive in rows:
        tof = dt_arrive - dt_deploy
        print(
            f"Deputy {idx} | deploy = {dt_deploy.isoformat(timespec='seconds').replace('+00:00','Z')}"
            f" | arrival = {dt_arrive.isoformat(timespec='seconds').replace('+00:00','Z')}"
            f" | TOF = {fmt_timedelta(tof)}"
        )


if __name__ == "__main__":
    main()
