"""
mc_campaign.py
--------------
NB2 Part 4's own campaign machinery -- run_one_sample, load_campaign, run_campaign -- unchanged
apart from one addition every notebook after NB2 needs: an `nlgeom` argument, since NB2's own
inline version only ever ran nlgeom=OFF. Imported here rather than redefined, exactly like
`kde.py`'s KDE functions: nothing below is new to any notebook after NB2, which already defined
and explained every one of these pieces in full (NB2 Part 4, "Launching one Abaqus run" through
"Running (and resuming) the whole campaign").

Because it never does `from abaqus import *`, this module is safe to import directly into a
live Jupyter kernel (unlike `build_fiber_tension.py`, which must only ever be invoked via
`subprocess`).
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from peak_reconstruction import reconstruct_peak

SCRIPTS_DIR = Path(__file__).parent  # where build_fiber_tension.py itself lives

# One column per quantity a completed sample needs: the sampled input, both definitions of the
# peak force (raw vs. reconstructed), the HSNFTCRT budget check, and the run's own wall-clock
# time. Matches the header already written to runs/nb02/summary.csv and runs/nb03/*.csv, so a
# csv.DictWriter appending to either file lines up with what is already on disk.
FIELDNAMES = ["sample_index", "x_t_mpa", "f_peak_raw_n", "u1_peak_raw_mm",
              "f_peak_recon_n", "u1_peak_recon_mm", "hsnftcrt_max", "hsnftcrt_ok", "run_seconds"]


def run_one_sample(x_t, run_json, target_u1, nlgeom="OFF"):
    """Run build_fiber_tension.py once for the given X_T (and nlgeom setting), and save its full
    recorded force-displacement/damage history straight to run_json. build_fiber_tension.py's
    own `out=` parameter creates run_json's parent folder automatically if it doesn't exist yet,
    so there is no separate directory setup here.

    If run_json already exists, the Abaqus call is skipped entirely and the saved history is
    read back instead -- this single file-existence check is what makes ONE run resumable.

    Returns (data, elapsed_seconds); elapsed_seconds is 0.0 for a skipped (already-run) sample.
    """
    run_json = Path(run_json)
    if run_json.exists():
        with open(run_json) as f:
            return json.load(f), 0.0

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "build_fiber_tension.py", f"nlgeom={nlgeom}", f"X_T={x_t:.6f}",
         f"U1={target_u1}", f"out={run_json.as_posix()}"],
        cwd=str(SCRIPTS_DIR),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    # result.returncode is NOT a reliable success signal on this Abaqus install: abqpy's `from abaqus import *` unconditionally calls sys.exit(0) once
    # it hands the script to real Abaqus, regardless of whether that Abaqus run itself
    # succeeded. The only trustworthy signal is whether the declared output file actually
    # appeared.
    if not run_json.exists():
        raise RuntimeError(
            f"X_T={x_t:.2f} MPa (nlgeom={nlgeom}) did not produce results -- stdout:\n"
            f"{result.stdout}\nstderr:\n{result.stderr}"
        )

    with open(run_json) as f:
        data = json.load(f)
    return data, elapsed


def load_campaign(summary_csv, n_max=None):
    """Read a campaign's summary CSV back into plain numpy arrays, sorted by sample index.

    `n_max`, if given, truncates to the first `n_max` samples (by sample_index) -- this is what
    lets a notebook report on exactly some earlier prefix of a campaign even after later batches
    have appended more rows to the same CSV.

    A campaign with no file yet (nothing run so far) reads back as empty arrays rather than
    raising, so run_campaign can call this before its own first sample has ever run.
    """
    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        return {"x_t": np.array([]), "f_peak_recon": np.array([]), "u1_peak_recon": np.array([]),
                "hsn_ok": np.array([], dtype=bool), "run_seconds": np.array([])}
    with open(summary_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["sample_index"]))  # rows are appended in order already, but
                                                       # sort explicitly rather than assume it
    if n_max is not None:
        rows = [r for r in rows if int(r["sample_index"]) < n_max]
    return {
        "x_t": np.array([float(r["x_t_mpa"]) for r in rows]),
        "f_peak_recon": np.array([float(r["f_peak_recon_n"]) for r in rows]),
        "u1_peak_recon": np.array([float(r["u1_peak_recon_mm"]) for r in rows]),
        "hsn_ok": np.array([r["hsnftcrt_ok"] == "True" for r in rows]),  # csv stores everything as text
        "run_seconds": np.array([float(r["run_seconds"]) for r in rows]),
    }


def run_campaign(x_t_campaign, runs_dir, summary_csv, target_u1, nlgeom="OFF", verbose=True):
    """Run every sample in x_t_campaign, in order, appending one summary row per sample to
    summary_csv. Already-recorded samples are skipped: because samples are always appended in
    order, "already run" simply means "the first n_done rows of summary_csv already match the
    first n_done values of x_t_campaign" -- checked explicitly before trusting it, so a changed
    seed can never silently mix two different campaigns' results.

    `verbose=False` silences the progress/summary printing, for a caller (NB3's batching driver)
    that calls this once per batch and reports its own progress instead.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(exist_ok=True, parents=True)
    summary_csv = Path(summary_csv)

    existing = load_campaign(summary_csv)
    n_done = len(existing["x_t"])
    # Compare only the OVERLAP between what's already on disk and what this call asked for: a
    # caller that hands x_t_campaign a shorter, still-growing prefix each time (an open-ended
    # batching loop, see NB3) can have n_done > len(x_t_campaign) -- that's not a mismatch, it
    # just means this batch's own samples are already all recorded, and the loop below (which
    # only ever runs range(n_done, len(x_t_campaign))) correctly does nothing in that case.
    n_check = min(n_done, len(x_t_campaign))
    if n_check:
        assert np.allclose(existing["x_t"][:n_check], x_t_campaign[:n_check]), (
            f"{summary_csv} already holds a different X_T draw than the one requested -- "
            "delete it (and its run_*.json files) to start a fresh campaign rather than mixing two."
        )

    write_header = not summary_csv.exists()
    t_segment_start = time.perf_counter()
    n_new = 0
    with open(summary_csv, "a", newline="") as f:  # "a": append, so already-written rows are untouched
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for i in range(n_done, len(x_t_campaign)):  # only the samples NOT already recorded
            x_t = x_t_campaign[i]
            run_json = runs_dir / f"run_{i:03d}.json"
            data, elapsed = run_one_sample(x_t, run_json, target_u1, nlgeom=nlgeom)

            u1 = np.array(data["u1"])
            rf1 = np.array(data["rf1"])
            hsn_ft = np.array(data["hsn_ft"])
            f_peak_raw = float(rf1.max())
            u1_peak_raw = float(u1[int(rf1.argmax())])
            u1_peak, f_peak = reconstruct_peak(u1, rf1, hsn_ft)
            hsn_max = float(np.max(hsn_ft))
            hsn_ok = hsn_max >= 0.99  # confirms the displacement budget reached damage initiation
            writer.writerow({
                "sample_index": i, "x_t_mpa": x_t,
                "f_peak_raw_n": f_peak_raw, "u1_peak_raw_mm": u1_peak_raw,
                "f_peak_recon_n": f_peak, "u1_peak_recon_mm": u1_peak,
                "hsnftcrt_max": hsn_max, "hsnftcrt_ok": hsn_ok, "run_seconds": elapsed,
            })
            n_new += 1

            if not hsn_ok:
                print(f"[nlgeom={nlgeom}][{i}] WARNING: HSNFTCRT_max={hsn_max:.3f} did not reach ~1.0 - investigate this sample.")
            if verbose and (i % 40 == 0 or i == len(x_t_campaign) - 1):
                print(f"[nlgeom={nlgeom}][{i + 1}/{len(x_t_campaign)}] X_T={x_t:7.2f} MPa -> "
                      f"F_peak_recon={f_peak:8.2f} N  ({elapsed:.1f}s)")

    if verbose:
        if n_new:
            print(f"\nRan {n_new} new sample(s) this session in {time.perf_counter() - t_segment_start:.0f}s.")
        else:
            print(f"\nAll {len(x_t_campaign)} samples already completed - read back from disk, no new Abaqus jobs.")
