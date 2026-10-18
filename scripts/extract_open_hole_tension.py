# Code under BSD-3-Clause (see LICENSE). (c) 2026 Francesco Mario Antonio Mitrotta.
"""
extract_open_hole_tension.py
----------------------------
Reads the results of one finished open-hole tension (OHT) job, built by
build_open_hole_tension.py, from its .sta, .dat and .odb files: completion, solver CPU time,
the load-displacement and energy histories, the peak, first-ply failure and the first 0-deg
fiber-tension failure (and, for full runs, the per-ply damage).

Abaqus-side (it imports odbAccess), and written for Abaqus 2023's Python 2.7. Two ways in:
  - build_open_hole_tension.py imports `extract` inside its Abaqus/CAE session, right after the
    job, so NB6's one-pass runs are unchanged;
  - a campaign that solves a ready-made input file (NB8, oht_campaign.py) runs it without CAE,
    and so without a CAE licence token, in the job's own folder:

        abaqus python extract_open_hole_tension.py job=s0042 meta=<template>.json out=s0042_extract.json

    `meta` is the JSON that build_open_hole_tension.py writes with inp_only=1: it supplies the
    gross section area and the ply angles. Light mode is implied (no damage field is read).

Never import this module in a notebook: `odbAccess` exists only inside Abaqus.
"""
import json
import math
import sys
import time

import numpy as np
from odbAccess import openOdb

CRIT_NAMES = ("HSNFTCRT", "HSNFCCRT", "HSNMTCRT", "HSNMCCRT")
DMG_NAMES = ("DAMAGEFT", "DAMAGEFC", "DAMAGEMT", "DAMAGEMC")
MODE_NAMES = {"HSNFTCRT": "fiber tension", "HSNFCCRT": "fiber compression", "HSNMTCRT": "matrix tension", "HSNMCCRT": "matrix compression"}


def ply_of(sp_number):
    return (sp_number - 1) // 3 + 1  # 1-based ply; 3 section points per ply


# ---------------------------------------------------------------------------------
# Event extraction. A Hashin criterion output stays at exactly 1.0 from the first frame after
# its mode initiates, so the frame after a crossing says only "it happened somewhere since the
# previous frame". Instead, extrapolate forward from frame(s) strictly BEFORE the crossing:
#   - quadratic=True (first-ply failure): before any damage the model is linear elastic, so
#     every stress scales with U2 and every (quadratic) Hashin criterion with U2^2. One frame
#     then gives the crossing exactly: U2* = U2_prev / sqrt(F_prev), RF2* = RF2_prev * U2*/U2_prev.
#   - quadratic=False (first 0-deg fiber-tension failure, after matrix cracking has made the
#     response nonlinear): straight line through sqrt(F) at the two frames before the
#     crossing, clamped inside the bracketing increment; RF2 interpolated linearly in U2.
# ---------------------------------------------------------------------------------
def extrapolate_event(u2_list, rf2_list, crit_history, quadratic, gross_area, threshold=1.0):
    for i in range(1, len(crit_history)):
        if crit_history[i - 1] < threshold <= crit_history[i]:
            if quadratic:
                c0 = crit_history[i - 1]
                if c0 <= 0.0:
                    return None  # crossed within the first increment: no elastic frame to scale from
                u2_star = u2_list[i - 1] / math.sqrt(c0)
                rf2_star = rf2_list[i - 1] * (u2_star / u2_list[i - 1])
            elif i >= 2 and crit_history[i - 2] < crit_history[i - 1]:
                s0, s1 = math.sqrt(crit_history[i - 2]), math.sqrt(crit_history[i - 1])
                u0, u1 = u2_list[i - 2], u2_list[i - 1]
                slope = (s1 - s0) / (u1 - u0)
                u2_star = min(max(u1 + (1.0 - s1) / slope, u1), u2_list[i])
                frac = (u2_star - u1) / (u2_list[i] - u1) if u2_list[i] != u1 else 0.0
                rf2_star = rf2_list[i - 1] + frac * (rf2_list[i] - rf2_list[i - 1])
            else:
                u2_star, rf2_star = u2_list[i - 1], rf2_list[i - 1]
            return {"u2_mm": u2_star, "rf2_n": rf2_star, "sigma_mpa": rf2_star / gross_area, "frame_before": i - 1}
    return None


def extract(job_name, light, gross_area, physical_angles, element_centroids=None, damage_csv=None):
    """Read job `job_name` (in the current folder) and return the results dict NB6 and NB8 read.

    gross_area [mm^2] turns loads into gross-section stresses; physical_angles (top ply first)
    says which plies are 0 deg. The damage CSV is written to `damage_csv` only for a full run
    (light=False), which also needs `element_centroids` {element label: (x, y)}.
    """
    sta_tail = ""
    try:
        with open(job_name + ".sta") as f:
            sta_tail = f.read()[-400:]
    except IOError:
        pass
    completed = "COMPLETED SUCCESSFULLY" in sta_tail

    # Solver cost: the "TOTAL CPU TIME" line of the .dat file's job time summary. CPU time, unlike
    # wall-clock time, is barely affected by other jobs sharing the machine.
    cpu_seconds = None
    try:
        with open(job_name + ".dat") as f:
            for line in f:
                if "TOTAL CPU TIME" in line:
                    cpu_seconds = float(line.split("=")[1])
    except IOError:
        pass

    odb = None
    for _attempt in range(6):  # openOdb can race the solver's final flush by a few seconds
        try:
            odb = openOdb(job_name + ".odb")
            break
        except Exception:
            if _attempt == 5:
                raise
            time.sleep(5)
    step = odb.steps["Tension"]

    # History output is recorded PER NODE on a multi-node set: U2 is identical at every Top-Edge
    # node, RF2 must be summed across all of them.
    top_edge_regions = [region for key, region in step.historyRegions.items() if key.startswith("Node ")]
    u2_history = top_edge_regions[0].historyOutputs["U2"].data
    rf2_per_node = [region.historyOutputs["RF2"].data for region in top_edge_regions]
    rf2_totals = []
    for i in range(len(u2_history)):
        total = 0.0
        for node_data in rf2_per_node:
            total += node_data[i][1]
        rf2_totals.append(total)
    u2 = [p[1] for p in u2_history]

    energy_regions = [region for key, region in step.historyRegions.items() if key.startswith("Assembly ")]
    allcd = [p[1] for p in energy_regions[0].historyOutputs["ALLCD"].data]
    alldmd = [p[1] for p in energy_regions[0].historyOutputs["ALLDMD"].data]

    peak_index = max(range(len(rf2_totals)), key=lambda i: rf2_totals[i])

    # Per-frame quantities: the model-wide maximum of each Hashin criterion, the maximum fiber-
    # tension criterion in the 0-deg plies, and, per ply, the maximum of each damage variable.
    n_plies = len(physical_angles)
    zero_deg_plies = set(i + 1 for i, a in enumerate(physical_angles) if a == 0.0)  # 1-based

    # bulkDataBlocks hands back numpy arrays, one block per section point -- orders of magnitude
    # faster than looping over FieldValue objects, which matters for runs with thousands of frames.
    crit_max_history = dict((name, []) for name in CRIT_NAMES)
    crit_ft_0deg_history = []
    ply_max_damage = dict((name, []) for name in DMG_NAMES)  # [frame][ply]
    frames = step.frames
    for frame in frames:
        for crit_name in CRIT_NAMES:
            m = 0.0
            m0 = 0.0
            for block in frame.fieldOutputs[crit_name].bulkDataBlocks:
                bmax = float(np.max(block.data))
                m = max(m, bmax)
                if crit_name == "HSNFTCRT" and ply_of(block.sectionPoint.number) in zero_deg_plies:
                    m0 = max(m0, bmax)
            crit_max_history[crit_name].append(m)
            if crit_name == "HSNFTCRT":
                crit_ft_0deg_history.append(m0)
        if light:
            continue  # the damage variables were not requested
        for dmg_name in DMG_NAMES:
            dmax = [0.0] * n_plies
            for block in frame.fieldOutputs[dmg_name].bulkDataBlocks:
                p = ply_of(block.sectionPoint.number) - 1
                dmax[p] = max(dmax[p], float(np.max(block.data)))
            ply_max_damage[dmg_name].append(dmax)

    overall_crit_history = [max(crit_max_history[name][i] for name in CRIT_NAMES) for i in range(len(frames))]
    fpf = extrapolate_event(u2, rf2_totals, overall_crit_history, True, gross_area)
    onset0 = extrapolate_event(u2, rf2_totals, crit_ft_0deg_history, False, gross_area)

    if fpf is not None:  # which mode and ply drove first-ply failure
        frame = frames[fpf["frame_before"] + 1]
        for crit_name in CRIT_NAMES:
            for fv in frame.fieldOutputs[crit_name].values:
                if fv.data >= 1.0:
                    fpf["mode"] = MODE_NAMES[crit_name]
                    fpf["ply"] = ply_of(fv.sectionPoint.number)
                    break
            if "mode" in fpf:
                break

    # Companion damage CSV at tagged frames: the first frame after first-ply failure, the first
    # frame after the first 0-deg fiber-tension failure, and the peak-load frame.
    tagged_frames = []
    if fpf is not None:
        tagged_frames.append(("fpf", fpf["frame_before"] + 1))
    if onset0 is not None:
        tagged_frames.append(("onset0", onset0["frame_before"] + 1))
    tagged_frames.append(("peak", peak_index))

    if not light:  # a campaign has no damage field to tabulate
        with open(damage_csv, "w") as f:
            f.write("frame_tag,element_label,x,y,ply,damage_ft,damage_fc,damage_mt,damage_mc\n")
            for tag, frame_index in tagged_frames:
                frame = frames[frame_index]
                values_by_dmg = {}
                for dmg_name in DMG_NAMES:
                    d = {}
                    for fv in frame.fieldOutputs[dmg_name].values:
                        d[(fv.elementLabel, fv.sectionPoint.number)] = fv.data
                    values_by_dmg[dmg_name] = d
                for key in values_by_dmg["DAMAGEFT"]:
                    ft, fc, mt, mc = [values_by_dmg[n].get(key, 0.0) for n in DMG_NAMES]
                    if ft == 0.0 and fc == 0.0 and mt == 0.0 and mc == 0.0:
                        continue  # undamaged rows are omitted to keep the file small
                    cx, cy = element_centroids[key[0]]
                    f.write("%s,%d,%.4f,%.4f,%d,%.6f,%.6f,%.6f,%.6f\n" % (tag, key[0], cx, cy, ply_of(key[1]), ft, fc, mt, mc))

    results = {
        "completed": completed,
        "n_increments": len(frames) - 1,
        "cpu_seconds": cpu_seconds,
        "u2": u2,
        "rf2": rf2_totals,
        "allcd": allcd, "alldmd": alldmd,
        "gross_area_mm2": gross_area,
        "peak_index": peak_index,
        "f_peak_n": rf2_totals[peak_index],
        "u2_peak_mm": u2[peak_index],
        "sigma_peak_mpa": rf2_totals[peak_index] / gross_area,
        "fpf": fpf,  # {u2_mm, rf2_n, sigma_mpa, frame_before, mode, ply}, or None
        "onset0": onset0,  # first 0-deg fiber-tension failure, same shape, or None
    }
    if not light:  # the per-frame histories are what makes a full JSON large
        results.update({
            "crit_max_history": crit_max_history,
            "hsnftcrt_0deg_history": crit_ft_0deg_history,
            "ply_max_damage": ply_max_damage,  # {DAMAGExx: [frame][ply]}
            "tagged_frames": dict(tagged_frames),
        })
    odb.close()
    return results


if __name__ == "__main__":
    # key=value tokens, in any order (also as "-key value", the shape Abaqus's launchers can produce).
    args, argv, i = {}, sys.argv[1:], 0
    while i < len(argv):
        if "=" in argv[i]:
            key, _, value = argv[i].partition("=")
            args[key.lstrip("-").lower()] = value
            i += 1
        elif argv[i].startswith("-") and i + 1 < len(argv):
            args[argv[i][1:].lower()] = argv[i + 1]
            i += 2
        else:
            i += 1
    with open(args["meta"]) as f:
        meta = json.load(f)
    out = extract(args["job"], True, meta["gross_area_mm2"], meta["physical_angles"])
    with open(args["out"], "w") as f:
        json.dump(out, f)
