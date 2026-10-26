# Code under BSD-3-Clause (see LICENSE). (c) 2026 Francesco Mario Antonio Mitrotta.
"""
oht_campaign.py
---------------
Runs rows of a design CSV through the M1 open-hole tension model, several Abaqus jobs at a time,
and saves one result JSON per sample. NB8's campaign driver.

Plain Python 3 and never Abaqus-side (it does not import abaqus), so it runs unchanged on
Windows and Linux, and may be imported or run from a shell. With no options it runs the whole
training design, runs/nb08/design_train.csv, into runs/nb08/train:

    python scripts/oht_campaign.py
    python scripts/oht_campaign.py --rows 0-31 --workers 5        # a part of it, 5 jobs at once

How a sample runs. Abaqus/CAE builds the model only once, writing a template input file
(runs/nb08/template/M1.inp, made by build_open_hole_tension.py with inp_only=1). Every sample
copies it with its own 15 material values on the three material data lines, solves it with
`abaqus job=` and reads the results with `abaqus python extract_open_hole_tension.py`, in its own
working folder, so concurrent jobs never share a file. No CAE session runs per job, so no CAE
licence token (pools QAE/QAX, 34 tokens university-wide) is held for the length of a solve.

How many at once. --workers caps the concurrent jobs (default: logical processors - 1, or fewer
if memory is short). Before each launch the driver also asks the licence server
(`abaqus licensing dslsstat`) and waits until the solver pools QSD and QXT both have a job's
tokens free, on top of an optional reserve left for other users (--reserve 0.1 keeps 10% of each
pool free; none by default).

Design CSV: one header line with the 15 input names below, one row per sample, physical values
(MPa, N/mm, ...). Row numbers count from 0 and name the samples: row 7 is sample s0007.

Each sample ends as <out-dir>/sNNNN.json: the build script's JSON keys (light mode) plus the
driver's measurements, and an `accepted` flag with the `reason` when it is false. A sample whose
JSON already exists is skipped, so an interrupted campaign resumes where it stopped. Rejected and
failed samples keep their JSON and are never retried automatically, except a sample whose job
got no licence: it gets no JSON, so running the same command again retries it.
<out-dir>/status.csv has one line per sample processed.

    python scripts/oht_campaign.py --selftest      # checks the pure functions; no Abaqus needed
"""
import argparse
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

SCRIPTS_DIR = Path(__file__).resolve().parent  # where the build and extraction scripts live
RUNS_DIR = SCRIPTS_DIR.parent / "runs" / "nb08"
TEMPLATE_DIR = RUNS_DIR / "template"

# The 15 uncertain inputs, in design-CSV column order (NB6's names, as the build script reads them).
INPUTS = ["E1", "E2", "nu12", "G12", "G23", "X_T", "X_C", "Y_T", "Y_C", "S_L", "S_T",
          "G_fT", "G_fC", "G_mT", "G_mC"]
# The three material data lines of the input file: keyword line -> the inputs on the line after it, in
# Abaqus's order. G13 = G12 in every sample: one variable (NB8).
MATERIAL_LINES = {"*Elastic, type=LAMINA": ("E1", "E2", "nu12", "G12", "G13", "G23"),
                  "*Damage Initiation, criterion=HASHIN": ("X_T", "X_C", "Y_T", "Y_C", "S_L", "S_T"),
                  "*Damage Evolution, type=ENERGY": ("G_fT", "G_fC", "G_mT", "G_mC")}
MESHES = {"M1": (20, 15)}  # (ncirc, nrad): NB6's coarsest mesh, 1200 elements; the only one NB8 uses
# Solver tokens of one job, from each of QSD and QXT, by cores per job (NB8's job logs).
TOKENS_PER_JOB = {1: 50, 2: 59, 4: 76, 6: 94}
PEAK_RAM_GB = 0.65  # largest process-tree memory of one M1 job (NB8, with CAE in the tree)
# A job checks out its tokens about 6-9 s after launch (NB8's job logs), so the licence server does not see it
# at once: a job launched less than this many seconds ago counts as already holding its tokens.
CHECKOUT_S = 30.0
STATUS_FIELDS = ["sample", "accepted", "reason", "wall_s", "cpu_s", "peak_ram_mb", "tokens"]


def parse_rows(text):
    """'a-b' (both ends included) or 'a' -> list of row numbers."""
    # 'a-b' splits into a and b; a single 'a' leaves b empty, so the range is a to a.
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def acceptance_reason(r):
    """Return '' if a sample's result is usable, else why it is rejected.

    Three tests (NB8, Part 4): first-ply failure was found; the load falls by at least 20% after the
    peak (so the peak is not just the last recorded frame); and the viscous energy at the peak is
    below 1% of the damage energy (NB6's check on the viscosity). A job that stopped before the
    target displacement passes if it stopped after such a drop: the outputs end at the peak.
    """
    # Test 1: the extraction found the first point where a Hashin criterion reaches 1.
    if r.get("fpf") is None:
        return "no first-ply failure found"
    # Test 2: the peak has later frames, and the lowest load among them is below 80% of the peak.
    rf2, peak = r.get("rf2") or [], r.get("peak_index")
    if peak is None or peak >= len(rf2) - 1 or min(rf2[peak:]) >= 0.8 * rf2[peak]:
        return "no clear load drop after the peak"
    # Test 3: viscous energy ALLCD below 1% of the damage energy ALLDMD at the peak frame.
    if r["alldmd"][peak] <= 0.0 or r["allcd"][peak] / r["alldmd"][peak] >= 0.01:
        return "ALLCD/ALLDMD at the peak is not below 1%"
    return ""


def parse_log(text):
    """(Abaqus release, tokens checked out) from a job's .log text; None where absent."""
    # The release is a line of its own, 'Abaqus 2023'; the token count is in the licence manager's line
    # 'Abaqus/Standard checked out 50 tokens from DSLS server ...'. No such line: the job got no licence.
    release = re.search(r"^Abaqus (\d{4}\S*)\s*$", text, re.M)
    tokens = re.search(r"checked out (\d+) tokens", text)
    return (release.group(1) if release else None, int(tokens.group(1)) if tokens else None)


def parse_dslsstat(text, features=("QSD", "QXT")):
    """{feature: (free tokens, pool size)} from `abaqus licensing dslsstat`, summed over the feature's pools.

    Its licence table has rows '| QSD | 28 | <release> | Token | STD | 2000 | 150 | ...': field 4 is the
    licence model, 6 the pool size, 7 the tokens in use.
    """
    out = {}
    for line in text.splitlines():
        # The fields of a table row; the leading '|' makes field 0 empty, so the feature name is field 1.
        f = [c.strip() for c in line.split("|")]
        # Only token rows of the wanted features count; a feature with two pools adds both.
        if len(f) > 7 and f[1] in features and f[4] == "Token":
            free, pool = out.get(f[1], (0, 0))
            out[f[1]] = (free + int(f[6]) - int(f[7]), pool + int(f[6]))
    return out


def room_for_job(pools, tokens, reserve, pending):
    """True if every pool ({feature: (free, size)}) has `tokens` free beyond the reserve (a share of the pool)
    and the tokens of `pending` jobs launched but not yet seen by the licence server."""
    # Free tokens, less the reserve and the pending jobs' tokens, must still hold one more job, in every pool.
    return all(free - reserve * size - pending * tokens >= tokens for free, size in pools.values())


def patch_inp(text, values):
    """The template input file with one sample's material values (dict by input name, G13 included)."""
    # Work line by line; the keyword match ignores case and leading spaces, as Abaqus does.
    lines = text.split("\n")
    for keyword, names in MATERIAL_LINES.items():
        # Each keyword line must occur once; its data line, the next line, is replaced whole.
        hits = [i for i, line in enumerate(lines) if line.strip().lower().startswith(keyword.lower())]
        assert len(hits) == 1, "%d lines start with %s" % (len(hits), keyword)
        # repr() writes every float in full precision (CAE's own writeInput rounds to 6 significant digits).
        lines[hits[0] + 1] = ", ".join(repr(values[n]) for n in names)
    return "\n".join(lines)


def ceiling_share(lc_values, e2, g_mt, y_t):
    """Share of elements above the sample's own matrix-tension ceiling Lc < 2 E2 G_mT / Y_T^2 (NB6)."""
    # The ceiling moves with the sample's E2, G_mT and Y_T; the element sizes Lc are fixed by the mesh.
    ceiling = 2.0 * g_mt * e2 / y_t ** 2
    return sum(lc > ceiling for lc in lc_values) / len(lc_values)


def run_monitored(cmd, cwd, env, stdout_path):
    """Run cmd, polling its whole process tree once a second.

    Returns (returncode, peak_ram_mb, cpu_seconds). Peak RAM is the largest *simultaneous* sum of
    resident memory over the tree, the same call on Windows and Linux. CPU time is the sum of the
    last CPU time seen for every process in the tree (a process shorter than the 1 s poll can be
    missed in part, so the solver's own `.dat` CPU time stays the reference for the solve).
    """
    with open(stdout_path, "w") as out:
        # Start the command with its output in a file, and follow its process from outside.
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT)
        root = psutil.Process(proc.pid)
        # Peak memory so far [bytes], and the latest CPU time of every process seen, by process id [s].
        peak, cpu = 0, {}
        while proc.poll() is None:
            # One poll: list the tree (the launcher, pre, standard, ...) and add up its memory now.
            rss = 0
            try:
                tree = [root] + root.children(recursive=True)
            except psutil.Error:  # the root ended between poll() and the listing: the job is over
                break
            for p in tree:
                try:
                    # Resident memory adds up over the tree; CPU time (user + system) is kept per process,
                    # overwritten at each poll, since it only grows.
                    rss += p.memory_info().rss
                    t = p.cpu_times()
                    cpu[p.pid] = t.user + t.system
                except psutil.Error:  # the process ended between listing and reading
                    pass
            # Keep the largest total so far, and wait for the next poll.
            peak = max(peak, rss)
            time.sleep(1.0)
        returncode = proc.wait()  # reap the root (a no-op if poll() already did)
    return returncode, peak / 2**20, sum(cpu.values())


class Launcher:
    """Spaces job launches by `stagger` seconds, so that dozens of jobs do not start (and ask the
    licence server for tokens) in the same second, and holds each launch until the solver pools have
    a job's tokens free on top of the reserve and of the jobs it launched in the last CHECKOUT_S seconds.
    ponytail: one global lock, fine for tens of workers; per-worker offsets if it ever matters."""

    def __init__(self, stagger, abaqus, tokens, reserve):
        # Seconds between launches, the time of the last launch, and the lock that lets one worker launch at a time.
        self.stagger, self.last, self.lock = stagger, 0.0, threading.Lock()
        # The abaqus command (for dslsstat), the tokens one job takes from each pool, and the reserve share.
        self.abaqus, self.tokens, self.reserve = abaqus, tokens, reserve
        self.launches = []  # launch times of the last CHECKOUT_S seconds

    def tokens_free(self):
        """True if QSD and QXT both have room for one more job (room_for_job); True also if the
        server cannot be read (the job then finds out itself, and a job without a licence is retried)."""
        # Ask the licence server for its table (about 0.5 s on the server, 1.5 s on the desktop).
        try:
            text = subprocess.run([self.abaqus, "licensing", "dslsstat"], capture_output=True, text=True,
                                  timeout=120).stdout
        except (OSError, subprocess.SubprocessError) as error:
            print("licence server not read (%r): launching anyway" % error, flush=True)
            return True
        pools = parse_dslsstat(text)
        if len(pools) < 2:
            print("licence server listing has no QSD/QXT rows: launching anyway", flush=True)
            return True
        # This driver's jobs launched too recently to show in the table yet.
        pending = sum(time.time() - t < CHECKOUT_S for t in self.launches)
        return room_for_job(pools, self.tokens, self.reserve, pending)

    def wait(self):
        """Block the calling worker until its job may be launched."""
        # One worker at a time: the others queue on the lock, so launches never overlap.
        with self.lock:
            # At least `stagger` seconds after the previous launch.
            time.sleep(max(0.0, self.last + self.stagger - time.time()))
            # Then wait for tokens, asking again once a minute.
            while not self.tokens_free():
                print("%s waiting for tokens" % time.strftime("%H:%M:%S"), flush=True)
                time.sleep(60.0)
            # Record the launch, and forget launches older than CHECKOUT_S (they now show in the table).
            self.last = time.time()
            self.launches = [t for t in self.launches if self.last - t < CHECKOUT_S] + [self.last]


def ensure_template(args, env):
    """Path of the template's JSON, building the template with Abaqus/CAE first if it is missing (run-or-load)."""
    # Load: the input file and its JSON (mesh statistics, element sizes Lc) are both there.
    meta_path = TEMPLATE_DIR / (args.mesh + ".json")
    if (TEMPLATE_DIR / (args.mesh + ".inp")).exists() and meta_path.exists():
        return meta_path
    # Run: the build script, in Abaqus/CAE through abqpy, writes the input file without solving (inp_only=1).
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    ncirc, nrad = MESHES[args.mesh]
    print("building the template %s.inp with Abaqus/CAE ..." % args.mesh, flush=True)
    with open(TEMPLATE_DIR / "driver_stdout.txt", "w") as out:
        subprocess.run([sys.executable, str(SCRIPTS_DIR / "build_open_hole_tension.py"), "ncirc=%d" % ncirc,
                        "nrad=%d" % nrad, "light=1", "inp_only=1", "out=" + str(meta_path)],
                       cwd=str(TEMPLATE_DIR), env=env, stdout=out, stderr=subprocess.STDOUT)
    if not (TEMPLATE_DIR / (args.mesh + ".inp")).exists():  # success = the file exists; returncode is unreliable
        sys.exit("no template written: see %s" % (TEMPLATE_DIR / "driver_stdout.txt"))
    return meta_path


def run_sample(row_no, row, args, launcher, keep, meta, template):
    """Run one sample to completion and return its status row. Never raises for a failed job."""
    # The sample's name, and its result file.
    name = "s%04d" % row_no
    out_dir = Path(args.out_dir)
    final_json = out_dir / (name + ".json")
    if final_json.exists():  # skip-if-exists: the campaign is resumable
        return None
    # A fresh working folder of its own, holding the template patched with this row's values.
    work = (out_dir / name).resolve()
    shutil.rmtree(work, ignore_errors=True)  # a killed job's leftovers (a .lck file) would block `abaqus job=`
    work.mkdir(parents=True)
    values = {k: float(row[k]) for k in INPUTS}
    values["G13"] = values["G12"]  # G13 = G12 in every sample: one variable (NB8)
    (work / (name + ".inp")).write_text(patch_inp(template, values))

    # Wait for the launch slot and the tokens, then solve: `interactive` keeps the command running until the job ends.
    launcher.wait()
    t_launch = time.time()
    # ponytail: cpus > 1 on the command line is untested in this path (the CAE path needed numDomains = cpus).
    code, peak_mb, cpu_s = run_monitored(
        [args.abaqus, "job=" + name, "input=%s.inp" % name, "cpus=%d" % args.cpus, "mp_mode=threads",
         "interactive", "ask_delete=OFF"], str(work), None, work / "driver_stdout.txt")
    t_solved = time.time()
    # Extract the results from the output database, if the job wrote one (no licence token needed).
    extract_json = work / (name + "_extract.json")
    if (work / (name + ".odb")).exists():
        _, peak2_mb, cpu2_s = run_monitored(
            [args.abaqus, "python", str(SCRIPTS_DIR / "extract_open_hole_tension.py"), "job=" + name,
             "meta=" + meta["meta_path"], "out=" + str(extract_json)], str(work), None,
            work / "extract_stdout.txt")
        # The sample's memory is the larger of the two stages' peaks; its CPU time is their sum.
        peak_mb, cpu_s = max(peak_mb, peak2_mb), cpu_s + cpu2_s
    wall_s = time.time() - t_launch

    # With `interactive` the job log goes to the job's stdout; read the .log file too, if one was written.
    log = "".join(p.read_text(errors="replace") for p in (work / "driver_stdout.txt", work / (name + ".log")) if p.exists())
    release, tokens = parse_log(log)
    if not extract_json.exists() and tokens is None:  # no licence: no JSON, so the next run retries it
        reason = "licence not obtained (rerun)"
        return {"sample": name, "accepted": False, "reason": reason, "wall_s": round(wall_s, 1),
                "cpu_s": round(cpu_s, 1), "peak_ram_mb": round(peak_mb), "tokens": None}

    # The build script's JSON keys: the template's mesh statistics and fixed settings, this sample's inputs
    # and its own ceiling share, the extracted results and the stage time stamps.
    result = {k: v for k, v in meta.items() if k not in ("lc_values", "physical_angles", "meta_path")}
    result.update({k.lower(): v for k, v in values.items()})
    result.update({"run": name, "cpus": args.cpus,
                   "frac_elements_over_ceiling": ceiling_share(meta["lc_values"], values["E2"], values["G_mT"], values["Y_T"])})
    # The extracted results and the screening verdict, or a failed job's placeholders.
    if extract_json.exists():  # success = the output file exists; returncode is unreliable
        with open(extract_json) as f:
            result.update(json.load(f))
        reason = acceptance_reason(result)
    else:
        result.update({"completed": False, "fpf": None})
        reason = "no result file (job failed)"
    # The driver's own record: stage time stamps (no CAE stages here, so start-up and build take 0 s), the design
    # row, where and with what it ran, its cost, and the verdict.
    result.update({
        "t_script": t_launch, "t_submit": t_launch, "t_solved": t_solved, "t_extracted": t_launch + wall_s,
        "sample": name, "design_row": {k: values[k] for k in INPUTS}, "host": socket.gethostname(),
        "abaqus_release": release, "tokens": tokens, "workers": args.workers,
        "t_launch": t_launch, "wall_s": wall_s, "driver_cpu_s": cpu_s, "peak_ram_mb": peak_mb,
        "accepted": reason == "", "reason": reason,
    })
    # Writing the JSON marks the sample as done for every later run of the driver.
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(final_json, "w") as f:
        json.dump(result, f)
    if not keep:  # keep the .odb (and every other job file) only for the first --keep-odb samples
        shutil.rmtree(work, ignore_errors=True)
    # The line for status.csv.
    return {"sample": name, "accepted": result["accepted"], "reason": reason, "wall_s": round(wall_s, 1),
            "cpu_s": round(cpu_s, 1), "peak_ram_mb": round(peak_mb), "tokens": tokens}


def default_workers():
    """Logical processors - 1 (one left for the machine), or fewer if memory is short (NB8, Part 2, Step 5)."""
    # The smaller of the processor limit and the memory limit (total RAM over one job's peak), and at least 1.
    return max(1, min(psutil.cpu_count(logical=True) - 1, int(psutil.virtual_memory().total / 2**30 / PEAK_RAM_GB)))


def main():
    # Command-line options; the docstring's first paragraph is the --help description.
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--design", default=str(RUNS_DIR / "design_train.csv"), help="design CSV (default: %(default)s)")
    ap.add_argument("--rows", help="'a-b' (both included) or 'a'; default: every row")
    ap.add_argument("--out-dir", default=str(RUNS_DIR / "train"), help="folder for the sample JSONs and status.csv")
    ap.add_argument("--workers", type=int, default=default_workers(), help="most concurrent jobs (default: %(default)s)")
    ap.add_argument("--cpus", type=int, default=1, choices=sorted(TOKENS_PER_JOB), help="solver cores per job (m)")
    ap.add_argument("--mesh", default="M1", choices=sorted(MESHES))
    ap.add_argument("--keep-odb", type=int, default=0, help="keep the job files of the first n samples")
    ap.add_argument("--abaqus", default="abaqus", help="abaqus command, if it is not `abaqus` on PATH")
    ap.add_argument("--reserve", type=float, default=0.0,
                    help="share of each solver pool left free for others, e.g. 0.1 (default: none)")
    # 1 s plus the licence query (0.5-1.5 s) is one launch every ~2 s, enough for ~170 jobs of 350 s; NB8's 5 s
    # kept the 120-thread server at 59 jobs on average.
    ap.add_argument("--stagger", type=float, default=1.0, help="seconds between job launches (default: %(default)s)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    # The full path of the abaqus command, so every subprocess finds it without a shell.
    abaqus = shutil.which(args.abaqus)
    if abaqus is None:
        sys.exit("no `%s` command on PATH: load the Abaqus module, or pass --abaqus" % args.abaqus)
    args.abaqus = abaqus
    env = dict(os.environ, ABAQUS_BAT_PATH=abaqus)  # how abqpy, in the template build, finds the abaqus command
    # The design rows (dicts by input name), the rows asked for, and those without a result file yet.
    with open(args.design, newline="") as f:
        design = list(csv.DictReader(f))
    rows = parse_rows(args.rows) if args.rows else list(range(len(design)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out_dir / ("s%04d.json" % r)).exists()]
    print("%d of %d samples to run (at most %d workers x %d cpus)" % (len(todo), len(rows), args.workers, args.cpus), flush=True)
    if not todo:
        return
    # The template input file (built once if missing) and its JSON, shared read-only by every worker.
    meta_path = ensure_template(args, env)
    with open(meta_path) as f:
        meta = dict(json.load(f), meta_path=str(meta_path))
    template = (TEMPLATE_DIR / (args.mesh + ".inp")).read_text()
    # status.csv is appended to, so a resumed campaign adds its lines below the earlier ones.
    status_path = out_dir / "status.csv"
    new_file = not status_path.exists()
    launcher = Launcher(args.stagger, abaqus, TOKENS_PER_JOB[args.cpus], args.reserve)
    with open(status_path, "a", newline="") as sf, ThreadPoolExecutor(max_workers=args.workers) as pool:
        status = csv.DictWriter(sf, fieldnames=STATUS_FIELDS)
        if new_file:
            status.writeheader()
        # Queue every sample at once; at most `workers` run, each waiting at the launcher for its turn.
        futures = [pool.submit(run_sample, r, design[r], args, launcher, i < args.keep_odb, meta, template)
                   for i, r in enumerate(todo)]
        try:
            # Collect the results in submission order and record each one as soon as it is in.
            for done, fut in enumerate(futures, start=1):
                try:
                    line = fut.result()
                except Exception as error:  # a driver fault in one sample must not stop the others being recorded
                    print("[%d/%d] driver error: %r (the sample has no result file and will be rerun next time)"
                          % (done, len(todo), error), flush=True)
                    continue
                if line is not None:
                    status.writerow(line)
                    sf.flush()
                    print("%s [%d/%d] %s accepted=%s %s wall %s s" % (time.strftime("%H:%M:%S"), done, len(todo),
                          line["sample"], line["accepted"], line["reason"], line["wall_s"]), flush=True)
        except KeyboardInterrupt:  # Ctrl-C: start no new sample (the running jobs get the same Ctrl-C)
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def selftest():
    """Checks the pure functions on made-up data; needs neither Abaqus nor a design file."""
    assert parse_rows("32-32") == [32] and parse_rows("2-4") == [2, 3, 4] and parse_rows("5") == [5]
    good = {"completed": True, "fpf": {"u2_mm": 0.2}, "rf2": [0, 10, 30, 12, 5], "peak_index": 2,
            "allcd": [0, 0, 0.001, 0, 0], "alldmd": [0, 0, 1.0, 0, 0]}
    assert acceptance_reason(good) == ""
    assert acceptance_reason(dict(good, completed=False)) == ""  # stopped early, but after a clear drop
    assert "load drop" in acceptance_reason(dict(good, rf2=[0, 10, 30, 25, 24]))  # a drop of only 20%
    assert acceptance_reason(dict(good, rf2=[0, 10, 30, 23, 22])) == ""  # a drop of more than 20%
    assert "first-ply" in acceptance_reason(dict(good, fpf=None))
    assert "load drop" in acceptance_reason(dict(good, rf2=[0, 10, 30, 29, 28]))  # no clear drop
    assert "load drop" in acceptance_reason(dict(good, rf2=[0, 10, 30], allcd=[0] * 3, alldmd=[1] * 3))  # peak last
    assert "1%" in acceptance_reason(dict(good, allcd=[0, 0, 0.05, 0, 0]))
    assert parse_log("Abaqus JOB x\nAbaqus 2023\nAbaqus/Standard checked out 50 tokens from DSLS") == ("2023", 50)
    assert parse_log("nothing here") == (None, None)
    # dslsstat: two pools per feature are summed; other features and non-token rows are ignored.
    listing = ("| QAE   | 28 | 01-ott-2027 00:59:00 | Token | STD |   34 |    0 | x | y | z\n"
               "| QSD   | 28 | 29-set-2027 00:59:00 | Token | STD | 2000 |  150 | x | y | z\n"
               "| QSD   | 28 | 01-ott-2027 00:59:00 | Token | STD | 3600 |    0 | x | y | z\n"
               "| QXT   | 28 | 29-set-2027 00:59:00 | Token | STD | 2000 | 1228 | x | y | z\n"
               "| QPL   | 28 | 01-ott-2027 00:59:00 | ConcurrentUser | STD | 16384 | 1 | x | y | z\n")
    assert parse_dslsstat(listing) == {"QSD": (5450, 5600), "QXT": (772, 2000)}
    assert parse_dslsstat("no table") == {}
    # Room for a job: QXT above has 772 free of 2000; 10% reserve leaves 572, i.e. 11 jobs of 50 tokens.
    pools = parse_dslsstat(listing)
    assert room_for_job(pools, 50, 0.0, 14) and not room_for_job(pools, 50, 0.0, 15)
    assert room_for_job(pools, 50, 0.1, 10) and not room_for_job(pools, 50, 0.1, 11)
    # .inp patching: the line after each material keyword is replaced, everything else kept.
    deck = ("*Material, name=M\n*Damage Initiation, criterion=HASHIN, alpha=1.\n 1., 2., 3., 4., 5., 6.\n"
            "*Damage Evolution, type=ENERGY\n 1., 2., 3., 4.\n*Damage Stabilization\n0.0001\n"
            "*Elastic, type=LAMINA\n1., 2., 3., 4., 5., 6.\n** BOUNDARY CONDITIONS")
    values = {n: (i + 1) * 10.5 for i, n in enumerate(INPUTS)}
    values["G13"] = 99.25
    lines = patch_inp(deck, values).split("\n")
    assert lines[2] == ", ".join(repr(values[n]) for n in MATERIAL_LINES["*Damage Initiation, criterion=HASHIN"])
    assert lines[4] == "126.0, 136.5, 147.0, 157.5" and lines[6] == "0.0001" and lines[-1] == "** BOUNDARY CONDITIONS"
    assert lines[8] == "10.5, 21.0, 31.5, 42.0, 99.25, 52.5"
    try:
        patch_inp(deck.replace("*Elastic, type=LAMINA", "*Elastic"), values)
        raise RuntimeError("a missing keyword must fail")
    except AssertionError:
        pass
    # Ceiling share: nominal ceiling 2 * 0.2774 * 10300 / 65.5^2 = 1.332 mm.
    assert ceiling_share([1.0, 1.3, 1.4, 2.0], 10300.0, 0.2774, 65.5) == 0.5
    print("selftest ok")


if __name__ == "__main__":
    main()
