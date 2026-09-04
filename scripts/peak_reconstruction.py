"""
peak_reconstruction.py
-----------------------
The two reconstruction helpers NB1 derives and fully explains ("Reconstructing the peak and
full-failure displacement"): fitting a line through the two increments either side of a kink in
a run's recorded force-displacement history and reading off where the fitted lines actually
meet or cross zero, rather than trusting the raw recorded increment nearest that event.

Both are needed by more than one notebook (NB2's single-input campaign, NB3's nlgeom=OFF/ON
convergence study, NB4's two-input campaign), so they live here rather than being redefined in
each one -- but nothing about the *idea* is new to any notebook after NB1; that is why this
module is not walked through again the way NB2's own, newly-introduced campaign machinery is.

Neither function touches Abaqus or imports `abaqus`/`odbAccess`, so this module is safe to
import directly into a live Jupyter kernel.
"""
import numpy as np

__all__ = ["reconstruct_peak", "reconstruct_full_failure"]


def reconstruct_peak(u1, rf1, hsn_ft):
    """Reconstruct the true peak by intersecting the elastic and softening branches either side
    of damage initiation -- NB1's fix for the solver's coarse increment spacing right around the
    peak (see NB1's "Reconstructing the peak" section). Works identically for nlgeom=OFF and
    nlgeom=ON runs; only the recorded u1/rf1/hsn_ft values themselves differ between the two.

    Returns (u1_peak, f_peak), or (nan, nan) if this run's recorded history doesn't have at
    least two increments on each side of initiation to fit a line through (should not happen for
    a well-resolved run, but guarded here rather than letting the campaign loop crash on a
    single bad sample).
    """
    u1 = np.asarray(u1, dtype=float)
    rf1 = np.asarray(rf1, dtype=float)
    hsn_ft = np.asarray(hsn_ft, dtype=float)

    init_idx = int(np.argmax(hsn_ft >= 1.0))  # first increment where the criterion reached 1.0
    if init_idx < 2 or init_idx + 1 >= len(u1):
        return float("nan"), float("nan")  # not enough elastic or softening points around initiation

    elastic_idx = [init_idx - 2, init_idx - 1]   # two increments immediately before initiation
    softening_idx = [init_idx, init_idx + 1]     # two increments immediately after (and including) it

    # np.polyfit(..., deg=1) through exactly two points is the exact line through them, not a
    # least-squares fit (see CLAUDE.md on the two-point-fit convention) -- np.roots then finds
    # where the elastic and softening lines meet.
    elastic = np.polyfit(u1[elastic_idx], rf1[elastic_idx], 1)
    softening = np.polyfit(u1[softening_idx], rf1[softening_idx], 1)

    u1_peak = np.roots(elastic - softening)[0]  # where the two lines meet
    f_peak = np.polyval(elastic, u1_peak)
    return float(u1_peak), float(f_peak)


def reconstruct_full_failure(u1, rf1, damage_ft):
    """Reconstruct the full-failure displacement -- NB1's method (see NB1's "Reconstructing the
    peak and full-failure displacement" section): fit a line through the last two increments
    still short of DAMAGEFT=1 (already on the softening branch) and extrapolate it to zero
    force. Works identically regardless of nlgeom or how many uncertain inputs fed the run.

    Returns nan if the run's recorded history never actually reached DAMAGEFT>=0.999 (the
    displacement budget was too short for this particular sample -- a real failure mode for a
    two-input campaign, since the binding case is a joint corner of the sample, not either
    input's own extreme) or has fewer than two increments short of it to fit a line through.
    """
    u1 = np.asarray(u1, dtype=float)
    rf1 = np.asarray(rf1, dtype=float)
    damage_ft = np.asarray(damage_ft, dtype=float)

    if damage_ft.max() < 0.999:
        return float("nan")  # never reached full failure within this run's displacement budget

    full_idx = np.where(damage_ft < 0.999)[0]
    if len(full_idx) < 2:
        return float("nan")  # not enough pre-failure softening points to fit a line through
    full_idx = full_idx[-2:]  # the two increments closest to (but before) full failure

    full = np.polyfit(u1[full_idx], rf1[full_idx], 1)  # exact line through those two points
    u1_full = np.roots(full)[0]  # where the fitted line crosses zero force
    return float(u1_full)
