"""
bootstrap.py
------------
NB3 Parts 1-2's own convergence machinery -- bootstrap_ci, ConvergenceChecker, and the
minimum-sample-size existence condition NB3 derives for a distribution-free quantile interval
-- imported here rather than redefined, exactly like `kde.py`'s KDE functions and
`mc_campaign.py`'s campaign loop. NB3 derives and explains every one of these in full (Part 1,
"The bootstrap"; Part 2, "Choosing a convergence criterion"), which is why this module is
imported rather than walked through again by any notebook after NB3.

What is deliberately NOT here: NB3's `run_until_converged`. That function's batching logic is
Abaqus-specific -- it grows a campaign by handing `mc_campaign.run_campaign` a longer prefix of
the input pool each time, i.e. by launching more Abaqus jobs. Everything in this module, by
contrast, only ever touches an array of numbers already in hand, so a notebook whose samples
come from somewhere else entirely (a closed-form map, say) can reuse the criterion without
inheriting the subprocess machinery.

Nothing in this module touches Abaqus, and nothing in it imports `abaqus` or `odbAccess`, so
unlike `build_fiber_tension.py` it is safe to import directly into a live Jupyter kernel.
"""
import numpy as np

__all__ = ["bootstrap_ci", "ConvergenceChecker", "min_samples_for_quantile_ci"]


def bootstrap_ci(x, statfunc, n_boot=1000, rng=None):
    '''Percentile-method bootstrap CI for any statistic. `statfunc(a, axis)` must compute the
    statistic ALONG `axis` -- np.mean and np.percentile both do this already, which is what
    lets every one of the n_boot replicates be computed in a single vectorised call instead of
    a Python loop over resamples.

    The point estimate is `statfunc` applied to the ORIGINAL sample; the replicates describe
    its bootstrap uncertainty (NB3 Part 1's own distinction, made explicit again here since
    this one function is what every convergence check calls).
    '''
    x = np.asarray(x, dtype=float)
    n = len(x)
    estimate = float(statfunc(x, axis=0))
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, n, size=(n_boot, n))     # WITH replacement, see NB3 Part 1
    replicates = statfunc(x[idx], axis=1)          # one replicate per resample, all n_boot rows at once
    ci_lo, ci_hi = np.percentile(replicates, [2.5, 97.5])
    half_width = 0.5 * (ci_hi - ci_lo)
    return {
        "n": n, "estimate": estimate, "ci_low": float(ci_lo), "ci_high": float(ci_hi),
        "half_width": float(half_width), "rel_half_width": float(half_width / abs(estimate)),
        "replicates": replicates,
    }


class ConvergenceChecker:
    '''Stateful convergence criterion for an open-ended Monte Carlo campaign: keep adding
    samples, bootstrapping a fresh confidence interval on the order-statistic Q5% after every
    batch, until that interval's relative half-width drops to (or below) the 1% target for the
    FIRST time. `n_min` is not this class's job to enforce -- NB3 Part 2's existence condition
    (see `min_samples_for_quantile_ci` below) means no check below n_min could ever pass
    anyway, so the caller's batching loop simply never calls `check` before then.

    Call `check(sample)` once per batch, with every completed sample so far.
    '''

    def __init__(self, label, p=0.05, target_rel_half_width=0.01, n_boot=1000, rng=None, unit="N",
                 verbose=True):
        self.label = label                              # a short tag, printed with every check, so several campaigns' logs are easy to tell apart
        self.p = p                                       # which percentile to track (0.05 -> Q5%)
        self.target_rel_half_width = target_rel_half_width  # the 1% stopping target
        self.n_boot = n_boot                             # bootstrap resamples per check (B, sized in NB3 Part 1)
        self.unit = unit                                 # printed after the estimate -- NB3's campaigns are forces (N); a later notebook (NB5) monitoring a displacement passes unit="mm"
        # One line is printed per check. That log IS the point in NB3, where the criterion
        # itself is what is being studied; a later notebook running several campaigns at once
        # only wants their final result, and passes verbose=False to silence the per-check line
        # without losing anything -- every check is still recorded in self.history either way.
        self.verbose = verbose
        # A dedicated random generator for THIS checker's own bootstrap draws, kept separate
        # from whatever generator drew the input samples themselves, so the two streams of
        # randomness cannot interfere with each other, and re-used (not re-seeded) across
        # calls so consecutive checks draw fresh, independent resamples.
        self.rng = rng if rng is not None else np.random.default_rng()
        self.history = []        # one bootstrap result dict per check() call, for later plotting

    def check(self, sample):
        # A sample that failed to produce a value contributes NaN; drop it before
        # bootstrapping, since np.percentile has no sensible answer for a NaN-contaminated row.
        sample = np.asarray(sample, dtype=float)
        sample = sample[~np.isnan(sample)]

        # The statistic being bootstrapped: the p-th percentile, read directly off the sample
        # (or off each resample) as an order statistic -- no smoothing, see NB3 Part 1.
        q_fn = lambda a, axis: np.percentile(a, 100 * self.p, axis=axis)
        result = bootstrap_ci(sample, q_fn, n_boot=self.n_boot, rng=self.rng)
        self.history.append(result)   # keep every check, so the convergence plots can show the whole trajectory

        passed = result["rel_half_width"] <= self.target_rel_half_width
        if self.verbose:
            print(f"[{self.label}] n={result['n']:4d}  "
                  f"Q{100 * self.p:.0f}%_hat={result['estimate']:8.2f} {self.unit}  "
                  f"95% CI=[{result['ci_low']:8.2f}, {result['ci_high']:8.2f}]  "
                  f"rel_half_width={100 * result['rel_half_width']:.3f}%  "
                  f"({'CONVERGED' if passed else 'not yet'})")
        return passed


def min_samples_for_quantile_ci(p=0.05, alpha=0.05, batch_size=None):
    """Smallest N below which a (1-alpha) distribution-free interval on Q_p cannot exist at all
    -- NB3 Part 2's own existence bound, as a function rather than a hard-coded 60.

    The derivation (NB3 Part 2, in full): an order-statistic interval for Q_p is bounded by the
    sample's own extremes, [x_(1), x_(N)]. That widest possible interval still fails to cover
    Q_p whenever every one of the N points happens to land on the same side of it, which
    happens with probability p**N + (1-p)**N (all above, or all below). A (1-alpha) interval
    can only exist once that failure probability is itself below alpha:

        p**N + (1-p)**N <= alpha   ==>   N >= ln(alpha) / ln(1-p)

    (the p**N term is negligible for small p). At p = alpha = 0.05 this gives N >= 58.4, i.e.
    N >= 59. `batch_size`, if given, rounds that up onto a check grid of that spacing -- 60 for
    NB3's own 10-sample batches -- since a criterion checked only every `batch_size` samples
    can never stop between grid points anyway.

    The bound is symmetric in p <-> 1-p: "all N points land on the same side of Q_p" has
    probability p**N + (1-p)**N either way, so Q_5% and Q_95% share the same n_min. The
    small-p approximation used just above only holds for the SMALLER of the two tails, though
    -- at p=0.95 the dropped term (0.95**N) is the dominant one, not the negligible one -- so
    `p` is first folded onto whichever side of 0.5 it actually is:
    """
    p = min(p, 1.0 - p)
    n_min = int(np.ceil(np.log(alpha) / np.log(1.0 - p)))
    if batch_size is not None:
        n_min = int(np.ceil(n_min / batch_size) * batch_size)
    return n_min
