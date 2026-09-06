"""
kde.py
------
Kernel density estimation: NB2 Part 5's own functions, unchanged, imported here rather than
redefined -- silverman_bandwidth, kde_loo_log_likelihood, kde_loo_bandwidth (bandwidth selection
by leave-one-out cross-validation), kde_pdf, kde_cdf, and kde_percentile (bisecting the KDE's
own closed-form CDF). NB2 derives and explains every one of these in full; nothing here is new
to any notebook after NB2, which is why this module is imported rather than walked through again.

Nothing in this module touches Abaqus, and nothing in it imports `abaqus` or `odbAccess`, so
unlike `build_fiber_tension.py` it is safe to import directly into a live Jupyter kernel.
"""
import numpy as np
from scipy.stats import norm


def silverman_bandwidth(x):
    """Silverman's rule-of-thumb bandwidth, h = 1.06 * s * n**(-1/5) (Silverman, 1986) -- a
    closed-form starting point for the grid search below, not the answer itself. It minimizes
    the *asymptotic mean integrated squared error* -- the expected squared gap between the
    estimated and true density, integrated over all x -- but only under the assumption that the
    true density is Normal. The n**(-1/5) factor is why it shrinks only slowly as more data
    arrives: more samples support finer detail, but only a little at a time."""
    x = np.asarray(x, dtype=float)
    return float(1.06 * x.std(ddof=1) * len(x) ** (-0.2))


def kde_loo_log_likelihood(x, h):
    """Leave-one-out log-likelihood of bandwidth h for sample x: for each point, build the
    density from every OTHER point and evaluate it there, then sum the logs."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    z = (x[:, None] - x[None, :]) / h                 # every pairwise standardised distance x_i - x_j
    k = np.exp(-0.5 * z ** 2)                          # phi(z) for every pair (missing its 1/(h*sqrt(2*pi)) normalization for now)
    np.fill_diagonal(k, 0.0)                           # this is the "leave one out": drop each point's own kernel from its own sum
    dens = k.sum(axis=1) / ((n - 1) * h * np.sqrt(2.0 * np.pi))  # normalize each row -> f_{-i}(x_i), one value per sample
    return float(np.log(np.maximum(dens, 1e-300)).sum())  # sum of logs; the tiny floor avoids log(0) if a density ever rounds to exactly zero


def kde_loo_bandwidth(x, n_grid=40, lo_factor=0.3, hi_factor=2.5):
    """Choose the KDE bandwidth by grid search on the leave-one-out log-likelihood. The grid
    spans lo_factor to hi_factor times Silverman's rule, which brackets the optimum comfortably
    for the unimodal samples this repository produces. Returns (h_best, h_grid, log_likelihoods)
    so a notebook can plot the criterion being maximised rather than just quoting its answer."""
    x = np.asarray(x, dtype=float)
    h_grid = silverman_bandwidth(x) * np.linspace(lo_factor, hi_factor, n_grid)  # candidate h's, centered on Silverman's rule
    ll = np.array([kde_loo_log_likelihood(x, h) for h in h_grid])  # the score for every candidate
    h_best = float(h_grid[int(np.argmax(ll))])  # the grid search itself: simply keep the best-scoring candidate
    return h_best, h_grid, ll


def kde_pdf(x_eval, samples, h):
    """Gaussian KDE evaluated at x_eval: put a Gaussian "bump" of width h on top of every
    sample point and average them. Smooth everywhere, unlike a histogram."""
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    samples = np.asarray(samples, dtype=float)
    z = (x_eval[:, None] - samples[None, :]) / h  # every (evaluation point, sample) pairwise distance, rescaled by h
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(samples) * h * np.sqrt(2.0 * np.pi))  # sum of kernels, normalized by N*h


def kde_cdf(x_eval, samples, h):
    """CDF of the Gaussian KDE above, in closed form: integrating a sum of Gaussian kernels
    term by term replaces each one by its own CDF."""
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    samples = np.asarray(samples, dtype=float)
    return norm.cdf((x_eval[:, None] - samples[None, :]) / h).mean(axis=1)


def kde_percentile(samples, h, p=0.05, tol=0.01):
    """The p-th percentile of the KDE, found by bisecting its closed-form CDF.

    The bisection bracket starts BRACKET_SIGMAS bandwidths outside the sample range on each
    side. A Gaussian kernel's tail is already below 1e-9 of its peak six standard deviations
    out, so starting the bracket that far beyond the data guarantees F_hat is essentially 0 at
    the low end and essentially 1 at the high end -- i.e. that the true root really does lie
    inside the starting bracket, for any percentile this notebook would ever ask for.

    The loop halves the bracket every iteration and stops once its width is below `tol`
    (newtons) -- narrow enough that the result is exact to well under a tenth of a percent of
    F_peak's own scale, several orders tighter than anything else this notebook estimates to.
    """
    samples = np.asarray(samples, dtype=float)
    BRACKET_SIGMAS = 6.0  # see docstring: far enough that the Gaussian tail there is negligible
    lo, hi = samples.min() - BRACKET_SIGMAS * h, samples.max() + BRACKET_SIGMAS * h
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if float(kde_cdf(mid, samples, h)[0]) < p:
            lo = mid  # F_hat(mid) is still below the target level -> the root is above mid
        else:
            hi = mid  # F_hat(mid) has already reached the target level -> the root is at or below mid
    return float(0.5 * (lo + hi))
