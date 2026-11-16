# Code under BSD-3-Clause (see LICENSE). (c) 2026 Francesco Mario Antonio Mitrotta.
"""
kde.py
------
Kernel density estimation: NB2 Part 6's own functions, unchanged, imported here rather than
redefined -- silverman_bandwidth, kde_loo_log_likelihood, kde_loo_bandwidth (bandwidth selection
by leave-one-out cross-validation), kde_pdf, kde_cdf, and kde_percentile (the root of the KDE's
own closed-form CDF, by SciPy's brentq). NB2 derives and explains every one of these in full; nothing here is new
to any notebook after NB2, which is why this module is imported rather than walked through again.

Nothing in this module touches Abaqus, and nothing in it imports `abaqus` or `odbAccess`, so
unlike `build_fiber_tension.py` it is safe to import directly into a live Jupyter kernel.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import gaussian_kde, norm


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
    """Gaussian KDE evaluated at x_eval, from SciPy: a Gaussian "bump" of width h on every
    sample, averaged. SciPy's bw_method is the bandwidth in units of the sample std, hence h / s."""
    s = np.std(samples, ddof=1)                               # sample std, the unit of SciPy's bandwidth
    return gaussian_kde(samples, bw_method=h / s)(x_eval)    # build f_hat with width h, evaluate at x_eval


def kde_cdf(x_eval, samples, h):
    """CDF of the Gaussian KDE above, in closed form: integrating a sum of Gaussian kernels
    term by term replaces each one by its own CDF."""
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    samples = np.asarray(samples, dtype=float)
    return norm.cdf((x_eval[:, None] - samples[None, :]) / h).mean(axis=1)


def kde_percentile(samples, h, p=0.05):
    """The p-th percentile of the KDE: the root of F_hat(x) - p. The bracket starts six bandwidths
    outside the sample range, where F_hat is essentially 0 (left) and 1 (right), so it holds the root."""
    lo, hi = np.min(samples) - 6 * h, np.max(samples) + 6 * h        # F_hat - p < 0 at lo, > 0 at hi
    return brentq(lambda x: kde_cdf(x, samples, h)[0] - p, lo, hi)
