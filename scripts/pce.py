# Code under BSD-3-Clause (see LICENSE). (c) 2026 Francesco Mario Antonio Mitrotta.
"""
pce.py
------
Polynomial chaos expansion: NB8 Part 2's own functions, unchanged, imported here rather than
redefined -- total_order_indices (the multi-indices of a total-order expansion), design_matrix
(every basis polynomial at every sample), basis_norms (E[psi^2] = prod alpha_k!), fit_pce (point
collocation by least squares) and relative_mse (the validation error epsilon, Thapa et al.
Eq. 34) and sobol_indices (first-order and total Sobol' indices from the coefficients, Thapa et al.
Eqs. 29-33). NB8 derives and explains every one of them and checks them against a function with known
coefficients, which is why later notebooks import this module rather than walk through it again.

Nothing in this module touches Abaqus, and nothing in it imports `abaqus` or `odbAccess`, so it is
safe to import directly into a live Jupyter kernel.
"""
import itertools
import math

import numpy as np
from numpy.polynomial import hermite_e as He


def total_order_indices(d, p):
    """All multi-indices alpha with sum(alpha) <= p for d inputs.

    Returns an integer array of shape (P+1, d), ordered by total degree, the constant term first.
    """
    rows = []
    # Loop over the total degree, 0..p.
    for degree in range(p + 1):
        # A combination (with repeats) of `degree` input numbers: input k appearing m times means alpha_k = m.
        for combo in itertools.combinations_with_replacement(range(d), degree):
            # Count how often each input appears; the constant term (degree 0) is the zero vector.
            rows.append(np.bincount(combo, minlength=d) if degree > 0 else np.zeros(d, dtype=int))
    indices = np.array(rows)
    # Sanity check: the enumeration must give as many multi-indices as the binomial formula (p+d)!/(p! d!).
    assert len(indices) == math.comb(p + d, d)
    return indices


def design_matrix(xi, alphas):
    """Evaluate every basis polynomial at every sample.

    Parameters
    ----------
    xi : (n, d) array
        Standardised inputs, one row per sample [-].
    alphas : (P+1, d) integer array
        Multi-indices, one row per basis polynomial.

    Returns
    -------
    (n, P+1) array
        Psi[i, j] = prod_k He_{alphas[j, k]}(xi[i, k]).
    """
    n, d = xi.shape
    # The highest degree needed, and He_m(xi[i, k]) for every degree m up to it (hermevander): shape (n, d, p_max+1).
    p_max = alphas.max()
    he = He.hermevander(xi, p_max)
    # Start from ones, then multiply in one factor He_{alpha_k}(xi_k) per input.
    psi = np.ones((n, len(alphas)))
    for j, alpha in enumerate(alphas):
        for k in range(d):
            psi[:, j] *= he[:, k, alpha[k]]
    return psi


def basis_norms(alphas):
    """E[psi_alpha^2] = prod_k alpha_k! for every multi-index (orthogonality of the He_n)."""
    # The product of the factorials of the degrees, one value per multi-index.
    return np.array([np.prod([math.factorial(m) for m in alpha]) for alpha in alphas], dtype=float)


def relative_mse(y_true, y_pred):
    """Relative mean-square error epsilon, Thapa et al. Eq. (34)."""
    # Squared error of the prediction over the squared spread of the true values around their mean.
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def fit_pce(xi, y, alphas):
    """Least-squares PCE coefficients from samples (xi, y): solves Psi a = y, Thapa et al. Eqs. (26)-(27)."""
    # lstsq returns the coefficients first; the other return values (residuals, rank, singular values) are not needed.
    coeffs, *_ = np.linalg.lstsq(design_matrix(xi, alphas), y, rcond=None)
    return coeffs


def sobol_indices(coeffs, alphas):
    """First-order and total Sobol' indices of every input, read from PCE coefficients (Thapa et al. Eqs. 29-33).

    Parameters
    ----------
    coeffs : (P+1,) array
        PCE coefficients a_alpha, in the order of `alphas`.
    alphas : (P+1, d) integer array
        Multi-indices, one row per basis polynomial (the constant term included).

    Returns
    -------
    S, S_tot : (d,) arrays
        S_i sums the variance shares a_alpha^2 E[psi_alpha^2] of the terms in xi_i alone; S_i^tot those of every
        term containing xi_i; both divided by the expansion's variance (the constant term excluded).
    """
    # Each term's share of the variance, a^2 E[psi^2]; the constant term carries none.
    active = alphas > 0
    shares = coeffs ** 2 * basis_norms(alphas) * active.any(axis=1)
    # Terms in xi_i alone (one active input) for S_i; every term containing xi_i for S_i^tot.
    alone = active & (active.sum(axis=1, keepdims=True) == 1)
    return alone.T @ shares / shares.sum(), active.T @ shares / shares.sum()
