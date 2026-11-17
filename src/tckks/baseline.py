"""Non-private baseline: the same computations on pooled plaintext data.

This is what a trusted data warehouse would compute. It is the reference for
every correctness test and the speed baseline for the performance comparison.
"""
from __future__ import annotations

import numpy as np

from .data import scale_public, standardize, with_intercept


def local_moments(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """mu_j = (n, sum u, sum u^2, sum y, sum y^2) on publicly scaled features u in [0,1].

    2d+3 = 17 numbers for 7 features. Working on u keeps every moment <= n,
    so after division by N_b each slot is <= 1 and cannot overflow CKKS.
    """
    U = scale_public(X)
    return np.concatenate([[len(y)], U.sum(0), (U ** 2).sum(0), [y.sum()], [(y ** 2).sum()]])


def pooled_moments(sites) -> np.ndarray:
    return sum(local_moments(X, y) for X, y in sites)


def moments_to_stats(m: np.ndarray, p: int = 7):
    """Turn pooled moments into (N, mean of u, std of u, mean y), u = scaled features."""
    n = m[0]
    mean = m[1:1 + p] / n
    var = m[1 + p:1 + 2 * p] / n - mean ** 2
    return n, mean, np.sqrt(np.maximum(var, 1e-12)), m[1 + 2 * p] / n


def design_matrix(X, mode: str, mean=None, std=None):
    if mode == "standardized":
        return with_intercept(standardize(scale_public(X), mean, std))
    if mode == "public":
        return with_intercept(scale_public(X))
    raise ValueError(mode)


def gram_terms(sites, mode, mean=None, std=None):
    G = sum(design_matrix(X, mode, mean, std).T @ design_matrix(X, mode, mean, std) for X, _ in sites)
    g = sum(design_matrix(X, mode, mean, std).T @ y for X, y in sites)
    return G, g


def ols(sites, mode, mean=None, std=None):
    """Exact least-squares coefficients on pooled data."""
    G, g = gram_terms(sites, mode, mean, std)
    return np.linalg.solve(G, g)


def gradient_descent(G_tilde, g_tilde, eta, iters, return_path=False):
    """beta_{t+1} = (I - eta*G~) beta_t + eta*g~, starting from beta_0 = 0.

    Uses exactly the recurrence the encrypted server runs, so the difference
    between this and the decrypted result isolates the CKKS error.
    """
    d = len(g_tilde)
    M = np.eye(d) - eta * G_tilde
    c = eta * g_tilde
    beta = np.zeros(d)
    path = []
    for _ in range(iters):
        beta = M @ beta + c
        path.append(beta.copy())
    return (beta, path) if return_path else beta
