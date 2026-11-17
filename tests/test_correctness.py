"""Correctness tests for the threshold-CKKS protocol.

Run with:  pytest -q tests/
Uses the small P1-compact parameter set and a small cohort so the whole suite
finishes in well under a minute.
"""
import numpy as np
import pytest

from tckks import baseline as B
from tckks.data import make_cohort
from tckks.params import PARAM_SETS, D
from tckks.protocol import IntegrityError, ThresholdCKKS, round_sig

PS = PARAM_SETS["P1-compact"]


@pytest.fixture(scope="module")
def cohort():
    return make_cohort(num_sites=3, n_per_site=400, seed=7)


@pytest.fixture(scope="module")
def proto(cohort):
    p = ThresholdCKKS(PS, cohort)
    p.keygen()
    return p


def test_diagonal_method_matches_matrix_vector(cohort):
    """Plaintext check of the packing used by the server: M b = sum_k diag_k(M) * rot_k(b)."""
    rng = np.random.default_rng(0)
    M, b = rng.normal(size=(D, D)), rng.normal(size=D)
    out = sum(np.array([M[i, (i + k) % D] for i in range(D)]) * np.roll(b, -k) for k in range(D))
    assert np.allclose(out, M @ b)


def test_mode_a_pooled_moments(proto, cohort):
    """Decrypted pooled moments equal the plaintext pooled moments."""
    proto.run_mode_a()
    truth = B.pooled_moments(cohort)
    assert np.max(np.abs(proto.moments_raw - truth) / np.maximum(1.0, np.abs(truth))) < 1e-3
    n, mean, std, _ = B.moments_to_stats(truth)
    assert proto.released["N"] == n
    assert np.allclose(proto.released["mean"], round_sig(mean), rtol=2e-3)
    assert np.allclose(proto.released["std"], round_sig(std), rtol=2e-3)


def test_threshold_decryption_needs_every_share(proto, cohort):
    """n-1 shares must not decrypt: the result is garbage or decoding fails."""
    x = np.linspace(-0.5, 0.5, D)
    ct = proto._encrypt(x, "test_encrypt")
    assert np.allclose(proto.decrypt(ct, D), x, atol=5e-3)
    try:
        partial = proto.decrypt(ct, D, shares_from=proto.sites[:-1])
        assert np.max(np.abs(partial - x)) > 1.0
    except RuntimeError:
        pass  # OpenFHE refuses to decode: also a pass


def test_collective_refresh_preserves_plaintext(proto):
    x = np.linspace(-0.9, 0.9, D)
    ct = proto._encrypt(x, "test_encrypt")
    for _ in range(PS.epoch_len):  # use up an epoch's worth of levels
        ct = proto.cc.Rescale(proto.cc.EvalMult(ct, proto._encrypt(np.ones(D), "test_encrypt")))
    fresh = proto.refresh(ct)
    assert fresh.GetLevel() < ct.GetLevel()
    assert np.allclose(proto.decrypt(fresh, D), x, atol=5e-3)


@pytest.mark.parametrize("hoisted", [True, False])
def test_encrypted_regression_matches_plaintext_gd(cohort, hoisted):
    p = ThresholdCKKS(PS, cohort, hoisted=hoisted)
    p.keygen()
    rel = p.run_mode_a()
    K = 10  # 9 products; refreshes after products 3 and 6 (epoch length 3)
    dc, cc = p.encrypt_regression_inputs("standardized", rel["mean"], rel["std"], rel["N"])
    beta = p.decrypt(p.run_regression(dc, cc, K), D)
    G, g = B.gram_terms(cohort, "standardized", rel["mean"], rel["std"])
    expected = B.gradient_descent(G / rel["N"], g / rel["N"], 1.0 / D, K)
    assert p.m.counts["refreshes"] == 2
    # P1 has ~12 bits of precision after flooding (Delta = 2^40, sigma_flood = 2^20)
    assert np.max(np.abs(beta - expected)) < 5e-3


def test_ciphertext_size_independent_of_site_size():
    """Leakage check: what the server sees does not depend on how many records a site has."""
    small = ThresholdCKKS(PS, make_cohort(3, 50, seed=1))
    small.keygen()
    big = ThresholdCKKS(PS, make_cohort(3, 2000, seed=1))
    big.keygen()
    small.run_mode_a()
    big.run_mode_a()
    assert small.m.sizes["ciphertext_fresh"] == big.m.sizes["ciphertext_fresh"]


def test_small_sites_are_excluded():
    data = make_cohort(3, 400, seed=3)
    data[1] = (data[1][0][:5], data[1][1][:5])  # 5 records < 11
    p = ThresholdCKKS(PS, data)
    assert p.n_parties == 2


def test_integrity_check_detects_dropped_ciphertext(cohort):
    """I1: if the server swaps out one site's ciphertext, no site releases a decryption share."""
    p = ThresholdCKKS(PS, cohort)
    p.keygen()
    rel = p.run_mode_a()
    dc, cc = p.encrypt_regression_inputs("standardized", rel["mean"], rel["std"], rel["N"])
    cc[1] = cc[0]                       # malicious server: drop site 1's input, double-count site 0's
    beta = p.run_regression(dc, cc, 3)
    with pytest.raises(IntegrityError):
        p.decrypt(beta, D)
