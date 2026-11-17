"""Threshold-CKKS protocol for multi-site analytics (Part 2 design, implemented).

Roles
-----
Site      holds raw data and one secret-key share; encrypts local statistics,
          contributes refresh shares and partial decryptions.
Server    untrusted; holds only public/evaluation keys; aggregates ciphertexts
          and runs encrypted gradient descent.
Analyst   fuses partial decryptions and applies the release policy.

All parties run in one process here (a simulation of the network), but every
object that would cross the network is serialized once so its size is measured,
and every party's computation is timed separately.
"""
from __future__ import annotations

import gc
import hashlib
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import openfhe as fhe

from .params import ParamSet, D, N_BOUND, MIN_CELL, ROUND_DIGITS
from .baseline import local_moments, design_matrix, moments_to_stats


# --------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------
@dataclass
class Metrics:
    times: dict = field(default_factory=lambda: defaultdict(list))   # seconds
    sizes: dict = field(default_factory=dict)                        # bytes
    counts: dict = field(default_factory=lambda: defaultdict(int))

    def timed(self, key):
        metrics = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                metrics.times[key].append(time.perf_counter() - self.t)
        return _T()

    def total(self, key):
        return float(sum(self.times.get(key, [])))


def nbytes(obj) -> int:
    """Size of an OpenFHE object in OpenFHE's binary serialization format."""
    return len(fhe.Serialize(obj, fhe.BINARY))


def digest(obj) -> str:
    """SHA-256 of an object's binary serialization (integrity layer I1)."""
    return hashlib.sha256(fhe.Serialize(obj, fhe.BINARY)).hexdigest()


class IntegrityError(RuntimeError):
    """Raised when a site's ciphertext is missing from, or duplicated in, the server transcript."""


def round_sig(x: np.ndarray, digits: int = ROUND_DIGITS) -> np.ndarray:
    """Release policy (Part 2, Phase 5): keep `digits` significant digits."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    nz = x != 0
    mag = np.floor(np.log10(np.abs(x[nz])))
    out[nz] = np.round(x[nz] / 10 ** (mag - digits + 1)) * 10 ** (mag - digits + 1)
    return out


# --------------------------------------------------------------------------
# Crypto context
# --------------------------------------------------------------------------
def make_context(ps: ParamSet, num_parties: int):
    p = fhe.CCParamsCKKSRNS()
    p.SetSecurityLevel(fhe.HEStd_128_classic)
    p.SetSecretKeyDist(fhe.UNIFORM_TERNARY)
    p.SetMultiplicativeDepth(ps.mult_depth)
    p.SetScalingModSize(ps.scale_bits)
    p.SetFirstModSize(ps.first_mod_bits)
    # FIXEDMANUAL: we rescale explicitly after every product, so every ciphertext
    # that is decrypted sits at scale Delta. Decrypting an un-rescaled ciphertext
    # (scale Delta^2) would let its own noise dwarf the 2^20 flooding noise.
    p.SetScalingTechnique(fhe.FIXEDMANUAL)
    p.SetKeySwitchTechnique(fhe.HYBRID)
    p.SetNumLargeDigits(ps.num_large_digits)
    p.SetInteractiveBootCompressionLevel(fhe.CompressionLevel.COMPACT)
    cc = fhe.GenCryptoContext(p)
    for feat in (fhe.PKE, fhe.KEYSWITCH, fhe.LEVELEDSHE, fhe.ADVANCEDSHE, fhe.MULTIPARTY):
        cc.Enable(feat)
    return cc


def context_info(cc, ps: ParamSet) -> dict:
    n = cc.GetRingDimension()
    q = cc.GetModulus()                          # ciphertext modulus Q (without P), as float
    log_q = math.log2(q) if math.isfinite(q) else ps.first_mod_bits + ps.mult_depth * ps.scale_bits
    return {"ring_dim": n, "slots": n // 2, "log_q": round(log_q, 1), **ps.as_dict()}


# --------------------------------------------------------------------------
# Parties
# --------------------------------------------------------------------------
class Site:
    def __init__(self, idx: int, X: np.ndarray, y: np.ndarray):
        self.idx, self.X, self.y = idx, X, y
        self.kp = None
        self.sent_digests = []      # I1: hashes of every ciphertext this site uploaded

    @property
    def n(self):
        return len(self.y)

    def participates(self) -> bool:
        return self.n >= MIN_CELL


class ThresholdCKKS:
    """Runs the full protocol for one parameter set and one cohort."""

    def __init__(self, ps: ParamSet, sites_data, hoisted: bool = True, integrity: bool = True):
        self.ps = ps
        self.integrity = integrity
        self.transcript = []        # I1: hashes of every ciphertext the server aggregated
        self.m = Metrics()
        self.sites = [Site(i, X, y) for i, (X, y) in enumerate(sites_data)]
        self.sites = [s for s in self.sites if s.participates()]
        self.n_parties = len(self.sites)
        self.hoisted = hoisted
        with self.m.timed("setup_context"):
            self.cc = make_context(ps, self.n_parties)
        self.slots = self.cc.GetRingDimension() // 2
        assert self.slots % D == 0
        self.info = context_info(self.cc, ps)

    # ---------------- Phase 1: collective key generation -------------------
    def keygen(self):
        cc, m = self.cc, self.m
        s = self.sites
        # Public key: chained generation, pk = (-a*sum(s_j) + e, a)
        with m.timed("kg_pk_site"):
            s[0].kp = cc.KeyGen()
        for j in range(1, len(s)):
            with m.timed("kg_pk_site"):
                s[j].kp = cc.MultipartyKeyGen(s[j - 1].kp.publicKey)
        self.pk = s[-1].kp.publicKey
        tag = self.pk.GetKeyTag()
        m.sizes["public_key"] = nbytes(self.pk)

        # Relinearization key, round 1 (key-switching shares for s^2)
        with m.timed("kg_relin_r1_site"):
            ek0 = cc.KeySwitchGen(s[0].kp.secretKey, s[0].kp.secretKey)
        m.sizes["relin_r1_share"] = nbytes(ek0)
        acc = ek0
        for j in range(1, len(s)):
            with m.timed("kg_relin_r1_site"):
                share = cc.MultiKeySwitchGen(s[j].kp.secretKey, s[j].kp.secretKey, ek0)
            with m.timed("kg_server_combine"):
                acc = cc.MultiAddEvalKeys(acc, share, s[j].kp.publicKey.GetKeyTag())
        # Round 2
        parts = []
        for j in range(len(s)):
            with m.timed("kg_relin_r2_site"):
                parts.append(cc.MultiMultEvalKey(s[j].kp.secretKey, acc, tag))
        m.sizes["relin_r2_share"] = nbytes(parts[0])
        with m.timed("kg_server_combine"):
            rlk = parts[0]
            for q in parts[1:]:
                rlk = cc.MultiAddEvalMultKeys(rlk, q, rlk.GetKeyTag())
            cc.InsertEvalMultKey([rlk])
        m.sizes["relin_key"] = nbytes(rlk)

        # Rotation keys for the diagonal method: all r = 1..d-1, or only powers of
        # two {1, 2, 4} (other shifts are composed from these; 3/7 of the key memory).
        idx = list(range(1, D)) if self.ps.rot_strategy == "all" else [1, 2, 4]
        with m.timed("kg_rot_site"):
            cc.EvalAtIndexKeyGen(s[0].kp.secretKey, idx)
        base = cc.GetEvalAutomorphismKeyMap(s[0].kp.secretKey.GetKeyTag())
        m.sizes["rot_share_all"] = nbytes(base)
        acc_rot = base
        for j in range(1, len(s)):
            with m.timed("kg_rot_site"):
                mk = cc.MultiEvalAtIndexKeyGen(s[j].kp.secretKey, base, idx, s[j].kp.publicKey.GetKeyTag())
            with m.timed("kg_server_combine"):
                acc_rot = cc.MultiAddEvalAutomorphismKeys(acc_rot, mk, s[j].kp.publicKey.GetKeyTag())
            del mk
            gc.collect()
        del base
        with m.timed("kg_server_combine"):
            cc.ClearEvalAutomorphismKeys()        # drop site 0's own keys held by the context
            cc.InsertEvalAutomorphismKey(acc_rot)
        m.sizes["rot_keys_all"] = nbytes(acc_rot)
        del acc_rot
        gc.collect()
        m.counts["rot_keys"] = len(idx)
        m.sizes["site_keygen_upload"] = (m.sizes["public_key"] + m.sizes["relin_r1_share"]
                                         + m.sizes["relin_r2_share"] + m.sizes["rot_share_all"])
        m.sizes["server_key_storage"] = m.sizes["relin_key"] + m.sizes["rot_keys_all"]

    # ---------------- helpers ---------------------------------------------
    def _replicate(self, vec):
        vec = list(map(float, vec))
        reps = self.slots // len(vec)
        return vec * reps

    def _encrypt(self, vec, timer_key, site=None):
        with self.m.timed(timer_key):
            pt = self.cc.MakeCKKSPackedPlaintext(self._replicate(vec))
            ct = self.cc.Encrypt(self.pk, pt)
        if site is not None and self.integrity:
            with self.m.timed("integrity_site"):
                site.sent_digests.append(digest(ct))
        return ct

    def _log_received(self, cts):
        """Server side of I1: publish the hash of every ciphertext it aggregates."""
        if self.integrity:
            with self.m.timed("integrity_server"):
                self.transcript.extend(digest(ct) for ct in cts)

    def verify_transcript(self):
        """Each site checks its uploads appear exactly once before releasing a share."""
        if not self.integrity:
            return
        published = Counter(self.transcript)
        for site in self.sites:
            with self.m.timed("integrity_site"):
                bad = [d for d in site.sent_digests if published[d] != 1]
            if bad:
                raise IntegrityError(f"site {site.idx}: {len(bad)} ciphertext(s) missing or duplicated "
                                     f"in the server transcript; refusing to decrypt")

    def decrypt(self, ct, length, shares_from=None, timer_prefix="dec"):
        """Threshold decryption. Every site's flooded share goes to the analyst."""
        cc, m = self.cc, self.m
        self.verify_transcript()
        sites = self.sites if shares_from is None else shares_from
        shares = []
        for j, site in enumerate(sites):
            with m.timed(f"{timer_prefix}_partial_site"):
                if j == 0:
                    sh = cc.MultipartyDecryptLead([ct], site.kp.secretKey)[0]
                else:
                    sh = cc.MultipartyDecryptMain([ct], site.kp.secretKey)[0]
            shares.append(sh)
        m.sizes[f"{timer_prefix}_share"] = nbytes(shares[-1])
        with m.timed(f"{timer_prefix}_fusion_analyst"):
            pt = cc.MultipartyDecryptFusion(shares)
            pt.SetLength(length)
            vals = np.array(pt.GetRealPackedValue())
        m.counts["decryptions"] += 1
        return vals

    # ---------------- Round 1 / Mode A: pooled moments --------------------
    def run_mode_a(self):
        m = self.m
        # Scale moments (each <= N_b) to use the plaintext headroom q0/Delta fully:
        # slot value <= 2^(q0 - Delta_bits - 2). Larger slot values = more precision.
        self.mode_a_scale = 2.0 ** (self.ps.first_mod_bits - self.ps.scale_bits - 2) / N_BOUND
        cts = []
        for site in self.sites:
            mu = local_moments(site.X, site.y) * self.mode_a_scale
            cts.append(self._encrypt(mu, "modeA_encrypt_site", site))
        m.sizes["ciphertext_fresh"] = nbytes(cts[0])
        self._log_received(cts)
        with m.timed("modeA_aggregate_server"):
            agg = cts[0]
            for ct in cts[1:]:
                agg = self.cc.EvalAdd(agg, ct)
        raw = self.decrypt(agg, len(mu), timer_prefix="modeA_dec") / self.mode_a_scale
        self.moments_raw = raw                      # decrypted, before any rounding
        n, mean, std, ybar = moments_to_stats(raw)
        if n < MIN_CELL:
            raise RuntimeError("pooled cohort below the small-cell threshold; nothing released")
        # Release policy: publish N, means and SDs rounded to 4 significant digits
        self.released = {"N": float(np.round(n)), "mean": round_sig(mean), "std": round_sig(std),
                         "ybar": float(round_sig(np.array([ybar]))[0])}
        return self.released

    # ---------------- Round 2 / Mode B: encrypted regression --------------
    def encrypt_regression_inputs(self, mode, mean=None, std=None, n_total=None, eta=1.0 / D):
        """Each site encrypts -(eta/N) diag_k(G_j) for k<d and (eta/N) g_j."""
        denom = n_total if mode == "standardized" else N_BOUND
        self.eta, self.denom, self.mode = eta, denom, mode
        diag_cts = [[] for _ in range(D)]
        c_cts = []
        for site in self.sites:
            Z = design_matrix(site.X, mode, mean, std)
            G = Z.T @ Z
            g = Z.T @ site.y
            for k in range(D):
                diag = np.array([G[i, (i + k) % D] for i in range(D)])
                diag_cts[k].append(self._encrypt(-eta / denom * diag, "modeB_encrypt_site", site))
            c_cts.append(self._encrypt(eta / denom * g, "modeB_encrypt_site", site))
        self.m.sizes["modeB_site_upload"] = (D + 1) * nbytes(c_cts[0])
        return diag_cts, c_cts

    def _rotations(self, ct):
        """Return [rot_0(ct), ..., rot_{d-1}(ct)] (rot_k = left shift by k slots)."""
        cc = self.cc
        m_order = 2 * cc.GetRingDimension()

        def rot_many(src, ks):
            if self.hoisted:                      # share one digit decomposition
                pre = cc.EvalFastRotationPrecompute(src)
                return [cc.EvalFastRotation(src, k, m_order, pre) for k in ks]
            return [cc.EvalRotate(src, k) for k in ks]

        if self.ps.rot_strategy == "all":
            return [ct] + rot_many(ct, range(1, D))
        r1, r2, r4 = rot_many(ct, [1, 2, 4])
        r3, r5 = rot_many(r1, [2, 4])
        r6, = rot_many(r2, [4])
        r7, = rot_many(r3, [4])
        return [ct, r1, r2, r3, r4, r5, r6, r7]

    def refresh(self, ct):
        """Collective refresh = interactive multiparty bootstrapping (Mouchet et al.)."""
        cc, m = self.cc, self.m
        with m.timed("refresh_server_prep"):
            ct_in = cc.IntMPBootAdjustScale(ct)
            a = cc.IntMPBootRandomElementGen(self.pk)
            c1 = ct_in.Clone()
            c1.RemoveElement(0)
        shares = []
        for site in self.sites:
            with m.timed("refresh_share_site"):
                shares.append(cc.IntMPBootDecrypt(site.kp.secretKey, c1, a))
        m.sizes["refresh_share"] = sum(nbytes(x) for x in shares[0])
        m.sizes["refresh_c1_download"] = nbytes(c1)
        with m.timed("refresh_server_combine"):
            agg = cc.IntMPBootAdd(shares)
            out = cc.IntMPBootEncrypt(self.pk, agg, a, ct_in)
        m.counts["refreshes"] += 1
        return out

    def run_regression(self, diag_cts, c_cts, iters, trace=False):
        """Algorithm 1 of the Part 2 design."""
        cc, m = self.cc, self.m
        self._log_received([ct for col in diag_cts for ct in col] + list(c_cts))
        with m.timed("modeB_aggregate_server"):
            diag = []
            for k in range(D):
                acc = diag_cts[k][0]
                for ct in diag_cts[k][1:]:
                    acc = cc.EvalAdd(acc, ct)
                diag.append(acc)
            ones = cc.MakeCKKSPackedPlaintext(self._replicate([1.0] * D))
            diag[0] = cc.EvalAdd(diag[0], ones)          # M = I - eta*G~
            c = c_cts[0]
            for ct in c_cts[1:]:
                c = cc.EvalAdd(c, ct)
        beta = c                                          # beta_1 = c
        E = self.ps.epoch_len
        levels, path = [], []
        for t in range(1, iters):
            with m.timed("gd_iteration_server"):
                rots = self._rotations(beta)
                acc = cc.EvalMultNoRelin(diag[0], rots[0])
                for k in range(1, D):
                    acc = cc.EvalAdd(acc, cc.EvalMultNoRelin(diag[k], rots[k]))
                beta = cc.EvalAdd(cc.Rescale(cc.Relinearize(acc)), c)
            levels.append(beta.GetLevel())
            if trace:
                path.append(self.decrypt(beta, D, timer_prefix="trace_dec"))
            if t % E == 0 and t < iters - 1:
                beta = self.refresh(beta)
        self.level_trace = levels
        m.sizes["ciphertext_out"] = nbytes(beta)
        return (beta, path) if trace else beta
