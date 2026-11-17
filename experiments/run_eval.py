"""Evaluation harness: every number and figure in the report comes from here.

Usage
-----
    python experiments/run_eval.py              # full evaluation (~10-15 min, ~4 GB RAM peak)
    python experiments/run_eval.py --quick      # P1 + P2, 1 trial, no scaling/ablation (~1 min)

Each configuration runs in its own subprocess so peak memory is measured per
configuration and one configuration cannot affect another's timings.
Results are written to results/*.json; experiments/make_figures.py turns them
into the figures and tables used in the report.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tckks import baseline as B                          # noqa: E402
from tckks.data import make_cohort                       # noqa: E402
from tckks.params import PARAM_SETS, D, K_DEFAULT        # noqa: E402
from tckks.protocol import ThresholdCKKS                 # noqa: E402

RESULTS = ROOT / "results"


# --------------------------------------------------------------------------
# One protocol run
# --------------------------------------------------------------------------
def run_once(set_name, num_sites, n_per_site, iters, mode, seed, hoisted=True, rot_strategy=None,
             trace=False):
    ps = PARAM_SETS[set_name]
    if rot_strategy:
        ps = dataclasses.replace(ps, rot_strategy=rot_strategy)
    sites = make_cohort(num_sites, n_per_site, seed=seed)

    t_all = time.perf_counter()
    P = ThresholdCKKS(ps, sites, hoisted=hoisted)
    P.keygen()
    rel = P.run_mode_a()
    if mode == "standardized":
        dc, cc = P.encrypt_regression_inputs("standardized", rel["mean"], rel["std"], rel["N"])
    else:  # the Part 2 design: public-bound scaling, no pooled statistics used
        dc, cc = P.encrypt_regression_inputs("public")
    out = P.run_regression(dc, cc, iters, trace=trace)
    beta_ct, enc_path = out if trace else (out, None)
    beta = P.decrypt(beta_ct, D, timer_prefix="modeB_dec")
    wall = time.perf_counter() - t_all

    # ---- plaintext references (non-private baseline) ----
    truth_m = B.pooled_moments(sites)
    if mode == "standardized":
        G, g = B.gram_terms(sites, "standardized", rel["mean"], rel["std"])
        denom = rel["N"]
    else:
        G, g = B.gram_terms(sites, "public")
        denom = P.denom
    beta_gd, pt_path = B.gradient_descent(G / denom, g / denom, 1.0 / D, iters, return_path=True)
    beta_ols = np.linalg.solve(G, g)

    m = P.m
    n = P.n_parties
    T = lambda k: m.total(k)                                     # noqa: E731
    per_site = lambda k: T(k) / n                                # noqa: E731
    res = {
        "set": set_name, "num_sites": n, "n_per_site": n_per_site, "iters": iters, "mode": mode,
        "seed": seed, "hoisted": hoisted, "rot_strategy": ps.rot_strategy,
        "context": P.info,
        "times": {
            "kg_site": per_site("kg_pk_site") + per_site("kg_relin_r1_site")
            + per_site("kg_relin_r2_site") + per_site("kg_rot_site"),
            "kg_server": T("kg_server_combine"),
            "modeA_encrypt_site": per_site("modeA_encrypt_site"),
            "modeA_aggregate_server": T("modeA_aggregate_server"),
            "modeA_partial_dec_site": per_site("modeA_dec_partial_site"),
            "modeA_fusion_analyst": T("modeA_dec_fusion_analyst"),
            "modeB_encrypt_site": per_site("modeB_encrypt_site"),
            "modeB_aggregate_server": T("modeB_aggregate_server"),
            "gd_total_server": T("gd_iteration_server"),
            "gd_iter_mean": float(np.mean(m.times["gd_iteration_server"])),
            "refresh_server": T("refresh_server_prep") + T("refresh_server_combine"),
            "refresh_site_per_refresh": (float(np.mean(m.times["refresh_share_site"]))
                                         if m.times.get("refresh_share_site") else 0.0),
            "modeB_partial_dec_site": per_site("modeB_dec_partial_site"),
            "modeB_fusion_analyst": T("modeB_dec_fusion_analyst"),
            "integrity_site": per_site("integrity_site"),
            "integrity_server": T("integrity_server"),
            "wall_clock_simulation": wall,
        },
        "sizes": dict(m.sizes),
        "counts": dict(m.counts),
        "level_trace": P.level_trace,
        "accuracy": {
            "modeA_max_rel_err": float(np.max(np.abs(P.moments_raw - truth_m) / np.abs(truth_m))),
            "beta_enc": beta.tolist(), "beta_pt_gd": beta_gd.tolist(), "beta_ols": beta_ols.tolist(),
            "beta_max_abs_err_vs_pt_gd": float(np.max(np.abs(beta - beta_gd))),
            "beta_rel_err_vs_ols": float(np.linalg.norm(beta - beta_ols) / np.linalg.norm(beta_ols)),
            "pt_gd_rel_err_vs_ols": float(np.linalg.norm(beta_gd - beta_ols) / np.linalg.norm(beta_ols)),
        },
    }
    t = res["times"]
    # Online latency if parties run in parallel on their own machines (network excluded):
    # one site's encryption + server work + per-refresh site share + one partial decryption + fusion
    t["online_parallel_modeB"] = (t["modeB_encrypt_site"] + t["modeB_aggregate_server"] + t["gd_total_server"]
                                  + t["refresh_server"] + m.counts.get("refreshes", 0) * t["refresh_site_per_refresh"]
                                  + t["modeB_partial_dec_site"] + t["modeB_fusion_analyst"]
                                  + t["integrity_site"] + t["integrity_server"])
    t["online_parallel_modeA"] = (t["modeA_encrypt_site"] + t["modeA_aggregate_server"]
                                  + t["modeA_partial_dec_site"] + t["modeA_fusion_analyst"])
    if trace:
        res["trace"] = {"enc": [p.tolist() for p in enc_path], "pt": [p.tolist() for p in pt_path],
                        "ols": beta_ols.tolist()}
    return res


def baseline_timing(num_sites, n_per_site, iters, seed, repeats=20):
    """Wall-clock of the same pipeline on pooled plaintext (numpy)."""
    sites = make_cohort(num_sites, n_per_site, seed=seed)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        n, mean, std, _ = B.moments_to_stats(B.pooled_moments(sites))
        G, g = B.gram_terms(sites, "standardized", mean, std)
        B.gradient_descent(G / n, g / n, 1.0 / D, iters)
        np.linalg.solve(G, g)
        ts.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(ts)), "min_s": float(np.min(ts)), "repeats": repeats}


def plaintext_convergence(num_sites, n_per_site, seed, iters=120):
    sites = make_cohort(num_sites, n_per_site, seed=seed)
    n, mean, std, _ = B.moments_to_stats(B.pooled_moments(sites))
    out = {}
    for mode in ("standardized", "public"):
        G, g = (B.gram_terms(sites, "standardized", mean, std) if mode == "standardized"
                else B.gram_terms(sites, "public"))
        denom = n if mode == "standardized" else 2 ** 16
        ols = np.linalg.solve(G, g)
        _, path = B.gradient_descent(G / denom, g / denom, 1.0 / D, iters, return_path=True)
        ev = np.linalg.eigvalsh(G / denom)
        out[mode] = {"rel_err": [float(np.linalg.norm(p - ols) / np.linalg.norm(ols)) for p in path],
                     "kappa": float(ev[-1] / ev[0]), "lambda_max": float(ev[-1]), "lambda_min": float(ev[0])}
    return out


# --------------------------------------------------------------------------
# Subprocess worker (isolates memory and timing per configuration)
# --------------------------------------------------------------------------
def worker(args):
    cfg = json.loads(args.worker)
    res = run_once(**cfg)
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("@@RESULT@@" + json.dumps(res))


def spawn(cfg, label):
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, __file__, "--worker", json.dumps(cfg)],
                          capture_output=True, text=True)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
    if proc.returncode != 0 or line is None:
        print(f"  [FAILED] {label} (exit {proc.returncode})\n{proc.stderr[-2000:]}")
        return None
    res = json.loads(line[len("@@RESULT@@"):])
    acc = res["accuracy"]
    print(f"  {label:<44} {time.perf_counter() - t0:6.1f}s  "
          f"beta err vs pt-GD {acc['beta_max_abs_err_vs_pt_gd']:.1e}  "
          f"peak {res['peak_rss_mb']:.0f} MB")
    return res


def env_info():
    import openfhe
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "processor": platform.processor() or platform.machine(),
            "openfhe_python": getattr(openfhe, "__version__", "1.6.1.0.24.4"),
            "numpy": np.__version__, "date": time.strftime("%Y-%m-%d %H:%M")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="P1+P2, one trial, skip scaling/ablation")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--sites", type=int, default=3)
    ap.add_argument("--n-per-site", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    RESULTS.mkdir(exist_ok=True)
    trials = 1 if args.quick else args.trials
    sets = ["P1-compact", "P2-design"] if args.quick else list(PARAM_SETS)
    base = dict(num_sites=args.sites, n_per_site=args.n_per_site, iters=K_DEFAULT, mode="standardized")
    out = {"env": env_info(), "config": {**base, "trials": trials}}

    print("[1/5] Parameter sets (correctness, accuracy, time, sizes)")
    out["param_sets"] = {}
    for s in sets:
        runs = [spawn({**base, "set_name": s, "seed": args.seed + t}, f"{s} trial {t + 1}")
                for t in range(trials)]
        out["param_sets"][s] = [r for r in runs if r]

    print("[2/5] Non-private baseline")
    out["baseline"] = baseline_timing(args.sites, args.n_per_site, K_DEFAULT, args.seed)
    print(f"  numpy pipeline median {out['baseline']['median_s'] * 1e3:.2f} ms")

    print("[3/5] Convergence (plaintext curves + encrypted trace on P2)")
    out["convergence"] = plaintext_convergence(args.sites, args.n_per_site, args.seed)
    if not args.quick:
        out["convergence"]["encrypted_trace"] = spawn(
            {**base, "set_name": "P2-design", "seed": args.seed, "trace": True}, "P2 encrypted trace (K=27)")
        out["convergence"]["encrypted_k54"] = spawn(
            {**base, "set_name": "P2-design", "seed": args.seed, "iters": 54}, "P2 standardized K=54")
        out["convergence"]["encrypted_public_k27"] = spawn(
            {**base, "set_name": "P2-design", "seed": args.seed, "mode": "public"}, "P2 Part-2 public scaling K=27")

    if not args.quick:
        print("[4/5] Scaling with number of sites (P2)")
        out["scaling"] = [spawn({**base, "set_name": "P2-design", "seed": args.seed, "num_sites": n},
                                f"P2 with {n} sites") for n in (3, 5, 10)]
        print("[5/5] Ablations (P2): hoisted rotations, rotation-key strategy")
        out["ablation"] = {
            "no_hoist": spawn({**base, "set_name": "P2-design", "seed": args.seed, "hoisted": False},
                              "P2 without hoisted rotations"),
            "pow2_keys": spawn({**base, "set_name": "P2-design", "seed": args.seed, "rot_strategy": "pow2"},
                               "P2 with power-of-two rotation keys"),
        }

    path = RESULTS / ("results_quick.json" if args.quick else "results.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
