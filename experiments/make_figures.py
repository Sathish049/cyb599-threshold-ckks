"""Build the report's figures and summary tables from results/results.json.

Usage:  python experiments/make_figures.py [results/results.json]
Writes figures to results/figures/*.pdf (+ .png) and a Markdown summary to
results/summary.md, so every number in the report can be traced to a file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COLORS = {"P1-compact": "#4C78A8", "P2-design": "#F58518", "P3-deep": "#54A24B"}
MB = 1e6


def mean_std(runs, getter):
    vals = np.array([getter(r) for r in runs], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def fig_time_breakdown(R):
    sets = list(R["param_sets"])
    parts = [("Site encryption", lambda t: t["modeB_encrypt_site"] + t["modeA_encrypt_site"]),
             ("Encrypted GD (server)", lambda t: t["gd_total_server"]),
             ("Collective refresh", lambda t: t["refresh_server"]),
             ("Threshold decryption", lambda t: t["modeA_partial_dec_site"] + t["modeA_fusion_analyst"]
              + t["modeB_partial_dec_site"] + t["modeB_fusion_analyst"]),
             ("Integrity check (I1)", lambda t: t.get("integrity_site", 0) + t.get("integrity_server", 0)),
             ("Key generation (one-time, per site)", lambda t: t["kg_site"])]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    cmap = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#B279A2", "#BAB0AC"]
    y = np.arange(len(sets))
    left = np.zeros(len(sets))
    for (label, f), c in zip(parts, cmap):
        vals = np.array([np.mean([f(r["times"]) for r in R["param_sets"][s]]) for s in sets])
        ax.barh(y, vals, left=left, color=c, label=label, height=0.55)
        left += vals
    for i, v in enumerate(left):
        ax.text(v + 0.3, i, f"{v:.1f} s", va="center", fontsize=8)
    ax.set_yticks(y, [f"{s}\n$N=2^{{{int(np.log2(R['param_sets'][s][0]['context']['ring_dim']))}}}$"
                      for s in sets])
    ax.invert_yaxis()
    ax.set_xlabel("time per party role (s), mean of 3 trials; parties simulated sequentially on a 2-vCPU VM")
    ax.set_xlim(0, left.max() * 1.12)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "time_breakdown")


def fig_convergence(R):
    C = R["convergence"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    it = np.arange(1, len(C["standardized"]["rel_err"]) + 1)
    ax.semilogy(it, C["standardized"]["rel_err"], color="#F58518",
                label=f"standardized ($\\kappa$={C['standardized']['kappa']:.1f})")
    ax.semilogy(it, C["public"]["rel_err"], color="#4C78A8",
                label=f"public bounds, Part 2 ($\\kappa$={C['public']['kappa']:.0f})")
    tr = C.get("encrypted_trace")
    if tr:
        ols = np.array(tr["trace"]["ols"])
        enc = [np.linalg.norm(np.array(b) - ols) / np.linalg.norm(ols) for b in tr["trace"]["enc"]]
        ax.semilogy(np.arange(2, len(enc) + 2), enc, "o", ms=2.5, color="#E45756", label="encrypted (P2)")
    ax.axvline(27, color="gray", lw=0.6, ls=":")
    ax.text(28, 0.5, "K=27", fontsize=7, color="gray")
    ax.set_xlabel("gradient-descent iteration")
    ax.set_ylabel(r"$\|\beta_t-\beta^\ast\|/\|\beta^\ast\|$")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    save(fig, "convergence")


def fig_scaling(R):
    S = [r for r in R.get("scaling", []) if r]
    if not S:
        return
    n = [r["num_sites"] for r in S]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(n, [r["times"]["kg_site"] * r["num_sites"] for r in S], "o-", label="key generation, all sites")
    ax.plot(n, [r["times"]["online_parallel_modeB"] for r in S], "s-", label="online latency (parallel)")
    ax.plot(n, [r["times"]["wall_clock_simulation"] for r in S], "^-", label="one-machine simulation")
    ax.set_xlabel("number of sites $n$")
    ax.set_ylabel("time (s)")
    ax.set_xticks(n)
    ax.legend(fontsize=7, frameon=False)
    save(fig, "scaling")


def summary(R):
    L = ["# Results summary (auto-generated)\n", f"Environment: {R['env']}\n", f"Config: {R['config']}\n"]
    L.append("\n## Parameter sets (mean ± sd over trials)\n")
    L.append("| set | N | log Q | levels | refreshes | keygen/site s | encrypt/site s | GD total s | "
             "GD iter s | refresh total s | decrypt s | integrity s | online s | sim wall s | peak MB | "
             "Mode A rel err | beta err vs pt-GD | beta rel err vs OLS |")
    L.append("|" + "---|" * 18)
    for s, runs in R["param_sets"].items():
        c = runs[0]["context"]
        ms = lambda g: "%.3g ± %.2g" % mean_std(runs, g)           # noqa: E731
        L.append(f"| {s} | {c['ring_dim']} | {c['log_q']:.0f} | {c['mult_depth']} | {runs[0]['counts'].get('refreshes', 0)} | "
                 f"{ms(lambda r: r['times']['kg_site'])} | {ms(lambda r: r['times']['modeB_encrypt_site'] + r['times']['modeA_encrypt_site'])} | "
                 f"{ms(lambda r: r['times']['gd_total_server'])} | {ms(lambda r: r['times']['gd_iter_mean'])} | "
                 f"{ms(lambda r: r['times']['refresh_server'] + r['counts'].get('refreshes', 0) * r['times']['refresh_site_per_refresh'])} | "
                 f"{ms(lambda r: r['times']['modeB_partial_dec_site'] + r['times']['modeB_fusion_analyst'])} | "
                 f"{ms(lambda r: r['times'].get('integrity_site', 0) + r['times'].get('integrity_server', 0))} | "
                 f"{ms(lambda r: r['times']['online_parallel_modeB'])} | {ms(lambda r: r['times']['wall_clock_simulation'])} | "
                 f"{ms(lambda r: r['peak_rss_mb'])} | {ms(lambda r: r['accuracy']['modeA_max_rel_err'])} | "
                 f"{ms(lambda r: r['accuracy']['beta_max_abs_err_vs_pt_gd'])} | {ms(lambda r: r['accuracy']['beta_rel_err_vs_ols'])} |")
    L.append("\n## Sizes (MB, first trial)\n")
    keys = ["public_key", "ciphertext_fresh", "ciphertext_out", "relin_key", "rot_keys_all",
            "site_keygen_upload", "modeB_site_upload", "server_key_storage", "refresh_share",
            "refresh_c1_download", "modeB_dec_share", "modeA_dec_share"]
    L.append("| set | " + " | ".join(keys) + " |")
    L.append("|" + "---|" * (len(keys) + 1))
    for s, runs in R["param_sets"].items():
        L.append(f"| {s} | " + " | ".join(f"{runs[0]['sizes'].get(k, 0) / MB:.2f}" for k in keys) + " |")
    L.append(f"\n## Baseline\n\nnumpy pipeline: {R['baseline']}\n")
    C = R["convergence"]
    L.append("\n## Convergence\n")
    for mode in ("standardized", "public"):
        e = C[mode]["rel_err"]
        L.append(f"- {mode}: kappa={C[mode]['kappa']:.1f}, rel err at K=27: {e[26]:.4f}, K=54: {e[53]:.5f}, K=120: {e[-1]:.5f}")
    for k in ("encrypted_k54", "encrypted_public_k27"):
        if C.get(k):
            a = C[k]["accuracy"]
            L.append(f"- {k}: beta rel err vs OLS {a['beta_rel_err_vs_ols']:.5f}, err vs pt-GD {a['beta_max_abs_err_vs_pt_gd']:.2e}, "
                     f"refreshes {C[k]['counts'].get('refreshes', 0)}, GD total {C[k]['times']['gd_total_server']:.2f}s")
    if R.get("scaling"):
        L.append("\n## Scaling (P2)\n\n| n | keygen/site s | site keygen upload MB | refresh total s | decrypt s | online s | sim wall s |")
        L.append("|---|---|---|---|---|---|---|")
        for r in R["scaling"]:
            if r:
                t = r["times"]
                L.append(f"| {r['num_sites']} | {t['kg_site']:.3f} | {r['sizes']['site_keygen_upload'] / MB:.1f} | "
                         f"{t['refresh_server'] + r['counts'].get('refreshes', 0) * t['refresh_site_per_refresh']:.3f} | "
                         f"{t['modeB_partial_dec_site'] + t['modeB_fusion_analyst']:.3f} | {t['online_parallel_modeB']:.2f} | {t['wall_clock_simulation']:.2f} |")
    if R.get("ablation"):
        L.append("\n## Ablations (P2)\n")
        ref = R["param_sets"]["P2-design"][0]
        for name, r in [("default (hoisted, all keys)", ref)] + list(R["ablation"].items()):
            if r:
                L.append(f"- {name}: GD iter {r['times']['gd_iter_mean']:.3f}s, rotation keys {r['sizes']['rot_keys_all'] / MB:.1f} MB, "
                         f"server key storage {r['sizes']['server_key_storage'] / MB:.1f} MB")
    (ROOT / "results" / "summary.md").write_text("\n".join(L) + "\n")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "results.json"
    if not path.exists():
        path = ROOT / "results" / "results_quick.json"
    R = json.loads(path.read_text())
    fig_time_breakdown(R)
    fig_convergence(R)
    fig_scaling(R)
    summary(R)
    print(f"Figures in {FIG.relative_to(ROOT)}, summary in results/summary.md (from {path.name})")


if __name__ == "__main__":
    main()
