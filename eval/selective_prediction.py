"""Selective-prediction / risk-coverage analysis.

The positive framing on the same data. Even though SAE features don't beat
input statistics for predicting forecast difficulty, the input-statistics
probe itself is a usable abstention signal: predicting only on windows where
P(hard) is low produces a lower mean CRPS than forecasting on everything.

Curve: for coverage c in [0.1, 1.0], sort test windows ascending by predicted
P(hard), retain the first c*N (predicted-easy), report mean CRPS on retained
with bootstrap 95% CI. Compared against:
  - Oracle: sort by TRUE CRPS (lower bound, perfect ranking)
  - No-abstention baseline (constant)

Headline number: AURC (area under the risk-coverage curve, lower is better)
and CRPS reduction at coverage 0.5 vs no abstention.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROBE_LABELS = {
    "pred_P1_InputStats":     ("P1 stats",       "#4c78a8"),
    "pred_P2_InputStats_Raw": ("P2 stats+raw",   "#f58518"),
    "pred_P3_InputStats_SAE": ("P3 stats+sae",   "#e45756"),
    "pred_P4_RawOnly":        ("P4 raw only",    "#72b7b2"),
    "pred_P5_SAEOnly":        ("P5 sae only",    "#54a24b"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_scores", default="activations/probe_scores.parquet")
    ap.add_argument("--metadata", default="activations/ETTh1_metadata.parquet")
    ap.add_argument("--out_dir", default="eval/results")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for p in (args.probe_scores, args.metadata):
        if not os.path.exists(p):
            sys.exit(f"[selective] missing: {p}. Run the full pipeline first.")
    os.makedirs(args.out_dir, exist_ok=True)

    scores = pd.read_parquet(args.probe_scores)
    meta = pd.read_parquet(args.metadata)
    test_meta = meta[meta["split"] == "test"][["start_ts", "crps_raw"]].rename(
        columns={"crps_raw": "crps"})
    df = scores.merge(test_meta, on="start_ts")
    if len(df) == 0:
        sys.exit("[selective] no overlap between probe_scores and test metadata.")

    probe_cols = [c for c in df.columns if c.startswith("pred_")]
    if not probe_cols:
        sys.exit("[selective] no pred_* columns in probe_scores.")

    n = len(df)
    crps_all = df["crps"].values.astype(float)
    mean_crps_all = float(crps_all.mean())
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 4)

    # Oracle: sort by TRUE CRPS ascending. Best possible ranking.
    sorted_truth = np.sort(crps_all)
    oracle_curve = np.array([
        sorted_truth[:max(1, int(round(c * n)))].mean() for c in coverages
    ])

    # Random baseline: averaged over many random orderings.
    rand_curves = []
    for _ in range(args.n_bootstrap):
        perm = rng.permutation(n)
        rand_crps = crps_all[perm]
        rand_curves.append([rand_crps[:max(1, int(round(c * n)))].mean()
                            for c in coverages])
    rand_curves = np.array(rand_curves)
    random_curve = rand_curves.mean(axis=0)

    results = {}
    for col in probe_cols:
        order = np.argsort(df[col].values)        # ascending P(hard)
        sorted_crps = crps_all[order]
        curve, lo, hi = [], [], []
        for c in coverages:
            k = max(1, int(round(c * n)))
            kept = sorted_crps[:k]
            curve.append(float(kept.mean()))
            boots = [kept[rng.integers(0, k, k)].mean()
                     for _ in range(args.n_bootstrap)]
            lo.append(float(np.percentile(boots, 2.5)))
            hi.append(float(np.percentile(boots, 97.5)))
        results[col] = {"curve": curve, "ci95_lower": lo, "ci95_upper": hi,
                        "aurc": float(np.trapezoid(curve, coverages))}

    oracle_aurc = float(np.trapezoid(oracle_curve, coverages))
    random_aurc = float(np.trapezoid(random_curve, coverages))

    # CRPS reduction at 50% coverage vs no-abstention baseline.
    i50 = int(np.argmin(np.abs(coverages - 0.5)))

    summary = {
        "n_test": n,
        "mean_crps_no_abstention": mean_crps_all,
        "coverages": coverages.tolist(),
        "oracle_curve": oracle_curve.tolist(),
        "oracle_aurc": oracle_aurc,
        "random_curve": random_curve.tolist(),
        "random_aurc": random_aurc,
        "probes": results,
        "at_coverage_0p5": {
            "no_abstention": mean_crps_all,
            "oracle": float(oracle_curve[i50]),
            "random": float(random_curve[i50]),
            **{col: {"mean_crps": results[col]["curve"][i50],
                     "reduction_pct": 100 * (mean_crps_all - results[col]["curve"][i50])
                                       / mean_crps_all}
               for col in probe_cols},
        },
    }
    with open(os.path.join(args.out_dir, "selective_prediction.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axhline(mean_crps_all, color="gray", linestyle=":",
               label=f"No abstention ({mean_crps_all:.3f})")
    ax.plot(coverages, random_curve, color="black", linestyle="--",
            label=f"Random (AURC {random_aurc:.3f})")
    ax.plot(coverages, oracle_curve, color="black", linestyle="-",
            linewidth=2, label=f"Oracle (AURC {oracle_aurc:.3f})")
    for col in probe_cols:
        lbl, c = PROBE_LABELS.get(col, (col, "purple"))
        ax.plot(coverages, results[col]["curve"], color=c, marker="o",
                markersize=4,
                label=f"{lbl} (AURC {results[col]['aurc']:.3f})")
        ax.fill_between(coverages, results[col]["ci95_lower"],
                        results[col]["ci95_upper"], alpha=0.12, color=c)
    ax.set_xlabel("Coverage (fraction of windows retained for forecasting)")
    ax.set_ylabel("Mean CRPS on retained windows (lower better)")
    ax.set_title("Selective prediction on ETTh1 — chronos-t5-small")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "risk_coverage.png")
    fig.savefig(out_png, dpi=150)

    # Console summary.
    print(f"n_test = {n}")
    print(f"No-abstention mean CRPS  = {mean_crps_all:.4f}")
    print(f"Oracle AURC              = {oracle_aurc:.4f}")
    print(f"Random AURC              = {random_aurc:.4f}")
    print("Probe AURCs (lower better):")
    for col in probe_cols:
        print(f"  {PROBE_LABELS.get(col, (col, ''))[0]:16s}  "
              f"AURC = {results[col]['aurc']:.4f}")
    print(f"\nAt 50% coverage:")
    for col in probe_cols:
        d = summary["at_coverage_0p5"][col]
        print(f"  {PROBE_LABELS.get(col, (col, ''))[0]:16s}  "
              f"mean CRPS = {d['mean_crps']:.4f}  "
              f"(ΔCRPS = {-d['reduction_pct']:+.1f}% vs no abstention)")
    print(f"  {'Oracle':16s}  mean CRPS = {oracle_curve[i50]:.4f}")
    print(f"  {'Random':16s}  mean CRPS = {random_curve[i50]:.4f}")
    print(f"\nSaved {out_png}\nSaved {os.path.join(args.out_dir, 'selective_prediction.json')}")


if __name__ == "__main__":
    main()
