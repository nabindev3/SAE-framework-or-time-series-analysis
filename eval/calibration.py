"""Probe calibration: reliability diagram + Expected Calibration Error (ECE) +
Brier score for each probe.

A probe with high AUROC can still be miscalibrated — its predicted P(hard)
might not match the actual frequency of hard windows in each predicted-bin.
For routing/abstention deployment, calibration matters as much as ranking:
"the probe says 80% hard" should actually mean "80% of those windows are
hard". Class-balanced L1 logistic (used in probing/probe.py) is famously
NOT well-calibrated; this script quantifies how bad it is and produces the
reliability diagram.

Pure analysis on existing `probe_scores.parquet` — no model load, no GPU.
"""
import os
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


def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
    """Returns per-bin (mean_pred, mean_actual, count) plus ECE and Brier."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
    bin_pred, bin_actual, bin_count = [], [], []
    ece_terms = []
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            bin_pred.append(np.nan); bin_actual.append(np.nan); bin_count.append(0)
            continue
        p_pred = float(y_pred[mask].mean())
        p_act  = float(y_true[mask].mean())
        bin_pred.append(p_pred); bin_actual.append(p_act); bin_count.append(cnt)
        ece_terms.append((cnt / n) * abs(p_pred - p_act))
    ece = float(sum(ece_terms))
    brier = float(np.mean((y_pred - y_true) ** 2))
    return {"bin_centers": [(bins[b] + bins[b + 1]) / 2 for b in range(n_bins)],
            "bin_pred_mean": bin_pred, "bin_actual_freq": bin_actual,
            "bin_count": bin_count, "ece": ece, "brier": brier}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_scores", default="activations/probe_scores.parquet")
    ap.add_argument("--metadata", default="activations/ETTh1_metadata.parquet")
    ap.add_argument("--hard_quantile", type=float, default=0.85,
                    help="Match probe.py's quantile so the y_hard label aligns.")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--out_dir", default="eval/results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scores = pd.read_parquet(args.probe_scores)
    meta = pd.read_parquet(args.metadata)
    label_col = "crps_norm" if "crps_norm" in meta.columns else "crps_raw"
    # Train-set threshold (matches probe.py and §4.2).
    tr = meta["split"].values == "train"
    thr = np.quantile(meta.loc[tr, label_col].values, args.hard_quantile)

    # probe_scores already carries crps_raw/crps_norm (copied from meta in probe.py).
    # Just use it directly — merging would collide column names and break the lookup.
    if label_col not in scores.columns:
        raise SystemExit(f"[calibration] {label_col} missing from probe_scores; "
                         "re-run probing/probe.py.")
    df = scores
    y = (df[label_col].values >= thr).astype(int)
    print(f"Test windows: {len(df)}   hard_fraction = {y.mean():.3f}")

    probe_cols = [c for c in df.columns if c.startswith("pred_")]
    if not probe_cols:
        raise SystemExit("No pred_* columns in probe_scores.")

    summary = {"n_test": len(df), "hard_fraction": float(y.mean()),
               "hard_quantile": args.hard_quantile, "thr_crps_norm": float(thr),
               "probes": {}}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls=":", label="perfect calibration")
    print(f"\n{'probe':<24}  {'ECE':>7}  {'Brier':>7}")
    for col in probe_cols:
        cal = compute_calibration(y, df[col].values, n_bins=args.n_bins)
        summary["probes"][col] = cal
        lbl, c = PROBE_LABELS.get(col, (col, "purple"))
        ax.plot(cal["bin_pred_mean"], cal["bin_actual_freq"],
                color=c, marker="o", markersize=5,
                label=f"{lbl}  (ECE {cal['ece']:.3f}, Brier {cal['brier']:.3f})")
        print(f"{lbl:<24}  {cal['ece']:>7.4f}  {cal['brier']:>7.4f}")

    ax.set_xlabel("predicted P(hard)")
    ax.set_ylabel("actual hard frequency in bin")
    ax.set_title(f"Reliability diagram — chronos-t5-small probes on ETTh1 test "
                 f"(n={len(df)}, hard_frac {y.mean():.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "reliability_diagram.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved {out_png}")

    with open(os.path.join(args.out_dir, "calibration_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'calibration_results.json')}")


if __name__ == "__main__":
    main()
