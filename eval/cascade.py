"""Feature-routed cascade: route between a cheap and an expensive TSFM using a
trained difficulty probe as the routing signal.

For each routing signal (a `pred_*` column in `probe_scores.parquet`) we sweep
the routing threshold τ in [0, 1] and plot the resulting Pareto curve on
(mean inference cost, mean CRPS). Compared against:
  - **always cheap**  : mean(crps_small), cost=cheap
  - **always base**   : mean(crps_base),  cost=base
  - **random routing**: average over permutations at each routing fraction
  - **oracle routing**: route to base the k windows where (crps_small −
                        crps_base) is largest — i.e. where base actually helps
                        most. This is the best any oracle-ranked router can do
                        for a given fraction routed.

A probe-driven Pareto point that sits BELOW the random curve and APPROACHES
the oracle curve is the real evidence that the probe's routing signal carries
deployable value.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _route_threshold(scores, tau, crps_cheap, crps_base, cost_cheap, cost_base):
    to_base = scores >= tau
    final = np.where(to_base, crps_base, crps_cheap)
    cost = np.where(to_base, cost_base, cost_cheap)
    return float(final.mean()), float(cost.mean()), float(to_base.mean())


def _route_random(crps_cheap, crps_base, cost_cheap, cost_base, n_trials=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(crps_cheap)
    fractions = np.linspace(0.0, 1.0, 21)
    curve = []
    for f in fractions:
        k = int(round(f * n))
        crpses, costs = [], []
        for _ in range(n_trials):
            idx = rng.choice(n, size=k, replace=False) if k > 0 else np.array([], dtype=int)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            crpses.append(np.where(mask, crps_base, crps_cheap).mean())
            costs.append(np.where(mask, cost_base, cost_cheap).mean())
        curve.append((float(np.mean(costs)), float(np.mean(crpses)), float(f)))
    return curve


def _route_oracle(crps_cheap, crps_base, cost_cheap, cost_base):
    # For each fraction f, route the top-(f*n) windows where base helps most.
    gap = crps_cheap - crps_base          # >0 means base is better here
    order = np.argsort(-gap)              # descending
    n = len(gap)
    fractions = np.linspace(0.0, 1.0, 21)
    curve = []
    for f in fractions:
        k = int(round(f * n))
        mask = np.zeros(n, dtype=bool)
        mask[order[:k]] = True
        crps = np.where(mask, crps_base, crps_cheap).mean()
        cost = np.where(mask, cost_base, cost_cheap).mean()
        curve.append((float(cost), float(crps), float(f)))
    return curve


def _probe_curve(scores, crps_cheap, crps_base, cost_cheap, cost_base, n_taus=41):
    taus = np.linspace(0.0, 1.0, n_taus)
    pts = []
    for tau in taus:
        crps, cost, frac = _route_threshold(scores, tau, crps_cheap, crps_base,
                                            cost_cheap, cost_base)
        pts.append({"tau": float(tau), "frac_to_base": frac,
                    "mean_cost": cost, "mean_crps": crps})
    return pts


def _dominating_points(pts, cheap_anchor, base_anchor):
    """Count probe-driven Pareto points strictly below the cheap↔base
    interpolation line at their cost. Real evidence the probe adds value."""
    c0, y0 = cheap_anchor
    c1, y1 = base_anchor
    dom = []
    for p in pts:
        c = p["mean_cost"]
        if not (c0 < c < c1):
            continue
        t = (c - c0) / (c1 - c0 + 1e-12)
        y_line = y0 + t * (y1 - y0)
        if p["mean_crps"] < y_line - 1e-9:
            dom.append(p)
    return dom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small_metadata", type=str, default="activations/ETTh1_metadata.parquet")
    p.add_argument("--base_metadata", type=str, default="activations_base/ETTh1_metadata.parquet")
    p.add_argument("--probe_scores", type=str, default="activations/probe_scores.parquet")
    p.add_argument("--score_cols", type=str, nargs="+",
                   default=["pred_P3_InputStats_SAE", "pred_P1_InputStats"],
                   help="Probe score columns to evaluate as routing signals.")
    p.add_argument("--cost_cheap", type=float, default=1.0)
    p.add_argument("--cost_base", type=float, default=5.0)
    p.add_argument("--output_dir", type=str, default="eval/results")
    p.add_argument("--n_random_trials", type=int, default=500)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for path in (args.small_metadata, args.base_metadata, args.probe_scores):
        if not os.path.exists(path):
            raise SystemExit(f"[cascade] missing input: {path}. "
                             "Run both extractions and probe.py first.")

    df_small = pd.read_parquet(args.small_metadata)[["window_id", "crps_raw"]].rename(
        columns={"crps_raw": "crps_small"})
    df_base = pd.read_parquet(args.base_metadata)[["window_id", "crps_raw"]].rename(
        columns={"crps_raw": "crps_base"})
    df_probe = pd.read_parquet(args.probe_scores)
    if "window_id" not in df_probe.columns:
        raise SystemExit("[cascade] probe_scores missing 'window_id' column.")

    available = [c for c in args.score_cols if c in df_probe.columns]
    missing = [c for c in args.score_cols if c not in df_probe.columns]
    if missing:
        print(f"[cascade] WARNING: missing score columns in probe_scores: {missing}")
    if not available:
        raise SystemExit("[cascade] no requested score columns present in probe_scores.")

    keep_cols = ["window_id"] + available
    df = df_probe[keep_cols].merge(df_small, on="window_id").merge(df_base, on="window_id")
    if len(df) == 0:
        raise SystemExit("[cascade] zero-row join — window_ids don't overlap.")
    n = len(df)
    crps_small = df["crps_small"].values.astype(float)
    crps_base = df["crps_base"].values.astype(float)
    print(f"[cascade] evaluating on {n} test windows")
    print(f"  mean CRPS small = {crps_small.mean():.4f}")
    print(f"  mean CRPS base  = {crps_base.mean():.4f}")
    print(f"  win rate base   = {(crps_base < crps_small).mean():.2%}  "
          f"(fraction of windows where base beats small)")

    cheap_anchor = (args.cost_cheap, float(crps_small.mean()))
    base_anchor = (args.cost_base,  float(crps_base.mean()))

    # Baselines
    random_curve = _route_random(crps_small, crps_base, args.cost_cheap,
                                 args.cost_base, n_trials=args.n_random_trials)
    oracle_curve = _route_oracle(crps_small, crps_base, args.cost_cheap, args.cost_base)

    # Probe-driven curves
    probe_pts = {}
    summary = {"n_windows": n,
               "always_cheap": {"mean_crps": cheap_anchor[1], "cost": cheap_anchor[0]},
               "always_base":  {"mean_crps": base_anchor[1],  "cost": base_anchor[0]},
               "win_rate_base": float((crps_base < crps_small).mean()),
               "random_curve": random_curve,
               "oracle_curve": oracle_curve,
               "probes": {}}
    for col in available:
        pts = _probe_curve(df[col].values, crps_small, crps_base,
                            args.cost_cheap, args.cost_base)
        dom = _dominating_points(pts, cheap_anchor, base_anchor)
        best_dom = min(dom, key=lambda p: p["mean_crps"]) if dom else None
        probe_pts[col] = pts
        summary["probes"][col] = {
            "frontier": pts,
            "n_dominating_points": len(dom),
            "best_dominating": best_dom,
        }
        print(f"  {col}: {len(dom)} Pareto-dominating points  "
              f"(best: {best_dom})" if dom else f"  {col}: 0 dominating points")

    # Save JSON
    with open(os.path.join(args.output_dir, "cascade_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Cheap-to-base interpolation line (linear ablation reference)
    ax.plot([cheap_anchor[0], base_anchor[0]],
            [cheap_anchor[1], base_anchor[1]],
            color="gray", linestyle=":", label="linear interp (random equiv.)")
    # Random and oracle
    rx, ry = [c for c, _, _ in random_curve], [y for _, y, _ in random_curve]
    ox, oy = [c for c, _, _ in oracle_curve], [y for _, y, _ in oracle_curve]
    ax.plot(rx, ry, color="#999", linestyle="--", linewidth=1.5,
            label=f"random routing (500-trial avg)")
    ax.plot(ox, oy, color="black", linewidth=2,
            label="oracle (best per-window choice)")

    colors = ["#4c78a8", "#e45756", "#54a24b", "#f58518", "#72b7b2"]
    for col, c in zip(available, colors):
        cx = [pt["mean_cost"] for pt in probe_pts[col]]
        cy = [pt["mean_crps"] for pt in probe_pts[col]]
        ax.plot(cx, cy, color=c, marker="o", markersize=4,
                label=f"routed by {col.replace('pred_', '')}")

    ax.scatter([cheap_anchor[0]], [cheap_anchor[1]], color="#4c78a8",
               s=80, zorder=6, edgecolor="black", label="always cheap")
    ax.scatter([base_anchor[0]], [base_anchor[1]], color="#e45756",
               s=80, zorder=6, edgecolor="black", label="always base")
    ax.set_xlabel(f"Mean inference cost  "
                  f"(cheap={args.cost_cheap}, base={args.cost_base})")
    ax.set_ylabel("Mean CRPS on test (lower better)")
    ax.set_title("Feature-routed cascade: chronos-t5-small  ↔  chronos-t5-base")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "pareto_frontier.png"), dpi=150)
    print(f"\nSaved {os.path.join(args.output_dir, 'pareto_frontier.png')}")
    print(f"Saved {os.path.join(args.output_dir, 'cascade_results.json')}")


if __name__ == "__main__":
    main()
