import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
import scipy.signal
import scipy.stats
from statsmodels.tsa.stattools import acf, adfuller
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sae')))
from sae_model import TopKSAE

def compute_spectral_entropy(ts):
    f, Pxx = scipy.signal.welch(ts)
    if np.sum(Pxx) == 0:
        return 0.0
    Pxx = Pxx / np.sum(Pxx)
    return scipy.stats.entropy(Pxx)

def compute_input_stats(df_meta, context_length=512, season_length=24):
    """Eight classical features per window. The report claims eight; the prior
    version computed four. The missing ones (volatility, seasonal autocorr,
    trend slope, range) are precisely the ones a regime-shift detector would
    target, so they need to be in the baseline for an honest comparison."""
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    df_raw = pd.read_csv(url)
    ts_data = df_raw["OT"].values.astype(np.float64)

    stats = []
    for _, row in df_meta.iterrows():
        start = int(row["start_ts"])
        x = ts_data[start:start + context_length]
        n = len(x)

        var = float(np.var(x))
        volatility = float(np.mean(np.abs(np.diff(x)))) if n > 1 else 0.0
        acf_vals = acf(x, nlags=max(1, season_length), fft=False) if n > season_length else np.zeros(season_length + 1)
        lag1_acf = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
        seasonal_acf = float(acf_vals[season_length]) if len(acf_vals) > season_length else 0.0
        try:
            adf_p = float(adfuller(x, autolag="AIC")[1])
        except Exception:
            adf_p = 1.0
        entropy = compute_spectral_entropy(x)
        trend_slope = float(np.polyfit(np.arange(n), x, 1)[0]) if n > 1 else 0.0
        rng = float(x.max() - x.min())

        stats.append([var, volatility, lag1_acf, seasonal_acf,
                      adf_p, entropy, trend_slope, rng])
    return np.array(stats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="activations/ETTh1_metadata.parquet")
    parser.add_argument("--activations", type=str, default="activations/ETTh1_activations.safetensors")
    parser.add_argument("--sae_ckpt", type=str, default="sae/checkpoints/sae_topk_32.pt")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_hidden", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    args = parser.parse_args()

    print("Loading data...")
    df_meta = pd.read_parquet(args.metadata)
    tensors = load_file(args.activations)
    raw_acts = tensors["encoder_embeddings"] # (batch, seq, d_model)
    
    print("Computing input statistics...")
    input_stats = compute_input_stats(df_meta)
    
    print("Loading SAE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hard-fail if the checkpoint can't be loaded. The probe must NEVER silently
    # run on a random SAE -- the resulting numbers would be noise but look
    # identical to a real result in the JSON and the report.
    if not os.path.exists(args.sae_ckpt):
        sys.exit(f"[probe] SAE checkpoint '{args.sae_ckpt}' not found. "
                 f"Train the SAE first; refusing to probe with random weights.")
    state = torch.load(args.sae_ckpt, map_location=device)
    if "W_enc" not in state:
        sys.exit(f"[probe] '{args.sae_ckpt}' is not a TopKSAE checkpoint (no W_enc).")
    d_model_ckpt, d_hidden_ckpt = state["W_enc"].shape
    print(f"Auto-detected SAE dims from checkpoint: d_model={d_model_ckpt}, d_hidden={d_hidden_ckpt}")
    sae = TopKSAE(d_model=d_model_ckpt, d_hidden=d_hidden_ckpt, k=args.k).to(device)
    sae.load_state_dict(state)
    sae.eval()
    
    print("Aggregating activations...")
    # Aggregation: concat(mean, max, last-token) per feature
    raw_mean = raw_acts.mean(dim=1).numpy()
    raw_max = raw_acts.max(dim=1).values.numpy()
    raw_last = raw_acts[:, -1, :].numpy()
    raw_agg = np.concatenate([raw_mean, raw_max, raw_last], axis=1)
    
    sae_acts_list = []
    with torch.no_grad():
        for i in range(raw_acts.shape[0]):
            x = raw_acts[i:i+1].to(device).to(torch.float32)
            acts, _, _ = sae(x)
            sae_acts_list.append(acts.cpu())
    sae_acts = torch.cat(sae_acts_list, dim=0) # (batch, seq, d_hidden)
    
    sae_mean = sae_acts.mean(dim=1).numpy()
    sae_max = sae_acts.max(dim=1).values.numpy()
    sae_last = sae_acts[:, -1, :].numpy()
    sae_agg = np.concatenate([sae_mean, sae_max, sae_last], axis=1)
    
    # Label definition: Top 25% CRPS in test set is "hard"
    threshold = df_meta['crps_norm'].quantile(0.75)
    y = (df_meta['crps_norm'] >= threshold).astype(int).values
    
    train_mask = (df_meta['split'] == 'train').values
    test_mask = (df_meta['split'] == 'test').values
    
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        print("Not enough train/test split data. Need full extraction.")
        return
    
    print(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")
    
    y_train, y_test = y[train_mask], y[test_mask]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("Warning: Only one class present in train or test split. AUROC cannot be computed properly.")
        
    probes = {
        "P1_InputStats":     input_stats,
        "P2_InputStats_Raw": np.concatenate([input_stats, raw_agg], axis=1),
        "P3_InputStats_SAE": np.concatenate([input_stats, sae_agg], axis=1),
        # Diagnostic probes (not in headline table): isolate where signal lives.
        # If P4 ~ chance, raw activations carry no difficulty signal at all.
        # If P5 ~ chance, SAE features carry no difficulty signal at all.
        # If P4/P5 are non-trivial but P2/P3 still lose to P1, the problem is
        # input-stats DOMINATING the L1 logistic when concatenated.
        "P4_RawOnly":        raw_agg,
        "P5_SAEOnly":        sae_agg,
    }
    
    results = {}
    preds = {}

    # Inner CV uses TimeSeriesSplit so consecutive (overlapping) windows do not
    # leak across folds when picking C. The outer temporal/purge split already
    # protects the test set, but a shuffled inner CV would still bias the
    # chosen regularization toward overfitting.
    n_splits = max(2, min(5, int(np.bincount(y_train).min()) - 1, train_mask.sum() // 3))
    # Extended downward: with 12k SAE features and 483 train samples the prior
    # grid's lower bound (0.01) was still too lax -- CV had no choice but to
    # pick C=3.0 (top of grid) and overfit. 1e-4 ... 1.0 covers the regime
    # actually relevant to high-dim sparse-feature probes.
    C_grid = {"C": [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]}

    for name, X in probes.items():
        print(f"Training probe: {name} (features: {X.shape[1]})")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_mask])
        X_test_s = scaler.transform(X[test_mask])

        if len(np.unique(y_train)) < 2:
            preds[name] = np.zeros_like(y_test, dtype=float)
            results[name] = {"AUROC": 0.0, "95%_CI_lower": 0.0, "95%_CI_upper": 0.0}
            continue

        base = LogisticRegression(penalty="l1", solver="liblinear",
                                   class_weight="balanced", max_iter=2000)
        gs = GridSearchCV(base, C_grid, scoring="roc_auc",
                          cv=TimeSeriesSplit(n_splits=n_splits))
        gs.fit(X_train_s, y_train)
        preds[name] = gs.predict_proba(X_test_s)[:, 1]
        # Point AUROC on the actual test set (not a bootstrap mean) for honesty.
        point = (roc_auc_score(y_test, preds[name])
                 if len(np.unique(y_test)) > 1 else 0.0)
        results[name] = {"AUROC": point, "best_C": gs.best_params_["C"]}
        print(f"  {name} point AUROC = {point:.3f}  (C={gs.best_params_['C']})")

    # PAIRED bootstrap: resample test indices ONCE per iteration, reuse for all
    # probes and all deltas. This is the only way to get a CI on the headline
    # P3 - P2 delta, which neutralizes the dimensionality argument.
    rng = np.random.default_rng(42)
    names = list(probes.keys())
    boot = {n: [] for n in names}
    pairs = [("P2_InputStats_Raw", "P1_InputStats"),
             ("P3_InputStats_SAE", "P1_InputStats"),
             ("P3_InputStats_SAE", "P2_InputStats_Raw")]
    boot_delta = {f"{a}-{b}": [] for a, b in pairs}
    idx_all = np.arange(len(y_test))
    if len(np.unique(y_test)) > 1:
        for _ in range(2000):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            if len(np.unique(y_test[idx])) < 2:
                continue
            per = {n: roc_auc_score(y_test[idx], preds[n][idx]) for n in names}
            for n in names:
                boot[n].append(per[n])
            for a, b in pairs:
                boot_delta[f"{a}-{b}"].append(per[a] - per[b])

    def _ci(arr):
        if not arr:
            return (np.nan, np.nan)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    for n in names:
        lo, hi = _ci(boot[n])
        results[n]["95%_CI_lower"] = lo
        results[n]["95%_CI_upper"] = hi
        print(f"  {n} AUROC 95% CI: [{lo:.3f}, {hi:.3f}]")

    delta_raw = results["P2_InputStats_Raw"]["AUROC"] - results["P1_InputStats"]["AUROC"]
    delta_sae = results["P3_InputStats_SAE"]["AUROC"] - results["P1_InputStats"]["AUROC"]
    delta_sae_over_raw = results["P3_InputStats_SAE"]["AUROC"] - results["P2_InputStats_Raw"]["AUROC"]
    d_raw_ci = _ci(boot_delta["P2_InputStats_Raw-P1_InputStats"])
    d_sae_ci = _ci(boot_delta["P3_InputStats_SAE-P1_InputStats"])
    d_sor_ci = _ci(boot_delta["P3_InputStats_SAE-P2_InputStats_Raw"])
    print("\n--- Incremental Predictive Power (ΔAUROC, paired bootstrap) ---")
    print(f"Δ Raw - Stats : {delta_raw:+.3f}  95% CI [{d_raw_ci[0]:+.3f}, {d_raw_ci[1]:+.3f}]")
    print(f"Δ SAE - Stats : {delta_sae:+.3f}  95% CI [{d_sae_ci[0]:+.3f}, {d_sae_ci[1]:+.3f}]")
    print(f"Δ SAE - Raw   : {delta_sae_over_raw:+.3f}  95% CI [{d_sor_ci[0]:+.3f}, {d_sor_ci[1]:+.3f}]")
        
    df_test = df_meta[test_mask].copy()
    if preds:
        for name, p in preds.items():
            df_test[f"pred_{name}"] = p
    df_test.to_parquet("activations/probe_scores.parquet")
    print("\nSaved probe_scores.parquet")

    # Save probe results JSON
    import json
    os.makedirs("probing/results", exist_ok=True)
    
    # Calculate n_train, n_test, hard_fraction
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())
    hard_fraction = float(y[test_mask].mean()) if n_test > 0 else 0.0
    n_total = len(df_meta)
    
    final_results = {
        "n_total": n_total,
        "n_train": n_train,
        "n_test": n_test,
        "hard_fraction": hard_fraction,
        "P1_AUROC": results.get("P1_InputStats", {}).get("AUROC", 0.0),
        "P1_CI_lower": results.get("P1_InputStats", {}).get("95%_CI_lower", 0.0),
        "P1_CI_upper": results.get("P1_InputStats", {}).get("95%_CI_upper", 0.0),
        "P2_AUROC": results.get("P2_InputStats_Raw", {}).get("AUROC", 0.0),
        "P2_CI_lower": results.get("P2_InputStats_Raw", {}).get("95%_CI_lower", 0.0),
        "P2_CI_upper": results.get("P2_InputStats_Raw", {}).get("95%_CI_upper", 0.0),
        "P3_AUROC": results.get("P3_InputStats_SAE", {}).get("AUROC", 0.0),
        "P3_CI_lower": results.get("P3_InputStats_SAE", {}).get("95%_CI_lower", 0.0),
        "P3_CI_upper": results.get("P3_InputStats_SAE", {}).get("95%_CI_upper", 0.0),
        "delta_raw": float(delta_raw),
        "delta_raw_CI_lower": d_raw_ci[0],
        "delta_raw_CI_upper": d_raw_ci[1],
        "delta_sae": float(delta_sae),
        "delta_sae_CI_lower": d_sae_ci[0],
        "delta_sae_CI_upper": d_sae_ci[1],
        "delta_sae_over_raw": float(delta_sae_over_raw),
        "delta_sae_over_raw_CI_lower": d_sor_ci[0],
        "delta_sae_over_raw_CI_upper": d_sor_ci[1],
        # Diagnostic probes (not in headline table)
        "P4_RawOnly_AUROC": results.get("P4_RawOnly", {}).get("AUROC", 0.0),
        "P4_RawOnly_CI_lower": results.get("P4_RawOnly", {}).get("95%_CI_lower", 0.0),
        "P4_RawOnly_CI_upper": results.get("P4_RawOnly", {}).get("95%_CI_upper", 0.0),
        "P5_SAEOnly_AUROC": results.get("P5_SAEOnly", {}).get("AUROC", 0.0),
        "P5_SAEOnly_CI_lower": results.get("P5_SAEOnly", {}).get("95%_CI_lower", 0.0),
        "P5_SAEOnly_CI_upper": results.get("P5_SAEOnly", {}).get("95%_CI_upper", 0.0),
        "chosen_C": {k: v.get("best_C") for k, v in results.items() if "best_C" in v},
    }
    with open("probing/results/probe_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("Saved probing/results/probe_results.json")

if __name__ == "__main__":
    main()
