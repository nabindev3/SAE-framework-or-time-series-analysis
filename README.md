# Feature-Routed Time-Series Forecasting

Sparse-autoencoder features from a Time-Series Foundation Model (TSFM) as a
learned signal for forecast difficulty, routing, and abstention.

## Question

Do SAE features add predictive power for forecast difficulty **on top of** cheap
input statistics and raw activations — i.e., does the model's internal
representation know something about its own future error that the input doesn't
already reveal?

The headline metric is **incremental** AUROC with bootstrap CIs:

- `P1` = input-stats only (the baseline that matters)
- `P2` = input-stats + raw activations
- `P3` = input-stats + SAE features
- deltas: `P2 − P1`, `P3 − P1`, and `P3 − P2` (neutralizes the dimensionality
  argument: SAE vs. raw, both high-dim)

Even a rigorously-reported null result is a credible signal — sloppy 0.85
< rigorous honest 0.62 in the eyes of the people we want to impress.

## Repo layout

```
extract_activations.py   # encoder block-6 activations, CRPS@100, seasonal MASE, temporal split + purge
sae/sae_model.py         # TopK SAE (768 -> 6144, k=32, aux-k revival)
sae/train_sae.py         # trains the SAE on the TRAIN split only
probing/features.py      # input statistics + (mean,max,last) sequence pooling
probing/probe.py         # P1/P2/P3 probes, paired-bootstrap ΔAUROC -- refuses to run on unlabeled data
probing/visualize_features.py  # Figure 2: top difficulty features on real series
eval/cascade.py          # Target-tier: cost-CRPS Pareto for a feature-routed cheap/base cascade
eval/report_template.md  # 6-page workshop report skeleton
reproduce.sh             # one-command pipeline (smoke -> extract -> SAE -> probe -> figures)
_stale/                  # quarantined prototype artifacts -- see _stale/README.md, do not load
```

## Methodology constraints (load-bearing)

1. **Temporal train/test split with a purge gap** ≥ `context + horizon` between
   train and test. Sliding windows overlap; a random split inflates AUROC.
2. **CRPS labels normalized using train-split stats only** (full-dataset
   normalization leaks test info into the label).
3. **Seasonal-naive MASE** (m=24 for hourly data) — comparable to the Chronos
   paper; lag-1 naive is not.
4. **SAE trained on train-split tokens only**. Fitting it on test-window
   activations is an unsupervised form of leakage an interviewer will probe.
5. **Same (mean,max,last) pooling for raw and SAE** so the comparison is fair.

## Running

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # add `statsmodels` for real ADF (optional)
bash reproduce.sh                 # full pipeline
```

`reproduce.sh` runs: smoke test → full extraction → SAE train → probe →
feature visualizations. The Target-tier cascade requires a second extraction
with `--model amazon/chronos-t5-small` — see the trailing note in
`reproduce.sh`.

### Compute branch

- A100 / 4090 / 3090: `--model amazon/chronos-t5-base` (default).
- Colab / Kaggle free tier: drop to `amazon/chronos-t5-small` (60 M). The
  science is identical; state the swap in the writeup.

## Status

| Stage                                 | Code   | Run on full data |
|---------------------------------------|--------|------------------|
| Smoke test                            | ✅     | ✅ (one window)  |
| Activation extraction + labels        | ✅     | ❌ (gate: GPU)   |
| SAE training (train split)            | ✅     | ❌               |
| Difficulty probe (headline ΔAUROC)    | ✅     | ❌               |
| Feature visualization                 | ✅     | ❌               |
| Cascade Pareto (Target tier)          | ✅     | ❌ (needs 2nd extraction) |
| 6-page report                         | template | ❌            |

The probe carries a built-in guardrail: it refuses to run on metadata that
lacks `split` and a `crps_*` column, so it cannot silently produce a fake
result on the quarantined prototype cache.
