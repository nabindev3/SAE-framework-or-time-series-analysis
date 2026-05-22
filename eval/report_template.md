# Sparse-Autoencoder Features from a Time-Series Foundation Model Predict Forecast Difficulty

*Workshop-style preliminary report. ~6 pages. Fill bracketed slots from `probing/results/` and `eval/results/`.*

## Abstract
We train a TopK sparse autoencoder on Chronos-T5 encoder activations and ask
whether the discovered features predict a forecast's own difficulty *beyond*
cheap input statistics and raw activations. On ETTh1, [P3−P1 ΔAUROC = X, 95% CI
(a,b)]; [SAE vs raw: P3−P2 ΔAUROC = Y]. [State honestly: positive / null.]

## 1. Introduction
- TSFMs are deployed black-box; knowing *when not to trust a forecast* has direct value (energy, finance, ops triage).
- Question: do model internals encode self-difficulty not trivially present in the input?
- Contribution: an *incremental*, leakage-controlled probe; [optional: a one-point feature-routed cascade].

## 2. Related work
- Mishra (2026): SAEs on Chronos, causal change-detection features. *(verify citation before submission)*
- TimeSAE (2026): SAE-based black-box TS explanation under shift. *(verify)*
- Chronos (Ansari et al.): tokenized T5 forecaster; benchmark protocol.
- Distinction: we target **label-free inference-time difficulty prediction / routing**, not causal ablation or post-hoc explanation.

## 3. Method
- Model: amazon/chronos-t5-[base|small]; hook encoder block 6, (B, 513, 768).
- Labels: CRPS @100 samples; seasonal-naive MASE (m=24); difficulty = top-15% train-normalized CRPS.
- SAE: TopK, 768→6144 (8×), k=32, aux-k revival; trained on **train-split tokens only**.
- Probe: L1 logistic, time-series-CV C, temporal train/test split with purge gap; concat(mean,max,last) pooling for raw and SAE alike.
- Metric: paired bootstrap ΔAUROC (P2−P1, P3−P1, P3−P2), 95% CI.

## 4. Experiments
- Setup: ETTh1, context 512, horizon 96, stride 24; n_train=[ ], n_test=[ ].
- Table 1: AUROC ± CI for P1/P2/P3. Figure 1: `probing/results/auroc.png`.
- Figure 2: top-5 difficulty features on real series — `probing/results/features/`.
- [Optional] Figure 3: cascade Pareto — `eval/results/cascade_pareto.png`; [n dominating points].

## 5. Limitations
- Single series / single model family; thin test set → wide CIs.
- Probes encoded context, not decoder sampling dynamics.
- ADF replaced by variance-ratio proxy if statsmodels absent.
- [If null:] SAE features are interpretable but not more predictive than raw — reported honestly.

## 6. Future work
Multi-model feature alignment (TimesFM/Moirai), cross-domain transfer, full
implemented cascade with end-to-end cost accounting, feature steering/abstention.
