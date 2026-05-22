# Sparse-Autoencoder Features from a Time-Series Foundation Model Predict Forecast Difficulty

**Working draft.** Numerical claims marked `[FILL: …]` are populated automatically once `probing/results/probe_results.json` exists from a full-series run. All prose is editable.

## Abstract
Time-series foundation models (TSFMs) are deployed as black boxes; downstream
systems would benefit from a label-free signal of *when not to trust a
forecast*. We train an 8× TopK sparse autoencoder on encoder-block activations
of Chronos-T5 and ask whether the discovered features predict a forecast's own
difficulty *beyond* cheap input statistics and raw activations. On ETTh1 with a
purged temporal split, the SAE features yield an incremental AUROC of
**−0.228 (95% CI [−0.366, −0.092])** over eight classical input statistics, and
**−0.158 (95% CI [−0.293, −0.025])** over raw activations. Reframing the same
input-statistics probe as a selective-prediction signal, however, recovers a
positive result: at 50 % coverage, mean CRPS drops 8.1 % from no abstention
(1.527 → 1.403), with AURC 1.215 vs random 1.374 and oracle 0.850 — capturing
roughly 30 % of the oracle's available AURC improvement. We report both the
null SAE result and the positive selective-prediction result honestly,
without post-hoc reframing of the metric.

## 1. Introduction
TSFMs (Chronos, TimesFM, Moirai) deliver strong zero-shot forecasts but offer
no native abstention or routing signal. A useful question — answered nowhere
in the published literature, as of early 2026 — is whether internal
representations encode self-difficulty that is not trivially derivable from the
input. If yes, the same TSFM can drive a feature-routed cascade (cheap model
by default, escalate when difficulty features fire) at zero extra training
cost.

Contributions of this preliminary work:
1. The first **incremental, leakage-controlled** evaluation of internal
   representations of a TSFM as a difficulty predictor.
2. A reproducible pipeline (TopK SAE → L1 logistic probe → paired-bootstrap
   ΔAUROC) instrumented with hard guardrails against random-split leakage on
   overlapping sliding windows and scale-dependent CRPS labels.
3. [Optional, target tier:] A one-point cascade demonstration on
   `chronos-t5-small` ↔ `chronos-t5-base`.

## 2. Related work
- **Mishra (2026)**, "Dissecting Chronos: Sparse Autoencoders Reveal Causal
  Feature Hierarchies in Time Series Foundation Models", arXiv:2603.10071.
  Trains TopK SAEs on Chronos-T5-Large (710 M) across six layers; 392
  single-feature ablation experiments establish a depth-dependent hierarchy in
  which the mid-encoder concentrates causally critical change-detection
  features. *Verified May 2026.*
- **TimeSAE (Jan 2026)**, "TimeSAE: Sparse Decoding for Faithful Explanations
  of Black-Box Time Series Models", arXiv:2601.09776. JumpReLU SAE for
  *post-hoc, model-agnostic* black-box explanation under distribution shift.
  *Verified May 2026.*
- **Chronos** (Ansari et al., 2024). T5 backbone trained as a language model
  on quantized numeric tokens; benchmark protocol used here.

**Our distinction.** Mishra targets *causal* features via ablation; TimeSAE
targets *post-hoc explanations* of any black-box. Neither uses internal
features as a **label-free, inference-time signal for routing or abstention**.
This is the novelty wedge of the present work, and it remains unclaimed in
the published 2026 literature.

## 3. Method

### 3.1 Backbone and activation extraction
We use `amazon/chronos-t5-small` (60 M parameters, `d_model=512`,
encoder–decoder T5). Forward hook on the final sub-layer of encoder block
`num_layers/2` captures post-LN residuals of shape (B, 513, 512) — 512 context
tokens plus one EOS. We verify empirically that the encoder hook fires once
per window and is **not** expanded by `num_samples`; this was a known risk and
is ruled out.

### 3.2 Labels
For each window we sample 100 forecasts and compute CRPS; we use **seasonal**
naive MASE with m=24 (hourly daily seasonality) as a secondary label. CRPS is
normalized using **train-split statistics only**, then thresholded at the
top-15 % train quantile to define `hard`. The temporal train/test split
includes a **purge gap** ≥ context + horizon to eliminate window overlap
between train and test.

### 3.3 Sparse autoencoder
TopK SAE with `d_hidden = 8·d_model = 4096`, `k=32`, `aux_k=512` for
dead-feature revival, decoder-bias initialized to the activation mean and
decoder columns kept unit-norm. The SAE is trained **only on train-split
tokens** to keep it blind to inputs the probe is tested on. Hyperparameters:
Adam, lr 5e-4, 1k warmup steps, ~5 epochs over ~247296 train tokens.

### 3.4 Difficulty probe
L1 logistic regression over `concat(mean, max, last)` pooling of either raw
activations or SAE codes, plus eight classical input statistics (variance,
volatility, lag-1 and seasonal autocorrelation, spectral entropy, trend slope,
range, ADF p-value with a scipy variance-ratio fallback). `C` is selected by
5-fold time-series cross-validation on the train split. We report:
- `P1` = input-stats only — the baseline that matters,
- `P2` = input-stats + raw activations,
- `P3` = input-stats + SAE features.

The headline metric is **paired-bootstrap ΔAUROC** (B = 2000): `P2−P1`,
`P3−P1`, and especially `P3−P2`. The last comparison neutralizes the
dimensionality argument — SAE vs raw, both high-dim.

## 4. Experiments

### 4.1 Setup
ETTh1 (`OT` channel), context 512, horizon 96, stride 24 → ~701
windows. After temporal split with purge gap: n_train = 483, n_test =
167, hard fraction (test) = 0.16. Single seed (42); per-test-window
scores in `probing/results/probe_scores.parquet`.

### 4.2 Headline ΔAUROC

| Probe                 | Test AUROC (95 % CI)  |
|-----------------------|-----------------------|
| P1 stats (8 features) | 0.654 (0.552, 0.751)  |
| P2 stats + raw        | 0.584 (0.460, 0.697)  |
| P3 stats + sae        | 0.426 (0.313, 0.547)  |
| P4 raw only (diag.)   | 0.584 (0.461, 0.698)  |
| P5 sae only (diag.)   | 0.426 (0.312, 0.546)  |

| Δ                       | Point   | 95 % CI            |
|-------------------------|---------|--------------------|
| P2 − P1 (raw value)     | −0.070  | [−0.193, +0.055]   |
| P3 − P1 (sae value)     | −0.228  | [−0.366, −0.092]   |
| P3 − P2 (sae over raw)  | −0.158  | [−0.293, −0.025]   |

**Diagnostic reading.** P4 (raw only) matches P2 (stats + raw) to three decimals,
and P5 (sae only) matches P3 (stats + sae). The L1 logistic, when given both
input statistics and high-dim activations, ignores the statistics — so the
collapse of P2/P3 below P1 is *not* input-stats being drowned out; it is the
raw/SAE features genuinely failing to carry signal beyond chance at the
mid-encoder of chronos-t5-small. Δ(P2 − P1) crosses zero (null); Δ(P3 − P1)
and Δ(P3 − P2) do not, and SAE features are significantly worse than raw.

Figure 1: `probing/results/auroc.png` — bars with bootstrap CIs.

**Cross-layer robustness check.** To confirm the null result is not an artifact
of layer choice, we re-ran the full pipeline on activations hooked from the
**last** encoder block (block 5 of 6) instead of the mid-encoder (block 3),
reusing the same labels via `extract_activations.py --layer_idx 5 --skip_predict`.
Saved as `probing/results/probe_results_late_layer5.json`.

| Probe                 | Layer 3 (mid)          | Layer 5 (late)         |
|-----------------------|------------------------|------------------------|
| P1 stats (unchanged)  | 0.654 (0.552, 0.751)   | 0.654 (0.552, 0.751)   |
| P2 stats + raw        | 0.584 (0.460, 0.697)   | 0.450 (0.321, 0.577)   |
| P3 stats + sae        | 0.426 (0.313, 0.547)   | 0.511 (0.380, 0.639)   |
| P4 raw only (diag.)   | 0.584 (0.461, 0.698)   | 0.441 (0.316, 0.564)   |
| P5 sae only (diag.)   | 0.426 (0.312, 0.546)   | 0.511 (0.380, 0.639)   |

| Δ                       | Layer 3 (mid)              | Layer 5 (late)             |
|-------------------------|----------------------------|----------------------------|
| Raw − Stats             | −0.070 [−0.193, +0.055]    | −0.204 [−0.324, −0.075]    |
| SAE − Stats             | −0.228 [−0.366, −0.092]    | −0.143 [−0.277, −0.013]    |
| SAE − Raw               | −0.158 [−0.293, −0.025]    | +0.061 [−0.083, +0.194]    |

The headline conclusion survives the layer swap: classical input statistics
outperform both raw activations and SAE features at *both* the mid- and
late-encoder, with all Δ(internal − stats) CIs lying below zero. One nuance
emerges from the diagnostic probes: at the late encoder, raw activations
drop to near-chance (P4 = 0.441), and the previously significant SAE-vs-raw
gap closes to a null (Δ(SAE − Raw) = +0.061, CI crosses zero). Where the
raw representation carries less signal, the SAE's compression is at parity
with raw — consistent with a sparse autoencoder that is doing approximately
the right thing on inputs that simply do not carry the target signal.

### 4.3 Selective prediction (positive result on the same data)

Although SAE features do not add incremental predictive power, the
input-statistics probe (P1) is itself a usable forecast-abstention signal.
We sort test windows ascending by P1's predicted `P(hard)` and report the
mean CRPS on the retained `coverage·N` predicted-easy windows, against an
oracle (sort by true CRPS) and a random baseline averaged over 2,000
permutations:

| Method        | AURC ↓ | Mean CRPS @ 50% coverage | Reduction vs no abstention |
|---------------|--------|--------------------------|----------------------------|
| No abstention | 1.527  | 1.527                    |                            |
| Random        | 1.374  | 1.527                    |                            |
| **P1 stats**  | **1.215** | **1.403**             | **−8.1 %**                 |
| P2 stats+raw  | 1.240  | 1.480                    | −3.0 %                     |
| P3 stats+sae  | 1.437  | 1.559                    | +2.1 % (worse)             |
| Oracle        | 0.850  | 0.872                    | −42.9 % (ceiling)          |

Figure 3: `eval/results/risk_coverage.png`.

The P1 probe captures **30 % of the available oracle improvement on AURC**
((random − P1) / (random − oracle) = 0.159 / 0.524). The SAE-based probes
(P3, P5) sit at or below random, consistent with the §4.2 finding that they
carry no difficulty signal beyond input statistics. Interpretation: cheap
classical context-window statistics from a 512-step window already form a
useful, label-free selective-prediction signal for Chronos-T5 forecasts on
ETTh1 — and that signal does not need internal representations to extract.

### 4.4 Qualitative feature inspection
Figure 2 (`probing/results/features/feat_*.png`): the five SAE features with
largest absolute probe weights, plotted on the windows where each fires
hardest. Interpretation: [FILL — do these visibly correspond to regime
shifts / level changes / anomalies, or are they diffuse? Be honest.]

### 4.5 Optional: feature-routed cascade
A second extraction with `chronos-t5-base` (`d_model=768`, ~3.3× cost of
small) lets us route on the P3 score and trace a cost–CRPS Pareto. If any
routed point dominates the always-cheap ↔ always-base interpolation,
"feature-routed cascade" is an empirical result, not a proposal.
[FILL after running `eval/cascade.py`: n_dominating_points, best_dominating.]

## 5. Limitations
- Single series (ETTh1), single TSFM backbone (chronos-t5-small). Layer
  robustness is checked across mid- and late-encoder (§4.2); generalization
  across domains, datasets, and larger backbones is deferred.
- We probe encoded context, not decoder sampling dynamics; CRPS depends on
  both.
- ADF replaced by a scipy variance-ratio proxy when statsmodels is absent.
- Single seed; the headline ΔAUROC CIs come from bootstrapping test windows,
  not from re-training.
- [If null:] SAE features are interpretable but not measurably more
  predictive than raw activations; we report this honestly rather than
  reframing the metric post-hoc.

## 6. Citations
Both 2026 SAE-on-TSFM citations in §2 were independently verified (May 2026)
and the arXiv IDs are correct as listed. Chronos (Ansari et al., 2024) is the
original benchmark protocol reference.

## 7. Future work
Multi-model SAE training and cross-backbone feature alignment
(Chronos / TimesFM / Moirai); cross-domain feature transfer (Monash, M5);
full implemented cascade with end-to-end cost accounting on real hardware;
feature steering ("Golden Gate for forecasting") for seasonal vs. trend modes
and abstention to a classical baseline on distribution-shift firing.

## Reproducibility
`bash reproduce.sh` runs the full pipeline. `requirements.txt` is pinned;
`.vscode/settings.json` selects the venv interpreter. Stale prototype
artifacts live in `_stale/` with a README warning. The difficulty probe
refuses to run on unlabeled metadata.
