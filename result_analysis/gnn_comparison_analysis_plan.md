# Frozen analysis plan: GNN versus baselines

## Primary estimand and population
- Binary rug-pull classification at fixed observation cutoffs; NOT first-alert time.
- Outcome: mean Average Precision (AP) over the five pre-existing trained seeds (0–4), with each seed evaluated on identical token IDs.
- Population: Panel, original cohort, no_mask, fixed six-file longitudinal intersection for each GNN versus the XGBoost node_graph baseline (q1/q2/q3 both model families). Intersection may differ by GNN; report IDs hash and class counts for each.
- Observation windows q1=12.5 min, q2=35.45 min, q3=93.08 min; confirm exact cutoff times against Trainer before submission.

## Nine prespecified AP contrasts (one multiplicity family)
- GNN models: GCN, GraphSAGE, GAT, **no post-Test model selection**.
- Comparator: XGBoost using node_graph features, original/no_mask Panel.
- For each GNN: q1 GNN minus q1 XGBoost; q1 GNN minus q2 XGBoost; q1 GNN minus q3 XGBoost.
- Directional research question: is the GNN AP superior to the reference? Positive differences support superiority. Negative/indeterminate cross-window differences do NOT establish equivalence.
- Formal primary inference: nine predeclared paired AP contrasts, with **Bonferroni simultaneous two-sided percentile bootstrap intervals**, familywise nominal alpha 0.05. Each marginal confidence = 1-0.05/9 = 99.444...%. Treat an AP superiority contrast as supported only if its Bonferroni-adjusted **lower** interval bound is strictly positive.
- Resample **token IDs** within true-label strata, jointly across compared models and all five seeds, 10,000 replicates. This measures uncertainty over tokens conditional on five fitted seeds and fixed prediction outputs; five seeds are NOT independent evaluation samples.
- Also report descriptive point effects and ordinary (unadjusted) 95% paired-bootstrap intervals. The simultaneous percentile-bootstrap CI approach is approximate, not a finite-sample exact guarantee; assess instability/small samples and do not claim equivalence if an interval includes zero.
- No score-swap permutation test or Holm p-values are part of the confirmatory primary analysis.


## Secondary and exploratory
- Remaining rows in gnn_baseline_pairs.csv (72 total), including Gradient Boosting, node_matched, q2 comparisons, and matched temporal curves; ROC-AUC is a secondary endpoint.
- Report effects and clearly labelled unadjusted 95% CIs; do not infer confirmatory significance for all 72 comparisons.
- Cross-model or cross-window comparisons use model-pair-specific six-file longitudinal intersection. Do not equate this with the entire Full token population.

## Interpretation constraints
- The q1 versus q2/q3 comparisons measure earlier **predictive performance**; do not report minutes of first true alert.
- Differences failing to show superiority are inconclusive about equivalence. No non-inferiority margin or formal equivalence hypothesis is prespecified, therefore do not claim equal or non-inferior performance.
- Differences from node_graph and node_matched do not isolate a pure causal topology contribution.
- Pointwise CIs on individual curve points are not paired-difference CIs and not simultaneous confidence bands.
- No threshold/model/primary-contrast changes after Test access. Report incomplete/missing primary comparisons, never quietly drop them.

## Freeze checklist
1. Verify GNN-vs-XGBoost matching and label/seed integrity on Validation and freeze implementation.
2. Freeze manifest, this plan, code revision, model selection protocol and hyperparameters before Test access.
3. Preflight complete Test data; report failures without silently dropping comparisons.
4. Generate full paired AP/ROC CI table (10k bootstrap), matched curves (10k bootstrap), and separately nine Bonferroni simultaneous-bootstrap interval results.
5. Include counts, uncertainty, multiplicity caveats, limitations and complete table in supplement.
