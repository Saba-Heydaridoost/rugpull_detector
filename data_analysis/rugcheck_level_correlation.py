"""Correlate RugCheck risk `level` with our rule-based rug-pull labels.

RugCheck reports a `level` (warn / danger) per entry in risks[]. Per token we use
the maximum level as an ordinal (none=0 < warn=1 < danger=2), plus danger/warn
counts. Labels: rug_pull_label from rule_based_labels.csv where label_status is
VALID (the label is identical across q1-q3 for every mint, so token-level is
used as primary; per-experiment subsets are reported too).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "dataset" / "rugcheck"
OUT = Path(__file__).resolve().parent / "output"


def cramers_v(ct):
    chi2, p, _, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1))), chi2, p


def correlate(df, x, y="rug_pull_label"):
    d = df[[x, y]].dropna()
    rho, p_rho = stats.spearmanr(d[x], d[y])
    tau, p_tau = stats.kendalltau(d[x], d[y])
    r, p_r = stats.pointbiserialr(d[y], d[x])
    auc = stats.mannwhitneyu(d.loc[d[y] == 1, x], d.loc[d[y] == 0, x]).statistic / (
        (d[y] == 1).sum() * (d[y] == 0).sum())
    return dict(feature=x, n=len(d), spearman_rho=rho, spearman_p=p_rho, kendall_tau_b=tau,
                kendall_p=p_tau, point_biserial_r=r, point_biserial_p=p_r, auc=auc)


def main():
    rc = pd.read_csv(RC / "rugcheck_reports.csv", low_memory=False,
                     usecols=["mint", "ok", "max_risk_level", "max_risk_level_ord", "n_danger",
                              "n_warn", "report.score_normalised", "report.rugged"])
    lab = pd.read_csv(ROOT / "dataset" / "rule_based_labels.csv",
                      usecols=["token_mint", "experiment_id", "label_status", "rug_pull_label"])
    valid = lab[lab.label_status == "VALID"]
    tok = valid.groupby("token_mint", as_index=False).rug_pull_label.first()
    df = tok.merge(rc[rc.ok], left_on="token_mint", right_on="mint", how="inner")
    print(f"tokens with VALID label: {len(tok)}; with RugCheck report: {len(df)}")

    ct = pd.crosstab(df.max_risk_level, df.rug_pull_label).reindex(["none", "warn", "danger"])
    ct.columns = ["not_rug", "rug"]
    ct["n"] = ct.sum(axis=1)
    ct["rug_rate"] = ct.rug / ct.n
    v, chi2, p = cramers_v(ct[["not_rug", "rug"]])
    print("\nmax risk level vs rule-based label\n", ct.to_string())
    print(f"chi2={chi2:.2f} p={p:.3g} Cramer's V={v:.3f}")

    feats = ["max_risk_level_ord", "n_danger", "n_warn", "report.score_normalised"]
    res = pd.DataFrame([correlate(df, f) for f in feats])
    df["any_danger"] = (df.max_risk_level_ord == 2).astype(int)
    res = pd.concat([res, pd.DataFrame([correlate(df, "any_danger")])], ignore_index=True)
    res.insert(0, "subset", "token_level")

    per_exp = []
    for q, g in valid.groupby("experiment_id"):
        d = g[["token_mint", "rug_pull_label"]].merge(rc[rc.ok], left_on="token_mint", right_on="mint")
        per_exp.append({"subset": q, **correlate(d, "max_risk_level_ord")})
    res = pd.concat([res, pd.DataFrame(per_exp)], ignore_index=True)
    print("\n", res.round(4).to_string(index=False))

    # Per-risk association: does having risk X at level L co-occur with rug labels?
    risks = pd.read_csv(RC / "rugcheck_risks.csv", usecols=["mint", "name", "level"])
    risks["risk"] = risks.name + " [" + risks.level + "]"
    ind = pd.crosstab(risks.mint, risks.risk).clip(upper=1)
    m = df[["mint", "rug_pull_label"]].set_index("mint").join(ind).fillna(0)
    per_risk = []
    for c in ind.columns:
        has = m[c] == 1
        if has.sum() < 10:
            continue
        r, p = stats.pearsonr(m[c], m.rug_pull_label)  # phi coefficient
        per_risk.append(dict(risk=c, n_tokens=int(has.sum()), rug_rate_with=m.rug_pull_label[has].mean(),
                             rug_rate_without=m.rug_pull_label[~has].mean(), phi=r, p=p))
    per_risk = pd.DataFrame(per_risk).sort_values("phi", ascending=False)
    print("\nper-risk phi with rug label\n", per_risk.round(4).to_string(index=False))

    OUT.mkdir(exist_ok=True)
    ct.to_csv(OUT / "rugcheck_level_crosstab.csv")
    res.to_csv(OUT / "rugcheck_level_correlation.csv", index=False)
    per_risk.to_csv(OUT / "rugcheck_per_risk_correlation.csv", index=False)


if __name__ == "__main__":
    main()
