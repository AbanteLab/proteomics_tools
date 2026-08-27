import numpy as np
import pandas as pd
import statsmodels.api as sm

from proteomics_analysis.glm import fit_GLMs, contrasts_delta_method


def _make_toy_df(seed=0, n_per_group=15):
    rng = np.random.default_rng(seed)
    rows = []
    for assay in ["prot_a", "prot_b"]:
        for group, effect in [("ctrl", 0.0), ("treated", 1.5)]:
            values = rng.normal(loc=effect, scale=1.0, size=n_per_group)
            for v in values:
                rows.append({"assay": assay, "group": group, "intensity": v})
    return pd.DataFrame(rows)


def test_contrasts_delta_method_recovers_group_effect():
    df = _make_toy_df()
    df_a = df[df["assay"] == "prot_a"]
    family = sm.families.Gaussian(link=sm.families.links.identity())
    model = sm.GLM.from_formula("intensity ~ C(group)", data=df_a, family=family).fit()

    contrasts = {"treated_vs_ctrl": {"coefs": {"C(group)[T.treated]": 1}}}
    result = contrasts_delta_method(model, contrasts)

    assert "treated_vs_ctrl" in result
    assert result["treated_vs_ctrl"]["coef"] == model.params["C(group)[T.treated]"]
    assert result["treated_vs_ctrl"]["pval"] < 0.05


def test_fit_GLMs_runs_end_to_end(tmp_path):
    df = _make_toy_df()
    family = sm.families.Gaussian(link=sm.families.links.identity())
    contrasts = {"treated_vs_ctrl": {"coefs": {"C(group)[T.treated]": 1}}}

    results_df, contrasts_df, exp_vars_df, normality_results = fit_GLMs(
        df,
        formula="intensity ~ C(group)",
        family=family,
        response_var="intensity",
        output_dir=str(tmp_path),
        contrasts=contrasts,
        make_plots=False,
    )

    assert set(results_df.index) == {"prot_a", "prot_b"}
    assert "treated_vs_ctrl_coef" in contrasts_df.columns
    assert "treated_vs_ctrl_fdr" in contrasts_df.columns
    assert len(normality_results) == 2
