"""Fit one GLM per assay/protein and compute delta-method contrasts.

This module has no knowledge of any particular study's groups, tissues, or
contrasts of interest — those are always supplied by the caller.
"""

import os

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from .stats_utils import add_fdr_columns


def contrasts_delta_method(model, contrasts: dict) -> dict:
    """Compute linear contrasts of a fitted GLM's coefficients via the delta method.

    Parameters
    ----------
    model : statsmodels fitted GLM results object
        Must expose ``.params`` and ``.cov_params()``.
    contrasts : dict
        Mapping of ``contrast_name -> {"coefs": {param_name: weight, ...}}``.
        Each contrast is the weighted sum ``sum(weight * param)`` over the
        named model parameters. Parameter names not present in the fitted
        model are silently ignored (their weight contributes 0), which lets
        the same contrast dict be reused across models that may be missing a
        coefficient (e.g. an unobserved interaction level).

    Returns
    -------
    dict
        ``contrast_name -> {"coef", "se", "z", "pval", "ci_lower", "ci_upper"}``.
    """
    params = model.params
    cov = model.cov_params()
    contrast_results = {}

    param_idx = {name: i for i, name in enumerate(params.index)}

    for cname, cinfo in contrasts.items():
        L = np.zeros(len(params))
        for coef, weight in cinfo["coefs"].items():
            if coef in param_idx:
                L[param_idx[coef]] = weight

        contrast_value = sum(
            params[coef] * weight for coef, weight in cinfo["coefs"].items() if coef in params
        )

        var_C = L.T @ cov @ L
        se_C = np.sqrt(var_C)

        z = contrast_value / se_C if se_C > 0 else np.nan
        # Use survival function for numerical stability in extreme tails:
        # 2 * (1 - cdf(|z|)) can underflow/cancel and become 0 for very small p-values.
        p = 2 * st.norm.sf(abs(z)) if not np.isnan(z) else np.nan
        if not np.isnan(p) and p == 0.0:
            p = np.finfo(float).tiny

        ci_lower = contrast_value - 1.96 * se_C
        ci_upper = contrast_value + 1.96 * se_C

        contrast_results[cname] = {
            "coef": contrast_value,
            "se": se_C,
            "z": z,
            "pval": p,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return contrast_results


def fit_GLMs(
    df: pd.DataFrame,
    formula: str,
    family,
    response_var: str,
    output_dir: str,
    contrasts: dict | None = None,
    id_col: str = "assay",
    label: str | None = None,
    make_plots: bool = True,
):
    """Fit one GLM per unique value of ``id_col`` and, optionally, delta-method contrasts.

    Parameters
    ----------
    df : DataFrame
        Long-format data with one row per (assay, sample).
    formula : str
        A patsy/statsmodels formula, e.g. ``"intensity ~ C(group) + C(sex)"``.
    family : statsmodels family instance
        e.g. ``sm.families.Gaussian(link=sm.families.links.identity())``.
    response_var : str
        Name of the response column (used to compute explained variance).
    output_dir : str
        Directory where diagnostic plots are written (created if needed).
    contrasts : dict, optional
        Passed to :func:`contrasts_delta_method` for each fitted model. If
        ``None``, no contrasts are computed and ``contrasts_df`` is empty.
    id_col : str
        Column identifying the unit to fit separately (default ``"assay"``).
    label : str, optional
        Human-readable label used in plot titles (defaults to ``response_var``).
    make_plots : bool
        Whether to save diagnostic plots (explained-variance histogram and an
        example fitted-vs-observed distribution plot) to ``output_dir``.

    Returns
    -------
    results_df, contrasts_df, exp_vars_df, normality_results
        ``results_df``: one row per unit, columns ``{param}_coef``,
        ``{param}_pval``, ``{param}_fdr`` for every model parameter.
        ``contrasts_df``: one row per unit, columns
        ``{contrast}_coef/se/z/pval/ci_lower/ci_upper/fdr`` for every
        requested contrast.
        ``exp_vars_df``: one row per unit with the model's explained variance.
        ``normality_results``: list of ``(unit, shapiro_W, shapiro_p)`` tuples
        from testing residual normality.
    """
    os.makedirs(output_dir, exist_ok=True)
    label = label or response_var

    results = []
    exp_vars = []
    contrasts_list = []
    normality_results = []
    last_successful_model = None
    last_successful_df_unit = None

    for unit in tqdm.tqdm(df[id_col].unique(), desc=f"Fitting GLMs per {id_col}"):
        df_unit = df[df[id_col] == unit]

        try:
            model = sm.GLM.from_formula(formula, data=df_unit, family=family).fit()

            errors = model.resid_response
            sst = np.sum((df_unit[response_var] - np.mean(df_unit[response_var])) ** 2)
            ssr = np.sum(errors**2)
            explained_variance = 1 - (ssr / sst)
            exp_vars.append((unit, explained_variance))

            residuals = model.resid_response
            stat, p = st.shapiro(residuals)
            normality_results.append((unit, stat, p))

            params = model.params.rename(index=lambda s: f"{s}_coef")
            pvals = model.pvalues.rename(index=lambda s: f"{s}_pval").astype(float)
            # keep tiny p-values visible (e.g. 1e-18) instead of collapsing to 0.0
            pvals = pvals.where((pvals != 0) | pvals.isna(), np.finfo(float).tiny)

            combined = pd.concat([params, pvals])
            combined.name = unit
        except Exception as exc:
            print(f"Model fitting failed for {unit}. Skipping. Reason: {exc}")
            combined = pd.Series(name=unit, dtype=float)
            contrasts_list.append({id_col.capitalize(): unit})
            results.append(combined)
            continue

        last_successful_model = model
        last_successful_df_unit = df_unit

        cont_dict = {id_col.capitalize(): unit}
        if contrasts:
            unit_contrasts = contrasts_delta_method(model, contrasts)
            for contrast_name, values in unit_contrasts.items():
                for key, val in values.items():
                    cont_dict[f"{contrast_name}_{key.lower()}"] = val
        contrasts_list.append(cont_dict)

        results.append(combined)

    id_col_title = id_col.capitalize()
    contrasts_df = pd.DataFrame(contrasts_list).set_index(id_col_title)

    exp_vars_df = pd.DataFrame(exp_vars, columns=[id_col_title, "Explained_Variance"]).set_index(id_col_title)
    if not exp_vars_df.empty:
        print(
            "Mean and median explained variance:",
            exp_vars_df["Explained_Variance"].mean(),
            exp_vars_df["Explained_Variance"].median(),
        )
        if make_plots:
            plt.figure(figsize=(6, 4))
            sns.histplot(exp_vars_df["Explained_Variance"], bins=30)
            plt.title(f"Explained Variance Distribution for {label}")
            plt.xlabel("Explained Variance")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{label}_explained_variance_histogram.png"), dpi=150)
            plt.close()

    results_df = pd.DataFrame(results)
    results_df.index.name = id_col

    param_bases = sorted({c.rsplit("_", 1)[0] for c in results_df.columns})
    cols_order = []
    for base in param_bases:
        if f"{base}_coef" in results_df.columns:
            cols_order.append(f"{base}_coef")
        if f"{base}_pval" in results_df.columns:
            cols_order.append(f"{base}_pval")
    if cols_order:
        results_df = results_df[cols_order]

    results_df = add_fdr_columns(results_df, alpha=0.05)
    contrasts_df = add_fdr_columns(contrasts_df, alpha=0.05)

    if make_plots and last_successful_model is not None and family.__class__.__name__ in ("Gaussian", "Gamma"):
        _plot_example_fit(last_successful_model, last_successful_df_unit, response_var, family)

    return results_df, contrasts_df, exp_vars_df, normality_results


def _plot_example_fit(model, df_unit: pd.DataFrame, response_var: str, family) -> None:
    """Plot the fitted distribution against observed data for one example unit.

    Purely a diagnostic sanity-check plot; shown inline / left on the current
    matplotlib figure rather than saved, since it's illustrative rather than
    a per-unit output.
    """
    mu = model.predict()
    y = df_unit[response_var].values

    if family.__class__.__name__ == "Gaussian":
        sigma2 = model.deviance / model._iweights.sum()
        i = 0
        x = np.linspace(mu[i] - 5 * np.sqrt(sigma2), mu[i] + 5 * np.sqrt(sigma2), 100)
        pdf = st.norm.pdf(x, loc=mu[i], scale=np.sqrt(sigma2))
        plt.figure()
        plt.hist(y, bins=10, density=True, alpha=0.4, label="data")
        plt.plot(x, pdf, lw=2, label="Gaussian fit")
        plt.legend()
        plt.close()

    elif family.__class__.__name__ == "Gamma":
        phi = model.scale
        shape = 1 / phi
        scale = phi * mu
        i = -1
        x = np.linspace(0, mu[i] * 5, 400)
        pdf = st.gamma.pdf(x, a=shape, scale=scale[i])
        plt.figure()
        plt.hist(y, bins=10, density=True, alpha=0.4, label="data")
        plt.plot(x, pdf, lw=2, label="Gamma fit")
        plt.legend()
        plt.close()
