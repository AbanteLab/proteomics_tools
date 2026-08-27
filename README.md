# Tools for proteomics data analysis

General-purpose methods for analyzing proteomics data, independent of any single
study. This package covers three stages of analysis:

1. **GLM fitting per assay** (`proteomics_tools.glm`) — fit one GLM per
   protein/assay for a user-supplied formula and family, identify DEGs and compute linear contrasts of the fitted coefficients via the delta
   method.
2. **Gene set enrichment analysis** (`proteomics_tools.gsea`) — run
   preranked GSEA against one or more `.gmt` gene-set collections, given a
   ranking metric per assay.
3. **Shared plotting utilities** (`proteomics_tools.plotting`) — PCA plots,
   volcano plots, and GSEA rank-change plots.

Study-specific details — which contrasts to compute, which gene-set
collections/paths to use, which covariates or groups exist, how to
integrate across tissues (e.g. MOFA) — are **not** part of this package.
Those live in the analysis repo for a given project (e.g. `repro_paper`),
which depends on `proteomics_analysis` and supplies that configuration as
arguments.

## Installation

```bash
pip install -e /path/to/proteomics_tools
```

or, from a downstream project's `pyproject.toml`:

```toml
dependencies = [
    "proteomics_tools @ git+https://github.com/<you>/proteomics_tools.git",
]
```

## Modules

- `proteomics_tools.stats_utils` — `add_fdr_columns`
- `proteomics_tools.glm` — `fit_GLMs`, `contrasts_delta_method`
- `proteomics_tools.gsea` — `run_gsea`
- `proteomics_tools.plotting` — `plot_pca`, `make_volcano_plots`, `plot_rank_change`

## Example

```python
import statsmodels.api as sm
from proteomics_tools.glm import fit_GLMs
from proteomics_tools.gsea import run_gsea

family = sm.families.Gaussian(link=sm.families.links.identity())

contrasts = {
    "treated_vs_ctrl": {"coefs": {"C(group)[T.treated]": 1}},
}

results_df, contrasts_df, exp_vars_df, normality_results = fit_GLMs(
    df,
    formula="intensity ~ C(group) + C(sex) + nan_count",
    family=family,
    response_var="intensity",
    output_dir="results/my_tissue",
    contrasts=contrasts,
)

gsea_results, sig_results = run_gsea(
    ranked_df=results_df[["treated_vs_ctrl_coef"]].rename(columns={"treated_vs_ctrl_coef": "rank_metric"}),
    base="treated_vs_ctrl",
    out_dir="results/my_tissue/GSEA_Results/treated_vs_ctrl",
    gmt_dir="/path/to/gene_sets",
    collections=["m5.go.bp", "mh.all"],
)
```
