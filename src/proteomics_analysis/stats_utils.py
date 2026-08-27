"""Generic statistical helpers shared across the package."""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def add_fdr_columns(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add a Benjamini-Hochberg FDR column right after every ``*_pval`` column.

    For each column named ``{base}_pval``, inserts a new column
    ``{base}_fdr`` immediately to its right containing the BH-adjusted
    p-values (NaNs are ignored and left as NaN).

    Parameters
    ----------
    results_df : DataFrame
        Table with one or more columns ending in ``_pval``.
    alpha : float
        Significance level passed to ``statsmodels.stats.multitest.multipletests``.
        Only affects the ``reject`` output, which is not used here, but is kept
        for parity with ``multipletests``' signature.

    Returns
    -------
    DataFrame
        Copy of ``results_df`` with the new ``_fdr`` columns inserted.
    """
    results_df = results_df.copy()
    for col in list(results_df.columns):
        if col.endswith("_pval"):
            base = col.rsplit("_", 1)[0]
            fdr_col = f"{base}_fdr"
            pvals = results_df[col]
            adj = pd.Series(np.nan, index=results_df.index)
            mask = pvals.notna()
            if mask.any():
                try:
                    _, pvals_adj, _, _ = multipletests(pvals[mask].values, alpha=alpha, method="fdr_bh")
                except Exception:
                    pvals_adj = np.full(mask.sum(), np.nan)
                adj.loc[mask] = pvals_adj
            loc = results_df.columns.get_loc(col)
            results_df.insert(loc + 1, fdr_col, adj)
    return results_df
