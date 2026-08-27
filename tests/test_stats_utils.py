import numpy as np
import pandas as pd

from proteomics_analysis.stats_utils import add_fdr_columns


def test_add_fdr_columns_inserts_fdr_next_to_pval():
    df = pd.DataFrame(
        {
            "groupA_coef": [1.0, 2.0, 3.0],
            "groupA_pval": [0.001, 0.04, 0.5],
        }
    )
    out = add_fdr_columns(df)
    assert list(out.columns) == ["groupA_coef", "groupA_pval", "groupA_fdr"]
    assert out["groupA_fdr"].notna().all()
    # BH-adjusted p-values should be >= raw p-values
    assert (out["groupA_fdr"].values >= out["groupA_pval"].values - 1e-12).all()


def test_add_fdr_columns_handles_nans():
    df = pd.DataFrame({"x_pval": [0.01, np.nan, 0.2]})
    out = add_fdr_columns(df)
    assert pd.isna(out["x_fdr"].iloc[1])
    assert out["x_fdr"].iloc[[0, 2]].notna().all()
