"""Shared plotting utilities: PCA, volcano plots, and GSEA rank-change plots.

Column names, group labels, and color choices are all parameters — nothing
here assumes a particular study's covariates.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def plot_pca(
    df: pd.DataFrame,
    color_by: str,
    color_dict: dict | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    id_col: str = "sample_id",
    assay_col: str = "assay",
    value_col: str = "intensity",
):
    """PCA of samples (rows) x assays (columns), colored by a metadata column.

    Parameters
    ----------
    df : DataFrame
        Long-format data containing ``id_col``, ``assay_col``, ``value_col``,
        and ``color_by``.
    color_by : str
        Name of the metadata column to color points by (case-insensitive
        match against ``df`` columns).
    color_dict : dict, optional
        Mapping of category value -> matplotlib color. Categories not in the
        dict fall back to gray. If omitted, colors are assigned automatically.
    output_dir, label : optional
        If both are given, the plot is saved to
        ``{output_dir}/pca_plot_{label}_{color_by}.png``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    color_by = color_by.lower()
    intensity_data = df.pivot(index=id_col, columns=assay_col, values=value_col)

    scaler = StandardScaler()
    intensity_data_scaled = scaler.fit_transform(intensity_data)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(intensity_data_scaled)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    sample_ids = intensity_data.index.tolist()

    cov_vals = []
    for sid in sample_ids:
        cov_value = df[df[id_col].astype(str) == str(sid)][color_by].values
        cov_vals.append(str(cov_value[0]).lower() if len(cov_value) > 0 else "unknown")

    unique_cov_vals = set(cov_vals)
    color_dict = color_dict or {}
    cov_colors = {val: color_dict.get(val, "gray") for val in unique_cov_vals}

    for i, sid in enumerate(sample_ids):
        color = cov_colors.get(cov_vals[i], "gray")
        plt.scatter(pca_result[i, 0], pca_result[i, 1], color=color)
    for cov_value, color in cov_colors.items():
        plt.scatter([], [], color=color, label=cov_value)
    plt.legend(title=color_by)

    for i, sample_id in enumerate(intensity_data.index):
        plt.text(pca_result[i, 0], pca_result[i, 1], sample_id)

    title_label = label or ""
    plt.title(f"PCA of {title_label} colored by {color_by}".strip())
    plt.xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")

    if output_dir and label:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"pca_plot_{label}_{color_by}.png"), dpi=300, bbox_inches="tight")

    return plt.gca()


def make_volcano_plots(
    results_df: pd.DataFrame,
    output_dir: str,
    fdr_threshold: float = 0.05,
    coef_threshold: float = 0.5,
    fig_size: tuple = (6, 5),
    annotate_top_n: int = 15,
):
    """Save one volcano plot per parameter/contrast with both ``_coef`` and ``_fdr`` columns.

    Parameters
    ----------
    results_df : DataFrame
        Must contain columns named ``{base}_coef`` and ``{base}_fdr`` for
        each parameter/contrast to plot.
    output_dir : str
        Plots are written to ``{output_dir}/volcano_plots/volcano_{base}.pdf``.

    Returns
    -------
    list of str
        Paths of the saved figures.
    """
    plots_out_dir = os.path.join(output_dir, "volcano_plots")
    os.makedirs(plots_out_dir, exist_ok=True)

    param_bases = sorted({c.rsplit("_", 1)[0] for c in results_df.columns if c.endswith("_coef")})
    saved = []

    for base in param_bases:
        coef_col = f"{base}_coef"
        fdr_col = f"{base}_fdr"
        if coef_col not in results_df.columns or fdr_col not in results_df.columns:
            continue

        df_plot = results_df[[coef_col, fdr_col]].dropna()
        if df_plot.empty:
            continue

        coefs = pd.to_numeric(df_plot[coef_col], errors="coerce")
        fdrs = pd.to_numeric(df_plot[fdr_col], errors="coerce").replace(0, np.nextafter(0, 1))
        mask = fdrs.notna() & coefs.notna()
        if not mask.any():
            continue
        coefs, fdrs = coefs.loc[mask], fdrs.loc[mask]
        y = -np.log10(fdrs)

        sig_mask = fdrs < fdr_threshold
        nonsig = ~sig_mask
        sig_in_region = sig_mask & (np.abs(coefs) <= coef_threshold)
        sig_out_region = sig_mask & ~sig_in_region
        pos_sig_out = sig_out_region & (coefs > 0)
        neg_sig_out = sig_out_region & (coefs < 0)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.fill_betweenx([0, y.max() if len(y) else 1], -coef_threshold, coef_threshold, color="lightgray", alpha=0.5)

        if nonsig.any():
            ax.scatter(coefs[nonsig], y[nonsig], color="grey", alpha=0.6, s=20, label="ns")
        if neg_sig_out.any():
            ax.scatter(coefs[neg_sig_out], y[neg_sig_out], color="blue", alpha=0.9, s=20, label=f"FDR < {fdr_threshold} (neg)")
        if pos_sig_out.any():
            ax.scatter(coefs[pos_sig_out], y[pos_sig_out], color="red", alpha=0.9, s=20, label=f"FDR < {fdr_threshold} (pos)")
        if sig_in_region.any():
            ax.scatter(
                coefs[sig_in_region], y[sig_in_region], color="dimgray", alpha=0.95, s=30,
                label=f"FDR < {fdr_threshold} (|coef| \u2264 {coef_threshold})",
            )

        ax.axhline(-np.log10(fdr_threshold), color="blue", lw=1, ls="--")
        ax.axvline(0, color="black", lw=0.7)
        ax.axvline(coef_threshold, color="gray", lw=0.7)
        ax.axvline(-coef_threshold, color="gray", lw=0.7)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("-log10(FDR)")
        ax.legend(frameon=False, fontsize=8)

        if sig_out_region.any():
            top_n = min(annotate_top_n, sig_out_region.sum())
            top_idx = y[sig_out_region].nlargest(top_n).index
            for idx in top_idx:
                ax.text(coefs.loc[idx], y.loc[idx], str(idx), fontsize=6, ha="right", va="bottom", color="black")

        plt.tight_layout()
        out_path = os.path.join(plots_out_dir, f"volcano_{base.replace('/', '_')}.pdf")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)

    return saved


def plot_rank_change(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    rank_col: str,
    ascending: bool,
    output_path: str,
    label_1: str,
    label_2: str,
    topn: int = 20,
):
    """Plot how the top-``topn`` GSEA terms' ranks shift between two result sets.

    Parameters
    ----------
    df_1, df_2 : DataFrame
        Each must contain a ``Term`` column and ``rank_col``.
    rank_col : str
        Column to rank terms by within each DataFrame (e.g. ``"NES"``).
    output_path : str
        Full path (including filename) to save the figure to.
    label_1, label_2 : str
        Axis labels identifying each result set.
    """
    top_df_1 = df_1.sort_values(rank_col, ascending=ascending).head(topn).assign(rank_1=lambda d: range(1, len(d) + 1))
    top_df_2 = df_2.sort_values(rank_col, ascending=ascending).head(topn).assign(rank_2=lambda d: range(1, len(d) + 1))
    top_df_1 = top_df_1[["Term", "rank_1"]]
    top_df_2 = top_df_2[["Term", "rank_2"]]

    off_rank = topn + 1
    terms_1, terms_2 = set(top_df_1["Term"]), set(top_df_2["Term"])
    dropped, entered = terms_1 - terms_2, terms_2 - terms_1

    df = pd.merge(top_df_1, top_df_2, on="Term", how="outer")
    df["rank_1_plot"] = df["rank_1"]
    df["rank_2_plot"] = df["rank_2"]
    df.loc[df["Term"].isin(dropped), "rank_2_plot"] = off_rank
    df.loc[df["Term"].isin(entered), "rank_1_plot"] = off_rank

    df["delta"] = df["rank_1"] - df["rank_2"]
    df["movement"] = np.where(
        df["delta"].isna(), "out", np.where(df["delta"] > 0, "up", np.where(df["delta"] < 0, "down", "same"))
    )

    df_score = pd.merge(
        df_1[["Term", rank_col]].rename(columns={rank_col: "score_1"}),
        df_2[["Term", rank_col]].rename(columns={rank_col: "score_2"}),
        on="Term",
        how="outer",
    )
    df = df.merge(df_score, on="Term", how="left")
    df["score_change"] = np.abs(df["score_2"] - df["score_1"])
    max_change = df["score_change"].max()
    df["arrow_width"] = 1 + 5 * df["score_change"] / max_change

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_axes([0.25, 0.05, 0.5, 0.8])
    x_1, x_2 = 0, 1
    colors = {"up": "green", "down": "red", "same": "tab:orange", "out": "gray"}

    for _, row in df.iterrows():
        ax.annotate(
            "",
            xy=(x_2, row["rank_2_plot"]),
            xytext=(x_1, row["rank_1_plot"]),
            arrowprops=dict(arrowstyle="->", color=colors.get(row["movement"], "gray"), lw=1.5, alpha=0.8),
        )
        ax.scatter(x_1, row["rank_1"], color="black", s=30, zorder=3)
        ax.scatter(x_2, row["rank_2"], color="black", s=30, zorder=3)
        ax.text(x_2 + 0.7, row["rank_2"], row["Term"], ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(off_rank + 0.5, 0.5)
    ax.set_xticks([x_1, x_2])
    ax.set_xticklabels([label_1, label_2])
    ax.set_ylabel(f"Rank (by {rank_col})")
    ax.set_title(f"Top {topn} GSEA terms: rank changes")
    ax.axhline(off_rank, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.text(x_2 + 0.7, off_rank, f"Outside top {topn}", va="center", fontsize=9, color="black")

    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="Moves up"),
        Line2D([0], [0], color="red", lw=2, label="Moves down"),
        Line2D([0], [0], color="tab:orange", lw=2, label="Same rank"),
        Line2D([0], [0], color="gray", lw=2, label="Moves in/out"),
        Line2D([0], [0], marker="o", color="black", lw=0, markersize=6, label=f"Top {topn}"),
        Line2D([0], [0], marker="o", color="black", lw=0, markersize=6, alpha=0.5, label=f"Outside top {topn}"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right", bbox_to_anchor=(0.25, 0.98),
        bbox_transform=ax.transAxes, frameon=False, fontsize=6,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
