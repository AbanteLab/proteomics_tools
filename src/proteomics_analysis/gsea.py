"""Preranked GSEA against one or more MSigDB-style .gmt gene-set collections.

No collection list, .gmt directory, or organism suffix is hardcoded here —
callers supply all of that, since it's specific to a given study/organism.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gseapy as gp
from gseapy import dotplot
from gseapy.parser import read_gmt


def run_gsea(
    ranked_series: pd.Series,
    base: str,
    out_dir: str,
    gmt_dir: str,
    collections: list[str],
    cutoff: float = 0.25,
    permutation_num: int = 10000,
    min_size: int = 2,
    max_size: int = 50,
    seed: int = 42,
    background_genes: set | None = None,
    min_overlap: int = 2,
    gmt_filename_template: str = "{collection}.gmt",
    make_plots: bool = True,
    top_n_plot: int = 20,
):
    """Run preranked GSEA for one contrast/coefficient against several collections.

    Parameters
    ----------
    ranked_series : Series
        Index = gene/assay names, values = the ranking metric (e.g. a model
        coefficient). Will be sorted descending and written to
        ``{out_dir}/{base}_coef_ranked.rnk``.
    base : str
        Name of the contrast/coefficient being tested (used for filenames and titles).
    out_dir : str
        Directory to write ranked lists, per-collection results, and plots to.
    gmt_dir : str
        Directory containing the ``.gmt`` gene-set files.
    collections : list of str
        Collection identifiers, e.g. ``["m5.go.bp", "mh.all"]``. Each is
        formatted into ``gmt_filename_template`` to locate its ``.gmt`` file.
    cutoff : float
        FDR q-value threshold used to select "significant" pathways for
        plotting and for the returned ``sig_results_all``.
    background_genes : set, optional
        Genes to use as the background when filtering gene sets by overlap
        (e.g. all genes on your panel/platform). Defaults to
        ``set(ranked_series.index)``.
    min_overlap : int
        Minimum number of background genes a gene set must contain to be tested.
    gmt_filename_template : str
        Format string mapping a collection name to its ``.gmt`` filename,
        e.g. MSigDB mouse files are typically
        ``"{collection}.v2025.1.Mm.symbols.gmt"``.
    make_plots : bool
        Whether to save a dotplot per collection and write combined result files.

    Returns
    -------
    gsea_results_all, sig_results_all : DataFrame, DataFrame
        Concatenated results (with a ``collection`` column) across all
        collections, and the subset with ``FDR q-val < cutoff``.
    """
    os.makedirs(out_dir, exist_ok=True)

    ranked_list = ranked_series.sort_values(ascending=False)
    rnk_path = os.path.join(out_dir, f"{base}_coef_ranked.rnk")
    ranked_list.to_csv(rnk_path, sep="\t", header=False)

    if background_genes is None:
        background_genes = set(ranked_series.index)

    results_collections = []
    sig_results_collections = []

    for coll in collections:
        csv_path = os.path.join(out_dir, f"{base}_gsea_{coll}_full_results.csv")

        if os.path.exists(csv_path):
            # Results already computed on a previous run -- reuse them and go
            # straight to plotting instead of re-running the permutation test.
            print(f"Reusing existing GSEA results for {base} ({coll}): {csv_path}")
            res_df = pd.read_csv(csv_path, index_col=0)
        else:
            gmt_file = os.path.join(gmt_dir, gmt_filename_template.format(collection=coll))
            gmt = read_gmt(gmt_file)

            gmt_filt = {}
            for gene_set, genes_in_set in gmt.items():
                n_overlap = len(set(genes_in_set) & background_genes)
                if n_overlap >= min_overlap:
                    gmt_filt[gene_set] = genes_in_set

            if not gmt_filt:
                print(f"No gene sets with >= {min_overlap} overlapping genes for {base} ({coll}), skipping.")
                continue

            pre_res = gp.prerank(
                rnk=rnk_path,
                gene_sets=gmt_filt,
                permutation_num=permutation_num,
                outdir=None,
                seed=seed,
                min_size=min_size,
                max_size=max_size,
            )

            res_df = pre_res.res2d
            res_df["collection"] = coll

            res_df.to_csv(csv_path)
            res_df.to_excel(os.path.join(out_dir, f"{base}_gsea_{coll}_full_results.xlsx"))

        results_collections.append(res_df)

        sig_df = res_df[res_df["FDR q-val"] < cutoff].copy()
        sig_df = sig_df.sort_values("NES", ascending=False)

        if sig_df.empty:
            print(f"No significant pathways for {base} ({coll})")
            continue

        print(f"Significant pathways for {base} ({coll}): {sig_df.shape[0]}")
        sig_results_collections.append(sig_df)

        if make_plots:
            # sig_df is already sorted by NES descending; cap the dotplot at the
            # top_n_plot highest-NES pathways rather than showing every hit.
            _save_dotplot(
                sig_df.head(top_n_plot),
                title=f"{coll} (top {top_n_plot} by NES)",
                out_path=os.path.join(out_dir, f"{base}_gsea_{coll}_dotplot.png"),
                cutoff=cutoff,
            )

    if not results_collections:
        return pd.DataFrame(), pd.DataFrame()

    gsea_results_all = pd.concat(results_collections, axis=0).sort_values("FDR q-val")
    gsea_results_all.to_csv(os.path.join(out_dir, f"{base}_gsea_combined_collections_full_results.csv"))
    gsea_results_all.to_excel(os.path.join(out_dir, f"{base}_gsea_combined_collections_full_results.xlsx"))

    if sig_results_collections:
        sig_results_all = pd.concat(sig_results_collections, axis=0)
    else:
        sig_results_all = pd.DataFrame()

    if make_plots and not gsea_results_all.empty:
        plot_df = gsea_results_all.copy()
        plot_df["FDR q-val"] = plot_df["FDR q-val"].replace(0, np.nextafter(0, 1))
        # Restrict to significant terms before picking the top_n_plot to show.
        # gseapy's dotplot re-applies `cutoff` on "FDR q-val" anyway, but doing
        # it here means head(top_n_plot) selects from significant terms only
        # (rather than top-FDR terms that might then all be dropped at render).
        plot_df = plot_df[plot_df["FDR q-val"] < cutoff]
        top_terms = plot_df.sort_values("FDR q-val").head(top_n_plot)
        if not top_terms.empty:
            _save_dotplot(
                top_terms,
                title=f"GSEA Top {top_n_plot} Terms for {base}",
                out_path=os.path.join(out_dir, f"{base}_gsea_combined_collections_top{top_n_plot}_dotplot.png"),
                cutoff=cutoff,
            )

    return gsea_results_all, sig_results_all


def _save_dotplot(df: pd.DataFrame, title: str, out_path: str, cutoff: float, size: int = 6) -> None:
    plot_df = df.copy()
    plot_df["FDR q-val"] = plot_df["FDR q-val"].astype(float).replace(0, np.nextafter(0, 1))

    ax = dotplot(
        plot_df,
        column="FDR q-val",
        title=title,
        cmap=plt.cm.viridis,
        top_term=len(plot_df),
        size=size,
        figsize=(6, max(4, len(plot_df) * 0.4)),
        cutoff=cutoff,
        show_ring=False,
        labels=plot_df["Term"].tolist(),
    )
    fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
