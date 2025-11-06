# ============================================================
# Unified Gene Combination Analysis Pipeline (with Visualization & Saving)
# ------------------------------------------------------------
# Performs:
#   1. Single-Gene Threshold Optimization
#   2. Two-Gene Combination Analysis
#   3. Three-Gene Combination Analysis
#   4. KS Distribution Visualization (shown & saved)
#
# Features:
# - Parallelized with joblib for thousands of genes
# - Logging for progress tracking
# - Saves all outputs as CSV + KS plots as PNG
# ============================================================

import numpy as np
import pandas as pd
import itertools
import logging
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from tqdm import tqdm
import os

# ------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# ------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("plots/single_gene", exist_ok=True)
os.makedirs("plots/two_gene", exist_ok=True)
os.makedirs("plots/three_gene", exist_ok=True)

# ------------------------------------------------------------
# Helper: KS test visualization (show + save)
# ------------------------------------------------------------
def plot_ks_distribution(group_high, group_low, title, save_path):
    """Plot KS test distributions for two groups, show and save."""
    plt.figure(figsize=(6, 4))
    plt.hist(group_high, bins=20, alpha=0.6, label="High", density=True)
    plt.hist(group_low, bins=20, alpha=0.6, label="Low", density=True)
    plt.title(title)
    plt.xlabel("Phenotype Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

# ------------------------------------------------------------
# 1. Single-Gene Threshold Optimization
# ------------------------------------------------------------
def best_threshold_for_gene(gene, expr_mat, phenotype, thresholds=np.arange(0.2, 0.9, 0.1)):
    expr = expr_mat[gene]
    best_p = 1.0
    best_t = None
    for t in thresholds:
        cutoff = np.quantile(expr, t)
        group_high = phenotype[expr >= cutoff]
        group_low = phenotype[expr < cutoff]
        if len(group_high) > 0 and len(group_low) > 0:
            p = ks_2samp(group_high, group_low).pvalue
            if p < best_p:
                best_p, best_t = p, t
    return {'gene': gene, 'best_threshold': best_t, 'lowest_p': best_p}


def scan_genes(expr_mat, phenotype, thresholds=np.arange(0.2, 0.9, 0.1), n_jobs=-1):
    logger.info("Running single-gene threshold optimization...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(best_threshold_for_gene)(gene, expr_mat, phenotype, thresholds)
        for gene in tqdm(expr_mat.columns)
    )
    df = pd.DataFrame(results).sort_values('lowest_p')
    df.to_csv("results/single_gene_results.csv", index=False)
    logger.info("✅ Single-gene analysis complete. Saved to results/single_gene_results.csv")
    return df

# ------------------------------------------------------------
# 2. Two-Gene Combination Analysis
# ------------------------------------------------------------
def combine_two_genes(g1, g2, expr_mat, phenotype, thresholds1=[0.7], thresholds2=[0.3]):
    results = []
    for t1 in thresholds1:
        for t2 in thresholds2:
            high1 = expr_mat[g1] >= np.quantile(expr_mat[g1], t1)
            high2 = expr_mat[g2] >= np.quantile(expr_mat[g2], t2)
            group = (high1 & high2)
            if group.sum() > 0 and (~group).sum() > 0:
                p = ks_2samp(phenotype[group], phenotype[~group]).pvalue
                results.append({
                    'gene1': g1, 'gene2': g2,
                    'threshold1': t1, 'threshold2': t2, 'p_value': p
                })
    return results


def run_two_gene_combinations(expr_mat, phenotype, gene_list, thresholds1=[0.7], thresholds2=[0.3], n_jobs=-1):
    logger.info("Running two-gene combination analysis...")
    combos = list(itertools.combinations(gene_list, 2))
    results = Parallel(n_jobs=n_jobs)(
        delayed(combine_two_genes)(g1, g2, expr_mat, phenotype, thresholds1, thresholds2)
        for g1, g2 in tqdm(combos)
    )
    df = pd.DataFrame([r for sublist in results for r in sublist]).sort_values('p_value')
    df.to_csv("results/two_gene_results.csv", index=False)
    logger.info("✅ Two-gene combination analysis complete. Saved to results/two_gene_results.csv")
    return df

# ------------------------------------------------------------
# 3. Three-Gene Combination Analysis
# ------------------------------------------------------------
def combine_three_genes(g1, g2, g3, expr_mat, phenotype, thresholds=[0.7]):
    results = []
    for t in thresholds:
        high_expr = [expr_mat[g] >= np.quantile(expr_mat[g], t) for g in [g1, g2, g3]]
        group = np.logical_and.reduce(high_expr)
        if group.sum() > 0 and (~group).sum() > 0:
            p = ks_2samp(phenotype[group], phenotype[~group]).pvalue
            results.append({
                'gene1': g1, 'gene2': g2, 'gene3': g3,
                'threshold': t, 'p_value': p
            })
    return results


def run_three_gene_combinations(expr_mat, phenotype, gene_list, thresholds=[0.7], n_jobs=-1):
    logger.info("Running three-gene combination analysis...")
    combos = list(itertools.combinations(gene_list, 3))
    results = Parallel(n_jobs=n_jobs)(
        delayed(combine_three_genes)(g1, g2, g3, expr_mat, phenotype, thresholds)
        for g1, g2, g3 in tqdm(combos)
    )
    df = pd.DataFrame([r for sublist in results for r in sublist]).sort_values('p_value')
    df.to_csv("results/three_gene_results.csv", index=False)
    logger.info("✅ Three-gene combination analysis complete. Saved to results/three_gene_results.csv")
    return df

# ------------------------------------------------------------
# 4. Visualization of Top KS Comparisons (show + save)
# ------------------------------------------------------------
def visualize_top_single_genes(expr_mat, phenotype, single_results, top_n=5):
    logger.info(f"Visualizing top {top_n} single genes...")
    for _, row in single_results.head(top_n).iterrows():
        gene, t = row['gene'], row['best_threshold']
        cutoff = np.quantile(expr_mat[gene], t)
        group_high = phenotype[expr_mat[gene] >= cutoff]
        group_low = phenotype[expr_mat[gene] < cutoff]
        save_path = f"plots/single_gene/{gene}_p{row['lowest_p']:.2e}.png"
        plot_ks_distribution(group_high, group_low, f"{gene} (t={t:.2f}, p={row['lowest_p']:.2e})", save_path)

def visualize_top_two_genes(expr_mat, phenotype, two_results, top_n=3):
    logger.info(f"Visualizing top {top_n} two-gene pairs...")
    for _, row in two_results.head(top_n).iterrows():
        g1, g2, t1, t2 = row['gene1'], row['gene2'], row['threshold1'], row['threshold2']
        high1 = expr_mat[g1] >= np.quantile(expr_mat[g1], t1)
        high2 = expr_mat[g2] >= np.quantile(expr_mat[g2], t2)
        group = (high1 & high2)
        save_path = f"plots/two_gene/{g1}_{g2}_p{row['p_value']:.2e}.png"
        plot_ks_distribution(phenotype[group], phenotype[~group],
                             f"{g1}+{g2} (p={row['p_value']:.2e})", save_path)

# ------------------------------------------------------------
# 5. Example Workflow
# ------------------------------------------------------------
"""
# Load expression and phenotype data
expr_mat = pd.read_csv("expression_data.csv", index_col=0)
phenotype = pd.read_csv("phenotype.csv").values.ravel()

# Step 1: Single-Gene Optimization
single_gene_results = scan_genes(expr_mat, phenotype)

# Step 2: Two-Gene Combinations (using top 20 genes)
top_genes = single_gene_results['gene'].head(20).tolist()
two_gene_results = run_two_gene_combinations(expr_mat, phenotype, top_genes)

# Step 3: Three-Gene Combinations
three_gene_results = run_three_gene_combinations(expr_mat, phenotype, top_genes)

# Step 4: Visualization (show + save)
visualize_top_single_genes(expr_mat, phenotype, single_gene_results)
visualize_top_two_genes(expr_mat, phenotype, two_gene_results)

logger.info("🎯 Full pipeline complete! Results saved as CSV and KS plots generated & saved.")
"""
