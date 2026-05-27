# =============================================================================
# step5_analysis.py
#
# OBJECTIVE: Load the results of previous steps and generate: 
#   - Table stats summary
#   - Tests t  (with vs without)
#   - Graphs saved in the folder plots/
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from config import PATHS


def print_section(title):
    """Print the section head on the terminal."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run():
    print("[Step 5] Generating graph analysis...")

    pairs_eval      = pd.read_csv(PATHS["pairs_eval"])
    normal_emoji_df = pd.read_csv(PATHS["normal_emoji"])

    os.makedirs(PATHS["plots"], exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Summary per congruence (artificial pairs)
    # -----------------------------------------------------------------------
    print_section("Efects of context per type of congruence  (artifical pairs)")

    summary_context = (
        pairs_eval.groupby("congruence_simple")
        .agg(
            n             = ("emoji_inserted", "size"),
            mean_p_noctx  = ("p_noctx",        "mean"),
            mean_p_ctx    = ("p_ctx",           "mean"),
            mean_delta    = ("delta_context",   "mean"),
            median_delta  = ("delta_context",   "median"),
        )
        .reset_index()
    )
    print(summary_context.to_string(index=False))

    # -----------------------------------------------------------------------
    # 2. T tests
    # -----------------------------------------------------------------------
    print_section("T tests (without and with contex)")

    t_art = ttest_rel(pairs_eval["p_noctx"], pairs_eval["p_ctx"])
    print(f"  Artifical pairs   → t = {t_art.statistic:.4f},  p = {t_art.pvalue:.4f}")

    t_spon = ttest_rel(normal_emoji_df["p_noctx"], normal_emoji_df["p_ctx"])
    print(f"  Spontaneous turns → t = {t_spon.statistic:.4f},  p = {t_spon.pvalue:.4f}")

    # -----------------------------------------------------------------------
    # 3. Extreme cases — context helps / hurts
    # -----------------------------------------------------------------------
    print_section("Artifical pairs: context helps (top 5)")
    top_art = pairs_eval.sort_values("delta_context", ascending=False)[[
        "intercepted_turn", "emoji_inserted", "p_noctx", "p_ctx",
        "delta_context", "Type of intervention"
    ]].head(5)
    print(top_art.to_string(index=False))

    print_section("Artifical pairs: context hurts (top 5)")
    worst_art = pairs_eval.sort_values("delta_context", ascending=True)[[
        "intercepted_turn", "emoji_inserted", "p_noctx", "p_ctx",
        "delta_context", "Type of intervention"
    ]].head(5)
    print(worst_art.to_string(index=False))

    print_section("Spontaneous turns: context helps (top 5)")
    top_spon = normal_emoji_df.sort_values("delta_context", ascending=False)[[
        "Turn No.", "user_emoji", "p_noctx", "p_ctx", "delta_context"
    ]].head(5)
    print(top_spon.to_string(index=False))

    print_section("Spontaneous turns: context hurts (top 5)")
    worst_spon = normal_emoji_df.sort_values("delta_context", ascending=True)[[
        "Turn No.", "user_emoji", "p_noctx", "p_ctx", "delta_context"
    ]].head(5)
    print(worst_spon.to_string(index=False))

    # -----------------------------------------------------------------------
    # 4. Graph 1  — Scatter: probability without and with context (artificial)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pairs_eval["p_noctx"], pairs_eval["p_ctx"], alpha=0.7)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Probability without context")
    ax.set_ylabel("Probability with context")
    ax.set_title("Artificial pairs: before vs after context")
    fig.tight_layout()
    fig.savefig(PATHS["plots"] + "artificial_scatter.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 5. Graph 2 — Histogram of the delta (artificial)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pairs_eval["delta_context"], bins=10, alpha=0.8)
    ax.axvline(0, linestyle="--", color="red")
    ax.set_xlabel("Variation of probability (context − without context)")
    ax.set_ylabel("Number of examples")
    ax.set_title("Context effect — artifical emojis ")
    fig.tight_layout()
    fig.savefig(PATHS["plots"] + "artificial_delta_hist.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 6. Graph 3 — Histogram of the delta (spontaneous)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(normal_emoji_df["delta_context"], bins=10, alpha=0.8)
    ax.axvline(0, linestyle="--", color="red")
    ax.set_xlabel("Variation of probability (context − without context)")
    ax.set_ylabel("Number of examples")
    ax.set_title("Context effect — spontaneous emojis")
    fig.tight_layout()
    fig.savefig(PATHS["plots"] + "spontaneous_delta_hist.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 7. Graph 4 — Grouped histogram: artificial vs. spontaneous (without context)
    # -----------------------------------------------------------------------
    artificial = pairs_eval["p_noctx"].dropna()
    user       = normal_emoji_df["p_noctx"].dropna()

    bins               = np.linspace(0, 1, 11)
    counts_artificial, edges = np.histogram(artificial, bins=bins)
    counts_user, _           = np.histogram(user, bins=bins)
    centers                  = (edges[:-1] + edges[1:]) / 2
    bar_width                = (edges[1] - edges[0]) * 0.4

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(centers - bar_width / 2, counts_artificial, width=bar_width,
           label="Artificially inserted emoji", color="blue")
    ax.bar(centers + bar_width / 2, counts_user,       width=bar_width,
           label="Naturally occurring emoji", color="green")
    ax.set_xlabel("Emoji probability (without context)")
    ax.set_ylabel("Number of examples")
    ax.set_title("Emoji prediction without conversational context")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATHS["plots"] + "comparison_histogram.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 8. Graph 5 — Grouped histogram: artificial vs. spontaneous (with context)
    # -----------------------------------------------------------------------
    artificial_ctx = pairs_eval["p_ctx"].dropna()
    user_ctx       = normal_emoji_df["p_ctx"].dropna()

    counts_artificial_ctx, _ = np.histogram(artificial_ctx, bins=bins)
    counts_user_ctx, _       = np.histogram(user_ctx, bins=bins)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(centers - bar_width / 2, counts_artificial_ctx, width=bar_width,
           label="Artificially inserted emoji", color="blue")
    ax.bar(centers + bar_width / 2, counts_user_ctx,       width=bar_width,
           label="Naturally occurring emoji", color="green")
    ax.set_xlabel("Probability of emoji (with context)")
    ax.set_ylabel("Number of examples")
    ax.set_title("Emoji prediction with conversational context")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATHS["plots"] + "comparison_histogram_ctx.png", dpi=150)
    plt.close(fig)

    print(f"\n  Graphs saved in: {PATHS['plots']}")


if __name__ == "__main__":
    run()
