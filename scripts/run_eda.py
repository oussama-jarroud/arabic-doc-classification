"""Exploratory Data Analysis script computing class metrics, vocabulary stats, and distributions."""

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path for direct script invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    build_split_if_needed,
    load_config,
    load_split_records,
)


def compute_text_statistics(
    file_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, Counter[str]]:
    """Extract character counts, word counts, and token frequencies to evaluate length variance."""
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    token_counter: Counter[str] = Counter()

    for path in file_paths:
        text = path.read_text(encoding="utf-8")
        tokens = text.split()
        char_lengths.append(len(text))
        word_lengths.append(len(tokens))
        token_counter.update(tokens)

    return np.array(char_lengths), np.array(word_lengths), token_counter


def plot_class_counts_and_duplicates(
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate bar chart comparing raw vs clean counts to visualize deduplication."""
    categories = [r["category"] for r in summary_rows]
    raw_counts = [r["raw_count"] for r in summary_rows]
    clean_counts = [r["clean_count"] for r in summary_rows]
    dup_counts = [r["duplicates_removed"] for r in summary_rows]

    x = np.arange(len(categories))
    width = 0.35

    has_seaborn = "seaborn-v0_8-whitegrid" in plt.style.available
    style = "seaborn-v0_8-whitegrid" if has_seaborn else "default"
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.bar(x - width / 2, raw_counts, width, label="Raw Documents", color="#4C72B0")
    ax.bar(
        x + width / 2,
        clean_counts,
        width,
        label="Clean (Deduplicated)",
        color="#55A868",
    )

    ax.set_ylabel("Document Count", fontsize=12, fontweight="bold")
    ax.set_title(
        "Khaleej Corpus: Raw vs. Deduplicated Counts per Category",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=11)

    for i, dup in enumerate(dup_counts):
        if dup > 0:
            ax.annotate(
                f"-{dup} dups",
                (x[i] + width / 2, clean_counts[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=9,
                color="#C44E52",
                fontweight="bold",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_doc_length_distribution(
    word_lengths_by_cat: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Produce boxplots of document word lengths across classes to identify domain length skews."""
    categories = list(word_lengths_by_cat.keys())
    data = [word_lengths_by_cat[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    box = ax.boxplot(
        data,
        tick_labels=categories,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markeredgecolor": "red",
            "markersize": 5,
        },
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.2},
    )

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD", "#8C8C8C"]
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Word Count per Document", fontsize=12, fontweight="bold")
    ax.set_title(
        "Document Word Length Distribution Across Khaleej Categories",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Execute complete EDA workflow generating summary tables and figures."""
    cfg = load_config()
    print("Building or validating fixed stratified 80/20 split...")
    split_file = build_split_if_needed(cfg)
    print(f"Split file ready: {split_file}")

    dev_records = load_split_records("dev", cfg)
    test_records = load_split_records("test", cfg)

    dev_count_by_cat: Counter[str] = Counter(r["category"] for r in dev_records)
    test_count_by_cat: Counter[str] = Counter(r["category"] for r in test_records)

    categories: list[str] = list(cfg["dataset"]["categories"])
    summary_rows: list[dict[str, Any]] = []
    top_terms_rows: list[dict[str, Any]] = []
    word_lengths_by_cat: dict[str, np.ndarray] = {}

    import json
    with split_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)["metadata"]
    dedup_stats = meta["deduplication_stats"]

    for cat in categories:
        cat_records = [r for r in dev_records + test_records if r["category"] == cat]
        cat_paths = [Path(r["path"]) for r in cat_records]

        char_arr, word_arr, token_counter = compute_text_statistics(cat_paths)
        word_lengths_by_cat[cat] = word_arr

        cat_stats = dedup_stats[cat]
        summary_rows.append({
            "category": cat,
            "raw_count": cat_stats["raw_count"],
            "empty_count": cat_stats["empty_count"],
            "duplicates_removed": cat_stats["duplicates_removed"],
            "clean_count": cat_stats["clean_count"],
            "dev_count": dev_count_by_cat[cat],
            "test_count": test_count_by_cat[cat],
            "mean_words": round(float(np.mean(word_arr)), 2),
            "std_words": round(float(np.std(word_arr)), 2),
            "median_words": round(float(np.median(word_arr)), 2),
            "mean_chars": round(float(np.mean(char_arr)), 2),
            "std_chars": round(float(np.std(char_arr)), 2),
            "median_chars": round(float(np.median(char_arr)), 2),
        })

        top_15 = [token for token, _ in token_counter.most_common(15)]
        top_terms_rows.append({
            "category": cat,
            "unique_tokens": len(token_counter),
            "total_tokens": sum(token_counter.values()),
            "top_15_frequent_tokens": " | ".join(top_15),
        })

    # 1. Save results/eda_class_summary.csv
    summary_path = Path(cfg["eda"]["class_summary_csv"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_cols = list(summary_rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_cols)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved EDA class summary to {summary_path}")

    # 2. Save results/eda_top_terms.csv
    top_terms_path = Path(cfg["eda"]["top_terms_csv"])
    top_terms_path.parent.mkdir(parents=True, exist_ok=True)
    terms_cols = list(top_terms_rows[0].keys())
    with top_terms_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=terms_cols)
        writer.writeheader()
        writer.writerows(top_terms_rows)
    print(f"Saved EDA vocabulary stats to {top_terms_path}")

    # 3. Generate charts
    class_counts_plot_path = Path(cfg["eda"]["class_counts_plot"])
    plot_class_counts_and_duplicates(summary_rows, class_counts_plot_path)
    print(f"Saved class counts plot to {class_counts_plot_path}")

    doc_length_plot_path = Path(cfg["eda"]["doc_length_plot"])
    plot_doc_length_distribution(word_lengths_by_cat, doc_length_plot_path)
    print(f"Saved doc length distribution to {doc_length_plot_path}")


if __name__ == "__main__":
    main()
