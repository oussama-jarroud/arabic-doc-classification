"""Phase 7: Comprehensive Evaluation Suite, Statistical Testing, and Graphics.

Evaluates the finalized models on the held-out test split (9,097 articles) under zero-leakage
conditions, generates explainability graphics, conducts McNemar's significance testing, evaluates
synthetic typographic noise robustness (Hypothesis 5), and benchmarks Pareto efficiency.
"""

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

# Reconfigure stdout for safe UTF-8 printing in Windows consoles
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path.cwd()))

from src.models_dl import ArabicCNN_1D, ArabicTextDataset, TextVocabulary  # noqa: E402

CLASS_NAMES: list[str] = [
    "Culture",
    "Finance",
    "Medical",
    "Politics",
    "Religion",
    "Sports",
    "Tech",
]

# Keyboard typographic substitution map for Arabic informal typing errors
ARABIC_TYPO_MAP: dict[str, str] = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ا": "أ",
    "ة": "ه",
    "ه": "ة",
    "ى": "ي",
    "ي": "ى",
    "ؤ": "و",
    "ئ": "ي",
}


def load_clean_test_split(
    test_path: Path = Path("data/processed/light_normalized_test.joblib"),
) -> tuple[list[str], list[int]]:
    """Load the held-out test split for strict inference evaluation."""
    if not test_path.is_file():
        raise FileNotFoundError(f"Test split not found at: {test_path}")
    payload = joblib.load(test_path)
    texts: list[str] = payload["texts"]
    labels: list[int] = payload["labels"]
    return texts, labels


def load_models(models_dir: Path = Path("models_saved")) -> dict[str, Any]:
    """Load the three primary benchmark models from saved artifacts."""
    svc_file = models_dir / "LinearSVC_sparse_fusion_none_light_normalized.joblib"
    stack_file = models_dir / "StackingEnsemble_sparse_fusion_none_light_normalized.joblib"
    cnn_file = models_dir / "ArabicCNN_1D_best_weights.pt"

    if not svc_file.is_file():
        raise FileNotFoundError(f"LinearSVC checkpoint not found at: {svc_file}")
    if not stack_file.is_file():
        raise FileNotFoundError(f"Stacking checkpoint not found at: {stack_file}")
    if not cnn_file.is_file():
        raise FileNotFoundError(f"ArabicCNN_1D checkpoint not found at: {cnn_file}")

    svc_obj = joblib.load(svc_file)
    stack_obj = joblib.load(stack_file)
    cnn_ckpt = torch.load(cnn_file, map_location="cpu", weights_only=False)

    vocab = TextVocabulary()
    vocab.token_to_idx = cnn_ckpt["vocab"]
    vocab.idx_to_token = {v: k for k, v in vocab.token_to_idx.items()}

    cnn_model = ArabicCNN_1D(
        vocab_size=len(vocab),
        embedding_dim=128,
        num_filters=128,
        filter_sizes=(3, 5, 7),
        num_classes=7,
        dropout=0.5,
    )
    cnn_model.load_state_dict(cnn_ckpt["state_dict"])
    cnn_model.eval()

    return {
        "LinearSVC": svc_obj["pipeline"],
        "StackingEnsemble": stack_obj["pipeline"],
        "ArabicCNN_1D": cnn_model,
        "cnn_vocab": vocab,
    }


def predict_cnn(
    model: ArabicCNN_1D,
    texts: Sequence[str],
    vocab: TextVocabulary,
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """Run batched inference using the 1D-CNN PyTorch model."""
    dummy_labels = [0] * len(texts)
    dataset = ArabicTextDataset(texts, dummy_labels, vocab=vocab, max_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()
    preds: list[int] = []
    with torch.no_grad():
        for x_b, _ in loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
    return np.array(preds, dtype=int)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Compute standard multi-class evaluation metrics with macro averaging."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_confusion_matrix_stacking(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: Path,
) -> None:
    """Generate high-resolution annotated confusion matrix for Stacking Ensemble."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cax = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row Normalized Accuracy", fontweight="bold", fontsize=10)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontweight="bold", fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontweight="bold", fontsize=10)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            text_str = f"{count}\n({pct:.1f}%)"
            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                fontweight="bold" if i == j else "normal",
            )

    ax.set_title(
        "Stacking Ensemble Confusion Matrix (Held-out Test Set, N=9,097)",
        fontweight="bold",
        fontsize=12,
        pad=15,
    )
    ax.set_xlabel("Predicted Class", fontweight="bold", fontsize=11, labelpad=8)
    ax.set_ylabel("True Class", fontweight="bold", fontsize=11, labelpad=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix saved to: {output_path}")


def plot_feature_importance(
    pipeline: Any,
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Extract top discriminative features per class from LinearSVC and plot bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clf = pipeline.named_steps["clf"]
    vectorizer = pipeline.named_steps["vectorizer"]
    names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Tahoma", "DejaVu Sans"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), dpi=300)
    axes_flat = axes.flatten()

    for idx, c_name in enumerate(CLASS_NAMES):
        ax = axes_flat[idx]
        top_idx = np.argsort(coef[idx])[-top_k:][::-1]
        top_names = names[top_idx]
        top_scores = coef[idx][top_idx]

        cleaned_names = []
        bar_colors = []
        for name in top_names:
            if name.startswith("word_tfidf__"):
                cleaned_names.append(f"[word] {name[len('word_tfidf__'):]}")
                bar_colors.append("#2b5c8f")
            elif name.startswith("char_tfidf__"):
                cleaned_names.append(f"[char] {name[len('char_tfidf__'):]}")
                bar_colors.append("#d95f02")
            else:
                cleaned_names.append(name)
                bar_colors.append("#7570b3")

        y_pos = np.arange(len(cleaned_names))
        ax.barh(y_pos, top_scores[::-1], color=bar_colors[::-1], alpha=0.85, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cleaned_names[::-1], fontsize=8, fontweight="bold")
        ax.set_title(f"Class: {c_name}", fontweight="bold", fontsize=11, pad=5)
        ax.set_xlabel("LinearSVC Coefficient Weight", fontsize=8, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Empty 8th subplot used as clean visual legend
    ax_legend = axes_flat[7]
    ax_legend.axis("off")
    ax_legend.plot([], [], color="#2b5c8f", lw=8, label="Word N-gram Token")
    ax_legend.plot([], [], color="#d95f02", lw=8, label="Character Subword N-gram")
    ax_legend.legend(loc="center", fontsize=11, frameon=True)

    fig.suptitle(
        "Top 15 Discriminative Features per Class (LinearSVC Sparse Fusion)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature importance chart saved to: {output_path}")


def run_mcnemar_test(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    name_a: str,
    name_b: str,
) -> dict[str, Any]:
    """Compute exact McNemar's test and contingency matrix between two models."""
    correct_a = np.array(y_pred_a) == np.array(y_true)
    correct_b = np.array(y_pred_b) == np.array(y_true)

    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & (~correct_b)))
    n10 = int(np.sum((~correct_a) & correct_b))
    n11 = int(np.sum((~correct_a) & (~correct_b)))

    total_discordant = n01 + n10
    if total_discordant == 0:
        stat = 0.0
        p_chi2 = 1.0
        p_binom = 1.0
    else:
        stat = float((abs(n01 - n10) - 1.0) ** 2 / total_discordant)
        p_chi2 = float(1.0 - stats.chi2.cdf(stat, df=1))
        p_binom = float(
            stats.binomtest(min(n01, n10), total_discordant, p=0.5, alternative="two-sided").pvalue
        )

    return {
        "model_a": name_a,
        "model_b": name_b,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "chi2_stat": stat,
        "p_chi2": p_chi2,
        "p_binom": p_binom,
        "significant": p_binom < 0.05,
    }


def inject_arabic_typographic_noise(
    text: str,
    noise_rate: float,
    rng: np.random.Generator,
) -> str:
    """Inject realistic Arabic keyboard typographical errors with given probability per char."""
    if noise_rate <= 0.0:
        return text
    chars = list(text)
    for i, c in enumerate(chars):
        if c in ARABIC_TYPO_MAP and rng.random() < noise_rate:
            chars[i] = ARABIC_TYPO_MAP[c]
    return "".join(chars)


def evaluate_noise_robustness(
    models: dict[str, Any],
    clean_texts: Sequence[str],
    labels: Sequence[int],
    noise_levels: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
    output_plot: Path = Path("graphics/noise_robustness_curve.png"),
    output_csv: Path = Path("results/results_phase7_noise.csv"),
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate models across synthetic typographic noise levels and plot decay curves."""
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    results_rows: list[dict[str, Any]] = []

    print("\n--- Component C: Synthetic Typographic Noise Robustness Testing ---")
    for rate in noise_levels:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        if rate == 0.0:
            eval_texts = clean_texts
        else:
            eval_texts = [inject_arabic_typographic_noise(t, rate, rng) for t in clean_texts]

        # Predict LinearSVC
        svc_preds = models["LinearSVC"].predict(eval_texts)
        svc_f1 = f1_score(labels, svc_preds, average="macro", zero_division=0)
        svc_acc = accuracy_score(labels, svc_preds)
        results_rows.append({
            "model": "LinearSVC",
            "representation": "sparse_fusion",
            "noise_level": rate,
            "accuracy": round(float(svc_acc), 6),
            "macro_f1": round(float(svc_f1), 6),
        })

        # Predict Stacking
        stack_preds = models["StackingEnsemble"].predict(eval_texts)
        stack_f1 = f1_score(labels, stack_preds, average="macro", zero_division=0)
        stack_acc = accuracy_score(labels, stack_preds)
        results_rows.append({
            "model": "StackingEnsemble",
            "representation": "sparse_fusion",
            "noise_level": rate,
            "accuracy": round(float(stack_acc), 6),
            "macro_f1": round(float(stack_f1), 6),
        })

        # Predict CNN
        cnn_preds = predict_cnn(
            models["ArabicCNN_1D"],
            eval_texts,
            models["cnn_vocab"],
            batch_size=128,
            device="cpu",
        )
        cnn_f1 = f1_score(labels, cnn_preds, average="macro", zero_division=0)
        cnn_acc = accuracy_score(labels, cnn_preds)
        results_rows.append({
            "model": "ArabicCNN_1D",
            "representation": "word_embeddings",
            "noise_level": rate,
            "accuracy": round(float(cnn_acc), 6),
            "macro_f1": round(float(cnn_f1), 6),
        })

        dt = time.perf_counter() - t0
        print(
            f"Noise [{rate * 100:04.1f}%] ({dt:.1f}s) | "
            f"Stacking F1: {stack_f1:.4f} | LinearSVC F1: {svc_f1:.4f} | CNN F1: {cnn_f1:.4f}"
        )

    df_noise = pd.DataFrame(results_rows)
    df_noise.to_csv(output_csv, index=False)
    print(f"Noise robustness metrics logged to: {output_csv}")

    # Plot Decay Curves
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    model_styles = {
        "StackingEnsemble": ("#1f77b4", "o-", "Stacking Ensemble (Word+Char Fusion)"),
        "LinearSVC": ("#2ca02c", "s--", "LinearSVC (Word+Char Fusion)"),
        "ArabicCNN_1D": ("#d62728", "^-.", "ArabicCNN_1D (Word Embeddings)"),
    }

    x_percents = [r * 100 for r in noise_levels]
    for model_name, (color, style, label) in model_styles.items():
        sub = df_noise[df_noise["model"] == model_name]
        f1_vals = [sub[sub["noise_level"] == r]["macro_f1"].values[0] for r in noise_levels]
        ax.plot(x_percents, f1_vals, style, color=color, lw=2.2, markersize=7, label=label)

    ax.set_title(
        "Model Robustness Under Synthetic Typographic Noise (Hypothesis 5)",
        fontweight="bold",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel(
        "Synthetic Typographic Corruption Rate (%)",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_ylabel("Held-Out Test Macro-F1", fontweight="bold", fontsize=11)
    ax.set_xticks(x_percents)
    ax.set_xticklabels([f"{p:.0f}%" for p in x_percents], fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="lower left")

    plt.tight_layout()
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)
    print(f"Noise robustness decay curve saved to: {output_plot}")

    return df_noise


def plot_efficiency_benchmark(
    csv_paths: Sequence[Path],
    output_plot: Path = Path("graphics/efficiency_benchmark.png"),
) -> None:
    """Generate Pareto efficiency scatter plot of Macro-F1 vs Latency across all models."""
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    dfs = []
    for p in csv_paths:
        if p.is_file():
            dfs.append(pd.read_csv(p))

    if not dfs:
        print("No CSV files found for efficiency benchmarking.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    grp = (
        full_df.groupby(["model", "representation", "dim_reduction"])
        .agg({"macro_f1": "mean", "inference_ms_per_doc": "mean"})
        .reset_index()
    )

    family_map = {
        "LinearSVC": ("Linear Classifiers", "#1f77b4", "s"),
        "LogisticRegression": ("Linear Classifiers", "#1f77b4", "s"),
        "SGDClassifier": ("Linear Classifiers", "#1f77b4", "s"),
        "MultinomialNB": ("Probabilistic Models", "#ff7f0e", "^"),
        "BernoulliNB": ("Probabilistic Models", "#ff7f0e", "^"),
        "LightGBM": ("Gradient Boosted Trees", "#2ca02c", "D"),
        "XGBoost": ("Gradient Boosted Trees", "#2ca02c", "D"),
        "HistGradientBoosting": ("Gradient Boosted Trees", "#2ca02c", "D"),
        "StackingEnsemble": ("Meta-Ensembles", "#9467bd", "*"),
        "ArabicCNN_1D": ("Deep Learning (1D-CNN)", "#d62728", "P"),
    }

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    plotted_families: set[str] = set()
    for _, row in grp.iterrows():
        m_name = str(row["model"])
        fam_name, color, marker = family_map.get(m_name, ("Other", "#7f7f7f", "o"))
        label = fam_name if fam_name not in plotted_families else ""
        if label:
            plotted_families.add(fam_name)

        size = 180 if marker == "*" else 90
        ax.scatter(
            row["inference_ms_per_doc"],
            row["macro_f1"],
            color=color,
            marker=marker,
            s=size,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.7,
            label=label if label else None,
        )

    # Highlight and annotate Pareto-optimal models
    annotations = [
        ("LinearSVC (sparse_fusion)", "LinearSVC", "sparse_fusion", (-40, 10)),
        ("Stacking Ensemble", "StackingEnsemble", "sparse_fusion", (-50, -22)),
        ("ArabicCNN_1D (GPU/CPU)", "ArabicCNN_1D", "word_embeddings", (12, -15)),
        ("MultinomialNB (word)", "MultinomialNB", "tfidf", (10, 5)),
    ]

    for label_text, m_name, rep_name, offset in annotations:
        match = grp[(grp["model"] == m_name) & (grp["representation"] == rep_name)]
        if not match.empty:
            x_val = match["inference_ms_per_doc"].values[0]
            y_val = match["macro_f1"].values[0]
            ax.annotate(
                label_text,
                xy=(x_val, y_val),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            )

    ax.set_xscale("log")
    ax.set_title(
        "Pareto Efficiency Frontier: Macro-F1 vs. Inference Latency",
        fontweight="bold",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel(
        "Inference Latency per Document (ms/doc, Log Scale)",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_ylabel("Macro-F1 Score", fontweight="bold", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5, loc="lower left")

    plt.tight_layout()
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)
    print(f"Efficiency benchmark plot saved to: {output_plot}")


def main() -> None:
    """Execute complete Phase 7 evaluation suite."""
    parser = argparse.ArgumentParser(description="Phase 7 Comprehensive Evaluation Suite")
    parser.add_argument("--test-file", default="data/processed/light_normalized_test.joblib")
    parser.add_argument("--models-dir", default="models_saved")
    parser.add_argument("--graphics-dir", default="graphics")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    graphics_dir = Path(args.graphics_dir)
    results_dir = Path(args.results_dir)
    graphics_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 7: COMPREHENSIVE EVALUATION SUITE & STATISTICAL TESTING")
    print("=" * 80)

    # 1. Ingest test partition (Strict zero data leakage)
    print("\n[1/6] Loading held-out test split (Strict inference mode)...")
    clean_texts, clean_labels = load_clean_test_split(Path(args.test_file))
    print(f"      Loaded {len(clean_texts):,} articles across {len(CLASS_NAMES)} classes.")

    # 2. Ingest models
    print("\n[2/6] Loading benchmark models from artifacts...")
    models = load_models(Path(args.models_dir))
    print("      Loaded LinearSVC, StackingEnsemble, and ArabicCNN_1D successfully.")

    # 3. Clean test evaluation
    print("\n[3/6] Evaluating models on clean held-out test set...")
    t0 = time.perf_counter()
    y_pred_svc = models["LinearSVC"].predict(clean_texts)
    svc_clean_metrics = compute_metrics(clean_labels, y_pred_svc)
    svc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_stack = models["StackingEnsemble"].predict(clean_texts)
    stack_clean_metrics = compute_metrics(clean_labels, y_pred_stack)
    stack_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_cnn = predict_cnn(
        models["ArabicCNN_1D"],
        clean_texts,
        models["cnn_vocab"],
        batch_size=128,
        device="cpu",
    )
    cnn_clean_metrics = compute_metrics(clean_labels, y_pred_cnn)
    cnn_time = time.perf_counter() - t0

    print("-" * 75)
    hdr = f"{'Model':<16} | {'Accuracy':<10} | {'Precision':<10} | "
    hdr += f"{'Recall':<10} | {'Macro-F1':<10}"
    print(hdr)
    print("-" * 75)
    print(
        f"{'LinearSVC':<20} | "
        f"{svc_clean_metrics['accuracy']:<10.4f} | "
        f"{svc_clean_metrics['macro_precision']:<10.4f} | "
        f"{svc_clean_metrics['macro_recall']:<10.4f} | "
        f"{svc_clean_metrics['macro_f1']:<10.4f}"
    )
    print(
        f"{'StackingEnsemble':<20} | "
        f"{stack_clean_metrics['accuracy']:<10.4f} | "
        f"{stack_clean_metrics['macro_precision']:<10.4f} | "
        f"{stack_clean_metrics['macro_recall']:<10.4f} | "
        f"{stack_clean_metrics['macro_f1']:<10.4f}"
    )
    print(
        f"{'ArabicCNN_1D':<20} | "
        f"{cnn_clean_metrics['accuracy']:<10.4f} | "
        f"{cnn_clean_metrics['macro_precision']:<10.4f} | "
        f"{cnn_clean_metrics['macro_recall']:<10.4f} | "
        f"{cnn_clean_metrics['macro_f1']:<10.4f}"
    )
    print("-" * 75)

    # Save clean test metrics to results_phase7_test_clean.csv
    clean_csv = results_dir / "results_phase7_test_clean.csv"
    with clean_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "eval_time_sec",
        ])
        writer.writerow([
            "LinearSVC",
            svc_clean_metrics["accuracy"],
            svc_clean_metrics["macro_precision"],
            svc_clean_metrics["macro_recall"],
            svc_clean_metrics["macro_f1"],
            round(svc_time, 2),
        ])
        writer.writerow([
            "StackingEnsemble",
            stack_clean_metrics["accuracy"],
            stack_clean_metrics["macro_precision"],
            stack_clean_metrics["macro_recall"],
            stack_clean_metrics["macro_f1"],
            round(stack_time, 2),
        ])
        writer.writerow([
            "ArabicCNN_1D",
            cnn_clean_metrics["accuracy"],
            cnn_clean_metrics["macro_precision"],
            cnn_clean_metrics["macro_recall"],
            cnn_clean_metrics["macro_f1"],
            round(cnn_time, 2),
        ])
    print(f"Clean test benchmark logged to: {clean_csv}")

    # 4. Component A: Error Analysis & Graphics
    print("\n[4/6] Generating Component A Graphics (Confusion Matrix & Feature Importance)...")
    plot_confusion_matrix_stacking(
        clean_labels,
        y_pred_stack,
        graphics_dir / "confusion_matrix_stacking.png",
    )
    plot_feature_importance(
        models["LinearSVC"],
        graphics_dir / "feature_importance.png",
        top_k=15,
    )

    # 5. Component B: Statistical Testing (McNemar's Test)
    print("\n[5/6] Conducting Component B: McNemar's Statistical Significance Testing...")
    mcnemar_cnn = run_mcnemar_test(
        clean_labels,
        y_pred_stack,
        y_pred_cnn,
        "StackingEnsemble",
        "ArabicCNN_1D",
    )
    mcnemar_svc = run_mcnemar_test(
        clean_labels,
        y_pred_stack,
        y_pred_svc,
        "StackingEnsemble",
        "LinearSVC",
    )

    mcnemar_file = results_dir / "mcnemar_tests.txt"
    with mcnemar_file.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MCNEMAR STATISTICAL SIGNIFICANCE TESTS ON HELD-OUT TEST SPLIT (N=9,097)\n")
        f.write("=" * 80 + "\n\n")

        for test_res in [mcnemar_cnn, mcnemar_svc]:
            f.write(f"Comparison: {test_res['model_a']} vs. {test_res['model_b']}\n")
            f.write("-" * 60 + "\n")
            f.write("  Contingency Table (2x2):\n")
            ma = test_res["model_a"]
            mb = test_res["model_b"]
            f.write(f"    Both Correct (n00)                       : {test_res['n00']}\n")
            f.write(f"    {ma} Correct, {mb} Incorrect (n01): {test_res['n01']}\n")
            f.write(f"    {ma} Incorrect, {mb} Correct (n10): {test_res['n10']}\n")
            f.write(f"    Both Incorrect (n11)                     : {test_res['n11']}\n")
            f.write(f"  McNemar Chi2 (Edwards CC)             : {test_res['chi2_stat']:.4f}\n")
            f.write(f"  Chi-Square Asymptotic p-value               : {test_res['p_chi2']:.4e}\n")
            f.write(f"  Exact Two-Sided Binomial p-value            : {test_res['p_binom']:.4e}\n")
            sig_str = (
                "STATISTICALLY SIGNIFICANT (p < 0.05)"
                if test_res["significant"]
                else "NOT SIGNIFICANT (p >= 0.05)"
            )
            f.write(f"  Conclusion at alpha = 0.05                  : {sig_str}\n\n")

    print(f"Statistical test results logged to: {mcnemar_file}")

    # 6. Component C: Synthetic Typographic Noise (Hypothesis 5)
    evaluate_noise_robustness(
        models=models,
        clean_texts=clean_texts,
        labels=clean_labels,
        noise_levels=(0.0, 0.05, 0.10, 0.20),
        output_plot=graphics_dir / "noise_robustness_curve.png",
        output_csv=results_dir / "results_phase7_noise.csv",
    )

    # 7. Component D: Efficiency Benchmarking
    print("\n[6/6] Generating Component D: Efficiency Benchmark Scatter Plot...")
    plot_efficiency_benchmark(
        csv_paths=[
            results_dir / "results_classical.csv",
            results_dir / "results_features.csv",
            results_dir / "results_advanced.csv",
            results_dir / "results_dl.csv",
        ],
        output_plot=graphics_dir / "efficiency_benchmark.png",
    )

    print("\n" + "=" * 80)
    print("PHASE 7 EVALUATION SUITE SUCCESSFULLY COMPLETED")
    print("All graphics written to graphics/ and all numerical logs written to results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
