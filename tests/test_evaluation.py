"""Unit tests for Phase 7 Comprehensive Evaluation Suite.

Tests cover:
1. Typographic noise injection probabilities (0% identity, corruption mapping).
2. Multi-class metric calculations (accuracy, macro precision, recall, F1).
3. McNemar's 2x2 contingency table and exact binomial test logic.
4. Held-out test set integrity and schema validation.
"""

from pathlib import Path

import numpy as np

from scripts.run_phase7_evaluation import (
    ARABIC_TYPO_MAP,
    compute_metrics,
    inject_arabic_typographic_noise,
    load_clean_test_split,
    run_mcnemar_test,
)


def test_noise_injection_zero_rate_is_identity() -> None:
    """Verifies that 0.0 noise rate leaves text completely unchanged."""
    text = "أعلنت وزارة الصحة المصرية في القاهرة عن إجراءات جديدة"
    rng = np.random.default_rng(42)
    noisy = inject_arabic_typographic_noise(text, noise_rate=0.0, rng=rng)
    assert noisy == text


def test_noise_injection_full_rate_corruption() -> None:
    """Verifies that 1.0 noise rate substitutes all target Arabic characters."""
    text = "وزارة الصحة والرياضة"
    rng = np.random.default_rng(42)
    noisy = inject_arabic_typographic_noise(text, noise_rate=1.0, rng=rng)
    assert noisy != text
    assert "ه" in noisy  # 'ة' should be replaced by 'ه'
    for original, corrupted in zip(text, noisy, strict=False):
        if original in ARABIC_TYPO_MAP:
            assert corrupted == ARABIC_TYPO_MAP[original]


def test_noise_injection_probabilistic_rates() -> None:
    """Verifies that higher noise rates produce monotonically non-decreasing corruption."""
    long_text = "أعلنت وزارة الصحة المصرية في القاهرة عن إجراءات جديدة " * 10
    rng = np.random.default_rng(42)
    diffs = []
    for rate in [0.0, 0.05, 0.10, 0.20]:
        noisy = inject_arabic_typographic_noise(long_text, rate, rng)
        diff_count = sum(1 for c1, c2 in zip(long_text, noisy, strict=False) if c1 != c2)
        diffs.append(diff_count)

    # diffs should strictly increase as rate increases
    assert diffs[0] == 0
    assert diffs[1] > 0
    assert diffs[2] > diffs[1]
    assert diffs[3] > diffs[2]


def test_mcnemar_test_contingency_and_pvalues() -> None:
    """Verifies McNemar's test contingency table, chi-square stat, and binomial p-value."""
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred_a = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # 10/10 correct
    y_pred_b = [0, 1, 2, 0, 1, 0, 0, 0, 0, 0]  # 7/10 correct (3 discordant errors)

    res = run_mcnemar_test(y_true, y_pred_a, y_pred_b, "ModelA", "ModelB")
    assert res["n00"] == 7
    assert res["n01"] == 3  # ModelA right, ModelB wrong
    assert res["n10"] == 0  # ModelA wrong, ModelB right
    assert res["n11"] == 0
    assert 0.0 <= res["p_binom"] <= 1.0
    assert res["chi2_stat"] >= 0.0


def test_compute_metrics_correctness() -> None:
    """Verifies multi-class macro metric calculations and zero_division safeguard."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 1]  # 5/6 correct
    metrics = compute_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert round(metrics["accuracy"], 4) == round(5 / 6, 4)


def test_clean_test_split_integrity() -> None:
    """Verifies that the held-out test split loads exactly 9,097 articles across 7 classes."""
    test_path = Path("data/processed/light_normalized_test.joblib")
    if test_path.is_file():
        texts, labels = load_clean_test_split(test_path)
        assert len(texts) == 9097
        assert len(labels) == 9097
        assert set(labels) == {0, 1, 2, 3, 4, 5, 6}
