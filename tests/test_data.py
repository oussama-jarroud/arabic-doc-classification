"""Unit tests verifying dataset scanning, SHA-256 deduplication, 80/20 splitting, and CV folds."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.data import (
    build_split_if_needed,
    get_cv_folds,
    load_config,
    load_split,
    load_split_paths,
    scan_and_deduplicate,
)


@pytest.fixture
def mock_dataset_env(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Create a synthetic Arabic corpus with duplicates and empty files for fast testing."""
    categories = ["Culture", "Finance", "Medical", "Politics", "Religion", "Sports", "Tech"]
    expected_per_class = 20
    raw_dir = tmp_path / "data" / "raw"

    for cat_idx, cat in enumerate(categories):
        cat_dir = raw_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        for i in range(expected_per_class):
            file_path = cat_dir / f"doc_{i:03d}.txt"
            # Ensure unique Arabic content per document
            file_path.write_text(
                f"مقال فريد رقم {i} يتبع تصنيف {cat} باللغة العربية الفصحى.",
                encoding="utf-8",
            )

        # In the first category ("Culture"), add 1 empty file and 2 duplicate files
        if cat_idx == 0:
            empty_file = cat_dir / "doc_empty.txt"
            empty_file.write_text("   \n  \t  ", encoding="utf-8")

            dup_1 = cat_dir / "doc_dup_1.txt"
            dup_2 = cat_dir / "doc_dup_2.txt"
            dup_content = "خبر مكرر تم نشره في وكالات أنباء متعددة بحذافيره."
            dup_1.write_text(dup_content, encoding="utf-8")
            dup_2.write_text(dup_content, encoding="utf-8")

    cfg: dict[str, Any] = {
        "dataset": {
            "dataset_root": "data/raw",
            "split_file": "splits/split_seed42.json",
            "expected_raw_per_class": expected_per_class,
            "categories": categories,
            "split_ratios": {
                "dev": 0.80,
                "test": 0.20,
            },
            "cv_folds": 5,
            "seed": 42,
        }
    }

    return tmp_path, cfg


def test_scan_and_deduplicate_filters_empty_and_duplicates(
    mock_dataset_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_dataset_env
    clean_files, stats = scan_and_deduplicate(cfg, project_root=tmp_path)

    # In Culture: 20 unique docs + 1 empty + 2 dups (one unique retained, one removed) = 21 clean
    assert stats["Culture"]["raw_count"] == 23
    assert stats["Culture"]["empty_count"] == 1
    assert stats["Culture"]["duplicates_removed"] == 1
    assert stats["Culture"]["clean_count"] == 21
    assert len(clean_files["Culture"]) == 21

    # Remaining 6 categories have 20 clean docs each
    for cat in cfg["dataset"]["categories"][1:]:
        assert stats[cat]["clean_count"] == 20
        assert stats[cat]["duplicates_removed"] == 0
        assert stats[cat]["empty_count"] == 0


def test_build_split_creates_stratified_80_20_split_with_posix_paths(
    mock_dataset_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_dataset_env
    split_file = build_split_if_needed(cfg, project_root=tmp_path)

    assert split_file.is_file()

    with split_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert "dev" in data
    assert "test" in data
    assert "metadata" in data

    # Verify POSIX relative paths format
    for record in data["dev"] + data["test"]:
        path_str = record["path"]
        assert "\\" not in path_str, f"Found Windows backslash in POSIX path: {path_str}"
        assert path_str.startswith("data/raw/")

    total_clean = 21 + (6 * 20)  # 141 docs
    assert len(data["dev"]) + len(data["test"]) == total_clean
    # Test set is 20% (+/- 1 due to per-class integer rounding in stratified split)
    assert abs(len(data["test"]) - (total_clean * 0.20)) <= 1.0
    assert abs(len(data["dev"]) - (total_clean * 0.80)) <= 1.0


def test_build_split_is_idempotent(
    mock_dataset_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_dataset_env
    split_file = build_split_if_needed(cfg, project_root=tmp_path)

    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["metadata"]["sentinel"] = "DO_NOT_OVERWRITE"
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    recalled_file = build_split_if_needed(cfg, project_root=tmp_path)
    assert recalled_file == split_file

    with split_file.open("r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["metadata"].get("sentinel") == "DO_NOT_OVERWRITE"


def test_zero_data_leakage_between_dev_and_test(
    mock_dataset_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_dataset_env

    dev_paths = set(load_split_paths("dev", cfg, project_root=tmp_path))
    test_paths = set(load_split_paths("test", cfg, project_root=tmp_path))

    for p in dev_paths | test_paths:
        assert p.is_file()

    assert len(dev_paths & test_paths) == 0, "Data leakage detected between dev and test splits!"


def test_get_cv_folds_yields_stratified_balanced_folds(
    mock_dataset_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_dataset_env
    _, labels = load_split("dev", cfg, project_root=tmp_path)

    folds = list(get_cv_folds(labels, cfg))
    assert len(folds) == 5

    all_val_indices: list[int] = []
    for train_idx, val_idx in folds:
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(train_idx) + len(val_idx) == len(labels)
        all_val_indices.extend(val_idx)

    # All dev indices must appear in validation exactly once
    assert sorted(all_val_indices) == list(range(len(labels)))


def test_load_split_returns_utf8_texts_and_integer_labels(
    mock_dataset_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_dataset_env
    texts, labels = load_split("dev", cfg, project_root=tmp_path)

    assert len(texts) == len(labels)
    assert all(isinstance(t, str) for t in texts)
    assert all(isinstance(lbl, int) for lbl in labels)
    assert all(0 <= lbl < len(cfg["dataset"]["categories"]) for lbl in labels)

    assert "مقال فريد" in texts[0] or "خبر مكرر" in texts[0]
    assert "باللغة العربية" in texts[0] or "وكالات أنباء" in texts[0]


def test_load_split_invalid_name_raises(
    mock_dataset_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_dataset_env
    with pytest.raises(ValueError, match="Unknown split name"):
        # Type ignore to test runtime argument safety
        load_split("validation", cfg, project_root=tmp_path)  # type: ignore[arg-type]


def test_load_config_default_fallback() -> None:
    config = load_config("configs/default.yaml")
    assert "dataset" in config
    assert len(config["dataset"]["categories"]) == 7
    assert config["dataset"]["cv_folds"] == 5
