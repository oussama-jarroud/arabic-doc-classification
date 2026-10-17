"""Unit tests verifying parallel batch preprocessing, disk serialization, and split loaders."""

import json
from pathlib import Path
from typing import Any

import joblib
import pytest

from src.data import load_processed_split, load_split
from src.preprocessing import batch_preprocess_texts, get_pipeline


@pytest.fixture
def mock_processed_env(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Build a mock dataset environment with dev/test partitions and configuration."""
    categories = ["Culture", "Finance", "Medical", "Politics", "Religion", "Sports", "Tech"]
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    splits_dir = tmp_path / "splits"
    results_dir = tmp_path / "results"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dev_records = []
    test_records = []

    for cat in categories:
        cat_dir = raw_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        # 3 dev files and 1 test file per category
        for i in range(3):
            file_name = f"dev_{i:02d}.txt"
            (cat_dir / file_name).write_text(f"مقال تجريبي رقم {i} في {cat}!", encoding="utf-8")
            dev_records.append({"path": f"data/raw/{cat}/{file_name}", "category": cat})

        test_file = "test_00.txt"
        (cat_dir / test_file).write_text(f"مقال اختبار نهائي في {cat}!", encoding="utf-8")
        test_records.append({"path": f"data/raw/{cat}/{test_file}", "category": cat})

    split_payload = {
        "metadata": {"seed": 42},
        "dev": dev_records,
        "test": test_records,
    }
    split_file = splits_dir / "split_seed42.json"
    split_file.write_text(json.dumps(split_payload, ensure_ascii=False), encoding="utf-8")

    cfg: dict[str, Any] = {
        "dataset": {
            "dataset_root": "data/raw",
            "split_file": "splits/split_seed42.json",
            "categories": categories,
            "processed_dir": "data/processed",
        },
        "preprocessing_pipelines": {
            "light_normalized": [
                "remove_diacritics",
                "remove_tatweel",
                "normalize_hamza_alef",
                "remove_non_arabic_punct",
                "remove_stopwords",
            ],
            "paper_heavy_stemming": [
                "remove_diacritics",
                "remove_tatweel",
                "normalize_hamza_alef",
                "remove_non_arabic_punct",
                "remove_stopwords",
                "isri_stem",
            ],
        },
        "preprocessing": {
            "target_pipelines": ["light_normalized", "paper_heavy_stemming"],
            "serialization_format": "joblib",
            "metadata_file": "results/preprocessing_metadata.json",
        },
    }

    return tmp_path, cfg


def test_batch_preprocess_texts_transforms_and_preserves_order() -> None:
    pipe = get_pipeline("light_normalized")
    texts = [
        "كَتَبَ الطَّالِبُ فِي المَدْرَسَةِ!",
        "ارْتَفَعَ سُوقُ المَالِ!",
        "جـامـعـة الـقـاهـرة",
    ]

    single_results = [pipe(t) for t in texts]
    batch_results = batch_preprocess_texts(texts, pipe, chunk_size=2, n_jobs=1)

    assert batch_results == single_results
    assert batch_results[0] == "كتب الطالب المدرسه"
    assert batch_results[1] == "ارتفع سوق المال"
    assert batch_results[2] == "جامعه القاهره"


def test_batch_preprocess_texts_empty_list() -> None:
    pipe = get_pipeline("light_normalized")
    assert batch_preprocess_texts([], pipe) == []


def test_load_processed_split_missing_file_raises(
    mock_processed_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_processed_env
    with pytest.raises(FileNotFoundError, match="Processed data file not found"):
        load_processed_split("light_normalized", "dev", cfg, project_root=tmp_path)


def test_load_processed_split_invalid_split_raises(
    mock_processed_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_processed_env
    with pytest.raises(ValueError, match="Unknown split name"):
        # Type ignore to test runtime argument safety
        load_processed_split("light_normalized", "val", cfg, project_root=tmp_path)  # type: ignore[arg-type]


def test_serialization_and_loader_roundtrip(
    mock_processed_env: tuple[Path, dict[str, Any]],
) -> None:
    tmp_path, cfg = mock_processed_env
    processed_dir = tmp_path / "data" / "processed"

    dev_texts, dev_labels = load_split("dev", cfg, project_root=tmp_path)
    pipe = get_pipeline("light_normalized", cfg)
    cleaned_dev = batch_preprocess_texts(dev_texts, pipe, n_jobs=1)

    # Save to disk
    target_file = processed_dir / "light_normalized_dev.joblib"
    joblib.dump({"texts": cleaned_dev, "labels": dev_labels}, target_file)
    assert target_file.is_file()

    # Load via load_processed_split
    loaded_texts, loaded_labels = load_processed_split(
        "light_normalized",
        "dev",
        cfg,
        project_root=tmp_path,
    )

    assert loaded_texts == cleaned_dev
    assert loaded_labels == dev_labels
    assert len(loaded_texts) == len(dev_texts)


def test_run_batch_preprocessing_end_to_end(
    mock_processed_env: tuple[Path, dict[str, Any]],
) -> None:
    from scripts.run_preprocessing import run_batch_preprocessing

    tmp_path, cfg = mock_processed_env
    meta = run_batch_preprocessing(cfg, project_root=tmp_path)

    assert meta["dev_document_count"] == 21
    assert meta["test_document_count"] == 7
    assert "light_normalized" in meta["pipelines"]
    assert "paper_heavy_stemming" in meta["pipelines"]

    processed_dir = tmp_path / "data" / "processed"
    assert (processed_dir / "light_normalized_dev.joblib").is_file()
    assert (processed_dir / "light_normalized_test.joblib").is_file()
    assert (processed_dir / "paper_heavy_stemming_dev.joblib").is_file()
    assert (processed_dir / "paper_heavy_stemming_test.joblib").is_file()

    dev_texts, dev_labels = load_processed_split(
        "paper_heavy_stemming",
        "dev",
        cfg,
        project_root=tmp_path,
    )
    assert len(dev_texts) == 21
    assert len(dev_labels) == 21

