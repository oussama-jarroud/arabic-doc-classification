"""Dataset scanning, exact SHA-256 deduplication, stratified partitioning, and split loading."""

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration dictionary from disk to avoid hardcoding experiment parameters."""
    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_category_directory(raw_dir: Path, category: str) -> Path:
    """Locate category folder supporting both flat raw and nested Khaleej folder structures."""
    candidate_khaleej = raw_dir / "Khaleej" / category
    if candidate_khaleej.is_dir():
        return candidate_khaleej
    candidate_direct = raw_dir / category
    if candidate_direct.is_dir():
        return candidate_direct
    raise FileNotFoundError(
        f"Category directory for '{category}' not found in '{raw_dir}' or '{raw_dir / 'Khaleej'}'."
    )


def scan_and_deduplicate(
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    """Filter empty files and remove exact content duplicates via SHA-256."""
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    dataset_root_rel = Path(cfg["dataset"]["dataset_root"])
    raw_dir_path = root / dataset_root_rel
    categories: list[str] = list(cfg["dataset"]["categories"])
    expected_per_class: int = int(cfg["dataset"]["expected_raw_per_class"])

    clean_files_by_cat: dict[str, list[Path]] = {}
    stats_by_cat: dict[str, dict[str, int]] = {}
    seen_hashes: set[str] = set()

    for cat in categories:
        cat_dir = _resolve_category_directory(raw_dir_path, cat)
        files = sorted(cat_dir.glob("*.txt"))
        raw_count = len(files)

        if raw_count < expected_per_class:
            raise ValueError(
                f"Category '{cat}' contains {raw_count} documents, fewer than "
                f"expected {expected_per_class}."
            )

        clean_files: list[Path] = []
        empty_count = 0
        duplicate_count = 0

        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            normalized_text = " ".join(content.split())

            if not normalized_text:
                empty_count += 1
                continue

            content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                duplicate_count += 1
                continue

            seen_hashes.add(content_hash)
            clean_files.append(file_path)

        clean_files_by_cat[cat] = clean_files
        stats_by_cat[cat] = {
            "raw_count": raw_count,
            "empty_count": empty_count,
            "duplicates_removed": duplicate_count,
            "clean_count": len(clean_files),
        }

    return clean_files_by_cat, stats_by_cat


def build_split_if_needed(
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> Path:
    """Create fixed 80/20 dev/test stratified split once on deduplicated corpus."""
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    split_file_rel = Path(cfg["dataset"]["split_file"])
    split_file_path = root / split_file_rel

    if split_file_path.is_file():
        return split_file_path

    clean_files_by_cat, stats_by_cat = scan_and_deduplicate(cfg, project_root=root)

    all_files: list[Path] = []
    all_categories: list[str] = []

    for cat, files in clean_files_by_cat.items():
        all_files.extend(files)
        all_categories.extend([cat] * len(files))

    split_ratios: dict[str, float] = cfg["dataset"]["split_ratios"]
    test_ratio = float(split_ratios["test"])
    seed = int(cfg["dataset"]["seed"])

    dev_files, test_files, dev_cats, test_cats = train_test_split(
        all_files,
        all_categories,
        test_size=test_ratio,
        random_state=seed,
        stratify=all_categories,
    )

    def _to_records(files: list[Path], cats: list[str]) -> list[dict[str, str]]:
        return [
            {
                "path": file.resolve().relative_to(root.resolve()).as_posix(),
                "category": cat,
            }
            for file, cat in zip(files, cats, strict=True)
        ]

    payload: dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "split_ratios": split_ratios,
            "cv_folds": int(cfg["dataset"]["cv_folds"]),
            "categories": list(cfg["dataset"]["categories"]),
            "total_raw": sum(s["raw_count"] for s in stats_by_cat.values()),
            "total_clean": len(all_files),
            "deduplication_stats": stats_by_cat,
        },
        "dev": _to_records(dev_files, dev_cats),
        "test": _to_records(test_files, test_cats),
    }

    split_file_path.parent.mkdir(parents=True, exist_ok=True)
    with split_file_path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, ensure_ascii=False)

    return split_file_path


def load_split_records(
    split_name: Literal["dev", "test"],
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> list[dict[str, str]]:
    """Retrieve raw partitioned split record metadata from the fixed split JSON."""
    valid_splits = ("dev", "test")
    if split_name not in valid_splits:
        raise ValueError(f"Unknown split name '{split_name}'. Expected one of {valid_splits}.")

    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    split_file_path = build_split_if_needed(cfg, project_root=root)
    with split_file_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records: list[dict[str, str]] = payload[split_name]
    return records


def load_split_paths(
    split_name: Literal["dev", "test"],
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> list[Path]:
    """Provide resolved file paths for a partition to support file-level inspection."""
    root = Path.cwd() if project_root is None else Path(project_root)
    records = load_split_records(split_name, config=config, project_root=root)
    return [root / record["path"] for record in records]


def load_split(
    split_name: Literal["dev", "test"],
    config: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> tuple[list[str], list[int]]:
    """Load text contents and categorical labels for a partition in deterministic integer format."""
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    categories: list[str] = list(cfg["dataset"]["categories"])
    category_to_id: dict[str, int] = {cat: idx for idx, cat in enumerate(categories)}

    records = load_split_records(split_name, config=cfg, project_root=root)

    texts: list[str] = []
    labels: list[int] = []

    for record in records:
        file_path = root / record["path"]
        texts.append(file_path.read_text(encoding="utf-8"))
        labels.append(category_to_id[record["category"]])

    return texts, labels


def get_cv_folds(
    labels: Sequence[int] | np.ndarray,
    config: dict[str, Any] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic StratifiedKFold index splits for leak-free cross-validation on dev."""
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    cv_folds = int(cfg["dataset"]["cv_folds"])
    seed = int(cfg["dataset"]["seed"])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(labels))
    yield from skf.split(dummy_x, labels)


def load_processed_split(
    pipeline_name: str,
    split_name: Literal["dev", "test"],
    config: dict[str, Any] | Path | str | None = None,
    project_root: Path | None = None,
) -> tuple[list[str], list[int]]:
    """Load serialized preprocessed texts and labels to bypass repetitive text transformations."""
    valid_splits = ("dev", "test")
    if split_name not in valid_splits:
        raise ValueError(f"Unknown split name '{split_name}'. Expected one of {valid_splits}.")

    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    processed_dir_rel = Path(cfg["dataset"].get("processed_dir", "data/processed"))
    processed_dir = root / processed_dir_rel

    file_name = f"{pipeline_name}_{split_name}.joblib"
    file_path = processed_dir / file_name

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Processed data file not found: '{file_path}'. "
            f"Please run 'python scripts/run_preprocessing.py' first."
        )

    payload = joblib.load(file_path)
    texts: list[str] = payload["texts"]
    labels: list[int] = payload["labels"]
    return texts, labels

