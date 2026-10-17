"""Batch text preprocessing across fixed split documents and serialization to disk."""

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_config, load_split
from src.preprocessing import batch_preprocess_texts, get_pipeline


def run_batch_preprocessing(
    config: dict[str, Any] | Path | str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Execute batch preprocessing for pipelines and persist compressed artifacts to disk."""
    cfg = load_config(config) if isinstance(config, (str, Path)) or config is None else config
    root = Path.cwd() if project_root is None else Path(project_root)

    processed_dir_rel = Path(cfg["dataset"].get("processed_dir", "data/processed"))
    processed_dir = root / processed_dir_rel
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path_rel = Path(
        cfg.get("preprocessing", {}).get("metadata_file", "results/preprocessing_metadata.json")
    )
    metadata_path = root / metadata_path_rel
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    target_pipelines: list[str] = list(
        cfg.get("preprocessing", {}).get(
            "target_pipelines", ["light_normalized", "paper_heavy_stemming"]
        )
    )

    print("=" * 80)
    print("PHASE 2: BATCH PREPROCESSING & SERIALIZATION TO DISK")
    print("=" * 80)
    print("Loading raw partitions from fixed split...")
    dev_texts, dev_labels = load_split("dev", cfg, project_root=root)
    test_texts, test_labels = load_split("test", cfg, project_root=root)
    print(f"Loaded: dev={len(dev_texts)} docs, test={len(test_texts)} docs.")

    pipeline_metadata: dict[str, Any] = {}

    for pipeline_name in target_pipelines:
        print(f"\n>>> Running pipeline: '{pipeline_name}'")
        pipe_fn = get_pipeline(pipeline_name, cfg)

        start_dev = time.perf_counter()
        dev_cleaned = batch_preprocess_texts(dev_texts, pipe_fn)
        elapsed_dev = time.perf_counter() - start_dev
        print(f"    - Preprocessed dev ({len(dev_cleaned)} docs) in {elapsed_dev:.2f}s")

        start_test = time.perf_counter()
        test_cleaned = batch_preprocess_texts(test_texts, pipe_fn)
        elapsed_test = time.perf_counter() - start_test
        print(f"    - Preprocessed test ({len(test_cleaned)} docs) in {elapsed_test:.2f}s")

        dev_file = processed_dir / f"{pipeline_name}_dev.joblib"
        test_file = processed_dir / f"{pipeline_name}_test.joblib"

        joblib.dump({"texts": dev_cleaned, "labels": dev_labels}, dev_file, compress=3)
        joblib.dump({"texts": test_cleaned, "labels": test_labels}, test_file, compress=3)

        dev_size_mb = dev_file.stat().st_size / (1024 * 1024)
        test_size_mb = test_file.stat().st_size / (1024 * 1024)
        print(f"    - Serialized: {dev_file.name} ({dev_size_mb:.2f} MB)")
        print(f"    - Serialized: {test_file.name} ({test_size_mb:.2f} MB)")

        pipeline_metadata[pipeline_name] = {
            "dev_file": dev_file.relative_to(root).as_posix(),
            "test_file": test_file.relative_to(root).as_posix(),
            "dev_size_mb": round(dev_size_mb, 2),
            "test_size_mb": round(test_size_mb, 2),
            "dev_time_sec": round(elapsed_dev, 2),
            "test_time_sec": round(elapsed_test, 2),
            "total_time_sec": round(elapsed_dev + elapsed_test, 2),
        }

    overall_metadata: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "split_file": cfg["dataset"]["split_file"],
        "dev_document_count": len(dev_texts),
        "test_document_count": len(test_texts),
        "pipelines": pipeline_metadata,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(overall_metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to: {metadata_path}")
    print("=" * 80)
    print("Batch preprocessing complete! All partitions serialized to data/processed/")
    print("=" * 80)

    return overall_metadata


def main() -> None:
    """CLI entrypoint for batch preprocessing workflow."""
    run_batch_preprocessing()


if __name__ == "__main__":
    main()
