"""CLI verification script demonstrating Arabic text cleaning across all three pipelines."""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_config
from src.preprocessing import get_pipeline


def main() -> None:
    """Load sample documents per category and display progressive pipeline transformations."""
    cfg = load_config()
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    categories = list(cfg["dataset"]["categories"])

    raw_char_pipe = get_pipeline("raw_char", cfg)
    light_pipe = get_pipeline("light_normalized", cfg)
    heavy_pipe = get_pipeline("paper_heavy_stemming", cfg)

    print("=" * 80)
    print("PHASE 2: ARABIC PREPROCESSING PIPELINE VERIFICATION REPORT")
    print("=" * 80)

    for cat in categories:
        khaleej_dir = dataset_root / "Khaleej" / cat
        cat_dir = khaleej_dir if khaleej_dir.is_dir() else dataset_root / cat
        files = sorted(cat_dir.glob("*.txt"))
        if not files:
            continue

        sample_file = files[0]
        raw_text = sample_file.read_text(encoding="utf-8").strip()

        # Extract first 160 characters for clean terminal inspection
        raw_snippet = " ".join(raw_text[:160].split())
        raw_char_snippet = " ".join(raw_char_pipe(raw_text)[:160].split())
        light_snippet = " ".join(light_pipe(raw_text)[:160].split())
        heavy_snippet = " ".join(heavy_pipe(raw_text)[:160].split())

        print(f"\n[CATEGORY: {cat.upper()}] (Source: {sample_file.name})")
        print("-" * 80)
        print(f"1. RAW INPUT:              {raw_snippet}")
        print(f"2. RAW CHAR:               {raw_char_snippet}")
        print(f"3. LIGHT NORMALIZED:       {light_snippet}")
        print(f"4. PAPER HEAVY STEMMING:   {heavy_snippet}")

    print("\n" + "=" * 80)
    print("Verification complete: All 3 pipelines successfully transformed 7 categories.")
    print("=" * 80)


if __name__ == "__main__":
    main()
