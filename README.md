# Arabic Document Classification — Research Benchmark

Reproduction, critique, and extension of Muaad et al. (2022) on the Khaleej (SANAD) dataset (7 categories, 45,500 Arabic news articles).

## Collaborator Setup
1. Clone the repository and create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Place the raw Khaleej dataset folders (Culture, Finance, Medical, Politics, Religion, Sports, Tech) inside `data/raw/`.

3. Run linter and unit tests before committing:
   ```powershell
   ruff check .
   pytest
   ```

## Repository Structure
- `configs/`: Central YAML experiment configs.
- `src/`: Modular codebase (`preprocessing`, `features`, `models`, `eval`).
- `splits/`: Fixed stratified split files (e.g., `split_seed42.json`).
- `results/`: Experiment metrics, tables, and figures.
- `docs/`: Hypotheses, problem statements, and literature benchmarks.
- `tests/`: Deterministic pytest test suites.
- `scripts/`: Standalone execution scripts (e.g. EDA, benchmark runners).
