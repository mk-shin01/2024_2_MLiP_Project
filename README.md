# Project Package

This repository was generated from `ML_Baseline8229temp.ipynb` by extracting code cells into Python modules under `src/project_pkg/`.

## Layout
```
repo/
  ├─ src/project_pkg/
  │   ├─ __init__.py
  │   ├─ data.py
  │   ├─ models.py
  │   ├─ train.py
  │   ├─ eval.py
  │   ├─ utils.py
  │   └─ main.py
  ├─ notebooks/ML_Baseline8229temp.ipynb
  ├─ scripts/
  ├─ tests/
  ├─ requirements.txt
  └─ README.md
```

## Usage
```bash
pip install -r requirements.txt
python -m project_pkg.main --epochs 5 --seed 42
```

> Adjust function names in `main.py` to match your extracted code (e.g., `load_data`, `build_model`, `train_loop`, `evaluate`).

