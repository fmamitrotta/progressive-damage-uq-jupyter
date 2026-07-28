# Introduction to Uncertainty Quantification of Progressive Damage Analysis with Jupyter Notebooks

An open-source, notebook-based introduction to uncertainty quantification (UQ) of progressive damage modeling of composite structures, aimed at PhD students and researchers with no assumed prior background in composites or damage mechanics. The course follows a learn-by-doing, computational approach: every concept is introduced alongside a runnable model, not just equations.

The material starts from the smallest possible verification case — a single finite element
— and builds up gradually toward laminate-level analysis and, eventually, a stochastic layer
on top of the deterministic damage model.

## What this course is (and isn't)

This course uses Abaqus's built-in progressive damage capabilities for fiber-reinforced
composites (Hashin failure criteria with energy-based damage evolution), automated through
Python rather than the Abaqus/CAE GUI. Two things follow from that which are worth stating
upfront:

- **A licensed local Abaqus installation is required** for any notebook that actually builds
  or solves a finite element model. This is different from a fully self-contained
  numerical-methods course — there is no way around needing Abaqus itself, and no version of
  this repository will run those notebooks without it.
- **Binder and Google Colab are not viable for the Abaqus-driving notebooks.** They can only
  be used for notebooks that are pure Python and operate on already-computed results (later
  UQ / post-processing notebooks, once those exist). Anything that builds or submits an
  Abaqus model needs to run somewhere with Abaqus actually installed.

## Requirements

- A working, licensed installation of **Abaqus** (developed and verified against **Abaqus
  2023**; nearby versions are likely to work but haven't been checked).
- **Python 3.9+** for the notebook-authoring environment (this is separate from, and does not
  need to match, Abaqus's own bundled Python interpreter).

## Setting up the Python environment

### Option A — Anaconda / conda

```bash
conda create -n progressive-damage-uq python=3.10
conda activate progressive-damage-uq
conda install abqpy==2023 numpy matplotlib ipympl ipykernel
```

### Option B — plain venv + pip

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install "abqpy==2023.*" numpy matplotlib ipympl ipykernel
```

A few notes on the `abqpy` pin:

- **Match the version number to your installed Abaqus version.** `abqpy` is versioned
  alongside Abaqus itself (e.g. `abqpy==2023.*` for Abaqus 2023); mismatches can lead to type
  hints and keyword defaults that don't correspond to what your actual installation supports.
- This environment does **not** need the `abqpy[jupyter]` extra (`ipynbname` + `nbconvert`).
  That extra exists to support writing Abaqus scripting code directly inside a live notebook
  cell — a pattern this repository deliberately avoids because Abaqus-related python code can
  only run inside Abaqus' kernel (it kills the notebook's kernel if used directly in the notebook).
  All Abaqus-facing code here lives in standalone `.py` scripts, invoked from a notebook via
  `subprocess`, which sidesteps the need for that extra entirely.
- Abaqus itself is a separate, licensed installation your institution provides — conda/pip
  cannot install or license it.
- If the `abaqus` command isn't on your system `PATH`, `abqpy` will also respect an
  `ABAQUS_BAT_PATH` environment variable pointing directly at your `abaqus.bat` (Windows) or
  `abaqus` executable, which can be more robust than relying on `PATH` resolution.

## Repository structure

```
.
├── README.md          — this file
├── notebooks/          — the course content, added incrementally
└── scripts/            — standalone Abaqus/Python scripts generated and used by the
                          notebooks
```

## Author

Francesco Mario Antonio Mitrotta (University of Bristol)
