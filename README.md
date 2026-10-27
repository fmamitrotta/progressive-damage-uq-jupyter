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

- **Running the analyses yourself needs a licensed local Abaqus installation.** This is
  different from a fully self-contained numerical-methods course: the notebooks that build or
  solve a finite element model call Abaqus itself.
- **Binder and Google Colab only work for the notebooks that need no Abaqus results at all:**
  today that is notebooks 04, 05 and 07 (they compute everything themselves). Notebooks 01,
  02, 03, 06 and 08 use Abaqus results, so they need either Abaqus or the result files (next
  section).

## Reading or running the notebooks

- **Reading:** every notebook is saved with its outputs, so you can read it on GitHub with
  nothing installed.
- **Running:** set up the Python environment below. Notebooks 01, 02, 03, 06 and 08 work by
  *run or load*: Abaqus runs an analysis only if its result file under `runs/nbNN/` is
  missing, and otherwise loads the file. To run them you need **either**
  - a licensed **Abaqus** installation, to run the analyses yourself (notebooks 06 and 08 take
    hours and need many licence tokens), **or**
  - the **result files**. The `runs/` folder is not part of this repository (about 180 MB of
    JSON and CSV files). While the series is in development, email me at
    fma.mitrotta@bristol.ac.uk with the notebooks you want to run, and put the folders I send
    you in `runs/` at the repository root (`runs/nb01/`, `runs/nb02/`, ...). Notebook 03 also
    reads the results of notebook 02.

## Requirements

- **Abaqus**, only if you want to run the analyses yourself rather than load result files
  (developed and verified against **Abaqus 2023**; nearby versions are likely to work but
  haven't been checked).
- **Python 3.10** for the notebook environment, pinned in `environment.yml` (this is separate
  from, and does not need to match, Abaqus's own bundled Python interpreter).

## Setting up the Python environment

### Option A — Anaconda / conda

```bash
conda env create -f environment.yml
conda activate progressive-damage-uq
```

[`environment.yml`](environment.yml) lists the direct dependencies (Python, NumPy, SciPy,
Matplotlib, `abqpy`, and the Jupyter pieces needed to run a notebook).

### Option B — plain venv + pip

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install "abqpy==2023.*" numpy scipy matplotlib ipykernel nbconvert
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
├── README.md           — this file
├── LICENSE             — CC BY 4.0 (text, figures) and BSD-3-Clause (code)
├── environment.yml     — conda environment for the notebooks
├── notebooks/          — the course content, added incrementally
├── scripts/            — helper modules imported by the notebooks, and the standalone
│                         Abaqus scripts they launch through `subprocess`
├── images/             — figures used in the notebooks
└── runs/               — saved analysis results; not in the repository (see above)
```

## License

The text and figures of the notebooks are released under the
[Creative Commons Attribution 4.0 license](http://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0). The code, including the code cells of the notebooks and the files in `scripts/`,
is released under the BSD-3-Clause license. See [`LICENSE`](LICENSE) for both.

## Author

Francesco Mario Antonio Mitrotta (University of Bristol)
