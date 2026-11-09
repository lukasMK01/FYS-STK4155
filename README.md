# FYS-STK4155 Machine Learning Projects

This repository contains coursework projects for the FYS-STK3155/FYS4155 Data
Analysis and Machine Learning course at the University of Oslo. Each project
has its own folder and README with details about its aims, files, and results:

- [Project 1](project_1/README.md): regression, resampling methods, and gradient descent.
- [Project 2](project_2/README_proj2.md): neural networks for regression and classification.

## Shared Python environment

The projects use the Conda environment defined in
[`lukas_fysstk.yaml`](lukas_fysstk.yaml). Create and activate it with:

```bash
conda env create -f lukas_fysstk.yaml
conda activate lmk_ML
```

The environment includes Python, Jupyter/IPython, NumPy, Matplotlib,
scikit-learn, and JAX. Open the relevant notebook in Jupyter or VS Code and
run its cells as described in that project's README.

## Repository contents

- `project_1/` — Project 1 notebooks, report, and figures.
- `project_2/` — Project 2 notebook, helper functions, and figures.
- `lukas_fysstk.yaml` — shared Conda environment specification.
- this README.

See each project's README for the project-specific notebook and report guide.
