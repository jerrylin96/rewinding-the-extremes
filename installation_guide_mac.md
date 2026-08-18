# Earth2Studio Custom Environment Installation Guide (macOS CPU/MPS)

This guide provides instructions to set up a macOS-compatible development and analytics environment for `earth2studio` using `conda` and `uv`. It is tailored for local desktops (e.g. Apple Silicon or Intel Macs) without local NVIDIA GPUs.

## Recommended Installation Order

### 1. Conda (Environment & C-Library Packages)
We use `conda` (via `conda-forge`) to set up Python, Jupyter, `cartopy` (which depends on C-libraries like GEOS and PROJ), and the `tempest-extremes` climate analysis binaries.

```bash
# Create the environment with Python 3.12, Cartopy, Matplotlib, Jupyter, and tempest-extremes
conda create -n e2s-mac -c conda-forge python=3.12 cartopy matplotlib jupyterlab ipykernel tempest-extremes
conda activate e2s-mac

# Prevent python from accidentally loading "shadow" packages from ~/.local
conda env config vars set PYTHONNOUSERSITE=1
# Workaround: OpenMP multiple initializations check (avoid macOS crashes)
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
# Reactivate environment for variable changes to take effect
conda activate e2s-mac

# Register conda environment as a custom Jupyter Kernel
python -m ipykernel install --user --name e2s-mac --display-name "Python (e2s-mac)"

# Verify TempestExtremes binaries are on PATH
which DetectNodes StitchNodes
```

### 2. uv (Earth2Studio & Python Packages)
With your conda environment active, use `uv` to install PyTorch (CPU/MPS), `earth2studio`, and remaining dependencies.

First, ensure `uv` is installed and updated to the latest version:
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Update uv to the latest version (ensures support for pyproject.toml settings)
uv self update
```

# Navigate to your local clone (e.g. dev-temp or dev-clean)
```bash
# Install build dependencies and standard PyTorch (CPU/MPS support is native on macOS)
uv pip install setuptools torch

# Navigate to your local clone (e.g. dev-temp or dev-clean)
cd /Users/jerrylin/Desktop/earth2studio-tinkering/dev-mac

# Install earth2studio in editable mode with macOS-compatible extras.
# Omit GPU-only extras (like `cyclone` and `cbottle` which require cupy/cucim/CUDA).
uv pip install -e ".[utils,perturbation]"

# Install custom analytical packages
uv pip install cmasher scikit-learn dask-jobqueue
```

### 3. Verify Installation
Verify the environment settings, PyTorch devices, package imports, and `tempest-extremes` binaries:

```bash
python verify_env_mac.py
```
