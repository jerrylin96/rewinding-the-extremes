# Earth2Studio Custom Environment Installation Guide

Based on an inspection of `custom_utils`, `scripts`, and `notebooks`, here are the packages your custom code uses that are not part of the standard Python library:
- **Core ML & Data**: `torch`, `numpy`, `scipy`
- **Earth2Studio Ecosystem**: `earth2studio`, `earth2grid`
- **Visualization**: `matplotlib`, `cartopy`, `cmasher`
- **I/O & Logging**: `zarr`, `numcodecs`, `loguru`
- **Other**: `scikit-learn`, `dask`
- **Notebooks**: `jupyter` / `ipython`

> [!NOTE]
> **What about torch, xarray, dask, etc.?**
> Packages like `torch`, `xarray`, `tqdm`, `zarr`, `loguru`, `numpy`, and `dask` (which is pulled by `xarray[parallel]`) are **core dependencies** of Earth2Studio. When we run the `uv pip install earth2studio` command in Step 2, `uv` will automatically read Earth2Studio's `pyproject.toml` and download and install them for you seamlessly. You do not need to specify them manually!

## Recommended Installation Order

To ensure everything works seamlessly on a Linux HPC cluster (like BU SCC), here is the recommended order using `conda` and `uv`. Since you want to use `uv`, we can completely replace `pip` with `uv pip`.

### 1. Conda (Environment & System/C-Library Packages)
Use `conda` (specifically the `conda-forge` channel) to set up Python, essential Jupyter packages, and `cartopy`. 

> [!IMPORTANT]
> **Why use Conda here?** `cartopy` requires underlying C-libraries (GEOS and PROJ) that can be frustrating to build from source via pip/uv on a cluster where you lack root/sudo privileges. `conda-forge` ships pre-compiled binaries that work smoothly.

```bash
# We include matplotlib here because cartopy depends on it, and installing it via pip later can corrupt the installation.
# We also explicitly pin gxx_linux-64=12 because newer GCCs (14+) are currently rejected by CUDA 12.8 nvcc.
# tempest-extremes ships the DetectNodes/StitchNodes binaries used by the
# `--tracker tempest` option in scripts/ensemble_analysis/dispatch_tc_tracks.py.
conda create -n e2s-custom -c conda-forge python=3.12 gxx_linux-64=12 cartopy matplotlib jupyterlab ipykernel tempest-extremes
conda activate e2s-custom

# Prevent python from accidentally loading "shadow" packages from ~/.local
conda env config vars set PYTHONNOUSERSITE=1
# You must reactivate the environment for the variable change to take effect immediately
conda activate e2s-custom

# Register your conda environment as a custom Jupyter Kernel
# This is required for VS Code / Jupyter on OnDemand to "see" your environment
python -m ipykernel install --user --name e2s-custom --display-name "Python (e2s-custom)"

# Verify TempestExtremes binaries are on PATH
which DetectNodes StitchNodes
```

### 2. uv (Earth2Studio & Pure Python Packages)
With your conda environment active, use `uv` to install Earth2Studio and your remaining Python dependencies. `uv pip install` will automatically use the Python interpreter from your active conda environment and is a lighting-fast drop-in replacement for `pip`.

> [!WARNING]
> **HPC Home Directory Quotas**
> By default, `uv` caches downloaded packages in `~/.cache/uv`. Because machine learning packages are massive, this cache can easily exceed 20GB and blow past your 10GB `$HOME` directory quota on the cluster. 
> To prevent a "Disk quota exceeded" error, set the `UV_CACHE_DIR` environment variable to a high-capacity project directory before running `uv`:
> ```bash
> export UV_CACHE_DIR="/projectnb/eb-general/jlin404/.cache/uv"
> ```

First, make sure `uv` is installed on your system:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, install Earth2Studio. The E2S docs recommend installing from the GitHub repo. Because the repository's `pyproject.toml` is configured for `uv`, `uv` will automatically handle pulling in `earth2grid` via its internal Git source mapping.

> [!IMPORTANT]
> **Git clone dependency**: `uv pip install -e` is **editable mode**, which writes a `.pth` file in the conda env's `site-packages` that points at the **exact filesystem path** of this git clone. `earth2grid`'s compiled C extensions are also placed inside that source tree.
>
> Consequences:
> - **Deleting or moving the clone breaks `import earth2studio`** because the `.pth` target no longer exists.
> - **The fix is cheap**: from any new clone (or worktree) at any path, re-run `uv pip install -e ".[utils,cbottle,cyclone,perturbation]"`. That rewrites the `.pth` and rebuilds the C extensions at the new location. You do **not** need to recreate the conda env.
> - The conda env only needs a full rebuild if something at the conda layer (Python version, cartopy, tempest-extremes, etc.) needs to change.

```bash
# Navigate to your local clone of your private branch
cd /projectnb/eb-general/jlin404/weather_interpolation/dev-clean

# PRE-REQUISITE: Ensure your HPC's CUDA module is loaded so `nvcc` is available for C++ compilation!
# The following line shows how it would be loaded on BU SCC:
module load cuda/12.8

# PRE-REQUISITE: Install torch (CUDA version) and setuptools first
# earth2grid requires CUDA-enabled PyTorch to build its C++ extensions correctly
uv pip install setuptools
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

> [!NOTE]
> **How to double-check this URL yourselves:**
> The `--index-url` format is always `https://download.pytorch.org/whl/cuXXX`. You can find the exact URL by visiting [PyTorch's Get Started Page](https://pytorch.org/get-started/locally/) and selecting "Pip" + your OS + "CUDA". 
> Since your cluster uses CUDA 12.8 (which we can see from your `/share/pkg.8/cuda/12.8` log path), use the `cu128` index which matches your CUDA version and has torch 2.9.1 available.

```bash
# Install Earth2Studio in editable mode, including utils, cbottle, cyclone, and perturbation.
# The `perturbation` extra pulls in `torch-harmonics`, which is required by
# `custom_utils/diagnostics.py` (spherical_power_spectrum → RealSHT).
# Because you are compiling on a login node with no GPUs, PyTorch cannot autodetect 
# target architectures. We must manually supply them (A100, RTX Ada, H100, Blackwell).
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"
uv pip install -e ".[utils,cbottle,cyclone,perturbation]"

# Install your extra custom dependencies (not covered by E2S core)
uv pip install cmasher numcodecs scipy scikit-learn dask distributed dask-jobqueue
```

> [!TIP]
> **What about `earth2grid` and `--no-build-isolation`?**
> The Earth2Studio developers already configured their `pyproject.toml` file to seamlessly handle this specifically for `uv`! If you look in the `[tool.uv]` section of the source code ([pyproject.toml#L266-L267](file:///Users/jlin404/Desktop/polished_repos/earth2studio_tinkering/earth2studio-private/dev-clean/pyproject.toml#L266-L267)), they have `no-build-isolation-package = ["earth2grid"]` and map it to a specific Git commit lower down ([pyproject.toml#L278](file:///Users/jlin404/Desktop/polished_repos/earth2studio_tinkering/earth2studio-private/dev-clean/pyproject.toml#L278)). So when you run `uv pip install -e ".[utils,cbottle,cyclone]"`, `uv` natively skips build isolation for you automatically. You don't have to manually figure this out!

> [!TIP]
> **Why no `pip`?** We completely skipped `pip`. Since `uv pip` replaces it, you achieve your goal of using `uv` exclusively for the Python package resolution.

### 3. Verify Installation
You can verify the installation by running `verify_env.py`.

```bash
python verify_env.py
```
