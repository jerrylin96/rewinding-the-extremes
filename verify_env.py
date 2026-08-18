import shutil
import sys, os, matplotlib, sklearn, dask, torch

def verify_hpc_env():
    # Update this to match your actual conda environment path if different
    env_root = "/projectnb/eb-general/jlin404/.conda/envs/e2s-custom"
    pkgs = [matplotlib, sklearn, dask]
    
    print(f"--- Environment Verification ---")
    for pkg in pkgs:
        path = pkg.__file__
        status = "✅ OK" if env_root in path else "❌ POISONED (External Path)"
        print(f"{pkg.__name__:12} : {path} [{status}]")
    
    # Test for the previous failure
    try:
        matplotlib.get_data_path()
        print("✅ Matplotlib: Data path initialized correctly.")
    except AttributeError:
        print("❌ Matplotlib: STILL CORRUPTED. Check for ~/.local files.")

    print(f"✅ CUDA Available: {torch.cuda.is_available()}")

    # Verify earth2grid CUDA extensions compiled correctly
    try:
        from earth2grid.healpix._padding.cuda import healpixpad_cuda
        if healpixpad_cuda is not None:
            print("✅ earth2grid: CUDA healpixpad extension loaded.")
        else:
            print("❌ earth2grid: healpixpad_cuda is None. Reinstall with CUDA module loaded.")
    except ImportError as e:
        print(f"❌ earth2grid: Failed to import healpixpad CUDA extension: {e}")

    # Verify torch-harmonics (required by custom_utils/diagnostics.py)
    try:
        from torch_harmonics import RealSHT  # noqa: F401
        print("✅ torch-harmonics: RealSHT imported.")
    except ImportError as e:
        print(f"❌ torch-harmonics: Failed to import RealSHT: {e}")

    # Verify TempestExtremes binaries (used by --tracker tempest in
    # scripts/ensemble_analysis/dispatch_tc_tracks.py).
    for binary in ("DetectNodes", "StitchNodes"):
        path = shutil.which(binary)
        if path is None:
            print(f"❌ tempest-extremes: {binary} not found on PATH. Reinstall via conda.")
        else:
            print(f"✅ tempest-extremes: {binary} -> {path}")

if __name__ == "__main__":
    verify_hpc_env()
