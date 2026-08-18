import shutil
import sys, os, matplotlib, sklearn, dask, torch

def verify_mac_env():
    # Detect the current conda env root dynamically if possible, or fallback to default
    env_root = os.environ.get("CONDA_PREFIX", "/Users/jerrylin/miniconda3/envs/e2s-mac")
    pkgs = [matplotlib, sklearn, dask]
    
    print(f"--- macOS Environment Verification ---")
    print(f"Target Conda Env Root: {env_root}")
    for pkg in pkgs:
        path = pkg.__file__
        status = "✅ OK" if env_root in path else "❌ POISONED (External Path)"
        print(f"{pkg.__name__:12} : {path} [{status}]")
    
    # Test for matplotlib issues
    try:
        matplotlib.get_data_path()
        print("✅ Matplotlib: Data path initialized correctly.")
    except AttributeError:
        print("❌ Matplotlib: STILL CORRUPTED. Check for ~/.local files.")

    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ MPS (Metal) Available: {torch.backends.mps.is_available()}")

    # Verify earth2grid base and CUDA extension status
    try:
        import earth2grid
        print("✅ earth2grid: Base package imported successfully.")
        
        try:
            from earth2grid.healpix._padding.cuda import healpixpad_cuda
            if healpixpad_cuda is not None:
                print("✅ earth2grid: CUDA healpixpad extension loaded.")
            else:
                print("ℹ️ earth2grid: CUDA healpixpad is None (expected on CPU/MPS).")
        except ImportError:
            print("ℹ️ earth2grid: CUDA healpixpad extension not loaded (expected on CPU/MPS).")
            
    except ImportError as e:
        print(f"❌ earth2grid: Failed to import base package: {e}")

    # Verify torch-harmonics (required by custom_utils/diagnostics.py)
    try:
        from torch_harmonics import RealSHT  # noqa: F401
        print("✅ torch-harmonics: RealSHT imported.")
    except ImportError as e:
        print(f"❌ torch-harmonics: Failed to import RealSHT: {e}")

    # Verify TempestExtremes binaries
    for binary in ("DetectNodes", "StitchNodes"):
        path = shutil.which(binary)
        if path is None:
            print(f"❌ tempest-extremes: {binary} not found on PATH. Ensure environment is activated.")
        else:
            print(f"✅ tempest-extremes: {binary} -> {path}")

if __name__ == "__main__":
    verify_mac_env()
