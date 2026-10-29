"""Post-install sanity check: can PyTorch actually run a kernel on this GPU?

Catches the classic "no kernel image is available for execution on the device"
(wrong CUDA build for the card) right at install time instead of mid-training.
Exit code 0 = OK (or CPU-only install), 1 = GPU present but unusable.
"""
import sys

try:
    import torch
except Exception as e:
    print(f"[ERROR] PyTorch could not be imported: {e}")
    sys.exit(1)

print(f"PyTorch {torch.__version__}  (CUDA build: {torch.version.cuda})")
if not torch.cuda.is_available():
    print("[INFO] CUDA is not available - training will run on the CPU (very slow).")
    sys.exit(0)

try:
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    x = torch.ones(1024, device="cuda") * 2
    torch.cuda.synchronize()
    assert float(x.sum()) == 2048.0
    print(f"[OK] GPU check passed: {name} (compute capability {cc[0]}.{cc[1]})")
except Exception as e:
    print(f"[ERROR] The GPU is visible but PyTorch cannot run on it: {e}")
    print("        This usually means the PyTorch build does not match the card or driver.")
    sys.exit(1)
