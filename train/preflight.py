"""Pre-flight checks shared by training and export: refuse early, with a clear message,
when a step cannot work on this machine (not enough GPU memory, RAM or disk, no usable GPU).

Testing / support overrides (environment variables):
  LLM_STUDIO_VRAM_GB       pretend the GPU has this much memory
  LLM_STUDIO_RAM_GB        pretend the machine has this much free RAM
  LLM_STUDIO_FREE_DISK_GB  pretend this much disk space is free
"""
import os
import shutil

GB = 1024 ** 3


def estimate_params_b(model_id):
    """Parameter count in billions: local safetensors headers / Hub metadata, then name, then 7."""
    import glob
    import json
    import re
    import struct
    try:
        if os.path.isdir(model_id):
            total = 0
            for f in glob.glob(os.path.join(model_id, "*.safetensors")):
                with open(f, "rb") as fh:
                    hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
                for k, v in hdr.items():
                    if k != "__metadata__" and v.get("dtype") in ("F16", "BF16", "F32"):
                        n = 1
                        for d in v["shape"]:
                            n *= d
                        total += n
            if total:
                return total / 1e9
        else:
            from huggingface_hub import model_info
            st = getattr(model_info(model_id), "safetensors", None)
            if st and getattr(st, "total", 0):
                return st.total / 1e9
    except Exception:
        pass
    m = re.findall(r"(\d+(?:\.\d+)?)\s*b(?![a-z])",
                   os.path.basename(model_id.rstrip("/\\")).lower().replace("-", " ").replace("_", " "))
    return float(m[-1]) if m else 7.0


def gpu_memory():
    """dict(n, free0, total0, free_all, total_all, other_gb) in GB, or None without a usable CUDA GPU.

    Free memory is read from the driver, so memory held by other programs (a game, a model loaded in
    Ollama or LM Studio, ...) is excluded. A simulated size (LLM_STUDIO_VRAM_GB) caps the result.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        n = torch.cuda.device_count()
        free, total = [], []
        for i in range(n):
            f, t = torch.cuda.mem_get_info(i)
            free.append(f / GB)
            total.append(t / GB)
    except Exception:
        return None
    if n == 1:
        # On Windows torch's figure ignores memory that OTHER processes hold (verified: nvidia-smi showed
        # 31.5 of 32 GB used while torch reported 29 GB free), so trust the driver's number as well.
        try:
            import subprocess
            row = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            free[0] = min(free[0], float(row.split(",")[0]) / 1024)
        except Exception:
            pass
    sim = os.environ.get("LLM_STUDIO_VRAM_GB")
    if sim:
        try:
            s = float(sim)
            return dict(n=1, free0=min(s, free[0]), total0=s, free_all=min(s, free[0]), total_all=s,
                        other_gb=max(0.0, total[0] - free[0]))
        except ValueError:
            pass
    return dict(n=n, free0=free[0], total0=total[0], free_all=sum(free), total_all=sum(total),
                other_gb=max(0.0, total[0] - free[0]))


def cuda_sanity():
    """None if PyTorch can run a kernel on the GPU, else a human-readable error."""
    try:
        import torch
        x = torch.ones(8, device="cuda") * 2
        torch.cuda.synchronize()
        if float(x.sum()) != 16.0:
            return "The GPU returned a wrong result on a basic test. Update your NVIDIA driver and reinstall."
    except Exception as e:
        return ("PyTorch cannot run on this GPU: " + str(e).splitlines()[0][:160] + "\n"
                "This usually means the installed PyTorch build does not match the card or driver. "
                "Delete the .venv folder and run install.bat again.")
    return None


def free_disk_gb(path):
    sim = os.environ.get("LLM_STUDIO_FREE_DISK_GB")
    if sim:
        return float(sim)
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p or ".").free / GB


def ram_gb():
    """(total, available) in GB, or (None, None) if psutil is unavailable."""
    sim = os.environ.get("LLM_STUDIO_RAM_GB")
    if sim:
        return float(sim), float(sim)
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / GB, vm.available / GB
    except Exception:
        return None, None


def hf_cache_dir():
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:
        return os.path.expanduser("~/.cache/huggingface/hub")


def model_is_cached(model_id):
    if os.path.isdir(model_id):
        return True
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, local_files_only=True, allow_patterns=["config.json"])
        # config present is not proof the weights are; check for at least one weight file
        import glob
        from huggingface_hub import constants
        folder = os.path.join(constants.HF_HUB_CACHE, "models--" + model_id.replace("/", "--"), "snapshots")
        return bool(glob.glob(os.path.join(folder, "*", "*.safetensors")))
    except Exception:
        return False
