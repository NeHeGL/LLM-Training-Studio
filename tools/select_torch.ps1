# Picks the right PyTorch build for this machine. Called by install.bat.
# Prints KEY=VALUE lines: GPU_NAME, CUDA_VER, GPU_TAG, TORCH_INDEX, TORCH_SPEC, ERROR
#
# Rules (PyTorch index as of late 2026 - re-check with tools/select_torch.ps1 -Explain):
#   cu130  newest; needs driver CUDA >= 13.0;  cards: Turing (RTX 20xx, cc 7.5) and newer
#   cu128  needs driver CUDA >= 12.8;          cards: Turing and newer (incl. Blackwell RTX 50xx)
#   cu126  needs driver CUDA >= 12.6;          the ONLY build that still covers Pascal/Volta
#          (GTX 10xx, Titan V, cc < 7.5); PyTorch stops publishing it after 2.14, so we pin torch~=2.14
# Older tags (cu118/cu121/cu124) are no longer published and are never chosen.

param([switch]$Explain)

function Out($k, $v) { Write-Output "$k=$v" }

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Out GPU_NAME "none"; Out CUDA_VER "none"; Out GPU_TAG "cpu"
    Out TORCH_INDEX "https://download.pytorch.org/whl/cpu"; Out TORCH_SPEC "torch torchvision torchaudio"
    exit 0
}

$line = (& nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
if (-not $line) {
    Out GPU_NAME "none"; Out CUDA_VER "none"; Out GPU_TAG "cpu"
    Out TORCH_INDEX "https://download.pytorch.org/whl/cpu"; Out TORCH_SPEC "torch torchvision torchaudio"
    exit 0
}
$name, $ccText = $line -split ',\s*', 2
$cc = [version]($ccText.Trim())

$text = (& nvidia-smi 2>$null) -join "`n"
if ($text -match 'CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)') { $drv = [version]"$($Matches[1]).$($Matches[2])" } else { $drv = [version]"0.0" }

Out GPU_NAME $name.Trim()
Out CUDA_VER "$drv"

$tag = $null; $spec = "torch torchvision torchaudio"
if ($cc -lt [version]"7.5") {
    # Pascal / Volta / Maxwell: only cu126 builds (torch <= 2.14) still contain their kernels
    if ($drv -ge [version]"12.6") { $tag = "cu126"; $spec = "torch~=2.14.0 torchvision torchaudio" }
} elseif ($drv -ge [version]"13.0") {
    $tag = "cu130"
} elseif ($drv -ge [version]"12.8") {
    $tag = "cu128"
} elseif ($drv -ge [version]"12.6" -and $cc -lt [version]"12.0") {
    $tag = "cu126"; $spec = "torch~=2.14.0 torchvision torchaudio"
}

if (-not $tag) {
    Out GPU_TAG "none"; Out TORCH_INDEX ""; Out TORCH_SPEC ""
    Out ERROR "Your NVIDIA driver (CUDA $drv) is too old for the $($name.Trim()). Update it from https://www.nvidia.com/drivers and run install.bat again."
    exit 0
}
Out GPU_TAG $tag
Out TORCH_INDEX "https://download.pytorch.org/whl/$tag"
Out TORCH_SPEC $spec
