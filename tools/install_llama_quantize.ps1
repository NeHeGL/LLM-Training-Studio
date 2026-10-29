# Installs llama.cpp's official Windows binaries next to the project (called by install.bat).
#
#   tools\llama       CPU build (~25 MB). Always installed. llama-quantize, llama-imatrix, llama-server.
#   tools\llama-cuda  CUDA build (~600 MB, optional). Same tools, GPU-accelerated: faster chat tester
#                     and faster imatrix calibration. Installed only when an NVIDIA GPU + a new enough
#                     driver are found, and removed again if it cannot actually see the GPU.
#
# Safe to run repeatedly: parts that are already installed are skipped.
#   -NoCuda   skip the CUDA build (also: set LLM_SKIP_CUDA_LLAMA=1)

param(
    [string]$LlamaDir = "$PSScriptRoot\llama",
    [string]$LlamaExe = "$PSScriptRoot\llama\llama-quantize.exe",
    [string]$CudaDir  = "$PSScriptRoot\llama-cuda",
    [switch]$NoCuda
)

$ProgressPreference = 'SilentlyContinue'     # Windows PowerShell 5.1 downloads are ~10x slower with the progress bar
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-Releases {
    # /releases/latest no longer reliably points at a binary release (llama.cpp tags builds bNNNN and
    # marks them pre-release), so scan the most recent releases instead.
    Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15'
}

function Save-And-Extract($asset, $dest) {
    if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Path $dest | Out-Null }
    $zip = Join-Path $dest "_download.zip"
    Write-Host ("  Downloading {0} ({1} MB) ..." -f $asset.name, [math]::Round($asset.size / 1MB))
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip
    $z = [System.IO.Compression.ZipFile]::OpenRead($zip)
    $n = 0
    foreach ($entry in $z.Entries) {
        if ($entry.Name -eq '') { continue }
        $ext = [System.IO.Path]::GetExtension($entry.Name).ToLower()
        if ($ext -ne '.exe' -and $ext -ne '.dll') { continue }
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $dest $entry.Name), $true)
        $n++
    }
    $z.Dispose()
    Remove-Item $zip -Force
    return $n
}

# ── 1. CPU build (required) ─────────────────────────────────────────────────────────
$cpuOk = Test-Path $LlamaExe
if ($cpuOk) {
    Write-Host "  [OK] llama.cpp CPU tools already installed"
} else {
    try {
        Write-Host "  Looking for the latest llama.cpp Windows build on GitHub..."
        $rels = Get-Releases
        $patterns = @('^llama-.*-bin-win-cpu-x64\.zip$', '^llama-.*-bin-win-avx2-x64\.zip$', '^llama-.*-bin-win-.*x64\.zip$')
        $asset = $null
        foreach ($pat in $patterns) {
            foreach ($rel in $rels) {
                $asset = $rel.assets | Where-Object { $_.name -match $pat -and $_.name -notmatch 'arm64|rocm|sycl|openvino|opencl|cudart|cuda|vulkan' } | Select-Object -First 1
                if ($asset) { break }
            }
            if ($asset) { break }
        }
        if (-not $asset) { throw "No Windows x64 build found in the recent llama.cpp releases" }
        $n = Save-And-Extract $asset $LlamaDir
        Write-Host "  Extracted $n files (exe + DLLs)"
        if (-not (Test-Path $LlamaExe)) { throw "llama-quantize.exe not found in zip" }
        Write-Host "  [OK] llama-quantize.exe installed successfully (with all required DLLs)"
        $cpuOk = $true
    } catch {
        Write-Host "  [WARN] $_"
    }
}

# ── 2. CUDA build (optional) ────────────────────────────────────────────────────────
function Install-CudaBuild {
    if ($NoCuda -or $env:LLM_SKIP_CUDA_LLAMA) { Write-Host "  (skipping the CUDA build as requested)"; return }
    if (Test-Path (Join-Path $CudaDir "llama-server.exe")) { Write-Host "  [OK] llama.cpp CUDA tools already installed"; return }
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return }

    $line = (& nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
    if (-not $line) { return }
    $name, $ccText = $line -split ',\s*', 2
    $cc = [version]($ccText.Trim())
    $text = (& nvidia-smi 2>$null) -join "`n"
    if ($text -notmatch 'CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)') { return }
    $drv = [version]"$($Matches[1]).$($Matches[2])"

    # Which llama.cpp CUDA build can this card + driver run?
    #   cuda-13.4: Turing (RTX 20xx) and newer, needs driver CUDA >= 13.4; required for Blackwell (RTX 50xx)
    #   cuda-12.4: older cards (GTX 10xx / Titan V) and drivers between 12.4 and 13.3 (not Blackwell)
    $tag = $null
    if ($cc -lt [version]"7.5") {
        if ($drv -ge [version]"12.4") { $tag = "12.4" }
    } elseif ($drv -ge [version]"13.4") {
        $tag = "13.4"
    } elseif ($cc -lt [version]"12.0" -and $drv -ge [version]"12.4") {
        $tag = "12.4"
    }
    if (-not $tag) {
        Write-Host "  (GPU llama.cpp build skipped: the NVIDIA driver is too old for it - the CPU build is used. Update the driver to enable it.)"
        return
    }

    try {
        Write-Host "  Adding the GPU-accelerated llama.cpp build (CUDA $tag) for $($name.Trim()) - a one-time download of roughly 600 MB..."
        $rels = Get-Releases
        $main = $null; $rt = $null
        foreach ($rel in $rels) {
            $m = $rel.assets | Where-Object { $_.name -match "^llama-.*-bin-win-cuda-$([regex]::Escape($tag))-x64\.zip$" } | Select-Object -First 1
            $r = $rel.assets | Where-Object { $_.name -match "^cudart-llama-bin-win-cuda-$([regex]::Escape($tag))-x64\.zip$" } | Select-Object -First 1
            if ($m -and $r) { $main = $m; $rt = $r; break }
        }
        if (-not $main) { throw "no CUDA $tag build found in the recent llama.cpp releases" }
        $null = Save-And-Extract $main $CudaDir
        $null = Save-And-Extract $rt $CudaDir

        # Prove it can actually see the GPU; otherwise throw it away and stay on the CPU build.
        $out = (& (Join-Path $CudaDir "llama-server.exe") --list-devices 2>&1) -join "`n"
        if ($out -notmatch 'CUDA') { throw "the CUDA build could not see the GPU:`n$out" }
        Write-Host "  [OK] GPU-accelerated llama.cpp installed (CUDA $tag)"
    } catch {
        Write-Host "  [WARN] GPU llama.cpp build not installed: $_"
        Write-Host "         The CPU build is used instead - everything still works, just slower."
        if (Test-Path $CudaDir) { Remove-Item $CudaDir -Recurse -Force -ErrorAction SilentlyContinue }
    }
}
Install-CudaBuild

if (-not $cpuOk) { exit 1 }
exit 0
