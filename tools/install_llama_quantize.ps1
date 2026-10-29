param(
    [string]$LlamaDir = "$PSScriptRoot\llama",
    [string]$LlamaExe = "$PSScriptRoot\llama\llama-quantize.exe"
)

if (-not (Test-Path $LlamaDir)) {
    New-Item -ItemType Directory -Path $LlamaDir | Out-Null
}

try {
    Write-Host "  Looking for the latest llama.cpp Windows build on GitHub..."
    # NOTE: /releases/latest no longer reliably points at a binary release (llama.cpp tags builds
    # bNNNN and marks them pre-release), so scan the most recent releases instead.
    $rels = Invoke-RestMethod 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15'

    # The CPU build contains llama-quantize and llama-imatrix, which is all we need.
    $patterns = @('^llama-.*-bin-win-cpu-x64\.zip$', '^llama-.*-bin-win-avx2-x64\.zip$', '^llama-.*-bin-win-.*x64\.zip$')
    $asset = $null
    foreach ($pat in $patterns) {
        foreach ($rel in $rels) {
            $asset = $rel.assets | Where-Object { $_.name -match $pat -and $_.name -notmatch 'arm64|rocm|sycl|openvino|opencl|cudart' } | Select-Object -First 1
            if ($asset) { break }
        }
        if ($asset) { break }
    }
    if (-not $asset) { throw "No Windows x64 build found in the recent llama.cpp releases" }

    $zip = Join-Path $LlamaDir "llama_release.zip"
    Write-Host "  Downloading $($asset.name) ..."
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip

    Write-Host "  Extracting llama-quantize.exe and required DLLs..."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $z = [System.IO.Compression.ZipFile]::OpenRead($zip)

    $extracted = 0
    foreach ($entry in $z.Entries) {
        if ($entry.Name -eq '') { continue }
        $ext = [System.IO.Path]::GetExtension($entry.Name).ToLower()
        if ($ext -ne '.exe' -and $ext -ne '.dll') { continue }
        $destPath = Join-Path $LlamaDir $entry.Name
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $destPath, $true)
        $extracted++
    }
    $z.Dispose()
    Remove-Item $zip -Force

    Write-Host "  Extracted $extracted files (exe + DLLs)"

    if (Test-Path $LlamaExe) {
        Write-Host "  [OK] llama-quantize.exe installed successfully (with all required DLLs)"
        exit 0
    } else {
        throw "llama-quantize.exe not found in zip"
    }
} catch {
    Write-Host "  [WARN] $_"
    exit 1
}
