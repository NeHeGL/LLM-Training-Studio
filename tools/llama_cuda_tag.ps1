# Which llama.cpp CUDA build can a given GPU + driver run?  Dot-sourced by install_llama_quantize.ps1.
#
#   $cc   GPU compute capability, e.g. [version]"12.0" for an RTX 5090, "8.9" for an RTX 4090, "6.1" for a GTX 1080
#   $drv  CUDA version the NVIDIA driver supports (nvidia-smi top banner), e.g. [version]"13.5"
#
# Returns "13.4", "12.4", or $null (= use the CPU build only).
#   cuda-13.4: Turing (cc 7.5, RTX 20xx) and newer, needs driver CUDA >= 13.4.  Native Blackwell (RTX 50xx) kernels.
#   cuda-12.4: older cards (cc < 7.5: GTX 10xx, Titan V) and drivers between 12.4 and 13.3.
#              Blackwell cards (driver >= 12.8, which Blackwell needs anyway) can run it too: it has no native
#              Blackwell kernels, so the driver compiles them on the first load (~20 s once, then cached) and it
#              then runs within ~2% of the native build. Verified on an RTX 5090 (driver 13.5).
function Select-CudaTag([version]$cc, [version]$drv) {
    if ($cc -lt [version]"7.5") {
        if ($drv -ge [version]"12.4") { return "12.4" }
        return $null
    }
    if ($drv -ge [version]"13.4") { return "13.4" }
    if ($cc -ge [version]"12.0") {
        if ($drv -ge [version]"12.8") { return "12.4" }
        return $null
    }
    if ($drv -ge [version]"12.4") { return "12.4" }
    return $null
}
