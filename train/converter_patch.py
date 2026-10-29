"""Small additive fix for llama.cpp's convert_hf_to_gguf.py.

The converter reads a model's hyper-parameters through AutoConfig(...).to_dict().
Newer transformers releases re-serialize some configs in that round-trip (Gemma 4
loses text_config.global_head_dim, for example), which makes the converter fail with
a KeyError. The patch fills in any key that is present in the raw config.json but
missing after the round-trip. It only adds keys, never changes existing ones.

ensure_patch() is idempotent and is re-run on every export, so the fix survives the
converter being re-downloaded from llama.cpp.
"""

MARKER = "# LLM-TRAINING-STUDIO-PATCH"

_ANCHOR = '                config = json.load(f)\n        if "llm_config" in config:'

_PATCH = (
    "        " + MARKER + ": AutoConfig.to_dict() on newer transformers rewrites some configs\n"
    "        # (e.g. Gemma 4 loses text_config.global_head_dim). Fill in any key present in the raw\n"
    "        # config.json but missing after that round-trip. Purely additive.\n"
    "        try:\n"
    '            with open(dir_model / "config.json", "r", encoding="utf-8") as _f:\n'
    "                _raw_cfg = json.load(_f)\n"
    "            def _fill_missing(dst, src):\n"
    "                for _k, _v in src.items():\n"
    "                    if _k not in dst:\n"
    "                        dst[_k] = _v\n"
    "                    elif isinstance(_v, dict) and isinstance(dst[_k], dict):\n"
    "                        _fill_missing(dst[_k], _v)\n"
    "            _fill_missing(config, _raw_cfg)\n"
    "        except Exception:\n"
    "            pass\n"
)


def ensure_patch(script_path):
    """Apply the patch if it is missing. Returns True if the file was changed."""
    try:
        with open(script_path, encoding="utf-8", newline="") as f:
            src = f.read()
        if MARKER in src:
            return False
        crlf = "\r\n" in src
        norm = src.replace("\r\n", "\n")
        if _ANCHOR not in norm:
            return False          # converter layout changed upstream; leave it alone
        head, tail = _ANCHOR.split('        if "llm_config"')
        norm = norm.replace(_ANCHOR, head + _PATCH + '        if "llm_config"' + tail, 1)
        with open(script_path, "w", encoding="utf-8", newline="") as f:
            f.write(norm.replace("\n", "\r\n") if crlf else norm)
        return True
    except Exception as e:
        print(f"[export] (could not patch converter: {e})")
        return False
