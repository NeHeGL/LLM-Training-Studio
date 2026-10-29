"""Optional Hugging Face token support + friendly model-access checks.

The default catalog is fully ungated, so no token is ever required.  A token is
only needed if the user types a gated model ID (Llama, Gemma 3, ...).  It is
stored in <project>/.hf_token (git-ignored) and exported as HF_TOKEN, which
huggingface_hub / transformers pick up automatically.
"""
import os

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_FILE = os.path.join(ROOT, ".hf_token")


def load_token():
    """Export the saved token as HF_TOKEN (unless one is already set). Returns it or ''."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        with open(TOKEN_FILE, encoding="utf-8") as f:
            tok = f.read().strip()
    except OSError:
        return ""
    if tok:
        os.environ["HF_TOKEN"] = tok
    return tok


def save_token(token):
    """Save (or, with an empty string, clear) the token."""
    token = (token or "").strip()
    if token:
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(token)
        os.environ["HF_TOKEN"] = token
    else:
        try:
            os.remove(TOKEN_FILE)
        except OSError:
            pass
        os.environ.pop("HF_TOKEN", None)


def check_model(model_id):
    """Return (ok, message). A local folder path always passes."""
    if os.path.isdir(model_id):
        return True, "local folder"
    load_token()
    try:
        from huggingface_hub import model_info
        from huggingface_hub.errors import (GatedRepoError, RepositoryNotFoundError,
                                            HfHubHTTPError)
    except ImportError:
        return True, "skipped (huggingface_hub missing)"
    try:
        info = model_info(model_id)
        if getattr(info, "gated", False):
            # Metadata is public even for gated repos; only file access is blocked.
            from huggingface_hub import hf_hub_download
            hf_hub_download(model_id, "config.json")   # raises GatedRepoError w/o access
        return True, "found"
    except GatedRepoError:
        return False, (f"'{model_id}' is a gated model. Accept its license on "
                       f"https://huggingface.co/{model_id} and add a Hugging Face token "
                       f"in the Studio (Model Settings > Hugging Face token). "
                       f"Ungated models in the catalog need no token.")
    except RepositoryNotFoundError:
        return False, (f"'{model_id}' was not found on Hugging Face, or it is private/gated "
                       f"and needs a token. Check the exact ID at https://huggingface.co/models")
    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code in (401, 403):
            return False, (f"Access to '{model_id}' was denied (HTTP {code}). It is probably "
                           f"gated: accept its license on huggingface.co and add a token.")
        return True, f"could not verify (HTTP {code}); continuing"
    except Exception as e:  # offline, DNS, etc. - cached models still work
        return True, f"could not verify ({type(e).__name__}); continuing"
