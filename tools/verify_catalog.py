"""Check every MODEL_CATALOG entry against the Hugging Face API.

Flags models that don't exist, are gated (need login / license acceptance), or
aren't Apache-2.0 / MIT / Qwen-licensed, and compares params_b with the real
safetensors parameter count.  The default catalog must stay download-and-go.

    python tools/verify_catalog.py            # check train/server.py catalog
    python tools/verify_catalog.py ID [ID..]  # probe candidate model IDs
"""
import ast, json, os, sys, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK_LICENSES = {"apache-2.0", "mit", "qwen", "other"}   # "other": inspect by hand


def load_catalog():
    src = open(os.path.join(ROOT, "train", "server.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "MODEL_CATALOG":
            return ast.literal_eval(node.value)
    raise SystemExit("MODEL_CATALOG not found")


def probe(mid):
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{mid}", timeout=20) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}"}
    except Exception as e:
        return {"error": str(e)}
    st = (d.get("safetensors") or {}).get("total")
    return {"gated": d.get("gated"), "license": (d.get("cardData") or {}).get("license"),
            "params_b": round(st / 1e9, 2) if st else None}


def main():
    entries = [{"id": i} for i in sys.argv[1:]] or load_catalog()
    bad = 0
    for e in entries:
        info = probe(e["id"])
        problems = []
        if "error" in info:
            problems.append(info["error"])
        else:
            if info["gated"]:
                problems.append(f"GATED ({info['gated']})")
            if (info["license"] or "").lower() not in OK_LICENSES:
                problems.append(f"license={info['license']}")
            want = e.get("params_b")
            if want and info["params_b"] and abs(info["params_b"] - want) / want > 0.30:
                problems.append(f"params_b={want} but HF says {info['params_b']}")
        bad += bool(problems)
        print(("FAIL " if problems else "ok   ") + e["id"] + "  " +
              ("; ".join(problems) if problems else f"{info['license']} {info['params_b']}B"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
