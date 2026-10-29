"""Chat-template helpers shared by training, export and chat.

Training renders every example with the base model's own chat template, so the
exported model must be prompted with that same template.  Thinking-capable
families (Qwen3/3.5/3.6, ...) are trained and prompted with thinking OFF so the
model answers directly instead of emitting an empty <think> block.
"""

THINK_OFF = {"enable_thinking": False}


def render(tokenizer, messages, add_generation_prompt=False):
    """apply_chat_template(tokenize=False) with thinking disabled where supported."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **THINK_OFF)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def is_chatml(tokenizer):
    """True when the template natively uses <|im_start|>/<|im_end|> turns."""
    tpl = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(tpl, dict):
        tpl = tpl.get("default") or next(iter(tpl.values()), "")
    return "<|im_start|>" in tpl


def to_messages(record):
    """Normalise a dataset row (chatml 'messages' or alpaca fields) to a messages list."""
    if "messages" in record and record["messages"]:
        return record["messages"]
    msgs = []
    if record.get("system"):
        msgs.append({"role": "system", "content": record["system"]})
    msgs.append({"role": "user", "content": record["instruction"]})
    msgs.append({"role": "assistant", "content": record["output"]})
    return msgs


def tokenize_masked(tokenizer, messages, max_len):
    """Return (input_ids, labels) with loss only on the final assistant reply.

    The prompt is everything up to the generation prompt; the labels for those
    tokens are -100.  Falls back to the longest common prefix if the template
    rewrites the prompt part when the assistant turn is present.
    """
    prompt = render(tokenizer, messages[:-1], add_generation_prompt=True)
    full = render(tokenizer, messages)
    split = 0
    for a, b in zip(prompt, full):
        if a != b:
            break
        split += 1
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels = [tid if end > split else -100 for tid, (_, end) in zip(ids, offs)]
    return ids[:max_len], labels[:max_len]


def _jinja_str(s):
    return "'" + s.replace("\\", "\\\\").replace("'", "\'").replace("\n", "\n") + "'"


def wrap_native(template, system_prompt):
    """Return `template` with two export-time guarantees prepended.

    1. Thinking stays OFF if the template supports it (matches how training rendered it).
    2. If the caller sends no system message, the model's configured system prompt is used
       (the model's identity travels inside the GGUF with no Modelfile needed).
    """
    pre = ""
    if "enable_thinking" in template:
        pre += "{%- set enable_thinking = false -%}"
    pre += ("{%- if messages and messages[0]['role'] != 'system' -%}"
            "{%- set messages = [{'role': 'system', 'content': " + _jinja_str(system_prompt) + "}] + messages -%}"
            "{%- endif -%}")
    return pre + template


def ollama_template(tokenizer):
    """Ollama Modelfile TEMPLATE (Go syntax) + stop strings, derived from the real template.

    Returns (template, stops) or (None, []) for families we don't have a verified mapping for;
    the caller then leaves TEMPLATE out and Ollama falls back to the template in the GGUF.
    The text the template emits after the assistant tag (e.g. an empty <think> block) is taken
    from the actual rendered generation prompt, so it always matches training.
    """
    tpl = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(tpl, dict):
        tpl = tpl.get("default") or next(iter(tpl.values()), "")
    gen = render(tokenizer, [{"role": "user", "content": "U"}], add_generation_prompt=True)

    def esc(s):
        return s.replace("\n", "\n").replace('"', '\\"')

    if "<|im_start|>" in tpl or "<|im_start|>" in gen:
        head = "<|im_start|>assistant\n"
        suffix = gen.rsplit(head, 1)[-1] if head in gen else ""
        return (
            "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
            "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}"
            "<|im_start|>assistant\n" + suffix
        ), ["<|im_end|>", "<|endoftext|>"]
    if "<|assistant|>" in tpl and "<|end|>" in tpl:                      # Phi-3 / Phi-4
        return (
            "{{- if .System }}<|system|>{{ .System }}<|end|>{{ end }}"
            "{{- range .Messages }}<|{{ .Role }}|>{{ .Content }}<|end|>{{ end }}<|assistant|>"
        ), ["<|end|>", "<|endoftext|>"]
    if "<|turn>" in tpl and "<turn|>" in tpl:                            # Gemma 4
        suffix = gen.rsplit("<|turn>model\n", 1)[-1]
        return (
            "{{- if .System }}<|turn>system\n{{ .System }}<turn|>\n{{ end }}"
            "{{- range .Messages }}<|turn>{{ if eq .Role \"assistant\" }}model{{ else }}{{ .Role }}{{ end }}\n"
            "{{ .Content }}<turn|>\n{{ end }}<|turn>model\n" + suffix
        ), ["<turn|>"]
    return None, []


# Gemma 4 E2B/E4B templates, unlike the 12B+ one, do not close an empty "thought" channel in the
# generation prompt when thinking is off, so those models open a thinking block anyway and burn the
# reply budget on it. Make them behave like the larger ones. Exact, narrow match; any other
# template is returned untouched.
_BS = chr(92)   # a literal backslash: Jinja source contains the two characters \ and n, not a newline
_G4_OLD = ("{{- '<|turn>model" + _BS + "n' -}}\n"
           "    {%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}")
_G4_NEW = ("{{- '<|turn>model" + _BS + "n' -}}\n"
           "        {%- if not enable_thinking -%}\n"
           "            {{- '<|channel>thought" + _BS + "n<channel|>' -}}\n"
           "        {%- endif -%}\n"
           "    {%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}")


def normalize_template(template):
    """Return `template` with known per-family fixes applied (idempotent)."""
    if isinstance(template, str) and _G4_OLD in template and "<|channel>thought" in template:
        return template.replace(_G4_OLD, _G4_NEW, 1)
    return template


def prepare_tokenizer(tokenizer):
    """Apply normalize_template() to a tokenizer's chat template in place."""
    tpl = getattr(tokenizer, "chat_template", None)
    new = normalize_template(tpl)
    if new is not tpl and new != tpl:
        tokenizer.chat_template = new
    return tokenizer
