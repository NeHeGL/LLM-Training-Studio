# Using your model in Ollama, LM Studio and other apps

When you press **Generate Model**, the Studio produces one file: a `.gguf`. It contains
the model, its tokenizer, the chat format and your system prompt, so it is self-contained.

You will find it in `models/<ModelName>/gguf/`, for example
`testai-4b-q4_k_m-qwen3.gguf`.

## Quick test inside the Studio

Press **Start Chat** (or run `python chat_test/server.py`). It opens a chat page at
http://localhost:5000 and loads your newest export.

## Ollama

If Ollama is installed and running, the Studio **registers the model for you**. When the
export finishes it prints a line like:

```
Load in Ollama : ollama run testai-4b-q4_k_m-qwen3
```

Run that command in a terminal and start chatting. A few handy extras:

```
ollama list                         # see your models
ollama rm testai-4b-q4_k_m-qwen3    # remove one
```

If Ollama wasn't running during the export, register it afterwards. Next to the `.gguf` the
Studio writes a recipe file with the same name ending in `.ollama`:

```
ollama create my-model -f models/TestAI/gguf/testai-4b-q4_k_m-qwen3.ollama
```

**Using it from your own programs:** Ollama serves a local API at `http://localhost:11434`.

```
curl http://localhost:11434/api/chat -d '{
  "model": "testai-4b-q4_k_m-qwen3",
  "stream": false,
  "messages": [{"role": "user", "content": "Who are you?"}]
}'
```

## LM Studio

1. Open LM Studio, go to **My Models**, and add the folder `models/<ModelName>/gguf/`
   (or use *Import model* and pick the `.gguf` file).
2. Load the model and chat. Your system prompt is already built in. The chat format comes from
   the file itself, so no manual template setup is needed.

## llama.cpp and other apps

Any app that reads GGUF files works, including `llama.cpp` (`llama-server -m your-file.gguf`),
KoboldCpp and Jan. Point it at the `.gguf`.

## Sharing your model

Send the `.gguf` file. That is all anyone needs. For a file you will share, export as
**q4_k_m**: it runs on modest hardware. A bigger format only helps people with the VRAM for it.

## Good to know

- **The model remembers its system prompt.** Whatever you set in *Model Settings* is baked in.
  Apps that let you override the system prompt will override it.
- **Models with a thinking mode** (the newer Qwen and Gemma 4 families) are trained and
  exported with thinking switched off. **Ollama turns thinking on by default** for these
  families and uses its own built-in prompt format for them, so you may see a "thinking"
  section before the answer, and a short token limit can run out before the answer starts.
  To switch it off: `ollama run <model> --think=false`, or type `/set nothink` inside a
  chat, or send `"think": false` in an API request. LM Studio and llama.cpp use the format
  stored in the file, which already has thinking off. Gemma 4 is still experimental.
- **Ollama and unusual chat formats.** For a few model families the Studio can't write a
  verified Ollama recipe. It leaves that part out and Ollama uses the format stored inside the
  `.gguf`. If answers look odd in Ollama, try the same file in LM Studio.
- **Re-exporting** a different format (say q6_k after q4_k_m) is quick and doesn't touch your
  trained adapter. The adapter stays in `models/<ModelName>/lora/`.
