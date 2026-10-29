# Choosing a model and settings for your GPU

The Studio detects your graphics card and picks safe settings, so you can usually just
pick a model and press Train. This guide explains the choices behind that, in case you
want to change them.

## Which graphics cards work

| Card family | Works? | Notes |
|---|---|---|
| RTX 50xx, 40xx, 30xx, 20xx | Yes | Recommended. `install.bat` picks the right PyTorch build automatically. |
| GTX 10xx (Pascal), Titan V (Volta) | Yes, with limits | Only older PyTorch builds still support these cards, so `install.bat` installs PyTorch 2.14 with CUDA 12.6. That is the last version that will run on them. Needs a recent NVIDIA driver. |
| AMD or Intel graphics, Macs | No | Training needs an NVIDIA GPU (4-bit training uses `bitsandbytes`). |
| No GPU | Technically | 50-100x slower. Not practical. |

If the installer says your **driver is too old**, update it from nvidia.com/drivers and run
`install.bat` again.

## How much video memory (VRAM) do I need?

The model's weights have to fit in VRAM, plus room for the work done while training.
The Studio loads weights in 4-bit ("QLoRA") by default, which needs roughly a quarter of
the memory of full precision with very little quality loss.

Rough guide for **4-bit training**:

| Your VRAM | Comfortable | Possible (slower, shorter sequences) |
|---|---|---|
| 8 GB | up to ~4B (Qwen3 4B, Qwen3.5 4B, Phi-4-mini) | 7-8B models with `--low-mem` |
| 12 GB | up to ~8B (Qwen3 8B, OLMo 3 7B) | 9B |
| 16 GB | 8-9B | 14B |
| 24 GB | 14B | 27B |
| 32 GB | up to ~13B at full precision (BF16); 14B-32B in 4-bit | |
| 48 GB | 32B | |
| 80-96 GB | 32B+ at full precision, 70B-class in 4-bit | |

These are starting points, not guarantees: sequence length, batch size and the model's
architecture all change the real number. The model list in the Studio helps you:

- **⚡** (lightning) = fits at full precision (fastest and best quality; needs a 24 GB+ card)
- **🔵** (blue) = fits with 4-bit QLoRA
- **⛔** = too big for your card (greyed out)
- **⭐** = a good all-round pick for the size
- **🧪** = experimental (newer architecture, less tested)

If training stops with an **out of memory** error, see
[Troubleshooting](troubleshooting.md).

## Picking a model

- **Not sure?** Take the model the Studio pre-selects for your card.
- **Small data, simple goal (a name and personality)?** A 1-4B model is plenty and trains in minutes.
- **Lots of facts or nuanced answers?** Use the biggest model that fits comfortably.
- **Base vs "instruct" models:** instruct versions already know how to chat, which helps with
  small datasets. Several catalog models are base models; the Studio handles both.
- **Gated models** (Llama, Gemma 3) need a Hugging Face login. The catalog avoids them so
  things work out of the box. You can add one yourself under *Model Settings > Advanced*.

## Training settings

The defaults suit most projects. When to change them:

| Setting | Default | Change it when |
|---|---|---|
| Epochs | Auto | Leave blank and the Studio chooses (3-10) from your dataset size, so small datasets get more passes. Type a number to override. If the model parrots answers word for word on unrelated questions, lower it. |
| Learning rate | 2e-4 | Rarely. Lower it (1e-4) if training looks unstable. |
| LoRA rank | 16 (32 for models of 1.5B or less) | You have a lot of varied data: try 32-64. |
| Max sequence length | 2048 | Your answers are long, or you are short on memory (1024 saves a lot). |
| Quantization | 4-bit | You have 24 GB+ and want maximum quality: choose BF16. |
| Hold out 5% | off | You want an overfitting check. It trains on 95% of your data, so about 5% of your facts will not be learned. |

**Advanced LoRA options** (collapsed by default):

- **rsLoRA**: steadier learning at higher ranks (32+). Safe to try.
- **DoRA**: slightly better quality, about 25% slower and a little more memory.
- **NEFTune**: adds noise during training for a chattier style. It can blur exact facts, so leave
  it off for Q&A about specifics.

## Picking an export format

When you export, the file size and quality trade off like this (for a 7B model):

| Format | Size | Quality |
|---|---|---|
| q4_k_m (default) | ~4.4 GB | Good. Runs almost anywhere. |
| q5_k_m | ~5 GB | Very close to the original. |
| q6_k | ~6 GB | Practically indistinguishable. |
| q8_0 | ~8 GB | Essentially lossless. |
| iq4_xs | ~4 GB | Smaller than q4_k_m at similar quality; can be slower on some hardware. |

Tick **Better quality (slower export)** to calibrate the 4/5/6-bit files on your own training
data. Same file size, a little more accuracy. Leave the format on **Auto** when testing on your own
PC; choose **q4_k_m** for a file you plan to share, since it runs on modest hardware.
