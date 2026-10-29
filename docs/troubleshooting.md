# Troubleshooting

Find your symptom, try the fix. Most problems show a clear message in the Studio's log panel.

## Installing

**"Your NVIDIA driver is too old for ..."**
Update your graphics driver from nvidia.com/drivers, then run `install.bat` again.
Current PyTorch builds need a reasonably recent driver (CUDA 12.6 or newer).

**"No kernel image is available for execution on the device"** (or "the GPU is visible but
PyTorch cannot run on it")
The installed PyTorch build doesn't match your card. Delete the `.venv` folder and run
`install.bat` again; it picks the right build from your card and driver. GTX 10-series
(Pascal) and Titan V (Volta) cards use an older PyTorch on purpose, because newer builds dropped them.

**Install fails halfway / something won't download**
Check your internet connection and run `install.bat` again; it picks up where it left off.

## Starting the Studio

**"Server already running on port 5001"**
An older Studio window is still open (or still running in the background). Close it, or end the
`python.exe` process that is running `train/server.py`, then start again. This matters after
updates: the old copy keeps running the old code.

## Choosing a model

**"... is a gated model"**
That model needs a Hugging Face login and license acceptance. Either pick a catalog model
(none of them need a login) or open the model's page on huggingface.co, accept its license, and
paste a token under *Model Settings > Advanced*.

**"... was not found on Hugging Face"**
Check the spelling (`organization/model-name`). Names are case-sensitive.

**The first training run sits at "Loading base model" for a long time**
It is downloading the model (a few GB to tens of GB). It only happens once per model. A free
Hugging Face token under *Model Settings > Advanced* can make downloads faster.

## Training

**"CUDA out of memory"**
Try these in order:
1. Lower **Max seq length** (2048 -> 1024 -> 512).
2. Use **4-bit** quantization instead of BF16/FP16.
3. Lower the batch size to 1.
4. Pick a smaller model.
5. Close other programs using the GPU (games, browsers with hardware acceleration, other AI tools).
The command-line flag `--low-mem` applies several of these at once.

**Training finishes instantly or says "dataset too small"**
Press **Build Dataset** first and make sure you have Q&A pairs. See
[Writing good training data](writing-good-training-data.md).

**Loss goes to 0 very quickly / the model parrots answers**
That's overfitting: too many epochs for too little data. Lower the epochs or add more varied pairs.

**Training was interrupted**
Press **Train** again; it resumes from the last checkpoint. Use **Reset + Train** to start over.

## Exporting

**"GGUF conversion failed"**
Read the last lines of the log. The merged model is kept in `models/<ModelName>/gguf/` so you
can retry. Common causes: low disk space (exports need several GB free) or a brand-new model
architecture the converter doesn't know yet. If it mentions a missing key, update the Studio.

**`q4_k_m` export falls back to a bigger file / "llama-quantize not found"**
`install.bat` downloads `llama-quantize` in its last step. If that step failed, run
`install.bat` again, or place `llama-quantize.exe` in `tools/llama/`.

**"Ollama registration failed"**
Make sure Ollama is installed and running, then use the `ollama create` command shown in
[Using your model](using-your-model.md). If no `.gguf` file was produced, fix the conversion error first.

## The model's answers

**It doesn't know my facts / gives generic answers**
- Add more pairs, with each important fact asked in several different ways.
- Try 4-5 epochs.
- Test with the exact questions from your data first, then reworded ones.
- Remember that fine-tuning teaches style and a focused set of facts; it isn't a search engine.

**It makes things up**
Normal for any language model, and worse when questions drift away from your data. Add a few
"I don't know" examples for things outside your topic.

**It shows a "thinking" section or `<think>` text**
Models with a thinking mode are trained with it off. If you still see it, retrain with a few
more epochs, or try the file in a different app. Gemma 4 is experimental in this respect.

## Still stuck?

Open an issue on GitHub and include the last 30 lines of the Studio log, your GPU and driver version
(`nvidia-smi`), and the model you were using.
