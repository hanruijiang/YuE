# YuE2 CLI inference

A command-line-only distribution of the [YuE2](https://github.com/multimodal-art-projection/YuE) inference runtime. Generate songs from style and lyrics, optionally conditioned on a native ABC score. No agent skill, transcription service, UI, or benchmark toolkit is included.

## Installation

Use Python 3.12 and a fresh virtual environment. The recommended starting point is a BF16-capable NVIDIA GPU with 24 GB VRAM.

```bash
git clone https://github.com/hanruijiang/YuE.git
cd YuE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# NVIDIA: choose the PyTorch wheel appropriate for your driver.
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
yue2 doctor
```

On Windows, activate with `.venv\Scripts\activate`. Install the appropriate PyTorch build before installing the package. The runtime pins PyTorch 2.10.0 on non-macOS and 2.14.1 on macOS. For Intel XPU on Linux, use `uv sync --extra xpu` (mutually exclusive with `fast`). For NVIDIA vLLM acceleration on Linux, install `.[fast]` in a compatible environment and pass `--backend vllm`.

## Generate

```bash
yue2 generate --request examples/song.json --device cuda --output runs/single
yue2 generate --style "English, acoustic pop, piano" --lyrics "[Verse] A new day begins" --seed 42 --device cuda --output runs/custom
```

The default model and decoder are [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) and [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae). They are downloaded from Hugging Face when needed. You can instead specify local checkpoint directories with `--model` and `--vae`, pin revisions with `--revision` and `--vae-revision`, or use `--offline` to forbid downloads.

The JSON request accepts `id`, `style` (or `tags`), `lyrics`, `cot`, `seed`, `cfg_scale`, optional `abc` or `abc_path`, and sampling overrides. IDs must be unique in a batch; paths in `abc_path` are relative to the request file.

- `--cot full`: melody and chords (default).
- `--cot melody`: melody only.
- `--cot off`: no symbolic plan.
- `--stage plan`: generate and save the plan without synthesizing audio or resolving VAE weights.

```bash
yue2 generate --request examples/song.json --abc-file examples/melody.abc --cot melody --output runs/melody
yue2 generate --request examples/song.json --abc-file examples/score.abc --cot full --output runs/score
yue2 generate --request examples/song.json --stage plan --output runs/plan
```

The native ABC score uses Vocal and Ins voices. Empty lyrics are accepted, but do not guarantee voice-free audio. The CLI does not transcribe source recordings.

## Batch inference and saved outputs

Create a JSONL file with one request object per line, each with a unique string ID. For example:

```bash
python -c 'import json; print(json.dumps(json.load(open("examples/song.json", encoding="utf-8"))))' > requests.jsonl
yue2 validate --input requests.jsonl
yue2 batch --input requests.jsonl --device cuda --output runs/batch
yue2 batch --input requests.jsonl --device cuda --output runs/batch --resume
```

Batch processing is sequential (`--concurrency 1`). Generation saves artifacts under the output directory and request ID, including `audio.flac` (48 kHz stereo with the default decoder), `score.abc` when applicable, `plan.json`, semantic tokens, `latent.npy`, and `result.json`. Check truncation flags in the result; a playable file does not guarantee the generation finished without token limits. Resume verifies request, configuration, model and artifact identities. Use a fresh output directory for changed requests.

## Inference utilities

```bash
yue2 validate --request examples/song.json
yue2 inspect runs/single/city_lights
yue2 decode runs/single/city_lights/latent.npy --vae m-a-p/YuE2-Vae --device cuda --output decoded.wav
yue2 export --output offline-pipeline
yue2 generate --request examples/song.json --model offline-pipeline --offline --device cuda --output runs/offline
```

`validate` checks request inputs before loading models. `inspect` verifies saved artifact hashes, not audio quality. `decode` loads only the VAE and accepts finite floating-point latent arrays in [T,C] or [1,C,T] layout. `export` prepares a portable local pipeline for offline inference without selecting a device or loading model tensors. `doctor` checks the eight required package metadata entries and reports available devices; it does not accept generation-specific options or validate optional backends. Existing nonempty output directories are refused unless the command supports verified resume.

## Devices and memory

The default `--device auto` selects CUDA/ROCm, MPS, XPU, then CPU. Explicit devices include `cuda`, `mps`, `xpu`, and `cpu`. ROCm uses the `cuda` device name. The model uses BF16 and the VAE uses FP32; non-NVIDIA paths are experimental. Install a compatible PyTorch wheel for the target device.

```bash
yue2 generate --request examples/song.json --device cuda --low-vram --budget 8 --output runs/low-memory
```

`--low-vram` uses host memory to keep only active AR/NAR weights on the accelerator; long sequences can still exhaust device memory. `--quantization fp8` is optional on supported NVIDIA hardware. Use `--backend torch-eager` for eager execution without CUDA/XPU graphs or `torch.compile`, even when `YUE2_COMPILE` is set. `--quiet` hides progress on stderr. Run `yue2 generate --help` or `yue2 --help` for all options.

## Tests

```bash
python -m pip install '.[test]'
python -m pytest -q
```

The CPU tests use small local models without downloading production weights. They do not establish full-model audio quality or hardware-specific performance.

## License and attribution

Code: [Apache 2.0](LICENSE). Model weights: [CC BY-NC 4.0 with additional creator permission](MODEL_LICENSE); consult the full terms before use. See [third-party notices](THIRD_PARTY_NOTICES.md). Copyright (c) 2026 the YuE2 authors.

Upstream: [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE). Cite the [YuE2 paper](https://arxiv.org/abs/2609.33757) when using YuE2.
