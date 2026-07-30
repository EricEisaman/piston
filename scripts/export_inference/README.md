# Browser Train → ONNX / Transformers.js export

Convert an inference package from Browser Train into:

1. **ORT** — onnxruntime-web (`examples/browser-train-infer`)
2. **Transformers.js** — Hub-style folder for decoder `AutoModelForCausalLM`
   (`examples/browser-train-infer-tjs`)

## Site users (recommended)

Do **not** clone this repo. On the deployed Browser Train site, open the **Docs**
tab and download **browser-train-onnx-toolkit.zip** (`/browser-train-onnx-toolkit.zip`).
That zip is packed by `scripts/pack-export-inference-toolkit.sh` and includes
`setup` / `convert` scripts that create a local `.venv` on the user’s machine.

## Install (developers with a piston checkout)

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
pip install -r scripts/export_inference/requirements.txt
```

## Convert

After downloading from Browser Train (**ONNX** download button):

- `my-run.inference.safetensors`
- `my-run.model.json`

```bash
cd /path/to/piston
PYTHONPATH=scripts python -m export_inference convert \
  ./my-run.inference.safetensors \
  -o ./out \
  --model-json ./my-run.model.json
# default --targets both
```

Default layout (`--targets both`):

```
out/
  ort/
    model.onnx
    ort-manifest.json
    vocab.json | tokenizer.note.json
    model.json
  transformers-js/
    config.json
    generation_config.json
    tokenizer.json
    tokenizer_config.json
    onnx/model.onnx
    browser_train_card.json
```

- `--targets ort` → flat ORT package at `out/` (back-compat)
- `--targets transformers-js` → HF layout at `out/`

### Decoder (TinyStories / FineWeb / decoder presets)

`transformers-js/config.json` uses `model_type: "gpt2"` with matching sizes. ONNX accepts
`input_ids` + `attention_mask` → `logits`. Load with `@huggingface/transformers`
`AutoTokenizer` + `AutoModelForCausalLM` (see `browser-train-infer-tjs`).

Natural-language HF tokenizers: copy the matching `tokenizer.json` from Browser Train
`static/tokenizer` into the package when `tokenizer.note.json` is present.

### Encoder-decoder toys

ORT `encodeDecode` is the supported generation path. The Transformers.js folder is still
emitted for packaging consistency but **does not** pretend to be T5/BART; Seq2Seq AutoModel
is out of v1.

## Exportable training presets

In Browser Train, use:

- **Toy: Sort Characters (ONNX exportable)** (`sort-characters-onnx`)
- **TinyStories (ONNX exportable)** (`tinystories-onnx`)
- **FineWeb GPT-2-sized (ONNX exportable)** (`fineweb-onnx`)
- Or layer **ONNX export-friendly (transformer)** on any decoder / encoder-decoder run

Unsupported (converter will refuse): RoPE, ALiBi, GQA, attention gating, sinks, softcap, RNN, encoder-only.

## Phase 3 (not in this package)

Encoder-only MLM and RNN export are deferred.
