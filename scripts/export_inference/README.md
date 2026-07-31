# Browser Train → ONNX / Transformers.js export

Convert an inference package from Browser Train into:

1. **ORT** — onnxruntime-web (`examples/browser-train-infer`)
2. **Transformers.js** — Hub-style folder (`examples/browser-train-infer-tjs`)

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
# optional, for parity_v1_ext ONNX graph checks:
pip install onnxruntime
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
    onnx/…                # model.onnx or EncDec split graphs
    browser_train_card.json
```

- `--targets ort` → flat ORT package at `out/` (back-compat)
- `--targets transformers-js` → HF layout at `out/`

### Decoder (TinyStories / FineWeb / decoder presets)

`transformers-js/config.json` uses `model_type: "gpt2"`. Load with
`AutoTokenizer` + `AutoModelForCausalLM`.

### Encoder-decoder toys (sort / reverse / two-sum)

ORT: `encodeDecode`. Transformers.js: BART-compatible packaging +
`AutoModelForSeq2SeqLM` (`encoder_model.onnx`, `decoder_model.onnx`,
`decoder_model_merged.onnx`).

### Encoder / Dyck (MLM)

ORT: `encodeMasked` (char vocab includes `<mask>`). Transformers.js:
BERT-compatible `BertForMaskedLM` + `AutoModelForMaskedLM`.

## Exportable training presets

In Browser Train, use:

- **Toy: Sort Characters (ONNX exportable)** (`sort-characters-onnx`)
- **Toy: Reverse Sequence (ONNX exportable)** (`reverse-sequence-onnx`)
- **Toy: Two Sum (ONNX exportable)** (`two-sum-onnx`)
- **Toy: Dyck (Encoder, ONNX exportable)** (`dyck-encoder-onnx`)
- **TinyStories / FineWeb / Lil Siggy (ONNX exportable)**
- Or layer **ONNX export-friendly (transformer)** on any decoder / EncDec / encoder run

Unsupported (converter will refuse): RoPE, ALiBi, GQA, attention gating, qkNorm, sinks, softcap, RNN.

## Parity (v1-ext)

```bash
PYTHONPATH=scripts python -m export_inference.parity_v1_ext
```

Checks EncDec monolithic ORT path vs split encode/decode (TJS graphs) and encoder
MLM ORT vs TJS ONNX argmax agreement on synthetic weights.
