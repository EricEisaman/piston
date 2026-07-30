# Browser Train → ONNX export

Convert an inference package from Browser Train into ONNX + tokenizer files for
`onnxruntime-web` (see `examples/browser-train-infer`).

## Install

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
  -o ./out-onnx \
  --model-json ./my-run.model.json
```

Outputs in `out-onnx/`:

| File | Purpose |
|------|---------|
| `model.onnx` | Graph for onnxruntime-web |
| `ort-manifest.json` | Input/output names + tokenizer pointers |
| `model.json` | Full Browser Train model card |
| `vocab.json` / `special_tokens.json` | Char / toy tokenizers |
| `tokenizer.note.json` | Hint for HF tokenizer copy (NL models) |

## Exportable training presets

In Browser Train, use:

- **Toy: Sort Characters (ONNX exportable)** (`sort-characters-onnx`)
- **TinyStories (ONNX exportable)** (`tinystories-onnx`)
- Or layer **ONNX export-friendly (transformer)** on any decoder / encoder-decoder run

Unsupported (converter will refuse): RoPE, ALiBi, GQA, attention gating, sinks, softcap, RNN, encoder-only.

## Phase 3 (not in this package)

Encoder-only MLM and RNN export are deferred.
