# Inference export (ONNX + Transformers.js)

Browser Train can export a **weights-only** package for use outside the trainer.

**In the app:** open the **Docs** tab for a visual end-to-end guide (train → purple
download → toolkit → convert → deploy ORT or Transformers.js) plus per-preset
copy-paste snippets for both runtimes.

## In the UI

1. Train with an exportable preset:
   - **Toy: Sort Characters (ONNX exportable)**
   - **TinyStories (ONNX exportable)**
   - **FineWeb GPT-2-sized (ONNX exportable)**
   - or apply **ONNX export-friendly (transformer)** on top of your config
2. While a run is active, click the purple **ONNX** download control (next to the normal checkpoint download).
3. You get:
   - `{runId}.inference.safetensors` — model weights, no optimizer
   - `{runId}.model.json` — architecture card + tokenizer metadata

The green/normal download remains a **full training checkpoint** for resume inside Browser Train.

## Convert (site users — no repo clone)

1. Open the **Docs** tab and download **browser-train-onnx-toolkit.zip**
   (also served at `/browser-train-onnx-toolkit.zip` on the deployed site).
2. Unzip, then one-time setup (creates a local `.venv`; do not redistribute the venv):

```bash
chmod +x setup.sh convert.sh
./setup.sh
```

3. Convert your purple ONNX download (default: **both** ORT + Transformers.js):

```bash
./convert.sh ~/Downloads/my-run.inference.safetensors -o ./out
```

Produces:

| Path | Consumer |
|------|----------|
| `out/ort/` | [`examples/browser-train-infer`](../browser-train-infer) (`loadModel` / `complete` / `encodeDecode`) |
| `out/transformers-js/` | [`examples/browser-train-infer-tjs`](../browser-train-infer-tjs) (decoder `AutoModelForCausalLM`) |

Encoder-decoder toys: use **ORT** for generation. The Transformers.js folder is layout/tokenizer only (no Seq2Seq AutoModel in v1).

Windows: `setup.bat` / `convert.bat`. See the README inside the zip.

The zip is built at deploy time by [`scripts/pack-export-inference-toolkit.sh`](../../scripts/pack-export-inference-toolkit.sh).

## Convert from a piston checkout (developers)

```bash
pip install -r scripts/export_inference/requirements.txt
PYTHONPATH=scripts python -m export_inference convert \
  ./my-run.inference.safetensors \
  -o ./out \
  --model-json ./my-run.model.json
# --targets ort|transformers-js|both
```

See [`scripts/export_inference/README.md`](../../scripts/export_inference/README.md).

## Unsupported in v1

- RoPE, ALiBi, GQA, attention gating, sinks, softcap
- Encoder-only MLM and RNN (Phase 3)
- EncDec → Transformers.js `AutoModelForSeq2SeqLM`
- Dropping an inference package back into Browser Train to resume training (use the full `.safetensors` for that)
