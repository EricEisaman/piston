# Inference export (ONNX)

Browser Train can export a **weights-only** package for use outside the trainer.

## In the UI

1. Train with an exportable preset:
   - **Toy: Sort Characters (ONNX exportable)**
   - **TinyStories (ONNX exportable)**
   - or apply **ONNX export-friendly (transformer)** on top of your config
2. While a run is active, click the purple **ONNX** download control (next to the normal checkpoint download).
3. You get:
   - `{runId}.inference.safetensors` — model weights, no optimizer
   - `{runId}.model.json` — architecture card + tokenizer metadata

The green/normal download remains a **full training checkpoint** for resume inside Browser Train.

## Convert to ONNX

```bash
pip install -r scripts/export_inference/requirements.txt
PYTHONPATH=scripts python -m export_inference convert \
  ./my-run.inference.safetensors \
  -o ./out-onnx \
  --model-json ./my-run.model.json
```

See [`scripts/export_inference/README.md`](../../scripts/export_inference/README.md).

## Run in another webapp

Use [`examples/browser-train-infer`](../browser-train-infer/README.md) as a template
(`onnxruntime-web` + `loadModel` / `complete` / `encodeDecode`).

## Unsupported in v1

- RoPE, ALiBi, GQA, attention gating, sinks, softcap
- Encoder-only MLM and RNN (Phase 3)
- Dropping an inference package back into Browser Train to resume training (use the full `.safetensors` for that)
