# Browser Train

Static Svelte app for training sequence models in the browser with WebGPU.

Deployed at [browser-train.netlify.app](https://browser-train.netlify.app).

## Natural-language data

Prepare local shards (no third-party demo CDN):

```bash
# from repo root
DATASETS=tinyshakespeare,tinystories,fineweb VOCABS=char,1024 bash scripts/prepare-natural-data.sh
```

See [scripts/DATA.md](../../scripts/DATA.md).

## Inference export (ONNX + Transformers.js)

Train with an ONNX-exportable preset, download via the purple **ONNX** button, then convert
with the **Docs** tab toolkit zip (`/browser-train-onnx-toolkit.zip` — no repo clone).
One convert writes `ort/` (onnxruntime-web) and `transformers-js/` (decoder AutoModel).
Developers: [`scripts/export_inference`](../../scripts/export_inference/README.md).
Demos: [`browser-train-infer`](../browser-train-infer/README.md),
[`browser-train-infer-tjs`](../browser-train-infer-tjs/README.md).

Details: [INFERENCE.md](./INFERENCE.md).

Thanks to Vin Howe for getting this project off the ground.
