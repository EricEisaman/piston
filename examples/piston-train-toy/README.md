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

## Inference export (ONNX)

Train with an ONNX-exportable preset, download via the purple **ONNX** button, convert with
[`scripts/export_inference`](../../scripts/export_inference/README.md), then run in
[`examples/browser-train-infer`](../browser-train-infer/README.md).

Details: [INFERENCE.md](./INFERENCE.md).

Thanks to Vin Howe for getting this project off the ground.
