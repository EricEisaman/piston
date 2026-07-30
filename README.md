# Browser Train + Piston

Train language models in your browser with WebGPU-powered autodiff.

**Browser Train** is the web playground (Transformers, LSTMs, GRUs, and vanilla RNNs). **Piston** is the WebGPU automatic differentiation library behind it.

Thanks to Vin Howe for getting this project off the ground.

## Develop

```bash
# Prepare natural-language datasets (Karpathy / Hugging Face — local shards only)
DATASETS=tinyshakespeare VOCABS=char,1024 bash scripts/prepare-natural-data.sh

# Full static build (WASM + pnpm)
bash scripts/build-piston-train-toy.sh
```

See [scripts/DATA.md](scripts/DATA.md) and [examples/piston-train-toy](examples/piston-train-toy).
