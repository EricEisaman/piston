# Browser Train Infer (onnxruntime-web)

Minimal webapp that loads an ORT package produced by
[`scripts/export_inference`](../../scripts/export_inference/README.md)
(`out/ort/` when converting with default `--targets both`).

## End-to-end (quick)

1. Train in Browser Train with an ONNX-ready preset.
2. Purple **ONNX** download → `{run}.inference.safetensors` + `{run}.model.json`.
3. Docs tab toolkit zip → `./setup.sh` →  
   `./convert.sh ~/Downloads/run.inference.safetensors -o ./out`
4. Copy `out/ort/*` into `public/model/`.
5. `pnpm install && pnpm dev` → Load model → Run.

Works for **decoder** (`complete`) and **encoder-decoder** (`encodeDecode`).
See Browser Train **Docs** for the visual guide and per-preset snippets.

## Setup

```bash
cd examples/browser-train-infer
pnpm install
```

Copy converter ORT output into `public/model/`:

```bash
mkdir -p public/model
cp -R /path/to/out/ort/* public/model/
# If you converted with --targets ort only, copy out/* instead.
```

## Develop

```bash
pnpm dev
```

Open the app, click **Load model**, enter a prompt (for sort-characters EncDec try
something like `CBA:`), then **Run**.

## Integrate into another webapp

Copy `src/infer.ts` + `src/types.ts`, depend on `onnxruntime-web`, and call:

```ts
import { loadModel, complete, encodeDecode } from './infer';

const model = await loadModel('/model/');

// Encoder-decoder toys (sort / reverse / two-sum):
const { text } = await encodeDecode(model, 'CBA:');

// Decoder LMs (TinyStories / FineWeb):
// const { text } = await complete(model, 'Once upon a time', { maxNewTokens: 64 });
```

Serve `model.onnx`, `ort-manifest.json`, and tokenizer files as static assets.
For Netlify/static hosts, no special headers are required beyond a correct
`Content-Type` for `.wasm` if you vendor ORT wasm files yourself (Vite handles
this in this example).

## Transformers.js

Decoder packages also emit `out/transformers-js/` from the same convert. Use
[`../browser-train-infer-tjs`](../browser-train-infer-tjs) for
`@huggingface/transformers`. EncDec generation stays in this ORT app.

## Non-goals

- EncDec → Transformers.js Seq2Seq AutoModel (v1)
- RNN / encoder-only MLM (Phase 3)
