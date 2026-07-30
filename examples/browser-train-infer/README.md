# Browser Train Infer (onnxruntime-web)

Minimal webapp that loads an ONNX package produced by
[`scripts/export_inference`](../../scripts/export_inference/README.md).

## Setup

```bash
cd examples/browser-train-infer
pnpm install
```

Copy converter output into `public/model/`:

```bash
mkdir -p public/model
cp -R /path/to/out-onnx/* public/model/
```

## Develop

```bash
pnpm dev
```

Open the app, click **Load model**, enter a prompt (for sort-characters EncDec try
something like `cba:`), then **Run**.

## Integrate into another webapp

Copy `src/infer.ts` + `src/types.ts`, depend on `onnxruntime-web`, and call:

```ts
import { loadModel, complete, encodeDecode } from './infer';

const model = await loadModel('/model/');
const { text } = await encodeDecode(model, 'cba:');
```

Serve `model.onnx`, `ort-manifest.json`, and tokenizer files as static assets.
For Netlify/static hosts, no special headers are required beyond a correct
`Content-Type` for `.wasm` if you vendor ORT wasm files yourself (Vite handles
this in this example).

## Non-goals

- Transformers.js GPT-2 remapping (optional later for narrow decoder-only cases)
- RNN / encoder-only MLM (Phase 3)
