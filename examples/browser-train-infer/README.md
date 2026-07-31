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

Works for **decoder** (`complete`), **encoder-decoder** (`encodeDecode`), and
**encoder / Dyck** (`encodeMasked`). See Browser Train **Docs** for the visual
guide and per-preset snippets.

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
`CBA:`; for Dyck try `(<mask>)`), then **Run**.

## Integrate into another webapp

Copy `src/infer.ts` + `src/types.ts`, depend on `onnxruntime-web`, and call:

```ts
import { loadModel, complete, encodeDecode, encodeMasked } from './infer';

const model = await loadModel('/model/');

// Encoder-decoder toys (sort / reverse / two-sum):
const { text } = await encodeDecode(model, 'CBA:');

// Encoder / Dyck MLM:
// const { text } = await encodeMasked(model, '(<mask>)');

// Decoder LMs (TinyStories / FineWeb):
// const { text } = await complete(model, 'Once upon a time', { maxNewTokens: 64 });
```

Serve `model.onnx`, `ort-manifest.json`, and tokenizer files as static assets.

## Transformers.js

The same convert also emits `out/transformers-js/`. Use
[`../browser-train-infer-tjs`](../browser-train-infer-tjs) for
`@huggingface/transformers` (CausalLM / Seq2SeqLM / MaskedLM).

## Non-goals

- Arbitrary GQA / gating / qkNorm graphs (use `*-onnx` presets)
- RNN export
