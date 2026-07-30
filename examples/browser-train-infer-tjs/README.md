# Browser Train Infer (Transformers.js)

Minimal webapp that loads a **decoder-only** package from
`export_inference convert` (`out/transformers-js/` or `--targets transformers-js`).

Encoder-decoder toys are **not** supported here — use
[`../browser-train-infer`](../browser-train-infer) (onnxruntime-web + `encodeDecode`).

## End-to-end (quick)

1. Train in Browser Train with an ONNX-ready **decoder** preset (e.g. TinyStories ONNX).
2. Purple **ONNX** download → `{run}.inference.safetensors` + `{run}.model.json`.
3. Download the Docs tab toolkit zip → `./setup.sh` once →  
   `./convert.sh ~/Downloads/run.inference.safetensors -o ./out`
4. Copy `out/transformers-js/*` into this app’s `public/models/browser-train/`.
5. `pnpm install && pnpm dev` → Load model → Generate.

See Browser Train **Docs** tab for the visual guide and per-preset copy-paste snippets.

## Setup

```bash
cd examples/browser-train-infer-tjs
pnpm install
```

Copy converter Transformers.js output:

```bash
mkdir -p public/models/browser-train
cp -R /path/to/out/transformers-js/* public/models/browser-train/
# If you converted with --targets transformers-js only, copy out/* instead.
```

Required files:

- `config.json` (`model_type: "gpt2"`)
- `tokenizer.json` / `tokenizer_config.json` (char toys) or a real HF `tokenizer.json` (NL)
- `onnx/model.onnx` (and optionally `onnx/decoder_model_merged.onnx`)

## Develop

```bash
pnpm dev
```

Open the app, **Load model**, enter a prompt, **Generate**.

## Integrate into another webapp

Depend on `@huggingface/transformers`, serve the Hub-style folder under `/models/<id>/`, then:

```ts
import {
	env,
	AutoTokenizer,
	AutoModelForCausalLM
} from '@huggingface/transformers';

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/models/';

const tokenizer = await AutoTokenizer.from_pretrained('browser-train');
const model = await AutoModelForCausalLM.from_pretrained('browser-train', {
	dtype: 'fp32'
});

const encoded = await tokenizer('Once upon a time');
const output = await model.generate({
	...encoded,
	max_new_tokens: 64,
	do_sample: false
});
const ids = output.tolist ? output.tolist()[0] : output[0];
console.log(tokenizer.decode(ids, { skip_special_tokens: true }));
```

For natural-language models, if convert wrote `tokenizer.note.json`, copy the matching
HF `tokenizer.json` from Browser Train’s `static/tokenizer` into the package folder.

## Convert reminder

```bash
# from toolkit zip or piston repo
./convert.sh ~/Downloads/run.inference.safetensors -o ./out
# → ./out/ort/                for browser-train-infer
# → ./out/transformers-js/    for this app
```
