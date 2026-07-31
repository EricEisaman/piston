# Browser Train Infer (Transformers.js)

Minimal webapp that loads packages from `export_inference convert`
(`out/transformers-js/` or `--targets transformers-js`).

| Architecture | AutoModel | Packaging |
|--------------|-----------|-----------|
| decoder | `AutoModelForCausalLM` | gpt2 |
| encoder-decoder | `AutoModelForSeq2SeqLM` | BART |
| encoder (Dyck) | `AutoModelForMaskedLM` | BERT |

ORT fallback for all three: [`../browser-train-infer`](../browser-train-infer).

## End-to-end (quick)

1. Train in Browser Train with an ONNX-ready preset (e.g. TinyStories ONNX,
   sort-characters-onnx, dyck-encoder-onnx).
2. Purple **ONNX** download → `{run}.inference.safetensors` + `{run}.model.json`.
3. Download the Docs tab toolkit zip → `./setup.sh` once →  
   `./convert.sh ~/Downloads/run.inference.safetensors -o ./out`
4. Copy `out/transformers-js/*` into this app’s `public/models/browser-train/`.
5. `pnpm install && pnpm dev` → Load model → Generate / fill masks.

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

Required files (decoder example):

- `config.json` (`model_type: "gpt2"` | `"bart"` | `"bert"`)
- `tokenizer.json` / `tokenizer_config.json`
- ONNX under `onnx/` (`model.onnx`, or EncDec split encoder/decoder graphs)

## Develop

```bash
pnpm dev
```

Open the app, **Load model**, enter a prompt, **Generate**.
For Dyck, include `<mask>` in the prompt (e.g. `(<mask>)`).

## Integrate into another webapp

```ts
import {
	env,
	AutoTokenizer,
	AutoModelForCausalLM,
	AutoModelForSeq2SeqLM,
	AutoModelForMaskedLM
} from '@huggingface/transformers';

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/models/';

const tokenizer = await AutoTokenizer.from_pretrained('browser-train');
// Pick AutoModel* from config.json model_type / browser_train_architecture
const model = await AutoModelForCausalLM.from_pretrained('browser-train', {
	dtype: 'fp32'
});
```

## Convert reminder

```bash
# from toolkit zip or piston repo
./convert.sh ~/Downloads/run.inference.safetensors -o ./out
# → ./out/ort/                for browser-train-infer
# → ./out/transformers-js/    for this app
```
