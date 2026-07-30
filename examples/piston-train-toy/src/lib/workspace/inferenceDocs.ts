import { getPresetOptions } from '$lib/workspace/presets';

export type InferenceArchitecture = 'decoder' | 'encoder-decoder' | 'encoder' | 'modifier';

export type InferenceSupport = 'exportable' | 'unsupported' | 'modifier';

export interface InferencePresetDoc {
	/** Matches PRESET_DEFINITIONS id */
	presetId: string;
	architecture: InferenceArchitecture;
	support: InferenceSupport;
	/** How to select this preset in the UI */
	howToSelect: string;
	/** Export checklist bullets */
	exportChecklist: string[];
	/** Sample prompt / input for this task */
	sampleInput: string;
	/** Expected output description */
	expectedOutput: string;
	/** Copy-pasteable TypeScript for onnxruntime-web (browser-train-infer) */
	ortSnippet: string;
	/**
	 * Copy-pasteable TypeScript for Transformers.js (browser-train-infer-tjs).
	 * Null when this architecture should stay on ORT for generation.
	 */
	transformersJsSnippet: string | null;
	/** Extra notes (unsupported, ONNX-friendly overlay, etc.) */
	notes?: string;
}

/** Static asset produced by scripts/pack-export-inference-toolkit.sh at build time. */
export const ONNX_TOOLKIT_ZIP_URL = '/browser-train-onnx-toolkit.zip';

/** Visual end-to-end guide steps (Docs tab). */
export const END_TO_END_FLOW: {
	id: string;
	title: string;
	where: 'browser' | 'local' | 'deploy';
	body: string;
}[] = [
	{
		id: 'train',
		title: 'Train',
		where: 'browser',
		body: 'In Browser Train, pick an ONNX-ready preset (*-onnx) or apply ONNX export-friendly. Start training until you like the metrics.'
	},
	{
		id: 'purple',
		title: 'Purple ONNX download',
		where: 'browser',
		body: 'While the run is active, download {run}.inference.safetensors + {run}.model.json (not the gray training checkpoint).'
	},
	{
		id: 'toolkit',
		title: 'Toolkit zip',
		where: 'browser',
		body: 'Download browser-train-onnx-toolkit.zip from this Docs tab. Unzip on your machine — no repo clone.'
	},
	{
		id: 'setup',
		title: 'Local setup (once)',
		where: 'local',
		body: './setup.sh creates a .venv and installs PyTorch. Re-run only on a new machine.'
	},
	{
		id: 'convert',
		title: 'Convert',
		where: 'local',
		body: './convert.sh ~/Downloads/your-run.inference.safetensors -o ./out → out/ort/ and out/transformers-js/.'
	},
	{
		id: 'ort',
		title: 'Deploy: onnxruntime-web',
		where: 'deploy',
		body: 'Copy out/ort/* → public/model/. Use loadModel + complete (decoder) or encodeDecode (EncDec). Example: browser-train-infer.'
	},
	{
		id: 'tjs',
		title: 'Deploy: Transformers.js',
		where: 'deploy',
		body: 'Decoder only: copy out/transformers-js/* → public/models/browser-train/. Use AutoTokenizer + AutoModelForCausalLM. Example: browser-train-infer-tjs. EncDec toys stay on ORT.'
	}
];

export const SHARED_PIPELINE_STEPS: { title: string; body: string }[] = [
	{
		title: '1. Train with exportable settings',
		body: 'Use an ONNX-ready preset (*-onnx) or apply the ONNX export-friendly overlay so attention uses none / none and qkNorm is off. Stock toy presets often enable gating or qkNorm, which block the purple ONNX download until you switch.'
	},
	{
		title: '2. Download the inference package',
		body: 'While a run is active, click the purple ONNX download next to the run id. You get {run}.inference.safetensors (weights + tokenizer) and {run}.model.json (architecture). This is not the same as the gray training checkpoint download.'
	},
	{
		title: '3. Download the conversion toolkit',
		body: 'Download browser-train-onnx-toolkit.zip from this Docs tab (no repo clone). Unzip, run ./setup.sh once (creates a local .venv and installs PyTorch), then convert with ./convert.sh.'
	},
	{
		title: '4. Convert locally (ORT + Transformers.js)',
		body: 'From the unzipped toolkit: ./convert.sh ~/Downloads/your-run.inference.safetensors -o ./out. Default --targets both writes out/ort/ (onnxruntime-web) and out/transformers-js/ (HF-style folder for decoder AutoModel). Matching .model.json beside the safetensors file is detected automatically.'
	},
	{
		title: '5. Run with ORT',
		body: 'Copy out/ort/* into a small onnxruntime-web app’s public/model/ (model.onnx, ort-manifest.json, tokenizer files). Call loadModel, then complete (decoder) or encodeDecode (encoder-decoder).'
	},
	{
		title: '6. Run with Transformers.js (decoder)',
		body: 'For decoder-only packages, copy out/transformers-js/* into public/models/browser-train/ and use examples/browser-train-infer-tjs (@huggingface/transformers AutoTokenizer + AutoModelForCausalLM). Encoder-decoder toys stay on the ORT path for generation (Seq2Seq AutoModel is out of v1).'
	}
];

const ENCDEC_TJS_NOTE = `// Encoder-decoder: use ORT encodeDecode (browser-train-infer).
// Transformers.js AutoModelForSeq2SeqLM is not supported in v1.
// Copy out/ort/* → public/model/ and call encodeDecode.`;

function ortEncodeDecode(sample: string): string {
	return `import { loadModel, encodeDecode } from './infer';

// Serve out/ort/* at /model/ (model.onnx, ort-manifest.json, vocab…)
const model = await loadModel('/model/');
const { text, tokens } = await encodeDecode(model, ${JSON.stringify(sample)});
console.log(text, tokens);`;
}

function ortComplete(sample: string, maxNewTokens: number): string {
	return `import { loadModel, complete } from './infer';

// Serve out/ort/* at /model/ (model.onnx, ort-manifest.json, vocab…)
const model = await loadModel('/model/');
const { text, tokens } = await complete(model, ${JSON.stringify(sample)}, {
  maxNewTokens: ${maxNewTokens}
});
console.log(text, tokens);`;
}

function tjsGenerate(sample: string, maxNewTokens: number): string {
	return `import {
  env,
  AutoTokenizer,
  AutoModelForCausalLM
} from '@huggingface/transformers';

// Copy out/transformers-js/* → public/models/browser-train/
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/models/';

const tokenizer = await AutoTokenizer.from_pretrained('browser-train');
const model = await AutoModelForCausalLM.from_pretrained('browser-train', {
  dtype: 'fp32'
});

const encoded = await tokenizer(${JSON.stringify(sample)});
const output = await model.generate({
  ...encoded,
  max_new_tokens: ${maxNewTokens},
  do_sample: false
});
const ids = output.tolist ? output.tolist()[0] : output[0];
console.log(tokenizer.decode(ids, { skip_special_tokens: true }));`;
}

/**
 * Per-visible-preset inference docs. Labels are resolved via getPresetOptions() at render time.
 */
export const INFERENCE_PRESET_DOCS: InferencePresetDoc[] = [
	{
		presetId: 'sort-characters',
		architecture: 'encoder-decoder',
		support: 'exportable',
		howToSelect:
			'Presets → Sort Characters (stock). For ONNX download, switch to Sort Characters (ONNX) or apply ONNX export-friendly first.',
		exportChecklist: [
			'Train with sort-characters-onnx, or apply onnx-export-friendly on top of this preset',
			'Purple ONNX download must succeed (validator rejects gating / qkNorm)',
			'Convert with ./convert.sh … -o ./out (use out/ort/ for this EncDec task)'
		],
		sampleInput: 'CBA:',
		expectedOutput: 'Decoded sorted tokens, e.g. ABC (exact string depends on vocab / commas).',
		ortSnippet: ortEncodeDecode('CBA:'),
		transformersJsSnippet: ENCDEC_TJS_NOTE,
		notes: 'Stock sort-characters may not be ONNX-exportable until you overlay onnx-export-friendly or use sort-characters-onnx. Deploy with ORT encodeDecode.'
	},
	{
		presetId: 'sort-characters-onnx',
		architecture: 'encoder-decoder',
		support: 'exportable',
		howToSelect: 'Presets → Sort Characters (ONNX).',
		exportChecklist: [
			'Already ONNX-friendly (attention none/none, qkNorm off)',
			'Purple ONNX download → .inference.safetensors + .model.json',
			'Convert → copy out/ort/* into your onnxruntime-web app'
		],
		sampleInput: 'CBA:',
		expectedOutput: 'Decoded sorted characters after the prompt colon.',
		ortSnippet: ortEncodeDecode('CBA:'),
		transformersJsSnippet: ENCDEC_TJS_NOTE
	},
	{
		presetId: 'reverse-sequence',
		architecture: 'encoder-decoder',
		support: 'exportable',
		howToSelect:
			'Presets → Reverse Sequence. Apply ONNX export-friendly (or equivalent settings) before purple ONNX download.',
		exportChecklist: [
			'Apply onnx-export-friendly if the purple ONNX button reports unsupported settings',
			'Match reverse toy format (letters + colon by default)',
			'Convert → out/ort/ + encodeDecode'
		],
		sampleInput: 'ABC:',
		expectedOutput: 'Reversed letter sequence (e.g. CBA for prompt ABC:).',
		ortSnippet: ortEncodeDecode('ABC:'),
		transformersJsSnippet: ENCDEC_TJS_NOTE,
		notes: 'Default reverse preset may need the ONNX export-friendly overlay before export.'
	},
	{
		presetId: 'two-sum',
		architecture: 'encoder-decoder',
		support: 'exportable',
		howToSelect:
			'Presets → Two Sum. Apply ONNX export-friendly before purple ONNX download if needed.',
		exportChecklist: [
			'Apply onnx-export-friendly when stock settings block export',
			'Prompt uses padded numbers + :target= (expression tokens on by default)',
			'Convert → out/ort/ + encodeDecode'
		],
		sampleInput: '03010705:08=',
		expectedOutput:
			'The two addends that sum to the target (format depends on commas / expression tokens).',
		ortSnippet: ortEncodeDecode('03010705:08='),
		transformersJsSnippet: ENCDEC_TJS_NOTE,
		notes: 'Two-sum token strings are zero-padded to maxNumber width; match your training config.'
	},
	{
		presetId: 'tinystories',
		architecture: 'decoder',
		support: 'exportable',
		howToSelect:
			'Presets → TinyStories. Prefer TinyStories (ONNX) or apply ONNX export-friendly before purple ONNX download.',
		exportChecklist: [
			'Use tinystories-onnx or onnx-export-friendly for a clean export path',
			'Purple ONNX download → convert → out/ort/ and/or out/transformers-js/',
			'ORT: complete(); Transformers.js: AutoModelForCausalLM.generate()'
		],
		sampleInput: 'Once upon a time',
		expectedOutput: 'Continuation text from the decoder LM (greedy / configured decoding).',
		ortSnippet: ortComplete('Once upon a time', 64),
		transformersJsSnippet: tjsGenerate('Once upon a time', 64),
		notes: 'Stock tinystories may need onnx-export-friendly before the purple ONNX download. NL models may need the HF tokenizer.json copied into the Transformers.js folder (see tokenizer.note.json).'
	},
	{
		presetId: 'tinystories-onnx',
		architecture: 'decoder',
		support: 'exportable',
		howToSelect: 'Presets → TinyStories (ONNX).',
		exportChecklist: [
			'Already ONNX-friendly',
			'Purple ONNX download → convert',
			'Deploy with ORT complete() or Transformers.js generate()'
		],
		sampleInput: 'Once upon a time',
		expectedOutput: 'Story continuation tokens decoded as text.',
		ortSnippet: ortComplete('Once upon a time', 64),
		transformersJsSnippet: tjsGenerate('Once upon a time', 64)
	},
	{
		presetId: 'fineweb',
		architecture: 'decoder',
		support: 'exportable',
		howToSelect: 'Presets → FineWeb. Large model — export works the same, but convert/run is heavier.',
		exportChecklist: [
			'Apply onnx-export-friendly if settings block purple ONNX download',
			'Expect large .inference.safetensors / ONNX files',
			'Use a short prompt when smoke-testing'
		],
		sampleInput: 'The history of computing',
		expectedOutput: 'Open-ended continuation from the FineWeb-trained decoder.',
		ortSnippet: ortComplete('The history of computing', 32),
		transformersJsSnippet: tjsGenerate('The history of computing', 32),
		notes: 'FineWeb runs are large; keep prompts short when smoke-testing in the browser.'
	},
	{
		presetId: 'dyck-encoder',
		architecture: 'encoder',
		support: 'unsupported',
		howToSelect: 'Presets → Dyck Encoder (MLM).',
		exportChecklist: [
			'Encoder / MLM export is not supported in v1 (Phase 3)',
			'Purple ONNX inference package is for decoder and encoder-decoder only today'
		],
		sampleInput: '( [ ) ]  (masked MLM positions — training-only)',
		expectedOutput: 'N/A until encoder export lands.',
		ortSnippet: `// Not supported yet (Phase 3).
// Intended future shape (illustrative only):
// const model = await loadModel('/model/');
// const logits = await encodeMasked(model, tokens, maskPositions);`,
		transformersJsSnippet: null,
		notes: 'Documented for completeness. Do not expect the purple ONNX download / ORT path to work for this preset yet.'
	},
	{
		presetId: 'onnx-export-friendly',
		architecture: 'modifier',
		support: 'modifier',
		howToSelect:
			'Presets → ONNX export-friendly (overlay). Apply after picking a toy/NL base preset, or start from a *-onnx preset.',
		exportChecklist: [
			'Sets attention / norms to values the ONNX exporter accepts',
			'Does not train a task by itself — combine with a base preset',
			'Then use purple ONNX download on the resulting run'
		],
		sampleInput: '(inherits the base preset task)',
		expectedOutput: '(inherits the base preset API)',
		ortSnippet: `// Modifier only — pick the base task's API after export.
// Encoder-decoder toys → encodeDecode(model, prompt)  // out/ort/
// Decoder LMs → complete(model, prompt)               // out/ort/`,
		transformersJsSnippet: `// Modifier only — after convert:
// Decoder LMs → AutoModelForCausalLM (out/transformers-js/)
// Encoder-decoder → stay on ORT encodeDecode (out/ort/)`,
		notes: 'Required overlay when a stock preset enables gating or qkNorm that block export.'
	}
];

/** Visible training presets from getPresetOptions() that must appear in Docs. */
export function getVisibleTrainingPresetIds(): string[] {
	return getPresetOptions().map((p) => p.value);
}

/** True when every visible preset has an INFERENCE_PRESET_DOCS entry. */
export function inferenceDocsCoverAllVisiblePresets(): boolean {
	const documented = new Set(INFERENCE_PRESET_DOCS.map((d) => d.presetId));
	return getVisibleTrainingPresetIds().every((id) => documented.has(id));
}

export function getInferenceDocForPreset(presetId: string): InferencePresetDoc | undefined {
	return INFERENCE_PRESET_DOCS.find((d) => d.presetId === presetId);
}
