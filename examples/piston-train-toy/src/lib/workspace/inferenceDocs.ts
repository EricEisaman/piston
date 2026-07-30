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
	/** Copy-pasteable TypeScript for browser-train-infer */
	integrationSnippet: string;
	/** Extra notes (unsupported, ONNX-friendly overlay, etc.) */
	notes?: string;
}

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
		title: '3. Convert to ONNX locally',
		body: 'From the piston repo root: PYTHONPATH=scripts python -m export_inference convert path/to/run.inference.safetensors --out-dir path/to/out. Install deps from scripts/export_inference/requirements.txt if needed.'
	},
	{
		title: '4. Drop into browser-train-infer',
		body: 'Copy the converter output into examples/browser-train-infer/public/model/ (model.onnx, tokenizer.json, config.json as produced). Serve that app and call loadModel, then complete or encodeDecode.'
	},
	{
		title: '5. Dual consumers',
		body: 'ORT (onnxruntime-web) via loadModel / complete / encodeDecode is shipped in examples/browser-train-infer. A Transformers.js package layout from the same converter is planned next and not available yet.'
	}
];

/**
 * Per-visible-preset inference docs. Labels are resolved via getPresetOptions() at render time.
 */
export const INFERENCE_PRESET_DOCS: InferencePresetDoc[] = [
	{
		presetId: 'sort-characters',
		architecture: 'encoder-decoder',
		support: 'exportable',
		howToSelect: 'Presets → Sort Characters (stock). For ONNX download, switch to Sort Characters (ONNX) or apply ONNX export-friendly first.',
		exportChecklist: [
			'Train with sort-characters-onnx, or apply onnx-export-friendly on top of this preset',
			'Purple ONNX download must succeed (validator rejects gating / qkNorm)',
			'Convert with export_inference convert'
		],
		sampleInput: 'CBA:',
		expectedOutput: 'Decoded sorted tokens, e.g. ABC (exact string depends on vocab / commas).',
		integrationSnippet: `import { loadModel, encodeDecode } from './infer';

const model = await loadModel('/model/');
const { text, tokens } = await encodeDecode(model, 'CBA:');
console.log(text, tokens);`,
		notes: 'Stock sort-characters may not be ONNX-exportable until you overlay onnx-export-friendly or use sort-characters-onnx.'
	},
	{
		presetId: 'sort-characters-onnx',
		architecture: 'encoder-decoder',
		support: 'exportable',
		howToSelect: 'Presets → Sort Characters (ONNX).',
		exportChecklist: [
			'Already ONNX-friendly (attention none/none, qkNorm off)',
			'Purple ONNX download → .inference.safetensors + .model.json',
			'Convert with export_inference convert'
		],
		sampleInput: 'CBA:',
		expectedOutput: 'Decoded sorted characters after the prompt colon.',
		integrationSnippet: `import { loadModel, encodeDecode } from './infer';

const model = await loadModel('/model/');
const { text, tokens } = await encodeDecode(model, 'CBA:');
console.log(text, tokens);`
	},
	{
		presetId: 'reverse-sequence',
		architecture: 'encoder-decoder',
		support: 'exportable',
		howToSelect: 'Presets → Reverse Sequence. Apply ONNX export-friendly (or equivalent settings) before purple ONNX download.',
		exportChecklist: [
			'Apply onnx-export-friendly if the purple ONNX button reports unsupported settings',
			'Match reverse toy format (letters + colon by default)',
			'Convert with export_inference convert'
		],
		sampleInput: 'ABC:',
		expectedOutput: 'Reversed letter sequence (e.g. CBA for prompt ABC:).',
		integrationSnippet: `import { loadModel, encodeDecode } from './infer';

const model = await loadModel('/model/');
const { text, tokens } = await encodeDecode(model, 'ABC:');
console.log(text, tokens);`,
		notes: 'Default reverse preset may need the ONNX export-friendly overlay before export.'
	},
	{
		presetId: 'two-sum',
		architecture: 'encoder-decoder',
		support: 'exportable',
		howToSelect: 'Presets → Two Sum. Apply ONNX export-friendly before purple ONNX download if needed.',
		exportChecklist: [
			'Apply onnx-export-friendly when stock settings block export',
			'Prompt uses padded numbers + :target= (expression tokens on by default)',
			'Convert with export_inference convert'
		],
		sampleInput: '03010705:08=',
		expectedOutput: 'The two addends that sum to the target (format depends on commas / expression tokens).',
		integrationSnippet: `import { loadModel, encodeDecode } from './infer';

const model = await loadModel('/model/');
// Default two-sum: zero-padded numbers, no commas, :target= expression tokens.
const { text, tokens } = await encodeDecode(model, '03010705:08=');
console.log(text, tokens);`,
		notes: 'Two-sum token strings are zero-padded to maxNumber width; match your training config.'
	},
	{
		presetId: 'tinystories',
		architecture: 'decoder',
		support: 'exportable',
		howToSelect: 'Presets → TinyStories. Prefer TinyStories (ONNX) or apply ONNX export-friendly before purple ONNX download.',
		exportChecklist: [
			'Use tinystories-onnx or onnx-export-friendly for a clean export path',
			'Purple ONNX download → convert → public/model/',
			'Call complete (decoder LM), not encodeDecode'
		],
		sampleInput: 'Once upon a time',
		expectedOutput: 'Continuation text from the decoder LM (greedy / configured decoding).',
		integrationSnippet: `import { loadModel, complete } from './infer';

const model = await loadModel('/model/');
const { text, tokens } = await complete(model, 'Once upon a time', { maxNewTokens: 64 });
console.log(text, tokens);`,
		notes: 'Stock tinystories may need onnx-export-friendly before the purple ONNX download.'
	},
	{
		presetId: 'tinystories-onnx',
		architecture: 'decoder',
		support: 'exportable',
		howToSelect: 'Presets → TinyStories (ONNX).',
		exportChecklist: [
			'Already ONNX-friendly',
			'Purple ONNX download → convert → public/model/',
			'Use complete for open-ended generation'
		],
		sampleInput: 'Once upon a time',
		expectedOutput: 'Story continuation tokens decoded as text.',
		integrationSnippet: `import { loadModel, complete } from './infer';

const model = await loadModel('/model/');
const { text, tokens } = await complete(model, 'Once upon a time', { maxNewTokens: 64 });
console.log(text, tokens);`
	},
	{
		presetId: 'fineweb',
		architecture: 'decoder',
		support: 'exportable',
		howToSelect: 'Presets → FineWeb. Large model — export works the same, but convert/run is heavier.',
		exportChecklist: [
			'Apply onnx-export-friendly if settings block purple ONNX download',
			'Expect large .inference.safetensors / ONNX files',
			'Use complete with a short prompt'
		],
		sampleInput: 'The history of computing',
		expectedOutput: 'Open-ended continuation from the FineWeb-trained decoder.',
		integrationSnippet: `import { loadModel, complete } from './infer';

const model = await loadModel('/model/');
const { text, tokens } = await complete(model, 'The history of computing', { maxNewTokens: 32 });
console.log(text, tokens);`,
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
		integrationSnippet: `// Not supported yet (Phase 3).
// Intended future shape (illustrative only):
// const model = await loadModel('/model/');
// const logits = await encodeMasked(model, tokens, maskPositions);`,
		notes: 'Documented for completeness. Do not expect the purple ONNX download / ORT path to work for this preset yet.'
	},
	{
		presetId: 'onnx-export-friendly',
		architecture: 'modifier',
		support: 'modifier',
		howToSelect: 'Presets → ONNX export-friendly (overlay). Apply after picking a toy/NL base preset, or start from a *-onnx preset.',
		exportChecklist: [
			'Sets attention / norms to values the ONNX exporter accepts',
			'Does not train a task by itself — combine with a base preset',
			'Then use purple ONNX download on the resulting run'
		],
		sampleInput: '(inherits the base preset task)',
		expectedOutput: '(inherits the base preset API: encodeDecode or complete)',
		integrationSnippet: `// Modifier only — pick the base task's API after export.
// Encoder-decoder toys → encodeDecode(model, prompt)
// Decoder LMs → complete(model, prompt)`,
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
