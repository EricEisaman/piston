import {
	AutoModelForCausalLM,
	AutoTokenizer,
	env,
	type PreTrainedModel,
	type PreTrainedTokenizer
} from '@huggingface/transformers';

// Local Hub-style packages under public/models/<id>/
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = '/models/';

let tokenizer: PreTrainedTokenizer | null = null;
let model: PreTrainedModel | null = null;

const statusEl = document.getElementById('status') as HTMLParagraphElement;
const outputEl = document.getElementById('output') as HTMLDivElement;
const modelIdEl = document.getElementById('modelId') as HTMLInputElement;
const promptEl = document.getElementById('prompt') as HTMLTextAreaElement;
const loadBtn = document.getElementById('loadBtn') as HTMLButtonElement;
const runBtn = document.getElementById('runBtn') as HTMLButtonElement;

const setStatus = (msg: string, isError = false) => {
	statusEl.textContent = msg;
	statusEl.classList.toggle('error', isError);
};

loadBtn.addEventListener('click', async () => {
	loadBtn.disabled = true;
	runBtn.disabled = true;
	tokenizer = null;
	model = null;
	const id = modelIdEl.value.trim() || 'browser-train';
	setStatus(`Loading ${id}…`);
	try {
		tokenizer = await AutoTokenizer.from_pretrained(id);
		model = await AutoModelForCausalLM.from_pretrained(id, {
			dtype: 'fp32'
		});
		const cfg = (model as { config?: { model_type?: string; browser_train_architecture?: string } })
			.config;
		if (cfg?.browser_train_architecture === 'encoder-decoder') {
			throw new Error(
				'This package is encoder-decoder. Use browser-train-infer (ORT encodeDecode), not Transformers.js AutoModel.'
			);
		}
		setStatus(`Loaded ${cfg?.model_type ?? 'model'} · ready`);
		runBtn.disabled = false;
	} catch (err) {
		tokenizer = null;
		model = null;
		setStatus(err instanceof Error ? err.message : String(err), true);
	} finally {
		loadBtn.disabled = false;
	}
});

runBtn.addEventListener('click', async () => {
	if (!tokenizer || !model) {
		return;
	}
	runBtn.disabled = true;
	setStatus('Generating…');
	try {
		const prompt = promptEl.value;
		const inputs = tokenizer(prompt, { return_tensor: false });
		const inputIds = Array.isArray(inputs.input_ids?.[0])
			? (inputs.input_ids as number[][])[0]
			: (inputs.input_ids as number[]);

		// Prefer generate() when available; fall back to greedy full-sequence steps.
		const anyModel = model as PreTrainedModel & {
			generate?: (opts: Record<string, unknown>) => Promise<{ tolist?: () => number[][] } | number[][]>;
			forward?: (opts: Record<string, unknown>) => Promise<{ logits: { data: Float32Array; dims: number[] } }>;
		};

		let text: string;
		if (typeof anyModel.generate === 'function') {
			const encoded = await tokenizer(prompt);
			const out = await anyModel.generate({
				...encoded,
				max_new_tokens: 48,
				do_sample: false
			});
			const sequences =
				typeof (out as { tolist?: () => number[][] }).tolist === 'function'
					? (out as { tolist: () => number[][] }).tolist()
					: (out as number[][]);
			text = tokenizer.decode(sequences[0], { skip_special_tokens: true });
		} else if (typeof anyModel.forward === 'function') {
			const ids = [...inputIds];
			const vocabHint = 0;
			for (let step = 0; step < 48; step++) {
				const attention_mask = ids.map(() => 1);
				const result = await anyModel.forward!({
					input_ids: [ids],
					attention_mask: [attention_mask]
				});
				const logits = result.logits;
				const dims = logits.dims;
				const vocab = dims[dims.length - 1] ?? vocabHint;
				const seqLen = dims.length >= 2 ? dims[dims.length - 2] : 1;
				const offset = (seqLen - 1) * vocab;
				let best = 0;
				let bestVal = -Infinity;
				for (let i = 0; i < vocab; i++) {
					const v = logits.data[offset + i];
					if (v > bestVal) {
						bestVal = v;
						best = i;
					}
				}
				ids.push(best);
			}
			text = tokenizer.decode(ids, { skip_special_tokens: true });
		} else {
			throw new Error('Model has neither generate() nor forward(); check Transformers.js version');
		}

		outputEl.textContent = text;
		setStatus('Done');
	} catch (err) {
		setStatus(err instanceof Error ? err.message : String(err), true);
	} finally {
		runBtn.disabled = false;
	}
});
