import {
	AutoModelForCausalLM,
	AutoModelForMaskedLM,
	AutoModelForSeq2SeqLM,
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
let architecture: 'decoder' | 'encoder-decoder' | 'encoder' = 'decoder';

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

const resolveArchitecture = async (
	id: string
): Promise<'decoder' | 'encoder-decoder' | 'encoder'> => {
	try {
		const res = await fetch(`/models/${id}/config.json`);
		if (!res.ok) {
			return 'decoder';
		}
		const cfg = (await res.json()) as {
			browser_train_architecture?: string;
			model_type?: string;
		};
		if (cfg.browser_train_architecture === 'encoder-decoder' || cfg.model_type === 'bart') {
			return 'encoder-decoder';
		}
		if (cfg.browser_train_architecture === 'encoder' || cfg.model_type === 'bert') {
			return 'encoder';
		}
	} catch {
		/* default decoder */
	}
	return 'decoder';
};

const resolveMaskTokenId = (tok: PreTrainedTokenizer): number | null => {
	const withMask = tok as PreTrainedTokenizer & {
		mask_token_id?: number | null;
		convert_tokens_to_ids?: (token: string) => number | undefined;
	};
	if (typeof withMask.mask_token_id === 'number') {
		return withMask.mask_token_id;
	}
	for (const candidate of ['<mask>', '[MASK]', '_']) {
		const id = withMask.convert_tokens_to_ids?.(candidate);
		if (typeof id === 'number' && id >= 0) {
			return id;
		}
	}
	return null;
};

const argmaxAt = (data: Float32Array, vocab: number, pos: number): number => {
	const offset = pos * vocab;
	let best = 0;
	let bestVal = -Infinity;
	for (let i = 0; i < vocab; i++) {
		const v = data[offset + i]!;
		if (v > bestVal) {
			bestVal = v;
			best = i;
		}
	}
	return best;
};

loadBtn.addEventListener('click', async () => {
	loadBtn.disabled = true;
	runBtn.disabled = true;
	tokenizer = null;
	model = null;
	const id = modelIdEl.value.trim() || 'browser-train';
	setStatus(`Loading ${id}…`);
	try {
		architecture = await resolveArchitecture(id);
		tokenizer = await AutoTokenizer.from_pretrained(id);
		if (architecture === 'encoder-decoder') {
			model = await AutoModelForSeq2SeqLM.from_pretrained(id, { dtype: 'fp32' });
		} else if (architecture === 'encoder') {
			model = await AutoModelForMaskedLM.from_pretrained(id, { dtype: 'fp32' });
		} else {
			model = await AutoModelForCausalLM.from_pretrained(id, { dtype: 'fp32' });
		}
		const cfg = (model as { config?: { model_type?: string } }).config;
		setStatus(`Loaded ${cfg?.model_type ?? architecture} · ready`);
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
	setStatus(architecture === 'encoder' ? 'Filling masks…' : 'Generating…');
	try {
		const prompt = promptEl.value;
		const anyModel = model as PreTrainedModel & {
			generate?: (opts: Record<string, unknown>) => Promise<{ tolist?: () => number[][] } | number[][]>;
			forward?: (opts: Record<string, unknown>) => Promise<{
				logits: { data: Float32Array; dims: number[] };
			}>;
		};

		let text: string;
		if (architecture === 'encoder-decoder' && typeof anyModel.generate === 'function') {
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
			text = tokenizer.decode(sequences[0]!, { skip_special_tokens: true });
		} else if (architecture === 'encoder' && typeof anyModel.forward === 'function') {
			const encoded = await tokenizer(prompt, { return_tensor: false });
			const inputIds = Array.isArray(encoded.input_ids?.[0])
				? (encoded.input_ids as number[][])[0]!
				: (encoded.input_ids as number[]);
			const attention_mask = inputIds.map(() => 1);
			const token_type_ids = inputIds.map(() => 0);
			const result = await anyModel.forward!({
				input_ids: [inputIds],
				attention_mask: [attention_mask],
				token_type_ids: [token_type_ids]
			});
			const logits = result.logits;
			const dims = logits.dims;
			const vocab = dims[dims.length - 1] ?? 0;
			const maskId = resolveMaskTokenId(tokenizer);
			const positions =
				maskId == null
					? []
					: inputIds.map((id, i) => (id === maskId ? i : -1)).filter((i) => i >= 0);
			if (positions.length === 0) {
				throw new Error(
					'No mask positions found. Include <mask> (Dyck) or [MASK] in the prompt.'
				);
			}
			const filled = [...inputIds];
			for (const pos of positions) {
				filled[pos] = argmaxAt(logits.data, vocab, pos);
			}
			text = tokenizer.decode(filled, { skip_special_tokens: true });
		} else if (typeof anyModel.generate === 'function') {
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
			text = tokenizer.decode(sequences[0]!, { skip_special_tokens: true });
		} else if (typeof anyModel.forward === 'function') {
			const inputs = tokenizer(prompt, { return_tensor: false });
			const inputIds = Array.isArray(inputs.input_ids?.[0])
				? (inputs.input_ids as number[][])[0]!
				: (inputs.input_ids as number[]);
			const ids = [...inputIds];
			for (let step = 0; step < 48; step++) {
				const attention_mask = ids.map(() => 1);
				const result = await anyModel.forward!({
					input_ids: [ids],
					attention_mask: [attention_mask]
				});
				const logits = result.logits;
				const dims = logits.dims;
				const vocab = dims[dims.length - 1] ?? 0;
				const seqLen = dims.length >= 2 ? dims[dims.length - 2]! : 1;
				ids.push(argmaxAt(logits.data, vocab, seqLen - 1));
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
