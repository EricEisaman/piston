import * as ort from 'onnxruntime-web';

import type { CharTokenizer, LoadedModel, OrtManifest } from './types';

const joinUrl = (base: string, path: string): string => {
	const normalized = base.endsWith('/') ? base : `${base}/`;
	return new URL(path.replace(/^\//, ''), normalized).toString();
};

const loadJson = async <T>(url: string): Promise<T> => {
	const res = await fetch(url);
	if (!res.ok) {
		throw new Error(`Failed to fetch ${url}: ${res.status}`);
	}
	return (await res.json()) as T;
};

const loadCharTokenizer = async (
	baseUrl: string,
	manifest: OrtManifest
): Promise<CharTokenizer | null> => {
	if (manifest.tokenizer.kind !== 'char') {
		return null;
	}
	const vocabFile = manifest.tokenizer.vocabFile ?? 'vocab.json';
	const specialFile = manifest.tokenizer.specialTokensFile ?? 'special_tokens.json';
	const vocab = await loadJson<Record<string, number>>(joinUrl(baseUrl, vocabFile));
	const special = await loadJson<{
		bosId: number | null;
		eosId: number | null;
		padId: number | null;
	}>(joinUrl(baseUrl, specialFile));
	const idToToken: Record<number, string> = {};
	for (const [token, id] of Object.entries(vocab)) {
		idToToken[id] = token;
	}
	return {
		kind: 'char',
		tokenToId: vocab,
		idToToken,
		bosId: special.bosId ?? null,
		eosId: special.eosId ?? null,
		padId: special.padId ?? null
	};
};

/**
 * Load an ONNX model + manifest from a directory URL (e.g. `/model/`).
 */
export const loadModel = async (baseUrl: string): Promise<LoadedModel> => {
	const manifest = await loadJson<OrtManifest>(joinUrl(baseUrl, 'ort-manifest.json'));
	const onnxUrl = joinUrl(baseUrl, manifest.onnx);
	const session = await ort.InferenceSession.create(onnxUrl, {
		executionProviders: ['wasm']
	});
	const tokenizer = await loadCharTokenizer(baseUrl, manifest);
	return {
		architecture: manifest.architecture,
		manifest,
		session,
		tokenizer
	};
};

export const encodeText = (tokenizer: CharTokenizer, text: string): number[] => {
	const ids: number[] = [];
	for (const ch of text) {
		const id = tokenizer.tokenToId[ch];
		if (id === undefined) {
			throw new Error(`Unknown token: ${JSON.stringify(ch)}`);
		}
		ids.push(id);
	}
	return ids;
};

export const decodeIds = (tokenizer: CharTokenizer, ids: number[]): string => {
	return ids.map((id) => tokenizer.idToToken[id] ?? '').join('');
};

const argmaxLast = (logits: Float32Array, vocabSize: number, seqLen: number): number => {
	const offset = (seqLen - 1) * vocabSize;
	let best = 0;
	let bestVal = -Infinity;
	for (let i = 0; i < vocabSize; i++) {
		const v = logits[offset + i];
		if (v > bestVal) {
			bestVal = v;
			best = i;
		}
	}
	return best;
};

/**
 * Autoregressive completion for decoder-only ONNX models.
 */
export const complete = async (
	model: LoadedModel,
	prompt: string | number[],
	options: { maxNewTokens?: number } = {}
): Promise<{ tokens: number[]; text: string | null }> => {
	if (model.architecture !== 'decoder') {
		throw new Error('complete() requires a decoder model; use encodeDecode() for encoder-decoder');
	}
	const maxNew = options.maxNewTokens ?? 32;
	let ids =
		typeof prompt === 'string'
			? model.tokenizer
				? encodeText(model.tokenizer, prompt)
				: (() => {
						throw new Error('String prompts require a char tokenizer');
					})()
			: [...prompt];

	for (let step = 0; step < maxNew; step++) {
		const input = new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
		const out = await model.session.run({ input_ids: input });
		const logits = out.logits.data as Float32Array;
		const vocabSize = model.manifest.vocabSize;
		const next = argmaxLast(logits, vocabSize, ids.length);
		ids.push(next);
		if (model.tokenizer?.eosId != null && next === model.tokenizer.eosId) {
			break;
		}
	}

	return {
		tokens: ids,
		text: model.tokenizer ? decodeIds(model.tokenizer, ids) : null
	};
};

/**
 * Greedy decode for encoder-decoder ONNX models (sort / reverse style).
 */
export const encodeDecode = async (
	model: LoadedModel,
	src: string | number[],
	options: { maxNewTokens?: number; bosId?: number } = {}
): Promise<{ tokens: number[]; text: string | null }> => {
	if (model.architecture !== 'encoder-decoder') {
		throw new Error('encodeDecode() requires an encoder-decoder model');
	}
	const maxNew = options.maxNewTokens ?? 32;
	const srcIds =
		typeof src === 'string'
			? model.tokenizer
				? encodeText(model.tokenizer, src)
				: (() => {
						throw new Error('String inputs require a char tokenizer');
					})()
			: [...src];

	const bos =
		options.bosId ??
		model.tokenizer?.bosId ??
		(model.tokenizer ? encodeText(model.tokenizer, '<bos>')[0] : 0);
	const decIds = [bos];

	for (let step = 0; step < maxNew; step++) {
		const enc = new ort.Tensor(
			'int64',
			BigInt64Array.from(srcIds.map(BigInt)),
			[1, srcIds.length]
		);
		const dec = new ort.Tensor(
			'int64',
			BigInt64Array.from(decIds.map(BigInt)),
			[1, decIds.length]
		);
		const out = await model.session.run({
			encoder_input_ids: enc,
			decoder_input_ids: dec
		});
		const logits = out.logits.data as Float32Array;
		const next = argmaxLast(logits, model.manifest.vocabSize, decIds.length);
		decIds.push(next);
		if (model.tokenizer?.eosId != null && next === model.tokenizer.eosId) {
			break;
		}
	}

	return {
		tokens: decIds,
		text: model.tokenizer ? decodeIds(model.tokenizer, decIds) : null
	};
};
