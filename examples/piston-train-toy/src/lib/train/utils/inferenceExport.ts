import type { Config } from '$lib/workspace/config';

export const INFERENCE_FORMAT = 'browser-train-inference-v1' as const;

export type InferenceArchitecture = 'decoder' | 'encoder-decoder' | 'encoder';

export type InferenceTokenizerSpec =
	| {
			kind: 'char';
			/** token id → character (or special token string) */
			idToToken: Record<string, string>;
			bosId: number | null;
			eosId: number | null;
			padId: number | null;
			/** MLM mask token id (`<mask>` for Dyck / encoder toys) */
			maskId?: number | null;
	  }
	| {
			kind: 'hf';
			/** Relative note: tokenizer.json should sit next to the checkpoint after convert */
			vocabSize: number;
			tokenizerFile?: string;
			bosId?: number | null;
			eosId?: number | null;
			padId?: number | null;
			maskId?: number | null;
	  };

export interface InferenceExportProfile {
	attentionGating: boolean;
	attentionPresent: boolean;
	gqa: boolean;
	mlpGated: boolean;
	mlpPresent: boolean;
	nHeads: number;
	nKvHeads: number;
	positionalEncoding: 'learned' | 'none' | 'sinusoidal' | 'rope' | 'alibi' | 'other';
	qkNorm: boolean;
	sinks: boolean;
	softcapAttention: boolean;
	softcapLogits: boolean;
}

export interface InferenceModelCard {
	architecture: InferenceArchitecture;
	blockSize: number | { source: number; target: number };
	config: Config;
	dataset: string;
	embeddingSize: number;
	exportProfile: InferenceExportProfile;
	format: typeof INFERENCE_FORMAT;
	layers: { decoder: number; encoder: number };
	numSteps: number;
	tokenizer: InferenceTokenizerSpec;
	vocabSize: number;
}

export interface InferenceCheckpointExtra {
	format: typeof INFERENCE_FORMAT;
	model: InferenceModelCard;
	/** Training resume fields omitted / null for inference packages */
	numSteps: number;
	optimizer: null;
}

const OPTIMIZER_PREFIX = 'optimizer.state';

/**
 * Split model weights from optimizer tensors without requiring optimizer extras.
 */
export const splitInferenceLoadedState = (loaded: {
	state: Record<string, unknown>;
	extra?: { config?: Config; model?: InferenceModelCard; numSteps?: number } | null;
}): {
	config: Config | null;
	modelCard: InferenceModelCard | null;
	modelState: Record<string, unknown>;
	numSteps: number;
} => {
	const modelState: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(loaded.state)) {
		if (!key.startsWith(OPTIMIZER_PREFIX)) {
			modelState[key] = value;
		}
	}

	const extra = loaded.extra ?? undefined;
	const modelCard =
		extra && 'model' in extra && extra.model && typeof extra.model === 'object'
			? (extra.model as InferenceModelCard)
			: null;
	const config = modelCard?.config ?? extra?.config ?? null;
	const numSteps =
		typeof extra?.numSteps === 'number'
			? extra.numSteps
			: typeof modelCard?.numSteps === 'number'
				? modelCard.numSteps
				: 0;

	return { config, modelCard, modelState, numSteps };
};

export const buildExportProfile = (config: Config): InferenceExportProfile => {
	const attn = config.model.transformer.attention;
	const pe = config.model.transformer.positionalEncoding;
	const norm = config.model.transformer.normalization;
	const mlp = config.model.transformer.mlp;

	const nKvHeads = attn.present ? attn.nKeyValueHeads : 1;
	const nHeads =
		attn.present && attn.groupedQueryAttention.present
			? nKvHeads * attn.groupedQueryAttention.queryHeadsPerKeyValueHead
			: nKvHeads;

	let positionalEncoding: InferenceExportProfile['positionalEncoding'] = 'none';
	if (pe.present) {
		if (pe.type === 'learned' || pe.type === 'sinusoidal' || pe.type === 'rope' || pe.type === 'alibi') {
			positionalEncoding = pe.type;
		} else {
			positionalEncoding = 'other';
		}
	}

	return {
		attentionGating: Boolean(attn.present && attn.gating?.present),
		attentionPresent: Boolean(attn.present),
		gqa: Boolean(attn.present && attn.groupedQueryAttention.present),
		mlpGated: Boolean(mlp.present && mlp.variant === 'gated'),
		mlpPresent: Boolean(mlp.present),
		nHeads,
		nKvHeads,
		positionalEncoding,
		qkNorm: Boolean(norm.qkNorm?.present),
		sinks: Boolean(attn.present && attn.sinks?.present),
		softcapAttention: Boolean(norm.softcap?.attention?.present),
		softcapLogits: Boolean(norm.softcap?.logits?.present)
	};
};

/**
 * Returns a list of human-readable reasons the config cannot be exported to ONNX v1.
 */
export const getInferenceExportBlockers = (config: Config): string[] => {
	const blockers: string[] = [];

	if (config.model.family !== 'transformer') {
		blockers.push(`model.family must be "transformer" (got "${config.model.family}")`);
	}

	const topology = config.model.topology;
	if (topology !== 'decoder' && topology !== 'encoder-decoder' && topology !== 'encoder') {
		blockers.push(
			`model.topology must be "decoder", "encoder-decoder", or "encoder" for v1-ext export (got "${topology}")`
		);
	}

	const profile = buildExportProfile(config);

	if (!profile.attentionPresent) {
		blockers.push('transformer attention must be enabled');
	}
	if (profile.gqa) {
		blockers.push('disable grouped-query attention (GQA) for ONNX export');
	}
	if (profile.attentionGating) {
		blockers.push('disable attention gating for ONNX export');
	}
	if (profile.qkNorm) {
		blockers.push('disable qkNorm for ONNX export');
	}
	if (profile.sinks) {
		blockers.push('disable attention sinks for ONNX export');
	}
	if (profile.softcapAttention || profile.softcapLogits) {
		blockers.push('disable attention/logits softcap for ONNX export');
	}
	if (
		profile.positionalEncoding === 'rope' ||
		profile.positionalEncoding === 'alibi' ||
		profile.positionalEncoding === 'other'
	) {
		blockers.push(
			`positional encoding must be learned, sinusoidal, or off (got "${profile.positionalEncoding}")`
		);
	}
	if (!profile.mlpPresent) {
		blockers.push('transformer MLP must be enabled');
	}

	return blockers;
};

export const assertInferenceExportable = (config: Config): void => {
	const blockers = getInferenceExportBlockers(config);
	if (blockers.length > 0) {
		throw new Error(
			`Checkpoint is not ONNX-exportable:\n- ${blockers.join('\n- ')}\n\nApply the "onnx-export-friendly" preset (or turn those settings off) and retrain.`
		);
	}
};

export const resolveInferenceArchitecture = (config: Config): InferenceArchitecture => {
	if (config.model.topology === 'encoder-decoder') {
		return 'encoder-decoder';
	}
	if (config.model.topology === 'encoder') {
		return 'encoder';
	}
	return 'decoder';
};
