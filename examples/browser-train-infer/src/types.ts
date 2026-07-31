export type OrtManifest = {
	architecture: 'decoder' | 'encoder-decoder' | 'encoder';
	onnx: string;
	inputs: Array<{ name: string; dtype: string }>;
	outputs: Array<{ name: string; dtype: string }>;
	vocabSize: number;
	embeddingSize: number;
	tokenizer: {
		kind: 'char' | 'hf';
		vocabFile?: string;
		specialTokensFile?: string;
		tokenizerFile?: string;
	};
	dataset?: string;
};

export type CharTokenizer = {
	kind: 'char';
	tokenToId: Record<string, number>;
	idToToken: Record<number, string>;
	bosId: number | null;
	eosId: number | null;
	padId: number | null;
};

export type LoadedModel = {
	architecture: 'decoder' | 'encoder-decoder' | 'encoder';
	manifest: OrtManifest;
	session: import('onnxruntime-web').InferenceSession;
	tokenizer: CharTokenizer | null;
};
