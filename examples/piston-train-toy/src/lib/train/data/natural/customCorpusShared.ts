/**
 * Shared Lil Siggy corpus helpers (main thread + ingest worker).
 */

import { openDb, txRequest } from '$lib/dataUtils';

export const LIL_SIGGY_DATASET = 'lil-siggy' as const;
/** Shipped tokenizer used to encode Lil Siggy uploads (no in-browser BPE train). */
export const LIL_SIGGY_TOKENIZER_NAME = 'fineweb/8192';
export const LIL_SIGGY_VOCAB_SIZE = 8192 as const;

export const MAX_CORPUS_BYTES = 16 * 1024 * 1024;
export const WARN_CORPUS_BYTES = 2 * 1024 * 1024;
export const TOKENS_PER_SHARD = 100_000;
export const VAL_FRACTION = 0.05;
export const CHUNK_CHARS = 4_000;

const HEADER_INTS = 256;
const MAGIC = 20251003;
const VERSION = 1;

export const DB_NAME = 'lil-siggy-corpus';
export const DB_VERSION = 1;
export const STORE_META = 'meta';
export const STORE_SHARDS = 'shards';
export const MANIFEST_KEY = 'manifest';

export type LilSiggyManifest = {
	byteLength: number;
	fileName: string;
	nTrainShards: number;
	nTrainTokens: number;
	nValShards: number;
	nValTokens: number;
	uploadedAt: number;
	vocabSize: typeof LIL_SIGGY_VOCAB_SIZE;
};

export type IngestWorkerIn =
	| {
			type: 'ingest';
			fileName: string;
			byteLength: number;
			buffer: ArrayBuffer;
	  }
	| { type: 'cancel' };

export type IngestWorkerOut =
	| { type: 'progress'; ratio: number }
	| { type: 'done'; manifest: LilSiggyManifest }
	| { type: 'error'; message: string };

function pad4(n: number): string {
	return n.toString().padStart(4, '0');
}

export function shardKey(split: 'train' | 'val', index: number): string {
	return `${split}-${pad4(index)}`;
}

export function getLilSiggyDb(): Promise<IDBDatabase> {
	return openDb(DB_NAME, DB_VERSION, (db) => {
		if (!db.objectStoreNames.contains(STORE_META)) {
			db.createObjectStore(STORE_META);
		}
		if (!db.objectStoreNames.contains(STORE_SHARDS)) {
			db.createObjectStore(STORE_SHARDS);
		}
	});
}

/**
 * Build an llm.c-style uint16 shard ArrayBuffer (magic 20251003).
 * Sample-checks endpoints instead of scanning every id (hot path).
 */
export function buildLlmCShardBuffer(tokenIds: ArrayLike<number>): ArrayBuffer {
	const n = tokenIds.length;
	if (n === 0) {
		throw new Error('Cannot write empty shard');
	}
	const first = tokenIds[0]!;
	const last = tokenIds[n - 1]!;
	if (first < 0 || first > 0xffff || last < 0 || last > 0xffff) {
		throw new Error('Token id out of uint16 range');
	}
	const bytes = new ArrayBuffer(HEADER_INTS * 4 + n * 2);
	const header = new Int32Array(bytes, 0, HEADER_INTS);
	header[0] = MAGIC;
	header[1] = VERSION;
	header[2] = n;
	const tokens = new Uint16Array(bytes, HEADER_INTS * 4);
	tokens.set(tokenIds);
	return bytes;
}

export async function getLilSiggyManifest(): Promise<LilSiggyManifest | null> {
	const db = await getLilSiggyDb();
	const manifest = await txRequest<LilSiggyManifest | undefined>(
		db,
		STORE_META,
		'readonly',
		(s) => s.get(MANIFEST_KEY)
	);
	return manifest ?? null;
}

export async function getLilSiggyShard(
	split: 'train' | 'val',
	index: number
): Promise<ArrayBuffer | null> {
	const db = await getLilSiggyDb();
	const buf = await txRequest<ArrayBuffer | undefined>(db, STORE_SHARDS, 'readonly', (s) =>
		s.get(shardKey(split, index))
	);
	return buf ?? null;
}

export async function clearLilSiggyCorpus(): Promise<void> {
	const db = await getLilSiggyDb();
	const manifest = await getLilSiggyManifest();
	if (manifest) {
		const keys: string[] = [];
		for (let i = 0; i < manifest.nTrainShards; i++) {
			keys.push(shardKey('train', i));
		}
		for (let i = 0; i < manifest.nValShards; i++) {
			keys.push(shardKey('val', i));
		}
		await Promise.all(
			keys.map((key) => txRequest(db, STORE_SHARDS, 'readwrite', (s) => s.delete(key)))
		);
	}
	await txRequest(db, STORE_META, 'readwrite', (s) => s.delete(MANIFEST_KEY));
}

export function splitParagraphs(text: string): string[] {
	return text
		.replace(/\r\n/g, '\n')
		.split(/\n{2,}/)
		.map((p) => p.trim())
		.filter((p) => p.length > 0);
}

export function* chunkText(text: string, maxChars: number): Generator<string> {
	if (text.length <= maxChars) {
		yield text;
		return;
	}
	let i = 0;
	while (i < text.length) {
		yield text.slice(i, i + maxChars);
		i += maxChars;
	}
}

/** Growable uint16 token buffer (avoids number[] + spread). */
export function createTokenBuffer(initialCapacity = 65_536): {
	buf: Uint16Array;
	length: number;
} {
	return { buf: new Uint16Array(initialCapacity), length: 0 };
}

export function appendTokenIds(
	state: { buf: Uint16Array; length: number },
	ids: ArrayLike<number>,
	eos: number
): void {
	const need = state.length + ids.length + 1;
	if (need > state.buf.length) {
		let cap = state.buf.length;
		while (cap < need) {
			cap *= 2;
		}
		const next = new Uint16Array(cap);
		next.set(state.buf.subarray(0, state.length));
		state.buf = next;
	}
	for (let i = 0; i < ids.length; i++) {
		state.buf[state.length++] = ids[i]!;
	}
	state.buf[state.length++] = eos;
}

export function chunkTokenSubarray(
	ids: Uint16Array,
	tokensPerShard: number
): Uint16Array[] {
	const chunks: Uint16Array[] = [];
	for (let i = 0; i < ids.length; i += tokensPerShard) {
		chunks.push(ids.subarray(i, Math.min(i + tokensPerShard, ids.length)));
	}
	return chunks;
}
