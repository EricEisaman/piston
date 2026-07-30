/**
 * Lil Siggy: user-uploaded corpus → FineWeb-8192 encode → llm.c shards in IndexedDB.
 */

import { openDb, txRequest } from '$lib/dataUtils';
import { PreTrainedTokenizer } from '$lib/train/tokenizer';

export const LIL_SIGGY_DATASET = 'lil-siggy' as const;
/** Shipped tokenizer used to encode Lil Siggy uploads (no in-browser BPE train). */
export const LIL_SIGGY_TOKENIZER_NAME = 'fineweb/8192';
export const LIL_SIGGY_VOCAB_SIZE = 8192 as const;

export const MAX_CORPUS_BYTES = 16 * 1024 * 1024;
export const WARN_CORPUS_BYTES = 2 * 1024 * 1024;
const TOKENS_PER_SHARD = 100_000;
const VAL_FRACTION = 0.05;
const HEADER_INTS = 256;
const MAGIC = 20251003;
const VERSION = 1;
const CHUNK_CHARS = 4_000;

const DB_NAME = 'lil-siggy-corpus';
const DB_VERSION = 1;
const STORE_META = 'meta';
const STORE_SHARDS = 'shards';
const MANIFEST_KEY = 'manifest';

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

function pad4(n: number): string {
	return n.toString().padStart(4, '0');
}

function shardKey(split: 'train' | 'val', index: number): string {
	return `${split}-${pad4(index)}`;
}

function getDb(): Promise<IDBDatabase> {
	return openDb(DB_NAME, DB_VERSION, (db) => {
		if (!db.objectStoreNames.contains(STORE_META)) {
			db.createObjectStore(STORE_META);
		}
		if (!db.objectStoreNames.contains(STORE_SHARDS)) {
			db.createObjectStore(STORE_SHARDS);
		}
	});
}

/** Build an llm.c-style uint16 shard ArrayBuffer (magic 20251003). */
export function buildLlmCShardBuffer(tokenIds: number[]): ArrayBuffer {
	if (tokenIds.length === 0) {
		throw new Error('Cannot write empty shard');
	}
	if (tokenIds.some((id) => id < 0 || id > 0xffff)) {
		throw new Error('Token id out of uint16 range');
	}
	const bytes = new ArrayBuffer(HEADER_INTS * 4 + tokenIds.length * 2);
	const header = new Int32Array(bytes, 0, HEADER_INTS);
	header[0] = MAGIC;
	header[1] = VERSION;
	header[2] = tokenIds.length;
	const tokens = new Uint16Array(bytes, HEADER_INTS * 4);
	tokens.set(tokenIds);
	return bytes;
}

export async function getLilSiggyManifest(): Promise<LilSiggyManifest | null> {
	const db = await getDb();
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
	const db = await getDb();
	const buf = await txRequest<ArrayBuffer | undefined>(db, STORE_SHARDS, 'readonly', (s) =>
		s.get(shardKey(split, index))
	);
	return buf ?? null;
}

export async function clearLilSiggyCorpus(): Promise<void> {
	const db = await getDb();
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

function splitParagraphs(text: string): string[] {
	return text
		.replace(/\r\n/g, '\n')
		.split(/\n{2,}/)
		.map((p) => p.trim())
		.filter((p) => p.length > 0);
}

function* chunkText(text: string, maxChars: number): Generator<string> {
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

async function tokenizeCorpus(
	text: string,
	tokenizer: PreTrainedTokenizer,
	onProgress?: (ratio: number) => void
): Promise<number[]> {
	const eos = tokenizer.eosTokenId ?? tokenizer.bosTokenId ?? 0;
	const paragraphs = splitParagraphs(text);
	const pieces: string[] = [];
	for (const paragraph of paragraphs) {
		for (const piece of chunkText(paragraph, CHUNK_CHARS)) {
			pieces.push(piece);
		}
	}
	if (pieces.length === 0) {
		throw new Error('Corpus has no tokenizable text');
	}

	const ids: number[] = [];
	for (let i = 0; i < pieces.length; i++) {
		const pieceIds = tokenizer.encode(pieces[i], { addSpecialTokens: false });
		ids.push(...pieceIds, eos);
		onProgress?.((i + 1) / pieces.length * 0.85);
		if (i % 8 === 0) {
			await new Promise((r) => setTimeout(r, 0));
		}
	}
	return ids;
}

function chunkTokenIds(ids: number[]): number[][] {
	const chunks: number[][] = [];
	for (let i = 0; i < ids.length; i += TOKENS_PER_SHARD) {
		chunks.push(ids.slice(i, i + TOKENS_PER_SHARD));
	}
	return chunks.length > 0 ? chunks : [];
}

/**
 * Ingest a UTF-8 text/markdown file into IndexedDB as Lil Siggy train/val shards.
 */
export async function ingestLilSiggyCorpus(
	file: File,
	onProgress?: (ratio: number) => void
): Promise<LilSiggyManifest> {
	if (file.size > MAX_CORPUS_BYTES) {
		throw new Error(
			`Corpus too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Max is ${MAX_CORPUS_BYTES / (1024 * 1024)} MB.`
		);
	}
	const text = await file.text();
	if (!text.trim()) {
		throw new Error('Corpus file is empty');
	}

	onProgress?.(0.02);
	await clearLilSiggyCorpus();
	onProgress?.(0.05);

	const tokenizer = await PreTrainedTokenizer.fromPretrained(LIL_SIGGY_TOKENIZER_NAME);
	const allIds = await tokenizeCorpus(text, tokenizer, onProgress);

	const splitAt = Math.max(1, Math.floor(allIds.length * (1 - VAL_FRACTION)));
	const trainIds = allIds.slice(0, splitAt);
	const valIds =
		allIds.slice(splitAt).length > 0
			? allIds.slice(splitAt)
			: allIds.slice(-Math.min(1024, allIds.length));

	const trainChunks = chunkTokenIds(trainIds);
	const valChunks = chunkTokenIds(valIds).slice(0, 1);

	const db = await getDb();
	let written = 0;
	const totalWrites = trainChunks.length + valChunks.length;

	for (let i = 0; i < trainChunks.length; i++) {
		const buf = buildLlmCShardBuffer(trainChunks[i]);
		await txRequest(db, STORE_SHARDS, 'readwrite', (s) => s.put(buf, shardKey('train', i)));
		written += 1;
		onProgress?.(0.85 + (written / totalWrites) * 0.12);
	}
	for (let i = 0; i < valChunks.length; i++) {
		const buf = buildLlmCShardBuffer(valChunks[i]);
		await txRequest(db, STORE_SHARDS, 'readwrite', (s) => s.put(buf, shardKey('val', i)));
		written += 1;
		onProgress?.(0.85 + (written / totalWrites) * 0.12);
	}

	const manifest: LilSiggyManifest = {
		byteLength: file.size,
		fileName: file.name,
		nTrainShards: trainChunks.length,
		nTrainTokens: trainIds.length,
		nValShards: valChunks.length,
		nValTokens: valIds.length,
		uploadedAt: Date.now(),
		vocabSize: LIL_SIGGY_VOCAB_SIZE
	};
	await txRequest(db, STORE_META, 'readwrite', (s) => s.put(manifest, MANIFEST_KEY));
	onProgress?.(1);
	return manifest;
}
