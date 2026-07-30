/**
 * Lil Siggy corpus ingest worker: FineWeb BPE encode + llm.c shard writes off the UI thread.
 */

import { txRequest } from '$lib/dataUtils';
import { PreTrainedTokenizer } from '$lib/train/tokenizer';

import {
	appendTokenIds,
	buildLlmCShardBuffer,
	CHUNK_CHARS,
	chunkText,
	chunkTokenSubarray,
	clearLilSiggyCorpus,
	createTokenBuffer,
	getLilSiggyDb,
	type IngestWorkerIn,
	type IngestWorkerOut,
	LIL_SIGGY_TOKENIZER_NAME,
	LIL_SIGGY_VOCAB_SIZE,
	type LilSiggyManifest,
	MANIFEST_KEY,
	MAX_CORPUS_BYTES,
	shardKey,
	splitParagraphs,
	STORE_META,
	STORE_SHARDS,
	TOKENS_PER_SHARD,
	VAL_FRACTION
} from './customCorpusShared';

const PROGRESS_THROTTLE_MS = 100;

function post(msg: IngestWorkerOut): void {
	self.postMessage(msg);
}

function postProgress(ratio: number, lastPostAt: { t: number }, force = false): void {
	const now = performance.now();
	if (!force && now - lastPostAt.t < PROGRESS_THROTTLE_MS) {
		return;
	}
	lastPostAt.t = now;
	post({ type: 'progress', ratio });
}

async function runIngest(fileName: string, byteLength: number, buffer: ArrayBuffer): Promise<LilSiggyManifest> {
	if (byteLength > MAX_CORPUS_BYTES) {
		throw new Error(
			`Corpus too large (${(byteLength / (1024 * 1024)).toFixed(1)} MB). Max is ${MAX_CORPUS_BYTES / (1024 * 1024)} MB.`
		);
	}

	const lastPostAt = { t: 0 };
	postProgress(0.02, lastPostAt, true);

	const text = new TextDecoder('utf-8', { fatal: false }).decode(buffer);
	if (!text.trim()) {
		throw new Error('Corpus file is empty');
	}

	await clearLilSiggyCorpus();
	postProgress(0.05, lastPostAt, true);

	const tokenizer = await PreTrainedTokenizer.fromPretrained(LIL_SIGGY_TOKENIZER_NAME);
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

	const tokenState = createTokenBuffer();
	for (let i = 0; i < pieces.length; i++) {
		const pieceIds = tokenizer.encode(pieces[i]!, { addSpecialTokens: false });
		appendTokenIds(tokenState, pieceIds, eos);
		postProgress(0.05 + ((i + 1) / pieces.length) * 0.8, lastPostAt);
	}
	postProgress(0.85, lastPostAt, true);

	const allIds = tokenState.buf.subarray(0, tokenState.length);
	const splitAt = Math.max(1, Math.floor(allIds.length * (1 - VAL_FRACTION)));
	const trainIds = allIds.subarray(0, splitAt);
	const valSlice = allIds.subarray(splitAt);
	const valIds =
		valSlice.length > 0
			? valSlice
			: allIds.subarray(Math.max(0, allIds.length - Math.min(1024, allIds.length)));

	const trainChunks = chunkTokenSubarray(trainIds, TOKENS_PER_SHARD);
	const valChunks = chunkTokenSubarray(valIds, TOKENS_PER_SHARD).slice(0, 1);

	const db = await getLilSiggyDb();
	let written = 0;
	const totalWrites = trainChunks.length + valChunks.length;

	for (let i = 0; i < trainChunks.length; i++) {
		const buf = buildLlmCShardBuffer(trainChunks[i]!);
		await txRequest(db, STORE_SHARDS, 'readwrite', (s) => s.put(buf, shardKey('train', i)));
		written += 1;
		postProgress(0.85 + (written / totalWrites) * 0.12, lastPostAt);
	}
	for (let i = 0; i < valChunks.length; i++) {
		const buf = buildLlmCShardBuffer(valChunks[i]!);
		await txRequest(db, STORE_SHARDS, 'readwrite', (s) => s.put(buf, shardKey('val', i)));
		written += 1;
		postProgress(0.85 + (written / totalWrites) * 0.12, lastPostAt);
	}

	const manifest: LilSiggyManifest = {
		byteLength,
		fileName,
		nTrainShards: trainChunks.length,
		nTrainTokens: trainIds.length,
		nValShards: valChunks.length,
		nValTokens: valIds.length,
		uploadedAt: Date.now(),
		vocabSize: LIL_SIGGY_VOCAB_SIZE
	};
	await txRequest(db, STORE_META, 'readwrite', (s) => s.put(manifest, MANIFEST_KEY));
	postProgress(1, lastPostAt, true);
	return manifest;
}

self.onmessage = async (ev: MessageEvent<IngestWorkerIn>) => {
	const msg = ev.data;
	if (!msg || msg.type !== 'ingest') {
		return;
	}
	try {
		const manifest = await runIngest(msg.fileName, msg.byteLength, msg.buffer);
		post({ type: 'done', manifest });
	} catch (err) {
		post({
			type: 'error',
			message: err instanceof Error ? err.message : String(err)
		});
	}
};
