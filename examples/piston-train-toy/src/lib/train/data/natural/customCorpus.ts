/**
 * Lil Siggy: user-uploaded corpus → FineWeb-8192 encode → llm.c shards in IndexedDB.
 * Heavy BPE encode runs in {@link ./customCorpusWorker.ts}.
 */

import {
	buildLlmCShardBuffer,
	clearLilSiggyCorpus,
	getLilSiggyManifest,
	getLilSiggyShard,
	type IngestWorkerIn,
	type IngestWorkerOut,
	LIL_SIGGY_DATASET,
	LIL_SIGGY_TOKENIZER_NAME,
	LIL_SIGGY_VOCAB_SIZE,
	type LilSiggyManifest,
	MAX_CORPUS_BYTES,
	WARN_CORPUS_BYTES
} from './customCorpusShared';

export {
	buildLlmCShardBuffer,
	clearLilSiggyCorpus,
	getLilSiggyManifest,
	getLilSiggyShard,
	LIL_SIGGY_DATASET,
	LIL_SIGGY_TOKENIZER_NAME,
	LIL_SIGGY_VOCAB_SIZE,
	MAX_CORPUS_BYTES,
	WARN_CORPUS_BYTES
};
export type { LilSiggyManifest };

/**
 * Ingest a UTF-8 text/markdown file into IndexedDB as Lil Siggy train/val shards.
 * Tokenization and shard writes run in a dedicated module worker.
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

	const buffer = await file.arrayBuffer();
	if (buffer.byteLength === 0) {
		throw new Error('Corpus file is empty');
	}

	return new Promise<LilSiggyManifest>((resolve, reject) => {
		const worker = new Worker(new URL('./customCorpusWorker.ts', import.meta.url), {
			type: 'module',
			name: 'customCorpusWorker'
		});

		const cleanup = () => {
			worker.terminate();
		};

		worker.onmessage = (ev: MessageEvent<IngestWorkerOut>) => {
			const msg = ev.data;
			if (!msg) {
				return;
			}
			if (msg.type === 'progress') {
				onProgress?.(msg.ratio);
				return;
			}
			if (msg.type === 'done') {
				cleanup();
				resolve(msg.manifest);
				return;
			}
			if (msg.type === 'error') {
				cleanup();
				reject(new Error(msg.message));
			}
		};

		worker.onerror = (ev) => {
			cleanup();
			reject(ev.error ?? new Error(ev.message || 'Corpus ingest worker failed'));
		};

		const payload: IngestWorkerIn = {
			type: 'ingest',
			fileName: file.name,
			byteLength: file.size,
			buffer
		};
		worker.postMessage(payload, [buffer]);
	});
}
