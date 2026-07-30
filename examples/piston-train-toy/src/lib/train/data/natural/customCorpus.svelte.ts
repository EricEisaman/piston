import {
	clearLilSiggyCorpus,
	getLilSiggyManifest,
	ingestLilSiggyCorpus,
	type LilSiggyManifest,
	MAX_CORPUS_BYTES,
	WARN_CORPUS_BYTES
} from './customCorpus';

export const lilSiggyCorpusState = $state<{
	error: string | null;
	ingesting: boolean;
	manifest: LilSiggyManifest | null;
	progress: number;
	ready: boolean;
	warnLarge: boolean;
}>({
	error: null,
	ingesting: false,
	manifest: null,
	progress: 0,
	ready: false,
	warnLarge: false
});

let hydrated = false;

const UI_PROGRESS_THROTTLE_MS = 100;

/** Load manifest from IndexedDB into reactive state (idempotent). */
export async function hydrateLilSiggyCorpusState(): Promise<void> {
	if (hydrated && lilSiggyCorpusState.manifest) {
		return;
	}
	try {
		const manifest = await getLilSiggyManifest();
		lilSiggyCorpusState.manifest = manifest;
		lilSiggyCorpusState.ready = manifest != null;
		lilSiggyCorpusState.error = null;
		hydrated = true;
	} catch (err) {
		lilSiggyCorpusState.error = err instanceof Error ? err.message : String(err);
		lilSiggyCorpusState.ready = false;
	}
}

export function hasLilSiggyCorpus(): boolean {
	return lilSiggyCorpusState.ready && lilSiggyCorpusState.manifest != null;
}

export async function uploadLilSiggyCorpus(file: File): Promise<void> {
	lilSiggyCorpusState.ingesting = true;
	lilSiggyCorpusState.error = null;
	lilSiggyCorpusState.progress = 0;
	lilSiggyCorpusState.warnLarge = file.size >= WARN_CORPUS_BYTES;
	let lastUiProgressAt = 0;
	try {
		if (file.size > MAX_CORPUS_BYTES) {
			throw new Error(
				`Corpus too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Max is ${MAX_CORPUS_BYTES / (1024 * 1024)} MB.`
			);
		}
		const manifest = await ingestLilSiggyCorpus(file, (ratio) => {
			const now = performance.now();
			if (ratio >= 1 || now - lastUiProgressAt >= UI_PROGRESS_THROTTLE_MS) {
				lastUiProgressAt = now;
				lilSiggyCorpusState.progress = ratio;
			}
		});
		lilSiggyCorpusState.progress = 1;
		lilSiggyCorpusState.manifest = manifest;
		lilSiggyCorpusState.ready = true;
		hydrated = true;
	} catch (err) {
		lilSiggyCorpusState.error = err instanceof Error ? err.message : String(err);
		lilSiggyCorpusState.ready = false;
		lilSiggyCorpusState.manifest = null;
		throw err;
	} finally {
		lilSiggyCorpusState.ingesting = false;
	}
}

export async function clearLilSiggyCorpusState(): Promise<void> {
	await clearLilSiggyCorpus();
	lilSiggyCorpusState.manifest = null;
	lilSiggyCorpusState.ready = false;
	lilSiggyCorpusState.progress = 0;
	lilSiggyCorpusState.error = null;
	lilSiggyCorpusState.warnLarge = false;
}
