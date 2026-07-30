/**
 * Wipe persisted Browser Train session / checkpoint / shard-cache data.
 * Does not clear the Lil Siggy uploaded corpus (use DatasetControls Clear for that).
 * Callers should also invoke clearPastRuns() for in-memory run history.
 */

import { globalShardCache } from '$lib/train/data/natural/shardCache';

import { checkpointStore } from './checkpointStore';
import { lastSessionStore } from './lastSessionStore';

export async function clearLocalTrainingData(): Promise<void> {
	await Promise.all([
		lastSessionStore.delete(),
		checkpointStore.clear(),
		globalShardCache.clear()
	]);
}
