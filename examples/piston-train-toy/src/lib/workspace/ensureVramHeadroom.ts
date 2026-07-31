import type { Config } from './config';

const MIN_VRAM_MB = 8192;
const HEAVY_VRAM_MB = 32_768;

const HEAVY_PRESETS = new Set([
	'tinystories',
	'tinystories-onnx',
	'fineweb',
	'fineweb-onnx',
	'lil-siggy',
	'lil-siggy-onnx'
]);

/**
 * Raise a restored/imported config's software VRAM ceiling if it is too low.
 * Mutates and returns the same config object.
 */
export function ensureVramHeadroom(cfg: Config): Config {
	if (!cfg.training.vramLimitMb.present) {
		return cfg;
	}
	const floor = cfg.preset && HEAVY_PRESETS.has(cfg.preset) ? HEAVY_VRAM_MB : MIN_VRAM_MB;
	const prev = cfg.training.vramLimitMb.value;
	if (prev < floor) {
		cfg.training.vramLimitMb.value = floor;
		console.info(
			`[VRAM] Raised software limit from ${prev} MB to ${floor} MB` +
				(cfg.preset ? ` for preset "${cfg.preset}"` : '')
		);
	}
	return cfg;
}
