import { complete, encodeDecode, encodeMasked, loadModel } from './infer';
import type { LoadedModel } from './types';

let model: LoadedModel | null = null;

const statusEl = document.getElementById('status') as HTMLParagraphElement;
const outputEl = document.getElementById('output') as HTMLDivElement;
const modelUrlEl = document.getElementById('modelUrl') as HTMLInputElement;
const promptEl = document.getElementById('prompt') as HTMLTextAreaElement;
const modeEl = document.getElementById('mode') as HTMLSelectElement;
const loadBtn = document.getElementById('loadBtn') as HTMLButtonElement;
const runBtn = document.getElementById('runBtn') as HTMLButtonElement;

const setStatus = (msg: string, isError = false) => {
	statusEl.textContent = msg;
	statusEl.classList.toggle('error', isError);
};

loadBtn.addEventListener('click', async () => {
	loadBtn.disabled = true;
	runBtn.disabled = true;
	setStatus('Loading ONNX model…');
	try {
		model = await loadModel(modelUrlEl.value.trim() || '/model/');
		setStatus(
			`Loaded ${model.architecture} · vocab ${model.manifest.vocabSize} · dataset ${model.manifest.dataset ?? '?'}`
		);
		runBtn.disabled = false;
	} catch (err) {
		model = null;
		setStatus(err instanceof Error ? err.message : String(err), true);
	} finally {
		loadBtn.disabled = false;
	}
});

runBtn.addEventListener('click', async () => {
	if (!model) {
		return;
	}
	runBtn.disabled = true;
	setStatus('Running…');
	try {
		const prompt = promptEl.value;
		const mode =
			modeEl.value === 'auto'
				? model.architecture === 'encoder-decoder'
					? 'encdec'
					: model.architecture === 'encoder'
						? 'encoder'
						: 'decoder'
				: modeEl.value;
		const result =
			mode === 'encdec'
				? await encodeDecode(model, prompt, { maxNewTokens: 48 })
				: mode === 'encoder'
					? await encodeMasked(model, prompt)
					: await complete(model, prompt, { maxNewTokens: 48 });
		outputEl.textContent = result.text ?? JSON.stringify(result.tokens);
		setStatus('Done');
	} catch (err) {
		setStatus(err instanceof Error ? err.message : String(err), true);
	} finally {
		runBtn.disabled = false;
	}
});
