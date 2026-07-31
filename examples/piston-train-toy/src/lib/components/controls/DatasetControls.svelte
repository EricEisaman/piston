<script lang="ts">
	import type { ConfigParameter } from '$lib/train/data/toy/config';

	import { DATASET_CONFIG_METADATA } from '$lib/train/data';
	import { NATURAL_DATASET_META } from '$lib/train/data/natural';
	import {
		clearLilSiggyCorpusState,
		hydrateLilSiggyCorpusState,
		lilSiggyCorpusState,
		uploadLilSiggyCorpus
	} from '$lib/train/data/natural/customCorpus.svelte';
	import { MAX_CORPUS_BYTES, WARN_CORPUS_BYTES } from '$lib/train/data/natural/customCorpus';
	import { config, equalsConfigDefault, resetConfigToDefaults } from '$lib/workspace/config.svelte';
	import { getShowLowDiversityDatasetError } from '$lib/workspace/ui.svelte';
	import { onMount } from 'svelte';

	import { ControlsNote } from 'example-common';
	import DatasetSample from './DatasetSample.svelte';
	import SelectDataset from './SelectDataset.svelte';
	import { Slider, RadioGroupInput, CheckboxInput } from 'example-common';
	let { datasetName }: { datasetName: typeof config.data.dataset } = $props();

	let datasetConfigMetadata = $derived(DATASET_CONFIG_METADATA[datasetName]);
	let parameters = $derived<Record<string, ConfigParameter> | undefined>(
		'parameters' in datasetConfigMetadata ? datasetConfigMetadata.parameters : undefined
	);
	let datasetConfig = $derived(
		datasetName in config.data.datasets
			? config.data.datasets[datasetName as keyof typeof config.data.datasets]
			: undefined
	);
	const showMaskRatio = $derived(config.model.topology === 'encoder');
	const isNatural = $derived(Object.keys(NATURAL_DATASET_META).includes(config.data.dataset));
	const isLilSiggy = $derived(config.data.dataset === 'lil-siggy');

	const showDivider = $derived(showMaskRatio);

	let fileInput: HTMLInputElement | undefined = $state();

	onMount(() => {
		void hydrateLilSiggyCorpusState();
	});

	$effect(() => {
		if (isLilSiggy) {
			void hydrateLilSiggyCorpusState();
			if (config.data.natural.vocabSize !== 8192) {
				config.data.natural.vocabSize = 8192;
			}
		}
	});

	async function onCorpusFileSelected(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file) return;
		try {
			await uploadLilSiggyCorpus(file);
		} catch {
			/* error surfaced in lilSiggyCorpusState */
		}
	}

	export function isSliderParameter(param: ConfigParameter): boolean {
		return param.type === 'number';
	}

	export function isCheckboxParameter(param: ConfigParameter): boolean {
		return param.type === 'boolean';
	}

	export function getSliderProps(param: ConfigParameter) {
		if (param.type !== 'number') return null;

		return {
			min: param.min || 0,
			max: param.max || 100,
			step: param.step || 1,
			default: param.default as number
		};
	}
</script>

<SelectDataset
	bind:value={config.data.dataset}
	id="dataset-control"
	hasDefaultValue={equalsConfigDefault('data.dataset')}
	onReset={() => resetConfigToDefaults('data.dataset')}
/>

<p class="-mt-1 mb-1 mx-0.5 text-sm py-1">{datasetConfigMetadata?.description}</p>

{#if isLilSiggy}
	<div class="mb-2 space-y-1.5 border border-dashed border-purple-400 bg-purple-50/60 p-2 rounded">
		<p class="text-xs font-medium text-purple-900">Upload corpus (.txt / .md)</p>
		<p class="text-[11px] text-neutral-600 leading-snug">
			Encoded in a background worker with FineWeb BPE vocab 8192 (max
			{MAX_CORPUS_BYTES / (1024 * 1024)} MB). Use Lil Siggy presets to train. The first GPT-2-sized
			step can take a while; the UI may pause while the GPU is busy.
		</p>
		<input
			bind:this={fileInput}
			type="file"
			accept=".txt,.md,text/plain,text/markdown"
			class="hidden"
			onchange={(e) => void onCorpusFileSelected(e)}
		/>
		<div class="flex flex-wrap items-center gap-1.5">
			<button
				type="button"
				class="px-2 py-1 text-xs font-medium rounded border border-purple-600 text-purple-800 bg-white hover:bg-purple-100 cursor-pointer disabled:opacity-50"
				disabled={lilSiggyCorpusState.ingesting}
				onclick={() => fileInput?.click()}
			>
				{lilSiggyCorpusState.manifest ? 'Replace file' : 'Choose file'}
			</button>
			{#if lilSiggyCorpusState.manifest}
				<button
					type="button"
					class="px-2 py-1 text-xs rounded border border-neutral-300 text-neutral-700 bg-white hover:bg-neutral-50 cursor-pointer disabled:opacity-50"
					disabled={lilSiggyCorpusState.ingesting}
					onclick={() => void clearLilSiggyCorpusState()}
				>
					Clear
				</button>
			{/if}
		</div>
		{#if lilSiggyCorpusState.ingesting}
			<p class="text-[11px] text-purple-800 font-mono">
				Encoding in background… {Math.round(lilSiggyCorpusState.progress * 100)}%
			</p>
		{/if}
		{#if lilSiggyCorpusState.error}
			<ControlsNote label="Corpus error" type="error">
				<p>{lilSiggyCorpusState.error}</p>
			</ControlsNote>
		{/if}
		{#if lilSiggyCorpusState.manifest}
			<p class="text-[11px] text-neutral-700 font-mono leading-snug">
				{lilSiggyCorpusState.manifest.fileName}
				· {(lilSiggyCorpusState.manifest.byteLength / 1024).toFixed(0)} KB · train
				{lilSiggyCorpusState.manifest.nTrainTokens.toLocaleString()} tok · val
				{lilSiggyCorpusState.manifest.nValTokens.toLocaleString()} tok
			</p>
		{:else if !lilSiggyCorpusState.ingesting}
			<ControlsNote label="Required" type="warning">
				<p>Upload a corpus before starting Lil Siggy training.</p>
			</ControlsNote>
		{/if}
		{#if lilSiggyCorpusState.warnLarge || (lilSiggyCorpusState.manifest && lilSiggyCorpusState.manifest.byteLength >= WARN_CORPUS_BYTES)}
			<p class="text-[11px] text-amber-800">
				Large corpus — tokenization and training may be slow in the browser.
			</p>
		{/if}
	</div>
{/if}

<DatasetSample />

{#if getShowLowDiversityDatasetError()}
	<div id="low-diversity-dataset-error" class="error-flash">
		<ControlsNote label="Low Diversity" type="error">
			<p>
				Not enough example diversity in the training dataset for a held-out validation set of size {config
					.training.validation.batchSize}. Consider changing dataset parameters or reducing the
				validation batch size.
			</p>
		</ControlsNote>
	</div>
{/if}

<div class="flex flex-col gap-1">
	{#key datasetName}
		{#if parameters && datasetConfig}
			<div class="flex flex-col gap-1">
				{#each Object.entries(parameters) as [paramKey, paramMeta] (paramKey)}
					{@const value = datasetConfig[paramKey as keyof typeof datasetConfig]}
					{#if isSliderParameter(paramMeta)}
						{@const sliderProps = getSliderProps(paramMeta)}
						{#if sliderProps}
							<Slider
								id={`dataset-${paramKey}`}
								label={paramMeta.name}
								bind:value={datasetConfig[paramKey as keyof typeof datasetConfig]}
								min={sliderProps.min}
								max={sliderProps.max}
								step={sliderProps.step}
								hasDefaultValue={value === paramMeta.default}
								onReset={() => {
									datasetConfig[paramKey as keyof typeof datasetConfig] =
										paramMeta.default as never;
								}}
							/>
						{/if}
					{:else if isCheckboxParameter(paramMeta)}
						<CheckboxInput
							id={`dataset-${paramKey}`}
							label={paramMeta.name}
							bind:checked={datasetConfig[paramKey as keyof typeof datasetConfig]}
							hasDefaultValue={value === paramMeta.default}
							onReset={() => {
								datasetConfig[paramKey as keyof typeof datasetConfig] = paramMeta.default as never;
							}}
						/>
					{/if}
				{/each}
			</div>
		{/if}
	{/key}
	{#if isNatural}
		<Slider
			id="dataset-context-size"
			label="Context Size"
			bind:value={config.data.natural.contextSize}
			min={8}
			max={1024}
			step={4}
			hasDefaultValue={equalsConfigDefault('data.natural.contextSize')}
			onReset={() => resetConfigToDefaults('data.natural.contextSize')}
		/>
		{#if isLilSiggy}
			<p class="text-xs text-neutral-600 mx-0.5">
				Tokenizer vocabulary locked to <span class="font-mono">8192</span> (FineWeb BPE used for
				uploads).
			</p>
		{:else}
			<RadioGroupInput
				id="dataset-natural-vocab-size"
				label="Tokenizer Vocabulary Size"
				name="natural-vocab-size"
				bind:value={
					() => String(config.data.natural.vocabSize),
					(v) =>
						(config.data.natural.vocabSize =
							v === 'char' ? 'char' : (parseInt(v) as typeof config.data.natural.vocabSize))
				}
				options={[
					{ value: 'char', label: 'Character-level' },
					{ value: '512', label: '512' },
					{ value: '1024', label: '1024' },
					{ value: '2048', label: '2048' },
					{ value: '4096', label: '4096' },
					{ value: '8192', label: '8192' },
					{ value: '16384', label: '16384' },
					{ value: '32768', label: '32768' },
					{ value: '65536', label: '65536' }
				]}
				hasDefaultValue={equalsConfigDefault('data.natural.vocabSize')}
				onReset={() => resetConfigToDefaults('data.natural.vocabSize')}
			/>
		{/if}
	{/if}
	{#if showDivider}
		<hr class="my-1 border-panel-border-base" />
	{/if}
	{#if showMaskRatio}
		<Slider
			id="dataset-mask-ratio"
			label="Mask Ratio"
			bind:value={config.data.maskRatio}
			min={0.01}
			max={1}
			step={0.01}
			hasDefaultValue={equalsConfigDefault('data.maskRatio')}
			onReset={() => resetConfigToDefaults('data.maskRatio')}
		/>
	{/if}
</div>
