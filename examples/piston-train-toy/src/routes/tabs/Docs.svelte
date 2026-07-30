<script lang="ts">
	import {
		INFERENCE_PRESET_DOCS,
		SHARED_PIPELINE_STEPS,
		type InferencePresetDoc,
		type InferenceSupport
	} from '$lib/workspace/inferenceDocs';
	import { getPresetOptions } from '$lib/workspace/presets';
	import { Check, ChevronDown, ChevronRight, Clipboard } from '@lucide/svelte/icons';

	const CONVERT_COMMAND = `PYTHONPATH=scripts python -m export_inference convert \\
  path/to/run.inference.safetensors \\
  --out-dir path/to/out`;

	const presetLabels = $derived(
		Object.fromEntries(getPresetOptions().map((p) => [p.value, p.text])) as Record<string, string>
	);

	const orderedDocs = $derived(
		getPresetOptions()
			.map((p) => INFERENCE_PRESET_DOCS.find((d) => d.presetId === p.value))
			.filter((d): d is InferencePresetDoc => d != null)
	);

	let pipelineOpen = $state(true);
	let openCards = $state<Record<string, boolean>>({});
	let copiedKey = $state<string | null>(null);
	let copiedClearHandle = 0;

	function isCardOpen(id: string): boolean {
		return openCards[id] ?? false;
	}

	function toggleCard(id: string) {
		openCards = { ...openCards, [id]: !isCardOpen(id) };
	}

	function supportBadge(support: InferenceSupport): { label: string; className: string } {
		if (support === 'exportable') {
			return { label: 'Exportable', className: 'bg-green-100 text-green-800 border-green-300' };
		}
		if (support === 'unsupported') {
			return {
				label: 'Not in v1',
				className: 'bg-amber-100 text-amber-900 border-amber-300'
			};
		}
		return { label: 'modifier', className: 'bg-purple-100 text-purple-800 border-purple-300' };
	}

	async function copySnippet(key: string, text: string) {
		try {
			await navigator.clipboard.writeText(text);
			copiedKey = key;
			window.clearTimeout(copiedClearHandle);
			copiedClearHandle = window.setTimeout(() => {
				if (copiedKey === key) {
					copiedKey = null;
				}
			}, 1500);
		} catch {
			/* ignore */
		}
	}
</script>

<div class="bg-neutral-100 h-full overflow-auto overscroll-contain flex flex-col flex-1 min-h-0">
	<div
		class="max-w-2xl mx-auto px-3 pb-3 @md:px-4 @md:pb-4 bg-white flex-1 flex flex-col w-full"
	>
		<article class="prose prose-sm max-w-none text-base pb-8">
			<h1 class="text-xl font-semibold mt-4 mb-2">Inference export</h1>
			<p class="text-neutral-700 leading-relaxed mb-4">
				Train in Browser Train, download the purple <span class="text-purple-700 font-medium"
					>ONNX</span
				> package, convert locally, then run with
				<code class="text-sm">examples/browser-train-infer</code> (onnxruntime-web). Transformers.js
				layout from the same converter is planned next.
			</p>

			<nav class="mb-4 not-prose">
				<p class="text-xs font-medium uppercase tracking-wide text-neutral-500 mb-1">Jump to</p>
				<ul class="flex flex-wrap gap-x-3 gap-y-1 text-sm">
					<li>
						<a href="#pipeline" class="text-purple-700 underline underline-offset-2">Pipeline</a>
					</li>
					{#each orderedDocs as doc (doc.presetId)}
						<li>
							<a
								href={`#preset-${doc.presetId}`}
								class="text-purple-700 underline underline-offset-2"
							>
								{presetLabels[doc.presetId] ?? doc.presetId}
							</a>
						</li>
					{/each}
				</ul>
			</nav>

			<section id="pipeline" class="not-prose border border-panel-border-base rounded mb-4">
				<button
					type="button"
					class="w-full flex items-center gap-2 px-3 py-2 text-left font-medium bg-neutral-50 hover:bg-neutral-100 cursor-pointer"
					onclick={() => (pipelineOpen = !pipelineOpen)}
				>
					{#if pipelineOpen}
						<ChevronDown class="w-4 h-4 shrink-0" />
					{:else}
						<ChevronRight class="w-4 h-4 shrink-0" />
					{/if}
					Shared pipeline
				</button>
				{#if pipelineOpen}
					<ol class="list-decimal list-inside px-3 pb-3 space-y-3 text-sm text-neutral-800">
						{#each SHARED_PIPELINE_STEPS as step (step.title)}
							<li>
								<span class="font-medium">{step.title.replace(/^\d+\.\s*/, '')}</span>
								<p class="mt-1 ml-5 text-neutral-600 leading-relaxed">{step.body}</p>
							</li>
						{/each}
					</ol>
					<div class="relative mx-3 mb-3">
						<pre
							class="p-2 pr-10 bg-neutral-900 text-neutral-100 text-xs overflow-x-auto rounded font-mono whitespace-pre-wrap">{CONVERT_COMMAND}</pre>
						<button
							type="button"
							class="absolute top-1.5 right-1.5 p-1 rounded text-neutral-300 hover:text-white hover:bg-neutral-700 cursor-pointer"
							title={copiedKey === 'convert' ? 'Copied' : 'Copy convert command'}
							aria-label={copiedKey === 'convert' ? 'Copied' : 'Copy convert command'}
							onclick={() => void copySnippet('convert', CONVERT_COMMAND)}
						>
							{#if copiedKey === 'convert'}
								<Check class="w-3.5 h-3.5 text-green-400" />
							{:else}
								<Clipboard class="w-3.5 h-3.5" />
							{/if}
						</button>
					</div>
				{/if}
			</section>

			<h2 class="text-lg font-semibold mt-6 mb-2">Per-preset examples</h2>
			<p class="text-sm text-neutral-600 mb-3">
				One section per visible training preset. Stock toys often need
				<strong>ONNX export-friendly</strong> (or a <code>*-onnx</code> preset) before the purple
				download works.
			</p>

			<div class="not-prose space-y-2">
				{#each orderedDocs as doc (doc.presetId)}
					{@const badge = supportBadge(doc.support)}
					{@const open = isCardOpen(doc.presetId)}
					<section
						id={`preset-${doc.presetId}`}
						class="border border-panel-border-base rounded overflow-hidden"
					>
						<button
							type="button"
							class="w-full flex items-start gap-2 px-3 py-2 text-left bg-neutral-50 hover:bg-neutral-100 cursor-pointer"
							onclick={() => toggleCard(doc.presetId)}
						>
							{#if open}
								<ChevronDown class="w-4 h-4 shrink-0 mt-0.5" />
							{:else}
								<ChevronRight class="w-4 h-4 shrink-0 mt-0.5" />
							{/if}
							<div class="flex-1 min-w-0">
								<div class="flex flex-wrap items-center gap-2">
									<span class="font-medium text-sm">
										{presetLabels[doc.presetId] ?? doc.presetId}
									</span>
									<span
										class={`text-[10px] uppercase tracking-wide px-1.5 py-0.5 border rounded ${badge.className}`}
									>
										{badge.label}
									</span>
									<span class="text-[10px] font-mono text-neutral-500">
										{doc.architecture}
									</span>
								</div>
								<p class="text-xs text-neutral-500 mt-0.5 truncate">{doc.howToSelect}</p>
							</div>
						</button>
						{#if open}
							<div class="px-3 pb-3 space-y-3 text-sm border-t border-panel-border-base pt-3">
								<div>
									<p class="text-xs font-medium uppercase tracking-wide text-neutral-500 mb-1">
										How to select
									</p>
									<p class="text-neutral-800">{doc.howToSelect}</p>
								</div>
								<div>
									<p class="text-xs font-medium uppercase tracking-wide text-neutral-500 mb-1">
										Export checklist
									</p>
									<ul class="list-disc list-inside space-y-1 text-neutral-700">
										{#each doc.exportChecklist as item (item)}
											<li>{item}</li>
										{/each}
									</ul>
								</div>
								<div>
									<p class="text-xs font-medium uppercase tracking-wide text-neutral-500 mb-1">
										Sample input
									</p>
									<div class="relative">
										<code
											class="block text-xs bg-neutral-100 px-2 py-1.5 pr-9 rounded break-all font-mono"
											>{doc.sampleInput}</code
										>
										<button
											type="button"
											class="absolute top-1 right-1 p-1 rounded text-neutral-500 hover:text-neutral-800 hover:bg-neutral-200 cursor-pointer"
											title={copiedKey === `sample-${doc.presetId}`
												? 'Copied'
												: 'Copy sample input'}
											aria-label={copiedKey === `sample-${doc.presetId}`
												? 'Copied'
												: 'Copy sample input'}
											onclick={() => void copySnippet(`sample-${doc.presetId}`, doc.sampleInput)}
										>
											{#if copiedKey === `sample-${doc.presetId}`}
												<Check class="w-3.5 h-3.5 text-green-600" />
											{:else}
												<Clipboard class="w-3.5 h-3.5" />
											{/if}
										</button>
									</div>
								</div>
								<div>
									<p class="text-xs font-medium uppercase tracking-wide text-neutral-500 mb-1">
										Expected output
									</p>
									<p class="text-neutral-700">{doc.expectedOutput}</p>
								</div>
								{#if doc.notes}
									<p
										class="text-amber-900 bg-amber-50 border border-amber-200 rounded px-2 py-1.5 text-xs leading-relaxed"
									>
										{doc.notes}
									</p>
								{/if}
								<div>
									<p class="text-xs font-medium uppercase tracking-wide text-neutral-500 mb-1">
										Integration (browser-train-infer)
									</p>
									<div class="relative">
										<pre
											class="p-2 pr-10 bg-neutral-900 text-neutral-100 text-xs overflow-x-auto rounded font-mono whitespace-pre-wrap">{doc.integrationSnippet}</pre>
										<button
											type="button"
											class="absolute top-1.5 right-1.5 p-1 rounded text-neutral-300 hover:text-white hover:bg-neutral-700 cursor-pointer"
											title={copiedKey === `snippet-${doc.presetId}`
												? 'Copied'
												: 'Copy integration snippet'}
											aria-label={copiedKey === `snippet-${doc.presetId}`
												? 'Copied'
												: 'Copy integration snippet'}
											onclick={() =>
												void copySnippet(`snippet-${doc.presetId}`, doc.integrationSnippet)}
										>
											{#if copiedKey === `snippet-${doc.presetId}`}
												<Check class="w-3.5 h-3.5 text-green-400" />
											{:else}
												<Clipboard class="w-3.5 h-3.5" />
											{/if}
										</button>
									</div>
								</div>
							</div>
						{/if}
					</section>
				{/each}
			</div>
		</article>
	</div>
</div>
