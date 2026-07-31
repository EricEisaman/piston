<script lang="ts">
	import {
		END_TO_END_FLOW,
		INFERENCE_PRESET_DOCS,
		ONNX_TOOLKIT_ZIP_URL,
		SHARED_PIPELINE_STEPS,
		type InferencePresetDoc,
		type InferenceSupport
	} from '$lib/workspace/inferenceDocs';
	import { getPresetOptions } from '$lib/workspace/presets';
	import { Check, ChevronDown, ChevronRight, Clipboard, Download } from '@lucide/svelte/icons';

	const CONVERT_COMMAND = `./setup.sh   # once: creates .venv + installs deps
./convert.sh ~/Downloads/your-run.inference.safetensors -o ./out
# → ./out/ort/                 onnxruntime-web
# → ./out/transformers-js/     decoder Transformers.js`;

	const presetLabels = $derived(
		Object.fromEntries(getPresetOptions().map((p) => [p.value, p.text])) as Record<string, string>
	);

	const orderedDocs = $derived(
		getPresetOptions()
			.map((p) => INFERENCE_PRESET_DOCS.find((d) => d.presetId === p.value))
			.filter((d): d is InferencePresetDoc => d != null)
	);

	const flowBrowser = $derived(END_TO_END_FLOW.filter((s) => s.where === 'browser'));
	const flowLocal = $derived(END_TO_END_FLOW.filter((s) => s.where === 'local'));
	const flowDeploy = $derived(END_TO_END_FLOW.filter((s) => s.where === 'deploy'));

	let flowOpen = $state(true);
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

	function whereBadge(where: 'browser' | 'local' | 'deploy'): { label: string; className: string } {
		if (where === 'browser') {
			return { label: 'In browser', className: 'bg-sky-100 text-sky-900 border-sky-300' };
		}
		if (where === 'local') {
			return { label: 'On your machine', className: 'bg-amber-100 text-amber-900 border-amber-300' };
		}
		return { label: 'Your webapp', className: 'bg-green-100 text-green-900 border-green-300' };
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

{#snippet codeBlock(key: string, code: string, title: string)}
	<div>
		<div class="flex items-center justify-between gap-2 mb-1">
			<p class="text-xs font-medium uppercase tracking-wide text-neutral-500">{title}</p>
		</div>
		<div class="relative">
			<pre
				class="p-2 pr-10 bg-neutral-900 text-neutral-100 text-xs overflow-x-auto rounded font-mono whitespace-pre-wrap">{code}</pre>
			<button
				type="button"
				class="absolute top-1.5 right-1.5 p-1 rounded text-neutral-300 hover:text-white hover:bg-neutral-700 cursor-pointer"
				title={copiedKey === key ? 'Copied' : `Copy ${title}`}
				aria-label={copiedKey === key ? 'Copied' : `Copy ${title}`}
				onclick={() => void copySnippet(key, code)}
			>
				{#if copiedKey === key}
					<Check class="w-3.5 h-3.5 text-green-400" />
				{:else}
					<Clipboard class="w-3.5 h-3.5" />
				{/if}
			</button>
		</div>
	</div>
{/snippet}

{#snippet flowStep(step: (typeof END_TO_END_FLOW)[number], indexLabel: string)}
	{@const badge = whereBadge(step.where)}
	<div
		class="rounded border border-panel-border-base bg-white px-2.5 py-2 shadow-sm min-w-0 flex-1"
	>
		<div class="flex items-center gap-1.5 flex-wrap mb-1">
			<span
				class="text-[10px] font-mono font-semibold text-purple-800 bg-purple-100 border border-purple-200 rounded px-1"
				>{indexLabel}</span
			>
			<span class={`text-[10px] uppercase tracking-wide px-1 py-0.5 border rounded ${badge.className}`}
				>{badge.label}</span
			>
		</div>
		<p class="text-sm font-medium text-neutral-900 leading-snug">{step.title}</p>
		<p class="text-xs text-neutral-600 mt-1 leading-relaxed">{step.body}</p>
	</div>
{/snippet}

{#snippet flowArrow()}
	<div
		class="flex items-center justify-center text-neutral-400 text-lg leading-none select-none shrink-0 py-0.5 @md:py-0 @md:px-0.5"
		aria-hidden="true"
	>
		<span class="@md:hidden">↓</span>
		<span class="hidden @md:inline">→</span>
	</div>
{/snippet}

<div class="bg-neutral-100 h-full overflow-auto overscroll-contain flex flex-col flex-1 min-h-0">
	<div
		class="max-w-2xl mx-auto px-3 pb-3 @md:px-4 @md:pb-4 bg-white flex-1 flex flex-col w-full"
	>
		<article class="prose prose-sm max-w-none text-base pb-8">
			<h1 class="text-xl font-semibold mt-4 mb-2">Train → export → deploy</h1>
			<p class="text-neutral-700 leading-relaxed mb-4">
				End-to-end path from Browser Train to a small inference webapp — either
				<strong>onnxruntime-web</strong> or <strong>Transformers.js</strong> (decoder). No repo
				clone; local convert uses the toolkit zip below.
			</p>

			<div
				class="not-prose mb-4 p-3 border border-purple-300 border-dashed bg-purple-50 rounded flex flex-col @sm:flex-row @sm:items-center gap-3"
			>
				<div class="flex-1 min-w-0">
					<p class="font-medium text-sm text-neutral-900">Conversion toolkit</p>
					<p class="text-xs text-neutral-600 mt-0.5 leading-relaxed">
						Small zip: converter + <code class="text-[11px]">setup</code> /
						<code class="text-[11px]">convert</code>. Creates a local
						<code class="text-[11px]">.venv</code> on your machine (not shipped in the zip).
					</p>
				</div>
				<a
					href={ONNX_TOOLKIT_ZIP_URL}
					download="browser-train-onnx-toolkit.zip"
					class="inline-flex items-center justify-center gap-1.5 shrink-0 px-3 py-2 rounded bg-purple-700 text-white text-sm font-medium hover:bg-purple-800 no-underline"
				>
					<Download class="w-4 h-4" />
					Download toolkit
				</a>
			</div>

			<nav class="mb-4 not-prose">
				<p class="text-xs font-medium uppercase tracking-wide text-neutral-500 mb-1">Jump to</p>
				<ul class="flex flex-wrap gap-x-3 gap-y-1 text-sm">
					<li>
						<a href="#flow" class="text-purple-700 underline underline-offset-2">Visual guide</a>
					</li>
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

			<!-- Visual end-to-end guide -->
			<section id="flow" class="not-prose border border-panel-border-base rounded mb-4 overflow-hidden">
				<button
					type="button"
					class="w-full flex items-center gap-2 px-3 py-2 text-left font-medium bg-neutral-50 hover:bg-neutral-100 cursor-pointer"
					onclick={() => (flowOpen = !flowOpen)}
				>
					{#if flowOpen}
						<ChevronDown class="w-4 h-4 shrink-0" />
					{:else}
						<ChevronRight class="w-4 h-4 shrink-0" />
					{/if}
					Visual guide — entire flow
				</button>
				{#if flowOpen}
					<div class="px-3 pb-4 pt-2 space-y-4">
						<div>
							<p class="text-[11px] font-semibold uppercase tracking-wide text-sky-800 mb-2">
								In the browser
							</p>
							<div class="flex flex-col @md:flex-row @md:items-stretch gap-1">
								{#each flowBrowser as step, i (step.id)}
									{@render flowStep(step, String(i + 1))}
									{#if i < flowBrowser.length - 1}
										{@render flowArrow()}
									{/if}
								{/each}
							</div>
						</div>

						<div class="flex justify-center text-neutral-400 text-lg" aria-hidden="true">↓</div>

						<div>
							<p class="text-[11px] font-semibold uppercase tracking-wide text-amber-900 mb-2">
								On your machine
							</p>
							<div class="flex flex-col @md:flex-row @md:items-stretch gap-1">
								{#each flowLocal as step, i (step.id)}
									{@render flowStep(step, String(flowBrowser.length + i + 1))}
									{#if i < flowLocal.length - 1}
										{@render flowArrow()}
									{/if}
								{/each}
							</div>
						</div>

						<div class="flex justify-center text-neutral-400 text-lg" aria-hidden="true">↓</div>

						<div>
							<p class="text-[11px] font-semibold uppercase tracking-wide text-green-900 mb-2">
								Pick a deploy path
							</p>
							<div class="grid grid-cols-1 @md:grid-cols-2 gap-2">
								{#each flowDeploy as step, i (step.id)}
									{@render flowStep(step, String.fromCharCode(65 + i))}
								{/each}
							</div>
							<p class="text-xs text-neutral-500 mt-2 leading-relaxed">
								<strong>A · ORT</strong> — <code>complete</code> / <code>encodeDecode</code> /
								<code>encodeMasked</code>.
								<strong>B · Transformers.js</strong> — CausalLM (decoder), Seq2SeqLM (EncDec /
								BART), MaskedLM (Dyck / BERT). Prefer <code>*-onnx</code> presets.
							</p>
						</div>

						{@render codeBlock('convert', CONVERT_COMMAND, 'Local convert commands')}
					</div>
				{/if}
			</section>

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
					Shared pipeline (detail)
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
				{/if}
			</section>

			<h2 class="text-lg font-semibold mt-6 mb-2">Per-preset examples</h2>
			<p class="text-sm text-neutral-600 mb-3">
				Each preset includes copyable snippets for <strong>onnxruntime-web</strong> and, where
				supported, <strong>Transformers.js</strong>. Stock toys often need
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
								{@render codeBlock(
									`ort-${doc.presetId}`,
									doc.ortSnippet,
									'onnxruntime-web (out/ort/ → public/model/)'
								)}
								{#if doc.transformersJsSnippet}
									{@render codeBlock(
										`tjs-${doc.presetId}`,
										doc.transformersJsSnippet,
										'Transformers.js (out/transformers-js/ → public/models/browser-train/)'
									)}
								{/if}
							</div>
						{/if}
					</section>
				{/each}
			</div>
		</article>
	</div>
</div>
