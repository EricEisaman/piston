<script lang="ts">
	import MermaidDiagram from '$lib/components/MermaidDiagram.svelte';
	import {
		ARCHITECTURE_DIAGRAMS,
		type ArchitectureDiagram
	} from '$lib/workspace/architectureDiagrams';
	import { getPresetOptions } from '$lib/workspace/presets';
	import { ChevronDown, ChevronRight } from '@lucide/svelte/icons';

	const presetLabels = $derived(
		Object.fromEntries(getPresetOptions().map((p) => [p.value, p.text])) as Record<string, string>
	);

	const ordered = $derived(
		getPresetOptions()
			.map((p) => ARCHITECTURE_DIAGRAMS.find((d) => d.presetId === p.value))
			.filter((d): d is ArchitectureDiagram => d != null)
	);

	let openCards = $state<Record<string, boolean>>({});
	let defaultsApplied = $state(false);

	$effect(() => {
		if (!defaultsApplied && ordered.length > 0) {
			openCards = { [ordered[0].presetId]: true };
			defaultsApplied = true;
		}
	});

	function isOpen(id: string): boolean {
		return openCards[id] ?? false;
	}

	function toggle(id: string) {
		openCards = { ...openCards, [id]: !isOpen(id) };
	}
</script>

<div class="h-full overflow-y-auto overscroll-contain p-3 space-y-3 text-sm">
	<header class="space-y-1">
		<h2 class="font-medium text-base">Architectures</h2>
		<p class="text-neutral-600 text-xs leading-relaxed">
			Colorized Mermaid views of each preset in the selector — encoder–decoder toys, encoder-only
			Dyck, and decoder-only TinyStories / FineWeb (plus the ONNX-friendly modifier).
		</p>
		<div class="flex flex-wrap gap-1.5 pt-1 text-[10px] font-mono uppercase tracking-wide">
			<span class="px-1.5 py-0.5 rounded border bg-blue-50 text-blue-900 border-blue-300"
				>tokens</span
			>
			<span class="px-1.5 py-0.5 rounded border bg-pink-50 text-pink-900 border-pink-300"
				>embed</span
			>
			<span class="px-1.5 py-0.5 rounded border bg-violet-50 text-violet-900 border-violet-300"
				>attention</span
			>
			<span class="px-1.5 py-0.5 rounded border bg-sky-50 text-sky-900 border-sky-300">encoder</span>
			<span class="px-1.5 py-0.5 rounded border bg-amber-50 text-amber-900 border-amber-300"
				>decoder</span
			>
			<span class="px-1.5 py-0.5 rounded border bg-green-50 text-green-900 border-green-300">mlp</span
			>
			<span class="px-1.5 py-0.5 rounded border bg-orange-50 text-orange-900 border-orange-300"
				>output</span
			>
		</div>
	</header>

	{#each ordered as diagram (diagram.presetId)}
		<section class="border border-panel-border-base rounded overflow-hidden bg-white">
			<button
				type="button"
				class="w-full flex items-start gap-2 p-2 text-left cursor-pointer hover:bg-neutral-50"
				onclick={() => toggle(diagram.presetId)}
				aria-expanded={isOpen(diagram.presetId)}
			>
				{#if isOpen(diagram.presetId)}
					<ChevronDown class="w-4 h-4 shrink-0 mt-0.5 text-neutral-500" />
				{:else}
					<ChevronRight class="w-4 h-4 shrink-0 mt-0.5 text-neutral-500" />
				{/if}
				<div class="min-w-0">
					<p class="font-medium text-sm leading-snug">
						{presetLabels[diagram.presetId] ?? diagram.presetId}
					</p>
					<p class="text-xs text-neutral-600 mt-0.5">{diagram.summary}</p>
				</div>
			</button>
			{#if isOpen(diagram.presetId)}
				<div class="border-t border-panel-border-base p-2 bg-slate-50/80">
					<MermaidDiagram definition={diagram.mermaid} />
				</div>
			{/if}
		</section>
	{/each}
</div>
