<script lang="ts">
	type CitationEntry = { name: string; url?: string };

	export type CitationEntries = {
		entries?: readonly CitationEntry[];
		extra?: string;
	};

	let { citations }: { citations: CitationEntries } = $props();
</script>

<span class="text-2xs text-neutral-500">
	{#each citations?.entries ?? [] as e, i (e.name)}
		{#if i > 0},
		{/if}
		{e.name}
	{/each}{#if citations?.extra}{citations.extra}{/if}
</span>
