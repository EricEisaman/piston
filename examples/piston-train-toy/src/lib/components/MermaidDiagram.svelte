<script lang="ts">
	import mermaid from 'mermaid';

	let { definition }: { definition: string } = $props();

	let svgHtml = $state('');
	let errorMessage = $state<string | null>(null);
	let renderGeneration = 0;

	let initialized = false;

	function ensureMermaid() {
		if (initialized) return;
		mermaid.initialize({
			startOnLoad: false,
			securityLevel: 'loose',
			theme: 'base',
			themeVariables: {
				background: '#ffffff',
				fontFamily: 'ui-sans-serif, system-ui, sans-serif',
				primaryColor: '#ede9fe',
				primaryTextColor: '#1e1b4b',
				primaryBorderColor: '#7c3aed',
				secondaryColor: '#dbeafe',
				secondaryTextColor: '#1e3a8a',
				secondaryBorderColor: '#2563eb',
				tertiaryColor: '#dcfce7',
				tertiaryTextColor: '#14532d',
				tertiaryBorderColor: '#16a34a',
				lineColor: '#64748b',
				textColor: '#0f172a',
				mainBkg: '#ede9fe',
				nodeBorder: '#7c3aed',
				clusterBkg: '#f8fafc',
				clusterBorder: '#cbd5e1',
				titleColor: '#334155'
			},
			flowchart: {
				curve: 'basis',
				padding: 12,
				htmlLabels: true,
				nodeSpacing: 36,
				rankSpacing: 40
			}
		});
		initialized = true;
	}

	$effect(() => {
		const source = definition;
		const generation = ++renderGeneration;
		errorMessage = null;
		ensureMermaid();
		void (async () => {
			try {
				const id = `arch-${crypto.randomUUID().replace(/-/g, '')}`;
				const { svg } = await mermaid.render(id, source);
				if (generation === renderGeneration) {
					svgHtml = svg;
				}
			} catch (err) {
				if (generation === renderGeneration) {
					svgHtml = '';
					errorMessage = err instanceof Error ? err.message : String(err);
				}
			}
		})();
	});
</script>

{#if errorMessage}
	<pre class="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2 whitespace-pre-wrap"
		>{errorMessage}</pre
	>
{:else if svgHtml}
	<div class="mermaid-diagram overflow-x-auto [&_svg]:max-w-full [&_svg]:h-auto">
		<!-- Mermaid emits trusted SVG from our static diagram strings -->
		{@html svgHtml}
	</div>
{:else}
	<p class="text-xs text-neutral-500">Rendering diagram…</p>
{/if}
