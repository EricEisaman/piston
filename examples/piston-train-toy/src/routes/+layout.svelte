<script lang="ts">
	import { pwaInfo } from 'virtual:pwa-info';
	import '../app.css';

	let { children } = $props();

	const webManifestLink = $derived(pwaInfo ? pwaInfo.webManifest.linkTag : '');
</script>

<svelte:head>
	<title>Browser Train</title>
	<meta name="description" content="Train language models in your browser with WebGPU" />
	<meta name="author" content="Eric Eisaman" />
	<meta name="keywords" content="machine learning, transformer, WebGPU, training, neural network" />
	<meta name="theme-color" content="oklch(0.438 0.218 303.724)" />
	<meta name="color-scheme" content="light" />
	<meta charset="utf-8" />
	<meta name="robots" content="index, follow" />
	<meta name="googlebot" content="index, follow" />
	<link rel="icon" href="/favicon.ico" sizes="32x32" />
	<link rel="icon" href="/favicon-32.png" type="image/png" sizes="32x32" />
	<link rel="apple-touch-icon" href="/apple-touch-icon.png" />
	<meta name="application-name" content="Browser Train" />
	<meta name="subject" content="Machine Learning Training Interface" />
	<meta name="rating" content="general" />
	<link rel="canonical" href="https://browser-train.netlify.app" />
	<link
		rel="preload"
		href="/Berkeley Mono Variable.woff2"
		as="font"
		type="font/woff2"
		crossorigin="anonymous"
	/>
	{#if webManifestLink}
		{@html webManifestLink}
	{/if}

	<!-- Open Graph -->
	<meta property="og:title" content="Browser Train" />
	<meta property="og:description" content="Train language models in your browser with WebGPU" />
	<meta property="og:type" content="website" />
	<meta property="og:url" content="https://browser-train.netlify.app" />
	<meta property="og:image" content="https://browser-train.netlify.app/og-image.png" />
	<meta property="og:image:alt" content="Browser Train decoder self-attention visualization" />

	<!-- Twitter -->
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content="Browser Train" />
	<meta name="twitter:description" content="Train language models in your browser with WebGPU" />
	<meta name="twitter:image" content="https://browser-train.netlify.app/og-image.png" />
	<meta name="twitter:creator" content="@SirFizX207" />
	<meta name="twitter:site" content="@SirFizX207" />

	<!-- Structured data -->
	<script type="application/ld+json">
		{
			"@context": "https://schema.org",
			"@type": "WebApplication",
			"name": "Browser Train",
			"url": "https://browser-train.netlify.app",
			"description": "Train language models in your browser with WebGPU",
			"inLanguage": "en",
			"applicationCategory": "EducationalApplication",
			"operatingSystem": "Any"
		}
	</script>
</svelte:head>

{@render children()}
