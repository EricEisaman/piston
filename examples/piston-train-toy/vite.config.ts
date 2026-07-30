import { sveltekit } from '@sveltejs/kit/vite';
import { SvelteKitPWA } from '@vite-pwa/sveltekit';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';
import sirv from 'sirv';
import { fileURLToPath } from 'url';
import { defineConfig, loadEnv, type ViteDevServer } from 'vite';
import wasm from 'vite-plugin-wasm';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const resolveCommitHash = (): string => {
	const candidates = [
		process.env.VITE_COMMIT_HASH,
		process.env.COMMIT_REF,
		process.env.RENDER_GIT_COMMIT,
		process.env.CF_PAGES_COMMIT_SHA
	];
	for (const value of candidates) {
		if (value && value.trim().length > 0) {
			return value.trim().slice(0, 7);
		}
	}
	return 'unknown';
};

const commitHash = resolveCommitHash();

// Dev-only mount for tokenizer and tokenized directories via env paths
const devStaticMount = (opts: { tokenizerDir?: string; tokenizedDir?: string }) => ({
	name: 'dev-static-mount',
	apply: 'serve' as const,
	configureServer(server: ViteDevServer) {
		const tokenizerDir = opts.tokenizerDir;
		const tokenizedDir = opts.tokenizedDir;
		if (tokenizerDir) {
			server.middlewares.use('/tokenizer', sirv(tokenizerDir, { dev: true, etag: true }));
		}
		if (tokenizedDir) {
			server.middlewares.use('/tokenized', sirv(tokenizedDir, { dev: true, etag: true }));
		}
	}
});

export default defineConfig(({ mode }) => {
	const envDir = path.dirname(fileURLToPath(import.meta.url));
	const env = loadEnv(mode, envDir, '');

	return {
		define: {
			__COMMIT_HASH__: JSON.stringify(commitHash)
		},
		plugins: [
			tailwindcss(),
			...(mode === 'development'
				? [
						devStaticMount({
							tokenizerDir: env.VITE_TOKENIZER_DIR,
							tokenizedDir: env.VITE_TOKENIZED_DIR
						})
					]
				: []),
			sveltekit(),
			wasm(),
			SvelteKitPWA({
				registerType: 'autoUpdate',
				injectRegister: 'auto',
				manifest: {
					name: 'Browser Train',
					short_name: 'Browser Train',
					description: 'Train language models in your browser with WebGPU',
					theme_color: '#6b21a8',
					background_color: '#f5f5f5',
					display: 'standalone',
					start_url: '/',
					scope: '/',
					icons: [
						{
							src: '/pwa-192.png',
							sizes: '192x192',
							type: 'image/png'
						},
						{
							src: '/pwa-512.png',
							sizes: '512x512',
							type: 'image/png'
						},
						{
							src: '/pwa-512.png',
							sizes: '512x512',
							type: 'image/png',
							purpose: 'maskable'
						}
					]
				},
				// Let @vite-pwa/sveltekit pick client/ + prerendered/ globs.
				workbox: {
					navigateFallback: undefined,
					runtimeCaching: [
						{
							urlPattern: ({ url }) =>
								url.pathname.startsWith('/tokenizer/') ||
								url.pathname.startsWith('/tokenized/'),
							handler: 'CacheFirst',
							options: {
								cacheName: 'browser-train-data',
								expiration: {
									maxEntries: 64,
									maxAgeSeconds: 60 * 60 * 24 * 30
								}
							}
						}
					]
				},
				devOptions: {
					enabled: false
				}
			})
		],
		worker: {
			format: 'es',
			plugins: () => [wasm(), sveltekit()]
		},
		resolve: {
			dedupe: [
				'svelte',
				'svelte/legacy',
				'@codemirror/state',
				'@codemirror/view',
				'@codemirror/language',
				'@codemirror/lang-javascript',
				'@codemirror/lint',
				'codemirror',
				'@lezer/highlight'
			]
		},
		esbuild: {
			supported: { 'top-level-await': true },
			keepNames: true
		},
		server: {
			fs: {
				allow: [
					path.resolve(path.dirname(fileURLToPath(import.meta.url))),
					projectRoot,
					path.resolve(projectRoot, 'target', 'pkg', 'piston-web')
				]
			},
			headers: {
				'Cross-Origin-Embedder-Policy': 'require-corp',
				'Cross-Origin-Opener-Policy': 'same-origin'
			}
		}
	};
});
