import type { WebAIWorkerSample, WebAIWorkerSampleAsset } from './types';

/** Download and verify the original encoded sample bytes, suitable for File inputs. */
export async function fetchWorkerSampleAsset(asset: WebAIWorkerSampleAsset, options: { signal?: AbortSignal; url?: string } = {}): Promise<ArrayBuffer> {
  const response = await fetch(options.url ?? asset.url, { signal: options.signal });
  if (!response.ok) throw new Error(`Sample download failed: ${asset.field} (${response.status})`);
  const bytes = await response.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-1', bytes);
  const sha1 = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
  if (bytes.byteLength !== asset.size || sha1 !== asset.sha1) throw new Error(`Sample integrity check failed: ${asset.field}`);
  options.signal?.throwIfAborted();
  return bytes;
}

/** Fetch a reusable sample without loading inference weights. Call dispose after generation. */
export async function prepareWorkerSample(
  sample: WebAIWorkerSample,
  options: { signal?: AbortSignal; resolveAssetUrl?: (asset: WebAIWorkerSampleAsset) => string } = {},
): Promise<{ request: { userInput: Record<string, unknown>; modelConfig: Record<string, unknown>; generateConfig: Record<string, unknown> }; dispose: () => void }> {
  const userInput = structuredClone(sample.userInput);
  const urls: string[] = [];
  const dispose = () => { for (const url of urls.splice(0)) URL.revokeObjectURL(url); };
  try {
    for (const asset of sample.assets) {
      const bytes = await fetchWorkerSampleAsset(asset, { signal: options.signal, url: options.resolveAssetUrl?.(asset) });
      options.signal?.throwIfAborted();
      if (asset.encoding === 'blob-url') {
        const url = URL.createObjectURL(new Blob([bytes], { type: asset.mediaType }));
        urls.push(url);
        userInput[asset.field] = url;
      } else if (asset.encoding === 'audio-pcm-f32') {
        if (!asset.sampleRate || !Number.isInteger(asset.sampleRate) || asset.sampleRate <= 0) throw new Error('Sample audio requires a valid sampleRate.');
        const context = new AudioContext();
        let decoded: AudioBuffer;
        try { decoded = await context.decodeAudioData(bytes); } finally { await context.close(); }
        const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * asset.sampleRate), asset.sampleRate);
        const source = offline.createBufferSource();
        source.buffer = decoded;
        source.connect(offline.destination);
        source.start();
        const rendered = await offline.startRendering();
        userInput[asset.field] = rendered.getChannelData(0).slice();
      } else throw new Error(`Unsupported sample encoding: ${asset.encoding}`);
      options.signal?.throwIfAborted();
    }
    return { request: { userInput, modelConfig: structuredClone(sample.modelConfig), generateConfig: structuredClone(sample.generateConfig) }, dispose };
  } catch (error) { dispose(); throw error; }
}
