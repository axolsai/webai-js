# WebAI worker contract v1.0

Every model worker is a self-contained adapter. The main library owns worker lifecycle,
request correlation, queueing, timeouts, cancellation, and the standard result envelope.
The worker owns model metadata, input/config validation, loading, inference, and disposal.

Start from `docs/worker-template.js`. Keep model-specific logic inside the adapter methods;
do not add ad-hoc message listeners or change response shapes.

## Required handshake

On boot, post `{ type: "worker initialized", data: { success: true, ... } }`.
For `checkModelSupports`, return the legacy runtime fields plus a `manifest` with
`contractVersion: "1.0"`. The manifest must document model identity, every accepted input,
every supported model/generation option, the native output, supported operations, and each
precision's devices, exact model-weight bytes, and estimated complete first-download bytes.
Identity metadata must include `displayName`, a plain-language `description`, `intendedUses`, and
honest `limitations`, in addition to `provider`, `providerUrl`, `license`, `licenseUrl`,
source/artifact repositories, and dates. `providerUrl` must identify the original provider, and
`licenseUrl` must link to the authoritative license text, including for non-standard licenses.

Consumers can inspect it immediately after `WebAI.create()`:

```js
const webai = await WebAI.create({ modelId: "my-model" });
console.log(webai.modelManifest);
```

## Request and response protocol

Every request is `{ requestId, type, data }`. Every correlated response must echo the same
`requestId`: `{ requestId, type, data }`. Progress uses `downloadProgress`, streaming uses
`streamChunk`, and failures use `error` with `{ message, context }`.

Required operations are `checkModelSupports`, `init`, `download`, `generate`, and
`clearMemory`. Advertise `generateStream` only when it is implemented.

If `generateStream` is not advertised, the public API must reject it before enqueueing worker work;
the worker may also return an explicit unsupported-operation error as a defensive fallback.

Do not implement `interrupt` as a worker message. A busy inference worker may not service its event
queue promptly. The framework interrupts by terminating the worker, rejecting its active request,
creating a fresh worker, and reinitializing the previous model configuration from browser cache.

`generate` success data is always `{ status: "success", result: nativeModelOutput }`.
The core library converts that into:

```js
{
  result: nativeModelOutput,
  runtime: { durationMs: 123.4 }
}
```

## Implementation rules

- Validate inputs and configurations before model inference.
- Do not collect telemetry, identifiers, authentication data, or analytics.
- Pin runtime dependencies to exact versions.
- Declare `manifest.runtime.engine` with the runtime name and exact version. Keep the legacy
  `transformersJsVersion` capability for compatibility, using `null` for non-Transformers workers.
- Runtime-specific loading, preprocessing, tensor conversion, execution, and disposal belong only
  inside the worker adapter; the public protocol and result envelope must remain unchanged.
- LiteRT.js adapters must serve the matching pinned WASM bundle, inspect real tensor signatures,
  dispose all input/output tensors, and report WebGPU fallback or full-delegation evidence.
- Report downloads through correlated progress messages.
- Dispose replaced pipelines and release model memory in `clearMemory`.
- Never swallow model-loading or disposal errors.
- Keep one message handler and one error boundary.
- Keep defaults and schemas together in the manifest to prevent documentation drift.

## Precision benchmark

The repository includes one standard 29.49-second MP3 fixture and a browser benchmark agent.
With the test app running on port 3001, execute:

```sh
npm run benchmark:precisions --prefix test-app
```

The agent discovers the worker manifest, tests every Hugging Face ONNX dtype mapping on
WASM and WebGPU, disposes each pipeline, clears that model's browser cache after every
case, and writes an extended manifest to
`benchmarks/whisper-tiny_timestamped.json`. Limit a diagnostic run with, for example,
`-- --precisions=fp32,q4 --devices=wasm`.

The report keeps raw timing under a standardized `runtime` object, a normalized performance grade, native output, and a correctness
assessment against the checked-in reference transcript. A combination is not supported when it
runs but fails the quality/gibberish gate.
