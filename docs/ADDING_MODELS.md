# Adding a model to WebAI

This is the required workflow for an agent adding a model. Whisper Tiny is the reference, not a
universal implementation: preserve its contract and test discipline while adapting inputs,
configuration, preprocessing, output, and fixtures to the new task.

The `test-app/` harness used below is local and Git-ignored; it is not part of the published package.

## 1. Establish provenance

Inspect the original model and browser-ready artifact repositories. Record the stable WebAI ID,
original creator/provider and `providerUrl`, original repository, SPDX license and authoritative
`licenseUrl`, original `lastModified`, artifact
repository, artifact `lastModified`, pipeline task, attribution requirements, and Transformers.js
compatibility. For a non-standard license, use an SPDX `LicenseRef-*` identifier and retain its
official license URL. Never infer licensing from the converter. Stop if provenance or licensing is absent.

Write model metadata for a developer who has not read the upstream model card:

- `displayName`: concise human-readable name and relevant variant;
- `description`: one plain-language paragraph covering what it does, where it runs, and its key output;
- `intendedUses`: concrete appropriate use cases, not marketing claims;
- `limitations`: accuracy tradeoffs, poor-input conditions, known output caveats, and safety boundaries.

Descriptions must be specific and factual. Do not claim production suitability, guaranteed accuracy,
privacy beyond the actual execution path, or benchmark superiority without evidence.

## 2. Select and pin the browser runtime

Choose the runtime supported by the browser artifact rather than assuming Transformers.js.
Transformers.js workers pin `@huggingface/transformers`; LiteRT.js workers pin `@litertjs/core`
and its matching WASM directory. Record the engine name/version in the manifest. Keep runtime code
inside the worker adapter and do not change WebAI managers, queues, messages, or result envelopes.
Run a minimal module-Web-Worker initialization probe before model implementation. If the runtime
depends on classic-worker-only APIs such as `importScripts()`, record the failure and stop unless
the user explicitly authorizes an infrastructure change or a separately tested worker bridge.
An approved bridge must stay inside the model adapter: use the normal module worker as the public
endpoint and a nested classic worker as the runtime host. Correlate bridge requests independently,
forward progress/errors, resolve relative WASM resources to the pinned directory, transfer large
tensor buffers, and confirm model/tensor cleanup on both normal disposal and outer-worker termination.

For LiteRT models, inspect the actual input/output tensor signatures and quantization parameters.
Implement preprocessing and postprocessing explicitly, delete every tensor, and use the LiteRT.js
model tester or an equivalent isolated-browser harness to measure WASM and WebGPU. A WebGPU run
must record whether the graph was fully delegated or fell back to WASM.

LiteRT language models use the separately versioned `@litert-lm/core` runtime and `.litertlm`
artifacts. Start from `model-workers-v4/gemma-4-e2b-it-litert-lm.worker.js` for these models. Preserve
the standard LLM input (`userInput.messages`) and result (`result`, `messages`, and
`structured_output`) shapes, and implement both `generate` and `generateStream`. Forward each native
streaming text delta as a correlated `streamChunk`, then finish with the same correlated `generated`
result used by non-streaming generation. Pin the runtime and web-specific artifact revision, verify
the runtime in a module worker, document JSPI/WebGPU/browser restrictions, and explicitly delete
each conversation and engine. Do not advertise WASM merely because LiteRT-LM supports CPU on native
platforms; browser support must be demonstrated using the selected web artifact.

## 3. Inventory actual artifacts

Use the repository file API to inventory every ONNX variant and exact byte size. Derive candidate
dtype mappings only from files that exist. Total shared configuration, tokenizer/vocabulary,
preprocessor, and generation assets required for a cold load.

For each precision, `weightsSize` is the exact selected model-weight sum. `size` is
`weightsSize + shared download bytes`; use it for storage checks and aggregate download progress.
Keep `modelKeys` exact for cache detection.

## 4. Implement the worker

Copy `docs/worker-template.js` and follow `docs/WORKER_CONTRACT.md`. Keep one correlated message
handler, echo `requestId`, pin dependencies, validate before inference, report progress, dispose
replaced pipelines, and never swallow load/disposal errors.

Use `model-workers-v4/whisper-tiny.worker.js` as the reference for environment configuration,
precision validation, downloads, typed inputs, preprocessing, output normalization, co-located
manifest/default definitions, lifecycle, and errors. Adapt task-specific behavior: chat, embeddings,
vision, audio, and other tasks require different input and output schemas.

## 5. Add a standard fixture

Add a small, redistributable, representative local fixture under `test-app/public/test-*` and
document its source, license, dimensions/duration, and preprocessing. Do not judge semantic quality
with synthetic data. Whisper uses `test-app/public/test-audio/whisper-standard.mp3`.
Upload the same immutable fixture to the project's public B2 `assets/benchmarks/` prefix and include
the public URL, exact byte size, and checksum in the worker and generated benchmark manifests. Keep
the local copy so tests do not depend on external availability.

## 6. Benchmark the complete matrix

Seed candidates from repository artifacts, not the worker allow-list. Test every precision on WASM
and WebGPU, including expected failures. Each case must use a fresh browser process and must:

1. start with a cold model cache;
2. load the model and record load duration;
3. run real inference and record duration plus native output;
4. compare output to a checked-in expected result using a task-appropriate quality metric;
5. detect degenerate output such as excessive repetition, empty results, NaN vectors, or invalid labels;
6. record the complete error on failure;
7. dispose the pipeline in `finally`;
8. delete only that model repository's Cache API entries in `finally`;
9. record cleanup success and deleted-entry count.

Keep unverified candidates in the benchmark page's matrix, not in the distributable worker's
`runtime.precisions`. Publish the allow-list only after the complete matrix and quality gate finish.

Advertise a combination only when loading and inference pass. Note slow or quality-degraded passes;
runtime success alone does not prove acceptable quality. Grade performance using a normalized task
metric, not raw milliseconds alone. Whisper records real-time factor (inference time divided by audio
duration) as excellent, good, slow, or very slow, and uses word error rate plus repetition checks for
quality. Other tasks must define an equivalent meaningful metric and correctness gate. For Whisper:

Treat implausibly fast inference as a reason to inspect output, not as proof of an optimization.
Whisper Base exposed FP16/Q4F16 WebGPU cases that ran quickly but returned unusable transcripts.
Do not assume smaller files are faster: Whisper Small's INT8/UINT8/Q8 and BNB4 WebGPU variants
were valid but substantially slower than Q4 and FP32 on the measured browser and hardware.

```sh
npm run benchmark:precisions --prefix test-app
```

The report is `benchmarks/whisper-tiny_timestamped.json`. Build an equivalent task-specific harness
for other model types instead of weakening the Whisper benchmark.

## 7. Publish evidence

Update `manifest.runtime.precisions` from measured results. Include report path, fixture metadata,
test timestamp/environment, pass/fail, timings, native output, errors, and cleanup evidence. Keep
failed combinations in the report even though the worker does not advertise them. Measurements are
specific to the tested browser and hardware, not universal guarantees.

Put per-case timing under `runtime`: model `loadMs`, `inferenceMs`, total case `totalMs`, normalized
real-time factor or equivalent task metric, and its grade. Do not label download time as inference
runtime or discard slow measurements.

Embed a compact summary of every tested combination directly in `manifest.benchmark.results`, while
keeping full native outputs/errors in the packaged report. Pin the exact artifact commit used for
the benchmark. Existing results may be reused only while artifact revision, Transformers.js/ONNX
Runtime, fixture checksum, quality gate, and relevant browser/backend behavior remain unchanged;
otherwise rerun the affected matrix cases and update `testedAt`.

## 8. Release gate

```sh
node --input-type=module --check < model-workers-v4/MODEL.worker.js
npm run typecheck
npm run build
npm run lint --prefix test-app
npm run build --prefix test-app
git diff --check
npm pack --dry-run --json
```

Finally, run one public-API browser smoke test with the recommended configuration and verify input,
native output, runtime metadata, cache behavior, and manifest discovery.

## 9. Mandatory final contract audit

The onboarding agent must perform this audit proactively for every added or changed model. The user
must never need to ask whether the worker conforms after implementation.

Before reporting completion:

1. Compare the finished worker and manifest field-by-field with `docs/WORKER_CONTRACT.md`.
2. Confirm the boot handshake and every correlated operation echo `requestId` with the standard
   response envelope, and that streaming is advertised only when implemented.
3. Confirm provenance, authoritative provider/license URLs, source and artifact revisions/dates,
   typed inputs/config/output, lifecycle validation, progress reporting, and disposal behavior.
4. Cross-check `runtime.precisions` against the full benchmark report and ensure no failed or
   unverified precision/device pair is advertised.
5. Verify the local and public fixture byte size and checksum, embedded benchmark summary, complete
   failure evidence, normalized performance grade, and task-specific quality gate.
6. Run the public `WebAI` API smoke test with the recommended configuration and all release-gate
   commands from the preceding section.
7. In the final response, explicitly say `v4 contract: conformant` only if every item passed. Also
   state the recommended configuration, tested unsupported combinations, and release checks. If an
   item fails, state `v4 contract: not yet conformant` and keep working or identify the blocker.
