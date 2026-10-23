# WebAI model integration instructions

When adding or changing a model, follow `docs/ADDING_MODELS.md` and
`docs/WORKER_CONTRACT.md`. Start from `docs/worker-template.js`; use
`model-workers-v4/whisper-tiny.worker.js` as the complete Transformers.js reference implementation.
For LiteRT-LM language models, use `model-workers-v4/gemma-4-e2b-it-litert-lm.worker.js` as the
reference and preserve the standard chat messages/result shapes plus correlated token streaming.
For LiteRT.js models, preserve the identical worker protocol and isolate all runtime differences
inside the worker adapter. Use a pinned `@litertjs/core` release, serve its matching WASM assets,
record `runtime.engine` metadata, inspect actual `.tflite` tensor signatures, and manually dispose
every LiteRT model and tensor. Never refactor the managers, queue, or public result envelope merely
to accommodate a different inference runtime.
Before implementation, prove that the pinned runtime initializes inside a module Web Worker.
LiteRT.js releases whose WASM loader calls `importScripts()` are incompatible with the current
module-worker host; stop and report that boundary instead of changing infrastructure or publishing
a worker that only runs on the main thread.
When the user authorizes a bridge, keep it adapter-local: the module worker may create a nested
classic worker that owns LiteRT, while the outer worker retains the standard WebAI contract.
Resolve Emscripten's relative WASM fetch/XHR paths against the pinned runtime directory and verify
that terminating the outer worker also terminates the nested runtime.

Do not advertise precision/device support from filenames alone. Discover ONNX or `.tflite` artifacts from the
model repository and run every candidate combination in an isolated browser with a real local
fixture. Support requires successful initialization and inference. Preserve failures in the
benchmark report. Inference success alone is insufficient: compare output to a checked-in expected
result, calculate a task-appropriate quality score, and reject repetitive or gibberish output.
Record a performance grade as well as raw load/inference durations.
Store these measurements under each benchmark result's `runtime` object. Upload the standard fixture
to the project's public B2 assets prefix and record its stable URL, byte size, and checksum while
retaining a local copy for reproducible/offline testing.

Reuse an embedded benchmark result when its artifact revision, runtime version, fixture checksum,
quality gate, and relevant browser/backend behavior are unchanged. Rerun only affected cases when
one of the manifest's `benchmark.invalidatedBy` conditions changes.

Every manifest must identify the original provider, SPDX license, source repository, and source
update timestamp separately from the browser/ONNX artifact repository and its update timestamp.
It must also include a clear display name, plain-language description, intended uses, and concrete
limitations covering accuracy, safety, input quality, and task-specific failure modes.
Every precision must include exact weight bytes and estimated complete first-download bytes. For
LiteRT.js, distinguish model quantization from runtime precision, test WebGPU delegation rather than
assuming it, and record whether execution was fully accelerated or fell back to WASM.
Every benchmark case must dispose the model and clear only that model's cache.

Before completion, run worker syntax checks, library typecheck/build, test-app lint/build, the
relevant benchmark matrix, JSON validation, `git diff --check`, and `npm pack --dry-run`.

Model onboarding is not complete until the agent performs and reports a final v4 contract audit
without waiting for the user to request one. Compare the finished worker and manifest against every
requirement in `docs/WORKER_CONTRACT.md`, verify that only benchmark-passing precision/device pairs
are advertised, and run the recommended configuration through the public `WebAI` API. The final
response must explicitly state whether the worker conforms, identify the verified recommended
configuration, summarize any tested-but-unsupported combinations, and list every release-gate
command that passed. If any required audit item or release check fails, do not call the onboarding
complete; fix it or clearly report the remaining blocker.
