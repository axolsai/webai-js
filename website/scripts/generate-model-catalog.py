#!/usr/bin/env python3
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKERS = ROOT / "model-workers-v4"
BENCHMARKS = ROOT / "benchmarks"
OUTPUT = Path(__file__).resolve().parents[1] / "app" / "models" / "models-data.json"
MANIFEST_EXTRACTOR = Path(__file__).resolve().parent / "extract-worker-manifest.mjs"

ALIASES = {
    "distilbert-base-uncased-finetuned-sst-2-english": "distilbert-sst2",
    "whisper-base": "whisper-base_timestamped",
    "whisper-large-v3-turbo": "whisper-large-v3-turbo_timestamped",
    "whisper-small": "whisper-small_timestamped",
    "whisper-tiny": "whisper-tiny_timestamped",
}

def compact_benchmark(benchmark):
    results = benchmark.get("results", []) if isinstance(benchmark, dict) else []
    passed = [r for r in results if r.get("status") == "pass"]
    failed = [r for r in results if r.get("status") == "fail"]
    return {
        "testedAt": benchmark.get("testedAt"),
        "environment": benchmark.get("environment"),
        "qualityGate": benchmark.get("qualityGate"),
        "report": benchmark.get("report"),
        "passed": len(passed),
        "failed": len(failed),
        "results": [
            {
                "precision": r.get("precision"),
                "device": r.get("device"),
                "status": r.get("status"),
                "grade": (r.get("runtime") or {}).get("grade"),
                "loadMs": (r.get("runtime") or {}).get("loadMs"),
                "inferenceMs": (r.get("runtime") or {}).get("inferenceMs"),
                "totalMs": (r.get("runtime") or {}).get("totalMs"),
                "tokensPerSecond": (r.get("runtime") or {}).get("tokensPerSecond"),
                "realTimeFactor": (r.get("runtime") or {}).get("realTimeFactor"),
                "timeToFirstTokenMs": (r.get("runtime") or {}).get("timeToFirstTokenMs"),
                "prefillTokensPerSecond": (r.get("runtime") or {}).get("prefillTokensPerSecond"),
                "decodeTokensPerSecond": (r.get("runtime") or {}).get("decodeTokensPerSecond"),
                "fullyAccelerated": r.get("fullyAccelerated"),
            }
            for r in results
        ],
    }

def enrich_model_from_worker(model, worker):
    source = worker.read_text()
    for key in ("displayName", "provider", "providerUrl", "license", "licenseUrl", "task", "description", "repository", "sourceRepository"):
        if model.get(key):
            continue
        match = re.search(rf"\b{key}\s*:\s*([\"'])(.*?)\1", source, re.S)
        if match:
            model[key] = match.group(2)
    model.setdefault("displayName", model.get("id", worker.stem).replace("-", " ").title())
    model.setdefault("task", "other")
    model.setdefault("provider", "Model provider")
    model.setdefault("description", f"WebAI v4 worker for {model['displayName']}.")
    return model

models = []
for worker in sorted(WORKERS.glob("*.worker.js")):
    worker_id = worker.name.removesuffix(".worker.js")
    if worker_id == "mediapipe-selfie-segmenter":
        continue
    report = BENCHMARKS / f"{ALIASES.get(worker_id, worker_id)}.json"
    if not report.exists():
        raise SystemExit(f"Missing manifest report for {worker_id}: {report}")
    data = json.loads(report.read_text())
    manifest = json.loads(subprocess.check_output(["node", str(MANIFEST_EXTRACTOR), str(worker)], text=True))

    # The executable worker manifest is the public contract. Benchmark reports add
    # measurements, but must not erase current capabilities with stale/empty data.
    model = dict(manifest.get("model", {}))
    for key, value in data.get("model", {}).items():
        model.setdefault(key, value)
    model = enrich_model_from_worker(model, worker)
    manifest_runtime = manifest.get("runtime", {})
    report_runtime = data.get("runtime", {})
    precisions = manifest_runtime.get("precisions") or report_runtime.get("precisions", {})
    engine = manifest_runtime.get("engine") or report_runtime.get("engine")
    sizes = [p.get("size") for p in precisions.values() if isinstance(p, dict) and isinstance(p.get("size"), (int, float))]
    models.append({
        "workerFile": worker.name,
        "contractVersion": manifest.get("contractVersion") or data.get("contractVersion"),
        "model": model,
        "runtime": {"engine": engine, "precisions": precisions},
        "input": manifest.get("input") or data.get("input"),
        "config": manifest.get("config") or data.get("config"),
        "output": manifest.get("output") or data.get("output"),
        "operations": manifest.get("operations") or data.get("operations", []),
        "benchmark": compact_benchmark(data.get("benchmark") or manifest.get("benchmark", {})),
        "smallestDownloadBytes": min(sizes) if sizes else None,
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(models, indent=2, ensure_ascii=False) + "\n")
print(f"Wrote {len(models)} models to {OUTPUT}")
