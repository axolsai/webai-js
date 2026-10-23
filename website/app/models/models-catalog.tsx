"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, Box, ChevronDown, Cpu, Database, Github, ImageIcon, MessageSquareText, Languages, Mic2, Search } from "lucide-react";

type Precision = { supportedDevices?: string[] };
type CatalogModel = {
  workerFile: string;
  model: { id: string; displayName: string; provider?: string; license?: string; task: string; description?: string; repository?: string; sourceRepository?: string };
  runtime: { engine?: { name?: string; version?: string } | null; precisions?: Record<string, Precision> };
  input?: { description?: string } | null;
  output?: { description?: string } | null;
  benchmark?: { passed?: number };
  smallestDownloadBytes?: number | null;
};

const taskNames: Record<string, string> = {
  "text-generation": "Text Generation", "feature-extraction": "Text Embeddings", "text-ranking": "Reranking",
  "text-classification": "Text Classification", "token-classification": "Token Classification",
  "automatic-speech-recognition": "Speech Recognition", "text-to-speech": "Text to Speech", "audio-classification": "Audio Classification",
  "image-classification": "Image Classification", "zero-shot-image-classification": "Zero Shot Vision", "object-detection": "Object Detection",
  "depth-estimation": "Depth Estimation", "image-segmentation": "Image Segmentation", "mask-generation": "Mask Generation",
  "background-removal": "Background Removal", "image-matting": "Image Matting", "image-text-to-text": "Vision to Text",
  "face-detection": "Face Detection", "hand-landmark-detection": "Hand Landmarks", "keypoint-detection": "Pose Estimation",
};

const groups = [
  { name: "LLM", icon: MessageSquareText, tasks: ["text-generation"] },
  { name: "NLP", icon: Languages, tasks: ["feature-extraction", "text-ranking", "text-classification", "token-classification"] },
  { name: "Audio", icon: Mic2, tasks: ["automatic-speech-recognition", "text-to-speech", "audio-classification"] },
  { name: "Image", icon: ImageIcon, tasks: ["image-classification", "zero-shot-image-classification", "object-detection", "depth-estimation", "image-segmentation", "mask-generation", "background-removal", "image-matting", "image-text-to-text", "face-detection", "hand-landmark-detection", "keypoint-detection"] },
];

const taskLabel = (task: string) => taskNames[task] ?? task.split("-").map((word) => word[0]?.toUpperCase() + word.slice(1)).join(" ");
const sectionId = (task: string) => `task-${task}`;
function formatBytes(bytes?: number | null) {
  if (!bytes) return "Size varies";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function ModelCard({ item }: { item: CatalogModel }) {
  const [open, setOpen] = useState(false);
  const precisions = Object.entries(item.runtime.precisions ?? {});
  const devices = [...new Set(precisions.flatMap(([, value]) => value.supportedDevices ?? []))];
  const source = item.model.sourceRepository ?? item.model.repository;
  return (
    <article className="group overflow-hidden rounded-xl border border-white/10 bg-white/[0.025] transition hover:border-primary/35 hover:bg-white/[0.04]">
      <Link href={`/models/${item.model.id}`} className="block p-4 sm:p-5">
        <div className="flex items-start gap-4">
          <div className="grid h-11 w-11 shrink-0 place-items-center rounded-lg border border-primary/20 bg-primary/10 text-primary"><Cpu className="h-5 w-5" /></div>
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0"><h3 className="truncate text-base font-semibold text-white transition group-hover:text-primary">{item.model.displayName}</h3><p className="mt-0.5 truncate font-mono text-[11px] text-neutral-400">{item.model.id}</p></div>
              <span className="shrink-0 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-1 text-[10px] font-medium text-emerald-300">v4 worker</span>
            </div>
            <p className="mt-3 line-clamp-2 min-h-10 text-sm leading-5 text-neutral-400">{item.model.description}</p>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-1.5">
          {precisions.map(([precision]) => <span key={precision} className="rounded-md border border-fuchsia-400/20 bg-fuchsia-400/10 px-2 py-1 font-mono text-[10px] text-fuchsia-200">{precision}</span>)}
          {devices.map((device) => <span key={device} className="rounded-md border border-cyan-400/20 bg-cyan-400/10 px-2 py-1 text-[10px] uppercase text-cyan-200">{device}</span>)}
        </div>
        <div className="mt-4 flex items-center gap-5 border-t border-white/10 pt-3 text-xs text-neutral-400">
          <span className="inline-flex items-center gap-1.5"><Database className="h-3.5 w-3.5" />{formatBytes(item.smallestDownloadBytes)}</span>
          <span>{item.benchmark?.passed ?? 0} tested configs</span><span className="ml-auto shrink-0 font-medium text-neutral-300">View model →</span>
        </div>
      </Link>
      <button type="button" onClick={() => setOpen(!open)} className="flex w-full items-center justify-between border-t border-white/10 px-4 py-2.5 text-xs font-medium text-neutral-400 hover:bg-white/[0.04] hover:text-white" aria-expanded={open}>Worker details<ChevronDown className={`h-3.5 w-3.5 transition ${open ? "rotate-180" : ""}`} /></button>
      {open && <div className="space-y-4 border-t border-white/10 bg-black/20 p-4 text-xs leading-5 text-neutral-400">
        <div className="grid gap-3 sm:grid-cols-2"><div><strong className="text-neutral-200">Runtime</strong><p>{item.runtime.engine?.name ?? "Worker runtime"}{item.runtime.engine?.version ? ` ${item.runtime.engine.version}` : ""}</p></div><div><strong className="text-neutral-200">License</strong><p>{item.model.license ?? "See manifest"}</p></div></div>
        {item.input?.description && <div><strong className="text-neutral-200">Input</strong><p>{item.input.description}</p></div>}
        {item.output?.description && <div><strong className="text-neutral-200">Output</strong><p>{item.output.description}</p></div>}
        <div className="flex flex-wrap gap-2"><a href={`https://github.com/axolsai/webai-js/blob/main/model-workers-v4/${item.workerFile}`} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 rounded-md border border-white/10 px-2.5 py-1.5 text-neutral-200 hover:border-primary/40"><Github className="h-3.5 w-3.5" />Worker</a>{source && <a href={`https://huggingface.co/${source}`} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 rounded-md border border-white/10 px-2.5 py-1.5 text-neutral-200 hover:border-primary/40">Model source<ArrowUpRight className="h-3.5 w-3.5" /></a>}</div>
      </div>}
    </article>
  );
}

export default function ModelsCatalog({ models: rawModels }: { models: unknown }) {
  const models = rawModels as CatalogModel[];
  const [query, setQuery] = useState("");
  const [activeTask, setActiveTask] = useState("all");
  const sidebarRef = useRef<HTMLElement>(null);
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return models.filter((item) => !q || `${item.model.displayName} ${item.model.id} ${item.model.provider} ${item.model.description} ${taskLabel(item.model.task)}`.toLowerCase().includes(q));
  }, [models, query]);
  const visibleGroups = groups.map((group) => ({ ...group, tasks: group.tasks.map((task) => ({ task, models: filtered.filter((model) => model.model.task === task) })).filter((entry) => entry.models.length) })).filter((group) => group.tasks.length);
  const countTask = (task: string) => models.filter((model) => model.model.task === task).length;
  const chooseTask = (task: string) => {
    setActiveTask(task);
    document.getElementById(task === "all" ? "catalog-top" : sectionId(task))?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  useEffect(() => {
    let frame = 0;
    const updateActiveTask = () => {
      frame = 0;
      const sections = groups.flatMap((group) => group.tasks)
        .map((task) => ({ task, element: document.getElementById(sectionId(task)) }))
        .filter((entry): entry is { task: string; element: HTMLElement } => Boolean(entry.element));
      if (window.scrollY < 120) {
        setActiveTask("all");
        return;
      }
      let current = sections[0]?.task ?? "all";
      for (const section of sections) {
        if (section.element.getBoundingClientRect().top <= 150) current = section.task;
        else break;
      }
      setActiveTask(current);
    };
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(updateActiveTask);
    };
    updateActiveTask();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [query]);

  useEffect(() => {
    if (activeTask === "all") return;
    const sidebar = sidebarRef.current;
    const item = sidebar?.querySelector<HTMLElement>(`[data-task="${activeTask}"]`);
    if (sidebar && item) sidebar.scrollTo({ top: item.offsetTop - sidebar.clientHeight / 2, behavior: "smooth" });
  }, [activeTask]);

  return <main id="catalog-top" className="min-h-[calc(100vh-52px)] bg-background text-foreground">
    <div className="border-b border-white/10 px-4 py-4 lg:hidden">
      <div className="relative"><Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search models..." className="h-11 w-full rounded-lg border border-white/10 bg-white/[0.04] pl-10 pr-3 text-sm outline-none focus:border-primary/50" /></div>
      <select value={activeTask} onChange={(e) => chooseTask(e.target.value)} className="mt-3 h-10 w-full rounded-lg border border-white/10 bg-neutral-950 px-3 text-sm text-neutral-200"><option value="all">All models ({models.length})</option>{groups.flatMap((group) => group.tasks).filter((task) => countTask(task)).map((task) => <option key={task} value={task}>{taskLabel(task)} ({countTask(task)})</option>)}</select>
    </div>
    <div className="mx-auto flex w-full max-w-[1600px]">
      <aside ref={sidebarRef} className="sticky top-[52px] hidden h-[calc(100vh-52px)] w-72 shrink-0 overflow-y-auto border-r border-white/10 px-5 py-7 lg:block">
        <div className="mb-7"><h1 className="text-2xl font-semibold tracking-tight text-white">Models</h1><p className="mt-1 text-sm text-neutral-400">{models.length} production ready v4 workers</p></div>
        <div className="relative mb-7"><Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search models..." className="h-10 w-full rounded-lg border border-white/10 bg-white/[0.04] pl-9 pr-3 text-sm outline-none placeholder:text-neutral-500 focus:border-primary/50" /></div>
        <p className="mb-2 px-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-neutral-500">Model categories</p>
        <button onClick={() => chooseTask("all")} className={`mb-3 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm transition ${activeTask === "all" ? "bg-primary/10 text-primary" : "text-neutral-400 hover:bg-white/[0.04] hover:text-white"}`}><Box className="h-4 w-4" /><span className="flex-1">All models</span><span className="text-xs text-neutral-500">{models.length}</span></button>
        <nav className="space-y-5">{groups.map((group) => { const Icon = group.icon; const count = models.filter((model) => group.tasks.includes(model.model.task)).length; if (!count) return null; return <div key={group.name}><div className="flex items-center gap-2 px-2.5 py-1.5 text-sm font-semibold text-neutral-200"><Icon className="h-4 w-4 text-primary" />{group.name}<span className="ml-auto text-xs font-normal text-neutral-500">{count}</span></div><div className="ml-4 mt-1 border-l border-white/10 pl-3">{group.tasks.filter((task) => countTask(task)).map((task) => <button key={task} data-task={task} onClick={() => chooseTask(task)} className={`flex w-full items-center rounded-md px-2 py-1.5 text-left text-xs transition ${activeTask === task ? "bg-primary/10 text-primary" : "text-neutral-400 hover:bg-white/[0.04] hover:text-neutral-200"}`}><span className="flex-1">{taskLabel(task)}</span><span>{countTask(task)}</span></button>)}</div></div>; })}</nav>
      </aside>
      <section className="min-w-0 flex-1 px-4 py-7 sm:px-6 lg:px-9 lg:py-8 xl:px-12">
        <div className="mb-8 flex flex-col gap-3 border-b border-white/10 pb-6 sm:flex-row sm:items-end sm:justify-between"><div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">WebAI.js model library</p><h2 className="mt-2 text-2xl font-semibold text-white sm:text-3xl">{query ? "Search results" : "All browser AI models"}</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-neutral-400">Verified workers from <span className="font-mono text-neutral-400">model-workers-v4</span>, organized by task.</p></div><span className="text-sm text-neutral-400">{filtered.length} {filtered.length === 1 ? "model" : "models"}</span></div>
        <div className="space-y-12">{visibleGroups.map((group) => { const Icon = group.icon; return <div key={group.name}><div className="mb-6 flex items-center gap-3"><Icon className="h-5 w-5 text-primary" /><h2 className="text-xl font-semibold text-white">{group.name}</h2><div className="h-px flex-1 bg-white/10" /></div><div className="space-y-9">{group.tasks.map(({ task, models: taskModels }) => <section key={task} id={sectionId(task)} className="scroll-mt-20"><div className="mb-4"><h3 className="font-medium text-neutral-200">{taskLabel(task)}</h3><p className="mt-0.5 text-xs text-neutral-500">{taskModels.length} {taskModels.length === 1 ? "worker" : "workers"}</p></div><div className="grid gap-4 xl:grid-cols-2">{taskModels.map((item) => <ModelCard key={item.workerFile} item={item} />)}</div></section>)}</div></div>; })}</div>
        {!filtered.length && <div className="rounded-xl border border-dashed border-white/15 py-20 text-center"><Search className="mx-auto h-6 w-6 text-neutral-500" /><p className="mt-3 text-sm text-neutral-400">No models match your search.</p></div>}
      </section>
    </div>
  </main>;
}
