import Link from "next/link";
import { notFound } from "next/navigation";
import data from "../../models-data.json";
import PlaygroundClient, { type Model } from "./playground-client";
import ModelInfoSidebar from "../model-info-sidebar";

const models=data as unknown as Model[];
export default async function Page({params}:{params:Promise<{id:string}>}){const {id}=await params,model=models.find(x=>x.model.id===id);if(!model)notFound();return <main className="flex h-[calc(100vh-52px)] w-full overflow-hidden px-3"><ModelInfoSidebar model={model}/><div className="flex min-w-0 flex-1 flex-col p-4"><nav className="mb-4 flex gap-2"><Link href={`/models/${id}/api-reference/get-started/basic-usage`} className="rounded-full px-4 py-2 text-sm text-neutral-400 hover:bg-white/5 hover:text-white">API Reference</Link><span className="rounded-full bg-primary px-4 py-2 text-sm text-white">Playground</span></nav><PlaygroundClient model={model}/></div></main>}
