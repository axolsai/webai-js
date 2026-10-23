// hero-code-section.tsx - Right code/demo section
"use client";
import { Safari } from "@/components/magicui/safari";
import SimpleCodeblock from "@/components/simple-codeblock";
import { motion } from "framer-motion";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  ChevronRight,
  Gamepad2,
  Github,
  Code,
  Terminal,
  Scale,
  PackageOpen,
} from "lucide-react";

const ease = [0.16, 1, 0.3, 1];

export default function HeroCodeSection() {
  const sampleCode = `import { WebAI } from '@axols/webai-js';
 
// 1 - Create
const webai = await WebAI.create({
  modelId: "qwen3.5-0.8b",
  workerPath: "/workers/qwen3.5-0.8b.worker.js",
});

// 2 - Init
await webai.init({
  precision: "q4f16",
  device: "webgpu",
});

// 3 - Generate
const { result } = await webai.generate({
  userInput: {
    messages: [{
      role: "user",
      content: "What is Web AI?",
    }],
  },
});

console.log(result.result);`;

  return (
    <motion.section
      className="w-full lg:w-[42%] relative mx-auto lg:mx-0 flex items-center justify-center"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.8, delay: 0.3, ease }}
    >
      {/* Desktop Layout - Code Block with Side Tags */}
      <div className="hidden lg:flex lg:absolute lg:top-[72px] lg:left-7 z-30 gap-4">
        {/* Code Block Section */}
        <motion.div
          className="relative w-[420px] flex flex-col -mt-4 -ml-4"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease }}
        >
          {/* Header Banner - Above Code */}
          <motion.div
            className="flex items-center justify-between gap-3 px-3 py-2 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10 border border-indigo-500/20 rounded-t-lg mb-0"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease }}
          >
            <div className="flex items-center gap-2">
              <div className="p-1.5 bg-indigo-500/20 rounded-lg">
                <Terminal className="w-3.5 h-3.5 text-indigo-400" />
              </div>
              <h3 className="text-white  text-xs font-mono">
                npm install @axols/webai-js
              </h3>
            </div>
            <motion.button
              onClick={() => {
                navigator.clipboard.writeText("npm install @axols/webai-js");
              }}
              className="inline-flex items-center gap-1.5 px-2 py-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded-md text-white/70 hover:text-white text-xs font-medium transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
              </svg>
              <span>Copy</span>
            </motion.button>
          </motion.div>

          <SimpleCodeblock
            code={sampleCode}
            language="javascript"
            className="bg-[#0c0c0c] z-50 rounded-none border-t-0 "
          />
          <motion.div
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-purple-500/5 via-indigo-500/5 to-purple-500/5 border-x border-b border-indigo-500/20 rounded-b-lg"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.35, ease }}
          >
            <div className="flex items-center gap-2">
              <PackageOpen className="w-3.5 h-3.5 text-purple-400" />
              <span className="text-xs font-medium text-purple-300">
                Open Source
              </span>
              <span className="text-xs text-muted-foreground">•</span>
              <Scale className="w-3.5 h-3.5 text-indigo-300" />
              <span className="text-xs font-medium text-indigo-400">
                Apache 2.0 License
              </span>
            </div>
          </motion.div>
        </motion.div>

        {/* Right Section - Tags and Buttons - Desktop only */}
        <div className="flex flex-col gap-6 pt-12">
          {/* Tags Section */}
          <div className="flex flex-col gap-3">
            <motion.div
              className="flex items-center justify-center rounded-full bg-primary/20 px-3 py-1 text-xs font-medium text-primary ring-1 ring-primary/30 w-fit"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.4, ease }}
            >
              LLM
            </motion.div>
            <motion.div
              className="flex items-center justify-center rounded-full bg-orange-500/20 px-3 py-1 text-xs font-medium text-orange-400 ring-1 ring-orange-500/30 w-fit"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.5, ease }}
            >
              NLP
            </motion.div>
            <motion.div
              className="flex items-center justify-center rounded-full bg-purple-500/20 px-3 py-1 text-xs font-medium text-purple-400 ring-1 ring-purple-500/30 w-fit"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.6, ease }}
            >
              GenAI
            </motion.div>
            <motion.div
              className="flex items-center justify-center rounded-full bg-indigo-500/20 px-3 py-1 text-xs font-medium text-indigo-400 ring-1 ring-indigo-500/30 w-fit"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.7, ease }}
            >
              Image
            </motion.div>
            <motion.div
              className="flex items-center justify-center rounded-full bg-amber-500/20 px-3 py-1 text-xs font-medium text-amber-400 ring-1 ring-amber-500/30 w-fit"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.8, ease }}
            >
              Video
            </motion.div>
            <motion.div
              className="flex items-center justify-center rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-400 ring-1 ring-emerald-500/30 w-fit"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.9, ease }}
            >
              Audio
            </motion.div>
            <motion.div
              className="flex items-center justify-center rounded-full bg-gray-500/20 px-3 py-1 text-xs font-medium text-gray-400 ring-1 ring-gray-500/30 w-fit"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 1.0, ease }}
            >
              ...
            </motion.div>
          </div>

          {/* Buttons Section - Updated styling */}
          <div className="flex flex-col gap-2 mt-6">
            <motion.button
              onClick={() => {
                window.open(
                  "https://github.com/axolsai/webai-js/blob/main/model-workers-v4/qwen3.5-0.8b.worker.js",
                  "_blank",
                );
              }}
              className="group relative inline-flex items-center justify-between gap-1.5 px-2 py-1.5 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 rounded-md text-white text-xs font-semibold shadow-md shadow-blue-500/30 hover:shadow-blue-500/50 transition-all duration-300 border border-blue-400/20 min-w-[120px]"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 1.1, ease }}
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="absolute inset-0 bg-gradient-to-t from-white/0 to-white/10 rounded-md" />
              <Gamepad2 className="w-3.5 h-3.5 relative z-10" />
              <span className="relative z-10">View Worker</span>
              <ChevronRight className="w-3.5 h-3.5 relative z-10 group-hover:translate-x-0.5 transition-transform" />
            </motion.button>

            <motion.a
              href="https://github.com/axolsai/webai-js"
              target="_blank"
              rel="noopener noreferrer"
              className="group relative inline-flex items-center justify-between gap-1.5 px-2 py-1.5 bg-gradient-to-br from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 rounded-md text-white text-xs font-semibold shadow-md shadow-purple-500/30 hover:shadow-purple-500/50 transition-all duration-300 border border-purple-400/20 min-w-[120px]"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 1.2, ease }}
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="absolute inset-0 bg-gradient-to-t from-white/0 to-white/10 rounded-md" />
              <Github className="w-3.5 h-3.5 relative z-10" />
              <span className="relative z-10">Star on GitHub</span>
              <ChevronRight className="w-3.5 h-3.5 relative z-10 group-hover:translate-x-0.5 transition-transform" />
            </motion.a>
          </div>
        </div>
      </div>

      {/* Mobile Layout - Original structure for mobile */}
      <motion.div
        className="lg:hidden relative mx-auto max-w-[420px] sm:w-[400px] md:w-[420px] w-full"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2, ease }}
      >
        {/* Header Banner - Above Code */}
        <motion.div
          className="flex items-center justify-between gap-3 px-3 py-2 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10 border border-indigo-500/20 rounded-t-lg mb-0"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3, ease }}
        >
          <div className="flex items-center gap-2">
            <div className="p-1.5 bg-indigo-500/20 rounded-lg">
              <Code className="w-3.5 h-3.5 text-indigo-400" />
            </div>
            <h3 className="text-white font-semibold text-xs">
              npm install @axols/webai-js
            </h3>
          </div>
          <motion.button
            onClick={() => {
              navigator.clipboard.writeText("npm install @axols/webai-js");
            }}
            className="inline-flex items-center gap-1.5 px-2 py-1 bg-white/5 hover:bg-white/10 border border-white/10 rounded-md text-white/70 hover:text-white text-xs font-medium transition-all"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
              <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
            </svg>
            <span>Copy</span>
          </motion.button>
        </motion.div>

        <SimpleCodeblock
          code={sampleCode}
          language="javascript"
          className="bg-[#0c0c0c] z-20 rounded-t-none"
        />

        {/* Open Source Badge - Below Code */}
        <motion.div
          className="flex items-center justify-center gap-2 px-4 py-2.5 bg-gradient-to-r from-purple-500/5 via-indigo-500/5 to-purple-500/5 border-x border-b border-indigo-500/20 rounded-b-lg"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.35, ease }}
        >
          <div className="flex items-center gap-2">
            <Github className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-xs font-medium text-purple-300">
              Open Source
            </span>
            <span className="text-xs text-muted-foreground">•</span>
            <span className="text-xs font-medium text-indigo-300">
              Apache 2.0 License
            </span>
          </div>
        </motion.div>

        {/* Mobile tags (horizontal row) */}
        <TooltipProvider>
          <motion.div
            className="flex flex-wrap justify-center gap-2 mt-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4, ease }}
          >
            <div className="flex items-center justify-center rounded-full bg-primary/20 px-3 py-1 text-xs font-medium text-primary ring-1 ring-primary/30">
              LLM
            </div>
            <div className="flex items-center justify-center rounded-full bg-orange-500/20 px-3 py-1 text-xs font-medium text-orange-400 ring-1 ring-orange-500/30">
              NLP
            </div>
            <div className="flex items-center justify-center rounded-full bg-purple-500/20 px-3 py-1 text-xs font-medium text-purple-400 ring-1 ring-purple-500/30">
              GenAI
            </div>
            <div className="flex items-center justify-center rounded-full bg-indigo-500/20 px-3 py-1 text-xs font-medium text-indigo-400 ring-1 ring-indigo-500/30">
              Image
            </div>
            <div className="flex items-center justify-center rounded-full bg-amber-500/20 px-3 py-1 text-xs font-medium text-amber-400 ring-1 ring-amber-500/30">
              Video
            </div>
            <div className="flex items-center justify-center rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-400 ring-1 ring-emerald-500/30">
              Audio
            </div>
            <div className="flex items-center justify-center rounded-full bg-gray-500/20 px-3 py-1 text-xs font-medium text-gray-400 ring-1 ring-gray-500/30">
              ...
            </div>
          </motion.div>
        </TooltipProvider>

        {/* Mobile Buttons - positioned below badges, centered */}
        <motion.div
          className="flex justify-center gap-3 mt-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6, ease }}
        >
          <motion.button
            onClick={() => {
              window.location.href =
                "https://github.com/axolsai/webai-js/blob/main/model-workers-v4/qwen3.5-0.8b.worker.js";
            }}
            className="group relative inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 rounded-lg text-white text-sm font-semibold shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 transition-all duration-300 border border-blue-400/20"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="absolute inset-0 bg-gradient-to-t from-white/0 to-white/10 rounded-lg" />
            <Gamepad2 className="w-4 h-4 relative z-10" />
            <span className="relative z-10">Demo</span>
          </motion.button>

          <motion.a
            href="https://github.com/axolsai/webai-js"
            target="_blank"
            rel="noopener noreferrer"
            className="group relative inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-br from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 rounded-lg text-white text-sm font-semibold shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50 transition-all duration-300 border border-purple-400/20"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="absolute inset-0 bg-gradient-to-t from-white/0 to-white/10 rounded-lg" />
            <Github className="w-4 h-4 relative z-10" />
            <span className="relative z-10">Star on GitHub</span>
          </motion.a>
        </motion.div>
      </motion.div>

      {/* Create a container with fade mask only on the right - Only visible on large screens */}
      <motion.div
        className="w-full h-full relative overflow-hidden hidden lg:block -mt-4 -ml-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 0.9, ease }}
      >
        <div className="w-full h-full relative">
          {/* Safari component */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 100, filter: "blur(8px)" }}
            animate={{ opacity: 1, scale: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 1.2, delay: 0.3, ease }}
            className="max-h-[560px] overflow-hidden "
          >
            <Safari
              url="www.webai-js.com"
              className="size-[180%] xl:size-[160%]"
            />
          </motion.div>

          {/* Right and bottom fade masks */}
          <motion.div className="absolute inset-0 pointer-events-none rounded-br-full">
            {/* Right fade - increased width for more pronounced effect */}
            <div className="absolute top-0 right-0 h-full w-24 bg-gradient-to-l from-background to-transparent"></div>
            {/* Bottom fade */}
            <div className="absolute bottom-0 left-0 w-full h-16 bg-gradient-to-t from-background to-transparent"></div>
          </motion.div>
        </div>
      </motion.div>
    </motion.section>
  );
}
