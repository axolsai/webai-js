"use client";

import { motion } from "framer-motion";

export default function SocialProof() {
  const technologies = [
    {
      name: "ONNX Runtime",
      logo: "/microsoft-color.png",
    },
    {
      name: "Transformers.js",
      logo: "/huggingface-color.png",
    },
    {
      name: "WebGPU",
      logo: "/webgpu-notext.svg",
    },
    {
      name: "LiteRT.js",
      logo: "/litert-js.png",
    },
    {
      name: "TensorFlow.js",
      logo: "/google-color.png",
    },
  ];

  return (
    <section className="w-full mb-4 bg-zinc-950 ">
      <div className="container mx-auto px-4">
        <motion.div
          className="text-center mb-8"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1 }}
        >
          <p className="text-sm md:text-xl text-zinc-400 font-bold tracking-wide uppercase">
            Built on industry-leading frameworks
          </p>
        </motion.div>

        <div className="flex items-center justify-center gap-x-12 md:gap-x-24">
          {technologies.map((tech, index) => (
            <motion.div
              key={index}
              className="group relative"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.4,
                delay: 1 + index * 0.1,
                ease: "easeOut",
              }}
            >
              <div className="relative bg-zinc-950 backdrop-blur-sm transition-all duration-300 group-hover:border-transparent flex gap-3 h-16">
                {/* Logo image */}
                <div className="rounded-xl flex items-center justify-center w-12">
                  <img
                    src={tech.logo}
                    alt={`${tech.name} logo`}
                    className="w-9 h-9 object-contain"
                  />
                </div>

                <div className="text-lg font-semibold text-white h-full flex items-center">
                  {tech.name}
                </div>

                {/* Subtle shine effect */}
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/[0.02] via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
