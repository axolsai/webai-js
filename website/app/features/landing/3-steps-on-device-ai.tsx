"use client";

import SimpleCodeblock from "@/components/simple-codeblock";

import { motion, useScroll, useTransform, cubicBezier } from "motion/react";
import { useRef, useMemo } from "react";
import { ChevronRight, Rocket, Book, Gamepad2 } from "lucide-react";
import CopyButton from "./copy-button";

export default function ThreeStepsIntegration() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  });
  const opacityHeader = useTransform(scrollYProgress, [0, 0.2], [0, 1]);

  // Card data - Desktop order: Create, Generate, Initiate (1, 3, 2)
  const cards = useMemo(
    () => [
      {
        number: 1,
        title: "Create",
        description: "Create a WebAI instance",
        gradient: "from-emerald-300 via-green-400 to-teal-500",
        buttonGradient: "from-green-500 to-emerald-600",
        icon: Rocket,
        buttonText: "Get Started",
        code: `import { WebAI } from '@axols/webai-js';

const webai = await WebAI.create({
  modelId: "qwen3.5-0.8b",
  workerPath: "/workers/qwen3.5-0.8b.worker.js",
});`,
        href: "https://github.com/axolsai/webai-js#quick-start",
        shadow:
          "rgba(39,245,76,0.2) 0px 25px 80px -15px, rgba(36,42,66,0.08) 0px 15px 30px -8px, rgba(36,42,66,0.1) 0px 2px 8px -1px",
      },
      {
        number: 3,
        title: "Generate",
        description: "Start to generate with WebAI already",
        gradient: "from-fuchsia-400 via-pink-500 to-rose-600",
        buttonGradient: "from-pink-500 to-rose-600",
        icon: Gamepad2,
        buttonText: "View Worker",
        code: `const { result } = await webai.generate({
  userInput: {
    messages: [{
      role: "user",
      content: "What is Web AI?",
    }],
  },
});

console.log(result.result);`,
        href: "https://github.com/axolsai/webai-js/blob/main/model-workers-v4/qwen3.5-0.8b.worker.js",
        shadow:
          "rgba(245,40,145,0.2) 0px 25px 80px -15px, rgba(36,42,66,0.08) 0px 15px 30px -8px, rgba(36,42,66,0.1) 0px 2px 8px -1px",
      },
      {
        number: 2,
        title: "Initiate",
        description: "Load WebAI models into memory",
        gradient: "from-indigo-400 via-blue-500 to-sky-600",
        buttonGradient: "from-blue-500 to-indigo-600",
        icon: Book,
        buttonText: "Learn More",
        code: `await webai.init({
  precision: "q4f16",
  device: "webgpu",
});`,
        href: "https://github.com/axolsai/webai-js/blob/main/benchmarks/qwen3.5-0.8b.json",
        shadow:
          "rgba(39,127,245,0.2) 0px 25px 80px -15px, rgba(36,42,66,0.08) 0px 15px 30px -8px, rgba(36,42,66,0.1) 0px 2px 8px -1px",
      },
    ],
    [],
  );

  // Mobile card data - Logical order: Create, Initiate, Generate (1, 2, 3)
  const mobileCards = useMemo(
    () => [
      cards[0], // Create (1)
      cards[2], // Initiate (2)
      cards[1], // Generate (3)
    ],
    [cards],
  );

  // Desktop variants with original complex animations
  const desktopVariants = useMemo(
    () => [
      {
        initial: {
          x: 35,
          y: 5,
          scale: 0.8,
          rotate: -3,
          zIndex: 1,
          transition: {
            delay: 0.05,
            duration: 0.15,
            ease: cubicBezier(0.22, 1, 0.36, 1),
          },
        },
        whileHover: {
          x: 0,
          y: 0,
          scale: 1,
          rotate: 0,
          transition: {
            delay: 0.05,
            duration: 0.15,
            ease: cubicBezier(0.22, 1, 0.36, 1),
          },
        },
        width: "w-[350px] xl:w-[350px] lg:w-[300px]",
      },
      {
        initial: {
          scale: 1.05,
          zIndex: 2,
          transition: {
            delay: 0.05,
            duration: 0.15,
            ease: cubicBezier(0.22, 1, 0.36, 1),
          },
        },
        whileHover: {
          scale: 1,
          transition: {
            delay: 0.05,
            duration: 0.15,
            ease: cubicBezier(0.22, 1, 0.36, 1),
          },
        },
        width: "w-[400px] xl:w-[400px] lg:w-[320px]",
      },
      {
        initial: {
          x: -35,
          y: 5,
          scale: 0.8,
          rotate: 3,
          zIndex: 1,
          transition: {
            delay: 0.05,
            duration: 0.15,
            ease: cubicBezier(0.22, 1, 0.36, 1),
          },
        },
        whileHover: {
          x: 0,
          y: 0,
          scale: 1,
          rotate: 0,
          transition: {
            delay: 0.05,
            duration: 0.15,
            ease: cubicBezier(0.22, 1, 0.36, 1),
          },
        },
        width: "w-[350px] xl:w-[350px] lg:w-[300px]",
      },
    ],
    [],
  );

  // Mobile variants - simple and uniform
  const mobileVariants = useMemo(
    () => ({
      initial: {
        scale: 1,
        transition: { duration: 0.15, ease: cubicBezier(0.22, 1, 0.36, 1) },
      },
      whileHover: {
        scale: 1.02,
        y: -5,
        transition: { duration: 0.15, ease: cubicBezier(0.22, 1, 0.36, 1) },
      },
    }),
    [],
  );

  const containerVariants = useMemo(
    () => ({
      initial: {},
      whileHover: { transition: { staggerChildren: 0.1 } },
    }),
    [],
  );

  const buttonVariants = useMemo(
    () => ({
      rest: { scale: 1, transition: { duration: 0.2, ease: "easeInOut" } },
      hover: { scale: 1.03, transition: { duration: 0.2, ease: "easeInOut" } },
      tap: { scale: 0.97, transition: { duration: 0.1, ease: "easeInOut" } },
    }),
    [],
  );

  const chevronVariants = useMemo(
    () => ({
      rest: { x: 0, transition: { duration: 0.2, ease: "easeInOut" } },
      hover: {
        x: 3,
        transition: { duration: 0.2, ease: "easeInOut", delay: 0.1 },
      },
    }),
    [],
  );

  // Separate copy button component to isolate state changes - NOW USING THE IMPORTED ONE

  // Static animated elements that won't re-render when copy state changes
  const AnimatedHeader = () => (
    <>
      <motion.h2
        className="mt-4 text-4xl font-bold text-white"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.7, delay: 1.2 }}
      >
        Running <span className="text-primary">AI on the Client</span> Has Never
        Been Easier
      </motion.h2>

      <motion.p
        className="mt-6 text-lg text-gray-300 max-w-2xl mx-auto"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.7, delay: 1.3 }}
      >
        A First-Class Developer Experience — fast, simple, affordable, and
        secure. <br /> And best of all,{" "}
        <span className="italic text-primary/90 font-medium">
          it just works.
        </span>
      </motion.p>
    </>
  );

  const AnimatedInstallCommand = () => (
    <motion.div
      className="flex justify-center items-center mt-[52px]"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.35 }}
    >
      <div
        className="inline-flex items-center rounded-md p-1.5 px-3 relative overflow-hidden shadow-md"
        style={{
          background:
            "linear-gradient(180deg, rgba(0, 0, 0, 0.9) 0%, rgba(5, 6, 15, 0.98) 100%)",
          border: "1px solid rgba(150, 160, 180, 0.35)",
          boxShadow:
            "0 2px 10px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.08)",
        }}
      >
        <div className="text-neutral-300 font-mono text-sm px-1">
          npm install @axols/webai-js
        </div>
        <CopyButton text="npm install @axols/webai-js" />
      </div>
    </motion.div>
  );

  // Shared header component
  const Header = () => (
    <motion.div
      className="mb-16 text-center relative"
      style={{ opacity: opacityHeader }}
    >
      <AnimatedHeader />
      <AnimatedInstallCommand />
    </motion.div>
  );

  // Shared card component
  type CardProps = {
    card: {
      number: number;
      title: string;
      description: string;
      gradient: string;
      buttonGradient: string;
      icon: React.ElementType;
      buttonText: string;
      code: string;
      href: string;
      shadow: string;
    };
    variant: any;
    isDesktop: boolean;
    index: number;
  };

  const Card = ({ card, variant, isDesktop }: CardProps) => {
    const IconComponent = card.icon;
    const cardWidth = isDesktop ? variant.width : "w-full max-w-[400px]";

    return (
      <motion.div
        variants={variant}
        whileHover={
          isDesktop
            ? { ...variant.whileHover, boxShadow: card.shadow }
            : { ...mobileVariants.whileHover, boxShadow: card.shadow }
        }
        className={`z-[3] flex h-[480px] ${cardWidth} flex-col items-center justify-between rounded-xl border border-neutral-400/20 bg-white p-6 dark:border-neutral-800 dark:bg-neutral-900`}
      >
        <div className="flex flex-col items-center relative">
          <div className="relative h-24 w-24 rounded-full mb-6 overflow-hidden">
            <div
              className={`absolute inset-0 bg-gradient-to-br ${card.gradient}`}
            ></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-white text-4xl font-bold">
                {card.number}
              </span>
            </div>
          </div>
          <div className="text-center mb-6">
            <h3 className="text-2xl font-bold text-neutral-900 dark:text-white">
              {card.title}
            </h3>
            <p className="text-neutral-600 dark:text-neutral-300 mt-2">
              {card.description}
            </p>
          </div>
        </div>
        <div className="flex-grow w-full flex items-center bg-muted/50 rounded-md">
          <SimpleCodeblock
            copiable={false}
            code={card.code}
            language="javascript"
            className="text-start h-full w-full"
          />
        </div>
        <motion.button
          initial="rest"
          whileHover="hover"
          whileTap="tap"
          variants={buttonVariants}
          onClick={() => {
            window.location.href = card.href;
          }}
          className={`h-12 shrink-0 px-6 bg-gradient-to-r ${card.buttonGradient} text-white font-medium rounded-lg w-full mt-6 relative group overflow-hidden flex items-center justify-center`}
        >
          <div className="absolute inset-0 bg-black/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          <div className="flex items-center justify-center">
            <IconComponent size={18} className="mr-2" />
            <span>{card.buttonText}</span>
            <motion.div variants={chevronVariants} className="ml-2">
              <ChevronRight size={18} />
            </motion.div>
          </div>
        </motion.button>
      </motion.div>
    );
  };

  return (
    <div
      className="relative w-full max-w-7xl mx-auto px-4 overflow-hidden"
      ref={containerRef}
    >
      <Header />

      {/* Desktop Version */}
      <motion.div
        variants={containerVariants}
        initial="initial"
        whileHover="whileHover"
        className="hidden lg:flex h-auto w-full max-w-6xl mx-auto mt-8 items-center justify-center"
      >
        <div className="flex flex-row h-auto w-full items-center justify-center gap-8 mt-10">
          {cards.map((card, index) => (
            <Card
              key={card.number}
              card={card}
              variant={desktopVariants[index]}
              isDesktop={true}
              index={index}
            />
          ))}
        </div>
      </motion.div>

      {/* Mobile/Tablet Version */}
      <div className="block lg:hidden w-full max-w-4xl mx-auto mt-8">
        <div className="flex flex-col items-center justify-center gap-8 mt-10">
          {mobileCards.map((card, index) => (
            <Card
              key={card.number}
              card={card}
              variant={mobileVariants}
              isDesktop={false}
              index={index}
            />
          ))}
        </div>
      </div>

      {/* Footer */}
      <motion.div
        className="mt-16"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.4 }}
      >
        {/* Uncomment if needed
        <h2 className="text-xl font-medium text-white mb-4">
          <span className="text-primary">3 Easy Steps</span> to Integrate AI into Modern Web Applications
        </h2>
        <p className="text-lg">Fully compatible with JavaScript and TypeScript</p>
        */}
      </motion.div>
    </div>
  );
}
