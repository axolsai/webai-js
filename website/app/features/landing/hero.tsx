// LandingHero.jsx - Left text section only
"use client";
import { InteractiveHoverButton } from "@/components/magicui/interactive-hover-button";
import { motion } from "framer-motion";
import { FaReact } from "react-icons/fa";
import { BiLogoTypescript, BiLogoJavascript } from "react-icons/bi";
import { SiNextdotjs, SiVuedotjs, SiElectron } from "react-icons/si";
import { HeroAuroraText } from "@/components/hero-aurora-text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import HeroCodeSection from "./hero-code-section";

const ease = [0.16, 1, 0.3, 1];

export default function LandingHero() {
  return (
    <motion.div
      className="w-full flex flex-col lg:flex-row my-12 justify-between min-h-[450px] lg:min-h-[520px] px-4 sm:px-0"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8, ease }}
    >
      <motion.section
        className="w-full lg:w-[53%] flex flex-col justify-between gap-y-6 lg:pl-8 mb-10 lg:mb-0"
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8, ease }}
      >
        <div className="flex w-full flex-col overflow-hidden pt-8 lg:pt-12">
          <motion.h1
            className="text-center lg:text-left text-3xl lg:text-4xl xl:text-6xl font-semibold leading-[1.1] text-foreground tracking-tighter"
            initial={{ filter: "blur(10px)", opacity: 0, y: 50 }}
            animate={{ filter: "blur(0px)", opacity: 1, y: 0 }}
            transition={{
              duration: 1,
              ease,
              staggerChildren: 0.2,
            }}
          >
            <motion.span
              className="inline-block text-balance"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 0.8,
                delay: 0.5,
                ease,
              }}
            >
              <HeroAuroraText className="leading-[1.2] border border-background">
                Run AI Models Locally in the Browser
              </HeroAuroraText>
            </motion.span>
          </motion.h1>
          <motion.p
            className="text-center lg:text-left leading-normal text-muted-foreground text-base sm:text-lg sm:leading-normal text-balance mt-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              delay: 0.6,
              duration: 0.8,
              ease,
            }}
          >
            The best way to embed AI in your production web app. <br />
            Scale instantly — no AI hosting headaches, no data privacy concerns.
          </motion.p>
          <motion.div
            className="mt-6 sm:mt-8 flex justify-center lg:justify-start gap-x-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.8, ease }}
          >
            <InteractiveHoverButton
              onClick={() => {
                window.location.href = "/models";
              }}
            >
              Find a Model
            </InteractiveHoverButton>
          </motion.div>
        </div>

        <motion.div
          className="flex flex-col mt-6 sm:mt-8 pb-4 overflow-visible"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.8, ease }}
        >
          <motion.p
            className="text-center lg:text-left text-muted-foreground"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.75, duration: 0.8, ease }}
          >
            Compatible with major front-end frameworks:
          </motion.p>
          <TooltipProvider>
            <motion.div
              className="flex flex-row justify-center lg:justify-start gap-4 sm:gap-6 mt-3 sm:mt-4 pb-[20px]"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                delay: 0.8,
                duration: 0.5,
                ease,
              }}
            >
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.85, duration: 0.5, ease }}
              >
                <Tooltip delayDuration={0}>
                  <TooltipTrigger>
                    <FaReact className="h-5 w-5 sm:h-6 sm:w-6 text-foreground/90" />
                  </TooltipTrigger>
                  <TooltipContent
                    side="bottom"
                    className="bg-muted/30 border border-border text-foreground"
                  >
                    React
                  </TooltipContent>
                </Tooltip>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9, duration: 0.5, ease }}
              >
                <Tooltip delayDuration={0}>
                  <TooltipTrigger>
                    <SiNextdotjs className="h-5 w-5 sm:h-6 sm:w-6 text-foreground/90" />
                  </TooltipTrigger>
                  <TooltipContent
                    side="bottom"
                    className="bg-muted/30 border border-border text-foreground"
                  >
                    Next.js
                  </TooltipContent>
                </Tooltip>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.95, duration: 0.5, ease }}
              >
                <Tooltip delayDuration={0}>
                  <TooltipTrigger>
                    <SiVuedotjs className="h-5 w-5 sm:h-6 sm:w-6 text-foreground/90" />
                  </TooltipTrigger>
                  <TooltipContent
                    side="bottom"
                    className="bg-muted/30 border border-border text-foreground"
                  >
                    Vue.js
                  </TooltipContent>
                </Tooltip>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0, duration: 0.5, ease }}
              >
                <Tooltip delayDuration={0}>
                  <TooltipTrigger>
                    <SiElectron className="h-5 w-5 sm:h-6 sm:w-6 text-foreground/90" />
                  </TooltipTrigger>
                  <TooltipContent
                    side="bottom"
                    className="bg-muted/30 border border-border text-foreground"
                  >
                    Electron
                  </TooltipContent>
                </Tooltip>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.05, duration: 0.5, ease }}
              >
                <Tooltip delayDuration={0}>
                  <TooltipTrigger>
                    <BiLogoJavascript className="h-5 w-5 sm:h-6 sm:w-6 text-foreground/90 scale-[137%]" />
                  </TooltipTrigger>
                  <TooltipContent
                    side="bottom"
                    className="bg-muted/30 border border-border text-foreground"
                  >
                    JavaScript
                  </TooltipContent>
                </Tooltip>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.1, duration: 0.5, ease }}
              >
                <Tooltip delayDuration={0}>
                  <TooltipTrigger>
                    <BiLogoTypescript className="h-5 w-5 sm:h-6 sm:w-6 text-foreground/90 scale-[137%]" />
                  </TooltipTrigger>
                  <TooltipContent
                    side="bottom"
                    className="bg-muted/30 border border-border text-foreground"
                  >
                    TypeScript
                  </TooltipContent>
                </Tooltip>
              </motion.div>
            </motion.div>
          </TooltipProvider>
        </motion.div>
      </motion.section>

      <HeroCodeSection />
    </motion.div>
  );
}
