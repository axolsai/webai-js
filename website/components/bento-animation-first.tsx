/* eslint-disable @next/next/no-img-element */
"use client";

import {
  Reasoning,
  ReasoningContent,
  ReasoningResponse,
} from "@/components/ui/reasoning";
import { Bot } from "lucide-react";
import { AnimatePresence, motion, useInView } from "framer-motion";
import { useEffect, useRef, useState } from "react";

export function ReasoningBasic() {
  const reasoningText = `Absolutely. it's completely private. Everything stays on your device—I process it all locally, and nothing is ever sent to a server or collected elsewhere. It's just between us. No one else has access, and your privacy is always my top priority.`;

  return (
    <Reasoning>
      <ReasoningContent className="">
        <ReasoningResponse text={reasoningText} />
      </ReasoningContent>
    </Reasoning>
  );
}

export default function FirstBentoAnimation() {
  const ref = useRef(null);
  const isInView = useInView(ref);
  const [shouldAnimate, setShouldAnimate] = useState(false);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    if (isInView) {
      timeoutId = setTimeout(() => {
        setShouldAnimate(true);
      }, 1000);
    } else {
      setShouldAnimate(false);
    }

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [isInView]);

  return (
    <div
      ref={ref}
      className="w-full h-full p-4 flex flex-col items-center justify-center gap-5"
    >
      <motion.div
        className="max-w-md mx-auto w-full flex flex-col gap-2"
        animate={{
          y: shouldAnimate ? -75 : 0,
        }}
        transition={{
          type: "spring",
          stiffness: 300,
          damping: 20,
        }}
      >
        <div className="flex items-end justify-end gap-3  mb-4">
          <motion.div
            className="max-w-[320px] bg-secondary text-white p-4 rounded-2xl ml-auto"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{
              duration: 0.3,
              ease: "easeOut",
            }}
          >
            <p className="text-sm">
              Hey, just to be sure… are our conversations private and secure?
            </p>
          </motion.div>
          <div className="flex items-center bg-background rounded-full w-fit border border-border flex-shrink-0">
            <img
              src="https://randomuser.me/api/portraits/women/79.jpg"
              alt="User Avatar"
              className="size-8 rounded-full flex-shrink-0"
            />
          </div>
        </div>
        <div className="flex items-start gap-2">
          <div className="flex items-center bg-background rounded-full size-10 flex-shrink-0 justify-center border border-border">
            <Bot />
          </div>

          <div className="relative">
            <AnimatePresence mode="wait">
              {!shouldAnimate ? (
                <motion.div
                  key="dots"
                  className="absolute left-0 top-0 bg-background p-4 rounded-2xl border border-border"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{
                    duration: 0.2,
                    ease: "easeOut",
                  }}
                >
                  <div className="flex gap-1">
                    {[0, 1, 2].map((index) => (
                      <motion.div
                        key={index}
                        className="w-2 h-2 bg-primary/50 rounded-full"
                        animate={{ y: [0, -5, 0] }}
                        transition={{
                          duration: 0.6,
                          repeat: Infinity,
                          delay: index * 0.2,
                          ease: "easeInOut",
                        }}
                      />
                    ))}
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="response"
                  layout
                  className="absolute left-0 top-0 md:min-w-[320px] min-w-[220px] p-4 bg-accent border border-border rounded-xl"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{
                    opacity: 1,
                    x: 0,
                  }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{
                    duration: 0.3,
                    ease: "easeOut",
                  }}
                >
                  <ReasoningBasic />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
