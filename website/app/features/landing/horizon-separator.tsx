"use client";

import React from "react";
import { motion } from "framer-motion";

interface HorizonSeparatorProps {
  className?: string;
  width?: string;
  height?: string;
}

const HorizonSeparator: React.FC<HorizonSeparatorProps> = ({
  className = "",
  width = "60%",
  height = "24",
}) => {
  return (
    <motion.div
      className={`relative h-${height} flex items-center justify-center ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1, delay: 0.5 }}
    >
      <div className={`w-[${width}] h-full flex items-center justify-center`}>
        {/* Enhanced horizon line with animated glow */}
        <motion.div
          className="w-full h-px bg-gradient-to-r from-transparent via-slate-300/90 to-transparent"
          animate={{
            opacity: [0.5, 1, 0.5],
            scaleY: [1, 1.5, 1],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            repeatType: "reverse",
          }}
        />
        <motion.div
          className="absolute w-[80%] h-1 bg-gradient-to-r from-transparent via-slate-300/40 to-transparent blur-sm"
          animate={{
            opacity: [0.3, 0.7, 0.3],
            width: ["70%", "80%", "70%"],
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            repeatType: "reverse",
          }}
        />
        <motion.div
          className="absolute w-[80%] h-1 bg-gradient-to-r from-transparent via-blue-50/30 to-transparent blur-md"
          animate={{
            opacity: [0.2, 0.5, 0.2],
            width: ["75%", "85%", "75%"],
          }}
          transition={{
            duration: 7,
            repeat: Infinity,
            repeatType: "reverse",
          }}
        />

        {/* Animated center sun element */}
        <div className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2">
          <motion.div
            className="relative"
            animate={{
              scale: [1, 1.1, 1],
              rotate: [0, 360],
            }}
            transition={{
              scale: {
                duration: 4,
                repeat: Infinity,
                repeatType: "reverse",
              },
              rotate: {
                duration: 20,
                repeat: Infinity,
                ease: "linear",
              },
            }}
          >
            {/* Larger, more radiant sun with silverish center */}
            <motion.div
              className="h-6 w-6 bg-slate-200 rounded-full blur-xl opacity-90"
              animate={{
                opacity: [0.7, 0.9, 0.7],
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                repeatType: "reverse",
              }}
            />
            <motion.div
              className="absolute inset-0 h-4 w-4 bg-slate-100 rounded-full left-1 top-1 blur-lg opacity-80"
              animate={{
                opacity: [0.6, 0.8, 0.6],
                scale: [1, 1.1, 1],
              }}
              transition={{
                duration: 2.5,
                repeat: Infinity,
                repeatType: "reverse",
                delay: 0.3,
              }}
            />
            <motion.div
              className="absolute inset-0 h-3 w-3 bg-white rounded-full left-2 top-2 blur-md opacity-100"
              animate={{
                opacity: [0.8, 1, 0.8],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                repeatType: "reverse",
                delay: 0.6,
              }}
            />

            {/* Intense bright center spot with pulse */}
            <motion.div
              className="absolute inset-0 h-1 w-1 bg-white rounded-full left-2.5 top-2.5"
              animate={{
                opacity: [0.9, 1, 0.9],
                scale: [1, 1.3, 1],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                repeatType: "reverse",
              }}
            />

            {/* Subtle glow around the sun with pulse */}
            <motion.div
              className="absolute -inset-2 bg-slate-100/30 rounded-full blur-xl"
              animate={{
                opacity: [0.2, 0.4, 0.2],
                scale: [1, 1.1, 1],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                repeatType: "reverse",
              }}
            />
            <motion.div
              className="absolute -inset-4 bg-slate-100/20 rounded-full blur-2xl"
              animate={{
                opacity: [0.1, 0.3, 0.1],
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: 5,
                repeat: Infinity,
                repeatType: "reverse",
                delay: 0.5,
              }}
            />
          </motion.div>
        </div>
      </div>

      {/* Animated side gradient fades */}
      <motion.div
        className="absolute left-1/4 top-1/2 transform -translate-y-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-slate-300/40 to-transparent blur-sm"
        animate={{
          opacity: [0.3, 0.5, 0.3],
          width: ["45%", "55%", "45%"],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />
      <motion.div
        className="absolute left-1/4 top-1/2 transform -translate-y-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-slate-300/20 to-transparent blur-md -translate-y-2"
        animate={{
          opacity: [0.2, 0.4, 0.2],
          width: ["48%", "52%", "48%"],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          repeatType: "reverse",
          delay: 1,
        }}
      />
      <motion.div
        className="absolute left-1/4 top-1/2 transform -translate-y-1/2 w-1/2 h-px bg-gradient-to-r from-transparent via-slate-300/20 to-transparent blur-md translate-y-2"
        animate={{
          opacity: [0.2, 0.4, 0.2],
          width: ["46%", "54%", "46%"],
        }}
        transition={{
          duration: 9,
          repeat: Infinity,
          repeatType: "reverse",
          delay: 2,
        }}
      />
    </motion.div>
  );
};

export default HorizonSeparator;
