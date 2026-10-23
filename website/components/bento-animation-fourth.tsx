"use client";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const seededValue = (index: number, salt: number) => {
  const value = Math.sin((index + 1) * 12.9898 + salt * 78.233) * 43758.5453;
  return Number((value - Math.floor(value)).toFixed(3));
};

const FourthBentoAnimation = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, {
    once: true, // Animation triggers only once
    margin: "-100px", // Trigger when 100px before entering viewport
  });

  // Container sizes
  const containerWidth = 110;
  const containerHeight = 180;
  const containerRadius = 6;

  // Wave animation properties
  const waveColors = {
    traditional: ["#f6555d", "#f47983"], // Updated to gradient with #f6555d base and darker top
    webAI: ["#10B981", "#6EE7B7"], // Adjusted emerald gradient for dark mode
  };

  return (
    <div
      ref={ref}
      className="flex flex-col items-center w-full h-64 pt-12 px-6 pb-6 rounded-xl mt-16"
    >
      <div className="flex w-full h-full justify-around items-end relative">
        {/* Traditional AI Container */}
        <div className="flex flex-col items-center">
          <motion.div
            className="relative bg-background rounded-md overflow-hidden shadow-lg border border-border"
            style={{
              width: containerWidth,
              height: containerHeight,
              borderRadius: containerRadius,
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 0.8 }}
          >
            {/* Water fill animation for Traditional AI */}
            <motion.div
              className="absolute bottom-0 w-full"
              style={{
                background: `linear-gradient(to top, ${waveColors.traditional[0]}, ${waveColors.traditional[1]})`,
              }}
              initial={{ height: 0 }}
              animate={isInView ? { height: "105%" } : { height: 0 }}
              transition={{ duration: 1.8, ease: "easeInOut", delay: 0.8 }}
            >
              {/* Animated water surface */}
              <svg
                className="absolute top-0 left-0 w-full"
                style={{ transform: "translateY(-98%)" }}
                viewBox="0 0 1200 120"
                preserveAspectRatio="none"
              >
                <motion.path
                  d="M0,0 L1200,0 L1200,120 L0,120 Z"
                  fill={waveColors.traditional[1]}
                  initial={{
                    d: "M0,40 C300,0 900,80 1200,40 L1200,120 L0,120 Z",
                  }}
                  animate={{
                    d: [
                      "M0,40 C300,0 900,80 1200,40 L1200,120 L0,120 Z",
                      "M0,60 C300,20 900,100 1200,60 L1200,120 L0,120 Z",
                      "M0,40 C300,0 900,80 1200,40 L1200,120 L0,120 Z",
                    ],
                  }}
                  transition={{
                    repeat: Infinity,
                    duration: 3,
                    ease: "easeInOut",
                  }}
                />
              </svg>

              {/* Bubbles animation */}
              {[...Array(6)].map((_, index) => (
                <motion.div
                  key={`trad-bubble-${index}`}
                  className="absolute rounded-full bg-white opacity-80"
                  style={{
                    width: seededValue(index, 1) * 6 + 4,
                    height: seededValue(index, 2) * 6 + 4,
                    left: `${seededValue(index, 3) * 80 + 10}%`,
                  }}
                  initial={{
                    bottom: 0,
                    opacity: 0.8,
                  }}
                  animate={{
                    bottom: "100%",
                    opacity: 0,
                  }}
                  transition={{
                    duration: seededValue(index, 4) * 2 + 2,
                    repeat: Infinity,
                    delay: seededValue(index, 5) * 2,
                    ease: "easeOut",
                  }}
                />
              ))}
            </motion.div>

            {/* Overflowing water droplets and splashes */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={isInView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ delay: 2.5 }}
            >
              {/* Left overflow stream */}
              <motion.div
                className="absolute top-0 left-0"
                style={{
                  width: 30,
                  height: 120,
                  overflow: "hidden",
                  zIndex: -1,
                }}
                initial={{ opacity: 0 }}
                animate={isInView ? { opacity: 1 } : { opacity: 0 }}
                transition={{ duration: 0.3, delay: 2.5 }}
              >
                <motion.div
                  className="absolute top-0 left-0 w-2 rounded-b-md"
                  style={{
                    background: `linear-gradient(to bottom, ${waveColors.traditional[1]}, ${waveColors.traditional[0]})`,
                    height: 60,
                    left: 10,
                  }}
                  initial={{ height: 0 }}
                  animate={isInView ? { height: 60 } : { height: 0 }}
                  transition={{ duration: 0.6, ease: "easeOut", delay: 2.5 }}
                />

                {/* Left side slow drip */}
                <motion.div
                  className="absolute top-0 left-0 w-1.5 rounded-b-md"
                  style={{
                    background: `linear-gradient(to bottom, ${waveColors.traditional[1]}, ${waveColors.traditional[0]})`,
                    left: 14,
                    zIndex: 5,
                  }}
                  initial={{ height: 0 }}
                  animate={isInView ? { height: 20 } : { height: 0 }}
                  transition={{
                    duration: 1,
                    ease: "easeOut",
                    repeat: isInView ? Infinity : 0,
                    repeatType: "loop",
                    repeatDelay: 0.5,
                    delay: 2.5,
                  }}
                />

                {/* Small droplet at the end of drip */}
                <motion.div
                  className="absolute rounded-full"
                  style={{
                    background: waveColors.traditional[1],
                    width: 4,
                    height: 4,
                    left: 14,
                    boxShadow: "0px 1px 3px rgba(0,0,0,0.3)",
                  }}
                  initial={{
                    top: 20,
                    opacity: 0,
                    scale: 0.5,
                  }}
                  animate={
                    isInView
                      ? {
                          top: 70,
                          opacity: [0, 1, 1, 0],
                          scale: [0.5, 1, 1, 0.7],
                        }
                      : {
                          top: 20,
                          opacity: 0,
                          scale: 0.5,
                        }
                  }
                  transition={{
                    duration: 1,
                    repeat: isInView ? Infinity : 0,
                    repeatDelay: 0.5,
                    times: [0, 0.1, 0.9, 1],
                    ease: "easeIn",
                    delay: 2.5,
                  }}
                />

                {/* Splash droplets */}
                {[...Array(4)].map((_, i) => (
                  <motion.div
                    key={`left-drop-${i}`}
                    className="absolute rounded-full"
                    style={{
                      background: waveColors.traditional[1],
                      width: seededValue(i, 6) * 4 + 3,
                      height: seededValue(i, 7) * 4 + 3,
                      left: seededValue(i, 8) * 20,
                    }}
                    initial={{
                      top: 0,
                      opacity: 0.8,
                      scale: 1,
                    }}
                    animate={
                      isInView
                        ? {
                            top: seededValue(i, 9) * 60 + 30,
                            opacity: 0,
                            scale: 0.5,
                          }
                        : {
                            top: 0,
                            opacity: 0.8,
                            scale: 1,
                          }
                    }
                    transition={{
                      duration: seededValue(i, 10) * 0.5 + 0.5,
                      repeat: isInView ? Infinity : 0,
                      delay: seededValue(i, 11) * 0.5 + 2.5,
                      ease: "easeOut",
                    }}
                  />
                ))}
              </motion.div>

              {/* Right overflow stream */}
              <motion.div
                className="absolute top-0 right-0"
                style={{
                  width: 30,
                  height: 120,
                  overflow: "hidden",
                  zIndex: -1,
                }}
                initial={{ opacity: 0 }}
                animate={isInView ? { opacity: 1 } : { opacity: 0 }}
                transition={{ duration: 0.3, delay: 2.7 }}
              >
                <motion.div
                  className="absolute top-0 right-0 w-3 rounded-b-md"
                  style={{
                    background: `linear-gradient(to bottom, ${waveColors.traditional[1]}, ${waveColors.traditional[0]})`,
                    height: 40,
                    right: 12,
                  }}
                  initial={{ height: 0 }}
                  animate={isInView ? { height: 40 } : { height: 0 }}
                  transition={{ duration: 0.5, ease: "easeOut", delay: 2.7 }}
                />

                {/* Right side slow drip */}
                <motion.div
                  className="absolute top-0 right-0 w-1.5 rounded-b-md"
                  style={{
                    background: `linear-gradient(to bottom, ${waveColors.traditional[1]}, ${waveColors.traditional[0]})`,
                    right: 8,
                    zIndex: 5,
                  }}
                  initial={{ height: 0 }}
                  animate={isInView ? { height: 25 } : { height: 0 }}
                  transition={{
                    duration: 1.2,
                    ease: "easeOut",
                    repeat: isInView ? Infinity : 0,
                    repeatType: "loop",
                    repeatDelay: 0.7,
                    delay: 3,
                  }}
                />

                {/* Small droplet at the end of right drip */}
                <motion.div
                  className="absolute rounded-full"
                  style={{
                    background: waveColors.traditional[1],
                    width: 4,
                    height: 4,
                    right: 8,
                    boxShadow: "0px 1px 3px rgba(0,0,0,0.3)",
                  }}
                  initial={{
                    top: 25,
                    opacity: 0,
                    scale: 0.5,
                  }}
                  animate={
                    isInView
                      ? {
                          top: 80,
                          opacity: [0, 1, 1, 0],
                          scale: [0.5, 1, 1, 0.7],
                        }
                      : {
                          top: 25,
                          opacity: 0,
                          scale: 0.5,
                        }
                  }
                  transition={{
                    duration: 1.2,
                    repeat: isInView ? Infinity : 0,
                    repeatDelay: 0.7,
                    delay: 3,
                    times: [0, 0.1, 0.9, 1],
                    ease: "easeIn",
                  }}
                />

                {/* Splash droplets */}
                {[...Array(3)].map((_, i) => (
                  <motion.div
                    key={`right-drop-${i}`}
                    className="absolute rounded-full"
                    style={{
                      background: waveColors.traditional[1],
                      width: seededValue(i, 12) * 4 + 2,
                      height: seededValue(i, 13) * 4 + 2,
                      right: seededValue(i, 14) * 20,
                    }}
                    initial={{
                      top: 0,
                      opacity: 0.8,
                      scale: 1,
                    }}
                    animate={
                      isInView
                        ? {
                            top: seededValue(i, 15) * 40 + 20,
                            opacity: 0,
                            scale: 0.5,
                          }
                        : {
                            top: 0,
                            opacity: 0.8,
                            scale: 1,
                          }
                    }
                    transition={{
                      duration: seededValue(i, 16) * 0.5 + 0.5,
                      repeat: isInView ? Infinity : 0,
                      delay: seededValue(i, 17) * 0.5 + 2.7,
                      ease: "easeOut",
                    }}
                  />
                ))}
              </motion.div>

              {/* Front overflow - only small animated droplets falling down */}
              <motion.div
                className="absolute top-0 left-1/2 transform -translate-x-1/2"
                style={{
                  width: 60,
                  height: 120,
                  overflow: "visible",
                  zIndex: 10,
                }}
              >
                {/* Falling droplets */}
                {[...Array(8)].map((_, i) => (
                  <motion.div
                    key={`front-drop-${i}`}
                    className="absolute rounded-full"
                    style={{
                      background: waveColors.traditional[1],
                      width: seededValue(i, 18) * 4 + 2,
                      height: seededValue(i, 19) * 4 + 2,
                      left: seededValue(i, 20) * 40 + 10,
                      boxShadow: "0px 1px 3px rgba(0,0,0,0.3)",
                    }}
                    initial={{
                      top: -5,
                      opacity: 0.9,
                      scale: 1,
                    }}
                    animate={
                      isInView
                        ? {
                            top: 90,
                            opacity: 0,
                            scale: 0.7,
                          }
                        : {
                            top: -5,
                            opacity: 0.9,
                            scale: 1,
                          }
                    }
                    transition={{
                      duration: seededValue(i, 21) * 0.8 + 0.7,
                      repeat: isInView ? Infinity : 0,
                      delay: seededValue(i, 22) * 1.5 + 2.5,
                      ease: "easeIn",
                    }}
                  />
                ))}
              </motion.div>
            </motion.div>

            {/* Cost label */}
            <motion.div
              className="absolute top-2 left-0 right-0 text-center text-white text-sm font-medium"
              initial={{ opacity: 0 }}
              animate={isInView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ duration: 0.3, delay: 1.8 }}
            >
              Cost
            </motion.div>

            {/* Glass reflection effect - subtler for dark theme */}
            <div
              className="absolute inset-0 bg-gradient-to-br from-white to-transparent opacity-10"
              style={{ borderRadius: containerRadius - 1 }}
            />

            {/* Puddle beneath the overflowing container */}
            <motion.div
              className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 rounded-full"
              style={{
                background: waveColors.traditional[0],
                opacity: 0.5,
              }}
              initial={{ width: 0, height: 0 }}
              animate={
                isInView ? { width: 100, height: 12 } : { width: 0, height: 0 }
              }
              transition={{ duration: 1, ease: "easeOut", delay: 2.5 }}
            />
          </motion.div>

          <motion.div
            className="mt-3 text-sm font-medium text-gray-200"
            initial={{ opacity: 0 }}
            animate={isInView ? { opacity: 1 } : { opacity: 0 }}
            transition={{ duration: 0.5, delay: 1.1 }}
          >
            Traditional AI
          </motion.div>
        </div>

        {/* webAI Container */}
        <div className="flex flex-col items-center">
          <motion.div
            className="relative bg-background rounded-md overflow-hidden shadow-lg border border-border"
            style={{
              width: containerWidth,
              height: containerHeight,
              borderRadius: containerRadius,
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
            transition={{ duration: 1.6 }}
          >
            {/* Water fill animation for webAI - only 10% of the height */}
            <motion.div
              className="absolute bottom-0 w-full"
              style={{
                background: `linear-gradient(to top, ${waveColors.webAI[0]}, ${waveColors.webAI[1]})`,
              }}
              initial={{ height: 0 }}
              animate={isInView ? { height: "10%" } : { height: 0 }}
              transition={{ duration: 1, ease: "easeOut", delay: 0.8 }}
            >
              {/* Animated water surface */}
              <svg
                className="absolute top-0 left-0 w-full"
                style={{ transform: "translateY(-98%)" }}
                viewBox="0 0 1200 120"
                preserveAspectRatio="none"
              >
                <motion.path
                  d="M0,0 L1200,0 L1200,120 L0,120 Z"
                  fill={waveColors.webAI[1]}
                  initial={{
                    d: "M0,40 C300,0 900,80 1200,40 L1200,120 L0,120 Z",
                  }}
                  animate={{
                    d: [
                      "M0,40 C300,0 900,80 1200,40 L1200,120 L0,120 Z",
                      "M0,60 C300,20 900,100 1200,60 L1200,120 L0,120 Z",
                      "M0,40 C300,0 900,80 1200,40 L1200,120 L0,120 Z",
                    ],
                  }}
                  transition={{
                    repeat: Infinity,
                    duration: 2.5,
                    ease: "easeInOut",
                  }}
                />
              </svg>

              {/* Bubbles animation */}
              {[...Array(2)].map((_, index) => (
                <motion.div
                  key={`web-bubble-${index}`}
                  className="absolute rounded-full bg-white opacity-80"
                  style={{
                    width: seededValue(index, 23) * 4 + 2,
                    height: seededValue(index, 24) * 4 + 2,
                    left: `${seededValue(index, 25) * 80 + 10}%`,
                  }}
                  initial={{
                    bottom: 0,
                    opacity: 0.8,
                  }}
                  animate={{
                    bottom: "100%",
                    opacity: 0,
                  }}
                  transition={{
                    duration: seededValue(index, 26) + 1,
                    repeat: Infinity,
                    delay: seededValue(index, 27) + 7,
                    ease: "easeOut",
                  }}
                />
              ))}
            </motion.div>

            {/* Cost label */}
            <motion.div
              className="absolute bottom-0 left-0 right-0 text-center text-white text-sm font-medium mb-1"
              initial={{ opacity: 0 }}
              animate={isInView ? { opacity: 1 } : { opacity: 0 }}
              transition={{ duration: 0.3, delay: 1.3 }}
            >
              Cost
            </motion.div>

            {/* Glass reflection effect - subtler for dark theme */}
            <div
              className="absolute inset-0 bg-gradient-to-br from-white to-transparent opacity-10"
              style={{ borderRadius: containerRadius - 1 }}
            />
          </motion.div>

          <motion.div
            className="mt-3 text-sm font-medium text-gray-200"
            initial={{ opacity: 0 }}
            animate={isInView ? { opacity: 1 } : { opacity: 0 }}
            transition={{ duration: 0.5, delay: 1.1 }}
          >
            Web AI
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default FourthBentoAnimation;
