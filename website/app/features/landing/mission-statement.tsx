"use client";

import React from "react";
import { motion, useInView } from "framer-motion";
import { Quote } from "lucide-react";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";

const MissionStatement: React.FC = () => {
  const ref = React.useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section ref={ref} className="py-8 md:py-12 mb-16">
      <div className="container px-4 md:px-6">
        <Card className="border-none shadow-none bg-background/50">
          <CardContent className="p-4 md:p-8 flex flex-col items-center text-center space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="flex flex-col items-center"
            >
              {/* Avatar */}
              <motion.div
                className="mb-4"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={
                  isInView
                    ? { opacity: 1, scale: 1 }
                    : { opacity: 0, scale: 0.8 }
                }
                transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
              >
                <Avatar
                  className="w-20 h-20 md:w-24 md:h-24 border-2 border-background shadow-sm hover:shadow-md cursor-pointer transition-transform transform hover:scale-105"
                  onClick={() => {
                    window.open(
                      "https://www.linkedin.com/in/peng-zhang-55a653113",
                      "_blank",
                    );
                  }}
                >
                  <AvatarImage
                    src="/peng-profile-pic.webp"
                    alt="Company Founder"
                    className="object-contain scale-[1.35]"
                  />
                  <AvatarFallback className="text-lg">PZ</AvatarFallback>
                </Avatar>
              </motion.div>

              {/* Name */}
              <motion.h3
                className="text-xl md:text-2xl font-bold text-foreground mb-1"
                initial={{ opacity: 0, y: 20 }}
                animate={
                  isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }
                }
                transition={{ duration: 0.6, delay: 0.4, ease: "easeOut" }}
              >
                Peng Zhang
              </motion.h3>

              {/* Job Title */}
              <motion.p
                className="text-sm md:text-base text-muted-foreground mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={
                  isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }
                }
                transition={{ duration: 0.6, delay: 0.5, ease: "easeOut" }}
              >
                Founder & Developer of Axols WebAI.js
              </motion.p>

              {/* Mission Statement (Quote Style) - with Lucide icons */}
              <motion.div
                className="relative max-w-[1050px]"
                initial={{ opacity: 0, y: 30 }}
                animate={
                  isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }
                }
                transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
              >
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={
                    isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: -20 }
                  }
                  transition={{ duration: 0.6, delay: 0.8, ease: "easeOut" }}
                >
                  <Quote className="w-8 h-8 md:w-10 md:h-10 text-primary absolute -top-6 -left-2 md:-left-4 rotate-180" />
                </motion.div>

                <motion.p
                  className="text-lg md:text-xl font-medium text-foreground/90 leading-relaxed px-6 md:px-8 italic"
                  initial={{ opacity: 0, y: 20 }}
                  animate={
                    isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }
                  }
                  transition={{ duration: 0.8, delay: 0.7, ease: "easeOut" }}
                >
                  As AI models get smaller and edge devices grow more powerful,
                  we&apos;re leading the Web AI ecosystem by making intelligence
                  accessible in the browser—fast, private, and affordable for
                  everyone.
                </motion.p>

                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={
                    isInView ? { opacity: 1, x: 0 } : { opacity: 0, x: 20 }
                  }
                  transition={{ duration: 0.6, delay: 0.9, ease: "easeOut" }}
                >
                  <Quote className="w-8 h-8 md:w-10 md:h-10 text-primary absolute -bottom-6 -right-2 md:-right-4" />
                </motion.div>
              </motion.div>

              {/* Decorative Element */}
              <motion.div
                className="h-0.5 w-16 bg-primary mx-auto rounded-full mt-10"
                initial={{ opacity: 0, scaleX: 0 }}
                animate={
                  isInView
                    ? { opacity: 1, scaleX: 1 }
                    : { opacity: 0, scaleX: 0 }
                }
                transition={{ duration: 0.8, delay: 1.0, ease: "easeOut" }}
              ></motion.div>
            </motion.div>
          </CardContent>
        </Card>
      </div>
    </section>
  );
};

export default MissionStatement;
