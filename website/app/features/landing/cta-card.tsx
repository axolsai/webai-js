"use client";

import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { ChevronRight, HeartHandshake } from "lucide-react";
import Link from "next/link";
import { ComponentPropsWithoutRef, useEffect } from "react";

// Self-contained Marquee component
interface MarqueeProps extends ComponentPropsWithoutRef<"div"> {
  className?: string;
  reverse?: boolean;
  pauseOnHover?: boolean;
  children: React.ReactNode;
  repeat?: number;
}

function Marquee({
  className,
  reverse = false,
  pauseOnHover = false,
  children,
  repeat = 4,
  ...props
}: MarqueeProps) {
  useEffect(() => {
    const styleId = "marquee-animations";
    if (!document.getElementById(styleId)) {
      const style = document.createElement("style");
      style.id = styleId;
      style.textContent = `
        @keyframes marquee-scroll {
          from {
            transform: translateX(0%);
          }
          to {
            transform: translateX(calc(-100% - 1rem));
          }
        }
        
        .marquee-item {
          animation: marquee-scroll 20s linear infinite;
        }
        
        .marquee-item-reverse {
          animation: marquee-scroll 20s linear infinite reverse;
        }
        
        .marquee-container:hover .marquee-item,
        .marquee-container:hover .marquee-item-reverse {
          animation-play-state: ${pauseOnHover ? "paused" : "running"};
        }
      `;
      document.head.appendChild(style);
    }
  }, [pauseOnHover]);

  return (
    <div
      {...props}
      className={cn(
        "marquee-container flex overflow-hidden p-2 gap-4",
        className,
      )}
    >
      {Array(repeat)
        .fill(0)
        .map((_, i) => (
          <div
            key={i}
            className={cn(
              "flex shrink-0 justify-around gap-4",
              reverse ? "marquee-item-reverse" : "marquee-item",
            )}
          >
            {children}
          </div>
        ))}
    </div>
  );
}

const reviews = [
  {
    name: "Sarah Chen",
    username: "@sarah_builds",
    body: "Honestly didn't think browser AI would be fast enough for our chat app, but WebAI proved me wrong!",
    img: "https://avatar.vercel.sh/sarah",
  },
  {
    name: "Mike Torres",
    username: "@miketorres",
    body: "Was burning $800/month on OpenAI calls. WebAI cut that to zero. Same quality, runs offline too.",
    img: "https://avatar.vercel.sh/mike",
  },
  {
    name: "Alex Kim",
    username: "@alexkimdev",
    body: "Deployed to 50k users without touching a single server. WebAI scales automatically with your user base.",
    img: "https://avatar.vercel.sh/alex",
  },
  {
    name: "James Wilson",
    username: "@jwilson_tech",
    body: "The WebAI model loads in 2 seconds and runs at 30 tokens/sec. Beats our old cloud setup by miles.",
    img: "https://avatar.vercel.sh/james",
  },
  {
    name: "Lisa Rodriguez",
    username: "@lisa_codes",
    body: "Built a medical app that works in remote areas with zero internet. WebAI + PWA = game changer.",
    img: "https://avatar.vercel.sh/lisa",
  },
  {
    name: "Tom Anderson",
    username: "@tom_builds",
    body: "Went from prototype to production in 3 days. No backend, no API keys, no headaches. Just works.",
    img: "https://avatar.vercel.sh/tom",
  },
  {
    name: "Maya Patel",
    username: "@maya_dev",
    body: "Client wanted AI features but couldn't send data to external APIs. WebAI solved it perfectly.",
    img: "https://avatar.vercel.sh/maya",
  },
  {
    name: "Chris Lee",
    username: "@chrislee_dev",
    body: "The documentation is actually good and the examples work out of the box. Rare in the AI space tbh.",
    img: "https://avatar.vercel.sh/chris",
  },
  {
    name: "Rachel Green",
    username: "@rachelgreen",
    body: "My users love that their data stays private. I love that my hosting costs stayed the same with 10x traffic.",
    img: "https://avatar.vercel.sh/rachel",
  },
];

const firstRow = reviews.slice(0, Math.ceil(reviews.length / 2));
const secondRow = reviews.slice(Math.ceil(reviews.length / 2));

const ReviewCard = ({
  img,
  name,
  username,
  body,
}: {
  img: string;
  name: string;
  username: string;
  body: string;
}) => {
  return (
    <figure
      className={cn(
        "relative w-64 cursor-pointer overflow-hidden rounded-[2rem] border p-4",
        "border-gray-950/[.1] bg-gray-950/[.01] hover:bg-gray-950/[.05]",
        "dark:border-gray-50/[.1] dark:bg-gray-50/[.10] dark:hover:bg-gray-50/[.15]",
      )}
    >
      <div className="flex flex-row items-center gap-2">
        <img className="rounded-full" width="32" height="32" alt="" src={img} />
        <div className="flex flex-col">
          <figcaption className="text-sm font-medium dark:text-white">
            {name}
          </figcaption>
          <p className="text-xs font-medium dark:text-white/40">{username}</p>
        </div>
      </div>
      <blockquote className="mt-2 text-sm">{body}</blockquote>
    </figure>
  );
};

const containerVariants = {
  hidden: { opacity: 0, y: 50 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.8,
      ease: "easeOut",
      staggerChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: "easeOut",
    },
  },
};

const marqueeVariants = {
  hidden: { opacity: 0, scale: 0.8, rotate: 35 },
  visible: {
    opacity: 1,
    scale: 1,
    rotate: 35,
    transition: {
      duration: 1,
      ease: "easeOut",
      delay: 0.3,
    },
  },
};

const iconVariants = {
  hidden: { opacity: 0, scale: 0, rotate: -180 },
  visible: {
    opacity: 1,
    scale: 1,
    rotate: 0,
    transition: {
      duration: 0.8,
      ease: [0.175, 0.885, 0.32, 1.275],
      delay: 0.5,
    },
  },
};

interface CallToActionProps {
  enterAnimation?: boolean;
}

export function CallToAction({ enterAnimation = true }: CallToActionProps) {
  const containerMotionProps = enterAnimation
    ? {
        initial: "hidden",
        whileInView: "visible",
        viewport: {
          once: true,
          amount: 0.3,
          margin: "-100px 0px -100px 0px",
        },
        variants: containerVariants,
      }
    : {};

  const itemMotionProps = enterAnimation ? { variants: itemVariants } : {};

  const marqueeMotionProps = enterAnimation
    ? { variants: marqueeVariants }
    : {};

  const iconMotionProps = enterAnimation ? { variants: iconVariants } : {};

  return (
    <section id="cta">
      <div className="py-14">
        <div className="container flex w-full flex-col items-center justify-center">
          <motion.div
            className="relative flex w-full max-w-[1100px] flex-col items-center justify-center overflow-hidden rounded-[2rem] border p-10 py-14"
            {...containerMotionProps}
          >
            <motion.div className="absolute" {...marqueeMotionProps}>
              <Marquee pauseOnHover repeat={3}>
                {firstRow.map((review) => (
                  <ReviewCard key={review.username} {...review} />
                ))}
              </Marquee>
              <Marquee reverse pauseOnHover repeat={3}>
                {secondRow.map((review) => (
                  <ReviewCard key={review.username} {...review} />
                ))}
              </Marquee>
              <Marquee pauseOnHover repeat={3}>
                {firstRow.map((review) => (
                  <ReviewCard key={review.username} {...review} />
                ))}
              </Marquee>
              <Marquee reverse pauseOnHover repeat={3}>
                {secondRow.map((review) => (
                  <ReviewCard key={review.username} {...review} />
                ))}
              </Marquee>
              <Marquee pauseOnHover repeat={3}>
                {firstRow.map((review) => (
                  <ReviewCard key={review.username} {...review} />
                ))}
              </Marquee>
              <Marquee reverse pauseOnHover repeat={3}>
                {secondRow.map((review) => (
                  <ReviewCard key={review.username} {...review} />
                ))}
              </Marquee>
            </motion.div>

            <motion.div
              className="z-10 mx-auto size-24 rounded-[2rem] border bg-white/10 p-3 shadow-2xl backdrop-blur-md dark:bg-black/10 lg:size-32"
              {...iconMotionProps}
            >
              <HeartHandshake className="mx-auto size-16 text-black dark:text-white lg:size-24" />
            </motion.div>

            <motion.div
              className="z-10 mt-4 flex flex-col items-center text-center text-black dark:text-white"
              {...itemMotionProps}
            >
              <motion.h1
                className="text-3xl font-bold lg:text-4xl"
                {...itemMotionProps}
              >
                Start Building with{" "}
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-[#f43e5c] to-[#ff758c] px-1">
                  WebAI.js
                </span>{" "}
                Today
              </motion.h1>
              <motion.p className="mt-2 max-w-2xl" {...itemMotionProps}>
                Join the growing community of developers using Web AI to create
                faster, more private AI applications at a fraction of the cost.
              </motion.p>
              <div className="flex gap-x-4">
                {" "}
                <motion.div {...itemMotionProps}>
                  <Link
                    href="https://www.webai-js.com/models"
                    className={cn(
                      buttonVariants({
                        size: "lg",
                        variant: "default",
                      }),
                      "group mt-4 rounded-[2rem] px-6",
                    )}
                  >
                    Get Started
                    <ChevronRight className="ml-1 size-4 transition-all duration-300 ease-out group-hover:translate-x-1" />
                  </Link>
                </motion.div>
                <motion.div {...itemMotionProps}>
                  <Link
                    href="https://demo.axolsai.com"
                    target="_blank"
                    className={cn(
                      buttonVariants({
                        size: "lg",
                        variant: "outline",
                      }),
                      "group mt-4 rounded-[2rem] px-6 ",
                    )}
                  >
                    View Demo
                    <ChevronRight className="ml-1 size-4 transition-all duration-300 ease-out group-hover:translate-x-1" />
                  </Link>
                </motion.div>
              </div>
            </motion.div>

            <div className="absolute inset-x-0 bottom-0 h-full bg-gradient-to-b from-transparent to-white to-70% dark:to-black rounded-b-[2rem]" />
          </motion.div>
        </div>
      </div>
    </section>
  );
}
