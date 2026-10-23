import React from "react";

import { cn } from "@/lib/utils";

export function RainbowContainer({
  children,
  className,
  ...props
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "h-full w-full group relative animate-rainbow border-0 pb-8 center transition-colors overflow-hidden ",

        // before styles
        "before:absolute before:top-[108%] before:z-0 before:h-[20%] before:w-full  before:animate-rainbow before:bg-[linear-gradient(90deg,hsl(var(--color-1)),hsl(var(--color-5)),hsl(var(--color-3)),hsl(var(--color-4)),hsl(var(--color-2)))] before:bg-[length:200%] before:[filter:blur(calc(6rem))]",

        // light mode colors
        "bg-[rgba(18,18,19,0)),linear-gradient(90deg,hsl(var(--color-1)),hsl(var(--color-5)),hsl(var(--color-3)),hsl(var(--color-4)),hsl(var(--color-2)))]",

        // dark mode colors
        "dark:bg-[rgba(0,0,0,0)),linear-gradient(90deg,hsl(var(--color-1)),hsl(var(--color-5)),hsl(var(--color-3)),hsl(var(--color-4)),hsl(var(--color-2)))]",
        "dark:bg-background",

        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
