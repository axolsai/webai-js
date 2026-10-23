"use client";

import { cn } from "@/lib/utils";
import dynamic from "next/dynamic";
import { memo, useId, useEffect, useState } from "react";
import type { Components } from "react-markdown";
import { CodeBlock, CodeBlockCode } from "./code-block";

// Dynamic imports with no SSR
const ReactMarkdown = dynamic(() => import("react-markdown"), {
  ssr: false,
});

export type MarkdownProps = {
  children: string;
  id?: string;
  className?: string;
  components?: Partial<Components>;
};

function extractLanguage(className?: string): string {
  if (!className) return "plaintext";
  const match = className.match(/language-(\w+)/);
  return match ? match[1] : "plaintext";
}

const INITIAL_COMPONENTS: Partial<Components> = {
  code: function CodeComponent({ className, children, ...props }: any) {
    const isInline =
      !props.node?.position?.start.line ||
      props.node?.position?.start.line === props.node?.position?.end.line;

    if (isInline) {
      return (
        <span
          className={cn(
            "bg-primary-foreground rounded-sm px-1 font-mono text-sm",
            className,
          )}
          {...props}
        >
          {children}
        </span>
      );
    }

    const language = extractLanguage(className);

    return (
      <CodeBlock className={className}>
        <CodeBlockCode code={children as string} language={language} />
      </CodeBlock>
    );
  },
  pre: function PreComponent({ children }: any) {
    return <>{children}</>;
  },
};

function MarkdownComponent({
  children,
  id,
  className,
  components = INITIAL_COMPONENTS,
}: MarkdownProps) {
  const generatedId = useId();
  const blockId = id ?? generatedId;
  const [remarkGfm, setRemarkGfm] = useState<any>(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    import("remark-gfm").then((mod) => setRemarkGfm(() => mod.default));
  }, []);

  // Don't render until client-side
  if (!isClient) {
    return <div className={className}>Loading...</div>;
  }

  return (
    <div className={className}>
      <ReactMarkdown
        key={blockId}
        remarkPlugins={remarkGfm ? [remarkGfm] : []}
        components={components}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

const Markdown = memo(MarkdownComponent);
Markdown.displayName = "Markdown";

export { Markdown };
