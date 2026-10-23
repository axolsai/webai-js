"use client";
import { useState } from "react";
import { Clipboard, Check } from "lucide-react";
import { Highlight } from "prism-react-renderer";
import { codeBlockTheme } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface SimpleCodeBlockProps {
  code: string;
  language: string;
  className?: string;
  copiable?: boolean;
}

const SimpleCodeBlock: React.FC<SimpleCodeBlockProps> = ({
  code,
  language,
  className,
  copiable = true,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  };

  return (
    <div className="relative w-full h-full">
      {copiable && (
        <button
          onClick={handleCopy}
          className="absolute top-1 right-2 h-8 w-8 p-0 z-20 text-muted-foreground bg-muted/30 border-none rounded-md hover:bg-muted/50 transition-colors cursor-pointer flex items-center justify-center"
          aria-label="Copy code to clipboard"
        >
          {copied ? (
            <Check className="h-4 w-4 text-green-500" />
          ) : (
            <Clipboard className="h-4 w-4" />
          )}
        </button>
      )}
      <div
        className={cn(
          "rounded-md border border-border w-full max-w-full overflow-auto",
          className,
        )}
      >
        <Highlight
          theme={codeBlockTheme}
          code={code}
          language={language as any}
        >
          {({ style, tokens, getLineProps, getTokenProps }) => (
            <pre
              className="p-3 overflow-auto rounded-md text-xs bg-transparent"
              style={{
                ...style,
                backgroundColor: undefined, // Remove inline style background
              }}
            >
              {tokens.map((line, i) => {
                // Get the props without passing key in the argument
                const linePropsWithoutKey = getLineProps({ line });

                // Don't use spread for the line
                return (
                  <div
                    key={i}
                    className={linePropsWithoutKey.className}
                    style={linePropsWithoutKey.style}
                  >
                    <span className="text-muted-foreground inline-block w-6 select-none text-xs">
                      {i + 1}
                    </span>

                    {line.map((token, j) => {
                      // Get the props without passing key in the argument
                      const tokenPropsWithoutKey = getTokenProps({ token });

                      // Don't use spread for the token either
                      return (
                        <span
                          key={j}
                          className={tokenPropsWithoutKey.className}
                          style={tokenPropsWithoutKey.style}
                        >
                          {tokenPropsWithoutKey.children}
                        </span>
                      );
                    })}
                  </div>
                );
              })}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
};

export default SimpleCodeBlock;
