import { motion, AnimatePresence } from "motion/react";
import { useState } from "react";
import { Copy, Check } from "lucide-react";

export default function CopyButton({ text = "npm install hubters-webai" }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.button
      onClick={copyToClipboard}
      className={`ml-2 p-1 rounded-md ${
        copied
          ? "bg-green-900/40 text-green-400"
          : "bg-neutral-800/60 text-neutral-400 hover:bg-neutral-700/60 hover:text-neutral-300"
      } flex items-center justify-center relative`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <AnimatePresence mode="wait">
        {copied ? (
          <motion.div
            key="check"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <Check size={14} />
          </motion.div>
        ) : (
          <motion.div
            key="copy"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <Copy size={14} />
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {copied && (
          <motion.div
            className="absolute inset-0 bg-green-500/5 pointer-events-none rounded-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          />
        )}
      </AnimatePresence>
    </motion.button>
  );
}
