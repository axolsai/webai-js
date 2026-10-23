import type { Metadata } from "next";
import ModelsCatalog from "./models-catalog";
import models from "./models-data.json";

export const metadata: Metadata = {
  title: "Models | WebAI.js",
  description: "Explore benchmarked WebAI.js v4 workers for local browser inference.",
};

export default function ModelsPage() {
  return <ModelsCatalog models={models} />;
}
