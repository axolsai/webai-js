"use client";

import FirstBentoAnimation from "@/components/bento-animation-first";
import FourthBentoAnimation from "@/components/bento-animation-fourth";
import { SecondBentoAnimation } from "@/components/bento-animation-second";
import { ThirdBentoAnimation } from "@/components/bento-animation-third";


export default function LandingFeaturesClient() {
  const bentoConfig = {
    title: "Absolutely Secure, Scalable and Affordable",
    description:
      "Ask your AI Agent for real-time collaboration, seamless integrations, and actionable insights to streamline your operations.",
    items: [
      {
        id: 1,
        content: <FirstBentoAnimation />,
        title: "Real-time AI Collaboration",
        description:
          "Experience real-time assistance. Ask your AI Agent to coordinate tasks, answer questions, and maintain team alignment.",
      },
      {
        id: 2,
        content: <SecondBentoAnimation />,
        title: "Seamless Integrations",
        description:
          "Unite your favorite tools for effortless connectivity. Boost productivity through interconnected workflows.",
      },
      {
        id: 3,
        content: (
          <ThirdBentoAnimation
            data={[20, 30, 25, 45, 40, 55, 75]}
            toolTipValues={[
              1234, 1678, 2101, 2534, 2967, 3400, 3833, 4266, 4700, 5133,
            ]}
          />
        ),
        title: "Instant Insight Reporting",
        description:
          "Transform raw data into clear insights in seconds. Empower smarter decisions with real-time, always-learning intelligence.",
      },
      {
        id: 4,
        content: <FourthBentoAnimation />,
        title: "Smart Automation",
        description:
          "Set it, forget it. Your AI Agent tackles repetitive tasks so you can focus on strategy, innovation, and growth.",
      },
    ],
  };
  const { items, title } = bentoConfig;

  return (
    <section className="flex flex-col items-center justify-center w-full relative px-5 md:px-10 my-16 p-16">
      <div className="w-full text-center mb-24">
        <h3 className="text-4xl md:text-4xl font-bold text-[#D3D3D3]">
          {title}
        </h3>
      </div>
      <div className=" mx-5 md:mx-10 relative">
        <div className="absolute top-0 -left-4 md:-left-14 h-full w-4 md:w-14 text-primary/5 bg-[size:10px_10px] [background-image:repeating-linear-gradient(315deg,currentColor_0_1px,#0000_0_50%)]"></div>
        <div className="absolute top-0 -right-4 md:-right-14 h-full w-4 md:w-14 text-primary/5 bg-[size:10px_10px] [background-image:repeating-linear-gradient(315deg,currentColor_0_1px,#0000_0_50%)]"></div>

        <div className="grid grid-cols-1 md:grid-cols-2 overflow-hidden">
          {items.map((item) => (
            <div
              key={item.id}
              className="flex flex-col items-start justify-end min-h-[600px] md:min-h-[500px] p-0.5 relative before:absolute before:-left-0.5 before:top-0 before:z-10 before:h-screen before:w-px before:bg-border before:content-[''] after:absolute after:-top-0.5 after:left-0 after:z-10 after:h-px after:w-screen after:bg-border after:content-[''] group cursor-pointer max-h-[400px] group"
            >
              <div className="relative flex size-full items-center justify-center h-full overflow-hidden">
                {item.content}
              </div>
              <div className="flex-1 flex-col gap-2 p-6">
                <h3 className="text-lg tracking-tighter font-semibold">
                  {item.title}
                </h3>
                <p className="text-muted-foreground">{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
