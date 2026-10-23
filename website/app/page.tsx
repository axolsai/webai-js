import { Footer } from "./features/footer/footer";
import LandingHero from "./features/landing/hero";
import { Particles } from "@/components/magicui/particles";
import HorizonSeparator from "./features/landing/horizon-separator";
import ThreeStepsIntegration from "./features/landing/3-steps-on-device-ai";
import LandingFeatures from "./features/landing/features";
import MissionStatement from "./features/landing/mission-statement";
import { CallToAction } from "./features/landing/cta-card";
import SocialProof from "./features/landing/social-proof";

export default function Home() {
  return (
    <>
      {/* Fixed particles background */}
      <div className="hidden sm:block">
        <Particles
          className="fixed inset-0 -z-10"
          quantity={150}
          ease={80}
          size={0.1}
          staticity={30}
        />
      </div>
      <div className=" sm:hidden">
        <Particles
          className="fixed inset-0 -z-10"
          quantity={80}
          ease={80}
          size={0.08}
          staticity={30}
        />
      </div>
      {/* Scrollable content */}
      <div className="relative z-10 flex flex-col w-full mx-auto min-h-screen overflow-auto">
        <main className="flex flex-col container mx-auto mb-12 pt-4">
          <LandingHero />
          <SocialProof />
          <div className="mb-4">
            {" "}
            <HorizonSeparator />
          </div>
          <ThreeStepsIntegration />
          <LandingFeatures />
          <MissionStatement />
          <CallToAction />
        </main>

        <div className="w-full mt-auto">
          <Footer />
        </div>
      </div>
    </>
  );
}
