import { RainbowContainer } from "@/components/rainbow-container";
import Link from "next/link";
import { FaXTwitter } from "react-icons/fa6";
import { BsLinkedin } from "react-icons/bs";
import Image from "next/image";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer
      className="w-full border-t border-border overflow-x-hidden bg-background/50"
      role="contentinfo"
      aria-label="Site footer"
    >
      <RainbowContainer className="h-full pb-0 before:top-[100%]  before:h-[40%]">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 md:py-1">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
            {/* Company Information Section */}
            <div className="flex flex-col sm:flex-row items-center gap-4 text-sm">
              <div className="flex items-center gap-3">
                <Link
                  href="/"
                  className="flex items-center gap-2 hover:opacity-90 transition-opacity"
                  aria-label="Return to Hubters WebaI homepage"
                >
                  <div className="relative w-5 h-5 flex-shrink-0">
                    <Image
                      src="/hubters-webai-logo.svg"
                      alt="Hubters WebAI Logo"
                      fill
                      priority
                      className="object-contain"
                    />
                  </div>
                  <span className="font-bold text-foreground text-sm">
                    Axols <span className="text-primary">WebAI.js</span>
                  </span>
                </Link>
              </div>

              <div className="flex flex-col sm:flex-row items-center gap-4 sm:gap-6">
                <p className="text-xs text-muted-foreground">
                  <span aria-label={`Copyright ${currentYear}`}>
                    © {currentYear}
                  </span>{" "}
                  HubtersAI LLC, All rights reserved.
                </p>

                <nav aria-label="Footer navigation" className="flex gap-6">
                  <Link
                    href="https://www.webai-js.com/terms-of-service"
                    className="text-xs hover:underline hover:text-primary transition-colors focus:outline-none"
                  >
                    Terms of Service
                  </Link>
                  <Link
                    href="https://www.webai-js.com/privacy-policy"
                    className="text-xs hover:underline hover:text-primary transition-colors focus:outline-none"
                  >
                    Privacy Policy
                  </Link>

                  <Link
                    href="https://www.webai-js.com/ethical-ai-terms"
                    className="text-xs hover:underline hover:text-primary transition-colors focus:outline-none"
                  >
                    Ethical AI
                  </Link>
                  <Link
                    href="https://demo.axolsai.com"
                    className="text-xs hover:underline hover:text-primary transition-colors focus:outline-none"
                  >
                    Web AI Demos
                  </Link>
                  <Link
                    href="https://localai.tools"
                    className="text-xs hover:underline hover:text-primary transition-colors focus:outline-none"
                  >
                    Local AI Tools
                  </Link>
                  <Link
                    href="https://www.webai-js.com/contact-us"
                    className="text-xs hover:underline hover:text-primary transition-colors focus:outline-none"
                  >
                    Contact Us
                  </Link>
                </nav>
              </div>
            </div>

            {/* Social Links Section */}
            <div
              className="flex items-center gap-1"
              aria-label="Social media links"
            >
              <Link
                href="https://x.com/AxolsAI"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Follow Hubters WebAI on X (formerly Twitter)"
                className="hover:text-primary transition-colors p-2 rounded-full hover:bg-muted "
              >
                <FaXTwitter className="w-4 h-4" aria-hidden="true" />
              </Link>
              <Link
                href="https://www.linkedin.com/company/hubters-webai"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Follow Hubters WebAI on Linkedin"
                className="hover:text-primary transition-colors p-2 rounded-full hover:bg-muted "
              >
                <BsLinkedin className="w-4 h-4" aria-hidden="true" />
              </Link>
            </div>
          </div>
        </div>
      </RainbowContainer>
    </footer>
  );
}
