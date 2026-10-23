"use client";

import { useState, useEffect } from "react";
import Logo from "./components/logo";
import NavBar from "./components/nav";
// import { ThemeToggle } from "./components/theme-toggle";
import { FaGithub, FaNpm } from "react-icons/fa6";
import { ArrowUpRight } from "lucide-react";
import Link from "next/link";

export default function Header() {
  const [isScrolled, setIsScrolled] = useState(false);

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={`sticky top-0 z-50 transition-shadow w-full border-b-[0.8px] flex justify-center ${
        isScrolled ? "shadow-sm" : ""
      }`}
    >
      <div className="bg-background/60 backdrop-blur w-full">
        <section className="relative w-full px-4 md:px-6 max-h-[52px] min-h-[52px] py-2 flex items-center justify-between ">
          <div className="center">
            <Logo />
          </div>
          <div className="center gap-x-4 md:gap-x-8">
            <NavBar />
            <div className="gap-x-5 flex items-center justify-center">
              <Link
                href="https://www.npmjs.com/package/@axols/webai-js"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View @axols/webai-js on npm"
                className="flex items-center hover:text-gray-400"
              >
                <FaNpm size={28} />
              </Link>
              <Link
                href="https://github.com/axolsai/webai-js"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View WebAI.js on GitHub"
                className="flex items-center gap-2 hover:text-gray-400"
              >
                <FaGithub className="" size={22} />
                {/* <span className="text-sm">12.5K</span> */}
              </Link>
            </div>

            <Link
              href="/models"
              className="group inline-flex items-center gap-1.5 rounded-full border border-rose-300/25 bg-gradient-to-r from-primary to-rose-500 px-4 py-2 text-sm font-semibold text-white shadow-[0_6px_22px_rgba(244,63,94,0.24)] transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_9px_28px_rgba(244,63,94,0.36)]"
            >
              Models
              <ArrowUpRight className="h-3.5 w-3.5 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </Link>

            {/* <ThemeToggle /> */}

          </div>
        </section>
      </div>
    </header>
  );
}
