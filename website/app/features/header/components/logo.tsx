import React from "react";
import Link from "next/link";
import Image from "next/image";
import { Archivo } from "next/font/google";

const archivo = Archivo({
  subsets: ["latin"],
  weight: ["700"],
});

const Logo = () => {
  return (
    <Link
      href="/"
      className="inline-flex items-center hover:opacity-90 transition-opacity"
      aria-label="Hubters WebAI Home"
    >
      <div className="flex items-center gap-2">
        <div className="relative w-[36px] h-6 -mr-1">
          <Image
            src="/hubters-webai-logo.svg"
            alt=""
            fill
            priority
            className="object-contain"
          />
          {/* <Image
            src="/hubters-webai-logo.svg"
            alt=""
            fill
            priority
            className="object-contain w-5 h-5"
          /> */}
        </div>

        <div className="flex items-baseline gap-1">
          <span
            className={`text-sm md:text-lg leading-none ${archivo.className}`}
          >
            Axols
          </span>
          <span
            className={`text-sm md:text-lg leading-none text-rose-500 ${archivo.className} font-extrabold`}
          >
            WebAI.js
          </span>
        </div>
      </div>
    </Link>
  );
};

export default Logo;
