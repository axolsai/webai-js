import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Header from "./features/header/header";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "WebAI.js - Run AI Locally in Browser",
  description: "Run AI models locally in the browser with WebAI.js.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} min-h-screen bg-background antialiased flex flex-col items-center overflow-x-hidden hide-scroll-bar max-w-[100vw] hide-scrollbar`}>
        <Header />
        <div className="w-full">{children}</div>
      </body>
    </html>
  );
}
