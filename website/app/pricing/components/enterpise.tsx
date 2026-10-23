"use client";
import React from "react";
import { Building2, ChevronRight } from "lucide-react";
import { motion } from "framer-motion";

const EnterpriseSection = () => {
  return (
    <motion.section
      className="mb-24 pt-16"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay: 0.2 }}
    >
      <div className="max-w-5xl mx-auto px-4">
        <motion.div
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-3xl md:text=-4xl font-bold mb-6 bg-gradient-to-r from-white to-zinc-400 bg-clip-text text-transparent">
            Need a custom Web AI solution?
          </h2>
          <p className="text-center text-lg max-w-2xl mx-auto text-zinc-400">
            Get tailored Web AI experiences optimized for your specific use case
          </p>
        </motion.div>

        <div className="relative">
          {/* Gradient background effect */}
          <div className="absolute -inset-4 bg-gradient-to-r from-[#ff758c]/25 via-[#f43e5c]/20 to-orange-500/10 rounded-3xl blur-xl" />

          <div className="relative w-full rounded-2xl bg-gradient-to-b from-zinc-900/90 to-zinc-950/90 border border-[#f43e5c]/20 backdrop-blur-sm p-8 overflow-hidden">
            {/* Floating particles effect */}
            <div className="absolute right-0 top-0 w-40 h-40 bg-[#ff758c]/5 rounded-full blur-3xl -mr-10 -mt-10" />
            <div className="absolute left-0 bottom-0 w-32 h-32 bg-[#f43e5c]/5 rounded-full blur-3xl -ml-10 -mb-10" />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Feature Cards */}
              {[
                {
                  id: "01",
                  title: "Model Conversion",
                  description:
                    "Convert your own AI models into browser runnable format customized to your requirements, and fully integrated with our toolkit.",
                  delay: 0.3,
                  color: "from-[#ff758c] to-[#f43e5c]",
                },
                {
                  id: "02",
                  title: "Advanced Security",
                  description:
                    "Deploy with enhanced security measures including end-to-end  model encryption and private model hosting and distribution channels.",
                  delay: 0.4,
                  color: "from-[#f43e5c] to-orange-500",
                },
                {
                  id: "03",
                  title: "Dedicated Support",
                  description:
                    "Receive priority technical support with a dedicated ticketing system tailored to your organization.",
                  delay: 0.5,
                  color: "from-orange-400 to-[#f43e5c]",
                },
              ].map((feature, index) => (
                <motion.div
                  key={index}
                  className="relative rounded-xl bg-gradient-to-r from-[#ff758c]/10 via-[#f43e5c]/5 to-orange-500/5 border border-[#f43e5c]/20 p-6 hover:border-[#f43e5c]/40 hover:from-[#ff758c]/20 hover:via-[#f43e5c]/15 hover:to-orange-500/10 transition-all duration-300 group"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: feature.delay }}
                >
                  <div className="absolute -top-3 left-4 px-3 py-1 bg-zinc-900 rounded-md border border-[#f43e5c]/30">
                    <span
                      className={`text-sm font-medium bg-gradient-to-r ${feature.color} bg-clip-text text-transparent`}
                    >
                      {feature.id}
                    </span>
                  </div>
                  <div className="pt-4">
                    <h3 className="text-xl font-semibold text-white mb-3">
                      {feature.title}
                    </h3>
                    <p className="text-zinc-400 text-sm mb-4 group-hover:text-zinc-300 transition-colors">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="mt-10 flex justify-center">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.6 }}
              >
                <a
                  href="https://www.webai-js.com/enterprise"
                  className="flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-[#ff758c] via-[#f43e5c] to-orange-500 border  border-[#f43e5c]/30 px-6 py-4 text-white text-base font-bold hover:from-[#ff758c]/80 hover:via-[#f43e5c]/80 hover:to-orange-500/80 hover:border-[#f43e5c]/40  transition-all duration-300 group shadow-lg hover:shadow-[#f43e5c]/10"
                >
                  <Building2 className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span>Explore Enterprise Solutions</span>
                  <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                </a>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </motion.section>
  );
};

export default EnterpriseSection;
