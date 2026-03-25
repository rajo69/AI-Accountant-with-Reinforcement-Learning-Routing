"use client";

import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import {
  BrainCircuit, Tag, GitCompare, FileText,
  ArrowRight, CheckCircle2, ChevronRight,
} from "lucide-react";
import GradientText from "@/components/gradient-text";
import FluidGlassButton from "@/components/fluid-glass-button";

const Aurora = dynamic(() => import("@/components/aurora"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Data ──────────────────────────────────────────────────────────────────────

const features = [
  {
    icon: Tag,
    title: "Transaction Categorisation",
    description: "AI assigns account codes automatically. Confidence-scored and explainable, every decision has a reason you can read.",
    accent: "text-emerald-400",
    hoverBorder: "hover:border-emerald-500/25",
    hoverGlow: "group-hover:bg-emerald-500/[0.04]",
  },
  {
    icon: GitCompare,
    title: "Bank Reconciliation",
    description: "Fuzzy-matching on amount, date, and description pairs statement lines to transactions in seconds.",
    accent: "text-violet-400",
    hoverBorder: "hover:border-violet-500/25",
    hoverGlow: "group-hover:bg-violet-500/[0.04]",
  },
  {
    icon: FileText,
    title: "Management Letters",
    description: "Professional PDF reports with AI-written narrative, generated from your actual figures, not lorem ipsum.",
    accent: "text-sky-400",
    hoverBorder: "hover:border-sky-500/25",
    hoverGlow: "group-hover:bg-sky-500/[0.04]",
  },
];

const steps = [
  { label: "Connect Xero",              detail: "OAuth2 in one click, no credentials stored" },
  { label: "AI processes transactions", detail: "Claude categorises with your chart-of-accounts context" },
  { label: "Review & correct",          detail: "Approve, reject, or fix suggestions in the UI" },
  { label: "Download reports",          detail: "Management letters ready to send to clients" },
];

const stats = [
  { value: "94%",    label: "Categorisation accuracy" },
  { value: "< 2s",   label: "Per transaction" },
  { value: "3+ hrs", label: "Saved per week" },
];

// ── Animations ────────────────────────────────────────────────────────────────

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.55, ease: "easeOut" as const } },
};

// ── Page ──────────────────────────────────────────────────────────────────────

export default function LandingPage() {
  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">

      {/* ── Aurora background ─────────────────────────────────────────────── */}
      <div className="absolute inset-x-0 top-0 h-[680px] pointer-events-none">
        <Aurora
          colorStops={["#4338CA", "#7C3AED", "#0EA5E9"]}
          amplitude={1.1}
          blend={0.55}
          speed={0.45}
        />
        {/* Fade to black at the bottom */}
        <div className="absolute inset-x-0 bottom-0 h-56 bg-gradient-to-b from-transparent to-black" />
      </div>

      {/* Dot grid overlay: very subtle texture */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.18]"
        style={{
          backgroundImage: "radial-gradient(rgba(255,255,255,0.15) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* ── Nav ───────────────────────────────────────────────────────────── */}
      <nav className="relative z-10 max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-[5px] bg-indigo-500 flex items-center justify-center shadow-md shadow-indigo-500/40">
            <BrainCircuit className="w-3.5 h-3.5 text-white" strokeWidth={2} />
          </div>
          <span className="text-sm font-semibold tracking-tight">AI Accountant</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-white/40 hidden sm:block">For UK accounting firms</span>
          <a href="/privacy" className="text-xs text-white/40 hover:text-white/70 transition-colors hidden sm:block">
            Privacy
          </a>
          <a href={`${API_BASE}/auth/xero/connect`}>
            <FluidGlassButton variant="glass" className="h-8 px-4 text-xs font-medium">
              Connect Xero <ArrowRight className="w-3 h-3" />
            </FluidGlassButton>
          </a>
        </div>
      </nav>

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-4xl mx-auto px-6 pt-20 pb-16 text-center">
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="flex flex-col items-center"
        >
          {/* Badge */}
          <motion.div variants={fadeUp} className="mb-8">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/[0.15] bg-white/[0.06] backdrop-blur-sm px-3.5 py-1.5 text-xs text-white/60">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />
              Beta · Free for UK accountants
            </span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            variants={fadeUp}
            className="text-[clamp(3rem,9vw,6rem)] font-bold tracking-tight leading-[1.02] mb-6"
          >
            <span className="text-white drop-shadow-sm">Accounting,</span>
            <br />
            <GradientText
              colors={["#818cf8", "#a78bfa", "#38bdf8", "#6366f1", "#818cf8"]}
              animationSpeed={6}
              className="font-bold"
            >
              automated.
            </GradientText>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            variants={fadeUp}
            className="text-base sm:text-lg text-white/55 max-w-lg mx-auto mb-10 leading-relaxed"
          >
            Connect Xero. AI categorises transactions, reconciles bank statements, and drafts management letters. Full explanations for every decision.
          </motion.p>

          {/* CTAs */}
          <motion.div variants={fadeUp} className="flex flex-col sm:flex-row gap-3 justify-center">
            <a href={`${API_BASE}/auth/xero/connect`}>
              <FluidGlassButton variant="primary" className="h-11 px-8 text-sm font-semibold shadow-xl shadow-indigo-600/30">
                Connect with Xero
                <ArrowRight className="w-4 h-4" />
              </FluidGlassButton>
            </a>
            <a href="#how-it-works">
              <FluidGlassButton variant="glass" className="h-11 px-6 text-sm">
                How it works <ChevronRight className="w-3.5 h-3.5" />
              </FluidGlassButton>
            </a>
          </motion.div>

          {/* Stats row */}
          <motion.div
            variants={fadeUp}
            className="mt-16 flex items-center justify-center gap-10 sm:gap-16 border-t border-white/[0.08] pt-10 w-full"
          >
            {stats.map((s, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl sm:text-3xl font-bold text-white tabular-nums tracking-tight">{s.value}</div>
                <div className="text-sm text-white/50 mt-1.5 font-medium">{s.label}</div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </section>

      {/* ── Features ──────────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-6xl mx-auto px-6 py-16">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          variants={container}
          className="grid sm:grid-cols-3 gap-4"
        >
          {features.map((f) => (
            <motion.div
              key={f.title}
              variants={fadeUp}
              className={`group relative rounded-xl border border-white/[0.07] bg-white/[0.03] backdrop-blur-sm p-6 ${f.hoverBorder} transition-all duration-300 cursor-default overflow-hidden`}
            >
              <div className={`absolute inset-0 ${f.hoverGlow} transition-colors duration-300`} />
              <div className="relative">
                <f.icon className={`w-5 h-5 ${f.accent} mb-4`} strokeWidth={1.75} />
                <h3 className="text-sm font-semibold text-white mb-2">{f.title}</h3>
                <p className="text-sm text-white/40 leading-relaxed">{f.description}</p>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ── How it works ──────────────────────────────────────────────────── */}
      <section id="how-it-works" className="relative z-10 max-w-2xl mx-auto px-6 py-12">
        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          variants={container}
        >
          <motion.p variants={fadeUp} className="text-[11px] font-semibold text-white/30 uppercase tracking-widest mb-6 text-center">
            How it works
          </motion.p>
          <div className="space-y-2">
            {steps.map((step, i) => (
              <motion.div
                key={i}
                variants={fadeUp}
                className="flex items-start gap-4 rounded-lg border border-white/[0.06] bg-white/[0.02] px-5 py-4"
              >
                <span className="shrink-0 mt-0.5 w-5 h-5 rounded-full border border-white/[0.15] flex items-center justify-center text-[10px] font-semibold text-white/40">
                  {i + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white">{step.label}</p>
                  <p className="text-xs text-white/40 mt-0.5">{step.detail}</p>
                </div>
                {i === steps.length - 1 && (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 ml-auto shrink-0 mt-0.5" />
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* ── Bottom CTA ────────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-xl mx-auto px-6 py-16 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45 }}
          className="rounded-2xl border border-white/[0.09] bg-white/[0.03] backdrop-blur-sm p-10 relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_0%,rgba(79,70,229,0.12),transparent)]" />
          <div className="relative">
            <h2 className="text-2xl font-bold text-white mb-3 tracking-tight">
              Ready to save hours every week?
            </h2>
            <p className="text-sm text-white/40 mb-8">
              Free beta. No credit card. Connect Xero in 30 seconds.
            </p>
            <a href={`${API_BASE}/auth/xero/connect`}>
              <FluidGlassButton variant="primary" className="h-11 px-10 text-sm font-semibold shadow-xl shadow-indigo-600/25">
                Get started free <ArrowRight className="w-4 h-4" />
              </FluidGlassButton>
            </a>
          </div>
        </motion.div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="relative z-10 border-t border-white/[0.05] py-8">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-white/25">
          <span>© 2026 AI Accountant · Built for UK accounting firms</span>
          <div className="flex gap-6">
            <a href="/privacy" className="hover:text-white/50 transition-colors">Privacy Policy</a>
            <a href="mailto:hello@aiaccountant.app" className="hover:text-white/50 transition-colors">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
