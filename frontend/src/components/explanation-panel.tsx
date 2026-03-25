"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRight, Brain, AlertTriangle, Info } from "lucide-react";
import { getExplanation, type ExplanationResponse } from "@/lib/api";

// ── Risk badge ────────────────────────────────────────────────────────────────

const RISK_STYLES: Record<string, { className: string; label: string }> = {
  low:    { className: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200", label: "Low risk" },
  medium: { className: "bg-amber-50 text-amber-700 ring-1 ring-amber-200",       label: "Medium risk" },
  high:   { className: "bg-rose-50 text-rose-700 ring-1 ring-rose-200",           label: "High risk" },
};

function RiskBadge({ label }: { label: string | null }) {
  if (!label) return null;
  const cfg = RISK_STYLES[label] ?? { className: "bg-slate-100 text-slate-600", label };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-medium ${cfg.className}`}>
      {label === "high" && <AlertTriangle className="w-2.5 h-2.5" />}
      {cfg.label}
    </span>
  );
}

// ── Confidence arc ────────────────────────────────────────────────────────────

function ConfidenceRing({ value }: { value: number }) {
  const r = 20;
  const circumference = 2 * Math.PI * r;
  const dash = circumference * value;
  const pct = Math.round(value * 100);
  const color = value >= 0.85 ? "#059669" : value >= 0.5 ? "#d97706" : "#e11d48";

  return (
    <div className="relative w-14 h-14 shrink-0">
      <svg className="w-full h-full -rotate-90" viewBox="0 0 48 48">
        <circle cx="24" cy="24" r={r} fill="none" stroke="#e2e8f0" strokeWidth={4} />
        <motion.circle
          cx="24" cy="24" r={r} fill="none"
          stroke={color} strokeWidth={4}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - dash }}
          transition={{ duration: 0.8, delay: 0.1 }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-bold text-slate-800 tabular-nums">{pct}%</span>
      </div>
    </div>
  );
}

// ── Feature bar ───────────────────────────────────────────────────────────────

function FeatureBar({
  name,
  value,
  contribution,
  maxContribution,
  delay,
}: {
  name: string;
  value: number;
  contribution: number;
  maxContribution: number;
  delay: number;
}) {
  const pct = maxContribution > 0 ? Math.abs(contribution) / maxContribution : 0;
  const isPositive = contribution >= 0;
  const label = name.replace(/_/g, " ");
  const formattedValue =
    name === "amount"
      ? new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP" }).format(value)
      : value.toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.2 }}
      className="space-y-1"
    >
      <div className="flex justify-between text-[11px] text-slate-500">
        <span className="font-medium capitalize">{label}</span>
        <span className="text-slate-400 font-mono tabular-nums">{formattedValue}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
          <motion.div
            className={`h-full rounded-full ${isPositive ? "bg-indigo-500" : "bg-amber-400"}`}
            initial={{ width: 0 }}
            animate={{ width: `${pct * 100}%` }}
            transition={{ delay: delay + 0.1, duration: 0.5 }}
          />
        </div>
        <span className={`text-[10px] w-8 text-right font-mono tabular-nums shrink-0 ${isPositive ? "text-indigo-600" : "text-amber-600"}`}>
          {isPositive ? "+" : ""}{contribution.toFixed(2)}
        </span>
      </div>
    </motion.div>
  );
}

// ── Expandable section ────────────────────────────────────────────────────────

function Expandable({ title, count, children }: { title: string; count?: number; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-slate-100 pt-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-slate-400 font-medium hover:text-slate-700 transition-colors w-full text-left"
      >
        <motion.span animate={{ rotate: open ? 90 : 0 }} transition={{ duration: 0.15 }}>
          <ChevronRight className="w-3 h-3" />
        </motion.span>
        {title}
        {count !== undefined && (
          <span className="ml-auto bg-slate-100 text-slate-500 rounded-full px-1.5 py-0.5 text-[10px]">{count}</span>
        )}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pt-2">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function ExplanationPanel({
  transactionId,
  inline = false,
}: {
  transactionId: string;
  inline?: boolean;
}) {
  const [data, setData] = useState<ExplanationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getExplanation(transactionId)
      .then((res) => { if (!cancelled) setData(res); })
      .catch((e) => { if (!cancelled) setError(String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [transactionId]);

  const wrapClass = inline
    ? "space-y-4 text-sm"
    : "bg-slate-50 rounded-xl border border-slate-200 p-5 space-y-4 text-sm";

  if (loading) {
    return (
      <div className={wrapClass}>
        <div className="flex items-center gap-3 animate-pulse">
          <div className="w-14 h-14 rounded-full bg-slate-200" />
          <div className="flex-1 space-y-2">
            <div className="h-3 bg-slate-200 rounded w-1/2" />
            <div className="h-5 bg-slate-200 rounded w-1/3" />
          </div>
        </div>
        <div className="space-y-2">
          {[0.7, 0.5, 0.4].map((w, i) => (
            <div key={i} className="space-y-1">
              <div className="h-2.5 bg-slate-200 rounded animate-pulse" style={{ width: `${w * 100}%` }} />
              <div className="h-1.5 bg-slate-100 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className={`${wrapClass} flex items-center gap-2 text-xs text-slate-400`}>
        <Info className="w-4 h-4 shrink-0" />
        {error ?? "No explanation available yet. Run categorisation first."}
      </div>
    );
  }

  const { prediction, xai, risk } = data;
  const confidence = prediction?.confidence ?? data.category_confidence;
  const topFeatures = xai?.top_features ?? [];
  const maxContrib = topFeatures.length > 0
    ? Math.max(...topFeatures.map((f) => Math.abs(f.contribution)), 0.001)
    : 1;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.25 }}
      className={wrapClass}
    >
      {/* Decision header */}
      <div className="flex items-start gap-3">
        {confidence !== null && confidence !== undefined && (
          <ConfidenceRing value={confidence} />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-0.5">
            <Brain className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest">AI Decision</span>
          </div>
          <p className="font-semibold text-slate-900 text-sm leading-snug truncate">
            {prediction?.category ?? data.category ?? "-"}
          </p>
          <div className="flex items-center gap-2 mt-1.5">
            <RiskBadge label={risk?.risk_label ?? null} />
            {xai.model_type === "ebm" && (
              <span className="text-[10px] bg-indigo-50 text-indigo-600 rounded-full px-2 py-0.5 font-medium">
                EBM
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Feature bars */}
      {topFeatures.length > 0 && (
        <div className="space-y-2.5">
          <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest">Feature contributions</p>
          {topFeatures.slice(0, 5).map((f, i) => (
            <FeatureBar
              key={f.name}
              name={f.name}
              value={f.value}
              contribution={f.contribution}
              maxContribution={maxContrib}
              delay={i * 0.05}
            />
          ))}
        </div>
      )}

      {/* AI reasoning */}
      {xai.explanation_text && (
        <div className="bg-white rounded-lg px-3 py-2.5 text-xs text-slate-500 leading-relaxed border border-slate-100">
          {xai.explanation_text}
        </div>
      )}

      {/* Why this risk level */}
      {risk?.fired_rules && risk.fired_rules.length > 0 && (
        <Expandable title="Why this risk level?" count={risk.fired_rules.length}>
          <ol className="space-y-1.5 border-l-2 border-slate-200 pl-3">
            {risk.fired_rules.map((rule, i) => (
              <li key={i} className="text-[11px] text-slate-500 leading-relaxed">{rule}</li>
            ))}
          </ol>
        </Expandable>
      )}

      {/* Full audit trail */}
      {data.audit_history.length > 0 && (
        <Expandable title="Full audit trail" count={data.audit_history.length}>
          <ol className="space-y-3 border-l-2 border-slate-200 pl-3">
            {data.audit_history.map((entry) => (
              <li key={entry.id} className="space-y-0.5">
                <p className="text-[11px] font-semibold text-slate-700">{entry.action}</p>
                {entry.ai_explanation && (
                  <p className="text-[11px] text-slate-500 italic leading-relaxed">{entry.ai_explanation}</p>
                )}
                <div className="flex items-center gap-3 text-[10px] text-slate-400">
                  {entry.ai_confidence !== null && entry.ai_confidence !== undefined && (
                    <span className="tabular-nums">{Math.round(entry.ai_confidence * 100)}% confidence</span>
                  )}
                  {entry.ai_model && <span>{entry.ai_model}</span>}
                  {entry.created_at && <span>{new Date(entry.created_at).toLocaleString("en-GB")}</span>}
                </div>
              </li>
            ))}
          </ol>
        </Expandable>
      )}
    </motion.div>
  );
}
