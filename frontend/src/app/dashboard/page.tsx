"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import {
  RefreshCw,
  Tag,
  GitCompare,
  Layers,
  AlertCircle,
  Clock,
  Sparkles,
  Unlink2,
} from "lucide-react";
import {
  getDashboardSummary,
  triggerSync,
  triggerCategorise,
  triggerReconcile,
  type DashboardSummary,
} from "@/lib/api";
import FluidGlassButton from "@/components/fluid-glass-button";

// ── Count-up hook ─────────────────────────────────────────────────────────────

function useCountUp(target: number, duration = 800) {
  const [value, setValue] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (typeof target !== "number") return;
    const start = performance.now();
    const tick = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(target * ease));
      if (progress < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [target, duration]);

  return value;
}

// ── Animation variants ────────────────────────────────────────────────────────

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
};

const item = {
  hidden: { opacity: 0, y: 16 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.3 } },
};

// ── Stat card ─────────────────────────────────────────────────────────────────

interface StatCardProps {
  label: string;
  value: number | string;
  icon: React.ElementType;
  iconColor: string;
  iconBg: string;
  isNumeric?: boolean;
}

function StatCard({ label, value, icon: Icon, iconColor, iconBg, isNumeric = true }: StatCardProps) {
  const counted = useCountUp(isNumeric ? (value as number) : 0);

  return (
    <motion.div
      variants={item}
      whileHover={{ y: -2, transition: { duration: 0.15 } }}
      className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm hover:shadow-md flex items-start justify-between gap-4 transition-all duration-200"
    >
      <div>
        <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</p>
        <p className="mt-2 text-3xl font-bold text-slate-900 tabular-nums">
          {isNumeric ? counted.toLocaleString() : value}
        </p>
      </div>
      <div className={`${iconBg} w-10 h-10 rounded-lg flex items-center justify-center shrink-0 mt-0.5`}>
        <Icon className={`w-5 h-5 ${iconColor}`} />
      </div>
    </motion.div>
  );
}

// ── Action button ─────────────────────────────────────────────────────────────

interface ActionBtnProps {
  label: string;
  loadingLabel: string;
  icon: React.ElementType;
  loading: boolean;
  onClick: () => void;
  variant?: "primary" | "secondary";
}

function ActionBtn({ label, loadingLabel, icon: Icon, loading, onClick, variant = "secondary" }: ActionBtnProps) {
  if (variant === "primary") {
    return (
      <FluidGlassButton
        onClick={onClick}
        disabled={loading}
        variant="primary"
        className="shadow-md shadow-indigo-600/20"
      >
        <Icon className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        {loading ? loadingLabel : label}
      </FluidGlassButton>
    );
  }
  return (
    <Button
      variant="outline"
      onClick={onClick}
      disabled={loading}
      className="gap-2 text-sm"
    >
      <Icon className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
      {loading ? loadingLabel : label}
    </Button>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [categorising, setCategorising] = useState(false);
  const [reconciling, setReconciling] = useState(false);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchSummary = useCallback(async () => {
    try {
      const data = await getDashboardSummary();
      setSummary(data);
      setError(false);
    } catch {
      setError(true);
      setSummary(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchSummary(); }, [fetchSummary]);

  const handleSync = async () => {
    setSyncing(true);
    try {
      const res = await triggerSync();
      toast.success(`Sync complete: ${res.synced_transactions} transactions, ${res.synced_accounts} accounts`);
      await fetchSummary();
    } catch (e) { toast.error(String(e)); } finally { setSyncing(false); }
  };

  const handleCategorise = async () => {
    setCategorising(true);
    try {
      const res = await triggerCategorise();
      toast.success(`Categorised ${res.total_processed}: ${res.auto_categorised} auto, ${res.suggested} suggested`);
      await fetchSummary();
    } catch (e) { toast.error(String(e)); } finally { setCategorising(false); }
  };

  const handleReconcile = async () => {
    setReconciling(true);
    try {
      const res = await triggerReconcile();
      toast.success(`Reconciled ${res.total_processed}: ${res.auto_matched} auto, ${res.suggested} suggested`);
      await fetchSummary();
    } catch (e) { toast.error(String(e)); } finally { setReconciling(false); }
  };

  if (loading) {
    return (
      <div className="space-y-6 max-w-5xl">
        <div className="h-7 w-36 bg-slate-200 rounded-lg animate-pulse" />
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-28 bg-white rounded-xl animate-pulse border border-slate-200" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center justify-center h-[60vh] gap-6"
      >
        <div className="w-16 h-16 rounded-2xl bg-indigo-50 flex items-center justify-center">
          <Unlink2 className="w-8 h-8 text-indigo-500" />
        </div>
        <div className="text-center">
          <h2 className="text-lg font-semibold text-slate-900">Xero not connected</h2>
          <p className="text-sm text-slate-500 mt-1">Connect your Xero account to start using the AI pipeline.</p>
        </div>
        <a href={`${API_BASE}/auth/xero/connect`}>
          <motion.div whileTap={{ scale: 0.97 }}>
            <Button className="bg-indigo-600 hover:bg-indigo-500 gap-2 px-6">
              <Sparkles className="w-4 h-4" />
              Connect with Xero
            </Button>
          </motion.div>
        </a>
      </motion.div>
    );
  }

  const isFirstRun = !summary?.last_sync_at && (summary?.total_transactions ?? 0) === 0;

  const lastSync = summary?.last_sync_at
    ? new Date(summary.last_sync_at).toLocaleString("en-GB", { dateStyle: "short", timeStyle: "short" })
    : "Never";

  const stats: StatCardProps[] = [
    {
      label: "Total Transactions",
      value: summary?.total_transactions ?? 0,
      icon: Layers,
      iconColor: "text-slate-500",
      iconBg: "bg-slate-100",
    },
    {
      label: "Uncategorised",
      value: summary?.uncategorised_count ?? 0,
      icon: AlertCircle,
      iconColor: (summary?.uncategorised_count ?? 0) > 0 ? "text-amber-500" : "text-emerald-500",
      iconBg:    (summary?.uncategorised_count ?? 0) > 0 ? "bg-amber-50" : "bg-emerald-50",
    },
    {
      label: "Unreconciled",
      value: summary?.unreconciled_count ?? 0,
      icon: GitCompare,
      iconColor: (summary?.unreconciled_count ?? 0) > 0 ? "text-rose-500" : "text-emerald-500",
      iconBg:    (summary?.unreconciled_count ?? 0) > 0 ? "bg-rose-50" : "bg-emerald-50",
    },
    {
      label: "Last Sync",
      value: lastSync,
      icon: Clock,
      iconColor: "text-slate-400",
      iconBg: "bg-slate-100",
      isNumeric: false,
    },
  ];

  return (
    <motion.div
      initial="hidden"
      animate="show"
      variants={container}
      className="space-y-8 max-w-5xl"
    >
      {/* Header */}
      <motion.div variants={item}>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Dashboard</h1>
        {summary?.organisation_name && (
          <p className="text-sm text-slate-500 mt-1">{summary.organisation_name}</p>
        )}
      </motion.div>

      {/* First-run prompt */}
      {isFirstRun && (
        <motion.div
          variants={item}
          className="bg-indigo-50 border border-indigo-100 rounded-xl p-5 flex items-start gap-4"
        >
          <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0 mt-0.5">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div className="flex-1">
            <p className="font-semibold text-indigo-900 text-sm">You&apos;re connected! Run your first sync</p>
            <p className="text-indigo-600/70 text-xs mt-1">
              Click &ldquo;Sync with Xero&rdquo; below to import your transactions, then run the AI pipeline.
            </p>
          </div>
        </motion.div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <StatCard key={s.label} {...s} />
        ))}
      </div>

      {/* Actions */}
      <motion.div variants={item} className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-4">AI Pipeline</p>
        <div className="flex flex-wrap gap-3 sm:flex-row flex-col sm:items-center">
          <ActionBtn
            label="Sync with Xero"
            loadingLabel="Syncing…"
            icon={RefreshCw}
            loading={syncing}
            onClick={handleSync}
            variant="primary"
          />
          <ActionBtn
            label="Categorise All"
            loadingLabel="Categorising…"
            icon={Tag}
            loading={categorising}
            onClick={handleCategorise}
          />
          <ActionBtn
            label="Reconcile All"
            loadingLabel="Reconciling…"
            icon={GitCompare}
            loading={reconciling}
            onClick={handleReconcile}
          />
        </div>
      </motion.div>
    </motion.div>
  );
}
