"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "sonner";
import {
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Search,
  Link2,
} from "lucide-react";
import {
  getBankStatements,
  getBankStatement,
  confirmMatch,
  unmatch,
  manualMatch,
  getTransactions,
  type BankStatement,
  type BankStatementDetail,
  type Transaction,
} from "@/lib/api";
import { ExplanationPanel } from "@/components/explanation-panel";

// ── Match badge ───────────────────────────────────────────────────────────────

const MATCH_CONFIG: Record<string, { label: string; className: string }> = {
  confirmed:    { label: "Confirmed",    className: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200" },
  auto_matched: { label: "Auto-matched", className: "bg-sky-50 text-sky-700 ring-1 ring-sky-200" },
  suggested:    { label: "Suggested",    className: "bg-amber-50 text-amber-700 ring-1 ring-amber-200" },
  unmatched:    { label: "Unmatched",    className: "bg-slate-100 text-slate-500 ring-1 ring-slate-200" },
};

function MatchBadge({ status }: { status: string }) {
  const cfg = MATCH_CONFIG[status] ?? { label: status, className: "bg-slate-100 text-slate-500" };
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-medium ${cfg.className}`}>
      {cfg.label}
    </span>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtAmount(amount: string) {
  return new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP" }).format(parseFloat(amount));
}
function fmtDate(d: string) {
  return new Date(d).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}
function fmtPct(n: number) {
  return `${(n * 100).toFixed(0)}%`;
}

// ── Score bar ─────────────────────────────────────────────────────────────────

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between text-[10px] text-slate-500">
        <span>{label}</span>
        <span className="font-mono">{fmtPct(value)}</span>
      </div>
      <div className="h-1 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          className="h-full bg-indigo-400 rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.5}}
        />
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function ReconciliationPage() {
  const [statements, setStatements] = useState<BankStatement[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [matchStatus, setMatchStatus] = useState("");
  const [loading, setLoading] = useState(true);

  const [selected, setSelected] = useState<BankStatementDetail | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  const [searchOpen, setSearchOpen] = useState(false);
  const [txSearch, setTxSearch] = useState("");
  const [txResults, setTxResults] = useState<Transaction[]>([]);
  const [txSearching, setTxSearching] = useState(false);

  const PAGE_SIZE = 20;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getBankStatements({ page, page_size: PAGE_SIZE, match_status: matchStatus || undefined });
      setStatements(res.items);
      setTotal(res.total);
    } catch (e) { toast.error(String(e)); } finally { setLoading(false); }
  }, [page, matchStatus]);

  useEffect(() => { load(); }, [load]);

  const openStatement = async (id: string) => {
    try {
      const detail = await getBankStatement(id);
      setSelected(detail);
    } catch (e) { toast.error(String(e)); }
  };

  const handleConfirm = async () => {
    if (!selected) return;
    setActionLoading(true);
    try {
      await confirmMatch(selected.id);
      toast.success("Match confirmed");
      load(); setSelected(null);
    } catch (e) { toast.error(String(e)); } finally { setActionLoading(false); }
  };

  const handleUnmatch = async () => {
    if (!selected) return;
    setActionLoading(true);
    try {
      await unmatch(selected.id);
      toast.success("Match removed");
      load(); setSelected(null);
    } catch (e) { toast.error(String(e)); } finally { setActionLoading(false); }
  };

  const handleManualMatch = async (txId: string) => {
    if (!selected) return;
    setActionLoading(true);
    try {
      await manualMatch(selected.id, txId);
      toast.success("Manually matched");
      load(); setSearchOpen(false); setSelected(null);
    } catch (e) { toast.error(String(e)); } finally { setActionLoading(false); }
  };

  const searchTransactions = async () => {
    if (!txSearch.trim()) return;
    setTxSearching(true);
    try {
      const res = await getTransactions({ search: txSearch, page_size: 10 });
      setTxResults(res.items);
    } catch (e) { toast.error(String(e)); } finally { setTxSearching(false); }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3}}
      className="max-w-6xl"
    >
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900">Reconciliation</h1>
        <p className="text-sm text-slate-500 mt-0.5">Match bank statement lines to your Xero transactions</p>
      </div>

      <div className="flex flex-col lg:flex-row gap-5">
        {/* Left panel: statements list */}
        <div className="flex-1 min-w-0 space-y-3">
          <Select
            value={matchStatus}
            onValueChange={(v) => { setMatchStatus(v === "all" ? "" : v ?? ""); setPage(1); }}
          >
            <SelectTrigger className="w-48 bg-white border-slate-200 text-sm">
              <SelectValue placeholder="All statuses" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              <SelectItem value="unmatched">Unmatched</SelectItem>
              <SelectItem value="suggested">Suggested</SelectItem>
              <SelectItem value="auto_matched">Auto-matched</SelectItem>
              <SelectItem value="confirmed">Confirmed</SelectItem>
            </SelectContent>
          </Select>

          <div className="bg-white rounded-xl shadow-sm ring-1 ring-slate-100 overflow-hidden">
            {loading ? (
              <div className="divide-y divide-slate-50">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="flex items-center justify-between px-4 py-3.5 gap-4">
                    <div className="space-y-1.5 flex-1">
                      <div className="h-3.5 bg-slate-100 rounded animate-pulse w-3/4" />
                      <div className="h-3 bg-slate-100 rounded animate-pulse w-1/3" />
                    </div>
                    <div className="h-5 w-16 bg-slate-100 rounded-full animate-pulse" />
                  </div>
                ))}
              </div>
            ) : statements.length === 0 ? (
              <div className="py-16 text-center space-y-3">
                <div className="w-12 h-12 rounded-xl bg-slate-50 flex items-center justify-center mx-auto">
                  <GitCompareIcon className="w-6 h-6 text-slate-300" />
                </div>
                <p className="text-sm font-medium text-slate-600">No bank statements</p>
                <p className="text-xs text-slate-400 max-w-xs mx-auto">
                  {matchStatus
                    ? "Try a different filter"
                    : "Sync with Xero to import bank statement lines, then run Reconcile All"}
                </p>
              </div>
            ) : (
              <motion.div
                className="divide-y divide-slate-50"
                initial="hidden"
                animate="show"
                variants={{ hidden: {}, show: { transition: { staggerChildren: 0.03 } } }}
              >
                {statements.map((s) => (
                  <motion.div
                    key={s.id}
                    variants={{ hidden: { opacity: 0, x: -8 }, show: { opacity: 1, x: 0, transition: { duration: 0.2 } } }}
                    onClick={() => openStatement(s.id)}
                    className={`flex items-center justify-between px-4 py-3.5 cursor-pointer transition-colors text-sm ${
                      selected?.id === s.id
                        ? "bg-indigo-50/60 border-l-2 border-indigo-500"
                        : "hover:bg-slate-50/80"
                    }`}
                  >
                    <div className="space-y-0.5 min-w-0">
                      <p className="font-medium text-slate-900 truncate">{s.description}</p>
                      <p className="text-slate-400 text-xs">{fmtDate(s.date)}</p>
                    </div>
                    <div className="flex flex-col items-end gap-1.5 shrink-0 ml-4">
                      <span className={`font-mono font-medium ${parseFloat(s.amount) >= 0 ? "text-emerald-700" : "text-slate-900"}`}>
                        {fmtAmount(s.amount)}
                      </span>
                      <MatchBadge status={s.match_status} />
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </div>

          {totalPages > 1 && (
            <div className="flex items-center gap-3 text-sm text-slate-500">
              <Button variant="outline" size="sm" className="bg-white gap-1" disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
                <ChevronLeft className="w-3.5 h-3.5" /> Previous
              </Button>
              <span className="text-xs">{page} / {totalPages}</span>
              <Button variant="outline" size="sm" className="bg-white gap-1" disabled={page === totalPages} onClick={() => setPage((p) => p + 1)}>
                Next <ChevronRight className="w-3.5 h-3.5" />
              </Button>
            </div>
          )}
        </div>

        {/* Right panel: detail */}
        <div className="lg:w-80 shrink-0">
          <AnimatePresence mode="wait">
            {selected ? (
              <motion.div
                key={selected.id}
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 12 }}
                transition={{ duration: 0.2}}
                className="bg-white rounded-xl shadow-sm ring-1 ring-slate-100 p-5 space-y-5 text-sm"
              >
                {/* Statement info */}
                <div>
                  <p className="font-semibold text-slate-900 text-base leading-snug">{selected.description}</p>
                  <p className="text-slate-400 text-xs mt-1">{fmtDate(selected.date)}</p>
                </div>

                <div className="grid grid-cols-2 gap-y-2.5 bg-slate-50 rounded-lg p-3">
                  <span className="text-slate-400 text-xs uppercase tracking-wide font-medium self-center">Amount</span>
                  <span className={`font-mono font-semibold ${parseFloat(selected.amount) >= 0 ? "text-emerald-700" : "text-slate-900"}`}>
                    {fmtAmount(selected.amount)}
                  </span>
                  <span className="text-slate-400 text-xs uppercase tracking-wide font-medium self-center">Status</span>
                  <MatchBadge status={selected.match_status} />
                  {selected.match_confidence && (
                    <>
                      <span className="text-slate-400 text-xs uppercase tracking-wide font-medium">Confidence</span>
                      <span className="font-mono text-indigo-600 font-semibold">{fmtPct(parseFloat(selected.match_confidence))}</span>
                    </>
                  )}
                </div>

                {/* Match candidates */}
                {selected.candidates.length > 0 && (
                  <div>
                    <p className="font-semibold text-slate-700 text-xs uppercase tracking-wide mb-2">Match candidates</p>
                    <div className="space-y-3">
                      {selected.candidates.map((c, i) => (
                        <motion.div
                          key={c.transaction_id}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className="rounded-lg border border-slate-100 p-3 space-y-2"
                        >
                          <div className="flex justify-between items-start gap-2">
                            <p className="font-medium text-xs text-slate-900 leading-snug truncate">{c.description}</p>
                            <span className={`font-mono text-xs shrink-0 ${parseFloat(c.amount) >= 0 ? "text-emerald-700" : "text-slate-900"}`}>
                              {fmtAmount(c.amount)}
                            </span>
                          </div>
                          <p className="text-xs text-slate-400">{fmtDate(c.date)}</p>
                          <div className="space-y-1">
                            <ScoreBar label="Amount" value={c.amount_score} />
                            <ScoreBar label="Date" value={c.date_score} />
                            <ScoreBar label="Description" value={c.description_score} />
                          </div>
                          <div className="flex justify-between items-center pt-1">
                            <span className="text-xs text-slate-400">Overall</span>
                            <span className="text-sm font-bold text-indigo-600">{fmtPct(c.combined_score)}</span>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Matched transaction AI explanation */}
                {selected.matched_transaction_id && (
                  <div>
                    <p className="font-semibold text-slate-700 text-xs uppercase tracking-wide mb-2">
                      Matched transaction: AI explanation
                    </p>
                    <ExplanationPanel transactionId={selected.matched_transaction_id} inline />
                  </div>
                )}

                {/* Actions */}
                <div className="flex flex-wrap gap-2 pt-1 border-t border-slate-100">
                  {(selected.match_status === "suggested" || selected.match_status === "auto_matched") && (
                    <Button size="sm" className="bg-indigo-600 hover:bg-indigo-700 gap-1.5" onClick={handleConfirm} disabled={actionLoading}>
                      <CheckCircle2 className="w-3.5 h-3.5" /> Confirm
                    </Button>
                  )}
                  {selected.match_status !== "unmatched" && (
                    <Button size="sm" variant="outline" className="text-rose-600 hover:text-rose-700 gap-1.5" onClick={handleUnmatch} disabled={actionLoading}>
                      <XCircle className="w-3.5 h-3.5" /> Unmatch
                    </Button>
                  )}
                  {selected.match_status === "unmatched" && (
                    <Button size="sm" variant="outline" className="gap-1.5" onClick={() => { setTxSearch(""); setTxResults([]); setSearchOpen(true); }}>
                      <Link2 className="w-3.5 h-3.5" /> Find &amp; Match
                    </Button>
                  )}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white rounded-xl shadow-sm ring-1 ring-slate-100 p-8 text-center"
              >
                <div className="w-10 h-10 rounded-lg bg-slate-50 flex items-center justify-center mx-auto mb-3">
                  <GitCompareIcon className="w-5 h-5 text-slate-300" />
                </div>
                <p className="text-sm text-slate-400">Select a statement to see details and match candidates</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Manual match search dialog */}
      <Dialog open={searchOpen} onOpenChange={setSearchOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="text-base">Find Transaction to Match</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400" />
                <Input
                  placeholder="Search by description…"
                  className="pl-9 bg-white border-slate-200"
                  value={txSearch}
                  onChange={(e) => setTxSearch(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && searchTransactions()}
                />
              </div>
              <Button onClick={searchTransactions} disabled={txSearching} className="bg-indigo-600 hover:bg-indigo-700 shrink-0">
                {txSearching ? "…" : "Search"}
              </Button>
            </div>
            {txResults.length > 0 && (
              <div className="divide-y divide-slate-50 border border-slate-100 rounded-xl overflow-hidden max-h-64 overflow-y-auto">
                {txResults.map((tx) => (
                  <div
                    key={tx.id}
                    className="flex items-center justify-between px-3 py-2.5 hover:bg-slate-50 cursor-pointer text-sm transition-colors"
                    onClick={() => handleManualMatch(tx.id)}
                  >
                    <div>
                      <p className="font-medium text-slate-900 truncate max-w-52">{tx.description}</p>
                      <p className="text-slate-400 text-xs">{fmtDate(tx.date)}</p>
                    </div>
                    <span className={`font-mono text-xs shrink-0 ml-3 ${parseFloat(tx.amount) >= 0 ? "text-emerald-700" : "text-slate-900"}`}>
                      {fmtAmount(tx.amount)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}

function GitCompareIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
      <circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="18" r="2.5" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 8.5v2a5 5 0 005 5H14M18 15.5v-2a5 5 0 00-5-5H10" />
    </svg>
  );
}
