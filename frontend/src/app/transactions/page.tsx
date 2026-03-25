"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
import { Search, ChevronLeft, ChevronRight, CheckCircle2, XCircle, Pencil, Layers } from "lucide-react";
import {
  getTransactions,
  getTransaction,
  approveTransaction,
  rejectTransaction,
  correctTransaction,
  type Transaction,
  type TransactionDetail,
} from "@/lib/api";
import { ExplanationPanel } from "@/components/explanation-panel";

// ── Status badge ──────────────────────────────────────────────────────────────

const STATUS_CONFIG: Record<string, { label: string; className: string }> = {
  confirmed:        { label: "Confirmed",        className: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200" },
  auto_categorised: { label: "Auto-categorised", className: "bg-sky-50 text-sky-700 ring-1 ring-sky-200" },
  suggested:        { label: "Suggested",        className: "bg-amber-50 text-amber-700 ring-1 ring-amber-200" },
  needs_review:     { label: "Needs review",     className: "bg-rose-50 text-rose-700 ring-1 ring-rose-200" },
  uncategorised:    { label: "Uncategorised",    className: "bg-slate-100 text-slate-500 ring-1 ring-slate-200" },
  rejected:         { label: "Rejected",         className: "bg-slate-100 text-slate-400 ring-1 ring-slate-200" },
};

function StatusBadge({ status }: { status: string }) {
  const cfg = STATUS_CONFIG[status] ?? { label: status, className: "bg-slate-100 text-slate-500" };
  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-medium ${cfg.className}`}>
      {cfg.label}
    </span>
  );
}

// ── Formatters ────────────────────────────────────────────────────────────────

function fmtAmount(amount: string) {
  return new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP" }).format(parseFloat(amount));
}
function fmtDate(d: string) {
  return new Date(d).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

function ConfidencePill({ confidence }: { confidence: string | null }) {
  if (!confidence) return <span className="text-slate-300">-</span>;
  const pct = parseFloat(confidence) * 100;
  const cls = pct >= 85 ? "text-emerald-600" : pct >= 50 ? "text-amber-600" : "text-rose-600";
  return <span className={`font-mono text-xs font-medium tabular-nums ${cls}`}>{pct.toFixed(0)}%</span>;
}

// ── Animations ────────────────────────────────────────────────────────────────

const tableVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.025 } },
};
const rowVariants = {
  hidden: { opacity: 0, x: -6 },
  show:   { opacity: 1, x: 0, transition: { duration: 0.18 } },
};

// ── Page ──────────────────────────────────────────────────────────────────────

export default function TransactionsPage() {
  const [items, setItems] = useState<Transaction[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(true);

  const [selected, setSelected] = useState<TransactionDetail | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [correctOpen, setCorrectOpen] = useState(false);
  const [correctCategory, setCorrectCategory] = useState("");
  const [actionLoading, setActionLoading] = useState(false);

  const PAGE_SIZE = 20;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getTransactions({
        page,
        page_size: PAGE_SIZE,
        search: search || undefined,
        status: status || undefined,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (e) {
      toast.error(String(e));
    } finally {
      setLoading(false);
    }
  }, [page, search, status]);

  useEffect(() => { load(); }, [load]);

  const openDetail = async (id: string) => {
    try {
      const detail = await getTransaction(id);
      setSelected(detail);
      setDetailOpen(true);
    } catch (e) { toast.error(String(e)); }
  };

  const handleApprove = async (id: string) => {
    setActionLoading(true);
    try {
      await approveTransaction(id);
      toast.success("Category confirmed");
      load();
      setDetailOpen(false);
    } catch (e) { toast.error(String(e)); } finally { setActionLoading(false); }
  };

  const handleReject = async (id: string) => {
    setActionLoading(true);
    try {
      await rejectTransaction(id);
      toast.success("Rejected. Transaction is uncategorised again.");
      load();
      setDetailOpen(false);
    } catch (e) { toast.error(String(e)); } finally { setActionLoading(false); }
  };

  const handleCorrect = async () => {
    if (!selected || !correctCategory.trim()) return;
    setActionLoading(true);
    try {
      await correctTransaction(selected.id, { category: correctCategory });
      toast.success("Category corrected");
      load();
      setCorrectOpen(false);
      setDetailOpen(false);
    } catch (e) { toast.error(String(e)); } finally { setActionLoading(false); }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-5 max-w-6xl"
    >
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Transactions</h1>
        <p className="text-sm text-slate-500 mt-0.5">Review, approve and correct AI categorisations</p>
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap items-center">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400" />
          <Input
            placeholder="Search description…"
            className="pl-9 w-64 text-sm"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          />
        </div>
        <Select
          value={status}
          onValueChange={(v) => { setStatus(v === "all" ? "" : v ?? ""); setPage(1); }}
        >
          <SelectTrigger className="w-48 text-sm">
            <SelectValue placeholder="All statuses" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All statuses</SelectItem>
            <SelectItem value="uncategorised">Uncategorised</SelectItem>
            <SelectItem value="suggested">Suggested</SelectItem>
            <SelectItem value="auto_categorised">Auto-categorised</SelectItem>
            <SelectItem value="confirmed">Confirmed</SelectItem>
            <SelectItem value="needs_review">Needs review</SelectItem>
            <SelectItem value="rejected">Rejected</SelectItem>
          </SelectContent>
        </Select>
        {total > 0 && (
          <span className="text-xs text-slate-400 ml-1 tabular-nums">{total.toLocaleString()} transactions</span>
        )}
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="bg-slate-50 hover:bg-slate-50 border-b border-slate-200">
              <TableHead className="text-xs font-semibold text-slate-500 uppercase tracking-wide py-3">Date</TableHead>
              <TableHead className="text-xs font-semibold text-slate-500 uppercase tracking-wide py-3">Description</TableHead>
              <TableHead className="text-xs font-semibold text-slate-500 uppercase tracking-wide py-3 text-right">Amount</TableHead>
              <TableHead className="text-xs font-semibold text-slate-500 uppercase tracking-wide py-3">Category</TableHead>
              <TableHead className="text-xs font-semibold text-slate-500 uppercase tracking-wide py-3">Status</TableHead>
              <TableHead className="text-xs font-semibold text-slate-500 uppercase tracking-wide py-3">Conf.</TableHead>
              <TableHead className="py-3" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              [...Array(6)].map((_, i) => (
                <TableRow key={i} className="border-b border-slate-100">
                  {[...Array(7)].map((_, j) => (
                    <TableCell key={j}>
                      <div className="h-4 bg-slate-100 rounded animate-pulse" style={{ width: `${60 + Math.random() * 40}%` }} />
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : items.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="py-20">
                  <div className="flex flex-col items-center gap-3 text-center">
                    <div className="w-12 h-12 rounded-xl bg-slate-100 flex items-center justify-center">
                      <Layers className="w-6 h-6 text-slate-400" />
                    </div>
                    <p className="text-sm font-medium text-slate-700">No transactions found</p>
                    <p className="text-xs text-slate-400 max-w-xs">
                      {search || status
                        ? "Try adjusting your filters"
                        : "Sync with Xero first, then run Categorise All from the dashboard"}
                    </p>
                  </div>
                </TableCell>
              </TableRow>
            ) : (
              <AnimatePresence>
                <motion.tbody variants={tableVariants} initial="hidden" animate="show">
                  {items.map((tx) => (
                    <motion.tr
                      key={tx.id}
                      variants={rowVariants}
                      className="border-b border-slate-100 hover:bg-slate-50 cursor-pointer transition-colors"
                      onClick={() => openDetail(tx.id)}
                    >
                      <TableCell className="text-xs text-slate-500 py-3 tabular-nums">{fmtDate(tx.date)}</TableCell>
                      <TableCell className="text-sm text-slate-900 font-medium max-w-xs truncate py-3">
                        {tx.description}
                      </TableCell>
                      <TableCell className="text-sm text-right font-mono tabular-nums py-3">
                        <span className={parseFloat(tx.amount) >= 0 ? "text-emerald-600" : "text-rose-600"}>
                          {fmtAmount(tx.amount)}
                        </span>
                      </TableCell>
                      <TableCell className="text-xs text-slate-500 py-3">{tx.category ?? <span className="text-slate-300">-</span>}</TableCell>
                      <TableCell className="py-3"><StatusBadge status={tx.categorisation_status} /></TableCell>
                      <TableCell className="py-3">
                        <ConfidencePill confidence={tx.category_confidence} />
                      </TableCell>
                      <TableCell className="py-3" onClick={(e) => e.stopPropagation()}>
                        {tx.categorisation_status === "suggested" && (
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-emerald-600 hover:text-emerald-700 hover:bg-emerald-50"
                              onClick={() => handleApprove(tx.id)}
                              title="Accept"
                            >
                              <CheckCircle2 className="w-4 h-4" />
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-7 w-7 p-0 text-rose-500 hover:text-rose-600 hover:bg-rose-50"
                              onClick={() => handleReject(tx.id)}
                              title="Reject"
                            >
                              <XCircle className="w-4 h-4" />
                            </Button>
                          </div>
                        )}
                      </TableCell>
                    </motion.tr>
                  ))}
                </motion.tbody>
              </AnimatePresence>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center gap-3 text-sm text-slate-500">
          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            disabled={page === 1}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="w-3.5 h-3.5" /> Previous
          </Button>
          <span className="text-xs tabular-nums">Page {page} of {totalPages}</span>
          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            disabled={page === totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            Next <ChevronRight className="w-3.5 h-3.5" />
          </Button>
        </div>
      )}

      {/* Detail dialog */}
      <Dialog open={detailOpen} onOpenChange={setDetailOpen}>
        <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-base">Transaction Detail</DialogTitle>
          </DialogHeader>
          {selected && (
            <div className="space-y-4 text-sm">
              {/* Fields */}
              <div className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2.5 bg-slate-50 rounded-lg p-4 border border-slate-100">
                <span className="text-slate-400 text-[10px] uppercase tracking-widest font-medium self-center">Date</span>
                <span className="text-slate-900 text-sm">{fmtDate(selected.date)}</span>

                <span className="text-slate-400 text-[10px] uppercase tracking-widest font-medium self-center">Amount</span>
                <span className={`font-mono font-semibold tabular-nums ${parseFloat(selected.amount) >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
                  {fmtAmount(selected.amount)}
                </span>

                <span className="text-slate-400 text-[10px] uppercase tracking-widest font-medium">Description</span>
                <span className="text-slate-900 leading-snug">{selected.description}</span>

                {selected.reference && (
                  <>
                    <span className="text-slate-400 text-[10px] uppercase tracking-widest font-medium">Ref</span>
                    <span className="text-slate-600">{selected.reference}</span>
                  </>
                )}

                <span className="text-slate-400 text-[10px] uppercase tracking-widest font-medium self-center">Category</span>
                <span className="text-slate-900">{selected.category ?? <span className="text-slate-300">-</span>}</span>

                <span className="text-slate-400 text-[10px] uppercase tracking-widest font-medium self-center">Status</span>
                <StatusBadge status={selected.categorisation_status} />
              </div>

              {/* XAI explanation panel */}
              {selected.categorisation_status !== "uncategorised" && (
                <ExplanationPanel transactionId={selected.id} inline />
              )}

              {/* Actions */}
              <div className="flex gap-2 pt-1 border-t border-slate-100">
                {(selected.categorisation_status === "suggested" ||
                  selected.categorisation_status === "auto_categorised") && (
                  <Button
                    size="sm"
                    className="bg-indigo-600 hover:bg-indigo-500 gap-1.5 border-0"
                    onClick={() => handleApprove(selected.id)}
                    disabled={actionLoading}
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" /> Accept
                  </Button>
                )}
                {selected.categorisation_status === "suggested" && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-rose-200 text-rose-600 hover:bg-rose-50 gap-1.5"
                    onClick={() => handleReject(selected.id)}
                    disabled={actionLoading}
                  >
                    <XCircle className="w-3.5 h-3.5" /> Reject
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="outline"
                  className="gap-1.5 ml-auto"
                  onClick={() => {
                    setCorrectCategory(selected.category ?? "");
                    setCorrectOpen(true);
                  }}
                >
                  <Pencil className="w-3.5 h-3.5" /> Edit Category
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Correct category dialog */}
      <Dialog open={correctOpen} onOpenChange={setCorrectOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle className="text-base">Correct Category</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <Input
              placeholder="Enter correct category name"
              value={correctCategory}
              onChange={(e) => setCorrectCategory(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCorrect()}
            />
            <div className="flex gap-2">
              <Button
                className="bg-indigo-600 hover:bg-indigo-500 border-0"
                onClick={handleCorrect}
                disabled={actionLoading || !correctCategory.trim()}
              >
                Save
              </Button>
              <Button variant="outline" onClick={() => setCorrectOpen(false)}>Cancel</Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}
