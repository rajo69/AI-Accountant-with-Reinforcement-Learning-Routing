const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("session_token");
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options?.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (res.status === 401) {
    if (typeof window !== "undefined") {
      localStorage.removeItem("session_token");
      document.cookie = "has_session=; Max-Age=0; path=/";
      window.location.href = "/";
    }
    throw new Error("Session expired. Please reconnect.");
  }

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Types ────────────────────────────────────────────────────────────────────

export interface DashboardSummary {
  total_accounts: number;
  total_transactions: number;
  uncategorised_count: number;
  unreconciled_count: number;
  last_sync_at: string | null;
  organisation_name: string | null;
}

export interface Transaction {
  id: string;
  organisation_id: string;
  account_id: string | null;
  xero_id: string;
  date: string;
  amount: string;
  description: string;
  reference: string | null;
  category: string | null;
  category_confidence: string | null;
  categorisation_status: string;
  is_reconciled: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuditLog {
  id: string;
  organisation_id: string;
  action: string;
  entity_type: string;
  entity_id: string;
  old_value: Record<string, unknown> | null;
  new_value: Record<string, unknown> | null;
  ai_model: string | null;
  ai_confidence: string | null;
  ai_explanation: string | null;
  ai_decision_data: Record<string, unknown> | null;
  created_at: string;
}

export interface TransactionDetail extends Transaction {
  audit_history: AuditLog[];
}

export interface TransactionListResponse {
  items: Transaction[];
  total: number;
  page: number;
  page_size: number;
}

export interface BankStatement {
  id: string;
  organisation_id: string;
  xero_id: string;
  date: string;
  amount: string;
  description: string;
  reference: string | null;
  matched_transaction_id: string | null;
  match_confidence: string | null;
  match_status: string;
  created_at: string;
  updated_at: string;
}

export interface MatchCandidate {
  transaction_id: string;
  description: string;
  date: string;
  amount: string;
  amount_score: number;
  date_score: number;
  description_score: number;
  combined_score: number;
}

export interface BankStatementDetail extends BankStatement {
  candidates: MatchCandidate[];
  audit_history: AuditLog[];
}

export interface BankStatementListResponse {
  items: BankStatement[];
  total: number;
  page: number;
  page_size: number;
}

export interface SyncResponse {
  status: string;
  synced_accounts: number;
  synced_transactions: number;
  synced_bank_statements: number;
  message: string | null;
}

export interface BatchCategoriseResponse {
  auto_categorised: number;
  suggested: number;
  needs_review: number;
  errors: number;
  total_processed: number;
}

export interface ReconcileBatchResponse {
  auto_matched: number;
  suggested: number;
  needs_review: number;
  errors: number;
  total_processed: number;
}

// ── Dashboard ────────────────────────────────────────────────────────────────

export const getDashboardSummary = () =>
  apiFetch<DashboardSummary>("/api/v1/dashboard/summary");

// ── Sync ─────────────────────────────────────────────────────────────────────

export const triggerSync = () =>
  apiFetch<SyncResponse>("/api/v1/sync", { method: "POST" });

// ── Categorise ───────────────────────────────────────────────────────────────

export const triggerCategorise = () =>
  apiFetch<BatchCategoriseResponse>("/api/v1/categorise", { method: "POST" });

export const getTransactions = (params: {
  page?: number;
  page_size?: number;
  status?: string;
  search?: string;
  date_from?: string;
  date_to?: string;
}) => {
  const q = new URLSearchParams();
  if (params.page) q.set("page", String(params.page));
  if (params.page_size) q.set("page_size", String(params.page_size));
  if (params.status) q.set("status", params.status);
  if (params.search) q.set("search", params.search);
  if (params.date_from) q.set("date_from", params.date_from);
  if (params.date_to) q.set("date_to", params.date_to);
  return apiFetch<TransactionListResponse>(`/api/v1/transactions?${q}`);
};

export const getTransaction = (id: string) =>
  apiFetch<TransactionDetail>(`/api/v1/transactions/${id}`);

export const approveTransaction = (id: string) =>
  apiFetch<Transaction>(`/api/v1/transactions/${id}/approve`, {
    method: "POST",
  });

export const rejectTransaction = (id: string) =>
  apiFetch<Transaction>(`/api/v1/transactions/${id}/reject`, {
    method: "POST",
  });

export const correctTransaction = (
  id: string,
  body: { category: string; category_code?: string }
) =>
  apiFetch<Transaction>(`/api/v1/transactions/${id}/correct`, {
    method: "POST",
    body: JSON.stringify(body),
  });

// ── Reconcile ────────────────────────────────────────────────────────────────

export const triggerReconcile = () =>
  apiFetch<ReconcileBatchResponse>("/api/v1/reconcile", { method: "POST" });

export const getBankStatements = (params: {
  page?: number;
  page_size?: number;
  match_status?: string;
  date_from?: string;
  date_to?: string;
}) => {
  const q = new URLSearchParams();
  if (params.page) q.set("page", String(params.page));
  if (params.page_size) q.set("page_size", String(params.page_size));
  if (params.match_status) q.set("match_status", params.match_status);
  if (params.date_from) q.set("date_from", params.date_from);
  if (params.date_to) q.set("date_to", params.date_to);
  return apiFetch<BankStatementListResponse>(`/api/v1/bank-statements?${q}`);
};

export const getBankStatement = (id: string) =>
  apiFetch<BankStatementDetail>(`/api/v1/bank-statements/${id}`);

export const confirmMatch = (id: string) =>
  apiFetch<BankStatement>(`/api/v1/bank-statements/${id}/confirm`, {
    method: "POST",
  });

export const unmatch = (id: string) =>
  apiFetch<BankStatement>(`/api/v1/bank-statements/${id}/unmatch`, {
    method: "POST",
  });

export const manualMatch = (id: string, transaction_id: string) =>
  apiFetch<BankStatement>(`/api/v1/bank-statements/${id}/match`, {
    method: "POST",
    body: JSON.stringify({ transaction_id }),
  });

// ── Explanations ─────────────────────────────────────────────────────────────

export interface FeatureContribution {
  name: string;
  value: number;
  contribution: number;
}

export interface XaiData {
  top_features: FeatureContribution[];
  explanation_text: string | null;
  model_type: "ebm" | "llm";
}

export interface RiskData {
  risk_score: number | null;
  risk_label: "low" | "medium" | "high" | null;
  fired_rules: string[];
  input_values: Record<string, number>;
}

export interface PredictionInfo {
  category: string | null;
  confidence: number | null;
  reasoning: string | null;
  model: string | null;
  category_code: string | null;
}

export interface ExplanationAuditEntry {
  id: string;
  action: string;
  ai_model: string | null;
  ai_confidence: number | null;
  ai_explanation: string | null;
  new_value: Record<string, unknown> | null;
  created_at: string | null;
}

export interface ExplanationResponse {
  transaction_id: string;
  category: string | null;
  category_confidence: number | null;
  categorisation_status: string;
  prediction: PredictionInfo | null;
  xai: XaiData;
  risk: RiskData;
  audit_history: ExplanationAuditEntry[];
}

export const getExplanation = (transactionId: string) =>
  apiFetch<ExplanationResponse>(`/api/v1/transactions/${transactionId}/explanation`);

// ── Documents ─────────────────────────────────────────────────────────────────

export interface GeneratedDocument {
  id: string;
  organisation_id: string;
  template: string;
  period_start: string;
  period_end: string;
  ai_model: string;
  figures: {
    total_income: number;
    total_expenses_abs: number;
    net: number;
    transaction_count: number;
    top_expense_categories: { name: string; total_abs: number; pct: number }[];
    largest_transactions: { date: string; description: string; amount: number }[];
  };
  generated_at: string;
}

export const listDocuments = () =>
  apiFetch<GeneratedDocument[]>("/api/v1/documents");

export const generateDocument = (body: {
  template: string;
  period_start: string;
  period_end: string;
}) => {
  const token = getToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return fetch(`${API_BASE}/api/v1/documents/generate`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
};
