import type {
  BenchmarksResponse,
  DecisionResponse,
  DemoGenerateResult,
  DemoMonitorResult,
  DemoResetResult,
  DemoScenarioInfo,
  DemoSession,
  DistributionFeature,
  DriftResult,
  ExplainResult,
  HealthSummary,
  HousePricePredictionInput,
  HousePricePredictionResponse,
  MonitoringRecord,
  PredictionRecord,
  ProductionInfo,
  PromoteResult,
  PromotionCheckResult,
  PromotionRecord,
  RegistryInfo,
  RetrainingRun,
  RetrainResult,
  RollbackResult,
} from '../types';

const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) || 'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number | null;
  constructor(status: number | null, message: string) {
    super(message);
    this.status = status;
  }
}

export function apiErrorStatus(e: unknown): number | null {
  if (e instanceof ApiError) return e.status;
  const m = /API (\d{3})\b/.exec(String((e as Error)?.message ?? e));
  return m ? Number(m[1]) : null;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  } catch (e) {
    throw new ApiError(null, `API network error ${path}: ${(e as Error).message}`);
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new ApiError(res.status, `API ${res.status} ${path} ${text.slice(0, 200)}`);
  }
  return (await res.json()) as T;
}

export const api = {
  base: BASE,
  health: () => req<{ status: string }>('/health'),
  production: () => req<ProductionInfo>('/models/house-price/production'),
  registry: (limit = 100) => req<RegistryInfo>(`/models/house-price/registry?limit=${limit}`),
  predict: (input: HousePricePredictionInput) =>
    req<HousePricePredictionResponse>('/predict/house-price', { method: 'POST', body: JSON.stringify(input) }),
  explain: (input: HousePricePredictionInput) =>
    req<ExplainResult>('/predict/house-price/explain', { method: 'POST', body: JSON.stringify(input) }),
  benchmarks: (location?: string) =>
    req<BenchmarksResponse>(location ? `/predict/house-price/benchmarks?location=${encodeURIComponent(location)}` : '/predict/house-price/benchmarks'),
  predictionHistory: (limit = 50, offset = 0) =>
    req<{ items: PredictionRecord[]; limit: number; offset: number }>(
      `/predict/house-price/history?limit=${limit}&offset=${offset}`,
    ),
  driftCheck: () => req<DriftResult>('/monitor/house-price', { method: 'POST' }),
  healthSummary: (limit = 1000) =>
    req<HealthSummary>(`/monitor/house-price/summary?limit=${limit}`),
  monitoringHistory: (limit = 50, offset = 0) =>
    req<{ items: MonitoringRecord[]; limit: number; offset: number }>(
      `/monitor/house-price/history?limit=${limit}&offset=${offset}`,
    ),
  distributions: (limit = 1000) =>
    req<{ features: DistributionFeature[]; reference_rows: number; production_rows: number }>(
      `/monitor/house-price/distributions?limit=${limit}`,
    ),
  retrain: (promote = false) =>
    req<RetrainResult>('/monitor/house-price/retrain', {
      method: 'POST',
      body: JSON.stringify({ promote }),
    }),
  retrainHistory: (limit = 50, offset = 0) =>
    req<{ items: PromotionRecord[]; limit: number; offset: number }>(
      `/monitor/house-price/retrain/history?limit=${limit}&offset=${offset}`,
    ),
  retrainStatus: () => req<Record<string, unknown>>('/monitor/house-price/retrain/status'),
  lifecycleDecision: (limit = 1000) =>
    req<DecisionResponse>(`/retrain/house-price/decision?limit=${limit}`),
  lifecycleStartRun: (production_limit = 1000, trigger = 'manual') =>
    req<RetrainingRun>('/retrain/house-price/runs', {
      method: 'POST',
      body: JSON.stringify({ production_limit, trigger }),
    }),
  lifecycleRuns: (limit = 50, offset = 0) =>
    req<{ items: RetrainingRun[]; limit: number; offset: number }>(
      `/retrain/house-price/runs?limit=${limit}&offset=${offset}`,
    ),
  lifecycleRun: (run_id: string) => req<RetrainingRun>(`/retrain/house-price/runs/${encodeURIComponent(run_id)}`),
  lifecyclePromotionCheck: (run_id: string, window_limit = 1000) =>
    req<PromotionCheckResult>(`/retrain/house-price/runs/${encodeURIComponent(run_id)}/promotion-check`, {
      method: 'POST',
      body: JSON.stringify({ window_limit }),
    }),
  lifecyclePromote: (run_id: string) =>
    req<PromoteResult>(`/retrain/house-price/runs/${encodeURIComponent(run_id)}/promote`, {
      method: 'POST',
    }),
  lifecycleRollback: (pointer_history_id: number) =>
    req<RollbackResult>('/retrain/house-price/rollback', {
      method: 'POST',
      body: JSON.stringify({ pointer_history_id }),
    }),
  demoScenarios: () =>
    req<{ items: DemoScenarioInfo[] }>('/demo/house-price/scenarios'),
  demoCreateSession: (scenario: string, sample_count = 100, seed = 42, params?: Record<string, unknown>) =>
    req<DemoSession>('/demo/house-price/sessions', {
      method: 'POST',
      body: JSON.stringify({ scenario, sample_count, seed, params: params ?? null }),
    }),
  demoSessions: (limit = 50, offset = 0) =>
    req<{ items: DemoSession[]; limit: number; offset: number }>(
      `/demo/house-price/sessions?limit=${limit}&offset=${offset}`,
    ),
  demoSession: (session_id: string) =>
    req<DemoSession>(`/demo/house-price/sessions/${encodeURIComponent(session_id)}`),
  demoGenerate: (session_id: string) =>
    req<DemoGenerateResult>(`/demo/house-price/sessions/${encodeURIComponent(session_id)}/generate`, {
      method: 'POST',
    }),
  demoMonitor: (session_id: string) =>
    req<DemoMonitorResult>(`/demo/house-price/sessions/${encodeURIComponent(session_id)}/monitor`, {
      method: 'POST',
    }),
  demoReset: (session_id: string) =>
    req<DemoResetResult>(`/demo/house-price/sessions/${encodeURIComponent(session_id)}/reset`, {
      method: 'POST',
    }),
  demoRetrain: (session_id: string, production_limit = 1000) =>
    req<RetrainingRun>(`/demo/house-price/sessions/${encodeURIComponent(session_id)}/retrain`, {
      method: 'POST',
      body: JSON.stringify({ production_limit }),
    }),
};

export function formatMoneyUSD(v: number | null | undefined, currency: 'USD' | 'INR'): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (currency === 'INR') return formatINR(v * 83);
  return `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
}

export function formatINR(amountInr: number): string {
  const neg = amountInr < 0;
  const s = String(Math.round(Math.abs(amountInr)));
  let grouped: string;
  if (s.length <= 3) grouped = s;
  else {
    const tail = s.slice(-3);
    const head = s.slice(0, -3);
    const chunks: string[] = [];
    const first = head.length % 2 === 0 ? 2 : head.length % 2;
    chunks.push(head.slice(0, first));
    for (let i = first; i < head.length; i += 2) chunks.push(head.slice(i, i + 2));
    grouped = `${chunks.filter(Boolean).join(',')},${tail}`;
  }
  return `₹${neg ? '-' : ''}${grouped}`;
}

export function formatNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits);
}

export function formatInt(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  return v.toLocaleString('en-US');
}

export function timeUTC(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toISOString().slice(11, 19) + ' UTC';
  } catch {
    return iso;
  }
}
