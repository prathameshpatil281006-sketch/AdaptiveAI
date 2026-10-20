import { useEffect, useMemo, useState } from 'react';
import { api, apiErrorStatus, formatINR, timeUTC } from '../api/client';
import { useApp } from '../state/store';
import type {
  ExplainResult,
  HousePricePredictionInput,
  HousePricePredictionResponse,
  LocalityBenchmark,
  PredictionRecord,
  ProductionInfo,
} from '../types';
import { EmptyState, KeyVal, PageHeader, SectionCard, StatusBadge } from '../components/ui';
import LocalityMarketMap from '../components/LocalityMarketMap';

const FALLBACK_LOCALITIES = [
  'Aundh', 'Balewadi', 'Baner', 'Hadapsar', 'Hinjawadi', 'Kharadi',
  'Kothrud', 'Punawale', 'Ravet', 'Viman Nagar', 'Wagholi', 'Wakad',
];
const PROPERTY_TYPES = ['Apartment', 'Villa', 'Row House'];
const FURNISHING_OPTIONS = ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'];

const DEFAULT_FORM: HousePricePredictionInput = {
  location: 'Wakad',
  property_type: 'Apartment',
  area_sqft: 1200,
  bedrooms: 3,
  bathrooms: 2,
  parking: 1,
  property_age: 5,
  floor_number: 6,
  total_floors: 12,
  furnishing_status: 'Semi-Furnished',
};

const PRESETS: Record<string, HousePricePredictionInput> = {
  'Wakad 3BHK': { ...DEFAULT_FORM },
  'Kothrud Villa': {
    location: 'Kothrud', property_type: 'Villa', area_sqft: 2400, bedrooms: 4,
    bathrooms: 4, parking: 2, property_age: 3, floor_number: 1, total_floors: 2,
    furnishing_status: 'Fully-Furnished',
  },
  'Wagholi 1BHK': {
    location: 'Wagholi', property_type: 'Apartment', area_sqft: 600, bedrooms: 1,
    bathrooms: 1, parking: 0, property_age: 2, floor_number: 4, total_floors: 14,
    furnishing_status: 'Unfurnished',
  },
};

type Phase = 'idle' | 'loading' | 'success' | 'error';

function methodLabel(method: string | null): string {
  if (method === 'linear_exact') return 'Linear exact contribution';
  if (method === 'baseline_substitution') return 'Baseline substitution';
  return method ?? '—';
}

function formatLakh(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 10000000) return `₹${(v / 10000000).toFixed(2)} Cr`;
  if (abs >= 100000) return `₹${(v / 100000).toFixed(1)} L`;
  return formatINR(v);
}

function formatTs(iso: string | null): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit', hour12: false,
    });
  } catch {
    return iso;
  }
}

function inputSummary(h: PredictionRecord): { text: string; legacy: boolean } {
  const d = h.input_data ?? {};
  const loc = typeof d.location === 'string' ? d.location : null;
  const area = typeof d.area_sqft === 'number' ? d.area_sqft
    : typeof d.area === 'number' ? d.area : null;
  if (loc && area !== null) {
    const bhk = typeof d.bedrooms === 'number' ? `${d.bedrooms}b` : '?b';
    const ba = typeof d.bathrooms === 'number' ? `${d.bathrooms}ba` : '?ba';
    return { text: `${loc} · ${area.toLocaleString()} sqft · ${bhk}/${ba}`, legacy: false };
  }
  if (area !== null) return { text: `Legacy prediction · ${area.toLocaleString()} sqft`, legacy: true };
  return { text: 'Legacy prediction', legacy: true };
}

export default function Predictions() {
  const { currency, refreshTick, bump } = useApp();
  const [form, setForm] = useState<HousePricePredictionInput>({ ...DEFAULT_FORM });
  const [preset, setPreset] = useState('Wakad 3BHK');
  const [phase, setPhase] = useState<Phase>('idle');
  const [formError, setFormError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [result, setResult] = useState<HousePricePredictionResponse | null>(null);
  const [submitted, setSubmitted] = useState<HousePricePredictionInput | null>(null);
  const [explain, setExplain] = useState<ExplainResult | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [localities, setLocalities] = useState<string[]>(FALLBACK_LOCALITIES);
  const [benchmarks, setBenchmarks] = useState<LocalityBenchmark[]>([]);
  const [serving, setServing] = useState<ProductionInfo | null>(null);

  const money = (v: number | null | undefined): string => {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    if (currency === 'INR') return formatINR(v);
    return `$${Math.round(v / 83).toLocaleString('en-US')}`;
  };

  const set = <K extends keyof HousePricePredictionInput>(k: K, v: HousePricePredictionInput[K]) => {
    setForm((f) => ({ ...f, [k]: v }));
    setFormError(null);
  };

  const loadHistory = async () => {
    try {
      const h = await api.predictionHistory(50, 0);
      setHistory(h.items);
    } catch {
      /* keep previous */
    }
  };

  useEffect(() => {
    loadHistory();
  }, [refreshTick]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const b = await api.benchmarks();
        if (!alive || !b.items?.length) return;
        setBenchmarks(b.items);
        setLocalities(b.items.map((i) => i.location).sort());
      } catch {
        /* keep fallback locality names (no rates duplicated) */
      }
      try {
        const p = await api.production();
        if (alive) setServing(p);
      } catch {
        /* serving badge degrades */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const applyPreset = (name: string) => {
    setPreset(name);
    const p = PRESETS[name];
    if (p) {
      setForm({ ...p });
      setFormError(null);
    }
  };

  const validate = (f: HousePricePredictionInput): string | null => {
    if (!localities.includes(f.location)) return 'Please select a supported Pune locality.';
    if (!PROPERTY_TYPES.includes(f.property_type)) return 'Please select a property type.';
    if (!FURNISHING_OPTIONS.includes(f.furnishing_status)) return 'Please select a furnishing status.';
    if (!Number.isFinite(f.area_sqft) || f.area_sqft <= 0) return 'Area must be greater than zero.';
    if (!Number.isInteger(f.bedrooms) || f.bedrooms < 1 || f.bedrooms > 6) return 'Bedrooms must be 1–6.';
    if (!Number.isInteger(f.bathrooms) || f.bathrooms < 1 || f.bathrooms > 6) return 'Bathrooms must be 1–6.';
    if (f.bathrooms > f.bedrooms + 2) return 'Bathrooms must be at most bedrooms + 2.';
    if (!Number.isInteger(f.parking) || f.parking < 0 || f.parking > 3) return 'Parking must be 0–3.';
    if (!Number.isFinite(f.property_age) || f.property_age < 0) return 'Property age cannot be negative.';
    if (!Number.isInteger(f.floor_number) || f.floor_number < 0) return 'Floor cannot be negative.';
    if (!Number.isInteger(f.total_floors) || f.total_floors < 1) return 'Total floors must be at least 1.';
    if (f.floor_number > f.total_floors) return 'Floor cannot exceed total floors.';
    return null;
  };

  const run = async () => {
    const problem = validate(form);
    if (problem) {
      setFormError(problem);
      return;
    }
    const payload = { ...form };
    setPhase('loading');
    setFormError(null);
    setApiError(null);
    setExplainError(null);
    try {
      const r = await api.predict(payload);
      setResult(r);
      setSubmitted(payload);
      setPhase('success');
      try {
        const ex = await api.explain(payload);
        setExplain(ex);
      } catch {
        setExplain(null);
        setExplainError('Explanation unavailable for this estimate.');
      }
      await loadHistory();
      bump();
    } catch (e) {
      const status = apiErrorStatus(e);
      if (status === 422) setApiError('Please check the property details.');
      else if (status === 503)
        setApiError('Current production model is not compatible with the Pune schema yet. An explicit model promotion is required.');
      else if (status === null) setApiError('Prediction service unavailable.');
      else setApiError('Unable to complete valuation.');
      setPhase('error');
    }
  };

  const contributions = useMemo(() => {
    if (!explain) return [];
    return [...explain.contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  }, [explain]);
  const maxAbs = useMemo(
    () => contributions.reduce((m, c) => Math.max(m, Math.abs(c.contribution)), 0),
    [contributions],
  );

  const benchMax = useMemo(() => {
    if (!result) return 1;
    return Math.max(result.prediction, result.locality_benchmark_price ?? 0, 1);
  }, [result]);
  const benchPct = (v: number | null) => (v === null ? 0 : Math.max(2, (v / benchMax) * 100));

  const diff = result?.benchmark_difference ?? null;
  const diffPct = result?.benchmark_difference_percent ?? null;

  const localityMarks = useMemo(() => {
    const ranked = [...benchmarks].sort((a, b) => b.benchmark_price_per_sqft - a.benchmark_price_per_sqft);
    const top = ranked[0]?.benchmark_price_per_sqft ?? 1;
    return ranked.map((b) => ({
      ...b,
      width: Math.max(4, (b.benchmark_price_per_sqft / top) * 100),
      selected: submitted?.location === b.location,
    }));
  }, [benchmarks, submitted]);

  const selectCls = 'aa-select mt-1';
  const labelCls = 'text-[12px] text-aa-text';
  const numCls = 'aa-input mt-1';

  return (
    <div>
      <div className="flex items-center justify-end mb-2">
        {serving ? (
          <StatusBadge tone="green">● Serving: {serving.model_name} v{serving.model_version ?? '—'}</StatusBadge>
        ) : (
          <StatusBadge tone="gray">● Serving: unavailable</StatusBadge>
        )}
      </div>
      <PageHeader
        title="Pune Property Valuation"
        desc="Model estimate, locality benchmark comparison, and model explanation for Pune residential property."
      />

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-3">
        <div className="xl:col-span-2">
          <SectionCard title="Property Details" subtitle="Pune schema v2 · validated against serving contract">
            <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
              <span className="text-aa-faint mr-1">Presets:</span>
              {Object.keys(PRESETS).map((k) => (
                <button
                  key={k}
                  onClick={() => applyPreset(k)}
                  className={`px-2.5 py-1 rounded-md border text-[11px] ${
                    preset === k ? 'border-aa-blue bg-aa-blue/15 text-white' : 'border-aa-border text-aa-muted hover:text-white'
                  }`}
                >
                  {k}
                </button>
              ))}
            </div>

            <div className="mt-4 text-[10px] tracking-[0.08em] font-semibold text-aa-faint">PROPERTY</div>
            <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <label className="block">
                <span className={labelCls}>Location</span>
                <select aria-label="Location" value={form.location} onChange={(e) => set('location', e.target.value)} className={selectCls}>
                  {localities.map((l) => (
                    <option key={l} value={l}>{l}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className={labelCls}>Property Type</span>
                <select aria-label="Property Type" value={form.property_type} onChange={(e) => set('property_type', e.target.value)} className={selectCls}>
                  {PROPERTY_TYPES.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </label>
              <label className="block sm:col-span-2">
                <span className={labelCls}>Area (sq ft)</span>
                <input
                  aria-label="Area in square feet" type="number" min={1} step={10} value={form.area_sqft}
                  onChange={(e) => set('area_sqft', Number(e.target.value))} className={numCls}
                />
              </label>
            </div>

            <div className="mt-4 text-[10px] tracking-[0.08em] font-semibold text-aa-faint">CONFIGURATION</div>
            <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              <label className="block">
                <span className={labelCls}>Bedrooms</span>
                <select aria-label="Bedrooms" value={form.bedrooms} onChange={(e) => set('bedrooms', Number(e.target.value))} className={selectCls}>
                  {[1, 2, 3, 4, 5, 6].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className={labelCls}>Bathrooms</span>
                <select aria-label="Bathrooms" value={form.bathrooms} onChange={(e) => set('bathrooms', Number(e.target.value))} className={selectCls}>
                  {[1, 2, 3, 4, 5, 6].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className={labelCls}>Parking</span>
                <select aria-label="Parking" value={form.parking} onChange={(e) => set('parking', Number(e.target.value))} className={selectCls}>
                  {[0, 1, 2, 3].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
            </div>

            <div className="mt-4 text-[10px] tracking-[0.08em] font-semibold text-aa-faint">BUILDING</div>
            <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              <label className="block">
                <span className={labelCls}>Property Age (yrs)</span>
                <input
                  aria-label="Property age in years" type="number" min={0} step={1} value={form.property_age}
                  onChange={(e) => set('property_age', Number(e.target.value))} className={numCls}
                />
              </label>
              <label className="block">
                <span className={labelCls}>Floor</span>
                <select aria-label="Floor number" value={form.floor_number} onChange={(e) => set('floor_number', Number(e.target.value))} className={selectCls}>
                  {Array.from({ length: 31 }, (_, n) => n).map((n) => (
                    <option key={n} value={n}>{n === 0 ? 'Ground' : n}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className={labelCls}>Total Floors</span>
                <select aria-label="Total floors" value={form.total_floors} onChange={(e) => set('total_floors', Number(e.target.value))} className={selectCls}>
                  {Array.from({ length: 30 }, (_, n) => n + 1).map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </label>
            </div>

            <div className="mt-4 text-[10px] tracking-[0.08em] font-semibold text-aa-faint">FINISHING</div>
            <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <label className="block">
                <span className={labelCls}>Furnishing</span>
                <select aria-label="Furnishing status" value={form.furnishing_status} onChange={(e) => set('furnishing_status', e.target.value)} className={selectCls}>
                  {FURNISHING_OPTIONS.map((f) => (
                    <option key={f} value={f}>{f}</option>
                  ))}
                </select>
              </label>
            </div>

            {formError && <div role="alert" className="mt-3 text-[12px] text-red-400">{formError}</div>}
            {apiError && <div role="alert" className="mt-3 text-[12px] text-red-400">{apiError}</div>}

            <button onClick={run} disabled={phase === 'loading'} className="aa-btn-primary w-full justify-center mt-4 !py-2.5">
              <span>◉</span> {phase === 'loading' ? 'Estimating…' : 'Run Valuation'}
            </button>
          </SectionCard>
        </div>

        <div className="xl:col-span-3 space-y-3">
          {phase !== 'success' || !result ? (
            <SectionCard title="Valuation Result">
              <EmptyState
                title={phase === 'error' ? 'Valuation did not complete' : 'Run a property valuation'}
                desc={
                  phase === 'error'
                    ? apiError ?? 'Unable to complete valuation.'
                    : 'Enter Pune property details to generate a model estimate, locality benchmark comparison, and model explanation.'
                }
              />
            </SectionCard>
          ) : (
            <>
              <SectionCard
                title="Estimated Property Value"
                subtitle="Model estimate · served inference"
                right={
                  <StatusBadge tone="green">
                    ● {result.model_name ?? 'served'}{result.model_version ? ` v${result.model_version}` : ''}
                  </StatusBadge>
                }
              >
                <div className="aa-mono text-[32px] font-bold text-white leading-tight">{money(result.prediction)}</div>
                <div className="mt-1 text-[12px] text-aa-muted">
                  {result.predicted_price_per_sqft !== null ? `${money(result.predicted_price_per_sqft)} / sq ft` : 'Price/sqft unavailable'}
                </div>
                <div className="mt-3 rounded-lg border border-aa-border bg-[#0a1222] p-3">
                  <div className="text-[10px] tracking-[0.08em] font-semibold text-aa-faint">ESTIMATED RANGE</div>
                  {result.prediction_lower !== null && result.prediction_upper !== null ? (
                    <>
                      <div className="mt-1 aa-mono text-[15px] font-semibold text-white">
                        {formatLakh(result.prediction_lower)} — {formatLakh(result.prediction_upper)}
                      </div>
                      <div className="mt-0.5 text-[10.5px] text-aa-faint">Model-derived prediction range</div>
                    </>
                  ) : (
                    <div className="mt-1 text-[12px] text-aa-muted">Prediction range unavailable</div>
                  )}
                </div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-[11.5px]">
                  <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                    <div className="text-[10px] text-aa-faint">Model</div>
                    <div className="aa-mono text-aa-text truncate">{result.model_name ?? '—'}</div>
                  </div>
                  <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                    <div className="text-[10px] text-aa-faint">Version</div>
                    <div className="aa-mono text-aa-text">{result.model_version ? `v${result.model_version}` : '—'}</div>
                  </div>
                  <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                    <div className="text-[10px] text-aa-faint">Inference ID</div>
                    <div className="aa-mono text-aa-text">{result.prediction_id !== null ? `#${result.prediction_id}` : '—'}</div>
                  </div>
                </div>
              </SectionCard>

              <SectionCard title="Locality Market Context" subtitle="Reference market calibration · not ground truth">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-[12px]">
                  <div className="rounded-lg border border-aa-blue/40 bg-aa-blue/[0.07] p-3">
                    <div className="text-[10px] tracking-[0.08em] font-semibold text-aa-blueText">MODEL ESTIMATE</div>
                    <div className="mt-1 aa-mono text-[17px] font-bold text-white">{money(result.prediction)}</div>
                    <div className="mt-0.5 aa-mono text-[11px] text-aa-muted">
                      {result.predicted_price_per_sqft !== null ? `${money(result.predicted_price_per_sqft)} / sq ft` : '—'}
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-white/5 overflow-hidden">
                      <div className="h-full rounded-full bg-aa-blue" style={{ width: `${benchPct(result.prediction)}%` }} />
                    </div>
                  </div>
                  <div className="rounded-lg border border-aa-border bg-[#0a1222] p-3">
                    <div className="text-[10px] tracking-[0.08em] font-semibold text-aa-faint">LOCALITY BENCHMARK</div>
                    <div className="mt-1 aa-mono text-[17px] font-bold text-aa-text">{money(result.locality_benchmark_price)}</div>
                    <div className="mt-0.5 aa-mono text-[11px] text-aa-muted">
                      {result.benchmark_price_per_sqft !== null ? `${money(result.benchmark_price_per_sqft)} / sq ft · ${result.location ?? ''}` : '—'}
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-white/5 overflow-hidden">
                      <div className="h-full rounded-full bg-slate-500/70" style={{ width: `${benchPct(result.locality_benchmark_price)}%` }} />
                    </div>
                  </div>
                </div>
                <div className="mt-2.5 flex items-center justify-between rounded-lg border border-aa-border bg-[#0a0e1a] px-3 py-2.5 text-[12px]">
                  <span className="text-aa-muted">Difference vs benchmark</span>
                  {diff !== null && diffPct !== null ? (
                    <span className={`aa-mono font-semibold ${diff >= 0 ? 'text-green-400' : 'text-amber-400'}`}>
                      {diff >= 0 ? '+' : '−'}{money(Math.abs(diff))} · {diff >= 0 ? '+' : '−'}{Math.abs(diffPct).toFixed(2)}%
                    </span>
                  ) : (
                    <span className="aa-mono text-aa-faint">—</span>
                  )}
                </div>
              </SectionCard>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <SectionCard
                  title="Property Snapshot"
                  subtitle="Inputs actually evaluated"
                >
                  {submitted ? (
                    <div className="text-[12.5px] text-aa-text space-y-1">
                      <div className="text-[14px] font-semibold text-white">{submitted.location} · {submitted.property_type}</div>
                      <div className="aa-mono text-aa-muted">
                        {submitted.area_sqft.toLocaleString()} sq ft · {submitted.bedrooms} BHK · {submitted.bathrooms} Bath · {submitted.parking} Parking
                      </div>
                      <div className="aa-mono text-aa-muted">
                        {submitted.property_age} yrs · Floor {submitted.floor_number} / {submitted.total_floors} · {submitted.furnishing_status}
                      </div>
                    </div>
                  ) : (
                    <div className="text-[12px] text-aa-faint">—</div>
                  )}
                </SectionCard>
                <SectionCard title="Model Information" subtitle="Serving metadata">
                  <KeyVal k="Model" v={<span className="aa-mono">{result.model_name ?? '—'}</span>} />
                  <KeyVal k="Model version" v={<span className="aa-mono">{result.model_version ? `v${result.model_version}` : '—'}</span>} />
                  <KeyVal k="Schema version" v={<span className="aa-mono">{result.schema_version !== null ? `v${result.schema_version}` : '—'}</span>} />
                  <KeyVal k="Inference ID" v={<span className="aa-mono">{result.prediction_id !== null ? `#${result.prediction_id}` : '—'}</span>} />
                  <KeyVal k="Inference time" v={<span className="aa-mono">{formatTs(result.inferred_at)}</span>} />
                </SectionCard>
              </div>
            </>
          )}
        </div>
      </div>

      {phase === 'success' && result && (
        <div className="mt-3 grid grid-cols-1 xl:grid-cols-5 gap-3">
          <div className="xl:col-span-3">
            <SectionCard
              title="Why This Estimate?"
              subtitle={`Explanation method: ${explain ? methodLabel(explain.method) : '—'}`}
              right={explain ? <StatusBadge tone="blue">● {explain.model_name}</StatusBadge> : undefined}
            >
              {!explain ? (
                <div className="text-[12px] text-aa-muted">{explainError ?? 'Explanation unavailable for this estimate.'}</div>
              ) : (
                <div className="space-y-1.5">
                  {contributions.map((c) => (
                    <div key={c.feature} className="flex items-center gap-2.5">
                      <div className="w-32 shrink-0 truncate text-[12px] text-aa-text" title={c.label}>{c.label}</div>
                      <div className="flex-1 h-2 rounded-full bg-white/5 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            c.direction === 'positive' ? 'bg-aa-blue' : c.direction === 'negative' ? 'bg-amber-400/80' : 'bg-slate-600/60'
                          }`}
                          style={{ width: `${maxAbs > 0 ? Math.max(3, (Math.abs(c.contribution) / maxAbs) * 100) : 0}%` }}
                        />
                      </div>
                      <div
                        className={`w-28 shrink-0 text-right aa-mono text-[11.5px] ${
                          c.direction === 'positive' ? 'text-aa-blueText' : c.direction === 'negative' ? 'text-amber-400' : 'text-aa-faint'
                        }`}
                      >
                        {c.contribution < 0 ? '−' : '+'}{money(Math.abs(c.contribution))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </SectionCard>
          </div>
          <div className="xl:col-span-2">
            <SectionCard title="Pune Locality Context" subtitle="Benchmark price/sqft · ranked">
              {localityMarks.length === 0 ? (
                <div className="text-[12px] text-aa-muted">Locality benchmarks unavailable.</div>
              ) : (
                <div className="space-y-1.5">
                  {localityMarks.map((b) => (
                    <div key={b.location} className={`flex items-center gap-2 rounded-md px-2 py-1 ${b.selected ? 'bg-aa-blue/10 border border-aa-blue/40' : 'border border-transparent'}`}>
                      <div className="w-24 shrink-0 truncate text-[11.5px] text-aa-text">{b.location}</div>
                      <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                        <div className={`h-full rounded-full ${b.selected ? 'bg-aa-blue' : 'bg-slate-600/60'}`} style={{ width: `${b.width}%` }} />
                      </div>
                      <div className="w-20 shrink-0 text-right aa-mono text-[10.5px] text-aa-muted">{money(b.benchmark_price_per_sqft)}</div>
                    </div>
                  ))}
                  <div className="pt-1 text-[10.5px] text-aa-faint">Reference market calibration · selected locality highlighted</div>
                </div>
              )}
            </SectionCard>
          </div>
        </div>
      )}

      <div className="mt-3">
        <SectionCard
          title="Pune Locality Market Map"
          subtitle="Locality benchmark visualization · schematic relative layout"
        >
          <LocalityMarketMap
            benchmarks={benchmarks}
            selected={submitted?.location ?? form.location}
            onSelect={(loc) => {
              set('location', loc);
              setFormError(null);
            }}
            formatRate={(v) => (v === null || v === undefined ? '—' : money(v))}
          />
        </SectionCard>
      </div>

      <div className="mt-3">
        <SectionCard title="Prediction History" subtitle="Newest first · legacy rows render gracefully">
          <div className="overflow-x-auto -mx-4 px-4">
            <table className="aa-table w-full min-w-[760px]">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Inference ID</th>
                  <th>Input Summary</th>
                  <th>Model</th>
                  <th>Version</th>
                  <th className="!text-right">Predicted Valuation</th>
                </tr>
              </thead>
              <tbody>
                {history.length === 0 && (
                  <tr>
                    <td colSpan={6} className="aa-mono text-aa-faint !py-6 text-center">
                      No predictions yet — run a valuation above.
                    </td>
                  </tr>
                )}
                {history.map((h) => {
                  const s = inputSummary(h);
                  return (
                    <tr key={h.id}>
                      <td className="aa-mono text-aa-faint">{timeUTC(h.timestamp)}</td>
                      <td className="aa-mono text-aa-blueText">pred_{String(h.id).padStart(8, '0')}</td>
                      <td className="aa-mono text-aa-muted">
                        {s.text}{' '}
                        {s.legacy && (
                          <span className="ml-1 px-1.5 py-[1px] rounded border border-aa-border text-aa-faint text-[10px]">Legacy</span>
                        )}
                      </td>
                      <td>{h.model_name || '—'}</td>
                      <td>
                        <span className="aa-mono text-[10.5px] px-1.5 py-[2px] rounded bg-aa-blue/15 border border-aa-blue/30 text-aa-blueText">
                          {h.model_version ? `v${h.model_version}` : '—'}
                        </span>
                      </td>
                      <td className="!text-right aa-mono font-semibold">{money(h.prediction)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
