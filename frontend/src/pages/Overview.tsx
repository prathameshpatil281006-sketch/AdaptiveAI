import { useEffect, useState } from 'react';
import { api, formatINR, formatNum, timeUTC } from '../api/client';
import { useApp } from '../state/store';
import type { MonitoringRecord, ProductionInfo, PromotionRecord } from '../types';
import { InfoPanel, KeyVal, MetricCard, PageHeader, SectionCard, StatusBadge } from '../components/ui';

function money(v: number | null | undefined, currency: 'INR' | 'USD'): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (currency === 'INR') return formatINR(v);
  return `$${Math.round(v / 83).toLocaleString('en-US')}`;
}

function shortId(id: string | null | undefined): string {
  if (!id) return '—';
  return id.length > 14 ? `${id.slice(0, 14)}…` : id;
}

export default function Overview() {
  const { currency, setView, refreshTick, bump } = useApp();
  const [prod, setProd] = useState<ProductionInfo | null>(null);
  const [latest, setLatest] = useState<MonitoringRecord | null>(null);
  const [recentPreds, setRecentPreds] = useState<number>(0);
  const [promos, setPromos] = useState<PromotionRecord[]>([]);
  const [monHistory, setMonHistory] = useState<MonitoringRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [driftRunning, setDriftRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      try {
        const [p, mh, ph, rh] = await Promise.all([
          api.production().catch(() => null),
          api.monitoringHistory(5, 0).catch(() => ({ items: [] })),
          api.predictionHistory(100, 0).catch(() => ({ items: [] })),
          api.retrainHistory(10, 0).catch(() => ({ items: [] })),
        ]);
        if (!alive) return;
        setProd(p);
        const mItems = (mh as { items: MonitoringRecord[] }).items ?? [];
        setMonHistory(mItems);
        setLatest(mItems[0] ?? null);
        setRecentPreds(((ph as { items: unknown[] }).items ?? []).length);
        setPromos(((rh as { items: PromotionRecord[] }).items ?? []));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [refreshTick]);

  const driftedCount = latest ? latest.features.filter((f) => f.drifted).length : 0;
  const totalFeatures = latest ? latest.features.length || 0 : 0;
  const r2 = prod?.metrics.r2 ?? null;
  const mae = prod?.metrics.mae ?? null;
  const rmse = prod?.metrics.rmse ?? null;
  const health = latest?.overall_health ?? null;
  const severity = latest?.severity ?? null;
  const dq = latest?.data_quality_status ?? null;

  const runDrift = async () => {
    setDriftRunning(true);
    setError(null);
    try {
      await api.driftCheck();
      bump();
    } catch (e) {
      setError(`Drift check failed: ${(e as Error).message}`);
    } finally {
      setDriftRunning(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Overview"
        desc="Command center for the House Price MLOps lifecycle — production model, prediction activity, and monitoring health."
        right={
          <>
            <button className="aa-btn-primary" onClick={() => setView('PREDICTIONS')}>
              <span>◉</span> New Prediction
            </button>
            <button className="aa-btn-ghost" onClick={runDrift} disabled={driftRunning}>
              <span>⟳</span> {driftRunning ? 'Checking…' : 'Run Drift Check'}
            </button>
          </>
        }
      />
      {error && <div role="alert" className="mb-3 text-[12px] text-red-400">{error}</div>}

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        <MetricCard label="PRODUCTION MODEL" topRight={<span className="text-aa-blueText">serving authority</span>}>
          <div className="flex items-center gap-2">
            <span className="text-[16px] font-bold text-white">{prod?.model_name ?? 'Unknown'}</span>
            <span className="aa-mono text-[10.5px] px-1.5 py-[2px] rounded border border-aa-blue/40 bg-aa-blue/10 text-aa-blueText">
              {prod?.model_version ? `v${prod.model_version}` : 'no version'}
            </span>
          </div>
          <div className="mt-1.5 text-[11.5px] text-aa-muted flex items-center gap-1.5">
            {prod ? (
              <>
                <span className="h-1.5 w-1.5 rounded-full bg-green-400" /> Serving Active
                <span className="aa-mono text-aa-faint truncate">· {shortId(prod.run_id)}</span>
              </>
            ) : (
              <span className="text-aa-faint">No production model deployed</span>
            )}
          </div>
        </MetricCard>
        <MetricCard label="MODEL PERFORMANCE" topRight={<span className="text-green-400">R² Metric</span>}>
          <div className="text-[22px] font-bold text-white aa-mono">{formatNum(r2, 4)}</div>
          <div className="mt-1 aa-mono text-[11px] text-aa-muted">
            MAE: {money(mae, currency)} <span className="text-aa-faint mx-1">·</span> RMSE: {money(rmse, currency)}
          </div>
        </MetricCard>
        <MetricCard label="MONITORING STATUS" topRight={<span className="aa-mono">α = 0.05</span>}>
          {!latest ? (
            <div className="text-[16px] font-bold text-aa-faint">Unknown</div>
          ) : (
            <div className={`text-[16px] font-bold flex items-center gap-2 ${latest.overall_drifted ? 'text-amber-400' : 'text-green-400'}`}>
              <span className={`h-2 w-2 rounded-full ${latest.overall_drifted ? 'bg-amber-400' : 'bg-green-400'}`} />
              {latest.overall_drifted ? 'Drift Detected' : 'No Drift'}
            </div>
          )}
          <div className="mt-1 aa-mono text-[11px] text-aa-muted">
            {latest ? `${driftedCount} of ${totalFeatures} features drifted · ${latest.production_rows} rows` : 'No monitoring runs yet'}
          </div>
        </MetricCard>
        <MetricCard label="PREDICTION ACTIVITY" topRight={<span>recent window</span>}>
          <div className="text-[22px] font-bold text-white">
            {recentPreds} <span className="text-[12px] font-normal text-aa-muted">records</span>
          </div>
          <div className="mt-1 flex items-center justify-between text-[11px]">
            <span className="aa-mono text-aa-muted">Latest 100 predictions</span>
            <button onClick={() => setView('PREDICTIONS')} className="text-aa-blueText hover:underline">
              Logs →
            </button>
          </div>
        </MetricCard>
      </div>

      <div className="mt-3 grid grid-cols-1 xl:grid-cols-3 gap-3">
        <div className="xl:col-span-2">
          <SectionCard
            title="Production Model Specification"
            subtitle={prod ? `${prod.registered_model_name} · Version ${prod.model_version ? `v${prod.model_version}` : '—'}` : 'No production model deployed'}
            right={
              <>
                {prod && <StatusBadge tone="green">● Production</StatusBadge>}
                <button onClick={() => setView('REGISTRY')} className="aa-btn-ghost !py-1 !px-2.5 !text-[11.5px]">
                  Lineage
                </button>
              </>
            }
          >
            {!prod ? (
              <div className="aa-mono text-[12px] text-aa-faint">
                {loading ? 'Loading production spec…' : 'No production model is deployed. Deploy one explicitly before serving predictions.'}
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                  <div className="rounded-lg border border-aa-border bg-[#0a1222] p-3">
                    <div className="text-[9.5px] tracking-[0.08em] text-aa-faint font-semibold">ALGORITHM FAMILY</div>
                    <div className="mt-1 text-[13px] font-semibold text-white">{prod.algorithm_family}</div>
                    <div className="aa-mono text-[10.5px] text-aa-faint">scikit-learn {prod.sklearn_version}</div>
                  </div>
                  <div className="rounded-lg border border-aa-border bg-[#0a1222] p-3">
                    <div className="text-[9.5px] tracking-[0.08em] text-aa-faint font-semibold">MLFLOW RUN ID</div>
                    <div className="mt-1 aa-mono text-[12px] text-aa-blueText truncate" title={prod.run_id ?? ''}>
                      {prod.run_id ?? '—'}
                    </div>
                    <div className="aa-mono text-[10.5px] text-aa-faint truncate">use case: {prod.use_case}</div>
                  </div>
                  <div className="rounded-lg border border-aa-border bg-[#0a1222] p-3">
                    <div className="text-[9.5px] tracking-[0.08em] text-aa-faint font-semibold">TRAINING DATASET</div>
                    <div className="mt-1 aa-mono text-[12px] text-white">
                      {prod.training_dataset.rows !== null ? `${prod.training_dataset.rows.toLocaleString()} rows` : '—'}
                    </div>
                    <div className="aa-mono text-[10.5px] text-aa-faint truncate">{prod.training_dataset.reference ?? '—'}</div>
                  </div>
                </div>

                <div className="mt-3 rounded-lg border border-aa-border bg-[#0a1222]/60 p-3">
                  <div className="flex items-center justify-between">
                    <div className="text-[12px] font-semibold text-white">Validation Benchmark Metrics</div>
                    <div className="aa-mono text-[10.5px] text-aa-faint">Offline Holdout Evaluation</div>
                  </div>
                  <div className="mt-2.5 grid grid-cols-1 md:grid-cols-3 gap-2.5">
                    <div className="rounded-md border border-aa-border bg-[#0c1425] p-3 text-center">
                      <div className="text-[9.5px] tracking-[0.08em] text-aa-faint">R² COEFFICIENT</div>
                      <div className="mt-1 text-[15px] font-bold text-aa-blueText aa-mono">{formatNum(r2, 4)}</div>
                    </div>
                    <div className="rounded-md border border-aa-border bg-[#0c1425] p-3 text-center">
                      <div className="text-[9.5px] tracking-[0.08em] text-aa-faint">MEAN ABSOLUTE ERROR</div>
                      <div className="mt-1 text-[15px] font-bold text-white aa-mono">{money(mae, currency)}</div>
                    </div>
                    <div className="rounded-md border border-aa-border bg-[#0c1425] p-3 text-center">
                      <div className="text-[9.5px] tracking-[0.08em] text-aa-faint">ROOT MEAN SQ ERROR</div>
                      <div className="mt-1 text-[15px] font-bold text-white aa-mono">{money(rmse, currency)}</div>
                    </div>
                  </div>
                </div>

                <div className="mt-3">
                  <div className="text-[12px] font-semibold text-white">Input Feature Signatures ({prod.features.length} Features)</div>
                  {prod.features.length === 0 ? (
                    <div className="mt-2 aa-mono text-[11px] text-aa-faint">Feature schema unavailable.</div>
                  ) : (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {prod.features.map((f) => (
                        <span key={f.name} className="aa-mono text-[11px] px-2.5 py-1 rounded-md border border-aa-border bg-[#0a1222] text-aa-text">
                          {f.name}: <span className="text-aa-faint">{f.dtype}</span>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </SectionCard>
        </div>

        <div>
          <SectionCard
            title="Model Health Guard"
            right={
              <button onClick={() => setView('MONITORING')} className="text-[11.5px] text-aa-blueText hover:underline">
                Full Observability →
              </button>
            }
          >
            <KeyVal
              k="Monitoring Status"
              v={
                !latest ? <span className="text-aa-faint">Unknown</span>
                : health === 'healthy' ? <span className="text-green-400 flex items-center gap-1.5">● Healthy</span>
                : health === 'warning' ? <span className="text-amber-400 flex items-center gap-1.5">● Warning</span>
                : health === 'critical' ? <span className="text-red-400 flex items-center gap-1.5">● Critical</span>
                : <span className="text-aa-faint">Unknown</span>
              }
            />
            <KeyVal k="Last Drift Check" v={<span className="aa-mono">{latest ? timeUTC(latest.timestamp) : '—'}</span>} />
            <KeyVal k="Drifted Feature Count" v={<span className="aa-mono">{latest ? `${driftedCount} / ${totalFeatures}` : '—'}</span>} />
            <KeyVal k="Production Stream Evaluated" v={<span className="aa-mono">{latest ? `${latest.production_rows} rows` : '—'}</span>} />
            <KeyVal k="Severity / Quality" v={<span className="aa-mono">{latest ? `${severity ?? '—'} / ${dq ?? '—'}` : '—'}</span>} />
            <KeyVal k="Multiple Testing Correction" v={<span className="aa-mono">Holm-Bonferroni (α=0.05)</span>} />
            <div className="mt-3">
              <InfoPanel>
                {!latest
                  ? 'No monitoring runs yet — run a drift check to establish the production baseline.'
                  : latest.overall_drifted
                    ? `Drift detected in ${driftedCount} of ${totalFeatures} monitored features. Review Monitoring before retraining.`
                    : 'Production feature distributions adhere to the reference baseline. No automated retraining triggered.'}
              </InfoPanel>
            </div>
          </SectionCard>
        </div>
      </div>

      <div className="mt-3">
        <SectionCard
          title="Recent Lifecycle Activity"
          subtitle="real promotion + monitoring events, newest first"
          right={
            <button onClick={() => setView('AUDIT')} className="text-[11.5px] text-aa-blueText hover:underline">
              Full Audit Log →
            </button>
          }
        >
          {promos.length === 0 && monHistory.length === 0 ? (
            <div className="aa-mono text-[12px] text-aa-faint !py-4 text-center">
              {loading ? 'Loading live state…' : 'No lifecycle events yet.'}
            </div>
          ) : (
            <div className="divide-y divide-white/[0.04]">
              {promos.slice(0, 3).map((r) => (
                <div key={r.id} className="py-3 flex flex-wrap items-start gap-3">
                  <StatusBadge tone={r.decision === 'promoted' ? 'green' : r.decision === 'rejected' ? 'amber' : 'gray'}>
                    {r.decision === 'promoted' ? '● Promoted' : r.decision === 'rejected' ? '● Rejected' : `● ${r.decision}`}
                  </StatusBadge>
                  <div className="flex-1 min-w-[220px]">
                    <div className="text-[12.5px] font-semibold text-white">
                      {r.model_name}{r.model_version ? ` v${r.model_version}` : ''} → {r.decision}
                    </div>
                    <div className="text-[11.5px] text-aa-muted">{r.reason.slice(0, 160)}</div>
                  </div>
                  <div className="aa-mono text-[11px] text-aa-faint">{timeUTC(r.timestamp)}</div>
                </div>
              ))}
              {monHistory.slice(0, 2).map((m) => (
                <div key={m.id} className="py-3 flex flex-wrap items-start gap-3">
                  <StatusBadge tone={m.overall_drifted ? 'amber' : 'green'}>
                    {m.overall_drifted ? '● Drift Detected' : '● Drift Check'}
                  </StatusBadge>
                  <div className="flex-1 min-w-[220px]">
                    <div className="text-[12.5px] font-semibold text-white">Drift analysis {m.status}</div>
                    <div className="text-[11.5px] text-aa-muted">
                      {m.production_rows} production rows · {m.features.filter((f) => f.drifted).length}/{m.features.length || 0} features drifted
                    </div>
                  </div>
                  <div className="aa-mono text-[11px] text-aa-faint">{timeUTC(m.timestamp)}</div>
                </div>
              ))}
            </div>
          )}
        </SectionCard>
      </div>
    </div>
  );
}
