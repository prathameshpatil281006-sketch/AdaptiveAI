import { useEffect, useMemo, useState } from 'react';
import { api, timeUTC } from '../api/client';
import { useApp } from '../state/store';
import type { DistributionFeature, DriftFeature, HealthSummary, MonitoringRecord } from '../types';
import { InfoPanel, MetricCard, PageHeader, SectionCard, StatusBadge } from '../components/ui';

const FEATURE_LABELS: Record<string, string> = {
  area_sqft: 'Living Space Area (sqft)',
  bedrooms: 'Bedrooms',
  bathrooms: 'Bathrooms',
  parking: 'Parking Spaces',
  property_age: 'Property Age (years)',
  floor_number: 'Floor Number',
  total_floors: 'Total Floors',
  location: 'Locality',
  furnishing_status: 'Furnishing Status',
  property_type: 'Property Type',
};

type Tone = 'green' | 'red' | 'amber' | 'blue' | 'gray';

function severityTone(s: string | null | undefined): Tone {
  if (s === 'healthy' || s === 'pass') return 'green';
  if (s === 'warning' || s === 'warn') return 'amber';
  if (s === 'critical' || s === 'fail') return 'red';
  if (s === 'ok') return 'green';
  return 'gray';
}

function healthLabel(h: string | null | undefined): string {
  if (h === 'healthy') return 'Healthy';
  if (h === 'warning') return 'Warning';
  if (h === 'critical') return 'Critical';
  return 'Unknown';
}

function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits);
}

function evidenceOf(f: DriftFeature): string {
  if (f.type === 'categorical') {
    const chi = fmtNum(f.statistic, 2);
    const p = fmtNum(f.p_value_corrected ?? f.p_value);
    const psi = fmtNum(f.effect_size, 3);
    return `χ²=${chi} · p=${p} · PSI=${psi}`;
  }
  const d = fmtNum(f.statistic);
  const p = fmtNum(f.p_value_corrected ?? f.p_value);
  return `D=${d} · p=${p}`;
}

function blockSeverity(features: DriftFeature[]): string {
  if (features.some((f) => f.severity === 'critical')) return 'critical';
  if (features.some((f) => f.severity === 'warning')) return 'warning';
  if (features.length > 0 && features.every((f) => f.severity === 'healthy')) return 'healthy';
  return 'unknown';
}

export default function Monitoring() {
  const { refreshTick, bump, setView } = useApp();
  const [summary, setSummary] = useState<HealthSummary | null>(null);
  const [dists, setDists] = useState<DistributionFeature[]>([]);
  const [history, setHistory] = useState<MonitoringRecord[]>([]);
  const [tab, setTab] = useState('area_sqft');
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const [s, d, h] = await Promise.all([
        api.healthSummary(1000).catch(() => null),
        api.distributions(1000).catch(() => null),
        api.monitoringHistory(20, 0).catch(() => null),
      ]);
      if (s) setSummary(s);
      if (d) setDists(d.features);
      const items = (h as { items: MonitoringRecord[] } | null)?.items;
      if (items) setHistory(items);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  useEffect(() => {
    setLoading(true);
    load().finally(() => setLoading(false));
  }, [refreshTick]);

  const runCheck = async () => {
    setRunning(true);
    setError(null);
    try {
      const s = await api.healthSummary(1000);
      setSummary(s);
      bump();
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const distByFeature = useMemo(() => {
    const m: Record<string, DistributionFeature> = {};
    dists.forEach((d) => {
      m[d.feature] = d;
    });
    return m;
  }, [dists]);

  const features = useMemo(() => summary?.features ?? [], [summary]);
  const numeric = useMemo(() => features.filter((f) => f.type !== 'categorical'), [features]);
  const categorical = useMemo(() => features.filter((f) => f.type === 'categorical'), [features]);
  const numericDrifted = numeric.filter((f) => f.drifted).length;
  const categoricalDrifted = categorical.filter((f) => f.drifted).length;

  const tabFeature = features.find((f) => f.feature_name === tab);
  const isCategoricalTab = tabFeature?.type === 'categorical';
  const active = distByFeature[tab];

  const maxCount = useMemo(() => {
    if (!active || active.type === 'categorical') return 1;
    return Math.max(1, ...active.reference.histogram.counts, ...active.production.histogram.counts);
  }, [active]);

  const maxShare = useMemo(() => {
    if (!active?.categories?.length) return 1;
    return Math.max(0.01, ...active.categories.map((c) => Math.max(c.reference_share, c.production_share)));
  }, [active]);

  const dq = summary?.data_quality ?? null;

  return (
    <div>
      <PageHeader
        title="ML Health Center"
        desc="Prediction traffic, distribution drift, and data quality for the deployed valuation model."
        right={
          <>
            <button onClick={() => setView('SIMULATOR')} className="aa-btn-ghost !border-aa-blue/40 !text-aa-blueText hover:!border-aa-blue">
              <span>✦</span> Run Scenario Simulator →
            </button>
            <button onClick={() => setView('SIMULATOR')} className="aa-btn-ghost !border-amber-500/40 !text-amber-300 hover:!border-amber-400">
              <span>✦</span> Simulate Drift (Isolated Demo)
            </button>
            <button onClick={runCheck} disabled={running} className="aa-btn-primary">
              <span>⟳</span> {running ? 'Checking…' : 'Run Health Check'}
            </button>
          </>
        }
      />
      {error && <div className="mb-3 text-[12px] text-red-400">{error}</div>}

      {loading && !summary ? (
        <SectionCard title="Health Guard">
          <div className="aa-mono text-[12px] text-aa-faint">Loading health summary…</div>
        </SectionCard>
      ) : !summary ? (
        <SectionCard title="Health Guard">
          <div className="aa-mono text-[12px] text-aa-faint">No health data yet — run a health check.</div>
        </SectionCard>
      ) : (
        <>
          <SectionCard
            title="Health Guard"
            subtitle={`Evaluated ${summary.timestamp ? timeUTC(summary.timestamp) : '—'}`}
            right={<StatusBadge tone={severityTone(summary.overall_health)}>● {healthLabel(summary.overall_health)}</StatusBadge>}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
              <MetricCard label="OVERALL HEALTH" topRight={<span>{summary.status}</span>}>
                <div className="text-[15px] font-bold text-white">{healthLabel(summary.overall_health)}</div>
                <div className="mt-1 text-[11px] text-aa-muted">{summary.health_reason}</div>
              </MetricCard>
              <MetricCard label="MONITORED MODEL" topRight={<span>serving identity</span>}>
                <div className="text-[15px] font-bold text-white aa-mono">{summary.model_name ?? '—'}</div>
                <div className="mt-1 aa-mono text-[11px] text-aa-muted">
                  {summary.model_version ? `v${summary.model_version}` : 'no version'} · schema {summary.schema_version !== null ? `v${summary.schema_version}` : '—'}
                </div>
              </MetricCard>
              <MetricCard label="PREDICTION TRAFFIC" topRight={<span>window</span>}>
                <div className="text-[20px] font-bold text-white aa-mono">
                  {summary.window_v2_records} <span className="text-[11px] font-normal text-aa-faint">/ {summary.total_predictions ?? '—'} total</span>
                </div>
                <div className="mt-1 aa-mono text-[11px] text-aa-muted">
                  {summary.window_records} window rows · {summary.window_legacy_records} legacy · {summary.window_malformed_records} malformed
                </div>
              </MetricCard>
              <MetricCard label="DRIFT & QUALITY" topRight={<span>{summary.total_monitored} features</span>}>
                <div className="text-[20px] font-bold text-white aa-mono">
                  {summary.drifted_count} <span className="text-[11px] font-normal text-aa-faint">drifted</span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <StatusBadge tone={severityTone(summary.severity)}>● drift: {summary.severity}</StatusBadge>
                  <StatusBadge tone={severityTone(summary.data_quality_status)}>● quality: {summary.data_quality_status}</StatusBadge>
                </div>
              </MetricCard>
            </div>
          </SectionCard>

          <div className="mt-3">
            <SectionCard
              title="Drift Overview"
              subtitle="Numeric KS + Holm-Bonferroni · categorical chi-square + Holm-Bonferroni"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="rounded-lg border border-aa-border bg-[#0a1222] p-3">
                  <div className="flex items-center justify-between">
                    <div className="text-[11px] tracking-[0.08em] font-semibold text-aa-faint">NUMERIC DRIFT</div>
                    <StatusBadge tone={severityTone(blockSeverity(numeric))}>● {blockSeverity(numeric)}</StatusBadge>
                  </div>
                  <div className="mt-2 aa-mono text-[13px] text-white">
                    {numericDrifted} / {numeric.length} drifted
                  </div>
                  <div className="mt-1 text-[11px] text-aa-muted">
                    Two-sample Kolmogorov-Smirnov per feature, Holm step-down at α={summary.alpha}. Reference n={summary.reference_rows}, current n={summary.production_rows}.
                  </div>
                </div>
                <div className="rounded-lg border border-aa-border bg-[#0a1222] p-3">
                  <div className="flex items-center justify-between">
                    <div className="text-[11px] tracking-[0.08em] font-semibold text-aa-faint">CATEGORICAL / LOCALITY DRIFT</div>
                    <StatusBadge tone={severityTone(blockSeverity(categorical))}>● {blockSeverity(categorical)}</StatusBadge>
                  </div>
                  <div className="mt-2 aa-mono text-[13px] text-white">
                    {categoricalDrifted} / {categorical.length} drifted
                  </div>
                  <div className="mt-1 text-[11px] text-aa-muted">
                    Chi-square homogeneity per feature with rare-bucket merging, PSI effect size. Locality mix monitored separately from price benchmarks.
                  </div>
                </div>
              </div>
            </SectionCard>
          </div>

          <div className="mt-3">
            <SectionCard
              title="Feature Drift Breakdown"
              subtitle="Status, severity, and statistical evidence per monitored feature"
            >
              <div className="overflow-x-auto -mx-4 px-4">
                <table className="aa-table w-full min-w-[900px]">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Type</th>
                      <th>Status</th>
                      <th>Severity</th>
                      <th>Evidence</th>
                      <th>Ref N</th>
                      <th>Current N</th>
                    </tr>
                  </thead>
                  <tbody>
                    {features.map((f) => (
                      <tr key={f.feature_name}>
                        <td>
                          <span className="flex items-center gap-2 text-aa-text">
                            {f.feature_name === tab && <span className="h-1.5 w-1.5 rounded-full bg-aa-blueText" />}
                            {FEATURE_LABELS[f.feature_name] ?? f.feature_name}
                          </span>
                        </td>
                        <td className="aa-mono text-aa-muted">{f.type ?? 'numeric'}</td>
                        <td>
                          {(f.status ?? 'ok') !== 'ok' ? (
                            <StatusBadge tone="gray">● No data</StatusBadge>
                          ) : f.drifted ? (
                            <StatusBadge tone="red">● Drifted</StatusBadge>
                          ) : (
                            <StatusBadge tone="green">● In Bounds</StatusBadge>
                          )}
                        </td>
                        <td>
                          <StatusBadge tone={severityTone(f.severity)}>● {f.severity ?? 'unknown'}</StatusBadge>
                        </td>
                        <td className="aa-mono text-aa-muted">{evidenceOf(f)}</td>
                        <td className="aa-mono text-aa-muted">{summary.reference_rows.toLocaleString()}</td>
                        <td className="aa-mono text-aa-muted">{summary.production_rows.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-3">
                <InfoPanel>
                  <span>
                    <strong className="text-white">Reading this table:</strong> drift flags come from corrected p-values at α={summary.alpha} (Holm-Bonferroni within each family). Severity reflects effect size (KS D ≥ 0.30 or PSI ≥ 0.25 reads as large) and spread — it is distribution-shift severity, not a model-quality score.
                  </span>
                </InfoPanel>
              </div>
            </SectionCard>
          </div>

          <div className="mt-3">
            <SectionCard
              title={`Distribution Comparison: ${FEATURE_LABELS[tab] ?? tab}`}
              subtitle="Reference population vs current production window"
              right={
                <div className="flex flex-wrap justify-end gap-1">
                  {features.map((f) => (
                    <button
                      key={f.feature_name}
                      onClick={() => setTab(f.feature_name)}
                      className={`aa-mono text-[10.5px] px-2.5 py-1 rounded-md border ${
                        tab === f.feature_name ? 'bg-aa-blue text-white border-aa-blue' : 'border-aa-border text-aa-muted hover:text-white'
                      }`}
                    >
                      {f.feature_name}
                    </button>
                  ))}
                </div>
              }
            >
              {!active ? (
                <div className="aa-mono text-[12px] text-aa-faint">No distribution data for this feature yet.</div>
              ) : isCategoricalTab ? (
                <div className="space-y-1.5">
                  {(active.categories ?? []).map((c) => (
                    <div key={c.value} className="flex items-center gap-2.5">
                      <div className="w-32 shrink-0 truncate text-[11.5px] text-aa-text" title={c.value}>{c.value}</div>
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                            <div className="h-full rounded-full bg-slate-500/70" style={{ width: `${Math.max(2, (c.reference_share / maxShare) * 100)}%` }} />
                          </div>
                          <div className="w-40 shrink-0 aa-mono text-[10px] text-aa-muted">ref {(c.reference_share * 100).toFixed(1)}% · {c.reference_count.toLocaleString()}</div>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                            <div className="h-full rounded-full bg-aa-blue/80" style={{ width: `${Math.max(2, (c.production_share / maxShare) * 100)}%` }} />
                          </div>
                          <div className="w-40 shrink-0 aa-mono text-[10px] text-aa-blueText">cur {(c.production_share * 100).toFixed(1)}% · {c.production_count.toLocaleString()}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                  {(!active.categories || active.categories.length === 0) && (
                    <div className="aa-mono text-[11px] text-aa-faint">No category data — insufficient current production rows.</div>
                  )}
                </div>
              ) : (
                <div className="rounded-lg border border-aa-border bg-[#080e1b] p-3">
                  <div className="flex flex-wrap items-center gap-3 aa-mono text-[10.5px]">
                    <span className="flex items-center gap-1.5 text-aa-muted">
                      <span className="h-2.5 w-2.5 rounded bg-slate-500" /> Reference (μ = {active.reference.mean?.toFixed(2) ?? '—'})
                    </span>
                    <span className="flex items-center gap-1.5 text-aa-blueText">
                      <span className="h-2.5 w-2.5 rounded bg-aa-blue" /> Current (μ = {active.production.mean?.toFixed(2) ?? '—'})
                    </span>
                  </div>
                  <div className="mt-3 h-[150px] flex items-end gap-[2px]">
                    {(() => {
                      const rc = active.reference.histogram.counts ?? [];
                      const pc = active.production.histogram.counts ?? [];
                      const n = Math.max(rc.length, pc.length, 20);
                      if (rc.length === 0) {
                        return (
                          <div key="empty" className="w-full h-full flex items-center justify-center aa-mono text-[11px] text-aa-faint">
                            No distribution data yet — run predictions, then re-check.
                          </div>
                        );
                      }
                      return Array.from({ length: n }).map((_, i) => {
                        const hR = ((rc[i] ?? 0) / maxCount) * 100;
                        const hP = ((pc[i] ?? 0) / maxCount) * 100;
                        return (
                          <div key={i} className="flex-1 flex items-end justify-center gap-[2px] h-full">
                            <div className="w-full rounded-t-[2px] bg-slate-600/60" style={{ height: `${Math.max(2, hR)}%` }} />
                            <div className="w-full rounded-t-[2px] bg-aa-blue/80" style={{ height: `${Math.max(2, hP)}%` }} />
                          </div>
                        );
                      });
                    })()}
                  </div>
                </div>
              )}
            </SectionCard>
          </div>

          <div className="mt-3">
            <SectionCard
              title="Data Quality"
              subtitle="Row-level checks over the current window · read-only, nothing is deleted"
              right={<StatusBadge tone={severityTone(dq?.status)}>● {dq?.status ?? 'unknown'}</StatusBadge>}
            >
              {!dq || dq.checks.length === 0 ? (
                <div className="aa-mono text-[12px] text-aa-faint">No data-quality assessment available.</div>
              ) : (
                <>
                  <div className="aa-mono text-[11px] text-aa-muted">
                    {dq.v2_rows} v2 rows · {dq.legacy_rows} legacy · {dq.malformed_rows} malformed of {dq.total_rows} window rows
                  </div>
                  <div className="mt-2 overflow-x-auto -mx-4 px-4">
                    <table className="aa-table w-full min-w-[760px]">
                      <thead>
                        <tr>
                          <th>Check</th>
                          <th>Status</th>
                          <th>Checked</th>
                          <th>Affected</th>
                          <th>Message</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dq.checks.map((c) => (
                          <tr key={c.name}>
                            <td className="aa-mono text-aa-text">{c.name}</td>
                            <td>
                              <StatusBadge tone={severityTone(c.status)}>● {c.status}</StatusBadge>
                            </td>
                            <td className="aa-mono text-aa-muted">{c.checked}</td>
                            <td className="aa-mono text-aa-muted">{c.affected}</td>
                            <td className="text-aa-muted">{c.message}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </SectionCard>
          </div>

          <div className="mt-3">
            <SectionCard title="Monitoring History" subtitle="Persisted health runs, newest first">
              <div className="overflow-x-auto -mx-4 px-4">
                <table className="aa-table w-full min-w-[860px]">
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Status</th>
                      <th>Severity</th>
                      <th>Health</th>
                      <th>Drifted</th>
                      <th>Data Quality</th>
                      <th>Rows</th>
                      <th>Model</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.length === 0 && (
                      <tr>
                        <td colSpan={8} className="aa-mono text-aa-faint !py-6 text-center">
                          No monitoring runs yet — run a health check above.
                        </td>
                      </tr>
                    )}
                    {history.map((h) => {
                      const drifted = h.drifted_features ?? h.features.filter((f) => f.drifted).length;
                      const total = h.features.length || summary.total_monitored;
                      return (
                        <tr key={h.id}>
                          <td className="aa-mono text-aa-faint whitespace-nowrap">{timeUTC(h.timestamp)}</td>
                          <td className="aa-mono text-aa-muted">{h.status}</td>
                          <td>
                            <StatusBadge tone={severityTone(h.severity)}>● {h.severity ?? '—'}</StatusBadge>
                          </td>
                          <td>
                            <StatusBadge tone={severityTone(h.overall_health)}>● {h.overall_health ?? '—'}</StatusBadge>
                          </td>
                          <td className="aa-mono text-aa-muted">{drifted}/{total}</td>
                          <td>
                            <StatusBadge tone={severityTone(h.data_quality_status)}>● {h.data_quality_status ?? '—'}</StatusBadge>
                          </td>
                          <td className="aa-mono text-aa-muted">{h.production_rows}</td>
                          <td className="aa-mono text-aa-muted">
                            {h.model_name ?? '—'}{h.model_version ? ` v${h.model_version}` : ''}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </SectionCard>
          </div>
        </>
      )}
      <div className="mt-3">
        <InfoPanel>
          <span>
            This page observes <strong className="text-white">production traffic only</strong> — simulated rows (source=demo) are excluded from every chart and check here. For isolated demonstrations use the <strong className="text-white">Scenario Simulator</strong>, where drift samples never enter production history. Severity describes distribution shift and sample adequacy only, never model quality.
          </span>
        </InfoPanel>
      </div>
    </div>
  );
}
