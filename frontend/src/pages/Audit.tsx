import { useEffect, useMemo, useState } from 'react';
import { api, formatINR } from '../api/client';
import { useApp } from '../state/store';
import type { DemoSession, MonitoringRecord, PredictionRecord, PromotionRecord } from '../types';
import { PageHeader, SectionCard, StatusBadge } from '../components/ui';

interface Row {
  ts: string;
  event: string;
  rawEvent: string;
  kind: 'production' | 'demo';
  model: string;
  version: string;
  actor: string;
  details: string;
  status: 'ok' | 'warn' | 'info';
}

const EVENT_LABELS: Record<string, string> = {
  PREDICT: 'Prediction served',
  DRIFT_CHECK: 'Drift check',
  DRIFT_DETECTED: 'Drift detected',
  MODEL_PROMOTED: 'Model promoted',
  PROMOTION_REJECTED: 'Promotion rejected',
  RETRAIN_CYCLE: 'Retraining cycle',
  DEMO_SESSION: 'Demo session',
};

function fullTs(iso: string): string {
  try {
    const d = new Date(iso);
    const date = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    return `${date}, ${d.toISOString().slice(11, 19)} UTC`;
  } catch {
    return iso;
  }
}

export default function Audit() {
  const { refreshTick } = useApp();
  const [preds, setPreds] = useState<PredictionRecord[]>([]);
  const [mons, setMons] = useState<MonitoringRecord[]>([]);
  const [promos, setPromos] = useState<PromotionRecord[]>([]);
  const [demos, setDemos] = useState<DemoSession[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      try {
        const [p, m, r, d] = await Promise.all([
          api.predictionHistory(50, 0).catch(() => ({ items: [] })),
          api.monitoringHistory(50, 0).catch(() => ({ items: [] })),
          api.retrainHistory(50, 0).catch(() => ({ items: [] })),
          api.demoSessions(20, 0).catch(() => ({ items: [] })),
        ]);
        if (!alive) return;
        setPreds((p as { items: PredictionRecord[] }).items);
        setMons((m as { items: MonitoringRecord[] }).items);
        setPromos((r as { items: PromotionRecord[] }).items);
        setDemos((d as { items: DemoSession[] }).items);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [refreshTick]);

  const rows: Row[] = useMemo(() => {
    const out: Row[] = [];
    const label = (raw: string) => EVENT_LABELS[raw] ?? raw;
    preds.forEach((p) => {
      const d = p.input_data ?? {};
      const loc = typeof d.location === 'string' ? d.location : null;
      const area =
        typeof d.area_sqft === 'number' ? d.area_sqft
        : typeof d.area === 'number' ? d.area : null;
      const legacy = !loc || area === null;
      const summary = legacy
        ? area !== null
          ? `Legacy prediction · ${area.toLocaleString()} sqft`
          : 'Legacy prediction'
        : `${loc} · ${area.toLocaleString()} sqft · ${d.bedrooms ?? '?'}b/${d.bathrooms ?? '?'}ba`;
      out.push({
        ts: p.timestamp,
        event: label('PREDICT'),
        rawEvent: 'PREDICT',
        kind: 'production',
        model: p.model_name || '—',
        version: p.model_version ? `v${p.model_version}` : '—',
        actor: 'serving-api',
        details: `${summary} → ${formatINR(p.prediction)}`,
        status: 'ok',
      });
    });
    mons.forEach((m) => {
      const raw = m.overall_drifted ? 'DRIFT_DETECTED' : 'DRIFT_CHECK';
      out.push({
        ts: m.timestamp,
        event: label(raw),
        rawEvent: raw,
        kind: 'production',
        model: '—',
        version: '—',
        actor: 'monitoring-svc',
        details: `${m.production_rows} rows · ${m.features.filter((f) => f.drifted).length}/${m.features.length || 7} drifted · ${m.status}`,
        status: m.overall_drifted ? 'warn' : 'ok',
      });
    });
    promos.forEach((r) => {
      const raw = r.decision === 'promoted' ? 'MODEL_PROMOTED' : r.decision === 'rejected' ? 'PROMOTION_REJECTED' : 'RETRAIN_CYCLE';
      out.push({
        ts: r.timestamp,
        event: label(raw),
        rawEvent: raw,
        kind: 'production',
        model: r.model_name,
        version: `v${r.model_version ?? '—'}`,
        actor: 'retraining-svc',
        details: r.reason.slice(0, 160),
        status: r.decision === 'promoted' ? 'ok' : r.decision === 'rejected' ? 'warn' : 'info',
      });
    });
    demos.forEach((s) =>
      out.push({
        ts: s.created_at,
        event: label('DEMO_SESSION'),
        rawEvent: 'DEMO_SESSION',
        kind: 'demo',
        model: '—',
        version: '—',
        actor: 'demo-simulator',
        details: `${s.session_id} · ${s.scenario} · ${s.samples_generated} demo samples · ${s.status}`.slice(0, 160),
        status: 'info',
      }),
    );
    // No fabricated demo rows: an empty trail renders as empty.
    if (out.length === 0) {
      return [];
    }
    return out.sort((a, b) => (a.ts < b.ts ? 1 : -1)).slice(0, 60);
  }, [preds, mons, promos, demos]);

  return (
    <div>
      <PageHeader
        title="Audit History"
        desc="Chronological, immutable trail of predictions, drift checks, retraining cycles and promotions."
        right={<span className="aa-mono text-[11px] text-aa-faint">{rows.length} events · {loading ? 'loading…' : 'live from SQLite + MLflow'}</span>}
      />
      <SectionCard title="Lifecycle Event Trail" subtitle="timestamp · event · source · model · version · actor · details · status">
        <div className="overflow-x-auto -mx-4 px-4">
          <table className="aa-table w-full min-w-[980px]">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Event</th>
                <th>Source</th>
                <th>Model</th>
                <th>Version</th>
                <th>Actor / Source</th>
                <th>Details</th>
                <th>Status</th>
              </tr>
            </thead>
              <tbody>
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={8} className="aa-mono text-aa-faint !py-6 text-center">
                      No lifecycle events yet.
                    </td>
                  </tr>
                )}
                {rows.map((r, i) => (
                <tr key={`${r.ts}-${i}`}>
                  <td className="aa-mono text-aa-faint whitespace-nowrap">{fullTs(r.ts)}</td>
                  <td>
                    <span className="text-[11.5px] text-aa-text whitespace-nowrap" title={`raw event: ${r.rawEvent}`}>
                      {r.event}
                    </span>
                    <div className="aa-mono text-[9.5px] text-aa-faint whitespace-nowrap">{r.rawEvent}</div>
                  </td>
                  <td>
                    {r.kind === 'demo' ? (
                      <StatusBadge tone="amber">● Demo</StatusBadge>
                    ) : (
                      <StatusBadge tone="blue">● Production</StatusBadge>
                    )}
                  </td>
                  <td className="whitespace-nowrap">{r.model}</td>
                  <td className="aa-mono whitespace-nowrap">{r.version}</td>
                  <td className="aa-mono text-aa-muted whitespace-nowrap">{r.actor}</td>
                  <td className="text-aa-muted max-w-[420px] truncate" title={r.details}>
                    {r.details}
                  </td>
                  <td>
                    {r.status === 'ok' ? (
                      <StatusBadge tone="green">● ok</StatusBadge>
                    ) : r.status === 'warn' ? (
                      <StatusBadge tone="amber">● review</StatusBadge>
                    ) : (
                      <StatusBadge tone="blue">● info</StatusBadge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}
