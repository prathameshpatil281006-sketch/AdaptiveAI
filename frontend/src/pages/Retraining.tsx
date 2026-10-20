import { useEffect, useState } from 'react';
import { api, formatINR, formatNum, timeUTC } from '../api/client';
import { useApp } from '../state/store';
import type {
  BaselineComparison,
  DecisionResponse,
  GateCheck,
  PointerHistoryEntry,
  RegistryInfo,
  RetrainingRun,
} from '../types';
import { InfoPanel, PageHeader, SectionCard, StatusBadge } from '../components/ui';

type Tone = 'green' | 'red' | 'amber' | 'blue' | 'gray';

function runTone(status: string | null): Tone {
  if (status === 'completed') return 'green';
  if (status === 'failed') return 'red';
  if (status === 'started') return 'blue';
  return 'gray';
}

function money(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return formatINR(v);
}

function ComparisonTable({ comparison }: { comparison: BaselineComparison | null }) {
  if (!comparison) return <div className="aa-mono text-[12px] text-aa-faint">No baseline comparison recorded for this run.</div>;
  const prod = comparison.production;
  const cand = comparison.candidate;
  const d = comparison.deltas;
  const rows: { metric: string; p: string; c: string; delta: string }[] = [
    {
      metric: 'MAE (INR)',
      p: prod?.metrics ? money(prod.metrics.mae) : '—',
      c: cand?.metrics ? money(cand.metrics.mae) : '—',
      delta: d ? `${d.mae_delta >= 0 ? '+' : '−'}${money(Math.abs(d.mae_delta)).replace('−', '')}${d.mae_reduction_pct !== null ? ` (${d.mae_reduction_pct >= 0 ? '+' : '−'}${Math.abs(d.mae_reduction_pct).toFixed(1)}%)` : ''}` : '—',
    },
    {
      metric: 'RMSE (INR)',
      p: prod?.metrics ? money(prod.metrics.rmse) : '—',
      c: cand?.metrics ? money(cand.metrics.rmse) : '—',
      delta: d ? `${d.rmse_delta >= 0 ? '+' : '−'}${money(Math.abs(d.rmse_delta)).replace('−', '')}${d.rmse_reduction_pct !== null ? ` (${d.rmse_reduction_pct >= 0 ? '+' : '−'}${Math.abs(d.rmse_reduction_pct).toFixed(1)}%)` : ''}` : '—',
    },
    {
      metric: 'R²',
      p: prod?.metrics ? formatNum(prod.metrics.r2, 4) : '—',
      c: cand?.metrics ? formatNum(cand.metrics.r2, 4) : '—',
      delta: d ? `${d.r2_improvement >= 0 ? '+' : '−'}${Math.abs(d.r2_improvement).toFixed(4)}${d.r2_improvement_pct !== null ? ` (${d.r2_improvement_pct >= 0 ? '+' : '−'}${Math.abs(d.r2_improvement_pct).toFixed(2)}%)` : ''}` : '—',
    },
  ];
  return (
    <div className="overflow-x-auto -mx-4 px-4">
      <table className="aa-table w-full min-w-[560px]">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Production{prod?.model_name ? ` (${prod.model_name})` : ''}</th>
            <th>Candidate{cand?.model_name ? ` (${cand.model_name})` : ''}</th>
            <th className="!text-right">Delta</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.metric}>
              <td className="text-aa-text">{r.metric}</td>
              <td className="aa-mono text-aa-muted">{r.p}</td>
              <td className="aa-mono text-white">{r.c}</td>
              <td className="!text-right aa-mono text-aa-blueText">{r.delta}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GateList({ checks }: { checks: GateCheck[] | null }) {
  if (!checks || checks.length === 0)
    return <div className="aa-mono text-[12px] text-aa-faint">No gate evaluation recorded.</div>;
  return (
    <div className="space-y-1.5">
      {checks.map((c) => (
        <div key={c.name} className="flex items-start gap-2.5 rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
          <span className={`aa-mono text-[12px] font-bold ${c.passed ? 'text-green-400' : 'text-red-400'}`}>
            {c.passed ? '✓' : '✗'}
          </span>
          <div className="min-w-0">
            <div className="aa-mono text-[11.5px] text-aa-text">{c.name}</div>
            <div className="text-[11px] text-aa-muted">{c.message}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function Retraining() {
  const { refreshTick, bump, setView } = useApp();
  const [decision, setDecision] = useState<DecisionResponse | null>(null);
  const [runs, setRuns] = useState<RetrainingRun[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<RetrainingRun | null>(null);
  const [registry, setRegistry] = useState<RegistryInfo | null>(null);
  const [running, setRunning] = useState(false);
  const [checking, setChecking] = useState(false);
  const [promoting, setPromoting] = useState(false);
  const [rollingBack, setRollingBack] = useState(false);
  const [rollbackTarget, setRollbackTarget] = useState<number | ''>('');
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    try {
      const [d, r, g] = await Promise.all([
        api.lifecycleDecision(1000).catch(() => null),
        api.lifecycleRuns(20, 0).catch(() => ({ items: [] })),
        api.registry(100).catch(() => null),
      ]);
      if (d) setDecision(d);
      const items = (r as { items: RetrainingRun[] }).items ?? [];
      setRuns(items);
      if (!selectedId && items.length > 0) {
        setSelectedId(items[0].run_id);
        setDetail(items[0]);
      }
      if (g) setRegistry(g);
    } catch {
      /* keep previous */
    }
  };

  useEffect(() => {
    load();
  }, [refreshTick]);

  const refreshDetail = async (runId: string) => {
    try {
      const run = await api.lifecycleRun(runId);
      setDetail(run);
      setSelectedId(runId);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const startRun = async () => {
    if (!window.confirm('Start an explicit retraining run? This trains all candidates and records a run; it never promotes.')) return;
    setRunning(true);
    setError(null);
    setNotice(null);
    try {
      const run = await api.lifecycleStartRun(1000, 'manual');
      setDetail(run);
      setSelectedId(run.run_id);
      setNotice(`Run ${run.run_id} ${run.status}.`);
      bump();
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const runCheck = async () => {
    if (!selectedId) return;
    setChecking(true);
    setError(null);
    try {
      const checked = await api.lifecyclePromotionCheck(selectedId, 1000);
      await refreshDetail(selectedId);
      setNotice(
        checked.eligible
          ? `Run ${selectedId} is promotion-eligible.`
          : `Run ${selectedId} is not eligible; see gate checks.`,
      );
      bump();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setChecking(false);
    }
  };

  const promote = async () => {
    if (!detail) return;
    const from = `${decision?.decision.production_model ?? 'unknown model'} v${decision?.decision.production_version ?? '—'}`;
    if (!window.confirm(`This action will change the production model from ${from} to candidate '${detail.best_model ?? '—'}'. Production traffic will be served by the new model. Continue?`)) return;
    setPromoting(true);
    setError(null);
    try {
      const out = await api.lifecyclePromote(detail.run_id);
      await refreshDetail(detail.run_id);
      setNotice(out.promoted ? `Promoted: production is now v${out.production.model_version}.` : `Not promoted: ${out.reason}`);
      bump();
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPromoting(false);
    }
  };

  const rollback = async () => {
    if (rollbackTarget === '') return;
    if (!window.confirm(`Roll back production to pointer-history entry #${rollbackTarget}? The current artifact is preserved.`)) return;
    setRollingBack(true);
    setError(null);
    try {
      const out = await api.lifecycleRollback(Number(rollbackTarget));
      setNotice(`Rolled back: production is now v${out.production.model_version}.`);
      setRollbackTarget('');
      bump();
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRollingBack(false);
    }
  };

  const pointerHistory: PointerHistoryEntry[] = registry?.pointer_history ?? [];
  const eligible = (detail?.promotion_eligible ?? false) && detail?.status === 'completed';

  return (
    <div>
      <PageHeader
        title="Retraining Lifecycle"
        desc="Explicit, audited model lifecycle: recommendation → training run → baseline comparison → promotion gate → explicit promotion → rollback."
        right={
          <div className="flex items-center gap-2">
            <button onClick={() => setView('SIMULATOR')} className="aa-btn-ghost !border-aa-blue/40 !text-aa-blueText hover:!border-aa-blue">
              <span>✦</span> Run Scenario Simulator →
            </button>
            <button onClick={startRun} disabled={running} className="aa-btn-primary">
              <span>◉</span> {running ? 'Training…' : 'Start Retraining'}
            </button>
          </div>
        }
      />
      {error && <div role="alert" className="mb-3 text-[12px] text-red-400">{error}</div>}
      {notice && <div className="mb-3 text-[12px] text-green-400">{notice}</div>}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        <SectionCard title="Retraining Status" subtitle="Monitoring recommendation · read-only advice">
          {!decision ? (
            <div className="aa-mono text-[12px] text-aa-faint">No recommendation yet.</div>
          ) : (
            <div className="space-y-2.5">
              <div className="flex items-center gap-2">
                <StatusBadge tone={decision.decision.recommended ? 'blue' : 'gray'}>
                  ● {decision.decision.recommended ? 'Retraining recommended' : 'No retraining recommended'}
                </StatusBadge>
              </div>
              <div className="text-[12.5px] text-aa-text leading-relaxed">{decision.decision.reason}</div>
              {decision.decision.triggers.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {decision.decision.triggers.map((t) => (
                    <span key={t} className="aa-mono text-[10.5px] px-2 py-[3px] rounded-md border border-amber-500/30 bg-aa-amberSoft text-amber-300">{t}</span>
                  ))}
                </div>
              )}
              {decision.decision.blocked_by.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {decision.decision.blocked_by.map((b) => (
                    <span key={b} className="aa-mono text-[10.5px] px-2 py-[3px] rounded-md border border-red-500/30 bg-red-500/10 text-red-400">blocked: {b}</span>
                  ))}
                </div>
              )}
              <div className="rounded-lg border border-aa-border bg-[#0a1222] p-3 aa-mono text-[11px] space-y-1.5">
                <div className="flex justify-between"><span className="text-aa-faint">Production</span><span className="text-aa-text">{decision.decision.production_model ?? '—'}{decision.decision.production_version ? ` v${decision.decision.production_version}` : ''}</span></div>
                <div className="flex justify-between"><span className="text-aa-faint">Monitored predictions</span><span className="text-aa-text">{decision.decision.monitored_prediction_count ?? '—'}</span></div>
                <div className="flex justify-between"><span className="text-aa-faint">Monitor status</span><span className="text-aa-text">{decision.monitoring.status} · {decision.monitoring.severity ?? '—'} · DQ {decision.monitoring.data_quality_status ?? '—'}</span></div>
                <div className="flex justify-between"><span className="text-aa-faint">Drifted</span><span className="text-aa-text">{decision.monitoring.drifted_count ?? '—'}/{decision.monitoring.total_monitored ?? '—'}</span></div>
              </div>
            </div>
          )}
        </SectionCard>

        <SectionCard
          title="Latest Run"
          subtitle={detail ? `${detail.run_id} · ${detail.status}` : 'No runs recorded'}
          right={detail ? <StatusBadge tone={runTone(detail.status)}>● {detail.status}</StatusBadge> : undefined}
        >
          {!detail ? (
            <div className="aa-mono text-[12px] text-aa-faint">No retraining runs yet — start one explicitly.</div>
          ) : (
            <div className="space-y-2">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                  <div className="text-[10px] text-aa-faint">Run ID</div>
                  <div className="aa-mono text-[12px] text-aa-text">{detail.run_id}</div>
                </div>
                <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                  <div className="text-[10px] text-aa-faint">Trigger</div>
                  <div className="aa-mono text-[12px] text-aa-text">{detail.trigger}</div>
                </div>
                <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                  <div className="text-[10px] text-aa-faint">Started / Completed</div>
                  <div className="aa-mono text-[11px] text-aa-text">{timeUTC(detail.started_at)} → {detail.completed_at ? timeUTC(detail.completed_at) : '—'}</div>
                </div>
                <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                  <div className="text-[10px] text-aa-faint">Dataset / Schema</div>
                  <div className="aa-mono text-[11px] text-aa-text">{detail.dataset_rows ?? '—'} rows · {detail.dataset_ref ?? '—'} · schema {detail.schema_version !== null ? `v${detail.schema_version}` : '—'}</div>
                </div>
                <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                  <div className="text-[10px] text-aa-faint">Candidate</div>
                  <div className="aa-mono text-[12px] text-aa-text">{detail.best_model ?? '—'}{detail.candidate_version ? ` v${detail.candidate_version}` : ''}</div>
                </div>
                <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                  <div className="text-[10px] text-aa-faint">Candidate R² / Train run</div>
                  <div className="aa-mono text-[11px] text-aa-text">{detail.best_metrics ? formatNum(detail.best_metrics.r2, 4) : '—'} · {detail.candidate_run_id ? `${detail.candidate_run_id.slice(0, 12)}…` : '—'}</div>
                </div>
              </div>
              {detail.status === 'failed' && detail.failure_reason && (
                <div role="alert" className="rounded-md border border-red-500/30 bg-red-500/10 px-2.5 py-2 text-[11.5px] text-red-300">
                  {detail.failure_reason}
                </div>
              )}
              {detail.best_metrics?.all_results && detail.best_metrics.all_results.length > 0 && (
                <div>
                  <div className="text-[11px] font-semibold text-aa-muted mb-1.5">Candidate roster</div>
                  <div className="space-y-1">
                    {detail.best_metrics.all_results.map((r) => (
                      <div key={r.model_name} className={`flex items-center gap-2.5 rounded-md border px-2.5 py-1.5 text-[11.5px] ${r.model_name === detail.best_model ? 'border-aa-blue/40 bg-aa-blue/[0.07]' : 'border-aa-border bg-[#0a0e1a]'}`}>
                        <span className="text-aa-text font-medium w-44 truncate">{r.model_name}{r.model_name === detail.best_model ? ' ★' : ''}</span>
                        <span className="aa-mono text-aa-muted">R² {formatNum(r.metrics.r2, 4)}</span>
                        <span className="aa-mono text-aa-muted">MAE {money(r.metrics.mae)}</span>
                        <span className="flex-1" />
                        <span className="aa-mono text-aa-faint">{r.run_id.slice(0, 12)}…</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="flex flex-wrap gap-2 pt-1">
                <select
                  aria-label="Select run"
                  value={selectedId ?? ''}
                  onChange={(e) => refreshDetail(e.target.value)}
                  className="aa-select !w-auto"
                >
                  {runs.map((r) => (
                    <option key={r.run_id} value={r.run_id}>{r.run_id} · {r.status}</option>
                  ))}
                </select>
                <button onClick={runCheck} disabled={checking || !selectedId} className="aa-btn-ghost !py-1.5 !text-[11.5px]">
                  {checking ? 'Checking…' : 'Re-check Gate'}
                </button>
              </div>
            </div>
          )}
        </SectionCard>
      </div>

      {detail && detail.status === 'completed' && (
        <div className="mt-3 grid grid-cols-1 xl:grid-cols-2 gap-3">
          <SectionCard title="Production vs Candidate" subtitle="Measured comparison · no subjective verdict">
            <ComparisonTable comparison={detail.comparison} />
          </SectionCard>
          <SectionCard
            title="Promotion Gate"
            subtitle={detail.promotion_eligible ? 'Eligible — explicit promotion required' : 'Gate evaluation'}
            right={
              detail.promotion_eligible ? (
                <StatusBadge tone="green">● Eligible</StatusBadge>
              ) : (
                <StatusBadge tone="gray">● Not eligible</StatusBadge>
              )
            }
          >
            <GateList checks={detail.gate_checks} />
            <div className="mt-3 rounded-lg border border-aa-border bg-[#0a1222] p-3">
              <div className="text-[11px] text-aa-muted">
                Promoting <span className="text-white font-medium">{detail.best_model ?? '—'}</span> over production{' '}
                <span className="aa-mono">{decision?.decision.production_model ?? '—'} v{decision?.decision.production_version ?? '—'}</span>.
                Eligibility never promotes by itself.
              </div>
              <button onClick={promote} disabled={promoting || !eligible} className="aa-btn-primary w-full justify-center mt-2.5 !py-2">
                {promoting ? 'Promoting…' : eligible ? `Promote ${detail.best_model}` : 'Promote (disabled — gate not satisfied)'}
              </button>
            </div>
          </SectionCard>
        </div>
      )}

      <div className="mt-3 grid grid-cols-1 xl:grid-cols-2 gap-3">
        <SectionCard title="Rollback" subtitle="Restore a history-recorded previous pointer · current artifact preserved">
          {pointerHistory.length === 0 ? (
            <div className="aa-mono text-[12px] text-aa-faint">No pointer transitions recorded yet — nothing to roll back to.</div>
          ) : (
            <div className="space-y-2">
              <label className="block">
                <span className="text-[12px] text-aa-text">Previous production state</span>
                <select
                  aria-label="Rollback target"
                  value={rollbackTarget}
                  onChange={(e) => setRollbackTarget(e.target.value === '' ? '' : Number(e.target.value))}
                  className="aa-select mt-1"
                >
                  <option value="">Select a transition…</option>
                  {pointerHistory.map((h) => {
                    const prev = h.previous_pointer as Record<string, string | null> | null;
                    return (
                      <option key={h.id} value={h.id}>
                        #{h.id} → restore {String(prev?.['model_name'] ?? '—')} v{String(prev?.['model_version'] ?? '—')} ({timeUTC(h.timestamp)})
                      </option>
                    );
                  })}
                </select>
              </label>
              <button onClick={rollback} disabled={rollingBack || rollbackTarget === ''} className="aa-btn-ghost w-full justify-center !border-amber-500/40 !text-amber-300">
                {rollingBack ? 'Rolling back…' : 'Rollback to Selected State'}
              </button>
            </div>
          )}
        </SectionCard>
        <SectionCard title="Run History" subtitle="Newest first · status preserved">
          {runs.length === 0 ? (
            <div className="aa-mono text-[12px] text-aa-faint">No runs recorded yet.</div>
          ) : (
            <div className="space-y-1.5">
              {runs.slice(0, 8).map((r) => (
                <button
                  key={r.run_id}
                  onClick={() => refreshDetail(r.run_id)}
                  className={`w-full flex items-center gap-2.5 rounded-md border px-2.5 py-2 text-left text-[11.5px] ${r.run_id === selectedId ? 'border-aa-blue/40 bg-aa-blue/[0.07]' : 'border-aa-border bg-[#0a0e1a] hover:border-aa-blue/30'}`}
                >
                  <StatusBadge tone={runTone(r.status)}>● {r.status}</StatusBadge>
                  <span className="aa-mono text-aa-text">{r.run_id}</span>
                  <span className="text-aa-muted truncate">{r.best_model ?? '—'}</span>
                  <span className="flex-1" />
                  <span className="aa-mono text-aa-faint">{timeUTC(r.started_at)}</span>
                </button>
              ))}
            </div>
          )}
        </SectionCard>
      </div>

      <div className="mt-3">
        <InfoPanel>
          <span>
            <strong className="text-white">Lifecycle invariant:</strong> runs never promote — only an explicit, confirmed
            promotion call moves the production pointer, and only after every gate check passes. Failures leave production untouched.
          </span>
        </InfoPanel>
      </div>
    </div>
  );
}
