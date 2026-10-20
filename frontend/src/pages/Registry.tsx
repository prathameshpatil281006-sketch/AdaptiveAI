import { useEffect, useState } from 'react';
import { api, formatINR, formatNum, timeUTC } from '../api/client';
import { useApp } from '../state/store';
import type { RegistryCandidate, RegistryInfo } from '../types';
import { KeyVal, PageHeader, SectionCard, StatusBadge } from '../components/ui';

export default function Registry() {
  const { refreshTick } = useApp();
  const [info, setInfo] = useState<RegistryInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<RegistryCandidate | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      setLoading(true);
      try {
        const r = await api.registry(100);
        if (alive) setInfo(r);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [refreshTick]);

  const prod = info?.production;

  return (
    <div>
      <PageHeader
        title="Model Registry"
        desc="Governed artifact lineage for HousePriceModel — production authority plus every promotion decision."
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">
        <SectionCard title="Production Model" subtitle="Serving authority · production.json">
          {prod ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-[16px] font-bold text-white">{prod.model_name}</span>
                <span className="aa-mono text-[10.5px] px-1.5 py-[2px] rounded border border-aa-blue/40 bg-aa-blue/10 text-aa-blueText">
                  v{prod.model_version ?? '—'}
                </span>
                <StatusBadge tone="green">● Production</StatusBadge>
              </div>
              <div className="rounded-lg border border-aa-border bg-[#0a1222] p-3 aa-mono text-[11px] space-y-1.5">
                <div className="flex justify-between"><span className="text-aa-faint">Registered family</span><span className="text-aa-text">{prod.registered_model_name}</span></div>
                <div className="flex justify-between"><span className="text-aa-faint">Run ID</span><span className="text-aa-blueText">{prod.run_id?.slice(0, 16)}…</span></div>
                <div className="flex justify-between"><span className="text-aa-faint">Deployed</span><span className="text-aa-text">{timeUTC(prod.deployed_at)}</span></div>
                <div className="flex justify-between"><span className="text-aa-faint">Artifact</span><span className="text-aa-text">{prod.model_name}.joblib</span></div>
              </div>
              <div className="text-[11px] text-aa-muted">
                Lineage is preserved end-to-end: every served prediction resolves exactly this artifact + version. A broken
                reference fails closed instead of silently serving another model.
              </div>
            </div>
          ) : (
            <div className="text-[12px] text-aa-muted">{loading ? 'Loading…' : 'No production model deployed.'}</div>
          )}
        </SectionCard>

        <div className="xl:col-span-2">
          <SectionCard
            title="Model Versions"
            subtitle={`HousePriceModel · ${info?.total ?? 0} recorded decisions`}
            right={<span className="aa-mono text-[10.5px] text-aa-faint">MLflow registry: {info?.mlflow_versions.length ?? 0} listed</span>}
          >
            <div className="overflow-x-auto -mx-4 px-4">
              <table className="aa-table w-full min-w-[720px]">
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Algorithm</th>
                    <th>Status</th>
                    <th>R² (cand → prod)</th>
                    <th>Run ID</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {(info?.versions ?? []).length === 0 && (
                    <tr>
                      <td colSpan={6} className="text-center text-aa-faint py-6">
                        {loading ? 'Loading versions…' : 'No versions recorded yet. Run the retraining pipeline.'}
                      </td>
                    </tr>
                  )}
                  {(info?.versions ?? []).slice(0, 12).map((v) => (
                    <tr key={v.id}>
                      <td>
                        <span className="aa-mono text-[11px] px-1.5 py-[2px] rounded bg-aa-blue/10 border border-aa-blue/30 text-aa-blueText">
                          v{v.model_version ?? '—'}
                        </span>
                      </td>
                      <td className="font-medium text-white">{v.model_name}</td>
                      <td>
                        {v.decision === 'promoted' ? (
                          <StatusBadge tone="green">● Promoted</StatusBadge>
                        ) : v.decision === 'rejected' ? (
                          <StatusBadge tone="amber">● Rejected</StatusBadge>
                        ) : (
                          <StatusBadge tone="gray">{v.decision}</StatusBadge>
                        )}
                      </td>
                      <td className="aa-mono text-[11px]">
                        {formatNum(v.candidate_score, 4)} → {formatNum(v.production_score, 4)}
                      </td>
                      <td className="aa-mono text-[11px] text-aa-faint">{v.run_id?.slice(0, 12)}…</td>
                      <td className="aa-mono text-[11px] text-aa-faint">{timeUTC(v.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(info?.mlflow_versions ?? []).length > 0 && (
              <div className="mt-3 aa-mono text-[10.5px] text-aa-faint">
                MLflow: {(info?.mlflow_versions ?? []).slice(0, 3).map((m) => `v${m.version} (${m.run_id.slice(0, 8)}…)`).join(' · ')}
                {info && info.mlflow_versions.length > 3 ? ` · +${info.mlflow_versions.length - 3} more` : ''}
              </div>
            )}
          </SectionCard>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-1 xl:grid-cols-3 gap-3">
        <div className="xl:col-span-2">
          <SectionCard
            title="Lifecycle Candidates"
            subtitle={`Retraining runs · ${info?.runs?.length ?? 0} recorded`}
          >
            {(info?.runs ?? []).length === 0 ? (
              <div className="aa-mono text-[12px] text-aa-faint py-4 text-center">
                {loading ? 'Loading candidates…' : 'No retraining runs recorded yet.'}
              </div>
            ) : (
              <div className="overflow-x-auto -mx-4 px-4">
                <table className="aa-table w-full min-w-[760px]">
                  <thead>
                    <tr>
                      <th>Run</th>
                      <th>Candidate</th>
                      <th>Status</th>
                      <th>R²</th>
                      <th>Schema</th>
                      <th>Artifact</th>
                      <th>Lifecycle</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(info?.runs ?? []).slice(0, 12).map((r) => (
                      <tr
                        key={r.run_id ?? `${r.started_at}`}
                        onClick={() => setSelected(r)}
                        className={`cursor-pointer ${selected?.run_id === r.run_id ? 'bg-aa-blue/[0.06]' : ''}`}
                      >
                        <td className="aa-mono text-[11px] text-aa-blueText">{r.run_id ?? '—'}</td>
                        <td className="font-medium text-white">{r.candidate_model ?? '—'}</td>
                        <td>
                          <StatusBadge tone={r.status === 'completed' ? 'green' : r.status === 'failed' ? 'red' : 'blue'}>
                            ● {r.status ?? '—'}
                          </StatusBadge>
                        </td>
                        <td className="aa-mono">{r.candidate_metrics ? formatNum(r.candidate_metrics.r2, 4) : '—'}</td>
                        <td className="aa-mono text-aa-muted">{r.schema_version !== null && r.schema_version !== undefined ? `v${r.schema_version}` : '—'}</td>
                        <td className="aa-mono text-[11px]">
                          {r.artifact_available === true ? (
                            <span className="text-green-400">available</span>
                          ) : r.artifact_available === false ? (
                            <span className="text-aa-faint">missing</span>
                          ) : (
                            <span className="text-aa-faint">—</span>
                          )}
                        </td>
                        <td>
                          {r.promotion_decision === 'promoted' ? (
                            <StatusBadge tone="green">● Promoted{r.candidate_version ? ` v${r.candidate_version}` : ''}</StatusBadge>
                          ) : r.promotion_decision === 'blocked' ? (
                            <StatusBadge tone="amber">● Blocked</StatusBadge>
                          ) : r.promotion_eligible ? (
                            <StatusBadge tone="blue">● Eligible</StatusBadge>
                          ) : (
                            <span className="text-aa-faint text-[11px]">—</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </SectionCard>
        </div>
        <SectionCard title="Candidate Detail" subtitle="Select a run to inspect">
          {!selected ? (
            <div className="aa-mono text-[12px] text-aa-faint">No candidate selected.</div>
          ) : (
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[15px] font-bold text-white">{selected.candidate_model ?? '—'}</span>
                <StatusBadge tone={selected.status === 'completed' ? 'green' : selected.status === 'failed' ? 'red' : 'blue'}>
                  ● {selected.status ?? '—'}
                </StatusBadge>
              </div>
              <div className="mt-2">
                <KeyVal k="Run ID" v={<span className="aa-mono">{selected.run_id ?? '—'}</span>} />
                <KeyVal k="Training run" v={<span className="aa-mono">{selected.candidate_run_id ? `${selected.candidate_run_id.slice(0, 16)}…` : '—'}</span>} />
                <KeyVal k="Dataset" v={<span className="aa-mono">{selected.dataset_rows ?? '—'} rows · schema {selected.schema_version !== null && selected.schema_version !== undefined ? `v${selected.schema_version}` : '—'}</span>} />
                <KeyVal k="R² / MAE / RMSE" v={<span className="aa-mono">{selected.candidate_metrics ? `${formatNum(selected.candidate_metrics.r2, 4)} / ${formatINR(selected.candidate_metrics.mae)} / ${formatINR(selected.candidate_metrics.rmse)}` : '—'}</span>} />
                <KeyVal k="R² improvement" v={<span className="aa-mono">{selected.comparison_deltas ? `${selected.comparison_deltas.r2_improvement >= 0 ? '+' : ''}${selected.comparison_deltas.r2_improvement.toFixed(4)}` : '—'}</span>} />
                <KeyVal k="Staged artifact" v={<span className="aa-mono">{selected.candidate_artifact ?? '—'}</span>} />
                <KeyVal k="Trigger" v={<span className="aa-mono">{selected.trigger ?? '—'}</span>} />
                <KeyVal k="Completed" v={<span className="aa-mono">{selected.completed_at ? timeUTC(selected.completed_at) : '—'}</span>} />
                {selected.failure_reason && (
                  <div role="alert" className="mt-2 rounded-md border border-red-500/30 bg-red-500/10 px-2.5 py-2 text-[11.5px] text-red-300">
                    {selected.failure_reason}
                  </div>
                )}
              </div>
            </div>
          )}
        </SectionCard>
      </div>

      <div className="mt-3">
        <SectionCard title="Pointer Lineage" subtitle="Production pointer transitions · rollback targets">
          {(info?.pointer_history ?? []).length === 0 ? (
            <div className="aa-mono text-[12px] text-aa-faint">No pointer transitions recorded yet.</div>
          ) : (
            <div className="space-y-1.5">
              {(info?.pointer_history ?? []).slice(0, 8).map((h) => {
                const prev = h.previous_pointer as Record<string, string | null> | null;
                const next = h.new_pointer as Record<string, string | null> | null;
                return (
                  <div key={h.id} className="flex flex-wrap items-center gap-2.5 rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2 text-[11.5px]">
                    <span className="aa-mono text-aa-faint">#{h.id}</span>
                    <span className="aa-mono text-aa-muted">{String(prev?.['model_name'] ?? '—')} v{String(prev?.['model_version'] ?? '—')}</span>
                    <span className="text-aa-faint">→</span>
                    <span className="aa-mono text-aa-text">{String(next?.['model_name'] ?? '—')} v{String(next?.['model_version'] ?? '—')}</span>
                    <span className="flex-1" />
                    <span className="aa-mono text-aa-faint">{h.run_id ?? h.reason ?? ''}</span>
                    <span className="aa-mono text-aa-faint">{timeUTC(h.timestamp)}</span>
                  </div>
                );
              })}
            </div>
          )}
        </SectionCard>
      </div>

      <div className="mt-3">
        <SectionCard title="Promotion Detail" subtitle="Reason and lineage for the latest decision">
          {info?.versions?.[0] ? (
            <div className="text-[12px] text-aa-muted leading-relaxed">
              <span className="text-white font-semibold">
                {info.versions[0].model_name} v{info.versions[0].model_version ?? '—'}
              </span>{' '}
              → <strong className="text-white">{info.versions[0].decision}</strong> · {info.versions[0].reason}
              <div className="mt-1 aa-mono text-[11px] text-aa-faint">
                run {info.versions[0].run_id} · {info.versions[0].registered_model_name} · metric{' '}
                {info.versions[0].primary_metric ?? 'r2'}
              </div>
            </div>
          ) : (
            <div className="text-[12px] text-aa-faint">No promotion detail available.</div>
          )}
        </SectionCard>
      </div>
    </div>
  );
}
