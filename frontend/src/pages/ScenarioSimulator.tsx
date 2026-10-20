import { useEffect, useState } from 'react';
import { api, formatNum, timeUTC } from '../api/client';
import { useApp } from '../state/store';
import type {
  DemoScenarioInfo,
  DemoSession,
  DriftFeature,
  HealthSummary,
  PromotionCheckResult,
  PromoteResult,
  RegistryInfo,
  RetrainingDecision,
  RetrainingRun,
} from '../types';
import { InfoPanel, PageHeader, SectionCard, StatusBadge } from '../components/ui';

const LOCALITIES = [
  'Wakad', 'Hinjawadi', 'Kharadi', 'Baner', 'Balewadi', 'Kothrud',
  'Hadapsar', 'Wagholi', 'Viman Nagar', 'Aundh', 'Punawale', 'Ravet',
];

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

function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits);
}

function evidenceOf(f: DriftFeature): string {
  if (f.type === 'categorical') {
    return `χ²=${fmtNum(f.statistic, 2)} · p=${fmtNum(f.p_value_corrected ?? f.p_value)} · PSI=${fmtNum(f.effect_size, 3)}`;
  }
  return `D=${fmtNum(f.statistic)} · p=${fmtNum(f.p_value_corrected ?? f.p_value)}`;
}

export default function ScenarioSimulator() {
  const { bump } = useApp();
  const [scenarios, setScenarios] = useState<DemoScenarioInfo[]>([]);
  const [selected, setSelected] = useState('healthy_baseline');
  const [sampleCount, setSampleCount] = useState(100);
  const [seed, setSeed] = useState(42);
  const [locality, setLocality] = useState('Kothrud');
  const [concentration, setConcentration] = useState(0.9);
  const [areaMin, setAreaMin] = useState(2400);
  const [areaMax, setAreaMax] = useState(3000);
  const [session, setSession] = useState<DemoSession | null>(null);
  const [monitoring, setMonitoring] = useState<HealthSummary | null>(null);
  const [decision, setDecision] = useState<RetrainingDecision | null>(null);
  const [run, setRun] = useState<RetrainingRun | null>(null);
  const [check, setCheck] = useState<PromotionCheckResult | null>(null);
  const [promoteOut, setPromoteOut] = useState<PromoteResult | null>(null);
  const [registry, setRegistry] = useState<RegistryInfo | null>(null);
  const [sessions, setSessions] = useState<DemoSession[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [s, list, reg] = await Promise.all([
          api.demoScenarios().catch(() => ({ items: [] })),
          api.demoSessions(10, 0).catch(() => ({ items: [] })),
          api.registry(20).catch(() => null),
        ]);
        setScenarios(s.items);
        setSessions(list.items);
        if (reg) setRegistry(reg);
      } catch {
        /* keep defaults */
      }
    })();
  }, []);

  const scenarioParams = (): Record<string, unknown> | undefined => {
    if (selected === 'locality_shift') return { locality, concentration };
    if (selected === 'area_shift') return { area_min: areaMin, area_max: areaMax };
    return undefined;
  };

  const refreshSessions = async () => {
    const list = await api.demoSessions(10, 0).catch(() => ({ items: [] as DemoSession[] }));
    setSessions(list.items);
  };

  const generate = async () => {
    setBusy('generate');
    setError(null);
    setNotice(null);
    try {
      const created = await api.demoCreateSession(selected, sampleCount, seed, scenarioParams());
      const out = await api.demoGenerate(created.session_id);
      setSession(out.session);
      setMonitoring(out.session.last_monitoring);
      setDecision(out.session.last_decision);
      setRun(null);
      setCheck(null);
      setPromoteOut(null);
      setNotice(`Generated ${out.samples_generated} isolated demo samples (session ${out.session.session_id}). Production traffic untouched.`);
      bump();
      await refreshSessions();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const monitor = async () => {
    if (!session) return;
    setBusy('monitor');
    setError(null);
    try {
      const out = await api.demoMonitor(session.session_id);
      setSession(out.session);
      setMonitoring(out.monitoring);
      setDecision(out.decision);
      bump();
      await refreshSessions();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const startRetraining = async () => {
    if (!session) return;
    if (!window.confirm(`Start retraining triggered by demo session ${session.session_id}? This trains candidates on the reference dataset; it never promotes.`)) return;
    setBusy('retrain');
    setError(null);
    try {
      const r = await api.demoRetrain(session.session_id, 1000);
      setRun(r);
      setNotice(`Retraining run ${r.run_id} ${r.status} (trigger: ${r.trigger}). Production pointer unchanged.`);
      bump();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const runPromotionCheck = async () => {
    if (!run) return;
    setBusy('check');
    setError(null);
    try {
      const c = await api.lifecyclePromotionCheck(run.run_id, 1000);
      setCheck(c);
      const detail = await api.lifecycleRun(run.run_id);
      setRun(detail);
      bump();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const promote = async () => {
    if (!run) return;
    const prod = registry?.production;
    if (!window.confirm(`DEMO PROMOTION — this changes the production model${prod ? ` from ${prod.model_name} v${prod.model_version ?? '—'}` : ''} to candidate '${run.best_model ?? '—'}'. Production traffic will be served by the new model. Continue?`)) return;
    setBusy('promote');
    setError(null);
    try {
      const out = await api.lifecyclePromote(run.run_id);
      setPromoteOut(out);
      const reg = await api.registry(20).catch(() => null);
      if (reg) setRegistry(reg);
      setNotice(out.promoted ? `Promoted: production is now v${out.production.model_version}. This is a REAL production change from a demo flow.` : `Not promoted: ${out.reason}`);
      bump();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const reset = async () => {
    if (!session) return;
    if (!window.confirm(`Reset demo session ${session.session_id}? Only its simulated rows are removed; production state is untouched.`)) return;
    setBusy('reset');
    setError(null);
    try {
      const out = await api.demoReset(session.session_id);
      setSession(out.session);
      setMonitoring(null);
      setDecision(null);
      setRun(null);
      setCheck(null);
      setNotice(`Session reset: ${out.removed_demo_rows} demo row(s) removed. Production unchanged.`);
      bump();
      await refreshSessions();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const openSession = async (id: string) => {
    setError(null);
    try {
      const s = await api.demoSession(id);
      setSession(s);
      setMonitoring(s.last_monitoring);
      setDecision(s.last_decision);
      setRun(null);
      setCheck(null);
      setPromoteOut(null);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const features = monitoring?.features ?? [];

  const stages: { key: string; label: string; done: boolean; active: boolean }[] = [
    { key: 'scenario', label: 'Scenario', done: true, active: !session },
    { key: 'generate', label: 'Generate', done: !!session && session.samples_generated > 0, active: !!session && session.samples_generated === 0 },
    { key: 'monitor', label: 'Monitor', done: !!monitoring, active: !!session && session.samples_generated > 0 && !monitoring },
    { key: 'recommend', label: 'Recommend', done: !!decision, active: !!monitoring && !decision },
    { key: 'retrain', label: 'Retrain', done: !!run && run.status === 'completed', active: !!decision && (!run || run.status !== 'completed') },
    { key: 'gate', label: 'Gate check', done: !!check, active: !!run && run.status === 'completed' && !check },
    { key: 'promote', label: 'Promote', done: !!promoteOut?.promoted, active: !!check?.eligible && !promoteOut?.promoted },
  ];

  return (
    <div>
      <PageHeader
        title="Scenario Simulator"
        desc="Safely demonstrate AdaptiveAI's MLOps lifecycle without changing production traffic."
        right={
          <span className="aa-mono text-[11px] font-bold px-2.5 py-1.5 rounded-md border border-amber-500/50 bg-amber-500/10 text-amber-300">
            ✦ DEMO MODE
          </span>
        }
      />
      {error && <div role="alert" className="mb-3 text-[12px] text-red-400">{error}</div>}
      {notice && <div className="mb-3 text-[12px] text-green-400">{notice}</div>}

      <SectionCard
        title="Active Session"
        subtitle="demo traffic is isolated by source + session id and never enters production reads"
        right={
          session ? (
            <StatusBadge tone={session.status === 'reset' ? 'gray' : session.status === 'completed' ? 'green' : 'blue'}>
              ● {session.session_id} · {session.status}
            </StatusBadge>
          ) : (
            <StatusBadge tone="gray">● No session</StatusBadge>
          )
        }
      >
        <div className="flex flex-wrap items-center gap-1.5 mb-3" aria-label="Demo lifecycle progress">
          {stages.map((s, i) => (
            <div key={s.key} className="flex items-center gap-1.5">
              {i > 0 && <span className="text-aa-faint text-[10px]">→</span>}
              <span
                className={`aa-mono text-[10.5px] px-2 py-1 rounded-md border ${
                  s.done
                    ? 'border-green-500/40 bg-green-500/10 text-green-400'
                    : s.active
                      ? 'border-aa-blue/50 bg-aa-blue/10 text-aa-blueText'
                      : 'border-aa-border text-aa-faint'
                }`}
              >
                {s.done ? '✓ ' : ''}{i + 1}. {s.label}
              </span>
            </div>
          ))}
        </div>
        <div className="aa-mono text-[12px] text-aa-muted">
          {session
            ? `${session.scenario} · ${session.samples_generated}/${session.sample_count} samples · seed ${session.seed} · created ${timeUTC(session.created_at)}`
            : 'Pick a scenario below and generate — production prediction history stays untouched.'}
        </div>
      </SectionCard>

      <div className="mt-3">
        <SectionCard title="Scenario Selector" subtitle="deterministic: same scenario + seed reproduces identical samples">
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-2">
            {(scenarios.length > 0 ? scenarios : [
              { name: 'healthy_baseline', title: 'Healthy Baseline', description: '', expected: '' },
              { name: 'area_shift', title: 'Area Shift', description: '', expected: '' },
              { name: 'locality_shift', title: 'Locality Shift', description: '', expected: '' },
              { name: 'config_shift', title: 'Configuration Shift', description: '', expected: '' },
              { name: 'data_quality', title: 'Data Quality Issue', description: '', expected: '' },
            ]).map((s) => (
              <button
                key={s.name}
                onClick={() => setSelected(s.name)}
                className={`text-left rounded-lg border px-3 py-2.5 transition-colors ${
                  selected === s.name ? 'border-aa-blue/60 bg-aa-blue/[0.08]' : 'border-aa-border bg-[#0a0e1a] hover:border-aa-blue/30'
                }`}
              >
                <div className="text-[12px] font-semibold text-white">{s.title}</div>
                {s.description && <div className="mt-1 text-[11px] text-aa-muted leading-snug">{s.description}</div>}
                {s.expected && <div className="mt-1.5 aa-mono text-[10px] text-aa-blueText">{s.expected}</div>}
              </button>
            ))}
          </div>
          <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2">
            <label className="block">
              <span className="text-[11.5px] text-aa-text">Sample Count</span>
              <input type="number" min={1} max={1000} value={sampleCount} onChange={(e) => setSampleCount(Number(e.target.value))} className="aa-select mt-1" />
            </label>
            <label className="block">
              <span className="text-[11.5px] text-aa-text">Seed</span>
              <input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} className="aa-select mt-1" />
            </label>
            {selected === 'locality_shift' && (
              <>
                <label className="block">
                  <span className="text-[11.5px] text-aa-text">Locality</span>
                  <select value={locality} onChange={(e) => setLocality(e.target.value)} className="aa-select mt-1">
                    {LOCALITIES.map((l) => <option key={l} value={l}>{l}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="text-[11.5px] text-aa-text">Concentration</span>
                  <input type="number" min={0.1} max={1} step={0.05} value={concentration} onChange={(e) => setConcentration(Number(e.target.value))} className="aa-select mt-1" />
                </label>
              </>
            )}
            {selected === 'area_shift' && (
              <>
                <label className="block">
                  <span className="text-[11.5px] text-aa-text">Area Min (sqft)</span>
                  <input type="number" min={1} value={areaMin} onChange={(e) => setAreaMin(Number(e.target.value))} className="aa-select mt-1" />
                </label>
                <label className="block">
                  <span className="text-[11.5px] text-aa-text">Area Max (sqft)</span>
                  <input type="number" min={1} value={areaMax} onChange={(e) => setAreaMax(Number(e.target.value))} className="aa-select mt-1" />
                </label>
              </>
            )}
          </div>
          <button onClick={generate} disabled={busy === 'generate'} className="aa-btn-primary mt-3">
            <span>✦</span> {busy === 'generate' ? 'Generating…' : 'Generate Scenario'}
          </button>
        </SectionCard>
      </div>

      {session && (
        <div className="mt-3">
          <SectionCard
            title="Simulation Result"
            subtitle={`scenario: ${session.scenario} · session: ${session.session_id}`}
            right={<StatusBadge tone="amber">● DEMO samples: {session.samples_generated}</StatusBadge>}
          >
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-2">
              <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                <div className="text-[10px] text-aa-faint">Scenario</div>
                <div className="text-[12px] text-aa-text">{session.scenario}</div>
              </div>
              <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                <div className="text-[10px] text-aa-faint">Samples (seed {session.seed})</div>
                <div className="aa-mono text-[12px] text-aa-text">{session.samples_generated}</div>
              </div>
              <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                <div className="text-[10px] text-aa-faint">Status</div>
                <div className="text-[12px] text-aa-text">{session.status}</div>
              </div>
              <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                <div className="text-[10px] text-aa-faint">Isolation</div>
                <div className="aa-mono text-[12px] text-amber-300">source=demo</div>
              </div>
            </div>
          </SectionCard>
        </div>
      )}

      {session && session.samples_generated > 0 && (
        <div className="mt-3">
          <SectionCard
            title="Monitor Scenario"
            subtitle="real Phase 4 engine over demo rows only — production monitoring history unchanged"
            right={<button onClick={monitor} disabled={busy === 'monitor'} className="aa-btn-primary"><span>⟳</span> {busy === 'monitor' ? 'Monitoring…' : 'Run Monitoring'}</button>}
          >
            {!monitoring ? (
              <div className="aa-mono text-[12px] text-aa-faint">No demo monitoring yet — run monitoring above.</div>
            ) : (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-2">
                  <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                    <div className="text-[10px] text-aa-faint">Health</div>
                    <div className="mt-1"><StatusBadge tone={severityTone(monitoring.overall_health)}>● {monitoring.overall_health}</StatusBadge></div>
                  </div>
                  <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                    <div className="text-[10px] text-aa-faint">Drift / Severity</div>
                    <div className="mt-1 flex gap-1.5">
                      <StatusBadge tone={monitoring.overall_drifted ? 'red' : 'green'}>● {monitoring.overall_drifted ? 'Drift' : 'No drift'}</StatusBadge>
                      <StatusBadge tone={severityTone(monitoring.severity)}>● {monitoring.severity}</StatusBadge>
                    </div>
                  </div>
                  <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                    <div className="text-[10px] text-aa-faint">Drifted Features</div>
                    <div className="aa-mono text-[12px] text-aa-text">{monitoring.drifted_count}/{monitoring.total_monitored}</div>
                    <div className="aa-mono text-[10.5px] text-aa-muted truncate" title={(monitoring.drifted_features ?? []).join(', ')}>{(monitoring.drifted_features ?? []).join(', ') || '—'}</div>
                  </div>
                  <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                    <div className="text-[10px] text-aa-faint">Data Quality</div>
                    <div className="mt-1"><StatusBadge tone={severityTone(monitoring.data_quality_status)}>● {monitoring.data_quality_status}</StatusBadge></div>
                  </div>
                </div>
                <div className="mt-3 overflow-x-auto -mx-4 px-4">
                  <table className="aa-table w-full min-w-[720px]">
                    <thead><tr><th>Feature</th><th>Type</th><th>Status</th><th>Evidence</th></tr></thead>
                    <tbody>
                      {features.map((f) => (
                        <tr key={f.feature_name}>
                          <td className="text-aa-text">{FEATURE_LABELS[f.feature_name] ?? f.feature_name}</td>
                          <td className="aa-mono text-aa-muted">{f.type ?? 'numeric'}</td>
                          <td>
                            {(f.status ?? 'ok') !== 'ok' ? <StatusBadge tone="gray">● No data</StatusBadge>
                              : f.drifted ? <StatusBadge tone="red">● Drifted</StatusBadge>
                              : <StatusBadge tone="green">● In Bounds</StatusBadge>}
                          </td>
                          <td className="aa-mono text-aa-muted">{evidenceOf(f)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </SectionCard>
        </div>
      )}

      {decision && (
        <div className="mt-3">
          <SectionCard title="What Happens Next?" subtitle="Phase 5 retraining recommendation from the demo monitoring result">
            <div className="flex items-center gap-2">
              <StatusBadge tone={decision.recommended ? 'blue' : 'gray'}>
                ● {decision.recommended ? 'Retraining recommended' : 'No retraining recommended'}
              </StatusBadge>
            </div>
            <div className="mt-2 text-[12.5px] text-aa-text leading-relaxed">{decision.reason}</div>
            {decision.triggers.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {decision.triggers.map((t) => (
                  <span key={t} className="aa-mono text-[10.5px] px-2 py-[3px] rounded-md border border-amber-500/30 bg-aa-amberSoft text-amber-300">{t}</span>
                ))}
              </div>
            )}
            <button onClick={startRetraining} disabled={busy === 'retrain'} className="aa-btn-primary mt-3">
              <span>◉</span> {busy === 'retrain' ? 'Training…' : 'Start Retraining'}
            </button>
            {run && (
              <div className="mt-2 aa-mono text-[11.5px] text-aa-muted">
                Run {run.run_id} · {run.status} · trigger {run.trigger} · best {run.best_model ?? '—'}
                {run.best_metrics ? ` · R² ${formatNum(run.best_metrics.r2, 4)}` : ''}
              </div>
            )}
          </SectionCard>
        </div>
      )}

      {run && run.status === 'completed' && (
        <div className="mt-3">
          <SectionCard
            title="Model Lifecycle"
            subtitle="candidate vs production · explicit promotion only"
            right={<StatusBadge tone="amber">● DEMO promotion will modify the production pointer</StatusBadge>}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                <div className="text-[10px] text-aa-faint">Candidate ({run.best_model ?? '—'})</div>
                <div className="aa-mono text-[12px] text-aa-text">R² {run.best_metrics ? formatNum(run.best_metrics.r2, 4) : '—'}</div>
              </div>
              <div className="rounded-md border border-aa-border bg-[#0a0e1a] px-2.5 py-2">
                <div className="text-[10px] text-aa-faint">Production</div>
                <div className="aa-mono text-[12px] text-aa-text">
                  {registry?.production ? `${registry.production.model_name} v${registry.production.model_version ?? '—'}` : '—'}
                </div>
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <button onClick={runPromotionCheck} disabled={busy === 'check'} className="aa-btn-ghost">
                {busy === 'check' ? 'Checking…' : 'Run Promotion Check'}
              </button>
              <button onClick={promote} disabled={busy === 'promote'} className="aa-btn-primary !bg-amber-600 hover:!bg-amber-500">
                {busy === 'promote' ? 'Promoting…' : 'Promote Candidate'}
              </button>
            </div>
            {check && (
              <div className="mt-2 aa-mono text-[11.5px] text-aa-muted">
                Gate: {check.eligible ? 'ELIGIBLE — still requires explicit promotion' : 'NOT eligible — production unchanged'}
              </div>
            )}
            {promoteOut && (
              <div className="mt-2 text-[12px] text-aa-text">{promoteOut.promoted ? `Promoted: production is now v${promoteOut.production.model_version}.` : `Not promoted: ${promoteOut.reason}`}</div>
            )}
            <div className="mt-3">
              <InfoPanel>
                <span>Promotion here is a <strong className="text-white">real production change</strong> executed through the existing Phase 5 gate. Generating, monitoring, retraining, and promotion checks never move the pointer — only this confirmed button does.</span>
              </InfoPanel>
            </div>
          </SectionCard>
        </div>
      )}

      <div className="mt-3 grid grid-cols-1 xl:grid-cols-2 gap-3">
        <SectionCard title="Session" subtitle="reset removes only this session's simulated rows">
          {!session ? (
            <div className="aa-mono text-[12px] text-aa-faint">No session selected.</div>
          ) : (
            <button onClick={reset} disabled={busy === 'reset'} className="aa-btn-ghost w-full justify-center !border-amber-500/40 !text-amber-300">
              {busy === 'reset' ? 'Resetting…' : `Reset Demo Session ${session.session_id}`}
            </button>
          )}
        </SectionCard>
        <SectionCard title="Recent Demo Sessions" subtitle="newest first · production history never listed here">
          {sessions.length === 0 ? (
            <div className="aa-mono text-[12px] text-aa-faint">No demo sessions yet.</div>
          ) : (
            <div className="space-y-1.5">
              {sessions.slice(0, 8).map((s) => (
                <button
                  key={s.session_id}
                  onClick={() => openSession(s.session_id)}
                  className={`w-full flex items-center gap-2.5 rounded-md border px-2.5 py-2 text-left text-[11.5px] ${s.session_id === session?.session_id ? 'border-aa-blue/40 bg-aa-blue/[0.07]' : 'border-aa-border bg-[#0a0e1a] hover:border-aa-blue/30'}`}
                >
                  <StatusBadge tone={s.status === 'reset' ? 'gray' : s.status === 'completed' ? 'green' : 'blue'}>● {s.status}</StatusBadge>
                  <span className="aa-mono text-aa-text">{s.session_id}</span>
                  <span className="text-aa-muted truncate">{s.scenario} · {s.samples_generated} samples</span>
                  <span className="flex-1" />
                  <span className="aa-mono text-aa-faint">{timeUTC(s.created_at)}</span>
                </button>
              ))}
            </div>
          )}
        </SectionCard>
      </div>

      <div className="mt-3">
        <InfoPanel>
          <span><strong className="text-white">DEMO MODE guarantees:</strong> simulated rows carry <span className="aa-mono">source=demo</span> and are excluded from production monitoring, prediction history, and KPI cards. Demo monitoring never writes to monitoring history. Reset deletes only demo rows. Only an explicitly confirmed promotion moves the production pointer.</span>
        </InfoPanel>
      </div>
    </div>
  );
}
