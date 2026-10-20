import { useState } from 'react';
import { useApp, DEMO_STEPS } from '../state/store';
import type { ViewKey } from '../types';

const NAV: { section: string; items: { key: ViewKey; label: string; icon: string }[] }[] = [
  { section: 'WORKSPACE', items: [{ key: 'OVERVIEW', label: 'Overview', icon: '▦' }, { key: 'PREDICTIONS', label: 'Predictions', icon: '◉' }] },
  { section: 'OBSERVABILITY', items: [{ key: 'MONITORING', label: 'Monitoring', icon: '∿' }, { key: 'SIMULATOR', label: 'Scenario Simulator', icon: '✦' }] },
  { section: 'LIFECYCLE', items: [{ key: 'RETRAINING', label: 'Retraining', icon: '⎇' }, { key: 'REGISTRY', label: 'Model Registry', icon: '▤' }] },
  { section: 'AUDIT', items: [{ key: 'AUDIT', label: 'Audit History', icon: '🗎' }] },
];

export function Sidebar({ productionLabel, productionVersion }: { productionLabel: string; productionVersion: string }) {
  const { view, setView } = useApp();
  const [collapsed, setCollapsed] = useState(true);

  const go = (v: ViewKey) => {
    setView(v);
    const idx = DEMO_STEPS.findIndex((s) => s.view === v);
    if (idx >= 0) {
      // keep walkthrough roughly in sync without forcing step 1
    }
  };

  return (
    <>
      {/* mobile toggle */}
      <button
        className="lg:hidden fixed bottom-4 left-4 z-50 aa-btn-ghost !py-2"
        onClick={() => setCollapsed((c) => !c)}
        aria-label="Toggle navigation"
      >
        {collapsed ? '☰' : '✕'}
      </button>
      <aside
        className={`fixed lg:static z-40 h-full w-[224px] shrink-0 bg-aa-sidebar border-r border-aa-borderSoft flex flex-col transition-transform ${
          collapsed ? '-translate-x-full lg:translate-x-0' : 'translate-x-0'
        }`}
      >
        <div className="px-4 pt-4 pb-3 flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-md bg-aa-blue/20 border border-aa-blue/40 flex items-center justify-center text-aa-blueText text-[15px]">
            ◈
          </div>
          <div>
            <div className="text-[13px] font-bold text-white leading-tight">AdaptiveAI</div>
            <div className="aa-mono text-[9.5px] text-aa-faint leading-tight">ML Operations Platform</div>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto px-2.5 pb-3">
          {NAV.map((sec) => (
            <div key={sec.section} className="mt-3">
              <div className="px-2 text-[9.5px] tracking-[0.12em] font-semibold text-aa-faint">{sec.section}</div>
              <div className="mt-1 space-y-[2px]">
                {sec.items.map((it) => {
                  const active = view === it.key;
                  return (
                    <button
                      key={it.key}
                      onClick={() => {
                        go(it.key);
                        setCollapsed(true);
                      }}
                      className={`w-full flex items-center gap-2.5 px-2.5 py-[7px] rounded-md text-[12.5px] font-medium transition-colors ${
                        active ? 'bg-aa-blue text-white' : 'text-aa-muted hover:bg-white/5 hover:text-white'
                      }`}
                    >
                      <span className="text-[13px] w-4 text-center">{it.icon}</span>
                      {it.label}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="p-2.5">
          <div className="rounded-lg border border-aa-border bg-[#0c1425] p-2.5">
            <div className="flex items-center justify-between">
              <span className="text-[9px] tracking-[0.1em] text-aa-faint font-semibold">PROJECT</span>
              <span className="text-[10px] text-green-400 flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-green-400" /> Active
              </span>
            </div>
            <div className="mt-1 text-[12px] font-semibold text-white">House Price Prediction</div>
            <div className="aa-mono text-[10.5px] text-aa-blueText">
              {productionLabel} · {productionVersion}
            </div>
          </div>
          <div className="mt-2 px-1 flex items-center justify-between text-[10.5px]">
            <span className="text-green-400 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-green-400" /> Serving Online
            </span>
            <span className="aa-mono text-aa-faint">v1.2-eng</span>
          </div>
        </div>
      </aside>
    </>
  );
}

export function TopBar({ monitoringLabel, monitoringTone }: { monitoringLabel: string; monitoringTone: 'ok' | 'warn' | 'info' }) {
  const { currency, setCurrency, step, view } = useApp();
  const dot = monitoringTone === 'ok' ? 'bg-green-400' : monitoringTone === 'warn' ? 'bg-amber-400' : 'bg-sky-400';
  const text = monitoringTone === 'ok' ? 'text-green-400' : monitoringTone === 'warn' ? 'text-amber-400' : 'text-sky-400';
  return (
    <header className="h-[52px] shrink-0 border-b border-aa-borderSoft bg-[#0a101d]/90 backdrop-blur flex items-center gap-3 px-4">
      <div className="flex items-center gap-1.5 text-[12.5px] min-w-0">
        <span className="font-semibold text-white">AdaptiveAI</span>
        <span className="text-aa-faint">›</span>
        <span className="text-aa-muted">Projects</span>
        <span className="text-aa-faint">›</span>
        <span className="font-semibold text-white truncate">House Price Prediction</span>
        <span className="ml-1 text-[10.5px] px-2 py-[2px] rounded-full border border-green-500/30 bg-green-500/10 text-green-400 flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-green-400" /> Active
        </span>
        {view === 'SIMULATOR' && (
          <span className="ml-1 text-[10.5px] font-bold px-2 py-[2px] rounded-full border border-amber-500/50 bg-amber-500/10 text-amber-300 flex items-center gap-1">
            ✦ DEMO MODE
          </span>
        )}
      </div>
      <div className="flex-1" />
      <div className="hidden sm:flex items-center rounded-md border border-aa-border overflow-hidden text-[11.5px] font-semibold">
        <button
          onClick={() => setCurrency('INR')}
          className={`px-2.5 py-1 ${currency === 'INR' ? 'bg-aa-blue text-white' : 'text-aa-muted hover:text-white'}`}
        >
          ₹ INR
        </button>
        <button
          onClick={() => setCurrency('USD')}
          className={`px-2.5 py-1 ${currency === 'USD' ? 'bg-aa-blue text-white' : 'text-aa-muted hover:text-white'}`}
        >
          $ USD
        </button>
      </div>
      <div className={`hidden md:flex items-center gap-1.5 text-[11.5px] px-2.5 py-1 rounded-md border border-aa-border bg-white/[0.02] ${text}`}>
        <span className={`h-1.5 w-1.5 rounded-full ${dot}`} /> Monitoring: {monitoringLabel}
      </div>
      <div className="hidden lg:flex items-center gap-1.5 text-[11.5px] font-semibold px-2.5 py-1 rounded-md bg-aa-blue text-white">
        <span>✦</span> Faculty Demo Guide <span className="aa-mono bg-black/30 rounded px-1.5 py-[1px] text-[10px]">{step}/17</span>
      </div>
      <button className="h-7 w-7 rounded-full border border-aa-border text-aa-muted hover:text-white" title="Help">
        ?
      </button>
      <div className="flex items-center gap-2 pl-2 border-l border-aa-borderSoft">
        <div className="h-7 w-7 rounded-md bg-indigo-500/30 border border-indigo-400/30 flex items-center justify-center text-[10px] font-bold text-indigo-200">
          OP
        </div>
        <div className="hidden xl:block leading-tight">
          <div className="text-[11.5px] font-semibold text-white">Ops Lead</div>
          <div className="aa-mono text-[9.5px] text-aa-faint">final-year-eng</div>
        </div>
      </div>
    </header>
  );
}

export function WalkthroughBar() {
  const { step, goStep, setStep } = useApp();
  const s = DEMO_STEPS[step - 1];
  const viewLabel = s.view;
  const options = DEMO_STEPS.map((d, i) => `${i + 1}. ${d.title}`);
  return (
    <div className="border-b border-aa-borderSoft bg-[#0b1222] px-4 py-2.5 flex flex-wrap items-center gap-3">
      <div className="flex items-center gap-3 min-w-0">
        <span className="text-[10px] font-bold tracking-wide px-2 py-1.5 rounded-md bg-aa-blue/20 border border-aa-blue/40 text-aa-blueText text-center leading-tight">
          STEP {step} OF
          <br />
          17
        </span>
        <div className="min-w-0">
          <div className="text-[13px] font-semibold text-white truncate">
            {s.title} <span className="aa-mono text-[10.5px] font-normal text-aa-faint">(View: {viewLabel})</span>
          </div>
          <div className="text-[11.5px] text-aa-muted truncate max-w-[560px]">{s.desc}</div>
        </div>
      </div>
      <div className="flex-1" />
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1 rounded-md border border-aa-border px-1.5 py-1">
          <button onClick={() => goStep(step - 1)} disabled={step <= 1} className="px-1.5 text-aa-muted hover:text-white disabled:opacity-40">
            ‹
          </button>
          <span className="aa-mono text-[11px] text-aa-text px-1">
            {step} / 17
          </span>
          <button onClick={() => goStep(step + 1)} disabled={step >= 17} className="px-1.5 text-aa-muted hover:text-white disabled:opacity-40">
            ›
          </button>
        </div>
        <select
          value={options[step - 1]}
          onChange={(e) => {
            const idx = options.indexOf(e.target.value);
            if (idx >= 0) goStep(idx + 1);
          }}
          className="aa-select !w-[210px] !py-1.5 aa-mono !text-[11px]"
          aria-label="Walkthrough step"
        >
          {options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
        <button onClick={() => setStep(1)} className="text-aa-faint hover:text-white px-1" title="Close guide">
          ✕
        </button>
      </div>
    </div>
  );
}
