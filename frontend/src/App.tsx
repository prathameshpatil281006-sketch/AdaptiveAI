import { useEffect, useState } from 'react';
import { Sidebar, TopBar, WalkthroughBar } from './components/shell';
import { AppProvider, useApp } from './state/store';
import { api } from './api/client';
import Overview from './pages/Overview';
import Predictions from './pages/Predictions';
import Monitoring from './pages/Monitoring';
import Retraining from './pages/Retraining';
import Registry from './pages/Registry';
import Audit from './pages/Audit';
import ScenarioSimulator from './pages/ScenarioSimulator';

function Shell() {
  const { view } = useApp();
  const [monLabel, setMonLabel] = useState('Nominal');
  const [monTone, setMonTone] = useState<'ok' | 'warn' | 'info'>('ok');
  const [prodLabel, setProdLabel] = useState('Unknown');
  const [prodVersion, setProdVersion] = useState('—');

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const p = await api.production();
        if (!alive) return;
        setProdLabel(p.model_name || 'Unknown');
        setProdVersion(p.model_version ? `v${p.model_version}` : '—');
      } catch {
        if (!alive) return;
        setProdLabel('Unknown');
        setProdVersion('—');
      }
    })();
    return () => {
      alive = false;
    };
  }, [view]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const h = await api.monitoringHistory(1, 0);
        if (!alive) return;
        const latest = h.items[0];
        if (!latest) {
          setMonLabel('Nominal');
          setMonTone('ok');
          return;
        }
        if (latest.status === 'insufficient_data') {
          setMonLabel('Insufficient');
          setMonTone('info');
        } else if (latest.overall_drifted) {
          setMonLabel('Drift');
          setMonTone('warn');
        } else {
          setMonLabel('Nominal');
          setMonTone('ok');
        }
      } catch {
        /* keep nominal */
      }
    })();
    return () => {
      alive = false;
    };
  }, [view]);

  return (
    <div className="h-screen w-screen overflow-hidden bg-aa-bg flex">
      <Sidebar productionLabel={prodLabel} productionVersion={prodVersion} />
      <div className="flex-1 min-w-0 flex flex-col h-full">
        <TopBar monitoringLabel={monLabel} monitoringTone={monTone} />
        <WalkthroughBar />
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-[1380px] mx-auto px-5 py-4">
            {view === 'OVERVIEW' && <Overview />}
            {view === 'PREDICTIONS' && <Predictions />}
            {view === 'MONITORING' && <Monitoring />}
            {view === 'RETRAINING' && <Retraining />}
            {view === 'REGISTRY' && <Registry />}
            {view === 'AUDIT' && <Audit />}
            {view === 'SIMULATOR' && <ScenarioSimulator />}
          </div>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
