import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';
import type { ViewKey } from '../types';

export const DEMO_STEPS: { title: string; desc: string; view: ViewKey }[] = [
  { title: 'Open Overview', desc: 'Inspect the primary MLOps Command Center, displaying system KPIs, production model status, and live health metrics.', view: 'OVERVIEW' },
  { title: 'Confirm production model', desc: 'Check family, version and run lineage in the spec panel.', view: 'OVERVIEW' },
  { title: 'Review performance', desc: 'Read R² / MAE / RMSE from the production evaluation on the holdout split.', view: 'OVERVIEW' },
  { title: 'Review monitoring status', desc: 'Confirm overall health, rows evaluated and drifted count.', view: 'OVERVIEW' },
  { title: 'Open Predictions', desc: 'Move to the managed estimate workspace.', view: 'PREDICTIONS' },
  { title: 'Choose a preset', desc: 'Start from a preset or enter property details manually.', view: 'PREDICTIONS' },
  { title: 'Run one estimate', desc: 'Choose Run Prediction for a single managed call.', view: 'PREDICTIONS' },
  { title: 'Review valuation', desc: 'Read the predicted valuation plus serving detail.', view: 'PREDICTIONS' },
  { title: 'Open Monitoring', desc: 'Move to the statistical observability view.', view: 'MONITORING' },
  { title: 'Run Drift Check', desc: 'Execute one on-demand check against stored history.', view: 'MONITORING' },
  { title: 'Read the drift table', desc: 'Review KS statistic, raw and corrected p-values per feature.', view: 'MONITORING' },
  { title: 'Create controlled drift', desc: 'Optional: send shifted area samples, then re-run the check.', view: 'MONITORING' },
  { title: 'Confirm area drift', desc: 'Verify area_sqft is flagged while other features stay in bounds.', view: 'MONITORING' },
  { title: 'Open Retraining', desc: 'Move to the explicit retraining pipeline.', view: 'RETRAINING' },
  { title: 'Run retraining cycle', desc: 'Train candidates once and review the roster and gate.', view: 'RETRAINING' },
  { title: 'Open Model Registry', desc: 'Inspect the active artifact and version lineage.', view: 'REGISTRY' },
  { title: 'Audit the lifecycle', desc: 'Open Audit History for the chronological event trail.', view: 'AUDIT' },
];

interface AppState {
  view: ViewKey;
  setView: (v: ViewKey) => void;
  currency: 'INR' | 'USD';
  setCurrency: (c: 'INR' | 'USD') => void;
  step: number;
  setStep: (n: number) => void;
  goStep: (n: number) => void;
  refreshTick: number;
  bump: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [view, setViewState] = useState<ViewKey>('OVERVIEW');
  const [currency, setCurrency] = useState<'INR' | 'USD'>('INR');
  const [step, setStepRaw] = useState(1);
  const [refreshTick, setRefreshTick] = useState(0);

  const setView = useCallback((v: ViewKey) => setViewState(v), []);
  const setStep = useCallback((n: number) => {
    setStepRaw(Math.min(17, Math.max(1, n)));
  }, []);
  const goStep = useCallback(
    (n: number) => {
      const clamped = Math.min(17, Math.max(1, n));
      setStepRaw(clamped);
      setViewState(DEMO_STEPS[clamped - 1].view);
    },
    [],
  );
  const bump = useCallback(() => setRefreshTick((t) => t + 1), []);

  const value = useMemo(
    () => ({ view, setView, currency, setCurrency, step, setStep, goStep, refreshTick, bump }),
    [view, setView, currency, step, setStep, goStep, refreshTick, bump],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp outside provider');
  return v;
}
