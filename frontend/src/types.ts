export type ViewKey = 'OVERVIEW' | 'PREDICTIONS' | 'MONITORING' | 'RETRAINING' | 'REGISTRY' | 'AUDIT' | 'SIMULATOR';

export interface ProductionInfo {
  model_name: string;
  model_version: string | null;
  registered_model_name: string;
  run_id: string | null;
  algorithm_family: string;
  sklearn_version: string;
  training_dataset: { rows: number | null; reference: string | null };
  metrics: { r2: number | null; mae: number | null; rmse: number | null };
  features: { name: string; dtype: string }[];
  deployed_at: string | null;
  use_case: string;
}

export interface DriftFeature {
  feature_name: string;
  type?: string;
  method?: string;
  statistic: number | null;
  p_value: number | null;
  p_value_corrected?: number | null;
  dof?: number | null;
  effect_size?: number | null;
  cramers_v?: number | null;
  drifted: boolean;
  status?: string;
  severity?: string;
}

export interface DriftResult {
  overall_drifted: boolean;
  alpha: number;
  correction?: string;
  min_production_rows: number;
  production_rows: number;
  sufficient_data: boolean;
  status: string;
  features: DriftFeature[];
}

export interface PredictionRecord {
  id: number;
  timestamp: string;
  use_case: string;
  model_name: string;
  model_version: string | null;
  input_data: Record<string, number | string>;
  prediction: number;
}

export interface HousePricePredictionInput {
  area_sqft: number;
  bedrooms: number;
  bathrooms: number;
  parking: number;
  location: string;
  property_age: number;
  floor_number: number;
  total_floors: number;
  furnishing_status: string;
  property_type: string;
}

export interface HousePricePredictionResponse {
  prediction: number;
  predicted_price_per_sqft: number | null;
  locality_benchmark_price: number | null;
  benchmark_price_per_sqft: number | null;
  benchmark_difference: number | null;
  benchmark_difference_percent: number | null;
  prediction_lower: number | null;
  prediction_upper: number | null;
  model_name: string | null;
  model_version: string | null;
  schema_version: number | null;
  prediction_id: number | null;
  inferred_at: string | null;
  location: string | null;
  area_sqft: number | null;
}

export interface ContributionEntry {
  feature: string;
  label: string;
  value: number | string;
  contribution: number;
  direction: 'positive' | 'negative' | 'neutral';
}

export interface MonitoringRecord {
  id: number;
  timestamp: string;
  use_case: string;
  overall_drifted: boolean;
  status: string;
  alpha: number;
  min_production_rows: number;
  production_rows: number;
  sufficient_data: boolean;
  features: DriftFeature[];
  model_name?: string | null;
  model_version?: string | null;
  schema_version?: number | null;
  severity?: string | null;
  overall_health?: string | null;
  data_quality_status?: string | null;
  drifted_features?: number | null;
}

export interface DataQualityCheck {
  name: string;
  status: string;
  checked: number;
  affected: number;
  message: string;
}

export interface DataQualityResult {
  status: string;
  total_rows: number;
  v2_rows: number;
  legacy_rows: number;
  malformed_rows: number;
  checks: DataQualityCheck[];
}

export interface DriftBlock {
  overall_drifted: boolean;
  method: string;
  correction: string;
  features: DriftFeature[];
}

export interface HealthSummary {
  timestamp: string;
  use_case: string;
  model_name: string | null;
  model_version: string | null;
  schema_version: number | null;
  total_predictions: number | null;
  window_limit: number;
  window_records: number;
  window_v2_records: number;
  window_legacy_records: number;
  window_malformed_records: number;
  status: string;
  overall_drifted: boolean;
  overall_health: string;
  health_reason: string;
  severity: string;
  severity_reason: string;
  drifted_features: string[];
  drifted_count: number;
  total_monitored: number;
  alpha: number;
  min_production_rows: number;
  production_rows: number;
  sufficient_data: boolean;
  reference_rows: number;
  numeric_drift: DriftBlock;
  categorical_drift: DriftBlock;
  features: DriftFeature[];
  data_quality: DataQualityResult;
  data_quality_status: string;
  monitoring_run_id?: number;
}

export interface PromotionRecord {
  id: number;
  timestamp: string;
  use_case: string;
  decision: string;
  model_name: string;
  run_id: string | null;
  registered_model_name: string | null;
  model_version: string | null;
  primary_metric: string | null;
  candidate_score: number | null;
  production_score: number | null;
  improvement: number | null;
  reason: string;
}

export interface ExplainResult {
  prediction: number;
  intercept: number | null;
  contributions: ContributionEntry[];
  method: string | null;
  model_name: string;
  model_version: string | null;
}

export interface LocalityBenchmark {
  location: string;
  benchmark_price_per_sqft: number;
  benchmark_source: string;
  calibration_note: string;
}

export interface BenchmarksResponse {
  items: LocalityBenchmark[];
  schema_version: number;
}

export interface CategoryShare {
  value: string;
  reference_count: number;
  reference_share: number;
  production_count: number;
  production_share: number;
}

export interface DistributionFeature {
  feature: string;
  type?: string;
  reference: { mean: number | null; std: number | null; count: number; histogram: { bins: number[]; counts: number[] } };
  production: { mean: number | null; std: number | null; count: number; histogram: { bins: number[]; counts: number[] } };
  categories?: CategoryShare[];
  reference_rows?: number;
  production_rows?: number;
}

export interface RegistryInfo {
  production: {
    model_name: string;
    model_version: string | null;
    registered_model_name: string;
    run_id: string | null;
    deployed_at: string | null;
    artifact_available?: boolean | null;
  } | null;
  runs?: RegistryCandidate[];
  pointer_history?: PointerHistoryEntry[];
  versions: PromotionRecord[];
  mlflow_versions: { name: string; version: string; run_id: string; status: string; stage: string }[];
  total: number;
}

export interface RetrainingDecision {
  recommended: boolean;
  reason: string;
  triggers: string[];
  blocked_by: string[];
  production_model: string | null;
  production_version: string | null;
  monitored_prediction_count: number | null;
}

export interface DecisionResponse {
  decision: RetrainingDecision;
  monitoring: {
    status: string;
    severity: string | null;
    overall_health: string | null;
    data_quality_status: string | null;
    production_rows: number | null;
    drifted_count: number | null;
    total_monitored: number | null;
  };
}

export interface MetricsTriple {
  mae: number;
  rmse: number;
  r2: number;
}

export interface ComparisonDeltas {
  mae_delta: number;
  rmse_delta: number;
  r2_improvement: number;
  mae_reduction_pct: number | null;
  rmse_reduction_pct: number | null;
  r2_improvement_pct: number | null;
}

export interface BaselineComparison {
  candidate: { model_name: string | null; metrics: MetricsTriple | null };
  production: { model_name: string | null; metrics: MetricsTriple | null } | null;
  deltas: ComparisonDeltas | null;
  status?: string;
}

export interface GateCheck {
  name: string;
  passed: boolean;
  message: string;
}

export interface RetrainingRun {
  id: number;
  run_id: string;
  use_case: string;
  status: string;
  trigger: string;
  started_at: string;
  completed_at: string | null;
  dataset_ref: string | null;
  dataset_rows: number | null;
  schema_version: number | null;
  train_rows: number | null;
  test_rows: number | null;
  random_seed: number | null;
  feature_schema: string[] | null;
  candidates: { model_type: string; hyperparameters: Record<string, unknown> } | null;
  production_model: string | null;
  production_version: string | null;
  best_model: string | null;
  best_metrics: (MetricsTriple & { all_results?: { model_name: string; metrics: MetricsTriple; run_id: string }[] }) | null;
  comparison: BaselineComparison | null;
  candidate_artifact: string | null;
  candidate_run_id: string | null;
  candidate_version: string | null;
  promotion_eligible: boolean | null;
  promotion_decision: string | null;
  gate_checks: GateCheck[] | null;
  failure_reason: string | null;
}

export interface PromotionCheckResult {
  run_id: string;
  eligible: boolean;
  checks: GateCheck[];
  comparison: BaselineComparison | null;
  production: { model_name: string | null; model_version: string | null };
}

export interface PromoteResult {
  run_id: string;
  promoted: boolean;
  eligible: boolean;
  checks: GateCheck[];
  comparison: BaselineComparison | null;
  production: { model_name: string | null; model_version: string | null; registered_model_name?: string | null; run_id?: string | null };
  previous_production_version: string | null;
  reason: string;
}

export interface PointerHistoryEntry {
  id: number;
  timestamp: string;
  use_case: string;
  previous_pointer: Record<string, unknown> | null;
  new_pointer: Record<string, unknown>;
  reason: string | null;
  run_id: string | null;
}

export interface RollbackResult {
  pointer_history_id: number;
  rolled_back: boolean;
  production: { model_name: string | null; model_version: string | null; registered_model_name?: string | null; run_id?: string | null };
  previous_production_version: string | null;
  reason: string;
}

export interface RegistryCandidate {
  run_id: string | null;
  status: string | null;
  started_at: string | null;
  completed_at: string | null;
  trigger: string | null;
  dataset_rows: number | null;
  schema_version: number | null;
  candidate_model: string | null;
  candidate_metrics: MetricsTriple | null;
  candidate_run_id: string | null;
  candidate_version: string | null;
  candidate_artifact: string | null;
  artifact_available: boolean | null;
  comparison_deltas: ComparisonDeltas | null;
  promotion_eligible: boolean | null;
  promotion_decision: string | null;
  failure_reason: string | null;
}

export interface RetrainResult {
  use_case?: string;
  best_model_name?: string;
  best_metrics?: { mae: number; rmse: number; r2: number };
  best_run_id?: string;
  all_results?: { model_name: string; metrics: { mae: number; rmse: number; r2: number }; run_id: string }[];
  promotion?: {
    decision: string;
    model_name?: string;
    run_id?: string | null;
    registered_model_name?: string | null;
    model_version?: string | null;
    primary_metric?: string | null;
    candidate_score?: number | null;
    production_score?: number | null;
    improvement?: number | null;
    reason?: string;
  };
  promotion_history_id?: number;
  monitoring?: DriftResult;
  trigger?: { decision?: string; retraining_required?: boolean } | string;
}

export interface DemoScenarioInfo {
  name: string;
  title: string;
  description: string;
  expected: string;
}

export interface DemoSession {
  id: number;
  session_id: string;
  use_case: string;
  scenario: string;
  status: 'active' | 'completed' | 'reset';
  sample_count: number;
  seed: number;
  params: Record<string, unknown> | null;
  samples_generated: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  reset_at: string | null;
  last_monitoring: HealthSummary | null;
  last_decision: RetrainingDecision | null;
  last_monitored_at: string | null;
}

export interface DemoGenerateResult {
  session: DemoSession;
  samples_generated: number;
  scenario: string;
}

export interface DemoMonitorResult {
  session: DemoSession;
  monitoring: HealthSummary;
  decision: RetrainingDecision;
}

export interface DemoResetResult {
  session: DemoSession;
  removed_demo_rows: number;
}
