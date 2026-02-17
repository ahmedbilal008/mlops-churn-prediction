// Type definitions for the Churn Intelligence Platform frontend

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool";
  content: string;
  toolCalls?: ToolCall[];
  timestamp: Date;
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  result?: string;
  status: "pending" | "success" | "error";
}

// MCP tool response types

export interface PredictionResult {
  churn_probability: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  top_drivers: { feature: string; shap_value: number }[];
  model_used: string;
}

export interface ModelMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  confusion_matrix?: Record<string, number>;
}

export interface LeaderboardEntry {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
}

export interface CompareModelsResult {
  leaderboard: LeaderboardEntry[];
  best_model: string;
  ranking_metric: string;
  total_models: number;
}

export interface DatasetSummary {
  total_rows: number;
  total_features: number;
  numerical_features: string[];
  categorical_features: string[];
  missing_values: Record<string, number>;
  class_distribution: Record<string, number>;
  churn_rate: number;
  tenure_stats: { mean: number; median: number; max: number };
  monthly_charges_stats: { mean: number; median: number; max: number };
}

export interface ActiveModelInfo {
  model_name: string;
  trained_at: string;
  dataset_size: number;
  feature_count: number;
  models_trained: string[];
  metrics: Record<string, number>;
}

export interface SystemStatus {
  health: "healthy" | "degraded" | "error";
  model_loaded: boolean;
  preprocessor_ready: boolean;
  shap_ready: boolean;
  data_available: boolean;
  last_training_time: string;
  retrain_in_progress: boolean;
  pipeline_available: boolean;
  server_uptime: string;
}

export interface RetrainResult {
  status: "success" | "rejected" | "error";
  model_type?: string;
  metrics?: Record<string, number>;
  mlflow_run_id?: string;
  trained_at?: string;
  message?: string;
  error?: string;
}
