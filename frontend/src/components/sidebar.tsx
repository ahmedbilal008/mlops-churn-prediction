"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Activity,
  BarChart3,
  Database,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
  Trophy,
  Cpu,
  Clock,
  Wrench,
  ChevronDown,
  ChevronRight,
  Info,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type {
  SystemStatus,
  ActiveModelInfo,
  CompareModelsResult,
  DatasetSummary,
} from "@/lib/types";

// Helper to call our MCP bridge
async function callTool<T>(tool: string, args: Record<string, unknown> = {}): Promise<T> {
  const res = await fetch(`/api/mcp/${tool}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ args }),
  });
  const data = await res.json();
  if (data.status === "error") throw new Error(data.error);
  return data.result as T;
}

// --- System Status Panel ---
function StatusPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await callTool<SystemStatus>("system_status");
      setStatus(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !status) {
    return (
      <div className="text-center py-6">
        <XCircle className="h-8 w-8 text-destructive mx-auto mb-2" />
        <p className="text-sm text-muted-foreground">
          {error || "Backend offline"}
        </p>
        <Button variant="ghost" size="sm" className="mt-2" onClick={refresh}>
          <RefreshCw className="h-3 w-3 mr-1" /> Retry
        </Button>
      </div>
    );
  }

  const healthColor =
    status.health === "healthy"
      ? "text-green-500"
      : status.health === "degraded"
        ? "text-yellow-500"
        : "text-red-500";

  const StatusRow = ({
    label,
    ok,
  }: {
    label: string;
    ok: boolean;
  }) => (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      {ok ? (
        <CheckCircle2 className="h-4 w-4 text-green-500" />
      ) : (
        <XCircle className="h-4 w-4 text-red-400" />
      )}
    </div>
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className={`h-4 w-4 ${healthColor}`} />
          <span className="font-medium text-sm capitalize">{status.health}</span>
        </div>
        <Button variant="ghost" size="icon" className="h-7 w-7" onClick={refresh}>
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
      </div>
      <Separator />
      <div className="space-y-2">
        <StatusRow label="Model Loaded" ok={status.model_loaded} />
        <StatusRow label="Preprocessor" ok={status.preprocessor_ready} />
        <StatusRow label="SHAP Ready" ok={status.shap_ready} />
        <StatusRow label="Data Available" ok={status.data_available} />
        <StatusRow label="Pipeline" ok={status.pipeline_available} />
      </div>
      {status.retrain_in_progress && (
        <>
          <Separator />
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-yellow-500" />
            <span className="text-yellow-600 font-medium">Retraining in progress...</span>
          </div>
        </>
      )}
    </div>
  );
}

// --- Active Model Panel ---
function ModelPanel() {
  const [model, setModel] = useState<ActiveModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    callTool<ActiveModelInfo>("get_active_model_info")
      .then(setModel)
      .catch(() => setModel(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!model) {
    return (
      <p className="text-sm text-muted-foreground text-center py-4">
        No model info available
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Trophy className="h-4 w-4 text-yellow-500" />
        <span className="font-medium text-sm font-mono">
          {model.model_name.replace(/_/g, " ")}
        </span>
      </div>
      <Separator />
      <div className="grid grid-cols-2 gap-y-2 text-sm">
        {Object.entries(model.metrics || {}).map(([key, val]) => (
          <div key={key}>
            <span className="text-muted-foreground text-xs uppercase">
              {key.replace(/_/g, " ")}
            </span>
            <p className="font-mono font-medium">
              {typeof val === "number" ? val.toFixed(3) : String(val)}
            </p>
          </div>
        ))}
      </div>
      <Separator />
      <div className="space-y-1 text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <Database className="h-3 w-3" />
          {model.dataset_size?.toLocaleString()} samples ·{" "}
          {model.feature_count} features
        </div>
        <div className="flex items-center gap-1.5">
          <Clock className="h-3 w-3" />
          {model.trained_at || "Unknown"}
        </div>
        <div className="flex items-center gap-1.5">
          <Cpu className="h-3 w-3" />
          {model.models_trained?.join(", ") || "N/A"}
        </div>
      </div>
    </div>
  );
}

// --- Leaderboard Panel ---
function LeaderboardPanel() {
  const [data, setData] = useState<CompareModelsResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    callTool<CompareModelsResult>("compare_models")
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!data || !data.leaderboard) {
    return (
      <p className="text-sm text-muted-foreground text-center py-4">
        No models to compare
      </p>
    );
  }

  return (
    <div className="space-y-2">
      {data.leaderboard.map((entry, idx) => (
        <div
          key={entry.model}
          className={`flex items-center gap-3 p-2 rounded-md ${
            idx === 0 ? "bg-yellow-500/10" : "bg-muted/30"
          }`}
        >
          <span className="text-lg font-bold text-muted-foreground w-6 text-center">
            {idx + 1}
          </span>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium font-mono truncate">
              {entry.model.replace(/_/g, " ")}
            </p>
            <div className="flex gap-3 text-xs text-muted-foreground">
              <span>F1: {entry.f1_score?.toFixed(3)}</span>
              <span>AUC: {entry.roc_auc?.toFixed(3)}</span>
            </div>
          </div>
          {idx === 0 && (
            <Badge variant="default" className="text-[10px]">
              Best
            </Badge>
          )}
        </div>
      ))}
    </div>
  );
}

// --- Dataset Summary Panel ---
function DataPanel() {
  const [data, setData] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    callTool<DatasetSummary>("get_dataset_summary")
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!data) {
    return (
      <p className="text-sm text-muted-foreground text-center py-4">
        No dataset info available
      </p>
    );
  }

  return (
    <div className="space-y-3 text-sm">
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-muted/30 rounded-md p-2 text-center">
          <p className="text-lg font-bold">{data.total_rows?.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground">Rows</p>
        </div>
        <div className="bg-muted/30 rounded-md p-2 text-center">
          <p className="text-lg font-bold">{data.total_features}</p>
          <p className="text-xs text-muted-foreground">Features</p>
        </div>
      </div>
      <div className="bg-muted/30 rounded-md p-2 text-center">
        <p className="text-lg font-bold">
          {data.churn_rate != null ? (data.churn_rate * 100).toFixed(1) : "N/A"}%
        </p>
        <p className="text-xs text-muted-foreground">Churn Rate</p>
      </div>
      <Separator />
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground font-medium uppercase">
          Class Distribution
        </p>
        {data.class_distribution &&
          Object.entries(data.class_distribution).map(([k, v]) => (
            <div key={k} className="flex justify-between text-xs">
              <span>{k}</span>
              <span className="font-mono">{v.toLocaleString()}</span>
            </div>
          ))}
      </div>
    </div>
  );
}

// --- Retrain Button ---
function RetrainButton() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleRetrain = async () => {
    setLoading(true);
    setResult(null);
    try {
      const data = await callTool<{ status: string; message?: string }>("retrain_model");
      setResult(
        data.status === "success"
          ? "Retrain complete!"
          : data.message || "Retrain rejected"
      );
    } catch (err) {
      setResult(err instanceof Error ? err.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2">
      <Button
        variant="outline"
        size="sm"
        className="w-full"
        disabled={loading}
        onClick={handleRetrain}
      >
        {loading ? (
          <>
            <Loader2 className="h-3.5 w-3.5 mr-2 animate-spin" />
            Retraining...
          </>
        ) : (
          <>
            <RefreshCw className="h-3.5 w-3.5 mr-2" />
            Retrain Pipeline
          </>
        )}
      </Button>
      {result && (
        <p className="text-xs text-center text-muted-foreground">{result}</p>
      )}
    </div>
  );
}

// --- MCP Tools Reference Data ---
interface ToolParam {
  name: string;
  type: string;
  desc: string;
  default?: string;
  key?: boolean;
}

interface ToolDef {
  name: string;
  description: string;
  params: ToolParam[];
}

const MCP_TOOLS: ToolDef[] = [
  {
    name: "predict_churn",
    description: "Predict customer churn probability with SHAP-based explanations of top contributing factors.",
    params: [
      { name: "tenure", type: "int", desc: "Months as customer (0-72)", default: "29", key: true },
      { name: "MonthlyCharges", type: "float", desc: "Monthly charge amount", default: "70.35", key: true },
      { name: "Contract", type: "string", desc: "Month-to-month / One year / Two year", default: "Month-to-month", key: true },
      { name: "InternetService", type: "string", desc: "DSL / Fiber optic / No", default: "Fiber optic", key: true },
      { name: "TotalCharges", type: "float", desc: "Total charges to date", default: "1397.47" },
      { name: "gender", type: "string", desc: "Male / Female", default: "Male" },
      { name: "SeniorCitizen", type: "int", desc: "0 or 1", default: "0" },
      { name: "Partner", type: "string", desc: "Yes / No", default: "No" },
      { name: "Dependents", type: "string", desc: "Yes / No", default: "No" },
      { name: "PhoneService", type: "string", desc: "Yes / No", default: "Yes" },
      { name: "MultipleLines", type: "string", desc: "Yes / No / No phone service", default: "No" },
      { name: "OnlineSecurity", type: "string", desc: "Yes / No / No internet service", default: "No" },
      { name: "OnlineBackup", type: "string", desc: "Yes / No / No internet service", default: "No" },
      { name: "DeviceProtection", type: "string", desc: "Yes / No / No internet service", default: "No" },
      { name: "TechSupport", type: "string", desc: "Yes / No / No internet service", default: "No" },
      { name: "StreamingTV", type: "string", desc: "Yes / No / No internet service", default: "No" },
      { name: "StreamingMovies", type: "string", desc: "Yes / No / No internet service", default: "No" },
      { name: "PaperlessBilling", type: "string", desc: "Yes / No", default: "Yes" },
      { name: "PaymentMethod", type: "string", desc: "Electronic check / Mailed check / Bank transfer / Credit card", default: "Electronic check" },
    ],
  },
  {
    name: "explain_prediction",
    description: "Get detailed SHAP explanation for a prediction. Same params as predict_churn — all optional with dataset defaults.",
    params: [],
  },
  {
    name: "get_model_metrics",
    description: "Get performance metrics (accuracy, precision, recall, F1, ROC-AUC) for a specific model.",
    params: [
      { name: "model_name", type: "string", desc: "logistic_regression / random_forest / xgboost", key: true },
    ],
  },
  {
    name: "compare_models",
    description: "Compare all trained models side-by-side, ranked by F1 score.",
    params: [],
  },
  {
    name: "get_dataset_summary",
    description: "Get dataset overview: row count, feature types, churn rate, and statistics.",
    params: [],
  },
  {
    name: "get_feature_importance",
    description: "Get global SHAP feature importance rankings across all customers.",
    params: [
      { name: "top_n", type: "int", desc: "Number of top features to return", default: "10" },
    ],
  },
  {
    name: "retrain_model",
    description: "Trigger full retraining pipeline. Trains all 3 models, selects best by F1, logs to MLflow.",
    params: [],
  },
  {
    name: "add_customer_record",
    description: "Add a new customer record to the dataset.",
    params: [
      { name: "customer data", type: "object", desc: "Full customer features + Churn label", key: true },
    ],
  },
  {
    name: "get_active_model_info",
    description: "Get info about the deployed model: name, metrics, training time, dataset size.",
    params: [],
  },
  {
    name: "system_status",
    description: "Check platform health: model loaded, SHAP ready, data available, retrain status.",
    params: [],
  },
];

// --- Single Tool Card ---
function ToolCard({ tool }: { tool: ToolDef }) {
  const [expanded, setExpanded] = useState(false);
  const hasParams = tool.params.length > 0;

  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={() => hasParams && setExpanded(!expanded)}
        className={`w-full text-left px-3 py-2.5 flex items-start gap-2 transition-colors ${
          hasParams ? "hover:bg-muted/50 cursor-pointer" : "cursor-default"
        }`}
      >
        <Wrench className="h-3.5 w-3.5 mt-0.5 text-muted-foreground shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-semibold truncate">
              {tool.name}
            </span>
            {hasParams && (
              <Badge variant="secondary" className="text-[9px] px-1 py-0 h-4 shrink-0">
                {tool.params.length} param{tool.params.length !== 1 ? "s" : ""}
              </Badge>
            )}
            {!hasParams && (
              <Badge variant="outline" className="text-[9px] px-1 py-0 h-4 shrink-0">
                no params
              </Badge>
            )}
          </div>
          <p className="text-[11px] text-muted-foreground mt-0.5 leading-tight">
            {tool.description}
          </p>
        </div>
        {hasParams && (
          <span className="shrink-0 mt-0.5">
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
            )}
          </span>
        )}
      </button>

      {expanded && hasParams && (
        <div className="border-t bg-muted/20 px-3 py-2 space-y-1.5">
          <div className="flex items-center gap-1 mb-1">
            <Info className="h-3 w-3 text-muted-foreground" />
            <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">
              Parameters
            </span>
          </div>
          {tool.params.map((p) => (
            <div key={p.name} className="flex items-start gap-2 text-[11px]">
              <code className="bg-background px-1 py-0.5 rounded text-[10px] font-mono shrink-0 border">
                {p.name}
              </code>
              <div className="flex-1 min-w-0">
                <span className="text-muted-foreground">{p.desc}</span>
                {p.default && (
                  <span className="text-muted-foreground/70 ml-1">
                    (default: {p.default})
                  </span>
                )}
              </div>
              {"key" in p && p.key && (
                <Badge variant="default" className="text-[8px] px-1 py-0 h-3.5 shrink-0">
                  key
                </Badge>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Tools Panel ---
function ToolsPanel() {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 mb-3">
        <Info className="h-3.5 w-3.5 text-muted-foreground" />
        <p className="text-[11px] text-muted-foreground leading-tight">
          These MCP tools are available to the AI assistant. Click any tool with parameters to see details.
        </p>
      </div>
      {MCP_TOOLS.map((tool) => (
        <ToolCard key={tool.name} tool={tool} />
      ))}
    </div>
  );
}

// --- Combined Sidebar ---
export default function Sidebar() {
  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="p-4 border-b">
        <h2 className="font-semibold text-sm flex items-center gap-2">
          <BarChart3 className="h-4 w-4" />
          Platform Dashboard
        </h2>
      </div>

      <Tabs defaultValue="tools" className="flex-1 flex flex-col overflow-hidden">
        <TabsList className="mx-4 mt-3 grid grid-cols-4 h-8">
          <TabsTrigger value="tools" className="text-xs">Tools</TabsTrigger>
          <TabsTrigger value="status" className="text-xs">Status</TabsTrigger>
          <TabsTrigger value="models" className="text-xs">Models</TabsTrigger>
          <TabsTrigger value="data" className="text-xs">Data</TabsTrigger>
        </TabsList>

        <div className="flex-1 overflow-y-auto px-4 py-3">
          <TabsContent value="tools" className="mt-0 space-y-4">
            <Card>
              <CardHeader className="pb-2 pt-3 px-3">
                <CardTitle className="text-xs font-medium uppercase text-muted-foreground tracking-wider">
                  MCP Tools Reference
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 pb-3">
                <ToolsPanel />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="status" className="mt-0 space-y-4">
            <Card>
              <CardHeader className="pb-2 pt-3 px-3">
                <CardTitle className="text-xs font-medium uppercase text-muted-foreground tracking-wider">
                  System Health
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 pb-3">
                <StatusPanel />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2 pt-3 px-3">
                <CardTitle className="text-xs font-medium uppercase text-muted-foreground tracking-wider">
                  Actions
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 pb-3">
                <RetrainButton />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="models" className="mt-0 space-y-4">
            <Card>
              <CardHeader className="pb-2 pt-3 px-3">
                <CardTitle className="text-xs font-medium uppercase text-muted-foreground tracking-wider">
                  Active Model
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 pb-3">
                <ModelPanel />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2 pt-3 px-3">
                <CardTitle className="text-xs font-medium uppercase text-muted-foreground tracking-wider">
                  Leaderboard
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 pb-3">
                <LeaderboardPanel />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="data" className="mt-0 space-y-4">
            <Card>
              <CardHeader className="pb-2 pt-3 px-3">
                <CardTitle className="text-xs font-medium uppercase text-muted-foreground tracking-wider">
                  Dataset Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="px-3 pb-3">
                <DataPanel />
              </CardContent>
            </Card>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
