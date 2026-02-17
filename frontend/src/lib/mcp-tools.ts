// MCP tool definitions formatted for Gemini function calling schema
// Maps all 9 MCP tools to Gemini-compatible function declarations

import { FunctionDeclaration, SchemaType } from "@google/generative-ai";

export const mcpToolDeclarations: FunctionDeclaration[] = [
  {
    name: "predict_churn",
    description:
      "Predict customer churn probability. Provide customer features to get a churn risk prediction with SHAP-based explanations of the top contributing factors.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        gender: { type: SchemaType.STRING, description: "Customer gender: Male or Female" },
        SeniorCitizen: { type: SchemaType.INTEGER, description: "Is senior citizen: 0 or 1" },
        Partner: { type: SchemaType.STRING, description: "Has partner: Yes or No" },
        Dependents: { type: SchemaType.STRING, description: "Has dependents: Yes or No" },
        tenure: { type: SchemaType.INTEGER, description: "Months as customer (0-72)" },
        PhoneService: { type: SchemaType.STRING, description: "Has phone service: Yes or No" },
        MultipleLines: { type: SchemaType.STRING, description: "Multiple lines: Yes, No, or No phone service" },
        InternetService: { type: SchemaType.STRING, description: "Internet service type: DSL, Fiber optic, or No" },
        OnlineSecurity: { type: SchemaType.STRING, description: "Has online security: Yes, No, or No internet service" },
        OnlineBackup: { type: SchemaType.STRING, description: "Has online backup: Yes, No, or No internet service" },
        DeviceProtection: { type: SchemaType.STRING, description: "Has device protection: Yes, No, or No internet service" },
        TechSupport: { type: SchemaType.STRING, description: "Has tech support: Yes, No, or No internet service" },
        StreamingTV: { type: SchemaType.STRING, description: "Has streaming TV: Yes, No, or No internet service" },
        StreamingMovies: { type: SchemaType.STRING, description: "Has streaming movies: Yes, No, or No internet service" },
        Contract: { type: SchemaType.STRING, description: "Contract type: Month-to-month, One year, or Two year" },
        PaperlessBilling: { type: SchemaType.STRING, description: "Has paperless billing: Yes or No" },
        PaymentMethod: { type: SchemaType.STRING, description: "Payment method: Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)" },
        MonthlyCharges: { type: SchemaType.NUMBER, description: "Monthly charge amount (18.25-118.75)" },
        TotalCharges: { type: SchemaType.NUMBER, description: "Total charges to date" },
      },
      required: [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
      ],
    },
  },
  {
    name: "explain_prediction",
    description:
      "Get detailed SHAP explanation for a churn prediction. Returns each feature's SHAP contribution, the base value, and top positive/negative drivers.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        gender: { type: SchemaType.STRING, description: "Customer gender: Male or Female" },
        SeniorCitizen: { type: SchemaType.INTEGER, description: "Is senior citizen: 0 or 1" },
        Partner: { type: SchemaType.STRING, description: "Has partner: Yes or No" },
        Dependents: { type: SchemaType.STRING, description: "Has dependents: Yes or No" },
        tenure: { type: SchemaType.INTEGER, description: "Months as customer (0-72)" },
        PhoneService: { type: SchemaType.STRING, description: "Has phone service: Yes or No" },
        MultipleLines: { type: SchemaType.STRING, description: "Multiple lines: Yes, No, or No phone service" },
        InternetService: { type: SchemaType.STRING, description: "Internet service type: DSL, Fiber optic, or No" },
        OnlineSecurity: { type: SchemaType.STRING, description: "Has online security: Yes, No, or No internet service" },
        OnlineBackup: { type: SchemaType.STRING, description: "Has online backup: Yes, No, or No internet service" },
        DeviceProtection: { type: SchemaType.STRING, description: "Has device protection: Yes, No, or No internet service" },
        TechSupport: { type: SchemaType.STRING, description: "Has tech support: Yes, No, or No internet service" },
        StreamingTV: { type: SchemaType.STRING, description: "Has streaming TV: Yes, No, or No internet service" },
        StreamingMovies: { type: SchemaType.STRING, description: "Has streaming movies: Yes, No, or No internet service" },
        Contract: { type: SchemaType.STRING, description: "Contract type: Month-to-month, One year, or Two year" },
        PaperlessBilling: { type: SchemaType.STRING, description: "Has paperless billing: Yes or No" },
        PaymentMethod: { type: SchemaType.STRING, description: "Payment method: Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)" },
        MonthlyCharges: { type: SchemaType.NUMBER, description: "Monthly charge amount (18.25-118.75)" },
        TotalCharges: { type: SchemaType.NUMBER, description: "Total charges to date" },
      },
      required: [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
      ],
    },
  },
  {
    name: "get_model_metrics",
    description:
      "Retrieve performance metrics for a specific model including accuracy, precision, recall, F1-score and ROC-AUC.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        model_name: {
          type: SchemaType.STRING,
          description: "Model name: logistic_regression, random_forest, or xgboost",
        },
      },
      required: ["model_name"],
    },
  },
  {
    name: "compare_models",
    description:
      "Compare all trained models side-by-side and return a leaderboard ranked by the specified metric.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        metric: {
          type: SchemaType.STRING,
          description: "Metric to rank by: accuracy, precision, recall, f1_score, or roc_auc (default: f1_score)",
        },
      },
    },
  },
  {
    name: "get_dataset_summary",
    description:
      "Get a comprehensive summary of the churn dataset including feature types, class distribution, churn rate, and descriptive statistics.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {},
    },
  },
  {
    name: "get_feature_importance",
    description:
      "Get global feature importance rankings from SHAP analysis, showing which features most influence churn predictions across all customers.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        top_n: { type: SchemaType.INTEGER, description: "Number of top features to return (default: 10)" },
      },
    },
  },
  {
    name: "retrain_model",
    description:
      "Trigger a full model retraining pipeline. Trains all models (logistic regression, random forest, xgboost), selects the best by F1-score, and registers it with MLflow. Only one retrain can run at a time.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {},
    },
  },
  {
    name: "get_active_model_info",
    description:
      "Get information about the currently active (deployed) model including name, training time, dataset size, metrics, and the list of all models trained in the last run.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {},
    },
  },
  {
    name: "system_status",
    description:
      "Get the health status of the MLOps platform including model availability, SHAP readiness, data availability, retrain lock status, and server uptime.",
    parameters: {
      type: SchemaType.OBJECT,
      properties: {},
    },
  },
];

// Map of tool names to their corresponding MCP server endpoints
export const MCP_TOOL_NAMES = mcpToolDeclarations.map((t) => t.name);
