// Chat API route: Gemini integration with MCP tool calling
// Route: POST /api/chat

import { NextRequest, NextResponse } from "next/server";
import {
  GoogleGenerativeAI,
  Content,
  Part,
  FunctionCallPart,
} from "@google/generative-ai";
import { mcpToolDeclarations } from "@/lib/mcp-tools";

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash-preview-05-20";

const SYSTEM_INSTRUCTION = `You are an AI assistant for the Agentic Churn Intelligence Platform — a production MLOps system for telecom customer churn prediction.

You have access to tools that interact with the ML backend:
- predict_churn: Predict churn probability for a customer
- explain_prediction: Explain the last prediction using SHAP
- get_model_metrics: Get metrics for a specific model
- compare_models: Compare all models on a leaderboard
- get_dataset_summary: Get dataset statistics
- get_feature_importance: Get global feature importance
- retrain_model: Retrigger the training pipeline
- get_active_model_info: Get info about the active model
- system_status: Check platform health

Guidelines:
- When users ask about a customer, use predict_churn with only the features they mention — all missing features automatically use dataset defaults (most common values from the training data)
- DO NOT fabricate values for features the user didn't mention — just omit them and the server will use defaults
- After predictions, offer to explain the drivers using explain_prediction
- Present data in a clear, formatted way using markdown
- For model comparisons, summarize the key findings
- Be concise but insightful — act like a data science colleague
- If the system is unhealthy, suggest checking the backend
- NEVER fabricate data — always use tools to get real information`;

// Build the Gemini client (lazy, fails gracefully if no key)
function getGenAI() {
  if (!GEMINI_API_KEY) {
    throw new Error(
      "GEMINI_API_KEY is not set. Add it to frontend/.env.local"
    );
  }
  return new GoogleGenerativeAI(GEMINI_API_KEY);
}

// Bridge a function call to the MCP backend via our internal API route
async function callMCPTool(
  toolName: string,
  args: Record<string, unknown>,
  origin: string
): Promise<{ result: unknown; error?: string }> {
  try {
    const res = await fetch(`${origin}/api/mcp/${toolName}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ args }),
    });

    const data = await res.json();
    if (data.status === "error") {
      return { result: null, error: data.error };
    }
    return { result: data.result };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "MCP call failed";
    return { result: null, error: msg };
  }
}

export async function POST(request: NextRequest) {
  try {
    const { messages, history } = await request.json();
    const userMessage: string = messages;

    if (!userMessage?.trim()) {
      return NextResponse.json(
        { error: "Empty message" },
        { status: 400 }
      );
    }

    const genAI = getGenAI();
    const model = genAI.getGenerativeModel({
      model: GEMINI_MODEL,
      systemInstruction: SYSTEM_INSTRUCTION,
      tools: [{ functionDeclarations: mcpToolDeclarations }],
    });

    // Build conversation history for context
    const contents: Content[] = [];
    if (Array.isArray(history)) {
      for (const msg of history) {
        if (msg.role === "user" || msg.role === "model") {
          contents.push({
            role: msg.role,
            parts: [{ text: msg.content }],
          });
        }
      }
    }

    // Add the current user message
    contents.push({
      role: "user",
      parts: [{ text: userMessage }],
    });

    // Start a chat session
    const chat = model.startChat({
      history: contents.slice(0, -1), // all except last
    });

    // Send the message
    let result = await chat.sendMessage(userMessage);
    let response = result.response;

    // Collect tool calls for the response
    const toolCalls: {
      name: string;
      args: Record<string, unknown>;
      result: unknown;
      status: "success" | "error";
    }[] = [];

    // Determine the origin for internal API calls
    const origin = request.nextUrl.origin;

    // Handle function calling loop (Gemini may request multiple tool calls)
    let maxIterations = 5;
    while (maxIterations > 0) {
      const candidate = response.candidates?.[0];
      const parts = candidate?.content?.parts || [];

      // Check if there are function calls
      const functionCalls = parts.filter(
        (p: Part): p is FunctionCallPart => "functionCall" in p
      );

      if (functionCalls.length === 0) break;

      // Execute all function calls
      const functionResponses: Part[] = [];

      for (const fc of functionCalls) {
        const { name, args } = fc.functionCall;
        console.log(`[Chat API] Calling MCP tool: ${name}`, args);

        const mcpResult = await callMCPTool(
          name,
          (args as Record<string, unknown>) || {},
          origin
        );

        const toolCall = {
          name,
          args: (args as Record<string, unknown>) || {},
          result: mcpResult.error || mcpResult.result,
          status: mcpResult.error ? ("error" as const) : ("success" as const),
        };
        toolCalls.push(toolCall);

        functionResponses.push({
          functionResponse: {
            name,
            response: mcpResult.error
              ? { error: mcpResult.error }
              : { result: mcpResult.result },
          },
        });
      }

      // Send function results back to Gemini
      result = await chat.sendMessage(functionResponses);
      response = result.response;
      maxIterations--;
    }

    // Extract the final text response
    const text =
      response.text() || "I processed your request but have no additional commentary.";

    return NextResponse.json({
      content: text,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error("[Chat API] Error:", message);
    return NextResponse.json(
      { error: message },
      { status: 500 }
    );
  }
}
