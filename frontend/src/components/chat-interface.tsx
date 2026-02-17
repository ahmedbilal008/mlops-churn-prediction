"use client";

import { useState, useRef, useEffect, useCallback, type FormEvent } from "react";
import { Send, Bot, User, Wrench, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { ChatMessage, ToolCall } from "@/lib/types";

function generateId(): string {
  return Math.random().toString(36).substring(2, 10);
}

// Render markdown-like formatting (bold, code blocks, lists)
function FormattedContent({ text }: { text: string }) {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let inCodeBlock = false;
  let codeLines: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith("```")) {
      if (inCodeBlock) {
        elements.push(
          <pre
            key={`code-${i}`}
            className="bg-muted rounded-md p-3 my-2 text-sm overflow-x-auto"
          >
            <code>{codeLines.join("\n")}</code>
          </pre>
        );
        codeLines = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    // Headers
    if (line.startsWith("### ")) {
      elements.push(
        <h3 key={i} className="font-semibold text-base mt-3 mb-1">
          {formatInline(line.slice(4))}
        </h3>
      );
    } else if (line.startsWith("## ")) {
      elements.push(
        <h2 key={i} className="font-bold text-lg mt-3 mb-1">
          {formatInline(line.slice(3))}
        </h2>
      );
    } else if (line.startsWith("# ")) {
      elements.push(
        <h1 key={i} className="font-bold text-xl mt-3 mb-1">
          {formatInline(line.slice(2))}
        </h1>
      );
    }
    // Bullet list
    else if (line.match(/^[-*]\s/)) {
      elements.push(
        <div key={i} className="flex gap-2 ml-2">
          <span className="text-muted-foreground">•</span>
          <span>{formatInline(line.slice(2))}</span>
        </div>
      );
    }
    // Numbered list
    else if (line.match(/^\d+\.\s/)) {
      const match = line.match(/^(\d+)\.\s(.*)/);
      if (match) {
        elements.push(
          <div key={i} className="flex gap-2 ml-2">
            <span className="text-muted-foreground font-mono text-sm">
              {match[1]}.
            </span>
            <span>{formatInline(match[2])}</span>
          </div>
        );
      }
    }
    // Empty line → spacer
    else if (line.trim() === "") {
      elements.push(<div key={i} className="h-2" />);
    }
    // Normal paragraph
    else {
      elements.push(
        <p key={i} className="leading-relaxed">
          {formatInline(line)}
        </p>
      );
    }
  }

  return <div className="space-y-0.5">{elements}</div>;
}

// Inline formatting: **bold**, `code`, *italic*
function formatInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*(.+?)\*\*|`(.+?)`|\*(.+?)\*)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(
        <strong key={match.index} className="font-semibold">
          {match[2]}
        </strong>
      );
    } else if (match[3]) {
      parts.push(
        <code
          key={match.index}
          className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono"
        >
          {match[3]}
        </code>
      );
    } else if (match[4]) {
      parts.push(
        <em key={match.index}>{match[4]}</em>
      );
    }
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length === 1 ? parts[0] : <>{parts}</>;
}

// Tool call visualization
function ToolCallBadge({ toolCall }: { toolCall: ToolCall }) {
  const statusColor =
    toolCall.status === "success"
      ? "default"
      : toolCall.status === "error"
        ? "destructive"
        : "secondary";

  return (
    <div className="flex items-center gap-2 my-1">
      <Wrench className="h-3.5 w-3.5 text-muted-foreground" />
      <Badge variant={statusColor} className="text-xs font-mono">
        {toolCall.name}
      </Badge>
      {toolCall.status === "pending" && (
        <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
      )}
    </div>
  );
}

// Single message bubble
function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-muted-foreground"
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div
        className={`flex flex-col gap-1 max-w-[80%] ${
          isUser ? "items-end" : "items-start"
        }`}
      >
        <Card
          className={`px-4 py-3 ${
            isUser
              ? "bg-primary text-primary-foreground rounded-br-sm"
              : "bg-muted/50 rounded-bl-sm"
          }`}
        >
          {message.toolCalls && message.toolCalls.length > 0 && (
            <div className="mb-2 pb-2 border-b border-border/50">
              <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
                Tools Used
              </span>
              {message.toolCalls.map((tc, idx) => (
                <ToolCallBadge key={idx} toolCall={tc} />
              ))}
            </div>
          )}
          <div className="text-sm">
            <FormattedContent text={message.content} />
          </div>
        </Card>
        <span className="text-[10px] text-muted-foreground px-1">
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </span>
      </div>
    </div>
  );
}

// Example prompt suggestions
const SUGGESTIONS = [
  "What's the current system status?",
  "Show me the model leaderboard",
  "Predict churn for a senior citizen with fiber optic internet on a month-to-month contract",
  "What are the most important features for churn?",
  "Tell me about the dataset",
];

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;

      setError(null);
      const userMsg: ChatMessage = {
        id: generateId(),
        role: "user",
        content: text.trim(),
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setIsLoading(true);

      try {
        // Build history for Gemini context
        const history = messages.map((m) => ({
          role: m.role === "user" ? "user" : "model",
          content: m.content,
        }));

        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: text.trim(),
            history,
          }),
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.error || `Request failed: ${res.status}`);
        }

        const assistantMsg: ChatMessage = {
          id: generateId(),
          role: "assistant",
          content: data.content,
          toolCalls: data.toolCalls?.map(
            (tc: { name: string; args: Record<string, unknown>; result: unknown; status: string }) => ({
              name: tc.name,
              args: tc.args,
              result: JSON.stringify(tc.result),
              status: tc.status as "success" | "error",
            })
          ),
          timestamp: new Date(),
        };

        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Failed to send message";
        setError(msg);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, messages]
  );

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4" ref={scrollRef}>
        <div className="max-w-3xl mx-auto py-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <Bot className="h-12 w-12 text-muted-foreground mb-4" />
              <h2 className="text-xl font-semibold mb-2">
                Agentic Churn Intelligence
              </h2>
              <p className="text-muted-foreground text-sm max-w-md mb-8">
                Interact with ML models through an AI agent. Ask about churn predictions,
                model performance, SHAP explanations, or dataset insights.
              </p>
              <div className="flex flex-wrap gap-2 justify-center max-w-lg">
                {SUGGESTIONS.map((s, i) => (
                  <Button
                    key={i}
                    variant="outline"
                    size="sm"
                    className="text-xs h-auto py-2 px-3"
                    onClick={() => sendMessage(s)}
                  >
                    {s}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
                <Bot className="h-4 w-4" />
              </div>
              <Card className="px-4 py-3 bg-muted/50">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Thinking...
                </div>
              </Card>
            </div>
          )}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-4 mb-2 flex items-center gap-2 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto h-6 text-xs"
            onClick={() => setError(null)}
          >
            Dismiss
          </Button>
        </div>
      )}

      {/* Input area */}
      <div className="border-t bg-background p-4">
        <form
          onSubmit={handleSubmit}
          className="max-w-3xl mx-auto flex gap-2 items-end"
        >
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about churn predictions, model metrics, or dataset insights..."
            className="min-h-11 max-h-35 resize-none"
            rows={1}
            disabled={isLoading}
          />
          <Button
            type="submit"
            size="icon"
            disabled={isLoading || !input.trim()}
            className="h-11 w-11 shrink-0"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </form>
      </div>
    </div>
  );
}
