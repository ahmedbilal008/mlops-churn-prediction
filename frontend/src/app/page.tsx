import ChatInterface from "@/components/chat-interface";
import Sidebar from "@/components/sidebar";

export default function Home() {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Main chat area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="border-b px-6 py-3 flex items-center gap-3 shrink-0">
          <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">CI</span>
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-none">
              Churn Intelligence Platform
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              MCP · MLflow · Gemini · SHAP
            </p>
          </div>
        </header>

        {/* Chat */}
        <div className="flex-1 overflow-hidden">
          <ChatInterface />
        </div>
      </main>

      {/* Sidebar */}
      <aside className="w-80 border-l bg-muted/20 hidden lg:flex flex-col shrink-0">
        <Sidebar />
      </aside>
    </div>
  );
}
