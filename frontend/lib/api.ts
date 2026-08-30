const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (process.env.NODE_ENV === "development"
    ? "http://localhost:8000/api"
    : "https://arxiv-rag-backend.onrender.com/api");

export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface ChatSource {
  title: string;
  content: string;
  id?: number;
  score?: number;
  source?: string;
  metadata?: Record<string, unknown>;
}

export interface ReasoningStep {
  thought: string;
  action?: string;
  observation?: string;
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  reasoning_steps: ReasoningStep[];
  sources: ChatSource[];
  execution_time: number;
  node_timings?: Record<string, number>;
}

export async function sendMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (typeof window !== "undefined") {
    const groqKey = localStorage.getItem("groq_api_key");
    const geminiKey = localStorage.getItem("gemini_api_key");
    const openaiKey = localStorage.getItem("openai_api_key");

    if (groqKey) headers["x-groq-api-key"] = groqKey;
    if (geminiKey) headers["x-gemini-api-key"] = geminiKey;
    if (openaiKey) headers["x-openai-api-key"] = openaiKey;
  }

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMessage = `Request failed with status ${response.status}`;

    try {
      const errorData = await response.json();
      if (errorData?.detail) {
        errorMessage = errorData.detail;
      }
    } catch {
      // Keep default error message
    }

    throw new Error(errorMessage);
  }

  return response.json();
}