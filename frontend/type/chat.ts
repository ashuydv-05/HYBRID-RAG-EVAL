export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  isStreaming?: boolean;
  isError?: boolean;
  timestamp: Date;
}

export interface Source {
  id: number;
  title: string;
  score: number;
  year?: number;
  arxiv_id?: string;
  paper_id?: string;
  section?: string;
  categories: string[];
  authors: string[];
  pdf_url?: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  stream?: boolean;
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  reasoning_steps: ReasoningStep[];
  sources: Source[];
  execution_time: number;
  total_tokens: number;
}

export interface ReasoningStep {
  thought: string;
  action?: string;
  action_input?: Record<string, unknown>;
  observation?: string;
}

export interface StreamChunk {
  type: 'token' | 'sources' | 'done' | 'error';
  data: string | Record<string, unknown>;
}
