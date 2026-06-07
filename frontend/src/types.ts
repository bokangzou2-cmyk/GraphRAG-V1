export type Role = 'user' | 'assistant';

export interface Citation {
  retrieval_id?: string;
  chunk_id?: string;
  title?: string;
  source_type?: string;
  field?: string;
  case_id?: string;
  source_file?: string;
  text_hash?: string;
  article_no?: string;
  law_version?: string;
  hybrid_score?: number;
  retrieval_sources?: string[];
  evidence_preview?: string;
}

export interface ChatResponse {
  answer: string;
  answer_type: string;
  citations: Citation[];
  graph_paths: Record<string, unknown>[];
  warnings: string[];
  confidence: Record<string, unknown>;
  refusal_reason?: string | null;
  retrieval_context: Record<string, unknown>[];
  orchestration_logs?: Record<string, unknown>[];
  metrics?: Record<string, unknown>;
}

export interface UserMessage {
  role: 'user';
  content: string;
}

export interface AssistantMessage {
  role: 'assistant';
  content: string;
  response?: ChatResponse;
}

export type ChatMessage = UserMessage | AssistantMessage;

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  lastResponse?: ChatResponse;
  createdAt: number;
  updatedAt?: number;
}

export interface AuthUser {
  id: string;
  username: string;
  role: 'admin' | 'user' | 'auditor';
  createdAt: number;
  lastLoginAt?: number | null;
}

export interface AuthSession {
  token: string;
  expiresAt: number;
  user: AuthUser;
}

export interface AuditRun {
  id: string;
  createdAt: number;
  path: 'stable' | 'orchestration';
  query: string;
  topK: number;
  answerType: string;
  confidenceLevel?: string | null;
  confidenceScore?: number | null;
  refusalReason?: string | null;
  warnings: string[];
  citationIds: string[];
  graphPathCount: number;
  retrievalContextCount: number;
  answerPreview: string;
  metrics?: Record<string, unknown>;
}
