import type { AuditRun, AuthSession, ChatMessage, ChatResponse, Conversation } from './types';

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

async function errorMessage(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) return `HTTP ${response.status}`;
  try {
    const data = JSON.parse(text) as { detail?: unknown };
    if (data.detail === 'invalid_credentials') return '用户名或密码错误';
    if (typeof data.detail === 'string') return data.detail;
  } catch {
    // Plain text response.
  }
  return text;
}

export async function login(username: string, password: string): Promise<AuthSession> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return response.json();
}

export async function logout(token: string): Promise<void> {
  await fetch('/api/auth/logout', {
    method: 'POST',
    headers: authHeaders(token)
  });
}

export async function listConversations(token: string): Promise<Conversation[]> {
  const response = await fetch('/api/conversations', { headers: authHeaders(token) });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const data = await response.json();
  return Array.isArray(data.conversations) ? data.conversations : [];
}

export async function saveConversation(conversation: Conversation, token: string): Promise<Conversation> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(conversation.id)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders(token) },
    body: JSON.stringify(conversation)
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

export async function deleteConversation(conversationId: string, token: string): Promise<void> {
  const response = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
    headers: authHeaders(token)
  });
  if (!response.ok && response.status !== 404) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
}

export async function listAuditRuns(token: string, limit = 8): Promise<AuditRun[]> {
  const response = await fetch(`/api/chat/audit-runs?limit=${limit}`, { headers: authHeaders(token) });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const data = await response.json();
  return Array.isArray(data.runs) ? data.runs : [];
}

export async function sendOrchestrationChat(
  messages: ChatMessage[],
  token: string,
  topK = 10,
  mockLlmMode = 'grounded',
  generationMode = 'provider'
): Promise<ChatResponse> {
  const response = await fetch('/api/chat/orchestration', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders(token) },
    body: JSON.stringify({
      messages: messages.map((message) => ({ role: message.role, content: message.content })),
      top_k: topK,
      mock_llm_mode: mockLlmMode,
      generation_mode: generationMode
    })
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}
