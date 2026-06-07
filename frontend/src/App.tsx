import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { Cloud, CloudOff, LogOut, Plus, RotateCcw, Search, Trash2 } from 'lucide-react';
import { deleteConversation, listConversations, login, logout, saveConversation, sendOrchestrationChat } from './api';
import type { AuthSession, ChatMessage, ChatResponse, Conversation } from './types';

const STORAGE_KEY = 'criminal-law-graphrag-conversations';
const DELETED_STORAGE_KEY = 'criminal-law-graphrag-deleted-conversations';
const AUTH_STORAGE_KEY = 'criminal-law-graphrag-auth';

function createConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: '新的对话',
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now()
  };
}

function findLatestAssistantIndex(messages: ChatMessage[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'assistant') {
      return index;
    }
  }
  return -1;
}

function getResponseAt(messages: ChatMessage[], index: number): ChatResponse | undefined {
  const message = messages[index];
  return message?.role === 'assistant' ? message.response : undefined;
}

function findLatestUserIndex(messages: ChatMessage[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') {
      return index;
    }
  }
  return -1;
}

function displayAnswerType(value?: string): string {
  const labels: Record<string, string> = {
    llm_grounded: '已根据资料回答',
    llm_direct: '直接回答',
    retrieval_fallback: '基于检索资料回答',
    low_confidence: '资料有限',
    no_answer: '无法确认',
    law_articles: '法条回答',
    applied_articles: '适用法条',
    applied_reasoning: '适用理由',
    fact: '案件事实',
    evidence: '证据材料',
    defense: '辩护意见',
    reasoning: '裁判理由',
    sentence: '量刑结果',
    amount: '涉案数额',
    fallback: '资料摘要'
  };
  return value ? labels[value] ?? value : '-';
}

function displayConfidence(value?: unknown): string {
  const labels: Record<string, string> = {
    high: '较高',
    medium: '中等',
    low: '待核验',
    none: '无'
  };
  return typeof value === 'string' ? labels[value] ?? value : '-';
}

function displayWarning(value: string): string {
  const labels: Record<string, string> = {
    insufficient_context: '资料不足',
    out_of_scope: '超出当前知识库范围',
    case_not_found: '未找到对应案例',
    historical_law_context_missing: '缺少历史法条原文',
    no_current_law_chunk: '未补充现行法条正文',
    fallback_answer: '使用资料摘要回答',
    direct_general_legal_answer: '刑法常识直接回答',
    provider_citation_count_decreased: '已补充引用资料'
  };
  return labels[value] ?? value;
}

function userVisibleWarnings(warnings?: string[]): string[] {
  const hidden = new Set([
    'low_confidence',
    'llm_call_failed',
    'llm_invalid_json',
    'llm_not_configured',
    'llm_skipped_by_confidence_gate',
    'query_rewriter:lora',
    'query_understanding:lora'
  ]);
  return (warnings ?? []).filter((warning) => !warning.startsWith('router:') && !hidden.has(warning));
}

function displayRefusal(value?: string | null): string {
  if (!value) return '-';
  const labels: Record<string, string> = {
    insufficient_context: '资料不足，无法可靠回答',
    low_confidence: '资料不足，无法可靠回答',
    out_of_scope: '超出当前刑法样本范围'
  };
  return labels[value] ?? value;
}

function graphPathSummary(path: Record<string, unknown>): string {
  const edge = String(path.edge_type ?? '关联');
  const article = String(path.article ?? path.article_no ?? '');
  const version = String(path.law_version ?? '');
  const caseTitle = String(path.case_title ?? path.case_id ?? '');
  return [caseTitle, edge, article, version].filter(Boolean).join(' / ');
}

function metricNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function formatMs(value: unknown): string {
  const number = metricNumber(value);
  return typeof number === 'number' ? `${number} ms` : '-';
}

function nodeDuration(metrics: Record<string, unknown> | undefined, node: string): number | undefined {
  const nodes = metrics?.nodes;
  if (!nodes || typeof nodes !== 'object') return undefined;
  const item = (nodes as Record<string, unknown>)[node];
  if (!item || typeof item !== 'object') return undefined;
  return metricNumber((item as Record<string, unknown>).duration_ms);
}

function providerUsage(metrics: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  const tokenUsage = metrics?.token_usage;
  if (!tokenUsage || typeof tokenUsage !== 'object') return undefined;
  const provider = (tokenUsage as Record<string, unknown>).provider_llm;
  return provider && typeof provider === 'object' ? provider as Record<string, unknown> : undefined;
}

function rewriterUsage(metrics: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  const tokenUsage = metrics?.token_usage;
  if (!tokenUsage || typeof tokenUsage !== 'object') return undefined;
  const rewriter = (tokenUsage as Record<string, unknown>).query_rewriter;
  return rewriter && typeof rewriter === 'object' ? rewriter as Record<string, unknown> : undefined;
}

function normalizeConversation(raw: Conversation): Conversation {
  const messages = Array.isArray(raw.messages) ? [...raw.messages] : [];
  const latestAssistantIndex = findLatestAssistantIndex(messages);

  if (raw.lastResponse && latestAssistantIndex >= 0) {
    const latestAssistant = messages[latestAssistantIndex];
    if (latestAssistant?.role === 'assistant' && !latestAssistant.response) {
      messages[latestAssistantIndex] = {
        ...latestAssistant,
        response: raw.lastResponse
      };
    }
  }

  return {
    ...raw,
    messages,
    updatedAt: raw.updatedAt ?? raw.createdAt
  };
}

function mergeConversations(localItems: Conversation[], remoteItems: Conversation[]): Conversation[] {
  const byId = new Map<string, Conversation>();
  const deletedIds = loadDeletedConversationIds();
  [...localItems, ...remoteItems].forEach((item) => {
    const normalized = normalizeConversation(item);
    if (deletedIds.has(normalized.id)) return;
    const current = byId.get(normalized.id);
    if (!current || (normalized.updatedAt ?? normalized.createdAt) >= (current.updatedAt ?? current.createdAt)) {
      byId.set(normalized.id, normalized);
    }
  });
  return [...byId.values()].sort((a, b) => (b.updatedAt ?? b.createdAt) - (a.updatedAt ?? a.createdAt));
}

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    if (!Array.isArray(parsed) || parsed.length === 0) return [createConversation()];
    const deletedIds = loadDeletedConversationIds();
    const items = parsed.map(normalizeConversation).filter((item) => !deletedIds.has(item.id));
    return items.length ? items : [createConversation()];
  } catch {
    return [createConversation()];
  }
}

function loadDeletedConversationIds(): Set<string> {
  try {
    const raw = localStorage.getItem(DELETED_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return new Set(Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : []);
  } catch {
    return new Set();
  }
}

function rememberDeletedConversation(conversationId: string) {
  const deletedIds = loadDeletedConversationIds();
  deletedIds.add(conversationId);
  localStorage.setItem(DELETED_STORAGE_KEY, JSON.stringify([...deletedIds]));
}

function loadSession(): AuthSession | null {
  try {
    const raw = localStorage.getItem(AUTH_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed?.token && parsed?.expiresAt > Date.now() ? parsed : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [session, setSession] = useState<AuthSession | null>(loadSession);
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState(conversations[0]?.id);
  const [selectedAssistantIndex, setSelectedAssistantIndex] = useState(() =>
    findLatestAssistantIndex(conversations[0]?.messages ?? [])
  );
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [historySync, setHistorySync] = useState<'syncing' | 'synced' | 'local'>('syncing');
  const [loginForm, setLoginForm] = useState({ username: 'admin', password: '' });
  const [loginError, setLoginError] = useState('');
  const didLoadRemote = useRef(false);

  const active = useMemo(
    () => conversations.find((item) => item.id === activeId) ?? conversations[0],
    [activeId, conversations]
  );
  const selectedResponse = useMemo(() => {
    const activeMessages = active?.messages ?? [];
    return getResponseAt(activeMessages, selectedAssistantIndex)
      ?? getResponseAt(activeMessages, findLatestAssistantIndex(activeMessages));
  }, [active, selectedAssistantIndex]);
  const canUndoLastMessage = !loading && findLatestUserIndex(active?.messages ?? []) >= 0;
  const visibleWarnings = userVisibleWarnings(selectedResponse?.warnings);
  const selectedMetrics = selectedResponse?.metrics as Record<string, unknown> | undefined;
  const selectedProviderUsage = providerUsage(selectedMetrics);
  const selectedRewriterUsage = rewriterUsage(selectedMetrics);

  useEffect(() => {
    if (!session?.token) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    if (!didLoadRemote.current) return;
    void Promise.all(conversations.map((conversation) => saveConversation(conversation, session.token)))
      .then(() => setHistorySync('synced'))
      .catch(() => setHistorySync('local'));
  }, [conversations, session?.token]);

  useEffect(() => {
    if (!session?.token) return;
    let cancelled = false;
    setHistorySync('syncing');
    void listConversations(session.token)
      .then((remoteItems) => {
        if (cancelled) return;
        setConversations((localItems) => {
          const merged = remoteItems.length ? mergeConversations(localItems, remoteItems) : localItems;
          setActiveId((current) => merged.find((item) => item.id === current)?.id ?? merged[0]?.id);
          return merged;
        });
        didLoadRemote.current = true;
        setHistorySync('synced');
      })
      .catch(() => {
        if (cancelled) return;
        didLoadRemote.current = true;
        setHistorySync('local');
      });
    return () => {
      cancelled = true;
    };
  }, [session?.token]);

  useEffect(() => {
    setSelectedAssistantIndex(findLatestAssistantIndex(active?.messages ?? []));
  }, [active?.id, active?.messages]);

  function updateActive(updater: (conversation: Conversation) => Conversation) {
    setConversations((items) => items.map((item) => (item.id === active.id ? updater(item) : item)));
  }

  function newChat() {
    const item = createConversation();
    setConversations((items) => [item, ...items]);
    setActiveId(item.id);
    setDraft('');
    setError('');
  }

  async function removeConversation(conversationId: string) {
    const remaining = conversations.filter((item) => item.id !== conversationId);
    const nextItems = remaining.length ? remaining : [createConversation()];
    rememberDeletedConversation(conversationId);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextItems));
    setConversations(nextItems);
    setActiveId((current) => (current === conversationId ? nextItems[0]?.id : current));
    setError('');
    if (session?.token) {
      await deleteConversation(conversationId, session.token).catch(() => setHistorySync('local'));
    }
  }

  function undoLastMessage() {
    if (loading) return;
    const messages = active?.messages ?? [];
    const latestUserIndex = findLatestUserIndex(messages);
    if (latestUserIndex < 0) return;
    const nextMessages = messages.slice(0, latestUserIndex);
    updateActive((conversation) => ({
      ...conversation,
      title: nextMessages.length === 0 ? '新的对话' : conversation.title,
      messages: nextMessages,
      updatedAt: Date.now()
    }));
    setSelectedAssistantIndex(findLatestAssistantIndex(nextMessages));
    setError('');
  }

  async function submit() {
    const content = draft.trim();
    if (!content || loading) return;
    const userMessage: ChatMessage = { role: 'user', content };
    const nextMessages = [...active.messages, userMessage];
    updateActive((conversation) => ({
      ...conversation,
      title: conversation.messages.length === 0 ? content.slice(0, 24) : conversation.title,
      messages: nextMessages,
      updatedAt: Date.now()
    }));
    setDraft('');
    setLoading(true);
    setError('');
    try {
      const response: ChatResponse = await sendOrchestrationChat(nextMessages, session?.token ?? '');
      const nextAssistantIndex = nextMessages.length;
      updateActive((conversation) => ({
        ...conversation,
        messages: [...nextMessages, { role: 'assistant', content: response.answer, response }],
        updatedAt: Date.now()
      }));
      setSelectedAssistantIndex(nextAssistantIndex);
    } catch (err) {
      setError(err instanceof Error ? err.message : '请求失败');
    } finally {
      setLoading(false);
    }
  }

  async function submitLogin(event: FormEvent) {
    event.preventDefault();
    setLoginError('');
    try {
      const nextSession = await login(loginForm.username.trim(), loginForm.password);
      localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(nextSession));
      setSession(nextSession);
      didLoadRemote.current = false;
    } catch (err) {
      setLoginError(err instanceof Error ? err.message : '登录失败');
    }
  }

  async function signOut() {
    if (session?.token) {
      await logout(session.token).catch(() => undefined);
    }
    localStorage.removeItem(AUTH_STORAGE_KEY);
    setSession(null);
    setHistorySync('local');
  }

  if (!session) {
    return (
      <main className="login-shell">
        <form className="login-panel" onSubmit={(event) => void submitLogin(event)}>
          <h1>刑事法律检索助手</h1>
          <p>登录后进入受控检索问答系统</p>
          <label>
            用户名
            <input
              value={loginForm.username}
              onChange={(event) => setLoginForm((current) => ({ ...current, username: event.target.value }))}
              autoComplete="username"
            />
          </label>
          <label>
            密码
            <input
              type="password"
              value={loginForm.password}
              onChange={(event) => setLoginForm((current) => ({ ...current, password: event.target.value }))}
              autoComplete="current-password"
            />
          </label>
          {loginError && <div className="error">{loginError}</div>}
          <button type="submit" disabled={!loginForm.username.trim() || !loginForm.password}>
            登录
          </button>
        </form>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <button className="new-chat" onClick={newChat} title="新建对话">
          <Plus size={18} />
          <span>新建对话</span>
        </button>
        <div className="history">
          {conversations.map((item) => (
            <div key={item.id} className={`history-row ${item.id === active.id ? 'active' : ''}`}>
              <button className="history-item" onClick={() => setActiveId(item.id)} title={item.title}>
                {item.title}
              </button>
              <button
                className="delete-chat"
                onClick={(event) => {
                  event.stopPropagation();
                  void removeConversation(item.id);
                }}
                title="删除对话"
                aria-label={`删除对话 ${item.title}`}
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <section className="chat-pane">
        <header className="topbar">
          <div>
            <h1>刑事法律检索助手</h1>
            <p>查询刑法条文、案例事实、证据、裁判理由和量刑信息</p>
          </div>
          <div className="topbar-actions">
            <span className="route-label" title="使用检索、图谱、引用校验和模型编排生成回答">智能编排</span>
            <div className="sync-status" title={historySync === 'synced' ? '后端会话存储已同步' : '后端会话存储不可用，正在使用浏览器本地缓存'}>
              {historySync === 'synced' ? <Cloud size={16} /> : <CloudOff size={16} />}
              <span>{historySync === 'synced' ? '已同步' : historySync === 'syncing' ? '同步中' : '本地'}</span>
            </div>
            <button className="icon-button" onClick={() => void signOut()} title={`退出 ${session.user.username}`}>
              <LogOut size={18} />
            </button>
            <Search size={20} />
          </div>
        </header>
        <div className="messages">
          {active.messages.length === 0 ? (
            <div className="empty-state">
              <h2>询问刑法条文、案例事实、裁判理由或适用法条</h2>
              <p>例如：吴必定案适用了哪些法条？故意伤害罪相关刑法条文是什么？</p>
            </div>
          ) : (
            active.messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                {message.role === 'assistant' ? (
                  <button
                    type="button"
                    className={`bubble assistant-bubble ${selectedAssistantIndex === index ? 'selected' : ''}`}
                    onClick={() => setSelectedAssistantIndex(index)}
                  >
                    {message.content}
                  </button>
                ) : (
                  <div className="bubble">{message.content}</div>
                )}
              </div>
            ))
          )}
          {loading && <div className="message assistant"><div className="bubble muted">检索并生成中...</div></div>}
          {error && <div className="error">{error}</div>}
        </div>
        <footer className="composer">
          <button
            className="composer-icon-button"
            onClick={undoLastMessage}
            disabled={!canUndoLastMessage}
            title="撤销上条消息"
            aria-label="撤销上条消息"
          >
            <RotateCcw size={18} />
          </button>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                void submit();
              }
            }}
            placeholder="输入刑法相关问题"
          />
          <button onClick={() => void submit()} disabled={loading || !draft.trim()}>
            发送
          </button>
        </footer>
      </section>

      <aside className="sources-panel">
        <h2>回答依据</h2>
        <div className="meta-row">
          <span>回答状态</span>
          <strong>{displayAnswerType(selectedResponse?.answer_type)}</strong>
        </div>
        <div className="meta-row">
          <span>置信度</span>
          <strong>
            {displayConfidence(selectedResponse?.confidence?.level)}
            {typeof selectedResponse?.confidence?.score === 'number' ? ` ${selectedResponse.confidence.score}` : ''}
          </strong>
        </div>
        <div className="meta-row">
          <span>无法回答原因</span>
          <strong>{displayRefusal(selectedResponse?.refusal_reason)}</strong>
        </div>
        <section>
          <h3>链路指标</h3>
          <div className="meta-row">
            <span>总耗时</span>
            <strong>{formatMs(selectedMetrics?.total_duration_ms)}</strong>
          </div>
          <div className="metric-grid">
            {[
              ['改写', 'start'],
              ['路由', 'router_result'],
              ['法条', 'law_retriever'],
              ['案例', 'case_retriever'],
              ['图谱', 'graph_retriever'],
              ['向量', 'vector_retriever'],
              ['融合', 'hybrid_merge_rerank'],
              ['回答', 'grounded_answer']
            ].map(([label, node]) => (
              <div className="metric-item" key={node}>
                <span>{label}</span>
                <strong>{formatMs(nodeDuration(selectedMetrics, node))}</strong>
              </div>
            ))}
          </div>
          {selectedProviderUsage && (
            <div className="token-row">
              <span>DeepSeek Token</span>
              <strong>
                {String(selectedProviderUsage.total_tokens ?? '-')}
                <small>
                  prompt {String(selectedProviderUsage.prompt_tokens ?? '-')} / completion {String(selectedProviderUsage.completion_tokens ?? '-')}
                </small>
              </strong>
            </div>
          )}
          {selectedRewriterUsage && (
            <div className="token-row">
              <span>Rewriter Token</span>
              <strong>
                {String(selectedRewriterUsage.total_tokens ?? '-')}
                <small>
                  prompt {String(selectedRewriterUsage.prompt_tokens ?? '-')} / completion {String(selectedRewriterUsage.completion_tokens ?? '-')}
                </small>
              </strong>
            </div>
          )}
        </section>
        {!!visibleWarnings.length && (
          <section className="warning-section">
            <h3>提示</h3>
            {visibleWarnings.map((warning) => <span key={warning} className="tag">{displayWarning(warning)}</span>)}
          </section>
        )}
        <section>
          <h3>引用资料</h3>
          {(selectedResponse?.citations ?? []).length ? (selectedResponse?.citations ?? []).map((citation, index) => (
            <article className="source-card" key={`${citation.retrieval_id}-${index}`}>
              <strong>{citation.title}</strong>
              <span>{citation.source_type === 'law_article' ? '刑法条文' : '案例材料'}{citation.field ? ` / ${citation.field}` : ''}</span>
              {(citation.article_no || citation.law_version) && (
                <span>条号 {citation.article_no ?? '-'} / 版本 {citation.law_version ?? '-'}</span>
              )}
              {citation.evidence_preview && <p>{citation.evidence_preview}</p>}
            </article>
          )) : <p className="empty-panel-text">当前回答没有可展示的引用资料。</p>}
        </section>
        <section>
          <h3>关联线索</h3>
          {(selectedResponse?.graph_paths ?? []).length ? (selectedResponse?.graph_paths ?? []).slice(0, 6).map((path, index) => (
            <div className="path-card readable-path" key={index}>{graphPathSummary(path)}</div>
          )) : <p className="empty-panel-text">当前回答没有额外的案例-法条关联线索。</p>}
        </section>
      </aside>
    </main>
  );
}
