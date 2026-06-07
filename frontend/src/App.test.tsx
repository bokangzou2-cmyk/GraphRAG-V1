import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import App from './App';

function mockChat(answer: string, suffix: string, confidence = 'high') {
  return {
    answer,
    answer_type: 'llm_grounded',
    citations: [{ title: `证据-${suffix}`, source_type: 'case_chunk', field: 'judgment_text', retrieval_id: `case:${suffix}` }],
    graph_paths: [{ edge_type: 'CASE_HAS_CHUNK', retrieval_id: `case:${suffix}` }],
    warnings: [`warning-${suffix}`],
    confidence: { level: confidence },
    retrieval_context: []
  };
}

function mockOrchestrationChat(answer: string, suffix: string) {
  return {
    ...mockChat(answer, suffix, 'high'),
    orchestration_logs: [
      { event: 'hybrid_retrieval', payload: { query_profile: { intents: ['sentence'] } } },
      { event: 'final_response', payload: { answer_type: 'llm_grounded' } }
    ]
  };
}

function mockConversationFetch(chatResponses: unknown[] = [], remoteConversations: unknown[] = []) {
  const queue = [...chatResponses];
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === '/api/auth/login' && init?.method === 'POST') {
      return {
        ok: true,
        json: async () => ({
          token: 'test-token',
          expiresAt: Date.now() + 60_000,
          user: { id: 'user-1', username: 'admin', role: 'admin', createdAt: 1 }
        })
      };
    }
    if (url === '/api/auth/logout' && init?.method === 'POST') {
      return { ok: true, json: async () => ({ ok: true }) };
    }
    if (url === '/api/conversations' && (!init?.method || init.method === 'GET')) {
      return { ok: true, json: async () => ({ conversations: remoteConversations }) };
    }
    if (url.startsWith('/api/chat/audit-runs')) {
      return {
        ok: true,
        json: async () => ({
          runs: [{
            id: 'audit-1',
            createdAt: 1,
            path: 'orchestration',
            query: '吴必定案判了多久',
            topK: 10,
            answerType: 'retrieval_fallback',
            confidenceLevel: 'medium',
            warnings: [],
            citationIds: ['case:first'],
            graphPathCount: 1,
            retrievalContextCount: 3,
            answerPreview: '审计摘要'
          }]
        })
      };
    }
    if (url.startsWith('/api/conversations/') && init?.method === 'PUT') {
      return { ok: true, json: async () => JSON.parse(String(init.body)) };
    }
    if (url.startsWith('/api/conversations/') && init?.method === 'DELETE') {
      return { ok: true, json: async () => ({ deleted: true }) };
    }
    if (url === '/api/chat/orchestration') {
      const payload = queue.shift();
      return { ok: true, json: async () => payload };
    }
    return { ok: false, status: 404, text: async () => 'not found' };
  });
}

async function loginUser(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText('用户名'), 'admin');
  await user.type(screen.getByLabelText('密码'), '1');
  await user.click(screen.getByRole('button', { name: '登录' }));
  expect(await screen.findByPlaceholderText('输入刑法相关问题')).toBeTruthy();
  expect(screen.queryByText('最近提问')).toBeNull();
}

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

afterEach(() => {
  cleanup();
});

test('stores sources per assistant turn and restores them after refresh', async () => {
  const fetchMock = mockConversationFetch([
    mockChat('第一轮回答', 'first', 'high'),
    mockChat('第二轮回答', 'second', 'medium')
  ]);
  vi.stubGlobal('fetch', fetchMock);
  const user = userEvent.setup();

  const view = render(<App />);
  await loginUser(user);
  await user.type(screen.getByPlaceholderText('输入刑法相关问题'), '吴必定案判了多久');
  await user.click(screen.getByRole('button', { name: '发送' }));
  expect(await screen.findByText('第一轮回答')).toBeTruthy();
  expect(screen.getByText('证据-first')).toBeTruthy();
  expect(screen.getByText('warning-first')).toBeTruthy();
  expect(screen.getByText('较高')).toBeTruthy();

  await user.type(screen.getByPlaceholderText('输入刑法相关问题'), '有哪些依据');
  await user.click(screen.getByRole('button', { name: '发送' }));
  expect(await screen.findByText('第二轮回答')).toBeTruthy();
  expect(screen.getByText('证据-second')).toBeTruthy();
  expect(screen.getByText('warning-second')).toBeTruthy();
  expect(screen.getByText('中等')).toBeTruthy();

  const chatCalls = fetchMock.mock.calls.filter((call) => call[0] === '/api/chat/orchestration');
  const secondPayload = JSON.parse(chatCalls[1][1].body);
  expect(chatCalls).toHaveLength(2);
  expect(secondPayload.messages.map((message: { role: string }) => message.role)).toEqual(['user', 'assistant', 'user']);

  await user.click(screen.getByRole('button', { name: '第一轮回答' }));
  expect(screen.getByText('证据-first')).toBeTruthy();
  expect(screen.getByText('warning-first')).toBeTruthy();
  expect(screen.getByText('较高')).toBeTruthy();

  view.unmount();
  render(<App />);
  expect(screen.getByText('第二轮回答')).toBeTruthy();
  expect(screen.getByText('证据-second')).toBeTruthy();
  expect(screen.getByText('warning-second')).toBeTruthy();

  await user.click(screen.getByRole('button', { name: '第一轮回答' }));
  expect(screen.getByText('证据-first')).toBeTruthy();
  expect(screen.getByText('warning-first')).toBeTruthy();
  expect(screen.getByText('较高')).toBeTruthy();
});

test('migrates legacy lastResponse into the latest assistant turn', async () => {
  vi.stubGlobal('fetch', mockConversationFetch());
  localStorage.setItem('criminal-law-graphrag-conversations', JSON.stringify([{
    id: 'legacy-conversation',
    title: '旧对话',
    createdAt: 1,
    messages: [
      { role: 'user', content: '第一问' },
      { role: 'assistant', content: '旧回答' }
    ],
    lastResponse: mockChat('旧回答', 'legacy', 'low')
  }]));

  render(<App />);
  await loginUser(userEvent.setup());
  expect(screen.getByText('旧回答')).toBeTruthy();
  expect(screen.getByText('证据-legacy')).toBeTruthy();
  expect(screen.getByText('warning-legacy')).toBeTruthy();
  expect(screen.getByText('待核验')).toBeTruthy();
});

test('uses orchestration path as the only frontend answer path', async () => {
  const fetchMock = mockConversationFetch([
    mockOrchestrationChat('第一轮编排回答', 'orchestration-first'),
    mockOrchestrationChat('第二轮编排回答', 'orchestration-second')
  ]);
  vi.stubGlobal('fetch', fetchMock);
  const user = userEvent.setup();

  render(<App />);
  await loginUser(user);
  await user.type(screen.getByPlaceholderText('输入刑法相关问题'), '吴必定案判了多久');
  await user.click(screen.getByRole('button', { name: '发送' }));
  expect(await screen.findByText('第一轮编排回答')).toBeTruthy();
  expect(fetchMock.mock.calls.some((call) => call[0] === '/api/chat')).toBe(false);
  expect(screen.queryByRole('button', { name: '稳定' })).toBeNull();
  expect(screen.queryByRole('button', { name: '实验' })).toBeNull();
  expect(screen.getByText('智能编排')).toBeTruthy();

  await user.type(screen.getByPlaceholderText('输入刑法相关问题'), '吴必定案有哪些证据');
  await user.click(screen.getByRole('button', { name: '发送' }));
  expect(await screen.findByText('第二轮编排回答')).toBeTruthy();
  expect(fetchMock.mock.calls.some((call) => call[0] === '/api/chat/orchestration')).toBe(true);
  expect(screen.getByText('关联线索')).toBeTruthy();

  const orchestrationCall = fetchMock.mock.calls.find((call) => call[0] === '/api/chat/orchestration');
  const payload = JSON.parse(orchestrationCall?.[1].body);
  expect(payload.mock_llm_mode).toBe('grounded');
  expect(payload.generation_mode).toBe('provider');
});

test('keeps deleted conversations hidden after refresh even if remote list is stale', async () => {
  const deletedConversation = {
    id: 'delete-me',
    title: '要删除的对话',
    createdAt: 1,
    updatedAt: 2,
    messages: [{ role: 'user', content: '盗窃罪怎么判' }]
  };
  const fetchMock = mockConversationFetch([], [deletedConversation]);
  vi.stubGlobal('fetch', fetchMock);
  const user = userEvent.setup();

  const view = render(<App />);
  await loginUser(user);
  expect(await screen.findByText('要删除的对话')).toBeTruthy();

  await user.click(screen.getByRole('button', { name: '删除对话 要删除的对话' }));
  expect(screen.queryByText('要删除的对话')).toBeNull();
  expect(fetchMock.mock.calls.some((call) => String(call[0]).includes('/api/conversations/delete-me') && call[1]?.method === 'DELETE')).toBe(true);

  view.unmount();
  render(<App />);
  await waitFor(() => {
    expect(fetchMock.mock.calls.filter((call) => call[0] === '/api/conversations')).toHaveLength(2);
  });
  expect(screen.queryByText('要删除的对话')).toBeNull();
});

test('undo removes the latest user turn and assistant response', async () => {
  const fetchMock = mockConversationFetch([
    mockOrchestrationChat('第一轮回答', 'first'),
    mockOrchestrationChat('第二轮回答', 'second')
  ]);
  vi.stubGlobal('fetch', fetchMock);
  const user = userEvent.setup();

  render(<App />);
  await loginUser(user);
  const undoButton = screen.getByRole('button', { name: '撤销上条消息' });
  expect((undoButton as HTMLButtonElement).disabled).toBe(true);

  await user.type(screen.getByPlaceholderText('输入刑法相关问题'), '盗窃罪怎么判');
  await user.click(screen.getByRole('button', { name: '发送' }));
  expect(await screen.findByText('第一轮回答')).toBeTruthy();

  await user.type(screen.getByPlaceholderText('输入刑法相关问题'), '有没有类似案件');
  await user.click(screen.getByRole('button', { name: '发送' }));
  expect(await screen.findByText('第二轮回答')).toBeTruthy();

  await user.click(undoButton);
  expect(screen.queryByText('有没有类似案件')).toBeNull();
  expect(screen.queryByText('第二轮回答')).toBeNull();
  expect(screen.getAllByText('盗窃罪怎么判').length).toBeGreaterThan(0);
  expect(screen.getByText('第一轮回答')).toBeTruthy();

  await waitFor(() => {
    const savedPayloads = fetchMock.mock.calls
      .filter((call) => String(call[0]).startsWith('/api/conversations/') && call[1]?.method === 'PUT')
      .map((call) => JSON.parse(String(call[1]?.body)));
    const latest = savedPayloads.at(-1);
    expect(latest?.messages.map((message: { content: string }) => message.content)).toEqual(['盗窃罪怎么判', '第一轮回答']);
  });
});
