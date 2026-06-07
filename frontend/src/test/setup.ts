import { vi } from 'vitest';

Object.defineProperty(globalThis, 'crypto', {
  value: {
    randomUUID: vi.fn(() => `test-${Math.random().toString(16).slice(2)}`)
  },
  configurable: true
});
