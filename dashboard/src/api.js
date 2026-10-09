export const API_BASE = import.meta.env?.VITE_API_BASE ?? '/api';

export async function api(path, token, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { Authorization: `Bearer ${token}`, ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || `요청 실패 (${response.status})`));
  }
  return response;
}

export async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split('\n');
      pending = lines.pop();
      if (done && pending.trim()) lines.push(pending);
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6));
        if (event.stage === 'error') throw new Error(event.detail);
        onEvent(event);
      }
      if (done) break;
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
