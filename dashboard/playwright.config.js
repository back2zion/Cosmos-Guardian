import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:5174', headless: true, screenshot: 'only-on-failure' },
  webServer: [
    {
      command: '../.venv/bin/python -m uvicorn browser_server:app --app-dir ../tests --host 127.0.0.1 --port 8899',
      env: { PYTHONPATH: '..', COSMOS_ALLOWED_ORIGINS: 'http://127.0.0.1:5174' },
      url: 'http://127.0.0.1:8899/health', reuseExistingServer: false,
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 5174 --strictPort',
      env: { VITE_API_BASE: 'http://127.0.0.1:8899' },
      url: 'http://127.0.0.1:5174', reuseExistingServer: false,
    },
  ],
});
