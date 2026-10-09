import { test, expect } from '@playwright/test';
import { readEvents } from '../src/api.js';

test('SSE parser preserves Korean characters and events across network boundaries', async () => {
  const bytes = new TextEncoder().encode('data: {"stage":"complete","result":{"text":"현장 확인"}}\n\n');
  const chunks = Array.from(bytes, byte => new Uint8Array([byte]));
  const response = new Response(new ReadableStream({ start(controller) { chunks.forEach(c => controller.enqueue(c)); controller.close(); } }));
  const received = [];
  await readEvents(response, event => received.push(event));
  expect(received).toEqual([{ stage: 'complete', result: { text: '현장 확인' } }]);
});

test('author submits, assigned supervisor approves, exports and verifies actions', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByLabel('접속 토큰', { exact: true }).fill('a'.repeat(32));
  await page.getByRole('button', { name: '로그인', exact: true }).click();
  await expect(page.getByText('author · 작성자')).toBeVisible();
  await page.locator('input[type=file]').setInputFiles('../assets/test_data/worker_safety_sample.png');
  await page.getByRole('button', { name: '분석 요청', exact: true }).click();
  await expect(page.getByRole('heading', { name: '초안', exact: true })).toBeVisible();
  for (const [label, value] of [
    ['사업장', '서울 현장'], ['공정', '프레스'], ['작업', '정비'], ['평가일', '2026-10-09'],
    ['평가 참여자', '작업자 및 감독자'], ['현장 위험도 척도 및 허용 판단 기준', '현장 자체 기준에 따라 조치 후 재확인'],
    ['현장 확인 의견 (누락·오탐 검토, 위험요인 미등록 시 그 근거)', '작업자와 현장을 확인함'],
    ['예상 피해', '손 부상'], ['현재 안전조치', '출입 제한'], ['가능성 (1–5)', '3'], ['중대성 (1–5)', '4'],
    ['조치 담당자', 'author'], ['조치 기한', '2026-10-10'],
  ]) await page.getByLabel(label, { exact: true }).fill(value);
  await page.getByLabel('담당 관리감독자', { exact: true }).selectOption('boss');
  await page.getByRole('button', { name: '초안 저장', exact: true }).click();
  await expect(page.getByRole('button', { name: '관리감독자 검토 요청', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: '관리감독자 검토 요청', exact: true }).click();
  await expect(page.getByText('boss 관리감독자의 확인을 기다리고 있습니다.')).toBeVisible();
  const id = await page.getByLabel('평가서 선택', { exact: true }).inputValue();
  await page.getByLabel('접속 토큰', { exact: true }).fill('b'.repeat(32));
  await page.getByRole('button', { name: '사용자 전환', exact: true }).click();
  await expect(page.getByText('boss · 관리감독자')).toBeVisible();
  await page.getByLabel('평가서 선택', { exact: true }).selectOption(id);
  await page.getByLabel('관리감독자 확인 의견 (현장 확인 및 평가 적정성)', { exact: true }).fill('현장 확인 및 개선계획 승인');
  await page.getByRole('button', { name: '원본 현장 영상·이미지 확인', exact: true }).click();
  await expect(page.getByAltText('분석 당시 원본 현장 이미지')).toBeVisible();
  for (const label of ['원본 현장자료 확인', '위험요인 누락·오탐 확인', '개선조치·담당자·기한 확인', '위험도·허용 기준 확인']) await page.getByLabel(label, { exact: true }).check();
  await page.getByRole('button', { name: '확인 후 승인', exact: true }).click();
  await expect(page.getByRole('heading', { name: '승인 · 조치 진행', exact: true })).toBeVisible();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Excel 평가서', exact: true }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toContain(id);
  await page.getByLabel('완료 내용', { exact: true }).fill('방호장치 설치 완료');
  await page.getByLabel('완료 증빙 (문서·사진 위치 또는 현장 확인 내용)', { exact: true }).fill('현장 사진 A');
  await page.getByLabel('조치 후 가능성 (1–5)', { exact: true }).fill('1');
  await page.getByLabel('조치 후 중대성 (1–5)', { exact: true }).fill('4');
  await page.getByRole('button', { name: '조치 완료 등록', exact: true }).click();
  await page.getByLabel('관리감독자 재확인 의견 (잔여 위험 허용 여부 포함)', { exact: true }).fill('설치 확인, 잔여 위험 허용');
  await page.getByRole('button', { name: '조치 재확인', exact: true }).click();
  await expect(page.getByRole('heading', { name: '끼임 위험 · 재확인 완료', exact: true })).toBeVisible();
  await page.getByLabel('종결 의견', { exact: true }).fill('개선조치 완료 및 현장 재확인');
  await page.getByRole('button', { name: '모든 조치 확인 후 종결', exact: true }).click();
  await expect(page.getByRole('heading', { name: '종결', exact: true })).toBeVisible();
  await page.getByText(/분석 원본과 변경 이력/).click();
  await page.getByRole('button', { name: '이력 무결성 확인', exact: true }).click();
  await expect(page.getByText('변경 이력 무결성 확인 완료', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('duplicate submission does not create another job and jobs survive browser reload', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('접속 토큰', { exact: true }).fill('a'.repeat(32));
  await page.getByRole('button', { name: '로그인', exact: true }).click();
  await expect(page.getByText('author · 작성자')).toBeVisible();
  await page.getByLabel('현장 사진', { exact: true }).setInputFiles('../assets/test_data/worker_safety_sample.png');
  const first = page.waitForResponse(r => r.url().endsWith('/jobs') && r.request().method() === 'POST');
  await page.getByRole('button', { name: '분석 요청', exact: true }).click();
  const job = await (await first).json();
  const again = page.waitForResponse(r => r.url().endsWith('/jobs') && r.request().method() === 'POST');
  await page.getByRole('button', { name: '분석 요청', exact: true }).click();
  expect((await (await again).json()).id).toBe(job.id);
  await expect(page.getByRole('status')).toContainText('중복 분석은 실행하지 않습니다');
  await page.reload();
  await page.getByLabel('접속 토큰', { exact: true }).fill('a'.repeat(32));
  await page.getByRole('button', { name: '로그인', exact: true }).click();
  await expect(page.getByText(/동일 요청 재전송 [1-9]/)).toBeVisible();
  await expect(page.getByRole('button', { name: '평가서 열기', exact: true }).first()).toBeVisible();
});

test('manual drafting creates an explicit non-AI record without an inference attempt', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('접속 토큰', { exact: true }).fill('a'.repeat(32));
  await page.getByRole('button', { name: '로그인', exact: true }).click();
  await expect(page.getByText('author · 작성자')).toBeVisible();
  await page.getByLabel('현장 사진', { exact: true }).setInputFiles('../assets/test_data/worker_safety_sample.png');
  const submitted = page.waitForResponse(r => r.url().endsWith('/jobs') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'AI 없이 직접 작성', exact: true }).click();
  const job = await (await submitted).json();
  expect(job.profile).toBe('manual-v1');
  expect(job.attempt).toBe(0);
  expect(job.attempts).toEqual([]);
  await expect(page.getByText('AI 분석 없이 직접 작성', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '위험요인 추가', exact: true })).toBeVisible();
});
