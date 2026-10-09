import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Shield, Upload, RefreshCw } from 'lucide-react';
import { api, API_BASE } from './api';
import ReviewPanel from './ReviewPanel';

const states = { queued: '대기', running: '분석 중', completed: '완료', failed: '실패', cancelled: '취소' };
const errors = {
  invalid_model_output: '결과 형식 오류 · 원본 확인 후 재시도하세요.',
  inference_failed: '모델 실행 실패 · 운영자에게 문의하세요.',
  worker_lost: '처리 서버 중단 · 재시도할 수 있습니다.',
  timeout: '분석 시간 초과 · 사진을 확인한 뒤 재시도하세요.',
  source_changed_or_missing: '원본 파일이 변경되었거나 없습니다.',
  processing_failed: '처리 실패 · 운영자에게 문의하세요.',
  worker_process_exited: '추론 프로세스가 종료됐습니다. 재시도할 수 있습니다.',
  cancelled: '요청자가 취소했습니다.',
};
const button = 'rounded-lg bg-blue-700 px-4 py-2 text-sm font-semibold text-white disabled:opacity-40';

export default function App() {
  const [token, setToken] = useState('');
  const [loginToken, setLoginToken] = useState('');
  const [user, setUser] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [ready, setReady] = useState(null);
  const [file, setFile] = useState(null);
  const [context, setContext] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [selectedJob, setSelectedJob] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const request = useRef(null);
  const activeToken = useRef('');
  const preview = useMemo(() => file ? URL.createObjectURL(file) : null, [file]);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const refresh = useCallback(async () => {
    if (!token) return;
    const [jobResponse, readiness] = await Promise.all([
      api('/jobs', token), fetch(`${API_BASE}/ready`),
    ]);
    const nextJobs = await jobResponse.json();
    const nextReady = await readiness.json();
    if (activeToken.current !== token) return;
    setJobs(nextJobs);
    setReady(nextReady);
  }, [token]);

  useEffect(() => {
    let disposed = false;
    const update = () => refresh().catch(e => { if (!disposed) setError(e.message); });
    update();
    const timer = setInterval(update, 2000);
    return () => { disposed = true; clearInterval(timer); };
  }, [refresh]);

  const completedId = jobs.find(job => job.id === selectedJob)?.assessment_id;
  useEffect(() => {
    if (completedId) Promise.resolve().then(() => setSelectedId(completedId));
  }, [completedId]);

  async function login(event) {
    event.preventDefault();
    setError('');
    try {
      const identity = await (await api('/me', loginToken)).json();
      activeToken.current = loginToken;
      setUser(identity); setToken(loginToken); setLoginToken('');
      setJobs([]); setReady(null); setNotice(''); setSelectedId(''); setSelectedJob(''); request.current = null;
    } catch (e) { setError(e.message); }
  }

  async function submit(event) {
    event.preventDefault();
    if (!file || !token) return;
    const mode = event.nativeEvent.submitter?.value === 'manual' ? 'manual' : 'helmet';
    setBusy(true); setError(''); setNotice('');
    if (!request.current || request.current.file !== file || request.current.context !== context || request.current.mode !== mode) {
      request.current = { file, context, mode, key: crypto.randomUUID() };
    }
    const data = new FormData(); data.append('file', file); data.append('context', context); data.append('mode', mode);
    try {
      const job = await (await api('/jobs', token, {
        method: 'POST', body: data, headers: { 'Idempotency-Key': request.current.key },
      })).json();
      setSelectedJob(job.id);
      setNotice(job.reused_request ? '이미 접수된 작업을 불러왔습니다. 중복 분석은 실행하지 않습니다.' : mode === 'manual' ? 'AI 분석 없이 평가서 초안을 만들었습니다. 위험요인을 직접 등록하세요.' : '분석 작업이 저장되었습니다. 창을 닫아도 처리 상태는 보존됩니다.');
      await refresh();
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function control(job, command) {
    setError('');
    try {
      await api(`/jobs/${job.id}/${command}`, token, { method: 'POST' });
      if (activeToken.current !== token) return;
      setSelectedJob(job.id); await refresh();
    } catch (e) { setError(e.message); }
  }

  const counts = jobs.reduce((total, job) => {
    total[job.state] = (total[job.state] || 0) + 1;
    total.repeated += job.duplicate_submissions;
    total.manual += job.profile === 'manual-v1' ? 1 : 0;
    total.seconds += job.attempts.reduce((sum, attempt) => sum + (attempt.elapsed_seconds || 0), 0);
    return total;
  }, { repeated: 0, seconds: 0, manual: 0 });

  return <div className="min-h-screen bg-slate-50 text-slate-900">
    <header className="border-b bg-white px-6 py-5">
      <div className="mx-auto flex max-w-7xl items-center gap-3">
        <Shield className="h-9 w-9 text-blue-700" />
        <div><h1 className="text-xl font-bold">Cosmos Guardian</h1><p className="text-sm text-slate-600">산업안전 점검 · 검토 · 개선조치</p></div>
        <span className="ml-auto rounded-full bg-amber-50 px-3 py-1 text-xs text-amber-900">개발 버전 · 현장 실증 전</span>
      </div>
    </header>
    <main className="mx-auto max-w-7xl space-y-6 px-6 py-8">
      <section className="rounded-2xl border bg-white p-5">
        <form onSubmit={login} className="flex flex-wrap items-center gap-3">
          <label className="text-sm font-semibold">접속 토큰 <input aria-label="접속 토큰" type="password" autoComplete="off" value={loginToken} onChange={e => setLoginToken(e.target.value)} className="ml-2 rounded-lg border px-3 py-2" /></label>
          <button className={button} disabled={!loginToken.trim() || busy}>{user ? '사용자 전환' : '로그인'}</button>
          {user && <span className="text-sm">{user.id} · {user.role === 'supervisor' ? '관리감독자' : '작성자'}</span>}
        </form>
        {ready && <p className="mt-3 text-sm text-slate-600">{!ready.inference_enabled ? '직접 작성·검토 모드 · AI 분석 꺼짐' : ready.worker_available ? '분석 서버 연결됨' : '분석 서버 연결 대기 · 직접 작성은 바로 사용할 수 있습니다.'}</p>}
        {error && <p role="alert" className="mt-3 text-sm text-red-700">{error}</p>}
        {notice && <p role="status" className="mt-3 text-sm text-blue-800">{notice}</p>}
      </section>

      <section className="grid gap-6 rounded-2xl border bg-white p-6 md:grid-cols-2">
        <form onSubmit={submit} className="space-y-4">
          <h2 className="text-lg font-bold">현장 사진으로 평가서 시작</h2>
          <p className="text-sm leading-6 text-slate-600">사진으로 평가서를 직접 작성하거나, 안전모 의무 구역에서 실험용 AI 판별을 요청할 수 있습니다. AI 미검출은 안전 판정이 아니며 모든 결과를 사람이 확인해야 합니다.</p>
          <label className="block text-sm font-semibold">현장 사진
            <input aria-label="현장 사진" type="file" accept="image/png,image/jpeg,image/webp" onChange={e => setFile(e.target.files[0] || null)} className="mt-2 block w-full rounded-lg border p-3" />
          </label>
          <p className="text-xs text-slate-500">PNG·JPEG·WebP, 최대 20 MiB · 2천만 화소. 영상 판별은 이번 제품 버전의 검증 범위에 포함되지 않습니다.</p>
          <label className="block text-sm font-semibold">작업·현장 메모
            <input aria-label="작업·현장 메모" placeholder="점검 위치와 작업 상황" maxLength={4000} value={context} onChange={e => setContext(e.target.value)} className="mt-2 w-full rounded-lg border px-3 py-2" />
          </label>
          <div className="flex flex-wrap gap-3">
            <button value="manual" className={button} disabled={!user || !file || busy}>AI 없이 직접 작성</button>
            <button value="helmet" className={button + ' inline-flex items-center gap-2'} disabled={!user || !file || busy || ready?.inference_enabled === false}><Upload size={16} />{busy ? '접수 중…' : '분석 요청'}</button>
          </div>
        </form>
        <div className="flex min-h-56 items-center justify-center rounded-xl bg-slate-100">
          {preview ? <img src={preview} alt="분석할 현장 이미지" className="max-h-80 w-full object-contain" /> : <p className="text-sm text-slate-500">기존 점검 사진을 업로드하세요.</p>}
        </div>
      </section>

      {user && <section className="space-y-4 rounded-2xl border bg-white p-6">
        <div className="flex items-center justify-between"><h2 className="text-lg font-bold">분석 작업</h2><button className="flex items-center gap-2 text-sm text-blue-700" onClick={() => refresh().catch(e => setError(e.message))}><RefreshCw size={15} />새로고침</button></div>
        <p className="text-sm text-slate-600">최근 100건 기준 · 완료 {counts.completed || 0} (직접 작성 {counts.manual}) · 대기 {counts.queued || 0} · 실패 {counts.failed || 0} · 동일 요청 재전송 {counts.repeated}건</p>
        <p className="text-xs text-slate-500">기록된 시도 처리시간 합계 {counts.seconds.toFixed(1)}초. 모델 로딩과 시간이 기록된 실패 시도를 포함합니다. 대기시간과 서버 중단으로 기록하지 못한 시간은 제외합니다.</p>
        <div className="overflow-x-auto"><table className="w-full text-left text-sm">
          <thead><tr className="border-b text-slate-500"><th className="py-3">자료</th><th>상태</th><th>시도</th><th>작업</th></tr></thead>
          <tbody>{jobs.map(job => <tr key={job.id} className="border-b align-top">
            <td className="max-w-sm py-3"><p className="break-all font-medium">{job.source.input_name}</p><p className="text-xs text-slate-500">{new Date(job.created_at).toLocaleString()} · {job.actor_id}</p>{job.error && <p className="mt-1 text-xs text-red-700">{errors[job.error] || '처리 오류가 발생했습니다.'}</p>}</td>
            <td className="py-3">{job.profile === 'manual-v1' ? '직접 작성' : states[job.state]}</td><td className="py-3">{job.profile === 'manual-v1' ? 'AI 미사용' : `${job.attempt}/3`}</td>
            <td className="space-x-3 py-3">
              {job.assessment_id && <button className="font-semibold text-blue-700" onClick={() => setSelectedId(job.assessment_id)}>평가서 열기</button>}
              {job.actor_id === user.id && ['queued', 'running'].includes(job.state) && <button className="text-red-700" onClick={() => control(job, 'cancel')}>취소</button>}
              {job.actor_id === user.id && ['failed', 'cancelled'].includes(job.state) && job.attempt < 3 && <button className="text-blue-700" onClick={() => control(job, 'retry')}>재시도</button>}
            </td>
          </tr>)}</tbody>
        </table>{!jobs.length && <p className="py-8 text-center text-sm text-slate-500">접수된 작업이 없습니다.</p>}</div>
      </section>}
    </main>
    {user && <ReviewPanel key={user.id} token={token} user={user} selectedId={selectedId} onSelect={setSelectedId} />}
    <footer className="px-6 py-8 text-center text-xs text-slate-500">검출 결과는 검토 자료입니다. 위험도·조치·종결은 담당자와 지정 관리감독자가 확인합니다.</footer>
  </div>;
}
