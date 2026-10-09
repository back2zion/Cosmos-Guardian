import { useEffect, useState } from 'react';
import { api } from './api';

const statuses = { draft: '초안', pending_review: '검토 대기', changes_requested: '보완 요청', approved: '승인 · 조치 진행', closed: '종결' };
const actionStatuses = { open: '미완료', completed: '재확인 대기', verified: '재확인 완료' };
const inputClass = 'w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm disabled:bg-zinc-100 disabled:text-zinc-600';
const buttonClass = 'rounded-lg bg-blue-700 px-4 py-2 text-sm font-bold text-white disabled:opacity-40 hover:bg-blue-800';

function SourceMedia({ document, token }) {
  const [url, setUrl] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  return <div className="space-y-3 rounded-xl bg-zinc-50 p-4">
    <button className={buttonClass} disabled={busy} onClick={async () => {
      setBusy(true); setError('');
      try { const response = await api(`/assessments/${document.id}/media`, token); setUrl(URL.createObjectURL(await response.blob())); }
      catch (e) { setError(e.message); }
      finally { setBusy(false); }
    }}>원본 현장 영상·이미지 확인</button>
    {error && <p role="alert" className="text-sm text-red-700">{error}</p>}
    {url && (document.source.media_type === 'video'
      ? <video src={url} controls className="max-h-96 w-full" />
      : <img src={url} alt="분석 당시 원본 현장 이미지" className="max-h-96 w-full object-contain" />)}
  </div>;
}

function Field({ label, value, onChange, type = 'text', disabled = false }) {
  return <label className="block space-y-1 text-sm text-zinc-600"><span>{label}</span>
    {type === 'textarea' ? <textarea className={inputClass} rows={3} value={value ?? ''} disabled={disabled} onChange={e => onChange(e.target.value)} />
      : <input className={inputClass} type={type} min={type === 'number' ? 1 : undefined} max={type === 'number' ? 5 : undefined} value={value ?? ''} disabled={disabled}
        onChange={e => onChange(type === 'number' ? (e.target.value ? Number(e.target.value) : null) : (type === 'date' ? e.target.value || null : e.target.value))} />}
  </label>;
}

function ActionEditor({ row, action, canReview, busy, run }) {
  const [completion, setCompletion] = useState({ note: '', evidence: '', residual_likelihood: null, residual_severity: null });
  const [note, setNote] = useState('');
  return <article className="space-y-3 rounded-xl border border-zinc-200 p-4">
    <h4 className="font-bold">{row.hazard} · {actionStatuses[action.status]}</h4>
    <p className="text-sm">{row.additional_controls} / 담당: {row.owner} / 기한: {row.due_date}</p>
    {action.reopen_note && <p className="text-amber-700">재조치 요청: {action.reopen_note}</p>}
    {action.status === 'open' ? <>
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="완료 내용" value={completion.note} onChange={v => setCompletion({ ...completion, note: v })} />
        <Field label="완료 증빙 (문서·사진 위치 또는 현장 확인 내용)" value={completion.evidence} onChange={v => setCompletion({ ...completion, evidence: v })} />
        <Field label="조치 후 가능성 (1–5)" type="number" value={completion.residual_likelihood} onChange={v => setCompletion({ ...completion, residual_likelihood: v })} />
        <Field label="조치 후 중대성 (1–5)" type="number" value={completion.residual_severity} onChange={v => setCompletion({ ...completion, residual_severity: v })} />
      </div>
      <button className={buttonClass} disabled={busy || !completion.note || !completion.evidence || !completion.residual_likelihood || !completion.residual_severity}
        onClick={() => run(`/actions/${row.id}/complete`, completion)}>조치 완료 등록</button>
    </> : <>
      <p className="text-sm whitespace-pre-wrap">완료: {action.note} · 증빙: {action.evidence}</p>
      <p className="text-sm">잔여 위험도: {action.residual_likelihood} × {action.residual_severity} = {action.residual_likelihood * action.residual_severity}</p>
      {action.status === 'completed' && canReview && <>
        <Field label="관리감독자 재확인 의견 (잔여 위험 허용 여부 포함)" disabled={busy} value={note} onChange={setNote} />
        <div className="flex flex-wrap gap-2">
          <button className={buttonClass} disabled={busy || !note.trim()} onClick={() => run(`/actions/${row.id}/verify`, { note })}>조치 재확인</button>
          <button className={buttonClass} disabled={busy || !note.trim()} onClick={() => run(`/actions/${row.id}/reopen`, { note })}>재조치 요청</button>
        </div>
      </>}
      {action.status === 'verified' && <p className="text-sm text-green-800">{action.verified_by} · {new Date(action.verified_at).toLocaleString()} · {action.verification_note}</p>}
    </>}
  </article>;
}

function AssessmentEditor({ document, supervisors, user, token, onUpdate }) {
  const [form, setForm] = useState(document.form);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [integrity, setIntegrity] = useState('');
  const [checks, setChecks] = useState({ source_checked: false, hazards_checked: false, controls_checked: false, criteria_checked: false });
  const editable = ['draft', 'changes_requested'].includes(document.status);
  const dirty = JSON.stringify(form) !== JSON.stringify(document.form);
  const canReview = user.role === 'supervisor' && user.id === document.form.supervisor_id;
  const change = (key, value) => setForm({ ...form, [key]: value });
  const changeRow = (index, key, value) => change('rows', form.rows.map((row, i) => i === index ? { ...row, [key]: value } : row));

  async function run(path, body = {}, method = 'POST') {
    setBusy(true); setError('');
    try {
      const response = await api(`/assessments/${document.id}${path}`, token, { method, body: JSON.stringify({ version: document.version, ...body }) });
      onUpdate(await response.json());
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function download(format) {
    setBusy(true); setError('');
    try {
      const response = await api(`/assessments/${document.id}/export/${format}`, token);
      const url = URL.createObjectURL(await response.blob());
      const anchor = window.document.createElement('a');
      anchor.href = url;
      anchor.download = `risk-assessment-${document.id}-v${document.version}.${format}`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  return <div className="space-y-6">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div><h3 className="text-xl font-bold">{statuses[document.status]}</h3><p className="text-xs text-zinc-500">{document.source.input_name} · 버전 {document.version} · {document.id}</p></div>
      {['approved', 'closed'].includes(document.status) && <div className="flex flex-wrap gap-2">
        <button className={buttonClass} disabled={busy} onClick={() => download('xlsx')}>Excel 평가서</button>
        <button className={buttonClass} disabled={busy} onClick={() => download('html')}>인쇄용 평가서 · PDF 저장</button>
      </div>}
    </div>
    {error && <p role="alert" className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}
    <SourceMedia document={document} token={token} />
    {document.original_result.assessment_status && <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm">
      <p className="font-bold">{document.original_result.analysis_performed === false ? 'AI 분석 없이 직접 작성' : `안전모 판별: ${{ violation: '미착용 의심', clear: '미착용 미검출 · 안전 판정 아님', unknown: '판단 불가 · 직접 확인 필요' }[document.original_result.assessment_status]}`}</p>
      <p className="mt-1">{document.original_result.scope}</p>
      <p className="mt-1">{document.original_result.evidence}</p>
    </div>}
    {document.status === 'changes_requested' && <p className="rounded-lg bg-amber-50 p-3">보완 의견: {document.history.findLast(e => e.type === 'request_changes')?.note}</p>}
    <p className="text-sm text-zinc-600">프로젝트 기본 양식 · 가능성(1–5) × 중대성(1–5). 현장 기준에 따라 직접 평가하세요. AI 안전점수는 위험도에 자동 반영되지 않습니다.</p>
    <fieldset disabled={!editable || busy} className="space-y-5">
      <div className="grid gap-4 md:grid-cols-3">
        {[['site', '사업장'], ['process', '공정'], ['task', '작업'], ['assessment_date', '평가일', 'date'], ['participants', '평가 참여자']].map(([key, label, type]) =>
          <Field key={key} label={label} type={type} value={form[key]} onChange={v => change(key, v)} />)}
        <label className="space-y-1 text-sm text-zinc-600">담당 관리감독자
          <select aria-label="담당 관리감독자" className={inputClass} value={form.supervisor_id} onChange={e => change('supervisor_id', e.target.value)}>
            <option value="">선택하세요</option>{supervisors.map(s => <option key={s.id}>{s.id}</option>)}
          </select>
        </label>
      </div>
      <Field label="현장 위험도 척도 및 허용 판단 기준" type="textarea" value={form.acceptance_criteria} onChange={v => change('acceptance_criteria', v)} />
      <Field label="현장 확인 의견 (누락·오탐 검토, 위험요인 미등록 시 그 근거)" type="textarea" value={form.review_note} onChange={v => change('review_note', v)} />
      {form.rows.map((row, index) => <article key={row.id} className="space-y-3 rounded-xl border border-zinc-200 p-4">
        <div className="flex justify-between"><h4 className="font-bold">위험요인 {index + 1}</h4>{editable && <button className="text-sm text-red-700" onClick={() => change('rows', form.rows.filter((_, i) => i !== index))}>항목 삭제</button>}</div>
        <div className="grid gap-3 md:grid-cols-2">
          {[['hazard', '위험요인'], ['consequence', '예상 피해'], ['existing_controls', '현재 안전조치'], ['additional_controls', '추가 개선조치'], ['likelihood', '가능성 (1–5)', 'number'], ['severity', '중대성 (1–5)', 'number'], ['owner', '조치 담당자'], ['due_date', '조치 기한', 'date']].map(([key, label, type]) =>
            <Field key={key} label={label} type={type} value={row[key]} onChange={v => changeRow(index, key, v)} />)}
        </div>
        <p className="text-sm font-bold">위험도: {row.likelihood && row.severity ? `${row.likelihood} × ${row.severity} = ${row.likelihood * row.severity}` : '평가 필요'}</p>
      </article>)}
      {editable && <button className={buttonClass} onClick={() => change('rows', [...form.rows, { id: crypto.randomUUID(), hazard: '', consequence: '', existing_controls: '', additional_controls: '', likelihood: null, severity: null, owner: '', due_date: null }])}>위험요인 추가</button>}
    </fieldset>
    {editable && <div className="flex items-center gap-3">
      <button className={buttonClass} disabled={busy} onClick={() => run('', { form }, 'PUT')}>초안 저장</button>
      <button className={buttonClass} disabled={busy || dirty} onClick={() => run('/submit')}>관리감독자 검토 요청</button>
      {dirty && <span className="text-sm text-amber-700">변경 내용을 먼저 저장하세요.</span>}
    </div>}
    {document.status === 'pending_review' && canReview && <div className="space-y-3 rounded-xl bg-blue-50 p-4">
      <div className="grid gap-2 md:grid-cols-2">{[
        ['source_checked', '원본 현장자료 확인'], ['hazards_checked', '위험요인 누락·오탐 확인'],
        ['controls_checked', '개선조치·담당자·기한 확인'], ['criteria_checked', '위험도·허용 기준 확인'],
      ].map(([key, label]) => <label key={key} className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={checks[key]} onChange={e => setChecks({ ...checks, [key]: e.target.checked })} />{label}
      </label>)}</div>
      <Field label="관리감독자 확인 의견 (현장 확인 및 평가 적정성)" disabled={busy} type="textarea" value={note} onChange={setNote} />
      <div className="flex gap-3"><button className={buttonClass} disabled={busy || !note.trim() || !Object.values(checks).every(Boolean)} onClick={() => run('/decisions/approve', { note, checks })}>확인 후 승인</button>
        <button className={buttonClass} disabled={busy || !note.trim()} onClick={() => run('/decisions/request_changes', { note })}>보완 요청</button></div>
    </div>}
    {document.status === 'pending_review' && !canReview && <p className="text-blue-700">{form.supervisor_id} 관리감독자의 확인을 기다리고 있습니다.</p>}
    {document.approval && <p className="rounded-lg bg-green-50 p-3 text-sm text-green-900">승인: {document.approval.actor} · {new Date(document.approval.timestamp).toLocaleString()} · {document.approval.note}</p>}
    {document.status === 'approved' && <section className="space-y-4"><h3 className="text-lg font-bold">개선조치 및 재확인</h3>
      {form.rows.map(row => <ActionEditor key={row.id} row={row} action={document.actions[row.id]} canReview={canReview} busy={busy} run={run} />)}
      {canReview && <div className="space-y-3">
        <Field label="종결 의견" disabled={busy} value={note} onChange={setNote} />
        <button className={buttonClass} disabled={busy || !note.trim() || Object.values(document.actions).some(a => a.status !== 'verified')} onClick={() => run('/decisions/close', { note })}>모든 조치 확인 후 종결</button>
      </div>}
    </section>}
    {document.closure && <p className="text-green-800">종결: {document.closure.actor} · {new Date(document.closure.timestamp).toLocaleString()} · {document.closure.note}</p>}
    <details className="rounded-xl border border-zinc-200 p-4"><summary className="cursor-pointer font-bold">분석 원본과 변경 이력 ({document.history.length}건)</summary>
      <pre className="my-3 max-h-64 overflow-auto whitespace-pre-wrap text-xs">{JSON.stringify(document.original_result, null, 2)}</pre>
      <ol className="space-y-2 text-sm">{document.history.map(event => <li key={event.seq}>{new Date(event.timestamp).toLocaleString()} · {event.actor.id} · {event.type} · {event.note}</li>)}</ol>
      <button className={`${buttonClass} mt-4`} onClick={async () => {
        try { const response = await api(`/assessments/${document.id}/integrity`, token); setIntegrity((await response.json()).valid ? '변경 이력 무결성 확인 완료' : '변경 이력 무결성 오류'); }
        catch (e) { setError(e.message); }
      }}>이력 무결성 확인</button><span className="ml-3 text-sm" role="status">{integrity}</span>
    </details>
  </div>;
}

export default function ReviewPanel({ token, user, selectedId, onSelect }) {
  const [records, setRecords] = useState([]);
  const [document, setDocument] = useState(null);
  const [supervisors, setSupervisors] = useState([]);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let active = true;
    Promise.all([api('/assessments', token).then(r => r.json()), api('/supervisors', token).then(r => r.json()),
      selectedId ? api(`/assessments/${selectedId}`, token).then(r => r.json()) : Promise.resolve(null)])
      .then(([list, people, detail]) => { if (active) { setRecords(list); setSupervisors(people); setDocument(detail); setError(''); } })
      .catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [token, selectedId, refresh]);
  return <section id="reviews" className="container relative mx-auto mb-16 space-y-6 rounded-3xl border border-zinc-200 bg-white p-6 md:p-8">
    <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-2xl font-bold">위험성평가 · 관리감독자 확인</h2>
      <button className={buttonClass} onClick={() => setRefresh(n => n + 1)}>목록 새로고침</button></div>
    <p className="text-sm text-zinc-500">분석 결과가 초안으로 저장됩니다. 현장 검토와 승인 후 평가서를 출력하고 개선조치를 추적하세요.</p>
    {error && <p role="alert" className="text-red-700">{error}</p>}
    <label className="block text-sm">평가서 선택<select aria-label="평가서 선택" className={`${inputClass} mt-1`} value={selectedId || ''} onChange={e => onSelect(e.target.value)}>
      <option value="">{records.length ? '평가서를 선택하세요' : '저장된 평가서 없음 — 미디어 분석을 먼저 실행하세요'}</option>
      {records.map(record => <option key={record.id} value={record.id}>{statuses[record.status]} · {record.form.site || record.source.input_name} · {new Date(record.created_at).toLocaleString()} · {record.id.slice(0, 8)}</option>)}
    </select></label>
    {document && document.id === selectedId && <AssessmentEditor key={`${document.id}-${document.version}`} document={document} supervisors={supervisors} token={token} user={user}
      onUpdate={next => { setDocument(next); setRecords(list => list.map(r => r.id === next.id ? next : r)); }} />}
  </section>;
}
