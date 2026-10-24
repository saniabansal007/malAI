import { api, post, stream } from './api.js';
import { md } from './md.js';
import { Spark } from './charts.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = n => Number(n || 0).toLocaleString();
const store = {
  get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
};
function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  kids.flat().forEach(c => c != null && e.append(c));
  return e;
}
const raf = (() => { let p = false, f; return fn => { f = fn; if (!p) { p = true; requestAnimationFrame(() => { p = false; f(); }); } }; })();

/* ---------- navigation ---------- */
function show(view) {
  $$('.view').forEach(v => v.hidden = v.id !== 'view-' + view);
  $$('#nav button').forEach(b => b.setAttribute('aria-current', b.dataset.view === view ? 'page' : 'false'));
  $('#chatsBox').style.display = view === 'chat' ? '' : 'none';
  store.set('view', view);
  if (view === 'library') loadDocs();
  if (view === 'memory') loadMemory();
}
$('#nav').addEventListener('click', e => e.target.dataset.view && show(e.target.dataset.view));

/* ---------- chat ---------- */
let chats = store.get('chats', []);
let cur = chats.find(c => c.id === store.get('cur')) || null;
let ctrl = null;
const msgBox = $('#messages');

const saveChats = () => { store.set('chats', chats.slice(0, 40)); store.set('cur', cur?.id); };

function renderList() {
  const ul = $('#chatList'); ul.innerHTML = '';
  chats.forEach(c => ul.append(h('li', { class: c === cur ? 'on' : '' },
    h('button', { class: 't', onclick: () => { cur = c; saveChats(); renderList(); renderChat(); } }, c.title),
    h('button', { class: 'x', 'aria-label': 'Delete chat', onclick: () => { chats = chats.filter(x => x !== c); if (cur === c) cur = null; saveChats(); renderList(); renderChat(); } }, '×'))));
}

function renderChat() {
  msgBox.innerHTML = '';
  if (!cur || !cur.messages.length) {
    msgBox.append(h('div', { class: 'empty' },
      h('h1', {}, 'Everything here runs on your machine.'),
      h('p', {}, 'Try one of these, then watch the cache tape under each answer:'),
      h('ul', {}, ['Summarise what you know about me, then ask a follow-up', 'Turn on Tools: what is 17.5% of 2,840, and what day is it?',
        'Upload a PDF in Library, switch on “Use my documents”, and ask about it'].map(t => h('li', {}, t)))));
    return;
  }
  cur.messages.forEach(m => { const n = h('div'); msgBox.append(n); renderMsg(m, n); });
  msgBox.scrollTop = msgBox.scrollHeight;
}

function toolCard(t) {
  return h('div', { class: 'tool' }, h('b', {}, t.name), ' ', t.input ? `“${t.input.slice(0, 120)}”` : '',
    t.output != null ? h('pre', {}, t.output) : h('div', { class: 'status-line' }, 'running…'));
}

function metaLine(m) {
  const cached = m.cached_tokens;
  const pct = cached != null && m.prompt_tokens ? Math.round(100 * cached / m.prompt_tokens) : 0;
  const txt = cached == null ? `cache stats unavailable (start SGLang with --enable-cache-report)`
    : `${fmt(cached)} of ${fmt(m.prompt_tokens)} prompt tokens reused from cache`;
  return h('div', { class: 'meta' },
    h('div', { class: 'tape', title: `${pct}% of the prompt was served from SGLang's prefix cache` }, h('i', { style: `width:${pct}%` })),
    h('span', {}, `${txt} · ${m.tps.toFixed(0)} tok/s · first token ${m.ttft.toFixed(2)}s${m.thinking ? ' · reasoning on' : ''}`));
}

function renderMsg(m, node) {
  const wasOpen = node.querySelector('details.think')?.open;
  node.className = 'msg ' + m.role; node.innerHTML = '';
  if (m.role === 'user') { node.append(h('div', { class: 'body' }, m.content)); return; }
  if (m.thinking) node.append(h('details', { class: 'think', ...((wasOpen ?? !m.content) ? { open: '' } : {}) }, h('summary', {}, m.content ? 'Reasoning' : 'Reasoning…'), h('div', { html: md(m.thinking) })));
  (m.tools || []).forEach(t => node.append(toolCard(t)));
  node.append(h('div', { class: 'body', html: md(m.content) || (m.status && !m.error ? '' : '') }));
  if (m.status && !m.content && !m.thinking) node.append(h('div', { class: 'status-line' }, m.status));
  if (m.error) node.append(h('div', { class: 'err' }, m.error));
  if (m.sources?.length && m.content) node.append(h('div', { class: 'chips' }, m.sources.map(s =>
    h('button', { class: 'chip', type: 'button', onclick: () => openSource(s) }, `[${s.n}] ${s.title}${s.page ? ' p.' + s.page : ''}`))));
  if (m.meta) node.append(metaLine(m.meta));
  node._sources = m.sources || [];
}

function openSource(s) {
  $('#srcTitle').textContent = `[${s.n}] ${s.title}${s.page ? ', page ' + s.page : ''}`;
  $('#srcText').textContent = s.text;
  $('#srcDialog').showModal();
}
document.addEventListener('click', e => {
  if (e.target.classList.contains('cite')) {
    const scope = e.target.closest('.msg')?._sources || researchSources;
    const s = scope.find(x => x.n === +e.target.dataset.n);
    if (s) openSource(s);
  } else if (e.target.classList.contains('copy')) {
    navigator.clipboard?.writeText(e.target.nextSibling.textContent);
    e.target.textContent = 'Copied'; setTimeout(() => e.target.textContent = 'Copy', 1200);
  }
});

const opts = () => ({ use_docs: $('#optDocs').checked, use_tools: $('#optTools').checked, use_memory: $('#optMemory').checked, thinking: $('#optThink').value });
['optDocs', 'optTools', 'optMemory', 'optThink'].forEach(id => {
  const el = $('#' + id), v = store.get('opt_' + id);
  if (v != null) el.type === 'checkbox' ? el.checked = v : el.value = v;
  el.addEventListener('change', () => store.set('opt_' + id, el.type === 'checkbox' ? el.checked : el.value));
});

function setBusy(b) {
  const btn = $('#send');
  btn.textContent = b ? 'Stop' : 'Send'; btn.classList.toggle('stop', b); btn.type = b ? 'button' : 'submit';
}
$('#send').addEventListener('click', () => ctrl?.abort());

async function sendChat(text) {
  if (!cur) { cur = { id: crypto.randomUUID(), title: text.slice(0, 40), messages: [] }; chats.unshift(cur); }
  cur.messages.push({ role: 'user', content: text });
  const m = { role: 'assistant', content: '', thinking: '', tools: [], sources: [], status: 'Starting…' };
  cur.messages.push(m);
  renderList(); renderChat();
  const node = msgBox.lastElementChild;
  ctrl = new AbortController(); setBusy(true);
  const draw = () => raf(() => { renderMsg(m, node); msgBox.scrollTop = msgBox.scrollHeight; });
  try {
    await stream('/chat', { messages: cur.messages.slice(0, -1).filter(x => !x.error).map(({ role, content }) => ({ role, content })), ...opts() }, ev => {
      if (ev.type === 'token') m.content += ev.text;
      else if (ev.type === 'thinking') m.thinking += ev.text;
      else if (ev.type === 'status') m.status = ev.text;
      else if (ev.type === 'sources') m.sources = ev.sources;
      else if (ev.type === 'tool_call') m.tools.push({ name: ev.name, input: ev.input });
      else if (ev.type === 'tool_result') m.tools[m.tools.length - 1].output = ev.output;
      else if (ev.type === 'meta') m.meta = ev;
      else if (ev.type === 'error') m.error = ev.message;
      draw();
    }, ctrl.signal);
  } catch (e) { if (e.name !== 'AbortError') m.error = e.message; }
  if (!m.content && !m.error) m.error = m.thinking ? '' : 'Stopped.';
  m.status = ''; ctrl = null; setBusy(false);
  renderMsg(m, node); saveChats(); pollMetrics();
}

$('#composer').addEventListener('submit', e => {
  e.preventDefault();
  const t = $('#input').value.trim(); if (!t || ctrl) return;
  $('#input').value = ''; $('#input').style.height = 'auto'; sendChat(t);
});
$('#input').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); } });
$('#input').addEventListener('input', e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 180) + 'px'; });
$('#newChat').addEventListener('click', () => { cur = null; saveChats(); renderList(); renderChat(); $('#input').focus(); });
document.addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); show('chat'); $('#newChat').click(); } });

/* ---------- deep research ---------- */
let researchSources = [];
let rCtrl = null;
$('#rForm').addEventListener('submit', async e => {
  e.preventDefault();
  if (rCtrl) { rCtrl.abort(); return; }
  const question = $('#rQ').value.trim(); if (!question) return;
  $('#graph').hidden = false; $('#researchers').innerHTML = ''; $('#rStats').textContent = ''; $('#report').innerHTML = '';
  $$('#graph .node').forEach(n => n.className = 'node');
  const node = id => $('#n-' + id) || $('#' + id);
  const setN = (id, cls) => { const n = node(id); if (n) n.className = 'node ' + cls; };
  let report = '';
  rCtrl = new AbortController(); $('#rGo').textContent = 'Stop'; $('#rGo').classList.add('stop');
  try {
    await stream('/research', { question, use_web: $('#rWeb').checked, n_sub: +$('#rN').value }, ev => {
      if (ev.type === 'sources') researchSources = ev.sources;
      else if (ev.type === 'stage') {
        if (ev.id === 'planner') setN('planner', ev.status);
        if (ev.subs) $('#researchers').replaceChildren(...ev.subs.map((s, i) => h('div', { class: 'node', id: 'n-r' + i }, h('b', {}, 'Researcher ' + (i + 1)), h('small', {}, s))));
        if (/^r\d+$/.test(ev.id)) {
          const n = $('#n-' + ev.id); if (!n) return;
          setN(ev.id, ev.status === 'critic' ? 'critic' : ev.status);
          if (ev.status === 'done') n.append(h('small', {}, `critic ${ev.score}/10${ev.issue && ev.issue !== 'none' ? ' · ' + ev.issue : ''}`));
        }
        if (ev.id === 'critic') setN('critic', ev.status === 'idle' ? '' : ev.status);
        if (ev.id === 'writer') setN('writer', ev.status);
      } else if (ev.type === 'token') { report += ev.text; raf(() => $('#report').innerHTML = `<div class="body">${md(report)}</div>`); }
      else if (ev.type === 'done') {
        researchSources = ev.sources; const s = ev.stats;
        $('#rStats').textContent = `${s.elapsed.toFixed(1)}s · ${s.calls} model calls · ${fmt(s.cached_tokens)} of ${fmt(s.prompt_tokens)} prompt tokens served from the prefix cache (${s.saved_pct.toFixed(0)}%)`;
        raf(() => $('#report').innerHTML = `<div class="body">${md(report)}</div>`);
        pollMetrics();
      } else if (ev.type === 'error') $('#report').innerHTML = `<div class="err">${ev.message}</div>`;
    }, rCtrl.signal);
  } catch (err) { if (err.name !== 'AbortError') $('#report').innerHTML = `<div class="err">${err.message}</div>`; }
  rCtrl = null; $('#rGo').textContent = 'Run research'; $('#rGo').classList.remove('stop');
});

/* ---------- extract ---------- */
$('#xSchema').value = JSON.stringify({
  type: 'object', required: ['invoice_number', 'vendor', 'customer', 'date', 'items', 'total'],
  properties: {
    invoice_number: { type: 'string' }, vendor: { type: 'string' }, customer: { type: 'string' }, date: { type: 'string' },
    items: { type: 'array', items: { type: 'object', required: ['name', 'quantity', 'unit_price'], properties: { name: { type: 'string' }, quantity: { type: 'integer' }, unit_price: { type: 'number' } } } },
    total: { type: 'number' },
  },
}, null, 2);
$('#xGo').addEventListener('click', async () => {
  const out = $('#xOut'); let schema;
  try { schema = JSON.parse($('#xSchema').value); } catch (e) { out.textContent = 'Schema is not valid JSON: ' + e.message; return; }
  out.textContent = 'Extracting…'; $('#xGo').disabled = true;
  const t = performance.now();
  try {
    const r = await post('/extract', { text: $('#xText').value, instruction: $('#xInst').value, schema });
    out.textContent = JSON.stringify(r.result, null, 2) + `\n\n// valid against your schema by construction · ${((performance.now() - t) / 1000).toFixed(2)}s`;
  } catch (e) { out.textContent = e.message; }
  $('#xGo').disabled = false; pollMetrics();
});

/* ---------- library ---------- */
async function loadDocs() {
  const d = await api('/docs');
  $('#libMsg').textContent = d.docs.length ? `${d.chunks} passages indexed · ${d.semantic ? 'semantic search' : 'keyword search (install sentence-transformers for semantic search)'}` : '';
  $('#docList').replaceChildren(...d.docs.map(x => h('li', {}, h('span', {}, x.name, ' ', h('small', {}, `${x.chunks} passages`)),
    h('button', { onclick: async () => { await api('/docs/' + x.id, { method: 'DELETE' }); loadDocs(); } }, 'Remove'))));
}
async function upload(files) {
  if (!files.length) return;
  $('#libMsg').textContent = 'Indexing… (the first run downloads the embedding model)';
  const fd = new FormData();[...files].forEach(f => fd.append('files', f));
  const r = await api('/docs', { method: 'POST', body: fd });
  await loadDocs();
  if (r.errors.length) $('#libMsg').textContent = r.errors.join(' · ');
}
$('#file').addEventListener('change', e => upload(e.target.files));
['dragover', 'dragenter'].forEach(t => $('#drop').addEventListener(t, e => { e.preventDefault(); $('#drop').classList.add('over'); }));
['dragleave', 'drop'].forEach(t => $('#drop').addEventListener(t, e => { e.preventDefault(); $('#drop').classList.remove('over'); }));
$('#drop').addEventListener('drop', e => upload(e.dataTransfer.files));

/* ---------- memory ---------- */
async function loadMemory() {
  const d = await api('/memory');
  $('#memList').replaceChildren(...(d.items.length ? d.items : []).map(x => h('li', {}, h('span', {}, x.text, ' ', h('small', {}, x.source === 'auto' ? 'learned from chat' : 'added by you')),
    h('button', { onclick: async () => { await api('/memory/' + x.id, { method: 'DELETE' }); loadMemory(); } }, 'Forget'))));
  if (!d.items.length) $('#memList').append(h('li', {}, h('span', { class: 'note' }, 'Nothing remembered yet. Turn on Memory in Chat and tell me about yourself.')));
}
$('#mForm').addEventListener('submit', async e => { e.preventDefault(); const t = $('#mText').value.trim(); if (t) { await post('/memory', { text: t }); $('#mText').value = ''; loadMemory(); } });
$('#mClear').addEventListener('click', async () => { if (confirm('Forget everything?')) { await api('/memory', { method: 'DELETE' }); loadMemory(); } });

/* ---------- benchmark ---------- */
$('#bForm').addEventListener('submit', async e => {
  e.preventDefault();
  const out = $('#bOut'); $('#bGo').disabled = true; out.textContent = 'Running both passes. This can take a minute on a laptop GPU…';
  try {
    const r = await post('/benchmark', { n: +$('#bN').value, prefix_tokens: +$('#bP').value });
    const a = r.no_sharing, b = r.shared_prefix, max = Math.max(a.total_s, b.total_s);
    const bar = (label, x, win) => h('div', { class: 'bar' + (win ? ' win' : '') }, h('span', {}, label), h('div', { style: `width:${100 * x.total_s / max}%` }), h('b', {}, x.total_s + 's'));
    out.replaceChildren(
      h('div', { class: 'big' }, r.speedup ? `${r.speedup}× faster with a shared prefix` : '–'),
      h('div', { class: 'bars' }, bar('A · unique prefixes', a, false), bar('B · shared prefix', b, true)),
      h('p', { class: 'note' }, `${r.requests} requests × ~${fmt(r.prefix_tokens)}-token prefix. Average latency ${a.avg_latency_s}s → ${b.avg_latency_s}s. ` +
        (b.cached_tokens ? `Run B reused ${fmt(b.cached_tokens)} of ${fmt(b.prompt_tokens)} prompt tokens from the cache.` : 'Cache usage not reported: start SGLang with --enable-cache-report.')));
  } catch (err) { out.textContent = err.message; }
  $('#bGo').disabled = false; pollMetrics();
});

/* ---------- live panel ---------- */
const sp = { cache: new Spark($('#cCache'), '#F2B544', 1), tps: new Spark($('#cTps'), '#8FC7FF'), lat: new Spark($('#cLat'), '#8FC7FF') };
let lastReq = -1;
async function pollMetrics() {
  try {
    const m = await api('/metrics');
    const s = m.sglang, ses = m.session;
    $('#status').className = 'status ' + (m.up ? 'up' : 'down');
    $('#status span').textContent = m.up ? 'SGLang connected' : 'SGLang not reachable. Run scripts/start_sglang.sh';
    const cache = s.cache_hit_rate ?? (ses.prompt_tokens ? ses.cached_tokens / ses.prompt_tokens : null);
    $('#mCache').textContent = cache != null ? Math.round(cache * 100) + '%' : '–';
    if (cache != null) sp.cache.push(cache);
    if (s.gen_throughput != null) { $('#mTps').textContent = s.gen_throughput.toFixed(0) + ' tok/s'; sp.tps.push(s.gen_throughput); } else $('#mTps').textContent = '–';
    const last = ses.recent[ses.recent.length - 1];
    if (last && ses.requests !== lastReq) { lastReq = ses.requests; $('#mLat').textContent = last.latency.toFixed(2) + 's'; sp.lat.push(last.latency); }
    $('#mQ').textContent = s.num_running_reqs != null ? `${s.num_running_reqs} / ${s.num_queue_reqs ?? 0}` : '–';
    $('#sReq').textContent = fmt(ses.requests);
    $('#sSaved').textContent = ses.cache_reported ? Math.round(ses.saved_pct) + '%' : '–';
    $('#sTape').style.width = (ses.cache_reported ? ses.saved_pct : 0) + '%';
    $('#liveNote').textContent = !m.up ? '' : !Object.keys(s).length ? 'Start SGLang with --enable-metrics for server-side gauges.' : !ses.cache_reported && ses.requests ? 'Start SGLang with --enable-cache-report to see reused tokens.' : '';
  } catch { $('#status').className = 'status down'; $('#status span').textContent = 'App backend not reachable'; }
}
setInterval(pollMetrics, 2500);

/* ---------- boot ---------- */
renderList(); renderChat(); show(store.get('view', 'chat')); pollMetrics();
