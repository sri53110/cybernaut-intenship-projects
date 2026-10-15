(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const pct = (x, d = 1) => (x * 100).toFixed(d).replace(/\.0+$/, '') + '%';
  const API = '';  // same origin; set to e.g. 'http://localhost:8000' if served separately

  const SAMPLES = [
    ['WINNER!! You have been selected to receive a $5000 cash prize. Call 08712345678 now or reply WIN to 81234 to claim your reward. T&Cs apply', 'Prize winner'],
    ['Your account has been suspended due to unusual activity. Verify your details immediately at http://secure-bank-login.top/verify or lose access', 'Action required'],
    ['Your parcel could not be delivered. Pay the $3 redelivery fee at http://parcel-pay.click/pay today', 'Parcel on hold'],
    ['Hey, are we still meeting for lunch at 1 pm tomorrow? I can book the table near the library.', ''],
    ['Reminder: the team meeting moved to 3:15 on Thursday, room 4B. Please bring the draft report.', 'Meeting moved'],
  ];

  let model = null, busy = false;

  async function api(path, body) {
    const res = await fetch(API + path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON error */ }
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status}). Is the server running?`);
    return data;
  }

  /* ---------- models selector ---------- */
  async function loadModels() {
    const { default: def, models } = await api('/api/models');
    model = def;
    const box = $('#models');
    models.forEach(m => {
      const b = el('button', null, m.name.replace('Support Vector Machine', 'SVM'));
      b.type = 'button'; b.setAttribute('role', 'radio'); b.dataset.id = m.id;
      b.setAttribute('aria-checked', m.id === def);
      b.addEventListener('click', () => {
        model = m.id;
        box.querySelectorAll('button').forEach(x => x.setAttribute('aria-checked', x.dataset.id === m.id));
      });
      box.appendChild(b);
    });
  }

  /* ---------- tabs ---------- */
  const tabs = [$('#tab-one'), $('#tab-many')];
  function showTab(i) {
    tabs.forEach((t, j) => { t.setAttribute('aria-selected', i === j); t.tabIndex = i === j ? 0 : -1; });
    $('#panel-one').hidden = i !== 0; $('#panel-many').hidden = i !== 1;
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => showTab(i));
    t.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { const n = 1 - i; showTab(n); tabs[n].focus(); }
    });
  });

  /* ---------- single message ---------- */
  const body = $('#body'), subject = $('#subject'), marked = $('#marked'), stamp = $('#stamp'), err = $('#err');

  function resetView() { marked.hidden = true; stamp.hidden = true; }
  body.addEventListener('input', resetView);

  function highlight(text, spamTerms, hamTerms) {
    const words = set => new Set(set.flatMap(t => t.term.split(' ')));
    const S = words(spamTerms), H = words(hamTerms);
    const has = (set, tok) => set.has(tok);
    marked.replaceChildren();
    const re = /(https?:\/\/\S+|www\.\S+|[$£€₹]\s?\d[\d,.]*|[A-Za-z']+|\d+)/g;
    let last = 0, m;
    const add = (piece, cls) => { if (!piece) return; if (cls) marked.appendChild(el('mark', cls, piece)); else marked.appendChild(document.createTextNode(piece)); };
    while ((m = re.exec(text))) {
      add(text.slice(last, m.index));
      const p = m[0], low = p.toLowerCase();
      let tok = low;
      if (/^(https?:|www\.)/.test(low)) tok = 'urltoken';
      else if (/^[$£€₹]/.test(low)) tok = 'moneytoken';
      else if (/^\d+$/.test(low)) tok = 'numtoken';
      add(p, has(S, tok) ? 's' : has(H, tok) ? 'h' : '');
      last = m.index + p.length;
    }
    add(text.slice(last));
    marked.hidden = false;
  }
  marked.addEventListener('click', () => { marked.hidden = true; body.focus(); });

  function gauge(p) {
    const svg = $('#gauge'); svg.replaceChildren();
    const ns = 'http://www.w3.org/2000/svg', cx = 110, cy = 110, r = 90;
    const pt = a => [cx + r * Math.cos(Math.PI - a), cy - r * Math.sin(Math.PI - a)];
    const arc = (a0, a1, color, w) => {
      const [x0, y0] = pt(a0), [x1, y1] = pt(a1);
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', `M${x0} ${y0} A${r} ${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1} ${y1}`);
      path.setAttribute('fill', 'none'); path.setAttribute('stroke', color); path.setAttribute('stroke-width', w); path.setAttribute('stroke-linecap', 'round');
      svg.appendChild(path);
    };
    arc(0, Math.PI, '#1d2b37', 16);
    if (p > 0.005) arc(0, Math.PI * Math.max(p, 0.01), p >= .5 ? '#e5484d' : p >= .2 ? '#e0a526' : '#3fb68b', 16);
  }

  function bars(ul, items, max, empty) {
    ul.replaceChildren();
    if (!items.length) { ul.appendChild(el('li', 'none', empty)); ul.lastChild.style.display = 'block'; return; }
    items.forEach(it => {
      const li = el('li'); const t = el('span', 't', it.term); t.title = it.term;
      const b = el('span', 'b'); const i = el('i'); i.style.width = Math.max(4, Math.abs(it.score || it.weight) / max * 100) + '%'; b.appendChild(i);
      li.append(t, b); ul.appendChild(li);
    });
  }

  function showResult(r, text) {
    const p = r.spam_probability;
    $('#result').hidden = false;
    $('#g-pct').textContent = pct(p, 0);
    gauge(p);
    const unsure = r.confidence === 'uncertain';
    stamp.className = 'stamp ' + (unsure ? 'unsure' : r.label);
    stamp.replaceChildren(document.createTextNode(unsure ? 'CHECK' : r.label === 'spam' ? 'SPAM' : 'CLEARED'), el('small', null, r.model_name));
    stamp.hidden = false;
    highlight(text, r.spam_signals, r.ham_signals);

    $('#why-line').textContent = r.label === 'spam'
      ? `This looks like spam (${pct(p, 0)} chance), judged by ${r.model_name}. The highlighted words are what tipped it.`
      : unsure
        ? `Not clearly spam, but not clearly safe either (${pct(p, 0)} chance of spam). Read it carefully before you click anything.`
        : `This looks legitimate (${pct(p, 0)} chance of spam), judged by ${r.model_name}.`;
    const mx = Math.max(.001, ...[...r.spam_signals, ...r.ham_signals].map(s => Math.abs(s.score)));
    bars($('#sig-spam'), r.spam_signals, mx, 'No spam-like words found.');
    bars($('#sig-ham'), r.ham_signals, mx, 'No clearly legitimate words found.');

    const v = $('#votes'); v.replaceChildren();
    Object.entries(r.votes).forEach(([k, val]) => {
      const name = { naive_bayes: 'Naive Bayes', logistic_regression: 'Logistic Regression', svm: 'SVM' }[k] || k;
      const li = el('li'); const head = el('div'); head.append(el('span', null, name), el('b', null, pct(val, 0)));
      const b = el('div', 'b'); const i = el('i', val >= .5 ? 'on' : ''); i.style.width = pct(val, 0); b.appendChild(i);
      li.append(head, b); v.appendChild(li);
    });
    const spamVotes = Object.values(r.votes).filter(x => x >= .5).length, n = Object.keys(r.votes).length;
    $('#vote-note').textContent = spamVotes === 0 || spamVotes === n ? 'All models agree.' : `${spamVotes} of ${n} models call it spam.`;
    $('#result').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'nearest' });
  }

  async function check() {
    const text = body.value.trim();
    err.hidden = true;
    if (!text) { err.textContent = 'Paste a message first, or pick an example below.'; err.hidden = false; body.focus(); return; }
    if (busy) return; busy = true;
    const btn = $('#go'); btn.disabled = true; btn.textContent = 'Checking…';
    try {
      const r = await api('/api/predict', { subject: subject.value, text, model });
      showResult(r, text);
      $('#f-ms').textContent = r.latency_ms + ' ms';
    } catch (e) { err.textContent = e.message; err.hidden = false; }
    finally { busy = false; btn.disabled = false; btn.textContent = 'Check message'; }
  }
  $('#go').addEventListener('click', check);
  body.addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') check(); });

  document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', () => {
    subject.value = ''; body.value = SAMPLES[+c.dataset.s][0]; resetView(); err.hidden = true; check();
  }));

  /* ---------- batch ---------- */
  const batch = $('#batch');
  const lines = () => batch.value.split('\n').map(s => s.trim()).filter(Boolean);
  batch.addEventListener('input', () => { const n = lines().length; $('#batch-count').textContent = `${n} message${n === 1 ? '' : 's'}`; });
  $('#go-batch').addEventListener('click', async () => {
    const e = $('#err-batch'); e.hidden = true;
    const texts = lines();
    if (!texts.length) { e.textContent = 'Add at least one message, one per line.'; e.hidden = false; return; }
    if (texts.length > 100) { e.textContent = 'That is more than 100 messages. Remove some and try again.'; e.hidden = false; return; }
    const btn = $('#go-batch'); btn.disabled = true; btn.textContent = 'Checking…';
    try {
      const r = await api('/api/predict/batch', { texts, model });
      $('#batch-sum').textContent = `${r.summary.spam} of ${r.summary.total} messages look like spam.`;
      const ul = $('#batch-list'); ul.replaceChildren();
      r.results.forEach(x => {
        const li = el('li'); const pill = el('span', 'pill ' + x.label, x.label === 'spam' ? 'Spam' : 'Legitimate');
        li.append(pill, el('b', null, pct(x.spam_probability, 0)), el('span', 'pv', x.preview)); ul.appendChild(li);
      });
      $('#batch-out').hidden = false;
    } catch (x) { e.textContent = x.message; e.hidden = false; }
    finally { btn.disabled = false; btn.textContent = 'Check all'; }
  });

  /* ---------- performance + keywords ---------- */
  async function loadMetrics() {
    const m = await api('/api/metrics');
    const best = m.models[m.default_model];
    $('#f-model').textContent = best.name.replace('Support Vector Machine', 'SVM');
    $('#f-prec').textContent = pct(best.precision);
    $('#sample-note').hidden = !m.is_sample_data;
    const tb = $('#perf-table tbody');
    Object.entries(m.models).forEach(([k, r]) => {
      const tr = el('tr', k === m.default_model ? 'best' : '');
      [r.name, pct(r.accuracy), pct(r.precision), pct(r.recall), pct(r.f1), pct(r.false_positive_rate)].forEach(v => tr.appendChild(el('td', null, v)));
      tb.appendChild(tr);
    });
    const c = best.confusion, g = $('#cm');
    $('#cm-cap').textContent = `${best.name} on ${m.test_rows} unseen messages`;
    [[c.tn, 'legitimate, correctly kept', 'ok'], [c.fp, 'legitimate, wrongly blocked', 'bad'], [c.fn, 'spam that slipped through', 'bad'], [c.tp, 'spam, correctly caught', 'ok']]
      .forEach(([n, label, cls]) => { const d = el('div', cls); d.append(el('b', null, n), el('span', null, label)); g.appendChild(d); });
  }
  async function loadKeywords() {
    const k = await api('/api/keywords');
    const mx = Math.max(...k.spam.slice(0, 12).map(x => x.weight), ...k.ham.slice(0, 12).map(x => -x.weight));
    bars($('#kw-spam'), k.spam.slice(0, 12), mx, ''); bars($('#kw-ham'), k.ham.slice(0, 12).map(x => ({ term: x.term, weight: -x.weight })), mx, '');
  }

  $('#origin').textContent = location.origin.startsWith('http') ? location.origin : 'http://localhost:8000';
  Promise.all([loadModels(), loadMetrics(), loadKeywords()]).then(async () => {
    // measure real round-trip latency once so the hero fact is a measurement, not a claim
    try { const t = performance.now(); await api('/api/predict', { text: 'hello', model }); $('#f-ms').textContent = Math.round(performance.now() - t) + ' ms'; } catch (_) {}
  }).catch(e => { err.textContent = 'Cannot reach the SpamShield server. ' + e.message; err.hidden = false; });
})();
