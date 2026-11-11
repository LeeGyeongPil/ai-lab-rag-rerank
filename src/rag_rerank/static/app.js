// RAG 리랭킹 전/후 비교 화면. 빌드 도구 없이 이 파일 하나가 화면 동작 전부다.
// 흐름: 질문 → POST /api/compare → 두 열 카드 + 연결선 → (버튼) POST /api/answer 스트림 → 답변 A/B
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = {
  status: null,       // /api/status 응답
  lastReq: null,      // 마지막 비교 요청(답변 만들 때 그대로 다시 씀)
  lastResult: null,   // 마지막 비교 결과
  retriever: "hybrid",
  answerCtrl: null,   // 답변 스트림 중지용 AbortController
  pollTimer: null,    // 리랭커 내려받기 진행률 폴링 타이머(준비되면 멈춘다)
  lastEval: null,     // 마지막 성적표(언어를 바꾸면 다시 그린다)
};
// 화면 문구는 전부 i18n.js 의 tr(문구 id) 로 가져온다(영어·한국어 표 한 곳).

// ---------- 공통 ----------
// 서버 오류 문구·모드 안내도 같은 언어로 받도록 모든 요청에 ?lang= 을 붙인다(JSON 모양은 그대로)
const withLang = (path) => `${path}${path.includes("?") ? "&" : "?"}lang=${LANG}`;
async function api(path, body) {
  const res = await fetch(withLang(path), body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`);
  return data;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;  // 항상 textContent — 문서 내용을 HTML 로 해석하지 않는다
  return e;
}
const fmtMB = (b) => (b / 1e6).toFixed(0) + "MB";
const moveText = (m) => (m > 0 ? `▲${m}` : m < 0 ? `▼${-m}` : "–");
const moveCls = (m) => (m > 0 ? "up" : m < 0 ? "down" : "same");

// ---------- 상태(모드·리랭커 내려받기) ----------
async function loadStatus() {
  const s = await api("/api/status");
  state.status = s;
  const badge = $("#mode-badge");
  badge.textContent = s.mode.toUpperCase();
  badge.className = `badge ${s.mode}`;
  $("#models").textContent = tr("models.line", { embed: s.models.embed, reranker: s.reranker.model, chat: s.models.chat });
  const note = $("#mode-note");
  if (s.mode === "mock") {
    note.hidden = false;
    note.textContent = "";
    note.append(el("b", "", `${tr("mode.mock_title")} — ${tr("mode.mock_body")} `), s.mode_reason + " ");
    for (const h of s.hints) { note.append(el("code", "", h), " "); }
  } else {
    note.hidden = true;
  }
  renderReranker(s.reranker);
  return s;
}

function renderReranker(r) {
  const box = $("#rr-status");
  const busy = r.state === "downloading" || r.state === "loading" || r.state === "idle";
  box.hidden = !(busy || r.state === "error");
  $("#rr-label").textContent =
    r.state === "downloading" ? tr("rr.downloading") :
    r.state === "loading" ? tr("rr.loading") :
    r.state === "error" ? tr("rr.error", { error: r.error || "" }) : tr("rr.idle");
  const pct = r.percent ?? (r.state === "loading" ? 100 : 0);
  $("#rr-bar").style.width = `${pct}%`;
  $("#rr-pct").textContent = r.total_bytes ? `${pct}% (${fmtMB(r.done_bytes)} / ${fmtMB(r.total_bytes)})` : "";
  $("#btn-rr-retry").hidden = !r.can_retry;
  // 준비 중이면 1초 뒤 다시 확인. 준비가 끝나면 타이머를 더 만들지 않는다.
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
  if (busy) {
    state.pollTimer = setTimeout(async () => {
      const s = await loadStatus().catch(() => null);
      if (s && s.reranker.state === "ready" && state.lastResult && state.lastResult.after === null) runCompare();
    }, 1000);
  }
}

function renderChips() {
  const box = $("#chips");
  box.textContent = "";
  for (const q of state.status.questions) {
    const b = el("button", "chip" + (q.note ? " limit" : ""), q.q);
    b.type = "button";
    if (q.note) b.title = tr("note.limit", { note: q.note });
    b.addEventListener("click", () => { $("#question").value = q.q; runCompare(); });
    box.append(b);
  }
}

// ---------- 비교 ----------
function currentRequest() {
  return {
    question: $("#question").value.trim(),
    collection: $("#collection").value,
    retriever: state.retriever,
    top_k: Number($("#top-k").value),
    chunking: $("#chunking").value,
    title_prefix: $("#title-prefix").checked,
  };
}

async function runCompare() {
  const req = currentRequest();
  if (!req.question) return;
  stopAnswers();
  document.body.dataset.state = "loading";
  $("#btn-compare").disabled = true;
  try {
    const result = await api("/api/compare", req);
    state.lastReq = req;
    state.lastResult = result;
    renderCompare(result);
    resetAnswers();
    $("#btn-answer").disabled = false;
    document.body.dataset.state = "compared";  // 스크린샷 스크립트가 기다리는 신호
  } catch (e) {
    document.body.dataset.state = "error";
    showNote(tr("error.prefix", { message: e.message }));
  } finally {
    $("#btn-compare").disabled = false;
  }
}

function showNote(text) {
  const n = $("#note");
  n.hidden = !text;
  n.textContent = text || "";
}

function goldAnchor(question) {
  if (!state.status || $("#collection").value !== "samples") return null;
  const q = state.status.questions.find((x) => x.q === question);
  return q ? q.gold : null;
}

// 카드 텍스트를 하이라이트 구간(노랑)과 근거 문장(밑줄)으로 나눠 그린다
function renderText(text, highlights, evidence) {
  const p = el("p", "txt");
  const cuts = new Set([0, text.length]);
  for (const [s, e] of highlights) { cuts.add(s); cuts.add(e); }
  if (evidence) { cuts.add(evidence[0]); cuts.add(evidence[1]); }
  const points = [...cuts].sort((a, b) => a - b);
  for (let i = 0; i < points.length - 1; i++) {
    const [s, e] = [points[i], points[i + 1]];
    const inHl = highlights.some(([a, b]) => a <= s && e <= b);
    const inEv = evidence && evidence[0] <= s && e <= evidence[1];
    const span = el(inHl ? "mark" : "span", [inHl ? "hl" : "", inEv ? "ev" : ""].join(" ").trim(), text.slice(s, e));
    p.append(span);
  }
  return p;
}

function renderCard(c, side, maxScore, gold) {
  const li = el("li", "card");
  li.dataset.chunkId = c.chunk_id;
  li.dataset.rank = c.rank;
  if (gold && c.text.includes(gold)) li.classList.add("gold");
  const top = el("div", "card-top");
  top.append(el("span", "rank", c.rank), el("span", "src", c.source));
  if (li.classList.contains("gold")) top.append(el("b", "gold-tag", tr("card.gold")));
  if (c.move !== undefined) {
    const m = el("b", `move ${moveCls(c.move)}`, moveText(c.move));
    m.title = side === "after" ? tr("card.move_after", { before: c.rank_before, after: c.rank }) : tr("card.move_before", { after: c.rank_after });
    top.append(m);
  }
  const bar = el("div", "scorebar");
  const track = el("div", "track");
  const fill = el("i");
  fill.style.width = `${Math.max(0, Math.min(1, c.score / (maxScore || 1))) * 100}%`;
  track.append(fill);
  bar.append(track, el("span", "num", String(c.score)));
  li.append(top, bar, renderText(c.text, c.highlights, c.evidence));
  return li;
}

function renderCompare(r) {
  const before = $("#col-before");
  const after = $("#col-after");
  before.textContent = "";
  after.textContent = "";
  // 척도 이름은 응답의 한국어 값 대신 화면 언어 표에서 찾는다(응답 JSON 은 그대로)
  $("#scale-before").textContent = tr(`scale.${r.retriever}`);
  $("#scale-after").textContent = tr("scale.after");
  const gold = goldAnchor(r.question);
  // 막대 길이: 코사인·리랭커는 0~1 그대로, BM25·RRF 는 상한이 없어 열의 최댓값 기준
  const maxB = r.retriever === "vector" ? 1 : Math.max(...r.before.map((c) => c.score), 1e-9);
  for (const c of r.before) before.append(renderCard(c, "before", maxB, gold));
  if (!r.before.length) before.append(el("li", "empty", tr("empty.no_candidates")));

  if (r.after === null) {
    after.append(el("li", "pending", tr("pending.reranker")));
  } else {
    for (const c of r.after) after.append(renderCard(c, "after", 1, gold));
  }
  const notes = [];
  if (r.no_evidence) notes.push(tr("note.no_evidence", { min: r.min_score }));
  if (gold && r.before.length && !r.before.some((c) => c.text.includes(gold)))
    notes.push(tr("note.gold_missing"));
  const q = state.status.questions.find((x) => x.q === r.question);
  if (q && q.note && $("#collection").value === "samples") notes.push(tr("note.limit", { note: q.note }));
  const t = r.timings_ms;
  notes.push(r.mode === "real" ? tr("note.timings", { embed: t.embed, retrieve: t.retrieve, rerank: t.rerank }) : "");
  showNote(notes.filter(Boolean).join("\n"));
  animateAfter();
  renderDistribution(r);
}

// '리랭킹 후' 카드를 '전' 위치에서 출발시켜 제자리로 미끄러지게 한다(FLIP 애니메이션)
function animateAfter() {
  const before = new Map([...$("#col-before").children].map((li) => [li.dataset.chunkId, li]));
  const cards = [...$("#col-after").querySelectorAll(".card")];
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  for (const li of cards) {
    const from = before.get(li.dataset.chunkId);
    if (!from || reduce) continue;
    const dy = from.getBoundingClientRect().top - li.getBoundingClientRect().top;
    li.style.transition = "none";
    li.style.transform = `translateY(${dy}px)`;
  }
  void document.body.offsetHeight;  // 위치를 먼저 확정시킨 뒤
  for (const li of cards) {
    li.style.transition = "transform .7s ease";
    li.style.transform = "";
  }
  drawLinks();
  setTimeout(drawLinks, reduce ? 0 : 750);
}

// 같은 청크끼리 선으로 잇는다: 초록=올라감, 빨강=내려감, 회색 점선=그대로
function drawLinks() {
  const svg = $("#links");
  svg.textContent = "";
  if (getComputedStyle(svg).display === "none") return;
  const box = svg.getBoundingClientRect();
  const w = box.width;
  const after = new Map([...$("#col-after").querySelectorAll(".card")].map((li) => [li.dataset.chunkId, li]));
  for (const li of $("#col-before").querySelectorAll(".card")) {
    const to = after.get(li.dataset.chunkId);
    if (!to) continue;
    const a = li.getBoundingClientRect();
    const b = to.getBoundingClientRect();
    const y1 = a.top + a.height / 2 - box.top;
    const y2 = b.top + b.height / 2 - box.top;
    const move = Number(li.dataset.rank) - Number(to.dataset.rank);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M0,${y1} C${w / 2},${y1} ${w / 2},${y2} ${w},${y2}`);
    path.setAttribute("class", moveCls(move));
    svg.append(path);
  }
  svg.style.height = `${Math.max($("#col-before").offsetHeight, $("#col-after").offsetHeight) + 30}px`;
}

// 점수 분포 미니 차트: 열마다 '1위 점수'를 오른쪽 끝(100%)에 두고 나머지를 그 비율 위치에 찍는다.
// 척도가 달라도(RRF 0.03 안팎, 리랭커 0~1) 모양을 그대로 비교할 수 있다:
// 점들이 오른쪽에 몰려 있으면 1위와 나머지가 비슷하고, 1위만 오른쪽에 떨어져 있으면 1위가 확실히 앞선다.
function renderDistribution(r) {
  const box = $("#dist");
  const svg = $("#score-chart");
  svg.textContent = "";
  const ratio = r.top2_ratio || {};
  if (!r.after || !r.before.length || ratio.before == null || ratio.after == null) { box.hidden = true; return; }
  box.hidden = false;
  const W = Math.max(320, Math.round(svg.getBoundingClientRect().width)) || 600;  // 실제 폭으로 그려 점이 찌그러지지 않게
  svg.setAttribute("viewBox", `0 0 ${W} 92`);
  const x0 = 70, span = W - x0 - 34;  // 오른쪽에 '1위' 글자 자리를 남긴다
  const ns = "http://www.w3.org/2000/svg";
  const gold = goldAnchor(r.question);
  const rows = [
    { label: tr("dist.stage1"), cls: "b", cards: r.before },
    { label: tr("dist.reranker"), cls: "a", cards: r.after },
  ];
  const text = (x, y, t, anchor = "start") => {
    const n = document.createElementNS(ns, "text");
    n.setAttribute("x", x); n.setAttribute("y", y); n.setAttribute("text-anchor", anchor);
    n.textContent = t;
    return n;
  };
  rows.forEach((row, i) => {
    const y = 26 + i * 40;
    const top = Math.max(...row.cards.map((c) => c.score));  // 이 열의 1위 점수 = 오른쪽 끝
    const line = document.createElementNS(ns, "line");
    Object.entries({ x1: x0, x2: x0 + span, y1: y, y2: y }).forEach(([k, v]) => line.setAttribute(k, v));
    svg.append(line, text(0, y + 4, row.label), text(x0 + span + 8, y + 4, tr("dist.top")));
    // 정답 점은 맨 위에 보이도록 마지막에 그린다
    const cards = [...row.cards].sort((a, b) => Number(gold && a.text.includes(gold)) - Number(gold && b.text.includes(gold)));
    for (const c of cards) {
      const pos = top > 0 ? Math.max(0, Math.min(1, c.score / top)) : 0;
      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", x0 + span * pos);
      dot.setAttribute("cy", y); dot.setAttribute("r", 6);
      dot.setAttribute("class", row.cls + (gold && c.text.includes(gold) ? " gold" : ""));
      dot.setAttribute("opacity", ".65");
      svg.append(dot);
    }
  });
  svg.append(text(x0, 88, "0"));
  // 요약 문장: 2위 점수가 1위의 몇 %인지(작을수록 1위가 확실히 앞선다) — 그림의 2위 점 위치와 같은 값
  const pb = Math.round(ratio.before * 100), pa = Math.round(ratio.after * 100);
  const verdict = pa < pb ? tr("dist.verdict_better") : tr("dist.verdict_same");
  $("#dist-text").textContent = `${tr("dist.sep", { before: pb, after: pa })} ${verdict}`;
}

// ---------- 답변 A/B 스트리밍 ----------
function resetAnswers() {
  for (const id of ["#answer-a", "#answer-b"]) {
    const a = $(id);
    a.dataset.done = "false";
    const sum = a.querySelector("summary");
    sum.dataset.i18n = "answer.think";  // 언어를 바꾸면 라벨도 따라 바뀐다(생각 중에는 아래에서 속성을 뗀다)
    delete sum.dataset.tokens;
    sum.textContent = tr("answer.think");
    a.querySelector(".notice").hidden = true;
    const t = a.querySelector(".text");
    t.className = "text dim";
    t.dataset.i18n = "answer.none";
    t.textContent = tr("answer.none");
  }
}

function stopAnswers() {
  if (state.answerCtrl) state.answerCtrl.abort();  // 연결을 끊으면 서버가 Ollama 스트림까지 닫는다
  state.answerCtrl = null;
}

async function runAnswers() {
  if (!state.lastReq) return;
  stopAnswers();
  resetAnswers();
  const ctrl = new AbortController();
  state.answerCtrl = ctrl;
  $("#btn-answer").disabled = true;
  $("#btn-stop").hidden = false;
  document.body.dataset.answer = "streaming";
  try {
    const res = await fetch(withLang("/api/answer"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...state.lastReq, sides: "AB" }), signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {  // 한 줄 = 이벤트 하나(NDJSON)
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        if (line.trim()) handleAnswerEvent(JSON.parse(line));
      }
    }
    document.body.dataset.answer = "done";
  } catch (e) {
    document.body.dataset.answer = e.name === "AbortError" ? "stopped" : "error";
    for (const id of ["#answer-a", "#answer-b"]) {
      const a = $(id);
      if (a.dataset.done !== "true") {
        const n = a.querySelector(".notice");
        n.hidden = false;
        n.textContent = e.name === "AbortError" ? tr("answer.stopped") : tr("error.prefix", { message: e.message });
      }
    }
  } finally {
    if (state.answerCtrl === ctrl) state.answerCtrl = null;
    $("#btn-answer").disabled = false;
    $("#btn-stop").hidden = true;
  }
}

function handleAnswerEvent(ev) {
  if (!ev.side) return;
  const box = $(ev.side === "A" ? "#answer-a" : "#answer-b");
  const text = box.querySelector(".text");
  if (text.classList.contains("dim")) { text.className = "text"; text.textContent = ""; text.removeAttribute("data-i18n"); }
  if (ev.type === "thinking") {
    const sum = box.querySelector("summary");
    sum.removeAttribute("data-i18n");
    sum.dataset.tokens = ev.tokens;
    sum.textContent = tr("answer.thinking_now", { tokens: ev.tokens });
  } else if (ev.type === "content") {
    text.textContent += localAnswer(ev.text);
  } else if (ev.type === "notice" || ev.type === "error") {
    const n = box.querySelector(".notice");
    n.hidden = false;
    n.textContent = ev.text;
  } else if (ev.type === "end") {
    const s = box.querySelector("summary");
    if (s.dataset.tokens) {
      s.textContent = tr("answer.thinking_done", { tokens: s.dataset.tokens }) + (ev.cached ? ` ${tr("answer.cached")}` : "");
    }
    renderCitations(text, localAnswer(ev.text) || text.textContent, ev.side);
    box.dataset.done = "true";  // 스크린샷 스크립트가 기다리는 신호
  }
}

// '근거 없음' 답은 서버가 한국어 고정 문구로 보낸다(답변 데이터). 화면 언어로 바꿔 보여 준다.
const NO_EVIDENCE_KO = "문서에서 근거를 찾지 못했습니다.";
const localAnswer = (text) => (text === NO_EVIDENCE_KO ? tr("answer.no_evidence") : text);

// 답 속의 [n]을 누르면 그 근거 카드가 깜빡인다(A=전 열, B=후 열의 n위)
function renderCitations(node, answer, side) {
  node.textContent = "";
  const parts = answer.split(/(\[\d+\])/);
  for (const part of parts) {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) { node.append(part); continue; }
    const a = el("a", "cite", part);
    a.href = "#";
    a.addEventListener("click", (e) => {
      e.preventDefault();
      const col = side === "A" ? "#col-before" : "#col-after";
      const card = $(col).querySelector(`.card[data-rank="${m[1]}"]`);
      if (!card) return;
      card.scrollIntoView({ block: "center", behavior: "smooth" });
      card.classList.remove("flash");
      void card.offsetWidth;
      card.classList.add("flash");
    });
    node.append(a);
  }
}

// ---------- 실험 탭 ----------
async function runEval() {
  const btn = $("#btn-eval");
  const table = $("#eval-table");
  btn.disabled = true;
  table.dataset.ready = "false";
  $("#eval-status").textContent = tr("eval.running");
  try {
    const { top_k, chunking, title_prefix } = currentRequest();
    const r = await api("/api/eval", { top_k, chunking, title_prefix });
    state.lastEval = r;
    renderEval(r);
    table.dataset.ready = "true";
  } catch (e) {
    $("#eval-status").textContent = tr("error.prefix", { message: e.message });
  } finally {
    btn.disabled = false;
  }
}

function cmpCell(before, after, fmt) {
  // 화면에 보이는 값(반올림 후)으로 비교해, 똑같아 보이는데 색만 다른 일이 없게 한다
  const [b, a] = [fmt(before), fmt(after)];
  const cls = a === b ? "" : after > before ? "better" : "worse";
  return el("td", cls, `${b} → ${a}`);
}

// 응답의 explain(한국어 값) 대신 순위로 화면 언어 문장을 만든다
function explainRanks(b, a) {
  if (b === null) return tr("explain.outside");
  if (a < b) return tr("explain.up", { before: b, after: a });
  if (a > b) return tr("explain.down", { before: b, after: a });
  return tr("explain.same", { before: b });
}

function renderEval(r) {
  $("#eval-status").textContent = tr("eval.status", {
    reranker: r.reranker, strategy: r.chunking.strategy, prefix: r.chunking.title_prefix ? tr("eval.prefix_on") : "",
    top_k: r.top_k, changed: r.orders_changed });
  const names = { keyword: tr("retriever.keyword"), vector: tr("retriever.vector"), hybrid: tr("retriever.hybrid") };
  const t = $("#eval-table");
  t.textContent = "";
  const head = el("tr");
  [tr("eval.col_retriever"), tr("eval.col_hit"), tr("eval.col_mrr"), tr("eval.col_changed")].forEach((h) => head.append(el("th", "", h)));
  t.append(head);
  for (const [name, row] of Object.entries(r.table)) {
    const rowEl = el("tr");
    rowEl.append(el("td", "", names[name] || name),
      cmpCell(row.hit1_before, row.hit1_after, (x) => `${x}/${r.n_questions}`),
      cmpCell(row.mrr_before, row.mrr_after, (x) => x.toFixed(2)),
      el("td", "", String(row.orders_changed)));
    t.append(rowEl);
  }
  const q = $("#eval-questions");
  q.textContent = "";
  const h2 = el("tr");
  [tr("eval.col_question"), ...Object.keys(r.table).map((n) => tr("eval.col_rank", { name: names[n] || n })), tr("eval.col_why")].forEach((h) => h2.append(el("th", "", h)));
  q.append(h2);
  for (const row of r.questions) {
    const rowEl = el("tr");
    rowEl.append(el("td", "", row.q));
    for (const n of Object.keys(r.table)) {
      const [b, a] = row.ranks[n];
      const cls = b === null ? "worse" : a < b ? "better" : a > b ? "worse" : "";
      const td = el("td", cls, `${b ?? tr("eval.outside")} → ${a ?? tr("eval.outside")}`);
      td.title = explainRanks(b, a);
      rowEl.append(td);
    }
    rowEl.append(el("td", "note", row.note || ""));
    q.append(rowEl);
  }
}

// ---------- 내 문서 탭 ----------
async function indexDocs(body) {
  const status = $("#index-status");
  status.textContent = tr("docs.indexing");
  try {
    const r = await api("/api/index", { ...body, chunking: $("#chunking").value, title_prefix: $("#title-prefix").checked });
    status.textContent = tr("docs.done", { docs: r.docs, chunks: r.chunks, files: r.files.join(", ") });
    const opt = $("#collection option[value=mine]");
    opt.disabled = false;
    $("#collection").value = "mine";
    showTab("compare");
  } catch (e) {
    status.textContent = tr("error.prefix", { message: e.message });
  }
}

async function readFiles(fileList) {
  const files = [];
  for (const f of [...fileList].slice(0, 200)) {
    if (!/\.(md|txt)$/i.test(f.name) || f.size > 1_000_000) continue;
    files.push({ name: f.name, text: await f.text() });  // 브라우저가 내용을 읽어 JSON 으로 보낸다
  }
  if (!files.length) { $("#index-status").textContent = tr("docs.no_files"); return; }
  indexDocs({ files });
}

// ---------- 언어 바꾸기 ----------
// 정적 문구는 applyStaticTexts() 가, 서버가 만든 문구(모드 안내)는 상태를 다시 받아서, 그린 결과는 다시 그려서 바꾼다.
async function changeLanguage(lang) {
  setLang(lang);
  try {
    await loadStatus();
    renderChips();
  } catch (e) { /* 서버 상태를 못 읽어도 정적 문구는 이미 바뀌었다 */ }
  if (state.lastResult) renderCompare(state.lastResult);
  if (state.lastEval) renderEval(state.lastEval);
}

// ---------- 탭·이벤트 연결 ----------
function showTab(name) {
  for (const b of document.querySelectorAll(".tab")) b.classList.toggle("active", b.dataset.tab === name);
  for (const p of ["compare", "eval", "docs"]) $(`#panel-${p}`).hidden = p !== name;
  if (name === "compare") requestAnimationFrame(drawLinks);
}

function wire() {
  $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); runCompare(); });
  for (const b of document.querySelectorAll(".tab")) b.addEventListener("click", () => showTab(b.dataset.tab));
  for (const b of document.querySelectorAll("#retriever button")) {
    b.addEventListener("click", () => {
      state.retriever = b.dataset.v;
      for (const x of document.querySelectorAll("#retriever button")) x.classList.toggle("on", x === b);
      runCompare();
    });
  }
  const topK = $("#top-k");
  topK.addEventListener("input", () => { $("#top-k-val").textContent = topK.value; });
  topK.addEventListener("change", runCompare);
  for (const id of ["#chunking", "#title-prefix", "#collection"]) $(id).addEventListener("change", runCompare);
  $("#btn-answer").addEventListener("click", runAnswers);
  $("#btn-stop").addEventListener("click", stopAnswers);
  $("#btn-eval").addEventListener("click", runEval);
  $("#lang-select").addEventListener("change", (e) => changeLanguage(e.target.value));
  $("#btn-rr-retry").addEventListener("click", async () => { await api("/api/reranker/retry", {}); loadStatus(); });
  $("#btn-index-folder").addEventListener("click", () => {
    const folder = $("#folder-path").value.trim();
    if (folder) indexDocs({ folder });
  });
  const drop = $("#dropzone");
  $("#file-input").addEventListener("change", (e) => readFiles(e.target.files));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); readFiles(e.dataTransfer.files); });
  window.addEventListener("resize", () => requestAnimationFrame(drawLinks));
  window.addEventListener("pagehide", () => { stopAnswers(); clearTimeout(state.pollTimer); });
}

(async function init() {
  applyStaticTexts();
  wire();
  try {
    await loadStatus();
    renderChips();
    if (state.status.mine.docs) $("#collection option[value=mine]").disabled = false;
    document.body.dataset.state = "ready";
  } catch (e) {
    showNote(tr("error.status", { message: e.message }));
  }
})();
