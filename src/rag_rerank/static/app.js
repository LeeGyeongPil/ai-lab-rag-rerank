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
};

// ---------- 공통 ----------
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
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
  $("#models").textContent = `임베딩 ${s.models.embed} · 리랭커 ${s.reranker.model} · 답변 ${s.models.chat}`;
  const note = $("#mode-note");
  if (s.mode === "mock") {
    note.hidden = false;
    note.textContent = "";
    note.append(el("b", "", "MOCK 모드 — 결과는 동작 확인용이며 품질을 뜻하지 않습니다. "), s.mode_reason + " ");
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
    r.state === "downloading" ? "리랭커 모델 내려받는 중" :
    r.state === "loading" ? "리랭커 모델 불러오는 중" :
    r.state === "error" ? `리랭커 준비 실패: ${r.error || ""}` : "리랭커 준비 중";
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
    if (q.note) b.title = "한계 사례: " + q.note;
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
    showNote(`오류: ${e.message}`);
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
  if (li.classList.contains("gold")) top.append(el("b", "gold-tag", "정답"));
  if (c.move !== undefined) {
    const m = el("b", `move ${moveCls(c.move)}`, moveText(c.move));
    m.title = side === "after" ? `리랭킹 전 ${c.rank_before}위 → 후 ${c.rank}위` : `리랭킹 후 ${c.rank_after}위`;
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
  $("#scale-before").textContent = r.scales.before;
  $("#scale-after").textContent = r.scales.after;
  const gold = goldAnchor(r.question);
  // 막대 길이: 코사인·리랭커는 0~1 그대로, BM25·RRF 는 상한이 없어 열의 최댓값 기준
  const maxB = r.retriever === "vector" ? 1 : Math.max(...r.before.map((c) => c.score), 1e-9);
  for (const c of r.before) before.append(renderCard(c, "before", maxB, gold));
  if (!r.before.length) before.append(el("li", "empty", "1단계 검색기가 찾은 후보가 없습니다."));

  if (r.after === null) {
    after.append(el("li", "pending", "리랭커 모델을 준비하는 중입니다. 준비되면 자동으로 채워집니다(위 진행 막대)."));
  } else {
    for (const c of r.after) after.append(renderCard(c, "after", 1, gold));
  }
  const notes = [];
  if (r.no_evidence) notes.push(`리랭커 최고 점수가 문턱값(${r.min_score})보다 낮아 '근거 부족'으로 판단했습니다 — 답변 B는 답을 지어내지 않습니다.`);
  if (gold && r.before.length && !r.before.some((c) => c.text.includes(gold)))
    notes.push("정답 문단이 1단계 후보 안에 없습니다. 리랭커는 후보 안에서만 순서를 바꾸므로 살릴 수 없습니다.");
  const q = state.status.questions.find((x) => x.q === r.question);
  if (q && q.note && $("#collection").value === "samples") notes.push("한계 사례: " + q.note);
  const t = r.timings_ms;
  notes.push(r.mode === "real" ? `시간: 임베딩 ${t.embed}ms · 1단계 ${t.retrieve}ms · 리랭킹 ${t.rerank}ms` : "");
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
    { label: "1단계", cls: "b", cards: r.before },
    { label: "리랭커", cls: "a", cards: r.after },
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
    svg.append(line, text(0, y + 4, row.label), text(x0 + span + 8, y + 4, "1위"));
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
  const verdict = pa < pb
    ? "리랭커가 1위를 나머지와 더 뚜렷하게 갈라놓았습니다."
    : "이 질문에서는 리랭커도 1위와 2위를 크게 가르지 못했습니다.";
  $("#dist-text").textContent =
    `1위와 2위 차이 — 2위 점수가 1위의 몇 %인가: 1단계 ${pb}% → 리랭커 ${pa}%. ` +
    `작을수록 1위가 확실히 앞선다는 뜻입니다. ${verdict}`;
}

// ---------- 답변 A/B 스트리밍 ----------
function resetAnswers() {
  for (const id of ["#answer-a", "#answer-b"]) {
    const a = $(id);
    a.dataset.done = "false";
    a.querySelector("summary").textContent = "생각 과정";
    a.querySelector(".notice").hidden = true;
    const t = a.querySelector(".text");
    t.className = "text dim";
    t.textContent = "아직 답변이 없습니다.";
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
    const res = await fetch("/api/answer", {
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
        n.textContent = e.name === "AbortError" ? "중지했습니다." : `오류: ${e.message}`;
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
  if (text.classList.contains("dim")) { text.className = "text"; text.textContent = ""; }
  if (ev.type === "thinking") {
    box.querySelector("summary").textContent = `생각 중… ${ev.tokens}토큰`;
  } else if (ev.type === "content") {
    text.textContent += ev.text;
  } else if (ev.type === "notice" || ev.type === "error") {
    const n = box.querySelector(".notice");
    n.hidden = false;
    n.textContent = ev.text;
  } else if (ev.type === "end") {
    const s = box.querySelector("summary");
    s.textContent = s.textContent.replace("생각 중…", "생각") + (ev.cached ? " (저장된 답)" : "");
    renderCitations(text, ev.text || text.textContent, ev.side);
    box.dataset.done = "true";  // 스크린샷 스크립트가 기다리는 신호
  }
}

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
  $("#eval-status").textContent = "계산 중…";
  try {
    const { top_k, chunking, title_prefix } = currentRequest();
    const r = await api("/api/eval", { top_k, chunking, title_prefix });
    renderEval(r);
    $("#eval-status").textContent = `리랭커 ${r.reranker} · 청킹 ${r.chunking.strategy}${r.chunking.title_prefix ? "(제목 붙임)" : ""} · top-k ${r.top_k} · 순서가 바뀐 (질문×검색기) 조합 ${r.orders_changed}개`;
    table.dataset.ready = "true";
  } catch (e) {
    $("#eval-status").textContent = `오류: ${e.message}`;
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

function renderEval(r) {
  const names = { keyword: "키워드", vector: "벡터", hybrid: "하이브리드" };
  const t = $("#eval-table");
  t.textContent = "";
  const head = el("tr");
  ["1단계 검색기", "hit@1 전 → 후", "MRR 전 → 후", "순서 바뀐 질문"].forEach((h) => head.append(el("th", "", h)));
  t.append(head);
  for (const [name, row] of Object.entries(r.table)) {
    const tr = el("tr");
    tr.append(el("td", "", names[name] || name),
      cmpCell(row.hit1_before, row.hit1_after, (x) => `${x}/${r.n_questions}`),
      cmpCell(row.mrr_before, row.mrr_after, (x) => x.toFixed(2)),
      el("td", "", String(row.orders_changed)));
    t.append(tr);
  }
  const q = $("#eval-questions");
  q.textContent = "";
  const h2 = el("tr");
  ["질문", ...Object.keys(r.table).map((n) => `${names[n] || n} 정답 순위`), "왜?"].forEach((h) => h2.append(el("th", "", h)));
  q.append(h2);
  for (const row of r.questions) {
    const tr = el("tr");
    tr.append(el("td", "", row.q));
    for (const n of Object.keys(r.table)) {
      const [b, a] = row.ranks[n];
      const cls = b === null ? "worse" : a < b ? "better" : a > b ? "worse" : "";
      const td = el("td", cls, `${b ?? "밖"} → ${a ?? "밖"}`);
      td.title = row.explain[n];
      tr.append(td);
    }
    tr.append(el("td", "note", row.note || ""));
    q.append(tr);
  }
}

// ---------- 내 문서 탭 ----------
async function indexDocs(body) {
  const status = $("#index-status");
  status.textContent = "색인하는 중…";
  try {
    const r = await api("/api/index", { ...body, chunking: $("#chunking").value, title_prefix: $("#title-prefix").checked });
    status.textContent = `문서 ${r.docs}개 → 청크 ${r.chunks}개 색인 완료: ${r.files.join(", ")}`;
    const opt = $("#collection option[value=mine]");
    opt.disabled = false;
    $("#collection").value = "mine";
    showTab("compare");
  } catch (e) {
    status.textContent = `오류: ${e.message}`;
  }
}

async function readFiles(fileList) {
  const files = [];
  for (const f of [...fileList].slice(0, 200)) {
    if (!/\.(md|txt)$/i.test(f.name) || f.size > 1_000_000) continue;
    files.push({ name: f.name, text: await f.text() });  // 브라우저가 내용을 읽어 JSON 으로 보낸다
  }
  if (!files.length) { $("#index-status").textContent = ".md / .txt 파일이 없습니다(파일당 1MB 이하)."; return; }
  indexDocs({ files });
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
  wire();
  try {
    await loadStatus();
    renderChips();
    if (state.status.mine.docs) $("#collection option[value=mine]").disabled = false;
    document.body.dataset.state = "ready";
  } catch (e) {
    showNote(`서버 상태를 읽지 못했습니다: ${e.message}`);
  }
})();
