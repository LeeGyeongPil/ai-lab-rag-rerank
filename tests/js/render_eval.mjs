// Runs the real static/i18n.js + static/app.js with a tiny fake DOM (no dependencies),
// then calls renderEval() with a fixture and prints what was drawn as JSON.
// usage: node render_eval.mjs <static dir> <en|ko> <eval-response.json>
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const [dir, lang, fixturePath] = process.argv.slice(2);
const fixture = JSON.parse(fs.readFileSync(fixturePath, "utf8"));

class FakeEl {
  constructor(tag) { this.tagName = tag; this.children = []; this.dataset = {}; this.style = {}; this.className = ""; this._text = ""; this.title = ""; this.value = ""; this.checked = false; this.disabled = false;
    this.classList = { add() {}, remove() {}, toggle() {}, contains: () => false }; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this._text = String(v); this.children = []; }
  append(...kids) { for (const k of kids) this.children.push(typeof k === "string" ? Object.assign(new FakeEl("#text"), { _text: k }) : k); }
  appendChild(k) { this.append(k); return k; }
  setAttribute() {} getAttribute() { return null; } addEventListener() {} removeEventListener() {}
  querySelector() { return new FakeEl("div"); } querySelectorAll() { return []; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 0, height: 0 }; }
}
const registry = new Map();
const byId = (sel) => { if (!registry.has(sel)) registry.set(sel, new FakeEl("div")); return registry.get(sel); };
const document = {
  documentElement: new FakeEl("html"), title: "",
  querySelector: byId, querySelectorAll: () => [], getElementById: byId,
  createElement: (t) => new FakeEl(t), createElementNS: (_n, t) => new FakeEl(t), addEventListener() {}, body: new FakeEl("body"),
};
const errors = [];
const win = { addEventListener() {}, requestAnimationFrame: (f) => f() };
const sandbox = {
  document, window: win, location: { search: `?lang=${lang}` }, navigator: { language: "en-US" },
  localStorage: { getItem: () => null, setItem() {} }, URLSearchParams, console,
  fetch: async () => { throw new Error("no network in this test"); },  // init() fails softly; we only test rendering
  setTimeout, clearTimeout, requestAnimationFrame: (f) => f(), AbortController, Math, JSON, Object, Array, String, Number, Promise,
};
vm.createContext(sandbox);
for (const f of ["i18n.js", "app.js"]) vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), sandbox, { filename: f });
await new Promise((r) => setTimeout(r, 20));  // let init() settle

let thrown = null;
try { vm.runInContext("renderEval(__fixture)", Object.assign(sandbox, { __fixture: fixture })); } catch (e) { thrown = String(e); }
const q = registry.get("#eval-questions");
const rows = q ? q.children.length - 1 : 0;  // minus the header row
console.log(JSON.stringify({
  thrown, rows, lang: sandbox.document.documentElement.lang,
  status: byId("#eval-status").textContent,
  summaryRows: byId("#eval-table").children.length - 1,
  outside: q ? q.textContent.includes(vm.runInContext('tr("eval.outside")', sandbox)) : false,
}));
