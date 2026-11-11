// README 장면 캡처(개발용). 서버를 먼저 띄워 두고 실행한다. 화면 언어별로 찍는다.
//   터미널 1: uv run rag-rerank serve --no-browser          (실제 모드 권장, mock 이면 --mock)
//   터미널 2: npm i --no-save playwright && npx playwright install chromium
//             node scripts/screenshots.mjs                   (결과: docs/images/*.png, *.ko.png)
// 환경변수: BASE_URL(기본 http://127.0.0.1:8765), OUT(기본 docs/images),
//          LANGS(기본 "en,ko" — 찍을 화면 언어), CHROMIUM_PATH(선택: 이미 설치된 크로미움 실행 파일)
// 파일 이름: 영어 화면 = <장면>.png (README.md), 한국어 화면 = <장면>.ko.png (README.ko.md).
// 언어는 주소 쿼리 ?lang=en|ko 로 정한다(화면의 언어 메뉴와 같은 값, localStorage 키: rag-rerank-lang).
// 실제 모드(Ollama·리랭커 모델 필요)에서만 의미 있는 장면(08-limit, 09-no-evidence)과 00-hero.gif 는 이 스크립트가 찍지 않는다.
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE_URL || "http://127.0.0.1:8765";
const OUT = process.env.OUT || "docs/images";
const LANGS = (process.env.LANGS || "en,ko").split(",").map((s) => s.trim()).filter(Boolean);
const QUESTION = "중학생도 공유자전거를 탈 수 있나요?"; // 예시 질문은 한국어 데이터라 두 언어 모두 같다
mkdirSync(OUT, { recursive: true });

// 내 문서 탭 장면에 쓸 임시 폴더(없으면 안내 문구만 찍힌다)
const demoDir = mkdtempSync(join(tmpdir(), "rag-rerank-shot-"));
writeFileSync(join(demoDir, "notes.md"), "# Notes\n\nThe weekly meeting is on Monday at 10.\n");

const suffix = (lang) => (lang === "en" ? "" : `.${lang}`);

async function shootLanguage(browser, lang) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 2 });
  const file = (name) => `${OUT}/${name}${suffix(lang)}.png`;
  try {
    await page.goto(`${BASE}/?lang=${lang}`);
    await page.waitForSelector('body[data-state="ready"]');
    // 리랭커가 준비될 때까지(첫 실행 내려받기) 기다린다
    await page.waitForFunction(async () => (await (await fetch("/api/status")).json()).reranker.state === "ready",
      null, { timeout: 600_000, polling: 1000 });

    // 01) 첫 화면: 예시 질문 칩
    await page.screenshot({ path: file("01-start") });

    // 02) 메인 비교 화면: 예시 질문 칩 → 두 열 + 연결선
    await page.getByRole("button", { name: QUESTION }).click();
    await page.waitForSelector('body[data-state="compared"]');
    await page.waitForTimeout(900); // 카드 이동 애니메이션(0.7초)이 끝나고 선이 그려질 때까지
    await page.locator("#board").screenshot({ path: file("02-compare") });

    // 03) 하이라이트 확대: 정답 카드(리랭킹 후 열). 정답이 후보 밖이면 1위 카드로 대신한다.
    const gold = page.locator("#col-after .card.gold");
    const card = (await gold.count()) ? gold.first() : page.locator("#col-after .card").first();
    await card.screenshot({ path: file("03-highlight") });

    // 04) 점수 분포
    await page.locator("#dist").screenshot({ path: file("04-distribution") });

    // 05) 답변 A/B (실제 모드는 두 답에 2~3분)
    await page.locator("#btn-answer").click();
    await page.waitForSelector('#answer-b[data-done="true"]', { timeout: 600_000 });
    await page.locator(".answers").screenshot({ path: file("05-answers") });

    // 06) 실험 탭 성적표
    await page.locator("#tab-eval").click();
    await page.locator("#btn-eval").click();
    await page.waitForSelector('#eval-table[data-ready="true"]', { timeout: 300_000 });
    await page.locator("#panel-eval").screenshot({ path: file("06-eval") });

    // 07) 내 문서 탭
    await page.locator("#tab-docs").click();
    await page.locator("#folder-path").fill(demoDir);
    await page.locator("#panel-docs").screenshot({ path: file("07-docs") });

    // 언어 메뉴로 바꾸면 화면이 실제로 바뀌고 저장되는지도 한 번 확인한다(영어 → 한국어)
    if (lang === "en") {
      await page.locator("#lang-select").selectOption("ko");
      const saved = await page.evaluate(() => localStorage.getItem("rag-rerank-lang"));
      if (saved !== "ko") throw new Error("language choice was not saved");
    }
    console.log(`saved ${lang} screens to ${OUT}/`);
  } finally {
    await page.close();
  }
}

const browser = await chromium.launch(
  process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {},
);
try {
  for (const lang of LANGS) await shootLanguage(browser, lang);
} finally {
  await browser.close();
}
