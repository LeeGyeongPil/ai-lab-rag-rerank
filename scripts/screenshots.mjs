// README 장면 캡처(개발용). 서버를 먼저 띄워 두고 실행한다.
//   터미널 1: uv run rag-rerank serve --no-browser          (실제 모드 권장, mock 이면 --mock)
//   터미널 2: npm i --no-save playwright && npx playwright install chromium
//             node scripts/screenshots.mjs                   (결과: docs/images/*.png)
// 환경변수: BASE_URL(기본 http://127.0.0.1:8765), OUT(기본 docs/images),
//          CHROMIUM_PATH(선택: 이미 설치된 크로미움 실행 파일을 쓰고 싶을 때)
import { mkdirSync } from "node:fs";
import { chromium } from "playwright";

const BASE = process.env.BASE_URL || "http://127.0.0.1:8765";
const OUT = process.env.OUT || "docs/images";
const QUESTION = "중학생도 공유자전거를 탈 수 있나요?";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch(
  process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {},
);
try {
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 2 });
  await page.goto(BASE);
  await page.waitForSelector('body[data-state="ready"]');

  // 리랭커가 준비될 때까지(첫 실행 내려받기) 기다린다
  await page.waitForFunction(async () => (await (await fetch("/api/status")).json()).reranker.state === "ready",
    null, { timeout: 600_000, polling: 1000 });

  // 1) 메인 비교 화면: 예시 질문 칩 → 두 열 + 연결선
  await page.getByRole("button", { name: QUESTION }).click();
  await page.waitForSelector('body[data-state="compared"]');
  await page.waitForTimeout(900); // 카드 이동 애니메이션(0.7초)이 끝나고 선이 그려질 때까지
  await page.locator("#board").screenshot({ path: `${OUT}/01-compare.png` });

  // 2) 하이라이트 확대: 정답 카드(리랭킹 후 열). mock 처럼 정답이 후보 밖이면 1위 카드로 대신한다.
  const gold = page.locator("#col-after .card.gold");
  const card = (await gold.count()) ? gold.first() : page.locator("#col-after .card").first();
  await card.screenshot({ path: `${OUT}/02-highlight.png` });

  // 3) 점수 분포
  await page.locator("#dist").screenshot({ path: `${OUT}/04-distribution.png` });

  // 4) 답변 A/B (실제 모드는 두 답에 2~3분)
  await page.locator("#btn-answer").click();
  await page.waitForSelector('#answer-b[data-done="true"]', { timeout: 600_000 });
  await page.locator(".answers").screenshot({ path: `${OUT}/03-answers.png` });

  // 5) 실험 탭 성적표
  await page.locator("#tab-eval").click();
  await page.locator("#btn-eval").click();
  await page.waitForSelector('#eval-table[data-ready="true"]', { timeout: 300_000 });
  await page.locator("#panel-eval").screenshot({ path: `${OUT}/05-eval.png` });
  console.log(`saved to ${OUT}/`);
} finally {
  await browser.close();
}
