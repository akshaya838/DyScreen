/* DyScreen - browser mini-games (Module 9) and report page (Module 11).
   Every question produces one record:
   { game_id, question_id, clicks, hits, misses, score, response_time_ms }
   Only these numbers and the age are sent to Flask. No personal data. */

const allResults = [];
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const shuffle = (a) => { const b = [...a]; for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; };

function record(game_id, question_id, clicks, hits, misses, startTime) {
  allResults.push({ game_id, question_id, clicks, hits, misses, score: hits,
                    response_time_ms: Math.round(performance.now() - startTime) });
}

function show(screenId) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  $(screenId).classList.add("active");
}

function tileButton(text, extraClass = "") {
  const b = document.createElement("button");
  b.className = "tile " + extraClass;
  b.textContent = text;
  return b;
}

/* ---------------- Game 1: Letter Hunt (visual discrimination, timed search) */
async function letterHunt(area, setStatus) {
  const rounds = [
    { target: "b", others: ["d", "p", "q"] },
    { target: "p", others: ["q", "b", "d"] },
    { target: "m", others: ["n", "w", "u"] },
  ];
  for (let q = 0; q < rounds.length; q++) {
    const { target, others } = rounds[q];
    const letters = shuffle([...Array(5).fill(target),
      ...Array.from({ length: 11 }, () => others[Math.floor(Math.random() * others.length)])]);
    let clicks = 0, hits = 0, misses = 0;
    area.innerHTML = `<p>Tap every <span class="target-letter">${target}</span></p>`;
    const grid = document.createElement("div");
    grid.className = "grid g4";
    area.appendChild(grid);
    const done = document.createElement("button");
    done.className = "btn"; done.textContent = "I found them all";
    area.appendChild(done);

    const start = performance.now();
    await new Promise((resolve) => {
      let timeLeft = 15;
      setStatus(`Round ${q + 1} of 3  |  ${timeLeft} s left`);
      const timer = setInterval(() => {
        timeLeft--; setStatus(`Round ${q + 1} of 3  |  ${timeLeft} s left`);
        if (timeLeft <= 0) finish();
      }, 1000);
      function finish() { clearInterval(timer); resolve(); }
      letters.forEach((ch) => {
        const b = tileButton(ch);
        b.onclick = () => {
          clicks++;
          if (ch === target) { hits++; b.classList.add("hit"); } else { misses++; b.classList.add("miss"); }
          b.disabled = true;
          if (hits === 5) finish();
        };
        grid.appendChild(b);
      });
      done.onclick = finish;
    });
    record("letter_hunt", q + 1, clicks, hits, misses, start);
    await sleep(500);
  }
}

/* ---------------- Game 2: Symbol Memory (working memory) */
async function symbolMemory(area, setStatus) {
  const symbols = ["★", "●", "▲", "■", "◆", "♥"];
  const lengths = [3, 4, 5];
  for (let q = 0; q < lengths.length; q++) {
    const seq = Array.from({ length: lengths[q] }, () => symbols[Math.floor(Math.random() * symbols.length)]);
    setStatus(`Round ${q + 1} of 3  |  Watch carefully`);
    area.innerHTML = `<div class="big-symbols">${seq.join("")}</div>`;
    await sleep(1200 + 700 * seq.length);
    area.innerHTML = `<div class="big-symbols" id="answerRow">&nbsp;</div><div class="options" id="palette"></div>`;
    setStatus(`Round ${q + 1} of 3  |  Tap the symbols in the same order`);
    let clicks = 0, hits = 0, misses = 0;
    const answer = [];
    const start = performance.now();
    await new Promise((resolve) => {
      symbols.forEach((s) => {
        const b = tileButton(s);
        b.onclick = () => {
          const pos = answer.length;
          clicks++;
          if (s === seq[pos]) hits++; else misses++;
          answer.push(s);
          $("answerRow").textContent = answer.join("");
          if (answer.length === seq.length) resolve();
        };
        $("palette").appendChild(b);
      });
    });
    record("symbol_memory", q + 1, clicks, hits, misses, start);
    await sleep(600);
  }
}

/* ---------------- Game 3: Odd One Out (visual discrimination) */
async function oddOneOut(area, setStatus) {
  const rounds = [["d", "b"], ["q", "p"], ["u", "n"], ["was", "saw"]];
  for (let q = 0; q < rounds.length; q++) {
    const [common, odd] = rounds[q];
    const items = shuffle([...Array(8).fill(common), odd]);
    setStatus(`Round ${q + 1} of ${rounds.length}`);
    area.innerHTML = `<p>Tap the one that is different</p>`;
    const grid = document.createElement("div");
    grid.className = "grid g3";
    area.appendChild(grid);
    const start = performance.now();
    await new Promise((resolve) => {
      items.forEach((it) => {
        const b = tileButton(it, it.length > 1 ? "word" : "");
        b.onclick = () => {
          const ok = it === odd;
          b.classList.add(ok ? "hit" : "miss");
          record("odd_one_out", q + 1, 1, ok ? 1 : 0, ok ? 0 : 1, start);
          grid.querySelectorAll("button").forEach((x) => (x.disabled = true));
          setTimeout(resolve, 500);
        };
        grid.appendChild(b);
      });
    });
  }
}

/* ---------------- Game 4: What Comes Next (executive function / reasoning) */
async function patternNext(area, setStatus) {
  const rounds = [
    { seq: ["▲", "●", "▲", "●", "▲"], answer: "●", options: ["▲", "●", "■"] },
    { seq: ["★", "★", "■", "★", "★"], answer: "■", options: ["★", "■", "●"] },
    { seq: ["1", "2", "3", "1", "2"], answer: "3", options: ["1", "3", "4"] },
    { seq: ["a", "b", "b", "a", "b"], answer: "b", options: ["a", "b", "d"] },
  ];
  for (let q = 0; q < rounds.length; q++) {
    const r = rounds[q];
    setStatus(`Round ${q + 1} of ${rounds.length}`);
    area.innerHTML = `<div class="big-symbols">${r.seq.join(" ")} ?</div><div class="options" id="opts"></div>`;
    const start = performance.now();
    await new Promise((resolve) => {
      shuffle(r.options).forEach((o) => {
        const b = tileButton(o);
        b.onclick = () => {
          const ok = o === r.answer;
          b.classList.add(ok ? "hit" : "miss");
          record("pattern_next", q + 1, 1, ok ? 1 : 0, ok ? 0 : 1, start);
          $("opts").querySelectorAll("button").forEach((x) => (x.disabled = true));
          setTimeout(resolve, 500);
        };
        $("opts").appendChild(b);
      });
    });
  }
}

/* ---------------- Game 5: Sound Match (auditory discrimination, language-independent) */
let audioCtx = null;
function beep(when, dur) {
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.frequency.value = 660;
  gain.gain.setValueAtTime(0.25, when);
  gain.gain.exponentialRampToValueAtTime(0.001, when + dur);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(when); osc.stop(when + dur);
}
function playRhythm(pattern, startAt) {        // pattern = gaps (s) between beeps
  let t = startAt;
  beep(t, 0.12);
  pattern.forEach((gap) => { t += gap; beep(t, 0.12); });
  return t + 0.3;
}
async function soundMatch(area, setStatus) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) { setStatus("Sound is not available in this browser - skipping this game."); await sleep(1500); return; }
  audioCtx = audioCtx || new Ctx();
  const rounds = [
    { a: [0.25, 0.25, 0.5], b: [0.25, 0.25, 0.5] },
    { a: [0.5, 0.25, 0.25], b: [0.25, 0.5, 0.25] },
    { a: [0.25, 0.5], b: [0.25, 0.5] },
    { a: [0.25, 0.25, 0.25, 0.5], b: [0.25, 0.5, 0.25, 0.25] },
  ];
  for (let q = 0; q < rounds.length; q++) {
    const r = rounds[q];
    const same = JSON.stringify(r.a) === JSON.stringify(r.b);
    setStatus(`Round ${q + 1} of ${rounds.length}`);
    area.innerHTML = `<p>Listen to two rhythms. Are they the same or different?</p>
      <button class="btn" id="playBtn">Play the sounds</button>
      <div class="options" id="opts"></div>`;
    const start = performance.now();
    $("playBtn").onclick = async () => {
      await audioCtx.resume();
      const t0 = audioCtx.currentTime + 0.1;
      playRhythm(r.b, playRhythm(r.a, t0) + 0.6);
    };
    await new Promise((resolve) => {
      ["Same", "Different"].forEach((label) => {
        const b = tileButton(label, "word");
        b.onclick = () => {
          const ok = (label === "Same") === same;
          b.classList.add(ok ? "hit" : "miss");
          record("sound_match", q + 1, 1, ok ? 1 : 0, ok ? 0 : 1, start);
          $("opts").querySelectorAll("button").forEach((x) => (x.disabled = true));
          setTimeout(resolve, 500);
        };
        $("opts").appendChild(b);
      });
    });
  }
}

/* ---------------- Game 6: Rhyme Time (phonological awareness, English) */
function speak(text) {
  if ("speechSynthesis" in window) {
    const u = new SpeechSynthesisUtterance(text); u.rate = 0.85; u.lang = "en-IN";
    speechSynthesis.cancel(); speechSynthesis.speak(u);
  }
}
async function rhymeTime(area, setStatus) {
  const rounds = [
    { word: "cat", options: ["hat", "dog", "sun"], answer: "hat" },
    { word: "bell", options: ["tree", "well", "cup"], answer: "well" },
    { word: "fish", options: ["dish", "ball", "car"], answer: "dish" },
    { word: "star", options: ["moon", "car", "pen"], answer: "car" },
  ];
  for (let q = 0; q < rounds.length; q++) {
    const r = rounds[q];
    setStatus(`Round ${q + 1} of ${rounds.length}`);
    area.innerHTML = `<p>Which word rhymes with</p><div class="target-letter">${r.word}</div>
      <button class="btn" id="hear">Hear the words</button><div class="options" id="opts"></div>`;
    $("hear").onclick = () => speak(`${r.word}. ${r.options.join(". ")}`);
    const start = performance.now();
    await new Promise((resolve) => {
      shuffle(r.options).forEach((o) => {
        const b = tileButton(o, "word");
        b.onclick = () => {
          const ok = o === r.answer;
          b.classList.add(ok ? "hit" : "miss");
          record("rhyme_time", q + 1, 1, ok ? 1 : 0, ok ? 0 : 1, start);
          $("opts").querySelectorAll("button").forEach((x) => (x.disabled = true));
          setTimeout(resolve, 500);
        };
        $("opts").appendChild(b);
      });
    });
  }
}

const GAMES = [
  { title: "Letter Hunt", text: "Find all the matching letters before time runs out.", run: letterHunt },
  { title: "Symbol Memory", text: "Remember the symbols, then tap them in the same order.", run: symbolMemory },
  { title: "Odd One Out", text: "One of these is different. Can you spot it?", run: oddOneOut },
  { title: "What Comes Next", text: "Look at the pattern and pick what comes next.", run: patternNext },
  { title: "Sound Match", text: "Listen closely to the beeps.", run: soundMatch },
  { title: "Rhyme Time", text: "Pick the word that sounds like the big word.", run: rhymeTime },
];

async function runGames(age) {
  show("screen-game");
  for (let g = 0; g < GAMES.length; g++) {
    $("progress").innerHTML = GAMES.map((_, i) =>
      `<span class="${i < g ? "done" : i === g ? "now" : ""}"></span>`).join("");
    $("gameTitle").textContent = GAMES[g].title;
    $("gameInstruction").textContent = GAMES[g].text;
    $("gameArea").innerHTML = `<button class="btn primary" id="go">Ready</button>`;
    $("gameStatus").textContent = `Game ${g + 1} of ${GAMES.length}`;
    await new Promise((r) => ($("go").onclick = r));
    await GAMES[g].run($("gameArea"), (t) => ($("gameStatus").textContent = t));
  }
  await submit(age);
}

async function submit(age) {
  show("screen-loading");
  try {
    const res = await fetch("/api/submit-game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ age, results: allResults }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Server error");
    renderReport(data);
  } catch (e) {
    $("screen-loading").innerHTML = `<h2>Could not create the report</h2>
      <p class="lead error">${e.message}. Check that "python app.py" is running, then play again.</p>`;
  }
}

function renderReport(d) {
  show("screen-report");
  if (d.mode === "demo") {
    $("demoBanner").textContent = d.demo_note;
    $("demoBanner").classList.remove("hidden");
  }
  $("riskValue").textContent = d.risk_level;
  $("riskBox").classList.add(d.risk_level);
  $("probValue").textContent = Math.round(d.probability * 100) + "%";
  $("modelInfo").textContent = d.mode === "real"
    ? `Prediction from the trained model: ${d.model_name}.`
    : "Demo estimate - no trained model loaded.";

  const weak = new Set(d.weak_skills), strong = new Set(d.strong_skills);
  $("skillBars").innerHTML = Object.entries(d.skill_scores).map(([name, v]) => `
    <div class="bar-row"><span>${name}</span>
      <div class="bar ${weak.has(name) ? "weak" : strong.has(name) ? "strong" : ""}">
        <div style="width:${Math.round(v * 100)}%"></div></div>
      <strong>${Math.round(v * 100)}%</strong></div>`).join("");
  $("weakList").innerHTML = d.weak_skills.length
    ? d.weak_skills.map((s) => `<li>${s}</li>`).join("") : "<li>None below the prototype threshold</li>";
  $("strongList").innerHTML = d.strong_skills.length
    ? d.strong_skills.map((s) => `<li>${s}</li>`).join("") : "<li>None above the prototype threshold</li>";

  const s = d.summary || {};
  const rows = [
    ["Questions answered", s.questions_answered],
    ["Correct taps", s.correct_answers],
    ["Incorrect taps", s.incorrect_answers],
    ["Total clicks", s.total_clicks],
    ["Click efficiency (correct / clicks)", s.click_efficiency],
    ["Average response time", s.avg_response_time_sec != null ? s.avg_response_time_sec + " s" : "-"],
    ["Overall accuracy (model input)", Math.round(d.features.overall_accuracy * 100) + "%"],
    ["Miss rate (model input)", Math.round(d.features.miss_rate * 100) + "%"],
  ];
  $("summaryTable").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  $("thresholdNote").textContent = `${d.threshold_note} Low < ${d.thresholds.LOW_MAX * 100}%, ` +
    `Moderate < ${d.thresholds.MODERATE_MAX * 100}%, otherwise High.`;
}

/* ---------------- Start-up */
window.addEventListener("DOMContentLoaded", async () => {
  for (let a = 6; a <= 17; a++) $("age").insertAdjacentHTML("beforeend", `<option value="${a}">${a} years</option>`);
  try {
    const h = await (await fetch("/api/health")).json();
    $("modeChip").textContent = h.mode === "real" ? `Real model: ${h.model_name}` : "Demo mode";
    $("modeChip").classList.add(h.mode);
  } catch { $("modeChip").textContent = "Server not reachable"; }
  $("startBtn").onclick = () => {
    const age = parseInt($("age").value, 10);
    if (!age) { $("startError").textContent = "Please select the child's age to continue."; return; }
    runGames(age);
  };
});
