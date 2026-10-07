#!/usr/bin/env node
/*
 * generate_chat.js — the transcripts of the chat question, written from the report corpus's cases. A transcript is a
 * system line (a persona and an instruction, in one of several phrasings), a user's turn and the assistant's turn,
 * ended by the end token; the model trains on all of it and, on the page, continues it. Three styles: brief (the
 * findings in the user's turn, a question, a short answer), full (the same, answered in a full sentence with the
 * reason), and agent (the question alone; the assistant has to write findings(), a tool call, and gets the findings
 * back in a TOOL turn before it answers). Eight questions, each answered from the findings by the rule of the report
 * corpus. One phrasing of every style's instruction is held out of training, to ask whether the instruction's words
 * carry any meaning beyond the transcripts they appeared in. Probe questions and lines the model never trained on
 * are kept for the page.
 *   node tools/generate_chat.js
 * writes data/chat/chat_data.js (train, test and rephrased transcripts) and data/chat/sample_transcripts.md.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const R = require('../js/reports.js');
const SEED = 20261007, OUT = path.join(__dirname, '..', 'data', 'chat');
function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rng = mulberry32(SEED), pickIndex = n => Math.floor(rng() * n);
const load = f => { const w = {}; new Function('window', fs.readFileSync(f, 'utf8'))(w); return w; };
const D = load(path.join(__dirname, '..', 'data', 'reports', 'reports_data.js')).LECTURE_REPORTS;

const C = require('../js/chat.js');
const { PERSONA, STYLES, QUESTIONS, PROBES } = C; // the persona, the styles, the questions with their answers from the findings, and the probes live in js/chat.js, for the page
const blockOf = rep => rep.text.split('\n').slice(0, 8).join('\n');
function transcript(rep, styleKey, line, qi) {
  const f = R.findingsOf(R.tokenize(rep.text)), r = R.ruleOf(f), style = STYLES[styleKey], qd = QUESTIONS[qi];
  const sys = `SYSTEM: ${PERSONA} ${line === 'held' ? style.held : style.lines[line]}`, answer = qd[styleKey === 'agent' ? 'brief' : styleKey](f, r), block = blockOf(rep);
  const text = styleKey === 'agent' ? `${sys}\nUSER: ${qd.q}\nASSISTANT: findings()\nTOOL:\n${block}\nASSISTANT: ${answer}` : `${sys}\nUSER: Here are the findings.\n${block}\n${qd.q}\nASSISTANT: ${answer}`;
  return { case: rep.id, dx: rep.dx, style: styleKey, line, qi, q: qd.q, answer, text };
}
const styleKeys = Object.keys(STYLES);
const make = (reps, prefix, lineOf) => { const out = []; let n = 0; for (const rep of reps) for (const sk of styleKeys) { const t = transcript(rep, sk, lineOf(sk), pickIndex(QUESTIONS.length)); t.id = `${prefix}${String(++n).padStart(4, '0')}`; t.name = t.id; out.push(t); } return out; };
const train = make(D.train, 'C', sk => pickIndex(STYLES[sk].lines.length)), test = make(D.test, 'T', sk => pickIndex(STYLES[sk].lines.length)), rephrased = make(D.test, 'H', () => 'held');
for (const t of train) t.split = 'train'; for (const t of test) t.split = 'test'; for (const t of rephrased) t.split = 'rephrased';
const meta = {
  question: 'Chat: ask the model about a case?',
  blurb: 'A chatbot is the next-word model trained on transcripts. Every transcript here is a system line (a persona and an instruction), the user\'s turn and the assistant\'s turn, and the model learns to continue it. In the brief and the full-sentence styles the user\'s turn holds the findings and a question; in the agent style it holds the question alone, and the assistant has to write findings(), a tool call, and gets the findings back in a TOOL turn before it answers. The eight questions are answered from the findings by the rule of the report corpus. The instruction comes in three phrasings per style, and a fourth is held out of training.',
  persona: PERSONA, styles: STYLES, questions: QUESTIONS.map(q => ({ key: q.key, q: q.q })), probes: PROBES,
  counts: { train: train.length, test: test.length, rephrased: rephrased.length },
};
fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, 'chat_data.js'), `window.LECTURE_CHAT = ${JSON.stringify({ meta, train, test, rephrased })};\n`);
// the sheet for vetting: every question with both answers for two cases, and one transcript of every style
const md = ['# Sample transcripts of the chat question', '', `${train.length} training transcripts from the ${D.train.length} training reports (one of each style per report), ${test.length} test transcripts on the trained phrasings and ${rephrased.length} on the held-out phrasings. The persona is "${PERSONA}"; the instructions:`, ''];
for (const sk of styleKeys) md.push(`- **${STYLES[sk].name}**: ${STYLES[sk].lines.map(l => `"${l}"`).join(', ')}; held out of training: "${STYLES[sk].held}"`);
md.push('', `Probe questions never asked in training: ${PROBES.questions.map(q => `"${q}"`).join(', ')}. Probe lines never seen: ${PROBES.lines.map(q => `"${q}"`).join(', ')}.`, '');
const sampleCases = [D.train.find(r => /invasive/.test(r.dx) && !/muscularis/.test(r.dx)), D.train.find(r => /^Benign urothelium\./.test(r.dx)), D.train.find(r => /suspicious/.test(r.dx)), D.train.find(r => /in situ\./.test(r.dx) && !/invasive/.test(r.dx))].filter(Boolean);
for (const rep of sampleCases) { const f = R.findingsOf(R.tokenize(rep.text)), r = R.ruleOf(f); md.push(`## ${rep.id} · ${rep.dx}`, '', '```', blockOf(rep), '```', '', '| question | brief | full sentence |', '|---|---|---|'); for (const qd of QUESTIONS) md.push(`| ${qd.q} | ${qd.brief(f, r)} | ${qd.full(f, r)} |`); md.push(''); }
md.push('## One transcript of each style', '');
for (const sk of styleKeys) { const t = train.find(x => x.style === sk); md.push(`### ${STYLES[sk].name} (${t.id}, from ${t.case})`, '', '```', t.text, '```', ''); }
fs.writeFileSync(path.join(OUT, 'sample_transcripts.md'), md.join('\n'));
const vocab = R.buildVocab(train.map(t => t.text)), lens = train.map(t => R.encode(vocab, t.text).length);
console.log(`wrote ${path.join(OUT, 'chat_data.js')} (${(fs.statSync(path.join(OUT, 'chat_data.js')).size / 1024).toFixed(0)} KB) and sample_transcripts.md · ${train.length} training, ${test.length} test, ${rephrased.length} rephrased · vocabulary ${vocab.size} · ${(lens.reduce((a, b) => a + b, 0) / lens.length).toFixed(0)} tokens per transcript (${Math.min(...lens)}–${Math.max(...lens)})`);
