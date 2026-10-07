/*
 * chat.js — the transcripts of the chat question as tokens: the roles of a transcript (the system line, the user's
 * turn, the findings block inside a user's or a tool's turn, the question, the assistant's turns), the assistant's
 * last answer, and a tool call. The tokenizer is the reports'. Works in the browser (window.NucleusChat) and in Node.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./reports.js'));
  else root.NucleusChat = factory(root.NucleusReports);
})(typeof self !== 'undefined' ? self : this, function (R) {
  'use strict';
  const ROLES = ['system', 'user', 'findings', 'question', 'assistant', 'tool'];
  const HEADERS = { SYSTEM: 'system', USER: 'user', ASSISTANT: 'assistant', TOOL: 'tool', FINDINGS: 'findings' };
  const BLOCK_LINES = 8; // FINDINGS and the seven lines of the block
  // the corpus: a persona, three styles of instruction (three phrasings each in training, a fourth held out), eight
  // questions answered from the findings by the rule of the report corpus, and the probes the page keeps for the
  // questions never asked and the lines never seen in training
  const PERSONA = 'You are a pathology assistant.';
  const STYLES = {
    brief: { key: 'brief', name: 'brief', lines: ['Answer briefly.', 'Be brief.', 'Reply briefly.'], held: 'Keep the answer brief.' },
    full: { key: 'full', name: 'full sentence', lines: ['Answer in a full sentence, with the reason.', 'Give a full sentence, with the reason.', 'Answer with the reason, in a full sentence.'], held: 'Explain the reason in a full sentence.' },
    agent: { key: 'agent', name: 'agent', lines: ['Call findings() to get the analyser\'s findings before you answer.', 'Use findings() to fetch the findings before answering.', 'Before you answer, call findings() for the analyser\'s findings.'], held: 'Get the findings with findings() first, then answer.' },
  };
  const inv = f => f.nests === 'present' && f.atypia === 'present' && f.contours === 'irregular' && f.stromal === 'desmoplasia';
  const susp = f => f.nests === 'present' && f.atypia === 'present' && !inv(f) && !(f.contours === 'rounded' && f.stromal === 'none');
  const cis = f => f.surface === 'atypia' || (f.nests === 'present' && f.atypia === 'present' && f.contours === 'rounded' && f.stromal === 'none');
  const benign = r => /^(Benign|Denuded)/.test(r.dx);
  const cap = s => s[0].toUpperCase() + s.slice(1);
  const QUESTIONS = [
    { key: 'dx', q: 'What is the diagnosis?', brief: (f, r) => r.dx, full: (f, r) => `The findings support the diagnosis: ${r.dx}` },
    { key: 'mp', q: 'Is muscularis propria present?', brief: f => (f.mp === 'absent' ? 'No, not identified.' : f.mp === 'present' ? 'Yes, present and not involved.' : 'Yes, present and involved.'), full: (f, r) => r.mp },
    { key: 'invasion', q: 'Is there invasion?', brief: f => (inv(f) ? 'Yes.' : susp(f) ? 'Suspicious, not established.' : 'No.'), full: f => (inv(f) ? 'Yes: the nests have irregular contours and a desmoplastic stromal reaction.' : susp(f) ? 'Suspicious: the atypical nests have irregular contours or a stromal reaction, but not both.' : f.nests === 'present' ? 'No: the nests are rounded, without a stromal reaction.' : 'No: there are no nests below the basement membrane.') },
    { key: 'cis', q: 'Is there carcinoma in situ?', brief: f => (cis(f) ? 'Yes.' : 'No.'), full: f => (f.surface === 'atypia' ? 'Yes: the surface urothelium shows full-thickness atypia.' : cis(f) ? 'Yes: atypical nests with rounded contours, carcinoma in situ involving von Brunn nests.' : 'No: the surface urothelium shows no atypia.') },
    { key: 'contours', q: 'What are the nest contours?', brief: f => (f.nests === 'present' ? `${cap(f.contours)}.` : 'There are no nests.'), full: f => (f.nests === 'present' ? `The nests below the basement membrane have ${f.contours} contours.` : 'There are no nests below the basement membrane.') },
    { key: 'surface', q: 'Is the surface urothelium normal?', brief: f => (f.surface === 'normal' ? 'Yes.' : f.surface === 'reactive' ? 'No, reactive.' : f.surface === 'atypia' ? 'No, atypical.' : 'It is denuded.'), full: f => (f.surface === 'normal' ? 'Yes: the surface urothelium is normal.' : f.surface === 'reactive' ? 'No: the surface urothelium shows reactive changes.' : f.surface === 'atypia' ? 'No: the surface urothelium shows atypia.' : 'The surface urothelium is denuded, so it cannot be assessed.') },
    { key: 'inflammation', q: 'Is there inflammation?', brief: f => (f.inflammation === 'none' ? 'No.' : `Yes, ${f.inflammation}.`), full: f => (f.inflammation === 'none' ? 'No: there is no inflammation.' : `Yes: there is ${f.inflammation} chronic inflammation in the lamina propria.`) },
    { key: 'benign', q: 'Can this be signed out as benign?', brief: (f, r) => (benign(r) ? 'Yes.' : 'No.'), full: (f, r) => (benign(r) ? `Yes: ${r.dx}` : `No: ${r.dx}`) },
  ];
  const PROBES = {
    questions: ['What is the patient\'s name?', 'Is this cancer?', 'What stage is it?', 'Is the margin clear?'], // never asked in training
    lines: ['Keep your answers short.', 'Please explain your reasoning.', 'You are a helpful assistant.'], // never seen as a system line
  };
  // the rule's answer to one of the eight questions, for the findings of a case, in a style; or null for any other question
  function answerOf(f, qi, style) { const qd = QUESTIONS[qi]; if (!qd || !f) return null; return qd[style === 'full' ? 'full' : 'brief'](f, R.ruleOf(f)); }
  // the text of a transcript's prompt, up to the assistant's first turn: the system line, then the user's turn
  function promptOf(style, line, block, q) { const sys = `SYSTEM: ${PERSONA} ${line}`; return style === 'agent' ? `${sys}\nUSER: ${q}\nASSISTANT:` : `${sys}\nUSER: Here are the findings.\n${block}\n${q}\nASSISTANT:`; }
  const blockOfText = text => { const m = text.match(/FINDINGS\n(?:[^\n]*\n){6}[^\n]*/); return m ? m[0] : null; }; // the findings block inside a transcript's text
  // the role of every token: a header at the start of a line opens its turn; a findings block inside a user's or a
  // tool's turn is 'findings' for its eight lines; what follows the block in a user's turn is the question
  function rolesOf(words) {
    let role = 'system', turn = 'system', lineStart = true, blockLeft = 0; const out = [];
    for (const w of words) {
      if (w === R.START) { out.push('system'); continue; }
      if (lineStart) {
        if (w === 'FINDINGS' && (turn === 'user' || turn === 'tool')) { role = 'findings'; blockLeft = BLOCK_LINES; }
        else if (HEADERS[w] && w !== 'FINDINGS') { turn = HEADERS[w]; role = turn; blockLeft = 0; }
        else if (blockLeft > 0) role = 'findings';
        else role = turn === 'user' ? 'question' : turn;
      }
      out.push(role);
      if (w === R.NL) { lineStart = true; if (blockLeft > 0) blockLeft--; } else lineStart = false;
    }
    return out;
  }
  // the assistant's last line: where it starts, its words and its text; or null
  function lastAssistant(words) {
    let i = -1; for (let k = 0; k + 1 < words.length; k++) if (words[k] === 'ASSISTANT' && words[k + 1] === ':' && (k === 0 || words[k - 1] === R.NL || words[k - 1] === R.START)) i = k;
    if (i < 0) return null;
    const out = []; for (let k = i + 2; k < words.length; k++) { const w = words[k]; if (w === R.END || w === R.NL) break; out.push(w); }
    return { at: i + 2, words: out, text: R.detokenize(out) };
  }
  const CALL = /^findings\s*\(\s*\)\s*$/;
  function callOf(text) { return text && CALL.test(text) ? 'findings' : null; } // the one tool: findings()
  function firstAnswerIndex(words) { const a = lastAssistant(words); return a && a.words.length ? a.at : words.length - 1; } // the word the diagrams follow by default
  const classOf = text => (text ? (text.split(/[ :,.]/)[0] || '') : '');
  const same = (a, b) => R.detokenize(R.tokenize(a || '')) === R.detokenize(R.tokenize(b || '')); // the same answer, spaced as the tokenizer spaces it
  // the style the assistant answered in, read off the transcript: agent if it wrote the tool call, full if the last answer runs to a sentence, else brief
  function styleOf(words) { const lines = assistantLines(words); if (lines.some(l => callOf(l.text))) return 'agent'; const last = lines[lines.length - 1]; return last && last.words.length > 3 ? 'full' : 'brief'; }
  function assistantLines(words) { // every assistant line: where it starts, its words and its text
    const out = []; for (let k = 0; k + 1 < words.length; k++) if (words[k] === 'ASSISTANT' && words[k + 1] === ':' && (k === 0 || words[k - 1] === R.NL || words[k - 1] === R.START)) { const ws = []; for (let j = k + 2; j < words.length; j++) { const w = words[j]; if (w === R.END || w === R.NL) break; ws.push(w); } out.push({ at: k + 2, words: ws, text: R.detokenize(ws) }); }
    return out;
  }
  return { ROLES, HEADERS, BLOCK_LINES, PERSONA, STYLES, QUESTIONS, PROBES, rolesOf, lastAssistant, assistantLines, callOf, firstAnswerIndex, classOf, same, styleOf, answerOf, promptOf, blockOfText };
});
