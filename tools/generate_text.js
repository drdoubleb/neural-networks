#!/usr/bin/env node
/*
 * generate_text.js — the default text of the Your-text question from a Project Gutenberg plain-text file: a book's
 * poems between two headings, or every book of the volume, written as a script the page loads (window.LECTURE_TEXT,
 * or window.LECTURE_TEXT_ALL for the on-demand volume), with what it is and where it came from.
 *   node tools/generate_text.js pg78327.txt --from "MOUNTAIN INTERVAL" --to "NEW HAMPSHIRE" --title "Mountain Interval" --year 1916 --out data/text/mountain_interval.js
 *   node tools/generate_text.js pg78327.txt --all --title "Collected Poems" --year 1930 --global LECTURE_TEXT_ALL --out data/text/frost_collected.js
 * The Gutenberg header, the contents and the licence are left out; the text is kept as it is otherwise (the page
 * normalises the quotes and the italics marks when it cuts it into documents).
 */
'use strict';
const fs = require('fs'), path = require('path');
const T = require(path.join(__dirname, '..', 'js', 'text.js'));
const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d), has = k => process.argv.includes(k);
const file = process.argv[2]; if (!file || !fs.existsSync(file)) { console.error('usage: node tools/generate_text.js <gutenberg.txt> [--from HEADING --to HEADING | --all] --title T --year Y [--global NAME] --out FILE'); process.exit(2); }
const lines = fs.readFileSync(file, 'utf8').replace(/\r\n?/g, '\n').split('\n');
const ebook = (fs.readFileSync(file, 'utf8').match(/\[eBook #(\d+)\]/) || [])[1] || null;
const start = lines.findIndex(l => /^\*\*\* START OF THE PROJECT GUTENBERG EBOOK/.test(l)), end = lines.findIndex(l => /^\*\*\* END OF THE PROJECT GUTENBERG EBOOK/.test(l));
const body = lines.slice(start + 1, end < 0 ? lines.length : end);
let from, to;
const heading = h => { const want = h.replace(/[\u2018\u2019']/g, "'"); let at = -1; body.forEach((l, i) => { if (l.trim().replace(/[\u2018\u2019']/g, "'") === want) at = i; }); return at; }; // a heading's last occurrence: the first is in the contents
if (has('--all')) { from = heading("A BOY'S WILL") + 1; to = body.length; } // from the first book's heading to the end
else { from = heading(arg('--from')); to = heading(arg('--to')); if (from < 0 || to < 0) { console.error('headings not found'); process.exit(2); } from++; }
const text = body.slice(from, to).join('\n').replace(/\n{3,}/g, '\n\n').replace(/^\n+|\n+$/g, '');
const st = T.stats(text), title = arg('--title', 'the text'), year = +arg('--year', 0) || null, global = arg('--global', 'LECTURE_TEXT');
const meta = { title, author: 'Robert Frost', year, source: `Project Gutenberg${ebook ? ` eBook #${ebook}` : ''}, Collected Poems of Robert Frost (Henry Holt, 1930), a text in the public domain in the United States`, url: ebook ? `https://www.gutenberg.org/ebooks/${ebook}` : null, documents: st.documents, poems: st.documents - (text.match(/\(\d+\)/g) || []).length, words: st.words, tokens: st.tokens, vocabulary: st.vocabulary };
const out = `// ${title} by Robert Frost (${year}): the default text of the Your-text question, from ${meta.source}, prepared by tools/generate_text.js. The page cuts it into documents itself.\nwindow.${global} = ${JSON.stringify({ meta, text })};\n`;
fs.writeFileSync(arg('--out'), out, 'utf8');
console.log(`${arg('--out')}: ${(out.length / 1024).toFixed(0)} kB · ${JSON.stringify(st)}`);
