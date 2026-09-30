#!/usr/bin/env node
/*
 * build_single_file.js — inlines the stylesheet, scripts and dataset into one portable HTML file.
 *   node tools/build_single_file.js            -> dist/nucleus-net.html  (open anywhere, no server needed)
 *   node tools/build_single_file.js --fragment -> prints the same page without the <html>/<head>/<body> wrapper
 *   --no-fields                                 -> without the fields of the invasion question inlined (loaded on demand instead)
 */
'use strict';
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
let html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
html = html.replace(/<link rel="stylesheet" href="css\/style.css">/, () => `<style>\n${fs.readFileSync(path.join(root, 'css/style.css'), 'utf8')}\n</style>`);
// the fields of the invasion question are loaded on demand by the page (5.7 MB); a single file has no folder to load
// them from, so they go in unless --no-fields (the artifact leaves them out and fetches them from the published page)
if (!process.argv.includes('--no-fields')) html = html.replace(/<script src="js\/features.js"><\/script>/, m => `<script src="data/fields/fields_data.js"></script>\n${m}`);
html = html.replace(/<script src="([^"]+)"><\/script>/g, (_, src) => `<script>\n${fs.readFileSync(path.join(root, src), 'utf8').replace(/<\/script/gi, '<\\/script')}\n</script>`);
if (process.argv.includes('--fragment')) {
  let frag = html.replace(/^[\s\S]*?<head>/, '').replace(/<\/head>\s*<body>/, '').replace(/<\/body>\s*<\/html>\s*$/, '');
  frag = frag.replace(/<meta charset="utf-8">\s*/, '').replace(/<meta name="viewport"[^>]*>\s*/, '');
  process.stdout.write(frag);
} else {
  fs.mkdirSync(path.join(root, 'dist'), { recursive: true });
  const out = path.join(root, 'dist', 'nucleus-net.html');
  fs.writeFileSync(out, html);
  console.log(`wrote ${out} (${(fs.statSync(out).size / 1024).toFixed(0)} KB)`);
}
