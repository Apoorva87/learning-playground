#!/usr/bin/env node
/* node check-links.cjs [--tracked] page.html...  Every relative href/src must resolve to a file,
   and every #fragment to an id in the target page. --tracked also requires git to track the file. */
const fs = require('node:fs'), path = require('node:path'), { spawnSync } = require('node:child_process');

const ids = file => { try { return new Set([...fs.readFileSync(file, 'utf8').matchAll(/\b(?:id|name)="([^"]+)"/g)].map(m => m[1])); } catch { return new Set(); } };
function checkLinks(files, { tracked = false } = {}) {
  const errors = [];
  for (const file of files) {
    const html = fs.readFileSync(file, 'utf8');
    for (const [, , url] of html.matchAll(/\b(href|src)="([^"]*)"/g)) {
      if (!url || /^(https?:|mailto:|data:|javascript:|\/\/)/.test(url)) continue;
      const [rest, frag] = url.split('#'), target = rest ? path.resolve(path.dirname(file), decodeURIComponent(rest.split('?')[0])) : file;
      if (!fs.existsSync(target)) { errors.push(`${path.basename(file)}: ${url} does not resolve`); continue; }
      if (tracked && rest && spawnSync('git', ['ls-files', '--error-unmatch', target], { cwd: path.dirname(target), stdio: 'ignore' }).status !== 0)
        errors.push(`${path.basename(file)}: ${url} is not tracked by git`);
      if (frag && /\.html?$/.test(target) && !ids(target).has(frag)) errors.push(`${path.basename(file)}: ${url} has no #${frag} in ${path.basename(target)}`);
    }
  }
  return errors;
}
if (require.main === module) {
  const args = process.argv.slice(2), tracked = args.includes('--tracked');
  const errs = checkLinks(args.filter(a => a !== '--tracked').map(f => path.resolve(f)), { tracked });
  errs.forEach(e => console.log('ERROR: ' + e));
  if (errs.length) process.exit(1);
  console.log('links ok');
}
module.exports = { checkLinks };
