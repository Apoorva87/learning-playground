/* Tiny static file server for the browser tests. serve(rootDir) -> { url, close }. */
const http = require('node:http'), fs = require('node:fs'), path = require('node:path');
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.cjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.md': 'text/markdown' };
function serve(root, prefix = '') {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (prefix && p.startsWith(prefix)) p = p.slice(prefix.length) || '/';
    else if (prefix) { res.writeHead(404); return res.end('outside prefix'); }
    const file = path.join(root, path.normalize(p));
    if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
    fs.stat(file, (e, st) => {
      const target = !e && st.isDirectory() ? path.join(file, 'index.html') : file;
      fs.readFile(target, (err, buf) => {
        if (err) { res.writeHead(404); return res.end('not found'); }
        res.writeHead(200, { 'content-type': TYPES[path.extname(target)] || 'application/octet-stream' });
        res.end(buf);
      });
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () =>
    resolve({ url: `http://127.0.0.1:${server.address().port}${prefix}`, close: () => new Promise(r => server.close(r)) })));
}
module.exports = { serve };
