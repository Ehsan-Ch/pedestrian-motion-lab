"""Serve only the public report assets, bound to loopback."""
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote

ROOT=Path(__file__).resolve().parents[1]
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        route=unquote(urlsplit(self.path).path)
        if route=='/':route='/index.html'
        path=(ROOT/route.lstrip('/')).resolve()
        allowed=path==ROOT/'index.html' or path in [ROOT/'web'/p for p in ('style.css','app.js')] or path==ROOT/'reports/results.json' or (path.parent==ROOT/'reports/images' and path.suffix=='.jpg')
        if not allowed or not path.is_file():self.send_error(404);return
        self.path=path.relative_to(ROOT).as_posix();super().do_GET()
    def do_HEAD(self):self.send_error(405)
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-cache')
        super().end_headers()
    def log_message(self,*args):pass

def serve(port=8786):
    handler=partial(Handler,directory=str(ROOT))
    with ThreadingHTTPServer(('127.0.0.1',port),handler) as server:
        print(f'Crosswalk: http://127.0.0.1:{port}/',flush=True)
        server.serve_forever()
