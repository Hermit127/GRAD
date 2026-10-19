"""Local HTTP scoring service for the GRAD gate (run with the mcp-exp venv).

  /score   POST {"nodes": [...]} -> {"p": float, "block": bool, "threshold": float}
  /threshold POST {"threshold": float} -> {"ok": true}
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code/repro/v10'))

from gnn_gate import GnnGate  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        req = json.loads(self.rfile.read(n) or b'{}')
        gate = GnnGate.get()
        if self.path == '/score':
            p, lat = gate.score_nodes(req['nodes'])
            self._send({'p': p, 'block': p >= gate.threshold,
                        'threshold': gate.threshold, 'latency_ms': round(lat, 2)})
        elif self.path == '/threshold':
            gate.threshold = float(req['threshold'])
            self._send({'ok': True, 'threshold': gate.threshold})
        else:
            self._send({'error': 'unknown path'}, 404)

    def log_message(self, *a):  # silence
        pass


if __name__ == '__main__':
    port = int(os.environ.get('GATE_PORT', '8765'))
    print(f'gate server on 127.0.0.1:{port} threshold={GnnGate.get().threshold}', flush=True)
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()
