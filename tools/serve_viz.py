#!/usr/bin/env python
"""Serve the test-result explorer on localhost.

Renders, per run, what the model did on the held-out test split: which events
flared up (flagged above the val-tuned threshold), which ground-truth
incidents were caught or missed, and the provenance chain inside each incident.

Standard library only — no extra dependencies.

Usage:
    python tools/export_predictions.py runs/<run_dir>     # once per run
    python tools/serve_viz.py [--port 8123] [--runs-dir runs]

Then open http://localhost:8123 and pick a run (or /#<run_name> to deep-link).
The server binds to 127.0.0.1 by default; pass --host 0.0.0.0 only if you
deliberately want to expose predictions on the network.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(_HERE, "viz", "index.html")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def scan_runs(runs_dir: str):
    """Runs that have an exported viz_data.json, newest first."""
    out = []
    if not os.path.isdir(runs_dir):
        return out
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name, "viz_data.json")
        if not RUN_NAME_RE.match(name) or not os.path.isfile(path):
            continue
        st = os.stat(path)
        entry = {"name": name, "exported_at": st.st_mtime, "bytes": st.st_size}
        try:  # meta only — cheap head read, tolerate truncated json otherwise
            with open(path, encoding="utf-8") as f:
                head = f.read(4096)
            m = re.search(r'"n_events":\s*(\d+)', head)
            if m:
                entry["n_events"] = int(m.group(1))
            p = re.search(r'"n_pos":\s*(\d+)', head)
            if p:
                entry["n_pos"] = int(p.group(1))
        except OSError:
            pass
        out.append(entry)
    out.sort(key=lambda e: -e["exported_at"])
    return out


def make_handler(runs_dir: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BiRGATViz/1.0"
        # HTTP/1.1 keep-alive: every response carries an exact Content-Length,
        # so this is safe and avoids a new connection per dashboard fetch
        protocol_version = "HTTP/1.1"

        # ------------------------------------------------------- responses
        def _send(self, code: int, body: bytes, ctype: str,
                  cache: str = "no-cache") -> None:
            # gzip only when we ACTUALLY compress: claiming the encoding for
            # an uncompressed body makes browsers fail to decode the response
            compressed = False
            if "gzip" in (self.headers.get("Accept-Encoding") or "") \
                    and len(body) > 1024:
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
                    gz.write(body)
                body = buf.getvalue()
                compressed = True
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _not_found(self, msg: str) -> None:
            self._json({"error": msg}, 404)

        # ------------------------------------------------------- routing
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    with open(INDEX_HTML, "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                elif path == "/api/runs":
                    self._json({"runs_dir": os.path.abspath(runs_dir),
                                "runs": scan_runs(runs_dir)})
                elif path.startswith("/api/data/"):
                    name = path[len("/api/data/"):]
                    if not RUN_NAME_RE.match(name):
                        return self._not_found("invalid run name")
                    fpath = os.path.join(runs_dir, name, "viz_data.json")
                    runs_root_real = os.path.realpath(runs_dir)
                    fpath_real = os.path.realpath(fpath)
                    if os.path.commonpath([runs_root_real, fpath_real]) != runs_root_real:
                        return self._not_found("invalid run name")
                    if not os.path.isfile(fpath_real):
                        return self._not_found(
                            f"no viz_data.json for '{name}' — run "
                            f"tools/export_predictions.py on it first")
                    with open(fpath_real, "rb") as f:
                        self._send(200, f.read(),
                                   "application/json; charset=utf-8")
                else:
                    self._not_found(f"unknown path {path}")
            except (OSError, BrokenPipeError) as e:
                try:
                    self._json({"error": str(e)}, 500)
                except OSError:
                    pass

        def log_message(self, fmt: str, *args) -> None:
            stamp = _dt.datetime.now().strftime("%H:%M:%S")
            print(f"[viz] {stamp} {self.address_string()} {fmt % args}")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Serve the BiRGAT test-result explorer on localhost.")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1 — localhost only)")
    ap.add_argument("--runs-dir", default="runs",
                    help="directory containing runs/<...>/viz_data.json")
    args = ap.parse_args()

    if not os.path.isdir(args.runs_dir):
        raise SystemExit(f"runs dir not found: {args.runs_dir}")
    available = scan_runs(args.runs_dir)
    if not available:
        print(f"[viz] no runs with viz_data.json under '{args.runs_dir}' yet.")
        print("[viz] export one first, e.g.:")
        print("      python tools/export_predictions.py runs/<your_run>")
        # still serve: the UI explains this and /api/runs updates live.
    else:
        print(f"[viz] {len(available)} run(s) with exported predictions:")
        for e in available:
            print(f"      - {e['name']}  ({e.get('n_events', '?')} test events)")

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(args.runs_dir))
    print(f"[viz] serving on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
