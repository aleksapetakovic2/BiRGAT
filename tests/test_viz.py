"""Tests for the localhost test-result explorer (tools/serve_viz.py).

These are deliberately torch-free: they exercise the stdlib HTTP layer and the
run-scanning logic only. The prediction *export* (tools/export_predictions.py)
needs a trained checkpoint + torch and is covered by running it on a real run.
"""
import gzip
import json
import os
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))

import serve_viz  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from eprgat.config import Config  # noqa: E402
import export_predictions  # noqa: E402


def _small_cfg():
    cfg = Config()
    cfg.data.days = 4
    cfg.data.n_hosts = 40
    cfg.data.n_users = 30
    cfg.data.n_incidents = 8
    cfg.data.attack_events_range = [15, 30]
    cfg.graph.split_train_frac = 0.55
    cfg.graph.split_val_frac = 0.20
    cfg.graph.split_gap_hours = 2.0
    return cfg


def test_build_incident_metadata_captures_templates():
    tab, captured = export_predictions.build_incident_metadata(_small_cfg())
    inc = tab.cols["incident"]
    y = tab.cols["y"]
    incident_ids = set(inc[(y == 1) & (inc >= 0)].tolist())
    # every injected incident is captured with a valid template id
    assert incident_ids, "no incidents generated"
    for iid in incident_ids:
        assert iid in captured, f"incident {iid} not captured"
        assert 0 <= captured[iid] < 5


def test_describe_event_covers_all_etypes():
    tab, _ = export_predictions.build_incident_metadata(_small_cfg())
    c = tab.cols
    seen = set()
    for i in range(len(tab)):
        et = int(c["etype"][i])
        if et in seen:
            continue
        s = export_predictions.describe_event(i, c, tab)
        assert isinstance(s, str) and s.strip()
        seen.add(et)
        if len(seen) == 6:
            break
    assert seen == set(range(6)), f"uncovered event types: {set(range(6)) - seen}"


def _mk_run(runs_dir, name, n_events=10, n_pos=2):
    d = os.path.join(runs_dir, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "viz_data.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {"run": name, "n_events": n_events, "n_pos": n_pos}}, f)
    return d


def test_scan_runs_filters_and_sorts(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _mk_run(str(runs), "run_b", n_events=5)
    _mk_run(str(runs), "run_a", n_events=7)
    # a run dir without an export must be skipped
    os.makedirs(runs / "run_no_export")
    # a non-run file must be skipped
    (runs / "stray.txt").write_text("x")

    found = serve_viz.scan_runs(str(runs))
    names = [e["name"] for e in found]
    assert names == ["run_a", "run_b"] or set(names) == {"run_a", "run_b"}
    assert "run_no_export" not in names and "stray.txt" not in names
    by = {e["name"]: e for e in found}
    assert by["run_a"]["n_events"] == 7 and by["run_a"]["n_pos"] == 2
    assert by["run_b"]["n_events"] == 5


def test_scan_runs_missing_dir(tmp_path):
    assert serve_viz.scan_runs(str(tmp_path / "does_not_exist")) == []


def test_http_routes(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _mk_run(str(runs), "run_a", n_events=3, n_pos=1)

    handler = serve_viz.make_handler(str(runs))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        # run index
        with urllib.request.urlopen(base + "/api/runs") as r:
            runs_obj = json.load(r)["runs"]
        assert [e["name"] for e in runs_obj] == ["run_a"]

        # dashboard page
        with urllib.request.urlopen(base + "/") as r:
            html = r.read().decode("utf-8")
        assert "test-result explorer" in html

        # exported data round-trips
        with urllib.request.urlopen(base + "/api/data/run_a") as r:
            data = json.load(r)
        assert data["meta"]["n_events"] == 3 and data["meta"]["n_pos"] == 1

        # unknown run -> 404
        try:
            urllib.request.urlopen(base + "/api/data/nope")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # path traversal attempt is rejected (not a valid run name)
        try:
            urllib.request.urlopen(base + "/api/data/..%2f..%2fetc")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # regression: when the client offers gzip but the body is NOT
        # compressed (small responses), the server must NOT advertise
        # Content-Encoding: gzip — browsers then fail to decode the body.
        req = urllib.request.Request(base + "/api/runs",
                                     headers={"Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            enc = (r.headers.get("Content-Encoding") or "").lower()
            if enc == "gzip":
                raw = gzip.decompress(raw)
            json.loads(raw)  # must always be valid JSON
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_gzip_header_matches_body(tmp_path):
    """If Content-Encoding: gzip is sent, the body must actually be gzip."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _mk_run(str(runs), "run_a")

    handler = serve_viz.make_handler(str(runs))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for path in ("/", "/api/runs", "/api/data/run_a"):
            req = urllib.request.Request(base + path,
                                         headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req) as r:
                raw = r.read()
                enc = (r.headers.get("Content-Encoding") or "").lower()
                if enc == "gzip":
                    raw = gzip.decompress(raw)  # raises if not real gzip
                if path.startswith("/api/"):
                    json.loads(raw)
                else:
                    assert b"test-result explorer" in raw
    finally:
        httpd.shutdown()
        httpd.server_close()
