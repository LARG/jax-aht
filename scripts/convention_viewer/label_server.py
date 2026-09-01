"""Serve the convention viewer site with convention-label saving.

Stdlib-only. Serves scripts/convention_viewer/site/ statically and accepts
POST /save_label with a JSON body {"task": ..., "key": ..., "label": ...},
writing the label into site/convention_labels_<task>.json so edits made in the
browser land directly in the repo. The page auto-detects this endpoint
(OPTIONS /save_label) and falls back to localStorage when served any other way.

Usage:
  python scripts/convention_viewer/label_server.py [--port 8000]
"""
import argparse
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_SITE_DIR = Path(__file__).resolve().parent / "site"
SITE_DIR = DEFAULT_SITE_DIR
NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


class Handler(SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        if self.path.lstrip("/") == "save_label":
            self.send_response(200)
            self.send_header("Allow", "OPTIONS, POST")
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.lstrip("/") != "save_label":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            task, key = body["task"], body["key"]
            label = str(body.get("label", "")).strip()
            if not NAME_RE.match(task) or not NAME_RE.match(key):
                raise ValueError(f"bad task/key: {task!r}/{key!r}")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self.send_error(400, str(e))
            return
        path = SITE_DIR / f"convention_labels_{task}.json"
        data = {"task": task, "labels": {}}
        if path.exists():
            data = json.loads(path.read_text())
        labels = data.setdefault("labels", {})
        if label:
            labels[key] = label
        else:
            labels.pop(key, None)
        data["labels"] = dict(sorted(labels.items()))
        path.write_text(json.dumps(data, indent=1) + "\n")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')
        print(f"saved label: {task} {key} -> {label!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR,
                    help="directory to serve (default: scripts/convention_viewer/site)")
    args = ap.parse_args()
    global SITE_DIR
    SITE_DIR = args.site_dir.resolve()
    handler = partial(Handler, directory=str(SITE_DIR))
    print(f"Serving {SITE_DIR} on http://localhost:{args.port} (labels saved to repo JSONs)")
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
