"""Simple web UI backend for error detection."""

import json
import os
import warnings
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.pipeline import build_classifier, check_text

PORT = 8811

# --------------- HTTP Server ---------------

classifier = None  # set at startup

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), "static"), **kwargs)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/check":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body)
                text = data.get("text", "")
                if len(text) > 5000:
                    self._json_response({"error": "Text too long (max 5000 chars)"}, 400)
                    return
                result = check_text(text, classifier)
                self._json_response(result)
            except json.JSONDecodeError:
                self._json_response({"error": "Invalid JSON"}, 400)
            except Exception as e:
                print(f"Error: {e}")
                self._json_response({"error": str(e)}, 500)
        else:
            self._json_response({"error": "Not found"}, 404)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Only log errors, skip GET noise
        if args and "200" not in str(args[1:2]):
            super().log_message(format, *args)


def main():
    global classifier
    print("=== Building classifier ===")
    classifier, *_ = build_classifier()
    print(f"\n=== Server ready on http://localhost:{PORT} ===")
    server = HTTPServer(("", PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
