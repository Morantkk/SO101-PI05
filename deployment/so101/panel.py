"""Deployment status, original receiver previews, and a stop-only motion latch."""

import json
import time
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        if self.path != "/stop-motion":
            self.send_error(404)
            return
        if self.headers.get("Origin") not in (None, "http://127.0.0.1:8766", "http://localhost:8766"):
            self.send_error(403)
            return
        (ROOT / "STOP_MOTION").touch()
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def stream_camera(self, camera):
        """Relay the original receiver's MJPEG stream without buffering frames."""
        if camera not in ("mid", "left", "right"):
            self.send_error(404)
            return
        try:
            upstream = urlopen(
                f"http://127.0.0.1:5082/api/v1/preview?cam={camera}&fps=30", timeout=5
            )
        except OSError:
            self.send_error(503, "Waiting for camera")
            return
        with upstream:
            self.send_response(200)
            self.send_header("Content-Type", upstream.headers["Content-Type"])
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while chunk := upstream.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except OSError:
                # A closed browser tab or stopped receiver ends this stream.
                pass

    def do_GET(self):
        if self.path.startswith("/stream/"):
            self.stream_camera(self.path.split("?")[0].rsplit("/", 1)[-1])
            return
        if self.path == "/":
            body = (ROOT / "panel.html").read_bytes()
            mime = "text/html; charset=utf-8"
        elif self.path == "/status":
            payload = {"now": time.time()}
            for key, name in [
                ("config", "config.json"),
                ("inference", "inference_status.json"),
                ("state", "state_status.json"),
                ("action", "received_action.json"),
                ("execution", "execution_status.json"),
            ]:
                try:
                    payload[key] = json.loads((ROOT / name).read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    payload[key] = None
            payload["camera_devices"] = {
                key: str(Path(path).resolve())
                for key, path in (payload.get("config") or {}).get("cameras", {}).items()
            }
            body = json.dumps(payload).encode()
            mime = "application/json"
        elif self.path.startswith("/camera/"):
            camera = self.path.split("?")[0].rsplit("/", 1)[-1]
            if camera not in ("mid", "left", "right"):
                self.send_error(404)
                return
            try:
                with urlopen(f"http://127.0.0.1:5082/api/v1/preview.jpg?cam={camera}", timeout=2) as response:
                    body = response.read()
            except OSError:
                self.send_error(503, "Waiting for camera")
                return
            mime = "image/jpeg"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8766), Handler).serve_forever()
