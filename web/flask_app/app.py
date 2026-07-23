"""
ChemSage web app — chemsage.mdeller.com
Flask + WebSocket PTY: each browser tab gets its own chat.py subprocess in a pseudo-terminal.
xterm.js renders it in the browser exactly as the CLI looks locally.
"""
from __future__ import annotations

import fcntl
import json
import os
import pty
import queue
import select
import struct
import subprocess
import sys
import termios
import threading
import uuid
from pathlib import Path

from flask import Flask, render_template, request, session
from flask_sock import Sock

import logging
logging.basicConfig(
    filename="/tmp/chemsage_server.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger("chemsage")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent
CHAT_PY    = BASE_DIR / "chat_remote.py"
PYTHON_BIN = BASE_DIR / ".venv" / "bin" / "python"
HF_API_URL = os.environ.get("HF_SPACE_URL", "")

app  = Flask(__name__)
sock = Sock(app)
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(32))

_sessions: dict[str, tuple[int, subprocess.Popen]] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_pty_size(fd: int, rows: int, cols: int) -> None:
    """Apply terminal window size to a PTY master fd via TIOCSWINSZ."""
    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template("index.html")


@sock.route("/ws")
def terminal(ws):
    """WebSocket endpoint: proxies keystrokes → PTY, PTY output → browser.

    Architecture note: simple_websocket (used by flask_sock 0.7) maintains its
    own internal reader thread that shares wsproto state with ws.send() /
    ws.receive().  wsproto is NOT thread-safe, so calling ws.send() from a
    background thread concurrently with ws.receive() in the main loop corrupts
    the WebSocket framing state and causes the connection to drop.

    Fix: _read_pty puts PTY output into a queue rather than calling ws.send()
    directly.  The main loop drains that queue and serialises ALL WebSocket I/O
    (send + receive) on a single greenlet, eliminating the race condition.
    """
    sid = session.get("sid", str(uuid.uuid4()))

    # Browser sends initial terminal dimensions as query params so the PTY
    # starts at the correct size and the banner renders without wrapping.
    try:
        init_cols = max(20, min(500, int(request.args.get("cols", 80))))
        init_rows = max(5,  min(200, int(request.args.get("rows", 24))))
    except (TypeError, ValueError):
        init_cols, init_rows = 80, 24

    if sid not in _sessions:
        master_fd, slave_fd = pty.openpty()
        # Set correct size BEFORE spawning so Python inherits it via TIOCGWINSZ.
        _set_pty_size(master_fd, init_rows, init_cols)
        env = {**os.environ, "HF_SPACE_URL": HF_API_URL, "TERM": "xterm-256color"}
        proc = subprocess.Popen(
            [str(PYTHON_BIN), str(CHAT_PY)],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            close_fds=True, env=env,
        )
        os.close(slave_fd)
        _sessions[sid] = (master_fd, proc)
    else:
        # Reconnection to an existing session (e.g. page refresh on same cookie).
        # Update the PTY to the client's current viewport size.
        _set_pty_size(_sessions[sid][0], init_rows, init_cols)

    master_fd, proc = _sessions[sid]

    # Queue for PTY output: _read_pty writes here; main loop drains and sends.
    # This keeps all ws.send() / ws.receive() calls on the main greenlet,
    # preventing concurrent wsproto state access.
    out_q: queue.Queue[str | None] = queue.Queue()

    def _read_pty():
        """Read PTY output into queue (never calls ws.send directly)."""
        while proc.poll() is None:
            r, _, _ = select.select([master_fd], [], [], 0.05)
            if r:
                try:
                    data = os.read(master_fd, 4096)
                    out_q.put(data.decode("utf-8", errors="replace"))
                except OSError:
                    break
        out_q.put(None)  # sentinel: PTY is done

    reader = threading.Thread(target=_read_pty, daemon=True)
    reader.start()
    _log.info("session %s started, pid=%s", sid[:8], proc.pid)

    try:
        while True:
            # 1. Flush all pending PTY output to the browser before waiting
            #    for client input.  Done on this greenlet → no concurrency.
            while True:
                try:
                    chunk = out_q.get_nowait()
                except queue.Empty:
                    break
                if chunk is None:
                    _log.info("session %s: PTY EOF sentinel", sid[:8])
                    return
                try:
                    ws.send(chunk)
                except Exception as e:
                    _log.warning("session %s: ws.send() raised %s: %s", sid[:8], type(e).__name__, e)
                    raise

            # 2. Child may have exited while we were draining
            rc = proc.poll()
            if rc is not None:
                _log.info("session %s: proc exited rc=%s", sid[:8], rc)
                break

            # 3. Wait briefly for a client message (keystroke / resize)
            try:
                msg = ws.receive(timeout=0.05)
            except Exception as e:
                _log.warning("session %s: ws.receive() raised %s: %s", sid[:8], type(e).__name__, e)
                raise
            if msg is None:
                continue

            # Resize control message — update PTY dimensions, do not write to stdin.
            # Guard: json.loads("3") returns int 3, and 3.get() raises AttributeError.
            try:
                obj = json.loads(msg)
                if isinstance(obj, dict) and obj.get("type") == "resize":
                    cols = max(20, min(500, int(obj["cols"])))
                    rows = max(5,  min(200, int(obj["rows"])))
                    _set_pty_size(master_fd, rows, cols)
                    continue   # only skip PTY write for actual resize messages
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
                pass

            # Regular keystroke — write to PTY stdin.
            try:
                os.write(master_fd, msg.encode("utf-8"))
            except OSError as e:
                _log.warning("session %s: os.write() raised OSError: %s", sid[:8], e)
                break
    except Exception as e:
        _log.warning("session %s: main loop exception %s: %s", sid[:8], type(e).__name__, e)
    finally:
        _log.info("session %s: finalizing, proc rc=%s", sid[:8], proc.poll())
        _sessions.pop(sid, None)
        try:
            proc.terminate()
            os.close(master_fd)
        except OSError:
            pass
