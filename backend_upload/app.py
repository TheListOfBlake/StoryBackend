import json
import hashlib
import hmac
import os
import secrets
import sqlite3
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from vosk import KaldiRecognizer, Model

MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
FRONTEND_ORIGINS = os.environ.get("FRONTEND_ORIGINS", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
PBKDF2_ROUNDS = int(os.environ.get("PBKDF2_ROUNDS", "200000"))
DB_PATH = os.environ.get("AUTH_DB_PATH", str(Path(__file__).resolve().parent / "data" / "app.db"))

app = FastAPI()

origins = [o.strip() for o in FRONTEND_ORIGINS.split(",") if o.strip()]
if not origins:
  origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
  ]


def parse_bool_env(name: str, default: bool) -> bool:
  raw = os.environ.get(name)
  if raw is None:
    return default
  value = raw.strip().lower()
  if value in {"1", "true", "yes", "on"}:
    return True
  if value in {"0", "false", "no", "off"}:
    return False
  return default


def is_local_origin(origin: str) -> bool:
  return origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1")


SESSION_HTTPS_ONLY_DEFAULT = not all(is_local_origin(origin) for origin in origins)
SESSION_HTTPS_ONLY = parse_bool_env("SESSION_HTTPS_ONLY", SESSION_HTTPS_ONLY_DEFAULT)
SESSION_SAME_SITE = os.environ.get("SESSION_SAME_SITE", "lax").strip().lower() or "lax"
if SESSION_SAME_SITE not in {"lax", "strict", "none"}:
  SESSION_SAME_SITE = "lax"

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["GET", "POST", "OPTIONS"],
  allow_headers=["*"],
)

if not SESSION_SECRET:
  # Dev fallback only. Set SESSION_SECRET in production.
  SESSION_SECRET = secrets.token_urlsafe(32)

app.add_middleware(
  SessionMiddleware,
  secret_key=SESSION_SECRET,
  https_only=SESSION_HTTPS_ONLY,
  same_site=SESSION_SAME_SITE,
)

_model = None


class AuthRequest(BaseModel):
  username: str
  password: str


class ContactCreateRequest(BaseModel):
  email: str
  comment: str


class ContactApproveRequest(BaseModel):
  approved: bool


def get_db_conn() -> sqlite3.Connection:
  db_path = Path(DB_PATH)
  db_path.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(str(db_path))
  conn.row_factory = sqlite3.Row
  return conn


def init_db() -> None:
  with get_db_conn() as conn:
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS contact_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        comment TEXT NOT NULL,
        approved INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.commit()


def hash_password(password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
  if not password:
    raise ValueError("Password is required")
  salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
  digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)
  return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
  _, computed = hash_password(password, salt_hex)
  return hmac.compare_digest(computed, expected_hash_hex)


def count_users() -> int:
  with get_db_conn() as conn:
    row = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
    return int(row["count"]) if row else 0


def get_user(username: str):
  with get_db_conn() as conn:
    return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def create_user(username: str, password: str) -> None:
  clean_username = (username or "").strip()
  if len(clean_username) < 3:
    raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
  if len(password or "") < 8:
    raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
  if get_user(clean_username):
    raise HTTPException(status_code=409, detail="Username already exists")

  salt, password_hash = hash_password(password)
  with get_db_conn() as conn:
    conn.execute(
      "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
      (clean_username, password_hash, salt),
    )
    conn.commit()


def require_admin(request: Request) -> str:
  username = request.session.get("username")
  if not username:
    raise HTTPException(status_code=401, detail="Not authenticated")
  return str(username)


@app.on_event("startup")
def startup() -> None:
  init_db()


@app.get("/auth/me")
async def auth_me(request: Request):
  username = request.session.get("username")
  if not username:
    return {"authenticated": False}
  return {"authenticated": True, "username": username}


@app.get("/auth/status")
async def auth_status(request: Request):
  username = request.session.get("username")
  return {
    "setup_required": count_users() == 0,
    "authenticated": bool(username),
    "username": username or "",
  }


@app.post("/auth/setup-admin")
async def auth_setup_admin(payload: AuthRequest, request: Request):
  # Initial bootstrap is open only when there are no users.
  # After bootstrap, only an authenticated admin can add another admin.
  existing_users = count_users()
  current_user = request.session.get("username")
  if existing_users > 0 and not current_user:
    raise HTTPException(status_code=403, detail="Admin already initialized")

  create_user(payload.username, payload.password)
  request.session["username"] = payload.username.strip()
  return {"ok": True, "username": payload.username.strip()}


@app.post("/auth/login")
async def auth_login(payload: AuthRequest, request: Request):
  user = get_user((payload.username or "").strip())
  if not user:
    raise HTTPException(status_code=401, detail="Invalid username or password")
  if not verify_password(payload.password or "", user["salt"], user["password_hash"]):
    raise HTTPException(status_code=401, detail="Invalid username or password")

  request.session["username"] = user["username"]
  return {"ok": True, "username": user["username"]}


@app.post("/auth/logout")
async def auth_logout(request: Request):
  request.session.clear()
  return {"ok": True}


@app.post("/api/contact")
async def create_contact_message(payload: ContactCreateRequest):
  email = (payload.email or "").strip()
  comment = (payload.comment or "").strip()
  if not email or "@" not in email:
    raise HTTPException(status_code=400, detail="Valid email is required")
  if len(comment) < 3:
    raise HTTPException(status_code=400, detail="Comment is too short")
  if len(comment) > 5000:
    raise HTTPException(status_code=400, detail="Comment is too long")

  with get_db_conn() as conn:
    conn.execute(
      "INSERT INTO contact_messages (email, comment, approved) VALUES (?, ?, 0)",
      (email, comment),
    )
    conn.commit()
  return {"ok": True}


@app.get("/api/contact/messages")
async def list_contact_messages(request: Request):
  require_admin(request)
  with get_db_conn() as conn:
    rows = conn.execute(
      """
      SELECT id, email, comment, approved, created_at
      FROM contact_messages
      ORDER BY datetime(created_at) DESC, id DESC
      """
    ).fetchall()
  return {
    "messages": [
      {
        "id": int(row["id"]),
        "email": row["email"],
        "comment": row["comment"],
        "approved": bool(row["approved"]),
        "created_at": row["created_at"],
      }
      for row in rows
    ]
  }


@app.post("/api/contact/messages/{message_id}/approve")
async def approve_contact_message(message_id: int, payload: ContactApproveRequest, request: Request):
  require_admin(request)
  with get_db_conn() as conn:
    row = conn.execute("SELECT id FROM contact_messages WHERE id = ?", (message_id,)).fetchone()
    if not row:
      raise HTTPException(status_code=404, detail="Message not found")
    conn.execute(
      "UPDATE contact_messages SET approved = ? WHERE id = ?",
      (1 if payload.approved else 0, message_id),
    )
    conn.commit()
  return {"ok": True}


def get_model() -> Model:
  global _model
  if _model is None:
    if not os.path.exists(MODEL_PATH):
      raise RuntimeError(f"Vosk model not found at {MODEL_PATH}")
    _model = Model(MODEL_PATH)
  return _model


def format_srt_time(seconds: float) -> str:
  hours = int(seconds // 3600)
  minutes = int((seconds % 3600) // 60)
  secs = seconds % 60
  return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def build_srt(cues):
  lines = []
  for idx, cue in enumerate(cues, start=1):
    lines.append(str(idx))
    lines.append(f"{format_srt_time(cue['start'])} --> {format_srt_time(cue['end'])}")
    lines.append(cue["text"])
    lines.append("")
  return "\n".join(lines)


def convert_to_wav(input_path: str, output_path: str) -> None:
  cmd = [
    "ffmpeg",
    "-y",
    "-i",
    input_path,
    "-ac",
    "1",
    "-ar",
    "16000",
    "-f",
    "wav",
    output_path,
  ]
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode != 0:
    raise RuntimeError(proc.stderr.strip() or "ffmpeg failed")


def transcribe_wav(path: str):
  model = get_model()
  with wave.open(path, "rb") as wf:
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
      raise RuntimeError("Audio must be 16kHz mono PCM WAV after conversion")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    words = []
    while True:
      data = wf.readframes(4000)
      if len(data) == 0:
        break
      if rec.AcceptWaveform(data):
        chunk = json.loads(rec.Result())
        words.extend(chunk.get("result", []))
    final = json.loads(rec.FinalResult())
    words.extend(final.get("result", []))

  cues = [
    {"start": w["start"], "end": w["end"], "text": w["word"]}
    for w in words
    if "start" in w and "end" in w and "word" in w
  ]
  if not cues:
    text = final.get("text", "")
    tokens = [t for t in text.split() if t]
    duration = max(0.6, (tokens and 0.6 * len(tokens)) or 0.6)
    cues = [
      {"start": idx * 0.6, "end": idx * 0.6 + 0.6, "text": t}
      for idx, t in enumerate(tokens)
    ]
  return cues


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
  size_header = file.headers.get("content-length")
  if size_header and int(size_header) > MAX_UPLOAD_BYTES:
    raise HTTPException(status_code=413, detail="File too large")

  with tempfile.TemporaryDirectory() as tmpdir:
    input_path = os.path.join(tmpdir, "input")
    output_path = os.path.join(tmpdir, "audio.wav")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
      raise HTTPException(status_code=413, detail="File too large")
    with open(input_path, "wb") as f:
      f.write(data)

    try:
      convert_to_wav(input_path, output_path)
      cues = transcribe_wav(output_path)
    except Exception as exc:
      raise HTTPException(status_code=400, detail=str(exc))

  return {"cues": cues, "srt": build_srt(cues)}


@app.get("/health")
async def health():
  return {"ok": True}
