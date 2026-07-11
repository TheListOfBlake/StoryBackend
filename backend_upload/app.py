from __future__ import annotations

import json
import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
import smtplib
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import wave
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from vosk import KaldiRecognizer, Model

MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
FRONTEND_ORIGINS = os.environ.get("FRONTEND_ORIGINS", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")
PBKDF2_ROUNDS = int(os.environ.get("PBKDF2_ROUNDS", "200000"))
DB_PATH = os.environ.get("AUTH_DB_PATH", str(Path(__file__).resolve().parent / "data" / "app.db"))
ADMIN_USERNAME = (
  os.environ.get("ADMIN_USERNAME")
  or os.environ.get("ADMIN_USER")
  or os.environ.get("VITE_ADMIN_USER")
  or ""
).strip()
ADMIN_PASSWORD = (
  os.environ.get("ADMIN_PASSWORD")
  or os.environ.get("ADMIN_PASS")
  or os.environ.get("VITE_ADMIN_PASS")
  or ""
)
PAYMENT_CONFIG_SECRET = os.environ.get("PAYMENT_CONFIG_SECRET", "")
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", SMTP_USERNAME).strip()
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME", "Storytime Captions").strip()

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


SMTP_USE_TLS = parse_bool_env("SMTP_USE_TLS", True)


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
  access_key: Optional[str] = None


class ContactCreateRequest(BaseModel):
  email: str
  comment: str


class ContactApproveRequest(BaseModel):
  approved: bool


class TemplateProduct(BaseModel):
  id: Optional[int] = None
  slug: str
  name: str
  description: str = ""
  old_price: str = ""
  price: str
  badge: str = ""
  stripe_url: str = ""
  paypal_url: str = ""
  r2_download_url: str = ""
  active: bool = True


class TemplateBulkUpdate(BaseModel):
  products: List[TemplateProduct]


class TemplateDownloadRequest(BaseModel):
  product_slug: str
  email: str


class TemplateCheckoutRequest(BaseModel):
  product_slug: str
  customer_first_name: str = ""
  customer_last_name: str = ""
  customer_email: str = ""
  success_url: str
  cancel_url: str


class PaymentSettingsUpdate(BaseModel):
  stripe_secret_key: Optional[str] = None
  stripe_webhook_secret: Optional[str] = None
  paypal_client_id: Optional[str] = None
  paypal_client_secret: Optional[str] = None
  paypal_webhook_id: Optional[str] = None


PAYMENT_SETTING_KEYS = {
  "stripe_secret_key",
  "stripe_webhook_secret",
  "paypal_client_id",
  "paypal_client_secret",
  "paypal_webhook_id",
}


DEFAULT_R2_DOWNLOAD_URLS = {
  f"month-{index}-pack": f"https://media.storytimecaptions.com/Month%20{index}.zip"
  for index in range(1, 13)
}


DEFAULT_TEMPLATE_PRODUCTS = [
  {
    "slug": "free-pack",
    "name": "Free Pack",
    "description": "Starter story template pack.",
    "old_price": "",
    "price": "Free",
    "badge": "Free starter pack",
  },
  {
    "slug": "full-12-month-pack",
    "name": "Full 12 Month Pack",
    "description": "Commercial use template bundle with 12 monthly packs.",
    "old_price": "$179.99",
    "price": "$99.99",
    "badge": "Save 44% • Instant Download • Commercial Use • 12 Monthly Packs",
  },
  *[
    {
      "slug": f"month-{index}-pack",
      "name": f"Month {index} Pack",
      "description": "Monthly story template pack.",
      "old_price": "",
      "price": "$14.99",
      "badge": "Instant Download • Commercial Use",
      "r2_download_url": DEFAULT_R2_DOWNLOAD_URLS[f"month-{index}-pack"],
    }
    for index in range(1, 13)
  ],
]


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
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS template_products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slug TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        old_price TEXT NOT NULL DEFAULT '',
        price TEXT NOT NULL,
        badge TEXT NOT NULL DEFAULT '',
        stripe_url TEXT NOT NULL DEFAULT '',
        paypal_url TEXT NOT NULL DEFAULT '',
        r2_download_url TEXT NOT NULL DEFAULT '',
        active INTEGER NOT NULL DEFAULT 1,
        sort_order INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS template_purchases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL,
        provider_event_id TEXT NOT NULL UNIQUE,
        provider_payment_id TEXT NOT NULL DEFAULT '',
        product_slug TEXT NOT NULL,
        customer_email TEXT NOT NULL,
        raw_event TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS payment_settings (
        key TEXT PRIMARY KEY,
        encrypted_value TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      )
      """
    )
    row = conn.execute("SELECT COUNT(*) AS count FROM template_products").fetchone()
    if row and int(row["count"]) == 0:
      for sort_order, product in enumerate(DEFAULT_TEMPLATE_PRODUCTS):
        conn.execute(
          """
          INSERT INTO template_products (
            slug, name, description, old_price, price, badge, r2_download_url, sort_order
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            product["slug"],
            product["name"],
            product["description"],
            product["old_price"],
            product["price"],
            product["badge"],
            product.get("r2_download_url", ""),
            sort_order,
          ),
        )
    for slug, download_url in DEFAULT_R2_DOWNLOAD_URLS.items():
      conn.execute(
        """
        UPDATE template_products
        SET r2_download_url = ?, updated_at = CURRENT_TIMESTAMP
        WHERE slug = ? AND (r2_download_url IS NULL OR r2_download_url = '')
        """,
        (download_url, slug),
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


def has_env_admin() -> bool:
  return bool(ADMIN_USERNAME and ADMIN_PASSWORD)


def verify_env_admin(username: str, password: str) -> bool:
  if not has_env_admin():
    return False
  return hmac.compare_digest(username.strip(), ADMIN_USERNAME) and hmac.compare_digest(password or "", ADMIN_PASSWORD)


def require_admin(request: Request) -> str:
  username = request.session.get("username")
  if not username:
    raise HTTPException(status_code=401, detail="Not authenticated")
  return str(username)


def require_admin_access_key(access_key: Optional[str]) -> None:
  configured_key = (os.environ.get("ADMIN_ACCESS_KEY") or "").strip()
  if not configured_key:
    return
  provided = (access_key or "").strip()
  if not provided or not hmac.compare_digest(provided, configured_key):
    raise HTTPException(status_code=403, detail="Invalid admin access key")


def get_payment_cipher() -> Fernet:
  secret = PAYMENT_CONFIG_SECRET.strip()
  if len(secret) < 32:
    raise HTTPException(
      status_code=500,
      detail="PAYMENT_CONFIG_SECRET must be set to a long random value before saving payment settings",
    )
  key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
  return Fernet(key)


def encrypt_payment_value(value: str) -> str:
  clean = (value or "").strip()
  if not clean:
    raise HTTPException(status_code=400, detail="Payment setting value cannot be empty")
  return get_payment_cipher().encrypt(clean.encode("utf-8")).decode("utf-8")


def decrypt_payment_value(encrypted_value: str) -> str:
  try:
    return get_payment_cipher().decrypt(encrypted_value.encode("utf-8")).decode("utf-8")
  except InvalidToken:
    raise HTTPException(status_code=500, detail="Unable to decrypt payment settings with the configured secret")


def mask_secret_value(value: str) -> str:
  clean = (value or "").strip()
  if not clean:
    return ""
  if len(clean) <= 8:
    return "•" * len(clean)
  return f"{clean[:4]}{'•' * 8}{clean[-4:]}"


def serialize_payment_setting(key: str, encrypted_value: Optional[str], updated_at: Optional[str]) -> dict:
  if not encrypted_value:
    return {"key": key, "configured": False, "masked_value": "", "updated_at": ""}
  value = decrypt_payment_value(encrypted_value)
  return {
    "key": key,
    "configured": True,
    "masked_value": mask_secret_value(value),
    "updated_at": updated_at or "",
  }


def get_payment_setting_value(key: str) -> str:
  if key not in PAYMENT_SETTING_KEYS:
    return ""
  with get_db_conn() as conn:
    row = conn.execute("SELECT encrypted_value FROM payment_settings WHERE key = ?", (key,)).fetchone()
  if not row:
    return ""
  return decrypt_payment_value(row["encrypted_value"]).strip()


def get_stripe_secret_key() -> str:
  saved = get_payment_setting_value("stripe_secret_key")
  return saved or (os.environ.get("STRIPE_SECRET_KEY") or "").strip()


def get_stripe_webhook_secret() -> str:
  saved = get_payment_setting_value("stripe_webhook_secret")
  return saved or (os.environ.get("STRIPE_WEBHOOK_SECRET") or "").strip()


def parse_price_cents(price: str) -> int:
  clean = (price or "").strip().replace("$", "").replace(",", "")
  try:
    amount = round(float(clean) * 100)
  except ValueError:
    raise HTTPException(status_code=400, detail="Product price must be a dollar amount like $14.99")
  if amount < 50:
    raise HTTPException(status_code=400, detail="Product price is too low for checkout")
  return int(amount)


def validate_checkout_url(value: str, fallback_origin: str) -> str:
  clean = (value or "").strip()
  parsed = urllib.parse.urlparse(clean)
  if parsed.scheme in {"http", "https"} and parsed.netloc:
    return clean
  if fallback_origin:
    origin = fallback_origin.rstrip("/")
    return f"{origin}/"
  raise HTTPException(status_code=400, detail="A valid checkout return URL is required")


def stripe_checkout_session(
  product: sqlite3.Row,
  success_url: str,
  cancel_url: str,
  customer_first_name: str = "",
  customer_last_name: str = "",
  customer_email: str = "",
) -> str:
  secret_key = get_stripe_secret_key()
  if not secret_key:
    raise HTTPException(status_code=500, detail="Stripe secret key is not configured")

  product_slug = product["slug"]
  first_name = customer_first_name.strip()
  last_name = customer_last_name.strip()
  customer_name = " ".join(part for part in [first_name, last_name] if part)
  if not first_name:
    raise HTTPException(status_code=400, detail="First name is required")
  data = {
    "mode": "payment",
    "success_url": success_url,
    "cancel_url": cancel_url,
    "payment_method_types[0]": "card",
    "customer_creation": "if_required",
    "line_items[0][quantity]": "1",
    "line_items[0][price_data][currency]": "usd",
    "line_items[0][price_data][unit_amount]": str(parse_price_cents(product["price"])),
    "line_items[0][price_data][product_data][name]": product["name"],
    "line_items[0][price_data][product_data][description]": product["description"] or product["badge"] or product["name"],
    "line_items[0][price_data][product_data][metadata][template_slug]": product_slug,
    "metadata[template_slug]": product_slug,
    "metadata[customer_name]": customer_name.strip(),
    "payment_intent_data[metadata][template_slug]": product_slug,
    "payment_intent_data[metadata][customer_name]": customer_name.strip(),
  }
  clean_email = customer_email.strip().lower()
  if clean_email:
    if "@" not in clean_email:
      raise HTTPException(status_code=400, detail="Enter a valid email address")
    data["customer_email"] = clean_email
  encoded = urllib.parse.urlencode(data).encode("utf-8")
  request = urllib.request.Request(
    "https://api.stripe.com/v1/checkout/sessions",
    data=encoded,
    headers={
      "Authorization": f"Bearer {secret_key}",
      "Content-Type": "application/x-www-form-urlencoded",
    },
    method="POST",
  )
  try:
    with urllib.request.urlopen(request, timeout=20) as response:
      payload = json.loads(response.read().decode("utf-8"))
  except urllib.error.HTTPError as exc:
    error_body = exc.read().decode("utf-8", errors="ignore")
    try:
      error_payload = json.loads(error_body)
      message = error_payload.get("error", {}).get("message") or "Stripe rejected the checkout request"
    except json.JSONDecodeError:
      message = "Stripe rejected the checkout request"
    raise HTTPException(status_code=502, detail=message)
  except urllib.error.URLError:
    raise HTTPException(status_code=502, detail="Unable to reach Stripe checkout service")

  checkout_url = payload.get("url")
  if not checkout_url:
    raise HTTPException(status_code=502, detail="Stripe did not return a checkout URL")
  return str(checkout_url)


def normalize_template_slug(value: str) -> str:
  clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in (value or "").strip())
  clean = "-".join(part for part in clean.split("-") if part)
  if not clean:
    raise HTTPException(status_code=400, detail="Template slug is required")
  return clean[:80]


def serialize_template_product(row: sqlite3.Row, include_private: bool = False) -> dict:
  product = {
    "id": int(row["id"]),
    "slug": row["slug"],
    "name": row["name"],
    "description": row["description"],
    "old_price": row["old_price"],
    "price": row["price"],
    "badge": row["badge"],
    "stripe_url": row["stripe_url"],
    "paypal_url": row["paypal_url"],
    "active": bool(row["active"]),
  }
  if include_private:
    product["r2_download_url"] = row["r2_download_url"]
  return product


def serialize_template_purchase(row: sqlite3.Row) -> dict:
  return {
    "id": int(row["id"]),
    "provider": row["provider"],
    "provider_event_id": row["provider_event_id"],
    "provider_payment_id": row["provider_payment_id"],
    "product_slug": row["product_slug"],
    "product_name": row["product_name"] or row["product_slug"],
    "product_price": row["product_price"] or "",
    "customer_email": row["customer_email"],
    "created_at": row["created_at"],
  }


def get_template_by_slug(slug: str):
  with get_db_conn() as conn:
    return conn.execute("SELECT * FROM template_products WHERE slug = ?", (slug,)).fetchone()


def month_pack_sort_key(row: sqlite3.Row) -> int:
  slug = str(row["slug"])
  parts = slug.split("-")
  for part in parts:
    if part.isdigit():
      return int(part)
  return 999


def get_template_download_links(product_slug: str) -> List[dict]:
  clean_slug = normalize_template_slug(product_slug)
  with get_db_conn() as conn:
    if clean_slug == "full-12-month-pack":
      rows = conn.execute(
        """
        SELECT name, slug, r2_download_url FROM template_products
        WHERE slug LIKE 'month-%-pack' AND active = 1 AND r2_download_url != ''
        """
      ).fetchall()
      return [
        {"name": row["name"], "url": row["r2_download_url"]}
        for row in sorted(rows, key=month_pack_sort_key)
      ]

    row = conn.execute(
      "SELECT name, slug, r2_download_url FROM template_products WHERE slug = ? AND active = 1",
      (clean_slug,),
    ).fetchone()
  if not row or not row["r2_download_url"]:
    return []
  return [{"name": row["name"], "url": row["r2_download_url"]}]


def send_template_download_email(product_slug: str, email: str) -> None:
  if not SMTP_HOST or not SMTP_FROM_EMAIL:
    return

  links = get_template_download_links(product_slug)
  if not links:
    return

  clean_email = email.strip().lower()
  product = get_template_by_slug(product_slug)
  product_name = product["name"] if product else "Storytime Template Pack"
  lines = [
    "Thanks for your purchase.",
    "",
    f"Your {product_name} download link{'s are' if len(links) > 1 else ' is'} below:",
    "",
  ]
  for link in links:
    lines.append(f"{link['name']}: {link['url']}")
  lines.extend(
    [
      "",
      "If you have trouble downloading, reply to this email with your purchase email.",
      "",
      "Storytime Captions",
    ]
  )

  message = EmailMessage()
  message["Subject"] = f"Your {product_name} download"
  message["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
  message["To"] = clean_email
  message.set_content("\n".join(lines))

  try:
    if SMTP_USE_TLS:
      with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
        smtp.starttls()
        if SMTP_USERNAME and SMTP_PASSWORD:
          smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
        smtp.send_message(message)
    else:
      with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
        if SMTP_USERNAME and SMTP_PASSWORD:
          smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
        smtp.send_message(message)
  except (OSError, smtplib.SMTPException):
    return


def extract_nested_value(data: Dict, paths: List[List[str]]) -> str:
  for path in paths:
    current = data
    for key in path:
      if isinstance(current, dict):
        current = current.get(key)
      elif isinstance(current, list) and key.isdigit():
        index = int(key)
        current = current[index] if index < len(current) else None
      else:
        current = None
        break
    if current:
      return str(current)
  return ""


def record_template_purchase(provider: str, event_id: str, payment_id: str, product_slug: str, email: str, raw_event: dict) -> None:
  clean_slug = normalize_template_slug(product_slug)
  clean_email = (email or "").strip().lower()
  if not clean_email or "@" not in clean_email:
    raise HTTPException(status_code=400, detail="Webhook did not include a customer email")
  if not get_template_by_slug(clean_slug):
    raise HTTPException(status_code=400, detail="Webhook referenced an unknown template")
  with get_db_conn() as conn:
    cursor = conn.execute(
      """
      INSERT OR IGNORE INTO template_purchases (
        provider, provider_event_id, provider_payment_id, product_slug, customer_email, raw_event
      ) VALUES (?, ?, ?, ?, ?, ?)
      """,
      (provider, event_id, payment_id or "", clean_slug, clean_email, json.dumps(raw_event)),
    )
    conn.commit()
  if cursor.rowcount > 0:
    send_template_download_email(clean_slug, clean_email)


def verify_stripe_signature(raw_body: bytes, signature_header: str) -> None:
  secret = get_stripe_webhook_secret()
  if not secret:
    return
  timestamp = ""
  signatures = []
  for part in signature_header.split(","):
    key, _, value = part.partition("=")
    if key == "t":
      timestamp = value
    if key == "v1":
      signatures.append(value)
  if not timestamp or not signatures:
    raise HTTPException(status_code=400, detail="Invalid Stripe signature header")
  signed_payload = f"{timestamp}.{raw_body.decode('utf-8')}".encode("utf-8")
  expected = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
  if not any(hmac.compare_digest(expected, signature) for signature in signatures):
    raise HTTPException(status_code=400, detail="Invalid Stripe webhook signature")


def verify_paypal_token(request: Request) -> None:
  expected = (os.environ.get("PAYPAL_WEBHOOK_TOKEN") or "").strip()
  if not expected:
    return
  provided = request.headers.get("x-paypal-webhook-token") or request.query_params.get("token") or ""
  if not hmac.compare_digest(provided, expected):
    raise HTTPException(status_code=403, detail="Invalid PayPal webhook token")


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
    "setup_required": False if has_env_admin() else count_users() == 0,
    "authenticated": bool(username),
    "username": username or "",
  }


@app.post("/auth/setup-admin")
async def auth_setup_admin(payload: AuthRequest, request: Request):
  require_admin_access_key(payload.access_key)
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
  require_admin_access_key(payload.access_key)
  if has_env_admin():
    if not verify_env_admin(payload.username or "", payload.password or ""):
      raise HTTPException(status_code=401, detail="Invalid username or password")
    request.session["username"] = ADMIN_USERNAME
    return {"ok": True, "username": ADMIN_USERNAME}

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


@app.get("/api/templates")
async def list_public_templates():
  with get_db_conn() as conn:
    rows = conn.execute(
      """
      SELECT * FROM template_products
      WHERE active = 1
      ORDER BY sort_order ASC, id ASC
      """
    ).fetchall()
  return {"products": [serialize_template_product(row) for row in rows]}


@app.post("/api/templates/checkout")
async def create_template_checkout(payload: TemplateCheckoutRequest, request: Request):
  product_slug = normalize_template_slug(payload.product_slug)
  with get_db_conn() as conn:
    product = conn.execute(
      "SELECT * FROM template_products WHERE slug = ? AND active = 1",
      (product_slug,),
    ).fetchone()
  if not product:
    raise HTTPException(status_code=404, detail="Template pack not found")
  if product["price"].strip().lower() == "free":
    raise HTTPException(status_code=400, detail="Free packs do not require checkout")

  fallback_origin = request.headers.get("origin") or ""
  success_url = validate_checkout_url(payload.success_url, fallback_origin)
  cancel_url = validate_checkout_url(payload.cancel_url, fallback_origin)
  checkout_url = stripe_checkout_session(
    product,
    success_url,
    cancel_url,
    payload.customer_first_name,
    payload.customer_last_name,
    payload.customer_email,
  )
  return {"provider": "stripe", "checkout_url": checkout_url}


@app.get("/api/admin/payment-settings")
async def get_admin_payment_settings(request: Request):
  require_admin(request)
  with get_db_conn() as conn:
    rows = conn.execute("SELECT key, encrypted_value, updated_at FROM payment_settings").fetchall()
  by_key = {row["key"]: row for row in rows}
  return {
    "encryption_configured": len(PAYMENT_CONFIG_SECRET.strip()) >= 32,
    "settings": {
      key: serialize_payment_setting(
        key,
        by_key[key]["encrypted_value"] if key in by_key else None,
        by_key[key]["updated_at"] if key in by_key else None,
      )
      for key in sorted(PAYMENT_SETTING_KEYS)
    },
  }


@app.post("/api/admin/payment-settings")
async def save_admin_payment_settings(payload: PaymentSettingsUpdate, request: Request):
  require_admin(request)
  updates = payload.dict()
  clean_updates = {
    key: value.strip()
    for key, value in updates.items()
    if key in PAYMENT_SETTING_KEYS and isinstance(value, str) and value.strip()
  }
  if not clean_updates:
    raise HTTPException(status_code=400, detail="Enter at least one payment setting to save")

  with get_db_conn() as conn:
    for key, value in clean_updates.items():
      conn.execute(
        """
        INSERT INTO payment_settings (key, encrypted_value, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET
          encrypted_value = excluded.encrypted_value,
          updated_at = CURRENT_TIMESTAMP
        """,
        (key, encrypt_payment_value(value)),
      )
    conn.commit()
  return await get_admin_payment_settings(request)


@app.get("/api/admin/templates")
async def list_admin_templates(request: Request):
  require_admin(request)
  with get_db_conn() as conn:
    rows = conn.execute("SELECT * FROM template_products ORDER BY sort_order ASC, id ASC").fetchall()
  return {"products": [serialize_template_product(row, include_private=True) for row in rows]}


@app.get("/api/admin/template-purchases")
async def list_admin_template_purchases(request: Request):
  require_admin(request)
  with get_db_conn() as conn:
    rows = conn.execute(
      """
      SELECT
        template_purchases.id,
        template_purchases.provider,
        template_purchases.provider_event_id,
        template_purchases.provider_payment_id,
        template_purchases.product_slug,
        template_purchases.customer_email,
        template_purchases.created_at,
        template_products.name AS product_name,
        template_products.price AS product_price
      FROM template_purchases
      LEFT JOIN template_products ON template_products.slug = template_purchases.product_slug
      ORDER BY datetime(template_purchases.created_at) DESC, template_purchases.id DESC
      LIMIT 500
      """
    ).fetchall()
  return {"purchases": [serialize_template_purchase(row) for row in rows]}


@app.post("/api/admin/templates")
async def save_admin_templates(payload: TemplateBulkUpdate, request: Request):
  require_admin(request)
  with get_db_conn() as conn:
    existing_ids = {
      int(row["id"])
      for row in conn.execute("SELECT id FROM template_products").fetchall()
    }
    kept_ids = set()
    for sort_order, product in enumerate(payload.products):
      slug = normalize_template_slug(product.slug or product.name)
      name = product.name.strip()
      price = product.price.strip()
      if not name:
        raise HTTPException(status_code=400, detail="Template name is required")
      if not price:
        raise HTTPException(status_code=400, detail=f"Price is required for {name}")
      values = (
        slug,
        name,
        product.description.strip(),
        product.old_price.strip(),
        price,
        product.badge.strip(),
        product.stripe_url.strip(),
        product.paypal_url.strip(),
        product.r2_download_url.strip(),
        1 if product.active else 0,
        sort_order,
      )
      if product.id and product.id in existing_ids:
        conn.execute(
          """
          UPDATE template_products
          SET slug = ?, name = ?, description = ?, old_price = ?, price = ?, badge = ?,
              stripe_url = ?, paypal_url = ?, r2_download_url = ?, active = ?,
              sort_order = ?, updated_at = CURRENT_TIMESTAMP
          WHERE id = ?
          """,
          (*values, product.id),
        )
        kept_ids.add(product.id)
      else:
        cursor = conn.execute(
          """
          INSERT INTO template_products (
            slug, name, description, old_price, price, badge, stripe_url,
            paypal_url, r2_download_url, active, sort_order
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          values,
        )
        kept_ids.add(int(cursor.lastrowid))
    ids_to_delete = existing_ids - kept_ids
    for product_id in ids_to_delete:
      conn.execute("DELETE FROM template_products WHERE id = ?", (product_id,))
    conn.commit()
  return await list_admin_templates(request)


@app.post("/api/templates/download")
async def get_template_download(payload: TemplateDownloadRequest):
  product_slug = normalize_template_slug(payload.product_slug)
  email = payload.email.strip().lower()
  if not email or "@" not in email:
    raise HTTPException(status_code=400, detail="Enter the purchase email")
  with get_db_conn() as conn:
    purchase = conn.execute(
      """
      SELECT id FROM template_purchases
      WHERE product_slug = ? AND lower(customer_email) = ?
      ORDER BY datetime(created_at) DESC, id DESC
      LIMIT 1
      """,
      (product_slug, email),
    ).fetchone()
    if not purchase:
      raise HTTPException(status_code=404, detail="No completed purchase found for that email")
  links = get_template_download_links(product_slug)
  if not links:
    raise HTTPException(status_code=404, detail="Download is not configured yet")
  response = {"download_links": links}
  if len(links) == 1:
    response["download_url"] = links[0]["url"]
  return response


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
  raw_body = await request.body()
  verify_stripe_signature(raw_body, request.headers.get("stripe-signature", ""))
  try:
    event = json.loads(raw_body.decode("utf-8"))
  except json.JSONDecodeError:
    raise HTTPException(status_code=400, detail="Invalid JSON")
  event_type = event.get("type", "")
  if event_type != "checkout.session.completed":
    return {"ok": True, "ignored": True}
  data_object = (event.get("data") or {}).get("object") or {}
  metadata = data_object.get("metadata") or {}
  product_slug = metadata.get("template_slug") or metadata.get("product_slug") or metadata.get("slug") or ""
  email = (
    data_object.get("customer_email")
    or data_object.get("receipt_email")
    or extract_nested_value(data_object, [["customer_details", "email"], ["billing_details", "email"]])
  )
  payment_id = str(data_object.get("payment_intent") or data_object.get("id") or "")
  record_template_purchase("stripe", str(event.get("id") or payment_id), payment_id, product_slug, email, event)
  return {"ok": True}


@app.post("/webhooks/paypal")
async def paypal_webhook(request: Request):
  verify_paypal_token(request)
  try:
    event = await request.json()
  except json.JSONDecodeError:
    raise HTTPException(status_code=400, detail="Invalid JSON")
  event_type = event.get("event_type", "")
  if event_type and event_type not in {"CHECKOUT.ORDER.APPROVED", "PAYMENT.CAPTURE.COMPLETED"}:
    return {"ok": True, "ignored": True}
  resource = event.get("resource") or {}
  product_slug = (
    resource.get("custom_id")
    or resource.get("invoice_id")
    or extract_nested_value(resource, [["purchase_units", "0", "custom_id"]])
    or ""
  )
  email = extract_nested_value(
    resource,
    [["payer", "email_address"], ["payment_source", "paypal", "email_address"], ["supplementary_data", "related_ids", "payer_email"]],
  )
  payment_id = str(resource.get("id") or "")
  record_template_purchase("paypal", str(event.get("id") or payment_id), payment_id, product_slug, email, event)
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

