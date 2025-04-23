"""
main/app.py
Gateway that exposes **one** FastAPI endpoint tree and mounts
the internal text and image routers.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
import httpx
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, start_http_server, generate_latest, CONTENT_TYPE_LATEST
import time
import json, os, bcrypt
import hashlib

app = FastAPI(title="FakeNewsDetectorAPI – Gateway")

# Allow any origin (adjust for production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Downstream service endpoints – overridable via environment variables
# ---------------------------------------------------------------------------
TEXT_SERVICE_BASE  = os.getenv("TEXT_SERVICE_URL", "http://text-service.fake-env.internal")
IMAGE_SERVICE_BASE = os.getenv("IMAGE_SERVICE_URL", "http://image-service.fake-env.internal")

TEXT_BACKEND_URL  = f"{TEXT_SERVICE_BASE}/text"
IMAGE_BACKEND_URL = f"{IMAGE_SERVICE_BASE}/image"
ADMIN_HEADER = "x-token"

# temporary in‑memory storage for image bytes awaiting user feedback
IMAGE_FEEDBACK_CACHE: dict[str, tuple[float, bytes]] = {}
EXPIRY_SEC = 300  # 10 minutes

# ─── Prometheus ──────────────────────────
start_http_server(9101)
REQ_COUNT   = Counter("http_requests_total", "Total HTTP requests", ["path"])
REQ_FAILED  = Counter("http_requests_failed", "Failed HTTP requests", ["path"])
LATENCY_SEC = Histogram("http_request_latency_seconds", "HTTP request latency", ["path"])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    path = request.url.path
    REQ_COUNT.labels(path).inc()
    start_t = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        REQ_FAILED.labels(path).inc()
        raise
    finally:
        latency = time.perf_counter() - start_t
        LATENCY_SEC.labels(path).observe(latency)
        if response.status_code >= 400:
            REQ_FAILED.labels(path).inc()
    return response

# -------------------------------- GUI templates setup ------------------------
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")
COOKIE_NAME = "fnd_auth"

def _load_users() -> dict[str, str]:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return {}

def _save_users(users: dict[str, str]):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(users, fh)

async def current_user(request: Request) -> str | None:
    user = request.cookies.get(COOKIE_NAME)
    if user and user in _load_users():
        return user
    return None

# ------------------------------ API schemas ----------------------------------

class TextRequestSchema(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def web_root(request: Request, user: str | None = Depends(current_user)):
    """Render home page if logged‑in else redirect to /login."""
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("home.html", {"request": request, "user": user, "thanks": request.query_params.get("thanks")})

# ------------------------------ Auth routes ----------------------------------

@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup_submit(username: str = Form(...), password: str = Form(...)):
    users = _load_users()
    if username in users:
        return RedirectResponse("/signup?error=exists", status_code=303)
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = hashed
    _save_users(users)
    resp = RedirectResponse("/login?msg=created", status_code=303)
    return resp

@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_submit(response: Response, username: str = Form(...), password: str = Form(...)):
    users = _load_users()
    if username not in users or not bcrypt.checkpw(password.encode(), users[username].encode()):
        return RedirectResponse("/login?error=invalid", status_code=303)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(COOKIE_NAME, username, max_age=24 * 3600, httponly=True)
    return response

@app.get("/logout")
async def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME)
    return resp

# ───────── Admin proxy routes ─────────
@app.post("/admin/text/role")
async def admin_text_role(body: dict, x_token: str | None = Header(None)):
    return await _proxy_admin("POST", f"{TEXT_SERVICE_BASE}/admin/role", body, x_token)

@app.post("/admin/text/reload")
async def admin_text_reload(x_token: str | None = Header(None)):
    return await _proxy_admin("POST", f"{TEXT_SERVICE_BASE}/admin/reload", None, x_token)

@app.get("/admin/text/status")
async def admin_text_status(x_token: str | None = Header(None)):
    return await _proxy_admin("GET", f"{TEXT_SERVICE_BASE}/admin/status", None, x_token)

@app.post("/admin/image/role")
async def admin_image_role(body: dict, x_token: str | None = Header(None)):
    return await _proxy_admin("POST", f"{IMAGE_SERVICE_BASE}/admin/role", body, x_token)

@app.post("/admin/image/reload")
async def admin_image_reload(x_token: str | None = Header(None)):
    return await _proxy_admin("POST", f"{IMAGE_SERVICE_BASE}/admin/reload", None, x_token)

@app.get("/admin/image/status")
async def admin_image_status(x_token: str | None = Header(None)):
    return await _proxy_admin("GET", f"{IMAGE_SERVICE_BASE}/admin/status", None, x_token)

# --------------------------- Web prediction handlers -------------------------

@app.post("/predict/text")
async def predict_text_web(request: Request, user: str | None = Depends(current_user), text: str = Form(...)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(TEXT_BACKEND_URL, json={"text": text})
            resp.raise_for_status()
        except httpx.RequestError:
            return templates.TemplateResponse("home.html", {"request": request, "user": user, "error": "text‑service unreachable"})
    result = resp.json()
    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user,
        "text_result": result,
        "orig_text": text,
        "thanks": request.query_params.get("thanks")
    })

@app.post("/predict/image")
async def predict_image_web(request: Request, user: str | None = Depends(current_user), file: UploadFile = File(...)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    content = await file.read()
    # compute hash to allow later feedback without re‑upload
    img_hash = hashlib.sha256(content).hexdigest()
    # prune old cache entries
    now = time.time()
    for h,(ts,_) in list(IMAGE_FEEDBACK_CACHE.items()):
        if now - ts > EXPIRY_SEC:
            IMAGE_FEEDBACK_CACHE.pop(h, None)

    IMAGE_FEEDBACK_CACHE[img_hash] = (now, content)
    files = {"file": (file.filename, content, file.content_type)}
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(IMAGE_BACKEND_URL, files=files)
            resp.raise_for_status()
        except httpx.RequestError:
            return templates.TemplateResponse("home.html", {"request": request, "user": user, "error": "image‑service unreachable"})
    result = resp.json()
    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user,
        "image_result": result,
        "img_hash": img_hash,
        "thanks": request.query_params.get("thanks")
    })

@app.post("/submit/text")
async def submit_text_feedback(request: Request, user: str | None = Depends(current_user), text: str = Form(...), label:int=Form(...)):
    if not user:
        return RedirectResponse("/login", status_code=303)
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(f"{TEXT_BACKEND_URL.rsplit('/text',1)[0]}/feedback/text", json={"text":text,"label":label})
    return RedirectResponse("/?thanks=1", status_code=303)

@app.post("/submit/image")
async def submit_image_feedback(request: Request, user: str|None=Depends(current_user), file_hash:str=Form(...), label:int=Form(...)):
    if not user:
        return RedirectResponse("/login",status_code=303)
    tup = IMAGE_FEEDBACK_CACHE.pop(file_hash, None)
    if tup is None:
        return RedirectResponse("/", status_code=303)
    ts, img_bytes = tup
    files = {"file": (f"{file_hash}.jpg", img_bytes, "image/jpeg")}
    data  = {"label": str(label)}
    async with httpx.AsyncClient(timeout=60) as client:
        await client.post(f"{IMAGE_BACKEND_URL.rsplit('/image',1)[0]}/feedback/image", files=files, data=data)
    return RedirectResponse("/?thanks=1", status_code=303)

async def _proxy_admin(method: str, url: str, data: dict | None, token: str | None):
    headers = {ADMIN_HEADER: token} if token else {}
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            if method == "POST":
                r = await client.post(url, json=data, headers=headers)
            else:
                r = await client.get(url, headers=headers)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"upstream admin error: {e}")
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics on the main API port."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

