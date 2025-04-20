"""
main/app.py
Gateway that exposes **one** FastAPI endpoint tree and mounts
the internal text and image routers.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, start_http_server
import time

app = FastAPI(title="FakeNewsDetectorAPI – Gateway")

# Allow any origin (adjust for production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

TEXT_BACKEND_URL  = "http://text-service:8001/text"
IMAGE_BACKEND_URL = "http://image-service:8002/image"
ADMIN_HEADER = "x-token"

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
        LATENCY_SEC.labels(path).observe(time.perf_counter() - start_t)
        if response.status_code >= 400:
            REQ_FAILED.labels(path).inc()
    return response

class TextRequestSchema(BaseModel):
    text: str

@app.post("/text")
async def proxy_text(req: TextRequestSchema):
    """Forward text to the text-service and return its response unchanged."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(TEXT_BACKEND_URL, json=req.model_dump())
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"text-service unreachable: {e}")
    return resp.json()

@app.post("/image")
async def proxy_image(file: UploadFile = File(...)):
    """Stream the uploaded image to the image-service and proxy back the JSON."""
    content = await file.read()
    files = {"file": (file.filename, content, file.content_type)}
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(IMAGE_BACKEND_URL, files=files)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"image-service unreachable: {e}")
    return resp.json()

@app.get("/")
def root():
    return {"service": "FakeNewsDetectionAPI – Gateway", "routes": ["/text", "/image"]}

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

# ───────── Admin proxy routes ─────────
@app.post("/admin/text/role")
async def admin_text_role(body: dict, x_token: str | None = Header(None)):
    return await _proxy_admin("POST", "http://text-service:8001/admin/role", body, x_token)

@app.post("/admin/text/reload")
async def admin_text_reload(x_token: str | None = Header(None)):
    return await _proxy_admin("POST", "http://text-service:8001/admin/reload", None, x_token)

@app.get("/admin/text/status")
async def admin_text_status(x_token: str | None = Header(None)):
    return await _proxy_admin("GET", "http://text-service:8001/admin/status", None, x_token)

@app.post("/admin/image/role")
async def admin_image_role(body: dict, x_token: str | None = Header(None)):
    return await _proxy_admin("POST", "http://image-service:8002/admin/role", body, x_token)

@app.post("/admin/image/reload")
async def admin_image_reload(x_token: str | None = Header(None)):
    return await _proxy_admin("POST", "http://image-service:8002/admin/reload", None, x_token)

@app.get("/admin/image/status")
async def admin_image_status(x_token: str | None = Header(None)):
    return await _proxy_admin("GET", "http://image-service:8002/admin/status", None, x_token)
