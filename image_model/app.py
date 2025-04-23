import os, io, tempfile, logging, hashlib, time
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File, Header, Form
from pydantic import BaseModel
import torch
from transformers import SiglipConfig, SiglipForImageClassification, AutoImageProcessor
from PIL import Image
from azure.storage.blob import BlobServiceClient
from prometheus_client import Counter, Histogram, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# ╭──────────────── Config ───────────────╮
load_dotenv()
AZ_CONN     = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER   = "fakenewsdetection-models"
PROD_BLOB   = "image.pt"
SHAD_BLOB   = "shadow_image.pt"
#LOCAL_WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", os.path.dirname(__file__))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ╰────────────────────────────────────────╯

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Start Prometheus exporter on port 9104
start_http_server(9104)

# Prod metrics per role
PROD_INF   = Counter("model_inferences_total", "Total model inferences", ["role"])
LATENCY    = Histogram("model_latency_seconds", "Inference latency", ["role"])
SH_INF     = Counter("shadow_inferences_total", "Shadow inferences", [])
SH_MATCH   = Counter("shadow_agree_total", "Shadow agrees with prod", [])

#
# Globals
#
CURRENT_ROLE    = "prod"
primary_model   = None
secondary_model = None
processor       = None

def download_blob(name):
    try:
        blob = BlobServiceClient.from_connection_string(AZ_CONN)\
               .get_container_client(CONTAINER).get_blob_client(name)
        with tempfile.NamedTemporaryFile(delete=False) as t:
            blob.download_blob().readinto(t)
            return torch.load(t.name, map_location=device)
    except Exception as e:
        log.warning(f"download {name}: {e}")
        return None
    

'''
def _load_local(blob_name):
    path = os.path.join(LOCAL_WEIGHTS_DIR, blob_name)
    if os.path.isfile(path):
        try:
            return torch.load(path, map_location=device)
        except Exception as e:
            log.warning(f"local load {blob_name}: {e}")
    return None
'''


def load_weights(role: str):
    global CURRENT_ROLE, primary_model, secondary_model, processor
    CURRENT_ROLE = role
    cfg = SiglipConfig.from_pretrained("Ateeqq/ai-vs-human-image-detector")
    proc= AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
    processor = proc

    prod_sd  = download_blob(PROD_BLOB)
    # Ensure we have prod_sd, with retry loop similar to earlier logic
    while prod_sd is None:
        log.warning("prod blob missing, retrying in 10 seconds")
        time.sleep(10)
        prod_sd = download_blob(PROD_BLOB)

    if role == "prod":
        prim = SiglipForImageClassification(cfg); prim.load_state_dict(prod_sd, strict=False)
        primary_model = prim.to(device).eval()
        secondary_model = None
    else:
        shad_sd =  download_blob(SHAD_BLOB)
        if shad_sd is None:
            raise RuntimeError("shadow blob missing for shadow role")
        prim = SiglipForImageClassification(cfg); prim.load_state_dict(prod_sd, strict=False)
        sec  = SiglipForImageClassification(cfg); sec.load_state_dict(shad_sd, strict=False)
        primary_model, secondary_model = prim.to(device).eval(), sec.to(device).eval()

    log.info(f"Loaded models – role={CURRENT_ROLE}  secondary={'yes' if secondary_model else 'none'}")

load_weights("prod")

# ╭────────────────── FastAPI ───────────────╮
router = APIRouter()
admin  = APIRouter()

class Resp(BaseModel):
    prediction: str
    confidence: float

@router.post("/image", response_model=Resp)
async def predict(file: UploadFile = File(...)):
    if primary_model is None or processor is None:
        raise HTTPException(503,"Model not ready")
    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400,"Bad image")

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        start_t = time.perf_counter()
        logits = primary_model(**inputs).logits
        latency = time.perf_counter() - start_t
        probs  = torch.softmax(logits, dim=-1)[0]
        idx    = int(probs.argmax())
        conf   = float(probs[idx])*100
        label  = primary_model.config.id2label[idx]
    PROD_INF.labels(CURRENT_ROLE).inc()
    LATENCY.labels(CURRENT_ROLE).observe(latency)

    if secondary_model:
        with torch.no_grad():
            s_idx = int(torch.argmax(secondary_model(**inputs).logits,1).item())
        SH_INF.inc()
        if s_idx == idx:
            SH_MATCH.inc()

    out = "AI" if label.lower()=="ai" else "Human"
    return Resp(prediction=out, confidence=round(conf,2))

# ---------- Feedback endpoint ----------
class FeedbackResp(BaseModel):
    status: str

@router.post("/feedback/image", response_model=FeedbackResp)
async def feedback_image(file: UploadFile = File(...), label: int = Form(...)):
    """Save user‑verified image with metadata if unique."""
    img_bytes = await file.read()
    img_hash = hashlib.sha256(img_bytes).hexdigest()
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    container = "fakenewsdetection-ai-imgs" if label == 1 else "fakenewsdetection-hum-imgs"
    try:
        blob_client = BlobServiceClient.from_connection_string(AZ_CONN) \
            .get_container_client(container) \
            .get_blob_client(f"{img_hash}{ext}")
        if not blob_client.exists():
            metadata = {"feedback": str(label), "filename": file.filename}
            blob_client.upload_blob(img_bytes, overwrite=False, metadata=metadata)
        else:
            log.info("duplicate feedback image ignored")
    except Exception as e:
        log.error(f"feedback upload image: {e}")
    return {"status":"ok"}

# ───── Admin
def chk(token): 
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(401,"bad token")

class RoleReq(BaseModel):
    role: str

@admin.post("/admin/role")
def set_role(r: RoleReq, x_token: str | None = Header(None)):
    chk(x_token); load_weights(r.role)
    return {"role":CURRENT_ROLE}

@admin.post("/admin/reload")
def reload(x_token: str | None = Header(None)):
    chk(x_token); load_weights(CURRENT_ROLE)
    return {"role":CURRENT_ROLE,"status":"reloaded"}

@admin.get("/admin/status")
def stat(x_token: str | None = Header(None)):
    chk(x_token)
    return {"role":CURRENT_ROLE,"secondary":bool(secondary_model)}

@admin.get("/metrics")
async def metrics():
    """Expose Prometheus metrics on the API port."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

app = FastAPI(title="FakeNews‑Image‑API")
app.include_router(router, tags=["image"])
app.include_router(admin,  tags=["admin"])