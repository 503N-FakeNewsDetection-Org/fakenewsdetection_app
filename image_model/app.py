import os, io, tempfile, logging, hashlib, time
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File, Header
from pydantic import BaseModel
import torch
from transformers import SiglipConfig, SiglipForImageClassification, AutoImageProcessor
from PIL import Image
from azure.storage.blob import BlobServiceClient, StorageStreamDownloader
from azure.core.exceptions import ResourceExistsError
from prometheus_client import Counter, Histogram, start_http_server

# ╭──────────────── Config ───────────────╮
load_dotenv()
AZ_CONN     = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER   = "fakenewsdetection-models"
PROD_BLOB   = "image.pt"
SHAD_BLOB   = "shadow_image.pt"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ╰────────────────────────────────────────╯

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Start Prometheus exporter on port 9103
start_http_server(9103)

# Prod metrics per role
PROD_INF   = Counter("model_inferences_total", "Total model inferences", ["role"])
LATENCY    = Histogram("model_latency_seconds", "Inference latency", ["role"])
SH_INF     = Counter("shadow_inferences_total", "Shadow inferences", ["service"])
SH_MATCH   = Counter("shadow_agree_total", "Shadow agrees with prod", ["service"])

#
# Globals
#
CURRENT_ROLE    = "prod"
primary_model   = None
secondary_model = None
processor       = None

def download_blob(name, timeout=300):
    try:
        blob = BlobServiceClient.from_connection_string(AZ_CONN)\
               .get_container_client(CONTAINER).get_blob_client(name)
        stream: StorageStreamDownloader = blob.download_blob(max_concurrency=4, timeout=timeout)
        with tempfile.NamedTemporaryFile(delete=False) as t:
            stream.readinto(t)
            return torch.load(t.name, map_location=device)
    except Exception as e:
        log.warning(f"download {name}: {e}")
        return None

def load_weights(role: str):
    global CURRENT_ROLE, primary_model, secondary_model, processor
    CURRENT_ROLE = role
    cfg = SiglipConfig.from_pretrained("Ateeqq/ai-vs-human-image-detector")
    proc= AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
    processor = proc

    # keep retrying until we successfully download the production blob
    while True:
        prod_sd = download_blob(PROD_BLOB)
        if prod_sd is not None:
            break
        log.warning(f"prod blob missing for role={role}, retrying in 5 seconds")
        time.sleep(5)

    if role == "prod":
        # production mode: only load prod model, no shadow
        prim = SiglipForImageClassification(cfg); prim.load_state_dict(prod_sd, strict=False)
        primary_model = prim.to(device).eval()
        secondary_model = None
    else:  # shadow role
        # shadow mode: download and load shadow blob
        shad_sd = download_blob(SHAD_BLOB)
        if shad_sd is None:
            raise RuntimeError("shadow blob missing for shadow role")
        prim = SiglipForImageClassification(cfg); prim.load_state_dict(prod_sd,  strict=False)
        sec  = SiglipForImageClassification(cfg); sec.load_state_dict(shad_sd, strict=False)
        primary_model   = prim.to(device).eval()
        secondary_model = sec.to(device).eval()

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
        SH_INF.labels("image").inc()
        if s_idx == idx:
            SH_MATCH.labels("image").inc()

    out = "AI" if label.lower()=="ai" else "Human"
    # Save user image to Azure Blob with unique hash filename and metadata
    try:
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        container = "fakenewsdetection-ai-imgs" if out == "AI" else "fakenewsdetection-hum-imgs"
        blob_client = BlobServiceClient.from_connection_string(AZ_CONN) \
            .get_container_client(container) \
            .get_blob_client(f"{img_hash}{ext}")
        metadata = out
        blob_client.upload_blob(img_bytes, overwrite=False, metadata=metadata)
    except ResourceExistsError:
        log.info(f"image already exists in blob: {img_hash}{ext}")
    except Exception as e:
        log.error(f"upload user image: {e}")
    return Resp(prediction=out, confidence=round(conf,2))

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

app = FastAPI(title="FakeNews‑Image‑API")
app.include_router(router, tags=["image"])
app.include_router(admin,  tags=["admin"])