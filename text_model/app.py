import os, tempfile, logging, time
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Header
from pydantic import BaseModel, field_validator
import torch, torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from azure.storage.blob import BlobServiceClient
from prometheus_client import Counter, Histogram, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# ╭──────────────── Config ───────────────╮
load_dotenv()
AZ_CONN     = "DefaultEndpointsProtocol=https;AccountName=modelscsv;AccountKey=hC/W3DHx3nxkNHhbJCiOBRICi56Cy/htx2lWQoI6LRO8hT5mKWVKIoIlEmHte6oLE6003sBGTalp+AStL5lznw==;EndpointSuffix=core.windows.net"
CONTAINER   = "fakenewsdetection-models"
PROD_BLOB   = "model.pt"
SHAD_BLOB   = "shadow_model.pt"
# path where local .pt files may reside
#LOCAL_WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", os.path.dirname(__file__))
ADMIN_TOKEN = "admin@123"            
MAX_LEN     = 15
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ╰────────────────────────────────────────╯

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Start Prometheus exporter on port 9103 for this container
start_http_server(9103)

# Production & latency metrics (labelled by CURRENT_ROLE)
PROD_INF   = Counter("model_inferences_total", "Total model inferences", ["role"])
LATENCY    = Histogram("model_latency_seconds", "Inference latency", ["role"])
# Shadow‑specific global counters (no labels)
SH_INF     = Counter("shadow_inferences_total", "Shadow inferences", [])
SH_MATCH   = Counter("shadow_agree_total", "Shadow agrees with prod", [])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ╭──────────────── Model skeleton ─────────╮
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert      = bert
        self.dropout   = nn.Dropout(0.1)
        self.relu      = nn.ReLU()
        self.fc1       = nn.Linear(768, 512)
        self.fc2       = nn.Linear(512, 2)
        self.logsoft   = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        # exactly as in your notebook
        cls_hs = self.bert(sent_id, attention_mask=mask)["pooler_output"]
        x      = self.fc1(cls_hs)
        x      = self.relu(x)
        x      = self.dropout(x)
        x      = self.fc2(x)
        return self.logsoft(x)
# ╰──────────────────────────────────────────╯

#
# Globals that change when we /admin/role or /admin/reload
#
CURRENT_ROLE   = "prod"      # "prod" | "shadow"
primary_model  = None        # answers returned to user
secondary_model= None        # only for comparison

'''
def _load_local(blob_name):
    "Return state‑dict from local file if present else None."
    path = os.path.join(LOCAL_WEIGHTS_DIR, blob_name)
    if os.path.isfile(path):
        try:
            return torch.load(path, map_location=device)
        except Exception as e:
            log.warning(f"local load {blob_name}: {e}")
    return None
'''

def download_blob(blob_name):
    try:
        svc  = BlobServiceClient.from_connection_string(AZ_CONN)
        blob = svc.get_container_client(CONTAINER).get_blob_client(blob_name)
        with tempfile.NamedTemporaryFile(delete=False) as t:
            blob.download_blob().readinto(t)
            return torch.load(t.name, map_location=device)
    except Exception as e:
        log.warning(f"download {blob_name}: {e}")
        return None

def load_weights(role: str):
    """(Re)load models according to desired role."""
    global primary_model, secondary_model, CURRENT_ROLE
    CURRENT_ROLE = role
    bert = AutoModel.from_pretrained("bert-base-uncased")

    prod_sd  = download_blob(PROD_BLOB)
    while prod_sd is None:
        log.warning("prod blob missing, retrying in 10 seconds")
        time.sleep(10)
        prod_sd = download_blob(PROD_BLOB)

    if role == "prod":
        prim = BERT_Arch(bert); prim.load_state_dict(prod_sd, strict=False)
        primary_model, secondary_model = prim.to(device).eval(), None

    else:  # shadow mode requires both blobs
        shadow_sd = download_blob(SHAD_BLOB)
        if shadow_sd is None:
            raise RuntimeError("shadow blob missing for shadow role")
        prim = BERT_Arch(bert); prim.load_state_dict(prod_sd,  strict=False)
        sec  = BERT_Arch(bert); sec.load_state_dict(shadow_sd, strict=False)
        primary_model, secondary_model = prim.to(device).eval(), sec.to(device).eval()

    log.info(f"Loaded models – role={CURRENT_ROLE}  secondary={'yes' if secondary_model else 'none'}")

# Initial load at startup
load_weights("prod")

# ╭────────────────── FastAPI  ───────────────╮
router = APIRouter()
admin  = APIRouter()

class Req(BaseModel):
    text: str
    @field_validator("text")
    def _not_empty(cls, v): 
        if not v.strip(): raise ValueError("Empty")
        return v.strip()

class Resp(BaseModel):
    prediction: str
    confidence: float

# Feedback schema from gateway
class FeedbackReq(BaseModel):
    text: str
    label: int  # 0=real 1=fake

@router.post("/text", response_model=Resp)
def predict(req: Req):
    if primary_model is None:
        raise HTTPException(503,"Model not loaded")

    toks = tokenizer.batch_encode_plus(
        [req.text], max_length=MAX_LEN,
        padding="max_length", truncation=True, return_tensors="pt")
    ids, mask = toks['input_ids'].to(device), toks['attention_mask'].to(device)

    with torch.no_grad():
        start_t = time.perf_counter()
        out   = primary_model(ids, mask)
        latency = time.perf_counter() - start_t
        probs = torch.exp(out)[0]
        idx   = int(probs.argmax())
        conf  = float(probs[idx])*100
    label = "Fake" if idx==1 else "Real"
    PROD_INF.labels(CURRENT_ROLE).inc()
    LATENCY.labels(CURRENT_ROLE).observe(latency)

    # secondary compare
    if secondary_model:
        with torch.no_grad():
            s_idx = int(torch.argmax(torch.exp(secondary_model(ids, mask)),1).item())
        # shadow statistics
        SH_INF.inc()
        if s_idx == idx:
            SH_MATCH.inc()

    return Resp(prediction=label, confidence=round(conf,2))

@router.post("/feedback/text")
def feedback_text(feed: FeedbackReq):
    """Append verified text to Azure CSV if not already present."""
    try:
        blob_client = BlobServiceClient.from_connection_string(AZ_CONN) \
            .get_container_client("fakenewsdetection-csv") \
            .get_blob_client("user_data.csv")
        try:
            existing = blob_client.download_blob().readall().decode('utf-8')
        except Exception:
            existing = ""
        lines = existing.splitlines()
        updated = False
        for i,l in enumerate(lines):
            if not l.strip():
                continue
            parts = l.split(',')
            if feed.text == parts[0]:
                # update label
                parts[1:] = [str(feed.label)]
                lines[i] = ','.join(parts)
                updated = True
                break
        if not updated:
            lines.append(f"{feed.text.replace(',', ' ')},{feed.label}")
        data = "\n".join(lines) + "\n"
        blob_client.upload_blob(data.encode('utf-8'), overwrite=True)
    except Exception as e:
        log.error(f"feedback upload: {e}")
    return {"status":"ok"}

# ───── Admin endpoints (token‑protected) ─────
def check_admin(token: str | None):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(401,"bad token")

class RoleReq(BaseModel):
    role: str # "prod" | "shadow"

@admin.post("/admin/role")
def set_role(r: RoleReq, x_token: str | None = Header(None)):
    check_admin(x_token)
    load_weights(r.role)
    return {"status":"ok","role":CURRENT_ROLE}

@admin.post("/admin/reload")
def reload_weights(x_token: str | None = Header(None)):
    check_admin(x_token)
    load_weights(CURRENT_ROLE)
    return {"status":"reloaded","role":CURRENT_ROLE}

@admin.get("/admin/status")
def status(x_token: str | None = Header(None)):
    check_admin(x_token)
    return {
        "role": CURRENT_ROLE,
        "secondary": bool(secondary_model)
    }

@admin.get("/metrics")
async def metrics():
    """Expose Prometheus metrics on the API port."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ─────────────────────────────────────────────
app = FastAPI(title="FakeNews‑Text‑API")
app.include_router(router, tags=["text"])
app.include_router(admin,  tags=["admin"])
