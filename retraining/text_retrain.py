"""
retraining/text_retrain.py
Retrains a BERT fake‑news classifier for titles.

New in this version
-------------------
• DRY_RUN=1 env var disables all external writes  
• Azure upload guarded when creds absent  
• Final promoted weights logged to MLflow  
• Loss averaging tolerant to skipped batches  
• Tokenizer & BERT loaded inside retrain()
"""

import os, uuid, base64, shutil, tempfile, logging, traceback
from datetime import datetime

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow, mlflow.pytorch
from azure.storage.blob import BlobServiceClient, BlobBlock
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────
# 0.  Config / DRY‑RUN
# ──────────────────────────────────────────────────────────────────────
load_dotenv()
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "8"))
EPOCHS             = int(os.getenv("EPOCHS", "2"))
MAX_LENGTH         = int(os.getenv("MAX_LENGTH", "15"))
LEARNING_RATE      = float(os.getenv("LEARNING_RATE", "1e-5"))
ACC_THRESH         = float(os.getenv("TEXT_ACCURACY_THRESHOLD", "0.85"))
F1_THRESH          = float(os.getenv("TEXT_F1_THRESHOLD",  "0.85"))

MLFLOW_URI         = os.getenv("MLFLOW_TEXT_TRACKING_URI", "file:retraining/mlflow-text/mlruns")
mlflow.set_tracking_uri(MLFLOW_URI)

AZURE_CONN         = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
MODELS_CONTAINER   = os.getenv("MODELS_CONTAINER", "fakenewsdetection-models")
MODEL_BLOB         = os.getenv("TEXT_MODEL_BLOB", "model.pt")
CSV_CONTAINER      = os.getenv("CSV_CONTAINER",  "fakenewsdetection-csv")
CSV_BLOB           = os.getenv("CSV_BLOB",       "user_data.csv")

DATA_PATH          = os.getenv("TEXT_DATA_PATH",  "datasets/text_data/user_data.csv")
ARCHIVE_DIR        = os.getenv("TEXT_ARCHIVE_DIR","datasets/text_data/archives")
MODEL_PATH         = os.getenv("TEXT_MODEL_PATH", "text_model/model.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
if DRY_RUN:
    logger.info("** DRY‑RUN mode – no files, no Azure, no MLflow **")

# ──────────────────────────────────────────────────────────────────────
# 1.  Helpers
# ──────────────────────────────────────────────────────────────────────
def safe_torch_save(sd, path):
    if DRY_RUN:
        print(f"[DRY‑RUN] would torch.save to {path}")
    else:
        torch.save(sd, path)

def load_data() -> pd.DataFrame:
    true_df = pd.read_csv("datasets/text_data/original_real.csv")[["title"]].copy()
    fake_df = pd.read_csv("datasets/text_data/original_fake.csv")[["title"]].copy()
    true_df["label"] = 0; fake_df["label"] = 1
    original = pd.concat([true_df, fake_df], ignore_index=True)

    user_df = pd.read_csv(DATA_PATH)
    if user_df.empty or not {"title","label"}.issubset(user_df.columns):
        raise ValueError("User CSV must contain non‑empty 'title' & 'label'")
    data = pd.concat([original, user_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
    return data

# ──────────────────────────────────────────────────────────────────────
# 2.  Model definition
# ──────────────────────────────────────────────────────────────────────
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Linear(768,512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512,2),  nn.LogSoftmax(dim=1)
        )
    def forward(self, ids, mask):
        pooled = self.bert(ids, attention_mask=mask)["pooler_output"]
        return self.classifier(pooled)

# ──────────────────────────────────────────────────────────────────────
# 3.  Azure helpers
# ──────────────────────────────────────────────────────────────────────
def push_to_azure(model) -> bool:
    if DRY_RUN or not AZURE_CONN:
        print("[DRY‑RUN] skip Azure upload");  return True
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_CONN)
        container    = blob_service.get_container_client(MODELS_CONTAINER)
        blob         = container.get_blob_client(MODEL_BLOB)

        tmp = tempfile.NamedTemporaryFile(delete=False).name
        torch.save(model.state_dict(), tmp)

        blocks=[]
        with open(tmp,"rb") as f:
            while chunk := f.read(4*1024*1024):
                b64 = base64.b64encode(uuid.uuid4().hex.encode()).decode()
                blob.stage_block(b64, chunk)
                blocks.append(BlobBlock(block_id=b64))
        blob.commit_block_list(blocks)
        os.unlink(tmp)
        logger.info("Model uploaded to Azure")
        return True
    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        return False

def reset_user_data():
    if DRY_RUN:
        print("[DRY‑RUN] would archive & reset user CSV on Azure")
        return True
    try:
        # archive local file
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(DATA_PATH, os.path.join(ARCHIVE_DIR, f"user_data_{stamp}.csv"))
        # reset Azure CSV
        if AZURE_CONN:
            blob_service = BlobServiceClient.from_connection_string(AZURE_CONN)
            blob = blob_service.get_container_client(CSV_CONTAINER).get_blob_client(CSV_BLOB)
            pd.DataFrame(columns=["title","label"]).to_csv("tmp_empty.csv", index=False)
            with open("tmp_empty.csv","rb") as f: blob.upload_blob(f, overwrite=True)
            os.remove("tmp_empty.csv")
        return True
    except Exception as e:
        logger.error(f"reset_user_data failed: {e}")
        return False

# ──────────────────────────────────────────────────────────────────────
# 4.  Train / eval loops
# ──────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, crit):
    model.train(); tot,n=0.0,0
    for ids,mask,lbl in loader:
        ids,mask,lbl = ids.to(device),mask.to(device),lbl.to(device)
        opt.zero_grad(); out=model(ids,mask); loss=crit(out,lbl)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tot+=loss.item(); n+=1
    return tot/max(n,1)

def eval_epoch(model, loader, crit):
    model.eval(); tot,n=0.0,0
    with torch.no_grad():
        for ids,mask,lbl in loader:
            ids,mask,lbl = ids.to(device),mask.to(device),lbl.to(device)
            loss = crit(model(ids,mask), lbl)
            tot+=loss.item(); n+=1
    return tot/max(n,1)

# ──────────────────────────────────────────────────────────────────────
# 5.  Main
# ──────────────────────────────────────────────────────────────────────
def retrain() -> bool:
    try:
        data = load_data()
        if len(data) < 10:
            logger.warning("Not enough data"); return False
        reset_user_data()

        train_txt, tmp_txt, train_lbl, tmp_lbl = train_test_split(
            data["title"], data["label"], test_size=0.3, stratify=data["label"], random_state=42)
        val_txt, test_txt, val_lbl, test_lbl = train_test_split(
            tmp_txt, tmp_lbl, test_size=0.5, stratify=tmp_lbl, random_state=42)

        # Load tokenizer & model here to avoid heavy import time
        from transformers import BertTokenizerFast, AutoModel
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        bert_base = AutoModel.from_pretrained("bert-base-uncased")

        def encode(texts):
            tk = tokenizer.batch_encode_plus(
                texts.tolist(), max_length=MAX_LENGTH, padding="max_length",
                truncation=True, return_tensors="pt")
            return tk["input_ids"], tk["attention_mask"]

        tr_ids,tr_mask = encode(train_txt); val_ids,val_mask = encode(val_txt)
        te_ids,te_mask = encode(test_txt)

        train_set = TensorDataset(tr_ids,tr_mask, torch.tensor(train_lbl.values))
        val_set   = TensorDataset(val_ids,val_mask, torch.tensor(val_lbl.values))
        test_set  = TensorDataset(te_ids,te_mask, torch.tensor(test_lbl.values))

        train_dl = DataLoader(train_set, sampler = RandomSampler(train_set), batch_size=BATCH_SIZE)
        val_dl   = DataLoader(val_set,   sampler = SequentialSampler(val_set), batch_size=BATCH_SIZE)
        test_dl  = DataLoader(test_set,  sampler = SequentialSampler(test_set), batch_size=BATCH_SIZE)

        model = BERT_Arch(bert_base).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        for p in model.bert.parameters(): p.requires_grad=False
        opt  = AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        crit = nn.NLLLoss()

        mlflow_active = not DRY_RUN
        if mlflow_active:
            mlflow.set_experiment("Text_Retraining")
            mlflow.start_run()
            mlflow.log_params({"epochs":EPOCHS,"batch_size":BATCH_SIZE,"lr":LEARNING_RATE,"max_len":MAX_LENGTH})

        best_loss, best_sd = float("inf"), None
        for ep in range(EPOCHS):
            tl = train_epoch(model,train_dl,opt,crit)
            vl = eval_epoch(model,val_dl,crit)
            logger.info(f"Epoch {ep+1}/{EPOCHS}  train={tl:.3f} val={vl:.3f}")
            if mlflow_active: mlflow.log_metric("val_loss", vl, step=ep)
            if vl < best_loss: best_loss, best_sd = vl, model.state_dict().copy()

        model.load_state_dict(best_sd)
        preds,gold=[],[]
        model.eval()
        with torch.no_grad():
            for ids,mask,lbl in test_dl:
                out = model(ids.to(device), mask.to(device))
                preds += torch.argmax(out,1).cpu().tolist()
                gold  += lbl.tolist()
        metrics = {"accuracy":accuracy_score(gold,preds),
                   "f1":f1_score(gold,preds),
                   "precision":precision_score(gold,preds),
                   "recall":recall_score(gold,preds)}
        if mlflow_active: mlflow.log_metrics({f"test_{k}":v for k,v in metrics.items()})
        logger.info(f"Test metrics: {metrics}")

        # Promotion
        promoted=False
        if metrics["accuracy"]>ACC_THRESH and metrics["f1"]>F1_THRESH:
            promoted=True
            push_to_azure(model)
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            safe_torch_save(model.state_dict(), MODEL_PATH)
            reset_user_data()
            # Log promoted weights
            if mlflow_active:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
                    torch.save(model.state_dict(), tf.name)
                    mlflow.log_artifact(tf.name, artifact_path="models/promoted")
                os.unlink(tf.name)
        else:
            logger.info("Thresholds not met – no promotion")

        if mlflow_active: mlflow.end_run()
        return promoted
    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    exit(0 if retrain() else 1)
