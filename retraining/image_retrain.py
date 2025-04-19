"""
retraining/image_retrain.py
Fine‑tune a SigLIP classifier and optionally push to Azure / MLflow.

DRY‑RUN mode
------------
Set the env‑var DRY_RUN=1 and **all** external writes (torch.save, Azure
Blob Storage, MLflow, user‑data reset) are replaced by printouts.
"""

import os, glob, uuid, base64, shutil, tempfile, logging, traceback
from datetime import datetime
from typing import List, Tuple

import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import SiglipConfig, SiglipForImageClassification, AutoImageProcessor
import mlflow, mlflow.pytorch
from azure.storage.blob import BlobServiceClient, BlobBlock
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────
# 0.  Configuration & DRY‑RUN switch
# ──────────────────────────────────────────────────────────────────────
load_dotenv()
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "8"))
EPOCHS             = int(os.getenv("EPOCHS", "5"))
LEARNING_RATE      = float(os.getenv("LEARNING_RATE", "5e-5"))
ACCURACY_THRESHOLD = float(os.getenv("IMAGE_ACCURACY_THRESHOLD", "0.80"))
F1_THRESHOLD       = float(os.getenv("IMAGE_F1_THRESHOLD", "0.80"))
MODEL_PATH         = os.getenv("MODEL_PATH", "image_model/image.pt")

# MLflow
MLFLOW_IMAGE_TRACKING_URI = os.getenv(
    "MLFLOW_IMAGE_TRACKING_URI",
    "file:retraining/mlflow-image/mlruns"
)
mlflow.set_tracking_uri(MLFLOW_IMAGE_TRACKING_URI)

# Azure
CONNECTION_STRING   = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
MODELS_CONTAINER    = os.getenv("MODELS_CONTAINER", "fakenewsdetection-models")
IMAGE_MODEL_BLOB    = os.getenv("IMAGE_MODEL_BLOB", "image.pt")
AI_IMAGE_CONTAINER  = os.getenv("AI_IMAGE_CONTAINER", "fakenewsdetection-ai-imgs")
HUMAN_IMAGE_CONTAINER = os.getenv("HUMAN_IMAGE_CONTAINER","fakenewsdetection-hum-imgs")

# Local data
DATA_PATH_AI   = os.getenv("DATA_PATH_AI",  "datasets/image_data/ai_user")
DATA_PATH_HUM  = os.getenv("DATA_PATH_HUM", "datasets/image_data/hum_user")
ARCHIVE_DIR_AI = os.getenv("ARCHIVE_DIR_AI", "datasets/image_data/archives")
ARCHIVE_DIR_HUM= os.getenv("ARCHIVE_DIR_HUM","datasets/image_data/archives")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if DRY_RUN:
    logger.info("** DRY‑RUN mode enabled **  — no files, no Azure, no MLflow")

# ──────────────────────────────────────────────────────────────────────
# 1.  Dataset with unified preprocessing
# ──────────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int],
                 processor: AutoImageProcessor, augment=None):
        self.paths, self.labels = paths, labels
        self.processor = processor
        self.augment   = augment

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.augment: img = self.augment(img)

        pix = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return {"pixel_values": pix,
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

# ──────────────────────────────────────────────────────────────────────
# 2.  Helpers: model + data + side‑effect wrappers
# ──────────────────────────────────────────────────────────────────────
def load_data() -> Tuple[List[str], List[int]]:
    img_ext = [".jpg", ".jpeg", ".png", ".webp"]
    folders = {0: ["datasets/image_data/ai_original", DATA_PATH_AI],
               1: ["datasets/image_data/hum_original", DATA_PATH_HUM]}

    paths, labels = [], []
    for label, dirs in folders.items():
        for d in dirs:
            for ext in img_ext:
                new_paths = glob.glob(os.path.join(d, f"*{ext}"))
                paths.extend(new_paths)
                labels.extend([label] * len(new_paths))

    if len(paths) < 10:
        logger.warning("Insufficient images")
        return None, None
    return paths, labels


def load_model_and_processor():
    cfg = SiglipConfig.from_pretrained("Ateeqq/ai-vs-human-image-detector")
    cfg.num_labels = 2
    proc = AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
    model= SiglipForImageClassification(cfg)
    return model, proc, cfg

def safe_torch_save(state_dict, path):
    if DRY_RUN:
        print(f"[DRY‑RUN] would torch.save to {path}")
    else:
        torch.save(state_dict, path)

# ──────────────────────────────────────────────────────────────────────
# 3.  Azure helpers (no‑ops in DRY mode)
# ──────────────────────────────────────────────────────────────────────
def push_to_production(model):
    if DRY_RUN:
        print("[DRY‑RUN] would push model to Azure")
        return True
    if not CONNECTION_STRING:
        logger.error("No Azure connection string")
        return False
    try:
        blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container    = blob_service.get_container_client(MODELS_CONTAINER)
        blob         = container.get_blob_client(IMAGE_MODEL_BLOB)

        tmp = tempfile.NamedTemporaryFile(delete=False).name
        torch.save(model.state_dict(), tmp)

        block_ids = []
        with open(tmp,"rb") as f:
            while chunk := f.read(4*1024*1024):
                block_id = base64.b64encode(uuid.uuid4().hex.encode()).decode()
                blob.stage_block(block_id=block_id, data=chunk)
                block_ids.append(BlobBlock(block_id=block_id))
        blob.commit_block_list(block_ids)
        os.unlink(tmp)
        logger.info("Model pushed to Azure")
        return True
    except Exception as e:
        logger.error(f"Azure upload failed: {e}")
        return False

# ──────────────────────────────────────────────────────────────────────
# 3b.  User‑data archiving & reset  (works in both DRY‑RUN and real mode)
# ──────────────────────────────────────────────────────────────────────
def archive_user_data() -> bool:
    """
    Copy every file in datasets/image_data/{ai_user,hum_user}
    to dated sub‑folders inside datasets/image_data/archives/.
    """
    if DRY_RUN:
        print("[DRY‑RUN] would archive user images to datasets/image_data/archives")
        return True

    try:
        if not os.path.exists(DATA_PATH_AI) and not os.path.exists(DATA_PATH_HUM):
            logger.info("No user data to archive.")
            return True

        os.makedirs(ARCHIVE_DIR_AI,  exist_ok=True)
        os.makedirs(ARCHIVE_DIR_HUM, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ------- AI user images -------
        if os.path.exists(DATA_PATH_AI):
            ai_dest = os.path.join(ARCHIVE_DIR_AI, f"ai_user_{stamp}")
            os.makedirs(ai_dest, exist_ok=True)
            for f in glob.glob(os.path.join(DATA_PATH_AI, "*")):
                if os.path.isfile(f):
                    shutil.copy2(f, os.path.join(ai_dest, os.path.basename(f)))
            logger.info(f"AI user images archived → {ai_dest}")

        # ------- Human user images ----
        if os.path.exists(DATA_PATH_HUM):
            hum_dest = os.path.join(ARCHIVE_DIR_HUM, f"hum_user_{stamp}")
            os.makedirs(hum_dest, exist_ok=True)
            for f in glob.glob(os.path.join(DATA_PATH_HUM, "*")):
                if os.path.isfile(f):
                    shutil.copy2(f, os.path.join(hum_dest, os.path.basename(f)))
            logger.info(f"Human user images archived → {hum_dest}")

        return True
    except Exception as e:
        logger.error(f"archive_user_data failed: {e}")
        return False


def reset_user_data() -> bool:
    """
    1. Archive local user images      (see above)
    2. Empty Azure blob containers    (ai & hum)  *if credentials exist*
    3. Optionally wipe local folders  (left as‑is here)
    """
    if DRY_RUN:
        print("[DRY‑RUN] would reset user data (archive + delete Azure blobs)")
        return True

    try:
        archive_user_data()

        if not CONNECTION_STRING:
            logger.warning("No Azure connection string – skipping cloud cleanup")
            return True

        blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)

        for container_name in (AI_IMAGE_CONTAINER, HUMAN_IMAGE_CONTAINER):
            cont = blob_service.get_container_client(container_name)
            blobs = list(cont.list_blobs())

            if not blobs:
                logger.info(f"Azure container '{container_name}' already empty")
                continue

            for blob in blobs:
                cont.delete_blob(blob.name)
            logger.info(f"Deleted {len(blobs)} blobs from '{container_name}'")

        return True
    except Exception as e:
        logger.error(f"reset_user_data failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# 4.  Training & eval
# ──────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optim, crit):
    model.train()
    tot, n = 0.0, 0
    for batch in tqdm(loader, desc="Training"):
        try:
            pix, lab = batch["pixel_values"].to(device), batch["label"].to(device)
            optim.zero_grad()
            loss = crit(model(pixel_values=pix).logits, lab)
            loss.backward(); optim.step()
            tot += loss.item(); n += 1
        except Exception as e:
            logger.error(f"Batch skipped: {e}")
    return tot/max(n,1)

def evaluate(model, loader):
    model.eval()
    p,g=[],[]
    with torch.no_grad():
        for b in tqdm(loader, desc="Evaluating"):
            pix = b["pixel_values"].to(device)
            g  += b["label"].tolist()
            p  += torch.argmax(model(pixel_values=pix).logits,1).cpu().tolist()
    return {"accuracy": accuracy_score(g,p),
            "f1": f1_score(g,p,average="weighted"),
            "precision": precision_score(g,p,average="weighted", zero_division=0),
            "recall": recall_score(g,p,average="weighted", zero_division=0)}

# ──────────────────────────────────────────────────────────────────────
# 5.  Main workflow
# ──────────────────────────────────────────────────────────────────────
def retrain() -> bool:
    try:
        paths, labels = load_data()
        if not paths: return False

        tr_p,tmp_p,tr_l,tmp_l = train_test_split(paths,labels, test_size=0.3,
                                                 stratify=labels,random_state=42)
        va_p,te_p,va_l,te_l = train_test_split(tmp_p,tmp_l, test_size=0.5,
                                               stratify=tmp_l,random_state=42)

        model, proc, _ = load_model_and_processor(); model.to(device)
        aug = transforms.RandomHorizontalFlip()
        train_ds = ImageDataset(tr_p,tr_l,proc,aug)
        val_ds   = ImageDataset(va_p,va_l,proc)
        test_ds  = ImageDataset(te_p,te_l,proc)
        train_dl = DataLoader(train_ds,sampler=RandomSampler(train_ds),batch_size=BATCH_SIZE)
        val_dl   = DataLoader(val_ds,batch_size=BATCH_SIZE)
        test_dl  = DataLoader(test_ds,batch_size=BATCH_SIZE)

        optim = AdamW(model.parameters(), lr=LEARNING_RATE)
        crit  = nn.CrossEntropyLoss()

        mlflow_active = not DRY_RUN
        if mlflow_active:
            mlflow.set_experiment("Image_Retraining")
            mlflow.start_run()
            mlflow.log_params({"epochs":EPOCHS,"batch_size":BATCH_SIZE,"lr":LEARNING_RATE})

        best_f1,best_state=0.0,None
        for ep in range(EPOCHS):
            loss=train_epoch(model,train_dl,optim,crit)
            val = evaluate(model,val_dl)
            logger.info(f"Epoch {ep+1}/{EPOCHS} loss={loss:.4f} val_f1={val['f1']:.4f}")
            if mlflow_active:
                mlflow.log_metrics({f"val_{k}":v for k,v in val.items()}, step=ep)
            if val["f1"]>best_f1:
                best_f1,best_state=val["f1"],model.state_dict().copy()

        if best_state: model.load_state_dict(best_state)
        test = evaluate(model,test_dl)
        logger.info(f"Test metrics: {test}")
        if mlflow_active:
            mlflow.log_metrics({f"test_{k}":v for k,v in test.items()})
            mlflow.log_artifact(MODEL_PATH)

        promoted=False
        if test["accuracy"]>ACCURACY_THRESHOLD and test["f1"]>F1_THRESHOLD:
            promoted=True
            push_to_production(model)
            local_dir=os.path.dirname(MODEL_PATH); os.makedirs(local_dir,exist_ok=True)
            safe_torch_save(model.state_dict(), MODEL_PATH)
            reset_user_data()
            if mlflow_active:
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
                    torch.save(model.state_dict(), tf.name)
                    mlflow.log_artifact(tf.name, artifact_path="models/promoted")
                os.unlink(tf.name)
        else:
            logger.info("Thresholds not met – no promotion")

        if mlflow_active: mlflow.end_run()
        return promoted
    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    exit(0 if retrain() else 1)