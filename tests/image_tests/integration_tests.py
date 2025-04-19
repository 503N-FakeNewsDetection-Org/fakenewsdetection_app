"""
DRY‑RUN integration test for retraining/image_retrain.py
Runs the full retrain() pipeline on 12 tiny pictures.
"""

import os, tempfile, unittest
from unittest.mock import patch
from PIL import Image
import torch, torch.nn as nn

# Force DRY and zero thresholds
os.environ["DRY_RUN"] = "1"
os.environ["IMAGE_ACCURACY_THRESHOLD"] = "0"
os.environ["IMAGE_F1_THRESHOLD"]       = "0"

from retraining import image_retrain as ir

# ──────────────────────────────────────────────────────────────
# Fakes for HF objects
# ──────────────────────────────────────────────────────────────
class FakeProcessor:
    def __call__(self, images, return_tensors=None):
        return {"pixel_values": torch.zeros(3, 224, 224)}

class FakeSiglip(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))      # so optimiser isn’t empty
    def forward(self, pixel_values=None, labels=None):
        bs = pixel_values.shape[0] if pixel_values.ndim == 4 else 1
        logits = self.w.repeat(bs, 2).view(bs, 2)
        return type("Out", (), {"logits": logits})

class FakeConfig(ir.SiglipConfig): pass           # passes config isinstance check

# ──────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────
class IntegrationSmokeTest(unittest.TestCase):

    @patch("retraining.image_retrain.load_model_and_processor",
           return_value=(FakeSiglip(), FakeProcessor(), FakeConfig()))
    @patch("mlflow.start_run")     # keep MLflow silent
    @patch("mlflow.log_params")
    @patch("mlflow.log_metrics")
    @patch("mlflow.log_artifact")
    def test_retrain_dry(self, *mlflow_stubs):
        with tempfile.TemporaryDirectory() as root:
            # Build 3 images per folder => 12 total
            folders = {
                "ai_orig":  os.path.join(root, "datasets/image_data/ai_original"),
                "ai_user":  os.path.join(root, "datasets/image_data/ai_user"),
                "hum_orig": os.path.join(root, "datasets/image_data/hum_original"),
                "hum_user": os.path.join(root, "datasets/image_data/hum_user"),
            }
            for d in folders.values():
                os.makedirs(d, exist_ok=True)
                for i in range(3):
                    Image.new("RGB", (16, 16), "red").save(os.path.join(d, f"{i}.jpg"))

            # Point module to user dirs
            ir.DATA_PATH_AI  = folders["ai_user"]
            ir.DATA_PATH_HUM = folders["hum_user"]
            os.environ["DATA_PATH_AI"]  = folders["ai_user"]
            os.environ["DATA_PATH_HUM"] = folders["hum_user"]

            result = ir.retrain()          # run everything
            self.assertTrue(result)        # promotion forced by thresholds

if __name__ == "__main__":
    unittest.main()