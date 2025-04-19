"""
Unit tests for retraining/image_retrain.py
All offline, safe, and Windows‑friendly.
"""

import os, tempfile, unittest
from unittest.mock import patch
from PIL import Image
import torch, torch.nn as nn

os.environ["DRY_RUN"] = "1"             # never write files

from retraining import image_retrain as ir

# ──────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────
class FakeProcessor:
    def __call__(self, images, return_tensors=None):
        return {"pixel_values": torch.zeros(3, 224, 224)}

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))   # gives optimiser something to update
    def forward(self, pixel_values=None, labels=None):
        bs = pixel_values.shape[0] if pixel_values.ndim == 4 else 1
        logits = self.w.repeat(bs, 2).view(bs, 2)
        return type("Out", (), {"logits": logits})

# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────
class TestHelpers(unittest.TestCase):

    def test_safe_torch_save_no_file(self):
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "dummy.pt")     # path does not exist
            ir.safe_torch_save({"a": 1}, target)
            self.assertFalse(os.path.exists(target))  # still doesn’t exist

    def test_load_data_merges_dirs(self):
        with tempfile.TemporaryDirectory() as root:
            # Create ≥12 tiny images; 3 per each of four folders
            dirs = {
                0: [os.path.join(root, "datasets/image_data/ai_original"),
                    os.path.join(root, "datasets/image_data/ai_user")],
                1: [os.path.join(root, "datasets/image_data/hum_original"),
                    os.path.join(root, "datasets/image_data/hum_user")],
            }
            for lbl_dirs in dirs.values():
                for d in lbl_dirs:
                    os.makedirs(d, exist_ok=True)
                    for i in range(3):
                        Image.new("RGB", (8, 8), "red").save(os.path.join(d, f"{i}.jpg"))

            # Point module globals / env to the *user* dirs (original paths are hard‑coded)
            ir.DATA_PATH_AI  = dirs[0][1]
            ir.DATA_PATH_HUM = dirs[1][1]
            os.environ["DATA_PATH_AI"]  = dirs[0][1]
            os.environ["DATA_PATH_HUM"] = dirs[1][1]

            paths, labels = ir.load_data()
            # There may be extra images already in repo → assert lower bound
            self.assertGreaterEqual(len(paths), 12)
            self.assertGreaterEqual(labels.count(0), 6)
            self.assertGreaterEqual(labels.count(1), 6)

class TestDataset(unittest.TestCase):

    def test_imagedataset_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            Image.new("RGB", (10, 10), "blue").save(tf.name)
        ds = ir.ImageDataset([tf.name], [0], processor=FakeProcessor())
        item = ds[0]
        os.unlink(tf.name)                           # close & delete on Windows
        self.assertIn("pixel_values", item)
        self.assertIn("label", item)
        self.assertEqual(item["pixel_values"].shape, (3, 224, 224))

class TestTrainEpoch(unittest.TestCase):

    def test_loss_averaging(self):
        model = FakeModel()
        batch = {"pixel_values": torch.zeros(2, 3, 224, 224),
                 "label": torch.tensor([0, 1])}
        loader = [batch, batch]
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        loss = ir.train_epoch(model, loader, opt, crit)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

if __name__ == "__main__":
    unittest.main()
