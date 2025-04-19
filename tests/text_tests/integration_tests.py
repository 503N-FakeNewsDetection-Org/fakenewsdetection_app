"""
test_text_integration.py
Light integration test: run tr.retrain() end‑to‑end with DRY_RUN.
Patches everything heavy so it runs in <10 s with no internet.
"""

import os, tempfile, pandas as pd, unittest
from unittest.mock import patch, MagicMock
import torch

os.environ["DRY_RUN"] = "1"
os.environ["TEXT_ACCURACY_THRESHOLD"] = "0"   # ensure promotion
os.environ["TEXT_F1_THRESHOLD"]       = "0"

from retraining import text_retrain as tr

# --- Fake tokenizer / model --------------------------------------- #
def fake_tokenizer(*a, **k):
    texts = a[0] if a else k["texts"]
    n, max_len = len(texts), k.get("max_length",15)
    return {"input_ids":torch.zeros(n,max_len,dtype=torch.long),
            "attention_mask":torch.ones(n,max_len,dtype=torch.long)}

class FakeBert(torch.nn.Module):
    def forward(self, ids, attention_mask=None):
        return {"pooler_output": torch.zeros(ids.size(0),768)}

# ------------------------------------------------------------------ #
class IntegrationSmokeTest(unittest.TestCase):

    @patch("transformers.BertTokenizerFast.from_pretrained", return_value=MagicMock(batch_encode_plus=fake_tokenizer))
    @patch("transformers.AutoModel.from_pretrained",        return_value=FakeBert())
    @patch("mlflow.start_run")    # suppress mlflow for speed
    @patch("mlflow.log_params")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_artifact")
    def test_retrain_dry(self,*patches):
        # synthetic user csv (5 real, 5 fake)
        tmpcsv = tempfile.NamedTemporaryFile(delete=False,suffix=".csv").name
        pd.DataFrame({"title":[f"t{i}" for i in range(10)],
                      "label":[0]*5+[1]*5}).to_csv(tmpcsv,index=False)
        os.environ["TEXT_DATA_PATH"] = tmpcsv

        result = tr.retrain()
        self.assertTrue(result)     # thresholds are 0 → promotion path

if __name__ == "__main__":
    unittest.main()