"""
test_text_unit.py
Unit‑level tests for retraining/text_retrain.py
– no internet, no Azure, no MLflow.
"""

import os, tempfile, pandas as pd, torch, unittest
from unittest.mock import patch

# ------------------------------------------------------------------ #
# 1) Prepare environment
# ------------------------------------------------------------------ #
os.environ["DRY_RUN"] = "1"          # safety

from retraining import text_retrain as tr

# ------------------------------------------------------------------ #
# 2) Synthetic helpers
# ------------------------------------------------------------------ #
def fake_tokenizer_encode_plus(texts, *a, **k):
    n = len(texts)
    max_len = k.get("max_length", 15)
    return {
        "input_ids":      torch.randint(0,100,(n,max_len)),
        "attention_mask": torch.ones(n,max_len, dtype=torch.long)
    }

class FakeBert(torch.nn.Module):
    def forward(self, ids, attention_mask=None):
        bs = ids.shape[0]
        # pooler_output shape (bs,768)
        return {"pooler_output": torch.zeros(bs,768)}

# ------------------------------------------------------------------ #
# 3) Tests
# ------------------------------------------------------------------ #
class TestHelpers(unittest.TestCase):

    def test_safe_torch_save_no_file(self):
        # Create a path inside a temp dir, but DO NOT touch the file
        with tempfile.TemporaryDirectory() as td:
            tmp_path = os.path.join(td, "dummy.pt")
            tr.safe_torch_save({"x": 1}, tmp_path)
            self.assertFalse(os.path.exists(tmp_path))  

    def test_load_data(self):
        # create minimal user csv
        with tempfile.NamedTemporaryFile("w",delete=False,suffix=".csv") as tf:
            pd.DataFrame({"title":["a","b"],"label":[0,1]}).to_csv(tf.name,index=False)
            os.environ["TEXT_DATA_PATH"] = tf.name
            df = tr.load_data()
            self.assertGreater(len(df), 2)   # original + user rows
        os.unlink(tf.name)

class TestModel(unittest.TestCase):

    @patch("transformers.AutoModel.from_pretrained", return_value=FakeBert())
    def test_bert_arch_forward(self, _):
        bert = FakeBert()
        model = tr.BERT_Arch(bert)
        out = model(torch.zeros(2,15,dtype=torch.long), torch.ones(2,15,dtype=torch.long))
        self.assertEqual(out.shape, (2,2))

class TestTrainLoop(unittest.TestCase):

    def setUp(self):
        self.model = tr.BERT_Arch(FakeBert())
        self.ids   = torch.zeros(4,15,dtype=torch.long)
        self.mask  = torch.ones(4,15,dtype=torch.long)
        self.lbl   = torch.tensor([0,1,0,1])
        ds = torch.utils.data.TensorDataset(self.ids,self.mask,self.lbl)
        self.dl = torch.utils.data.DataLoader(ds,batch_size=2)
        self.opt  = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.crit = torch.nn.NLLLoss()

    def test_train_epoch_avg(self):
        loss = tr.train_epoch(self.model,self.dl,self.opt,self.crit)
        self.assertIsInstance(loss,float)
        self.assertGreaterEqual(loss,0)

if __name__ == "__main__":
    unittest.main()
