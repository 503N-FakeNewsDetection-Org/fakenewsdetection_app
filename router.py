from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, AutoModel

# Define the model architecture matching your training notebook
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Create an instance of an APIRouter
router = APIRouter()

# Load the tokenizer and pretrained BERT model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased')

# Instantiate your custom model and load saved weights (adjust the file name/path as needed)
model = BERT_Arch(bert)
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")), strict = False)
model.eval()  # set to evaluation mode

# Define a request model for prediction input
class PredictionRequest(BaseModel):
    text: str

# Create the /text endpoint on this router
@router.post("/text")
def predict_text(request: PredictionRequest):
    MAX_LENGTH = 15  # use the same max length as in training
    tokens = tokenizer.batch_encode_plus(
        [request.text],
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    
    # Run inference without gradients
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    prediction = torch.argmax(outputs, dim=1).item()
    
    # Map prediction to labels (for example, Fake=1 and True=0)
    label = "Fake" if prediction == 1 else "True"
    return {"prediction": label, "raw": prediction}
