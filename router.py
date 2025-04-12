from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, AutoModel
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import tempfile
import os
import time
import logging
import pandas as pd

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure Blob Storage configuration
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODELS_CONTAINER = "fakenewsdetection-models"
CSV_CONTAINER = "fakenewsdetection-csv"
MODEL_BLOB = "model.pt"
CSV_BLOB = "user_data.csv"

# Constants
MODEL_CHECK_INTERVAL = 300  # Check for new model every 5 minutes
MAX_LENGTH = 15  # Same as in training code

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the text model architecture
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

# Create router instance for text detection
router = APIRouter(prefix="/text")

# Global variables for model management
model = None
last_model_check = None
current_model_version = None  # Will store the last modification time

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def load_model_from_storage():
    """Load model from Azure Blob Storage and return the model instance"""
    global model, current_model_version
    
    try:
        # Create BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
        blob_client = container_client.get_blob_client(MODEL_BLOB)
        
        # Get blob properties to check if it's a new version
        properties = blob_client.get_blob_properties()
        last_modified = properties.last_modified
        
        # Only proceed with model operations if we have a new version
        if last_modified != current_model_version:
            logger.info(f"New model version detected. Last modified: {last_modified}")
            
            # Initialize model with BERT base
            bert = AutoModel.from_pretrained('bert-base-uncased')
            new_model = BERT_Arch(bert)
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # Download blob to temp file
                download_stream = blob_client.download_blob()
                download_stream.readinto(temp_file)
                
                # Load model from temp file
                new_model.load_state_dict(torch.load(temp_file.name, map_location=device), strict=False)
                os.unlink(temp_file.name)
            
            # Move model to appropriate device
            new_model = new_model.to(device)
            new_model.eval()
            model = new_model
            current_model_version = last_modified
            logger.info(f"Model loaded successfully. Version timestamp: {current_model_version}")
        else:
            logger.info("Model is up to date")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

def save_user_input(text: str, prediction: str):
    """Save user input and prediction to Azure Blob Storage"""
    try:
        # Create DataFrame with new input
        new_data = pd.DataFrame({
            'title': [text],
            'label': [1 if prediction == "Fake" else 0]  # Convert prediction to label
        })
        
        # Create BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CSV_CONTAINER)
        blob_client = container_client.get_blob_client(CSV_BLOB)
        
        try:
            # Try to download existing data
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                download_stream = blob_client.download_blob()
                download_stream.readinto(temp_file)
                existing_data = pd.read_csv(temp_file.name)
                os.unlink(temp_file.name)
                
                # Check if the text already exists
                if text not in existing_data['title'].values:
                    # Append new data
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        combined_data.to_csv(temp_file.name, index=False)
                        with open(temp_file.name, "rb") as data:
                            blob_client.upload_blob(data, overwrite=True)
                        os.unlink(temp_file.name)
                        logger.info(f"New input saved to Azure Blob Storage")
                else:
                    logger.info("Input already exists in user_data.csv")
        except Exception as e:
            # If file doesn't exist or error reading, create new file
            logger.warning(f"Error reading existing file: {e}. Creating new file.")
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                new_data.to_csv(temp_file.name, index=False)
                with open(temp_file.name, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                os.unlink(temp_file.name)
                logger.info("Created new user_data.csv in Azure Blob Storage")
            
    except Exception as e:
        logger.error(f"Error saving user input: {e}")
        # Don't raise the error, just log it

def check_and_load_new_model():
    """Check if model needs to be reloaded"""
    global last_model_check
    
    current_time = time.time()
    if last_model_check is None or (current_time - last_model_check) > MODEL_CHECK_INTERVAL:
        load_model_from_storage()
        last_model_check = current_time

# Define request and response models
class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    prediction: str
    confidence: float
    raw_prediction: int

# Text detection endpoint
@router.post("/predict", response_model=TextResponse)
def predict_text(request: TextRequest):
    # Check for new model before prediction
    check_and_load_new_model()
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Tokenize input text
        tokens = tokenizer.batch_encode_plus(
            [request.text],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        
        # Move tensors to the same device as the model
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            # Convert log probabilities to probabilities
            probabilities = torch.exp(outputs)
            # Get the prediction and its probability
            prediction = torch.argmax(probabilities, dim=1).item()
            # Get the probability of the predicted class
            confidence = probabilities[0][prediction].item()
        
        label = "Fake" if prediction == 1 else "Real"
        logger.info(f"Prediction made: {label} with confidence {confidence:.2f}")
        save_user_input(request.text, label)
        return TextResponse(
            prediction=label,
            confidence=round(confidence * 100, 2),
            raw_prediction=prediction
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize model on startup
load_model_from_storage()
