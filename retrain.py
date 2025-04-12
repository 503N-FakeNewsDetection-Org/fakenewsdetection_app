import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import AutoModel, BertTokenizerFast, AdamW
import mlflow
import mlflow.pytorch
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import tempfile
from datetime import datetime

# MLflow configuration for Azure Blob Storage
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

# Azure Blob Storage configuration for MLflow
MLFLOW_CONTAINER = "fakenewsdetection-mlflow"
MLFLOW_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if MLFLOW_CONNECTION_STRING:
    # Configure MLflow to use Azure Blob Storage
    from azure.storage.blob import BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(MLFLOW_CONNECTION_STRING)
    
    # Create container if it doesn't exist
    try:
        container_client = blob_service_client.create_container(MLFLOW_CONTAINER)
    except Exception as e:
        if "ContainerAlreadyExists" not in str(e):
            print(f"Error creating MLflow container: {e}")
    
    # Set MLflow tracking URI to Azure Blob Storage
    MLFLOW_TRACKING_URI = f"wasbs://{MLFLOW_CONTAINER}@{blob_service_client.account_name}.blob.core.windows.net"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI set to Azure Blob Storage: {MLFLOW_TRACKING_URI}")

# Azure Blob Storage configuration for models and data
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODELS_CONTAINER = "fakenewsdetection-models"
CSV_CONTAINER = "fakenewsdetection-csv"
MODEL_BLOB = "model.pt"
CSV_BLOB = "user_data.csv"
SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID')
RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP')
WORKSPACE_NAME = os.getenv('AZURE_ML_WORKSPACE_NAME')

# Use GPU if available; otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")

# =============================================================================
# 1. Data Loading & Preparation
# =============================================================================
def load_data_from_storage():
    """Load user data from Azure Blob Storage"""
    try:
        # Create BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CSV_CONTAINER)
        blob_client = container_client.get_blob_client(CSV_BLOB)
        
        # Download data to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            download_stream = blob_client.download_blob()
            download_stream.readinto(temp_file)
            data = pd.read_csv(temp_file.name)
            os.unlink(temp_file.name)
            return data
    except Exception as e:
        print(f"Error loading data from Azure Blob Storage: {e}")
        return None

data = load_data_from_storage()
if data is None:
    print("No data found in Azure Blob Storage. Exiting...")
    exit(1)

data = data.sample(frac=1).reset_index(drop=True)

# Split the data into train (70%), validation (15%), and test (15%) sets
train_text, temp_text, train_labels, temp_labels = train_test_split(
    data['title'], data['label'], test_size=0.3, random_state=42, stratify=data['label']
)
val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# =============================================================================
# 2. Tokenization Setup
# =============================================================================
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased')
MAX_LENGTH = 15  # Same token length used in your original training

def tokenize_texts(texts):
    tokens = tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return tokens['input_ids'], tokens['attention_mask']

train_seq, train_mask = tokenize_texts(train_text)
val_seq, val_mask = tokenize_texts(val_text)
test_seq, test_mask = tokenize_texts(test_text)

# Convert labels to tensors (assuming labels are 0 or 1)
train_y = torch.tensor(train_labels.tolist())
val_y = torch.tensor(val_labels.tolist())
test_y = torch.tensor(test_labels.tolist())

batch_size = 32
train_data = TensorDataset(train_seq, train_mask, train_y)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_data = TensorDataset(val_seq, val_mask, val_y)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# =============================================================================
# 3. Model Architecture Definition
# =============================================================================
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

# =============================================================================
# 4. Function to Load Model from Storage
# =============================================================================
def load_model_from_storage():
    """Loads a model from Azure Blob Storage."""
    try:
        # Create BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
        blob_client = container_client.get_blob_client(MODEL_BLOB)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Download blob to temp file
            download_stream = blob_client.download_blob()
            download_stream.readinto(temp_file)
            
            # Initialize model with BERT base
            model = BERT_Arch(bert).to(device)
            # Load the saved state dict
            model.load_state_dict(torch.load(temp_file.name, map_location=device), strict=False)
            os.unlink(temp_file.name)
            return model
    except Exception as e:
        print(f"Error loading model from Azure Blob Storage: {e}")
        return None

def push_to_production(model):
    """Uploads the model to Azure Blob Storage."""
    try:
        # Create BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
        blob_client = container_client.get_blob_client(MODEL_BLOB)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Save model to temp file
            torch.save(model.state_dict(), temp_file.name)
            
            # Upload the model file
            with open(temp_file.name, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            os.unlink(temp_file.name)
            print(f"Model uploaded to Azure Blob Storage: {MODEL_BLOB}")
    except Exception as e:
        print(f"Error uploading model to Azure Blob Storage: {e}")

def reset_user_data():
    """Reset the user data CSV in Azure Blob Storage"""
    try:
        # Create BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CSV_CONTAINER)
        blob_client = container_client.get_blob_client(CSV_BLOB)
        
        # Create empty DataFrame and upload
        empty_df = pd.DataFrame(columns=['title', 'label'])
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            empty_df.to_csv(temp_file.name, index=False)
            with open(temp_file.name, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            os.unlink(temp_file.name)
            print("User data CSV reset in Azure Blob Storage")
    except Exception as e:
        print(f"Error resetting user data: {e}")

# =============================================================================
# 5. Training and Evaluation Functions
# =============================================================================
def train_epoch():
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        sent_id, mask, labels = batch
        optimizer.zero_grad()
        outputs = model(sent_id, mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return total_loss / len(train_dataloader)

def evaluate_epoch():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            sent_id, mask, labels = batch
            outputs = model(sent_id, mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_dataloader)

# =============================================================================
# 6. Load Existing Model and Retrain
# =============================================================================
if __name__ == "__main__":
    # Use a temporary directory for all file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Load the existing model
            model = load_model_from_storage()
            if model is None:
                print("No existing model found. Initializing new model...")
                model = BERT_Arch(bert).to(device)
            
            # Freeze BERT parameters; update only the classifier layers
            for param in model.bert.parameters():
                param.requires_grad = False
            
            optimizer = AdamW(model.parameters(), lr=1e-5)
            criterion = nn.NLLLoss()
            
            # =============================================================================
            # 7. Retraining Loop with MLflow Integration & Model Promotion Decision
            # =============================================================================
            # Set up MLflow experiment
            EXPERIMENT_NAME = "FakeNewsDetection_Retraining"
            epochs = 2
            best_valid_loss = float('inf')
            
            # Try to get the experiment, create it if it doesn't exist
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment is None:
                experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
                print(f"Created new MLflow experiment: {EXPERIMENT_NAME}")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing MLflow experiment: {EXPERIMENT_NAME}")
            
            # Set the experiment
            mlflow.set_experiment(EXPERIMENT_NAME)
            
            # Start the run with tags for CI/CD
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log CI/CD information
                mlflow.set_tag("mlflow.runName", f"retrain-{run.info.run_id[:8]}")
                mlflow.set_tag("mlflow.source.type", "PROJECT")
                mlflow.set_tag("mlflow.source.git.commit", os.getenv('GIT_COMMIT', 'unknown'))
                mlflow.set_tag("mlflow.source.git.branch", os.getenv('GIT_BRANCH', 'unknown'))
                
                # Log hyperparameters only once at the start of the run
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", 1e-5)
                mlflow.log_param("max_length", MAX_LENGTH)
                
                for epoch in range(epochs):
                    print(f"\nEpoch {epoch+1} of {epochs}")
                    train_loss = train_epoch()
                    valid_loss = evaluate_epoch()
                    print(f"Training Loss: {train_loss:.3f}")
                    print(f"Validation Loss: {valid_loss:.3f}")
                    # Log metrics per epoch
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("valid_loss", valid_loss, step=epoch)
                    
                    # Save the best model checkpoint
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        # Save model to temporary file for MLflow
                        temp_model_path = os.path.join(temp_dir, "model.pt")
                        torch.save(model.state_dict(), temp_model_path)
                        mlflow.log_artifact(temp_model_path, artifact_path="models")
                        print("New best model saved!")
                
                # Evaluate final model on test set
                model.eval()
                test_seq_device = test_seq.to(device)
                test_mask_device = test_mask.to(device)
                with torch.no_grad():
                    outputs = model(test_seq_device, test_mask_device)
                    preds = torch.argmax(outputs, dim=1)
                preds = preds.cpu().numpy()
                
                new_accuracy = accuracy_score(test_y, preds)
                new_f1 = f1_score(test_y, preds)
                new_precision = precision_score(test_y, preds)
                new_recall = recall_score(test_y, preds)
                
                # Log final evaluation metrics
                mlflow.log_metric("test_accuracy", new_accuracy)
                mlflow.log_metric("test_f1", new_f1)
                mlflow.log_metric("test_precision", new_precision)
                mlflow.log_metric("test_recall", new_recall)
                
                print("Final Test Metrics:")
                print(f"Accuracy: {new_accuracy:.3f}")
                print(f"F1 Score: {new_f1:.3f}")
                print(f"Precision: {new_precision:.3f}")
                print(f"Recall: {new_recall:.3f}")
                
                # Define static thresholds for model promotion
                ACCURACY_THRESHOLD = 0.88  # 88%
                F1_THRESHOLD = 0.88        # 88%
                
                # Promote the new model if both accuracy and F1 score are above 88%
                if new_accuracy > ACCURACY_THRESHOLD and new_f1 > F1_THRESHOLD:
                    print("New model meets performance thresholds. Promoting to production...")
                    # Save model to temporary file before pushing to production
                    temp_model_path = os.path.join(temp_dir, "model.pt")
                    torch.save(model.state_dict(), temp_model_path)
                    
                    # Create BlobServiceClient
                    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
                    container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
                    blob_client = container_client.get_blob_client(MODEL_BLOB)
                    
                    # Upload the model file
                    with open(temp_model_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                    
                    print(f"Model promoted with accuracy: {new_accuracy:.3f} and F1: {new_f1:.3f}")
                    mlflow.log_metric("production_model_promoted", 1)
                    
                    # Reset user data after successful retraining
                    reset_user_data()
                else:
                    print("New model did not meet performance thresholds. No promotion.")
                    print(f"Required: accuracy > {ACCURACY_THRESHOLD:.3f} and F1 > {F1_THRESHOLD:.3f}")
                    print(f"Actual: accuracy = {new_accuracy:.3f}, F1 = {new_f1:.3f}")
                    mlflow.log_metric("production_model_promoted", 0)
                
                # Log the final model with date-based versioning
                current_date = datetime.now().strftime("%d%m%Y")
                temp_model_path = os.path.join(temp_dir, "model.pt")
                torch.save(model.state_dict(), temp_model_path)
                mlflow.pytorch.log_model(model, f"model_{current_date}")
                
        except Exception as e:
            print(f"Error in retraining process: {str(e)}")
            raise  # Re-raise the exception to ensure proper error handling
            
        

