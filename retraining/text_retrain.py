import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModel, BertTokenizerFast, AdamW
import mlflow
import mlflow.pytorch
from azure.storage.blob import BlobServiceClient, BlobBlock
import tempfile
from datetime import datetime
from dotenv import load_dotenv
import uuid
import shutil

# Load environment variables from .env file
load_dotenv()

# Get environment variables with defaults
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
EPOCHS = int(os.getenv('EPOCHS', '3'))
MAX_LENGTH = int(os.getenv('MAX_LENGTH', '15'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '1e-5'))
ACCURACY_THRESHOLD = float(os.getenv('ACCURACY_THRESHOLD', '0.88'))
F1_THRESHOLD = float(os.getenv('F1_THRESHOLD', '0.88'))

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:retraining/mlflow-text/mlruns')
print(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Azure Blob Storage configuration
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODELS_CONTAINER = os.getenv('MODELS_CONTAINER', 'fakenewsdetection-models')
MLFLOW_CONTAINER = os.getenv('MLFLOW_CONTAINER', 'fakenewsdetection-mlflow')  
CSV_CONTAINER = os.getenv('CSV_CONTAINER', 'fakenewsdetection-csv')
MODEL_BLOB = os.getenv('MODEL_BLOB', 'model.pt')
CSV_BLOB = os.getenv('CSV_BLOB', 'user_data.csv')

# Data path configuration
DATA_PATH = os.getenv('DATA_PATH', 'datasets/text_data/user_data.csv')
ARCHIVE_DIR = os.getenv('ARCHIVE_DIR', 'datasets/text_data/')

# Model path configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'text_model/model.pt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 1. Data Loading 
# =============================================================================
def load_data_from_file(data_path=DATA_PATH):
    """Load user data from local file."""
    try:
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        if data.empty:
            raise ValueError("Data file is empty")
        if 'title' not in data.columns or 'label' not in data.columns:
            raise ValueError("Data file must contain 'title' and 'label' columns")
        print(f"Successfully loaded training data from local file: {len(data)} samples")
        return data
    except Exception as e:
        print(f"Error loading data from local file: {e}")
        raise

# =============================================================================
# 2. Tokenization Setup
# =============================================================================
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased')

def tokenize_texts(texts):
    """Tokenize input texts using BERT tokenizer."""
    if texts is None or len(texts) == 0:
        raise ValueError("Input texts cannot be None or empty")
    tokens = tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return tokens['input_ids'], tokens['attention_mask']

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
# 4. Functions for Model Loading, Production Push, and Data Management
# =============================================================================
def load_model_from_file(model_path=None):
    """Load model from local file."""
    try:
        model = BERT_Arch(bert).to(device)
        actual_model_path = model_path if model_path is not None else MODEL_PATH
        model.load_state_dict(torch.load(actual_model_path, map_location=device), strict=False)
        print(f"Successfully loaded model from {actual_model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path if model_path else MODEL_PATH}: {e}")
        return None

def push_to_production(model):
    """Uploads the model to Azure Blob Storage using chunked upload."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
        blob_client = container_client.get_blob_client(MODEL_BLOB)
        
        # Create a tempfile and close it after saving the model to it
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()  # Close immediately to avoid file locking issues
        
        # Save model to the closed temp file
        torch.save(model.state_dict(), temp_path)
        
        # Process in chunks
        block_list = []
        chunk_size = 4 * 1024 * 1024  # 4MB
        with open(temp_path, 'rb') as f:
            while True:
                read_data = f.read(chunk_size)
                if not read_data:
                    break
                blk_id = str(uuid.uuid4())
                blob_client.stage_block(block_id=blk_id, data=read_data)
                block_list.append(BlobBlock(block_id=blk_id))
        
        # Clean up the temp file
        os.unlink(temp_path)
        
        # Commit the blocks
        blob_client.commit_block_list(block_list)
        print(f"Model successfully uploaded to Azure Blob Storage: {MODEL_BLOB}")
        return True
    except Exception as e:
        print(f"Error uploading model to Azure Blob Storage: {str(e)}")
        # Try to clean up if possible
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass
        print(f"Container: {MODELS_CONTAINER}, Blob: {MODEL_BLOB}")
        raise
    
def promote_to_production(model, metrics, temp_dir):
    """
    Promotes a model to production if it meets performance criteria.
    
    Args:
        model: The model to potentially promote to production
        metrics: Dictionary of evaluation metrics
        temp_dir: Temporary directory for file operations
        
    Returns:
        bool: True if model was promoted, False otherwise
    """
    if metrics["test_accuracy"] > ACCURACY_THRESHOLD and metrics["test_f1"] > F1_THRESHOLD:
        print("New model meets performance thresholds. Promoting to production...")
        
        # Push model to Azure
        push_to_production(model)
        
        # Also save the model to local file for Docker deployment
        local_model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(local_model_dir, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model also saved locally to {MODEL_PATH} for Docker deployment")
        
        print(f"Model promoted with accuracy: {metrics['test_accuracy']:.3f} and F1: {metrics['test_f1']:.3f}")
        mlflow.log_metric("production_model_promoted", 1)
        reset_user_data()
        return True
    else:
        print("New model did not meet performance thresholds. No promotion.")
        print(f"Required: accuracy > {ACCURACY_THRESHOLD:.3f} and F1 > {F1_THRESHOLD:.3f}")
        print(f"Actual: accuracy = {metrics['test_accuracy']:.3f}, F1 = {metrics['test_f1']:.3f}")
        mlflow.log_metric("production_model_promoted", 0)
        return False

def archive_user_data(data_path=None, archive_dir=None):
    """Archives the current user data before resetting.
    
    Args:
        data_path (str, optional): Path to the data file to archive. Defaults to DATA_PATH.
        archive_dir (str, optional): Directory to store the archive. Defaults to ARCHIVE_DIR.
        
    Returns:
        bool: True if archiving was successful, False otherwise.
    """
    try:
        # Use provided paths or defaults
        actual_data_path = data_path if data_path is not None else DATA_PATH
        actual_archive_dir = archive_dir if archive_dir is not None else ARCHIVE_DIR
        
        # Create archive directory if it doesn't exist
        os.makedirs(actual_archive_dir, exist_ok=True)
        
        # Archive the current user data with date stamp
        current_date = datetime.now().strftime("%d%m%Y")
        archive_path = os.path.join(actual_archive_dir, f"user_data_{current_date}.csv")
        
        # Copy the current data to archive
        shutil.copy2(actual_data_path, archive_path)
        print(f"User data archived to {archive_path}")
        return True
    except Exception as e:
        print(f"Error archiving user data: {e}")
        return False

def reset_user_data():
    """Reset the user data CSV in Azure Blob Storage."""
    try:
        # First archive the existing data
        archive_user_data(DATA_PATH, ARCHIVE_DIR)
        
        # Then reset in Azure
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CSV_CONTAINER)
        blob_client = container_client.get_blob_client(CSV_BLOB)
        empty_df = pd.DataFrame(columns=['title', 'label'])
        
        # Create a temp file, close it, then write to it to avoid file locking
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()  # Close immediately to avoid file locking issues
        
        # Write to the closed temp file
        empty_df.to_csv(temp_path, index=False)
        
        # Upload and clean up
        with open(temp_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        # Remove the temp file
        os.unlink(temp_path)
        
        print("User data CSV reset in Azure Blob Storage")
        return True
    except Exception as e:
        print(f"Error resetting user data: {e}")
        # Try to clean up if possible
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass
        return False

# =============================================================================
# 5. Training and Evaluation Functions
# =============================================================================
def train_epoch(model, train_dataloader, optimizer, criterion):
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

def evaluate_epoch(model, val_dataloader, criterion):
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
# 6. Main Retraining Function
# =============================================================================
def retrain():
    """Main retraining function that handles the entire workflow."""
    try:
        # Load training data
        data = load_data_from_file()
        if data is None or len(data) < 10:  # Ensure enough data for training
            print("Insufficient data for retraining. Exiting...")
            return False
        
        reset_user_data()
        
        # Split data
        data = data.sample(frac=1).reset_index(drop=True)  # Shuffle
        train_text, temp_text, train_labels, temp_labels = train_test_split(
            data['title'], data['label'], test_size=0.3, random_state=42, stratify=data['label']
        )
        val_text, test_text, val_labels, test_labels = train_test_split(
            temp_text, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        # Tokenize
        train_seq, train_mask = tokenize_texts(train_text)
        val_seq, val_mask = tokenize_texts(val_text)
        test_seq, test_mask = tokenize_texts(test_text)
        
        # Convert labels to tensors
        train_y = torch.tensor(train_labels.tolist())
        val_y = torch.tensor(val_labels.tolist())
        test_y = torch.tensor(test_labels.tolist())
        
        # Create dataloaders
        train_data = TensorDataset(train_seq, train_mask, train_y)
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
        val_data = TensorDataset(val_seq, val_mask, val_y)
        val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)
        
        # Use a temporary directory for file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load existing model; if not found, initialize a new model
            model = load_model_from_file(MODEL_PATH)
            if model is None:
                print("No existing model found.")
                exit(1)
            
            # Freeze BERT parameters; update only classifier layers
            for param in model.bert.parameters():
                param.requires_grad = False
            
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.NLLLoss()
            
            # MLflow experiment setup
            EXPERIMENT_NAME = "Text_Retraining"
            experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if experiment is None:
                experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
                print(f"Created new MLflow experiment: {EXPERIMENT_NAME}")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing MLflow experiment: {EXPERIMENT_NAME}")
            
            mlflow.set_experiment(EXPERIMENT_NAME)
            
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log metadata and hyperparameters
                current_date = datetime.now().strftime("%d%m%Y")
                mlflow.set_tag("mlflow.runName", f"retrain-{current_date}")
                mlflow.set_tag("mlflow.source.type", "PROJECT")
                mlflow.set_tag("mlflow.source.git.commit", os.getenv('GIT_COMMIT', 'unknown'))
                mlflow.set_tag("mlflow.source.git.branch", os.getenv('GIT_BRANCH', 'unknown'))
                
                mlflow.log_param("epochs", EPOCHS)
                mlflow.log_param("batch_size", BATCH_SIZE)
                mlflow.log_param("learning_rate", LEARNING_RATE)
                mlflow.log_param("max_length", MAX_LENGTH)
                
                # Training loop
                best_valid_loss = float('inf')
                best_model_path = os.path.join(temp_dir, "best_model.pt")
                best_model_state = None
                
                for epoch in range(EPOCHS):
                    print(f"\nEpoch {epoch+1} of {EPOCHS}")
                    train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
                    valid_loss = evaluate_epoch(model, val_dataloader, criterion)
                    print(f"Training Loss: {train_loss:.3f}")
                    print(f"Validation Loss: {valid_loss:.3f}")
                    
                    # Save best model if validation improves
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        
                        # Create a date-based folder name
                        current_date = datetime.now().strftime("%d%m%Y")
                        
                        # Save the model state dict
                        best_model_path = os.path.join(temp_dir, "best_model.pt")
                        torch.save(model.state_dict(), best_model_path)
                        
                        # Store the best model state for later use
                        best_model_state = {key: val.cpu().clone() for key, val in model.state_dict().items()}
                        
                        # Log the artifact to MLflow with date in path
                        mlflow.log_artifact(best_model_path, artifact_path=f"models/best_{current_date}")
                        print(f"New best model saved to models/best_{current_date}/best_model.pt!")
                
                # Restore the best model for evaluation
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("Restored best model for evaluation")
                
                # Evaluate best model on test set
                model.eval()
                test_seq_device = test_seq.to(device)
                test_mask_device = test_mask.to(device)
                with torch.no_grad():
                    outputs = model(test_seq_device, test_mask_device)
                    preds = torch.argmax(outputs, dim=1)
                preds = preds.cpu().numpy()
                
                # Calculate metrics
                metrics = {
                    "test_accuracy": accuracy_score(test_y, preds),
                    "test_f1": f1_score(test_y, preds),
                    "test_precision": precision_score(test_y, preds),
                    "test_recall": recall_score(test_y, preds)
                }
                
                # Log metrics and promote if successful
                mlflow.log_metrics(metrics)
                print("Final Test Metrics for Best Model:")
                for name, value in metrics.items():
                    print(f"{name}: {value:.3f}")
                
                # Promote the best model (not the final one)
                promoted = promote_to_production(model, metrics, temp_dir)
                
                # Log the best model with date-based versioning using more efficient API
                current_date = datetime.now().strftime("%d%m%Y")
                mlflow.pytorch.log_model(model, f"models/production_{current_date}")
                print(f"Final model saved to models/production_{current_date}")
                
                return promoted
                
    except Exception as e:
        print(f"Error in retraining process: {str(e)}")
        raise

# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    success = retrain()
    exit(0 if success else 1)