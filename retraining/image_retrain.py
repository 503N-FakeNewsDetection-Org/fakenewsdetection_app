import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import SiglipConfig, SiglipForImageClassification, AutoImageProcessor
from PIL import Image
import mlflow
import mlflow.pytorch
from azure.storage.blob import BlobServiceClient, BlobBlock
import tempfile
from datetime import datetime
from dotenv import load_dotenv
import uuid
import shutil
from torchvision import transforms
import glob
from tqdm import tqdm
import logging
import traceback

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get environment variables with defaults
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))
EPOCHS = int(os.getenv('EPOCHS', '5'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '5e-5'))
ACCURACY_THRESHOLD = float(os.getenv('IMAGE_ACCURACY_THRESHOLD', '0.80'))
F1_THRESHOLD = float(os.getenv('IMAGE_F1_THRESHOLD', '0.80'))

# MLflow configuration
MLFLOW_IMAGE_TRACKING_URI = os.getenv('MLFLOW_IMAGE_TRACKING_URI', 'file:retraining/mlflow-image/mlruns')
logger.info(f"Using MLflow tracking URI: {MLFLOW_IMAGE_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_IMAGE_TRACKING_URI)

# Azure Blob Storage configuration
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODELS_CONTAINER = os.getenv('MODELS_CONTAINER', 'fakenewsdetection-models')
IMAGE_MODEL_BLOB = os.getenv('IMAGE_MODEL_BLOB', 'image.pt')
AI_IMAGE_CONTAINER= os.getenv('AI_IMAGE_CONTAINER', "fakenewsdetection-ai-imgs")
HUMAN_IMAGE_CONTAINER= os.getenv('HUMAN_IMAGE_CONTAINER', "fakenewsdetection-hum-imgs")

# Data path configuration
DATA_PATH_AI = os.getenv('DATA_PATH_AI', 'datasets/image_data/ai_user')
DATA_PATH_HUM = os.getenv('DATA_PATH_HUM', 'datasets/image_data/hum_user')
ARCHIVE_DIR_AI = os.getenv('ARCHIVE_DIR_AI', 'datasets/image_data/archives')
ARCHIVE_DIR_HUM = os.getenv('ARCHIVE_DIR_HUM', 'datasets/image_data/archives')

# Model path configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'image_model/image.pt')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# =============================================================================
# 1. Custom Dataset Class
# =============================================================================
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, processor=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.processor = processor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            
        # Apply processor if provided (for inference)
        if self.processor and not self.transform:
            # During inference we use the processor directly
            inputs = self.processor(images=image, return_tensors="pt")
            # Remove batch dimension
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
            inputs['label'] = torch.tensor(label, dtype=torch.long)
            return inputs
        
        # During training we use the transform and return a dict with pixel_values
        return {
            'pixel_values': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# 2. Data Loading and Preprocessing Functions
# =============================================================================
def load_data():
    """Load image data from local directories and combine with original data."""
    try:
        # Define image file extensions to look for
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        # Load original AI images
        ai_original_dir = 'datasets/image_data/ai_original'
        ai_original_paths = []
        if os.path.exists(ai_original_dir):
            for ext in image_extensions:
                ai_original_paths.extend(glob.glob(os.path.join(ai_original_dir, f'*{ext}')))
        else:
            logger.warning(f"Original AI images directory not found: {ai_original_dir}")
        
        # Load original human images
        hum_original_dir = 'datasets/image_data/hum_original'
        hum_original_paths = []
        if os.path.exists(hum_original_dir):
            for ext in image_extensions:
                hum_original_paths.extend(glob.glob(os.path.join(hum_original_dir, f'*{ext}')))
        else:
            logger.warning(f"Original human images directory not found: {hum_original_dir}")
        
        # Load user AI images
        ai_user_paths = []
        if os.path.exists(DATA_PATH_AI):
            for ext in image_extensions:
                ai_user_paths.extend(glob.glob(os.path.join(DATA_PATH_AI, f'*{ext}')))
        else:
            logger.warning(f"User AI images directory not found: {DATA_PATH_AI}")
        
        # Load user human images
        hum_user_paths = []
        if os.path.exists(DATA_PATH_HUM):
            for ext in image_extensions:
                hum_user_paths.extend(glob.glob(os.path.join(DATA_PATH_HUM, f'*{ext}')))
        else:
            logger.warning(f"User human images directory not found: {DATA_PATH_HUM}")
        
        # Combine paths and create labels (0 for AI, 1 for Human)
        all_image_paths = ai_original_paths + ai_user_paths + hum_original_paths + hum_user_paths
        all_labels = [0] * (len(ai_original_paths) + len(ai_user_paths)) + [1] * (len(hum_original_paths) + len(hum_user_paths))
        
        logger.info(f"Total images loaded: {len(all_image_paths)}")
        logger.info(f"AI images: {len(ai_original_paths) + len(ai_user_paths)} (Original: {len(ai_original_paths)}, User: {len(ai_user_paths)})")
        logger.info(f"Human images: {len(hum_original_paths) + len(hum_user_paths)} (Original: {len(hum_original_paths)}, User: {len(hum_user_paths)})")
        
        if len(all_image_paths) < 10:
            logger.warning("Insufficient data for training")
            return None, None
        
        return all_image_paths, all_labels
    
    except Exception as e:
        logger.error(f"Error loading image data: {e}")
        raise

def setup_transformations():
    """Setup image transformations for training and validation."""
    # Define transformations - similar to the fine-tuning approach
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform

# =============================================================================
# 3. Model Loading and Management Functions
# =============================================================================
def load_model_and_processor():
    """Load SigLIP config, processor, and model architecture (without weights)."""
    try:
        # Initialize SigLIP config and processor
        config = SiglipConfig.from_pretrained("Ateeqq/ai-vs-human-image-detector")
        processor = AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
        
        # Create model with the config
        model = SiglipForImageClassification(config)
        
        return model, processor, config
    except Exception as e:
        logger.error(f"Error loading model architecture and processor: {e}")
        return None, None, None

def load_model_from_file(model_path=None):
    """Load SigLIP model and weights from local file."""
    try:
        # Load model architecture and processor
        model, processor, config = load_model_and_processor()
        if model is None or processor is None:
            return None, None, None
        
        # Load state dict from file
        actual_model_path = model_path if model_path is not None else MODEL_PATH
        
        # Check if file exists
        if not os.path.exists(actual_model_path):
            logger.error(f"Model file {actual_model_path} not found.")
            return None, None, None
            
        state_dict = torch.load(actual_model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        logger.info(f"Successfully loaded model from {actual_model_path}")
        return model, processor, config
    except Exception as e:
        logger.error(f"Error loading model from {model_path if model_path else MODEL_PATH}: {e}")
        return None, None, None

def push_to_production(model):
    """Uploads the model to Azure Blob Storage using chunked upload."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
        blob_client = container_client.get_blob_client(IMAGE_MODEL_BLOB)
        
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
        logger.info(f"Model successfully uploaded to Azure Blob Storage: {IMAGE_MODEL_BLOB}")
        return True
    except Exception as e:
        logger.error(f"Error uploading model to Azure Blob Storage: {str(e)}")
        # Try to clean up if possible
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass
        logger.error(f"Container: {MODELS_CONTAINER}, Blob: {IMAGE_MODEL_BLOB}")
        raise

def promote_to_production(model, metrics):
    """
    Promotes a model to production if it meets performance criteria.
    
    Args:
        model: The model to potentially promote to production
        metrics: Dictionary of evaluation metrics
        temp_dir: Temporary directory for file operations
        
    Returns:
        bool: True if model was promoted, False otherwise
    """
    if metrics["accuracy"] > ACCURACY_THRESHOLD and metrics["f1"] > F1_THRESHOLD:
        logger.info("New model meets performance thresholds. Promoting to production...")
        
        # Push model to Azure
        push_to_production(model)
        
        # Also save the model to local file for Docker deployment
        local_model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(local_model_dir, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        logger.info(f"Model also saved locally to {MODEL_PATH} for Docker deployment")
        
        logger.info(f"Model promoted with accuracy: {metrics['test_accuracy']:.3f} and F1: {metrics['test_f1']:.3f}")
        mlflow.log_metric("production_model_promoted", 1)
        reset_user_data()
        return True
    else:
        logger.info("New model did not meet performance thresholds. No promotion.")
        logger.info(f"Required: accuracy > {ACCURACY_THRESHOLD:.3f} and F1 > {F1_THRESHOLD:.3f}")
        logger.info(f"Actual: accuracy = {metrics['test_accuracy']:.3f}, F1 = {metrics['test_f1']:.3f}")
        mlflow.log_metric("production_model_promoted", 0)
        return False

def archive_user_data():
    """Archives the current user data before resetting."""
    try:
        # Verify directories exist
        if not os.path.exists(DATA_PATH_AI) and not os.path.exists(DATA_PATH_HUM):
            logger.warning("User data directories not found, nothing to archive.")
            return True
            
        # Create archive directories if they don't exist
        os.makedirs(ARCHIVE_DIR_AI, exist_ok=True)
        os.makedirs(ARCHIVE_DIR_HUM, exist_ok=True)
        
        # Generate date string for archive folder names
        current_date = datetime.now().strftime("%d%m%Y")
        
        # Archive AI user images
        ai_archive_dir = os.path.join(ARCHIVE_DIR_AI, f"ai_user_{current_date}")
        os.makedirs(ai_archive_dir, exist_ok=True)
        
        # Archive HUM user images
        hum_archive_dir = os.path.join(ARCHIVE_DIR_HUM, f"hum_user_{current_date}")
        os.makedirs(hum_archive_dir, exist_ok=True)
        
        # Check if directories exist before archiving
        if os.path.exists(DATA_PATH_AI):
            # Move AI user images to archive
            for img_file in glob.glob(os.path.join(DATA_PATH_AI, "*")):
                if os.path.isfile(img_file):
                    shutil.copy2(img_file, os.path.join(ai_archive_dir, os.path.basename(img_file)))
            
        if os.path.exists(DATA_PATH_HUM):
            # Move HUM user images to archive
            for img_file in glob.glob(os.path.join(DATA_PATH_HUM, "*")):
                if os.path.isfile(img_file):
                    shutil.copy2(img_file, os.path.join(hum_archive_dir, os.path.basename(img_file)))
        
        logger.info(f"User data archived to {ai_archive_dir} and {hum_archive_dir}")
        return True
    except Exception as e:
        logger.error(f"Error archiving user data: {e}")
        return False

def reset_user_data():
    """Reset (remove) the user data after archiving.
        Archives local data and resets specific Azure storage containers."""
    try:
        # First archive the existing data locally
        archive_user_data()
        
        # Now reset Azure storage containers
        try:
            if CONNECTION_STRING:
                blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

                # Handle AI container
                try:
                    ai_container_client = blob_service_client.get_container_client(AI_IMAGE_CONTAINER)
                    
                    # List all blobs in the container
                    ai_blobs = list(ai_container_client.list_blobs())
                    
                    if ai_blobs:
                        # Delete each blob from Azure
                        deleted_count = 0
                        for blob in ai_blobs:
                            blob_client = ai_container_client.get_blob_client(blob.name)
                            blob_client.delete_blob()
                            deleted_count += 1
                            
                        logger.info(f"Deleted {deleted_count} images from Azure AI container '{AI_IMAGE_CONTAINER}'")
                    else:
                        logger.info(f"Azure AI container '{AI_IMAGE_CONTAINER}' is already empty")
                        
                except Exception as e:
                    logger.error(f"Error resetting Azure AI container: {e}")
                
                # Handle Human container
                try:
                    hum_container_client = blob_service_client.get_container_client(HUMAN_IMAGE_CONTAINER)
                    
                    # List all blobs in the container
                    hum_blobs = list(hum_container_client.list_blobs())
                    
                    if hum_blobs:
                        # Delete each blob from Azure
                        deleted_count = 0
                        for blob in hum_blobs:
                            blob_client = hum_container_client.get_blob_client(blob.name)
                            blob_client.delete_blob()
                            deleted_count += 1
                            
                        logger.info(f"Deleted {deleted_count} images from Azure Human container '{HUMAN_IMAGE_CONTAINER}'")
                    else:
                        logger.info(f"Azure Human container '{HUMAN_IMAGE_CONTAINER}' is already empty")
                        
                except Exception as e:
                    logger.error(f"Error resetting Azure Human container: {e}")
            else:
                logger.warning("No Azure connection string provided, skipping Azure storage reset")
        
        except Exception as e:
            logger.error(f"Error resetting Azure storage: {e}")
        
        logger.info("User image data reset completed (local and Azure)")
        return True
    except Exception as e:
        logger.error(f"Error resetting user data: {e}")
        return False

# =============================================================================
# 4. Training and Evaluation Functions
# =============================================================================
def train_epoch(model, train_dataloader, optimizer):
    model.train()
    total_loss = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for batch in progress_bar:
        try:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            continue
    
    return total_loss / len(train_dataloader)

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            # Ensure labels is explicitly a tensor
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=device)
                
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1)
            
            # Ensure predictions is explicitly a tensor
            if not isinstance(preds, torch.Tensor):
                preds = torch.tensor(preds, device=device)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
    return metrics

# =============================================================================
# 5. Main Retraining Function
# =============================================================================
def retrain():
    """Main retraining function that handles the entire workflow."""
    try:
        # 1. Load data
        image_paths, labels = load_data()
        if image_paths is None or len(image_paths) < 10:
            logger.warning("Insufficient data for retraining. Exiting...")
            return False
        
        # 2. Split data into train/val/test sets
        try:
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                image_paths, labels, test_size=0.3, random_state=42, stratify=labels
            )
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
            )
            
            logger.info(f"Data split - Training: {len(train_paths)}, Validation: {len(val_paths)}, Testing: {len(test_paths)}")
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return False
        
        # 3. Set up transformations
        train_transform, val_transform = setup_transformations()
        
        # 4. Load model, processor and config
        model, processor, config = load_model_from_file(MODEL_PATH)
        if model is None or processor is None or config is None:
            logger.error("Failed to load model, processor, or config. Exiting...")
            return False
        
        # 5. Create datasets
        try:
            train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform, processor=None)
            val_dataset = ImageDataset(val_paths, val_labels, transform=val_transform, processor=None) 
            test_dataset = ImageDataset(test_paths, test_labels, transform=val_transform, processor=None)
            
            # 6. Create dataloaders
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(
                train_dataset, 
                sampler=train_sampler, 
                batch_size=BATCH_SIZE
            )
            
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False
            )
            
            test_dataloader = DataLoader(
                test_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False
            )
        except Exception as e:
            logger.error(f"Error creating datasets or dataloaders: {e}")
            return False
        
        # Use a temporary directory for file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            # 7. Set up optimizer and loss function
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()
            
            # 8. MLflow experiment setup
            try:
                EXPERIMENT_NAME = "Image_Retraining"
                experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
                    logger.info(f"Created new MLflow experiment: {EXPERIMENT_NAME}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {EXPERIMENT_NAME}")
                
                mlflow.set_experiment(EXPERIMENT_NAME)
            except Exception as e:
                logger.error(f"Error setting up MLflow experiment: {e}")
                logger.info("Continuing without MLflow tracking...")
                experiment_id = None
            
            mlflow_active = experiment_id is not None
            
            # Start MLflow run if available
            try:
                if mlflow_active:
                    mlflow_run = mlflow.start_run(experiment_id=experiment_id)
                    
                    # Log metadata and hyperparameters
                    current_date = datetime.now().strftime("%d%m%Y")
                    mlflow.set_tag("mlflow.runName", f"retrain-{current_date}")
                    mlflow.set_tag("mlflow.source.type", "PROJECT")
                    mlflow.set_tag("mlflow.source.git.commit", os.getenv('GIT_COMMIT', 'unknown'))
                    mlflow.set_tag("mlflow.source.git.branch", os.getenv('GIT_BRANCH', 'unknown'))
                    
                    mlflow.log_param("epochs", EPOCHS)
                    mlflow.log_param("batch_size", BATCH_SIZE)
                    mlflow.log_param("learning_rate", LEARNING_RATE)
                    mlflow.log_param("model_architecture", "SigLIP")
                else:
                    mlflow_run = None
            except Exception as e:
                logger.error(f"Error starting MLflow run: {e}")
                mlflow_active = False
                mlflow_run = None
                
            try:
                # 9. Training loop
                best_val_metrics = None
                best_model_path = os.path.join(temp_dir, "best_model.pt")
                best_model_state = None
                
                for epoch in range(EPOCHS):
                    logger.info(f"\nEpoch {epoch+1} of {EPOCHS}")
                    
                    # Train for one epoch
                    train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
                    logger.info(f"Training Loss: {train_loss:.4f}")
                    
                    # Evaluate on validation set
                    val_metrics = evaluate(model, val_dataloader)
                    logger.info(f"Validation Metrics:")
                    for name, value in val_metrics.items():
                        logger.info(f"  {name}: {value:.4f}")
                        if mlflow_active:
                            try:
                                mlflow.log_metric(f"val_{name}", value, step=epoch)
                            except Exception as e:
                                logger.error(f"Error logging metrics to MLflow: {e}")
                    
                    # Save best model
                    if best_val_metrics is None or val_metrics["f1"] > best_val_metrics["f1"]:
                        best_val_metrics = val_metrics
                        
                        # Save the model state dict
                        best_model_path = os.path.join(temp_dir, "best_model.pt")
                        torch.save(model.state_dict(), best_model_path)
                        
                        # Store the best model state for later use
                        best_model_state = {key: val.cpu().clone() for key, val in model.state_dict().items()}
                        
                        # Log the best model state dict to MLflow
                        if mlflow_active:
                            try:
                                mlflow.log_artifact(best_model_path, artifact_path=f"models/best_{current_date}")
                                logger.info(f"New best model saved with F1: {val_metrics['f1']:.4f}")
                            except Exception as e:
                                logger.error(f"Error logging model artifact to MLflow: {e}")
                        else:
                            logger.info(f"New best model saved with F1: {val_metrics['f1']:.4f}")
                
                # 10. Restore the best model for evaluation
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    logger.info("Restored best model for final evaluation")
                
                # 11. Evaluate on test set
                test_metrics = evaluate(model, test_dataloader)
                if mlflow_active:
                    try:
                        for name, value in test_metrics.items():
                            mlflow.log_metric(f"test_{name}", value)
                    except Exception as e:
                        logger.error(f"Error logging test metrics to MLflow: {e}")
                
                logger.info("\nFinal Test Metrics:")
                for name, value in test_metrics.items():
                    logger.info(f"  {name}: {value:.4f}")
                
                # 12. Promote to production if metrics are good enough
                promoted = promote_to_production(model, test_metrics)
                
                # Clean up MLflow run if active
                if mlflow_active and mlflow_run:
                    try:
                        mlflow.end_run()
                    except Exception as e:
                        logger.error(f"Error ending MLflow run: {e}")
                
                return promoted
                
            except Exception as e:
                logger.error(f"Error during training/evaluation: {e}")
                # Clean up MLflow run if active and not ended
                if mlflow_active and mlflow_run:
                    try:
                        mlflow.end_run()
                    except:
                        pass
                return False
    
    except Exception as e:
        logger.error(f"Error in retraining process: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    success = retrain()
    exit(0 if success else 1)