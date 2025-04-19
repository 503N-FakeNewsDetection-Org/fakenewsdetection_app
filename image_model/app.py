import os
import io
import hashlib
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import SiglipConfig, SiglipForImageClassification, AutoImageProcessor
from PIL import Image
from azure.storage.blob import BlobServiceClient
import tempfile
import logging
from torchvision import transforms

# Load environment variables from .env file (assuming it's in the root)
load_dotenv()

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Azure Blob Storage configuration
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODELS_CONTAINER = "fakenewsdetection-models"
IMAGE_MODEL_BLOB = "image.pt"  # Blob name for the image model weights

# Define target containers for saving images
AI_IMAGE_CONTAINER = os.getenv("AI_IMAGE_CONTAINER", "fakenewsdetection-ai-imgs")
HUMAN_IMAGE_CONTAINER = os.getenv("HUMAN_IMAGE_CONTAINER", "fakenewsdetection-hum-imgs")

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create router instance for image detection
# Using empty prefix since the main_app will mount this with the /image prefix
router = APIRouter()

# Global variables for model and processor management
model = None
processor = None
model_config = None
transform = None

def load_image_model():
    """Load image model from local file or Azure Blob Storage"""
    global model, processor, model_config, transform
    try:
        # 1. Load the model config & processor
        config = SiglipConfig.from_pretrained("Ateeqq/ai-vs-human-image-detector")
        image_processor = AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
        
        # 2. Instantiate the model (no weights yet)
        new_model = SiglipForImageClassification(config)
        
        # 3. Set up image transformation pipeline (matching fine-tuning approach)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # 4. Try to load weights from local file first
        local_model_path = os.path.join(os.path.dirname(__file__), "image.pt")
        if os.path.exists(local_model_path):
            logger.info(f"Found local image model weights at {local_model_path}")
            state_dict = torch.load(local_model_path, map_location=device)
            new_model.load_state_dict(state_dict)
            new_model = new_model.to(device)
            new_model.eval()
            model = new_model
            processor = image_processor
            model_config = config
            logger.info("Image model loaded successfully from local file")
            return

        logger.info("No local image model weights found, downloading from Azure...")
        # 5. Download from Azure if local file not found
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(MODELS_CONTAINER)
        blob_client = container_client.get_blob_client(IMAGE_MODEL_BLOB) # Use image model blob name

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Download blob to temp file
            logger.info(f"Downloading {IMAGE_MODEL_BLOB} from Azure container {MODELS_CONTAINER}...")
            download_stream = blob_client.download_blob()
            download_stream.readinto(temp_file)
            
            # Load state dict from temp file
            state_dict = torch.load(temp_file.name, map_location=device)
            new_model.load_state_dict(state_dict)
            os.unlink(temp_file.name)
            logger.info("Downloaded and loaded state dict from Azure.")

        # Move model to appropriate device
        new_model = new_model.to(device)
        new_model.eval()
        model = new_model
        processor = image_processor
        model_config = config
        logger.info("Image model loaded successfully from Azure")

    except Exception as e:
        logger.error(f"Error loading image model: {e}")
        # Allow the app to start, but log the error. Prediction endpoint will fail.
        model = None
        processor = None
        model_config = None
        transform = None
        # raise HTTPException(status_code=500, detail="Image model could not be loaded") # Optional: block startup

# New function to save image bytes to the appropriate container
def save_image_to_blob(image_bytes: bytes, filename: str, prediction: str):
    """Saves unique image bytes to Azure Blob Storage based on prediction, using SHA256 hash as blob name."""
    if not CONNECTION_STRING:
        logger.warning("Azure connection string not configured. Skipping image file saving.")
        return
    try:
        # Determine target container
        if prediction == "AI":
            target_container = AI_IMAGE_CONTAINER
        elif prediction == "Human":
            target_container = HUMAN_IMAGE_CONTAINER
        else:
            logger.warning(f"Unknown prediction '{prediction}', cannot save image.")
            return

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        blob_name = image_hash
        metadata = {"original_filename": filename}
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

        try:
            container_client = blob_service_client.get_container_client(target_container)
            # Check if container exists by trying to get properties
            container_client.get_container_properties()
        except Exception as container_error:

            logger.info(f"Container '{target_container}' not found or error accessing: {container_error}. Attempting to create it.")
            try:
                 container_client = blob_service_client.create_container(target_container)
            except Exception as creation_error:
                 logger.error(f"Failed to create container '{target_container}': {creation_error}")
                 return # Cannot proceed without container

        blob_client = container_client.get_blob_client(blob_name)
        if blob_client.exists():
            logger.info(f"Duplicate image detected (Hash: {image_hash}). Skipping upload to '{target_container}'. Original filename: '{filename}'")
            return
        else:
            # Upload the image if it doesn't exist
            logger.info(f"Uploading new image (Hash: {image_hash}) as '{blob_name}' to container '{target_container}'. Original filename: '{filename}'")
            blob_client.upload_blob(image_bytes, metadata=metadata)
            logger.info(f"Image successfully uploaded.")

    except Exception as e:
        logger.error(f"Error saving image file to blob storage: {e}", exc_info=True)


class ImageResponse(BaseModel):
    prediction: str # "AI" or "Human"
    confidence: float # Percentage

# Image detection endpoint
@router.post("/image", response_model=ImageResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None or processor is None or model_config is None or transform is None:
        logger.error("Image model not loaded properly.")
        raise HTTPException(status_code=503, detail="Image model is not available")

    try:
        # Read image contents
        contents = await file.read()
        # Open image using PIL
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as img_err:
            logger.error(f"Error opening image: {img_err}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        inputs = processor(images=img, return_tensors="pt").to(device)
        # Run inference
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)[0]
            idx    = int(probs.argmax())
            label  = model_config.id2label[idx] # Get label from config ("ai", "human")
            conf   = float(probs[idx] * 100)

        # Standardize label output
        prediction_label = "AI" if label.lower() == "ai" else "Human"
        logger.info(f"Prediction made: {prediction_label} with confidence {conf:.2f}%")
        save_image_to_blob(contents, file.filename, prediction_label)

        return ImageResponse(
            prediction=prediction_label,
            confidence=round(conf, 2)
        )
        
    except Exception as e:
        logger.error(f"Image prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during image prediction: {e}")


# Initialize model on startup 
load_image_model()