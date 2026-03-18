from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import shutil
import base64
from io import BytesIO
from PIL import Image
import tempfile

from model_manager import ModelManager
from grad_cam import create_grad_cam_visualization

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize model manager
model_manager = ModelManager()

# Create the main app without a prefix
app = FastAPI(title="Bone Fracture Detection API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Training state
training_state = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_accuracy": 0.0,
    "status": "idle"
}

# Define Models
class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction: str
    confidence: float
    probabilities: List[float]
    grad_cam_image: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    filename: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: List[float]
    grad_cam_image: Optional[str] = None
    class_idx: int

class TrainingRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.0001

class TrainingStatus(BaseModel):
    is_training: bool
    progress: float
    current_epoch: int
    total_epochs: int
    train_loss: float
    val_accuracy: float
    status: str

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]

@api_router.get("/")
async def root():
    return {"message": "Bone Fracture Detection API", "status": "healthy"}

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_fracture(file: UploadFile = File(...)):
    """
    Upload X-ray image and get fracture prediction with Grad-CAM visualization
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save temporary file
    #temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    temp_dir = tempfile.gettempdir()   
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Run inference
        result = model_manager.predict(temp_path, return_cam=True)
        
        # Store prediction in database
        prediction_doc = {
            "id": str(uuid.uuid4()),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "filename": file.filename,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        await db.predictions.insert_one(prediction_doc)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@api_router.get("/predictions-history")
async def get_predictions_history(limit: int = 50):
    """
    Get recent prediction history
    """
    predictions = await db.predictions.find(
        {}, 
        {"_id": 0}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    
    return {"predictions": predictions, "total": len(predictions)}

@api_router.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Start model training in background
    """
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Reset training state
    training_state["is_training"] = True
    training_state["progress"] = 0
    training_state["current_epoch"] = 0
    training_state["total_epochs"] = request.epochs
    training_state["status"] = "starting"
    
    # Start training in background
    background_tasks.add_task(
        model_manager.train_model,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        training_state=training_state
    )
    
    return {"message": "Training started", "epochs": request.epochs}

@api_router.get("/training-status", response_model=TrainingStatus)
async def get_training_status():
    """
    Get current training status
    """
    return TrainingStatus(**training_state)

@api_router.get("/model-metrics")
async def get_model_metrics():
    """
    Get model evaluation metrics
    """
    metrics = model_manager.get_metrics()
    if metrics is None:
        raise HTTPException(status_code=404, detail="No trained model found")
    return metrics

@api_router.get("/dataset-info")
async def get_dataset_info():
    """
    Get information about the training dataset
    """
    info = model_manager.get_dataset_info()
    return info

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Bone Fracture Detection API")
    logger.info(f"Model device: {model_manager.device}")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
