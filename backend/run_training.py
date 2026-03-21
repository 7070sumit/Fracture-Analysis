import logging
from model_manager import ModelManager

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_script")

if __name__ == "__main__":
    MURA_DATA_DIR = "/Users/piyushkumar/Project/Fracture-Analysis-main/datasets/MURA/muramskxrays/MURA-v1.1"
    
    logger.info(f"Initializing ModelManager with MURA dataset at {MURA_DATA_DIR}")
    
    # Initialize the model manager. It will automatically detect MPS (Metal GPU) due to our updates.
    manager = ModelManager(data_dir=MURA_DATA_DIR)
    
    logger.info("Starting training process with 10 epochs and batch size of 16...")
    
    try:
        # Start training
        manager.train_model(epochs=10, batch_size=16, learning_rate=0.0001)
        logger.info("Training complete!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
