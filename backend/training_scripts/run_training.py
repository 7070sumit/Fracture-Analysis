import logging
import argparse
from model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_script")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the OAI Fracture Detection Model")
    parser.add_argument("--epochs",     type=int,   default=20,     help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=16,     help="Batch size")
    parser.add_argument("--lr",         type=float, default=0.0001, help="Learning rate")
    args = parser.parse_args()

    OAI_DATA_DIR = "/Users/piyushkumar/Project/Fracture-Analysis-main/backend/datasets/OAI_Processed"

    logger.info(f"Starting training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    manager = ModelManager(data_dir=OAI_DATA_DIR)

    try:
        manager.train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
        logger.info("Training complete!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
