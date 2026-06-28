import argparse
import sys
from src import train
from src import inference
from src import utils

logger = utils.setup_logger("main")

def main():
    parser = argparse.ArgumentParser(description="V5 Crop Disease Outbreak Prediction System")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "predict"],
        help="Mode to run the system in: 'train' or 'predict'"
    )
    
    args = parser.parse_args()
    
    utils.ensure_dirs()
    
    if args.mode == "train":
        logger.info("Executing TRAIN mode...")
        train.run_training()
    elif args.mode == "predict":
        logger.info("Executing PREDICT mode...")
        inference.run_inference()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
