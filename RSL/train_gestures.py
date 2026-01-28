#!/usr/bin/env python3
import os
import numpy as np
from main_fixed import AppConfig, DatasetManager, ModelHandler

def train_system():
    # 1. Configuration
    cfg = AppConfig()
    
    # 2. Define Gestures
    # DEFAULT list (smaller for testing)
    signs = ["hello", "thanks", "iloveyou"]
    
    # UNCOMMENT the line below for the FULL ASL Alphabet:
    # signs = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # You can also add custom gestures like this:
    # signs = ["hello", "thanks", "iloveyou", "please", "yes", "no", "help"]
    
    print(f"Gestures to process: {', '.join(signs)}")

    # 3. Handle Dataset
    cfg.training_epochs = 500  # Lowered for faster training, adjust as needed
    dm = DatasetManager(cfg, signs)
    
    # Check if data exists for all signs
    missing_data = False
    for sign in signs:
        sign_path = os.path.join(cfg.data_path, sign)
        if not os.path.exists(sign_path) or len(os.listdir(sign_path)) < cfg.sequences_per_sign:
            print(f"Missing or incomplete data for: {sign}")
            missing_data = True
            # Create directory for collection
            os.makedirs(sign_path, exist_ok=True)
            for i in range(cfg.sequences_per_sign):
                os.makedirs(os.path.join(sign_path, str(i)), exist_ok=True)

    if missing_data:
        print("\nStarting DATA COLLECTION phase...")
        print("Prepare to perform the signs when prompted.")
        success = dm.collect()
        if not success:
            print("\nData collection was interrupted. Training aborted.")
            return
    
    # 4. Load Data
    print("\nLoading dataset...")
    X, y = dm.load_dataset()
    if X.size == 0:
        print("Error: No valid sequences found in dataset. Please collect more data.")
        return
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # 5. Build and Train Model
    print("\nBuilding and training LSTM model...")
    model_handler = ModelHandler(cfg, len(signs))
    model_handler.build_lstm_model()
    model_handler.train(X, y)

    # 6. Save Model
    print(f"\nSaving model to: {cfg.model_path}")
    model_handler.save()
    print("Training complete!")

if __name__ == "__main__":
    train_system()
