#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Collection and Training Script for Tetris AI
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('supervised_learning.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))

try:
    from core.tetris_supervised_fixed import TetrisDataset, TetrisAI, TetrisNet, save_model
    from core.enhanced_training import EnhancedDataCollector
    from Tetris import shapes, rotate, check, join_matrix, clear_rows
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def collect_data(num_games: int = 5, max_steps: int = 1000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Collect training data using enhanced collector"""
    logger.info("=== Starting Data Collection ===")
    logger.info(f"Configuration: {num_games} games, {max_steps} max steps per game")
    
    try:
        collector = EnhancedDataCollector(
            num_games=num_games,
            max_moves=max_steps,
            timeout=300
        )
        
        game_states, moves = collector.collect_enhanced_data()
        
        if isinstance(game_states, np.ndarray) and len(game_states) > 0:
            # Save data
            data_dir = os.path.join(parent_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            data_path = os.path.join(data_dir, "tetris_training_data.npz")
            
            np.savez(data_path, game_states=game_states, moves=moves)
            logger.info(f"Successfully collected and saved {len(game_states)} training samples")
            return game_states, moves
        else:
            logger.error("No valid training data collected")
            return None, None
            
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        return None, None

def train_model(game_states: np.ndarray, moves: np.ndarray, epochs: int = 50, batch_size: int = 64) -> Optional[TetrisNet]:
    """Train the model using collected data"""
    logger.info("=== Starting Model Training ===")
    logger.info(f"Training configuration: {epochs} epochs, batch size {batch_size}")
    logger.info(f"Training data: {len(game_states)} samples")
    
    try:
        from core.train_full_model import train_full_model
        model = train_full_model(epochs=epochs, batch_size=batch_size)
        
        if model is not None:
            logger.info("Training completed successfully")
            return model
        else:
            logger.error("Training failed to produce a valid model")
            return None
            
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return None

def main():
    """Main function to run the complete training pipeline"""
    logger.info("Starting Tetris AI training pipeline")
    
    # Step 1: Collect training data
    game_states, moves = collect_data(num_games=5, max_steps=1000)
    if game_states is None or moves is None:
        logger.error("Failed to collect training data")
        return
    
    # Step 2: Train model
    model = train_model(game_states, moves, epochs=50, batch_size=64)
    if model is None:
        logger.error("Failed to train model")
        return
    
    logger.info("=== Training Pipeline Complete ===")
    logger.info("To test the model, run:")
    logger.info("python test_models.py supervised_learning/models/tetris_model_new_full.pth")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
# -*- coding: utf-8 -*-
"""
Data Collection and Training Script
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(parent_dir))

from core.tetris_supervised_fixed import TetrisDataset, TetrisAI, TetrisNet, save_model
from core.enhanced_training import EnhancedDataCollector
from Tetris import shapes, rotate, check, join_matrix, clear_rows

def collect_data(num_games=5, max_steps=1000):
    """Collect training data using enhanced collector"""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("\n=== Collecting Training Data ===")
    logger.info(f"Games: {num_games}, Max steps per game: {max_steps}")
    
    collector = EnhancedDataCollector(
        num_games=num_games,
        max_moves=max_steps,
        timeout=300
    )
    
    game_states, moves = collector.collect_enhanced_data()
    if len(game_states) > 0:
        # Save data
        data_path = os.path.join(parent_dir, "data", "tetris_training_data.npz")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.savez(data_path, game_states=game_states, moves=moves)
        print(f"\nCollected and saved {len(game_states)} training samples")
        return game_states, moves
    return None, None

def train_model(game_states, moves, epochs=50, batch_size=64):
    """Train the model using collected data"""
    print("\n=== Training Model ===")
    print(f"Training samples: {len(game_states)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    model = TetrisNet()
    dataset = TetrisDataset(game_states, moves)
    
    try:
        from core.train_full_model import train_full_model
        model = train_full_model(epochs=epochs, batch_size=batch_size)
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def main():
    """Main function"""
    # Step 1: Collect data
    game_states, moves = collect_data(num_games=5, max_steps=1000)
    if game_states is None:
        print("Failed to collect training data")
        return
        
    # Step 2: Train model
    model = train_model(game_states, moves, epochs=50, batch_size=64)
    if model is None:
        print("Failed to train model")
        return
        
    print("\n=== Training Complete ===")
    print("You can now test the model using:")
    print("python test_models.py supervised_learning/models/tetris_model_new_full.pth")

if __name__ == "__main__":
    main()
