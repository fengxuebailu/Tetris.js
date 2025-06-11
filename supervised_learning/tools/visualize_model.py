#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tetris Supervised Learning System - Neural Network Visualization Script
Used for visualizing and analyzing model internal structure, weights, and activations
"""

# Import Matplotlib configuration
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tools.matplotlibrc import *

import torch
import numpy as np
import seaborn as sns
from core.tetris_supervised_fixed import TetrisNet, TetrisAI
from Tetris import shapes, rotate, check, join_matrix, clear_rows

def visualize_model_structure(model_path):
    """Visualize model structure"""
    try:
        # Load model
        model = TetrisNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Print model structure
        print(f"\n=== {os.path.basename(model_path)} Model Structure ===")
        print(model)
        
        # Calculate number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    except Exception as e:
        print(f"Error visualizing model structure: {e}")
        return None

def visualize_model_weights(model):
    """Visualize model weights"""
    if model is None:
        return
    
    try:
        # Create directory
        if not os.path.exists("model_visualization"):
            os.makedirs("model_visualization")
        
        # Visualize weight distribution for each layer
        plt.figure(figsize=(15, 10))
        layer_idx = 0
        
        # Function to plot weights in a module
        def plot_module_weights(module, prefix="", max_plots=9):
            nonlocal layer_idx
            for name, param in module.named_parameters():
                if 'weight' in name:
                    layer_idx += 1
                    if layer_idx <= max_plots:
                        plt.subplot(3, 3, layer_idx)
                        weights = param.data.cpu().numpy().flatten()
                        sns.histplot(weights, bins=50, kde=True)
                        plt.title(f"{prefix}{name}")
                        plt.xlabel("Weight Value")
                        plt.ylabel("Frequency")
        
        # Visualize weights of main modules
        plot_module_weights(model.board_features, "board_features.")
        plot_module_weights(model.piece_features, "piece_features.")
        plot_module_weights(model.combined_network, "combined_network.")
        
        plt.tight_layout()
        plt.savefig("model_visualization/weight_distributions.png")
        
        # Visualize first layer weights
        plt.figure(figsize=(12, 5))
        
        # Board features first layer
        plt.subplot(1, 2, 1)
        board_weights = model.board_features[0].weight.data.cpu().numpy()
        plt.imshow(board_weights, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("Board Features First Layer Weights")
        plt.xlabel("Input Features")
        plt.ylabel("Neurons")
        
        # Piece features first layer
        plt.subplot(1, 2, 2)
        piece_weights = model.piece_features[0].weight.data.cpu().numpy()
        plt.imshow(piece_weights, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("Piece Features First Layer Weights")
        plt.xlabel("Input Features")
        plt.ylabel("Neurons")
        
        plt.tight_layout()
        plt.savefig("model_visualization/first_layer_weights.png")
        
    except Exception as e:
        print(f"Error visualizing model weights: {e}")

def visualize_activations(model_path):
    """Visualize model activations"""
    try:
        # Load AI and model
        ai = TetrisAI(model_path)
        
        # Create an empty game board and an example piece
        board = [[0 for _ in range(10)] for _ in range(20)]
        piece = shapes[0]  # I-shape piece
        
        # Get state vector
        state_vector = ai.create_state_vector(board, piece)
        state_tensor = torch.FloatTensor([state_vector])
        
        # Get activations for different parts of the model
        board_input = state_tensor[:, :200]
        piece_input = state_tensor[:, 200:]
        
        # Get intermediate activations
        model = ai.model
        board_features = model.board_features(board_input).detach().cpu().numpy()[0]
        piece_features = model.piece_features(piece_input).detach().cpu().numpy()[0]
        
        # Visualize activations
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(board_features)), board_features)
        plt.title("Board Feature Extraction Activations")
        plt.xlabel("Feature Index")
        plt.ylabel("Activation Value")
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(piece_features)), piece_features)
        plt.title("Piece Feature Extraction Activations")
        plt.xlabel("Feature Index")
        plt.ylabel("Activation Value")
        
        plt.tight_layout()
        plt.savefig("model_visualization/activations.png")
        
    except Exception as e:
        print(f"Error visualizing activations: {e}")

def visualize_decision_heatmap(model_path):
    """Generate decision heatmap"""
    try:
        # Load AI
        ai = TetrisAI(model_path)
        
        # Create a game board with some blocks
        board = [[0 for _ in range(10)] for _ in range(20)]
        
        # Add some blocks at the bottom
        for j in range(10):
            board[19][j] = 1
        
        for j in range(8):
            board[18][j] = 1
        
        # Create decision heatmap for I-shape piece
        piece = shapes[0]  # I-shape piece
        
        # Create heatmap data for different x positions and rotations
        heatmap_data = np.zeros((4, 10))  # 4 rotations, 10 possible x positions
        
        # Generate all possible rotated pieces
        rotated_pieces = [piece]
        for _ in range(3):
            rotated_pieces.append(rotate(rotated_pieces[-1]))
        
        # Fill heatmap
        for rot in range(4):
            rotated_piece = rotated_pieces[rot]
            for x in range(-2, 8):
                # Try to place the piece
                piece_width = len(rotated_piece[0])
                if x + piece_width > 10:
                    continue
                    
                # Calculate score (find lowest valid y)
                y = 0
                while y < 20 and check(board, rotated_piece, [x, y+1]):
                    y += 1
                
                # Check if placement is valid
                if check(board, rotated_piece, [x, y]):
                    # Simulate placement and evaluate
                    temp_board = [row[:] for row in board]
                    join_matrix(temp_board, rotated_piece, [x, y])
                    new_board, cleared = clear_rows(temp_board)
                    
                    # Use model to predict score for this position
                    state_vector = ai.create_state_vector(board, rotated_piece)
                    state_tensor = torch.FloatTensor([state_vector])
                    
                    with torch.no_grad():
                        prediction = ai.model(state_tensor).cpu().numpy()[0]
                        # Use prediction as score (lower is better for this metric)
                        score = abs(prediction[0] - x) + abs(prediction[1] - rot)
                        # Map to 0-1 range (higher is better for heatmap)
                        score = 1.0 / (1.0 + score)
                        
                        # Fill heatmap data
                        x_idx = x + 2  # Adjust index as x can be negative
                        if 0 <= x_idx < 10:
                            heatmap_data[rot, x_idx] = score
        
        # Visualize heatmap
        plt.figure(figsize=(10, 6))
        x_tick_labels = list(range(-2, 8)) # Actual game x-coordinates corresponding to heatmap_data columns
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", xticklabels=x_tick_labels)
        plt.title(f"{os.path.basename(model_path)}: I-Shape Piece Decision Heatmap")
        plt.xlabel("X Position (Game Coordinate)") # Clarified x-axis label
        plt.ylabel("Rotation")
        plt.tight_layout()
        plt.savefig("model_visualization/decision_heatmap.png")
        
    except Exception as e:
        print(f"Error generating decision heatmap: {e}")
        import traceback
        traceback.print_exc()

def compare_model_outputs(model_paths):
    """Compare outputs of different models"""
    models = []
    model_names = []
    
    # Load all models
    for path in model_paths:
        try:
            ai = TetrisAI(path)
            models.append(ai)
            model_names.append(os.path.basename(path))
        except Exception as e:
            print(f"Failed to load model {path}: {e}")
    
    if len(models) < 2:
        print("At least two models are required for comparison")
        return
    
    # Create some test scenarios
    test_scenarios = []
    
    # Scenario 1: Empty board
    board1 = [[0 for _ in range(10)] for _ in range(20)]
    test_scenarios.append(("Empty Board", board1))
    
    # Scenario 2: Bottom filled
    board2 = [[0 for _ in range(10)] for _ in range(20)]
    for j in range(10):
        board2[19][j] = 1
    test_scenarios.append(("Bottom Filled", board2))
    
    # Scenario 3: Hole at the bottom
    board3 = [[0 for _ in range(10)] for _ in range(20)]
    for j in range(10):
        if j != 5:
            board3[19][j] = 1
    test_scenarios.append(("Bottom with a Hole", board3))
    
    # Compare outputs for each scenario
    print("\n=== Model Output Comparison ===")
    
    for scenario_name, board in test_scenarios:
        print(f"\nScenario: {scenario_name}")
        
        for piece_idx, piece in enumerate(shapes[:3]):  # Test first three piece types
            print(f"Piece type: {piece_idx}")
            
            for i, ai in enumerate(models):
                move = ai.predict_move(board, piece)
                print(f"- {model_names[i]}: x={move['x']}, rotation={move['rotation']}")

# Import required functions (already imported at the top, but kept for original structure)
from Tetris import rotate, check, join_matrix, clear_rows

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_model.py <model_path>")
        print("Example: python visualize_model.py tetris_model.pth")
        
        # Use the latest .pth file
        import os
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if not model_files:
            print("Error: No model files found (.pth)")
            sys.exit(1)
            
        # Sort by modification time, select the newest
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model_path = model_files[0]
        print(f"Using latest model file: {model_path}")
    else:
        model_path = sys.argv[1]
    
    # Ensure visualization directory exists
    if not os.path.exists("model_visualization"):
        os.makedirs("model_visualization")
    
    # Visualize model structure
    print(f"Visualizing model: {model_path}")
    model = visualize_model_structure(model_path)
    
    # Visualize model weights
    print("Visualizing model weights...")
    visualize_model_weights(model)
    
    # Visualize model activations
    print("Visualizing model activations...")
    visualize_activations(model_path)
    
    # Generate decision heatmap
    print("Generating decision heatmap...")
    visualize_decision_heatmap(model_path)
    
    # If multiple models are provided, compare their outputs
    if len(sys.argv) > 2:
        compare_model_outputs(sys.argv[1:])
    elif len([f for f in os.listdir('.') if f.endswith('.pth')]) > 1:
        # Use the two latest models
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if len(model_files) >=2: # Ensure at least two models exist for comparison
            print("\nComparing with the second latest model as well...")
            compare_model_outputs(model_files[:2])
        
    print("\nVisualization complete. Results saved in model_visualization directory.")
