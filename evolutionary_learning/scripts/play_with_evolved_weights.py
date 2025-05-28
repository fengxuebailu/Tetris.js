\
import random
import json
import time
import os
import sys
from copy import deepcopy

# Path setup to import Tetris game logic from evolutionary_learning/core
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVOLUTIONARY_CORE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'core')
if EVOLUTIONARY_CORE_DIR not in sys.path:
    sys.path.insert(0, EVOLUTIONARY_CORE_DIR)

try:
    from Tetris import (shapes, rotate, check, join_matrix, clear_rows,
                        get_height, count_holes, get_bumpiness)
    print(f"Successfully imported Tetris game logic from: {EVOLUTIONARY_CORE_DIR}")
except ImportError as e_main:
    print(f"Error: Could not import Tetris game logic from {EVOLUTIONARY_CORE_DIR}: {e_main}")
    # Fallback: try to import from supervised_learning/core/Tetris.py
    # This requires knowing the relative path from this script to supervised_learning/core
    PROJECT_ROOT_FROM_SCRIPT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))) # Tetris.js directory
    SUPERVISED_CORE_DIR_FALLBACK = os.path.join(PROJECT_ROOT_FROM_SCRIPT, 'supervised_learning', 'core')
    if SUPERVISED_CORE_DIR_FALLBACK not in sys.path:
        sys.path.insert(0, SUPERVISED_CORE_DIR_FALLBACK)
    try:
        from Tetris import (shapes, rotate, check, join_matrix, clear_rows,
                            get_height, count_holes, get_bumpiness)
        print(f"Successfully imported Tetris game logic from fallback: {SUPERVISED_CORE_DIR_FALLBACK}")
    except ImportError as e_fallback:
        print(f"Fallback import also failed from {SUPERVISED_CORE_DIR_FALLBACK}: {e_fallback}")
        # As a last resort, try importing from the project root if Tetris.py is there
        if PROJECT_ROOT_FROM_SCRIPT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT_FROM_SCRIPT)
        try:
            from Tetris import (shapes, rotate, check, join_matrix, clear_rows,
                                get_height, count_holes, get_bumpiness)
            print(f"Successfully imported Tetris game logic from project root: {PROJECT_ROOT_FROM_SCRIPT}")
        except ImportError as e_root:
            print(f"Root import also failed from {PROJECT_ROOT_FROM_SCRIPT}: {e_root}")
            sys.exit(1)


def print_board_simple(board_state):
    """Helper function to print the board (text-based)."""
    if not board_state or not board_state[0]:
        print("Empty or invalid board state.")
        return
    print("-" * (len(board_state[0]) * 2 + 1))
    for row in board_state:
        print("|" + "|".join(["X" if cell else " " for cell in row]) + "|")
    print("-" * (len(board_state[0]) * 2 + 1))

def load_evolved_weights(filename="best_weights_evolved.json"):
    """Loads evolved weights from a JSON file located in the project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Should be Tetris.js
    weights_path = os.path.join(project_root, filename)
    try:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        print(f"Successfully loaded weights from {weights_path}")
        return weights
    except FileNotFoundError:
        print(f"Error: Weights file not found at {weights_path}")
        print("Using default weights as fallback.")
        return {
            'cleared_lines': 160.0, # Default from tetris_supervised_fixed
            'holes': -50.0,
            'bumpiness': -20.0,
            'height': -30.0
        }
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {weights_path}. Check the file content.")
        sys.exit(1)

def evaluate_position_evolved(board, cleared_lines, weights):
    """Evaluates a board position based on evolved weights."""
    height_val = get_height(board)
    holes_val = count_holes(board)
    bumpiness_val = get_bumpiness(board)

    score = (weights.get('cleared_lines', 0) * cleared_lines +
             weights.get('holes', 0) * holes_val +
             weights.get('bumpiness', 0) * bumpiness_val +
             weights.get('height', 0) * height_val)
    return score

def find_best_move_evolved(board, piece, weights):
    """Finds the best move for a given piece and board state using evolved weights."""
    best_eval_score = float('-inf')
    best_final_x = None
    best_rotation_count = 0
    
    original_piece = deepcopy(piece)

    for r_count in range(4): # 0, 1, 2, 3 rotations
        current_rotated_piece = deepcopy(original_piece)
        for _ in range(r_count):
            current_rotated_piece = rotate(current_rotated_piece)
        
        if not current_rotated_piece or not current_rotated_piece[0]: continue # Skip if piece is empty

        # Use the x-range from tetris_evolution.py for consistency
        # x_candidate is the column for the top-left of the piece matrix
        for x_candidate in range(-len(current_rotated_piece[0]) + 1, len(board[0]) + 2):
            # Check if the piece can be placed at (x_candidate, 0) (top of the board)
            if check(board, current_rotated_piece, [x_candidate, 0]):
                # Simulate dropping the piece
                y_final = 0
                while check(board, current_rotated_piece, [x_candidate, y_final + 1]):
                    y_final += 1
                
                # Create a temporary board to simulate the move
                temp_board = deepcopy(board)
                join_matrix(temp_board, current_rotated_piece, [x_candidate, y_final])
                
                new_board_after_clear, cleared_count = clear_rows(temp_board)
                
                current_eval_score = evaluate_position_evolved(new_board_after_clear, cleared_count, weights)
                
                if current_eval_score > best_eval_score:
                    best_eval_score = current_eval_score
                    best_final_x = x_candidate
                    best_rotation_count = r_count
    
    if best_final_x is not None:
        return {'x': best_final_x, 'rotation': best_rotation_count}
    else:
        # Fallback: if no move is found, try to place original piece at first valid spot
        # This is a very basic fallback, game might end soon.
        unrotated_piece = deepcopy(original_piece)
        if not unrotated_piece or not unrotated_piece[0]: return None
        for x_fb in range(-len(unrotated_piece[0]) + 1, len(board[0]) + 2):
            if check(board, unrotated_piece, [x_fb, 0]):
                return {'x': x_fb, 'rotation': 0} # Return first valid placement
        return None # No move possible

def play_game_with_evolved_ai(weights, game_id=0, max_moves=500, print_game=False):
    """Plays a single game of Tetris using the evolved AI."""
    board = [[0 for _ in range(10)] for _ in range(20)]
    game_score = 0 
    lines_cleared_total = 0
    moves_count = 0
    
    print(f"\\n--- Starting Game {game_id} with Evolved Weights ---")
    if print_game:
        print_board_simple(board)

    for move_idx in range(max_moves):
        current_piece_shape = random.choice(shapes)
        current_piece = deepcopy(current_piece_shape) 

        start_x_nominal = len(board[0]) // 2 - (len(current_piece[0]) // 2 if current_piece and current_piece[0] else 0)
        if not check(board, current_piece, [start_x_nominal, 0]):
            print(f"Game Over (Game {game_id}): Cannot place new piece. Moves: {moves_count}, Lines: {lines_cleared_total}, Score: {game_score}")
            break

        move_info = find_best_move_evolved(board, current_piece, weights)

        if move_info is None:
            print(f"Game Over (Game {game_id}): AI could not find a valid move. Moves: {moves_count}, Lines: {lines_cleared_total}, Score: {game_score}")
            break

        final_x = move_info['x']
        final_rotation = move_info['rotation']

        piece_to_place = deepcopy(current_piece_shape)
        for _ in range(final_rotation):
            piece_to_place = rotate(piece_to_place)
        
        final_y = 0
        # It's crucial that check is robust for x values outside 0..board_width-piece_width
        # if find_best_move_evolved can return them.
        # The y is determined by dropping from y=0.
        if check(board, piece_to_place, [final_x, 0]):
            while check(board, piece_to_place, [final_x, final_y + 1]):
                final_y += 1
        else:
            # This case implies find_best_move_evolved returned a move that's invalid at the top.
            # This could happen if the piece is wider than the board and x is such that no part is on board.
            # Or if check function has issues with certain x values.
            # For safety, if this happens, we consider it a game-ending error for the AI.
            print(f"Critical Error (Game {game_id}): Chosen move x={final_x}, rot={final_rotation} for piece {piece_to_place} is invalid at board top. Game Over.")
            # print_board_simple(board) # Print board for debugging
            break
        
        join_matrix(board, piece_to_place, [final_x, final_y])
        
        cleared_this_step = 0
        board, cleared_this_step = clear_rows(board)
        
        lines_cleared_total += cleared_this_step
        
        if cleared_this_step == 1: game_score += 40
        elif cleared_this_step == 2: game_score += 100
        elif cleared_this_step == 3: game_score += 300
        elif cleared_this_step >= 4: game_score += 1200 # Tetris
        game_score += 1 # Survival points per piece
        
        moves_count += 1

        if print_game:
            print(f"\\nMove {moves_count}: Piece placed at x={final_x}, y={final_y}, rotation={final_rotation}")
            print(f"Lines cleared this step: {cleared_this_step}, Total lines: {lines_cleared_total}, Score: {game_score}")
            print_board_simple(board)
            if sys.platform == "win32": # Basic clear screen for better viewing
                os.system('cls')
                print_board_simple(board) # Reprint after clear
            else:
                os.system('clear')
                print_board_simple(board) # Reprint after clear
            time.sleep(0.2) 

        if move_idx >= max_moves -1 :
            print(f"Game {game_id} ended: Max moves ({max_moves}) reached.")
            break
            
    print(f"--- Game {game_id} Finished ---")
    print(f"Final Score: {game_score}, Total Lines Cleared: {lines_cleared_total}, Moves: {moves_count}")
    if print_game and (sys.platform == "win32" or sys.platform == "linux"): # Reprint final board if it was cleared
        print("Final Board:")
        print_board_simple(board)
    return game_score, lines_cleared_total, moves_count

if __name__ == "__main__":
    evolved_weights = load_evolved_weights() 

    if evolved_weights:
        num_test_games = 3
        max_steps_per_game = 500 
        print_one_game_detail = True 

        all_scores = []
        all_lines = []
        all_moves = []

        for i in range(num_test_games):
            s, l, m = play_game_with_evolved_ai(
                evolved_weights, 
                game_id=i, 
                max_moves=max_steps_per_game,
                print_game=(i == 0 and print_one_game_detail) 
            )
            all_scores.append(s)
            all_lines.append(l)
            all_moves.append(m)
            if i == 0 and print_one_game_detail: # Pause after the detailed game
                input("Press Enter to continue with other games...")


        print("\\n--- Overall Test Results ---")
        if num_test_games > 0 and all_scores:
            print(f"Games played: {len(all_scores)}") # Use len(all_scores) in case some games failed early
            print(f"Average Score: {sum(all_scores)/len(all_scores):.2f}")
            print(f"Average Lines Cleared: {sum(all_lines)/len(all_lines):.2f}")
            print(f"Average Moves: {sum(all_moves)/len(all_moves):.2f}")
        else:
            print("No games were played or no results to average.")
