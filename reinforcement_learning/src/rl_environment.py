import numpy as np
import random
from collections import deque

# --- Constants ---
BOARD_WIDTH = 10
BOARD_HEIGHT = 20 # This is the visible part of the board

# Shapes of the Tetris pieces
PIECES = {
    0: {'shape': [[1, 1, 1, 1]], 'color': (255, 0, 0), 'name': 'I'}, # I
    1: {'shape': [[1, 1], [1, 1]], 'color': (0, 255, 0), 'name': 'O'},    # O
    2: {'shape': [[0, 1, 0], [1, 1, 1]], 'color': (0, 0, 255), 'name': 'T'}, # T
    3: {'shape': [[1, 0, 0], [1, 1, 1]], 'color': (255, 255, 0), 'name': 'L'},# L
    4: {'shape': [[0, 0, 1], [1, 1, 1]], 'color': (255, 0, 255), 'name': 'J'},# J
    5: {'shape': [[1, 1, 0], [0, 1, 1]], 'color': (0, 255, 255), 'name': 'S'},# S
    6: {'shape': [[0, 1, 1], [1, 1, 0]], 'color': (128, 0, 128), 'name': 'Z'} # Z
}
NUM_PIECES = len(PIECES)
NUM_ROTATIONS = 4 # Max rotations for any piece
STATE_DIM = 219

# --- Helper Functions ---
def rotate_piece(shape):
    return [list(row) for row in zip(*shape[::-1])]

class TetrisEnv:
    def __init__(self):
        self.board_width = BOARD_WIDTH
        self.board_height = BOARD_HEIGHT
        self.action_space_n = NUM_ROTATIONS * BOARD_WIDTH
        self.observation_space_shape = (STATE_DIM,)
        self.piece_shapes = [PIECES[i]['shape'] for i in range(NUM_PIECES)]
        self.reset()

    def _initialize_board(self):
        return np.zeros((self.board_height, self.board_width), dtype=np.int8)

    def _new_piece(self):
        if not self.next_piece_idx_queue:
            self.next_piece_idx_queue.append(random.randrange(NUM_PIECES))
            self.next_piece_idx_queue.append(random.randrange(NUM_PIECES))

        self.current_piece_idx = self.next_piece_idx_queue.popleft()
        self.current_piece_shape = self.piece_shapes[self.current_piece_idx]
        self.next_piece_idx_queue.append(random.randrange(NUM_PIECES))
        
        self.current_piece_rotation = 0
        self.current_piece_x = self.board_width // 2 - len(self.current_piece_shape[0]) // 2
        self.current_piece_y = 0 

        if self._check_collision(self.current_piece_shape, (self.current_piece_x, self.current_piece_y)):
            self.game_over = True

    def reset(self):
        self.board = self._initialize_board()
        self.score = 0
        self.lines_cleared_total = 0
        self.game_over = False
        self.next_piece_idx_queue = deque()
        self.next_piece_idx_queue.append(random.randrange(NUM_PIECES))
        self.next_piece_idx_queue.append(random.randrange(NUM_PIECES))
        self.held_piece_idx = None
        self.can_hold = True
        self._new_piece()
        # Game over can be set in _new_piece if the board is full.
        # If game over, the mask might not be relevant, but we provide a default one.
        valid_actions_mask = self.get_valid_actions() if not self.game_over else [False] * self.action_space_n
        return self._get_state(), valid_actions_mask

    def _check_collision(self, shape, offset):
        off_x, off_y = offset
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    board_r, board_c = r + off_y, c + off_x
                    if not (0 <= board_c < self.board_width and 0 <= board_r < self.board_height):
                        return True
                    if self.board[board_r, board_c] != 0:
                        return True
        return False

    def _lock_piece(self):
        shape = self.current_piece_shape
        off_x, off_y = self.current_piece_x, self.current_piece_y
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    self.board[r + off_y, c + off_x] = 1
        lines_cleared_this_turn = self._clear_lines()
        self.lines_cleared_total += lines_cleared_this_turn
        if lines_cleared_this_turn == 1: self.score += 40
        elif lines_cleared_this_turn == 2: self.score += 100
        elif lines_cleared_this_turn == 3: self.score += 300
        elif lines_cleared_this_turn >= 4: self.score += 1200
        self.can_hold = True
        return lines_cleared_this_turn

    def _clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.board) if np.all(row)]
        if not lines_to_clear: return 0
        for r_idx in sorted(lines_to_clear, reverse=True):
            self.board = np.delete(self.board, r_idx, axis=0)
            new_line = np.zeros((1, self.board_width), dtype=np.int8)
            self.board = np.vstack([new_line, self.board])
        return len(lines_to_clear)

    def _get_board_features(self):
        return self.board.flatten().tolist()

    def _get_game_stats(self):
        heights = np.zeros(self.board_width, dtype=int)
        for c in range(self.board_width):
            for r in range(self.board_height):
                if self.board[r, c] != 0:
                    heights[c] = self.board_height - r
                    break
        aggregated_height = np.sum(heights)
        holes = 0
        for c in range(self.board_width):
            col_has_block = False
            for r in range(self.board_height):
                if self.board[r, c] != 0: col_has_block = True
                elif col_has_block and self.board[r, c] == 0: holes += 1
        bumpiness = 0
        for i in range(self.board_width - 1):
            bumpiness += abs(heights[i] - heights[i+1])
        return [aggregated_height, holes, bumpiness]

    def _get_piece_features(self):
        current_piece_one_hot = np.zeros(NUM_PIECES, dtype=int)
        if self.current_piece_idx is not None: current_piece_one_hot[self.current_piece_idx] = 1
        next_piece_one_hot = np.zeros(NUM_PIECES, dtype=int)
        if self.next_piece_idx_queue: next_piece_one_hot[self.next_piece_idx_queue[0]] = 1
        hold_exists = 1 if self.held_piece_idx is not None else 0
        can_hold_feature = 1 if self.can_hold else 0
        return current_piece_one_hot.tolist() + next_piece_one_hot.tolist() + [hold_exists, can_hold_feature]

    def _get_state(self):
        if self.game_over: return np.zeros(STATE_DIM, dtype=float).tolist()
        board_feats = self._get_board_features()
        game_stats = self._get_game_stats() # Still useful for info, even if not directly in this simplified reward
        piece_feats = self._get_piece_features()
        return board_feats + game_stats + piece_feats

    def get_valid_actions(self):
        """
        Determines which actions (rotation, x-position combinations) are valid
        for the current piece at its spawn y-coordinate.
        An action is valid if it doesn't cause the piece to collide with board boundaries
        or existing pieces at spawn height (y=0).
        Returns a list of booleans of length self.action_space_n.
        """
        if self.game_over or self.current_piece_idx is None: # Should not happen if game_over is handled before call
            return [False] * self.action_space_n

        valid_actions_mask = [False] * self.action_space_n
        base_shape = self.piece_shapes[self.current_piece_idx]

        for r_idx in range(NUM_ROTATIONS): # 0, 1, 2, 3 rotations
            temp_shape = base_shape
            for _ in range(r_idx):
                temp_shape = rotate_piece(temp_shape) # Global helper

            for x_pos in range(self.board_width): # 0 to 9 for board_width=10
                action_idx = r_idx * self.board_width + x_pos
                # Use _check_collision at spawn height (y=0)
                # _check_collision returns True if there IS a collision (invalid placement)
                if not self._check_collision(temp_shape, (x_pos, 0)):
                    valid_actions_mask[action_idx] = True
        return valid_actions_mask

    def _apply_action_to_piece(self, action_idx):
        """
        Applies the action (rotation, horizontal position) to the current piece
        at its current y-level (spawn level).
        The action_idx maps to one of 4 rotations and 10 horizontal positions.
        Modifies self.current_piece_shape, self.current_piece_rotation, self.current_piece_x if valid.
        Returns True if placement is valid, False if it results in immediate collision.
        """
        target_rotation_times = action_idx // self.board_width
        target_x = action_idx % self.board_width

        # print(f"DEBUG: _apply_action_to_piece: action_idx={action_idx}, target_rot_times={target_rotation_times}, target_x={target_x}")

        # Get the base shape for the current piece type
        current_base_shape = self.piece_shapes[self.current_piece_idx]
        
        # Rotate the shape
        temp_shape = current_base_shape
        for _ in range(target_rotation_times):
            temp_shape = rotate_piece(temp_shape) # Use global helper

        # Check for collision with the new shape, target_x, at current_piece_y
        if self._check_collision(temp_shape, (target_x, self.current_piece_y)):
            print(f"DEBUG: Initial collision! Target Rotation Times: {target_rotation_times}, Target X: {target_x}, Y: {self.current_piece_y}") # MODIFIED: Uncommented
            print(f"DEBUG: Piece trying to be: {temp_shape}") # MODIFIED: Uncommented
            print(f"DEBUG: Board state at collision (around piece y={self.current_piece_y}):") # MODIFIED: Uncommented
            for r_idx, row in enumerate(self.board[max(0, self.current_piece_y -1) : self.current_piece_y + 4]): # MODIFIED: Uncommented
                print(f"DEBUG: Board row {max(0, self.current_piece_y -1) + r_idx}: {row}") # MODIFIED: Uncommented
            return False # Invalid placement due to immediate collision
        else:
            # Valid: update current piece's state
            self.current_piece_shape = temp_shape
            self.current_piece_rotation = target_rotation_times
            self.current_piece_x = target_x
            # self.current_piece_y remains the same (spawn y)
            print(f"DEBUG: Valid initial placement. Rotation: {self.current_piece_rotation}, X: {self.current_piece_x}, Y: {self.current_piece_y}, Shape: {self.current_piece_shape}") # MODIFIED: Uncommented
            return True


    def step(self, action_idx: int):
        if self.game_over: 
            # Provide a default mask even if game is over
            # This state might be stored in replay buffer, though next_state is None
            dummy_mask = [False] * self.action_space_n 
            return self._get_state(), 0, True, {"info": "Game Over, no action taken"}, dummy_mask

        if not isinstance(action_idx, int):
            print(f"DEBUG: Invalid action_idx type: {type(action_idx)}, value: {action_idx}. Defaulting to action 0.") # MODIFIED: Uncommented
            action_idx = 0

        # --- Reward Parameters (defined locally as per existing structure) ---
        REWARD_PLACE_SUCCESS = 0.0 
        REWARD_CLEAR_1_LINE = 10
        REWARD_CLEAR_2_LINES = 30
        REWARD_CLEAR_3_LINES = 60
        REWARD_CLEAR_4_LINES = 100 
        PENALTY_GAME_OVER = -50 

        # 1. Apply rotation and horizontal position from action_idx.
        # This modifies self.current_piece_* attributes if valid at spawn.
        initial_placement_valid = self._apply_action_to_piece(action_idx)

        if not initial_placement_valid:
            self.game_over = True
            reward = PENALTY_GAME_OVER
            print(f"DEBUG: Invalid initial placement chosen by action {action_idx}. Game Over.") # MODIFIED: Uncommented
            print(f"DEBUG: Board state at invalid initial placement Game Over:") # MODIFIED: Uncommented
            for r_idx, row_val in enumerate(self.board): # MODIFIED: Uncommented
                 print(f"DEBUG: Board row {r_idx}: {row_val}") # MODIFIED: Uncommented
            return self._get_state(), reward, self.game_over, {"info": "invalid_initial_placement_game_over"}, [False] * self.action_space_n

        # 2. Piece's initial rotation and x-position are set and valid at spawn y. Now drop it.
        # self.current_piece_y is currently at spawn (e.g., 0)
        print(f"DEBUG: Piece before drop: Y={self.current_piece_y}, X={self.current_piece_x}, Shape={self.current_piece_shape}") # MODIFIED: Added and Uncommented
        while not self._check_collision(self.current_piece_shape, (self.current_piece_x, self.current_piece_y + 1)):
            self.current_piece_y += 1
        
        print(f"DEBUG: Piece landed at Y: {self.current_piece_y}") # MODIFIED: Uncommented

        # 3. Lock the piece at its final position and clear lines.
        lines_cleared_this_turn_val = self._lock_piece() 
            
        if lines_cleared_this_turn_val == 1: reward = REWARD_CLEAR_1_LINE
        elif lines_cleared_this_turn_val == 2: reward = REWARD_CLEAR_2_LINES
        elif lines_cleared_this_turn_val == 3: reward = REWARD_CLEAR_3_LINES
        elif lines_cleared_this_turn_val >= 4: reward = REWARD_CLEAR_4_LINES 
        else: 
            reward = REWARD_PLACE_SUCCESS
            
        # 4. Spawn a new piece. This might set self.game_over to True.
        self._new_piece() 
        
        # Get valid actions for the NEW piece/state
        next_valid_actions_mask = self.get_valid_actions() if not self.game_over else [False] * self.action_space_n
        
        if self.game_over: 
            # If game over happened during _new_piece() or because the board is full.
            # The reward for the placement is already set.
            # Apply game over penalty. This should override the placement reward.
            print(f"DEBUG: Game over after _new_piece call. Original reward was {reward}, setting to {PENALTY_GAME_OVER}") # MODIFIED: Uncommented
            reward = PENALTY_GAME_OVER 
                
        info = {
            "lines_cleared": lines_cleared_this_turn_val,
            "score": self.score,
            "holes": self._get_game_stats()[1], # Assuming _get_game_stats is efficient enough
            "bumpiness": self._get_game_stats()[2],
            "current_piece_idx": self.current_piece_idx, # This will be the *new* piece after _new_piece()
            "next_piece_idx": self.next_piece_idx_queue[0] if self.next_piece_idx_queue else None,
            "held_piece_idx": self.held_piece_idx,
            "can_hold": self.can_hold,
            "game_over": self.game_over
        }

        next_state = self._get_state()
        done = self.game_over

        print(f"DEBUG: Step end. Reward: {reward}, Done: {done}, Lines: {lines_cleared_this_turn_val}") # MODIFIED: Uncommented
        return next_state, reward, done, info, next_valid_actions_mask

    def render(self, mode='human'):
        if mode == 'human':
            render_board = self.board.copy()
            shape = self.current_piece_shape
            off_x, off_y = self.current_piece_x, self.current_piece_y
            for r, row_data in enumerate(shape):
                for c, cell in enumerate(row_data):
                    if cell:
                        if 0 <= r + off_y < self.board_height and 0 <= c + off_x < self.board_width:
                            render_board[r + off_y, c + off_x] = 2
            print("+" + "--" * self.board_width + "+")
            for r_idx in range(self.board_height):
                row_str = "|"
                for c_idx in range(self.board_width):
                    if render_board[r_idx,c_idx] == 1: row_str += "XX"
                    elif render_board[r_idx,c_idx] == 2: row_str += "##"
                    else: row_str += "  "
                row_str += "|"
                print(row_str)
            print("+" + "--" * self.board_width + "+")
            print(f"Score: {self.score}, Lines: {self.lines_cleared_total}")
            if self.game_over: print("GAME OVER")
            print("\n")

    def close(self):
        """
        Closes the environment. 
        For console-based rendering, this might not do much, 
        but it's good practice for API consistency.
        """
        pass

if __name__ == '__main__':
    env = TetrisEnv()
    print("Tetris Environment Initialized.")
    # Modified to handle tuple unpacking from env.reset()
    initial_state, initial_valid_actions_mask = env.reset()
    print(f"Initial state shape: {np.array(initial_state).shape}")
    print(f"Initial valid actions mask (first 10): {initial_valid_actions_mask[:10]}") # Example print

    for i in range(5):
        # Example of how to use the mask, though here we just pick a random action from all for simplicity in test
        # In a real scenario, you might want to pick from valid actions or pass the mask to an agent
        action = random.randrange(env.action_space_n)
        print(f"Step {i+1}, Action: {action}")
        # Modified to handle tuple unpacking from env.step()
        next_state, reward, done, info, next_valid_actions_mask = env.step(action)
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        print(f"Next valid actions mask (first 10): {next_valid_actions_mask[:10]}") # Example print
        if done: 
            print("Game Over during test.")
            break
    print("\nTesting specific piece placement and line clear:")
    # Modified to handle tuple unpacking from env.reset()
    state, valid_actions_mask = env.reset()
    env.render()
    test_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    for i, action in enumerate(test_actions):
        if env.game_over: 
            print("Game ended early in specific test.")
            break
        print(f"Test Action {i}: {action} (Rot: {action // env.board_width}, Col: {action % env.board_width})")
        # Here, we are not using an agent, so we don't strictly need to use the mask for action selection in this test loop.
        # The environment internally handles the action.
        # Modified to handle tuple unpacking from env.step()
        next_state, reward, done, info, next_valid_actions_mask = env.step(action)
        env.render()
        print(f"  Reward: {reward}, Done: {done}, Lines: {info.get('lines_cleared',0)}")
        # Update state and mask for the next iteration if needed by a more complex test logic
        state = next_state
        valid_actions_mask = next_valid_actions_mask
    print("Environment testing finished.")
