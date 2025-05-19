import curses
import random
import time

shapes = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 0, 0], [1, 1, 1]],
    [[0, 0, 1], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
]

def rotate(shape):
    return [list(row) for row in zip(*shape[::-1])]

def check(board, shape, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                if y + off_y >= len(board) or x + off_x < 0 or x + off_x >= len(board[0]) or board[y + off_y][x + off_x]:
                    return False
    return True

def join_matrix(board, shape, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                board[y + off_y][x + off_x] = cell

def clear_rows(board):
    new_board = [row for row in board if any(cell == 0 for cell in row)]
    cleared = len(board) - len(new_board)
    for _ in range(cleared):
        new_board.insert(0, [0 for _ in range(10)])
    return new_board, cleared

def get_height(board):
    """获取当前面板的最高点"""
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j]:
                return len(board) - i
    return 0

def count_holes(board):
    """计算空洞数量"""
    holes = 0
    for j in range(len(board[0])):
        found_block = False
        for i in range(len(board)):
            if board[i][j]:
                found_block = True
            elif found_block:
                holes += 1
    return holes

def get_bumpiness(board):
    """计算表面平整度"""
    heights = []
    for j in range(len(board[0])):
        for i in range(len(board)):
            if board[i][j]:
                heights.append(len(board) - i)
                break
        else:
            heights.append(0)
    
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i + 1])
    return bumpiness

def evaluate_position(board, cleared_lines):
    """评估当前位置的分数"""
    height = get_height(board)
    holes = count_holes(board)
    bumpiness = get_bumpiness(board)
    
    # 权重设置
    weights = {
        'cleared_lines': 100,
        'holes': -40,
        'bumpiness': -10,
        'height': -20
    }
    
    return (weights['cleared_lines'] * cleared_lines +
            weights['holes'] * holes +
            weights['bumpiness'] * bumpiness +
            weights['height'] * height)

def find_best_move(board, piece):
    """找到最佳落点"""
    best_score = float('-inf')
    best_move = None
    best_rotation = 0
    
    current_piece = piece
    for rotation in range(4):
        for x in range(-2, len(board[0])+2):
            offset = [x, 0]
            if check(board, current_piece, offset):
                # 模拟下落
                while check(board, current_piece, [offset[0], offset[1]+1]):
                    offset[1] += 1
                
                # 模拟放置
                temp_board = [row[:] for row in board]
                join_matrix(temp_board, current_piece, offset)
                new_board, cleared = clear_rows(temp_board)
                
                # 评估位置
                score = evaluate_position(new_board, cleared)
                
                if score > best_score:
                    best_score = score
                    best_move = offset
                    best_rotation = rotation
        
        current_piece = rotate(current_piece)
    
    return best_rotation, best_move

def main(stdscr):
    curses.curs_set(0)
    board = [[0 for _ in range(10)] for _ in range(20)]
    current = random.choice(shapes)
    offset = [3, 0]
    score = 0
    auto_mode = False  # 自动模式开关
    stdscr.nodelay(True)
    last_move = time.time()
    
    while True:
        stdscr.clear()
        # 绘制上边框
        stdscr.addstr(0, 0, "+" + "-" * 20 + "+")
        
        # 绘制游戏区域和左右边框
        for y, row in enumerate(board):
            stdscr.addstr(y+1, 0, "|")  # 左边框
            for x, cell in enumerate(row):
                if cell:
                    stdscr.addstr(y+1, x*2+1, "[]")
            stdscr.addstr(y+1, 21, "|")  # 右边框
        
        # 绘制当前方块
        for y, row in enumerate(current):
            for x, cell in enumerate(row):
                if cell and 0 <= y+offset[1] < 20 and 0 <= x+offset[0] < 10:
                    stdscr.addstr(y+offset[1]+1, (x+offset[0])*2+1, "[]")
        
        # 绘制下边框
        stdscr.addstr(21, 0, "+" + "-" * 20 + "+")
        
        # 显示分数和模式
        stdscr.addstr(1, 24, f"Score: {score}")
        stdscr.addstr(3, 24, f"Mode: {'Auto' if auto_mode else 'Manual'}")
        stdscr.refresh()
        
        key = stdscr.getch()
        if key == ord('a'):  # 按'a'键切换自动/手动模式
            auto_mode = not auto_mode
        
        if auto_mode:
            # 在自动模式下，计算最佳移动
            rotations, target = find_best_move(board, current)
            
            # 执行旋转
            while rotations > 0:
                current = rotate(current)
                rotations -= 1
            
            # 执行水平移动
            if offset[0] < target[0] and check(board, current, [offset[0]+1, offset[1]]):
                offset[0] += 1
            elif offset[0] > target[0] and check(board, current, [offset[0]-1, offset[1]]):
                offset[0] -= 1
            
            # 执行下落
            if check(board, current, [offset[0], offset[1]+1]):
                offset[1] += 1
        else:
            # 手动模式的控制
            if key == curses.KEY_LEFT and check(board, current, [offset[0]-1, offset[1]]):
                offset[0] -= 1
            elif key == curses.KEY_RIGHT and check(board, current, [offset[0]+1, offset[1]]):
                offset[0] += 1
            elif key == curses.KEY_DOWN and check(board, current, [offset[0], offset[1]+1]):
                offset[1] += 1
            elif key == curses.KEY_UP:
                rotated = rotate(current)
                if check(board, rotated, offset):
                    current = rotated
        
        # 自动下落逻辑
        if time.time() - last_move > 0.5:
            if check(board, current, [offset[0], offset[1]+1]):
                offset[1] += 1
            else:
                join_matrix(board, current, offset)
                board, cleared = clear_rows(board)
                score += cleared * 100  # 每消除一行得100分
                current = random.choice(shapes)
                offset = [3, 0]
                if not check(board, current, offset):
                    stdscr.addstr(10, 10, "GAME OVER")
                    stdscr.refresh()
                    time.sleep(2)
                    break
            last_move = time.time()
        
        time.sleep(0.05)

if __name__ == "__main__":
    curses.wrapper(main)