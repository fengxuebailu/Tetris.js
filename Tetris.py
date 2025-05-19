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

def main(stdscr):
    curses.curs_set(0)
    board = [[0 for _ in range(10)] for _ in range(20)]
    current = random.choice(shapes)
    offset = [3, 0]
    score = 0
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
        
        # 显示分数
        stdscr.addstr(1, 24, f"Score: {score}")
        stdscr.refresh()
        key = stdscr.getch()
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
        if time.time() - last_move > 0.5:
            if check(board, current, [offset[0], offset[1]+1]):
                offset[1] += 1
            else:
                join_matrix(board, current, offset)
                board, cleared = clear_rows(board)
                score += cleared
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