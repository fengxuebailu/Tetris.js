import random
import numpy as np
from copy import deepcopy
import json
import time
from Tetris import (shapes, rotate, check, join_matrix, clear_rows, 
                   get_height, count_holes, get_bumpiness)

class TetrisEvolution:
    def __init__(self, population_size=50, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_fitness = float('-inf')
        self.best_weights = None
        self.initialize_population()
    
    def initialize_population(self):
        """初始化种群，每个个体是一组权重参数"""
        for _ in range(self.population_size):
            weights = {
                'cleared_lines': random.uniform(0, 200),    # 消行权重
                'holes': random.uniform(-100, 0),           # 空洞权重
                'bumpiness': random.uniform(-50, 0),        # 平整度权重
                'height': random.uniform(-50, 0)            # 高度权重
            }
            self.population.append(weights)
    
    def play_game(self, weights, max_moves=500):
        """使用给定权重玩一局游戏"""
        board = [[0 for _ in range(10)] for _ in range(20)]
        score = 0
        moves = 0
        total_cleared_lines = 0
        
        while moves < max_moves:
            current_piece = random.choice(shapes)
            if not self.make_move(board, current_piece, weights):
                break
            moves += 1
        
        # 返回游戏分数和其他指标
        game_stats = {
            'score': score,
            'moves': moves,
            'final_height': get_height(board),
            'final_holes': count_holes(board),
            'final_bumpiness': get_bumpiness(board),
            'cleared_lines': total_cleared_lines
        }
        return game_stats
    
    def evaluate_position(self, board, cleared_lines, weights):
        """使用给定权重评估局面"""
        height = get_height(board)
        holes = count_holes(board)
        bumpiness = get_bumpiness(board)
        
        return (weights['cleared_lines'] * cleared_lines +
                weights['holes'] * holes +
                weights['bumpiness'] * bumpiness +
                weights['height'] * height)
    
    def make_move(self, board, piece, weights):
        """找到并执行最佳移动"""
        best_score = float('-inf')
        best_move = None
        best_rotation = 0
        best_new_board = None
        best_cleared = 0
        
        # 尝试所有可能的移动
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
                    score = self.evaluate_position(new_board, cleared, weights)
                    
                    if score > best_score:
                        best_score = score
                        best_move = offset
                        best_rotation = rotation
                        best_new_board = new_board
                        best_cleared = cleared
            
            current_piece = rotate(current_piece)
        
        if best_move is None:
            return False
        
        # 执行最佳移动
        for _ in range(best_rotation):
            piece = rotate(piece)
        
        board[:] = best_new_board[:]
        return True
    
    def evaluate_fitness(self, weights, num_games=3):
        """评估一组权重的适应度"""
        total_stats = {
            'score': 0,
            'moves': 0,
            'cleared_lines': 0
        }
        
        for _ in range(num_games):
            stats = self.play_game(weights)
            for key in total_stats:
                total_stats[key] += stats[key]
        
        # 计算平均值
        for key in total_stats:
            total_stats[key] /= num_games
        
        # 综合评分
        fitness = (total_stats['score'] * 0.4 + 
                  total_stats['moves'] * 0.3 +
                  total_stats['cleared_lines'] * 0.3)
        
        return fitness, total_stats
    
    def select_parent(self, fitness_scores):
        """使用轮盘赌选择父代"""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.randint(0, len(fitness_scores)-1)
        
        r = random.uniform(0, total_fitness)
        current_sum = 0
        for i, fitness in enumerate(fitness_scores):
            current_sum += fitness
            if current_sum > r:
                return i
        return len(fitness_scores) - 1
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        child = {}
        for key in parent1:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def mutate(self, weights):
        """变异操作"""
        mutated = weights.copy()
        for key in mutated:
            if random.random() < self.mutation_rate:
                # 根据不同参数设置不同的变异范围
                if key == 'cleared_lines':
                    mutated[key] += random.gauss(0, 20)
                elif key == 'holes':
                    mutated[key] += random.gauss(0, 10)
                else:
                    mutated[key] += random.gauss(0, 5)
        return mutated
    
    def evolve(self, generations=50):
        """进化过程"""
        generation_stats = []
        
        for generation in range(generations):
            print(f"\n进化代数: {generation + 1}/{generations}")
            
            # 评估当前种群
            fitness_scores = []
            generation_data = []
            
            for i, weights in enumerate(self.population):
                fitness, stats = self.evaluate_fitness(weights)
                fitness_scores.append(fitness)
                generation_data.append({
                    'weights': weights,
                    'fitness': fitness,
                    'stats': stats
                })
                print(f"个体 {i+1}/{self.population_size} - 适应度: {fitness:.2f}")
            
            # 更新最佳记录
            best_idx = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[best_idx]
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_weights = deepcopy(self.population[best_idx])
                print(f"\n发现新的最佳权重！适应度: {self.best_fitness:.2f}")
                print("权重:", self.best_weights)
            
            # 生成新一代
            new_population = [self.best_weights]  # 精英保留
            
            while len(new_population) < self.population_size:
                # 选择父代
                parent1_idx = self.select_parent(fitness_scores)
                parent2_idx = self.select_parent(fitness_scores)
                
                # 交叉
                child = self.crossover(
                    self.population[parent1_idx],
                    self.population[parent2_idx]
                )
                
                # 变异
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            generation_stats.append({
                'generation': generation + 1,
                'best_fitness': max(fitness_scores),
                'avg_fitness': sum(fitness_scores) / len(fitness_scores),
                'best_weights': self.best_weights
            })
            
            # 保存当前最佳权重
            self.save_weights()
            
            # 输出当前代的统计信息
            print(f"\n当前代最佳适应度: {max(fitness_scores):.2f}")
            print(f"当前代平均适应度: {sum(fitness_scores)/len(fitness_scores):.2f}")
        
        return self.best_weights, generation_stats
    
    def save_weights(self, filename='best_weights_evolved.json'):
        """保存最佳权重到文件"""
        if self.best_weights:
            with open(filename, 'w') as f:
                json.dump(self.best_weights, f, indent=4)
            print(f"\n最佳权重已保存到 {filename}")

def main():
    # 设置进化参数
    population_size = 50
    generations = 30
    mutation_rate = 0.1
    
    print("开始俄罗斯方块AI进化优化...")
    print(f"种群大小: {population_size}")
    print(f"进化代数: {generations}")
    print(f"变异率: {mutation_rate}")
    
    # 创建优化器并开始进化
    optimizer = TetrisEvolution(
        population_size=population_size,
        mutation_rate=mutation_rate
    )
    
    start_time = time.time()
    best_weights, stats = optimizer.evolve(generations=generations)
    end_time = time.time()
    
    print("\n进化完成!")
    print(f"总用时: {end_time - start_time:.2f} 秒")
    print("\n最终最佳权重:")
    for key, value in best_weights.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    main()
