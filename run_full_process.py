#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tetris AI Main Process Runner
This script provides a unified way to run both supervised learning and evolutionary learning processes
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_supervised_learning(args):
    """Run supervised learning pipeline with specified arguments"""
    supervised_path = os.path.join(os.path.dirname(__file__), "supervised_learning")
    script_path = os.path.join(supervised_path, "core", "training_pipeline.py")
    
    cmd = [sys.executable, script_path]
    
    if args.collect:
        cmd.append("--collect")
    if args.games:
        cmd.extend(["--games", str(args.games)])
    if args.no_merge:
        cmd.append("--no-merge")
    if args.architecture:
        cmd.extend(["--architecture", args.architecture])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch:
        cmd.extend(["--batch", str(args.batch)])
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
        
    print(f"Running supervised learning with command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_evolutionary_learning(args):
    """Run evolutionary learning pipeline with specified arguments"""
    evolutionary_path = os.path.join(os.path.dirname(__file__), "evolutionary_learning")
    script_path = os.path.join(evolutionary_path, "core", "tetris_evolution.py")
    
    cmd = [sys.executable, script_path]
    
    if args.population:
        cmd.extend(["--population", str(args.population)])
    if args.generations:
        cmd.extend(["--generations", str(args.generations)])
    if args.mutation_rate:
        cmd.extend(["--mutation-rate", str(args.mutation_rate)])
        
    print(f"Running evolutionary learning with command: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Tetris AI Full Process Runner")
    subparsers = parser.add_subparsers(dest="mode", help="Learning mode")
    
    # Supervised learning arguments
    sup_parser = subparsers.add_parser("supervised", help="Run supervised learning")
    sup_parser.add_argument("--collect", action="store_true", help="Collect new training data")
    sup_parser.add_argument("--games", type=int, help="Number of games for data collection")
    sup_parser.add_argument("--no-merge", action="store_true", help="Do not merge with existing data")
    sup_parser.add_argument("--architecture", choices=["standard", "improved", "robust", "all"], help="Model architecture to train")
    sup_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    sup_parser.add_argument("--batch", type=int, help="Batch size")
    sup_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    
    # Evolutionary learning arguments
    evo_parser = subparsers.add_parser("evolutionary", help="Run evolutionary learning")
    evo_parser.add_argument("--population", type=int, help="Population size")
    evo_parser.add_argument("--generations", type=int, help="Number of generations")
    evo_parser.add_argument("--mutation-rate", type=float, help="Mutation rate")
    
    args = parser.parse_args()
    
    if args.mode == "supervised":
        run_supervised_learning(args)
    elif args.mode == "evolutionary":
        run_evolutionary_learning(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()