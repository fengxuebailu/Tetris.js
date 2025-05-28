#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger configuration module for the Tetris Supervised Learning System.
Provides unified logging setup with proper UTF-8 encoding.
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
import time

def setup_logger(name='tetris_training', log_dir='logs'):
    """
    Set up a logger with both file and console output using UTF-8 encoding.
    
    Args:
        name (str): Logger name
        log_dir (str): Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
        
    # Log file name with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # File handler with UTF-8 encoding
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
