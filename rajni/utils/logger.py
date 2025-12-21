"""
Logging utilities for RAJNI-ViT experiments.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "rajni",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, no file logging)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console
        
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamp() -> str:
    """
    Get current timestamp string for log files.
    
    Returns:
        timestamp: Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
