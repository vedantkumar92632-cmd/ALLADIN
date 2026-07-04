import logging
import json
import os
import sys
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging.
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        # Add any extra fields passed via extra=
        for key, val in record.__dict__.items():
            if key not in ("args", "asctime", "created", "exc_info", "exc_text", 
                           "filename", "funcName", "levelname", "levelno", "lineno", 
                           "module", "msecs", "message", "msg", "name", "pathname", 
                           "process", "processName", "relativeCreated", "stack_info", "thread", "threadName"):
                log_record[key] = val
        return json.dumps(log_record)

def get_logger(name: str) -> logging.Logger:
    """
    Get or configure a structured logger.
    """
    logger = logging.getLogger(name)
    # Check if logger is already configured
    if logger.handlers:
        return logger
        
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level_str, logging.INFO))
    
    # Console Handler (Human-readable structured output or JSON depending on environment)
    console_handler = logging.StreamHandler(sys.stdout)
    
    env = os.environ.get("ENV", "development").lower()
    if env == "production":
        formatter = JsonFormatter()
    else:
        # Development: readable text-based format
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s [%(name)s] [%(filename)s:%(lineno)d]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler for persistent logs
    log_dir = "data/records"
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{log_dir}/system.log", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    except Exception:
        # If running in environment where directories can't be created, fallback gracefully
        pass
        
    return logger
