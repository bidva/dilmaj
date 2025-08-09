"""Utility functions for PDF Translator."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def validate_api_key() -> str:
    """Validate OpenAI API key from environment variables.
    
    Returns:
        The valid API key string
        
    Raises:
        ConfigurationError: If API key is missing, invalid, or is a placeholder
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if API key exists
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set your OpenAI API key in the .env file or environment."
        )
    
    # Remove whitespace
    api_key = api_key.strip()
    
    # Check if API key is empty after stripping whitespace
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is empty. "
            "Please set a valid OpenAI API key in the .env file or environment."
        )
    
    # Check for common placeholder values
    placeholder_values = {
        "your_openai_api_key_here",
        "your_api_key_here", 
        "sk-your-api-key-here",
        "replace_with_your_api_key",
        "your-openai-api-key",
        "put_your_api_key_here"
    }
    
    if api_key.lower() in placeholder_values:
        raise ConfigurationError(
            f"OPENAI_API_KEY appears to be a placeholder value: '{api_key}'. "
            "Please replace it with your actual OpenAI API key from https://platform.openai.com/account/api-keys"
        )
    
    # Basic format validation - OpenAI API keys should start with 'sk-'
    if not api_key.startswith('sk-'):
        logger.warning(f"API key does not start with 'sk-'. Please verify it's a valid OpenAI API key.")
    
    # Basic length validation - OpenAI API keys are typically 51 characters
    if len(api_key) < 20:
        logger.warning(f"API key appears unusually short ({len(api_key)} characters). Please verify it's complete.")
    
    return api_key


def generate_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of a file for caching purposes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to JSON file with proper error handling.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load data from JSON file with error handling.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_cost(
    input_tokens: int, 
    output_tokens: int, 
    model: str = "gpt-4o-mini"
) -> float:
    """Estimate the cost of API calls based on token usage.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    # Pricing as of 2025 (in USD per 1K tokens)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4.1": {"input": 2.0, "output": 8.0},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4.5": {"input": 75.0, "output": 150.0},
    }
    
    if model not in pricing:
        logger.warning(f"Unknown model for cost estimation: {model}")
        return 0.0
    
    input_cost = (input_tokens / 1000) * pricing[model]["input"]
    output_cost = (output_tokens / 1000) * pricing[model]["output"]
    
    return input_cost + output_cost


def create_progress_tracker():
    """Create a simple progress tracking context manager."""
    
    class ProgressTracker:
        def __init__(self):
            self.start_time = None
            self.current_step = 0
            self.total_steps = 0
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.time() - self.start_time
                logger.info(f"Operation completed in {format_duration(duration)}")
        
        def update(self, step: int, total: int, description: str = ""):
            self.current_step = step
            self.total_steps = total
            percentage = (step / total) * 100 if total > 0 else 0
            logger.info(f"{description} ({step}/{total}) - {percentage:.1f}%")
    
    return ProgressTracker()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Characters that are invalid in filenames
    invalid_chars = '<>:"/\\|?*'
    
    sanitized = filename
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def calculate_text_tokens(text: str) -> int:
    """Rough estimation of token count for text.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4
