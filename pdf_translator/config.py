"""Configuration module for PDF Translator."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for PDF processing."""
    
    model: str = "gpt-3.5-turbo"
    prompt: str = "Please translate this text to English and provide a clean, formatted version."
    max_retries: int = 3
    rate_limit_rpm: int = 60
    concurrent_requests: int = 3
    verbose: bool = False
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    start_page: Optional[int] = None  # First page to translate (1-based), None means start from page 1
    end_page: Optional[int] = None    # Last page to translate (1-based), None means translate to last page
    
    @property
    def rate_limit_delay(self) -> float:
        """Calculate delay between requests in seconds based on rate limit."""
        return 60.0 / self.rate_limit_rpm
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "prompt": self.prompt,
            "max_retries": self.max_retries,
            "rate_limit_rpm": self.rate_limit_rpm,
            "concurrent_requests": self.concurrent_requests,
            "verbose": self.verbose,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "start_page": self.start_page,
            "end_page": self.end_page,
        }
    
    def get_page_range(self, total_pages: int) -> tuple[int, int]:
        """Get the normalized page range for processing.
        
        Args:
            total_pages: Total number of pages in the PDF
            
        Returns:
            Tuple of (start_page, end_page) in 1-based indexing
            
        Raises:
            ValueError: If page range is invalid
        """
        # Default values
        start = self.start_page if self.start_page is not None else 1
        end = self.end_page if self.end_page is not None else total_pages
        
        # Validate range
        if start < 1:
            raise ValueError(f"Start page must be >= 1, got {start}")
        if end > total_pages:
            raise ValueError(f"End page {end} exceeds total pages {total_pages}")
        if start > end:
            raise ValueError(f"Start page {start} cannot be greater than end page {end}")
        
        return start, end
