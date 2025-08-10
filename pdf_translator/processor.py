"""PDF processing module with GPT and LLaMA integration."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain.schema import HumanMessage
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from pypdf import PdfReader
from rich.console import Console
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config
from .exceptions import GPTProcessingError, PDFProcessingError

logger = logging.getLogger(__name__)
console = Console()


class PDFProcessor:
    """Process PDF files by slicing pages and processing with GPT or local LLMs."""
    
    def __init__(self, config: Config, init_llm: bool = True):
        """Initialize the PDF processor.
        
        Args:
            config: Configuration object containing processing parameters
            init_llm: Whether to initialize the LLM (set to False for dry runs)
        """
        self.config = config
        self.llm = None
        
        if init_llm:
            if config.is_local_model:
                self._init_local_llm()
            else:
                self._init_openai_llm()
        
        self._setup_logging()
    
    def _init_openai_llm(self) -> None:
        """Initialize OpenAI LLM."""
        self.llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
    
    def _init_local_llm(self) -> None:
        """Initialize local LLaMA LLM."""
        if not self.config.model_path:
            raise ValueError("model_path must be specified for local models")
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.llm = LlamaCpp(
            model_path=str(model_path),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens or 512,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            verbose=self.config.verbose,
        )
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get the total number of pages in a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Total number of pages in the PDF
            
        Raises:
            PDFProcessingError: If PDF cannot be read
        """
        try:
            reader = PdfReader(str(pdf_path))
            return len(reader.pages)
        except Exception as e:
            raise PDFProcessingError(f"Failed to read PDF {pdf_path}: {str(e)}")

    def extract_pages(self, pdf_path: Path) -> List[str]:
        """Extract text content from specified pages of the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text content for each page in the specified range
            
        Raises:
            PDFProcessingError: If PDF cannot be processed
        """
        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            
            # Get page range from config
            try:
                start_page, end_page = self.config.get_page_range(total_pages)
            except ValueError as e:
                raise PDFProcessingError(f"Invalid page range: {str(e)}")
            
            logger.info(f"Processing pages {start_page} to {end_page} of {total_pages} total pages")
            
            pages = []
            
            # Extract only the specified page range (convert to 0-based indexing)
            for page_num in range(start_page - 1, end_page):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        pages.append(text.strip())
                        logger.debug(f"Extracted page {page_num + 1}: {len(text)} characters")
                    else:
                        logger.warning(f"Page {page_num + 1} appears to be empty")
                        # Add empty string to maintain page numbering consistency
                        pages.append("")
                except Exception as e:
                    logger.error(f"Failed to extract text from page {page_num + 1}: {e}")
                    # Add empty string to maintain page numbering consistency
                    pages.append("")
                    continue
            
            if not any(page.strip() for page in pages):
                raise PDFProcessingError(f"No text content found in pages {start_page}-{end_page} of PDF: {pdf_path}")
            
            return pages
            
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Failed to process PDF {pdf_path}: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _process_single_page(
        self, 
        page_content: str, 
        page_num: int
    ) -> Dict[str, Any]:
        """Process a single page with LLM using retry logic.
        
        Args:
            page_content: Text content of the page
            page_num: Page number for reference
            
        Returns:
            Dictionary containing processed result
            
        Raises:
            GPTProcessingError: If LLM processing fails after retries
        """
        if self.llm is None:
            raise GPTProcessingError("LLM not initialized. Cannot process pages in dry run mode.")
            
        try:
            # Apply rate limiting
            await asyncio.sleep(self.config.rate_limit_delay)
            
            # Use the config's prompt formatting method
            input_text = self.config.format_prompt(page_content)
            
            # Process based on model type
            if self.config.is_local_model:
                # For local models (LlamaCpp), use direct invoke in executor
                def run_local_model():
                    return self.llm.invoke(input_text)  # type: ignore
                
                response_content = await asyncio.get_event_loop().run_in_executor(
                    None, run_local_model
                )
            else:
                # For OpenAI models, use LangChain chat format
                message = HumanMessage(content=input_text)
                response = await self.llm.ainvoke([message])
                response_content = response.content
            
            result = {
                "page_number": page_num,
                "original_content": page_content,
                "processed_content": response_content,
                "timestamp": time.time(),
                "model": self.config.model,
                "model_type": self.config.model_type,
                "model_path": self.config.model_path if self.config.is_local_model else None,
                "prompt": self.config.prompt,
            }
            
            logger.info(f"Successfully processed page {page_num}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process page {page_num}: {str(e)}"
            logger.error(error_msg)
            raise GPTProcessingError(error_msg)
    
    async def _process_pages_batch(
        self,
        pages: List[str],
        output_dir: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple pages concurrently with rate limiting.
        
        Args:
            pages: List of page contents
            output_dir: Directory to save results
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        results = []
        completed = 0
        
        # Get the actual starting page number for correct file naming
        start_page = self.config.start_page if self.config.start_page is not None else 1
        
        async def process_with_semaphore(page_content: str, page_index: int) -> Dict[str, Any]:
            async with semaphore:
                actual_page_num = start_page + page_index
                result = await self._process_single_page(page_content, actual_page_num)
                
                # Save individual result
                result_file = output_dir / f"page_{actual_page_num:03d}.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Save processed content as text file
                text_file = output_dir / f"page_{actual_page_num:03d}_processed.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result['processed_content'])
                
                nonlocal completed
                completed += 1
                if progress_callback:
                    progress_callback(completed)
                
                return result
        
        # Create tasks for all pages
        tasks = [
            process_with_semaphore(page_content, page_index)
            for page_index, page_content in enumerate(pages)
            if page_content.strip()  # Skip empty pages
        ]
        
        # Process all pages
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Page {start_page + i} failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    def process_pages_async(
        self,
        pages: List[str],
        output_dir: Path,
        pdf_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async page processing.
        
        Args:
            pages: List of page contents
            output_dir: Directory to save results
            pdf_path: Optional path to PDF file for metadata
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processing results
        """
        # Run the async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                self._process_pages_batch(pages, output_dir, progress_callback)
            )
            
            # Save combined results
            summary_file = output_dir / "processing_summary.json"
            
            # Get page range info for summary
            total_pages_in_pdf = len(pages)
            start_page = 1
            end_page = len(pages)
            
            if pdf_path and pdf_path.exists():
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(pdf_path))
                    total_pages_in_pdf = len(reader.pages)
                    start_page, end_page = self.config.get_page_range(total_pages_in_pdf)
                except Exception:
                    pass  # Use defaults
            
            summary = {
                "total_pages_in_pdf": total_pages_in_pdf,
                "page_range_requested": f"{start_page}-{end_page}",
                "pages_processed": len(pages),
                "successful_pages": len(results),
                "failed_pages": len(pages) - len(results),
                "config": self.config.to_dict(),
                "results": results,
                "timestamp": time.time(),
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            # Create combined text file
            combined_file = output_dir / "combined_processed.txt"
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write("# Combined Processed Results\n\n")
                for result in sorted(results, key=lambda x: x['page_number']):
                    f.write(f"## Page {result['page_number']}\n\n")
                    f.write(result['processed_content'])
                    f.write("\n\n" + "="*50 + "\n\n")
            
            return results
            
        finally:
            loop.close()
