"""Document processing module with pluggable LLM providers.

Now processes extracted text in paragraph units (not per page) and supports
multiple document types (PDF, DOCX/DOC) via pluggable extractors.
Backends for text generation are abstracted behind a provider interface,
currently implemented for OpenAI (future providers can be added later).
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

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
from .extractors import (
    DocumentExtractor,
    ExtractionOptions,
    PDFExtractor,
    WordExtractor,
)
from .providers import LLMProvider, ProviderFactory

logger = logging.getLogger(__name__)
console = Console()


class PDFProcessor:
    """Process documents by extracting paragraphs and processing with LLMs."""

    def __init__(self, config: Config, init_llm: bool = True):
        """Initialize the PDF processor.

        Args:
            config: Configuration object containing processing parameters
            init_llm: Whether to initialize the LLM (set to False for dry runs)
        """
        self.config = config
        # LLM provider implementing a unified interface across backends
        self.provider: Optional[LLMProvider] = None
        # Available extractors; selection is based on file suffix
        self.extractors: List[DocumentExtractor] = [
            PDFExtractor(),
            WordExtractor(),
        ]
        # Default to PDF extractor for legacy flows
        self.extractor: DocumentExtractor = PDFExtractor()

        if init_llm:
            self._init_provider()

        self._setup_logging()

    def _init_provider(self) -> None:
        """Initialize the LLM provider based on configuration."""
        # Initialize provider based on configuration
        self.provider = ProviderFactory.from_config(self.config)

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format=("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        )

    def _select_extractor(self, path: Path) -> DocumentExtractor:
        """Select an extractor that supports the given file.

        Falls back to the first extractor (PDF) if none explicitly support,
        which will then raise a clearer error downstream.
        """
        for ex in self.extractors:
            try:
                if ex.supports(path):
                    return ex
            except Exception:
                continue
        return self.extractor

    def extract_paragraphs(self, path: Path) -> List[str]:
        """Extract text content as paragraphs for the entire document.

        Args:
            path: Path to the input document (PDF/DOCX/DOC)

        Returns:
            List of paragraph strings for the whole document

        Raises:
            PDFProcessingError: If the document cannot be processed
        """
        try:
            # Ensure extractor is selected based on file type
            extractor = self._select_extractor(path)
            self.extractor = extractor

            options = ExtractionOptions(
                preprocess_text=self.config.preprocess_text,
                remove_headers_footers=self.config.remove_headers_footers,
                chunk_paragraphs=self.config.chunk_paragraphs,
            )

            paragraphs = extractor.extract_paragraphs(path, options)
            if not any(p.strip() for p in paragraphs):
                raise PDFProcessingError(
                    f"No paragraph content found in document: {path}"
                )
            return paragraphs

        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Failed to process document {path}: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _process_single_page(
        self, page_content: str, page_num: int
    ) -> Dict[str, Any]:
        """Process a single paragraph with LLM using retry logic.

        Args:
            page_content: Text content of the paragraph
            page_num: Paragraph number for reference

        Returns:
            Dictionary containing processed result

        Raises:
            GPTProcessingError: If LLM processing fails after retries
        """
        if self.provider is None:
            raise GPTProcessingError(
                "LLM not initialized. Cannot process paragraphs in " "dry run mode."
            )

        try:
            # Apply rate limiting
            await asyncio.sleep(self.config.rate_limit_delay)

            input_text = page_content

            # Delegate generation to the provider (uniform API)
            provider = self.provider
            assert provider is not None
            response_content = await provider.ainvoke(input_text)

            result: Dict[str, Any] = {
                "paragraph_number": page_num,
                "original_content": page_content,
                "processed_content": response_content,
                "timestamp": time.time(),
                "model": self.config.model,
                "model_type": "openai",
                "prompt": self.config.prompt,
            }

            logger.info(f"Successfully processed paragraph {page_num}")
            return result

        except Exception as e:
            error_msg = f"Failed to process paragraph {page_num}: {str(e)}"
            logger.error(error_msg)
            raise GPTProcessingError(error_msg)

    async def _process_pages_batch(
        self,
        pages: List[str],
        output_dir: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple paragraphs concurrently with rate limiting.

        Args:
            pages: List of paragraph contents
            output_dir: Directory to save results
            progress_callback: Optional callback for progress updates

        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        # Collect successful results only; errors are logged and filtered
        completed = 0

        # Paragraph numbering starts at 1 within the extracted range
        start_paragraph = 1

        async def process_with_semaphore(
            page_content: str, page_index: int
        ) -> Dict[str, Any]:
            nonlocal completed
            async with semaphore:
                actual_paragraph_num = start_paragraph + page_index
                result: Dict[str, Any] = await self._process_single_page(
                    page_content, actual_paragraph_num
                )

                # Save individual result
                result_file = output_dir / f"paragraph_{actual_paragraph_num:03d}.json"
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                # Save processed content as text file
                text_file = (
                    output_dir / f"paragraph_{actual_paragraph_num:03d}_processed.txt"
                )
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(result["processed_content"])

                completed += 1
                if progress_callback:
                    progress_callback(completed)

                return result

        # Create tasks for all paragraphs
        tasks = [
            process_with_semaphore(page_content, page_index)
            for page_index, page_content in enumerate(pages)
            if page_content.strip()  # Skip empty paragraphs
        ]

        # Process all paragraphs
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        successful_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results_raw):
            if isinstance(result, Exception):
                logger.error(f"Paragraph {start_paragraph + i} failed: {result}")
            else:
                successful_results.append(cast(Dict[str, Any], result))

        return successful_results

    def process_pages_async(
        self,
        pages: List[str],
        output_dir: Path,
        pdf_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for async paragraph processing.

        Args:
            pages: List of paragraph contents
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
            summary = {
                "paragraphs_total": len(pages),
                "successful_paragraphs": len(results),
                "failed_paragraphs": len(pages) - len(results),
                "config": self.config.to_dict(),
                "results": results,
                "timestamp": time.time(),
            }

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Create combined text file
            combined_file = output_dir / "combined_processed.txt"
            with open(combined_file, "w", encoding="utf-8") as f:
                f.write("# Combined Processed Results\n\n")
                for result in sorted(results, key=lambda x: x["paragraph_number"]):
                    f.write(f"## Paragraph {result['paragraph_number']}\n\n")
                    f.write(result["processed_content"])
                    f.write("\n\n" + "=" * 50 + "\n\n")

            return results

        finally:
            loop.close()
