"""Document processing module with GPT and LLaMA integration.

Now processes extracted text in paragraph units (not per page) and supports
multiple document types (PDF, DOCX/DOC) via pluggable extractors.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from langchain.schema import HumanMessage
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
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
        self.llm: Optional[Union[ChatOpenAI, LlamaCpp]] = None
        # Available extractors; selection is based on file suffix
        self.extractors: list[DocumentExtractor] = [
            PDFExtractor(),
            WordExtractor(),
        ]
        # Default to PDF extractor for legacy flows
        self.extractor: DocumentExtractor = PDFExtractor()

        if init_llm:
            if self.config.is_local_model:
                self._init_local_llm()
            else:
                self._init_openai_llm()

        self._setup_logging()

    def _init_openai_llm(self) -> None:
        """Initialize OpenAI LLM."""
        # Create ChatOpenAI instance with basic parameters
        if self.config.max_tokens is not None:
            self.llm = ChatOpenAI(  # type: ignore
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,  # type: ignore
            )
        else:
            self.llm = ChatOpenAI(  # type: ignore
                model=self.config.model, temperature=self.config.temperature
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

    def get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get the total number of pages/units in a document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Total number of pages in the PDF

        Raises:
            PDFProcessingError: If PDF cannot be read
        """
        try:
            extractor = self._select_extractor(pdf_path)
            if not extractor.supports(pdf_path):
                raise PDFProcessingError(
                    f"No extractor available for file type: {pdf_path.suffix}"
                )
            # Update default extractor to the selected one
            self.extractor = extractor
            return extractor.get_page_count(pdf_path)
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Failed to read PDF {pdf_path}: {str(e)}")

    def extract_pages(self, pdf_path: Path) -> List[str]:
        """Extract text content as paragraphs from the specified page range.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of paragraph strings derived from the specified page range

        Raises:
            PDFProcessingError: If PDF cannot be processed
        """
        try:
            total_pages = self.get_pdf_page_count(pdf_path)

            # Get page range from config
            try:
                start_page, end_page = self.config.get_page_range(total_pages)
            except ValueError as e:
                raise PDFProcessingError(f"Invalid page range: {str(e)}")

            logger.info(
                f"Extracting paragraphs from pages {start_page} to {end_page} "
                f"of {total_pages} total pages"
            )

            options = ExtractionOptions(
                start_page=start_page,
                end_page=end_page,
                preprocess_text=self.config.preprocess_text,
                remove_headers_footers=self.config.remove_headers_footers,
                chunk_paragraphs=self.config.chunk_paragraphs,
            )
            # Ensure extractor is selected based on file type
            extractor = self._select_extractor(pdf_path)
            self.extractor = extractor
            pages = extractor.extract_pages(pdf_path, options)
            if not any(page.strip() for page in pages):
                raise PDFProcessingError(
                    "No paragraph content found in pages "
                    f"{start_page}-{end_page} of PDF: {pdf_path}"
                )
            return pages

        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Failed to process document {pdf_path}: {str(e)}")

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
        if self.llm is None:
            raise GPTProcessingError(
                "LLM not initialized. Cannot process paragraphs in " "dry run mode."
            )

        try:
            # Apply rate limiting
            await asyncio.sleep(self.config.rate_limit_delay)

            # Use the config's prompt formatting method
            input_text = self.config.format_prompt(page_content)

            # Process based on model type
            if self.config.is_local_model:
                # For local models (LlamaCpp), use direct invoke in executor
                def run_local_model() -> str:
                    return self.llm.invoke(input_text)  # type: ignore

                loop = asyncio.get_event_loop()
                response_content = await loop.run_in_executor(None, run_local_model)
                # Ensure response_content is a string
                if not isinstance(response_content, str):
                    response_content = str(response_content)
            else:
                # For OpenAI models, use LangChain chat format
                message = HumanMessage(content=input_text)
                response = await self.llm.ainvoke([message])
                response_content = str(response.content) if response.content else ""

            result: Dict[str, Any] = {
                "paragraph_number": page_num,
                "original_content": page_content,
                "processed_content": response_content,
                "timestamp": time.time(),
                "model": self.config.model,
                "model_type": self.config.model_type,
                "model_path": self.config.model_path
                if self.config.is_local_model
                else None,
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

            # Get page range info for summary
            total_pages_in_pdf = len(pages)
            start_page = 1
            end_page = len(pages)

            if pdf_path and pdf_path.exists():
                try:
                    total_pages_in_pdf = self.get_pdf_page_count(pdf_path)
                    start_page, end_page = self.config.get_page_range(
                        total_pages_in_pdf
                    )
                except Exception:
                    pass  # Use defaults

            summary = {
                "total_pages_in_pdf": total_pages_in_pdf,
                "page_range_requested": f"{start_page}-{end_page}",
                "paragraphs_processed": len(pages),
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
