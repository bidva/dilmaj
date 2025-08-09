"""CLI interface for PDF Translator."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .config import Config
from .exceptions import ConfigurationError
from .processor import PDFProcessor
from .utils import validate_api_key

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="pdf-translator")
def cli() -> None:
    """PDF Translator CLI - Slice PDFs and process with GPT models."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for results",
)
@click.option(
    "--model",
    "-m",
    default="gpt-3.5-turbo",
    help="GPT model to use for processing",
)
@click.option(
    "--prompt",
    "-p",
    default="Please translate this text to English and provide a clean, formatted version.",
    help="Custom prompt for processing pages",
)
@click.option(
    "--max-retries",
    "-r",
    type=int,
    default=3,
    help="Maximum number of retry attempts",
)
@click.option(
    "--rate-limit",
    "-l",
    type=int,
    default=60,
    help="Rate limit in requests per minute",
)
@click.option(
    "--concurrent",
    "-c",
    type=int,
    default=3,
    help="Number of concurrent requests",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--start-page",
    "-s",
    type=int,
    help="First page to translate (1-based). Default: 1",
)
@click.option(
    "--end-page",
    "-e",
    type=int,
    help="Last page to translate (1-based). Default: last page",
)
def process(
    pdf_path: Path,
    output_dir: Path,
    model: str,
    prompt: str,
    max_retries: int,
    rate_limit: int,
    concurrent: int,
    verbose: bool,
    start_page: Optional[int],
    end_page: Optional[int],
) -> None:
    """Process a PDF file by slicing it into pages and processing each with GPT."""
    
    # Validate API key
    try:
        api_key = validate_api_key()
    except ConfigurationError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)
    
    # Create configuration
    config = Config(
        model=model,
        prompt=prompt,
        max_retries=max_retries,
        rate_limit_rpm=rate_limit,
        concurrent_requests=concurrent,
        verbose=verbose,
        start_page=start_page,
        end_page=end_page,
    )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[green]Processing PDF:[/green] {pdf_path}")
    console.print(f"[green]Output directory:[/green] {output_dir}")
    console.print(f"[green]Model:[/green] {model}")
    console.print(f"[green]Rate limit:[/green] {rate_limit} requests/minute")
    console.print(f"[green]Concurrent requests:[/green] {concurrent}")
    
    if start_page is not None or end_page is not None:
        page_range_text = f"Pages: {start_page or 1}-{end_page or 'last'}"
        console.print(f"[green]Page range:[/green] {page_range_text}")
    else:
        console.print(f"[green]Page range:[/green] All pages")
    
    # Initialize processor
    processor = PDFProcessor(config)
    
    try:
        # Get total page count for validation and display
        total_pages = processor.get_pdf_page_count(pdf_path)
        
        # Validate page range
        try:
            actual_start, actual_end = config.get_page_range(total_pages)
            page_count_to_process = actual_end - actual_start + 1
        except ValueError as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            sys.exit(1)
        
        console.print(f"[green]PDF has {total_pages} pages total[/green]")
        console.print(f"[green]Will process pages {actual_start}-{actual_end} ({page_count_to_process} pages)[/green]")
        
        # Process the PDF
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            # Extract pages
            extract_task = progress.add_task("Extracting PDF pages...", total=None)
            pages = processor.extract_pages(pdf_path)
            progress.update(extract_task, total=len(pages), completed=len(pages))
            
            # Get actual page range for display
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf_path))
                total_pages = len(reader.pages)
                actual_start, actual_end = config.get_page_range(total_pages)
                console.print(f"[green]Extracted pages {actual_start}-{actual_end} ({len([p for p in pages if p.strip()])} non-empty pages)[/green]")
            except Exception:
                console.print(f"[green]Extracted {len([p for p in pages if p.strip()])} non-empty pages from PDF[/green]")
            
            # Process pages
            non_empty_pages = [p for p in pages if p.strip()]
            process_task = progress.add_task("Processing pages with GPT...", total=len(non_empty_pages))
            
            results = processor.process_pages_async(
                pages, 
                output_dir,
                pdf_path,
                progress_callback=lambda completed: progress.update(process_task, completed=completed)
            )
            
            console.print(f"[green]Successfully processed {len(results)} pages[/green]")
            console.print(f"[green]Results saved to:[/green] {output_dir}")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error processing PDF: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
