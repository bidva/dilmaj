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
from .utils import calculate_text_tokens, estimate_cost, validate_api_key

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

console = Console()


def _estimate_processing_cost(pages: list[str], prompt: str, model: str) -> dict:
    """Estimate the cost of processing pages with the given model.
    
    Args:
        pages: List of page contents
        prompt: Processing prompt
        model: Model name
        
    Returns:
        Dictionary with cost estimation details
    """
    total_input_tokens = 0
    total_output_tokens_estimate = 0
    
    # Calculate input tokens for each page
    for page in pages:
        if page.strip():  # Only count non-empty pages
            # Input tokens = prompt + page content
            page_input_tokens = calculate_text_tokens(prompt + "\n\nPage Content:\n" + page)
            total_input_tokens += page_input_tokens
            
            # Rough estimate: output is typically 0.5-1.5x input for translation tasks
            # Using 1.0x as a reasonable estimate
            estimated_output_tokens = int(page_input_tokens * 1.0)
            total_output_tokens_estimate += estimated_output_tokens
    
    # Calculate cost
    estimated_cost = estimate_cost(total_input_tokens, total_output_tokens_estimate, model)
    
    non_empty_pages = len([p for p in pages if p.strip()])
    
    return {
        "pages_to_process": non_empty_pages,
        "total_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens_estimate,
        "estimated_total_tokens": total_input_tokens + total_output_tokens_estimate,
        "estimated_cost_usd": estimated_cost,
        "avg_tokens_per_page": (total_input_tokens + total_output_tokens_estimate) // non_empty_pages if non_empty_pages > 0 else 0,
    }


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
    default="gpt-4o-mini",
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
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without making API calls",
)
@click.option(
    "--estimate-cost",
    is_flag=True,
    help="Estimate API costs before processing",
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
    dry_run: bool,
    estimate_cost: bool,
) -> None:
    """Process a PDF file by slicing it into pages and processing each with GPT."""
    
    # Validate API key (only if not dry run)
    if not dry_run:
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
    
    # Create output directory (only if not dry run)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display basic info
    console.print(f"[green]PDF file:[/green] {pdf_path}")
    if not dry_run:
        console.print(f"[green]Output directory:[/green] {output_dir}")
    console.print(f"[green]Model:[/green] {model}")
    console.print(f"[green]Rate limit:[/green] {rate_limit} requests/minute")
    console.print(f"[green]Concurrent requests:[/green] {concurrent}")
    
    if start_page is not None or end_page is not None:
        page_range_text = f"Pages: {start_page or 1}-{end_page or 'last'}"
        console.print(f"[green]Page range:[/green] {page_range_text}")
    else:
        console.print(f"[green]Page range:[/green] All pages")
    
    if dry_run:
        console.print(f"[yellow]Mode:[/yellow] Dry run (no API calls will be made)")
    
    # Initialize processor
    processor = PDFProcessor(config, init_llm=not dry_run)
    
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
        
        # Extract pages for analysis
        console.print("[blue]Extracting pages for analysis...[/blue]")
        pages = processor.extract_pages(pdf_path)
        non_empty_pages = [p for p in pages if p.strip()]
        console.print(f"[green]Found {len(non_empty_pages)} non-empty pages to process[/green]")
        
        # Cost estimation
        if estimate_cost or dry_run:
            console.print("\n[blue]📊 Cost Estimation[/blue]")
            cost_info = _estimate_processing_cost(pages, prompt, model)
            
            console.print(f"[cyan]Pages to process:[/cyan] {cost_info['pages_to_process']}")
            console.print(f"[cyan]Estimated input tokens:[/cyan] {cost_info['total_input_tokens']:,}")
            console.print(f"[cyan]Estimated output tokens:[/cyan] {cost_info['estimated_output_tokens']:,}")
            console.print(f"[cyan]Estimated total tokens:[/cyan] {cost_info['estimated_total_tokens']:,}")
            console.print(f"[cyan]Average tokens per page:[/cyan] {cost_info['avg_tokens_per_page']:,}")
            console.print(f"[cyan]Estimated cost:[/cyan] ${cost_info['estimated_cost_usd']:.4f} USD")
            
            if cost_info['estimated_cost_usd'] > 1.0:
                console.print(f"[yellow]⚠️  High cost estimate (>${cost_info['estimated_cost_usd']:.2f}). Consider using a smaller page range or cheaper model.[/yellow]")
            
            if not dry_run and estimate_cost:
                # Ask for confirmation if cost is significant
                if cost_info['estimated_cost_usd'] > 0.50:
                    proceed = click.confirm(f"\nEstimated cost is ${cost_info['estimated_cost_usd']:.4f}. Do you want to proceed?")
                    if not proceed:
                        console.print("[yellow]Operation cancelled by user.[/yellow]")
                        return
        
        # Exit if dry run
        if dry_run:
            console.print("\n[yellow]🔍 Dry run complete - no API calls were made[/yellow]")
            return
        
        # Process the PDF
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            # Skip extraction since we already did it for cost estimation
            extract_task = progress.add_task("Pages already extracted", total=len(pages), completed=len(pages))
            
            # Display extracted page info
            console.print(f"[green]Processing {len(non_empty_pages)} non-empty pages from the extracted range[/green]")
            
            # Process pages
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


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    "-m",
    default="gpt-4o-mini",
    help="GPT model to use for cost estimation",
)
@click.option(
    "--prompt",
    "-p",
    default="Please translate this text to English and provide a clean, formatted version.",
    help="Custom prompt for processing pages",
)
@click.option(
    "--start-page",
    "-s",
    type=int,
    help="First page to analyze (1-based). Default: 1",
)
@click.option(
    "--end-page",
    "-e",
    type=int,
    help="Last page to analyze (1-based). Default: last page",
)
def estimate(
    pdf_path: Path,
    model: str,
    prompt: str,
    start_page: Optional[int],
    end_page: Optional[int],
) -> None:
    """Estimate the cost of processing a PDF without making any API calls."""
    
    # Create configuration for analysis
    config = Config(
        model=model,
        prompt=prompt,
        start_page=start_page,
        end_page=end_page,
    )
    
    console.print(f"[green]📄 PDF Cost Estimation[/green]")
    console.print(f"[green]File:[/green] {pdf_path}")
    console.print(f"[green]Model:[/green] {model}")
    
    if start_page is not None or end_page is not None:
        page_range_text = f"Pages: {start_page or 1}-{end_page or 'last'}"
        console.print(f"[green]Page range:[/green] {page_range_text}")
    else:
        console.print(f"[green]Page range:[/green] All pages")
    
    try:
        # Initialize processor (no API key needed for estimation)
        processor = PDFProcessor(config, init_llm=False)
        
        # Get page info
        total_pages = processor.get_pdf_page_count(pdf_path)
        actual_start, actual_end = config.get_page_range(total_pages)
        page_count_to_process = actual_end - actual_start + 1
        
        console.print(f"[green]PDF has {total_pages} pages total[/green]")
        console.print(f"[green]Will analyze pages {actual_start}-{actual_end} ({page_count_to_process} pages)[/green]")
        
        # Extract pages
        console.print("\n[blue]📖 Extracting pages for analysis...[/blue]")
        pages = processor.extract_pages(pdf_path)
        non_empty_pages = [p for p in pages if p.strip()]
        
        if not non_empty_pages:
            console.print("[red]No text content found in the specified page range.[/red]")
            return
        
        console.print(f"[green]Found {len(non_empty_pages)} non-empty pages[/green]")
        
        # Cost estimation
        console.print("\n[blue]💰 Cost Analysis[/blue]")
        cost_info = _estimate_processing_cost(pages, prompt, model)
        
        console.print(f"[cyan]Pages with content:[/cyan] {cost_info['pages_to_process']}")
        console.print(f"[cyan]Estimated input tokens:[/cyan] {cost_info['total_input_tokens']:,}")
        console.print(f"[cyan]Estimated output tokens:[/cyan] {cost_info['estimated_output_tokens']:,}")
        console.print(f"[cyan]Total estimated tokens:[/cyan] {cost_info['estimated_total_tokens']:,}")
        console.print(f"[cyan]Average tokens per page:[/cyan] {cost_info['avg_tokens_per_page']:,}")
        console.print(f"[cyan]Estimated cost:[/cyan] ${cost_info['estimated_cost_usd']:.4f} USD")
        
        # Cost warnings and recommendations
        if cost_info['estimated_cost_usd'] > 5.0:
            console.print(f"[red]⚠️  Very high cost estimate (>${cost_info['estimated_cost_usd']:.2f})[/red]")
            console.print("[yellow]💡 Consider:[/yellow]")
            console.print("   • Processing fewer pages at a time")
            console.print("   • Using a cheaper model (e.g., gpt-3.5-turbo)")
            console.print("   • Splitting into smaller batches")
        elif cost_info['estimated_cost_usd'] > 1.0:
            console.print(f"[yellow]⚠️  High cost estimate (>${cost_info['estimated_cost_usd']:.2f})[/yellow]")
            console.print("[yellow]💡 Consider using gpt-3.5-turbo for lower costs[/yellow]")
        elif cost_info['estimated_cost_usd'] < 0.01:
            console.print("[green]✅ Very low cost estimate - good to go![/green]")
        else:
            console.print("[green]✅ Reasonable cost estimate[/green]")
            
        # Token efficiency info
        if cost_info['avg_tokens_per_page'] > 3000:
            console.print("[yellow]💡 Pages have a lot of text. Consider shorter prompts to reduce costs.[/yellow]")
        
        console.print(f"\n[blue]To proceed with processing, run:[/blue]")
        console.print(f"[dim]pdf-translator process {pdf_path} --model {model}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error analyzing PDF: {str(e)}[/red]")
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
