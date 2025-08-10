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
from .utils import (
    calculate_text_tokens,
    detect_local_models,
    estimate_cost,
    get_suggested_local_models,
    validate_api_key,
)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

console = Console()


def _estimate_processing_cost(pages: list[str], prompt: str, model: str, model_type: str = "openai") -> dict:
    """Estimate the cost of processing pages with the given model.
    
    Args:
        pages: List of page contents
        prompt: Processing prompt
        model: Model name
        model_type: Type of model ("openai" or "local")
        
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
    estimated_cost = estimate_cost(total_input_tokens, total_output_tokens_estimate, model, model_type)
    
    non_empty_pages = len([p for p in pages if p.strip()])
    
    return {
        "pages_to_process": non_empty_pages,
        "total_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens_estimate,
        "estimated_total_tokens": total_input_tokens + total_output_tokens_estimate,
        "estimated_cost_usd": estimated_cost,
        "avg_tokens_per_page": (total_input_tokens + total_output_tokens_estimate) // non_empty_pages if non_empty_pages > 0 else 0,
        "model_type": model_type,
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
    help="Model to use for processing (OpenAI model name or 'local' for local model)",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to local model file (.gguf format) - required when using --local",
)
@click.option(
    "--local",
    is_flag=True,
    help="Use local model instead of OpenAI API",
)
@click.option(
    "--n-gpu-layers",
    type=int,
    default=0,
    help="Number of layers to offload to GPU (for local models)",
)
@click.option(
    "--n-ctx",
    type=int,
    default=2048,
    help="Context size for local models",
)
@click.option(
    "--prompt-template",
    type=click.Choice(['standard', 'persian', 'custom'], case_sensitive=False),
    default="standard",
    help="Prompt template format: 'standard' (OpenAI-style), 'persian' (optimized for Persian translation), 'custom' (simple format)",
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
    help="Rate limit in requests per minute (ignored for local models)",
)
@click.option(
    "--concurrent",
    "-c",
    type=int,
    default=3,
    help="Number of concurrent requests",
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.1,
    help="Temperature for model generation (0.0 to 1.0, lower = more deterministic)",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum tokens to generate per page (default: 512 for local, None for OpenAI)",
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
    help="Estimate API costs before processing (always $0.00 for local models)",
)
def process(
    pdf_path: Path,
    output_dir: Path,
    model: str,
    model_path: Optional[Path],
    local: bool,
    n_gpu_layers: int,
    n_ctx: int,
    prompt_template: str,
    prompt: str,
    max_retries: int,
    rate_limit: int,
    concurrent: int,
    temperature: float,
    max_tokens: Optional[int],
    verbose: bool,
    start_page: Optional[int],
    end_page: Optional[int],
    dry_run: bool,
    estimate_cost: bool,
) -> None:
    """Process a PDF file by slicing it into pages and processing each with LLMs."""
    
    # Determine model type
    model_type = "local" if local else "openai"
    
    # Validate local model setup
    if local and not model_path:
        console.print("[red]Error: --model-path is required when using --local flag[/red]")
        console.print("[yellow]💡 Tip: Download a .gguf model from Hugging Face and specify its path[/yellow]")
        sys.exit(1)
    
    # Validate API key (only if not dry run and not local)
    if not dry_run and not local:
        try:
            api_key = validate_api_key(model_type)
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
        temperature=temperature,
        max_tokens=max_tokens or (512 if local else None),
        start_page=start_page,
        end_page=end_page,
        model_path=str(model_path) if model_path else None,
        model_type=model_type,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        prompt_template=prompt_template,
    )
    
    # Create output directory (only if not dry run)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Display basic info
    console.print(f"[green]PDF file:[/green] {pdf_path}")
    if not dry_run:
        console.print(f"[green]Output directory:[/green] {output_dir}")
    
    if local:
        console.print(f"[green]Model type:[/green] Local (llama-cpp)")
        console.print(f"[green]Model path:[/green] {model_path}")
        console.print(f"[green]GPU layers:[/green] {n_gpu_layers}")
        console.print(f"[green]Context size:[/green] {n_ctx}")
        console.print(f"[green]💰 Cost:[/green] $0.00 (Local model - FREE!)")
    else:
        console.print(f"[green]Model type:[/green] OpenAI API")
        console.print(f"[green]Model:[/green] {model}")
        console.print(f"[green]Rate limit:[/green] {rate_limit} requests/minute")
    
    console.print(f"[green]Concurrent requests:[/green] {concurrent}")
    
    if start_page is not None or end_page is not None:
        page_range_text = f"Pages: {start_page or 1}-{end_page or 'last'}"
        console.print(f"[green]Page range:[/green] {page_range_text}")
    else:
        console.print(f"[green]Page range:[/green] All pages")
    
    if dry_run:
        console.print(f"[yellow]Mode:[/yellow] Dry run (no processing will be done)")
    
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
            console.print("\n[blue]📊 Processing Estimation[/blue]")
            cost_info = _estimate_processing_cost(pages, prompt, model, model_type)
            
            console.print(f"[cyan]Pages to process:[/cyan] {cost_info['pages_to_process']}")
            console.print(f"[cyan]Estimated input tokens:[/cyan] {cost_info['total_input_tokens']:,}")
            console.print(f"[cyan]Estimated output tokens:[/cyan] {cost_info['estimated_output_tokens']:,}")
            console.print(f"[cyan]Estimated total tokens:[/cyan] {cost_info['estimated_total_tokens']:,}")
            console.print(f"[cyan]Average tokens per page:[/cyan] {cost_info['avg_tokens_per_page']:,}")
            
            if local:
                console.print(f"[green]💰 Estimated cost:[/green] $0.00 (Local model - FREE!)")
                console.print(f"[green]✅ No API charges for local models![/green]")
            else:
                console.print(f"[cyan]Estimated cost:[/cyan] ${cost_info['estimated_cost_usd']:.4f} USD")
                
                if cost_info['estimated_cost_usd'] > 1.0:
                    console.print(f"[yellow]⚠️  High cost estimate (${cost_info['estimated_cost_usd']:.2f}). Consider using a local model for free processing![/yellow]")
            
            if not dry_run and estimate_cost and not local:
                # Ask for confirmation if cost is significant for OpenAI models
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
    help="Model to use for cost estimation",
)
@click.option(
    "--local",
    is_flag=True,
    help="Estimate for local model (always $0.00)",
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
    local: bool,
    prompt: str,
    start_page: Optional[int],
    end_page: Optional[int],
) -> None:
    """Estimate the cost of processing a PDF without making any API calls."""
    
    model_type = "local" if local else "openai"
    
    # Create configuration for analysis
    config = Config(
        model=model,
        prompt=prompt,
        start_page=start_page,
        end_page=end_page,
        model_type=model_type,
        prompt_template="standard",  # Use default for estimation
    )
    
    console.print(f"[green]📄 PDF Processing Estimation[/green]")
    console.print(f"[green]File:[/green] {pdf_path}")
    
    if local:
        console.print(f"[green]Model type:[/green] Local (FREE)")
    else:
        console.print(f"[green]Model type:[/green] OpenAI API")
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
        console.print("\n[blue]💰 Processing Analysis[/blue]")
        cost_info = _estimate_processing_cost(pages, prompt, model, model_type)
        
        console.print(f"[cyan]Pages with content:[/cyan] {cost_info['pages_to_process']}")
        console.print(f"[cyan]Estimated input tokens:[/cyan] {cost_info['total_input_tokens']:,}")
        console.print(f"[cyan]Estimated output tokens:[/cyan] {cost_info['estimated_output_tokens']:,}")
        console.print(f"[cyan]Total estimated tokens:[/cyan] {cost_info['estimated_total_tokens']:,}")
        console.print(f"[cyan]Average tokens per page:[/cyan] {cost_info['avg_tokens_per_page']:,}")
        
        if local:
            console.print(f"[green]💰 Estimated cost:[/green] $0.00 (Local model - FREE!)")
            console.print("[green]✅ Local models run on your hardware at no API cost![/green]")
            console.print("[yellow]💡 Consider local models for:[/yellow]")
            console.print("   • Cost-free processing")
            console.print("   • Privacy (no data sent to external services)")
            console.print("   • Offline processing")
        else:
            console.print(f"[cyan]Estimated cost:[/cyan] ${cost_info['estimated_cost_usd']:.4f} USD")
            
            # Cost warnings and recommendations
            if cost_info['estimated_cost_usd'] > 5.0:
                console.print(f"[red]⚠️  Very high cost estimate (${cost_info['estimated_cost_usd']:.2f})[/red]")
                console.print("[yellow]💡 Consider:[/yellow]")
                console.print("   • Using a local model (FREE)")
                console.print("   • Processing fewer pages at a time")
                console.print("   • Using a cheaper model (e.g., gpt-3.5-turbo)")
                console.print("   • Splitting into smaller batches")
            elif cost_info['estimated_cost_usd'] > 1.0:
                console.print(f"[yellow]⚠️  High cost estimate (${cost_info['estimated_cost_usd']:.2f})[/yellow]")
                console.print("[yellow]💡 Consider using a local model for free processing or gpt-3.5-turbo for lower costs[/yellow>")
            elif cost_info['estimated_cost_usd'] < 0.01:
                console.print("[green]✅ Very low cost estimate - good to go![/green]")
            else:
                console.print("[green]✅ Reasonable cost estimate[/green]")
                
            # Token efficiency info
            if cost_info['avg_tokens_per_page'] > 3000:
                console.print("[yellow]💡 Pages have a lot of text. Consider shorter prompts to reduce costs.[/yellow]")
        
        console.print(f"\n[blue]To proceed with processing, run:[/blue]")
        if local:
            console.print(f"[dim]pdf-translator process {pdf_path} --local --model-path /path/to/model.gguf[/dim]")
        else:
            console.print(f"[dim]pdf-translator process {pdf_path} --model {model}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error analyzing PDF: {str(e)}[/red>")
        sys.exit(1)


@cli.command()
@click.option(
    "--search-paths",
    "-p",
    multiple=True,
    help="Additional paths to search for model files",
)
def models(search_paths: tuple[str, ...]) -> None:
    """List available local models and suggestions for download."""
    
    console.print("[green]🤖 Local Model Management[/green]\n")
    
    # Show suggested models
    console.print("[blue]📋 Suggested Models for Download:[/blue]")
    suggested = get_suggested_local_models()
    for model_name, description in suggested.items():
        console.print(f"  • [cyan]{model_name}[/cyan]")
        console.print(f"    {description}")
    
    console.print(f"\n[yellow]💡 Download models from Hugging Face in .gguf format[/yellow]")
    console.print(f"[dim]Example: wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf[/dim]")
    
    # Detect available models
    console.print(f"\n[blue]🔍 Scanning for Local Models:[/blue]")
    search_list = list(search_paths) if search_paths else None
    found_models = detect_local_models(search_list)
    
    if found_models:
        console.print(f"[green]Found {len(found_models)} model(s):[/green]")
        for model_path in found_models:
            model_file = Path(model_path)
            size_mb = model_file.stat().st_size / (1024 * 1024)
            console.print(f"  • [cyan]{model_path}[/cyan] ({size_mb:.1f} MB)")
        
        console.print(f"\n[blue]💡 To use a local model:[/blue]")
        console.print(f"[dim]pdf-translator process your_file.pdf --local --model-path /path/to/model.gguf[/dim]")
    else:
        console.print("[yellow]No local models found.[/yellow]")
        console.print("[yellow]Download a .gguf model and place it in one of these locations:[/yellow]")
        default_paths = [
            "~/models",
            "~/.cache/huggingface/transformers", 
            "~/.ollama/models",
            "/usr/local/share/models",
            "./models",
        ]
        for path in default_paths:
            console.print(f"  • {path}")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
