"""CLI interface for PDF Translator."""

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
from .utils import detect_local_models, get_suggested_local_models, validate_api_key

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="pdf-translator")
def cli() -> None:
    """PDF Translator CLI - Slice PDFs and process with GPT models."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./output",
    help="Output directory for results",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o-mini",
    help=(
        "Model to use for processing (OpenAI model name or 'local' " "for local model)"
    ),
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    help=("Path to local model file (.gguf format) - required when using " "--local"),
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
    type=click.Choice(["standard", "persian", "custom"], case_sensitive=False),
    default="standard",
    help=(
        "Prompt template format: 'standard' (OpenAI-style), "
        "'persian' (optimized for Persian translation), "
        "'custom' (simple format)"
    ),
)
@click.option(
    "--prompt",
    "-p",
    default=(
        "Please translate this text to English and provide a "
        "clean, formatted version."
    ),
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
    help=(
        "Temperature for model generation (0.0 to 1.0, lower = more " "deterministic)"
    ),
)
@click.option(
    "--max-tokens",
    type=int,
    help=(
        "Maximum tokens to generate per page "
        "(default: 512 for local, None for OpenAI)"
    ),
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
    "--no-preprocess",
    is_flag=True,
    help="Disable text preprocessing (skip cleaning headers/footers, etc.)",
)
@click.option(
    "--keep-headers-footers",
    is_flag=True,
    help="Keep headers and footers during preprocessing",
)
@click.option(
    "--no-paragraph-chunking",
    is_flag=True,
    help="Disable paragraph chunking during preprocessing",
)
@click.option(
    "--from-extracted-dir",
    type=click.Path(exists=True, file_okay=False),
    help=(
        "Directory containing pre-extracted paragraph_*.txt files to process "
        "(skips in-PDF extraction)"
    ),
)
def process(
    pdf_path: str,
    output_dir: str,
    model: str,
    model_path: Optional[str],
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
    no_preprocess: bool,
    keep_headers_footers: bool,
    no_paragraph_chunking: bool,
    from_extracted_dir: Optional[str],
) -> None:
    """Process a PDF file by slicing it into pages and
    processing each with LLMs."""

    # Convert string paths to Path objects
    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(output_dir)
    model_path_obj = Path(model_path) if model_path else None
    extracted_dir_obj = Path(from_extracted_dir) if from_extracted_dir else None

    # Determine model type
    model_type = "local" if local else "openai"

    # Validate local model setup
    if local and not model_path:
        console.print(
            ("[red]Error: --model-path is required when using " "--local flag[/red]")
        )
        console.print(
            "[yellow]💡 Tip: Download a .gguf model from Hugging Face and "
            "specify its path[/yellow]"
        )
        sys.exit(1)

    # Validate API key (only if not dry run and not local)
    if not dry_run and not local:
        try:
            validate_api_key(model_type)
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
        model_path=str(model_path_obj) if model_path_obj else None,
        model_type=model_type,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        prompt_template=prompt_template,
        preprocess_text=not no_preprocess,
        remove_headers_footers=not keep_headers_footers,
        chunk_paragraphs=not no_paragraph_chunking,
    )

    # Always ensure output/extracted directories exist (we save extracted text
    # even in dry-run to make the default flow "extract to files, then read")
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    default_extracted_dir = output_dir_obj / "extracted"

    # Display basic info
    console.print(f"[green]PDF file:[/green] {pdf_path_obj}")
    if not dry_run:
        console.print(f"[green]Output directory:[/green] {output_dir_obj}")

    if local:
        console.print("[green]Model type:[/green] Local (llama-cpp)")
        console.print(f"[green]Model path:[/green] {model_path_obj}")
        console.print(f"[green]GPU layers:[/green] {n_gpu_layers}")
        console.print(f"[green]Context size:[/green] {n_ctx}")
        console.print("[green]💰 Cost:[/green] $0.00 (Local model - FREE!)")
    else:
        console.print("[green]Model type:[/green] OpenAI API")
        console.print(f"[green]Model:[/green] {model}")
        console.print(f"[green]Rate limit:[/green] {rate_limit} requests/minute")

    console.print(f"[green]Concurrent requests:[/green] {concurrent}")

    if start_page is not None or end_page is not None:
        page_range_text = f"Pages: {start_page or 1}-{end_page or 'last'}"
        console.print(
            (
                "[green]Page range (source pages to extract paragraphs "
                "from):[/green] "
                f"{page_range_text}"
            )
        )
    else:
        console.print("[green]Page range:[/green] All pages")

    if dry_run:
        console.print("[yellow]Mode:[/yellow] Dry run (no LLM calls)")

    # Display text preprocessing settings
    if config.preprocess_text:
        preprocessing_status = []
        if config.remove_headers_footers:
            preprocessing_status.append("remove headers/footers")
        if config.chunk_paragraphs:
            preprocessing_status.append("chunk paragraphs")

        status_text = (
            ", ".join(preprocessing_status)
            if preprocessing_status
            else "basic cleaning"
        )
        console.print(f"[green]Text preprocessing:[/green] Enabled ({status_text})")
    else:
        console.print("[yellow]Text preprocessing:[/yellow] Disabled")

    # Initialize processor
    processor = PDFProcessor(config, init_llm=not dry_run)

    try:
        # Determine extracted directory (given or default under output)
        if extracted_dir_obj is None:
            extracted_dir_obj = default_extracted_dir

        # If user didn't provide a pre-extracted directory,
        # perform extraction now
        if from_extracted_dir is None:
            try:
                # Get total page count for validation and display
                total_pages = processor.get_pdf_page_count(pdf_path_obj)

                # Validate page range
                try:
                    (
                        actual_start,
                        actual_end,
                    ) = config.get_page_range(total_pages)
                    page_count_to_process = actual_end - actual_start + 1
                except ValueError as e:
                    console.print(f"[red]Error: {str(e)}[/red]")
                    sys.exit(1)

                console.print(f"[green]PDF has {total_pages} pages total[/green]")
                console.print(
                    (
                        "[green]Will extract paragraphs from pages "
                        f"{actual_start}-{actual_end} "
                        f"({page_count_to_process} pages in range)"
                        "[/green]"
                    )
                )

                # Ensure extracted dir exists
                extracted_dir_obj.mkdir(parents=True, exist_ok=True)

                # Extract and save paragraphs to files
                console.print(
                    "[blue]Extracting and saving paragraphs to "
                    f"{extracted_dir_obj}...[/blue]"
                )
                extracted_paragraphs = processor.extract_pages(pdf_path_obj)

                # Write paragraph files
                written = 0
                for idx, content in enumerate(extracted_paragraphs, start=1):
                    if not content.strip():
                        continue
                    out_file = extracted_dir_obj / f"paragraph_{idx:03d}.txt"
                    out_file.write_text(content, encoding="utf-8")
                    written += 1

                console.print(
                    ("[green]Saved {n} paragraph files to " "{path}[/green]").format(
                        n=written, path=extracted_dir_obj
                    )
                )
            except Exception as e:
                console.print(f"[red]Error during extraction: {str(e)}[/red]")
                if verbose:
                    console.print_exception()
                sys.exit(1)

        # Load paragraphs from extracted directory (either user-provided
        # or newly created)
        console.print("[blue]Loading paragraphs from extracted directory...[/blue]")
        txt_files = sorted(extracted_dir_obj.glob("paragraph_*.txt"))
        pages = []
        for fp in txt_files:
            try:
                pages.append(fp.read_text(encoding="utf-8"))
            except Exception:
                pages.append("")
        non_empty_pages = [p for p in pages if p.strip()]
        console.print(
            (
                f"[green]Found {len(non_empty_pages)} paragraphs in "
                f"{extracted_dir_obj}[/green]"
            )
        )

        # Dry run: show basic info and exit without processing
        if dry_run:
            console.print("\n[blue]🔍 Dry run summary[/blue]")
            console.print(
                (
                    f"[cyan]Non-empty paragraphs to process:[/cyan] "
                    f"{len(non_empty_pages)}"
                )
            )
            console.print("[yellow]Dry run complete - no API calls were made[/yellow]")
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
            progress.add_task(
                "Paragraphs already extracted",
                total=len(pages),
                completed=len(pages),
            )

            # Display extracted page info
            console.print(
                (
                    f"[green]Processing {len(non_empty_pages)} non-empty "
                    "paragraphs from the extracted range[/green]"
                )
            )

            # Process pages
            process_task = progress.add_task(
                "Processing paragraphs with GPT...", total=len(non_empty_pages)
            )

            results = processor.process_pages_async(
                pages,
                output_dir_obj,
                None if extracted_dir_obj is not None else pdf_path_obj,
                progress_callback=lambda completed: progress.update(
                    process_task, completed=completed
                ),
            )

            console.print(
                f"[green]Successfully processed {len(results)} " "paragraphs[/green]"
            )
            console.print(f"[green]Results saved to:[/green] {output_dir_obj}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error processing PDF: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./output",
    help="Output directory where extracted chunks will be saved",
)
@click.option(
    "--start-page",
    "-s",
    type=int,
    help="First page to extract from (1-based). Default: 1",
)
@click.option(
    "--end-page",
    "-e",
    type=int,
    help="Last page to extract from (1-based). Default: last page",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--no-preprocess",
    is_flag=True,
    help="Disable text preprocessing (skip cleaning headers/footers, etc.)",
)
@click.option(
    "--keep-headers-footers",
    is_flag=True,
    help="Keep headers and footers during preprocessing",
)
@click.option(
    "--no-paragraph-chunking",
    is_flag=True,
    help="Disable paragraph chunking during preprocessing",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help=(
        "Auto-confirm cleaning if output directory already contains " "extracted files"
    ),
)
def extract(
    input_path: str,
    output_dir: str,
    start_page: Optional[int],
    end_page: Optional[int],
    verbose: bool,
    no_preprocess: bool,
    keep_headers_footers: bool,
    no_paragraph_chunking: bool,
    yes: bool,
) -> None:
    """Extract and chunk text from a document into files.

    If the output directory already contains extracted text, you'll be asked
    whether to clean it before proceeding (use -y to auto-confirm).
    """

    input_path_obj = Path(input_path)
    base_output_dir = Path(output_dir)
    extracted_dir = base_output_dir / "extracted"

    # Prepare config for extraction-only flow
    config = Config(
        verbose=verbose,
        start_page=start_page,
        end_page=end_page,
        preprocess_text=not no_preprocess,
        remove_headers_footers=not keep_headers_footers,
        chunk_paragraphs=not no_paragraph_chunking,
    )

    # Ensure parent dir exists
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Detect existing extracted content
    existing_files = list(extracted_dir.glob("paragraph_*.txt")) + list(
        extracted_dir.glob("*.json")
    )
    if existing_files:
        if yes or click.confirm(
            (
                f"[bold yellow]Extracted directory {extracted_dir} already "
                "contains files.[/bold yellow]\nDo you want to clean "
                "extracted text before continuing?"
            ),
            default=False,
        ):
            # Clean only extracted artifacts
            for p in extracted_dir.glob("**/*"):
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        # Only remove empty subdirs
                        try:
                            p.rmdir()
                        except OSError:
                            pass
                except Exception:
                    # Best-effort cleanup; continue
                    pass
            console.print(
                ("[green]Cleaned extracted artifacts in " f"{extracted_dir}[/green]")
            )
        else:
            console.print("[yellow]Keeping existing extracted files[/yellow]")

    # Initialize processor without LLM for extraction only
    processor = PDFProcessor(config, init_llm=False)

    try:
        # Extract paragraphs
        console.print("[blue]Extracting and chunking text...[/blue]")
        paragraphs = processor.extract_pages(input_path_obj)

        # Save paragraphs to files
        for idx, content in enumerate(paragraphs, start=1):
            if not content.strip():
                continue
            out_file = extracted_dir / f"paragraph_{idx:03d}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(content)

        # Save summary
        summary = {
            "source_file": str(input_path_obj),
            "output_directory": str(extracted_dir),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "config": config.to_dict(),
        }
        with open(
            extracted_dir / "extraction_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            import json as _json

            _json.dump(summary, f, indent=2, ensure_ascii=False)

        console.print(
            (
                f"[green]Extraction complete:[/green] Saved "
                f"{summary['paragraph_count']} paragraphs to "
                f"{extracted_dir}"
            )
        )

        # Ask user to proceed
        console.print(
            (
                "[cyan]You can now proceed to processing using the "
                "'process' command.[/cyan]"
            )
        )
        console.print(
            ("Example: pdf-translator process INPUT.pdf " "--output-dir ./output ...")
        )

    except Exception as e:
        console.print(f"[red]Error extracting text: {str(e)}[/red]")
        if verbose:
            console.print_exception()
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

    console.print(
        ("\n[yellow]💡 Download models from Hugging Face in .gguf " "format[/yellow]")
    )
    console.print(
        (
            "[dim]Example: wget https://huggingface.co/TheBloke/"
            "Llama-2-7B-Chat-GGUF/resolve/main/"
            "llama-2-7b-chat.Q4_K_M.gguf[/dim]"
        )
    )

    # Detect available models
    console.print("\n[blue]🔍 Scanning for Local Models:[/blue]")
    search_list = list(search_paths) if search_paths else None
    found_models = detect_local_models(search_list)

    if found_models:
        console.print(f"[green]Found {len(found_models)} model(s):[/green]")
        for model_path in found_models:
            model_file = Path(model_path)
            size_mb = model_file.stat().st_size / (1024 * 1024)
            console.print(f"  • [cyan]{model_path}[/cyan] ({size_mb:.1f} MB)")

        console.print("\n[blue]💡 To use a local model:[/blue]")
        console.print(
            "[dim]pdf-translator process your_file.pdf --local "
            "--model-path /path/to/model.gguf[/dim]"
        )
    else:
        console.print("[yellow]No local models found.[/yellow]")
        console.print(
            "[yellow]Download a .gguf model and place it in one of "
            "these locations:[/yellow]"
        )
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
