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
from .utils import validate_api_key

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="dilmaj")
def cli() -> None:
    """PDF Translator CLI - Slice PDFs and process with GPT models."""
    pass


def process_core(
    file_path: str,
    output_dir: str,
    model: str,
    prompt: str,
    max_retries: int,
    rate_limit: int,
    concurrent: int,
    temperature: float,
    max_tokens: Optional[int],
    verbose: bool,
    dry_run: bool,
    no_preprocess: bool,
    keep_headers_footers: bool,
    no_paragraph_chunking: bool,
    from_extracted_dir: Optional[str],
) -> None:
    """Core processing logic used by provider-specific commands."""

    # Convert string paths to Path objects
    file_path_obj = Path(file_path)
    output_dir_obj = Path(output_dir)
    extracted_dir_obj = Path(from_extracted_dir) if from_extracted_dir else None
    # Validate API key (only if not dry run)
    if not dry_run:
        try:
            validate_api_key("openai")
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
        max_tokens=max_tokens,
        preprocess_text=not no_preprocess,
        remove_headers_footers=not keep_headers_footers,
        chunk_paragraphs=not no_paragraph_chunking,
    )

    # Always ensure output/extracted directories exist (we save extracted text
    # even in dry-run to make the default flow "extract to files, then read")
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    default_extracted_dir = output_dir_obj / "extracted"

    # Display basic info
    console.print(f"[green]PDF file:[/green] {file_path_obj}")
    if not dry_run:
        console.print(f"[green]Output directory:[/green] {output_dir_obj}")

    console.print("[green]Model type:[/green] OpenAI API")
    console.print(f"[green]Model:[/green] {model}")
    console.print(f"[green]Rate limit:[/green] {rate_limit} requests/minute")

    console.print(f"[green]Concurrent requests:[/green] {concurrent}")

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
                console.print(
                    "[green]Extracting paragraphs from entire document[/green]"
                )

                # Ensure extracted dir exists
                extracted_dir_obj.mkdir(parents=True, exist_ok=True)

                # Extract and save paragraphs to files
                console.print(
                    "[blue]Extracting and saving paragraphs to "
                    f"{extracted_dir_obj}...[/blue]"
                )
                extracted_paragraphs = processor.extract_paragraphs(file_path_obj)

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

            # Display extracted paragraph info
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
                None if extracted_dir_obj is not None else file_path_obj,
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
@click.argument("file_path", type=click.Path(exists=True))
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
    help="OpenAI model to use for processing",
)
@click.option(
    "--prompt",
    "-p",
    default=(
        "Please translate this text to English and provide a "
        "clean, formatted version."
    ),
    help="Custom prompt for processing paragraphs",
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
    "--temperature",
    "-t",
    type=float,
    default=0.1,
    help=("Temperature for model generation (0.0 to 1.0, lower = more deterministic)"),
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum tokens to generate per paragraph (default: None)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
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
    file_path: str,
    output_dir: str,
    model: str,
    prompt: str,
    max_retries: int,
    rate_limit: int,
    concurrent: int,
    temperature: float,
    max_tokens: Optional[int],
    verbose: bool,
    dry_run: bool,
    no_preprocess: bool,
    keep_headers_footers: bool,
    no_paragraph_chunking: bool,
    from_extracted_dir: Optional[str],
) -> None:
    """Process a PDF file by extracting paragraphs and processing them with OpenAI."""
    process_core(
        file_path=file_path,
        output_dir=output_dir,
        model=model,
        prompt=prompt,
        max_retries=max_retries,
        rate_limit=rate_limit,
        concurrent=concurrent,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
        dry_run=dry_run,
        no_preprocess=no_preprocess,
        keep_headers_footers=keep_headers_footers,
        no_paragraph_chunking=no_paragraph_chunking,
        from_extracted_dir=from_extracted_dir,
    )


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
    # removed page args
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
        paragraphs = processor.extract_paragraphs(input_path_obj)

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
            ("Example: dilmaj process INPUT.pdf " "--output-dir ./output ...")
        )

    except Exception as e:
        console.print(f"[red]Error extracting text: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
