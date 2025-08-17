"""CLI extract and dry-run flows to improve coverage without network calls."""

import os
from pathlib import Path

from click.testing import CliRunner

from dilmaj.cli import cli


def _make_minimal_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)


def test_extract_on_blank_pdf_shows_error(tmp_path: Path):
    runner = CliRunner()
    blank = tmp_path / "b.pdf"
    _make_minimal_pdf(blank)

    outdir = tmp_path / "out"
    res = runner.invoke(cli, ["extract", str(blank), "--output-dir", str(outdir), "-y"])
    # extraction should fail gracefully and exit 1
    assert res.exit_code == 1


def test_process_dry_run_with_extracted_dir(tmp_path: Path):
    runner = CliRunner()

    # Prepare extracted dir with paragraphs
    extracted = tmp_path / "out" / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "paragraph_001.txt").write_text("Hello world", encoding="utf-8")
    (extracted / "paragraph_002.txt").write_text("\n\n", encoding="utf-8")
    (extracted / "paragraph_003.txt").write_text("Another one", encoding="utf-8")

    # In dry-run we don't need OPENAI_API_KEY
    # Provide an existing input file (Click requires exists=True)
    dummy_input = tmp_path / "in.pdf"
    dummy_input.write_text("not a real pdf", encoding="utf-8")

    res = runner.invoke(
        cli,
        [
            "process",
            str(dummy_input),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--from-extracted-dir",
            str(extracted),
        ],
    )
    assert res.exit_code == 0
    assert "Dry run" in res.output
