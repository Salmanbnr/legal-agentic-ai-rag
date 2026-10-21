"""Generate small PDF fixtures for upload and ingestion tests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

TEST_USER_PASSWORD = "test-user-password"
TEST_OWNER_PASSWORD = "test-owner-password"


@dataclass(frozen=True, slots=True)
class GeneratedPdfs:
    """Paths to the three generated PDF fixtures."""

    clean_native: Path
    image_only: Path
    encrypted: Path


def _save_clean_native(path: Path) -> None:
    """Write a one-page PDF with a realistic native text layer."""
    lines = [
        "IN THE MATTER OF A TEST LEGAL PROCEEDING",
        "STATEMENT OF FACTS",
        "",
        "1. This document is a locally generated native-text PDF fixture.",
        "2. It exists only to exercise upload validation and ingestion tests.",
        "3. The text layer is intentionally long enough to clear the native",
        "   text-density threshold without requiring OCR.",
        "4. No real person, claim, amount, or court is represented here.",
        "5. The fixture contains ordinary paragraphs and readable text.",
        "6. It should be extracted as a native page by PyMuPDF.",
        "",
        "Generated for the legal evidence workspace test suite.",
    ]
    document = pymupdf.open()
    try:
        page = document.new_page()
        for index, line in enumerate(lines):
            if line:
                page.insert_text(
                    (72, 72 + index * 18),
                    line,
                    fontsize=12,
                )
        document.save(path)
    finally:
        document.close()


def _save_image_only(path: Path) -> None:
    """Write a one-page PDF containing only a raster image."""
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=42)
    draw.rectangle((60, 60, 1140, 1540), outline="black", width=4)
    lines = [
        "IN THE MATTER OF A TEST LEGAL PROCEEDING",
        "SCANNED STATEMENT OF FACTS",
        "",
        "1. This page contains only a raster image.",
        "2. OCR must recover this text from the scan.",
        "3. The original page remains unchanged.",
        "4. No real person, date, or amount is represented.",
    ]
    for index, line in enumerate(lines):
        if line:
            draw.text((100, 120 + index * 120), line, fill="black", font=font)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    document = pymupdf.open()
    try:
        page = document.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=buffer.getvalue())
        document.save(path)
    finally:
        document.close()


def _save_encrypted(path: Path) -> None:
    """Write a one-page PDF that requires a user password."""
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "Encrypted PDF fixture.", fontsize=12)
        document.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw=TEST_OWNER_PASSWORD,
            user_pw=TEST_USER_PASSWORD,
        )
    finally:
        document.close()


def generate_test_pdfs(output_dir: Path | str) -> GeneratedPdfs:
    """Generate clean, image-only, and encrypted PDF fixtures."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = GeneratedPdfs(
        clean_native=directory / "clean_native.pdf",
        image_only=directory / "image_only.pdf",
        encrypted=directory / "encrypted.pdf",
    )
    _save_clean_native(paths.clean_native)
    _save_image_only(paths.image_only)
    _save_encrypted(paths.encrypted)
    return paths


def main() -> None:
    """Generate fixtures into the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures",
    )
    args = parser.parse_args()
    generated = generate_test_pdfs(args.output_dir)
    print(generated.clean_native)
    print(generated.image_only)
    print(generated.encrypted)


if __name__ == "__main__":
    main()
