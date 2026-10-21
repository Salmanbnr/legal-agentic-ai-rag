"""Image preprocessing and bounded OCR fallback for scanned pages.

The original rendered page is always copied before any transformation. When
OCR confidence is low, exactly one preprocessed variant is attempted and the
higher-confidence result wins. No retry loop is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

from app.ingestion.ocr import (
    OcrError,
    OcrResult,
    configure_tesseract,
    ocr_image,
)

_OSD_ROTATIONS = {0, 90, 180, 270}
_MAX_SKEW_SAMPLE_DIMENSION = 900


@dataclass(frozen=True, slots=True)
class PreprocessedPage:
    """A preprocessed image copy and the transformations applied to it."""

    image: np.ndarray
    transformations: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class OcrPageResult:
    """Winner of the original-vs-preprocessed OCR comparison."""

    result: OcrResult
    variant: Literal["original", "preprocessed"]
    transformations: list[dict[str, Any]]
    original_confidence: float
    preprocessed_confidence: float | None


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return a writable RGB uint8 copy of a PIL image or numpy array."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), copy=True, dtype=np.uint8)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            converted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            converted = image.copy()
        elif image.ndim == 3 and image.shape[2] == 4:
            converted = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            raise OcrError("OCR_FAILED")
        if converted.dtype != np.uint8:
            converted = np.clip(converted, 0, 255).astype(np.uint8)
        return np.array(converted, copy=True, dtype=np.uint8)
    raise OcrError("OCR_FAILED")


def _rotate_quarter_turns(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by 90, 180, or 270 degrees."""
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def _rotate_expanded(
    image: np.ndarray,
    angle_degrees: float,
    *,
    border_value: tuple[int, int, int] | int = (255, 255, 255),
) -> np.ndarray:
    """Rotate without cropping and fill exposed corners."""
    if abs(angle_degrees) < 0.01:
        return image.copy()
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = max(1, int(height * sine + width * cosine))
    new_height = max(1, int(height * cosine + width * sine))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _rotate_for_projection(binary: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a same-size binary sample for projection scoring."""
    height, width = binary.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), angle_degrees, 1.0
    )
    return cv2.warpAffine(
        binary,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def estimate_skew_angle(image: np.ndarray, *, max_degrees: float = 15.0) -> float:
    """Estimate the correction angle using horizontal projection variance.

    The image is downsampled before trying angles, so this is suitable for
    300-DPI scans. A positive result is the angle to apply with OpenCV's
    counter-clockwise rotation convention.
    """
    if max_degrees <= 0:
        return 0.0
    rgb = _as_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, _MAX_SKEW_SAMPLE_DIMENSION / float(max(height, width)))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    if cv2.countNonZero(binary) < 20:
        return 0.0

    best_angle = 0.0
    best_score = -1.0
    steps = int(round((max_degrees * 2.0) / 0.5))
    for index in range(steps + 1):
        angle = -max_degrees + index * 0.5
        rotated = _rotate_for_projection(binary, angle)
        projection = np.sum(rotated, axis=1, dtype=np.float64)
        score = float(np.var(projection))
        if score > best_score:
            best_score = score
            best_angle = angle
    return round(best_angle, 2)


def _osd_rotation(
    image: np.ndarray, *, tesseract_cmd: str | None
) -> tuple[int, dict[str, Any]]:
    """Read Tesseract orientation, falling back to no rotation."""
    configure_tesseract(tesseract_cmd)
    try:
        data = pytesseract.image_to_osd(
            image, config="--psm 0", output_type=Output.DICT
        )
        rotation = int(data.get("rotate", 0))
        if rotation not in _OSD_ROTATIONS:
            raise ValueError("unsupported rotation")
        return rotation, {
            "stage": "preprocess",
            "step": "orientation",
            "applied": rotation != 0,
            "rotate_degrees": rotation,
            "method": "tesseract_osd",
        }
    except Exception:
        # Orientation detection is best-effort; OCR still runs on the original.
        return 0, {
            "stage": "preprocess",
            "step": "orientation",
            "applied": False,
            "reason": "osd_unavailable",
        }


def preprocess_page(
    image: Image.Image | np.ndarray,
    *,
    tesseract_cmd: str | None = None,
    max_deskew_degrees: float = 15.0,
) -> PreprocessedPage:
    """Apply orientation, deskew, denoise, CLAHE, and adaptive threshold.

    The returned array is a new copy. The caller's image is never modified.
    """
    current = _as_rgb_array(image)
    transformations: list[dict[str, Any]] = []

    rotation, orientation_log = _osd_rotation(
        current, tesseract_cmd=tesseract_cmd
    )
    transformations.append(orientation_log)
    if rotation:
        current = _rotate_quarter_turns(current, rotation)

    skew_angle = estimate_skew_angle(
        current, max_degrees=max_deskew_degrees
    )
    applied_skew = bool(
        abs(skew_angle) >= 0.5 and abs(skew_angle) <= max_deskew_degrees
    )
    if applied_skew:
        current = _rotate_expanded(current, skew_angle)
    transformations.append(
        {
            "stage": "preprocess",
            "step": "deskew",
            "applied": applied_skew,
            "angle_degrees": skew_angle,
            "max_degrees": max_deskew_degrees,
            "method": "projection_variance",
        }
    )

    gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    transformations.append(
        {
            "stage": "preprocess",
            "step": "grayscale",
            "applied": True,
            "method": "rgb_to_gray",
        }
    )

    denoised = cv2.fastNlMeansDenoising(
        gray, None, h=7, templateWindowSize=7, searchWindowSize=21
    )
    transformations.append(
        {
            "stage": "preprocess",
            "step": "denoise",
            "applied": True,
            "method": "fast_nl_means",
            "strength": 7,
        }
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    transformations.append(
        {
            "stage": "preprocess",
            "step": "contrast",
            "applied": True,
            "method": "clahe",
            "clip_limit": 2.0,
            "tile_grid": [8, 8],
        }
    )

    thresholded = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    transformations.append(
        {
            "stage": "preprocess",
            "step": "threshold",
            "applied": True,
            "method": "adaptive_gaussian",
            "block_size": 31,
            "constant": 10,
        }
    )

    output = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
    return PreprocessedPage(image=output, transformations=transformations)


def ocr_page_with_fallback(
    image: Image.Image | np.ndarray,
    *,
    language: str = "eng",
    confidence_threshold: float = 70.0,
    tesseract_cmd: str | None = None,
    enable_preprocess: bool = True,
    max_deskew_degrees: float = 15.0,
) -> OcrPageResult:
    """OCR the original image and at most one preprocessed variant.

    The original result is kept when the threshold is met, when preprocessing
    fails, or when the preprocessed variant does not improve confidence.
    """
    if not 0.0 <= confidence_threshold <= 100.0:
        raise ValueError("confidence_threshold must be between 0 and 100")

    rgb = _as_rgb_array(image)
    original = ocr_image(
        rgb, language=language, tesseract_cmd=tesseract_cmd
    )
    if original.mean_confidence >= confidence_threshold or not enable_preprocess:
        return OcrPageResult(
            result=original,
            variant="original",
            transformations=[],
            original_confidence=original.mean_confidence,
            preprocessed_confidence=None,
        )

    try:
        preprocessed = preprocess_page(
            rgb,
            tesseract_cmd=tesseract_cmd,
            max_deskew_degrees=max_deskew_degrees,
        )
    except Exception:
        return OcrPageResult(
            result=original,
            variant="original",
            transformations=[
                {
                    "stage": "preprocess",
                    "step": "preprocess",
                    "applied": False,
                    "reason": "preprocess_failed",
                }
            ],
            original_confidence=original.mean_confidence,
            preprocessed_confidence=None,
        )

    try:
        processed = ocr_image(
            preprocessed.image,
            language=language,
            tesseract_cmd=tesseract_cmd,
        )
    except OcrError as exc:
        return OcrPageResult(
            result=original,
            variant="original",
            transformations=preprocessed.transformations
            + [
                {
                    "stage": "preprocess",
                    "step": "ocr_preprocessed",
                    "applied": False,
                    "reason": exc.error_code,
                }
            ],
            original_confidence=original.mean_confidence,
            preprocessed_confidence=None,
        )

    winner: Literal["original", "preprocessed"] = (
        "preprocessed"
        if processed.mean_confidence > original.mean_confidence
        else "original"
    )
    return OcrPageResult(
        result=processed if winner == "preprocessed" else original,
        variant=winner,
        transformations=preprocessed.transformations,
        original_confidence=original.mean_confidence,
        preprocessed_confidence=processed.mean_confidence,
    )
