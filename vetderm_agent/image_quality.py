from __future__ import annotations

import io

from PIL import Image


def basic_image_quality_warning(image_bytes: bytes, *, min_side: int = 224) -> str | None:
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size
    if min(width, height) < min_side:
        return f"image_short_side_below_{min_side}px"
    ratio = max(width, height) / max(1, min(width, height))
    if ratio > 3.0:
        return "extreme_aspect_ratio"
    return None
