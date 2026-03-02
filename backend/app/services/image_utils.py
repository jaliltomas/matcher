import base64
import io
from typing import Optional

import requests
from PIL import Image


def _decode_base64_image(raw: str) -> Optional[Image.Image]:
    try:
        cleaned = raw.split(",", maxsplit=1)[1] if raw.startswith("data:image") else raw
        binary = base64.b64decode(cleaned)
        return Image.open(io.BytesIO(binary)).convert("RGB")
    except Exception:
        return None


def load_image(image_value: str | None, timeout: int = 10) -> Optional[Image.Image]:
    if not image_value:
        return None

    if image_value.startswith("http://") or image_value.startswith("https://"):
        try:
            response = requests.get(image_value, timeout=timeout)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            return None

    return _decode_base64_image(image_value)
