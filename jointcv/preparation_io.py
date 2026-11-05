from pathlib import Path

import cv2
import numpy as np
def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read images reliably on Windows, including paths with Chinese characters."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))
