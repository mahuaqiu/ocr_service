"""
图像处理工具函数。
"""

import base64
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def decode_image(image_data: bytes | str) -> np.ndarray:
    """解码图像数据为 OpenCV 格式。

    Args:
        image_data: 图像字节数据或 Base64 编码字符串。

    Returns:
        np.ndarray: OpenCV 图像数组 (BGR 格式)。
    """
    if isinstance(image_data, str):
        # Base64 解码
        image_data = base64.b64decode(image_data)

    # 转换为 numpy 数组
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image")

    return image


def encode_image(image: np.ndarray, format: str = ".png") -> str:
    """将 OpenCV 图像编码为 Base64 字符串。

    Args:
        image: OpenCV 图像数组。
        format: 图像格式，如 '.png', '.jpg'。

    Returns:
        str: Base64 编码的图像字符串。
    """
    success, buffer = cv2.imencode(format, image)
    if not success:
        raise ValueError("Failed to encode image")

    return base64.b64encode(buffer).decode("utf-8")


def resize_image(
    image: np.ndarray,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """调整图像大小。

    Args:
        image: OpenCV 图像数组。
        max_width: 最大宽度。
        max_height: 最大高度。
        scale: 缩放比例。

    Returns:
        np.ndarray: 调整后的图像。
    """
    if scale is not None:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        return cv2.resize(image, (new_width, new_height))

    if max_width is None and max_height is None:
        return image

    h, w = image.shape[:2]

    if max_width is not None and w > max_width:
        scale = max_width / w
        w = max_width
        h = int(h * scale)

    if max_height is not None and h > max_height:
        scale = max_height / h
        h = max_height
        w = int(w * scale)

    return cv2.resize(image, (w, h))


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """将图像转换为灰度图。

    Args:
        image: OpenCV 图像数组。

    Returns:
        np.ndarray: 灰度图像。
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def get_image_size(image: np.ndarray) -> Tuple[int, int]:
    """获取图像尺寸。

    Args:
        image: OpenCV 图像数组。

    Returns:
        Tuple[int, int]: (width, height)
    """
    h, w = image.shape[:2]
    return w, h


def resize_image_with_scale(
    image: np.ndarray,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """调整图像大小并返回缩放比例。

    Args:
        image: OpenCV 图像数组。
        max_width: 最大宽度。
        max_height: 最大高度。
        scale: 缩放比例。

    Returns:
        Tuple[np.ndarray, float]: (调整后的图像, 缩放比例)
    """
    if scale is not None:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        return cv2.resize(image, (new_width, new_height)), scale

    if max_width is None and max_height is None:
        return image, 1.0

    h, w = image.shape[:2]
    actual_scale = 1.0

    if max_width is not None and w > max_width:
        actual_scale = max_width / w
        w = max_width
        h = int(h * actual_scale)

    if max_height is not None and h > max_height:
        actual_scale = max_height / (h / actual_scale) if actual_scale < 1.0 else max_height / image.shape[0]
        h = max_height
        w = int(w * actual_scale) if actual_scale < 1.0 else w

    if actual_scale < 1.0:
        return cv2.resize(image, (w, h)), actual_scale
    return image, 1.0


def denoise_image(image: np.ndarray, strength: str = "light") -> np.ndarray:
    """降噪处理。

    Args:
        image: OpenCV 图像数组。
        strength: 降噪强度，可选 "light", "medium", "strong"。

    Returns:
        np.ndarray: 降噪后的图像。
    """
    strength_map = {
        "light": 3,
        "medium": 7,
        "strong": 10,
    }
    h = strength_map.get(strength, 3)
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """使用 CLAHE 增强对比度。

    Args:
        image: OpenCV 图像数组。

    Returns:
        np.ndarray: 增强后的图像。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """锐化图像。

    Args:
        image: OpenCV 图像数组。

    Returns:
        np.ndarray: 锐化后的图像。
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, kernel)


def binarize_image(image: np.ndarray) -> np.ndarray:
    """自适应二值化。

    Args:
        image: OpenCV 图像数组。

    Returns:
        np.ndarray: 二值化后的图像（BGR格式）。
    """
    gray = convert_to_grayscale(image)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


class ResolutionType:
    """分辨率类型常量。"""
    DESKTOP_2K = "desktop_2k"
    DESKTOP_1080P = "desktop_1080p"
    MOBILE = "mobile"
    SMALL = "small"


def get_resolution_type(image: np.ndarray) -> str:
    """检测图像分辨率类型。

    Args:
        image: OpenCV 图像数组。

    Returns:
        str: 分辨率类型。
    """
    h, w = image.shape[:2]

    if w > 2000 or h > 1200:
        return ResolutionType.DESKTOP_2K
    elif w > 1200 or h > 800:
        return ResolutionType.DESKTOP_1080P
    elif w < h and w <= 1200:
        return ResolutionType.MOBILE
    else:
        return ResolutionType.SMALL