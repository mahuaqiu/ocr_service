"""
图像预处理模块。

提供自适应图像预处理功能，支持多种预设模式。
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

from ocr_service.config import ServiceConfig, get_config
from ocr_service.utils.image_utils import (
    resize_image_with_scale,
    denoise_image,
    enhance_contrast,
    sharpen_image,
    binarize_image,
    get_resolution_type,
    ResolutionType,
    get_image_size,
)


@dataclass
class PreprocessResult:
    """预处理结果。"""

    image: np.ndarray
    scale: float
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    steps_applied: List[str]


class PreprocessMode:
    """预处理模式常量。"""

    NONE = "none"
    AUTO = "auto"
    SCREENSHOT = "screenshot"
    MOBILE = "mobile"
    DOCUMENT = "document"
    LOW_QUALITY = "low_quality"


# 预设配置
PRESET_CONFIGS = {
    PreprocessMode.AUTO: {
        "auto_resize": True,
        "denoise": False,
        "contrast_enhance": False,
        "sharpen": False,
        "binarize": False,
    },
    PreprocessMode.SCREENSHOT: {
        "auto_resize": True,
        "denoise": False,
        "contrast_enhance": False,
        "sharpen": True,  # 桌面截图轻度锐化
        "binarize": False,
    },
    PreprocessMode.MOBILE: {
        "auto_resize": True,
        "denoise": False,
        "contrast_enhance": True,  # 移动端增强对比度
        "sharpen": False,
        "binarize": False,
    },
    PreprocessMode.DOCUMENT: {
        "auto_resize": True,
        "denoise": False,
        "contrast_enhance": True,
        "sharpen": False,
        "binarize": True,  # 文档二值化
    },
    PreprocessMode.LOW_QUALITY: {
        "auto_resize": True,
        "denoise": True,  # 低质量图片降噪
        "contrast_enhance": True,
        "sharpen": True,
        "binarize": False,
    },
}


class ImagePreprocessor:
    """图像预处理器。

    根据不同模式和分辨率类型自适应处理图像。
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        """初始化预处理器。

        Args:
            config: 服务配置，默认使用全局配置。
        """
        self.config = config or get_config()

    def preprocess(
        self,
        image: np.ndarray,
        mode: str = PreprocessMode.AUTO,
    ) -> PreprocessResult:
        """预处理图像。

        Args:
            image: OpenCV 图像数组。
            mode: 预处理模式。

        Returns:
            PreprocessResult: 预处理结果。
        """
        steps_applied = []
        original_size = get_image_size(image)
        processed_image = image.copy()
        scale = 1.0

        # 不预处理模式
        if mode == PreprocessMode.NONE or not self.config.preprocess_enabled:
            return PreprocessResult(
                image=processed_image,
                scale=scale,
                original_size=original_size,
                processed_size=original_size,
                steps_applied=[],
            )

        # 获取预设配置
        preset = PRESET_CONFIGS.get(mode, PRESET_CONFIGS[PreprocessMode.AUTO])

        # 分辨率检测
        resolution_type = get_resolution_type(image)

        # 自适应缩放
        if preset["auto_resize"] and self.config.auto_resize_enabled:
            processed_image, scale = self._auto_resize(processed_image, resolution_type)
            if scale < 1.0:
                steps_applied.append(f"resize({scale:.2f})")

        # 降噪
        if preset["denoise"]:
            processed_image = denoise_image(processed_image, strength="light")
            steps_applied.append("denoise")

        # 对比度增强
        if preset["contrast_enhance"]:
            processed_image = enhance_contrast(processed_image)
            steps_applied.append("contrast_enhance")

        # 锐化
        if preset["sharpen"]:
            processed_image = sharpen_image(processed_image)
            steps_applied.append("sharpen")

        # 二值化
        if preset["binarize"]:
            processed_image = binarize_image(processed_image)
            steps_applied.append("binarize")

        processed_size = get_image_size(processed_image)

        return PreprocessResult(
            image=processed_image,
            scale=scale,
            original_size=original_size,
            processed_size=processed_size,
            steps_applied=steps_applied,
        )

    def _auto_resize(
        self,
        image: np.ndarray,
        resolution_type: str,
    ) -> Tuple[np.ndarray, float]:
        """自适应缩放图像。

        Args:
            image: OpenCV 图像数组。
            resolution_type: 分辨率类型。

        Returns:
            Tuple[np.ndarray, float]: (缩放后的图像, 缩放比例)
        """
        h, w = image.shape[:2]
        max_width = self.config.max_image_width
        max_height = self.config.max_image_height

        # 根据分辨率类型决定是否缩放
        if resolution_type == ResolutionType.DESKTOP_2K:
            # 2K 图像需要缩放
            return resize_image_with_scale(image, max_width=max_width, max_height=max_height)
        elif resolution_type == ResolutionType.DESKTOP_1080P:
            # 1080P 图像一般不需要缩放
            if w > max_width or h > max_height:
                return resize_image_with_scale(image, max_width=max_width, max_height=max_height)
            return image, 1.0
        elif resolution_type == ResolutionType.MOBILE:
            # 移动端图像通常较小，不缩放
            return image, 1.0
        else:
            # 小图像不缩放
            return image, 1.0


# 全局预处理器实例
_preprocessor: Optional[ImagePreprocessor] = None


def get_image_preprocessor() -> ImagePreprocessor:
    """获取全局图像预处理器实例。"""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = ImagePreprocessor()
    return _preprocessor