"""
图像缩放坐标还原测试。

复现并验证修复：当图像宽高都超过限制时，resize_image_with_scale
必须返回统一的、可用于还原 x/y 坐标的单一缩放比例（等比缩放）。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.utils.image_utils import resize_image_with_scale


@pytest.mark.parametrize(
    "w,h,max_width,max_height",
    [
        (2560, 1600, 1920, 1080),  # 宽高都超限，且不是 16:9，触发二次缩放 bug
        (1800, 1300, 1920, 1080),  # 宽未超，高超限
        (3840, 2160, 1920, 1080),  # 标准 16:9，恰好不触发 bug（用于回归对比）
        (3000, 1000, 1920, 1080),  # 超宽，高未超
    ],
)
def test_resize_keeps_uniform_scale_for_both_axes(w, h, max_width, max_height):
    """缩放后，x 轴和 y 轴的实际缩放比例必须一致，且等于返回的 scale。"""
    image = np.zeros((h, w, 3), dtype=np.uint8)

    resized, scale = resize_image_with_scale(
        image, max_width=max_width, max_height=max_height
    )

    resized_h, resized_w = resized.shape[:2]
    actual_scale_x = resized_w / w
    actual_scale_y = resized_h / h

    # x 轴和 y 轴的实际缩放比例必须一致（等比缩放，不能变形）
    assert actual_scale_x == pytest.approx(actual_scale_y, abs=1e-3), (
        f"x/y 轴缩放比例不一致: x={actual_scale_x}, y={actual_scale_y}"
    )

    # 返回的 scale 必须能同时用于还原 x 和 y 坐标
    assert actual_scale_x == pytest.approx(scale, abs=1e-3)
    assert actual_scale_y == pytest.approx(scale, abs=1e-3)

    # 缩放后的尺寸不能超过限制
    assert resized_w <= max_width
    assert resized_h <= max_height
