"""
图像匹配颜色校验测试。

验证方案3：灰度粗匹配 + 颜色二次校验。
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# 确保可导入项目包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.config import ServiceConfig
from ocr_service.core.image_matcher import ImageMatcher


def _encode_bgr(image: np.ndarray) -> str:
    """将 BGR 图像编码为 Base64 字符串。"""
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _make_panel(with_red_line: bool) -> np.ndarray:
    """构造带文字的面板，可选红色/黑色横线。"""
    img = np.ones((60, 140, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (5, 5), (134, 54), (180, 180, 180), 1)
    line_color = (0, 0, 255) if with_red_line else (0, 0, 0)  # BGR
    cv2.line(img, (15, 25), (120, 25), line_color, 2)
    cv2.putText(img, "ABC", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return img


@pytest.fixture
def matcher() -> ImageMatcher:
    """使用固定阈值的匹配器，避免受全局配置影响。"""
    config = ServiceConfig(default_match_threshold=0.8, default_match_method="template")
    return ImageMatcher(config=config)


def test_red_line_template_should_not_match_black_line_source(matcher: ImageMatcher):
    """小图有红线、大图同结构但无红线时，不应匹配成功。"""
    template = _make_panel(with_red_line=True)

    source = np.ones((220, 320, 3), dtype=np.uint8) * 245
    # 大图粘贴“无红线”版本到已知位置
    panel = _make_panel(with_red_line=False)
    source[80:140, 90:230] = panel

    result = matcher.match_template(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.8,
    )

    assert result.status == "success"
    assert result.matches == [], (
        f"颜色不同应被过滤，却得到匹配: "
        f"{[m.to_dict() for m in result.matches]}"
    )


def test_red_line_template_should_match_red_line_source(matcher: ImageMatcher):
    """小图和大图都有红线时，应匹配成功。"""
    template = _make_panel(with_red_line=True)

    source = np.ones((220, 320, 3), dtype=np.uint8) * 245
    panel = _make_panel(with_red_line=True)
    source[80:140, 90:230] = panel

    result = matcher.match_template(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.8,
    )

    assert result.status == "success"
    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.confidence >= 0.8
    # 粘贴位置 (90, 80)
    assert abs(match.bbox.x - 90) <= 2
    assert abs(match.bbox.y - 80) <= 2


def test_match_all_also_filters_color_mismatch(matcher: ImageMatcher):
    """多目标匹配同样要做颜色二次校验。"""
    template = _make_panel(with_red_line=True)

    source = np.ones((300, 400, 3), dtype=np.uint8) * 245
    # 正确目标：有红线
    source[40:100, 40:180] = _make_panel(with_red_line=True)
    # 干扰目标：结构相同但无红线
    source[160:220, 200:340] = _make_panel(with_red_line=False)

    result = matcher.match_all(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.8,
    )

    assert result.status == "success"
    assert len(result.matches) == 1
    match = result.matches[0]
    assert abs(match.bbox.x - 40) <= 2
    assert abs(match.bbox.y - 40) <= 2


def test_default_match_threshold_is_0_9():
    """默认匹配置信度应为 0.9。"""
    from ocr_service.config import ServiceConfig

    config = ServiceConfig()
    assert config.default_match_threshold == 0.9


def test_color_score_calibration():
    """颜色打分按极大色差像素占比线性计：1%→0.85、2%→0.70、≥6.67%→0。"""
    matcher = ImageMatcher()
    h, w = 100, 100  # 10000 像素

    def make(ratio_red: float):
        """构造 template=纯灰底；patch=灰底 + ratio_red 比例的纯红像素（ΔE 远>30）。"""
        tpl = np.full((h, w, 3), 128, dtype=np.uint8)
        patch = tpl.copy()
        n = int(h * w * ratio_red)
        if n > 0:
            # 无重复采样，保证 frac 精确等于 ratio_red
            idx = np.random.default_rng(0).choice(h * w, n, replace=False)
            ys, xs = np.divmod(idx, w)
            patch[ys, xs] = (0, 0, 255)  # BGR 红
        return patch, tpl

    # 0% -> 1.0
    s, _ = matcher._color_score(*make(0.0))
    assert abs(s - 1.0) < 1e-6
    # 1% -> 0.85
    s, frac = matcher._color_score(*make(0.01))
    assert abs(frac - 0.01) < 1e-3
    assert abs(s - 0.85) < 1e-3
    # 2% -> 0.70
    s, frac = matcher._color_score(*make(0.02))
    assert abs(frac - 0.02) < 1e-3
    assert abs(s - 0.70) < 1e-3
    # 6.67% -> 0
    s, _ = matcher._color_score(*make(0.0667))
    assert s == 0.0
    # 尺寸不匹配 -> 0
    s, _ = matcher._color_score(
        np.zeros((10, 10, 3), np.uint8), np.zeros((11, 11, 3), np.uint8)
    )
    assert s == 0.0
