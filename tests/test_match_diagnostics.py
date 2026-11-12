"""图像匹配未命中诊断字段测试（not_found_reason / best_confidence）。"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.config import ServiceConfig
from ocr_service.core.image_matcher import ImageMatcher


def _encode_bgr(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _make_panel(with_red_line: bool) -> np.ndarray:
    img = np.ones((60, 140, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (5, 5), (134, 54), (180, 180, 180), 1)
    line_color = (0, 0, 255) if with_red_line else (0, 0, 0)
    cv2.line(img, (15, 25), (120, 25), line_color, 2)
    cv2.putText(img, "ABC", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return img


@pytest.fixture
def matcher() -> ImageMatcher:
    config = ServiceConfig(default_match_threshold=0.8, default_match_method="template")
    return ImageMatcher(config=config)


def test_gray_below_threshold_reports_reason_and_best(matcher: ImageMatcher):
    """内容完全不同的模板：below_threshold + best_confidence。"""
    template = _make_panel(with_red_line=True)
    # 大图只有随机噪点，与模板无结构相似性（固定种子，避免偶发高相似）
    rng = np.random.default_rng(42)
    source = rng.integers(0, 255, (220, 320, 3), dtype=np.uint8)

    result = matcher.match_template(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.99,
    )

    assert result.status == "success"
    assert result.matches == []
    assert result.not_found_reason == "below_threshold"
    assert result.best_confidence is not None
    assert result.best_confidence < 0.99


def test_color_fail_reports_color_check_failed(matcher: ImageMatcher):
    """灰度过、颜色关不过：color_check_failed，best 为灰度分。"""
    template = _make_panel(with_red_line=True)
    source = np.ones((220, 320, 3), dtype=np.uint8) * 245
    source[80:140, 90:230] = _make_panel(with_red_line=False)

    result = matcher.match_template(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.8,
    )

    assert result.status == "success"
    assert result.matches == []
    assert result.not_found_reason == "color_check_failed"
    assert result.best_confidence is not None
    assert result.best_confidence >= 0.8


def test_template_larger_than_source_reports_reason(matcher: ImageMatcher):
    template = _make_panel(with_red_line=True)
    source = np.ones((40, 80, 3), dtype=np.uint8) * 245  # 比模板(60x140)小

    result = matcher.match_template(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.8,
    )

    assert result.status == "success"
    assert result.matches == []
    assert result.not_found_reason == "template_not_matched"


def test_match_success_has_no_diagnostic_fields(matcher: ImageMatcher):
    """命中时不携带诊断字段。"""
    template = _make_panel(with_red_line=True)
    source = np.ones((220, 320, 3), dtype=np.uint8) * 245
    source[80:140, 90:230] = _make_panel(with_red_line=True)

    result = matcher.match_template(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.8,
    )

    assert len(result.matches) == 1
    assert result.not_found_reason is None
    assert result.best_confidence is None


def test_match_all_gray_fail_reports_below_threshold(matcher: ImageMatcher):
    """多目标路径：全图无灰度候选时 below_threshold。"""
    template = _make_panel(with_red_line=True)
    rng = np.random.default_rng(42)
    source = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)

    result = matcher.match_all(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.99,
    )

    assert result.status == "success"
    assert result.matches == []
    assert result.not_found_reason == "below_threshold"
    assert result.best_confidence is not None


def test_to_dict_includes_diagnostics(matcher: ImageMatcher):
    template = _make_panel(with_red_line=True)
    source = np.ones((220, 320, 3), dtype=np.uint8) * 245
    source[80:140, 90:230] = _make_panel(with_red_line=False)

    result = matcher.match_template(
        source_data=_encode_bgr(source),
        template_data=_encode_bgr(template),
        threshold=0.8,
    )

    d = result.to_dict()
    assert d["not_found_reason"] == "color_check_failed"
    assert "best_confidence" in d
