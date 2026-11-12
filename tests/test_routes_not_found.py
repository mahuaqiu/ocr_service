"""路由层未命中诊断与吞错误修复测试。"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.api.routes import image_match, image_match_near_text, ocr_get_coord_by_text
from ocr_service.api.schemas import (
    ImageMatchRequest,
    OCRRequest,
    PointModel,
    TextBlockModel,
    TextNearImageRequest,
)


def _engine_result(status="success", texts=None, error=None):
    return SimpleNamespace(status=status, texts=texts or [], duration_ms=12, error=error)


def _text_block(text, x, y):
    return TextBlockModel(
        text=text,
        confidence=0.98,
        bbox=[[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]],
        center=PointModel(x=x, y=y),
    )


def test_coord_by_text_engine_error_returns_error_status():
    """引擎报错必须透传 status=error，不得吞成 success+空。"""
    with patch("ocr_service.api.routes.get_ocr_engine") as g:
        g.return_value.recognize.return_value = _engine_result(status="error", error="模型加载失败")
        req = OCRRequest(image="aaa", filter_text="密码")
        resp = asyncio.run(ocr_get_coord_by_text(req))

    assert resp.status == "error"
    assert resp.error == "模型加载失败"


def test_coord_by_text_no_text_recognized():
    with patch("ocr_service.api.routes.get_ocr_engine") as g:
        g.return_value.recognize.return_value = _engine_result(status="success", texts=[])
        req = OCRRequest(image="aaa", filter_text="密码")
        resp = asyncio.run(ocr_get_coord_by_text(req))

    assert resp.status == "success"
    assert resp.texts == []
    assert resp.not_found_reason == "no_text_recognized"


def test_coord_by_text_text_not_found():
    with patch("ocr_service.api.routes.get_ocr_engine") as g:
        g.return_value.recognize.return_value = _engine_result(
            status="success", texts=[_text_block("用户名", 100, 200)]
        )
        req = OCRRequest(image="aaa", filter_text="密码")
        resp = asyncio.run(ocr_get_coord_by_text(req))

    assert resp.status == "success"
    assert resp.not_found_reason == "text_not_found"
    assert len(resp.ocr_info) == 1  # 屏幕实际识别内容仍返回


def test_image_match_passthrough_diagnostics():
    matcher_result = SimpleNamespace(
        status="success", matches=[], duration_ms=9,
        not_found_reason="below_threshold", best_confidence=0.7231, error=None,
    )
    with patch("ocr_service.api.routes.get_image_matcher") as g:
        g.return_value.match.return_value = matcher_result
        req = ImageMatchRequest(source_image="aaa", template_image="bbb", confidence_threshold=0.9)
        resp = asyncio.run(image_match(req))

    assert resp.status == "success"
    assert resp.not_found_reason == "below_threshold"
    assert resp.best_confidence == 0.7231


def test_near_text_distance_exceeded_reports_nearest():
    """文字找到、模板也匹配到，但全部超出 max_distance：distance_exceeded + 最近距离。"""
    engine_result = _engine_result(status="success", texts=[_text_block("设置", 100, 100)])
    matched = SimpleNamespace(
        confidence=0.95,
        bbox=SimpleNamespace(x=0, y=0, width=20, height=20),
        center=SimpleNamespace(x=10, y=10),
    )
    matcher_result = SimpleNamespace(
        status="success", matches=[matched], duration_ms=9,
        not_found_reason=None, best_confidence=None, error=None,
    )
    with patch("ocr_service.api.routes.get_ocr_engine") as ge, \
         patch("ocr_service.api.routes.get_image_matcher") as gm:
        ge.return_value.recognize.return_value = engine_result
        gm.return_value.match.return_value = matcher_result
        req = TextNearImageRequest(
            image="aaa", target_image="bbb", filter_text="设置", max_distance=5,
        )
        resp = asyncio.run(image_match_near_text(req))

    assert resp.status == "success"
    assert resp.not_found_reason == "distance_exceeded"
    assert resp.nearest_distance is not None and resp.nearest_distance > 5
