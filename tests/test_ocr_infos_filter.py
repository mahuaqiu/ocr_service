"""get_ocr_infos filter_text 匹配逻辑测试：精确优先，包含降级（与 get_coord_by_text 一致）。"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.api.routes import get_ocr_infos
from ocr_service.api.schemas import OCRRequest, PointModel, TextBlockModel


def _engine_result(status="success", texts=None, error=None):
    return SimpleNamespace(status=status, texts=texts or [], duration_ms=12, error=error)


def _text_block(text, x, y):
    return TextBlockModel(
        text=text,
        confidence=0.98,
        bbox=[[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]],
        center=PointModel(x=x, y=y),
    )


def _run(texts, filter_text):
    with patch("ocr_service.api.routes.get_ocr_engine") as g:
        g.return_value.recognize.return_value = _engine_result(status="success", texts=texts)
        req = OCRRequest(image="aaa", filter_text=filter_text)
        return asyncio.run(get_ocr_infos(req))


def test_ocr_infos_exact_match_wins_over_containment():
    """屏幕同时有"开始"和"开始游戏"时，filter_text="开始" 只应命中精确匹配。"""
    resp = _run([_text_block("开始", 10, 10), _text_block("开始游戏", 50, 10)], "开始")

    assert [t.text for t in resp.texts] == ["开始"]
    assert [c for c in resp.coords] == [PointModel(x=10, y=10)]
    # ocr_info 仍返回全部识别结果，便于排查
    assert [o.text for o in resp.ocr_info] == ["开始", "开始游戏"]


def test_ocr_infos_containment_fallback_when_no_exact():
    """无精确匹配时降级为包含匹配。"""
    resp = _run([_text_block("开始游戏", 50, 10)], "开始")

    assert [t.text for t in resp.texts] == ["开始游戏"]


def test_ocr_infos_multiple_exact_matches_returned_together():
    """多个完全相等的块都返回，与 get_coord_by_text 行为一致。"""
    resp = _run(
        [_text_block("确认", 10, 10), _text_block("确认", 50, 10), _text_block("确认取消", 90, 10)],
        "确认",
    )

    assert [t.text for t in resp.texts] == ["确认", "确认"]
    assert len(resp.coords) == 2


def test_ocr_infos_regex_mode_unaffected():
    """reg_ 前缀正则匹配行为不变。"""
    resp = _run([_text_block("雨大", 10, 10), _text_block("雨小", 50, 10), _text_block("晴天", 90, 10)], "reg_雨[大中小]")

    assert [t.text for t in resp.texts] == ["雨大", "雨小"]


def test_ocr_infos_filter_no_hit_returns_empty_texts():
    """无命中时 texts/coords 为空，ocr_info 保留全部识别内容。"""
    resp = _run([_text_block("用户名", 10, 10)], "密码")

    assert resp.texts == []
    assert resp.coords == []
    assert [o.text for o in resp.ocr_info] == ["用户名"]
