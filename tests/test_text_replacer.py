"""text_replacer 单元测试：替换逻辑 + 配置拉取解析 + 定时刷新语义。"""
import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service import text_replacer
from ocr_service.text_replacer import (
    apply_replacements,
    fetch_replace_map,
    parse_replace_map,
    refresh_replace_map,
    set_replace_map,
)


@pytest.fixture(autouse=True)
def clean_replace_map():
    """每个用例前后清空全局替换字典，避免串扰。"""
    set_replace_map({})
    yield
    set_replace_map({})


# ---------------- apply_replacements ----------------

def test_apply_replacements_basic():
    set_replace_map({"充许": "允许"})
    assert apply_replacements("充许用户聊天") == "允许用户聊天"


def test_apply_replacements_multiple_pairs():
    set_replace_map({"充许": "允许", "聊关": "聊天"})
    assert apply_replacements("充许聊关") == "允许聊天"


def test_apply_replacements_empty_map_passthrough():
    assert apply_replacements("充许") == "充许"


def test_apply_replacements_order_overlap():
    """按配置顺序执行：A->B 且 B->C 会链式叠加（预期行为）。"""
    set_replace_map({"A": "B", "B": "C"})
    assert apply_replacements("A") == "C"


def test_apply_replacements_skips_empty_key():
    set_replace_map({"": "X", "充许": "允许"})
    assert apply_replacements("充许") == "允许"


# ---------------- parse_replace_map ----------------

def test_parse_replace_map_valid():
    assert parse_replace_map({"充许": "允许"}) == {"充许": "允许"}


def test_parse_replace_map_rejects_non_dict():
    with pytest.raises(ValueError):
        parse_replace_map(["充许", "允许"])


def test_parse_replace_map_rejects_non_str_value():
    with pytest.raises(ValueError):
        parse_replace_map({"a": 1})


def test_parse_replace_map_rejects_non_str_key():
    with pytest.raises(ValueError):
        parse_replace_map({1: "a"})


# ---------------- fetch_replace_map ----------------

def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_fetch_replace_map_success(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["key"] == "ocr_config"
        return httpx.Response(200, json={"充许": "允许", "聊关": "聊天"})

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    mapping = asyncio.run(
        fetch_replace_map(
            "http://platform/api/public/config-center/query", "ocr_config", 5.0
        )
    )
    assert mapping == {"充许": "允许", "聊关": "聊天"}


def test_fetch_replace_map_404(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "配置项不存在: ocr_config"})

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    with pytest.raises(RuntimeError, match="404"):
        asyncio.run(
            fetch_replace_map(
                "http://platform/api/public/config-center/query", "ocr_config", 5.0
            )
        )


def test_fetch_replace_map_invalid_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    with pytest.raises(Exception):
        asyncio.run(
            fetch_replace_map(
                "http://platform/api/public/config-center/query", "ocr_config", 5.0
            )
        )


def test_fetch_replace_map_non_dict(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["a", "b"])

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    with pytest.raises(ValueError):
        asyncio.run(
            fetch_replace_map(
                "http://platform/api/public/config-center/query", "ocr_config", 5.0
            )
        )


# ---------------- refresh_replace_map ----------------

def test_refresh_replace_map_success_updates_map(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"充许": "允许"})

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    assert (
        asyncio.run(refresh_replace_map("http://platform/q", "ocr_config", 5.0))
        is True
    )
    assert apply_replacements("充许") == "允许"


def test_refresh_replace_map_failure_keeps_old(monkeypatch):
    set_replace_map({"充许": "允许"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    assert (
        asyncio.run(refresh_replace_map("http://platform/q", "ocr_config", 5.0))
        is False
    )
    assert apply_replacements("充许") == "允许"
