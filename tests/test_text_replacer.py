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


def test_apply_replacements_no_chaining():
    """单趟替换：替换结果不会被其它 key 再次扫描（A->B 后不再触发 B->C）。"""
    set_replace_map({"A": "B", "B": "C"})
    assert apply_replacements("A") == "B"


def test_apply_replacements_value_not_rescanned():
    """替换产物中即使包含其它 key，也不会被二次替换。"""
    set_replace_map({"聊关": "聊天", "天": "无"})
    assert apply_replacements("聊关") == "聊天"


def test_apply_replacements_longest_match_wins_regardless_of_order():
    """重叠 key 最长匹配优先，与配置顺序无关。"""
    set_replace_map({"许": "x", "充许": "允许"})
    assert apply_replacements("充许") == "允许"
    set_replace_map({"充许": "允许", "许": "x"})
    assert apply_replacements("充许") == "允许"


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
        fetch_replace_map("http://platform/api/public/config-center/query")
    )
    assert mapping == {"充许": "允许", "聊关": "聊天"}


def test_fetch_replace_map_404(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "配置项不存在: ocr_config"})

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    with pytest.raises(RuntimeError, match="404"):
        asyncio.run(fetch_replace_map("http://platform/api/public/config-center/query"))


def test_fetch_replace_map_invalid_json(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    with pytest.raises(Exception):
        asyncio.run(fetch_replace_map("http://platform/api/public/config-center/query"))


def test_fetch_replace_map_non_dict(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["a", "b"])

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    with pytest.raises(ValueError):
        asyncio.run(fetch_replace_map("http://platform/api/public/config-center/query"))


# ---------------- refresh_replace_map ----------------

def test_refresh_replace_map_success_updates_map(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"充许": "允许"})

    monkeypatch.setattr(
        text_replacer, "_create_client", lambda timeout: _mock_client(handler)
    )
    assert (
        asyncio.run(refresh_replace_map("http://platform/q"))
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
        asyncio.run(refresh_replace_map("http://platform/q"))
        is False
    )
    assert apply_replacements("充许") == "允许"


# ---------------- _seconds_until_next_noon ----------------

def test_seconds_until_next_noon_before_noon():
    from datetime import datetime

    from ocr_service.server import _seconds_until_next_noon

    assert _seconds_until_next_noon(datetime(2026, 9, 11, 6, 0, 0)) == 6 * 3600


def test_seconds_until_next_noon_after_noon():
    from datetime import datetime

    from ocr_service.server import _seconds_until_next_noon

    assert _seconds_until_next_noon(datetime(2026, 9, 11, 13, 0, 0)) == 23 * 3600


def test_seconds_until_next_noon_exactly_noon():
    from datetime import datetime

    from ocr_service.server import _seconds_until_next_noon

    assert _seconds_until_next_noon(datetime(2026, 9, 11, 12, 0, 0)) == 24 * 3600


# ---------------- 启动拉取失败后的重试节奏 ----------------


class _StopLoop(Exception):
    pass


def test_next_refresh_wait_after_failure_retries_in_10_minutes(monkeypatch):
    from ocr_service import server

    monkeypatch.setattr(server, "_seconds_until_next_noon", lambda: 99999.0)
    assert server._next_refresh_wait(False) == 600.0


def test_next_refresh_wait_after_success_waits_until_next_noon(monkeypatch):
    from ocr_service import server

    monkeypatch.setattr(server, "_seconds_until_next_noon", lambda: 3600.0)
    assert server._next_refresh_wait(True) == 3600.0


def test_config_refresh_loop_retries_soon_after_failed_startup(monkeypatch):
    """启动拉取失败后，首次重试应在 10 分钟内，而不是等到下一个 12:00。"""
    from ocr_service import server

    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)
        if len(waits) >= 2:
            raise _StopLoop

    async def fake_refresh(url):
        return True

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(server, "refresh_replace_map", fake_refresh)
    monkeypatch.setattr(server, "_seconds_until_next_noon", lambda: 86400.0)

    with pytest.raises(_StopLoop):
        asyncio.run(
            server._config_refresh_loop("http://platform/q", startup_success=False)
        )

    assert waits[0] == 600.0  # 失败 → 10 分钟后首次重试
    assert waits[1] == 86400.0  # 成功 → 等到次日 12:00


def test_pull_config_on_startup_returns_success_flag(monkeypatch):
    from ocr_service import server

    attempts = []

    async def fake_refresh(url):
        attempts.append(1)
        return len(attempts) >= 2

    async def fake_sleep(_seconds):
        return None

    monkeypatch.setattr(server, "refresh_replace_map", fake_refresh)
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    assert asyncio.run(server._pull_config_on_startup("http://p/q")) is True
    assert len(attempts) == 2
