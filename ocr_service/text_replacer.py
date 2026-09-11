"""
OCR 识别文本替换配置。

从测试平台配置中心拉取替换规则（如 {"充许":"允许","聊关":"聊天"}），
在识别结果进入任何 reg_/文本匹配之前对文本做整串替换。
"""

import logging
from typing import Dict

import httpx

logger = logging.getLogger(__name__)

# 全局替换字典（键=待替换文本，值=替换后文本）；空字典表示不替换
_replace_map: Dict[str, str] = {}


def get_replace_map() -> Dict[str, str]:
    """获取当前替换字典。"""
    return _replace_map


def set_replace_map(mapping: Dict[str, str]) -> None:
    """设置替换字典。"""
    global _replace_map
    _replace_map = dict(mapping)


def apply_replacements(text: str) -> str:
    """按配置顺序对文本逐对替换。"""
    for wrong, right in _replace_map.items():
        if wrong:
            text = text.replace(wrong, right)
    return text


def parse_replace_map(data) -> Dict[str, str]:
    """校验接口返回数据为 {str: str} 字典。"""
    if not isinstance(data, dict):
        raise ValueError(f"配置中心返回的不是 JSON 对象: {type(data).__name__}")
    result: Dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"配置值键值对必须是字符串: {key!r}={value!r}")
        result[key] = value
    return result


def _create_client(timeout: float) -> httpx.AsyncClient:
    """创建 HTTP 客户端（独立函数便于测试注入 MockTransport）。"""
    return httpx.AsyncClient(timeout=timeout)


async def fetch_replace_map(url: str, key: str, timeout: float) -> Dict[str, str]:
    """从配置中心拉取替换字典。"""
    async with _create_client(timeout) as client:
        response = await client.get(url, params={"key": key})
    if response.status_code != 200:
        raise RuntimeError(
            f"配置中心返回 HTTP {response.status_code}: {response.text[:200]}"
        )
    return parse_replace_map(response.json())


async def refresh_replace_map(url: str, key: str, timeout: float) -> bool:
    """拉取并应用替换配置；失败保留旧配置。返回是否成功。"""
    try:
        mapping = await fetch_replace_map(url, key, timeout)
    except Exception as exc:
        logger.error(
            "[CONFIG] 替换配置拉取失败，保留旧配置(当前 %d 对): %s",
            len(_replace_map),
            exc,
        )
        return False
    set_replace_map(mapping)
    logger.info("[CONFIG] 替换配置已更新: key=%s, 共 %d 对规则", key, len(mapping))
    return True
