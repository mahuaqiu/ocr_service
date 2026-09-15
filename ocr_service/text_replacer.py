"""
OCR 识别文本替换配置。

从测试平台配置中心拉取替换规则（如 {"充许":"允许","聊关":"聊天"}），
在识别结果进入任何 reg_/文本匹配之前对文本做整串替换。
"""

import logging
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# 配置中心查询参数（固定值，不作可配置项）：配置键固定 ocr_config，拉取超时 5 秒
CONFIG_CENTER_KEY = "ocr_config"
FETCH_TIMEOUT = 5.0

# 全局替换状态（字典 + 预编译模式）作为整体原子换引用，读者只会看到完整旧表或完整新表。
# 空字典表示不替换。模式为按 key 长度降序的 alternation：
# 单趟替换保证 1) 最长匹配优先（与配置顺序无关）2) 替换结果不会被其它 key 再次扫描。
_replace_state: Tuple[Dict[str, str], Optional[re.Pattern]] = ({}, None)

# 上次成功应用替换配置的时间（ISO 格式）；None 表示尚未拉到过配置
_last_updated_at: Optional[str] = None


def set_replace_map(mapping: Dict[str, str]) -> None:
    """设置替换字典（空 key 直接过滤，预编译单趟替换模式）。"""
    global _replace_state, _last_updated_at
    rules = {key: value for key, value in mapping.items() if key}
    if rules:
        alternation = "|".join(
            re.escape(key) for key in sorted(rules, key=len, reverse=True)
        )
        pattern: Optional[re.Pattern] = re.compile(alternation)
    else:
        pattern = None
    _replace_state = (rules, pattern)
    _last_updated_at = datetime.now().isoformat(timespec="seconds")


def get_replace_stats() -> Dict[str, Optional[object]]:
    """当前规则数与上次成功应用配置的时间（供 /health 诊断"为什么没替换"）。"""
    rules, _ = _replace_state
    return {"rules": len(rules), "updated_at": _last_updated_at}


def apply_replacements(text: str) -> str:
    """对文本做单趟替换：最长匹配优先，替换产物不再参与匹配。"""
    rules, pattern = _replace_state
    if pattern is None or not text:
        return text
    return pattern.sub(lambda match: rules[match.group(0)], text)


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


async def fetch_replace_map(url: str) -> Dict[str, str]:
    """从配置中心拉取替换字典。"""
    async with _create_client(FETCH_TIMEOUT) as client:
        response = await client.get(url, params={"key": CONFIG_CENTER_KEY})
    if response.status_code != 200:
        raise RuntimeError(
            f"配置中心返回 HTTP {response.status_code}: {response.text[:200]}"
        )
    return parse_replace_map(response.json())


async def refresh_replace_map(url: str) -> bool:
    """拉取并应用替换配置；失败保留旧配置。返回是否成功。"""
    try:
        mapping = await fetch_replace_map(url)
    except Exception as exc:
        logger.error(
            "[CONFIG] 替换配置拉取失败，保留旧配置(当前 %d 对): %s",
            len(_replace_state[0]),
            exc,
        )
        return False
    set_replace_map(mapping)
    logger.info(
        "[CONFIG] 替换配置已更新: key=%s, 共 %d 对规则",
        CONFIG_CENTER_KEY,
        len(mapping),
    )
    return True
