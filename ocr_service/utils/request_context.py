"""
请求上下文管理。

使用 contextvars 在异步请求中传递 request-id。
"""

import contextvars

# 请求 ID 上下文变量
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'request_id', default=None
)


def get_request_id() -> str | None:
    """获取当前请求的 request-id"""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """设置当前请求的 request-id"""
    request_id_var.set(request_id)


def clear_request_id() -> None:
    """清除当前请求的 request-id"""
    request_id_var.set(None)