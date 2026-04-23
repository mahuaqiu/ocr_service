---
name: request-id 日志追踪
description: 在请求 header 中提取 request-id 并注入所有日志，方便 grep 查询同一请求的全链路日志
type: project
---

# Request-ID 日志追踪设计

## 目标

将请求 header 中的 `request-id` 注入到所有日志中，格式为 `[abc123] 日志内容`，方便用 grep 查询同一请求的全链路日志。

## 约束

- 没有 request-id header 时，不记录标记，日志保持原格式
- 全部模块的日志都需要带上 request-id（包括 OCR 引擎、图像匹配等）
- Header 参数兼容三种格式：`request-id`、`request_id`、`X-Request-Id`（优先级：X-Request-Id > request-id > request_id）

## 技术方案

使用 Python `contextvars` 在请求上下文中传递 request-id，结合日志 Formatter 自动注入标记。

### 1. 创建请求上下文变量

新建 `ocr_service/utils/request_context.py`：

```python
import contextvars

# 请求 ID 上下文变量
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default=None)

def get_request_id() -> str | None:
    """获取当前请求的 request-id"""
    return request_id_var.get()

def set_request_id(request_id: str) -> None:
    """设置当前请求的 request-id"""
    request_id_var.set(request_id)

def clear_request_id() -> None:
    """清除当前请求的 request-id"""
    request_id_var.set(None)
```

**Why:** contextvars 是 Python 3.7+ 的标准库，专为异步上下文设计，在 FastAPI 的异步请求中能正确传递，不会被其他请求干扰。

**How to apply:** 在中间件中设置，在 Formatter 中读取，在请求结束时清理。

### 2. 中间件提取 header 并设置上下文

修改 `ocr_service/server.py` 的 `RequestLoggingMiddleware`：

```python
from ocr_service.utils.request_context import set_request_id, clear_request_id

def extract_request_id(request: Request) -> str | None:
    """从 header 中提取 request-id，兼容三种参数名"""
    # 优先级：X-Request-Id > request-id > request_id
    return (
        request.headers.get("X-Request-Id") or
        request.headers.get("request-id") or
        request.headers.get("request_id")
    )

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 提取 request-id，兼容三种参数名
        request_id = extract_request_id(request)
        if request_id:
            set_request_id(request_id)

        # ... 原有请求处理逻辑 ...

        response = await call_next(request)

        # 清理上下文
        clear_request_id()

        return response
```

**Why:** 中间件是请求入口点，能最早获取 header，确保后续所有日志都能访问到 request-id。

**How to apply:** 在请求开始时设置，在请求结束时清理，避免影响后续请求。

### 3. 自定义 Formatter 注入标记

在 `ocr_service/server.py` 中创建自定义 Formatter：

```python
from ocr_service.utils.request_context import get_request_id

class RequestIdFormatter(logging.Formatter):
    """自动注入 request-id 的日志格式化器"""

    def format(self, record):
        request_id = get_request_id()
        if request_id:
            # 在消息前添加标记
            record.msg = f"[{request_id}] {record.msg}"
        return super().format(record)
```

**Why:** Formatter 是日志输出的最后一环，在此处注入标记可以覆盖所有模块的日志，无需修改各模块的 logger 调用。

**How to apply:** 替换现有的 file_handler 和 console_handler 的 Formatter。

### 4. 更换日志处理器

修改 `setup_logging()` 函数，将 Formatter 替换为 `RequestIdFormatter`：

```python
def setup_logging():
    # ... 创建 handlers ...

    # 文件处理器
    file_formatter = RequestIdFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_formatter = RequestIdFormatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
```

**Why:** 确保所有日志输出都经过 RequestIdFormatter，实现统一的 request-id 注入。

**How to apply:** 在 setup_logging() 中替换 Formatter 实例。

## 涉及文件

| 文件 | 操作 |
|------|------|
| `ocr_service/utils/request_context.py` | 新建 |
| `ocr_service/server.py` | 修改（Formatter、中间件） |

## 查询示例

```bash
# 查询特定请求的所有日志
grep "[abc123]" logs/ocr.log

# 查询所有带 request-id 的日志
grep "\[[a-zA-Z0-9-]+\]" logs/ocr.log
```

## 日志输出示例

**有 request-id 时：**
```
2024-01-01 10:00:00 - ocr_service - INFO - [abc123] [REQUEST] POST /ocr/get_ocr_infos
2024-01-01 10:00:01 - ocr_service.core.ocr_engine - INFO - [abc123] OCR识别成功: 识别到 5 个文字块
2024-01-01 10:00:02 - ocr_service - INFO - [abc123] [RESPONSE] POST /ocr/get_ocr_infos -> 200
```

**没有 request-id 时：**
```
2024-01-01 10:00:00 - ocr_service - INFO - [REQUEST] POST /ocr/get_ocr_infos
2024-01-01 10:00:01 - ocr_service.core.ocr_engine - INFO - OCR识别成功: 识别到 5 个文字块
```