# Request-ID 日志追踪实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在请求 header 中提取 request-id 并注入所有日志，格式为 `[abc123] 日志内容`

**Architecture:** 使用 Python contextvars 在请求上下文中传递 request-id，结合自定义日志 Formatter 自动注入标记

**Tech Stack:** Python contextvars、logging.Formatter、FastAPI middleware

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `ocr_service/utils/request_context.py` | 新建 | 上下文变量和辅助函数 |
| `ocr_service/server.py` | 修改 | Formatter、提取函数、中间件 |

---

### Task 1: 创建请求上下文模块

**Files:**
- Create: `ocr_service/utils/request_context.py`

- [ ] **Step 1: 创建上下文变量模块**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add ocr_service/utils/request_context.py
git commit -m "feat: 添加 request-id 上下文管理模块"
```

---

### Task 2: 创建 RequestIdFormatter 和提取函数

**Files:**
- Modify: `ocr_service/server.py`

- [ ] **Step 1: 添加导入和 Formatter 类**

在文件开头添加导入（约第 16 行后）：

```python
import contextvars
```

在 `truncate_base64` 函数之前（约第 37 行）添加 Formatter 类：

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


def extract_request_id(request: Request) -> str | None:
    """从 header 中提取 request-id，兼容三种参数名"""
    # 优先级：X-Request-Id > request-id > request_id
    return (
        request.headers.get("X-Request-Id") or
        request.headers.get("request-id") or
        request.headers.get("request_id")
    )
```

- [ ] **Step 2: Commit**

```bash
git add ocr_service/server.py
git commit -m "feat: 添加 RequestIdFormatter 和 extract_request_id 函数"
```

---

### Task 3: 修改中间件提取 request-id

**Files:**
- Modify: `ocr_service/server.py:101-143`

- [ ] **Step 1: 修改 RequestLoggingMiddleware**

修改 `RequestLoggingMiddleware` 类，添加 request-id 提取和清理：

```python
from ocr_service.utils.request_context import set_request_id, clear_request_id

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求/响应日志中间件"""

    async def dispatch(self, request: Request, call_next):
        # 提取 request-id，兼容三种参数名
        request_id = extract_request_id(request)
        if request_id:
            set_request_id(request_id)

        # 读取请求体
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            # 重新构建请求以允许后续读取
            async def receive():
                return {"type": "http.request", "body": body}
            request = Request(request.scope, receive)

        # 记录请求日志
        try:
            if body:
                body_str = body.decode() if isinstance(body, bytes) else body
                body = json.loads(body_str)
        except Exception:
            body = None

        logger.info(format_request_log(request, body))

        # 调用处理函数
        response = await call_next(request)

        # 读取响应体
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        # 记录响应日志
        body_json = json.loads(response_body.decode()) if response_body else {}
        logger.info(format_response_log(request, body_json, response.status_code))

        # 清理上下文
        clear_request_id()

        # 返回响应
        from starlette.responses import Response
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
```

- [ ] **Step 2: Commit**

```bash
git add ocr_service/server.py
git commit -m "feat: 中间件提取 request-id 并注入上下文"
```

---

### Task 4: 更换日志 Formatter

**Files:**
- Modify: `ocr_service/server.py:155-192`

- [ ] **Step 1: 修改 setup_logging 函数**

将 Formatter 替换为 `RequestIdFormatter`：

```python
def setup_logging():
    """配置日志系统，实现持久化和轮转。"""
    # 创建 logs 目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "ocr.log")

    # 获取应用 logger
    app_logger = logging.getLogger("ocr_service")
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False  # 先关闭传播

    # 清除现有处理器
    app_logger.handlers.clear()

    # 文件处理器 - 轮转日志，最大50M，保留3个备份
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(RequestResponseFilter())
    file_formatter = RequestIdFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    app_logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(RequestResponseFilter())
    console_formatter = RequestIdFormatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    app_logger.addHandler(console_handler)
```

- [ ] **Step 2: Commit**

```bash
git add ocr_service/server.py
git commit -m "feat: 使用 RequestIdFormatter 替换日志格式化器"
```

---

### Task 5: 测试验证

- [ ] **Step 1: 启动服务**

```bash
cd D:/code/ocr_service
source venv/Scripts/activate
python -m ocr_service.server
```

Expected: 服务正常启动，监听 8081 端口

- [ ] **Step 2: 发送带 request-id 的请求测试**

```bash
curl -X POST http://localhost:8081/ocr/get_ocr_infos \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: test-abc123" \
  -d '{"image": "<base64_image_data>"}'
```

Expected: 日志输出包含 `[test-abc123]` 标记

- [ ] **Step 3: 验证日志格式**

检查 `logs/ocr.log`，确认日志格式为：
```
2024-04-24 10:00:00 - ocr_service - INFO - [test-abc123] [REQUEST] POST /ocr/get_ocr_infos
```

- [ ] **Step 4: Commit 最终变更**

```bash
git add -A
git commit -m "feat: 完成 request-id 日志追踪功能"
```