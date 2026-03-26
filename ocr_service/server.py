"""
OCR 服务 FastAPI 入口。

Usage:
    # 直接运行
    python -m ocr_service.server

    # 指定端口
    python -m ocr_service.server --port 8081

    # 使用 uvicorn
    uvicorn ocr_service.server:app --host 0.0.0.0 --port 8081
"""

import argparse
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from ocr_service import __version__
from ocr_service.api.routes import router
from ocr_service.config import ServiceConfig, get_config, set_config

# 使用根日志记录器，确保日志能写入文件和控制台
logger = logging.getLogger("ocr_service")

# Base64 缩写的最大长度
TRUNCATE_LENGTH = 50


def truncate_base64(data: Any, max_length: int = TRUNCATE_LENGTH) -> Any:
    """
    递归处理数据，对 Base64 字符串进行缩写。

    Args:
        data: 输入数据
        max_length: 最大保留长度

    Returns:
        处理后的数据
    """
    if isinstance(data, str):
        # 检测是否为 Base64 数据（包含常见图片前缀或纯 Base64）
        if len(data) > max_length:
            return f"{data[:max_length]}...[truncated, total={len(data)}]"
        return data
    elif isinstance(data, dict):
        return {k: truncate_base64(v, max_length) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [truncate_base64(item, max_length) for item in data]
    else:
        return data


def format_request_log(request: Request, body: Any = None) -> str:
    """格式化请求日志"""
    lines = [f"[REQUEST] {request.method} {request.url.path}"]

    # 从请求体中提取参数
    if body:
        truncated_body = truncate_base64(body)
        # 转换为可读格式
        if isinstance(truncated_body, dict):
            for key, value in truncated_body.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  body: {truncated_body}")

    return "\n".join(lines)


def format_response_log(request: Request, response_body: Any, status_code: int) -> str:
    """格式化响应日志"""
    lines = [f"[RESPONSE] {request.method} {request.url.path} -> {status_code}"]

    if isinstance(response_body, dict):
        # 只记录关键字段
        for key in ["status", "texts", "coords", "duration_ms", "error", "text", "matches"]:
            if key in response_body:
                value = response_body[key]
                if key == "texts":
                    value = f"[{len(value)} items]" if isinstance(value, list) else value
                elif key == "coords":
                    value = f"[{len(value)} items]" if isinstance(value, list) else value
                elif key == "matches":
                    value = f"[{len(value)} items]" if isinstance(value, list) else value
                lines.append(f"  {key}: {value}")
    else:
        lines.append(f"  body: {truncate_base64(response_body)}")

    return "\n".join(lines)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求/响应日志中间件"""

    async def dispatch(self, request: Request, call_next):
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

        # 返回响应
        from starlette.responses import Response
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


class RequestResponseFilter(logging.Filter):
    """日志过滤器：允许 REQUEST/RESPONSE、OCR_RAW 和错误日志通过"""

    def filter(self, record):
        msg = record.getMessage()
        # 允许包含 [REQUEST]、[RESPONSE]、[OCR_RAW] 或 ERROR 级别的日志
        return "[REQUEST]" in msg or "[RESPONSE]" in msg or "[OCR_RAW]" in msg or record.levelno >= logging.ERROR


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
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    app_logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(RequestResponseFilter())
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    app_logger.addHandler(console_handler)

def create_app(config: ServiceConfig = None) -> FastAPI:
    """
    创建 FastAPI 应用实例。

    Args:
        config: 服务配置。

    Returns:
        FastAPI: 应用实例。
    """
    # 初始化日志系统
    setup_logging()

    if config:
        set_config(config)
    else:
        config = get_config()

    app = FastAPI(
        title="OCR Service",
        description="基于 PaddleOCR 的文字识别和 OpenCV 图像匹配服务",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求/响应日志中间件
    app.add_middleware(RequestLoggingMiddleware)

    # 注册路由
    app.include_router(router, prefix="")

    return app


# 创建默认应用实例
app = create_app()


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="OCR Service")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("OCR_HOST", "0.0.0.0"),
        help="服务监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("OCR_PORT", "8081")),
        help="服务监听端口",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式，自动重载",
    )
    args = parser.parse_args()

    uvicorn.run(
        "ocr_service.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()