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
import os
import logging
from logging.handlers import RotatingFileHandler

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocr_service import __version__
from ocr_service.api.routes import router
from ocr_service.config import ServiceConfig, get_config, set_config


def setup_logging():
    """配置日志系统，实现持久化和轮转。"""
    # 创建 logs 目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "ocr_service.log")

    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有处理器
    logger.handlers.clear()

    # 文件处理器 - 轮转日志，最大50M，保留3个备份
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"日志系统初始化完成，日志文件: {log_file}")


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