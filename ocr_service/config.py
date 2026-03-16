"""
OCR 服务配置。
"""

import os
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """OCR 服务配置。"""

    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8081
    debug: bool = False

    # OCR 引擎配置
    ocr_lang: str = "ch"  # 默认中文
    ocr_use_gpu: bool = False
    ocr_model_dir: str | None = None  # 自定义模型目录

    # OCR 高级参数
    ocr_det_db_thresh: float = 0.3  # 文本检测阈值
    ocr_det_db_box_thresh: float = 0.6  # 文本框置信度阈值
    ocr_det_db_unclip_ratio: float = 1.5  # 文本框扩展比例
    ocr_drop_score: float = 0.5  # 低置信度结果过滤阈值

    # 图像预处理配置
    preprocess_enabled: bool = True  # 是否启用预处理
    auto_resize_enabled: bool = True  # 是否自动缩放大图
    max_image_width: int = 1920  # 最大图像宽度
    max_image_height: int = 1080  # 最大图像高度

    # 图像匹配配置
    default_match_threshold: float = 0.8
    default_match_method: str = "template"  # template / feature

    # 缓存目录
    cache_dir: str = "cache"

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """从环境变量加载配置。"""
        return cls(
            host=os.getenv("OCR_HOST", "0.0.0.0"),
            port=int(os.getenv("OCR_PORT", "8081")),
            debug=os.getenv("OCR_DEBUG", "false").lower() == "true",
            ocr_lang=os.getenv("OCR_LANG", "ch"),
            ocr_use_gpu=os.getenv("OCR_USE_GPU", "false").lower() == "true",
            ocr_model_dir=os.getenv("OCR_MODEL_DIR") or None,
            # OCR 高级参数
            ocr_det_db_thresh=float(os.getenv("OCR_DET_DB_THRESH", "0.3")),
            ocr_det_db_box_thresh=float(os.getenv("OCR_DET_DB_BOX_THRESH", "0.6")),
            ocr_det_db_unclip_ratio=float(os.getenv("OCR_DET_DB_UNCLIP_RATIO", "1.5")),
            ocr_drop_score=float(os.getenv("OCR_DROP_SCORE", "0.5")),
            # 预处理配置
            preprocess_enabled=os.getenv("OCR_PREPROCESS_ENABLED", "true").lower() == "true",
            auto_resize_enabled=os.getenv("OCR_AUTO_RESIZE_ENABLED", "true").lower() == "true",
            max_image_width=int(os.getenv("OCR_MAX_IMAGE_WIDTH", "1920")),
            max_image_height=int(os.getenv("OCR_MAX_IMAGE_HEIGHT", "1080")),
            # 其他配置
            default_match_threshold=float(os.getenv("OCR_MATCH_THRESHOLD", "0.8")),
            default_match_method=os.getenv("OCR_MATCH_METHOD", "template"),
            cache_dir=os.getenv("OCR_CACHE_DIR", "cache"),
        )


# 全局配置实例
_config: ServiceConfig | None = None


def get_config() -> ServiceConfig:
    """获取配置实例。"""
    global _config
    if _config is None:
        _config = ServiceConfig.from_env()
    return _config


def set_config(config: ServiceConfig) -> None:
    """设置配置实例。"""
    global _config
    _config = config