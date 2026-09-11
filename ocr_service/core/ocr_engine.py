"""
OCR 引擎封装。

基于 PaddleOCR 实现文字识别，支持中英文等多语言。
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

# 获取日志记录器
logger = logging.getLogger(__name__)

from ocr_service.config import ServiceConfig, get_config
from ocr_service.models.ocr_result import OCRResult
from ocr_service.text_replacer import apply_replacements, get_replace_map
from ocr_service.utils.image_utils import decode_image
from ocr_service.core.image_preprocessor import (
    ImagePreprocessor,
    PreprocessMode,
    get_image_preprocessor,
)


# OCR 预设配置 (PaddleOCR 3.3 兼容)
OCR_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "text_det_thresh": 0.3,
        "text_det_box_thresh": 0.6,
        "text_det_unclip_ratio": 1.5,
        "text_rec_score_thresh": 0.5,
    },
    "screenshot": {
        "text_det_thresh": 0.2,  # 降低阈值，提高小字检测
        "text_det_box_thresh": 0.5,
        "text_det_unclip_ratio": 1.8,  # 扩大文本框
        "text_rec_score_thresh": 0.4,
    },
    "mobile": {
        "text_det_thresh": 0.2,
        "text_det_box_thresh": 0.5,
        "text_det_unclip_ratio": 1.6,
        "text_rec_score_thresh": 0.4,
    },
    "low_quality": {
        "text_det_thresh": 0.15,  # 更低的阈值
        "text_det_box_thresh": 0.4,
        "text_det_unclip_ratio": 2.0,
        "text_rec_score_thresh": 0.3,
    },
}


class OCREngine:
    """
    OCR 引擎类。

    封装 PaddleOCR，提供文字识别能力。

    Attributes:
        config: 服务配置。
        _ocr: PaddleOCR 实例（延迟加载）。
        _preprocessor: 图像预处理器。
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        """
        初始化 OCR 引擎。

        Args:
            config: 服务配置，默认使用全局配置。
        """
        self.config = config or get_config()
        self._ocr = None
        self._preprocessor: Optional[ImagePreprocessor] = None

    @property
    def preprocessor(self) -> ImagePreprocessor:
        """延迟加载预处理器。"""
        if self._preprocessor is None:
            self._preprocessor = get_image_preprocessor()
        return self._preprocessor

    @property
    def ocr(self):
        """延迟加载 PaddleOCR 实例（带锁保护）。"""
        if self._ocr is None:
            with _ocr_lock:
                # 双重检查，避免重复初始化
                if self._ocr is None:
                    self._ocr = self._create_ocr_instance()
        return self._ocr

    def _create_ocr_instance(self, custom_params: Optional[Dict[str, Any]] = None):
        """创建 PaddleOCR 实例。

        Args:
            custom_params: 自定义参数，会覆盖默认配置。

        Returns:
            PaddleOCR 实例。
        """
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x 参数
        params = {
            "lang": self.config.ocr_lang,
            "device": "gpu" if self.config.ocr_use_gpu else "cpu",
            "ocr_version": self.config.ocr_version,
            # 禁用不必要的功能
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }

        # 高级参数
        if hasattr(self.config, 'ocr_det_db_thresh'):
            params["text_det_thresh"] = self.config.ocr_det_db_thresh
        if hasattr(self.config, 'ocr_det_db_box_thresh'):
            params["text_det_box_thresh"] = self.config.ocr_det_db_box_thresh
        if hasattr(self.config, 'ocr_det_db_unclip_ratio'):
            params["text_det_unclip_ratio"] = self.config.ocr_det_db_unclip_ratio
        if hasattr(self.config, 'ocr_drop_score'):
            params["text_rec_score_thresh"] = self.config.ocr_drop_score

        # 自定义参数覆盖
        if custom_params:
            params.update(custom_params)

        return PaddleOCR(**params)

    def recognize(
        self,
        image_data: bytes | str,
        lang: Optional[str] = None,
        confidence_threshold: float = 0.0,
        preprocess_mode: str = PreprocessMode.AUTO,
        ocr_preset: str = "default",
        custom_ocr_params: Optional[Dict[str, Any]] = None,
    ) -> OCRResult:
        """
        识别图片中的所有文字。

        Args:
            image_data: 图像字节数据或 Base64 编码字符串。
            lang: 语言代码，默认使用配置中的语言。
            confidence_threshold: 置信度阈值，低于此值的结果将被过滤。
            preprocess_mode: 预处理模式。
            ocr_preset: OCR 预设配置。
            custom_ocr_params: 自定义 OCR 参数。

        Returns:
            OCRResult: 识别结果。
        """
        start_time = time.time()

        try:
            # 解码图像
            image = decode_image(image_data)

            # 预处理图像
            preprocess_result = self.preprocessor.preprocess(image, mode=preprocess_mode)
            processed_image = preprocess_result.image
            scale = preprocess_result.scale

            # 获取 OCR 参数
            ocr_params = OCR_PRESETS.get(ocr_preset, OCR_PRESETS["default"])
            if custom_ocr_params:
                ocr_params = {**ocr_params, **custom_ocr_params}

            # 选择 OCR 实例并执行 OCR（锁保护区域）
            with _ocr_lock:
                if ocr_preset == "default" and custom_ocr_params is None:
                    ocr_instance = self.ocr
                else:
                    ocr_instance = self._create_ocr_instance(ocr_params)

                # 执行 OCR（核心保护区域）
                ocr_result = ocr_instance.predict(processed_image)

            # 打印 OCR 识别结果（只显示文字和置信度，不打印数组坐标）
            if ocr_result:
                for item in ocr_result:
                    rec_texts = item.get('rec_texts', []) if hasattr(item, 'keys') else []
                    rec_scores = item.get('rec_scores', []) if hasattr(item, 'keys') else []
                    logger.info(f"[OCR_RAW] 识别到 {len(rec_texts)} 个文字块: {rec_texts}, 置信度: {rec_scores}")

            # 解析 PaddleOCR 3.x 结果
            texts = OCRResult.parse_from_paddleocr(ocr_result, confidence_threshold, scale)

            # 应用配置中心替换规则（在所有 reg_/文本匹配之前）
            replace_map = get_replace_map()
            if replace_map:
                for text_block in texts:
                    text_block.text = apply_replacements(text_block.text)

            duration_ms = int((time.time() - start_time) * 1000)

            # 记录识别结果日志
            if texts:
                logger.info(f"OCR识别成功: 识别到 {len(texts)} 个文字块, 耗时 {duration_ms}ms")
                for i, tb in enumerate(texts):
                    logger.info(f"  [{i+1}] 文本: {tb.text}, 置信度: {tb.confidence:.2f}, 坐标: ({tb.center.x}, {tb.center.y})")
            else:
                logger.warning(f"OCR识别成功但未检测到文字, 耗时 {duration_ms}ms")

            return OCRResult(
                status="success",
                texts=texts,
                duration_ms=duration_ms,
            )

        except Exception as e:
            logger.exception("OCR识别失败")
            duration_ms = int((time.time() - start_time) * 1000)
            return OCRResult(
                status="error",
                texts=[],
                duration_ms=duration_ms,
                error=str(e),
            )


# 全局 OCR 锁（保护并发调用，使用可重入锁避免死锁）
_ocr_lock = threading.RLock()


# 全局引擎实例
_engine: Optional[OCREngine] = None


def get_ocr_engine() -> OCREngine:
    """获取全局 OCR 引擎实例。"""
    global _engine
    if _engine is None:
        with _ocr_lock:
            # 双重检查，避免重复初始化
            if _engine is None:
                _engine = OCREngine()
    return _engine