"""
OCR 引擎封装。

基于 PaddleOCR 实现文字识别，支持中英文等多语言。
"""

import logging
import re
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# 获取日志记录器
logger = logging.getLogger(__name__)

from ocr_service.config import ServiceConfig, get_config
from ocr_service.models.ocr_result import TextBlock, OCRResult, Point
from ocr_service.utils.image_utils import decode_image
from ocr_service.core.image_preprocessor import (
    ImagePreprocessor,
    PreprocessMode,
    PreprocessResult,
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
        """延迟加载 PaddleOCR 实例。"""
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

            # 选择 OCR 实例
            if ocr_preset == "default" and custom_ocr_params is None:
                ocr_instance = self.ocr
            else:
                ocr_instance = self._create_ocr_instance(ocr_params)

            # 执行 OCR (PaddleOCR 3.x 使用 predict 方法)
            ocr_result = ocr_instance.predict(processed_image)

            # 解析 PaddleOCR 3.x 结果
            texts = OCRResult.parse_from_paddleocr(ocr_result, confidence_threshold, scale)

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

    def _restore_coordinates(self, text_block: TextBlock, scale: float) -> TextBlock:
        """还原缩放后的坐标。

        Args:
            text_block: 原始文字块。
            scale: 缩放比例。

        Returns:
            TextBlock: 坐标还原后的文字块。
        """
        # 还原边界框
        restored_bbox = []
        for point in text_block.bbox:
            restored_bbox.append(Point(
                x=int(point.x / scale),
                y=int(point.y / scale),
            ))

        # 还原边界矩形
        restored_bounding_box = TextBlock._compute_bounding_box(restored_bbox)

        # 还原中心点
        restored_center = Point(
            x=int(text_block.center.x / scale),
            y=int(text_block.center.y / scale),
        )

        return TextBlock(
            text=text_block.text,
            confidence=text_block.confidence,
            bbox=restored_bbox,
            bounding_box=restored_bounding_box,
            center=restored_center,
        )

    def find_text(
        self,
        image_data: bytes | str,
        target_text: str,
        match_mode: str = "exact",
        confidence_threshold: float = 0.0,
        prefer_exact: bool = True,
        preprocess_mode: str = PreprocessMode.AUTO,
        ocr_preset: str = "default",
    ) -> Optional[TextBlock]:
        """
        在图片中查找指定文字。

        Args:
            image_data: 图像字节数据或 Base64 编码字符串。
            target_text: 目标文字。
            match_mode: 匹配模式，支持 exact（精确）、fuzzy（模糊）、regex（正则）。
            confidence_threshold: 置信度阈值。
            prefer_exact: 是否优先精确匹配。若为 True，先查找完全相等的文字，
                          未找到再查找包含匹配的文字。
            preprocess_mode: 预处理模式。
            ocr_preset: OCR 预设配置。

        Returns:
            TextBlock | None: 找到的文字块，未找到返回 None。
        """
        result = self.recognize(
            image_data,
            confidence_threshold=confidence_threshold,
            preprocess_mode=preprocess_mode,
            ocr_preset=ocr_preset,
        )

        if result.status != "success":
            return None

        # 精确匹配优先模式：先找完全相等，再找包含匹配
        if prefer_exact and match_mode == "exact":
            # 第一阶段：精确匹配（text == target）
            for text_block in result.texts:
                if text_block.text == target_text:
                    return text_block
            # 第二阶段：包含匹配（target in text）
            for text_block in result.texts:
                if target_text in text_block.text:
                    return text_block
            return None

        # 非精确优先模式，使用原有匹配逻辑
        for text_block in result.texts:
            if self._match_text(text_block.text, target_text, match_mode):
                return text_block

        return None

    def find_all_texts(
        self,
        image_data: bytes | str,
        target_text: str,
        match_mode: str = "exact",
        confidence_threshold: float = 0.0,
        prefer_exact: bool = True,
        preprocess_mode: str = PreprocessMode.AUTO,
        ocr_preset: str = "default",
    ) -> tuple[list[TextBlock], int]:
        """
        在图片中查找所有匹配的文字。

        Args:
            image_data: 图像字节数据或 Base64 编码字符串。
            target_text: 目标文字。
            match_mode: 匹配模式。
            confidence_threshold: 置信度阈值。
            prefer_exact: 是否优先精确匹配。若为 True，先查找完全相等的文字，
                          未找到再查找包含匹配的文字。
            preprocess_mode: 预处理模式。
            ocr_preset: OCR 预设配置。

        Returns:
            tuple[list[TextBlock], int]: 匹配的文字块列表和耗时(毫秒)。
        """
        result = self.recognize(
            image_data,
            confidence_threshold=confidence_threshold,
            preprocess_mode=preprocess_mode,
            ocr_preset=ocr_preset,
        )

        if result.status != "success":
            return [], result.duration_ms

        # 精确匹配优先模式：先找完全相等，再找包含匹配
        if prefer_exact and match_mode == "exact":
            matches = []
            # 第一阶段：精确匹配（text == target）
            for text_block in result.texts:
                if text_block.text == target_text:
                    matches.append(text_block)
            # 第二阶段：包含匹配（target in text），排除已匹配的
            if not matches:
                for text_block in result.texts:
                    if target_text in text_block.text:
                        matches.append(text_block)
            return matches, result.duration_ms

        # 非精确优先模式，使用原有匹配逻辑
        matches = []
        for text_block in result.texts:
            if self._match_text(text_block.text, target_text, match_mode):
                matches.append(text_block)

        return matches, result.duration_ms

    def _match_text(self, text: str, target: str, mode: str) -> bool:
        """
        匹配文字。

        Args:
            text: 实际识别的文字。
            target: 目标文字。
            mode: 匹配模式。

        Returns:
            bool: 是否匹配。
        """
        if mode == "exact":
            return target in text
        elif mode == "fuzzy":
            # 模糊匹配：忽略标点和空白
            import unicodedata

            def normalize(s):
                # 移除标点和空白
                return "".join(
                    c
                    for c in unicodedata.normalize("NFKC", s)
                    if not unicodedata.category(c).startswith("P")
                    and not unicodedata.category(c).startswith("Z")
                )

            return normalize(target) in normalize(text)
        elif mode == "regex":
            try:
                return bool(re.search(target, text))
            except re.error:
                return False
        else:
            return target in text

    def get_text_center(
        self,
        image_data: bytes | str,
        target_text: str,
        match_mode: str = "exact",
    ) -> Optional[Point]:
        """
        获取指定文字的中心坐标。

        Args:
            image_data: 图像字节数据或 Base64 编码字符串。
            target_text: 目标文字。
            match_mode: 匹配模式。

        Returns:
            Point | None: 中心坐标，未找到返回 None。
        """
        text_block = self.find_text(image_data, target_text, match_mode)
        return text_block.center if text_block else None


# 全局引擎实例
_engine: Optional[OCREngine] = None


def get_ocr_engine() -> OCREngine:
    """获取全局 OCR 引擎实例。"""
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine