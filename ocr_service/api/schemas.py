"""
API 请求/响应模型（Pydantic）。
"""

from enum import Enum
from typing import Optional, Tuple, List

from pydantic import BaseModel, Field


# ==================== 枚举类型 ====================

class PreprocessMode(str, Enum):
    """预处理模式。"""
    NONE = "none"  # 不预处理
    AUTO = "auto"  # 自动检测（推荐）
    SCREENSHOT = "screenshot"  # 桌面截图
    MOBILE = "mobile"  # 移动端截图
    DOCUMENT = "document"  # 文档图片
    LOW_QUALITY = "low_quality"  # 低质量图片


class OCRPreset(str, Enum):
    """OCR 预设配置。"""
    DEFAULT = "default"  # 默认配置
    SCREENSHOT = "screenshot"  # 桌面截图优化
    MOBILE = "mobile"  # 移动端优化
    LOW_QUALITY = "low_quality"  # 低质量图片优化


# ==================== 通用模型 ====================

class PointModel(BaseModel):
    """坐标点。"""

    x: int
    y: int


class TextBlockModel(BaseModel):
    """识别到的文字块。"""

    text: str
    confidence: float
    bbox: list[list[int]]
    center: PointModel


class BoundingBoxModel(BaseModel):
    """边界框。"""

    x: int
    y: int
    width: int
    height: int


class MatchItemModel(BaseModel):
    """单个匹配结果。"""

    confidence: float
    bbox: BoundingBoxModel
    center: PointModel


# ==================== OCR 接口 ====================

class OCRRequest(BaseModel):
    """OCR 识别请求。"""

    image: str = Field(..., description="Base64 编码的图像数据")
    lang: str = Field(default="ch", description="语言代码，默认中文")
    filter_text: Optional[str] = Field(default=None, description="过滤关键词，只返回包含此文字的结果")
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度阈值")

    # 新增参数
    preprocess_mode: PreprocessMode = Field(
        default=PreprocessMode.AUTO,
        description="预处理模式: none/auto/screenshot/mobile/document/low_quality"
    )
    ocr_preset: OCRPreset = Field(
        default=OCRPreset.DEFAULT,
        description="OCR预设配置: default/screenshot/mobile/low_quality"
    )

    # 高级用户自定义参数 (PaddleOCR 3.3 兼容)
    text_det_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="文本检测阈值")
    text_det_box_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="文本框置信度阈值")
    text_det_unclip_ratio: Optional[float] = Field(default=None, ge=1.0, le=3.0, description="文本框扩展比例")
    text_rec_score_thresh: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="低置信度过滤阈值")


class OCRResponse(BaseModel):
    """OCR 识别响应。"""

    status: str
    texts: list[TextBlockModel] = []
    coords: list[PointModel] = []  # 简洁的坐标数组 [(x1,y1),(x2,y2),...]
    duration_ms: int = 0
    error: Optional[str] = None


class OCRTextRequest(BaseModel):
    """OCR 纯文本请求。"""

    image: str = Field(..., description="Base64 编码的图像数据")
    lang: str = Field(default="ch", description="语言代码，默认中文")
    separator: str = Field(default="\n", description="文本分隔符")
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="置信度阈值")

    # 新增参数
    preprocess_mode: PreprocessMode = Field(
        default=PreprocessMode.AUTO,
        description="预处理模式: none/auto/screenshot/mobile/document/low_quality"
    )
    ocr_preset: OCRPreset = Field(
        default=OCRPreset.DEFAULT,
        description="OCR预设配置: default/screenshot/mobile/low_quality"
    )


class OCRTextResponse(BaseModel):
    """OCR 纯文本响应。"""

    status: str
    text: str = ""
    duration_ms: int = 0
    error: Optional[str] = None


# ==================== 图像匹配接口 ====================

class ImageMatchRequest(BaseModel):
    """图像匹配请求。"""

    source_image: str = Field(..., description="源图像（大图）Base64 编码")
    template_image: str = Field(..., description="模板图像（小图）Base64 编码")
    confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="匹配阈值"
    )
    method: str = Field(
        default="template", description="匹配方法: template(精确) / feature(特征)"
    )
    multi_target: bool = Field(default=True, description="是否多目标匹配")


class ImageMatchResponse(BaseModel):
    """图像匹配响应。"""

    status: str
    matches: list[MatchItemModel] = []
    coords: list[PointModel] = []  # 简洁的坐标数组
    duration_ms: int = 0
    error: Optional[str] = None


class TextNearImageRequest(BaseModel):
    """文本附近图片匹配请求。"""

    source_image: str = Field(..., description="源图像（大图）Base64 编码")
    template_image: str = Field(..., description="模板图像（小图）Base64 编码")
    text: str = Field(..., description="目标文字")
    max_distance: int = Field(default=200, ge=0, description="最大搜索距离（像素）")
    confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="匹配阈值"
    )
    method: str = Field(
        default="template", description="匹配方法: template(精确) / feature(特征)"
    )


class TextNearImageResponse(BaseModel):
    """文本附近图片匹配响应。"""

    status: str
    text_position: Optional[PointModel] = Field(default=None, description="文字位置")
    match: Optional[MatchItemModel] = Field(default=None, description="最近的匹配图片")
    coords: list[PointModel] = []  # 简洁的坐标数组
    distance: Optional[int] = Field(default=None, description="距离（像素）")
    duration_ms: int = 0
    error: Optional[str] = None


# ==================== 健康检查 ====================

class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    version: str