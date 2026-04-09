"""
API 路由定义。
"""

import math
import re

from fastapi import APIRouter, HTTPException

from ocr_service import __version__
from ocr_service.api.schemas import (
    HealthResponse,
    ImageMatchRequest,
    ImageMatchResponse,
    OCRRequest,
    OCRResponse,
    OCRTextRequest,
    OCRTextResponse,
    TextNearImageRequest,
    TextNearImageResponse,
    TextBlockModel,
    PointModel,
    BoundingBoxModel,
    MatchItemModel,
)
from ocr_service.core.ocr_engine import get_ocr_engine
from ocr_service.core.image_matcher import get_image_matcher

router = APIRouter()


def parse_filter_text(filter_text: str) -> tuple[str, str]:
    """
    解析 filter_text，支持正则表达式格式。

    Args:
        filter_text: 过滤文本，支持 "reg_xxx" 格式表示正则表达式

    Returns:
        (match_mode, pattern): match_mode 为 "regex" 或 "exact"，pattern 为匹配模式
    """
    if filter_text.startswith("reg_"):
        return ("regex", filter_text[4:])  # 去掉 "reg_" 前缀
    return ("exact", filter_text)


def filter_texts(texts: list, filter_text: str) -> list:
    """
    根据 filter_text 过滤文字块。

    Args:
        texts: 文字块列表
        filter_text: 过滤文本，支持 "reg_xxx" 格式表示正则表达式

    Returns:
        过滤后的文字块列表
    """
    match_mode, pattern = parse_filter_text(filter_text)

    if match_mode == "regex":
        try:
            regex = re.compile(pattern)
            return [t for t in texts if regex.search(t.text)]
        except re.error:
            # 正则表达式无效，回退到包含匹配
            return [t for t in texts if pattern in t.text]
    else:
        return [t for t in texts if pattern in t.text]


# ==================== OCR 接口 ====================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    健康检查。

    Returns:
        HealthResponse: 服务状态。
    """
    return HealthResponse(status="healthy", version=__version__)


@router.post("/ocr/get_ocr_infos", response_model=OCRResponse, tags=["OCR"])
async def get_ocr_infos(request: OCRRequest):
    """
    文字识别。

    识别图片中的所有文字，返回文字内容和坐标。

    Args:
        request: OCR 请求，包含 Base64 编码的图像。

    Returns:
        OCRResponse: 识别结果，包含文字列表和坐标。
    """
    engine = get_ocr_engine()

    # 构建自定义 OCR 参数 (PaddleOCR 3.3 兼容)
    custom_params = None
    if any([request.text_det_thresh, request.text_det_box_thresh,
            request.text_det_unclip_ratio, request.text_rec_score_thresh]):
        custom_params = {}
        if request.text_det_thresh is not None:
            custom_params["text_det_thresh"] = request.text_det_thresh
        if request.text_det_box_thresh is not None:
            custom_params["text_det_box_thresh"] = request.text_det_box_thresh
        if request.text_det_unclip_ratio is not None:
            custom_params["text_det_unclip_ratio"] = request.text_det_unclip_ratio
        if request.text_rec_score_thresh is not None:
            custom_params["text_rec_score_thresh"] = request.text_rec_score_thresh

    result = engine.recognize(
        image_data=request.image,
        lang=request.lang,
        confidence_threshold=request.confidence_threshold,
        preprocess_mode=request.preprocess_mode.value,
        ocr_preset=request.ocr_preset.value,
        custom_ocr_params=custom_params,
    )

    # 过滤文字
    texts = result.texts
    if request.filter_text:
        texts = filter_texts(texts, request.filter_text)

    return OCRResponse(
        status=result.status,
        texts=[
            TextBlockModel(
                text=t.text,
                confidence=t.confidence,
                bbox=t.bbox,
                center=PointModel(x=t.center.x, y=t.center.y),
            )
            for t in texts
        ],
        coords=[
            PointModel(x=t.center.x, y=t.center.y)
            for t in texts
        ],
        duration_ms=result.duration_ms,
        error=result.error,
    )


@router.post("/ocr/get_coord_by_text", response_model=OCRResponse, tags=["OCR"])
async def ocr_get_coord_by_text(request: OCRRequest):
    """
    查找指定文字。

    在图片中查找指定的文字，返回匹配的文字和坐标。
    返回所有匹配结果，texts 按顺序 [1,2,...]，coords 为对应的中心点坐标 [(x1,y1),(x2,y2),...]

    filter_text 支持格式：
    - 普通文本：直接进行包含匹配
    - 正则表达式：以 "reg_" 开头，如 "reg_雨[大中小]" 表示匹配 "雨大"、"雨中"、"雨小"

    Args:
        request: OCR 请求，filter_text 为必填字段。

    Returns:
        OCRResponse: 匹配结果，包含 texts 和 coords 字段。
    """
    if not request.filter_text:
        raise HTTPException(status_code=400, detail="filter_text is required")

    engine = get_ocr_engine()
    match_mode, pattern = parse_filter_text(request.filter_text)

    # 使用正则匹配模式
    if match_mode == "regex":
        # 先识别所有文字，再进行正则过滤
        result = engine.recognize(
            image_data=request.image,
            confidence_threshold=request.confidence_threshold,
            preprocess_mode=request.preprocess_mode.value,
            ocr_preset=request.ocr_preset.value,
        )
        if result.status != "success" or not result.texts:
            return OCRResponse(
                status="success",
                texts=[],
                coords=[],
                duration_ms=result.duration_ms,
            )

        # 正则过滤
        try:
            regex = re.compile(pattern)
            text_blocks = [t for t in result.texts if regex.search(t.text)]
        except re.error:
            # 正则表达式无效，回退到包含匹配
            text_blocks = [t for t in result.texts if pattern in t.text]

        duration_ms = result.duration_ms
    else:
        # 普通模式：使用 find_all_texts
        text_blocks, duration_ms = engine.find_all_texts(
            image_data=request.image,
            target_text=pattern,
            confidence_threshold=request.confidence_threshold,
            prefer_exact=True,
            preprocess_mode=request.preprocess_mode.value,
            ocr_preset=request.ocr_preset.value,
        )

    if not text_blocks:
        return OCRResponse(
            status="success",
            texts=[],
            coords=[],
            duration_ms=duration_ms,
        )

    return OCRResponse(
        status="success",
        texts=[
            TextBlockModel(
                text=tb.text,
                confidence=tb.confidence,
                bbox=tb.bbox,
                center=PointModel(x=tb.center.x, y=tb.center.y),
            )
            for tb in text_blocks
        ],
        coords=[
            PointModel(x=tb.center.x, y=tb.center.y)
            for tb in text_blocks
        ],
        duration_ms=duration_ms,
    )


@router.post("/ocr/get_ocr_texts", response_model=OCRTextResponse, tags=["OCR"])
async def ocr_text(request: OCRTextRequest):
    """
    获取图片中的所有文本。

    识别图片中的所有文字，返回拼接后的纯文本字符串。

    Args:
        request: OCR 文本请求，包含 Base64 编码的图像。

    Returns:
        OCRTextResponse: 拼接后的文本字符串。
    """
    engine = get_ocr_engine()
    result = engine.recognize(
        image_data=request.image,
        lang=request.lang,
        confidence_threshold=request.confidence_threshold,
        preprocess_mode=request.preprocess_mode.value,
        ocr_preset=request.ocr_preset.value,
    )

    if result.status != "success":
        return OCRTextResponse(
            status=result.status,
            text="",
            duration_ms=result.duration_ms,
            error=result.error,
        )

    # 拼接文本
    text = request.separator.join(t.text for t in result.texts)

    return OCRTextResponse(
        status="success",
        text=text,
        duration_ms=result.duration_ms,
    )


# ==================== 图像匹配接口 ====================

@router.post("/image/match", response_model=ImageMatchResponse, tags=["Image"])
async def image_match(request: ImageMatchRequest):
    """
    图像匹配。

    在源图像中查找模板图像的位置。

    Args:
        request: 匹配请求。

    Returns:
        ImageMatchResponse: 匹配结果，包含坐标和置信度。
    """
    matcher = get_image_matcher()
    result = matcher.match(
        source_data=request.source_image,
        template_data=request.template_image,
        threshold=request.confidence_threshold,
        method=request.method,
        multi_target=request.multi_target,
    )

    return ImageMatchResponse(
        status=result.status,
        matches=[
            MatchItemModel(
                confidence=m.confidence,
                bbox=BoundingBoxModel(
                    x=m.bbox.x,
                    y=m.bbox.y,
                    width=m.bbox.width,
                    height=m.bbox.height,
                ),
                center=PointModel(x=m.center.x, y=m.center.y),
            )
            for m in result.matches
        ],
        coords=[
            PointModel(x=m.center.x, y=m.center.y)
            for m in result.matches
        ],
        duration_ms=result.duration_ms,
        error=result.error,
    )


@router.post("/image/match_near_text", response_model=TextNearImageResponse, tags=["Image"])
async def image_match_near_text(request: TextNearImageRequest):
    """
    查找文本附近最近的图片。

    在源图像中查找目标文字位置，然后查找距离文字最近的模板图像位置。

    Args:
        request: 匹配请求。

    Returns:
        TextNearImageResponse: 匹配结果，包含文字位置、图片位置和距离。
    """
    import time
    start_time = time.time()

    # 1. 查找文字位置
    # 判断是否为正则表达式（以 reg_ 开头）
    filter_text = request.filter_text
    if filter_text.startswith("reg_"):
        match_mode = "regex"
        target_text = filter_text[4:]  # 去掉 reg_ 前缀
    else:
        match_mode = "exact"
        target_text = filter_text

    engine = get_ocr_engine()
    text_block = engine.find_text(
        image_data=request.image,
        target_text=target_text,
        match_mode=match_mode,
        prefer_exact=(match_mode == "exact"),
    )

    if text_block is None:
        return TextNearImageResponse(
            status="success",
            text_position=None,
            match=None,
            coords=[],
            distance=None,
            duration_ms=int((time.time() - start_time) * 1000),
        )

    text_center = text_block.center

    # 2. 查找所有模板图片位置
    matcher = get_image_matcher()
    match_result = matcher.match(
        source_data=request.image,
        template_data=request.target_image,
        threshold=request.confidence_threshold,
        method=request.method,
        multi_target=True,
    )

    if not match_result.matches:
        return TextNearImageResponse(
            status="success",
            text_position=PointModel(x=text_center.x, y=text_center.y),
            match=None,
            coords=[],
            distance=None,
            duration_ms=int((time.time() - start_time) * 1000),
        )

    # 3. 计算每个匹配图片与文字的距离，找到最近的
    def calc_distance(m):
        return math.sqrt((m.center.x - text_center.x) ** 2 + (m.center.y - text_center.y) ** 2)

    # 过滤超出最大距离的匹配，然后按距离排序
    valid_matches = [
        m for m in match_result.matches
        if calc_distance(m) <= request.max_distance
    ]

    if not valid_matches:
        return TextNearImageResponse(
            status="success",
            text_position=PointModel(x=text_center.x, y=text_center.y),
            match=None,
            coords=[],
            distance=None,
            duration_ms=int((time.time() - start_time) * 1000),
        )

    # 找到距离最近的匹配
    nearest_match = min(valid_matches, key=calc_distance)
    distance = int(calc_distance(nearest_match))

    return TextNearImageResponse(
        status="success",
        text_position=PointModel(x=text_center.x, y=text_center.y),
        match=MatchItemModel(
            confidence=nearest_match.confidence,
            bbox=BoundingBoxModel(
                x=nearest_match.bbox.x,
                y=nearest_match.bbox.y,
                width=nearest_match.bbox.width,
                height=nearest_match.bbox.height,
            ),
            center=PointModel(x=nearest_match.center.x, y=nearest_match.center.y),
        ),
        coords=[PointModel(x=nearest_match.center.x, y=nearest_match.center.y)],
        distance=distance,
        duration_ms=int((time.time() - start_time) * 1000),
    )