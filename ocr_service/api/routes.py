"""
API 路由定义。
"""

import math

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
async def ocr_recognize(request: OCRRequest):
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
        texts = [t for t in texts if request.filter_text in t.text]

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
        duration_ms=result.duration_ms,
        error=result.error,
    )


@router.post("/ocr/get_coord_by_text", response_model=OCRResponse, tags=["OCR"])
async def ocr_get_coord_by_text(request: OCRRequest):
    """
    查找指定文字。

    在图片中查找指定的文字，返回匹配的文字和坐标。

    Args:
        request: OCR 请求，filter_text 为必填字段。

    Returns:
        OCRResponse: 匹配结果。
    """
    if not request.filter_text:
        raise HTTPException(status_code=400, detail="filter_text is required")

    engine = get_ocr_engine()
    text_block = engine.find_text(
        image_data=request.image,
        target_text=request.filter_text,
        confidence_threshold=request.confidence_threshold,
        prefer_exact=True,
        preprocess_mode=request.preprocess_mode.value,
        ocr_preset=request.ocr_preset.value,
    )

    if text_block is None:
        return OCRResponse(
            status="success",
            texts=[],
            duration_ms=0,
        )

    return OCRResponse(
        status="success",
        texts=[
            TextBlockModel(
                text=text_block.text,
                confidence=text_block.confidence,
                bbox=text_block.bbox,
                center=PointModel(x=text_block.center.x, y=text_block.center.y),
            )
        ],
        duration_ms=0,
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

    # 按阅读顺序排序（从上到下，从左到右）
    sorted_texts = sorted(result.texts, key=lambda t: (t.center.y, t.center.x))

    # 拼接文本
    text = request.separator.join(t.text for t in sorted_texts)

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

    # 1. 查找文字位置（精确匹配优先）
    engine = get_ocr_engine()
    text_block = engine.find_text(
        image_data=request.source_image,
        target_text=request.text,
        prefer_exact=True,
    )

    if text_block is None:
        return TextNearImageResponse(
            status="success",
            text_position=None,
            match=None,
            distance=None,
            duration_ms=int((time.time() - start_time) * 1000),
        )

    text_center = text_block.center

    # 2. 查找所有模板图片位置
    matcher = get_image_matcher()
    match_result = matcher.match(
        source_data=request.source_image,
        template_data=request.template_image,
        threshold=request.confidence_threshold,
        method=request.method,
        multi_target=True,
    )

    if not match_result.matches:
        return TextNearImageResponse(
            status="success",
            text_position=PointModel(x=text_center.x, y=text_center.y),
            match=None,
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
        distance=distance,
        duration_ms=int((time.time() - start_time) * 1000),
    )