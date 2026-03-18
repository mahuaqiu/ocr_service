"""
OCR 结果模型。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class Point:
    """坐标点。"""

    x: int
    y: int

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}


@dataclass
class TextBlock:
    """识别到的文字块。"""

    text: str
    confidence: float
    bbox: list[list[int]]  # 四角坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    center: Point

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "center": self.center.to_dict(),
        }


@dataclass
class OCRResult:
    """OCR 识别结果。"""

    status: str
    texts: list[TextBlock] = field(default_factory=list)
    duration_ms: int = 0
    error: Optional[str] = None

    @staticmethod
    def parse_from_paddleocr(
        ocr_result: Any,
        confidence_threshold: float = 0.0,
        scale: float = 1.0,
    ) -> List[TextBlock]:
        """解析 PaddleOCR 3.x 结果为 TextBlock 列表。

        Args:
            ocr_result: PaddleOCR predict() 返回的结果
            confidence_threshold: 置信度阈值
            scale: 图像缩放比例，用于还原坐标

        Returns:
            List[TextBlock]: 识别到的文字块列表
        """
        texts = []
        if not ocr_result:
            return texts

        for result_item in ocr_result:
            # PaddleOCR 3.x 返回 Result 对象或字典
            rec_texts = result_item.rec_texts if hasattr(result_item, 'rec_texts') else result_item.get('rec_texts', [])
            rec_scores = result_item.rec_scores if hasattr(result_item, 'rec_scores') else result_item.get('rec_scores', [])
            rec_boxes = result_item.rec_boxes if hasattr(result_item, 'rec_boxes') else result_item.get('rec_boxes', [])

            for i, text in enumerate(rec_texts):
                if not text:
                    continue
                confidence = rec_scores[i] if i < len(rec_scores) else 0.0

                if confidence < confidence_threshold:
                    continue

                # 将矩形框 [x_min, y_min, x_max, y_max] 转换为 4 点坐标
                if i < len(rec_boxes):
                    box = rec_boxes[i]
                    bbox = [
                        [int(box[0]), int(box[1])],  # 左上
                        [int(box[2]), int(box[1])],  # 右上
                        [int(box[2]), int(box[3])],  # 右下
                        [int(box[0]), int(box[3])],  # 左下
                    ]
                else:
                    bbox = [[0, 0], [100, 0], [100, 50], [0, 50]]

                # 计算中心点
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                center_x = sum(x_coords) // 4
                center_y = sum(y_coords) // 4

                # 如果图像被缩放，还原坐标
                if scale < 1.0:
                    center_x = int(center_x / scale)
                    center_y = int(center_y / scale)
                    bbox = [[int(p[0] / scale), int(p[1] / scale)] for p in bbox]

                text_block = TextBlock(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    center=Point(x=center_x, y=center_y),
                )
                texts.append(text_block)

        return texts

    def to_dict(self) -> dict:
        result = {
            "status": self.status,
            "texts": [t.to_dict() for t in self.texts],
            "duration_ms": self.duration_ms,
        }
        if self.error:
            result["error"] = self.error
        return result