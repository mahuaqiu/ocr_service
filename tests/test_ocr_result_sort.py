"""ocr_result 排序单元测试：按行重排（行间从上到下，行内从左到右）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.models.ocr_result import (
    OCRResult,
    Point,
    TextBlock,
    sort_by_line,
)


def _block(text: str, x: int, y: int) -> TextBlock:
    """构造中心点为 (x, y) 的文字块。"""
    return TextBlock(
        text=text,
        confidence=0.9,
        bbox=[[x - 10, y - 5], [x + 10, y - 5], [x + 10, y + 5], [x - 10, y + 5]],
        center=Point(x=x, y=y),
    )


def test_sort_same_line_left_to_right():
    blocks = [_block("C", 200, 100), _block("A", 10, 100), _block("B", 100, 100)]
    assert [b.text for b in sort_by_line(blocks)] == ["A", "B", "C"]


def test_sort_lines_top_to_bottom():
    blocks = [_block("下", 50, 140), _block("上", 50, 40)]
    assert [b.text for b in sort_by_line(blocks)] == ["上", "下"]


def test_sort_threshold_30px_against_line_head():
    # 组参照为该行首个（最靠上）块的中心 y：B 与 A 差 25 同行；
    # C 与 A 差 35 超阈值，换行——即使它与 B 只差 10
    a = _block("A", 10, 100)
    b = _block("B", 200, 125)
    c = _block("C", 20, 135)
    assert [t.text for t in sort_by_line([c, b, a])] == ["A", "B", "C"]


def test_sort_two_lines_mixed_input():
    # 第一行: 甲(60,100) 乙(250,100)；第二行: 丙(70,140) 丁(260,140)
    blocks = [_block("乙", 250, 100), _block("丁", 260, 140), _block("丙", 70, 140), _block("甲", 60, 100)]
    assert [b.text for b in sort_by_line(blocks)] == ["甲", "乙", "丙", "丁"]


def test_sort_empty_and_single():
    assert sort_by_line([]) == []
    single = [_block("单", 5, 5)]
    assert [b.text for b in sort_by_line(single)] == ["单"]


def test_parse_from_paddleocr_returns_line_order():
    # 输入乱序：乙(250,100) 丁(260,140) 丙(70,140) 甲(60,100)
    ocr_result = [
        {
            "rec_texts": ["乙", "丁", "丙", "甲"],
            "rec_scores": [0.9, 0.9, 0.9, 0.9],
            "rec_boxes": [
                [200, 90, 300, 110],
                [210, 135, 310, 145],
                [20, 135, 120, 145],
                [10, 95, 110, 105],
            ],
        }
    ]
    result = OCRResult.parse_from_paddleocr(ocr_result)
    assert [t.text for t in result] == ["甲", "乙", "丙", "丁"]
