"""[RESPONSE] 日志字段覆盖测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.server import format_response_log


def test_response_log_includes_diagnostics_and_ocr_info():
    body = {
        "status": "success",
        "texts": [],
        "coords": [],
        "matches": [],
        "ocr_info": [{"text": "a", "center": {"x": 1, "y": 2}},
                     {"text": "b", "center": {"x": 3, "y": 4}}],
        "not_found_reason": "below_threshold",
        "best_confidence": 0.7231,
        "nearest_distance": None,
        "duration_ms": 9,
        "error": None,
    }
    request = type("R", (), {"method": "POST", "url": type("U", (), {"path": "/image/match"})()})()
    out = format_response_log(request, body, 200)

    assert "ocr_info: [2 items]" in out
    assert "not_found_reason: below_threshold" in out
    assert "best_confidence: 0.7231" in out
    assert "nearest_distance" not in out  # None 值不输出
