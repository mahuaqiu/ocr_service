"""decode_image 失败原因细分测试。"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.utils.image_utils import decode_image


def _tiny_png_b64() -> str:
    img = np.ones((8, 8, 3), dtype=np.uint8) * 255
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def test_empty_string_raises_empty_error():
    with pytest.raises(ValueError, match="图片数据为空"):
        decode_image("")


def test_data_uri_prefix_is_stripped():
    """yaml 声明支持 data:image/...;base64, 前缀，必须能解码。"""
    prefixed = "data:image/png;base64," + _tiny_png_b64()
    image = decode_image(prefixed)
    assert image.shape[:2] == (8, 8)


def test_invalid_base64_message_contains_length():
    with pytest.raises(ValueError, match="Base64 解码失败"):
        decode_image("!!!not-base64!!!")


def test_corrupt_bytes_message_contains_length():
    with pytest.raises(ValueError, match="图片解码失败"):
        decode_image(b"\x00\x01\x02garbage-not-an-image")


def test_valid_bytes_still_works():
    image = decode_image(base64.b64decode(_tiny_png_b64()))
    assert image.shape[:2] == (8, 8)
