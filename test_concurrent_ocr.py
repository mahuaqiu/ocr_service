"""
OCR服务并发测试脚本。
使用多线程同时发送15个请求到OCR服务。
"""

import base64
import time
import sys
import threading
from pathlib import Path
import requests

# 设置标准输出编码为UTF-8（解决Windows中文显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def send_ocr_request(image_base64: str, request_id: int):
    """
    发送单个OCR请求（线程函数）。

    Args:
        image_base64: Base64编码的图片
        request_id: 请求编号
    """
    url = "http://localhost:9021/ocr/get_ocr_infos"

    payload = {
        "image": image_base64,
        "lang": "ch",
        "confidence_threshold": 0.0,
    }

    # 记录请求发送时间（包含毫秒）
    request_time_full = time.time()
    request_time_str = time.strftime("%H:%M:%S", time.localtime(request_time_full))
    request_ms = int((request_time_full - int(request_time_full)) * 1000)
    request_time = f"{request_time_str}.{request_ms:03d}"

    try:
        response = requests.post(url, json=payload, timeout=30)
        result = response.json()

        # 记录响应返回时间（包含毫秒）
        response_time_full = time.time()
        response_time_str = time.strftime("%H:%M:%S", time.localtime(response_time_full))
        response_ms = int((response_time_full - int(response_time_full)) * 1000)
        response_time = f"{response_time_str}.{response_ms:03d}"

        print(f"请求 {request_id}: 发送时间 {request_time} | 返回时间 {response_time}")
    except Exception as e:
        response_time_full = time.time()
        response_time_str = time.strftime("%H:%M:%S", time.localtime(response_time_full))
        response_ms = int((response_time_full - int(response_time_full)) * 1000)
        response_time = f"{response_time_str}.{response_ms:03d}"
        print(f"请求 {request_id}: 发送时间 {request_time} | 返回时间 {response_time} | 错误: {str(e)}")


def concurrent_ocr_requests(image_path: str, num_requests: int = 15):
    """
    使用多线程发送多个并发OCR请求。

    Args:
        image_path: 图片路径
        num_requests: 并发请求数量
    """
    # 读取图片并转换为Base64
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"错误：图片文件不存在 - {image_path}")
        return

    with open(image_file, "rb") as f:
        image_data = f.read()

    image_base64 = base64.b64encode(image_data).decode("utf-8")

    print(f"开始发送 {num_requests} 个并发请求...")

    # 创建线程列表
    threads = []
    for i in range(num_requests):
        thread = threading.Thread(
            target=send_ocr_request,
            args=(image_base64, i + 1)
        )
        threads.append(thread)

    # 同时启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def main():
    """主函数"""
    # 图片路径
    image_path = r"C:\Users\Administrator\Pictures\微信图片_20220221223112.jpg"

    # 执行并发请求（使用多线程）
    concurrent_ocr_requests(image_path, num_requests=15)


if __name__ == "__main__":
    main()