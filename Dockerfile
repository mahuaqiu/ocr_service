# OCR Service Dockerfile (GPU 版本)
# 基于 PaddleOCR 的文字识别和图像匹配服务
# 使用 CUDA 12.9 + Python 3.12 支持 GPU 加速

FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

LABEL maintainer="OCR Service"
LABEL description="OCR and Image Matching Service (GPU Enabled)"

# 设置工作目录
WORKDIR /service

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/service \
    OCR_HOST=0.0.0.0 \
    OCR_PORT=8081 \
    OCR_LANG=ch \
    OCR_USE_GPU=true \
    TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

# 更换为阿里云国内源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources

# 安装系统依赖
# Ubuntu 24.04 自带 Python 3.12
# 注意: libgl1-mesa-glx 在 Ubuntu 24.04 已废弃，改用 libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 创建 venv 并激活（避免 pip externally-managed-environment 报错）
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制依赖文件
COPY requirements.txt .

# 先安装 paddlepaddle-gpu（使用官方源，阿里云源无此版本）
RUN pip install --no-cache-dir paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 安装其他 Python 依赖（排除 paddlepaddle-gpu，已单独安装）
RUN grep -v "paddlepaddle-gpu" requirements.txt | pip install --no-cache-dir -r /dev/stdin -i https://mirrors.aliyun.com/pypi/simple/

# 复制应用代码
COPY ocr_service ./ocr_service
COPY pyproject.toml .

# 暴露端口
EXPOSE 8081

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# 启动命令
CMD ["python", "-m", "ocr_service.server"]