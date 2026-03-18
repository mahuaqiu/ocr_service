# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## 项目概述

基于 PaddleOCR 的文字识别和 OpenCV 图像匹配 REST API 服务。

## 常用命令

### 运行服务

```bash
# 直接运行（默认端口 8081）
python -m ocr_service.server

# 指定端口
python -m ocr_service.server --port 8081

# 开发模式（自动重载）
python -m ocr_service.server --reload

# 使用 uvicorn
uvicorn ocr_service.server:app --host 0.0.0.0 --port 8081
```

### Docker

```bash
# 构建镜像
docker build -t ocr-service .

# 运行容器
docker run -p 8081:8081 ocr-service
```

### 依赖安装

```bash
pip install -r requirements.txt
```

## 架构

```
ocr_service/
├── server.py          # FastAPI 应用入口，创建 app 实例
├── config.py          # 配置管理，支持环境变量
├── api/
│   ├── routes.py      # API 路由定义
│   └── schemas.py     # Pydantic 请求/响应模型
├── core/
│   ├── ocr_engine.py      # OCR 引擎（PaddleOCR 封装）
│   └── image_matcher.py   # 图像匹配（OpenCV 模板/特征匹配）
├── models/
│   ├── ocr_result.py      # OCR 结果数据类
│   └── match_result.py    # 图像匹配结果数据类
└── utils/
    └── image_utils.py     # 图像编解码工具
```

### 核心组件

- **OCREngine**: 封装 PaddleOCR，延迟加载，支持中英文等多语言识别。通过 `get_ocr_engine()` 获取单例实例。

- **ImageMatcher**: 封装 OpenCV，支持两种匹配模式：
  - `template`: 精确模板匹配（`cv2.matchTemplate`）
  - `feature`: 特征匹配（SIFT），支持缩放/旋转场景

### 配置

通过环境变量配置，主要变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| OCR_HOST | 0.0.0.0 | 监听地址 |
| OCR_PORT | 8081 | 监听端口 |
| OCR_LANG | ch | OCR 语言 |
| OCR_USE_GPU | false | 是否使用 GPU |
| OCR_MATCH_THRESHOLD | 0.8 | 默认匹配阈值 |
| OCR_MATCH_METHOD | template | 默认匹配方法 |

## API 端点

- `GET /health` - 健康检查
- `POST /ocr/get_ocr_infos` - 文字识别，返回所有文字及坐标
- `POST /ocr/get_coord_by_text` - 查找指定文字（filter_text 必填）
- `POST /ocr/get_ocr_texts` - 获取图片中的所有文本（返回拼接后的纯文本）
- `POST /image/match` - 图像匹配，在源图像中查找模板位置
- `POST /image/match_near_text` - 文本附近图片匹配，先查找源图像中目标文字位置，再查找距离文字最近的模板图像位置

所有图像输入均为 Base64 编码字符串。

## 开发注意事项

- API 文档访问：`/docs`（Swagger UI）或 `/redoc`
- 核心引擎使用全局单例，首次调用时初始化
- PaddleOCR 模型会自动下载到 `cache/` 目录