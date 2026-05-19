# OCR 接口添加 ocr_info 字段实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在所有 OCR 相关接口的返回中添加 ocr_info 字段，包含识别内容和中心点坐标（不含置信度）。

**Architecture:** 新增 OCRInfoItem 模型，修改三个响应模型，在各接口返回时构建 ocr_info 字段。

**Tech Stack:** Python, FastAPI, Pydantic

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `ocr_service/api/schemas.py` | 修改 | 新增 OCRInfoItem 模型，修改响应模型 |
| `ocr_service/api/routes.py` | 修改 | 添加导入，修改各接口返回 |

---

### Task 1: 新增 OCRInfoItem 模型

**文件:**
- Modify: `ocr_service/api/schemas.py:40-46`（在 TextBlockModel 之后）

- [ ] **Step 1: 在 schemas.py 中添加 OCRInfoItem 模型**

在 TextBlockModel 类定义之后（约第46行），添加：

```python
class OCRInfoItem(BaseModel):
    """OCR 识别信息项（简化版，不含置信度）。"""

    text: str = Field(description="识别的文字内容")
    center: PointModel = Field(description="文字中心点坐标")
```

- [ ] **Step 2: 提交代码**

```bash
git add ocr_service/api/schemas.py
git commit -m "feat: 新增 OCRInfoItem 模型"
```

---

### Task 2: 修改响应模型添加 ocr_info 字段

**文件:**
- Modify: `ocr_service/api/schemas.py`

- [ ] **Step 1: 在 OCRResponse 中添加 ocr_info 字段**

修改 `OCRResponse` 类（约第93-100行），在 `coords` 字段后添加：

```python
class OCRResponse(BaseModel):
    """OCR 识别响应。"""

    status: str
    texts: list[TextBlockModel] = []
    coords: list[PointModel] = []  # 简洁的坐标数组 [(x1,y1),(x2,y2),...]
    ocr_info: list[OCRInfoItem] = []  # OCR识别信息（文字+坐标，不含置信度）
    duration_ms: int = 0
    error: Optional[str] = None
```

- [ ] **Step 2: 在 OCRTextResponse 中添加 ocr_info 字段**

修改 `OCRTextResponse` 类（约第122-128行），在 `text` 字段后添加：

```python
class OCRTextResponse(BaseModel):
    """OCR 纯文本响应。"""

    status: str
    text: str = ""
    ocr_info: list[OCRInfoItem] = []  # OCR识别信息（文字+坐标，不含置信度）
    duration_ms: int = 0
    error: Optional[str] = None
```

- [ ] **Step 3: 在 TextNearImageResponse 中添加 ocr_info 字段**

修改 `TextNearImageResponse` 类（约第172-181行），在 `coords` 字段后添加：

```python
class TextNearImageResponse(BaseModel):
    """文本附近图片匹配响应。"""

    status: str
    text_position: Optional[PointModel] = Field(default=None, description="文字位置")
    match: Optional[MatchItemModel] = Field(default=None, description="最近的匹配图片")
    coords: list[PointModel] = []  # 简洁的坐标数组
    ocr_info: list[OCRInfoItem] = []  # OCR识别信息（最多一个元素）
    distance: Optional[int] = Field(default=None, description="距离（像素）")
    duration_ms: int = 0
    error: Optional[str] = None
```

- [ ] **Step 4: 提交代码**

```bash
git add ocr_service/api/schemas.py
git commit -m "feat: 响应模型添加 ocr_info 字段"
```

---

### Task 3: 修改 routes.py 导入

**文件:**
- Modify: `ocr_service/api/routes.py:11-25`

- [ ] **Step 1: 添加 OCRInfoItem 导入**

在导入区域添加 `OCRInfoItem`：

```python
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
    OCRInfoItem,  # 新增
)
```

- [ ] **Step 2: 提交代码**

```bash
git add ocr_service/api/routes.py
git commit -m "feat: routes.py 添加 OCRInfoItem 导入"
```

---

### Task 4: 修改 /ocr/get_ocr_infos 接口

**文件:**
- Modify: `ocr_service/api/routes.py:87-147`

- [ ] **Step 1: 修改接口返回，添加 ocr_info 字段**

修改返回部分（约第130-147行），添加 ocr_info 构建：

```python
    # 构建 ocr_info（不含置信度）
    ocr_info = [
        OCRInfoItem(text=t.text, center=PointModel(x=t.center.x, y=t.center.y))
        for t in texts
    ]

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
        ocr_info=ocr_info,
        duration_ms=result.duration_ms,
        error=result.error,
    )
```

- [ ] **Step 2: 提交代码**

```bash
git add ocr_service/api/routes.py
git commit -m "feat: /ocr/get_ocr_infos 接口添加 ocr_info 字段"
```

---

### Task 5: 修改 /ocr/get_coord_by_text 接口

**文件:**
- Modify: `ocr_service/api/routes.py:150-236`

- [ ] **Step 1: 修改正则模式 OCR 失败时的返回**

修改第184-189行的返回，添加 `ocr_info=[]`：

```python
        if result.status != "success" or not result.texts:
            return OCRResponse(
                status="success",
                texts=[],
                coords=[],
                ocr_info=[],
                duration_ms=result.duration_ms,
            )
```

- [ ] **Step 2: 修改普通模式空结果时的返回**

修改第213-218行的返回，添加 `ocr_info=[]`：

```python
    if not text_blocks:
        return OCRResponse(
            status="success",
            texts=[],
            coords=[],
            ocr_info=[],
            duration_ms=duration_ms,
        )
```

- [ ] **Step 3: 修改成功匹配时的返回**

修改第220-235行的返回，添加 ocr_info 构建：

```python
    # 构建 ocr_info（不含置信度）
    ocr_info = [
        OCRInfoItem(text=tb.text, center=PointModel(x=tb.center.x, y=tb.center.y))
        for tb in text_blocks
    ]

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
        ocr_info=ocr_info,
        duration_ms=duration_ms,
    )
```

- [ ] **Step 4: 提交代码**

```bash
git add ocr_service/api/routes.py
git commit -m "feat: /ocr/get_coord_by_text 接口添加 ocr_info 字段"
```

---

### Task 6: 修改 /ocr/get_ocr_texts 接口

**文件:**
- Modify: `ocr_service/api/routes.py:239-276`

- [ ] **Step 1: 修改 OCR 失败时的返回**

修改第262-267行的返回，添加 `ocr_info=[]`：

```python
    if result.status != "success":
        return OCRTextResponse(
            status=result.status,
            text="",
            ocr_info=[],
            duration_ms=result.duration_ms,
            error=result.error,
        )
```

- [ ] **Step 2: 修改成功时的返回**

修改第269-276行，添加 ocr_info 构建：

```python
    # 拼接文本
    text = request.separator.join(t.text for t in result.texts)

    # 构建 ocr_info（不含置信度）
    ocr_info = [
        OCRInfoItem(text=t.text, center=PointModel(x=t.center.x, y=t.center.y))
        for t in result.texts
    ]

    return OCRTextResponse(
        status="success",
        text=text,
        ocr_info=ocr_info,
        duration_ms=result.duration_ms,
    )
```

- [ ] **Step 3: 提交代码**

```bash
git add ocr_service/api/routes.py
git commit -m "feat: /ocr/get_ocr_texts 接口添加 ocr_info 字段"
```

---

### Task 7: 修改 /image/match_near_text 接口

**文件:**
- Modify: `ocr_service/api/routes.py:327-433`

**注意:** 该接口的 ocr_info 最多包含一个元素（匹配到的文字块）。

- [ ] **Step 1: 修改文字未找到时的返回**

修改第362-369行的返回，添加 `ocr_info=[]`：

```python
    if text_block is None:
        return TextNearImageResponse(
            status="success",
            text_position=None,
            match=None,
            coords=[],
            ocr_info=[],
            distance=None,
            duration_ms=int((time.time() - start_time) * 1000),
        )
```

- [ ] **Step 2: 修改图片未匹配时的返回**

修改第384-391行的返回，添加 ocr_info：

```python
    if not match_result.matches:
        return TextNearImageResponse(
            status="success",
            text_position=PointModel(x=text_center.x, y=text_center.y),
            match=None,
            coords=[],
            ocr_info=[
                OCRInfoItem(text=text_block.text, center=PointModel(x=text_center.x, y=text_center.y))
            ],
            distance=None,
            duration_ms=int((time.time() - start_time) * 1000),
        )
```

- [ ] **Step 3: 修改超出距离时的返回**

修改第404-411行的返回，添加 ocr_info：

```python
    if not valid_matches:
        return TextNearImageResponse(
            status="success",
            text_position=PointModel(x=text_center.x, y=text_center.y),
            match=None,
            coords=[],
            ocr_info=[
                OCRInfoItem(text=text_block.text, center=PointModel(x=text_center.x, y=text_center.y))
            ],
            distance=None,
            duration_ms=int((time.time() - start_time) * 1000),
        )
```

- [ ] **Step 4: 修改成功匹配时的返回**

修改第417-433行的返回，添加 ocr_info：

```python
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
        ocr_info=[
            OCRInfoItem(text=text_block.text, center=PointModel(x=text_center.x, y=text_center.y))
        ],
        distance=distance,
        duration_ms=int((time.time() - start_time) * 1000),
    )
```

- [ ] **Step 5: 提交代码**

```bash
git add ocr_service/api/routes.py
git commit -m "feat: /image/match_near_text 接口添加 ocr_info 字段"
```

---

### Task 8: 验证测试

**文件:**
- 无文件修改，仅验证

- [ ] **Step 1: 启动服务**

```bash
python ocr_service/server.py
```

预期：服务正常启动，无报错

- [ ] **Step 2: 验证接口返回结构**

使用 curl 或 Postman 调用各接口，确认返回包含 `ocr_info` 字段：

1. `/ocr/get_ocr_infos` - ocr_info 应包含所有识别文字
2. `/ocr/get_coord_by_text` - ocr_info 应包含匹配的文字
3. `/ocr/get_ocr_texts` - ocr_info 应包含所有识别文字
4. `/image/match_near_text` - ocr_info 最多包含一个元素

- [ ] **Step 3: 验证数据一致性**

确认 `ocr_info` 的 text 和 center 与 `texts` 字段对应

- [ ] **Step 4: 最终提交**

```bash
git add -A
git commit -m "feat: 完成 OCR 接口 ocr_info 字段添加"
```

---

## 测试要点

1. 各接口返回结构包含 `ocr_info` 字段
2. `ocr_info` 内容与 OCR 结果一致
3. 空结果时 `ocr_info` 为空列表
4. 现有字段不受影响，向后兼容
5. `ocr_info` 与 `texts` 数据一致性
6. `/image/match_near_text` 的 `ocr_info` 最多一个元素