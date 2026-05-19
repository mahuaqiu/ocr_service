# OCR 接口添加 ocr_info 字段设计

## 背景

当前 OCR 服务有多个接口返回识别结果，但部分接口缺少完整的 OCR 识别信息（文字内容+坐标）。
用户需要在所有 OCR 相关接口中统一添加 `ocr_info` 字段，包含识别内容和中心点坐标，不返回置信度。

## 涉及接口

| 接口 | 当前返回 | 添加字段后 |
|------|---------|-----------|
| `/ocr/get_ocr_infos` | texts (含text、confidence、bbox、center) + coords | 新增 ocr_info |
| `/ocr/get_coord_by_text` | texts + coords | 新增 ocr_info |
| `/ocr/get_ocr_texts` | 只有 text (纯文本字符串) | 新增 ocr_info |
| `/image/match_near_text` | 只有 text_position 中心点 | 新增 ocr_info |

## 设计方案

### 新增模型

**文件**: `ocr_service/api/schemas.py`

新增 `OCRInfoItem` 模型，用于表示简化的 OCR 识别信息（不含置信度）：

```python
class OCRInfoItem(BaseModel):
    """OCR 识别信息项（简化版，不含置信度）。"""
    text: str = Field(description="识别的文字内容")
    center: PointModel = Field(description="文字中心点坐标")
```

### 修改响应模型

**文件**: `ocr_service/api/schemas.py`

在三个响应模型中添加 `ocr_info` 字段：

1. **OCRResponse**
   ```python
   class OCRResponse(BaseModel):
       status: str
       texts: list[TextBlockModel] = []
       coords: list[PointModel] = []
       ocr_info: list[OCRInfoItem] = []  # 新增
       duration_ms: int = 0
       error: Optional[str] = None
   ```

2. **OCRTextResponse**
   ```python
   class OCRTextResponse(BaseModel):
       status: str
       text: str = ""
       ocr_info: list[OCRInfoItem] = []  # 新增
       duration_ms: int = 0
       error: Optional[str] = None
   ```

3. **TextNearImageResponse**
   ```python
   class TextNearImageResponse(BaseModel):
       status: str
       text_position: Optional[PointModel] = None
       match: Optional[MatchItemModel] = None
       coords: list[PointModel] = []
       ocr_info: list[OCRInfoItem] = []  # 新增
       distance: Optional[int] = None
       duration_ms: int = 0
       error: Optional[str] = None
   ```

### 修改路由处理

**文件**: `ocr_service/api/routes.py`

在每个接口返回时，从 OCR 结果构建 `ocr_info` 字段：

**通用构建逻辑**:
```python
ocr_info = [
    OCRInfoItem(text=t.text, center=PointModel(x=t.center.x, y=t.center.y))
    for t in texts  # 或 result.texts / text_blocks
]
```

**各接口修改点**:

1. `/ocr/get_ocr_infos` (第130-147行)
   - 返回时添加 `ocr_info=ocr_info`

2. `/ocr/get_coord_by_text` (第184-236行)
   - 成功匹配时添加 `ocr_info=ocr_info`
   - 空结果时添加 `ocr_info=[]`

3. `/ocr/get_ocr_texts` (第239-276行)
   - 成功时添加 `ocr_info=ocr_info`
   - 失败时添加 `ocr_info=[]`

4. `/image/match_near_text` (第327-433行)
   - 各返回分支添加 `ocr_info=ocr_info` 或 `ocr_info=[]`

## 数据流

```
用户请求 → OCR引擎识别 → TextBlock列表
                         ↓
         构建响应时同时生成:
         - texts (含confidence、bbox) ← 保持不变
         - coords (只有中心点坐标) ← 保持不变
         - ocr_info (文字+中心点) ← 新增
```

## 向后兼容性

- 现有字段（texts、coords、text 等）保持不变
- 新增字段 `ocr_info` 默认值为空列表
- 现有调用方无需修改代码即可继续使用

## 测试要点

1. 各接口返回结构包含 `ocr_info` 字段
2. `ocr_info` 内容与 OCR 结果一致
3. 空结果时 `ocr_info` 为空列表
4. 现有字段不受影响，向后兼容