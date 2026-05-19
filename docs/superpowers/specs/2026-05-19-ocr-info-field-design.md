# OCR 接口添加 ocr_info 字段设计

## 背景

当前 OCR 服务有多个接口返回识别结果，但部分接口缺少完整的 OCR 识别信息（文字内容+坐标）。
用户需要在所有 OCR 相关接口中统一添加 `ocr_info` 字段，包含识别内容和中心点坐标，不返回置信度。

## 涉及接口

| 接口 | 当前返回 | 添加字段后 |
|------|---------|-----------|
| `/ocr/get_ocr_infos` | texts (含text、confidence、bbox、center) + coords | 新增 ocr_info |
| `/ocr/get_coord_by_text` | texts + coords | 新增 ocr_info |
| `/ocr/get_ocr_texts` | 纯文本字符串 text（无坐标信息） | 新增 ocr_info |
| `/image/match_near_text` | 只有 text_position 中心点 | 新增 ocr_info |

## 设计方案

### 新增模型

**文件**: `ocr_service/api/schemas.py`（第33行附近，PointModel 定义之后）

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

1. **OCRResponse**（第93-100行）
   ```python
   class OCRResponse(BaseModel):
       status: str
       texts: list[TextBlockModel] = []
       coords: list[PointModel] = []
       ocr_info: list[OCRInfoItem] = []  # 新增
       duration_ms: int = 0
       error: Optional[str] = None
   ```

2. **OCRTextResponse**（第122-128行）
   ```python
   class OCRTextResponse(BaseModel):
       status: str
       text: str = ""
       ocr_info: list[OCRInfoItem] = []  # 新增
       duration_ms: int = 0
       error: Optional[str] = None
   ```

3. **TextNearImageResponse**（第172-181行）
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

#### 1. 添加导入语句（第11-25行导入区域）

```python
from ocr_service.api.schemas import (
    # ... 现有导入
    OCRInfoItem,  # 新增
)
```

#### 2. 各接口修改点

**通用构建逻辑**:
```python
ocr_info = [
    OCRInfoItem(text=t.text, center=PointModel(x=t.center.x, y=t.center.y))
    for t in texts  # 或 result.texts / text_blocks
]
```

**接口 1: `/ocr/get_ocr_infos`**（第87-147行）

返回时添加 `ocr_info=ocr_info`，从过滤后的 texts 构建。

**接口 2: `/ocr/get_coord_by_text`**（第150-236行）

- 正则模式 OCR 失败时（第184-189行）：添加 `ocr_info=[]`
- 正则模式匹配成功时（第220-235行）：添加 `ocr_info=ocr_info`
- 普通模式空结果时（第213-218行）：添加 `ocr_info=[]`
- 普通模式匹配成功时（第220-235行）：添加 `ocr_info=ocr_info`

**接口 3: `/ocr/get_ocr_texts`**（第239-276行）

- OCR 失败时（第262-267行）：添加 `ocr_info=[]`
- 成功时（第272-276行）：添加 `ocr_info=ocr_info`

**接口 4: `/image/match_near_text`**（第327-433行）

**注意**: 该接口使用 `find_text` 查找**单个匹配文字**，不是 `recognize` 全图识别。
因此 `ocr_info` **最多包含一个元素**（匹配到的文字块），而非全图所有 OCR 结果。

- 文字未找到时（第362-369行）：添加 `ocr_info=[]`
- 图片未匹配时（第384-391行）：添加 `ocr_info=[OCRInfoItem(...)]`（文字已找到）
- 超出距离时（第404-411行）：添加 `ocr_info=[OCRInfoItem(...)]`（文字已找到）
- 成功匹配时（第417-433行）：添加 `ocr_info=[OCRInfoItem(...)]`

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
5. 验证 `ocr_info` 字段与 `texts` 字段数据一致性（文字内容和坐标应对应）
6. `/image/match_near_text` 的 `ocr_info` 最多包含一个元素