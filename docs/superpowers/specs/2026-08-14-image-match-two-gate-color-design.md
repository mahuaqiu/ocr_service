# 图像匹配两道关卡（灰度 + 颜色）设计

日期：2026-08-14  
状态：已定稿（待实现）  
范围：`ocr_service` 模板匹配（`match_template` / `match_all`）

## 1. 背景与问题

当前 `ImageMatcher` 已是「灰度粗匹配 + BGR 三通道相关取 min」的颜色二次校验，并用 `confidence = min(gray, color)` 与阈值比较。

实测问题：

1. **细红线负样本误匹配**：红线只占少量像素，通道相关被大面积相似背景抬高（如 5.png 颜色分仍 ~0.93），默认阈值 0.9 挡不住。
2. **调用链存在 JPEG 压缩**：autotest worker 的 `OCRClient.match_image` 会把 source、template **都**压成 JPEG q=90 再请求 OCR。按无损 PNG 调参会过于乐观。
3. **过程日志不足**：无法从服务端日志看到 gray / color 各自多少、卡在哪一关。

测试图目录：`C:\Users\Administrator\Downloads\test`

| 小图 | 大图 | 预期 |
|------|------|------|
| 22.png、33.png | 11.png | 命中 |
| 2.png、3.png、4.png | 1.png | 命中 |
| 5.png、6.png | 1.png | 不命中 |

## 2. 目标

- 灰度先过滤结构不像的位置。
- 对过灰度的候选，用「极大色差像素占比」做颜色关，能区分细红线有/无。
- 请求阈值（worker `threshold` → API `confidence_threshold`）**同时**作为两关门槛。
- 日志分别输出两关分数与是否通过。
- 在 worker 真实压缩路径（双方 JPEG q=90）下，上述测试集全对。

非目标：

- 不改 worker / `ocr_client` 压缩策略（本次只让服务端在真实输入下分得开）。
- 不改 API schema 字段结构。
- 不改 `match_feature`（SIFT）的颜色逻辑。

## 3. 参数对齐（调用链）

```
Action.threshold (默认 0.9)
  → ocr_client.match_image(..., threshold=...)
  → HTTP body: confidence_threshold
  → ImageMatcher.match(..., threshold=request.confidence_threshold)
```

| 层级 | 字段名 | 默认 |
|------|--------|------|
| worker 用例 | `threshold` | 0.9 |
| OCR 请求 | `confidence_threshold` | 0.9 |
| 服务内部 | `threshold` | 请求值；缺省用配置 `default_match_threshold=0.9` |
| 响应 | `matches[].confidence` | 命中时的 gray 分 |

约定：同一 `threshold` 同时控制：

- 第 1 关：`gray_score >= threshold`
- 第 2 关：`color_score >= threshold`

## 4. 算法设计

### 4.1 总流程（两道独立关卡）

```
解码 BGR
  → 第1关 灰度 TM_CCOEFF_NORMED
       gray < threshold → 淘汰（不再算颜色）
  → 第2关 颜色（仅候选 patch）
       color < threshold → 淘汰
  → 两关都过 → 命中
```

**不做** `confidence = min(gray, color)` 再与阈值比一次。决策是两道门，不是合成一个分。

### 4.2 第 1 关：灰度

- 方法：`cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`
- 单目标（`match_template`）：取全局 max；`gray_score < threshold` 则无匹配。
- 多目标（`match_all`）：`gray_score >= threshold` 的位置 → 现有简易 NMS 去重叠 → 逐个进第 2 关。
- template 任一边大于 source：无匹配 + 警告日志。

### 4.3 第 2 关：颜色（极大色差像素占比）

对候选 `patch` 与 `template`（同高宽）：

1. 转 Lab。
2. 每像素 \(\Delta E = \|Lab_{patch} - Lab_{template}\|_2\)。
3. `frac = mean(ΔE > DE_THRESHOLD)`，默认 `DE_THRESHOLD = 30`。  
   - 只统计「极大」色差，忽略 JPEG q=90 常见小抖动（正样本 JPEG 下 ΔE>30 的 frac 实测为 0）。
4. 线性打分（按产品要求校准）：

\[
color\_score = \max(0,\ 1 - \frac{frac}{\alpha}),\quad \alpha = \frac{1}{15} \approx 0.0667
\]

等价：`color_score = max(0, 1 - 15 * frac)`。

标定点：

| frac(ΔE>30) | color |
|-------------|-------|
| 0% | 1.00 |
| 0.5% | 0.925 |
| **1%** | **0.85** |
| **2%** | **0.70** |
| 5% | 0.25 |
| ≥6.67% | 0.00 |

5. `color_score >= threshold` 才通过第 2 关。

### 4.4 返回值

- 命中：`MatchResult.confidence = gray_score`（兼容现字段；调用方阈值语义仍是「相似度门槛」）。
- 是否命中：两关都过，与 `min` 无关。
- color / frac 只写日志；本次不新增 API 字段。

### 4.5 常量（先写死在 `image_matcher`）

```python
COLOR_DE_THRESHOLD = 30      # 极大色差 ΔE 门槛
COLOR_FRAC_ALPHA = 1.0 / 15  # frac 到该比例时 color 归零；1%→0.85，2%→0.70
```

不新增环境变量/配置项（后续若要调再暴露）。

## 5. 日志

使用 `logging.getLogger(__name__)`，与 `ocr_engine` 一致，挂到 `ocr_service` 日志体系。

每次匹配建议输出：

1. **开始**：src/tpl 尺寸、threshold、method、multi_target  
2. **第 1 关**：gray 峰值与坐标；未过则 `gray fail` 并结束；multi 时 NMS 前后候选数  
3. **第 2 关**：每个候选的 `color`、`frac`、`de_thr`、`alpha`、pass/fail  
4. **结束**：命中数、各命中 bbox + gray + color、耗时 ms  

示例：

```
[MATCH] start src=1919x1079 tpl=39x50 thr=0.9 multi=False
[MATCH] gray pass: 0.9983 @ (449,792)
[MATCH] color pass: 1.0000 frac=0.0000 @ (449,792) (de>30, a=0.0667)
[MATCH] done hits=1 duration_ms=12 | hit gray=0.9983 color=1.0000 bbox=(449,792,39,50)
```

负样本示例：

```
[MATCH] gray pass: 0.9437 @ (386,798)
[MATCH] color fail: 0.1316 frac=0.0579 < thr=0.9 @ (386,798)
[MATCH] done hits=0 duration_ms=10
```

## 6. 校准数据（双方 JPEG q=90，threshold=0.9）

公式：`color = max(0, 1 - frac/ (1/15))`，`ΔE>30`。

| 样本 | gray | frac | color | 第1关 | 第2关 | 命中 | 预期 |
|------|------|------|-------|-------|-------|------|------|
| 22 in 11 | 0.9916 | 0 | 1.000 | 过 | 过 | 有 | 有 |
| 33 in 11 | 0.9936 | 0 | 1.000 | 过 | 过 | 有 | 有 |
| 4 in 1 | 0.9980 | 0 | 1.000 | 过 | 过 | 有 | 有 |
| 2 in 1 | 0.9980 | 0 | 1.000 | 过 | 过 | 有 | 有 |
| 3 in 1 | 0.9983 | 0 | 1.000 | 过 | 过 | 有 | 有 |
| 5 in 1 | 0.9437 | 0.0579 | 0.132 | 过 | 淘汰 | 无 | 无 |
| 6 in 1 | 0.9112 | 0.0721 | 0.000 | 过 | 淘汰 | 无 | 无 |

结果：7/7。原始 PNG 同样 7/7。

## 7. 测试计划

1. 更新 `tests/test_image_matcher_color.py`  
   - 构造图与真实图路径测试均应 **模拟 worker：source、template 先 JPEG q=90 再 match**。  
   - 红线负样本：允许 gray 过阈值，但最终 `matches` 为空。  
   - 同结构同色正样本：命中且 confidence（gray）≥ threshold。  
2. 多目标 `match_all`：同色目标保留，异色结构相似目标剔除。  
3. 默认阈值仍为 0.9 的配置断言保留。  
4. 可用 caplog 断言日志中出现 gray/color 关键字（可选，不强制绑死文案）。

## 8. 改动范围

| 文件 | 变更 |
|------|------|
| `ocr_service/core/image_matcher.py` | 替换颜色打分；两关判定；补日志 |
| `tests/test_image_matcher_color.py` | JPEG 模拟 + 断言对齐新行为 |

不改：`ocr_client`、worker action、API schema、`match_feature`。

## 9. 风险与后续

- **中间地带**：1% 极大色差 → color=0.85，在 threshold=0.9 时第 2 关淘汰。若业务存在「允许极少装饰色差」的用例，需降低用例 `threshold` 或再调 α。  
- **极细线 + 更大模板**：frac 被稀释时颜色关变松；当前测试集红线 frac≈6%–7% 仍足够。若线上出现漏拦，可再评估按「高饱和像素掩膜」加权。  
- **feature 匹配**未纳入本设计；若同样出现色差误匹配，另开任务。
