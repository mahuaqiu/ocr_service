# 图像匹配两道关卡（灰度 + 颜色）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `ImageMatcher` 模板匹配改为「灰度关 + 颜色关」两道独立关卡判定，颜色关用「极大色差像素占比」打分，并补全过程日志，使其在 autotest worker 双方 JPEG q=90 链路下能区分细红线有/无。

**Architecture:** 保持 `match_template` / `match_all` 入参与返回结构不变；内部用 `threshold` 同时卡灰度关（`TM_CCOEFF_NORMED`）和颜色关（`color = max(0, 1 - 15*frac(ΔE>30))`）；命中时 `confidence` 返回 gray 分；color/frac 进日志。日志经 `ocr_service` logger 输出，前缀 `[MATCH]`，并加入 `RequestResponseFilter` 白名单。

**Tech Stack:** Python 3.12、OpenCV（`cv2.matchTemplate`、`cv2.cvtColor(...COLOR_BGR2LAB)`）、numpy、pytest。

**参考 spec:** `docs/superpowers/specs/2026-08-14-image-match-two-gate-color-design.md`

## Global Constraints

- 项目虚拟环境位于 `venv/`，所有 Python/pytest 命令须先 `.\venv\Scripts\Activate.ps1`（PowerShell）再执行。
- 运行脚本/测试用 `python`、`pytest`，不要用绝对路径解释器。
- 所有代码注释用中文。
- 常量写死：`COLOR_DE_THRESHOLD = 30`、`COLOR_FRAC_ALPHA = 1.0 / 15`。
- 默认阈值 `0.9` 不变；`threshold` 同时作为灰度关和颜色关门槛。
- 不改 `ocr_service/api/schemas.py`、不改 `common/ocr_client.py`（autotest 侧）、不改 `match_feature`。
- 测试必须模拟 worker 真实链路：source 与 template 都先 JPEG q=90 再匹配（封装到测试 helper）。
- Windows 平台，PowerShell 为主 shell；`Bash` 工具用于 POSIX 语法时注意换行。
- 提交粒度：每个 Task 一次 commit；commit message 中文。

## File Structure

| 文件 | 责任 | 改动 |
|------|------|------|
| `ocr_service/core/image_matcher.py` | 模板匹配引擎 | 改：颜色打分、两关判定、日志、常量 |
| `ocr_service/server.py` | 日志过滤器白名单 | 改：`RequestResponseFilter` 增加 `[MATCH]` |
| `tests/test_image_matcher_color.py` | 颜色校验单测 | 改：JPEG helper、用例对齐新行为、新增真实图断言 |

不新增文件。`match_result.py` 模型不变（`confidence` 仍为 float）。

---

## Task 1: 日志过滤器放行 `[MATCH]`

为后续 image_matcher 日志能写入文件/控制台做准备。`server.py` 的 `RequestResponseFilter` 目前只放行 `[REQUEST]/[RESPONSE]/[OCR_RAW]` 与 ERROR+，普通 INFO 会被丢弃。

**Files:**
- Modify: `ocr_service/server.py:178-184`

**Interfaces:**
- Produces: `RequestResponseFilter.filter` 放行含 `[MATCH]` 的日志记录，供 Task 3 的 logger 使用。

- [ ] **Step 1: 修改过滤器白名单**

把 `ocr_service/server.py` 中：

```python
    def filter(self, record):
        msg = record.getMessage()
        # 允许包含 [REQUEST]、[RESPONSE]、[OCR_RAW] 或 ERROR 级别的日志
        return "[REQUEST]" in msg or "[RESPONSE]" in msg or "[OCR_RAW]" in msg or record.levelno >= logging.ERROR
```

改为：

```python
    def filter(self, record):
        msg = record.getMessage()
        # 允许包含 [REQUEST]、[RESPONSE]、[OCR_RAW]、[MATCH] 或 ERROR 级别的日志
        return "[REQUEST]" in msg or "[RESPONSE]" in msg or "[OCR_RAW]" in msg or "[MATCH]" in msg or record.levelno >= logging.ERROR
```

- [ ] **Step 2: 验证过滤器放行 MATCH 日志**

运行：

```bash
.\venv\Scripts\Activate.ps1
python -c "import logging; from ocr_service.server import RequestResponseFilter; rec=logging.LogRecord('x',logging.INFO,'',0,'[MATCH] hello',None,None); print('PASS' if RequestResponseFilter().filter(rec) else 'FAIL'); rec2=logging.LogRecord('x',logging.INFO,'',0,'no-prefix hello',None,None); print('PASS' if not RequestResponseFilter().filter(rec2) else 'FAIL')"
```

Expected: 两行均为 `PASS`（带前缀放行、无前缀被挡）。

- [ ] **Step 3: Commit**

```bash
git add ocr_service/server.py
git commit -m "feat: 日志过滤器放行 [MATCH] 前缀"
```

---

## Task 2: 重写颜色打分为「极大色差像素占比」

把 `_color_similarity` 替换为 `_color_score`（Lab ΔE 占比线性打分），并对外暴露 `(score, frac)` 便于日志。先只改方法，不改调用点。

**Files:**
- Modify: `ocr_service/core/image_matcher.py:43-71`（替换 `_color_similarity`）

**Interfaces:**
- Produces: `ImageMatcher._color_score(patch, template) -> tuple[float, float]`，返回 `(color_score, frac)`；`color_score` ∈ [0,1]，`frac` ∈ [0,1]。
- Consumes: 无新依赖。

- [ ] **Step 1: 写失败测试 — 颜色打分公式标定**

在 `tests/test_image_matcher_color.py` 顶部 import 区追加（若已有 `import cv2, numpy as np` 则不重复）：

```python
from ocr_service.core.image_matcher import ImageMatcher
```

在文件末尾追加：

```python
def test_color_score_calibration():
    """颜色打分按极大色差像素占比线性计：1%→0.85、2%→0.70、≥6.67%→0。"""
    import numpy as np
    import cv2

    matcher = ImageMatcher()
    h, w = 100, 100  # 10000 像素

    def make(ratio_red: float):
        """构造 template=纯灰底；patch=灰底 + ratio_red 比例的纯红像素（ΔE 远>30）。"""
        tpl = np.full((h, w, 3), 128, dtype=np.uint8)
        patch = tpl.copy()
        n = int(h * w * ratio_red)
        if n > 0:
            ys = np.random.default_rng(0).integers(0, h, n)
            xs = np.random.default_rng(1).integers(0, w, n)
            patch[ys, xs] = (0, 0, 255)  # BGR 红
        return patch, tpl

    # 0% -> 1.0
    s, _ = matcher._color_score(*make(0.0))
    assert abs(s - 1.0) < 1e-6
    # 1% -> 0.85
    s, frac = matcher._color_score(*make(0.01))
    assert abs(frac - 0.01) < 1e-3
    assert abs(s - 0.85) < 1e-3
    # 2% -> 0.70
    s, frac = matcher._color_score(*make(0.02))
    assert abs(frac - 0.02) < 1e-3
    assert abs(s - 0.70) < 1e-3
    # 6.67% -> 0
    s, _ = matcher._color_score(*make(0.0667))
    assert s == 0.0
    # 尺寸不匹配 -> 0
    s, _ = matcher._color_score(np.zeros((10, 10, 3), np.uint8), np.zeros((11, 11, 3), np.uint8))
    assert s == 0.0
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_image_matcher_color.py::test_color_score_calibration -v`
Expected: FAIL，`AttributeError: 'ImageMatcher' object has no attribute '_color_score'`（旧方法叫 `_color_similarity`）。

- [ ] **Step 3: 替换 `_color_similarity` 为 `_color_score`**

把 `ocr_service/core/image_matcher.py` 第 43-71 行（`_color_similarity` 整个方法）替换为：

```python
    # 颜色关参数：极大色差 ΔE 门槛；frac 到该比例时 color 归零（1%→0.85，2%→0.70）
    COLOR_DE_THRESHOLD: int = 30
    COLOR_FRAC_ALPHA: float = 1.0 / 15

    @staticmethod
    def _color_score(patch: np.ndarray, template: np.ndarray) -> tuple[float, float]:
        """
        计算候选区域与模板的颜色相似度（极大色差像素占比线性打分）。

        转到 Lab 空间逐像素算 ΔE，统计 ΔE > COLOR_DE_THRESHOLD 的像素占比 frac，
        再线性打分 color = max(0, 1 - frac / COLOR_FRAC_ALPHA)。
        - JPEG q=90 的常见小抖动 ΔE 多 <30，不计入；
        - 细红线等颜色差异 ΔE 常远大于 30，会被计入，从而压低颜色分。

        Args:
            patch: 大图裁出的候选区域（BGR）。
            template: 模板图像（BGR）。

        Returns:
            tuple[float, float]: (color_score, frac)。color_score ∈ [0,1]，frac ∈ [0,1]。
        """
        if patch.shape[:2] != template.shape[:2]:
            return 0.0, 1.0
        if patch.size == 0 or template.size == 0:
            return 0.0, 1.0

        patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        template_lab = cv2.cvtColor(template, cv2.COLOR_BGR2LAB).astype(np.float32)
        de = np.sqrt(np.sum((patch_lab - template_lab) ** 2, axis=2))
        frac = float((de > ImageMatcher.COLOR_DE_THRESHOLD).mean())
        score = max(0.0, 1.0 - frac / ImageMatcher.COLOR_FRAC_ALPHA)
        return score, frac
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_image_matcher_color.py::test_color_score_calibration -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add ocr_service/core/image_matcher.py tests/test_image_matcher_color.py
git commit -m "feat: 颜色打分改为极大色差像素占比线性计"
```

---

## Task 3: 两道关卡判定 + 过程日志

把 `_verify_candidate` 改为两道关卡（gray≥threshold 且 color≥threshold），命中 `confidence` 返回 gray；`match_template` / `match_all` 补日志。导入 `logging`。

**Files:**
- Modify: `ocr_service/core/image_matcher.py:1-21`（import + 模块 logger）
- Modify: `ocr_service/core/image_matcher.py:73-111`（`_verify_candidate`）
- Modify: `ocr_service/core/image_matcher.py:113-179`（`match_template`）
- Modify: `ocr_service/core/image_matcher.py:181-280`（`match_all`）

**Interfaces:**
- Produces: `ImageMatcher._verify_candidate(...)` 仍返回 `Optional[MatchResult]`；`MatchResult.confidence` = gray_score（命中时）。`match_template` / `match_all` 行为不变（返回 `ImageMatchResult`），但内部判定与日志改变。

- [ ] **Step 1: 写失败测试 — 真实图 JPEG 链路两关判定**

在 `tests/test_image_matcher_color.py` 顶部 import 区确保有：

```python
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_service.config import ServiceConfig
from ocr_service.core.image_matcher import ImageMatcher
```

在文件末尾追加测试 helper 与用例：

```python
# 测试图目录：C:\Users\Administrator\Downloads\test
TEST_IMG_DIR = Path(r"C:\Users\Administrator\Downloads\test")


def _encode_file_jpeg90(name: str) -> str:
    """读取测试图 -> JPEG q=90 重编码 -> Base64（模拟 worker ocr_client.match_image 链路）。"""
    img = cv2.imdecode(np.fromfile(str(TEST_IMG_DIR / name), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"load fail: {name}"
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _encode_bgr_jpeg90(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@pytest.fixture
def real_matcher() -> ImageMatcher:
    return ImageMatcher(ServiceConfig(default_match_threshold=0.9, default_match_method="template"))


@pytest.mark.skipif(not TEST_IMG_DIR.exists(), reason="测试图目录不存在")
@pytest.mark.parametrize("src,tpl", [
    ("11.png", "22.png"),
    ("11.png", "33.png"),
    ("1.png", "4.png"),
    ("1.png", "2.png"),
    ("1.png", "3.png"),
])
def test_real_positive_pairs_match_under_jpeg(real_matcher, src, tpl):
    """正样本：双方 JPEG q=90 后仍应命中（confidence=gray≥0.9，且颜色关通过）。"""
    result = real_matcher.match_template(
        source_data=_encode_file_jpeg90(src),
        template_data=_encode_file_jpeg90(tpl),
        threshold=0.9,
    )
    assert result.status == "success"
    assert len(result.matches) == 1, f"{tpl} in {src} 应命中，实际 {len(result.matches)}"
    assert result.matches[0].confidence >= 0.9


@pytest.mark.skipif(not TEST_IMG_DIR.exists(), reason="测试图目录不存在")
@pytest.mark.parametrize("src,tpl", [
    ("1.png", "5.png"),
    ("1.png", "6.png"),
])
def test_real_negative_pairs_rejected_by_color_gate(real_matcher, src, tpl):
    """负样本（细红线）：灰度关可能过，但颜色关必须淘汰，最终无匹配。"""
    result = real_matcher.match_template(
        source_data=_encode_file_jpeg90(src),
        template_data=_encode_file_jpeg90(tpl),
        threshold=0.9,
    )
    assert result.status == "success"
    assert result.matches == [], f"{tpl} in {src} 不应命中，实际 {result.matches}"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_image_matcher_color.py -v -k "real"`
Expected: 正样本可能 PASS（旧实现通道相关在 JPEG 下正样本仍过），负样本 FAIL（5.png 旧实现 color≈0.13<0.9 但旧 `min(gray,color)` 会判不过——若旧实现恰好也挡住则负样本 PASS，但正样本 3.png 在旧实现下可能因通道相关 color<0.9 而 FAIL）。无论如何，需看到至少一个 FAIL 触发改造。若全部 PASS，仍继续 Step 3（新逻辑要保证全 PASS）。

- [ ] **Step 3: 添加 logger 与常量到模块顶部**

把 `ocr_service/core/image_matcher.py` 第 1-21 行（文件头 + import）替换为：

```python
"""
图像匹配引擎。

基于 OpenCV 实现模板匹配和特征匹配。
模板匹配采用「灰度关 + 颜色关」两道独立关卡：
1) 灰度 TM_CCOEFF_NORMED 定位候选；
2) 候选区域 Lab ΔE 极大色差像素占比线性打分过滤。
两关都过阈值才命中，命中 confidence 返回灰度分。
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from ocr_service.config import ServiceConfig, get_config
from ocr_service.models.match_result import (
    BoundingBox,
    ImageMatchResult,
    MatchResult,
    Point,
)
from ocr_service.utils.image_utils import decode_image

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: 重写 `_verify_candidate` 为两道关卡**

把 `ocr_service/core/image_matcher.py` 中 `_verify_candidate` 方法（原第 73-111 行）替换为：

```python
    def _verify_candidate(
        self,
        source: np.ndarray,
        template: np.ndarray,
        x: int,
        y: int,
        gray_score: float,
        threshold: float,
    ) -> Optional[MatchResult]:
        """
        对灰度关通过的候选做颜色关二次校验。

        两道关卡：
        - 灰度关已在外层通过（gray_score >= threshold）；
        - 颜色关：color_score >= threshold 才通过。
        命中时 confidence 返回灰度分（兼容调用方「相似度」语义）。

        Args:
            source: 源图像（BGR）。
            template: 模板图像（BGR）。
            x: 候选左上角 x。
            y: 候选左上角 y。
            gray_score: 灰度匹配分数（第1关分数）。
            threshold: 两关共同门槛。

        Returns:
            Optional[MatchResult]: 两关都过则返回结果，否则 None。
        """
        h, w = template.shape[:2]
        patch = source[y : y + h, x : x + w]
        if patch.shape[0] != h or patch.shape[1] != w:
            return None

        color_score, frac = self._color_score(patch, template)
        passed = color_score >= threshold
        if passed:
            logger.info(
                f"[MATCH] color pass: {color_score:.4f} frac={frac:.4f} "
                f"@ ({x},{y}) (de>{self.COLOR_DE_THRESHOLD}, a={self.COLOR_FRAC_ALPHA:.4f}, thr={threshold})"
            )
            return MatchResult(
                confidence=float(gray_score),
                bbox=BoundingBox(x=int(x), y=int(y), width=w, height=h),
                center=Point(x=int(x) + w // 2, y=int(y) + h // 2),
            )
        logger.info(
            f"[MATCH] color fail: {color_score:.4f} frac={frac:.4f} < thr={threshold} "
            f"@ ({x},{y}) (de>{self.COLOR_DE_THRESHOLD})"
        )
        return None
```

- [ ] **Step 5: 重写 `match_template` 补日志**

把 `match_template` 方法（原第 113-179 行）替换为：

```python
    def match_template(
        self,
        source_data: bytes | str,
        template_data: bytes | str,
        threshold: Optional[float] = None,
    ) -> ImageMatchResult:
        """
        精确模板匹配。

        流程：灰度关定位 → 颜色关二次校验，两关都过才命中。

        Args:
            source_data: 源图像（大图）。
            template_data: 模板图像（小图）。
            threshold: 两关共同阈值，默认使用配置中的阈值。

        Returns:
            ImageMatchResult: 匹配结果。
        """
        start_time = time.time()
        threshold = threshold or self.config.default_match_threshold

        try:
            source = decode_image(source_data)
            template = decode_image(template_data)

            logger.info(
                f"[MATCH] start src={source.shape[1]}x{source.shape[0]} "
                f"tpl={template.shape[1]}x{template.shape[0]} thr={threshold} multi=False"
            )

            # template 任一边大于 source：无匹配
            if template.shape[0] > source.shape[0] or template.shape[1] > source.shape[1]:
                logger.warning(
                    f"[MATCH] template larger than source: "
                    f"tpl={template.shape[:2]} src={source.shape[:2]}"
                )
                return ImageMatchResult(
                    status="success", matches=[], duration_ms=int((time.time() - start_time) * 1000)
                )

            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
            gray_score = float(max_val)

            matches = []
            if gray_score >= threshold:
                logger.info(
                    f"[MATCH] gray pass: {gray_score:.4f} @ ({max_loc[0]},{max_loc[1]}) gate={threshold}"
                )
                verified = self._verify_candidate(
                    source=source,
                    template=template,
                    x=max_loc[0],
                    y=max_loc[1],
                    gray_score=gray_score,
                    threshold=threshold,
                )
                if verified is not None:
                    matches.append(verified)
            else:
                logger.info(
                    f"[MATCH] gray fail: {gray_score:.4f} < thr={threshold} @ ({max_loc[0]},{max_loc[1]})"
                )

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[MATCH] done hits={len(matches)} duration_ms={duration_ms}"
                + (
                    f" | hit gray={matches[0].confidence:.4f} bbox=({matches[0].bbox.x},{matches[0].bbox.y},{matches[0].bbox.width},{matches[0].bbox.height})"
                    if matches else ""
                )
            )
            return ImageMatchResult(
                status="success", matches=matches, duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.exception(f"[MATCH] error duration_ms={duration_ms}")
            return ImageMatchResult(
                status="error", matches=[], duration_ms=duration_ms, error=str(e)
            )
```

- [ ] **Step 6: 重写 `match_all` 补日志**

把 `match_all` 方法（原第 181-280 行，含到 `return` 与 `except`）替换为：

```python
    def match_all(
        self,
        source_data: bytes | str,
        template_data: bytes | str,
        threshold: Optional[float] = None,
    ) -> ImageMatchResult:
        """
        多目标模板匹配。

        流程：灰度关（>=阈值的位置）→ 非极大值抑制 → 颜色关二次校验。

        Args:
            source_data: 源图像（大图）。
            template_data: 模板图像（小图）。
            threshold: 两关共同阈值。

        Returns:
            ImageMatchResult: 匹配结果（可能包含多个匹配）。
        """
        start_time = time.time()
        threshold = threshold or self.config.default_match_threshold

        try:
            source = decode_image(source_data)
            template = decode_image(template_data)

            logger.info(
                f"[MATCH] start src={source.shape[1]}x{source.shape[0]} "
                f"tpl={template.shape[1]}x{template.shape[0]} thr={threshold} multi=True"
            )

            if template.shape[0] > source.shape[0] or template.shape[1] > source.shape[1]:
                logger.warning(
                    f"[MATCH] template larger than source: "
                    f"tpl={template.shape[:2]} src={source.shape[:2]}"
                )
                return ImageMatchResult(
                    status="success", matches=[], duration_ms=int((time.time() - start_time) * 1000)
                )

            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)

            h, w = template_gray.shape
            locations = np.where(result >= threshold)
            rectangles = []
            for pt in zip(*locations[::-1]):
                rectangles.append([pt[0], pt[1], w, h, result[pt[1], pt[0]]])

            logger.info(
                f"[MATCH] gray candidates: {len(rectangles)} (>=thr={threshold})"
            )

            # 非极大值抑制
            picked = []
            if rectangles:
                rectangles.sort(key=lambda x: (x[1], x[0]))
                for rect in rectangles:
                    x, y, rw, rh, conf = rect
                    overlap = False
                    for p in picked:
                        px, py, pw, ph, _ = p
                        x1 = max(x, px)
                        y1 = max(y, py)
                        x2 = min(x + rw, px + pw)
                        y2 = min(y + rh, py + ph)
                        if x1 < x2 and y1 < y2:
                            overlap = True
                            break
                    if not overlap:
                        picked.append(rect)

            logger.info(f"[MATCH] nms picked: {len(picked)}")

            matches = []
            for rect in picked:
                x, y, _rw, _rh, conf = rect
                verified = self._verify_candidate(
                    source=source,
                    template=template,
                    x=int(x),
                    y=int(y),
                    gray_score=float(conf),
                    threshold=threshold,
                )
                if verified is not None:
                    matches.append(verified)

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[MATCH] done hits={len(matches)} duration_ms={duration_ms}"
            )
            return ImageMatchResult(
                status="success", matches=matches, duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.exception(f"[MATCH] error duration_ms={duration_ms}")
            return ImageMatchResult(
                status="error", matches=[], duration_ms=duration_ms, error=str(e)
            )
```

- [ ] **Step 7: 运行全部颜色测试**

Run: `pytest tests/test_image_matcher_color.py -v`
Expected: 全部 PASS（含标定、正样本、负样本、原有 `test_red_line_*`、`test_match_all_also_filters_color_mismatch`、`test_default_match_threshold_is_0_9`）。

注意：原有 `test_red_line_template_should_match_red_line_source` / `test_match_all_also_filters_color_mismatch` 用的是 `_make_panel` 合成图、`threshold=0.8`，新公式下应仍通过（同色 color=1.0，异色 frac>6.67% → color=0）。若某个旧用例因 JPEG 未压缩 + 红线占比不同而边界变化，检查红线像素占比：合成面板 60×140=8400 像素，红线宽 2 像素横贯（约 2×106=212 像素 ≈ 2.5%），`color = 1 - 15×0.025 = 0.625 < 0.8` 会淘汰异色目标，符合预期；同色目标 color=1.0 通过。若断言失败，先用 Step 8 的诊断脚本核对 frac。

- [ ] **Step 8: 诊断脚本核对真实图分数（可选但推荐）**

Run:

```bash
.\venv\Scripts\Activate.ps1
python -c "import cv2,numpy as np,base64; from pathlib import Path; from io import BytesIO; from ocr_service.core.image_matcher import ImageMatcher; from ocr_service.config import ServiceConfig; td=Path(r'C:\Users\Administrator\Downloads\test'); m=ImageMatcher(ServiceConfig(default_match_threshold=0.9)); 
pairs=[('11.png','22.png',True),('11.png','33.png',True),('1.png','4.png',True),('1.png','2.png',True),('1.png','3.png',True),('1.png','5.png',False),('1.png','6.png',False)];
[print(p,'exp',e,m.match_template(base64.b64encode(cv2.imencode('.jpg',cv2.imdecode(np.fromfile(str(td/s),np.uint8),cv2.IMREAD_COLOR))[1].tobytes()).decode(),base64.b64encode(cv2.imencode('.jpg',cv2.imdecode(np.fromfile(str(td/t),np.uint8),cv2.IMREAD_COLOR))[1].tobytes()).decode(),threshold=0.9).to_dict()) for s,t,e in pairs for p in [t+' in '+s]]"
```

Expected: 正样本 hits=1、confidence≈0.99+；负样本 matches=[]。

- [ ] **Step 9: Commit**

```bash
git add ocr_service/core/image_matcher.py tests/test_image_matcher_color.py
git commit -m "feat: 模板匹配改为两道关卡判定并补过程日志"
```

---

## Task 4: 清理旧 `_color_similarity` 残留与回归

确认旧方法已无引用、测试全绿、spec 中的 7/7 用例对齐。

**Files:**
- Verify: `ocr_service/core/image_matcher.py`、`tests/test_image_matcher_color.py`

**Interfaces:**
- 无新增。

- [ ] **Step 1: 确认无 `_color_similarity` 残留引用**

Run: `grep -rn "_color_similarity" ocr_service tests`
Expected: 无输出（或仅注释，若有需删除）。若仍有引用，删除之。

- [ ] **Step 2: 运行整个测试套件回归**

Run: `pytest tests/ -v --ignore=tests/test_concurrent_ocr.py 2>&1 | tail -40`（若 `test_concurrent_ocr.py` 需起服务则忽略，否则一并跑）
Expected: 无新增失败。关注 `tests/test_image_matcher_color.py` 全绿。

- [ ] **Step 3: Commit（如有清理改动）**

```bash
git add -A
git commit -m "chore: 清理旧颜色相关方法残留"
```

若无改动则跳过。

---

## Self-Review 结果

**1. Spec 覆盖**
- §4.1 两道关卡流程 → Task 3 `_verify_candidate` / `match_template` / `match_all`
- §4.2 灰度关 → Task 3 Step 5/6（含 template 大于 source 边界）
- §4.3 颜色公式 + 标定（1%→0.85, 2%→0.70, 6.67%→0）→ Task 2 标定测试 + `_color_score`
- §4.4 返回 confidence=gray → Task 3 `_verify_candidate`
- §4.5 常量 → Task 2 类常量
- §5 日志 → Task 1 过滤器 + Task 3 各日志点
- §6 7/7 校准 → Task 3 Step 1 真实图参数化用例
- §7 测试（JPEG 模拟、红线负样本、多目标、默认阈值）→ Task 2/3/4

**2. 占位符扫描**：无 TBD/TODO；每步含具体代码或命令。

**3. 类型一致性**：`_color_score -> tuple[float,float]` 在 Task 2 定义、Task 3 `_verify_candidate` 使用一致；`COLOR_DE_THRESHOLD` / `COLOR_FRAC_ALPHA` 作为类属性在 `_color_score` 与 `_verify_candidate` 中引用一致；`MatchResult.confidence=float(gray_score)` 与 spec §4.4 一致。
