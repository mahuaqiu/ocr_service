# OCR 识别文本替换（对接测试平台配置中心）设计

日期：2026-09-11
状态：已评审定稿（完整设计见 zq-platform 仓库 `docs/superpowers/specs/2026-09-11-config-center-design.md`，本文为 ocr_service 侧摘要）

## 背景

OCR 识别偶发错字（「允许」→「充许」、「聊天」→「聊关」），影响下游使用。测试平台新增配置中心（免鉴权查询接口）维护替换规则，本服务在拿到原始识别结果后、`reg_` 等匹配之前做整串替换，并每天 12:00 定时刷新。

## 需求点

1. **启动拉取**：服务启动时从平台拉取 `ocr_config`（重试 3 次，失败不阻塞启动）。
2. **识别替换**：`recognize()` 解析出 TextBlock 后立即对每个 text 应用替换（早于一切 `reg_`/exact 匹配；`find_text`/`find_all_texts`/`get_text_center` 经由 recognize 自动覆盖）。
3. **定时刷新**：每天本地时间 12:00 拉取；失败每 10 分钟重试直到成功；成功前保留旧配置。

## 实现

- 新文件 `ocr_service/text_replacer.py`：
  - 替换字典单例 `get_replace_map()/set_replace_map()`（空字典=直通不替换）；
  - `apply_replacements(text)`：按配置顺序逐对 `str.replace(键, 值)`；
  - `fetch_replace_map()`（async httpx）：`GET {OCR_CONFIG_CENTER_URL}?key={OCR_CONFIG_CENTER_KEY}`，校验 `dict[str,str]`；
  - `refresh_replace_map()`：成功换新 + `[CONFIG]` INFO 日志，失败保旧 + ERROR 日志。
- `config.py` 新增：`OCR_CONFIG_CENTER_URL`（空=禁用）/ `OCR_CONFIG_CENTER_KEY`（默认 ocr_config）/ `OCR_CONFIG_CENTER_TIMEOUT`（默认 5s）。
- `core/ocr_engine.py`：`recognize()` 中 parse 之后逐块替换；`[OCR_RAW]` 日志仍打印替换前原文，便于排查。
- `server.py`：`create_app()` 增加 lifespan——启动拉取 + asyncio 后台任务（睡到下一个 12:00 → 刷新；失败 10 分钟后重试；关闭取消）；日志过滤器放行 `[CONFIG]` 标签；未配置 URL 时整段跳过。
- `docker-compose.yml`：透传 `OCR_CONFIG_CENTER_URL`/`OCR_CONFIG_CENTER_KEY`。

## 接口契约（测试平台）

`GET /api/public/config-center/query?key=ocr_config`（免鉴权）：
- 200 → 字典本身：`{"充许":"允许","聊关":"聊天"}`（顺序即配置顺序）
- 404 → `{"detail":"配置项不存在: ocr_config"}`

## 测试

`tests/test_text_replacer.py`：apply_replacements（基本/多对/顺序叠加/空直通）、fetch 解析（httpx.MockTransport：200 字典/非 dict/非法 JSON/404）、refresh 失败保旧、下一个 12:00 计算（含跨天）。不触碰真实 PaddleOCR。

## 验收

1. 启动日志 `[CONFIG]` 显示拉取成功与规则数。
2. 识别含「充许」的图：`get_ocr_texts` 返回「允许」；`get_coord_by_text` 用 `filter_text=允许` 或 `reg_允[许]` 命中，用「充许」不命中（预期）。
3. 平台不可达时：启动不阻塞、服务可用（空规则或旧规则），日志有 ERROR。

## 部署

设置 `OCR_CONFIG_CENTER_URL=http://<platform-host>:8000/api/public/config-center/query` 后重启。
