# FinGround KPI Extraction Agent — Design Specification

## 1. Overview

FinGround 是一个基于 Google ADK 的多 Agent 年报 KPI 提取系统。用户提供一份年报引用、
目标财年和一个或多个 KPI；系统为每个 KPI 建立独立任务，从年报中找到可审计证据，完成
数值归一化，并返回结构化结果。

年报可能远大于模型上下文窗口。任何情况下都不得把年报全文放入 Root 或 Worker 的 prompt、
对话历史或单次工具响应。年报由工具侧存储并分块检索；模型在每个 task 中只能看到有严格大小
上限的候选摘要和证据片段。

系统只包含两个 Agent 角色：

1. `root_agent`：负责理解请求、规划任务、委派任务、监督进度和汇总结果。它不能读取年报，
   也不能判断 KPI 数值。
2. `kpi_worker`：唯一的执行型 Agent，使用 ADK `mode="task"`。Root 通过一个 ADK Workflow
   dispatcher 提交一批单 KPI 输入；Workflow 使用 `parallel_worker` 在隔离分支中并发运行多个
   Worker 实例。每个实例仍只处理一个 KPI，必须先读取该 KPI 的知识条目，再读取年报并返回
   一个结构化结果。

任务状态由 session 内的通用 Task Store 保存。它遵循 Claude Code 公开 SDK 中的任务协议，
KPI 输入、Worker 结果和错误只作为任务 `metadata`，不进入任务工具的数据模型。一个 ADK Plugin
在运行期间检查任务状态，并向 Root 注入未完成任务提醒。KPI 的定义和提取规则是结构化数据，
不写入 Root prompt，也不拆成多个 KPI Agent。

本阶段只实现本地原型。部署、CI/CD、外部数据库、向量检索、SEC/XBRL 数据和 benchmark
框架均不在范围内。

本地原型同时提供一个面向最终用户的对话式 Web 页面。它复用 ADK 的 session、artifact 和
`/run_sse` 接口，支持多轮对话、附件、历史 session 和实时任务进度，但不展示 thought、原始
tool call、trace 或 performance warning 等开发调试信息。KPI 提取是对话中可发起的一类任务，
不是固定表单或唯一页面流程；ADK Web 继续只用于开发调试。页面按照 ADK 事件的 `partial`
语义合并流式文本，并以最终非 partial 事件替换对应的临时输出，不能把两者重复追加。
消息正文支持安全的 Markdown 标题、列表、代码块和 GFM 表格；KPI 结果表使用语义化状态徽标，
证据列允许换行，窄屏或超宽内容通过表格容器横向滚动，不能把 Markdown 分隔符作为正文展示。

## 2. Goals

- 使用 ADK 原生混合编排：一个自由规划的 Root、一个 Workflow dispatcher、一个 task-mode
  Worker 定义。
- Root 只规划和监督，不接触年报内容。
- 一个任务只负责一个 KPI，任务边界清晰、可重试、可审计。
- Worker 在执行前必须通过工具读取对应 KPI 的知识。
- 支持正文大小超过模型上下文窗口的单份年报，且资源消耗不随全文长度线性进入模型上下文。
- 每个结果必须包含来源证据，不能只返回数值。
- 任务进度可查询，Root 不能在仍有未完成任务时宣称全部完成。
- 代码和状态模型保持最小，不为未来需求预建抽象层。
- 最终用户无需理解 Agent、Tool 或 Task Store，即可通过自然语言持续提出任务、上传资料并复核结果。

## 3. Non-goals

- 不为每个 KPI 创建独立 Agent、prompt 文件或 Python 模块。
- 不实现通用工作流引擎、消息队列或持久化任务平台。
- 不实现网页下载。PDF 上传通过外部 Unlimited-OCR 服务转为 Markdown，OCR 服务本身不在本项目中实现。
- 不使用 RAG、向量数据库或 embeddings。
- 不接入 SEC Facts、XBRL、第三方财务 API 或外部搜索。
- 不实现自建并发调度器、消息队列、benchmark CLI 或评分框架；并发只使用 ADK Workflow
  `parallel_worker`。
- 不实现部署、鉴权、租户隔离或生产级持久化。
- 不保留旧架构的兼容层。

## 4. Input Contract

一次用户请求包含：

```json
{
  "report_ref": "ACME_2025",
  "target_year": 2025,
  "kpis": ["revenue", "net_income"]
}
```

- `report_ref`：指向当前 session 已加载年报 artifact 的稳定标识。
- `target_year`：需要提取的财年。
- `kpis`：规范化 KPI key 列表。

ADK Web 用户可以直接把 UTF-8 Markdown 或 PDF 年报附加到消息。PDF 会在插件侧逐页渲染、默认
每 20 页一批通过 OpenAI 兼容的 Unlimited-OCR 服务转成 Markdown。渲染与远端 OCR 按批次流水线执行，
默认最多并行 8 个请求；多页使用 `base` 模式，拆分到单页时使用 `gundam` 模式。批次输出无法可靠按页
映射时递归二分，只在最后必要时退化到单页；批次响应截断时保留已完整返回的页面，只重试未完成尾部。
单页响应达到 8,192 token 输出上限时，只对该页使用 24,576 token 预算重试一次；重试仍截断则拒绝
整份结果，不能保存不完整年报。多页请求超时时同样递归二分为更小批次，单页请求超时时只重试一次；
最终错误必须包含 HTTP 异常类型、超时配置和起始页码，不能因底层异常消息为空而丢失诊断信息。
`ReadError` 等瞬时传输错误先对原请求重试一次；多页请求仍失败时最多额外二分一次，避免网络整体
故障引发指数级请求。OCR HTTP 客户端默认不继承系统代理；只有明确设置
`FINGROUND_OCR_TRUST_ENV=true` 时才读取代理环境变量。

成功的 PDF OCR Markdown 以 PDF SHA-256 和输出相关 OCR 配置的指纹作为缓存键，保存为当前用户范围的
ADK artifact。同一用户跨 session 上传内容相同的 PDF 时直接复用缓存，不跨用户共享；文件名变化不影响
命中。模型、服务地址、渲染 DPI、批大小、页数限制或 `FINGROUND_OCR_CACHE_VERSION` 变化时缓存自动
失效。失败、空内容和截断结果不能写入缓存；缓存读取或写入失败必须降级为正常 OCR，不能阻断上传。

`ReportUploadPlugin` 在 Root 调用模型前将
Markdown/OCR 结果保存为 ADK artifact、从文件名生成稳定 `report_ref`、写入 session manifest，并用一条
只包含 artifact 名称和 report_ref 的短占位文本替换原始附件。完整正文不会进入模型上下文。
程序化调用也可以预先保存 Markdown 或分块 JSONL artifact 并设置相同 manifest：

```json
{
  "report": {
    "report_ref": "ACME_2025",
    "artifact_name": "ACME_2025.md",
    "mime_type": "text/markdown",
    "sha256": "...",
    "total_pages": 214,
    "total_chunks": 487,
    "total_chars": 2361048
  }
}
```

Markdown 使用 `<--- Page Split --->` 标记分页；report tool 在模型上下文之外将每页继续切成不
超过 6,000 字符的稳定 chunk。也继续支持由调用方预先生成的 JSONL chunk 记录：

```json
{
  "chunk_id": "ACME_2025:p72:c1",
  "page": 72,
  "heading": "Consolidated Statements of Operations",
  "text": "..."
}
```

分块保留页码和当前页首个 Markdown 标题；超长页面优先按空段落边界切分，没有合适边界时按
字符硬上限切分。Agent 不读取任意文件路径，也不访问网络。

## 5. Architecture

```text
User
  │
  ▼
root_agent
  ├── TaskCreate
  ├── TaskList
  ├── TaskGet
  ├── TaskUpdate
  └── dispatch_kpi_tasks  ← ADK Workflow tool
         └── parallel_worker(max_parallel_workers=4)
               └── kpi_worker × N  ← isolated ADK task-mode runs
                     ├── GetKpiKnowledge
                     ├── SearchReport
                     ├── ReadReportChunks
                     └── finish_task  ← ADK built-in

TaskProgressPlugin
  ├── observes root task-tool calls
  ├── validates completion claims
  └── reminds root about unfinished tasks

ReportUploadPlugin
  ├── saves Markdown chat attachments as session artifacts
  ├── reuses successful PDF OCR from user-scoped content-addressed artifacts
  ├── replaces full attachment content with a short report reference
  └── initializes the current report manifest

ScopedContextCompactionPlugin
  ├── runs after ADK has filtered context by branch and isolation scope
  ├── replaces old visible contents with a rolling LLM summary
  └── preserves recent tool calls and responses verbatim
```

### 5.1 Root Agent

Root 的唯一职责：

1. 理解请求并按用户给出的 KPI key 规划任务；KPI 是否受支持由 Worker 的知识工具判断。
2. 为每个 KPI 建立一个通用任务，把 Worker 输入放入 `metadata.task_input`。
3. 按顺序将任务标记为 `in_progress` 并设置 owner；任务工具不得并行写 session state。
4. 一次调用 `dispatch_kpi_tasks`，传递所有独立的单 KPI 输入。
5. dispatcher 按输入顺序返回逐项 outcome 后，Root 将成功项的 `result` 原样写入对应任务的
   `metadata.result` 并完成任务；只把失败项返回 `pending`，写入该项的 `metadata.error`。
   只有 dispatcher 本身未能返回任何 outcomes 时，才把整批受影响任务返回 `pending`。
6. 检查是否还有未完成任务。
7. 汇总结果并回答用户。

Root 不拥有以下能力：

- 搜索或读取年报；
- 读取 KPI 知识；
- 选择证据；
- 计算、缩放或修正财务数值；
- 直接创建 KPI 结果。

### 5.2 KPI Worker

`kpi_worker` 配置：

- ADK mode：`task`
- 每次输入：一个 `KpiTaskInput`
- 每次输出：一个 `KpiTaskResult`
- 通过 ADK 自动提供的 `finish_task` 结束任务
- 不与其他 Agent transfer
- 不管理 Root 的 Task Store

固定执行顺序：

1. 调用 `GetKpiKnowledge(kpi_key)`。
2. 根据知识条目构造检索词，调用 `SearchReport`。工具在模型上下文之外扫描全部 chunks，
   只返回数量受限的候选摘要。
3. 对最相关候选调用 `ReadReportChunks`，只读取形成判断所需的少量 chunks。
4. 判断 `found`、`explicit_zero`、`absent` 或 `ambiguous`。
5. 归一化数值并组装证据。
6. 调用 `finish_task` 返回 `KpiTaskResult`。

Worker 不得处理输入范围外的 KPI，也不得凭模型记忆替代 KPI 知识工具。

### 5.3 KPI Dispatcher

`dispatch_kpi_tasks` 是作为 Root 普通 Tool 暴露的 ADK Workflow：

- 输入为非空 `list[KpiTaskInput]`；
- 通过原生 `parallel_worker` 为每个输入建立隔离分支；
- 最大同时运行 4 个 Worker，避免年报扫描和模型请求无限放大；
- Worker 因模型生成非法 tool-call JSON 而抛出 `JSONDecodeError` 时，由 ADK node
  `retry_config` 最多执行 3 次（包含首次执行）；
- 等待所有分支完成，并按输入顺序返回 `list[KpiDispatchOutcome]`；
- 不创建或修改 Task Store，不改变 Worker 结果；
- 一个分支在重试后仍失败时，dispatcher 将该异常转换为该 KPI 的 failed outcome；成功的 sibling
  outcomes 必须保留。dispatcher 自身无法形成 outcome 列表的异常才采用 Workflow fail-fast 语义；
  业务上的缺失或歧义必须由 Worker 返回 `absent` 或 `ambiguous`，不能抛出异常。

Root 只调用一次 dispatcher，因此不依赖模型生成并行 function calls。并发是确定性的 Workflow
行为，而不是 prompt 建议。

Root 和 Worker 的 ADK node 都对模型生成的非法 tool-call JSON 进行有限重试：最多执行 3 次
（包含首次执行），且只匹配 `JSONDecodeError`。Root 重试用于避免任意规划轮次中的瞬时空白或
截断参数直接终止 SSE；Worker 重试耗尽后仍由 dispatcher 转换为对应 KPI 的 failed outcome。

### 5.4 Task Progress Plugin

`TaskProgressPlugin` 是 ADK App 级插件，只处理横切的任务监督逻辑：

- 在 Root 调用任务工具后生成最新的状态摘要；
- 提醒 Root 仍处于 `pending` 或 `in_progress` 的任务；
- 当 Root 准备输出“全部完成”但仍有未完成任务时，阻止该完成声明并要求检查任务列表；
- 记录任务工具调用次数，便于调试。

Plugin 不创建任务、不调用 Worker、不修改 KPI 结果，也不读取年报。

### 5.5 Large-report Context Strategy

大年报通过“全文在工具侧、证据在上下文内”的方式处理：

1. **全文不进上下文**：完整 artifact 只能由 report tools 读取，不能插入 Agent instruction、
   session conversation events 或 Task input。
2. **工具侧全量扫描**：`SearchReport` 可以扫描任意数量的 chunks；全文大小影响工具运行时间，
   但不增加单次模型输入大小。
3. **紧凑候选返回**：一次搜索最多返回 8 个候选，每个候选只包含 chunk id、页码、标题、匹配词
   和最多 600 字符的命中窗口。
4. **受控证据读取**：一次最多读取 3 个已命中的 chunks，单次正文返回总量不超过 18,000 字符。
5. **Task 隔离**：`parallel_worker` 为每个 KPI 使用独立 branch；一个 Worker 读取过的年报
   文本、检索授权和读取预算不会进入另一个 Worker 的上下文。
6. **累计预算**：每个 task 最多调用 60 次 `SearchReport`、40 次 `ReadReportChunks`，所有
   report tool 返回正文累计不超过 480,000 字符。工具在 state 中确定性计数并拒绝超额调用。
7. **分页而非扩容**：搜索结果通过 opaque cursor 分页；Worker 只在当前结果不足以判断时请求
   下一页，不能通过提高 limit 绕过预算。
8. **预算耗尽时保守结束**：如果在预算内无法解决年份、范围、单位或数值歧义，Worker 返回
   `ambiguous`；不得为了继续读取而突破上下文预算。

### 5.6 Long-conversation Context Compaction

所有 LLM Agent 的实际模型请求都通过 `ScopedContextCompactionPlugin`：

1. **先隔离，后压缩**：Plugin 在 ADK 完成 branch 和 `isolation_scope` 过滤后运行，只能看到
   当前 Agent 本来就有权访问的内容。Root 和不同 KPI Worker 的摘要不得互相污染。
2. **Token 阈值**：当当前可见内容的保守估算达到 64,000 tokens 时触发滚动压缩。
3. **保留窗口**：最近约 16,000 tokens 保持原样；压缩切分点只能放在已闭合的 tool-call /
   tool-response 边界，不得产生孤立工具响应。
4. **滚动摘要**：后续请求复用上一次摘要，只在“旧摘要 + 新增原始内容”再次超阈值时调用
   摘要模型，避免每个 tool turn 都额外调用 LLM。
5. **生命周期**：Root 的摘要保存在 session state 中并跨用户轮次复用；task-mode Worker 的摘要
   按隔离 scope 保存，任务结束后立即清理。
6. **失败降级**：摘要模型失败时不改写当前请求，也不丢弃任何原始事件。

不直接使用 ADK 2.6.2 的 App 级 `EventsCompactionConfig`：它的摘要输入覆盖整个 session
event stream，生成的 compaction event 没有当前 `isolation_scope`，会将 Worker 证据泄露到 Root
可见上下文。

Context compaction 只是长会话和长工具链的容量保护，不是读取超长年报的正确性机制。
系统即使关闭 compaction，也必须满足 5.5 的分块和硬预算。

## 6. Data Models

### 6.1 KPI Knowledge

所有 KPI 存放在一个结构化目录中：

```python
KPI_CATALOG: dict[str, KpiKnowledge]
```

每条记录必须包含：

```json
{
  "key": "revenue",
  "name": "Revenue",
  "definition": "Consolidated top-line operating revenue for the fiscal year.",
  "accepted_labels": ["Revenue", "Net sales"],
  "rejected_labels": ["Segment revenue", "Adjusted revenue"],
  "preferred_statements": ["Consolidated Statements of Operations"],
  "retrieval_hints": ["Search the audited annual income statement first."],
  "normalization_rule": "Apply the unit governing the selected row.",
  "cautions": ["Do not use quarterly or segment values."]
}
```

KPI knowledge 是只读数据。任何新增或修改 KPI 都只修改目录和对应目录测试，不增加 Agent。

### 6.2 Task Record

```json
{
  "id": "1",
  "subject": "Extract revenue",
  "description": "Extract revenue from ACME_2025 for fiscal year 2025.",
  "activeForm": "Extracting revenue",
  "status": "pending",
  "owner": null,
  "blocks": [],
  "blockedBy": [],
  "metadata": {
    "task_input": {
      "report_ref": "ACME_2025",
      "target_year": 2025,
      "kpi_key": "revenue"
    }
  }
}
```

任务工具只定义三个持久状态：

```text
pending ──► in_progress ──► completed
```

`TaskUpdate(status="deleted")` 删除任务和其他任务中指向它的依赖边。任务图不包含 KPI 语义，
可通过 `addBlocks`、`addBlockedBy` 和 `owner` 表达通用计划。metadata 更新采用浅合并；值为
`null` 时删除该 key。

### 6.3 Worker Input

```json
{
  "task_id": "1",
  "report_ref": "ACME_2025",
  "target_year": 2025,
  "kpi_key": "revenue"
}
```

### 6.4 Worker Result

```json
{
  "task_id": "1",
  "report_ref": "ACME_2025",
  "target_year": 2025,
  "kpi_key": "revenue",
  "status": "found",
  "value": 1234000000.0,
  "unit": "USD",
  "source_value": "1,234",
  "source_unit": "USD millions",
  "evidence": {
    "chunk_id": "ACME_2025:p72:c1",
    "page": 72,
    "statement": "Consolidated Statements of Operations",
    "label": "Revenue",
    "text": "Revenue  1,234"
  },
  "notes": []
}
```

`status` 只能是：

- `found`：找到明确的非零目标值；
- `explicit_zero`：年报明确披露为零；
- `absent`：完成规定检索后未发现该 KPI；
- `ambiguous`：存在相关证据，但年份、范围、单位或数值无法可靠确定。

当状态不是 `found` 或 `explicit_zero` 时，`value` 必须为 `null`。`found` 必须提供完整
evidence；`absent` 必须在 notes 中说明已检查的范围。

### 6.5 Dispatcher Outcome

成功项：

```json
{
  "task_id": "1",
  "kpi_key": "revenue",
  "status": "succeeded",
  "result": {"...": "KpiTaskResult"},
  "error": null
}
```

重试后仍失败的单项：

```json
{
  "task_id": "2",
  "kpi_key": "capex",
  "status": "failed",
  "result": null,
  "error": "JSONDecodeError: Unterminated string ..."
}
```

Outcome 必须与输入顺序一致。`succeeded` 必须包含且只能包含 `result`；`failed` 必须包含且只能
包含 `error`。

## 7. Tools Required

### 7.1 Root Tools

#### `TaskCreate`

创建通用任务。

输入：必需 `subject`, `description`；可选 `activeForm`, `metadata`
输出：`{"task": {"id", "subject"}}`

任务工具不校验 KPI、不执行去重；这些是 Root 的规划职责。

#### `TaskList`

列出当前 session 的任务。

输入：空对象
输出：所有任务的 `id`, `subject`, `status`, 可选 `owner` 和 `blockedBy`

#### `TaskGet`

读取单个任务的完整记录。

输入：`taskId`
输出：任务的 `id`, `subject`, `description`, `status`, `blocks`, `blockedBy`；不存在时为 `null`

#### `TaskUpdate`

更新通用任务。

输入：必需 `taskId`；可选 `subject`, `description`, `activeForm`, `status`, `addBlocks`,
`addBlockedBy`, `owner`, `metadata`
输出：`success`, `taskId`, `updatedFields`，可选 `error` 和 `statusChange`

`status` 接受 `pending`, `in_progress`, `completed`, `deleted`。业务输入和 Worker 结果分别写入
`metadata.task_input` 与 `metadata.result`。

#### `dispatch_kpi_tasks`

执行一批已经创建并标记为 `in_progress` 的 KPI 任务。

输入：`{"tasks": list[KpiTaskInput]}`，列表不能为空
输出：与输入顺序一致的 `list[KpiDispatchOutcome]`

该 Tool 由 ADK Workflow 提供，内部使用最多 4 路 `parallel_worker`。Root 不得把它与其他 Tool
放在同一轮并行调用；dispatcher 运行期间不写 Task Store。

### 7.2 Worker Tools

#### `GetKpiKnowledge`

读取一个 KPI 的完整知识记录。

输入：`kpi_key`
输出：`KpiKnowledge`

未知 KPI 返回明确错误和支持的 key，不进行模糊匹配。

#### `SearchReport`

在模型上下文之外对当前年报的全部 chunks 执行确定性的文本检索。

输入：`report_ref`, `query`, `cursor`, `limit`
输出：`scanned_chunks`、`total_matches`、按相关度排序的候选摘要以及可选 `next_cursor`

调用前置条件：当前 task 已成功读取其 KPI knowledge。

约束：

- 每次 `limit` 最大为 8；
- 每个命中窗口最大为 600 字符；
- 即使年报大于模型上下文，也必须扫描完整索引或完整 artifact；
- cursor 只表示同一查询的后续结果，不能由模型构造或修改。

#### `ReadReportChunks`

读取指定候选 chunks 的证据正文。

输入：`report_ref`, `chunk_ids`
输出：带 chunk id、页码、标题和正文的记录

约束：一次最多读取 3 个 chunks，返回正文总量最多 18,000 字符；只能读取当前 task 中
`SearchReport` 返回过的 chunk ids。

### 7.3 Authentication

任务工具和 KPI knowledge 工具只访问当前 ADK session state；report tools 通过当前 Runner 的
artifact service 读取 manifest 指定的年报 artifact。它们都不需要外部业务认证。模型认证沿用
ADK 标准环境配置，不由业务代码管理。

运行时配置只从环境变量读取，不在源码中保存模型地址、模型名称或密钥：

- `FINGROUND_MODEL`、`FINGROUND_MODEL_BASE_URL`、`FINGROUND_MODEL_API_KEY`：Root、Worker 和
  上下文摘要共用的 OpenAI-compatible 模型配置，三项均为必需；
- `FINGROUND_OCR_BASE_URL`、`FINGROUND_OCR_MODEL`：PDF OCR 地址与模型，均为必需；
- `FINGROUND_OCR_API_KEY`：OCR 鉴权密钥，可为空以支持无鉴权的本地服务；
- 其余 OCR 超时、并发、批大小、渲染和缓存参数继续使用 `FINGROUND_OCR_*` 环境变量。

仓库只提交 `.env.example` 的非敏感占位配置；`.env` 和所有实际密钥由 `.gitignore` 排除。

## 8. End-to-End Workflow

以请求 `revenue` 和 `net_income` 为例：

1. Root 对两个 KPI key 规划独立工作；Root 本身不读取或校验 KPI 知识。
2. Root 分别创建任务 `1` 和 `2`，业务输入保存在 metadata。
3. Root 按顺序把任务 `1` 和 `2` 更新为 `in_progress`，owner 设为 `kpi_worker`。
4. Root 调用一次 `dispatch_kpi_tasks`，传入两个带 task ID 的单 KPI 输入。
5. Workflow 在隔离分支中并行运行两个 Worker；每个 Worker 读取自己的 KPI knowledge、检索
   年报、读取证据并调用 `finish_task`。
6. Workflow 按输入顺序聚合逐项 outcomes 并返回 Root。
7. Root 把成功 outcome 的 result 原样写入对应任务的 `metadata.result` 并完成任务；
   失败 outcome 只更新对应任务的 `metadata.error` 并返回 `pending`。
8. Plugin 在每次任务工具调用后提醒剩余任务。
9. Root 调用 `TaskList`；只有所有任务都为 `completed` 时才宣称全部完成。
10. 最终回答列出成功结果和仍未完成的任务，不隐藏不完整状态。

## 9. Constraints & Safety Rules

- Root 不能拥有或调用 Worker tools。
- Worker 不能拥有或调用 Root task tools。
- Root 只能通过 `dispatch_kpi_tasks` 运行 Worker，不能直接调用 Worker。
- Worker 必须是 ADK `mode="task"` Workflow node，必须通过 `finish_task` 返回。
- 每次 Worker 调用只允许一个 KPI。
- dispatcher 必须使用 ADK 原生 `parallel_worker`，并限制最大并发数。
- Worker 分支不得直接修改 Root Task Store；Root 的任务更新必须串行。
- Worker 必须先调用 `GetKpiKnowledge`，再调用任何年报工具。
- 工具必须校验 `report_ref`，禁止读取其他 session 或任意文件。
- 完整年报正文不得出现在 Agent instruction、Task input、conversation history 或搜索结果中。
- Report tools 必须在返回前执行单次和累计字符预算；不能依赖模型自行节制。
- 不得把模型记忆、搜索片段或未经读取的页面当作最终证据。
- 不得跨年份、跨列、跨报表范围拼接数值。
- 不得在没有明确单位依据时擅自放大或缩小数值。
- `absent` 和 `ambiguous` 不能伪装成数值 0。
- Root 不得修改 Worker 返回的 KPI 数值或证据。
- Plugin 不得执行业务工作或静默更改任务结果。

## 10. Success Criteria

### Structural

- App 中恰好存在一个 Root Agent、一个 dispatcher Workflow 和一个 `mode="task"` 的 KPI Worker
  定义。
- Root 暴露且仅暴露四个任务工具和一个 `dispatch_kpi_tasks` Workflow Tool。
- Worker 暴露且仅暴露三个业务工具及 ADK 的 `finish_task`。
- 代码库中不存在 KPI-specific Agent、旧兼容层或 benchmark 代码。

### Behavioral

- 一个两 KPI 请求产生两个独立任务。
- 一个多 KPI 请求只调用一次 dispatcher；dispatcher 确实同时运行多个 Worker，最大并发数为 4，
  并按输入顺序返回 outcomes。
- Root 或单个 Worker 的非法 tool-call JSON 最多尝试 3 次；Worker 最终失败不得丢弃已成功的
  sibling outcomes。
- ADK Web 消息中的 Markdown 附件会自动保存为 artifact、初始化 report manifest，并从模型输入
  中移除完整正文。
- 同一用户跨 session 重复上传相同 PDF 时只执行一次 OCR；不同用户不能共享该缓存。
- Worker 尝试在读取 KPI knowledge 前搜索年报时会被工具拒绝。
- Worker 不能读取未经过搜索返回的 chunk。
- 对大于模型上下文窗口的年报，搜索工具仍能扫描全部 chunks，而单次模型可见的 report tool
  输出不超过规定上限。
- 达到单次或累计字符预算后，report tools 必须拒绝继续返回正文。
- 并行 KPI branches 之间不共享之前读取的年报正文、搜索授权或 report tool 预算。
- 非法依赖更新会被拒绝且不改变 session state。
- Plugin 能准确报告所有未完成任务。
- Root 在任务未完成时不能输出“全部完成”。
- 每个 `found` 结果都能定位到具体页码和原始文本。

### Quality

- 至少建立一个单 KPI 正常提取 eval case。
- 至少建立一个 KPI 缺失或证据含糊的 eval case。
- 至少建立一个正文大小超过模型上下文窗口、但目标证据位于年报尾部的 eval case。
- ADK eval 中必须验证最终结果和关键工具轨迹：
  `TaskCreate × N → TaskUpdate(in_progress) × N → dispatch_kpi_tasks →
  TaskUpdate(completed|pending) × N`。

## 11. Edge Cases

- 用户提交未知 KPI key：Worker 的知识工具返回明确错误；Root 将错误写入 metadata，任务保持
  pending，并向用户报告该项未完成。
- KPI 列表包含重复项：Root 去重，保持首次出现顺序。
- session 中没有年报：Worker 返回失败，Root 将错误记入 metadata 并保持任务为 pending。
- `report_ref` 与 session 年报不一致：年报工具拒绝读取。
- 目标年份不在年报中：Worker 返回 absent 或 ambiguous，并说明原因。
- 搜索无结果：Worker 按 KPI knowledge 的检索提示完成必要尝试后返回 absent。
- 年报远大于模型上下文：搜索在工具侧覆盖全文，Worker 只分页获取候选摘要和必要 chunks。
- 目标证据位于最后一个 chunk：不得因只检索年报开头而漏掉。
- 单页或单个表格超过 chunk 上限：加载器继续分块并保留相同页码和结构元数据。
- 候选数量超过单页搜索结果：使用 cursor 读取后续候选，不增加单次返回大小。
- Report tool 累计预算耗尽：Worker 返回 ambiguous，并在 notes 中记录预算耗尽。
- 搜索命中多个年份或多个范围：Worker 读取证据页；无法消除歧义时返回 ambiguous。
- 单位缺失或作用范围不清：不得推断单位，返回 ambiguous。
- Task-mode Worker 生成非法 tool-call JSON：ADK node 最多尝试 3 次；仍失败时只将该项
  错误记入 metadata 并返回 pending，不回滚成功 sibling。
- Root 生成空白、截断或其他非法 tool-call JSON：ADK node 最多尝试 3 次；不得立即终止 SSE。
- 部分 KPI 成功、部分未完成：Root 返回部分结果和明确的失败清单。
- PDF OCR 缓存损坏、不可读或写入失败：忽略缓存并完成正常 OCR 流程；不得缓存失败或截断输出。
- 多页 OCR 超时：自动缩小批次继续执行；单页连续两次超时后返回包含异常类型、超时值和页码的错误。
- OCR 连接中断：原请求有限重试，多页最多额外二分一次；永久 HTTP 错误和非法响应不得盲目重试。

## 12. Minimal Project Layout

```text
finground/
├── __init__.py
├── agent.py              # ADK root_agent and App entry point
├── config.py             # Environment-backed model and service configuration
├── root_agent.py         # Root definition and instruction
├── kpi_worker.py         # Task-mode Worker and schemas
├── kpi_dispatcher.py     # Workflow parallel fan-out/fan-in tool
├── kpi_catalog.py        # Read-only KPI knowledge data
├── report_plugin.py      # Markdown attachment ingestion and report manifest
├── task_store.py         # Task models and four Root tools
├── task_plugin.py        # Progress reminders and completion guard
└── report_tools.py       # Artifact-backed full scan and bounded chunk reads

web/
├── index.html            # 最终用户对话页面
├── styles.css            # 响应式视觉样式
└── app.js                # ADK session、SSE、附件、消息与任务进度交互

tests/
├── test_architecture.py
├── test_task_store.py
├── test_kpi_catalog.py
├── test_report_tools.py
└── test_agent_flow.py
```

除上述文件、项目依赖文件和最小 eval 数据外，不增加其他模块。

## 13. Deferred Decisions

以下事项必须在原型通过评估后单独决策，不在本次实现中默认加入：

- 年报 PDF/HTML 的加载和分页方式；
- Task Store 是否需要跨 session 持久化；
- 是否增加结构化 SEC/XBRL 数据作为辅助来源；
- 部署到 Agent Engine、Cloud Run 或其他环境；
- 生产级日志、监控和成本控制。
