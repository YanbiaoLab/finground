# Multi-KPI 优化轨迹

本文是 Multi-KPI agent 的持续变更记录，也是后续优化路线。每次影响 prompt、
工具、上下文、执行流程或评分的修改，都必须新增一条记录，不覆盖历史结果。

## 记录规则

每轮变更遵循同一闭环：

1. 用轨迹或评分结果说明为什么改。
2. 写清假设、代码与 prompt 的具体改动。
3. 先跑单元测试，再跑固定 20 份 2017 报告。
4. 同时记录完成度、调用行为、召回率和准确率；不能只看单一指标。
5. 根据结果决定保留、回滚或继续实验。
6. 每个可评估版本先形成独立 Git commit；评估记录同时写入 commit SHA、prompt
   版本、输出目录和运行参数，保证代码与结果可追溯。

固定回归集为
[`tests/fixtures/ledger/eval-2017-trace-20.txt`](../tests/fixtures/ledger/eval-2017-trace-20.txt)，
正式评估并行度固定为 20。单份或两份 smoke 可以降低并行度用于诊断，但不能替代
固定 20 份回归。只有通过该回归集的改动才进入 76 份 2017 报告评估。

## 当前基线：evidence-v7

评估日期：2026-07-28

模型：`qwen36-27b-fp8`

模型上下文：131,072 tokens

agent 调用上限：每份报告 50 次

输出：
`outputs/ledger/multi-eval-2017-v7-trace20`

| 指标 | 结果 |
| --- | ---: |
| 报告数 | 20 |
| 完成 / incomplete / failed | 0 / 20 / 0 |
| LLM 调用 | 1,000，全部达到 50 次上限 |
| GT / prediction | 551 / 362 |
| matched / wrong / missing / extra | 279 / 60 / 212 / 23 |
| 召回率 | 0.5064 |
| 准确率 | 0.8230 |
| `record_multi_kpi_progress` partial success | 271 |
| 字段校验错误 | 560 |
| `read_report_pages` 调用 | 435 |
| 最大单次 prompt | 24,544 tokens |

作为方向性参照，LEDGER 本地官方复现的全量结果是召回率 0.7141、准确率
0.8055；它与固定 20 份子集不是同一评估范围，不能直接作为 A/B 结论。

### 轨迹结论

- 50 次调用并不少，但没有形成稳定的“读取 → 记录 → 收口 → 提交”状态机。
  20 份报告全部耗尽预算，说明主要问题是流程失控，而不是简单增加调用次数。
- 560 个字段校验错误和 271 次 partial success 消耗了大量调用。agent 会在当前证据
  仍然有效时反复提交不完整或不对齐的行，而不是按工具返回逐行修复。
- 435 次页面读取明显偏高。轨迹中存在重复读取和重复检索，证明上下文压缩只控制了
  token 体积，没有充分保留“已经做过什么、下一步必须做什么”的工作记忆。
- 最大 prompt 只有 24,544 tokens，远低于 128k 上限。本轮完成度低不是上下文窗口
  超限导致，而是压缩后的执行状态不够明确。
- 当前准确率尚可但召回率低，主要损失来自未完成覆盖、债务/研发/现金含受限现金/
  流通股等 note 型指标，以及在预算末端没有完成提交。

## 已实施变更

### ARCH-001：按职责拆分 agent 模块

日期：2026-07-28

状态：已实施

为什么改：

- 原 `finground/agent.py` 同时包含模型配置、Needle prompt、Multi-KPI prompt 和两个
  agent 工厂。修改 Multi-KPI 时需要理解并承担 Needle 与模型配置的连带风险。
- 后续将频繁迭代 Multi-KPI prompt 和工作流，需要让相关代码集中在一个明确边界内。

怎么改：

- `finground/agents/common.py`：模型与共享配置。
- `finground/agents/needle.py`：Needle prompt、预算回调与 agent 工厂。
- `finground/agents/multi_kpi.py`：Multi-KPI 常量、prompt、agent 与 App 工厂。
- `finground/agents/__init__.py`：公开 API。
- 删除原 `finground/agent.py`；所有调用方直接使用 `finground.agents` 下的新模块。
- benchmark runner 改为直接依赖对应的新模块。

结果：

- 这是纯结构调整，没有改 prompt、工具列表、模型参数、调用预算或上下文逻辑。
- 修改前测试基线：109 passed。
- 修改后：109 passed，`ruff check finground tests` 通过。
- 结构调整本身不重复运行模型评估，因为模型可见 prompt、工具和参数均未改变。

### FLOW-001 / CTX-001 / REPAIR-001：显式工作流与有限修复

日期：2026-07-28

状态：两例 smoke 通过，等待固定 20 份评估

prompt 版本：`evidence-v8`

为什么改：

- v7 的 20 份报告全部 incomplete，每份都耗尽 50 次调用。
- `NASDAQ_APA_2017` 在元数据和同一页之间振荡，覆盖数为 0。
- `NASDAQ_ORLY_2017` 会连续提交相同字段错误或相同成功记录，无法进入新来源。
- 第 45 次才开始 closure，而最坏仍需 4 次覆盖记录、1 次查询和 1 次提交，窗口不足。

怎么改：

- 始终保留紧凑的 `get_report_info` 结果，不再随旧页面一起压缩。
- execution guard 每轮注入 authoritative workflow state，并通过 ADK tool schema 表达
  metadata、source read、checkpoint、repair、recovery、closure 和 submit 阶段。
- 同一覆盖数下对完全相同的 read、search、record 和 progress query 去重；发生重复后
  临时移除该工具，迫使 agent 换来源或收口。
- `partial_success` 返回 `accepted_kpis` 和逐 KPI `repair_queue`。同一错误只强制修复
  一次，仍失败则释放当前来源，允许换来源或记录 ambiguous。
- closure 从第 45 次提前到第 40 次，留下至少 10 次调用完成覆盖和提交。
- 总搜索上限设为 7，与“最多 3 个缺失主表搜索 + 4 个 grouped note cycles”一致。

两例结果分别取最终有效运行：

- APA：`outputs/ledger/multi-eval-2017-v8-smoke2-r5`
- ORLY：`outputs/ledger/multi-eval-2017-v8-orly-r6`

| 指标 | v7 同两例 | v8 同两例 | 变化 |
| --- | ---: | ---: | ---: |
| complete / incomplete / failed | 0 / 2 / 0 | 2 / 0 / 0 | 完成度 0% → 100% |
| 总 LLM 调用 | 100 | 69 | -31% |
| 平均 LLM 调用 | 50.0 | 34.5 | -15.5 |
| read 调用 | 45 | 18 | -60% |
| partial success | 23 | 14 | -39% |
| validation errors | 25 | 24 | -4% |
| matched / wrong / missing / extra | 17 / 0 / 39 / 0 | 26 / 3 / 27 / 1 | matched +9 |
| 召回率 | 0.3036 | 0.4643 | +0.1607 |
| 准确率 | 1.0000 | 0.8966 | -0.1034 |

单例诊断：

- APA 从覆盖 0、50 次耗尽，变为覆盖 31、28 次提交；召回率 0.1923。它是只有摘要
  页的异常短报告，仍是后续 source-quality 优化样例。
- ORLY 覆盖 31、41 次提交，召回率 0.7000、准确率 0.9545；成功读取 9 个页面，
  现金流、capex、折旧摊销等指标不再被流程循环挤掉。

结论：

- 保留显式阶段、动作去重、40 次 closure 和 7 次搜索上限。
- FLOW、CTX、CLOSE 的 smoke 验证达到目标；尚不能在 20 份结果出来前标记最终完成。
- validation errors 仅下降 4%，下一轮主目标转为 EVID-001：减少行、年份和单位字段
  对齐错误，而不是继续增加流程限制。

## 后续优化路线

| ID | 优先级 | 假设 | 预期验证信号 | 状态 |
| --- | --- | --- | --- | --- |
| FLOW-001 | P0 | 用代码维护显式阶段与下一动作，可阻止模型重复读取并保证收口 | incomplete 降到 0；平均调用低于 40 | smoke 达标，待 20 份 |
| REPAIR-001 | P0 | 将工具校验错误压缩成逐行、可执行的修复队列，可减少无效重试 | validation errors 和 partial success 至少下降 60% | 部分实施，未达目标 |
| CTX-001 | P0 | checkpoint 应保留覆盖矩阵、已查来源、失败原因与下一动作，而非只保留 KPI 结果 | 重复 read/search 至少下降 50% | smoke 达标，待 20 份 |
| EVID-001 | P1 | 给表格单元格稳定 source ID，并让工具从 source ID 解析行/年/单位，可减少字段错位 | wrong 下降且准确率不低于当前基线 | 待实施 |
| NOTE-001 | P1 | 按来源组处理债务、现金、股数和费用 note，可提升长尾 KPI 召回 | 对应 KPI 的 missing 显著下降 | 待实施 |
| CLOSE-001 | P1 | 在剩余预算阈值处由代码限制动作并强制覆盖收口，可避免耗尽 50 次仍未完成 | 每份报告都有最终 submission | smoke 达标，待 20 份 |

原则仍然是单个 ADK agent：路线中的状态机、checkpoint 和动作约束都作为 agent
运行时机制实现，不拆分子 agent，也不绕过 agent 直接调用 LLM。

## 固定评估命令

```bash
uv run finground ledger-multi \
  --parquet data/ledger/raw/multi_kpi/eval/data.parquet \
  --reports-file tests/fixtures/ledger/eval-2017-trace-20.txt \
  --output-dir outputs/ledger/<version>-trace20 \
  --concurrency 20

uv run finground ledger-score-multi \
  --output-dir outputs/ledger/<version>-trace20 \
  --parquet data/ledger/raw/multi_kpi/eval/data.parquet
```

## 后续变更模板

```markdown
### <CHANGE-ID>：<标题>

日期：
状态：实验中 / 保留 / 回滚
代码版本：
prompt 版本：

为什么改：
- 轨迹证据：
- 失败样例：
- 假设：

怎么改：
- 代码：
- prompt：
- 工具/上下文/流程：

结果：
| 指标 | 基线 | 当前 | 差值 |
| --- | ---: | ---: | ---: |
| complete / incomplete / failed | | | |
| 平均 LLM 调用 | | | |
| validation errors | | | |
| recall | | | |
| precision | | | |

结论：
- 保留 / 回滚：
- 下一步：
```
