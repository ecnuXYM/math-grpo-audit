# Round5 Reward QA 实验记录

## 0. 目的

这份记录补充项目中 Reward QA 的实际离线实验, 不只描述方法论。

关于经验文档中 round1 到 round5 reward 演化在本项目 case 簇上的实际对比, 见:

```text
21_reward_round1_to_round5_comparison.md
```

实验问题:

```text
当前 round5 reward 能否把高质量正确回答、可疑正确回答、
错误但格式漂亮回答、明显 hack、重复和截断回答区分开?
```

这不是模型能力评估, 而是 reward 上线前的规则验收。

## 1. 可复跑脚本和产物

脚本:

```text
/root/own/scripts/run_reward_qa.py
```

运行命令:

```bash
python /root/own/scripts/run_reward_qa.py \
  --output-dir /root/own/reports/reward_qa_round5_2026-07-11
```

产物:

```text
/root/own/reports/reward_qa_round5_2026-07-11/reward_qa_cases.csv
/root/own/reports/reward_qa_round5_2026-07-11/reward_qa_summary.json
```

## 2. 实验配置

使用当前 round5 reward:

```text
gamma = 3.0
alpha = 0.5
wrong_base_penalty = -0.05
format_weight = 0.05
length_weight = 0.1
anti_hack_mode = absorbed
```

`absorbed` 表示 hedge、多候选、低信息、重复、hack phrase 等不作为独立 anti-hack 加项, 而是先汇总成 `severity`, 再门控 `gated_accuracy`。

## 3. Case 簇

所有 case 使用同一 label:

```text
42
```

覆盖的回答类型:

| case | 目的 |
|---|---|
| `best_honest` | 正确、完整、清晰、boxed final answer |
| `plain_correct_unboxed` | 正确但未使用 boxed 格式 |
| `suspicious_correct_multi_candidate` | 正确但有 hedge 和多候选倾向 |
| `short_low_information_correct` | 正确但只给最终答案 |
| `long_repetitive_correct` | 正确但明显重复 |
| `wrong_formatted` | 错误但 boxed 格式规范 |
| `hack_wrong_formatted` | 错误且含明显 hack phrase |
| `truncated_correct` | 正确但生成状态标为 truncated |

## 4. 实际结果

本次脚本输出的 reward breakdown 如下。

| case | correct | severity | gated_accuracy | format | length | raw_reward |
|---|---:|---:|---:|---:|---:|---:|
| `best_honest` | True | 0.00 | 1.0000 | 1.00 | 0.00 | 1.0500 |
| `plain_correct_unboxed` | True | 0.25 | 0.4219 | 0.00 | 0.00 | 0.4219 |
| `suspicious_correct_multi_candidate` | True | 1.00 | 0.0000 | 1.00 | 0.00 | 0.0500 |
| `short_low_information_correct` | True | 0.25 | 0.4219 | 1.00 | 0.00 | 0.4719 |
| `long_repetitive_correct` | True | 0.25 | 0.4219 | 1.00 | 0.00 | 0.4719 |
| `wrong_formatted` | False | 0.25 | -0.0578 | -0.25 | 0.00 | -0.0703 |
| `hack_wrong_formatted` | False | 1.00 | -0.5500 | -0.25 | 0.00 | -0.5625 |
| `truncated_correct` | True | 0.25 | 0.4219 | 1.00 | -1.00 | 0.3719 |

所有 8 个 case 的 `answer_parse_ok=True`。

## 5. 验收检查

脚本执行的检查全部通过:

```text
wrong_formatted_nonpositive = true
hack_wrong_formatted_nonpositive = true
best_honest_beats_plain_correct = true
best_honest_beats_suspicious_correct = true
best_honest_beats_short_low_information = true
best_honest_beats_repetitive = true
suspicious_correct_beats_wrong_formatted = true
truncated_correct_below_best_honest = true
wrong_answer_positive_count = 0
```

这证明当前 case 簇下:

```text
1. 错误但格式漂亮的回答没有拿到正 raw reward。
2. 明显 hack 的错误回答被进一步压低。
3. 高质量正确回答高于可疑、低信息、重复和截断正确回答。
4. 多候选和 hedge 会通过 severity 明显压低正确回答的奖励。
```

## 6. 这次实验暴露的边界

这份 QA 同时显示两个重要边界。

第一, `format` 是 correctness-gated 的, 但它仍会影响两个正确低质量 case 的相对分数:

```text
plain_correct_unboxed = 0.4219
short_low_information_correct = 0.4719
```

后者虽然低信息, 但因 boxed 格式得到 `0.05` 小奖励。这不构成错误答案刷正分, 但说明 format 是一个真实的辅助偏好, 不能把 format improvement 当作数学能力提升。

第二, truncated correct 的 raw reward 仍为正:

```text
truncated_correct = 0.3719
```

原因是 length component 对截断只扣 `0.1 * -1 = -0.1`, 不会覆盖正确答案的 gated accuracy。当前设计表达的是:

```text
截断正确回答低于高质量完整回答;
但不是“只要截断就绝对不可学习”。
```

这是一个合理但需要明确的软约束。若未来希望截断样本绝不进入正 reward 区域, 需要把截断改为 correctness gate 或更强的 post-process 约束, 不能只依赖现有 length weight。

## 7. 与训练后 audit 的关系

这次 Reward QA 证明的是规则在人工构造的风险 case 上排序合理。它不能替代训练后的真实 rollout audit。

训练后仍需要检查:

```text
positive_wrong_raw / zero_wrong_raw
severity / format / length 分布
response_len_mean / truncated_rate
高分错误样本回放
fixed eval correct_rate 和 heldout 表现
```

本项目正式 eval 和 rollout audit 中, `positive_wrong_raw=0`、`zero_wrong_raw=0`，与这次离线 QA 的安全线一致。

## 8. 面试说法

> 我没有只写 Reward QA 的原则, 还把它做成了可复跑的 round5 离线 case 簇实验。8 个 case 覆盖高质量正确、无 boxed 的普通正确、hedge 多候选、极短低信息、重复、错误格式漂亮、hack 和截断。当前配置下错误格式和 hack 的 raw reward 都非正, 高质量正确回答排在可疑、低信息、重复和截断回答之前, `wrong_answer_positive_count=0`。同时我保留了一个边界结论: 截断但正确的回答仍可能是正 raw reward, 因为当前 length 是软惩罚; 这说明 Reward QA 不只是证明规则通过, 也能暴露规则还没有覆盖的设计选择。
