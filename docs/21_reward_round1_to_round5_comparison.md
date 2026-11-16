# 项目化 Reward Round1-5 离线对比实验

## 0. 结论

这次对比在同一批项目 Reward QA case 上运行了五种 reward 结构。结果显示:

```text
round1:
  2 个错误样本拿到正 raw reward。

round2:
  仍有 1 个错误但格式漂亮样本拿到正 raw reward。

round3 / round4 / round5:
  错误样本正 raw reward 数量均为 0。
```

因此, 在当前项目的 case 簇上, 从线性 accuracy + format 过渡到 severity 进入主正确性项, 是堵住“错误格式样本刷正分”漏洞的关键变化。

这是一份**离线 reward-design ablation**, 不是五次完整 GRPO 长训。它回答的是“哪种 reward 结构值得进入训练”, 不直接证明哪一轮训练后的 eval600 一定最高。

## 1. 实验范围与可复跑方式

脚本:

```text
/root/own/scripts/run_reward_round_comparison.py
```

运行命令:

```bash
python /root/own/scripts/run_reward_round_comparison.py \
  --output-dir /root/own/reports/reward_round_comparison_2026-07-11
```

产物:

```text
/root/own/reports/reward_round_comparison_2026-07-11/round_comparison_cases.csv
/root/own/reports/reward_round_comparison_2026-07-11/round_comparison_summary.json
```

本实验复用当前项目的:

```text
answer extraction / equivalence 判断
severity 特征
format 判断
length / truncated 定义
wrong_base_penalty = -0.05 (round5)
```

只改变 reward 的组合结构。因为 `/root/GRPO实验的经验与坑.md` 描述的是另一条实验历史, 并没有提供可直接加载的各轮源码, 下列 round1-4 是根据其公式做的**项目化近似实现**，不是对其历史数值的声称性复现。

## 2. 五轮定义

| round | 项目化实现 |
|---|---|
| round1 | `accuracy + 0.05 * unconditional_format_pass` |
| round2 | `accuracy + 0.05 * unconditional_format_pass - 0.15 * severity + 0.1 * length` |
| round3 | `gated_accuracy(gamma=3) + 0.05 * correctness_gated_format - 0.15 * severity + 0.1 * length` |
| round4 | `symmetric_gated_accuracy(gamma=1.5) + 0.05 * correctness_gated_format - 0.15 * severity + 0.1 * length` |
| round5 | 当前项目 reward: `gamma=3`, `wrong_base_penalty=-0.05`, `anti_hack_mode=absorbed` |

round1/2 的 `unconditional_format_pass` 故意模拟早期线性设计: 只要格式合格就有格式加分, 不先检查答案正确性。

round3/4 的 `correctness_gated_format` 使用当前项目的格式门控:

```text
correct + format pass -> 1.0
correct + format fail -> 0.0
wrong + format pass   -> -0.25
wrong + format fail   -> 0.0
```

round5 另外保留当前 Slime post-process:

```text
raw_reward <= 0 -> final_reward <= 0
```

## 3. 固定 QA Case 簇

8 个 case 都用 label `42`:

```text
best_honest
plain_correct_unboxed
suspicious_correct_multi_candidate
short_low_information_correct
long_repetitive_correct
wrong_formatted
hack_wrong_formatted
truncated_correct
```

它们覆盖正确性、格式、hedge、多候选、低信息、重复、明显 hack 和截断。

## 4. 逐 Case 原始 Reward 对比

| case | correct | severity | length | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best honest | T | 0.00 | 0.00 | 1.0500 | 1.0500 | 1.0500 | 1.0500 | 1.0500 |
| plain correct, unboxed | T | 0.25 | 0.00 | 1.0000 | 0.9625 | 0.3844 | 0.6120 | 0.4219 |
| suspicious correct | T | 1.00 | 0.00 | 1.0500 | 0.9000 | -0.1000 | -0.1000 | 0.0500 |
| short low-information correct | T | 0.25 | 0.00 | 1.0500 | 1.0125 | 0.4344 | 0.6620 | 0.4719 |
| repetitive correct | T | 0.25 | 0.00 | 1.0500 | 1.0125 | 0.4344 | 0.6620 | 0.4719 |
| wrong formatted | F | 0.25 | 0.00 | 0.0500 | 0.0125 | -0.0578 | -0.1125 | -0.0703 |
| hack wrong formatted | F | 1.00 | 0.00 | 0.0500 | -0.1000 | -0.6625 | -0.6625 | -0.5625 |
| truncated correct | T | 0.25 | -1.00 | 1.0500 | 0.9125 | 0.3344 | 0.5620 | 0.3719 |

## 5. 汇总结果

| round | wrong answer positive count | best-honest minus risk gap | wrong formatted | hack wrong formatted |
|---|---:|---:|---:|---:|
| round1 | 2 | 1.0000 | +0.0500 | +0.0500 |
| round2 | 1 | 1.0938 | +0.0125 | -0.1000 |
| round3 | 0 | 1.4102 | -0.0578 | -0.6625 |
| round4 | 0 | 1.4375 | -0.1125 | -0.6625 |
| round5 | 0 | 1.3664 | -0.0703 | -0.5625 |

`best-honest minus risk gap` 中的 risk 是 `wrong_formatted` 与 `hack_wrong_formatted` 的平均 reward。

## 6. 结果解释

### 6.1 round1 和 round2 的线性可补偿漏洞被实际复现

round1 的格式奖励只看格式是否通过, 因此两个错误但 boxed 的 case 都获得:

```text
+0.05 raw reward
```

round2 加入 severity 和长度后, 明显 hack 已被压到负分, 但普通的 `wrong_formatted` 仍为:

```text
+0.0125 raw reward
```

这正是经验文档所说的线性可补偿问题: 格式正项仍能把低 severity 的错误样本抬到正区。

### 6.2 round3 开始, severity 进入主项后漏洞被堵住

round3 对错误样本不再使用单纯的 0/1 accuracy, 而是让 severity 进入 gated accuracy。结果:

```text
wrong_formatted:
  +0.0125 (round2)
  -> -0.0578 (round3)
```

`wrong_answer_positive_count` 由 1 降为 0。这说明主项门控比继续增大线性 anti-hack 权重更根本。

### 6.3 round4 的风险间隔最大, 但不能据此宣布它训练效果最好

在这个小 case 簇上, round4 的 best-honest minus risk gap 是 1.4375, 略高于 round5 的 1.3664。

但这不代表 round4 一定优于当前 round5:

```text
round4:
  使用显式 anti-hack 和 gamma=1.5。

round5:
  使用当前项目 gamma=3、wrong_base_penalty=-0.05、absorbed severity,
  并额外拥有 group normalization 后的非正约束和完整 breakdown 落盘。
```

这次实验只证明 case 簇内的规则排序，不证明实际 GRPO 训练后的泛化曲线。

### 6.4 round5 的工程安全线单独通过

用 `best_honest / wrong_formatted / hack_wrong_formatted` 构造同组后, round5 的 post-process final reward 是:

```text
best_honest:          +0.9109
wrong_formatted:      -0.2094
hack_wrong_formatted: -0.7016
```

因此两个 raw reward 非正的错误样本在 group normalization 后仍未被翻成正训练信号。

## 7. 当前 round5 的两个可见边界

第一, `suspicious_correct_multi_candidate` 在 round5 中仍为小正值:

```text
round5 raw reward = +0.0500
```

这是 `anti_hack_mode=absorbed` 的直接结果: severity 把正确主项压到 0, 但 boxed 格式仍提供 `+0.05`。这符合“正确但可疑低于 best honest”的排序，但也说明它不是硬拒绝。

第二, `truncated_correct` 仍为正值:

```text
round5 raw reward = +0.3719
```

当前 length 是软惩罚。它让截断正确样本低于高质量正确样本，但不把截断直接设为不可学习样本。

## 8. 验收检查

脚本检查全部通过:

```text
round1_exposes_wrong_formatted_positive = true
round2_exposes_wrong_formatted_positive = true
round3_blocks_wrong_formatted_positive = true
round4_blocks_wrong_formatted_positive = true
round5_blocks_wrong_formatted_positive = true
round5_nonpositive_constraint_holds = true
```

## 9. 下一步边界

这五轮离线对比是启动训练前的 reward 选择证据。若要比较五轮训练效果, 后续应该只挑通过 QA 的 round3/4/5, 在相同 seed、相同数据、相同 rollout 数、相同 checkpoint 间隔下分别训练，再用同一套 fixed eval 和 heldout 评估。

不建议为 round1/2 开正式长训, 因为它们已经在离线 QA 中出现错误答案正 reward。
