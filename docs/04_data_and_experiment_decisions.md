# 数据选择与实验决策记录

这份文档帮助你回答:

- 为什么选这个数据集?
- 为什么选这个模型?
- 为什么不是 DAPO / GSM8K / 7B?
- 什么时候可以进入更长训练?

## 1. 后训练任务的数据不是越难越好

GRPO 依赖同一 prompt 下多次采样的相对 reward。

因此数据难度要匹配模型初始能力:

```text
太难:
  多数 sample 错
  多数 group 全错
  GRPO 学不到正确行为

太简单:
  多数 sample 对
  多数 group 全对
  advantage 接近 0

中等:
  同一题经常有对有错
  有 mixed group
  适合 GRPO
```

## 2. 为什么 DAPO 不适合 1.5B 起步

DAPO 64 sample 结果:

```text
samples = 64
correct = 4
correct_rate = 6.25%
truncated = 9 / 64 = 14.06%
response_len_mean = 612.48
positive_wrong_raw = 0
zero_wrong_raw = 0
```

group 结构:

```text
32 groups
只有 3 组出现过正确答案
只有 2 组是 mixed
29 组全错
```

判断:

```text
这不是 reward 主问题, 是 1.5B 对 DAPO 长推理题初始能力不足。
```

面试回答:

> 我没有直接把 DAPO 低正确率归因到 reward, 而是构造 simple math sanity 和后续 MATH 分级实验来拆分模型能力、数据难度和 reward 抽取问题。

## 3. 为什么 simple math / GSM8K 又不适合

simple math 修复抽取后:

```text
correct = 63 / 64
correct_rate = 98.44%
truncated = 0
```

GSM8K 结果:

```text
Qwen2.5-7B GSM8K: 90%+
Qwen3-4B GSM8K: 96%+
```

判断:

```text
这些适合 pipeline sanity check, 不适合继续做 GRPO 训练。
```

原因:

```text
多数 group 全对
组内 reward 差异小
advantage 接近 0
训练 loss 可能接近 0
```

## 4. 为什么选 MATH l3_4_core + Qwen2.5-1.5B

这个组合的意义:

```text
1.5B 还有提升空间
题目不是太简单
correct_rate 落在可训练区间
截断低
reward 安全线可控
```

已观察结果:

```text
Qwen2.5-1.5B + MATH l3_4_core, 32p2 len1536:
correct_rate 约 37%~53% 之间, 取决于抽样和采样温度
truncated 很低
mixed group 有一定数量
```

Slime v2 smoke:

```text
correct = 24 / 64 = 37.5%
truncated = 1 / 64 = 1.56%
groups:
  all_correct = 8
  mixed = 8
  all_wrong = 16
positive_wrong_raw = 0
zero_wrong_raw = 0
```

10 rollout 小训:

```text
total samples = 640
overall correct = 238 / 640 = 37.19%
truncated = 7 / 640 = 1.09%
positive_wrong_raw = 0
zero_wrong_raw = 0
```

旧数据池 16x4 20 rollout:

```text
total samples = 1280
overall correct = 656 / 1280 = 51.25%
mixed group = 168 / 320 = 52.5%
truncated = 35 / 1280 = 2.73%
positive_wrong_raw = 0
zero_wrong_raw = 0
```

旧数据池固定 greedy eval:

```text
前 128 题:
before = 64 / 128 = 50.00%
after  = 70 / 128 = 54.69%

offset128 后 22 题:
before = 9 / 22 = 40.91%
after  = 13 / 22 = 59.09%

合计 150 题:
before = 73 / 150 = 48.67%
after  = 83 / 150 = 55.33%
净提升 = +10 题, +6.67 points
raw_reward_mean = 0.1814 -> 0.2081
truncated_rate = 2.00% -> 2.00%
```

但这轮后续 held-out seed 检查不稳定:

```text
heldout seed2026:
before = 115 / 200 = 57.50%
after  = 118 / 200 = 59.00%

heldout seed3030:
before = 120 / 200 = 60.00%
after  = 115 / 200 = 57.50%

combined:
before = 235 / 400 = 58.75%
after  = 233 / 400 = 58.25%
```

因此更严谨的做法是重新切 clean train/eval split, 从 base model 再训:

```text
train1000:
  /root/own/data/math_train_l3_4_core_train1000_seed4040.jsonl
eval200:
  /root/own/data/math_train_l3_4_core_eval200_seed4040.jsonl
```

clean split 16x4 20 rollout, latest recompute:

```text
训练 rollout:
total samples = 1280
overall correct = 665 / 1280 = 51.95%
mixed group = 155 / 320 = 48.44%
truncated = 37 / 1280 = 2.89%
positive_wrong_raw = 0
zero_wrong_raw = 0

clean greedy eval200:
before = 109 / 200 = 54.50%
after  = 113 / 200 = 56.50%
净提升 = +4 题, +2.00 points
raw_reward_mean = 0.1770 -> 0.1969
truncated_rate = 4.50% -> 2.50%
```

结论:

```text
可以作为后训练工程验证和短训数据。
旧数据池 in-pool 提升说明训练有正向信号, 但 held-out 不稳定。
clean split 重新训练后, 非重叠 eval200 上有小幅正向提升。
继续扩大到 50 rollout 后, clean eval200 回落到 53.00%, 说明必须用固定 eval 控制过训风险。
```

## 5. 为什么不是 4B / 7B

7B 在很多较简单数据上正确率高, 后训练提升空间小。

4B 在 GSM8K 上也偏高, 在 MATH 低级别上可能过简单。

对于项目目标:

```text
不是追求最高数学正确率
而是验证 reward + Slime GRPO 闭环
```

因此 1.5B 更合适:

- 成本低。
- 训练更快。
- 有明显不会做的空间。
- 中等难度数据能产生 mixed group。

面试回答:

> 我选择 1.5B 不是因为它最强, 而是因为它更适合验证 GRPO 信号。太强的模型在简单数据上大多全对, 反而没有组内 advantage。

## 6. 为什么要用 `16 prompts x 4 samples`

当前 `32x2`:

```text
32 个 prompt
每题 2 个回答
总样本 64
```

问题:

```text
每题只有两个 sample, mixed group 概率有限。
```

改成 `16x4`:

```text
16 个 prompt
每题 4 个回答
总样本仍然 64
```

优势:

```text
同题采样更多
更容易出现对有错
GRPO 组内信号更强
```

实测结果:

```text
32x2 10 rollout:
mixed group = 98 / 320 = 30.6%

16x4 20 rollout:
old pool mixed group = 168 / 320 = 52.5%
clean split mixed group = 155 / 320 = 48.44%
```

代价:

```text
每个 rollout 覆盖题目更少
需要固定 eval 补充泛化判断
```

## 7. 为什么 temperature 要调低

当前 Slime 默认实验曾用:

```text
rollout-temperature = 1
```

问题:

```text
采样更散
错误率和输出格式波动更大
```

建议:

```text
temperature = 0.7 或 0.8
```

原因:

```text
保留一定多样性
但减少过度发散
更容易产生稳定 mixed group
```

## 8. 长训前的门槛

不建议直接跳到 100 rollout。clean split 16x4 20 rollout 后已经验证过 50 rollout, 结果回落, 所以当前应停止单纯扩大 rollout。

长训前应满足:

```text
固定 eval set 上前后可比
短训 10~20 rollout 没有 reward hacking
16x4 mixed group 明显增加
truncated_rate < 5%
raw_reward_mean 不下降
response_len 不明显变长
```

如果短训后:

```text
fixed eval correct_rate +3%~5%
raw_reward_mean 上升
positive_wrong_raw = 0
zero_wrong_raw = 0
```

当前已经观察到:

```text
clean eval200 correct_rate +2.00 points
raw_reward_mean 0.1770 -> 0.1969
truncated_rate 4.50% -> 2.50%
positive_wrong_raw = 0
zero_wrong_raw = 0
```

50 rollout 后的更新判断:

```text
保留 iter19 作为当前最佳 checkpoint
不要直接 NUM_ROLLOUT=100+
做 checkpoint curve 和 correct_to_wrong 错误分析
如果再训, 优先调 lr / KL / early stopping
```

关键不是单次提升有多大, 而是扩大 rollout 后是否仍然:

```text
correct_rate 不回落
raw_reward_mean 不下降
truncated_rate 不上升
positive_wrong_raw = 0
zero_wrong_raw = 0
response_len 不明显变长
```

## 9. 固定 eval 的必要性

训练内 rollout 开了:

```bash
--rollout-shuffle
```

因此:

```text
rollout_0 和 rollout_9 不是同一批题
正确率波动不能直接解释为模型变好或变坏
```

固定 eval 应该:

```text
固定数据文件
固定 num_prompts
固定 offset 或 seed
固定采样参数
训练前跑一次
训练后跑一次
```

这样才能回答:

```text
后训练是否真的提升了模型在同一评估集上的表现?
```

## 10. 可以对外讲的实验结论

当前版本:

> 我完成了 Slime GRPO 后训练链路和 reward 审计闭环。实验表明 DAPO 对 1.5B 太难, simple math / GSM8K 又太简单, MATH l3_4_core 更适合作为中等难度训练集。10 rollout 证明训练链路和 reward 安全线稳定; 旧 16x4 20 rollout 在训练池内固定 eval 上从 48.67% 提升到 55.33%, 但 held-out seed 不稳定。随后我切了 clean train1000/eval200, 从 base 重新训练 20 rollout, 非重叠 eval200 从 54.50% 到 56.50%; 扩到 50 rollout 后回落到 53.00%。所以当前结论不是“越训越好”, 而是 20 rollout 附近可能是更好的 early checkpoint。

不要说:

```text
我已经训练出了更强数学模型
10 rollout 正确率某一轮到 46.88%, 所以训练有效
```

应该说:

```text
训练链路有效
reward 安全性有效
数据选择逻辑有效
16x4 20 rollout 已在 clean eval 上显示小幅正向效果
50 rollout 未继续提升, 后续应做 checkpoint curve 和错误分析
```
