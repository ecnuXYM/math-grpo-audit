# 实验指标与记录对应

## 首页展示口径

项目阶段记录：MATH eval600 从 289/600（48.17%）到 370/600（61.67%），净增 81 题；正式 rollout audit 的 positive-wrong-raw 为 0。此行为阶段汇总记录，附带文件的可复算范围如下。

## 仓库附带的复测摘要

`evidence/server_eval_summaries/` 保留九份服务器 split summary，`python3 -m own.scripts.summarize_eval600` 对这些已保存文件做汇总，输出为：

| 记录 | 汇总正确数 | 准确率 |
|---|---:|---:|
| Base sequential-greedy | 346/600 | 57.67% |
| iter49 sequential-greedy | 370/600 | 61.67% |
| iter99 sequential-greedy | 253/600 | 42.17% |

这组复测文件包含 eval200 与 heldout400，eval200 参与 checkpoint 选点。它与首页的阶段基线分别记录；当前选入的原始 summary 不能直接复算 289/600，也尚未完成两个基线的逐样本和生成配置对应。不能仅用上述九份文件证明 289→370 来自同一次同协议对照。

汇总脚本按原始摘要计算。后续补齐 289/600 对应的生成结果、模型 checkpoint、数据哈希与评测配置后，可增加完整复算入口。

## Reward QA 口径

round1–5 是同一固定风险集上的离线 Reward 结构对照。positive-wrong-raw=0 说明所检查的错误轨迹没有获得正 raw reward，不代表所有未来输出均无投机风险。

固定 QA、正式 rollout 审计与模型准确率是三类不同结果，报告时保留对应范围。
