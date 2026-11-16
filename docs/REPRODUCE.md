# 复现与环境边界

## CPU：公式与历史数值

    python3 -m pip install .
    python3 -m own.scripts.test_reward_rounds --skip-rollout
    python3 -m own.scripts.run_reward_qa --output-dir outputs/qa
    python3 -m own.scripts.run_reward_round_comparison --output-dir outputs/comparison
    python3 -m own.scripts.summarize_eval600

最后一条从九份服务器原始 split summary 汇总 base、iter49、iter99，不重新生成答案。首页展示项目阶段指标，汇总脚本输出附带 sequential-greedy 记录；文件对应与尚待补齐的基线证据见 [实验口径](EXPERIMENTS.md)。

## GPU：历史启动脚本

own/scripts/run-qwen2.5-1.5B-round-smoke.sh 是服务器实际使用的 launcher，默认路径为 /root 布局。SLIME_ROOT、MEGATRON_ROOT、MODEL_DIR、REF_LOAD、PROMPT_DATA、SAVE_DIR、DEBUG_DIR、ROLLOUT_BATCH_SIZE、N_SAMPLES_PER_PROMPT、NUM_ROLLOUT 可配置。

源码框架和模型不在本仓库；需准备兼容版本。Ray runtime env 还保留 /root 作为 own 的导入父目录，换机器时须改成仓库路径。仅在独占训练节点使用默认的 CLEANUP_EXISTING=1；共享节点设置为 0 并确认 Ray 实例归属。

own/scripts/eval_dapo_hf.py 使用模型执行独立生成评测。own/scripts/recompute_eval_jsonl.py 仅重算保存的输出，二者不是同一次生成。

CPU 入口用于 Reward 检查与历史结果汇总；重新训练需另行配置 GPU 环境。
