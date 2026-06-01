# exp34 vs exp33 对比

## 概述

- **exp33**: Offline Distillation（离线蒸馏）- 保留所有教师模型生成的响应
- **exp34**: RFT (Rejection Sampling Fine-Tuning) - 只保留 DeepScaler 判定为正确的响应

## 关键差异

| 维度 | exp33 | exp34 |
|------|-------|-------|
| **Rollout 函数** | `examples.offline_distillation.offline_distillation_rollout:generate_rollout` | `examples.rft.rft_rollout:generate_rollout` |
| **每个 prompt 生成** | 4 个候选 | 4 个候选 |
| **Reward 评估** | 无（`reward=0`） | DeepScaler 规则评估 |
| **样本筛选** | 全部保留 | 只保留 `reward >= 0.5` |
| **Over-sampling** | 无（`rollout_batch_size=128`） | 有（`over_sampling_batch_size=256`） |
| **每个 group 样本数** | 固定 4 个 | 可变 1-4 个（只保留正确的） |
| **最终训练样本** | 128 × 4 = 512 样本 | 128 groups × 平均 2 正确 ≈ 256 样本 |
| **数据质量** | 包含错误响应 | 只有正确响应 |

## 参数改动

### ROLLOUT_ARGS

```bash
# exp33
--rollout-function-path examples.offline_distillation.offline_distillation_rollout.generate_rollout
--rollout-batch-size 128
--n-samples-per-prompt 4
# 无 over-sampling-batch-size

# exp34
--rollout-function-path examples.rft.rft_rollout:generate_rollout  # 改
--rollout-batch-size 128
--n-samples-per-prompt 4
--over-sampling-batch-size 256                                      # 新增
```

### RM_ARGS

```bash
# exp33
--rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
--rm-max-concurrent-requests 128
--teacher-model-name ${TEACHER_NAME}
# 无 rm-type，无 rft-reward-threshold

# exp34
--rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
--rm-type deepscaler                                                # 新增
--rft-reward-threshold 0.5                                          # 新增
--rm-max-concurrent-requests 128
--teacher-model-name ${TEACHER_NAME}
```

## 工作流程对比

### exp33 流程

```
For each rollout iteration:
  1. 采样 128 个 prompts
  2. 每个 prompt 调用 teacher 生成 4 个响应（并发 128）
  3. 所有 512 个响应直接用于 SFT
  4. 无筛选，无 reward 计算
```

### exp34 流程

```
For each rollout iteration:
  1. 采样 256 个 prompts（over-sampling）
  2. 每个 prompt 调用 teacher 生成 4 个响应（并发 128）
  3. 对每个响应用 DeepScaler 评估 reward（本地 CPU 计算）
  4. 只保留 reward >= 0.5 的响应
  5. 如果某个 prompt 的 4 个候选都错误，丢弃整个 group
  6. 重复采样直到凑够 128 个 groups（有至少 1 个正确响应）
  7. 最终 ~256 个正确响应用于 SFT
```

## 并发控制

### exp33

```
教师生成：受 rm_max_concurrent_requests=128 控制
Reward 评估：无
```

### exp34

```
教师生成：受 rm_max_concurrent_requests=128 控制（semaphore）
Reward 评估：DeepScaler 本地计算，无限制（CPU bound）
```

## 预期效果

| 指标 | exp33 | exp34 |
|------|-------|-------|
| **训练样本数/rollout** | 512 | ~256（取决于过滤率） |
| **数据质量** | 混合（正确+错误） | 纯正确 |
| **训练速度** | 快（更多样本） | 慢（更少样本） |
| **模型性能** | 可能学到错误模式 | 理论上更好（只学正确的） |
| **过滤率** | 0% | 预计 30-70%（取决于教师质量） |

## 监控指标

在 wandb 中关注：

### exp34 特有指标

- `rollout/rft/prompts_tried`: 尝试了多少个 prompts
- `rollout/rft/candidates_generated`: 总共生成了多少个候选
- `rollout/rft/candidates_correct`: 有多少个候选正确
- `rollout/rft/filter_rate`: 过滤率（越高说明教师质量越差）

### 共同指标

- `eval/aime/pass@8`: AIME 数据集 pass rate
- `train/loss`: SFT loss
- `train/lr`: Learning rate

## 调优建议

### 如果 filter_rate 太高（>70%）

说明大部分 prompts 的所有候选都错误，需要：

1. 增加 `--over-sampling-batch-size`（例如改为 384）
2. 降低 `--rft-reward-threshold`（例如改为 0.3）
3. 检查教师模型质量
4. 增加 `--n-samples-per-prompt`（例如改为 8）

### 如果 filter_rate 太低（<10%）

说明几乎每个 prompt 都有正确答案，可以：

1. 降低 `--over-sampling-batch-size`（节省计算）
2. 提高 `--rft-reward-threshold`（更严格筛选）

## 文件对比

```
exp33:
  /mnt/.../shell/sh/exp33-qwen3-4B.sh
  examples/offline_distillation/offline_distillation_rollout.py

exp34:
  /mnt/.../shell/sh/exp34-qwen3-4B-rft.sh
  examples/rft/rft_rollout.py
  examples/rft/README.md
  slime/utils/arguments.py (+2 新参数)
```

## 运行命令

```bash
# exp33
bash /mnt/tidal-alsh-share2/usr/liuyanjiang601/project/shell/sh/exp33-qwen3-4B.sh

# exp34
bash /mnt/tidal-alsh-share2/usr/liuyanjiang601/project/shell/sh/exp34-qwen3-4B-rft.sh
```
