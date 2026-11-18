# Hiking：地形世界模型与深度 Actor

本设计取代 `hiking_imagined_depth_training.md` 作为新训练的入口；旧文档保留描述 depth-conditioned WM 实验。新模式为 `--tgi_depth`，检查点格式为 `native_hiking_tgi_terrain_depth_actor_v1`，从头训练，不加载旧模型。

## 目标与借鉴边界

以 TGI 从 Watch Your Step 接入物理想象训练的结构为参考，保留 Hiking 的机器人、PD 控制、目标到速度指令、原生深度处理和地形课程。物理世界模型使用训练时可查询的地形，部署只需要深度编码器和 Actor。当前不引入 FPM 或动作参考库，不要求机器人踩预先指定的坐标。

TGI 的 FPM 标签来自真实未来触地位置，而不是地形给出的最优落脚点。地形对物理模型的帮助独立于 FPM。本实现只借鉴其物理模型、地形查询和想象回报结构，不声称复现完整 TGI。

## 信息和梯度

- 物理模型 F：输入物理状态、本次动作、上次动作、随机化动力学参数，以及 68 维地形特征；不输入深度、编码器输出或动作参考。输出下一物理状态、接触 logits、力矩和关节加速度。
- 地形特征：机身中心离地高度 1 维，机身周围 7×5 个相对地形高度，左右脚各含中心净空和 5×3 个相对鞋底地形高度。机身范围沿用 TGI 的 x=[-0.3,0.9]、y=[-0.4,0.4]；脚部网格适配原生 G1 鞋底尺寸与 URDF 坐标。
- F 每个预测步通过 FK 重算查询位置，双线性地形查询保留位置和关节梯度。无效查询明确拒绝；策略 rollout 对越界预测吸收、取消奖励和 bootstrap，不把 raster padding 当作地面。
- Actor：8 帧本体历史和深度 z32。深度历史为原生 50 Hz 的 36 帧，按步长 5 取 8 帧。Actor 没有地形特征输入。
- 想象视觉：用预测姿态查询独立 raycast camera，推进完整历史，然后编码。raycast/FK 的相机姿态导数停止，编码器参数导数保留，本体反馈仍通过 F 反传。此处不能声称拥有穿过渲染器的完整动作梯度。
- 编码器：真实 replay 上的 L1 重建辅助学习，加上想象回报的任务梯度。初始 replay 深度也在策略图内重新编码，避免 H1 时完全没有任务梯度。
- 优化器：F、深度编码器/解码器、Actor、Critic 各自唯一拥有一组参数。模型阶段只更新 F 和深度重建；策略阶段冻结 F/Critic，更新 Actor 和深度编码器；Critic 随后用 detached lambda-return 拟合。解码器只接收重建梯度。

## 奖励与训练

真实采集和想象共用 Hiking 任务奖励和几何落脚读出。保留速度跟踪及动作/动力学约束；避免叠加重复的原生脚滑、空中时间和脚平整奖励。落脚要求包括支撑、摆动净空、脚滑和地形支撑场，不提供指定落脚坐标。

想象 H32，gamma=0.99，lambda=0.95；使用物理终止和地形失效屏蔽，以及阈值 0.3 的不可靠触地事件 bootstrap 截断。无 AMP、无教师、无未校准 failure head。

正式 Hiking 训练预算为 14,000 轮，2048 个并行环境。先做 500 轮物理模型/重建预热，再更新 Actor 和编码器；每轮采集 8 个控制步、模型更新 10 次，模型 batch32/H8，策略 batch128/H32。replay128 个控制步，固定 1/8 环境作为模型留出集。每 100 轮写入模型与完整恢复检查点并报告 H1/H8/H16/H32 预测误差。

```bash
source logs/hiking_environment/activate.sh
export CUDA_VISIBLE_DEVICES=GPU-35d42f8f-79ad-6118-f5df-d3f85899947b
export PYGLET_HEADLESS=true
export OMP_NUM_THREADS=2
python -u scripts/instinct_rl/train_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --num_envs 2048 --device cuda:0 --max_iterations 14000 \
  --tgi_depth --model_only_iterations 500 --model_updates 10 \
  --model_batch 32 --actor_batch 128 --collect_steps 8 \
  --output logs/hiking_supermimic/tgi_depth_20261116/hiking_train_2048
```

启动时核验新进程、环境初始化和实际优化更新；之后不自动巡检旧训练。验证启动成功与学会上下楼梯是不同验收阶段：后者需要真实 episode 中的稳定性、目标推进、跟踪误差、脚滑、净空，以及按地形拆分的成功率。重建误差降低不能证明感知已对控制有效，模型误差降低也不能替代真实动作扰动的梯度检查。

## 本轮验证记录

相关单元测试 75 项通过，包括地形坐标梯度、编码器任务梯度、模型/Critic 冻结、原有模式行为。8 环境平地全流程完成，更新次数 F1、Actor1、Critic1、深度2，控制器导出回读逐位一致。8 环境原生 Hiking 全流程也已完成相同更新计数及控制器导出校验。原生 Hiking 全流程诊断和正式启动记录位于 `logs/hiking_supermimic/tgi_depth_20261116/`。

仓库全量 pre-commit 已执行，现有 `scripts/interactive_batch_motion_filter.py` 的已有拼写及其他格式问题导致不通过；自动格式化对无关 tracked 文件的改动已恢复。此次修改单独执行 hooks 并记录结果。

正式 2048 环境进程 PID 1984777 已进入优化；启动核验时完成 23 轮，每轮 F10/深度10，Actor0/Critic0，符合 500 轮预热配置。启动清单位于 `logs/hiking_supermimic/tgi_depth_20261116/training_launch.json`，日志为同目录的 `hiking_train_2048.log`。此后不自动巡检。
