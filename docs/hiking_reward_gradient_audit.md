# Hiking TGI 奖励梯度检查

当前正式训练的 `tgi_aligned` 分支在真实采集与想象中共用 `tgi_core_reward`，不使用原生 RewardManager 的旧奖励作为策略目标。原生 `Episode_Reward/rewards_*` 日志仍保留，不能据其名称判断当前优化的奖励。

新增评估参数 `--reward_gradients`，适用于 TGI-aligned 检查点。评估过程在真实控制步 1、20、60 的当前状态上，检查固定动作序列 H1/H8/H32。若第一回合全部提前结束，后续检查点不会生成。每个奖励项保存动作梯度、首动作梯度、三种方向的有限差分和均值；三种方向为总奖励梯度、固定种子随机方向、只扰动第一个动作的随机方向。每个环境分别归一化方向，避免环境间梯度抵消掩盖问题。

`total` 是不含生存权重的折扣奖励；`survival_total` 额外包含 soft termination 的生存乘积；`final_survival` 与 `final_terrain_supported` 是诊断项，不属于奖励。梯度检查不更新参数或归一化统计，并逐元素确认 F 的 state_dict 未变化。Actor 反馈、Critic bootstrap、似然比和落脚事件截断未计入本检查，因此该结果不能代表完整策略目标的梯度，更不能证明相对于真实仿真的模型梯度准确。

地形 `exact=True` 的前向取最近邻高度，反向用双线性坐标梯度。这是显式代理梯度，地形奖励与有限差分不一致不自动视为 bug。脚底 `amin`、L1、ReLU 和边界 mask 也有折点，需要结合扰动尺度及状态判断。命令决定的站立/移动开关不依赖本次动作，本身不会截断状态到动作的梯度。

本次冻结第 900 轮检查点，8 个原生 Hiking 环境，无动作探索噪声，保留观测噪声和初始状态随机化。评估使用 seed1654，与训练 seed1652 不同；地形由评估场景重新生成，不宣称为训练回放原位置的重放。正式 2048 环境训练保持运行，未改奖励或重启。独立小批诊断与训练共用 GPU，运行期间可能暂时影响训练吞吐。

```bash
source logs/hiking_environment/activate.sh
export CUDA_VISIBLE_DEVICES=GPU-35d42f8f-79ad-6118-f5df-d3f85899947b
export PYGLET_HEADLESS=true
export OMP_NUM_THREADS=2
python -u scripts/instinct_rl/evaluate_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --num_envs 8 --device cuda:0 \
  --checkpoint logs/hiking_supermimic/tgi_speed_fresh_20261116/hiking_train_2048/model_900.pt \
  --terrain native_hiking --steps 320 --reward_gradients \
  --output logs/hiking_supermimic/reward_gradient_audit_20261116/native_hiking
```

JSON 分项汇总和 PT 原始导数保存在输出目录，真实首回合评估保存在 `results.json` 和 `trace.pt`。该目录已存在时需为复测指定新目录。

验证：22 项相关测试通过，覆盖解析梯度、被 detach 的错误路径、代理梯度与有限差分差异、原生奖励状态读取及现有地形/评估检查。修改文件 hooks 通过；全仓 hooks 仍有既有格式/拼写问题，无关 formatter 改动已按执行前快照恢复。

## 本次结果

9 组检查全部完成（3 个真实控制步 × 3 个想象窗口），F 权重和归一化状态均未改变，优化器更新数为 0。原始结果目录为 `logs/hiking_supermimic/reward_gradient_audit_20261116/native_hiking/`，机器可读汇总为上一级 `summary.json`。

没有发现总奖励的梯度断路：三组快照的 H1/H8 总奖励均有动作梯度，小扰动下总奖励的方向符号一致率为 100%。站立姿态及站立接触奖励为零，是这批移动命令的门控结果；未触发关节限位时，限位惩罚梯度为零同样符合定义，不能据此认定 detach。

H32 的主要数值差分差异集中在 `locomotion_support`。epsilon=0.001 时，三组总奖励的导数误差 RMS 分别为 0.13264 / 0.16337 / 0.12531；在已保存的导数中仅扣除该项，误差降为 0.00272 / 0.00363 / 0.00308。该操作只分析导数，不改变训练奖励。TGI 参考 `mdp/terrain.py` 的 exact-support 分支也明确采用最近邻前向、双线性反向，因此这类差异符合代理梯度的性质，不足以证明本实现的支撑项有 bug，更不能证明该项导致训练慢。

角速度跟踪在这批样本中提供的动作梯度明显大于线速度跟踪。例如控制步 20 的 H1 梯度 RMS 为 0.01214 对 0.00082，约 15 倍。该差异也受 F 的动作敏感度、当前跟踪误差和状态分布影响，不能只靠这一比值就降低角速度权重。

无动作探索噪声的 8 环境首回合截尾存活时间平均 4.05 s，范围 1.32–6.40 s；1/8 到 6.40 s 预算结束仍存活，因此 6.40 s 不是该回合的真实终止时长。5/8 因 root_height 终止，2/8 因 base_contact 终止。场景包含不同初始地形等级（0 和 3），样本少，不能与训练日志的回合移动均值或 TGI 历史结果直接横向比较。

结论：本轮没有证据支持整体替换核心奖励以修复“不可微”；当前奖励配方保持不变。下一项应核对实际下降/失去支撑前的真实与模型状态，尤其 root_height、脚底净空和支撑预测，验证模型梯度是否对应真实改善。只有模型内的自动求导与有限差分检查，不能完成这一物理正确性验证。
