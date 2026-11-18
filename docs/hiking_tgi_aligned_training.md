# Hiking：TGI 优化配方对齐

第一个 2048 环境实验在第 900 轮检查时，最近 50 轮记录的平均真实存活约 1.08 s，最近 10 轮约 1.23 s，地形等级仍为 0。该实验已停止；第 900 轮检查点保留在 `logs/hiking_supermimic/tgi_depth_20261116/hiking_train_2048/model_900.pt`。本轮按用户要求开始对齐 TGI，从头训练，不把旧策略或旧 replay 带入新实验。

新入口为 `--tgi_aligned`，自动启用 `--tgi_depth`。训练检查点格式为 `native_hiking_tgi_aligned_depth_v2`，部署控制器格式为 `hiking_depth_latent_controller_v2`。旧模式与旧导出仍可加载，跨版本直接 resume 会拒绝。

## 已对齐的部分

| 项目 | 旧实验 | 新实验 |
|---|---|---|
| 并行环境 | 2048 | 2048 |
| 世界模型 | 2048×3，Adam 1e-4 | 1024×3，Adam 2e-4（TGI 支持的 Adam 选项） |
| 模型 batch/H | 32 / 8 | 2048 / 8 |
| 模型更新/轮 | 10 | 8 |
| 策略有效 batch/H | 128 / 32 | 2048 / 32 |
| 模型预热 | 500 轮 | 300 轮，第 301 轮开始策略更新 |
| 真实动作噪声标准差 | 0.2 | 0.1 |
| 想象动作噪声标准差 | 0 | 0.2，重参数化 |
| 策略目标 | 物理模型反传 | 模型反传 + 0.1×似然比项 + 2×镜像一致性 |
| 策略 Adam betas | 0.9 / 0.999 | 0.9 / 0.95 |
| Actor 输入 | 固定尺度 | 本体/视觉各自运行均值方差，统计只来自真实采集 |
| Actor 输出初始化 | 普通初始化 | 末层权重×0.01，偏置归零 |
| 接触损失 | 62 通道求和×0.1 | 62 通道均值×0.5 |
| 核心奖励 | Hiking 多项惩罚叠加 | 共用 TGI 核心奖励公式，适配 native 根坐标和鞋底 |
| 脚滑权重 | -1 | -0.1 |
| 净空目标 | 0.08 m，摆动接触门控 | 0.1 m，按 TGI 水平脚速加权 |
| Critic targets | 更新输入统计后构造 | 用策略阶段相同的旧统计构造，再更新统计与拟合 |
| 梯度裁剪 | norm + value | norm=1，Actor 取消额外 value 裁剪 |

奖励包含线速度/角速度跟踪、直立、站立姿态、关节限位、动作变化、躯干角速度、关节偏移、垂直速度、步态、净空、脚滑和落脚支撑。真实 replay 和想象共用同一套公式；不再叠加旧 Hiking 的身体姿态、undesired contact 和 alive 等奖励。原生 RewardManager 的分项日志仍属于旧任务诊断，不能当作新策略目标；`mean_task_reward` 和 replay 奖励使用新奖励。

另外修复真实奖励统计中的 N×1 与 N 相加导致 N×N 广播的问题。该问题影响旧实验的总奖励日志，不影响存活步数统计；旧总奖励不适合与新实验比较。

## 分块梯度与镜像

有效策略 batch 始终为 2048。为了限制显存，按 `--policy_microbatch` 分块，每轮 Actor/编码器各只 step 一次，不能把分块误算成多次更新。

第一遍无梯度 rollout 保存每块 RNG 状态、Critic 特征与 lambda-return，优势在全体 2048×32 个转移上统一归一化。第二遍恢复 RNG，重算相同的带噪声 rollout 并累积梯度。Critic 参数和 F 始终冻结，Actor 和编码器的计算图保留。所有分块完成后才裁剪梯度、更新参数。随后 Critic 进行 5×4 次拟合。

镜像按 Hiking 的真实关节名称建立左右置换和 roll/yaw 符号，不能直接照抄 TGI 不同关节顺序的索引。角速度按轴向量反射，重力按普通向量反射，命令按 vx/vy/wz 反射。视觉镜像对原始深度历史水平翻转后重新编码，不假设 z32 本身具有镜像对应。镜像最多抽取约 4096 个想象观测。

## 保留的 Hiking 差异与验收

保留 native torso 根坐标、原有机器人/PD/动作尺度、动力学上下文、目标到速度命令、深度相机历史、原有地形和速度跟踪课程，以及原有真实/想象终止条件。TGI 使用另一套根坐标、地形分布和 SGS 课程，因此这不是完整 TGI 环境复现。Critic 也保留 native 的物理上下文、本体/AMP 历史与视觉特征布局，而非 TGI 的纯当前物理特征布局。

目前采集每轮 3 个控制步，replay200，初始64步；TGI 默认 UTD=0.4 在每轮交替采集2/3步。当前使用固定学习率；TGI 有余弦衰减。世界模型保留原生身体接触通道及力矩/加速度辅助头，以维持 native 接口；已修正通道归约方式，但没有声称与 TGI 双脚接触头等价。

深度编码器保留真实重建与策略回报两种学习信号；相机姿态到图像的导数停止，本体反馈仍经 F 反传。FPM 尚未接入。

运行检查必须覆盖真实采集、2048模型 batch、2048策略有效 batch、完整反传、参数冻结、Critic 更新以及控制器导出回读。梯度累积与未分块计算的一致性有独立测试。运行通过只证明新配方可执行；稳定站立/行走和6秒存活仍需真实回合验证。应按真实交互数、Actor 更新数和训练时间共同比较，不以轮次单独宣称追平 TGI。

```bash
source logs/hiking_environment/activate.sh
export CUDA_VISIBLE_DEVICES=GPU-35d42f8f-79ad-6118-f5df-d3f85899947b
export PYGLET_HEADLESS=true
export OMP_NUM_THREADS=2
python -u scripts/instinct_rl/train_hiking_native_latent.py \
  --task Instinct-Parkour-Target-Amp-G1-Newton-v0 \
  --num_envs 2048 --device cuda:0 --max_iterations 14000 \
  --tgi_aligned --policy_microbatch 512 \
  --output logs/hiking_supermimic/tgi_aligned_20261116/hiking_train_2048
```

当前验证与启动产物统一放在 `logs/hiking_supermimic/tgi_aligned_20261116/`。启动核验结束后不自动轮询旧训练。

验证记录：81 项相关测试通过；本次修改 hooks 全部通过，全仓 hooks 仍有既有格式/拼写问题，无关自动修改已恢复。平地及原生 Hiking 均以模型 batch2048、策略有效 batch2048 完成完整更新和导出一致性校验；512 分块额外完成平地验证。正式进程 PID 2046740 已完成至少3轮，每轮F8/深度8、Actor0/Critic0，符合预热计划。启动记录在同目录 `training_launch.json`，日志为 `hiking_train_2048.log`。

## 2026-11-16 回放采样提速

`NativeTrainingReplay.sample_physics` 改为按保留的时间行批量 gather，消除逐个 batch×horizon 样本构造 Python 字典和 stack 的开销。只搬运选中的图像，避免先堆叠整个图像回放库。随机数抽样顺序、环境顺序、终止转移、episode/segment/控制步连续性以及尾部填充均保留；无效位置中的 NaN 不会被当成有效目标。未改变精度、图像重建样本数、2048 模型/策略有效 batch 或每轮优化器更新次数。

每轮日志新增 `update_timing`：回放采样、模型训练、策略/风格阶段的主机墙钟耗时。没有为计时添加 GPU 同步，因此这些分项包含异步调度边界的影响，整轮耗时才是吞吐比较依据。

独立 CPU 基准使用 200 行、2048 环境、batch2048/H8、2 个 CPU 线程，各测四次。带两组深度图的采样平均从 0.509 s 降至 0.177 s（约 2.9 倍）；不带图像从 0.333 s 降至 0.049 s。该合成基准共用图像存储，仅用于隔离采样实现开销，不能当作训练整轮收益。产物在 `logs/hiking_supermimic/tgi_speed_20261116/`。

53 项相关测试通过，其中新增固定随机种子逐元素对比测试，覆盖 H1/H8/H20、环境顺序、终止/截断、时钟中断、采集段边界和 NaN 填充。修改文件 hooks 通过；全仓 hooks 的既有格式/拼写问题仍存在，无关自动修改已恢复。

用户随后要求从头训练：恢复第 100 轮的临时进程已停止，最终正式任务为 `logs/hiking_supermimic/tgi_speed_fresh_20261116/`，不带 `--resume` 或 `--initialize_from`。模型、Actor、编码器、Critic、优化器和 replay 全部重新初始化，仍先预热 300 轮，第 301 轮开始策略训练，总预算 14000 轮。启动信息在 `training_launch.json`，训练日志为 `hiking_train_2048.log`，指标为 `hiking_train_2048/metrics.jsonl`。旧文件保留供对比。

正式从头训练已核验至第 11 轮，PID 2071671，`resumed_from=null`。比较旧/新任务各自第 2–11 轮（排除首轮初始化，均为模型预热）：平均整轮耗时 10.519 s → 8.117 s，耗时下降约 22.8%，吞吐约 1.30 倍。新任务每轮采样约 1.700 s、模型训练约 5.991 s。此为同一 GPU 上先后运行的短期测量，耗时存在波动；策略阶段尚未开始，不能据此推断第 301 轮后的加速比。完整对比记录见正式目录 `performance_comparison.json`。启动和提速核验完成后保持任务运行，不持续自动监测。
