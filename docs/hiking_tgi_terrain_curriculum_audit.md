# Hiking 与 TGI 的地形课程和评估对照

## 结构对照

TGI 把地形表格的**行**作为难度等级，把**列**作为地形类型。每个配置还拆成出生 patch 和行走方向；Success-Guided Sampling（SGS）在每个地形类型内记录配置的成功率，并按目标成功率附近的权重采样。成功定义是走完至少 80% 的命令距离，或在近零命令下存活到 episode 末尾。不同类型的采样份额由原始 terrain proportion 固定。

Hiking 当前地形生成器同样有 10 行和 20 列，配置名称为 `perlin_rough`、`perlin_rough_stand`、`square_gaps`、`pyramid_stairs`、`pyramid_stairs_high`、`pyramid_stairs_inv`、`pyramid_stairs_inv_high`、`boxes`、`mesh_boxes`、`hf_pyramid_slope_inv`。训练使用 Isaac Lab 的 `tracking_exp_vel` curriculum：每次 reset 按当前列保留地形类型，只根据速度跟踪分数上调或下调该环境的行。它没有 TGI 的每配置成功率窗口、类型内成功引导采样、出生 patch/方向配置和 80% 距离成功判定。

因此，Hiking 的训练 level 并不是“一个 level 只有一种地形”；同一难度行会横跨多个地形列。这个结构可以用于训练，但评估时把同一行的所有列合并会混淆楼梯、坑、箱体和石块的结果。

## 当前改动

`evaluate_hiking_native_latent.py` 增加了固定地形评估参数：

```bash
--terrain native_hiking \
--terrain_type pyramid_stairs \
--terrain_level 3
```

评估会把所有环境固定到同一个 `terrain_type × terrain_level`，关闭 terrain-level curriculum，报告中写入 `fixed_terrain`、列名映射和 `by_terrain_type`。不带这两个参数时仍保留混合评估，但结果同时按类型分组，不能只看一个总 level 均值。地形列名称按 generator proportion 使用与 TGI 相同的累计列分组规则推导，并由测试覆盖。

楼梯应至少分别评估：`pyramid_stairs`、`pyramid_stairs_high`、`pyramid_stairs_inv` 和 `pyramid_stairs_inv_high`；坑、箱体、石块也应各自建立同样的 level 曲线。每个类型建议评估 level 0、3、6、9，保持 seed、环境数、无探索噪声和 episode 时长一致。

## SGS 课程切换（2026-11-17）

TGI aligned 的原生 Hiking 训练现在默认使用 SGS，替换 `tracking_exp_vel`，平地诊断仍关闭地形课程。配置为 4 个出生 patch、4 个世界行走方向、16 次结果窗口、0.8 目标成功率、0.25 相邻简单等级传播；权重公式、下限和温度与 TGI 相同。类型采样份额保持 terrain proportion（仅对生成器实际包含的类型归一化）。

出生 patch 对原有 reset 范围做 2×2 切分，不扩大原有出生区域。方向用于选择该扇区最远的原生 flat-patch 目标；若该扇区没有候选，则使用同方向、位于地形内部的虚拟导航目标。站立地形保留原有零速度命令。每次地形类型改变，命令速度范围和站立地形标志同步刷新；它们不再固定在初始化时的地形类型上。

成功判定使用回合实际出生位置，逐步累计实际命令的水平距离，加上剩余回合时间的命令距离预算；位移超过其 80%（上限为地形半宽）则成功。近零距离命令要求存活到完整回合末尾。显式的零长度 reset 不记录成功/失败，避免初始化和恢复污染窗口。该判定不再使用每一步都会更新的 `progress_start_pos_w`。

Checkpoint 保存成功率窗口和计数；恢复时加载窗口并从恢复后的分布重新 reset。契约写入 SGS 版本，阻止与普通 terrain-level 课程误续训。训练 metrics 每轮记录 SGS 总体统计，每 100 轮记录各地形类型内按 level 的成功率、采样份额和结果数。固定类型/等级评估继续关闭训练课程。

验证：25 项地形/评估相关测试通过；修改文件 hooks 通过。正式训练已按用户要求停止，未用新评估参数修改任何权重。

已用 4300 轮检查点实际运行 `pyramid_stairs × level 3` 固定评估：8 个环境全部报告 level 3，平均首回合存活 5.11 s，4/8 到达 6.4 s 评估上限；结果保存在 `logs/hiking_supermimic/eval_model4300_20261117_pyramid_stairs_l3/results.json`。这证明固定类型/等级路径在真实场景中生效，仍不代表已经稳定完成楼梯攀爬。

## Reward alignment update

The TGI recipe now computes clearance from the center shoe sample, matching TGI's
`foot_center` definition. The remaining shoe samples continue to feed the support
and terrain roughness terms. Legacy non-TGI locomotion keeps its previous lowest
sample behavior. The reward contract is now `v3_center_clearance_pose_std`; checkpoints
created with the previous `v1` or `v2` contracts should not be resumed after this change.
The pose term also uses TGI's standing, walking, and running joint standard deviations.
