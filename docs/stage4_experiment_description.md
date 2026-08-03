# 阶段四实验说明：MMSP 未见站点的 Chronos-2 多模态 Adapter

## 目标

验证冻结 Chronos-2 在未见光伏电站、完整模态和卫星模态缺失条件下，是否能从冻结 FusionSF 的 token 级多模态表示中获得可靠增强。Stage 3 的 mean-pooled embedding + past covariates 注入不再使用。

## 数据与公平性约束

- 数据集：MMSP；历史长度 24，预测长度 24，频率 1 小时。
- 站点隔离：训练站点 10–19，验证站点 20–21，未见测试站点 0–9。
- FusionSF checkpoint 与 Chronos-2 主体均冻结；仅训练投影、Adapter 和 gate 参数。
- 不使用目标区间真实功率或未来卫星图像；NWP 按 forecast origin 可用语义处理。

## Stage 4A：UniCA-inspired token 融合

FusionSF 正式 fusion 层输出 `[B,24,64]` token，不做 mean pooling；统一投影到 Chronos 隐藏维度后作为 Cross-Attention 的 Key/Value，Chronos token 作为 Query。残差缩放参数 α 零初始化：`H_new = H_chronos + α·CrossAttention(H_chronos,Z_fusion)`。

三种子 Stage 4A 的 aligned MAE 均值为 `0.050854`，标准差为 `0.000288`。

## Stage 4B：CoRA-inspired 相关性 Adapter

在 Stage 4A 动态 Cross-Attention 上增加 `Pool(Z_fusion) → MLP` 全局分支，使用零初始化 α、β。三种子 CoRA−Stage 4A MAE 均值为 `-0.001061`，3/3 seeds 为负，因此通过 Go 条件。

## Stage 4C：缺失感知 gate

gate 输入为卫星可用性、卫星缺失比例、NWP 可用性和 NWP 缺失比例。测试完整模态、卫星随机缺失 50%、卫星完全缺失。完整模态/50%/100% 缺失 MAE 分别为 `0.048831`、`0.093932`、`0.138627`。gate 均值变化范围仅 `0.000920`，未表现出可靠的缺失自适应，因此 Stage 4C 为负结果。

## 结果文件

- `reports/stage4_summary.csv`：Stage 4A/4B 三种子和 Stage 4C 场景级汇总。
- `reports/stage4_per_site_summary.csv`：Stage 4A/4B 逐站点指标。
- `results/stage4a/`、`results/stage4b/`、`results/stage4c/`：原始 manifest、metrics、audit、逐窗口结果和模型文件。

## 最终结论

token 级 FusionSF Cross-Attention 能稳定增强冻结 Chronos-2，CoRA-inspired Adapter 在三种子上进一步改善；当前缺失感知 gate 未学到有效的模态质量控制，后续最小行动是重设计 gate 映射，不增加多层 Transformer、MoE 或复杂损失。
