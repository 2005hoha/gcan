# 训练日志 01：QMIX SimpleSpread N=5 基线

> 启动时间：2026-07-05 19:48
> 完成时间：2026-07-06 02:38
> 状态：3/3 完成

## 实验配置

| 参数 | 值 |
|------|-----|
| 算法 | QMIX |
| 环境 | MPE SimpleSpread |
| 智能体数 N | 5 |
| 训练步数 t_max | 2,050,000 |
| 随机种子 | 42, 123, 999（还需补2个至5） |
| target_update_interval_or_tau | 800（RES论文修复） |
| epsilon_anneal_time | 50000 |
| buffer_size | 5000 |
| batch_size | 32 |
| lr | 0.0005 |
| mixer | qmix |
| mixing_embed_dim | 32 |
| hypernet_layers | 2 |
| hypernet_embed | 64 |
| double_q | True |
| use_rnn | False |
| gamma | 0.99 |
| grad_norm_clip | 10 |

## 关键修复

- `target_update_interval_or_tau: 800`（RES NeurIPS 2021证明MPE需要800，默认200会导致灾难性过估计）

## 验证结果（1万步冒烟测试）

| 步数 | test_return_mean | loss | grad_norm |
|------|-----------------|------|-----------|
| 5000 | -4318 | 1.37 | 17.27 |
| 10000 | -3750 | 1.37 | 17.27 |

- 训练不崩溃，梯度稳定，测试回报在改善

## 最终结果

| Seed | 最优 test_return | @步数 | 最终 test_return | 训练时长 |
|------|-----------------|-------|-----------------|---------|
| 42 | **-2,110.8** | 50,100 | -4,850.4 | 2h48m |
| 123 | **-1,130.0** | 900,100 | -3,387.7 | 2h43m |
| 999 | **-896.9** | 250,100 | -9,793.7 | ~2h45m |

### 诊断信号

所有三个seed一致的过估计崩溃模式：

1. **Q值持续膨胀**（seed=123: q_taken 0.24→30.24, target 0.26→30.13）
2. **梯度爆炸**（seed=42: grad_norm峰值~38,000, seed=123: grad_norm峰值~38,160）
3. **早期最优→中期崩溃→晚期不恢复**
4. **target_update_interval=800 不足以阻止过估计**

### 聚合统计（3 seeds）

| 指标 | 值 |
|------|-----|
| Best 均值 | -1,379.2 |
| Best 标准差 | 655.2 |
| Final 均值 | -6,010.6 |
| Final 标准差 | 3,010.5 |

### 结论

QMIX在MPE SimpleSpread N=5上出现灾难性过估计，RES论文的target_update_interval修复不足以防止。这直接支撑GCAN的核心动机——用GAT注意力驱动的信用分配替代单调超网络约束。

### 待完成

- [ ] 补跑 seed=7, seed=2024（共5个seed）
- [ ] 用 rliable 库计算 IQM + 95% bootstrap CI
- [ ] 画出训练曲线对比图
