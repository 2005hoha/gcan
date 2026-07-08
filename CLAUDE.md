# CLAUDE.md — GNN-LLM-MARL 协同优化框架

> 项目代号：GCAN (GAT Credit Assignment Network)
> 日期：2026-07-08
> 阶段：Q1 完成 / Q2 核心进行中

---

## 一、项目目标

用 **GAT 注意力机制替代 QMIX 单调超网络**，解决多智能体强化学习中的信用分配问题。
最终愿景是 GNN-LLM-MARL 三层协同：GNN 做信用分配、LLM 做语义先验、MARL 做策略优化。

**当前焦点**：在 MPE SimpleSpread N=5 上验证 GCAN 替代 QMIX 的有效性。

---

## 二、技术栈

| 层 | 技术 | 用途 |
|------|------|------|
| MARL 框架 | EPyMARL (PyMARL fork) | 训练循环、环境交互、算法配置 |
| 图神经网络 | PyTorch Geometric (PyG) | GATConv、图构建、消息传递 |
| 深度学习 | PyTorch 2.6 + CUDA 12.4 | 模型定义、自动微分 |
| 环境 | MPE (Multi-Particle Environment) | SimpleSpread 验证任务 |
| 实验管理 | Sacred + FileStorageObserver | 配置管理、指标记录 |
| 包管理 | Conda (gcan env, Python 3.10) | 环境隔离 |
| LLM | 待接入 (Sentence-BERT + API) | 信用原型生成 |

---

## 三、目录结构

```
GNN-LLM-MARL协同优化框架/
│
├── CLAUDE.md                    ← 本文件
│
├── gcan/                        ← GCAN 核心包（零 EPyMARL 依赖）
│   ├── __init__.py              # 包入口，导出 GCANMixer, GraphBuilder
│   ├── graph_builder.py         # 图构建器：obs → PyG Data/Batch
│   ├── gcan_mixer.py            # GCANMixer: 2-layer GAT + Softmax credit + V(s)
│   └── tests/
│       ├── test_graph_builder.py # GraphBuilder 单元测试
│       └── test_gcan_mixer.py    # GCANMixer 单元测试
│
├── configs/                     ← 配置文件（独立于 EPyMARL）
│   ├── gcan_qmix.yaml           # GCAN 算法配置
│   ├── qmix_mpe.yaml            # QMIX 基线配置
│   ├── mpe_simple_spread.yaml   # SimpleSpread 默认 N 配置
│   └── mpe_simple_spread_n5.yaml # SimpleSpread N=5 环境配置
│
├── epymarl/                     ← EPyMARL 框架（fork, 独立 git repo）
│   └── src/
│       ├── main.py              # 训练入口
│       ├── run.py               # 实验运行器（已修改：传递 obs_shape）
│       ├── learners/
│       │   └── q_learner.py     # Q Learner（已修改：注册 GCANMixer）
│       ├── modules/
│       │   └── mixers/
│       │       ├── qmix.py      # QMixer（已修改：forward 加 obs=None）
│       │       └── vdn.py       # VDNMixer（已修改：forward 加 obs=None）
│       └── config/
│           ├── algs/
│           │   ├── gcan_qmix.yaml   # GCAN 算法配置
│           │   └── qmix_mpe.yaml    # QMIX MPE 配置
│           └── envs/
│               ├── mpe_simple_spread.yaml
│               └── mpe_simple_spread_n5.yaml
│
├── plan/                        ← 项目规划文档
│   └── Q1-Q3实施计划.md
│
├── experiments/                 ← 实验日志
│   └── train_log_01_qmix_n5.md
│
└── 00-05*.md                    ← 项目前期分析与设计文档
```

---

## 四、模块职责与接口

### 4.1 分层架构

```
┌──────────────────────────────────────────────┐
│                  gcan/                        │  ← 纯 PyTorch+PyG，零框架依赖
│   GraphBuilder  │  GCANMixer                 │
└────────────────────┬─────────────────────────┘
                     │ import
┌────────────────────▼─────────────────────────┐
│               epymarl/                        │  ← EPyMARL 框架
│   q_learner.py  │  qmix.py  │  run.py       │
└──────────────────────────────────────────────┘
```

### 4.2 GraphBuilder

**职责**：将多智能体观测转换为 PyG 图结构。

```
输入: obs [B*T, N, obs_dim], agent_qs [B*T, N]
输出: PyG Batch (x=[B*T*N, node_feat_dim], edge_index=[2, B*T*N*(N-1)])

接口:
  __init__(n_agents, obs_dim, structured_features, graph_type, distance_threshold, device)
  build(obs, agent_qs) → Batch
  @property node_feat_dim → int
```

**设计约束**：
- 不依赖 EPyMARL，只接受普通 tensor
- `structured_features=True` 时解析 MPE 格式 obs（位置、速度、地标距离等）
- `graph_type="full"` 用于 N≤10，全连接图让 GAT 注意力自然稀疏化

### 4.3 GCANMixer

**职责**：GAT 驱动的信用分配，替代 QMIX 超网络。

```
输入: agent_qs [B, T, 1, N], states [B, T, state_dim], obs [B, T, N, obs_dim]
输出: q_tot [B, T, 1]

内部数据流:
  obs → GraphBuilder.build() → graph
  graph → GATConv(4 heads) → GATConv(1 head) → credit_head(Softplus)
  credits = Softmax(credit_head_output)  # [N, 1], Σ=1
  q_tot = Σ(credits * agent_qs) + V(state)
```

**关键设计决策**：
- Softmax 归一化（非 abs）：允许 agent 间有正有负交互
- V(s) 基线项：全局状态值函数，类似 QMIX 的 V
- 2 层 GAT：层 1 多头(4)→拼接，层 2 单头→1 维
- `forward(agent_qs, states, obs=None)`：obs=None 时从 states 重建（向后兼容 QMIX 接口）

### 4.4 EPyMARL 修改点

| 文件 | 修改 | 原因 |
|------|------|------|
| `q_learner.py:27` | 注册 `mixer=="gcan"` 分支 | 实例化 GCANMixer |
| `q_learner.py:108,113` | mixer 调用加 `obs=` 参数 | 传递 per-agent 观测 |
| `qmix.py:47` | `forward` 签名加 `obs=None` | 保持接口一致 |
| `vdn.py` | `forward` 签名加 `obs=None` | 保持接口一致 |
| `run.py` | 添加 `args.obs_shape = env_info["obs_shape"]` | 传递观测维度给 GCANMixer |

### 4.5 数据流（完整训练一步）

```
EpisodeRunner.run()
  │
  ├── env.step() → obs [B, N, obs_dim], state [B, state_dim]
  │
  ├── mac.select_actions() → agent_qs [B, N]
  │
  └── batch = {obs, state, agent_qs, reward, ...}
         │
         ▼
      QLearner.train(batch)
         │
         ├── chosen_action_qvals = mac.agent.forward(batch)[actions]  # [B, T, 1, N]
         │
         ├── q_tot = self.mixer(chosen_action_qvals, batch["state"][:,:-1], 
         │                       obs=batch["obs"][:,:-1])
         │         │
         │         ▼
         │      GCANMixer.forward()
         │         ├── GraphBuilder.build(obs, qs) → graph
         │         ├── GAT1(x, edge_index) → x1  [B*T*N, hidden*4]
         │         ├── GAT2(x1, edge_index) → x2  [B*T*N, hidden]
         │         ├── credit_head(x2) → credits  [B*T, N, 1]
         │         ├── credits = Softmax(credits)  (dim=agents)
         │         ├── q_tot = Σ(credits * qs) + V(state)
         │         └── return q_tot
         │
         ├── loss = MSE(q_tot, reward + γ * target_q_tot)
         └── loss.backward() → GAT 注意力权重通过梯度更新
```

---

## 五、开发规范

### 5.1 环境

```bash
# 激活 conda 环境
conda activate gcan          # 或直接使用 E:/e_conda/gcan/python.exe

# 运行训练
cd epymarl/src
python main.py --config=gcan_qmix --env-config=mpe_simple_spread_n5 \
  with t_max=2050000 seed=42

# 运行测试
cd ../../gcan
pytest tests/ -v
```

### 5.2 代码风格

- `gcan/` 包：纯 PyTorch+PyG，不 import EPyMARL 任何模块
- 类型注解：关键接口用 typing 标注 `→`, `Optional`, `Union`
- 命名：文件 snake_case，类 PascalCase，函数 snake_case
- 注释：不写冗余注释，只在 WHY 非显而易见时写

### 5.3 Git

- 根目录是主 repo（`https://github.com/2005hoha/gcan`）
- `epymarl/` 是独立 git repo（fork 自 uoe-agents/epymarl），根目录只存 gitlink
- epymarl 修改需在 epymarl 内部 commit
- `results/`、`__pycache__/`、第三方 repo 在 `.gitignore` 中

### 5.4 实验管理

- Sacred 自动管理实验目录：`results/sacred/gcan/<env>/<run_id>/`
- 每个 run 产生：`metrics.json`（指标）、`run.json`（元数据）、`cout.txt`（控制台输出）
- 训练指标：`return_mean`（训练回报）、`test_return_mean`（最终评价指标）
- 诊断指标：`grad_norm`、`loss`、`q_taken_mean`、`target_mean`、`td_error_abs`

---

## 六、分阶段路线图

### Q1：基线复现与环境搭建 ✅ 完成

| 子优化 | 状态 | 关键产出 |
|--------|------|---------|
| 0-A QMIX 基线 | ✅ | QMIX N=5 确认过估计崩溃（q_taken 30+） |
| 0-B 图构建 | ✅ | GraphBuilder + 单元测试 |

**结论**：QMIX 单调信用分配在 N=5 失效，验证 GCAN 必要性。

### Q2：GCAN 核心实现 🔄 进行中

| 子优化 | 状态 | 关键产出 |
|--------|------|---------|
| 1-A GAT 信用分配 | 🔄 | GCANMixer 完成、EPyMARL 集成完成、seed=42 2M 训练完成 |
| 1-B LLM 原型 | ⏳ | 待启动 |
| 1-C 对比实验 | ⏳ | 待 3 种子完成后 |

**当前结果（seed=42, 2M 步）**：
- test_return: -4319 → -532.6（最终）/ -495.2（最优 @800K）
- Q 值稳定收敛至 2.36，无过估计爆炸
- 对比 QMIX（q_taken 爆炸至 30+，grad_norm 38000+），GCAN 根本上解决了问题

**待完成**：
- [ ] 补种子 seed=2023, 2024 → 3 种子完整报告
- [ ] GAT 注意力权重可视化
- [ ] 消融实验：GCAN vs GCAN-no-V(s)

### Q3：LLM 智能调度 ⏳ 待启动

| 子优化 | 状态 | 关键产出 |
|--------|------|---------|
| 2-A HTA-Lite | ⏳ | 事件驱动 LLM 调度器 |
| 2-B GateNet | ⏳ | 轻量 GNN 门控 |
| 2-C 决策蒸馏 | ⏳ | 可选 |

### Q4：信号健康度 ⏸ 暂缓

| 子优化 | 状态 | 关键产出 |
|--------|------|---------|
| 3-A CoEvo-Lite | ⏸ | LLM 信号衰减监测 |

---

## 七、关键设计决策记录

1. **图建在 Mixer 中而非环境中**：图是信用分配的内部表示，不应暴露给环境
2. **全连接图 + GAT 自然稀疏化**：避免手工设计边类型，GAT 注意力自动学习重要边
3. **Softmax 归一化（非 abs）**：允许差异化信用分配，不强制单调
4. **GAT 层 1 多头→层 2 单头**：层 1 捕获多种交互模式，层 2 聚合为标量信用
5. **obs=None 向后兼容**：GCANMixer 接口兼容 QMIX，方便对比实验
6. **不并行训练**：单种子充分利用 GPU，总耗时更优（3×11h vs 3×55h 并行）
