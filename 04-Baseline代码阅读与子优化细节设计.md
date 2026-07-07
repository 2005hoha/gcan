# Baseline代码阅读后：子优化细节设计

> 日期：2026-07-03
> 代码基线：EPyMARL (QMIX MARL框架) + Qedgix (GNN+QMIX UAV系统)
> 目标：基于对两个代码库的深入阅读，对第三章中的每个子优化进行代码级细节设计
> 原则：只做分析设计，不修改代码

---

## 一、代码基线关键发现

### 1.1 EPyMARL 关键接口

| 接口 | 文件:行号 | 作用 | 我们的修改点 |
|------|----------|------|-----------|
| QMixer.forward | `qmix.py:47-66` | agent_qs + states → q_tot | → GCANMixer.forward 替代 |
| QLearner mixer选择 | `q_learner.py:23-31` | 根据args.mixer实例化mixer | → 添加 `elif args.mixer == "gcan_qmix"` |
| 状态表示 | `gymma.py:113-114` | state = concat(all_obs) | → 需要改为返回per-agent obs给mixer |
| Batch中obs存储 | `run.py:111` | obs以`group:agents`存储，shape=(B,T,N,obs_dim) | → 已可用，只需传给mixer |
| 算法配置 | `config/algs/qmix.yaml` | mixer超参数 | → 新建 `config/algs/gcan_qmix.yaml` |

### 1.2 Qedgix 关键发现

| 发现 | 含义 |
|------|------|
| GNN在Agent层（编码观察） | 信用分配仍然靠QMIX hypernetwork |
| EdgeConv是默认GNN | 不是GAT！没有注意力权重暴露 |
| MHAconv在gnns.py中已有 | 最接近GAT，可以改造为暴露α_ij |
| 图来自环境(info dict) | adjacency由环境提供（基于距离阈值） |
| GenAgg在genagg.py中 | 可微的广义均值聚合，适合做信用加权 |

**关键差异**：Qedgix的GNN是做"更好的特征提取"，我们要的GCAN是做"更好的信用分配"。GNN的位置从Agent层→Mixer层。

### 1.3 为什么不在EPyMARL的Agent层加GNN

如果GNN放在Agent层（像Qedgix那样），GNN输出的更好特征确实能提升每个Agent的Q值估计——但信用分配问题没有解决，QMIX的超网络仍然在用简单的全局状态→单调权重的映射。GCAN把GAT放在Mixer层，直接用GAT的注意力权重做信用分配。

---

## 二、子优化0-B（重做）：图构建在图构建层的定位

### 基线阅读后的重新设计

在EPyMARL中，MPE环境没有原生的图结构。我们需要自己构建。

**关键决策**：图构建的位置

| 位置 | 优点 | 缺点 |
|------|------|------|
| 在环境中构建 | 接近Qedgix的做法 | 需要修改或包装每个环境 |
| 在Mixer中构建 | 不侵入环境代码 | 需要per-agent位置/观测信息 |
| 在Learner中构建 | 灵活 | 增加Learner复杂度 |

**选择：在Mixer中构建**——因为我们的GAT在Mixer中，图结构是信用分配的内部表示，不应暴露给环境。

```python
# 文件: src/modules/mixers/gcan_qmix.py

class GraphBuilder:
    """
    在Mixer内部将per-agent状态构建为图。
    输入: obs [B, T, N, obs_dim], agent_qs [B, T, N]
    输出: PyG Batch对象 (每个batch元素一个图)
    """
    
    def build(self, obs, agent_qs):
        batch_size, T, N, obs_dim = obs.shape
        
        graphs = []
        for b in range(batch_size):
            for t in range(T):
                # 节点特征: [观测 | Q值 | 位置(如果有)]
                node_feats = []
                for i in range(N):
                    feat = torch.cat([
                        obs[b, t, i],           # 局部观测
                        agent_qs[b, t, i:i+1],   # 该agent在当前动作上的Q值
                    ])
                    node_feats.append(feat)
                
                x = torch.stack(node_feats)  # [N, obs_dim+1]
                
                # 全连接图(小N) 或 K近邻图(大N)
                if N <= 10:
                    # 全连接: 所有agent两两相连
                    edge_index = torch.combinations(
                        torch.arange(N), r=2
                    ).t().contiguous()
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                else:
                    # K近邻: 基于位置相似度
                    edge_index = self._knn_graph(obs[b, t], k=5)
                
                graphs.append(Data(x=x, edge_index=edge_index))
        
        return Batch.from_data_list(graphs)  # PyG的批次化
```

**简化的起点**：在MPE的SimpleSpread（N=3-8）中，直接使用全连接图。不需要K近邻逻辑。让GAT的注意力机制自然稀疏化不重要的边。

---

## 三、子优化1-A：GCAN Mixer 代码级设计

### 3.1 现有QMIX Mixer的代码结构

```python
# EPyMARL: src/modules/mixers/qmix.py (完整关键代码)

class QMixer(nn.Module):
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim  # 32
        
        # 超网络: 状态 → 混合权重
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        
        # 偏置项
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs, states):
        # agent_qs: [batch, 1, n_agents]
        # states:   [batch, state_dim]
        
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # [batch, 1, n_agents]
        
        # 第一层: 状态→权重(取绝对值保证单调性)
        w1 = abs(self.hyper_w_1(states))  # [batch, embed_dim * n_agents]
        b1 = self.hyper_b_1(states)       # [batch, embed_dim]
        w1 = w1.view(-1, self.n_agents, self.embed_dim)  # [batch, n_agents, embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)              # [batch, 1, embed_dim]
        
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [batch, 1, embed_dim]
        
        # 第二层
        w_final = abs(self.hyper_w_final(states))  # [batch, embed_dim]
        w_final = w_final.view(-1, self.embed_dim, 1)  # [batch, embed_dim, 1]
        
        v = self.V(states).view(-1, 1, 1)  # [batch, 1, 1]
        
        y = torch.bmm(hidden, w_final) + v  # [batch, 1, 1]
        q_tot = y.view(bs, -1, 1)
        return q_tot
```

### 3.2 GCAN Mixer：替换超网络为GAT

```python
# 新文件: src/modules/mixers/gcan_qmix.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GCANMixer(nn.Module):
    """
    GAT驱动的信用分配混合网络。
    
    核心改动：用GAT的注意力权重替代QMIX的hypernetwork（状态→权重）。
    
    数据流:
      agent_qs [B, T, 1, N]  +  obs [B, T, N, obs_dim]
        → GraphBuilder构建每时刻的图
        → GAT消息传递（学习智能体间的注意力=信用）
        → 信用加权求和 → Q_tot
    """
    
    def __init__(self, args):
        super().__init__()
        self.n_agents = args.n_agents
        self.obs_dim = int(np.prod(args.obs_shape))  # 单个智能体观测维度
        self.q_input_dim = self.obs_dim + 1           # obs + Q值
        
        # GAT层
        self.gat1 = GATConv(
            self.q_input_dim,
            args.gcan_hidden_dim,   # 默认64
            heads=args.gcan_heads,   # 默认4
            dropout=args.gcan_dropout,  # 默认0.1
            concat=True              # 多头拼接
        )
        # 第二层GAT：head=1，输出标量信用分数
        self.gat2 = GATConv(
            args.gcan_hidden_dim * args.gcan_heads,
            1,                        # → 1维信用分数
            heads=1,
            dropout=args.gcan_dropout,
            concat=False
        )
        
        # 全局状态偏置（类似QMIX的V(s)项）
        self.state_dim = int(np.prod(args.state_shape))
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, args.gcan_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.gcan_hidden_dim, 1)
        )
    
    def forward(self, agent_qs, states, obs=None):
        """
        Args:
            agent_qs: [B, T, 1, N] — 每个智能体在被选动作上的Q值
            states:   [B, T, state_dim] — 全局状态
            obs:      [B, T, N, obs_dim] — 每个智能体的观测（新增参数）
        Returns:
            q_tot:    [B, T, 1]
            credits:  [B, T, N] — 每个智能体的信用分数（可解释性输出）
            att_weights: GAT第一层的注意力权重（用于可视化）
        """
        B, T, _, N = agent_qs.shape
        
        # 如果观测不可用（向后兼容），从states恢复
        if obs is None:
            obs_dim_per_agent = self.obs_dim
            obs = states.view(B, T, N, obs_dim_per_agent)
        
        q_tots = []
        all_credits = []
        all_att = []
        
        for t in range(T):
            # 步骤1: 获取当前时刻的数据
            qs_t = agent_qs[:, t, 0, :]    # [B, N]
            obs_t = obs[:, t]               # [B, N, obs_dim]
            state_t = states[:, t]          # [B, state_dim]
            
            # 步骤2: 构建每batch的图
            batch_graphs = []
            for b in range(B):
                # 节点特征: 观测 + Q值
                node_feats = torch.cat([
                    obs_t[b],           # [N, obs_dim]
                    qs_t[b].unsqueeze(-1)  # [N, 1]
                ], dim=-1)              # [N, obs_dim + 1]
                
                # 边: 全连接 (N ≤ 10时)
                edge_index = self._fully_connected(N).to(node_feats.device)
                
                batch_graphs.append(Data(x=node_feats, edge_index=edge_index))
            
            # 步骤3: GAT前向传播
            pyg_batch = Batch.from_data_list(batch_graphs)
            
            # 第一层GAT (多头)
            x1, att1 = self.gat1(
                pyg_batch.x, 
                pyg_batch.edge_index,
                return_attention_weights=True
            )
            # x1: [B*N, hidden_dim*heads]
            # att1: (edge_index, alpha) — 注意力权重
            
            x1 = F.elu(x1)
            
            # 第二层GAT (单头→信用分数)
            x2, att2 = self.gat2(
                x1, 
                pyg_batch.edge_index,
                return_attention_weights=True
            )
            # x2: [B*N, 1] — 每个智能体的信用分数（在消息传递后的嵌入）
            
            # 步骤4: 整理per-agent信用
            credits = x2.view(B, N)                    # [B, N]
            credits = F.softplus(credits)               # 保证≥0
            credits = credits / credits.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # 归一化
            
            # 步骤5: 信用加权Q_tot
            q_tot = (credits * qs_t).sum(dim=-1, keepdim=True)  # [B, 1]
            
            # 步骤6: 全局状态偏置
            v = self.V(state_t)  # [B, 1]
            q_tot = q_tot + v
            
            q_tots.append(q_tot)
            all_credits.append(credits)
            all_att.append(att1[1])  # 保存注意力权重
        
        # 堆叠时间维
        q_tot = torch.stack(q_tots, dim=1)       # [B, T, 1]
        credits = torch.stack(all_credits, dim=1)  # [B, T, N]
        
        return q_tot, credits, all_att
    
    def _fully_connected(self, N):
        """生成N个节点的全连接边（无自环）"""
        import itertools
        edges = list(itertools.combinations(range(N), 2))
        edge_index = torch.tensor(
            edges + [(j, i) for i, j in edges],  # 双向
            dtype=torch.long
        ).t().contiguous()
        return edge_index  # [2, N*(N-1)]


# 用于消融实验的简化版：不使用per-agent obs
class GCANMixerSimple(GCANMixer):
    """
    简化版：仅使用Q值作为节点特征（不需要per-agent观测）。
    用于消融实验——验证观测信息对信用分配的贡献。
    """
    def __init__(self, args):
        # 覆盖输入维度：只用Q值(1维)
        self.q_input_dim_simple = 1
        # 重新定义第一层GAT
        args_copy = copy.deepcopy(args)
        # ... （省略细节）
```

### 3.3 GCAN vs QMIX 代码级对比

| 组件 | QMIX | GCAN |
|------|------|------|
| 权重生成 | hyper_w_1(states): state→linear | GATConv(x, edge): graph→attention |
| 非线性 | abs()保证单调性 | softplus(credits)保证非负 |
| 信用解释 | 无法解释（权重是状态的黑盒函数） | attention weights可直接可视化 |
| 参数数量 | state_dim × embed_dim × n_agents | GAT参数≈O(in_dim × hidden × heads) |
| 对N的扩展 | 参数数量随N线性增长 | 参数数量与N无关（归纳式） |
| 所需额外输入 | states（全局状态） | obs（per-agent，构建图用） |

### 3.4 Learner中的集成修改

```python
# 修改: src/learners/q_learner.py (第23-31行附近)

# 原代码:
# elif args.mixer == "qmix":
#     self.mixer = QMixer(args)

# 新代码:
elif args.mixer == "qmix":
    self.mixer = QMixer(args)
elif args.mixer == "gcan_qmix":
    from modules.mixers.gcan_qmix import GCANMixer
    self.mixer = GCANMixer(args)

# ...

# 修改: mixer调用处 (第107-113行附近)
# 原代码:
# q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

# 新代码:
if args.mixer == "gcan_qmix":
    q_tot, credits, att_weights = self.mixer(
        chosen_action_qvals, 
        batch["state"][:, :-1],
        batch["obs"][:, :-1]  # 新增：per-agent观测
    )
    # 可选：保存credit分数到日志
    self.log_credits(credits)
else:
    q_tot = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
```

### 3.5 配置文件

```yaml
# 新文件: src/config/algs/gcan_qmix.yaml

# GCAN + QMIX 混合配置
# 继承自qmix.yaml，增加GAT特定参数

mixer: "gcan_qmix"

# GAT参数
gcan_hidden_dim: 64        # GAT隐藏维度
gcan_heads: 4               # 第一层注意力头数
gcan_dropout: 0.1           # GAT dropout
gcan_graph_type: "full"     # 图类型: full(全连接)/knn/distance
gcan_k_neighbors: 5         # KNN的K值（仅graph_type=knn时使用）

# 继承QMIX的通用参数
mixing_embed_dim: 32        # 不再使用（GCAN不需要hypernet的embed dim）
hypernet_layers: 2          # 不再使用
hypernet_embed: 64          # 不再使用
use_rnn: False
double_q: True
target_update_interval_or_tau: 200
```

---

## 四、子优化2-A：HTA-Lite 代码级设计

### 4.1 事件触发器的精确实现

```python
# 新文件: src/components/llm_scheduler.py

class HTALiteScheduler:
    """
    事件驱动的LLM调用调度器。
    纯规则逻辑，不需要torch（非神经网络组件）。
    """
    
    def __init__(self, 
                 deviation_threshold: float = 5.0,
                 safety_threshold: float = 3.0,
                 max_steps_without_llm: int = 200,
                 goal_coverage_threshold: float = 0.9):
        self.deviation_threshold = deviation_threshold
        self.safety_threshold = safety_threshold
        self.max_steps_without_llm = max_steps_without_llm
        self.goal_coverage_threshold = goal_coverage_threshold
        
        self.last_llm_call_step = 0
        self.llm_plan = None  # LLM最近输出的规划
        self.call_history = []  # 记录每次调用的事件类型
        
    def should_call_llm(self, 
                        step: int, 
                        agent_positions: np.ndarray,  # [N, 2/3]
                        goal_positions: np.ndarray,    # [N, 2/3]
                        obstacle_distances: np.ndarray, # [N]
                        inter_agent_distances: np.ndarray  # [N, N]
                        ) -> Tuple[bool, Optional[str]]:
        """
        返回 (是否调用LLM, 触发原因)
        """
        
        # 事件1: 任务偏离检测
        if self.llm_plan is not None:
            target_positions = self.llm_plan["target_positions"]  # [N, 2/3]
            deviation = np.linalg.norm(agent_positions - target_positions, axis=-1)
            
            if deviation.mean() > self.deviation_threshold:
                return True, f"deviation_mean_{deviation.mean():.1f}"
            if deviation.max() > self.deviation_threshold * 1.5:
                return True, f"deviation_max_{deviation.max():.1f}"
        
        # 事件2: 安全临界
        if obstacle_distances.min() < self.safety_threshold:
            return True, f"safety_obstacle_{obstacle_distances.min():.1f}"
        
        # 排除对角线（自己到自己）
        mask = ~np.eye(inter_agent_distances.shape[0], dtype=bool)
        min_inter_agent = inter_agent_distances[mask].min()
        if min_inter_agent < self.safety_threshold:
            return True, f"safety_collision_{min_inter_agent:.1f}"
        
        # 事件3: 目标达成（所有agent接近其目标）
        goal_distances = np.linalg.norm(agent_positions - goal_positions, axis=-1)
        if goal_distances.mean() < self.safety_threshold:
            return True, "goal_achieved"
        
        # 事件4: 超时兜底
        steps_since_last_llm = step - self.last_llm_call_step
        if steps_since_last_llm > self.max_steps_without_llm:
            return True, f"timeout_{steps_since_last_llm}"
        
        return False, None
    
    def update_plan(self, plan: dict, step: int):
        """更新LLM规划"""
        self.llm_plan = plan
        self.last_llm_call_step = step
    
    # ... (状态保存/加载方法)
```

### 4.2 与EPyMARL训练循环的集成点

在EPyMARL中，LLM的参与位置不在`q_learner.py`（那个是纯MARL训练），而是在环境交互层面。如果LLM提供的是子目标/奖励塑形，应该在`EpisodeRunner.run()`中集成。

```python
# 修改点: src/runners/episode_runner.py (环境交互循环中)

# 伪代码（展示集成位置，不修改原文件）:
# 
# while not terminated:
#     # HTA-Lite调度器判断
#     should_call, reason = hta_scheduler.should_call_llm(
#         step=t, agent_positions=..., goal_positions=..., ...
#     )
#     
#     if should_call:
#         llm_output = call_llm_api(current_state, history)
#         update_sub_goals(llm_output)  # 更新agent的子目标
#         hta_scheduler.update_plan(llm_output, t)
#     
#     # 正常MARL执行（使用当前子目标作为条件）
#     actions = mac.select_actions(batch, t, t_env, 
#                                    extra_condition=current_sub_goals)
#     ...
```

---

## 五、子优化2-B：GateNet 代码级设计

### 5.1 轻量GNN门控器

```python
# 新文件: src/modules/agents/gatenet.py

class GateNet(nn.Module):
    """
    轻量GNN：预测每个智能体是否需要LLM。
    
    设计原则：
    - 小的隐藏维度（64 vs GCAN的128）
    - 少的注意力头（2 vs GCAN的4）
    - 简单的输出头（sigmoid二分类）
    """
    
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.gat = GATConv(obs_dim, hidden_dim, heads=2, dropout=0.0)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, graph):
        x = F.elu(self.gat(graph.x, graph.edge_index))
        gate = self.gate_head(x)  # [N, 1]
        return gate
```

### 5.2 训练数据生成方案

```python
# 伪代码: 离线生成GateNet训练数据

def generate_gatenet_training_data():
    """
    通过多次训练运行（不同LLM调用频率下），
    收集(state, 是否调用LLM, 效果)的三元组。
    """
    dataset = []
    
    # 配置1: 每20步调用LLM
    run_training(llm_freq=20, collect_fn=lambda state, perf:
        dataset.append((state, called=True, perf_gain=perf))
    )
    
    # 配置2: 每100步调用LLM
    run_training(llm_freq=100, collect_fn=lambda state, perf:
        dataset.append((state, called=True, perf_gain=perf))
    )
    
    # 配置3: 不调用LLM（baseline）
    run_training(llm_freq=None, collect_fn=lambda state, perf:
        dataset.append((state, called=False, perf_gain=0))
    )
    
    # 标签化
    for state, called, perf_gain in dataset:
        if called and perf_gain > threshold:
            label = 1.0  # 调用LLM有用
        else:
            label = 0.0  # 调用LLM无用或未调用
    
    return dataset
```

### 5.3 与HTA-Lite的双重门控

```python
# 综合决策伪代码:

def should_call_llm_hybrid(step, state, graph):
    # 第一优先级：安全规则（必须遵守）
    if is_safety_critical(state):
        return True, "safety_rule"
    
    # 第二优先级：任务偏离规则
    if is_deviated_from_plan(state):
        return True, "deviation_rule"
    
    # 第三优先级：学习到的模式
    gate_value = gatenet(graph).mean().item()
    if gate_value > 0.7:
        return True, f"learned_high_{gate_value:.2f}"
    
    # 第四优先级：超时兜底
    if step - last_llm_call > 200:
        return True, "timeout_fallback"
    
    return False, None
```

---

## 六、子优化3-A：CoEvo-Lite 代码级设计

### 6.1 信号健康度监测器

```python
# 新文件: src/components/signal_health.py

class SignalHealthMonitor:
    """
    监测LLM信号的三种衰减模式。
    纯统计分析（不需要GPU）。
    """
    
    def __init__(self, window_size=20, saturation_var_ratio=0.1,
                 entropy_threshold=0.01):
        self.window_size = window_size
        self.saturation_var_ratio = saturation_var_ratio
        self.entropy_threshold = entropy_threshold
        
        # 历史缓冲区
        self.reward_mean_history = deque(maxlen=200)
        self.reward_var_history = deque(maxlen=200)
        self.entropy_history = deque(maxlen=200)
        self.credit_distribution_history = deque(maxlen=200)
        
    def update(self, rewards, policy_entropy, credits=None):
        self.reward_mean_history.append(rewards.mean())
        self.reward_var_history.append(rewards.var())
        self.entropy_history.append(policy_entropy)
        if credits is not None:
            self.credit_distribution_history.append(credits)
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断当前信号健康状态。
        返回: {"status": "healthy"|"saturated"|"misleading"|"forgotten",
                "confidence": float,
                "details": dict}
        """
        
        if len(self.reward_var_history) < 2 * self.window_size:
            return {"status": "healthy", "confidence": 0.0,
                    "reason": "insufficient_data"}
        
        # 检测1: 饱和型衰减
        recent_vars = list(self.reward_var_history)[-self.window_size:]
        early_vars = list(self.reward_var_history)[-2*self.window_size:-self.window_size]
        
        var_ratio = np.mean(recent_vars) / (np.mean(early_vars) + 1e-8)
        if var_ratio < self.saturation_var_ratio:
            return {"status": "saturated", "confidence": 1 - var_ratio,
                    "var_ratio": var_ratio}
        
        # 检测2: 遗忘型衰减
        recent_entropy = list(self.entropy_history)[-self.window_size:]
        if np.mean(recent_entropy) < self.entropy_threshold:
            return {"status": "forgotten", "confidence": 0.8,
                    "entropy": np.mean(recent_entropy)}
        
        # 检测3: 信用分布警报
        if len(self.credit_distribution_history) >= self.window_size:
            recent_credits = np.stack(
                list(self.credit_distribution_history)[-self.window_size:]
            )
            credit_gini = self._gini(recent_credits.mean(axis=0))
            if credit_gini > 0.7:  # 极度不均衡
                return {"status": "stale", "confidence": credit_gini,
                        "credit_gini": credit_gini, 
                        "warning": "credit distribution too skewed"}
        
        return {"status": "healthy", "confidence": 0.9}
    
    def _gini(self, x):
        """计算基尼系数（衡量分布不均衡度）"""
        sorted_x = np.sort(x)
        n = len(x)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n
```

### 6.2 与EPyMARL的集成

CoEvo-Lite在EPyMARL的训练循环中是一个独立的监测钩子：

```python
# 伪代码: 在q_learner.py的train()方法末尾

# 每N轮训练后检查一次
if episode % args.signal_health_check_interval == 0:  # 默认每100轮
    diagnosis = signal_health_monitor.diagnose()
    
    if diagnosis["status"] != "healthy":
        wandb.log({"signal_health/status": diagnosis["status"],
                   "signal_health/confidence": diagnosis["confidence"]})
        
        if diagnosis["confidence"] > 0.8:
            # 触发LLM信号重新生成
            logger.warning(f"Signal health degraded: {diagnosis}")
            # 在实际系统中，这里会触发LLM重生成
            # trigger_llm_signal_regeneration()
```

---

## 七、关键实现风险与技术决策

### 7.1 PyG的Batch处理

EPyMARL的batch是 `[batch_size, time_steps, n_agents, dim]`。PyG的Batch需要 `[总节点数, dim]` + `batch`索引。

**风险**：循环处理T维可能导致训练变慢。

**缓解**：
- 对于T≤100的环境（MPE的time_limit=100），循环开销可接受
- 对于更大T，可以在时间维上采样（不展开所有T，随机采样一个子序列）

### 7.2 全连接图的扩展性

对于N=3-8的全连接图，edge_index的元素数 ≈ N² ≈ 9-64，GAT前向传播约1-5ms。对于N>20，全连接图的边数→400+，开始成为瓶颈。

**缓解**：N>10时自动切换到K近邻图（K=5），边数从N²降到NK≈50-100。

### 7.3 GAT注意力权重作为信用的理论保证

GAT的注意力权重α_ij衡量"节点j对节点i的信息传递有多重要"。这与信用分配的目标一致——"智能体j对智能体i的决策有多重要"。

但GAT的α是在**特征空间**中计算的（基于node embedding的相似度），而非**奖励空间**。需要通过TD误差的反向传播来对齐这两个空间——当α_ij的分配导致更好的Q_tot时，梯度会强化这种分配。

**验证方式**：在已知最优信用分配的环境（人工设定ground truth）上测试GCAN的注意力是否收敛到ground truth。

### 7.4 为什么不用GRPO

在补充分析报告中提到了MAGPO的组相对优势方法，但GCAN不使用它，而是使用标准TD学习。原因：
1. GRPO需要每个状态采样G个动作→G倍的环境交互
2. 物理多智能体仿真中，环境交互是瓶颈（不是网络推理）
3. TD学习在EPyMARL中已高度优化

---

## 八、从设计方案到可运行代码的步骤清单

```
□ Step 1: 安装PyG + 验证EPyMARL可运行 (1天)
□ Step 2: 创建 src/config/algs/gcan_qmix.yaml (0.5天)
□ Step 3: 创建 src/modules/mixers/gcan_qmix.py → GCANMixer类 (2天)
□ Step 4: 修改 src/learners/q_learner.py → 添加gcan_qmix分支 (0.5天)
□ Step 5: 确保 batch["obs"] 传递给mixer (0.5天)
□ Step 6: 在SimpleSpread(3 agents)上调试GCAN训练 (2天)
□ Step 7: 验证注意力权重可视化 → 确认信用分配有意义 (1天)
□ Step 8: GCAN vs QMIX对比实验 (2天)
□ Step 9: 扩展到5/8 agents (2天)
□ Step 10: 添加LLM原型（可选）(2天)

总计：约13个工作日（2.5周）达到GCAN核心完成
```
