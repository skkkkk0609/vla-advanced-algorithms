# pi0 学习笔记（完整版）

论文：[pi0-a-vision-language-action-flow-model-for-general-robot-control.pdf](/home/robot/projects/vla-advanced-algorithms/pi0/pi0-a-vision-language-action-flow-model-for-general-robot-control.pdf)

> 按论文章节顺序编写。标注了各部分在原文中的位置，方便对照阅读。

## 目录

- [1. Section III：Overview — pi0 的整体框架](#1-section-iiioverview--pi0-的整体框架)
- [2. Section IV：pi0 要建模什么](#2-section-ivpi0-要建模什么)
- [3. Section IV：action token](#3-section-ivaction-token)
- [4. Section IV：conditional flow matching loss](#4-section-ivconditional-flow-matching-loss)
- [5. Section IV：noisy action 怎么构造](#5-section-ivnoisy-action-怎么构造)
- [6. 你问的公式：u(A_t^tau | A_t) = A_t - epsilon](#6-你问的公式ua_t^tau--a_t--a_t---epsilon)
- [7. Section IV：推理过程](#7-section-iv推理过程)
- [8. Section V：Data & Training — 数据与训练策略](#8-section-vdata--training--数据与训练策略)
- [9. Section VI：Experiments — 实验评估](#9-section-viexperiments--实验评估)
- [10. Section VII：Discussion, Limitations](#10-section-viidiscussion-limitations-and-future-work)
- [11. Appendix B：输入 token 布局](#11-appendix-b输入-token-布局)
- [12. Appendix B：flow timestep 怎么进入模型](#12-appendix-bflow-timestep-怎么进入模型)
- [13. Appendix B：attention mask](#13-appendix-battention-mask)
- [14. Appendix B：action expert](#14-appendix-baction-expert)
- [15. Appendix D：推理缓存和执行](#15-appendix-d推理缓存和执行)
- [16. 一句话总结](#16-一句话总结)

---

## 1. Section III：Overview — pi0 的整体框架

### Section III：Overview

原文位置：Section III, Overview，Figure 3 附近。

论文先说明 pi0 的整体框架：

- 使用 PaliGemma 初始化的 VLM backbone 处理图像和语言。
- 新增 action expert 处理机器人状态和动作。
- 使用 flow matching 生成连续 action chunk。

直观理解：

```text
图像 + 语言 -> VLM backbone -> 任务/场景语义上下文
机器人状态 + noisy action chunk -> action expert -> 动作向量场
```

pi0 的关键不是把动作离散化成语言 token，而是直接建模连续动作分布。

## 2. Section IV：pi0 要建模什么

原文位置：Section IV, The pi0 Model，公式 `p(A_t | o_t)` 附近。

论文定义要建模的条件动作分布：

```text
p(A_t | o_t)
```

其中：

```text
A_t = [a_t, a_{t+1}, ..., a_{t+H-1}]
```

这是未来 H 步动作，也就是 action chunk。论文中使用：

```text
H = 50
```

观测为：

```text
o_t = [I_t^1, ..., I_t^n, l_t, q_t]
```

含义：

- `I_t`：多路 RGB 图像，通常 2 或 3 个相机。
- `l_t`：语言指令 token。
- `q_t`：机器人 proprioceptive state，例如关节角。

所以 pi0 学的是：

```text
给定 图像 + 语言指令 + 机器人状态，生成未来 50 步连续动作 chunk。
```

## 3. Section IV：action token

原文位置：Section IV，conditional flow matching loss 公式前一段。

论文说每个 action chunk 里的 action 都对应一个 action token：

```text
A_t = [a_t, ..., a_{t+49}]
```

对应：

```text
[action_token_1, ..., action_token_50]
```

这些不是文本 token，而是连续动作 slot。训练时，这些 action token 输入的是 noisy action，而不是最终动作。

## 4. Section IV：conditional flow matching loss

原文位置：Section IV，conditional flow matching loss 公式。

论文给出的 loss：

```text
L_tau(theta) = E || v_theta(A_t^tau, o_t) - u(A_t^tau | A_t) ||^2
```

变量含义：

- `A_t`：真实专家 action chunk。
- `epsilon`：随机高斯噪声 action chunk。
- `tau`：flow matching timestep，范围 `[0, 1]`。
- `A_t^tau`：噪声和真实动作之间的中间动作。
- `v_theta(A_t^tau, o_t)`：模型预测的向量场。
- `u(A_t^tau | A_t)`：目标向量场。

重点：模型不是直接预测动作 `A_t`，而是预测当前 noisy action 应该朝哪个方向移动。

## 5. noisy action 怎么构造

原文位置：Section IV，`q(A_t^tau | A_t)` 和 `A_t^tau = ...` 段落。

论文采用 linear-Gaussian path：

```text
A_t^tau = tau A_t + (1 - tau) epsilon
```

其中：

```text
epsilon ~ N(0, I)
```

直观理解：

```text
tau = 0   -> A_t^tau = epsilon，几乎是纯噪声
tau = 0.5 -> A_t^tau = 0.5 A_t + 0.5 epsilon
tau = 1   -> A_t^tau = A_t，真实动作
```

所以 `A_t^tau` 是从噪声到真实动作之间的一条直线路径上的点。

## 6. 你问的公式：u(A_t^tau | A_t) = A_t - epsilon

原文位置：Section IV，`u(A_t^tau | A_t) = A_t - epsilon`。

### epsilon 是逐渐向 A_t 靠拢的吗？

严格说：

```text
epsilon 本身不会变。
```

`epsilon` 是训练或推理开始时采样出来的一段随机噪声 action chunk。真正逐渐变化的是：

```text
A_t^tau
```

也就是当前的 noisy action chunk。

在论文设定的线性路径里：

```text
A_t^tau = tau A_t + (1 - tau) epsilon
```

当 `tau` 从 0 增加到 1：

```text
A_t^tau 从 epsilon 移动到 A_t
```

所以更准确的说法是：

```text
不是 epsilon 自己靠近 A_t，
而是当前动作样本 A_t^tau 沿着 epsilon -> A_t 的路径移动。
```

### 为什么目标向量是 A_t - epsilon？

因为线性路径是：

```text
A_t^tau = tau A_t + (1 - tau) epsilon
```

对 `tau` 求导：

```text
d A_t^tau / d tau = A_t - epsilon
```

所以 `A_t - epsilon` 就是这条路径上的速度方向。模型要学的就是这个速度场。

### 是 Transformer 让它靠拢的吗？

推理时，是 action expert Transformer 预测速度场：

```text
v_theta(A_t^tau, o_t)
```

然后用 Euler 积分更新当前动作：

```text
A_t^{tau + delta} = A_t^tau + delta * v_theta(A_t^tau, o_t)
```

所以可以说：

```text
Transformer 不直接“拖动 epsilon”，
而是每一步根据 observation、当前 noisy action 和 tau 预测一个更新方向，
Euler 积分用这个方向把当前 action chunk 推向真实动作分布。
```

训练时，Transformer 学习让：

```text
v_theta(A_t^tau, o_t) ≈ A_t - epsilon
```

推理时，真实 `A_t` 不存在，模型只能根据学到的条件向量场，把随机噪声动作逐步推成合理动作。

## 7. Section IV：推理过程

原文位置：Section IV，`At inference time...` 段落。

推理从纯噪声开始：

```text
A_t^0 ~ N(0, I)
```

然后用 forward Euler 积分：

```text
A_t^{tau + delta} = A_t^tau + delta * v_theta(A_t^tau, o_t)
```

论文实验中：

```text
10 integration steps
delta = 0.1
```

伪代码：

```python
action = random_noise()

for step in range(10):
    tau = step / 10
    velocity = model(observation, action, tau)
    action = action + 0.1 * velocity

final_action_chunk = action
```

最终得到未来 50 步 action chunk。

---

## 8. Section V：Data & Training — 数据与训练策略

原文位置：Section V, Data & Training，Figure 4 附近。

这是 pi0 论文非常重要但之前笔记跳过的章节。pi0 的训练分为两个阶段：

```text
Pre-training（预训练）：大规模、多样化数据，训练一个通用 base model
    ↓
Post-training（后训练/微调）：小规模、高质量任务数据，专精到具体下游任务
```

论文明确提出这个设计理念来自 LLM 的训练范式：

> Our framework broadly resembles the training procedures employed for large language models, which typically consist of pre-training a base model on very large datasets ... followed by a post-training procedure.

（原文位置：Section VII, Discussion）

### 8.1 Pre-training 数据混合（原文：Section V-A, Figure 4）

预训练数据由两部分组成：

| 数据来源 | 占比 | 说明 |
|---|---|---|
| OXE Magic Soup | 9.1% | Open X-Embodiment、Bridge v2、DROID 等开源数据 |
| π dataset（自采） | ~90.9% | Physical Intelligence 自己采集的灵巧操作数据 |

自采数据规模：

```text
总步数：903M timesteps（约 10,000 小时）
  - 单臂：106M steps
  - 双臂：797M steps

任务数：68 tasks（注意：pi0 对 task 的定义很宽，一个 task 内包含大量不同物体和场景变化）
```

数据加权策略：

```text
每个 task-robot 组合按 n^0.43 加权，n 为该组合的样本数。
目的：防止高重复任务（如叠衣服）过度主导训练。
```

**关键理解**：预训练数据质量参差不齐但覆盖面广。论文认为这恰好有好处：

> the diverse (but lower quality) pre-training data allows the model to recover from mistakes and handle highly varied situations

（原文位置：PDF 第 5 页，Section V 开头右栏，`A. Pre-training and post-training` 小节标题正上方；该句跨 4 行排版。）

### 8.2 Post-training 微调（原文：Section V-A）

预训练完成后，用少量高质量任务数据微调：

```text
最简单任务：仅需 5 小时数据
最复杂任务：需要 100+ 小时数据
```

论文的核心观点：

```text
只用高质量数据训练 → 模型脆弱，不能从错误中恢复
只用预训练 zero-shot → 缺乏流畅的任务策略
两者结合（pre-train + fine-tune）→ 既稳健又精准
```

### 8.3 语言指令与高层策略（原文：Section V-B）

对于复杂任务（如 table bussing），pi0 可以接入一个 high-level VLM policy：

```text
高层任务（"清理桌子"）
    ↓ 高层 VLM 分解
子任务序列（"拿起餐巾纸" → "扔掉餐巾纸" → "拿起盘子" → ...）
    ↓ pi0 执行
底层动作
```

这个设计类似于 SayCan 的思路：VLM 负责语义分解和规划，pi0 负责执行。

（原文位置：Section V-B, Language and high-level policies）

### 8.4 机器人平台（原文：Section V-C, Figure 5）

pi0 在 **7 种不同的机器人配置** 上联合训练：

| 平台 | 类型 | 相机数 | 动作维度 |
|---|---|---|---|
| UR5e | 单臂 | 2 | 7 |
| Bimanual UR5e | 双臂 | 3 | 14 |
| Franka | 单臂 | 2 | 8 |
| Bimanual Trossen | 双臂（类 ALOHA） | 3 | 14 |
| Bimanual ARX/AgileX | 双臂 | 3 | 14 |
| Mobile Trossen/ARX | 双臂+移动底盘（非全向） | 3 | 16 |
| Mobile Fibocom | 双臂+移动底盘（全向） | 3 | 17 |

**跨 embodiment 训练的关键设计**：

```text
动作空间统一 pad 到最大维度（18 维）
不够的机器人补零
不够的相机 masked out
```

所以 pi0 在所有平台上联合训练，一个模型控制所有机器人。

---

## 9. Section VI：Experiments — 实验评估

原文位置：Section VI, Experimental Evaluation，Figure 6-13。

实验按照从简单到复杂的顺序组织，正好对应从 pre-training 到 fine-tuning 的递进。

### 9.1 基础模型 out-of-box 评估（原文：Section VI-A, Figure 7）

**实验目的**：预训练完成后直接评估（不微调），看 base model 的 zero-shot 能力。

**5 个任务**（Figure 6）：
- Shirt folding（叠 T 恤）
- Bussing easy（简单桌面清理）
- Bussing hard（困难桌面清理，更多物体、更复杂摆放）
- Grocery bagging（杂货装袋）
- Toast out of toaster（从烤面包机取吐司）

**对比基线**：
| 基线 | 说明 |
|---|---|
| OpenVLA (7B) | 训练在同样数据上，但不支持 action chunk、不支持高频控制 |
| OpenVLA (UR5e only) | 仅在 UR5e 数据上训练 |
| Octo (93M) | 使用 diffusion 生成动作，但模型容量小 |
| π0-small | π0 的缩小版，无 VLM 初始化 |
| π0 (parity) | π0 只训练 160k 步（匹配基线训练量） |
| π0 (full) | π0 完整训练 700k 步 |

**关键结果**（Figure 7）：

```text
π0 (full) >> π0 (parity) > π0-small > OpenVLA > Octo
```

结论：
1. **大模型 + VLM 预训练至关重要**（π0 >> π0-small）
2. **flow matching > autoregressive discretization**（π0 >> OpenVLA）
3. **模型容量很重要**（π0 >> Octo）

### 9.2 语言指令跟随（原文：Section VI-B, Figure 8-9）

**实验目的**：评估 pi0 理解语言指令的能力，特别是 VLM 初始化带来的增益。

**3 个任务**（Figure 8）：
- Bussing（清理桌子）
- Table setting（摆放餐具）
- Grocery bagging（杂货装袋）

**5 种指令条件**：
| 条件 | 说明 |
|---|---|
| π0-flat | 只给最终任务描述（"bag the groceries"） |
| π0-human | 人类专家提供中间步骤指令 |
| π0-HL | 高层 VLM 自主提供中间步骤指令 |
| π0-small-flat | 小模型 + flat 指令 |
| π0-small-human | 小模型 + 人类指令 |

**关键结果**（Figure 9）：

```text
π0 的语言理解能力显著优于 π0-small
人类中间指令的帮助 > 高层 VLM 的自主分解
π0-small 由于语言理解弱，即使有人类指令也提升有限
```

结论：**VLM 预训练直接转化为更好的语言指令跟随能力**。

### 9.3 学习新的灵巧操作任务（原文：Section VI-C, Figure 10-11）

**实验目的**：在预训练未见过的任务上微调，看 pre-training 的正迁移效果。

**5 个任务**，按与预训练数据的相似度分 tier（Figure 10）：

| 任务 | Tier | 说明 |
|---|---|---|
| Stack bowls（叠碗） | Easy | 类似预训练中的 bussing |
| Towel folding（叠毛巾） | Easy | 类似预训练中的 shirt folding |
| Tupperware in microwave | Medium | 容器操作类似预训练，但微波炉是新物体 |
| Paper towel replacement | Hard | 完全新物体、新动作 |
| Franka items in drawer | Hard | Franka 上无类似预训练任务 |

**微调数据量**：1 / 5 / 10 小时

**对比基线**：OpenVLA、Octo、ACT、Diffusion Policy

**关键结果**（Figure 11）：

```text
π0（pre-trained + fine-tuned）在大多数任务上最优
有趣的是：ACT 和 Diffusion Policy（从头训）在个别任务上也很强
这说明从预训练模型中做正迁移在之前的方法上是个难点
π0 的预训练在数据量少（1小时）时提升最明显
```

### 9.4 复杂多阶段任务（原文：Section VI-D, Figure 12-13）

**实验目的**：π0 能处理多长时间的复杂任务？需要多少预训练才能支撑？

**6 个任务**（Figure 12），**每个需要 5-20 分钟完成**：

| 任务 | 预训练中存在？ | 关键难点 |
|---|---|---|
| Laundry folding | 是 | 随机揉皱的初始状态 |
| Mobile laundry | 是 | 移动底盘的折叠任务 |
| Dryer unloading | 是 | 从烘干机取衣物 |
| Table bussing | **否** | 全新杂乱物体、复杂排序 |
| Box building | **否** | 需要双手配合折叠纸箱 |
| To-go box | **否** | 食物装盒 + 合盖 |
| Packing eggs | **否** | 易碎鸡蛋、精准放置 |

**对比**：只有 π0 自身消融（full / out-of-box / scratch）

**关键结果**（Figure 13）：

```text
pre-train + fine-tune > out-of-box > scratch
任务越难，预训练带来的提升越大
在大多数任务上达到 50%+ 完成度
```

---

## 10. Section VII：Discussion, Limitations, and Future Work

原文位置：Section VII。

### 核心结论

论文将 pi0 类比为机器人领域的 LLM 训练范式：

```text
预训练阶段 → 模型获得大量"知识"
后训练阶段 → 告诉模型如何利用知识完成具体任务
```

### 局限性

1. **数据配比不清楚**：不知道应该加什么类型的数据、如何加权最优
2. **任务成功率不完美**：部分任务仍然不可靠
3. **跨域泛化未知**：不同任务、不同机器人之间的正迁移程度需要进一步研究
4. **更远领域未验证**：自主驾驶、导航、腿足运动等是否也能受益于同样的预训练范式

---

## 11. Appendix B：输入 token 布局

原文位置：Appendix B, Model Architecture Details, Additional inputs and outputs。

输入顺序：

```text
[I_t^1, ..., I_t^n, language tokens, q_t, A_t^tau]
```

分成三块：

```text
Block 1: images + language
Block 2: robot state q_t
Block 3: noisy action chunk A_t^tau
```

其中：

- `images + language` routed to VLM backbone。
- `q_t + A_t^tau` routed to action expert。

## 12. Appendix B：flow timestep 怎么进入模型

原文位置：Appendix B, Incorporating the flow matching timestep。

每个 noisy action token 的 embedding 包含：

```text
当前 noisy action 数值 + tau 的 sinusoidal embedding
```

论文公式：

```text
W3 * swish(W2 * concat(W1 * a_t'^tau, phi(tau)))
```

其中：

- `a_t'^tau`：某一个 noisy action token。
- `phi(tau)`：tau 的 sinusoidal positional encoding。
- `W1, W2, W3`：MLP 参数。

这一步很重要，因为模型必须知道当前处于 flow 的哪一步。

## 13. Appendix B：attention mask

原文位置：Appendix B, Attention mask。

pi0 使用 blockwise causal attention mask：

```text
Block 1: images + language
Block 2: robot state
Block 3: noisy actions
```

规则：

- 每个 block 内部双向 attention。
- 前面的 block 不能看未来 block。
- 后面的 block 可以看前面 block。

因此：

```text
images/language 不看 robot state/action，减少 PaliGemma 分布偏移。
robot state 不看 noisy actions，方便缓存。
action tokens 可以看 images/language/state，也可以互相看。
```

## 14. Appendix B：action expert

原文位置：Appendix B, Action expert。

pi0 是一个 single transformer with two sets of weights：

```text
VLM expert：处理 images + language
action expert：处理 q_t + A_t^tau
```

两套权重通过 self-attention 交互。

参数规模：

```text
VLM backbone: 约 3B
action expert: 约 300M
总模型: 约 3.3B
```

action expert 做小，是因为推理时需要多次执行 flow matching forward pass。

## 15. Appendix D：推理缓存和执行

原文位置：Appendix D, Inference, Table I。

每次生成 action chunk：

```text
1. 编码图像
2. 对 observation tokens 做 forward
3. 对 action tokens 做 10 步 flow matching
```

为了提速，observation 部分的 KV cache 可以复用，每一步 flow 只重新计算 action suffix。

论文给出 RTX 4090 上的时间：

```text
image encoders: 14 ms
observation forward pass: 32 ms
10x action forward pass: 27 ms
total on-board inference: 73 ms
```

执行策略：

- 一次生成 H=50 步 action chunk。
- 不一定执行完整 50 步。
- 20Hz 机器人执行 16 步后重新推理。
- 50Hz 机器人执行 25 步后重新推理。
- 论文尝试 temporal ensembling 后发现性能变差，所以最终 open-loop 执行当前 chunk 的前半段。

## 16. 一句话总结

```text
pi0 的 flow matching 是：
用 action expert Transformer 学习条件向量场，
把随机 noisy action chunk 通过 10 步 Euler 积分推成可执行的 50-step action chunk。
```
