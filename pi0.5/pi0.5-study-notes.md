# π0.5 学习笔记（完整版）

论文：[pi0.5-a-vla-with-open-world-generalization.pdf](/home/robot/projects/vla-advanced-algorithms/pi0.5/pi0.5-a-vla-with-open-world-generalization.pdf)

> 按论文章节顺序编写。每个重点标注 PDF 页码、Section/Figure/Table 及栏位或段落锚点，方便在 IDE 中逐行选中提问。
>
> 记号提醒：本文正文把插值写成 `a^{τ,ω}=τa+(1-τ)ω`，但把训练目标写成 `ω-a`。这与对该插值按递增 `τ` 求导得到的 `a-ω` 符号相反。下文严格保留原文记号，并专门解释其方向约定，绝不静默套用 π0 的写法。

## 目录

- [1. Introduction：问题定义与 open-world generalization](#1-introduction问题定义与-open-world-generalization)
- [2. Preliminaries：VLA 与动作表示](#2-preliminariesvla-与动作表示)
- [3. Section IV-A：π0.5 架构](#3-section-iv-aπ05-架构)
- [4. Section IV-B：离散与连续动作的混合目标](#4-section-iv-b离散与连续动作的混合目标)
- [5. Section IV-C：Pre-training](#5-section-iv-cpre-training)
- [6. Section IV-D：Post-training](#6-section-iv-dpost-training)
- [7. Section IV-E：机器人系统](#7-section-iv-e机器人系统)
- [8. Section V：实验问题与设置](#8-section-v实验问题与设置)
- [9. Section V-A：能否泛化到真实新家](#9-section-v-a能否泛化到真实新家)
- [10. Section V-B：泛化如何随场景数变化](#10-section-v-b泛化如何随场景数变化)
- [11. Section V-C：Co-training recipe 消融](#11-section-v-cco-training-recipe-消融)
- [12. Section V-D：与 π0 等 VLA 比较](#12-section-v-d与-π0-等-vla-比较)
- [13. Section V-E：高层推理有多重要](#13-section-v-e高层推理有多重要)
- [14. Section VI：Discussion、局限与未来工作](#14-section-vidiscussion局限与未来工作)
- [15. Appendix：评测与模型技术细节](#15-appendix评测与模型技术细节)
- [16. 训练与推理总览](#16-训练与推理总览)
- [17. 一句话总结](#17-一句话总结)

---

## 1. Introduction：问题定义与 open-world generalization

原文位置：PDF 第 1 页，Abstract、Figure 1；PDF 第 1–2 页，Section I，左/右栏。

### 1.1 论文要解决什么问题

论文把 **open-world generalization（开放世界泛化）** 定义为物理智能中的核心开放问题：机器人离开实验室后，仍能处理现实世界中多样、未预见的场景和事件。

这里的“开放世界”不是声称机器人能完成任意任务，而是论文实际评测范围内的更具体命题：

```text
训练：许多机器人、场景、任务、语义监督与网页多模态数据
测试：训练中从未出现的新厨房和新卧室
变化：新物体实例/类别、新背景、新布局、新家具机构
任务：数分钟乃至 10–15 分钟的多阶段家务操作
```

原文位置：PDF 第 1 页，Figure 1 caption 与 Introduction 左栏首段；PDF 第 2 页，Introduction 左栏末段。

论文认为，只靠穷举式扩大目标机器人的第一手数据难以覆盖复杂长程任务中的所有情形。所提出的替代思路是：让统一 VLA 从异构来源迁移知识，包括其他机器人、其他场景和任务、语义子任务标注、人类口头指导以及网页视觉语言数据。

### 1.2 π0.5 与 π0 的关系

原文位置：PDF 第 1 页，Abstract；PDF 第 2 页，Introduction 右栏；PDF 第 4–5 页，Section III 与 IV 开头。

π0.5 **基于 π0 构建**，不是与 π0 无关的新模型：

| 方面 | π0 | π0.5 |
|---|---|---|
| VLM backbone | PaliGemma | 沿用 PaliGemma 初始化 |
| 连续动作 | action expert + flow matching | 后训练阶段加入同类 action expert + flow matching |
| 预训练动作表示 | 主要持续使用 action expert | 先把动作全部编码为 FAST 离散 token |
| 训练数据 | 多机器人动作数据为核心 | MM/ME/CE/HL/WD，后训练再加 VI |
| 高层推理 | 通常由任务 prompt 直接出低层动作 | 同一个模型先生成文本子任务，再生成连续动作 |
| 目标 | 通用机器人控制 | 强调全新家庭环境中的 open-world generalization |

核心继承关系可以概括为：

```text
π0 的连续 action expert / flow matching
              +
FAST 离散动作预训练 + 异构 co-training + 统一高低层推理
              =
             π0.5
```

### 1.3 97.6% 的准确语境

原文位置：PDF 第 2 页，Section I，右栏第一段，`The overwhelming majority...`。

论文原话的限定非常重要：

```text
π0.5 第一训练阶段（pre-training）的训练样例中，97.6%
不是“移动机械臂执行家庭任务”的目标域样例。
```

因此：

- 97.6% **不是网页数据占比**；它包含其他机器人数据、网页数据等非目标移动家务来源。
- 97.6% **不是机器人数据之外的数据占比**；其他机器人数据仍是机器人数据。
- 97.6% **不是成功率或泛化率**。
- 它说明目标域 MM 样例在第一阶段只占约 2.4%，论文用此强调异构迁移，而不是说目标域数据不重要。
- 论文同时说明仍收集了约 400 小时目标移动机械臂数据，因此不能解读为“无需目标 embodiment 经验”。

### 1.4 层级式问题分解

原文位置：PDF 第 2 页，Section I，右栏第二段至页末；Figure 2。

论文将联合分布分成两步：先判断“现在应该做什么子任务”，再判断“如何用机器人动作完成它”。

```text
用户高层命令 ℓ："clean the bedroom"
      ↓ 同一个 π0.5，文本自回归
高层子任务 ℓ̂："pick up pillow"
      ↓ 同一个 π0.5，action expert + flow matching
低层 action chunk a_{t:t+H}
```

高层过程运行频率低于低层动作推理。它类似 chain-of-thought/test-time compute，但中间“思考”是可读的机器人子任务，并直接作为低层策略条件。

> 论文结论与解释须区分：实验显示含显式高层推理的完整模型最好；将其解释为“层级分解帮助长程规划”符合作者论述，但不能据此声称每一次成功都由某条高层文本严格因果地造成。

---

## 2. Preliminaries：VLA 与动作表示

原文位置：PDF 第 4 页，Section III，左栏下半至右栏。

### 2.1 模仿学习目标

论文先给出一般 VLA 的最大似然目标：

```text
max_θ E_{(a_{t:t+H}, o_t, ℓ) ~ D} [ log π_θ(a_{t:t+H} | o_t, ℓ) ]
```

符号逐项解释：

- `D`：机器人示范数据集。
- `t`：当前控制时刻。
- `a_{t:t+H}`：从 `t` 到 `t+H` 的动作 chunk。
- `o_t`：当前观测。
- `ℓ`：自然语言任务指令。
- `π_θ`：参数为 `θ` 的策略。

观测通常包含：

```text
o_t = [I_t^1, ..., I_t^n, q_t]
```

- `I_t^i`：第 `i` 路相机图像。
- `q_t`：proprioceptive state，即机器人本体状态/配置。

### 2.2 两种动作表示的取舍

原文位置：PDF 第 4 页，Section III，右栏中下部。

1. **FAST 离散动作 token**：把动作 chunk 压缩成离散 token，可直接做 next-token prediction，训练高效、容易与文本/视觉任务统一。
2. **连续 action expert + flow matching**：输入部分去噪的连续动作，输出向量场；连续细粒度、推理可并行生成整个 chunk，适合实时控制。

π0.5 的关键不是二选一，而是分阶段结合：

```text
Pre-training：用 FAST 把动作当作离散 token，扩大高效 co-training
Post-training：保留离散目标，同时新增连续 action expert
Inference：文本自回归；低层动作采用 10 步 flow matching
```

---

## 3. Section IV-A：π0.5 架构

原文位置：PDF 第 4 页 Figure 3；PDF 第 5 页，Section IV-A，左右栏。

### 3.1 联合分布与两阶段分解

模型表示：

```text
π_θ(a_{t:t+H}, ℓ̂ | o_t, ℓ)
```

其中：

- `o_t=[I_t^1,...,I_t^n,q_t]`：多相机图像和机器人状态。
- `ℓ`：总任务 prompt，例如 “put away the dishes”。
- `ℓ̂`：tokenized textual output；可以是高层子任务，也可以是网页 VQA 的回答。
- `a_{t:t+H}`：连续动作 chunk。

论文明确分解为：

```text
π_θ(a_{t:t+H}, ℓ̂ | o_t, ℓ)
= π_θ(a_{t:t+H} | o_t, ℓ̂) · π_θ(ℓ̂ | o_t, ℓ)
```

注意原文做了一个明确建模假设：动作分布条件于 `o_t, ℓ̂`，**不再直接条件于原始总任务 `ℓ`**。也就是高层文本 `ℓ̂` 承担从总任务到当前低层动作的接口。

### 3.2 单一 Transformer 如何容纳四类 token

论文把“token”作广义使用。Transformer 输入为 `x_{1:N}`，输出为：

```text
y_{1:N} = f(x_{1:N}, A(x_{1:N}), ρ(x_{1:N}))
```

输入可包括：

| token 类型 | 形式 | 处理方式 | 主要用途 |
|---|---|---|---|
| 文本 token | `x_i^w ∈ N` | embedding matrix + VLM expert | prompt、状态、子任务、答案、FAST token |
| 图像 patch | `x_i^I ∈ R^{p×p×3}` | vision encoder + VLM expert | 多相机视觉输入 |
| FAST action token | 离散整数，属于文本序列 | VLM expert，自回归 | 预训练/联合离散动作预测 |
| 连续 action token | `x_i^a ∈ R^d` | 线性投影 + action expert | flow matching 的 noisy action slot |

这里“单一 Transformer”指一个统一的 Transformer 计算图与共享 self-attention 交互，并不意味着所有 token 使用完全相同参数：连续动作 token 会路由到较小的 **action expert 权重**。

### 3.3 token type routing `ρ`

原文位置：PDF 第 5 页，Section IV-A，左栏末段。

`ρ(x_i)` 表示 token 类型路由。它决定两件事：

1. token 使用哪一种输入 encoder/embedding；
2. token 在 Transformer 层内使用哪一组 expert 权重。

```text
ρ(image patch)       -> vision encoder -> VLM Transformer weights
ρ(text/FAST token)   -> text embedding -> VLM Transformer weights
ρ(continuous action) -> linear project -> action-expert weights
```

`ρ` 不是随机 MoE gate，也不是根据内容学习出的概率路由；按论文描述，它由 token 类型指示。

### 3.4 Attention `A` 与信息隔离

原文位置：PDF 第 5 页，Section IV-A；PDF 第 19 页，Appendix E，Figure 18。

- 图像、prompt、状态组成 full-prefix，彼此双向注意。
- FAST action token 看 prefix，并对先前 FAST token 做因果注意。
- 连续 action expert token 看 prefix 和彼此，但不看 FAST action token。
- VLM token 不看 action expert token。

```text
                            可注意到
查询 token        prefix        earlier FAST       action expert
---------------------------------------------------------------
prefix               ✓               ✗                  ✗
FAST action           ✓               ✓(causal)          ✗
continuous action     ✓               ✗                  ✓(双向)
```

这样避免同一真实动作的离散表示与连续 noisy 表示在训练中互相泄漏答案。VLM 与 action expert 仅通过 self-attention 交互，整体信息从 VLM prefix 单向流向 action expert。

### 3.5 输出头与机器人状态输入

原文位置：PDF 第 5 页，Section IV-A，右栏上半。

输出拆成：

```text
y_{1:M}^ℓ：文本 token logits

y_{1:H}^a：action expert 连续输出 token
```

- 文本 logits 用于自回归采样 `ℓ̂`，也用于预测 FAST action token。
- action expert 输出经线性映射后成为连续 flow vector field。
- `M+H≤N`：不是每个输出位置都有 loss。

**机器人状态的输入方式**：`q_t` 先离散化，再作为文本 token 输入 VLM prefix。状态内容包括关节角、夹爪姿态、躯干升降姿态和底盘速度。它不是像 π0 的 action expert state token 那样以连续向量单独注入。

---

## 4. Section IV-B：离散与连续动作的混合目标

原文位置：PDF 第 5 页，Section IV-B，右栏，Equation (1)。

### 4.1 原文的 flow matching 插值与目标

论文逐字对应的公式是：

```text
a_{t:t+H}^{τ,ω} = τ a_{t:t+H} + (1-τ)ω
ω ~ N(0,I),  τ ∈ [0,1]
```

端点：

```text
τ=0 -> a^{0,ω}=ω       （纯高斯噪声）
τ=1 -> a^{1,ω}=a       （真实动作）
```

但是，紧接着原文写模型预测的 flow vector field 为：

```text
ω - a_t
```

Equation (1) 中则对整个 chunk 写为：

```text
ω - a_{t:t+H}
```

### 4.2 必须明确的符号方向

对原文插值按 **递增 `τ`** 求导：

```text
d a^{τ,ω} / dτ = a - ω
```

这与原文训练目标 `ω-a` 恰好相反。因此不能同时把下面两句话都不加说明地写成真：

```text
1. 推理让 τ 从 0 增加到 1；
2. 直接用原文预测量 ω-a 做 x <- x + δv。
```

严格解释只有两种等价方向约定：

```text
数据方向参数化：τ 从 0 -> 1，速度应为 a-ω；
噪声方向参数化：沿反向时间积分，速度写为 ω-a。
```

本文 PDF 正文明确印出插值 `τa+(1-τ)ω` 和监督 `ω-a`，但没有在 Equation (1) 附近写出 Euler 更新式来消除这个表面符号冲突。附录 E 又说 timestep 采样强调 low timesteps，并与 π0 一致。因此本笔记保留原文公式，并把它标为**需要结合实现确认积分变量/更新符号的方向约定**，不擅自把 `ω-a` 改成 `a-ω`。

> 与旧 π0 笔记对照时尤其要小心：旧笔记按 `τ:0→1` 的前向导数解释 `a-ε`；π0.5 PDF 的 Equation (1) 明写相反目标 `ω-a`。若实现采用从噪声到数据的正步长更新，更新式必须相应使用负号，或改用反向时间变量。

个人观点为论文中顺序错误
### 4.3 混合训练目标 Equation (1)

原文目标可整理为：

```text
E_{D,τ,ω} [
    H(x_{1:M}, f_θ^ℓ(o_t,ℓ))
    + α ||ω-a_{t:t+H} - f_θ^a(a_{t:t+H}^{τ,ω},o_t,ℓ)||²
]
```

符号逐项解释：

- `H(·,·)`：离散 token 的交叉熵。
- `x_{1:M}`：目标文本序列，其中也包括 FAST 编码动作 token。
- `f_θ^ℓ`：VLM 文本 logits 输出。
- `f_θ^a`：较小 action expert 输出的连续向量场。
- `a^{τ,ω}`：真实动作与噪声之间的中间连续动作。
- `α∈R`：离散 CE 与连续 flow matching MSE 的权衡系数。

### 4.4 为什么要混合

```text
FAST 离散动作：训练快、适合大规模异构 next-token co-training
连续 flow：整段并行、10 次去噪、适合实时且细粒度动作
```

注意混合目标不是让 FAST token 和连续 token 互相预测。二者共享视觉/文本 prefix 和大部分语义表示，但 attention mask 阻止动作表示之间直接注意。

### 4.5 两个阶段中 α 的含义

- Pre-training：`α=0`，没有连续 flow matching loss，也没有用于最终动作的 action expert 训练。
- Post-training：`α=10.0`，同时训练 next-token CE 和 action expert 的 flow matching MSE。

原文位置：PDF 第 5 页，Equation (1) 后；PDF 第 7 页，Section IV-D 左栏。

---

## 5. Section IV-C：Pre-training

原文位置：PDF 第 6–7 页，Section IV-C，Figure 4。

第一阶段训练 280k gradient steps。模型作为标准自回归 Transformer，预测文本、目标位置和 FAST 动作 token。

### 5.1 六类数据先总览

| 缩写 | 全称 | 内容 | Pre-training | Post-training |
|---|---|---|---|---|
| MM | Diverse Mobile Manipulator | 约 400 小时、约 100 个家庭环境中的移动机械臂家务 | ✓ | ✓，筛成功且时长不过阈值 |
| ME | Diverse Multi-Environment non-mobile robot | 多家庭环境中的固定单/双臂 | ✓ | ✓，筛成功且时长不过阈值 |
| CE | Cross-Embodiment laboratory | 实验室中多机器人、多任务，含 OXE | ✓ | ✗ |
| HL | High-Level subtask prediction | 子任务文本，及相关 bounding box | ✓ | ✓，只保留 multi-environment slice |
| WD | Multi-modal Web Data | caption、VQA、object localization | ✓ | ✓，维持语义视觉能力 |
| VI | Verbal Instructions | 人类用语言逐步遥控已学低层策略 | ✗ | ✓ |

### 5.2 MM：目标移动机械臂数据

原文位置：PDF 第 6 页，Section IV-C，左栏 `Diverse Mobile Manipulator data (MM)`。

- 约 400 小时。
- 约 100 个家庭环境；实验缩放部分具体使用最多 104 locations。
- 任务与新家评测最直接相关，如清洁、整理。
- 这是“目标平台第一手经验”，虽只占预训练样例约 2.4%，仍提供目标 embodiment 和任务分布锚点。

### 5.3 ME：多环境固定机械臂数据

原文位置：PDF 第 6 页，Section IV-C，左栏 `Diverse Multi-Environment...`。

固定单臂或双臂比移动平台轻，易搬到更多家庭采集，因此场景更丰富；代价是 embodiment 不同。它用于研究“环境多样性 + 跨 embodiment”能否迁移给移动平台。

### 5.4 CE：实验室跨 embodiment 数据

原文位置：PDF 第 6 页，Section IV-C，左栏末至右栏上半。

- 多机器人、多任务、较简单桌面环境。
- 有些任务与评测相关，如把餐具放进容器；另一些不相关，如磨咖啡豆。
- 包含单臂、双臂、静态和移动底盘。
- 包含开源 OXE。
- 是 π0 数据集的扩展版本。

### 5.5 HL：高层子任务预测

原文位置：PDF 第 6 页，Section IV-C，右栏 `High-Level subtask prediction (HL)`。

对 MM/ME/CE 中含多个子任务的数据，人工标注语义子任务。模型联合学习：

```text
当前图像 + 总任务 -> 相关物体 bounding box -> 当前子任务文本
当前图像 + 当前子任务 -> FAST 动作 token
```

所以同一个模型自然兼具高层 policy 与低层 policy。论文还训练模型在子任务前预测当前观测中的相关 bounding box，增强定位语义监督。

### 5.6 WD：网页多模态数据

原文位置：PDF 第 6–7 页，Section IV-C，右栏 `Multi-modal Web Data (WD)`。

包括：

- image captioning：CapsFusion、COCO；
- question answering：Cambrian-7M、PixMo、VQAv2；
- object localization：并扩展了室内场景和家居物体 bbox 网页数据。

WD 的作用不能简单归结为“提高所有任务成功率”。实验中，去掉 WD 对四项 mock-home 总分的差异不显著，但对 OOD 物体语言跟随和高层推理影响显著；作者推测其广泛物体知识帮助理解未见类别。

### 5.7 动作统一处理

原文位置：PDF 第 7 页，Section IV-C，左栏上半。

- 所有 action data 都预测目标关节姿态或末端执行器姿态。
- prompt 加入 `<control mode> joint/end effector <control mode>` 区分控制模式。
- 每个数据集、每个动作维度用 1% 与 99% 分位数归一化到 `[-1,1]`。
- 动作维度固定为可容纳最大 action space；低维机器人补零。
- 预训练阶段所有动作都经 FAST 编码成离散 token。

---

## 6. Section IV-D：Post-training

原文位置：PDF 第 7 页，Section IV-D，左栏。

### 6.1 目标与训练设置

预训练 280k 步后，再训练 80k 步，目的有两个：

1. 专门适配家庭移动操作；
2. 新增能通过 flow matching 生成连续 action chunk 的 action expert。

关键设置：

```text
action expert：post-training 开始时随机初始化
训练目标：next-token CE + flow matching MSE
α：10.0
训练步数：80k
```

这说明 π0.5 不是在预训练阶段就持续训练连续 action expert；连续专家是在后训练才加入。与此同时仍保留 next-token prediction，以维持文本预测能力。

### 6.2 后训练数据变化

```text
保留 MM + ME：只取成功 episode，且长度低于固定阈值
去掉 CE：聚焦移动操作和多样家庭环境
保留 WD：维持视觉与语义能力
保留 HL：仅 multi-environment 数据对应部分
新增 VI：强化恰当高层子任务输出
```

论文没有在正文给出过滤长度阈值的具体数值，因此不应自行补写。

### 6.3 VI：verbal instruction demonstrations

VI 由专家用户实时选择合适子任务命令，一步步指挥已经学会的低层 policy 完成长任务。它类似“用语言遥操作”：

```text
人类观察场景
  -> 给当前子任务文本
  -> 已训练低层策略执行
  -> 人类再给下一子任务
```

收集到的不是关节级手工遥操作轨迹，而是“什么时刻该下达什么子任务”的高层示范。

原文位置：PDF 第 7 页，Section IV-D，左栏末段。

---

## 7. Section IV-E：机器人系统

原文位置：PDF 第 7 页，Section IV-E，Figure 5。

论文使用两种移动机械臂平台，均包括：

- 两条 6-DoF 机械臂和平行夹爪；
- 两个腕部单目 RGB 相机；
- 前向、后向相机；
- 三自由度全向底盘（2D 线速度 + 1D 角速度）；
- 1D 或 2D 躯干升降机构。

状态和动作空间总维度为 18 或 19，取决于平台。

相机使用不同：

```text
高层推理：全部四个相机
低层推理：前向相机 + 两个腕部相机
```

控制系统：模型以 action chunking 方式直接给出机械臂、夹爪、躯干的目标姿态和底盘目标速度，由简单 PD controller 以 50 Hz 跟踪；没有额外轨迹规划或碰撞检测。论文因此称操作与导航控制是端到端的。

---

## 8. Section V：实验问题与设置

原文位置：PDF 第 7 页，Section V，右栏；PDF 第 8 页 Figure 6。

论文提出五个实验问题：

1. π0.5 能否在全新家庭中泛化到复杂多阶段任务？
2. 泛化如何随训练环境数量增长？
3. co-training mixture 各成分各有什么贡献？
4. π0.5 与 π0 相比如何？
5. 高层推理有多重要，相比扁平低层推理和 oracle 高层基线如何？

π0.5 主模型的评测环境都未出现在其训练中。mock homes 用于可控、可复现的定量比较；三个真实家庭用于最终现实评估。Section V-B 另设“直接在测试家庭训练”的诊断性 control，用来衡量泛化差距，不能与主模型的未见环境条件混为一谈。

> 读图原则：论文图 7–13 多以柱状图/曲线呈现，正文并未逐项列出精确数字。本笔记只记录可由正文和 caption 明确核实的比较、显著性和试验次数，不把图形目测值写成精确结果。

---

## 9. Section V-A：能否泛化到真实新家

原文位置：PDF 第 7 页右栏至第 8 页左栏，Section V-A；Figures 6–7。

### 9.1 设置

- 三个训练中未出现的真实家庭，共三个厨房和三个卧室场景。
- 使用两种机器人平台。
- 定量任务包括 items in drawer、laundry basket、dishes in sink。
- 每个 task/environment 组合 10 trials（Figure 7 caption）。
- rubric 大致表示任务步骤完成百分比，细则见 Appendix B。

### 9.2 结果与边界

论文报告 π0.5 在各家庭的多种任务上能持续取得成功。多阶段任务一般持续约 2–5 分钟；Figure 1 展示的整屋清理可达 10–15 分钟。模型只接收简单总命令，高层过程自主生成如 “pick up the cup” 的步骤。

准确表述是：

```text
在这组新家庭、新物体、新背景和新布局评测中表现成功，
并且 mock setup 的表现对真实家庭具有代表性。
```

不应扩大为“已经证明对任意真实家庭或任意开放世界任务均可泛化”。

---

## 10. Section V-B：泛化如何随场景数变化

原文位置：PDF 第 8–9 页，Section V-B；Figures 8–9。

### 10.1 场景缩放设置

移动操作训练环境数：

```text
3, 12, 22, 53, 82, 104 locations
```

为降低重复完整 recipe 的成本，这组实验先用**不含 mobile manipulation data** 的 robot-action mixture 预训练，再分别用不同数量环境的移动操作数据后训练。每个模型后训练 40k 步，并让各模型看到相同数量的 unique samples，以控制样本量影响。

因此 Figure 8/9 研究的是“在该受控 recipe 下环境数量变化的相关效果”，不是只改变一个完全独立因子的严格现实因果实验。

### 10.2 两类评测

1. 四项端到端任务：dishes in sink、items in drawer、laundry basket、make bed。
2. 语言跟随：按指令选指定物体并放入 drawer/sink；分别测训练见过类别的新实例（ID）与训练未见类别（OOD）。

指标：

- language following rate：是否选中了语言指定物体；
- success rate：是否把该物体成功放到指定位置。

### 10.3 结果

- 四任务平均表现随训练 locations 增多总体提高。
- 104-location 模型与直接在测试家庭训练的 control 表现相近，尽管前者没见测试家庭。
- 只用目标测试环境数据、或只用 104 个 MM 环境而没有其他预训练 co-training task 的基线都明显更差。
- 语言跟随与成功率随 locations 增多而稳步提高；ID 类别提升快于 OOD 类别。

这些结果支持“更多环境 + 完整 co-training recipe 与更强泛化相关”。论文用 control 和固定 unique samples 增强解释力，但仍不应写成脱离该训练设置的普适缩放定律。

---

## 11. Section V-C：Co-training recipe 消融

原文位置：PDF 第 9–10 页，Section V-C；Figures 10–11；PDF 第 18–19 页 Appendix D，Figure 16。

消融项：

```text
no WD       去掉网页数据
no ME       去掉多环境固定机器人数据
no CE       去掉实验室跨 embodiment 数据
no ME or CE 同时去掉二者，仅保留目标移动平台机器人数据和 WD
```

### 11.1 Mock-home 四任务结果

Figure 10：每个 policy/task 做 10 trials。

- 去掉 ME 或 CE 都显著降低表现；二者都去掉更差。
- 说明模型从“其他环境的其他机器人”（ME）和“实验室其他任务/机器人”（CE）都获得了明显迁移收益。
- `no WD` 在这组四任务 aggregate 中差异**不具有统计显著性**。

### 11.2 语言跟随结果

Figure 11：

- 去掉 ME、CE 或二者，ID 与 OOD 表现均明显下降。
- 去掉 WD 对 OOD 物体造成显著下降。
- 作者**推测**网页数据中广泛的物体知识帮助理解未见物体类别；这是论文解释，不是由该消融唯一证明的机制因果。

### 11.3 Appendix 的逐任务补充

Figure 16：

- Items in Drawer 对 ME、CE、WD 都敏感，可能因其需要广泛物体理解。
- Dishes in Sink 对去掉 WD 相对稳健，但去掉 ME/CE 会下降，更依赖机器人操作策略。
- Laundry Basket、Make Bed 去掉跨 embodiment 数据也下降，但对其他 mixture 改动通常不那么敏感。

---

## 12. Section V-D：与 π0 等 VLA 比较

原文位置：PDF 第 10 页，Section V-D；Figure 12；PDF 第 18 页 Appendix C，Figure 15。

比较对象：

1. `π0`：原始模型，始终使用 action expert/flow-based 训练。
2. `π0-FAST+Flow`：加入 Equation (1) 的 FAST + flow 混合训练，但只用含 robot actions 的数据，不含 HL/WD，不能做高层推理。
3. `π0.5`：混合阶段 recipe + HL/WD + 高低层统一推理。

公平性控制：模型得到相同 cross-embodiment robot training set，训练步数可比。

结果：

- Figure 12 中 π0.5 在 mock-home 测试显著优于 π0 和 π0-FAST+Flow。
- 即使 π0 训练延长到 300k 步，该结论仍成立。
- 作者据此并结合 FAST 工作，认为离散 token 训练比纯 diffusion/flow 式训练具有更好的 compute efficiency。
- Figure 15 中 π0.5 的语言跟随略高于 π0-FAST+Flow，且远高于 π0；论文将其作为离散 token 训练有利于语言跟随的证据。

谨慎点：π0.5 与 π0 的差异同时包括数据 mixture、高层任务和训练阶段设计，因此 Figure 12 不是对某一个单独组件的纯净因果比较；相关机制需结合 Section V-C/E 消融理解。

---

## 13. Section V-E：高层推理有多重要

原文位置：PDF 第 10–11 页，Section V-E；Figure 13；PDF 第 19 页 Appendix D，Figure 17。

所有方法都使用完整 π0.5 低层推理，只改变高层 policy 或训练数据：

| 方法 | 含义 |
|---|---|
| π0.5 | 同一模型做显式高层和低层推理 |
| no WD | 训练去掉网页数据 |
| no VI | 训练去掉 verbal instruction 数据 |
| implicit HL | 训练含 HL，但运行时不显式生成子任务，原始 task prompt 直接进低层 |
| no HL | 训练和推理都没有高层数据/过程 |
| GPT-4 | GPT-4 作高层 policy，并提示任务说明与常见 label 列表 |
| human HL | 专家人类作“oracle”高层 policy，作为上界参考 |

### 13.1 主要结果

- 完整 π0.5 最好，甚至高于 human-HL “oracle”基线。
- 第二好是 implicit HL：不在运行时显式推高层，但训练时见过完整子任务预测 mixture。
- no HL 显著更差，说明 HL 训练数据本身很重要。
- no VI 显著更弱。VI 只占 high-level mobile manipulation examples 的约 11%，却对表现关键。
- no WD 也显著更差，表明 WD 的主要收益之一落在高层 policy。
- zero-shot GPT-4 最差，论文据此强调用 robot data 适配高层 VLM 的重要性。

### 13.2 如何理解 human oracle 被超过

这不意味着模型的抽象规划能力普遍超过人类。合理限定是：在这一固定接口和评测中，模型自产生的高层标签可能与自己训练过的低层 policy 分布更匹配；“human HL”只控制高层文本，低层执行仍由 π0.5 完成。

### 13.3 逐任务结果

Appendix D Figure 17：

- Items in Drawer 与 Dishes in Sink 中 no-HL 降幅明显，显式结构化子任务很重要。
- 这两项中 π0.5 高层也优于 GPT-4 HL，显示 in-domain fine-tuning 的收益。
- Items in Drawer 去掉 WD 也明显下降。
- **原文疑似笔误**：Appendix 前一句明确说 `Items in Drawer` 和 `Dishes in Sink` 的 `no HL` 性能大幅下降，Figure 17 也显示 `Dishes in Sink` 明显依赖高层推理；但紧接着却写 `Laundry Basket and Dishes in Sink` 对高层 policy 较不敏感。结合 Figure 17，后一个 `Dishes in Sink` 很可能应为 `Make Bed`。因此应按图表理解为：`Items in Drawer` 与 `Dishes in Sink` 对高层推理较敏感，而 `Laundry Basket` 与 `Make Bed` 相对不敏感。

---

## 14. Section VI：Discussion、局限与未来工作

原文位置：PDF 第 11 页，Section VI，右栏。

### 14.1 论文总结

论文主张，约 400 小时目标移动操作数据配合更大规模的其他机器人、HL、VI 和 WD co-training，能够让 π0.5 在未见家庭中执行多阶段、灵巧的厨房和卧室任务。实验支持这种特定异构 recipe 带来有效迁移。

这不是“异构数据必然导致开放世界泛化”的一般定理；它是针对该模型、数据 mixture 和家庭任务评测的 proof of concept。

### 14.2 明确局限

论文列出的失败来源：

- 某些新机构持续困难，如陌生抽屉把手或物理上难开的柜门。
- 部分可观测问题，如机械臂挡住待擦污渍。
- 高层子任务容易被干扰，如收纳时反复关开抽屉。
- prompt 相对简单；可接受的复杂度由训练标注决定，尚不能自然覆盖复杂偏好。
- context 较有限，缺少丰富记忆；跨房间导航或记住物品位置受限。
- 只探索了异构数据源的一种组合，更多来源及配比仍待研究。
- VI 很有潜力，但监督形式和规模仍有待扩展。

另外，机器人系统没有独立轨迹规划和碰撞检测，这凸显端到端能力，也意味着安全与鲁棒性不能由额外规划层兜底。

---

## 15. Appendix：评测与模型技术细节

### 15.1 Appendix B：任务评分 rubric

原文位置：PDF 第 17–18 页，Appendix B。

标准定量评测包含四任务。通常每个 task 做 10 次，每个 policy 的标准比较跨四个固定 locations，总计 40 evaluations；policy 交错执行以控制环境变化。机器人故障、时间等导致的取消 episode 会移除；样本量尽量接近，并用允许 trial 数不同的双侧 t-test 报告显著性。

评分：

| 任务 | 最大分 | 关键得分项 |
|---|---:|---|
| Dishes in Sink | 8 | 4 件物品：每件拿起 +1、放入水槽 +1 |
| Items in Drawer | 4 | 拿起、开抽屉、放入、物品已在内时关抽屉，各 +1 |
| Laundry in Basket | 3 | 导航并拿起、放入/放上、完全位于篮内，各 +1 |
| Make the Bed | 5 | 拉直毯子、两个枕头各归位、毯子很整齐、两枕头很整齐 |

结果以 rubric 总分百分比呈现，不是简单二元成功率。

### 15.2 Appendix C：语言跟随设置

原文位置：PDF 第 18 页，Appendix C，Figures 14–15。

- 两个未见厨房场景。
- 每次摆五个物体，指令要求移动一个；目标刻意放得比 distractor 更远，减少“拿最近物体”的捷径。
- 不理解语言的随机选取基线约为 20%。
- ID drawer 物体：tongs、wooden serving spoon、can opener、scissors、small yellow mustard。
- ID sink 物体：cup、bowl、plate、plastic spoon、cutting board。
- OOD drawer 物体：funnel、pill bottle、grill lighter、lighter、safety goggles；这些类别训练中未出现。

### 15.3 Appendix E：模型规模

原文位置：PDF 第 19 页，Appendix E，左栏 `Model technical details`。

VLM：

```text
PaliGemma 初始化，约 2B 参数
width=2048, depth=18, mlp_dim=16384,
num_heads=18, num_kv_heads=1, head_dim=256
```

Action expert：

```text
约 300M 参数
同样 depth/head 配置，width=1024, mlp_dim=4096
```

### 15.4 Action horizon 与 token 数

原文写 action horizon 为 50，并注明 `H=49`，即索引 `t:t+H` 含首尾共 50 个动作：

```text
[a_t, a_{t+1}, ..., a_{t+49}]
```

这能避免把 `H=49` 误读成只有 49 个动作。

### 15.5 Timestep 注入：π0.5 与 π0 的差异

原文位置：PDF 第 19 页，Appendix E，左栏中段。

- π0：把 flow timestep `τ` 与 noisy action 融合后输入 Transformer。
- π0.5：用单独 MLP 投影 `τ`，再通过 adaptive RMSNorm 注入 action expert 的每一层。

```text
MLP(τ) = swish(W2 · swish(W1 · φ(τ)))
```

- `φ:R→R^w`：sinusoidal positional encoding。
- `W1,W2∈R^{w×w}`。

连续 noisy action chunk 先经一个线性层投影到 Transformer embedding dimension；action expert 输出再经最后线性层解码为目标向量场。

### 15.6 Timestep 采样

原文位置：PDF 第 19 页，Appendix E，右栏，Figure 18 后。

论文不使用标准均匀 `τ~U(0,1)`，而沿用 π0 的 low-timestep-heavy 分布：

```text
p(τ) = Beta((s-τ)/s; α=1.5, β=1)
s = 0.999
```

- 超过阈值 `s` 的 timestep 不采样。
- 若积分步长 `δ > 1-s`，无需训练该末端区间。
- `s=0.999` 可支持最多 1000 个积分步，因为要求 `δ>0.001`。

这里 Beta 分布参数中的 `α=1.5` 与 Equation (1) 的 loss 权重 `α=10.0` **同名但不是同一个量**。

### 15.7 图像增强

原文位置：PDF 第 19 页，Appendix E，右栏末。

按顺序：

```text
RandomCrop 到宽高的 95%
Resize 回原尺寸
Rotate(-5°, +5°)
ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
```

---

## 16. 训练与推理总览

### 16.1 预训练伪代码

原文依据：PDF 第 4–7 页，Figure 3、Section IV-B/C。

```python
# 仅表达论文数据流，不代表官方实现 API
initialize_vlm_from_paligemma()

for step in range(280_000):
    sample = sample_mixture(MM, ME, CE, HL, WD)
    prefix, target_sequence = multimodal_format(sample)

    if sample.has_robot_action:
        normalized_action = quantile_normalize(sample.action)
        fast_tokens = FAST.encode(normalized_action)
        target_sequence += fast_tokens

    logits = vlm_autoregressive(prefix, routing=rho, mask=A)
    loss = cross_entropy(target_sequence, logits)  # α = 0
    update(loss)
```

### 16.2 后训练伪代码

原文依据：PDF 第 5、7 页，Equation (1)、Section IV-D。

```python
initialize_action_expert_randomly()

for step in range(80_000):
    sample = sample_mixture(filtered_MM, filtered_ME, HL, WD, VI)

    # 离散分支：保持文本、高层语义与 FAST 预测能力
    discrete_loss = cross_entropy(target_tokens, text_logits)

    # 连续分支：按原文符号
    tau = sample_low_timestep_heavy_distribution()
    omega = standard_normal_like(action_chunk)
    noisy_action = tau * action_chunk + (1 - tau) * omega
    predicted_field = action_expert(noisy_action, observation, prompt, tau)
    flow_loss = squared_norm((omega - action_chunk) - predicted_field)

    loss = discrete_loss + 10.0 * flow_loss
    update(loss)
```

注意：这段训练伪代码严格照 Equation (1) 使用 `ω-a`。它没有暗示推理时可以用正步长直接相加；积分符号必须与模型向量场定义匹配。

### 16.3 两阶段推理伪代码

原文依据：PDF 第 2、5 页，Figure 3、Section IV-A/B；正文明确写 10 denoising steps。

```python
observation_high = all_four_cameras + tokenized_robot_state
subtask = autoregressive_text_decode(
    observation_high,
    overall_task_prompt,
)

observation_low = front_and_wrist_cameras + tokenized_robot_state
current = standard_normal_action_chunk()

for k in range(10):
    tau = integration_schedule[k]
    field = action_expert(current, observation_low, subtask, tau)
    current = integrate_with_matching_sign_convention(current, field, tau)

execute_action_chunk_with_pd_tracking(current, control_rate_hz=50)
```

`integrate_with_matching_sign_convention` 是刻意写出的占位名：论文当前 PDF 没在正文给出 Euler 更新式，而 Equation (1) 的 `ω-a` 与递增 `τ` 导数相反。实现时必须选择一致的反向积分或负号更新，不能凭旧 π0 公式猜测。

### 16.4 统一数据流图

```text
                         PRE-TRAINING (280k)
MM + ME + CE robot actions --normalize--> FAST discrete tokens --┐
HL subtasks + boxes ---------------------------------------------├--> PaliGemma VLM
WD caption/VQA/localization -------------------------------------┘    next-token CE, α=0
                                                                        |
                                                                        v
                         POST-TRAINING (80k)
filtered MM + ME + HL + WD + VI --> text/FAST CE -----------------------┐
                                                                        ├--> shared attention
real action + τ + noise --> continuous noisy action --> action expert --┘    + 10× flow MSE

                            INFERENCE
all cameras + state + overall task --> same VLM --> subtask text ℓ̂
front/wrist cameras + state + ℓ̂ + noise --> same model's action expert
                                        --> 10 denoising steps
                                        --> 50-action chunk
                                        --> PD tracking at 50 Hz
```

---

## 17. 一句话总结

```text
π0.5 以 π0 的 PaliGemma + action expert 为基础，
先用 FAST 把异构机器人动作、语义任务和网页任务统一成高效的离散 token 预训练，
再加入连续 flow-matching action expert 做家庭移动操作后训练；
推理时同一个模型先生成当前高层子任务，再以该子任务为条件生成连续动作 chunk，
从而在论文限定的新家庭长程任务中展示 open-world generalization。
```
