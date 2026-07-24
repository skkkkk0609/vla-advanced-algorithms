# SmolVLA 论文伴读笔记（中文版）

论文：[smolvla-a-vision-language-action-model-for-affordable-and-efficient-robotics.pdf](/home/robot/projects/vla-advanced-algorithms/smolvla/smolvla-a-vision-language-action-model-for-affordable-and-efficient-robotics.pdf)

> 本笔记严格按论文顺序整理，重要内容标注 PDF 阅读器页码及 Section、Figure、Table 或 Algorithm。本文 PDF 共 24 页；当前版本中 PDF 阅读器页码与页面底部的论文印刷页码基本一致。为避免歧义，下文默认“PDF 第 x 页”均指 PDF 阅读器显示页码。
>
> **公式勘误提醒**：Section 3.1 原文同时写出 `A^τ = τA + (1-τ)ε` 和目标向量场 `u(A^τ|A)=ε-A`。若 `τ` 从 0 递增到 1，对前一插值求导应得到 `A-ε`，与原文目标相反。论文该处没有交代反向时间或负步长约定。下文保留原文公式并明确讨论这一符号一致性疑点，不静默改写。

## 目录

- [1. Section 1：Introduction](#1-section-1introduction)
- [2. Section 2：Related Work](#2-section-2related-work)
- [3. Section 3：SmolVLA 总览](#3-section-3smolvla-总览)
- [4. Section 3.1：模型架构](#4-section-31模型架构)
- [5. Flow Matching：公式与符号疑点](#5-flow-matching公式与符号疑点)
- [6. 交替 Cross-Attention 与 Causal Self-Attention](#6-交替-cross-attention-与-causal-self-attention)
- [7. Section 3.2：社区预训练数据](#7-section-32社区预训练数据)
- [8. Section 3.3：异步推理](#8-section-33异步推理)
- [9. Section 4：实验设置、机器人与实现细节](#9-section-4实验设置机器人与实现细节)
- [10. Section 4.4–4.5：基线与主结果](#10-section-44–45基线与主结果)
- [11. Section 4.6：异步推理实验](#11-section-46异步推理实验)
- [12. Section 4.7：完整消融 Table 6–13](#12-section-47完整消融-table-6–13)
- [13. Section 5：Discussion 与 Limitations](#13-section-5discussion-与-limitations)
- [14. Appendix A：社区数据集](#14-appendix-a社区数据集)
- [15. 理解映射：从论文流程到 LeRobot／真机部署](#15-理解映射从论文流程到-lerobot真机部署)
- [16. 关键配置速查](#16-关键配置速查)

---

## 1. Section 1：Introduction

原文位置：PDF 第 1–2 页，Abstract、Figure 1、Section 1。

### 1.1 问题背景

论文从 foundation model（基础模型）在语言和多模态领域的进展出发，指出机器人策略仍难以跨物体、位置、环境和任务泛化。Vision-Language-Action model（视觉-语言-动作模型，VLA）把预训练 VLM 的视觉、语言和世界知识用于机器人控制，但当前 VLA 常有数十亿参数，训练昂贵、部署门槛高，而且很多成果并未完整公开训练细节。

SmolVLA 的目标不是追求更大的模型，而是建立一个：

- 小型、训练和推理高效的 VLA；
- 可在单张 GPU 上训练，可部署于消费级 GPU，甚至 CPU；
- 使用低成本机器人和社区公开数据；
- 模型、代码、训练数据与硬件均开放；
- 在明显小于现有 VLA 的参数规模下保持有竞争力的表现。

### 1.2 三项主要贡献

原文位置：PDF 第 2 页，Section 1 末尾。

1. **轻量架构（lightweight architecture）**：跳过 VLM 后部层、减少视觉 token、使用小型预训练 VLM，并在 Action Expert 中交替使用 self-attention 与更轻量的 cross-attention。
2. **社区数据预训练（community-driven pretraining）**：使用不足 30K episodes 的公开社区数据；精确统计见 Table 1，为 481 个数据集、22.9K episodes、10.6M frames。
3. **异步推理（asynchronous inference）**：把动作执行与观测处理／动作预测解耦，减少推理造成的停顿，提高控制响应速度。

> “训练在单张 GPU 上可行”是模型可训练性的主张；论文实际预训练为容纳大 global batch 使用了 4 张 GPU，项目总计约 30K GPU hours（Section 4.3）。两者不能混为“论文的正式预训练只用了一张卡”。

---

## 2. Section 2：Related Work

原文位置：PDF 第 2–3 页，Section 2。

### 2.1 Vision-Language Models（VLMs）

典型 VLM 把预训练 vision encoder（视觉编码器）与预训练 LLM 组合，先在图文和交错视觉语言语料上预训练，再做 instruction tuning。相关研究也在探索不依赖预训练视觉编码器、把图像与文本统一为离散 token，以及通过小数据、小模型或参数高效微调降低成本。

### 2.2 Vision-Language-Action Models（VLAs）

- Octo、RT-1 等从大规模机器人示范训练 Transformer policy。
- RT-2 把预训练 VLM 进一步用于机器人数据。
- OpenVLA 是 7B 开放模型，生成离散 action token。
- π0、DexVLA 使用 diffusion/flow 类 Action Expert 生成连续动作。
- 自回归动作 tokenizer 虽改善传统分箱，但推理仍受逐 token 生成限制。
- TinyVLA 追求 sub-1B 规模，但论文认为缺少大规模机器人预训练会限制更广泛泛化。

SmolVLA 与这些工作的共同方向是开放和高效，差异重点在小型预训练 VLM、轻量 Action Expert、社区数据预训练及异步执行栈。

---

## 3. Section 3：SmolVLA 总览

原文位置：PDF 第 3 页，Section 3 Overview；PDF 第 1 页，Figure 1。

SmolVLA 由两个主要组件组成：

```text
多路 RGB 图像 + 自然语言任务 + 机器人 sensorimotor state
                         ↓
              compact pretrained VLM
             （负责 perception）
                         ↓ VLM features
                 Action Expert
       （flow matching，负责 action generation）
                         ↓
                 action chunk A_t
```

训练和使用顺序为：

1. 在社区收集的数据上用 imitation learning（模仿学习）预训练 Action Expert。
2. 在仿真与真实任务上微调、评估。
3. 推理时可采用同步执行；论文另提出异步执行栈，让预测和执行并行。

Figure 1 的关键信息是：VLM 最后的 `L-N` 层被丢弃，只保留前部层；语言、RGB 图像和状态由 VLM 编码；Action Expert 交替使用 cross-attention 与 self-attention，并通过 flow matching 输出动作 chunk。

---

## 4. Section 3.1：模型架构

原文位置：PDF 第 3–4 页，Section 3.1；PDF 第 1 页，Figure 1；相关消融见 PDF 第 12–13 页，Table 6–11。

### 4.1 SmolVLM2、SigLIP 与 SmolLM2

SmolVLA 选择 **SmolVLM-2** 作为 perception backbone。其组成是：

```text
图像 → SigLIP vision encoder → visual features
文本 → tokenizer → text tokens
visual features + text tokens → SmolLM2 language decoder
```

- **SigLIP**：编码视觉输入。
- **SmolLM2**：语言解码器，融合视觉、语言及状态 token。
- SmolVLM-2 原本针对多图和视频输入优化。
- SmolVLA 训练时冻结 VLM，只训练 Action Expert；这不是对 SmolVLM-2 端到端微调。

### 4.2 输入与三个 projector

论文提到三类线性投影层：

| 投影器 | 输入 → 输出 | 作用 |
|---|---|---|
| state projector | sensorimotor state → VLM token dimension | 把整个机器人状态投影成一个 state token，加入 VLM prefix |
| action projector | noisy continuous actions → Action Expert dimension | 把动作向量变成 Action Expert 可处理的 action token |
| feature projector | VLM features → Action Expert dimension | 对齐 VLM 与 Action Expert 的隐藏维度 |

VLM 侧最终拼接：

```text
visual tokens + language tokens + one state token
                         ↓
                 SmolLM2 decoder
                         ↓
          第 N 层及之前得到的 VLM features
```

这里的 state 是 VLM 的 prefix，而非只在 Action Expert 后缀中输入。Table 11 进一步验证该选择。

### 4.3 每帧 64 个 visual tokens

SmolVLM-2 原训练方案支持 image tiling，即全局图外再处理多个 crop。SmolVLA 为降低推理开销：

- 不使用 tiling；
- 只使用 global image；
- 结合 pixel shuffle；
- 将每帧视觉 token 限制为 **64**。

注意：Section 4.3 同时规定输入图像 resize 到 **512×512**。512×512 是图像输入尺寸，64 是每帧进入语言模型的视觉 token 数，两者含义不同。

### 4.4 只使用前 `N=L/2` 层

论文不从完整 VLM 的最后层提取特征，而是在训练开始前丢弃顶端 `L-N` 层，只使用前 `N` 层产生的特征。默认：

```text
N = L / 2
```

主模型的具体配置是 **SmolLM2 前 16 层**（Section 4.3）。因此，在该主配置中可对应理解为总层数 `L=32`、保留 `N=16`；“32”也出现在 Table 8 的完整层数实验中。

论文声称这种做法在速度与性能间取得平衡，并“有效减半 LLM 和 Action Expert 的计算成本”。Table 8 显示，使用前半层与隔层抽取（Skip %2）不是同一方案；前半层方案的效果更好。

### 4.5 Action Expert

Action Expert 记为 `v_θ`，是一个 Transformer。它接收：

- VLM 第 `N` 层得到的条件特征；
- 经 action projector 投影的 noisy action chunk；
- flow timestep `τ`（公式中作为向量场输入的一部分表达）。

它输出连续动作的 vector field（向量场），再通过 flow integration 得到最终 action chunk。默认 expert hidden size 为 VLM hidden dimension `d` 的 **0.75 倍**；主模型总参数约 **450M**，其中 Action Expert 约 **100M**。

---

## 5. Flow Matching：公式与符号疑点

原文位置：PDF 第 4 页，Section 3.1，`Flow matching action expert` 段落。

### 5.1 论文给出的训练目标

论文写为：

```text
L^τ(θ) = E_{p(A_t|o_t), q(A_t^τ|A_t)} [ ||v_θ(A_t^τ, o_t) - u(A_t^τ|A_t)||² ]
```

并定义：

```text
A_t^τ = τ A_t + (1-τ) ε
ε ~ N(0, I)
u(A_t^τ|A_t) = ε - A_t
```

`τ` 从 Beta distribution（Beta 分布）采样，论文称这一点沿用 Black et al. (2024)。

### 5.2 逐符号解释

- `t`：机器人控制时刻。
- `o_t`：原始 observation，包含图像、语言任务和 sensorimotor state。
- `o_t`（原文在公式说明中复用同一字形）：从 observation 经 VLM 到第 `N` 层提取的 VLM features。原文排版对原始观测与特征使用了容易混淆的记号，阅读时应依上下文区分。
- `A_t=(a_t,...,a_{t+n})`：从当前时刻开始的真实示范 action chunk。论文文字称 chunk 含 `n` 个时间步，但公式端点写到 `t+n`，按通常计数会产生 `n+1` 项；实现配置明确写 `n=50 actions`，宜以“50 个动作的 chunk”理解，不自行扩展为 51。
- `a_t`：某一个时间步的低层连续动作。
- `ε`：与 action chunk 同形状的标准高斯噪声，`N(0,I)`。
- `τ`：flow timestep／插值系数。
- `A_t^τ`：真实动作与噪声之间的中间样本。
- `q(A_t^τ|A_t)`：给定真实动作后产生中间 noisy action 的条件路径分布。
- `p(A_t|o_t)`：数据中的、以观测为条件的动作分布。
- `v_θ(A_t^τ,o_t)`：Action Expert 预测的条件向量场。
- `u(A_t^τ|A_t)`：监督用目标向量场。
- `θ`：Action Expert 的可训练参数；VLM 冻结。
- `||·||²`：预测向量场与目标向量场之间的平方 L2 误差。
- `E`：对数据动作、噪声路径及 timestep 采样取期望。

### 5.3 插值端点

按论文的插值式：

```text
τ = 0  → A_t^τ = ε
τ = 1  → A_t^τ = A_t
```

因此，若推理沿 `τ: 0 → 1` 前进，样本应从噪声走向动作数据。

### 5.4 原文目标符号的一致性疑点

对原文插值直接求导：

```text
dA_t^τ / dτ = A_t - ε
```

但论文写的目标是：

```text
u(A_t^τ|A_t) = ε - A_t
```

两者互为相反数。要让 `ε-A_t` 与动力学一致，需要额外采用反向的时间参数化、从 `τ=1` 积分到 `τ=0`、或在更新中使用负步长／负号；然而 Section 3.1 没有明确给出这样的约定，Section 4.3 只说明推理固定为 **10 个 flow steps**，没有补充积分方向。

因此能严格依据论文得到的结论是：

- 原文确实同时给出上述插值与 `ε-A_t` 目标；
- 两者在“`τ` 递增且速度等于路径导数”的通常解释下符号不一致；
- 这是需要对照作者实现或勘误确认的疑点；
- 不能在伴读笔记中擅自把目标改成 `A_t-ε`，也不能未经依据断言实现一定采用某种反向积分。

---

## 6. 交替 Cross-Attention 与 Causal Self-Attention

原文位置：PDF 第 4 页，Section 3.1；PDF 第 1 页，Figure 1；PDF 第 12 页，Table 6–7。

Action Expert 的 block 不是每个 block 同时含 SA 和 CA，而是让 block **交替**包含其中一种注意力：

```text
CA block → SA block → CA block → SA block → ...
```

### 6.1 Cross-Attention（CA）

- action token 形成 query；
- VLM features 提供 key 和 value；
- 用途是让动作生成条件于图像、语言和状态的 VLM 表征。

### 6.2 Causal Self-Attention（causal SA）

- query、key、value 均来自 Action Expert 内的 action tokens；
- 使用 causal mask；
- 第 `i` 个 action token 只能看当前及过去 action token，不能看 chunk 中未来 token；
- 论文称这可防止 future action dependencies，并观察到 SA 使真实机器人上的 action chunk 更平滑。

Table 6 表明 CA 单独优于 SA 单独，交替 CA+SA 最好。Table 7 表明 causal SA 明显优于 bidirectional SA。这里“causal”限制的是一个 chunk 内 action token 之间的可见性，不表示 VLM 文本一定在做自回归动作生成。

---

## 7. Section 3.2：社区预训练数据

原文位置：PDF 第 4–5 页，Section 3.2、Table 1；PDF 第 20–24 页，Appendix A.1。

### 7.1 数据规模

Table 1 的精确统计为：

| datasets | episodes | frames |
|---:|---:|---:|
| 481 | 22.9K | 10.6M |

Table 1 caption 的正文写法出现“`At ~10M episodes`”，但表中明确是 **10.6M frames**、22.9K episodes；结合上下文，这是 caption 中单位／措辞不一致，不能把数据规模写成 10M episodes。

数据来自 Hugging Face 社区贡献，并按 robot embodiment、episode 数、整体数据质量和 frame coverage 过滤。社区数据的价值在于覆盖多样控制方式、相机视角、任务、环境与物体交互，但其噪声和异构性也需要清洗。

### 7.2 任务标注清洗

社区数据里存在：

- `task desc` 一类占位文本；
- `Hold`、`Up` 等过于模糊的指令；
- 完全缺失任务指令的数据。

作者使用现成的 **Qwen2.5-VL-3B-Instruct** 自动生成简短、面向动作的任务描述。输入包括每个数据集采样的 representative frames 和原始 instruction，输出为概括机器人行为的短句。完整 prompt 见 Appendix A.1。

### 7.3 相机视角规范化

不同数据集的相机字段名不能可靠表示实际视角，例如 `images.laptop` 可能对应 top、side 或 wrist。作者手动映射为统一顺序：

```text
优先级：top → wrist → side
字段名：OBS_IMAGE_1, OBS_IMAGE_2, OBS_IMAGE_3
```

额外视角保持其原顺序，但训练时不用的视角会被丢弃。论文只说未来可用 VLM 自动化该过程，没有声称本文已经自动识别视角。

---

## 8. Section 3.3：异步推理

原文位置：PDF 第 5–7 页，Section 3.3、Figure 2、Algorithm 1、Figure 3。

### 8.1 Action queue 与同步推理

policy 根据观测输出 action chunk：

```text
π(o_t) = A_t
A_t = (a_t, a_{t+1}, ..., a_{t+n})
a_t ← PopFront(A_t)
```

论文将**耗尽整个 chunk 后才采集新观测并预测下一 chunk**称为 synchronous inference（同步推理，sync）。优点是每 `n` 步才分配一次推理计算；缺点是：

- chunk 执行期间是 open-loop；
- 计算下一 chunk 时队列已经空，机器人会等待，形成 blind lag。

另一极端是每个控制步都采集观测、预测并聚合重叠 chunk，响应更及时，但资源消耗高。

### 8.2 异步推理的核心

异步栈将两个循环解耦：

```text
RobotClient：持续 PopFront(queue) 并执行动作
PolicyServer：接收 observation，计算新 action chunk
```

RobotClient 在旧队列尚未耗尽时就触发非阻塞预测；新 chunk 返回后，通过聚合函数 `f(old_queue, new_queue)` 处理重叠并更新队列。PolicyServer 可以在远端 GPU 上运行。

### 8.3 阈值 `g`

触发条件为：

```text
|A_t| / n < g,    g ∈ [0,1]
```

- `|A_t|`：当前队列剩余动作数。
- `n`：完整 chunk size。
- `g`：剩余比例阈值。

三种代表情形（Figure 3A）：

- `g=0`：顺序极限。队列完全耗尽才请求，推理期间机器人停顿。
- `g=0.7`：消耗约 `(1-g)=0.3` 的旧队列后触发；推理与剩余动作执行重叠。
- `g=1`：每个 timestep 都触发观测／推理，响应最强但计算最贵。

论文给出避免队列耗尽的近似条件：

```text
g ≥ E[ℓ] / (n Δt)
```

其中：

- `ℓ`：发送观测、server inference、返回动作 chunk 的总延迟随机变量；
- `E[ℓ] ≃ E[ℓ_S]`：若双向通信时间相等且相对 server inference 可忽略；
- `Δt`：控制周期；30 FPS 时为 33 ms。

无 similarity filter 时，平均每 `(1-g)nΔt` 秒发送观测，约在 `(1-g)nΔt + E[ℓ_S]` 后收到新 chunk。

### 8.4 相似度过滤

为减少重复 server 调用和运行时不稳定，作者在 **joint space（关节空间）** 比较观测：

- 距离低于预设 `ε∈R+`，视为 near-duplicate 并丢弃；
- 当 client queue 最终为空时，无论相似度如何，都强制处理最近观测，避免永远没有新动作。

Figure 3B 展示 similarity filter 防止近乎相同的观测持续产生并聚合近似动作队列；图中红箭头表示队列为空时绕过过滤器。

### 8.5 Algorithm 1 逐步解读

Algorithm 1 输入 horizon `T`、chunk size `n`、阈值 `g`：

1. 捕获 `o_0`，发给 PolicyServer，阻塞接收初始 `A_0=π(o_0)`。
2. 每个控制步从队首取 `a_t` 并执行。
3. 若剩余比例低于 `g`，捕获 `o_{t+1}`。
4. 若 `NeedsProcessing(o_{t+1})` 通过相似度判断或被强制触发，则以 `AsyncInfer` 非阻塞预测新队列 `Ã_{t+1}`。
5. 推理完成后，用 `f(A_t,Ã_{t+1})` 聚合可能重叠的队列。
6. 若 async handle 尚未完成，则 `A_{t+1}=A_t`，继续消费旧队列。

> Algorithm 1 是论文层面的伪代码。它没有定义 `f` 的具体加权公式、网络协议、线程模型或 LeRobot 类名，因此不能从论文单独推断这些实现 API。

---

## 9. Section 4：实验设置、机器人与实现细节

### 9.1 实验设置与指标

原文位置：PDF 第 7–9 页，Section 4.1、Figure 4。

除非另有说明，SmolVLA 采用 multi-task training。

**仿真：**

- LIBERO：Spatial、Object、Goal、Long 四类，每类 10 个任务，共 40 个；数据集 1,693 episodes；每任务评估 10 trials。
- Meta-World：50 个任务，按 easy、medium、hard、very hard 分类；作者新收集每任务 50 demonstrations，共 2,500 episodes；每任务评估 10 trials。
- 仿真 success rate（SR）为二元：完整完成记 1，否则 0。

**真实世界：**

- SO100：Pick-Place、Stacking、Sorting 三个数据集。
- SO101：Pick-Place-Lego 一个数据集；SmolVLA 预训练中没有 SO101 数据。
- 每个目标数据集通常为 5 个起始位置、每位置 10 trajectories，共 50 demonstrations。
- Pick-Place／Stacking：抓取成功 0.5，放置／堆叠成功再加 0.5。
- Sorting：两个方块的抓取与正确盒子匹配拆成四项，每项 0.25。
- Figure 4 显示 SO100 使用 top+wrist cameras，SO101 使用 top+side cameras。

### 9.2 机器人

原文位置：PDF 第 9 页，Section 4.2。

- **SO100 / SO101**：低成本、开源、可 3D 打印的 6-DOF 机械臂，以位置命令控制低成本舵机；SO101 改进装配和电机，运动更平滑。
- **Panda**：7-DOF torque-controlled Franka Emika Panda，用于 LIBERO 仿真。
- **Sawyer**：论文写作 `Swayer`，用于 Meta-World；文中描述为 4-DOF 控制，policy 控制位置和 gripper state。

### 9.3 实现细节

原文位置：PDF 第 10 页，Section 4.3。

| 项目 | 论文配置 |
|---|---|
| 框架 | LeRobot / PyTorch |
| 社区数据预训练 | **200,000 steps** |
| global batch size | **256** |
| warmup | 100 steps |
| learning rate | cosine，`1e-4 → 2.5e-6` |
| optimizer | AdamW，`β1=0.9, β2=0.95` |
| image size | **512×512** |
| VLM | SmolVLM-2 |
| chunk size | **n=50 actions** |
| inference flow | **10 steps** |
| 可训练模块 | **仅 Action Expert** |
| VLM | **frozen** |
| 主模型参数 | **450M total / 约 100M Action Expert** |
| 使用 LLM 层 | **前 16 层** |
| 仿真微调 | 100,000 steps，batch 64 |
| 真实任务微调 | 200,000 steps |
| 数值／编译 | bfloat16、`torch.compile()` |
| 分布式训练 | Hugging Face Accelerate + mixed precision |

为固定 sequence length 和 batch size，不能组成完整 batch 的 episode 多余 frames 会被丢弃。预训练实际使用 4 GPUs 以容纳大 batch，但论文称小模型也可在单 GPU 训练。项目总消耗约 30K GPU hours。

### 9.4 必须区分的推理协议

Section 4.3 明确说明：

- **真实世界主评估使用 synchronous inference**：完整执行 action chunk 后才采样新观测。
- **仿真评估**：每执行一个 action 就采样新观测并重新预测。
- **async 不是所有真实世界主表的默认方式**，而是在独立的 Section 4.6 / Figure 5 中比较。

因此 Table 3、Table 4、Table 5 的主结果不能描述成“异步推理下的主评估”。

---

## 10. Section 4.4–4.5：基线与主结果

### 10.1 基线

原文位置：PDF 第 10 页，Section 4.4。

- **π0**：3.3B 级 VLA，PaliGemma backbone，约 10,000 小时跨 embodiment 机器人数据预训练，使用 flow matching 预测 action chunk，输入三路 RGB、状态和语言。
- **ACT**：约 80M 参数的 CVAE Transformer，ImageNet 预训练 ResNet vision encoder，连续动作回归，输入图像和状态。

真实世界比较中，SmolVLA 先做社区数据预训练；π0 在各目标数据上微调；ACT 在每个数据集上从头训练。

### 10.2 仿真主结果（Table 2）

原文位置：PDF 第 10–11 页，Section 4.5、Table 2。

**LIBERO success rate (%)：**

| Policy | VLA robotics pretraining | Spatial | Object | Goal | Long | Avg. |
|---|---|---:|---:|---:|---:|---:|
| Diffusion Policy | No | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo (0.09B) | Yes | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA (7B) | Yes | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 (PaliGemma-3B) | No | 87 | 63 | 89 | 48 | 71.8 |
| π0 (3.3B) | Yes | 90 | 86 | 95 | 73 | 86.0 |
| SmolVLA (0.24B) | No | 87 | 93 | 88 | 63 | 82.75 |
| SmolVLA (0.45B) | No | 90 | 96 | 92 | 71 | 87.3 |
| SmolVLA (2.25B) | No | 93 | 94 | 91 | 77 | 88.75 |

**Meta-World success rate (%)：**

| Policy | VLA robotics pretraining | Easy | Medium | Hard | Very Hard | Avg. |
|---|---|---:|---:|---:|---:|---:|
| Diffusion Policy | No | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA | No | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π0 (3.5B-PaliGemma) | No | 80.4 | 40.9 | 36.7 | 44.0 | 50.5 |
| π0 (3.5B) | Yes | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA (0.24B) | No | 86.43 | 46.36 | 35 | 60 | 56.95 |
| SmolVLA (0.45B) | No | 82.5 | 41.8 | 45.0 | 60.0 | 57.3 |
| SmolVLA (2.25B) | No | 87.14 | 51.82 | 70 | 64 | 68.24 |

论文称，相比 π0，SmolVLA 训练约快 40%，内存消耗低 6 倍。Table 2 中 `VLA Pt.` 指机器人数据预训练；SmolVLA 这些仿真行只从 VLM 初始化，因此标为 No。

### 10.3 真实 SO100 主结果（Table 3）

原文位置：PDF 第 11 页，Table 3。

| Training | Policy | Pick-Place | Stacking | Sorting | Avg. |
|---|---|---:|---:|---:|---:|
| Single-task | ACT | 70 | 50 | 25 | 48.3 |
| Multi-task | π0 (3.5B) | 100 | 40 | 45 | 61.7 |
| Multi-task | SmolVLA (0.45B) | 75 | 90 | 70 | 78.3 |

### 10.4 真实 SO101 主结果（Table 4）

原文位置：PDF 第 11 页，Table 4。

Pick-Place-Lego，single-task training：

| Policy | In Distribution | Out of Distribution |
|---|---:|---:|
| ACT | 70 | 40 |
| SmolVLA (0.45B) | 90 | 50 |

OOD 指 Lego 放在训练未见的新位置。论文借此说明社区预训练只含 SO100 时，SmolVLA 仍可微调到 SO101；这不等于 zero-shot 跨机器人迁移。

### 10.5 预训练与多任务（Table 5）

原文位置：PDF 第 11 页，Section 4.5、Table 5。

| Training | VLA pretraining | Pick-Place | Stacking | Sorting | Avg. |
|---|---|---:|---:|---:|---:|
| Single-task | No | 55 | 45 | 20 | 40.0 |
| Multi-task | No | 80 | 40 | 35 | 51.7 |
| Multi-task | Yes | 75 | 90 | 70 | 78.3 |

论文结论：无预训练时 multi-task 相比 single-task 提高平均分；社区数据预训练再把平均分从 51.7 提升至 78.3。Pick-Place 单项从 80 降到 75，增益主要来自 Stacking 和 Sorting，因此“预训练所有任务都逐项提高”不符合表格。

---

## 11. Section 4.6：异步推理实验

原文位置：PDF 第 11–12 页，Section 4.6、Figure 5。

这是与真实世界 synchronous 主评估分开的专门实验。作者在三个 SO100 任务上比较 sync 与 async；Figure 5 caption 说明超参数在 Pick-Place 上优化后复用于其他任务。

### 11.1 成功率

| Inference | Pick-Place | Stacking | Sorting | Avg. |
|---|---:|---:|---:|---:|
| Sync | 75 | 90 | 70 | 78.3 |
| Async | 80 | 90 | 50 | 73.3 |

论文概括为 comparable success rates，但数值上 async 平均分低 5.0，Sorting 低 20；因此更准确的表述是：Pick-Place 略升、Stacking 持平、Sorting 下降，平均成功率接近但并非更高。

### 11.2 完成时间与固定时间吞吐

Pick-Place 速度实验：10 trials、5 个 cube positions，从机器人开始移动计时。

| Inference | Total time (s) | Avg. (s) | Std. |
|---|---:|---:|---:|
| Sync | 137.5 | 13.75 | 2.42 |
| Async | 97.0 | 9.70 | 2.95 |

async 平均 9.7 秒，sync 13.75 秒，约快 30%。

固定时间实验（正文举例为 60 秒）：

| Inference | Total cubes | Avg. | Std. |
|---|---:|---:|---:|
| Sync | 9 | 1.8 | 0.45 |
| Async | 19 | 3.8 | 1.3 |

论文还定性观察到 async 对物体位置变化和外部扰动反应更快；这是作者的 qualitative observation，不应扩展成未量化的普适鲁棒性结论。

---

## 12. Section 4.7：完整消融 Table 6–13

原文位置：PDF 第 12–14 页，Section 4.7、Table 6–13。

统一条件：全部消融在 LIBERO 上；除非另有说明，从头训练、无机器人数据预训练；VLM frozen，仅 Action Expert 从头训练。列 `S/O/G/10` 分别对应 Spatial/Object/Goal/Long；PDF 表头因字体提取可能把 Long 显示为 `10`，数值语境实际是 LIBERO Long 类别。

### 12.1 Table 6：Cross vs Self-Attention

| Attention mechanism | S | O | G | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| CA | 87 | 92 | 83 | 54 | 79.0 |
| SA | 80 | 94 | 84 | 40 | 74.5 |
| CA+SA (ours) | 86 | 99 | 90 | 67 | 85.5 |

结论：CA 单独优于 SA 单独；交替 CA+SA 平均最好，尤其 Long 为 67。

### 12.2 Table 7：Bidirectional vs Causal Attention

| Attention mask | S | O | G | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| Bidir | 79 | 86 | 82 | 23 | 67.5 |
| Causal | 80 | 94 | 84 | 40 | 74.5 |

结论：阻止 future action leakage 的 causal mask 明显更好。正文也把 pure CA 视为“action token 间无交互”的参照，并指出其表现仍有竞争力。

### 12.3 Table 8：使用多少个 VLM LLM 层

| `N` / variant | S | O | G | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| 8 | 77 | 88 | 86 | 49 | 75.0 |
| 16 | 88 | 91 | 91 | 44 | 78.5 |
| 24 | 86 | 97 | 86 | 49 | 79.5 |
| 32 | 89 | 94 | 85 | 53 | 80.3 |
| Skip %2 | 84 | 90 | 83 | 45 | 75.5 |
| VLM-256M | 86 | 83 | 75 | 59 | 75.8 |

结论不是“16 层绝对成功率最高”：全 32 层平均 80.3，高于 16 层的 78.5；论文选择前半层是性能与算力折中。保留大 VLM 的前层也优于将 backbone 直接缩成 VLM-256M；隔层 Skip %2 不及直接取前 `N` 层。

### 12.4 Table 9：Action Expert 宽度

| Expert width (w.r.t. VLM) | S | O | G | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| ×1.00 | 87 | 96 | 90 | 56 | 82.3 |
| ×0.75 | 82 | 89 | 84 | 55 | 77.5 |
| ×0.50 | 89 | 94 | 85 | 53 | 80.3 |
| ×0.25 | 76 | 97 | 83 | 39 | 73.8 |

论文采用 `0.75×d` 作为性能与效率平衡。需注意表中 `0.50×` 平均 80.3 高于 `0.75×` 的 77.5，因此 caption 的“larger capacities yield better success rates”并非每档单调；`1.00×` 的确最高。

### 12.5 Table 10：Flow Matching vs Regression

| Training objective | S | O | G | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| Flow matching | 89 | 94 | 85 | 53 | 80.25 |
| Regression (L1) | 92 | 85 | 86 | 38 | 75.25 |

Flow matching 平均高 5.0，主要差距在 Object 和 Long；Regression 的 Spatial 反而更高。

### 12.6 Table 11：State 放在 VLM prefix 还是 expert suffix

| States | Attention | S | O | G | Long | Avg. |
|---|---|---:|---:|---:|---:|---:|
| Prefix | CA | 89 | 94 | 85 | 53 | 80.3 |
| Suffix | CA | 86 | 82 | 78 | 47 | 73.3 |
| Prefix | SA | 62 | 74 | 57 | 20 | 53.3 |
| Suffix | SA | 80 | 92 | 80 | 47 | 74.8 |

Table 11 caption 笼统概括“将 state 输入 VLM（prefix）优于输入 expert（suffix）”，但表格对 SA 恰好相反：SA 的 Prefix 53.3，Suffix 74.8。严格按表可得：**CA 时 prefix 更好；SA 时 suffix 更好**。因此 caption 的概括不能同时覆盖两种 attention 设置，这是文字结论与表中数值的不一致。

### 12.7 Table 12：Action chunk size `n`

| Chunk Size | S | O | G | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| 1 | 45 | 77 | 54 | 24 | 50.0 |
| 10 | 90 | 94 | 94 | 58 | 84.0 |
| 30 | 85 | 94 | 87 | 48 | 78.5 |
| 50 | 89 | 94 | 85 | 53 | 80.3 |
| 100 | 83 | 88 | 85 | 42 | 74.5 |

过短或过长都下降；10–50 是论文给出的平衡区间。表中 `n=10` 平均最高，但主模型训练配置选择 `n=50`，兼顾减少预测频率和性能。

### 12.8 Table 13：更新观测前执行多少动作

| Action Steps | S | O | G | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| 1 | 89 | 94 | 85 | 53 | 80.3 |
| 10 | 89 | 94 | 91 | 57 | 82.8 |
| 30 | 76 | 91 | 74 | 42 | 70.8 |
| 50 | 54 | 70 | 58 | 25 | 51.8 |

此处 Action Steps 是拿到一个预测 chunk 后，在更新 observation／覆盖当前 chunk 前实际执行的动作数，不是 flow integration steps。结果显示每 1 或 10 个动作更新观测明显优于执行 30 或 50 个，揭示推理速度与闭环响应能力的取舍。

---

## 13. Section 5：Discussion 与 Limitations

原文位置：PDF 第 14–15 页，Section 5、5.1。

### 13.1 Discussion

论文总结 SmolVLA 能在消费级硬件和低成本机器人上运行，并与更大 VLA 竞争；异步执行栈与模型无关，原则上可用于任何输出 action chunk 的 policy。作者还开放模型、代码、训练数据、机器人硬件与复现说明。

### 13.2 Limitations

1. **数据多样性与跨 embodiment**：预训练数据只来自 SO100。Table 4 展示的是微调到 SO101 后的结果，而非预训练已覆盖多机器人。未来需加入多种 embodiment。
2. **数据规模与可扩展性**：约 23K trajectories，远小于 OpenVLA 约 1M trajectories。扩大数据可能提升任务和环境泛化。
3. **模型规模与硬件效率**：小于 0.5B 便于消费级推理，但未来需要研究如何在不牺牲速度和可访问性的前提下扩展。
4. **VLM backbone 选择**：SmolVLM-2 主要针对 document reading 和 OCR 预训练，未证明是现实机器人交互的最优 VLM；可探索更专门的预训练。

这些限制意味着，论文结果支持的是“在所评估仿真和低成本机械臂任务上，小模型具有竞争力”，不应外推为任意机器人、任意场景的通用控制能力。

---

## 14. Appendix A：社区数据集

原文位置：PDF 第 20–24 页，Appendix A、A.1。

Appendix A.1 包含两部分：任务标注 prompt 和全部社区数据集 handle 清单。

### 14.1 原文任务标注 prompt 的约束

作者给 Qwen2.5-VL-3B-Instruct 的 prompt 要求：

- 输入当前 task description；
- 生成非常短、清晰、完整的一句话；
- 描述机器人手臂执行的动作；
- 最多 30 characters；
- 删除不必要词语；
- 直接以 `Pick`、`Place`、`Open` 等动作动词开始；
- 示例包括“Pick up the cube and place it in the box”“open the drawer”。

该 prompt 与 representative frames 一起用于 Section 3.2 的任务标注清洗。它做的是单句行为概括，不是轨迹分段、成功检测或 reward 标注。

### 14.2 数据集清单

Appendix A.1 在 PDF 第 20–24 页逐项列出 481 个 Hugging Face dataset handles，覆盖 pick-place、push、stack、fold、pour、button、sorting、双臂等社区任务。完整清单已在论文原文保留，本笔记不重复抄写五页 handle，以免把名称抄录错误；精确审计应直接对照 Appendix A.1。

需要与正文统一理解：

- Table 1 的 481 是选中数据集数量；
- 统计总量是 22.9K episodes、10.6M frames；
- Section 5.1 说明这些预训练轨迹来自单一 robot type：SO100；
- 清单中即使个别名称含 dual-arm、aloha 或其他字样，也不能仅凭 handle 推翻作者对实际预训练 embodiment 的明确说明。

---

## 15. 理解映射：从论文流程到 LeRobot／真机部署

> **本章是理解映射，不是论文 API。** 论文说明实验使用 LeRobot，并给出 Algorithm 1 伪代码，但没有规定具体 Python 类、字段名、RPC 协议、线程模型或 queue aggregation 实现。下面只把论文概念映射到常见 LeRobot／真机控制流程，不能当作原文接口文档。

原文依据位置：PDF 第 5–7 页，Section 3.3、Algorithm 1；PDF 第 10 页，Section 4.3。

### 15.1 Policy 输入映射

论文概念：

```text
observation o_t
├── one or more RGB images
├── natural-language task instruction
└── sensorimotor state
```

在 LeRobot 风格数据／运行循环中，可概念性映射为：

```text
policy input batch
├── 规范顺序的 camera tensors（top / wrist / side）
├── task text
└── robot state tensor（例如关节位置）
```

进入模型前图像按论文配置 resize 为 512×512；每帧经视觉编码和 pixel shuffle 后形成 64 visual tokens。状态经 state projector 成为单 token。具体 tensor key 必须以所用 LeRobot 版本、dataset feature schema 和 checkpoint config 为准，论文没有提供统一 key 名。

### 15.2 Policy 输出与 action queue

模型一次输出 `n=50` 个连续动作：

```text
policy(observation) → action chunk → queue
控制循环每一步：action = queue.pop_front() → robot.send_action(action)
```

这里必须区分：

- **chunk size 50**：模型一次预测多少个动作；
- **flow steps 10**：生成这段 chunk 时向量场积分／去噪迭代次数；
- **action execution steps**：采新观测前实际执行 queue 中多少动作，Table 13 对此消融。

### 15.3 同步真机循环

论文真实世界主评估采用：

```text
采集 observation
→ policy 以 10 个 flow steps 生成 50-action chunk
→ 依次执行完整 chunk
→ 再采集下一 observation
```

这种实现简单、平均计算负担低，但 chunk 内 open-loop，且 queue 耗尽后再推理会产生停顿。

### 15.4 异步真机循环

论文 Algorithm 1 可映射为两个并行角色：

```text
RobotClient / control process
  按固定 control period 消费本地 action queue
  当 remaining / 50 < g 时采集 observation
  经 joint-space similarity filter 后提交异步请求
  新 chunk 返回后调用 aggregation f 更新 queue

PolicyServer / inference worker
  接收 observation
  执行 VLM + 10-step flow inference
  返回 action chunk
```

工程上必须维持的论文语义：

- 推理请求非阻塞，旧 queue 在 server 计算时继续执行；
- 同一时间的 in-flight request 状态需被追踪；
- 新 chunk 未完成时不能清空旧 queue；
- queue 空时强制处理最新 observation，即使 similarity filter 判为近重复；
- `g` 需结合推理延迟、chunk size 和 control period 调整；
- 新旧 chunk 重叠时通过 `f` 聚合，但论文未定义 `f` 的具体实现，部署时必须查作者代码／所用 LeRobot 版本。

### 15.5 部署边界

论文支持 PolicyServer 远程运行以使用更强 GPU，但其延迟近似假设是网络通信相对 server inference 可忽略。真实系统若网络抖动明显，不能直接套用 `E[ℓ]≈E[ℓ_S]`；应测量完整 round-trip latency 后再选择 `g`。这是对论文公式条件的工程化使用，不是论文报告过的额外实验结论。

---

## 16. 关键配置速查

原文位置：PDF 第 3–4 页 Section 3.1；PDF 第 5 页 Table 1；PDF 第 10 页 Section 4.3；PDF 第 11–12 页 Section 4.6。

```text
Backbone                 SmolVLM-2 = SigLIP + SmolLM2
VLM training             frozen
Trainable module         Action Expert only
LLM layers used          first 16（N=L/2 for main setup）
Visual tokens            64 per frame
Input image              512×512
Model size               450M total / ~100M Action Expert
Expert hidden width      0.75 × VLM hidden dimension
Pretraining data         481 datasets / 22.9K episodes / 10.6M frames
Pretraining              200,000 steps / global batch 256
Action chunk             n=50 actions
Inference flow           10 steps
Real-world main eval     synchronous
Async evaluation         separate Section 4.6 / Figure 5
```

一句话概括：**SmolVLA 冻结并截短小型 SmolVLM-2，以每帧 64 个视觉 token 和单个状态 token 形成条件特征，再由约 100M 参数、交替 CA 与 causal SA 的 Action Expert 通过 flow matching 生成 50 步连续动作；其社区数据和异步 queue 执行栈分别解决低成本预训练与运行时停顿问题，但论文公式及个别表述存在必须保留的符号／数值一致性疑点。**
