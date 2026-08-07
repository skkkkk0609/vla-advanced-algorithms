# Pi0 代码流程复习

本文只保留代码调用关系。辅助函数放在对应主流程旁，不展开公式推导。

## 1. 模块层级

```text
PI0Policy                         LeRobot 对外接口
|
+-- 数据预处理
+-- 训练入口 forward()
+-- 动作块入口 predict_action_chunk()
+-- 单步入口 select_action()
|
`-- PI0Pytorch                    Flow Matching 核心
    |
    +-- PaliGemmaWithExpertModel
    |   |
    |   +-- PaliGemma             图像和语言
    |   `-- Gemma Action Expert   状态、带噪动作和时间
    |
    +-- state_proj                state: 32 -> 1024
    +-- action_in_proj            action: 32 -> 1024
    +-- action_time_mlp           融合动作和时间
    `-- action_out_proj           速度: 1024 -> 32
```

## 2. 训练流程

```text
LeRobot batch
  |
  |  images / language / state / expert actions
  v
PI0Policy.forward(batch)
  |
  +-- _preprocess_images()
  |     resize + padding
  |     [0,1] -> [-1,1]
  |
  +-- prepare_state()
  |     [B,state_dim] -> [B,32]
  |
  +-- prepare_action()
  |     [B,50,action_dim] -> [B,50,32]
  |
  +-- sample_noise()
  |     noise ~ N(0,I), shape [B,50,32]
  |
  `-- sample_time()
        t ~ Beta(1.5,1.0), shape [B]
  |
  v
PI0Pytorch.forward(...)
  |
  +-- 构造 Flow Matching 样本
  |     x_t = t * noise + (1-t) * action
  |     u_t = noise - action
  |
  +-- embed_prefix()
  |     images -> Vision Tower -> visual tokens
  |     language IDs -> language embeddings
  |     输出 prefix
  |
  +-- embed_suffix()
  |     state -> state token
  |     x_t -> action tokens
  |     t -> time embedding
  |     输出 suffix
  |
  +-- PaliGemmaWithExpertModel.forward()
  |     输入 [prefix, suffix]
  |     18 层联合 Transformer
  |     输出 suffix hidden states
  |
  +-- action_out_proj()
  |     [B,50,1024] -> v_t [B,50,32]
  |
  `-- MSE(u_t, v_t, reduction="none")
        输出 [B,50,32]
  |
  v
PI0Policy.forward()
  |
  +-- 裁掉补零动作维度
  `-- losses.mean() -> 标量 loss
  |
  v
外部 Trainer
  |
  +-- optimizer.zero_grad()
  +-- loss.backward()
  `-- optimizer.step()
```

## 3. 动作块推理流程

```text
当前环境观测 batch
  |
  v
PI0Policy.predict_action_chunk(batch)
  |
  +-- 预处理 images
  +-- 读取 language tokens
  `-- 补齐 state
  |
  v
PI0Pytorch.sample_actions(...)
  |
  +-- 生成初始噪声
  |     x_1 [B,50,32]
  |
  +-- embed_prefix()
  |     图像 + 语言
  |
  +-- Prefix Prefill（仅一次）
  |     PaliGemmaWithExpertModel.forward([prefix, None])
  |     输出 prefix KV Cache
  |
  +-- dt = -1 / num_inference_steps
  |
  `-- 去噪循环，默认 10 次
        |
        +-- 当前时间 t: 1.0 -> 0.9 -> ... -> 0.1
        |
        +-- denoise_step(state, x_t, t, prefix KV)
        |     |
        |     +-- embed_suffix(state, x_t, t)
        |     +-- 构造 suffix 到 prefix/suffix 的 Mask
        |     +-- clone prefix KV Cache
        |     +-- Action Expert forward([None, suffix])
        |     `-- action_out_proj() -> v_t
        |
        `-- x_t = x_t + dt * v_t
  |
  v
最终 x_0 [B,50,32]
  |
  v
PI0Policy 裁掉补零维度
  |
  v
动作块 [B,50,actual_action_dim]
```

## 4. 单步动作执行流程

```text
PI0Policy.select_action(batch)
  |
  +-- 动作队列为空？
  |     |
  |     `-- 是
  |           predict_action_chunk(batch)
  |           -> 得到 50 步动作
  |           -> 取前 n_action_steps
  |           -> 放入 deque
  |
  `-- _action_queue.popleft()
        -> 返回当前 1 步动作

后续调用：
  队列未空 -> 继续取下一步
  队列为空 -> 用新观测重新预测动作块
```

## 5. 联合 Transformer 内部流程

```text
prefix: 图像 + 语言，hidden width 2048
suffix: state + x_t + t，hidden width 1024
  |
  +-- 各自 LayerNorm
  +-- 各自 q_proj / k_proj / v_proj
  +-- 转为兼容的多头 Q/K/V
  +-- 沿 token 维拼接 prefix 和 suffix
  +-- RoPE 给 Q/K 加入序列位置信息
  +-- Attention Mask 限制 token 访问权限
  +-- Attention 让 token 之间交换信息
  +-- 拆回 prefix 和 suffix
  +-- 各自 o_proj
  +-- 残差连接
  +-- 各自 MLP，处理 token 内部特征
  `-- 第二次残差连接
```

## 6. 三种模型调用模式

```text
训练：
  forward([prefix, suffix])
  -> 联合计算
  -> 不使用 KV Cache

推理初始化：
  forward([prefix, None])
  -> 图像语言只计算一次
  -> 生成 prefix KV Cache

推理每轮去噪：
  forward([None, suffix])
  -> 复用 prefix KV Cache
  -> 只重算变化的 x_t 和 t
```

## 7. 关键概念分工

```text
Policy             输入输出适配和动作队列
PI0Pytorch         Flow Matching 训练与推理
PaliGemma          编码图像和语言条件
Action Expert      预测动作速度场
Attention Mask     决定 token 能看谁
RoPE               编码 token 顺序和相对距离
Attention          token 之间交换信息
MLP                处理单个 token 内部特征
KV Cache           复用不变的 prefix 计算
RTC                实时衔接新旧动作块
Trainer            backward 和 optimizer.step
```
