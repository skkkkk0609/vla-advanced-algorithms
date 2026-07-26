# PyTorch 学习目标与进度

最后更新：2026-07-26

## 1. 学习目标

面向 VLA 实习工作，目标不是从零手写完整项目，而是能够：

1. 阅读并逐行解释常见 PyTorch 与 LeRobot 代码。
2. 识别模块在数据、模型、训练、评估或推理链路中的职责。
3. 跟踪关键 Tensor 的 `shape`、`dtype` 和 `device`。
4. 理解模型所需的配置、数据输入、预处理、编码器、核心网络、输出头、损失、优化器、checkpoint 与推理模块。
5. 审查 AI 生成的代码改动和命令，说明修改前后行为、问题根因和验证方法。
6. 完成 PyTorch Basics 后，按真实调用链拆解 LeRobot ACT，再拆解 SmolVLA。

不以以下事项作为当前验收要求：

- 从空白文件手写完整训练项目。
- 记忆全部 PyTorch API。
- 从零实现 Transformer、ACT 或 SmolVLA。
- 完整推导反向传播、Attention 或 Flow Matching 的全部数学公式。

## 2. 学习方法

阅读每段代码时固定回答：

1. 它属于数据、模型、训练、评估还是推理？
2. 主要变量是什么 Python 类型？
3. Tensor 的 `shape`、`dtype`、`device` 是什么？
4. 调用了哪个类或函数？
5. 输入经过它后发生了什么变化？
6. 输出继续传给哪里？

允许并鼓励使用 AI，但必须逐步从“执行 AI 给出的方案”转向：

1. 先提出自己的判断或假设。
2. 使用 AI 检查判断并辅助定位源码。
3. 执行前理解命令和代码修改的作用。
4. 修改后阅读 diff，并用日志、shape、指标或测试验证。
5. 能复述问题根因，而不只记录最终命令。

## 3. PyTorch Basics 路线

- [x] Quickstart：数据、模型、训练、评估、保存与加载的完整流程
- [ ] Tensors：创建、属性、索引、切片、拼接、运算、NumPy 互操作
- [ ] Datasets & DataLoaders：样本、batch、shuffle、迭代与自定义数据集
- [ ] Transforms：输入与标签预处理
- [ ] Build Model：`nn.Module`、层、参数、forward 与设备
- [ ] Autograd：计算图、梯度、`requires_grad`、`backward`
- [ ] Optimization：训练循环、loss、optimizer 与超参数
- [ ] Save & Load：`state_dict`、checkpoint 与推理恢复
- [ ] Basics 总复盘：能沿 `data -> forward -> loss -> backward -> optimizer` 解释代码

## 4. 当前掌握情况

### 已理解

- `[B,C,H,W]` 图像 shape 和 `[B,T,A]` 动作 chunk shape。
- batch、step、epoch 的区别。
- `Dataset` 保存样本，`DataLoader` 组织 batch 的基本职责。
- `Flatten`、`Linear`、ReLU 的功能及 shape 变化。
- logits、Softmax、类别标签与 CrossEntropyLoss 的关系。
- `forward`、loss、`backward()`、`optimizer.step()`、梯度清零的训练闭环。
- `model.train()`、`model.eval()`、`torch.no_grad()` 的基本区别。
- `state_dict` 保存训练参数，以及先构造模型再加载参数的原因。
- Tensor 的 `shape`、`dtype`、`device`。
- 整数索引会移除对应维度，范围切片会保留对应维度。

### 需要巩固

- Python 类、对象、属性、方法、`self` 与模块导入。
- Tensor 索引和切片的维度变化。
- `reshape`、`view`、`unsqueeze`、`squeeze`、`transpose`、`permute`。
- `cat`、`stack`、广播、矩阵乘法和逐元素运算。
- Dataset/DataLoader 的真实 batch 组织。
- Autograd 和计算图。

## 5. 当前学习位置

当前文件：`beginner_source/basics/tensorqs_tutorial.py`

当前主题：Tensor 索引与切片。

已确认示例：

- `actions[0]` 从 `[32,50,7]` 得到 `[50,7]`。
- `actions[:,0]` 从 `[32,50,7]` 得到 `[32,7]`。
- `actions[0:1]` 从 `[32,50,7]` 得到 `[1,50,7]`。

下一主题：`torch.cat` 与 `torch.stack`，然后区分矩阵乘法 `@` 和逐元素乘法 `*`。

## 6. 进入 LeRobot 源码的门槛

完成 PyTorch Basics 后，不要求从零手写项目，但应能够：

1. 从调用位置跳转到函数或类定义。
2. 说明关键配置项如何影响模型与数据。
3. 查看真实 batch 的 key、shape、dtype、device。
4. 沿 `forward` 跟踪主要 Tensor 变化。
5. 找到 loss、`optimizer.step()`、checkpoint 和 `select_action()`。
6. 通过打印、断点、日志或 diff 验证自己的理解。

## 7. LeRobot 实战路线

### ACT

- [ ] 找到实际训练命令并解释参数
- [ ] 找到配置类和数据集入口
- [ ] 查看真实 batch
- [ ] 跟踪 ACT forward 与动作输出 shape
- [ ] 找到重建损失、KL loss 和总 loss
- [ ] 找到训练循环与 checkpoint
- [ ] 跟踪 `select_action()` 推理调用链
- [ ] 用自己的语言复述完整流程

### SmolVLA

- [ ] 找到实际训练命令并解释参数
- [ ] 找到配置类、processor 和数据集入口
- [ ] 查看图像、语言、state、action 的真实 batch
- [ ] 跟踪 Vision Encoder、VLM、Action Expert 与输出
- [ ] 理解 Flow Matching 训练目标与推理积分
- [ ] 找到训练循环、checkpoint 和 `select_action()`
- [ ] 对比 ACT 与 SmolVLA 的动作生成方式
- [ ] 用自己的语言复述完整流程

## 8. 进度更新规则

每次完成一个学习节点后更新：

- 最后更新日期。
- 已完成的路线复选框。
- 当前学习位置和下一主题。
- 新掌握的概念。
- 仍不清楚或需要复习的概念。
- 进入 ACT/SmolVLA 后记录已追踪的真实调用链。
