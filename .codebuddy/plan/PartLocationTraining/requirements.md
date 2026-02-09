# 需求文档：Part Location 训练与推理系统

## 引言

本文档定义了 Part Location 项目中训练和推理脚本的详细需求。该系统的核心目标是：给定一张整体图（whole image）和一张部件图（part image），预测部件在整体3D模型中的平移位置（translation）和缩放（scale）。

> **当前阶段简化说明：**
> - **忽略旋转**：当前数据集中旋转值均为单位四元数 [1,0,0,0]（无旋转），暂不学习旋转参数
> - **缩放为标量**：数据集中 scale 的 x/y/z 三个分量始终相等（uniform 等比缩放），因此 scale 作为 1 维标量学习
> - 模型最终输出 **4 个参数**：translation [3] + scale [1]

> **显存约束：**
> - 目标硬件：16GB 显存 GPU（如 RTX 4080 / RTX A4000）
> - 必须在 16GB 显存内完成训练，通过多种显存优化策略实现

### 数据现状

数据已在 `data/images/` 目录中准备就绪，组织方式如下：
- 每个3D模型对应一个子目录，如 `data/images/{model_name}/`
- 整体图命名为 `{model_name}-whole.png`
- 部件图命名为 `{model_name}-{part_name}.png`
- 每个部件有对应的 JSON 文件 `{model_name}-{part_name}.json`，包含 transform 标签：
  ```json
  {
    "model_name": "xxx",
    "part_name": "yyy",
    "translation": [tx, ty, tz],
    "rotation": [1.0, 0.0, 0.0, 0.0],
    "scale": [s, s, s]
  }
  ```
- 所有3D模型均已归一化到 [-0.5, 0.5] 范围
- rotation 固定为 [1,0,0,0]，当前阶段忽略
- scale 的三个分量始终相等，取其中任一分量作为标量标签

### 模型输出含义

- **translation** [3]：归一化部件模型经过缩放后，其中心需要移动到整体模型坐标系中的位置，值域在 [-0.5, 0.5] 范围内
- **scale** [1]：归一化部件需要缩放到的大小比例（标量，uniform 等比缩放）

### 技术选型

- 图像特征提取器：DINOv2 预训练视觉基础模型（ViT-B/14 为默认）
- 图像输入分辨率：1024×1024（DINOv2 支持任意分辨率输入，patch_size=14）
- 深度学习框架：PyTorch（≥ 2.0，使用内置 memory-efficient attention）
- 训练日志：TensorBoard

### 显存优化策略总览

在 1024×1024 分辨率下，DINOv2 ViT-B/14 产生 73×73 = 5329 个 patch tokens。未优化时，两张图的编码 + Cross-Attention 在 FP16 下就需要 ~10GB+。为适配 16GB 显存，采用以下策略：

| 策略 | 节省效果 | 说明 |
|------|---------|------|
| **AMP (FP16)** | ~50% 参数/激活内存 | 混合精度训练 |
| **Gradient Checkpointing** | ~60% 编码器激活内存 | 对 DINOv2 Transformer 层开启，用重计算换显存 |
| **Patch Token 空间降采样** | ~80% Cross-Attention 显存 | 在 Cross-Attention 前将 5329 tokens 降采样到 ~1024 tokens（32×32），注意力矩阵从 5329² 降到 1024² |
| **Memory-Efficient Attention** | ~40% 注意力显存 | 使用 PyTorch 2.0 `F.scaled_dot_product_attention`（Flash Attention / Memory-Efficient 内核） |
| **梯度累积** | 允许小 batch 模拟大 batch | batch_size=1 或 2，通过梯度累积等效更大 batch |

预计优化后显存占用（batch_size=1, FP16）：
```
DINOv2 参数:                  ~170 MB
DINOv2 激活值 (带 checkpoint): ~1.5-2 GB × 2 张图 = ~3-4 GB
Cross-Attention (降采样后):    ~0.3-0.5 GB
优化器状态 (AdamW FP32):       ~500 MB
梯度 (FP16):                  ~170 MB
───────────────────────────────
总计约: ~5-6 GB (batch_size=1)
```

这为 batch_size=2~4 留出了充足空间。

### 项目现有结构

```
part_location/
├── data/           # 数据加载模块（待实现）
├── models/         # 模型架构（待实现）
├── training/       # 训练逻辑（待实现）
├── inference/      # 推理逻辑（待实现）
├── evaluation/     # 评估指标（待实现）
└── utils/          # 工具函数
scripts/            # 运行脚本
configs/            # 配置文件
```

---

## 需求

### 需求 1：数据加载与预处理

**用户故事：** 作为一名深度学习工程师，我希望有一个高效的数据加载模块，能够从现有的 `data/images/` 目录中加载整体图和部件图，以及对应的 transform 标签，以便为模型训练提供数据。

#### 验收标准

1. WHEN 加载数据集 THEN 系统 SHALL 自动扫描 `data/images/` 下所有模型子目录，识别整体图（`*-whole.png`）和部件图及其对应 JSON 标注
2. WHEN 构建训练样本 THEN 系统 SHALL 将每一个 (整体图, 部件图, JSON标注) 三元组作为一个训练样本
3. WHEN 解析标签 THEN 系统 SHALL 从 JSON 中提取 translation [3] 和 scale [1]（取 scale 数组的第一个元素）作为回归目标，忽略 rotation 字段
4. WHEN 预处理图像 THEN 系统 SHALL 对图像进行 resize 到 1024×1024、归一化（ImageNet 均值和标准差）
5. WHEN 划分数据集 THEN 系统 SHALL 支持按模型级别（而非样本级别）划分 train/val/test，避免同一模型的不同部件分属不同集合造成数据泄露
6. IF 部件图没有对应的 JSON 文件 THEN 系统 SHALL 跳过该样本并记录警告
7. IF 整体图缺失 THEN 系统 SHALL 跳过该模型目录下的所有样本并记录警告
8. WHEN 加载数据 THEN 系统 SHALL 支持训练时数据增强（颜色抖动等不影响空间语义的增强），验证/测试时仅做标准预处理

---

### 需求 2：模型架构

**用户故事：** 作为一名深度学习工程师，我希望有一个能够从整体图和部件图中提取特征并预测3D变换参数的网络架构，以便准确学习部件在整体中的空间位置关系。

#### 验收标准

1. WHEN 设计编码器 THEN 模型 SHALL 使用 DINOv2 ViT-B/14 作为共享权重的图像编码器，分别编码整体图和部件图，输入分辨率为 1024×1024
2. WHEN 编码图像 THEN 模型 SHALL 对 DINOv2 开启 Gradient Checkpointing，将 Transformer 层的激活值改为重计算方式，以大幅降低编码器部分的显存占用
3. WHEN 提取特征 THEN 模型 SHALL 提取 DINOv2 的 CLS token 和 patch tokens 作为图像特征表示
4. WHEN 准备 Cross-Attention 输入 THEN 模型 SHALL 对 patch tokens 进行空间降采样（Spatial Downsampling），将 73×73 = 5329 个 tokens 降采样到约 32×32 = 1024 个 tokens，降采样方式采用 2D Adaptive Average Pooling（将 patch tokens reshape 为 73×73 空间网格后 pooling 到 32×32，再 flatten 回序列）
5. WHEN 融合特征 THEN 模型 SHALL 使用交叉注意力（Cross-Attention）模块，以降采样后的部件 patch tokens 为 Query、降采样后的整体 patch tokens 为 Key/Value，建立语义对应关系。Cross-Attention 中的注意力计算 SHALL 使用 PyTorch 2.0 内置的 `F.scaled_dot_product_attention`（自动选择 Flash Attention 或 Memory-Efficient Attention 内核）以进一步节省显存
6. WHEN 预测变换参数 THEN 模型 SHALL 通过 MLP 预测头输出 4 个参数：
   - translation [3]: 平移向量
   - scale [1]: 缩放标量（uniform 等比缩放）
7. WHEN 配置模型 THEN 系统 SHALL 支持通过 YAML 配置文件指定 DINOv2 模型规模（ViT-S/14, ViT-B/14, ViT-L/14）、交叉注意力层数、MLP 隐藏层维度、降采样目标尺寸等超参数

---

### 需求 3：损失函数

**用户故事：** 作为一名机器学习研究员，我希望有针对3D变换预测任务设计的损失函数，以便有效优化模型的预测精度。

#### 验收标准

1. WHEN 计算平移损失 THEN 系统 SHALL 使用 L1 损失（MAE）衡量预测平移与真实平移的误差
2. WHEN 计算缩放损失 THEN 系统 SHALL 使用 L1 损失衡量缩放误差
3. WHEN 组合总损失 THEN 系统 SHALL 支持可配置的损失权重：`L_total = w_trans * L_trans + w_scale * L_scale`
4. WHEN 默认配置 THEN 系统 SHALL 设置默认权重使两项损失在训练初期量级大致平衡

---

### 需求 4：训练流程

**用户故事：** 作为一名深度学习工程师，我希望有完整的训练流程脚本，以便从配置文件启动训练、监控训练过程并管理模型检查点。

#### 验收标准

1. WHEN 启动训练 THEN 系统 SHALL 通过命令行指定 YAML 配置文件来配置所有训练超参数
2. WHEN 训练模型 THEN 系统 SHALL 使用 AdamW 优化器，支持余弦退火学习率调度
3. WHEN 训练模型 THEN 系统 SHALL 使用混合精度训练（AMP, FP16/BF16）以加速训练和节省显存
4. WHEN 训练模型 THEN 系统 SHALL 对 DINOv2 编码器开启 Gradient Checkpointing 以节省显存
5. WHEN 训练模型 THEN 系统 SHALL 支持梯度累积（Gradient Accumulation），通过配置 `accumulation_steps` 参数，在小 batch_size 下模拟大 batch 训练效果
6. WHEN 训练模型 THEN 系统 SHALL 支持梯度裁剪防止梯度爆炸
7. WHEN 监控训练 THEN 系统 SHALL 集成 TensorBoard，记录每个 epoch 的 train loss、val loss 以及各分项损失（translation loss、scale loss），并记录当前学习率和显存占用峰值
8. WHEN 保存检查点 THEN 系统 SHALL 在每个 epoch 结束时保存模型检查点（包含模型权重、优化器状态、epoch 号、最佳指标）
9. WHEN 保存检查点 THEN 系统 SHALL 自动保存验证集上表现最好的模型为 `best_model.pth`
10. IF 训练中断 THEN 系统 SHALL 支持通过 `--resume` 参数从最近的检查点恢复训练
11. WHEN 验证模型 THEN 系统 SHALL 在每个 epoch 结束时在验证集上评估，并输出各项指标

---

### 需求 5：推理脚本

**用户故事：** 作为一名应用开发者，我希望有简洁的推理脚本，能够加载训练好的模型，输入整体图和部件图，输出预测的变换参数。

#### 验收标准

1. WHEN 进行推理 THEN 系统 SHALL 提供命令行推理脚本，接受整体图路径、部件图路径和模型检查点路径作为输入
2. WHEN 输出结果 THEN 系统 SHALL 以 JSON 格式输出预测的 translation [3] 和 scale [1]
3. WHEN 批量推理 THEN 系统 SHALL 支持指定一个数据目录，自动遍历所有模型进行推理并输出结果文件
4. WHEN 加载模型 THEN 系统 SHALL 自动从检查点恢复模型架构和权重
5. IF 输入图像尺寸与训练时不同 THEN 系统 SHALL 自动 resize 到 1024×1024

---

### 需求 6：评估指标

**用户故事：** 作为一名研究员，我希望有标准化的评估指标来衡量模型的预测质量，以便对比不同模型和配置的效果。

#### 验收标准

1. WHEN 评估模型 THEN 系统 SHALL 计算以下指标：
   - 平均平移误差（Mean Translation Error）：L2 距离，单位为模型坐标
   - 平均缩放误差（Mean Scale Error）：绝对误差和相对百分比误差
2. WHEN 评估完成 THEN 系统 SHALL 输出整体指标和每个样本的详细指标
3. WHEN 评估 THEN 系统 SHALL 支持在测试集上运行完整评估并输出报告

---

### 需求 7：配置管理

**用户故事：** 作为一名开发者，我希望所有超参数和配置都通过 YAML 文件管理，以便灵活调整实验设置。

#### 验收标准

1. WHEN 配置训练 THEN 系统 SHALL 使用 YAML 配置文件包含以下内容：
   - 数据路径：data_dir, output_dir
   - 数据集划分比例：train_ratio, val_ratio, test_ratio
   - 模型配置：backbone_name, cross_attention_layers, mlp_hidden_dim, **downsample_size**（降采样目标尺寸，默认 32）
   - 训练配置：batch_size, num_epochs, learning_rate, weight_decay, grad_clip, **accumulation_steps**（梯度累积步数，默认 1）
   - 损失权重：w_trans, w_scale
   - 混合精度：use_amp
   - Gradient Checkpointing：**use_gradient_checkpointing**（默认 True）
   - 图像尺寸：image_size（默认 1024）
2. WHEN 解析配置 THEN 系统 SHALL 为所有配置项提供合理的默认值
3. WHEN 配置不完整 THEN 系统 SHALL 使用默认值填充缺失项

---

## 边界情况与技术限制

### 边界情况
1. **数据不均衡**：不同模型的部件数量可能差异较大（少则1个，多则几十个），需要考虑采样策略
2. **缩放接近零**：极小的部件缩放值可能导致数值不稳定
3. **数据量有限**：当前 `data/images/` 中模型目录数量较少，可能需要考虑小数据集下的训练策略（如更强的正则化、数据增强）

### 技术限制
1. **2D 到 3D 的信息损失**：从单视图 2D 图像推断 3D 位置存在固有的信息不足
2. **显存限制**：目标在 16GB 显存 GPU 上训练。1024×1024 输入下 DINOv2 ViT-B/14 产生 5329 个 patch tokens，通过 Gradient Checkpointing + 空间降采样 + Memory-Efficient Attention + AMP 组合优化，预计 batch_size=2~4 可在 16GB 内运行
3. **Gradient Checkpointing 的计算代价**：开启后训练速度约降低 20-30%，这是用时间换显存的权衡
4. **收敛速度**：小数据集上可能收敛较慢或过拟合

### 成功标准
1. 平移预测误差 < 0.05（模型坐标单位，整体归一化到 [-0.5, 0.5]）
2. 缩放预测误差 < 15%（相对误差）
3. 训练脚本能够在 16GB 显存 GPU 上以 batch_size ≥ 1 正常运行完整训练流程
4. 推理脚本能够加载模型并输出合理的预测结果
