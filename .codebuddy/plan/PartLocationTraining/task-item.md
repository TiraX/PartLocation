# 实施计划

- [ ] 1. 配置管理模块与默认训练配置文件
   - 在 `configs/` 目录下创建 `train_config.yaml`，包含所有训练超参数的默认值（data_dir, output_dir, backbone_name, image_size=1024, downsample_size=32, batch_size, num_epochs, learning_rate, weight_decay, grad_clip, accumulation_steps=1, use_amp=true, use_gradient_checkpointing=true, w_trans, w_scale, train/val/test_ratio 等）
   - 在 `part_location/utils/` 中实现 `config.py`，提供 YAML 配置加载函数，支持默认值填充和配置验证
   - _需求：7.1、7.2、7.3_

- [ ] 2. 数据集与数据加载模块
   - 在 `part_location/data/` 中实现 `dataset.py`，编写 `PartLocationDataset(torch.utils.data.Dataset)` 类
   - 自动扫描 `data/images/` 子目录，匹配整体图（`*-whole.png`）、部件图和 JSON 标注，构建 (整体图, 部件图, 标签) 三元组
   - 从 JSON 中提取 translation[3] 和 scale[1]（取第一个元素），忽略 rotation
   - 处理边界情况：跳过缺失 JSON 或整体图的样本，记录警告
   - 支持按模型级别划分 train/val/test（同一模型的所有部件归入同一集合）
   - 实现图像预处理（resize 1024×1024、ImageNet 归一化）和训练时数据增强（颜色抖动等）
   - _需求：1.1、1.2、1.3、1.4、1.5、1.6、1.7、1.8_

- [ ] 3. DINOv2 编码器封装
   - 在 `part_location/models/` 中实现 `encoder.py`，封装 DINOv2 ViT 编码器
   - 通过 `torch.hub` 加载 DINOv2 预训练权重（支持 ViT-S/14、ViT-B/14、ViT-L/14 可配置）
   - 提供接口返回 CLS token 和 patch tokens
   - 实现 Gradient Checkpointing 开关，对 Transformer 层启用 `torch.utils.checkpoint`
   - _需求：2.1、2.2、2.3、2.7_

- [ ] 4. Cross-Attention 融合模块
   - 在 `part_location/models/` 中实现 `cross_attention.py`
   - 实现空间降采样：将 patch tokens reshape 为 2D 空间网格（73×73），通过 `AdaptiveAvgPool2d` 降采样到可配置的目标尺寸（默认 32×32），再 flatten 回序列
   - 实现 Cross-Attention 层，以降采样后的部件 tokens 为 Query、整体 tokens 为 Key/Value
   - 注意力计算使用 `F.scaled_dot_product_attention`（Memory-Efficient Attention）
   - 支持多层堆叠（层数可配置）
   - _需求：2.4、2.5、2.7_

- [ ] 5. 完整模型组装与预测头
   - 在 `part_location/models/` 中实现 `part_location_model.py`，组装完整的 `PartLocationModel(nn.Module)`
   - 组合：共享 DINOv2 编码器 → 空间降采样 → Cross-Attention 融合 → 全局池化 → MLP 预测头
   - MLP 预测头输出 4 个参数（translation[3] + scale[1]），隐藏层维度可配置
   - 提供统一的 `forward(whole_image, part_image)` 接口
   - 在 `part_location/models/__init__.py` 中导出模型构建工厂函数
   - _需求：2.1、2.4、2.5、2.6、2.7_

- [ ] 6. 损失函数模块
   - 在 `part_location/training/` 中实现 `losses.py`
   - 实现 `PartLocationLoss` 类：translation L1 损失 + scale L1 损失的加权组合
   - 权重 `w_trans`、`w_scale` 从配置读取，并提供合理默认值使两项损失量级平衡
   - 返回总损失和各分项损失（用于 TensorBoard 记录）
   - _需求：3.1、3.2、3.3、3.4_

- [ ] 7. 训练流程脚本
   - 在 `part_location/training/` 中实现 `trainer.py`，编写核心训练循环
   - 在 `scripts/` 中实现 `train.py` 入口脚本（命令行参数: `--config`, `--resume`）
   - 实现：AdamW 优化器 + 余弦退火调度器 + AMP（GradScaler）+ Gradient Checkpointing 启用 + 梯度累积 + 梯度裁剪
   - 集成 TensorBoard：记录 train/val 的总损失、分项损失、学习率、显存峰值
   - 实现检查点管理：每 epoch 保存、自动维护 best_model.pth、支持 `--resume` 恢复
   - 每 epoch 结束后在验证集上评估
   - _需求：4.1、4.2、4.3、4.4、4.5、4.6、4.7、4.8、4.9、4.10、4.11_

- [ ] 8. 评估指标模块
   - 在 `part_location/evaluation/` 中实现 `metrics.py`
   - 实现平均平移误差（L2 距离）和平均缩放误差（绝对误差 + 相对百分比误差）
   - 支持输出整体汇总指标和每个样本的详细指标
   - _需求：6.1、6.2_

- [ ] 9. 推理脚本
   - 在 `part_location/inference/` 中实现 `predictor.py`，封装模型加载和推理逻辑
   - 在 `scripts/` 中实现 `predict.py` 入口脚本
   - 支持单样本推理（`--whole_image`, `--part_image`, `--checkpoint`）和批量推理（`--data_dir`）
   - 输出 JSON 格式结果（translation[3] + scale[1]），批量模式输出结果文件
   - 自动 resize 非 1024×1024 的输入图像
   - _需求：5.1、5.2、5.3、5.4、5.5_

- [ ] 10. 评估脚本与依赖管理
   - 在 `scripts/` 中实现 `evaluate.py` 入口脚本，在测试集上运行完整评估并输出报告
   - 创建 `requirements.txt`，列出所有 Python 依赖（torch, torchvision, pyyaml, tensorboard, Pillow 等）及版本
   - _需求：6.3_
