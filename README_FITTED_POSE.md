# UP2You 拟合姿态推理

本文档介绍如何使用 SAM-3D-Body 获取的拟合姿态进行 UP2You 推理。

## 概述

标准的 UP2You 推理流程使用 A-Pose（标准姿态）生成多视角图像。这个新的推理脚本支持使用 SAM-3D-Body 检测到的实际人体姿态（拟合姿态）进行推理，生成更贴合输入图像中人物实际姿势的 3D 模型。

## 流程对比

### 标准流程 (inference_thg.py)
```
输入图像 → 形状预测 → A-Pose 生成 → 多视角渲染 → RGB/法线生成 → 3D重建
```

### 拟合姿态流程 (inference_thg_fitted.py)
```
输入图像 → SAM-3D-Body 姿态估计 → 拟合姿态渲染 → RGB/法线生成 → 3D重建
```

## 安装依赖

1. 安装 SAM-3D-Body:
```bash
cd thirdparties/sam-3d-body
pip install -e .
```

2. 下载 SAM-3D-Body 模型权重:
```bash
# 设置环境变量
export SAM3D_MHR_PATH=/path/to/mhr/assets
```

## 使用方法

### 基本用法

```bash
python inference_thg_fitted.py \\
    --data_dir ./input_images \\
    --output_dir ./output_fitted \\
    --sam3d_checkpoint /path/to/sam3d/checkpoint.ckpt \\
    --mhr_path /path/to/mhr/assets
```

### 完整参数示例

```bash
python inference_thg_fitted.py \\
    --data_dir ./input_images \\
    --output_dir ./output_fitted \\
    --sam3d_checkpoint checkpoints/sam3d_body.ckpt \\
    --mhr_path human_models/mhr_assets \\
    --base_model_path stabilityai/stable-diffusion-2-1-base \\
    --feature_aggregator_path checkpoints/feature_aggregator.pth \\
    --rgb_adapter_path checkpoints/rgb_adapter \\
    --normal_adapter_path checkpoints/normal_adapter \\
    --thg_interval 3 \\
    --num_inference_steps 50 \\
    --guidance_scale 3.0 \\
    --seed 42 \\
    --gpu_id 0 \\
    --use_fp16
```

## 主要参数说明

### SAM-3D-Body 相关
- `--sam3d_checkpoint`: SAM-3D-Body 模型 checkpoint 路径 (必需)
- `--mhr_path`: MHR 模型路径，也可通过 `SAM3D_MHR_PATH` 环境变量设置

### UP2You 模型路径
- `--base_model_path`: Stable Diffusion 2.1 基础模型
- `--feature_aggregator_path`: 特征聚合器权重
- `--rgb_adapter_path`: RGB 生成适配器
- `--normal_adapter_path`: 法线生成适配器

### 推理参数
- `--thg_interval`: THG 优化间隔 (推荐 2-4)
- `--num_inference_steps`: 扩散步数 (默认 50)
- `--guidance_scale`: CFG 引导强度 (默认 3.0)
- `--use_fp16`: 使用 FP16 半精度推理以节省显存

## 输出文件

推理完成后，输出目录包含：

```
output_fitted/
├── ref_imgs/              # 预处理后的参考图像
├── fitted_mesh/           # SAM-3D-Body 拟合的姿态 mesh
│   └── fitted_pose.obj
├── rgb/                   # 生成的 RGB 图像 (20视角)
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── normal/                # 生成的法线图 (20视角)
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── pred_meshes/           # 重建的 3D 网格
│   └── final_mesh.obj
├── result_rgb.mp4         # RGB 旋转视频
└── result_normal.mp4      # 法线旋转视频
```

## 核心组件

### Sam3DToSMPLXAdapter

位于 `up2you/utils/smpl_utils/sam3d_to_smplx.py`

功能：
- 将 SAM-3D-Body 的 MHR 模型输出转换为渲染兼容格式
- 标准化顶点坐标
- 提供多视角渲染功能
- 支持法线和语义图生成

主要方法：
```python
adapter = Sam3DToSMPLXAdapter(device="cuda")

# 转换 sam3d 输出
converted = adapter.convert_sam3d_output(sam3d_output)

# 渲染多视角
normals, semantics = adapter.render_multiview(
    vertices=converted['vertices'],
    faces=faces,
    num_views=20,
    height=768,
    width=768
)
```

## 与标准推理的区别

| 特性 | 标准推理 | 拟合姿态推理 |
|------|----------|--------------|
| 姿态来源 | A-Pose (固定) | SAM-3D-Body 检测 |
| 姿态真实性 | 标准姿态 | 与输入图像一致 |
| 依赖模块 | ShapePredictor | SAM-3D-Body |
| 适用场景 | 通用 3D 重建 | 保持原始姿态的重建 |

## 注意事项

1. **显存需求**: SAM-3D-Body 需要额外的显存，建议至少 16GB VRAM
2. **姿态检测**: 需要输入图像中人体姿态清晰可见
3. **环境变量**: 确保正确设置 `SAM3D_MHR_PATH`
4. **模型下载**: 首次运行需要下载 SAM-3D-Body 相关模型

## 故障排除

### 1. SAM-3D-Body 未能检测到人体
```
RuntimeError: SAM-3D-Body 未能检测到人体，请检查输入图像！
```
**解决方法**:
- 确保输入图像中人体清晰可见
- 尝试使用更高质量的输入图像
- 检查图像中人体是否被遮挡

### 2. MHR 模型加载失败
```
FileNotFoundError: MHR model not found
```
**解决方法**:
- 检查 `--mhr_path` 参数是否正确
- 确认 `SAM3D_MHR_PATH` 环境变量设置
- 下载 MHR 模型到指定路径

### 3. 显存不足
```
RuntimeError: CUDA out of memory
```
**解决方法**:
- 使用 `--use_fp16` 启用半精度推理
- 减少 `--num_inference_steps`
- 使用更小的输入图像

## 性能优化

1. **使用 THG 优化**: 设置 `--thg_interval 3` 可加速约 1.4 倍
2. **半精度推理**: 添加 `--use_fp16` 可节省约 50% 显存
3. **批处理**: 如需处理多张图像，可编写脚本批量调用

## 示例脚本

创建 `run_fitted_inference.sh`:

```bash
#!/bin/bash

# 设置环境变量
export SAM3D_MHR_PATH=/path/to/mhr/assets

# 运行推理
python inference_thg_fitted.py \\
    --data_dir ./examples/person1 \\
    --output_dir ./output/person1_fitted \\
    --sam3d_checkpoint checkpoints/sam3d_body.ckpt \\
    --thg_interval 3 \\
    --use_fp16 \\
    --seed 42

echo "拟合姿态推理完成！"
```

## 后续改进方向

- [ ] 支持批量处理多个人体
- [ ] 添加姿态平滑选项
- [ ] 支持自定义视角数量
- [ ] 集成姿态编辑功能
- [ ] 优化显存使用

## 参考文献

- SAM-3D-Body: https://github.com/facebookresearch/sam-3d-body
- UP2You: 原项目文档
- MHR: Monocular Human Reconstruction
