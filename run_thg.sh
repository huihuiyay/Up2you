#!/bin/bash

# UP2You THG优化推理脚本
# 相比标准推理，预期加速1.4倍，显存降低22%

export CUDA_VISIBLE_DEVICES=0

# 显存碎片优化（PyTorch 2.0+）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python inference_thg.py \
    --base_model_path stabilityai/stable-diffusion-2-1-base \
    --shape_predictor_path pretrained_models/shape_predictor.pt \
    --rgb_adapter_path pretrained_models/rgb_adapter.safetensors \
    --feature_aggregator_path pretrained_models/feature_aggregator.pt \
    --normal_adapter_path pretrained_models/normal_adapter.safetensors \
    --segment_model_name ZhengPeng7/BiRefNet \
    --data_dir examples \
    --output_dir outputs_thg \
    --thg_interval 3 \
    --num_inference_steps 50 \
    --guidance_scale 3.0 \
    --use_fp16

# 参数说明：
# --thg_interval 3      : THG更新间隔，推荐2-4（越大越快但质量略降）
# --use_fp16            : 使用FP16加速（进一步降低显存）
# --num_inference_steps : 扩散步数，默认50（THG在长序列上效果更好）
# PYTORCH_CUDA_ALLOC_CONF : 减少显存碎片
