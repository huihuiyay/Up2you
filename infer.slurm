#!/bin/bash
#SBATCH -J diffusion_infer
#SBATCH --cpus-per-task=8
#SBATCH -p DGX,HGX
#SBATCH -t 12:00:00
#SBATCH --qos=lv1
#SBATCH -o infer_diffusion.log
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --account=research

# 原始低显存推理脚本
# python inference_low_gpu.py \
#   --base_model_path /share/mas/huangping/yuzihui/up2you/models/stable-diffusion-2-1-base \
#   --segment_model_name /share/mas/huangping/yuzihui/up2you/models/BiRefNet \
#   --data_dir examples \
#   --output_dir outputs

python inference_thg.py \
    --base_model_path /share/mas/huangping/yuzihui/up2you/models/stable-diffusion-2-1-base \
    --shape_predictor_path pretrained_models/shape_predictor.pt \
    --rgb_adapter_path pretrained_models/rgb_adapter.safetensors \
    --feature_aggregator_path pretrained_models/feature_aggregator.pt \
    --normal_adapter_path pretrained_models/normal_adapter.safetensors \
    --segment_model_name /share/mas/huangping/yuzihui/up2you/models/BiRefNet \
    --data_dir examples \
    --output_dir outputs_thg \
    --thg_interval 3 \
    --num_inference_steps 50 \
    --guidance_scale 3.0 \
    --use_fp16

# 本地测试
# 龟兔算法模式
# python inference_thg.py \
#     --base_model_path stabilityai/stable-diffusion-2-1-base\
#     --shape_predictor_path pretrained_models/shape_predictor.pt \
#     --rgb_adapter_path pretrained_models/rgb_adapter.safetensors \
#     --feature_aggregator_path pretrained_models/feature_aggregator.pt \
#     --normal_adapter_path pretrained_models/normal_adapter.safetensors \
#     --segment_model_name ZhengPeng7/BiRefNet\
#     --data_dir examples \
#     --output_dir outputs_thg \
#     --thg_interval 3 \
#     --num_inference_steps 50 \
#     --guidance_scale 3.0 \
#     --use_fp16 

# 拟合姿态做几何引导模式
# python inference_thg_fitted.py \
#   --base_model_path stabilityai/stable-diffusion-2-1-base \
#   --rgb_adapter_path pretrained_models/rgb_adapter.safetensors \
#   --feature_aggregator_path pretrained_models/feature_aggregator.pt \
#   --normal_adapter_path pretrained_models/normal_adapter.safetensors \
#   --segment_model_name ZhengPeng7/BiRefNet \
#   --data_dir examples \
#   --output_dir outputs_thg \
#   --fitted_mesh_path outputs/sam3dbody/meshes/img_mesh_000.obj \
#   --thg_interval 3 \
#   --num_inference_steps 50 \
#   --guidance_scale 2.0 \
#   --use_fp16
