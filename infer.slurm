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

# 关键：关闭/覆盖 HF 镜像
unset HF_ENDPOINT
unset HF_HUB_ENABLE_HF_MIRROR
export HF_ENDPOINT="https://huggingface.co"

# 如果模型是 gated 的，建议提前在登录节点上 huggingface-cli login 一次
# 或者这里显式带 token（不推荐写死在脚本里）：
# export HF_TOKEN="你的token"

python inference_low_gpu.py \
    --base_model_path stabilityai/stable-diffusion-2-1-base \
    --segment_model_name ZhengPeng7/BiRefNet \
    --data_dir examples \
    --output_dir outputs
