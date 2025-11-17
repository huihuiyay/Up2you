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


python inference_low_gpu.py \
  --base_model_path /share/mas/huangping/yuzihui/models/stable-diffusion-2-1-base \
  --segment_model_name /share/mas/huangping/yuzihui/models/BiRefNet \
  --data_dir examples \
  --output_dir outputs

