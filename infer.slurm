#!/bin/bash
#SBATCH -J diffusion_infer               # 作业名
#SBATCH --cpus-per-task=32               # 分配 CPU 数
#SBATCH -p DGX,HGX                       # 使用的分区
#SBATCH -t 12:00:00                      # 运行时间上限
#SBATCH --qos=lv1                        # QOS
#SBATCH -o infer_diffusion.log           # 日志输出文件
#SBATCH --gres=gpu:1                     # 使用 1 块 GPU
#SBATCH -N 1                             # 1 个节点
#SBATCH --account=research               # 账⼾/项目名

python inference_low_gpu.py \
    --base_model_path stabilityai/stable-diffusion-2-1-base \
    --segment_model_name ZhengPeng7/BiRefNet \
    --data_dir examples \
    --output_dir outputs
