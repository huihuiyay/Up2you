"""
UP2You 推理脚本 - 集成Tortoise and Hare Guidance (THG) 优化
相比标准推理，THG可减少约30%的UNet前向计算，加速1.4倍

使用方法：
    python inference_thg.py --data_dir <input_dir> --output_dir <output_dir> --thg_interval 3
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision.utils import save_image
from up2you.utils.img_utils import load_image, process_image_rgba
from up2you.models.feature_aggregator import FeatureAggregator
from up2you.models.encoder.dinov2_wrapper import Dinov2Wrapper
from up2you.models.shape_predictor import ShapePredictor
from up2you.pipelines.pipeline_mvpuzzle_i2mv_sd21 import UP2YouI2MVSDPipeline
from up2you.pipelines.pipeline_mvpuzzle_mv2normal_sd21 import UP2YouMV2NormalPipeline
from up2you.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from up2you.schedulers.scheduling_thg import TortoiseHareGuidanceScheduler, enable_thg_for_pipeline
from up2you.utils.weight_map_utils import weight_map_to_heatmap
from up2you.utils.smpl_utils.apose_renderer import AposeRenderer
from up2you.utils.mesh_utils.reconstructor import Reconstructor
from diffusers import DDPMScheduler
from einops import rearrange
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import random
from up2you.utils.mesh_utils.mesh_util import save_obj_mesh
import shutil
from up2you.utils.mesh_utils.mesh_common_renderer import CommonRenderer
from up2you.utils.video_utils import tensor_to_video


def manual_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def has_alpha_any(im):
    if 'A' in im.getbands():
        return True
    if im.mode == 'P' and 'transparency' in im.info:
        return True
    return False


def segment_rgbs(rgbs, seg_model, device):
    out = []
    for image in rgbs:
        if has_alpha_any(image):
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
        else:
            x = transform_image(image).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = seg_model(x)[-1].sigmoid().cpu()[0].squeeze()
            mask = transforms.ToPILImage()(pred).resize(image.size)
            image.putalpha(mask)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
        out.append(image)
    return out


def preprocess_ref_imgs(
    data_dir,
    seg_model,
    device,
    output_ref_imgs_dir=None,
):
    ref_img_names = os.listdir(data_dir)
    ref_img_names.sort()
    ref_pils = []
    for ref_img_name in ref_img_names:
        ref_img_path = os.path.join(data_dir, ref_img_name)
        ref_pil = Image.open(ref_img_path)
        ref_pil = ImageOps.exif_transpose(ref_pil)
        ref_pils.append(ref_pil)
    ref_rgbas = segment_rgbs(ref_pils, seg_model, device)
    ref_imgs = [process_image_rgba(ref_rgba, ratio=1.85/2) for ref_rgba in ref_rgbas]
    if output_ref_imgs_dir is not None:
        for idx, ref_img in enumerate(ref_imgs):
            ref_img.save(os.path.join(output_ref_imgs_dir, f"{idx}.png"))

    ref_img_tensor_list = []
    ref_alpha_tensor_list = []
    for ref_img in ref_imgs:
        ref_img_tensor, ref_alpha_tensor = load_image(ref_img, 768, 768, return_alpha=True)
        ref_img_tensor_list.append(ref_img_tensor)
        ref_alpha_tensor_list.append(ref_alpha_tensor)
    ref_img_tensor = torch.stack(ref_img_tensor_list)
    ref_alpha_tensor = torch.stack(ref_alpha_tensor_list)

    return ref_img_tensor.permute(0, 3, 1, 2).to(device), ref_alpha_tensor.to(device)


def main(args):
    manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.use_fp16 else torch.float32

    print("=" * 80)
    print("UP2You 推理 - THG优化版本")
    print("=" * 80)
    print(f"输入目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"THG引导更新间隔: {args.thg_interval} (推荐2-4)")
    print(f"设备: {device}, 数据类型: {dtype}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载分割模型
    print("\n[1/8] 加载背景分割模型...")
    seg_model = AutoModelForImageSegmentation.from_pretrained(
        args.segment_model_name,  # 使用命令行参数
        trust_remote_code=True,
    ).to(device).eval()

    # 2. 预处理参考图像
    print("\n[2/8] 预处理参考图像...")
    output_ref_imgs_dir = os.path.join(args.output_dir, "ref_imgs")
    os.makedirs(output_ref_imgs_dir, exist_ok=True)
    ref_rgbs, ref_alphas = preprocess_ref_imgs(
        args.data_dir,
        seg_model,
        device,
        output_ref_imgs_dir=output_ref_imgs_dir,
    )
    # 注意：seg_model稍后还需要用于RGB和法线图的分割，暂不删除

    # 3. 特征提取和形状预测
    print("\n[3/8] 提取图像特征和预测身体形状...")
    image_encoder = Dinov2Wrapper(
        device=device,
        model_name="dinov2_vitl14",
        image_size=518,
    )
    image_encoder.to(device)

    shape_predictor = ShapePredictor(
        embed_dim=1024,
        depth=2,
        trunk_depth=2
    )
    shape_predictor.load_state_dict(
        torch.load(args.shape_predictor_path, map_location="cpu")
    )
    shape_predictor.to(device)
    shape_predictor.eval()

    with torch.no_grad():
        ref_img_feats = image_encoder(ref_rgbs)
        ref_img_feats = rearrange(
            ref_img_feats,
            "(B Nr) H W C -> B Nr H W C",
            B=1
        )
        betas = shape_predictor(ref_img_feats)

    del image_encoder
    del shape_predictor
    torch.cuda.empty_cache()

    # 4. 生成A-Pose
    print("\n[4/8] 生成SMPL A-Pose (20视角)...")
    apose_renderer = AposeRenderer(device=device)
    target_poses, _, smplx_v, smplx_f = apose_renderer(
        betas.reshape(1, 10),
        height=768,
        width=768,
        num_views=20,  # 20视角
        return_mesh=True
    )

    smplx_mesh_dir = os.path.join(args.output_dir, "smplx_mesh")
    os.makedirs(smplx_mesh_dir, exist_ok=True)
    smplx_mesh_path = os.path.join(smplx_mesh_dir, "smplx_apose.obj")
    save_obj_mesh(smplx_mesh_path, smplx_v, smplx_f)

    del apose_renderer
    torch.cuda.empty_cache()

    # 5. 生成权重图
    print("\n[5/8] 生成多视角权重图...")
    feature_aggregator = FeatureAggregator(
        pose_img_size=512,
        pose_img_in_chans=3,
        pose_patch_embed_type="pose_encoder",
        embed_dim=1024,
        patch_size=16,
        depth=1,
        aa_order=['self', 'self', 'cross', 'cross'],
        weight_norm="none",
        use_mask=True,
        smooth_method="avgpool",
        kernel_size=3,
    )

    feature_aggregator.load_state_dict(
        torch.load(args.feature_aggregator_path, map_location="cpu")
    )
    feature_aggregator.to(device)
    feature_aggregator.eval()

    target_pose_imgs = rearrange(
        target_poses,
        "(B Nv) C H W -> B Nv H W C",
        Nv=20
    )
    ref_alphas_rearranged = rearrange(ref_alphas, "(B Nr) H W -> B Nr H W", B=1)

    with torch.no_grad():
        weight_maps = feature_aggregator(
            target_pose_imgs=target_pose_imgs,
            ref_img_feats=ref_img_feats,
            ref_alphas=ref_alphas_rearranged,
        )

    del feature_aggregator
    torch.cuda.empty_cache()

    # 6. RGB生成 - 启用THG优化
    print(f"\n[6/8] 生成RGB图像 (20视角, THG间隔={args.thg_interval})...")
    rgb_pipe = UP2YouI2MVSDPipeline.from_pretrained(
        args.base_model_path,
    )

    # 使用THG包装的调度器
    base_scheduler = ShiftSNRScheduler.from_scheduler(
        rgb_pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    rgb_pipe.scheduler = TortoiseHareGuidanceScheduler.from_scheduler(
        base_scheduler,
        guidance_update_interval=args.thg_interval,
    )

    rgb_pipe.init_custom_adapter(
        num_views=20,
        mode='topk',
    )
    rgb_pipe.load_custom_adapter(
        args.rgb_adapter_path, weight_name='custom_adapter.safetensors'
    )

    rgb_pipe.to(device=device, dtype=dtype)
    rgb_pipe.cond_encoder.to(device=device, dtype=dtype)  # 修复：adapter也需要转换为fp16
    rgb_pipe.enable_vae_slicing()

    with torch.no_grad():
        images = rgb_pipe(
            prompt=["Multi-view Human, Full Body, High Quality, HDR"],
            reference_rgbs=ref_rgbs,
            control_image=target_poses,  # 注意：参数名是 control_image 不是 control_images
            weight_maps=weight_maps,
            num_images_per_prompt=20,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=768,
            width=768,
        ).images

    # 打印THG统计信息
    thg_stats = rgb_pipe.scheduler.get_statistics()
    print(f"\n  THG统计 (RGB):")
    print(f"    总NFE: {thg_stats['total_nfe']}")
    print(f"    节省NFE: {thg_stats['saved_nfe']}")
    print(f"    效率提升: {thg_stats['efficiency']}")

    output_rgb_dir = os.path.join(args.output_dir, "rgb")
    os.makedirs(output_rgb_dir, exist_ok=True)

    # 为RGB图像添加alpha通道（使用分割模型）
    images_rgba = segment_rgbs(images, seg_model, device)
    for idx, image in enumerate(images_rgba):
        image.save(os.path.join(output_rgb_dir, f"{idx}.png"))

    # 转换RGBA为RGB tensor（用于法线生成）
    # 使用load_image函数处理RGBA图像，自动blend alpha通道
    mv_rgbs = []
    for image_rgba in images_rgba:
        mv_rgbs.append(load_image(image_rgba, 768, 768))
    mv_rgbs = torch.stack(mv_rgbs)
    mv_rgbs = mv_rgbs.permute(0, 3, 1, 2).to(device)

    del rgb_pipe
    torch.cuda.empty_cache()

    # 7. 法线生成 - 启用THG优化
    print(f"\n[7/8] 生成法线图 (20视角, THG间隔={args.thg_interval})...")
    normal_pipe = UP2YouMV2NormalPipeline.from_pretrained(
        args.base_model_path,
    )

    # 同样使用THG
    base_scheduler = ShiftSNRScheduler.from_scheduler(
        normal_pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    normal_pipe.scheduler = TortoiseHareGuidanceScheduler.from_scheduler(
        base_scheduler,
        guidance_update_interval=args.thg_interval,
    )

    normal_pipe.init_custom_adapter(num_views=20)
    normal_pipe.load_custom_adapter(
        args.normal_adapter_path, weight_name='custom_adapter.safetensors'
    )

    normal_pipe.to(device=device, dtype=dtype)
    normal_pipe.cond_encoder.to(device=device, dtype=dtype)  # 修复：adapter也需要转换为fp16
    normal_pipe.enable_vae_slicing()

    with torch.no_grad():
        normals = normal_pipe(
            prompt=["Multi-view Human, Full Body, Normal Map, High Quality, HDR"],
            reference_rgbs=mv_rgbs,  # 修复：使用转换后的mv_rgbs
            control_image=target_poses,
            num_images_per_prompt=20,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=768,
            width=768,
        ).images

    # 打印THG统计信息
    thg_stats = normal_pipe.scheduler.get_statistics()
    print(f"\n  THG统计 (法线):")
    print(f"    总NFE: {thg_stats['total_nfe']}")
    print(f"    节省NFE: {thg_stats['saved_nfe']}")
    print(f"    效率提升: {thg_stats['efficiency']}")

    output_normal_dir = os.path.join(args.output_dir, "normal")
    os.makedirs(output_normal_dir, exist_ok=True)

    # 为法线图添加alpha通道（使用分割模型）
    normals_rgba = segment_rgbs(normals, seg_model, device)
    for idx, normal in enumerate(normals_rgba):
        normal.save(os.path.join(output_normal_dir, f"{idx}.png"))

    # 清理显存
    del seg_model
    del normal_pipe
    torch.cuda.empty_cache()

    # 8. 网格重建
    print("\n[8/8] 重建3D网格...")
    reconstructor = Reconstructor(device=device)

    pred_meshes_output_dir = os.path.join(args.output_dir, "pred_meshes")
    os.makedirs(pred_meshes_output_dir, exist_ok=True)

    gen_obj_path = reconstructor.run(
        color_pils=images_rgba,
        normal_pils=normals_rgba,
        smplx_obj_path=smplx_mesh_path,
        output_dir=pred_meshes_output_dir,
        replace_hand=True,
    )

    del reconstructor
    torch.cuda.empty_cache()

    # 9. 渲染视频
    print("\n[9/8] 渲染360度旋转视频...")
    video_renderer = CommonRenderer(resolution=1024, return_rgba=True)
    render_rgbs, render_normals = video_renderer.render_video(gen_obj_path, 360)

    rgb_video_path = os.path.join(args.output_dir, "result_rgb.mp4")
    normal_video_path = os.path.join(args.output_dir, "result_normal.mp4")
    tensor_to_video(render_rgbs, rgb_video_path)
    tensor_to_video(render_normals, normal_video_path)

    del video_renderer
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("推理完成!")
    print(f"输出目录: {args.output_dir}")
    print(f"最终网格: {gen_obj_path}")
    print(f"RGB视频: {rgb_video_path}")
    print(f"法线视频: {normal_video_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UP2You推理 - THG优化版本")

    # 输入输出
    parser.add_argument("--data_dir", type=str, required=True, help="参考图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")

    # 模型路径
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="基础SD模型路径"
    )
    parser.add_argument(
        "--shape_predictor_path",
        type=str,
        default="checkpoints/shape_predictor_dinov2.pth",
        help="形状预测器权重路径"
    )
    parser.add_argument(
        "--feature_aggregator_path",
        type=str,
        default="checkpoints/feature_aggregator.pth",
        help="特征聚合器权重路径"
    )
    parser.add_argument(
        "--rgb_adapter_path",
        type=str,
        default="checkpoints/rgb_adapter",
        help="RGB适配器路径"
    )
    parser.add_argument(
        "--normal_adapter_path",
        type=str,
        default="checkpoints/normal_adapter",
        help="法线适配器路径"
    )

    # THG参数
    parser.add_argument(
        "--thg_interval",
        type=int,
        default=3,
        help="THG引导更新间隔 (2-4推荐，越大越快但可能影响质量)"
    )

    # 推理参数
    parser.add_argument("--num_inference_steps", type=int, default=50, help="扩散步数")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG引导强度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--use_fp16", action="store_true", help="使用FP16推理")
    parser.add_argument(
        "--segment_model_name",
        type=str,
        default="ZhengPeng7/BiRefNet",
        help="背景分割模型名称"
    )

    args = parser.parse_args()
    main(args)
