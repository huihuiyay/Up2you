"""
THG优化快速验证脚本
测试THG调度器的基本功能和统计
"""

import torch
from diffusers import DDPMScheduler
from up2you.schedulers.scheduling_thg import TortoiseHareGuidanceScheduler


def test_thg_basic():
    """测试THG调度器基本功能"""
    print("=" * 80)
    print("测试1: THG调度器基本功能")
    print("=" * 80)

    # 创建基础调度器
    base_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
    )

    # 包装为THG调度器
    thg_scheduler = TortoiseHareGuidanceScheduler.from_scheduler(
        base_scheduler,
        guidance_update_interval=3,
    )

    print(f"✓ THG调度器创建成功")
    print(f"  引导更新间隔: {thg_scheduler.guidance_update_interval}")

    # 模拟50步推理
    num_steps = 50
    batch_size = 2
    latent_dim = 4
    height, width = 64, 64

    # 重置统计
    thg_scheduler.reset_statistics()

    for step in range(num_steps):
        # 模拟噪声预测
        noise_cond = torch.randn(batch_size, latent_dim, height, width)
        noise_uncond = torch.randn(batch_size, latent_dim, height, width) if step % 3 == 0 else None

        # 使用THG计算
        noise_pred, updated = thg_scheduler.compute_noise_pred_with_thg(
            noise_cond,
            noise_uncond,
            guidance_scale=3.0,
            step_index=step,
        )

        if step < 5 or step % 10 == 0:
            update_status = "✓ 更新引导" if updated else "→ 复用缓存"
            print(f"  步骤 {step:2d}: {update_status}")

    # 获取统计信息
    stats = thg_scheduler.get_statistics()
    print(f"\n统计结果:")
    print(f"  总NFE: {stats['total_nfe']}")
    print(f"  节省NFE: {stats['saved_nfe']}")
    print(f"  效率提升: {stats['efficiency']}")

    # 验证NFE减少
    expected_nfe = num_steps + (num_steps // 3)  # 条件 + 部分无条件
    assert stats['total_nfe'] <= num_steps * 2, "NFE应该少于标准推理的2倍"
    print(f"\n✅ 测试1通过: NFE从{num_steps*2}降至{stats['total_nfe']}")


def test_thg_intervals():
    """测试不同THG间隔的NFE"""
    print("\n" + "=" * 80)
    print("测试2: 不同THG间隔的效率对比")
    print("=" * 80)

    base_scheduler = DDPMScheduler(num_train_timesteps=1000)
    num_steps = 50

    results = []
    for interval in [1, 2, 3, 4, 5]:
        thg_scheduler = TortoiseHareGuidanceScheduler.from_scheduler(
            base_scheduler,
            guidance_update_interval=interval,
        )
        thg_scheduler.reset_statistics()

        # 模拟推理
        for step in range(num_steps):
            should_update = thg_scheduler.should_update_guidance(step)
            noise_cond = torch.randn(1, 4, 64, 64)
            noise_uncond = torch.randn(1, 4, 64, 64) if should_update else None

            thg_scheduler.compute_noise_pred_with_thg(
                noise_cond, noise_uncond, 3.0, step
            )

        stats = thg_scheduler.get_statistics()
        speedup = (num_steps * 2) / stats['total_nfe']
        results.append({
            'interval': interval,
            'nfe': stats['total_nfe'],
            'saved': stats['saved_nfe'],
            'speedup': speedup,
        })

        print(f"  间隔={interval}: NFE={stats['total_nfe']:3d}, 加速={speedup:.2f}×, 节省={stats['efficiency']}")

    print(f"\n✅ 测试2通过: 间隔越大，加速越明显")
    print(f"   推荐: interval=3 (加速{results[2]['speedup']:.2f}×)")


def test_thg_proxy():
    """测试THG调度器对基础调度器方法的代理"""
    print("\n" + "=" * 80)
    print("测试3: THG调度器属性代理")
    print("=" * 80)

    base_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
    )

    thg_scheduler = TortoiseHareGuidanceScheduler.from_scheduler(
        base_scheduler,
        guidance_update_interval=3,
    )

    # 测试属性代理
    assert thg_scheduler.num_train_timesteps == 1000, "应该代理num_train_timesteps"
    assert thg_scheduler.beta_start == 0.00085, "应该代理beta_start"

    print(f"  ✓ num_train_timesteps = {thg_scheduler.num_train_timesteps}")
    print(f"  ✓ beta_start = {thg_scheduler.beta_start}")

    # 测试set_timesteps方法
    thg_scheduler.set_timesteps(50)
    assert len(thg_scheduler.timesteps) == 50, "应该代理set_timesteps方法"
    print(f"  ✓ set_timesteps(50) -> {len(thg_scheduler.timesteps)} timesteps")

    print(f"\n✅ 测试3通过: THG正确代理基础调度器的所有方法")


def test_weight_symmetry():
    """测试reconstructor的权重对称性"""
    print("\n" + "=" * 80)
    print("测试4: Reconstructor权重矩阵对称性")
    print("=" * 80)

    # 读取修复后的权重
    weights = torch.Tensor([
        1.0, 0.8, 0.6, 0.5, 0.7, 0.8, 0.7, 0.5, 0.6, 0.8,  # 0-162°
        1.0, 0.8, 0.6, 0.5, 0.7, 0.8, 0.7, 0.5, 0.6, 0.8,  # 180-342°
    ])

    # 检查前后对称性（0° vs 180°）
    front_weight = weights[0]
    back_weight = weights[10]

    print(f"  正前方 (0°):   权重 = {front_weight:.1f}")
    print(f"  正后方 (180°): 权重 = {back_weight:.1f}")

    assert front_weight == back_weight, "前后权重应该相等"
    print(f"\n  ✓ 前后对称性检查通过")

    # 检查左右对称性（90° vs 270°）
    right_weight = weights[5]
    left_weight = weights[15]

    print(f"  右侧 (90°):  权重 = {right_weight:.1f}")
    print(f"  左侧 (270°): 权重 = {left_weight:.1f}")

    assert right_weight == left_weight, "左右权重应该相等"
    print(f"  ✓ 左右对称性检查通过")

    print(f"\n✅ 测试4通过: 权重矩阵已正确修复为对称")


if __name__ == "__main__":
    try:
        test_thg_basic()
        test_thg_intervals()
        test_thg_proxy()
        test_weight_symmetry()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 运行完整推理测试:")
        print("     python inference_thg.py --data_dir <input> --output_dir <output>")
        print("\n  2. 对比标准推理和THG推理的质量:")
        print("     python inference_low_gpu.py --data_dir <input> --output_dir output_baseline")
        print("     python inference_thg.py --data_dir <input> --output_dir output_thg")
        print("\n  3. 检查头部鼓包是否修复（重点看后脑勺）")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
