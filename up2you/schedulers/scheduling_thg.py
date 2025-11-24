"""
Tortoise and Hare Guidance (THG) Scheduler
龟兔赛跑引导调度器 - 减少扩散模型的NFE（网络前向评估次数）

原理：
- 乌龟分支：基础噪声估计 ε̂_c，每个细时间步都更新（慢而精确）
- 兔子分支：引导项 Δε̂ = ε̂_c - ε̂_∅，只在粗时间步更新（快速跳跃）
- 最终噪声预测：ε̂ = ε̂_c + guidance_scale * Δε̂

参考论文：Tortoise and Hare: Efficient Guidance for Diffusion Models
预期加速：~1.4x（NFE从100降到~70）
"""

import torch
from typing import Optional, Tuple, Union
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class TortoiseHareGuidanceScheduler:
    """
    Tortoise and Hare Guidance包装器

    参数：
        base_scheduler: 基础调度器（如DDPMScheduler）
        guidance_update_interval: 兔子分支更新间隔（粗时间步）
        tortoise_update_interval: 乌龟分支更新间隔（默认1，每步都更新）
    """

    def __init__(
        self,
        base_scheduler: DDPMScheduler,
        guidance_update_interval: int = 3,  # 每3步更新一次无条件分支
        tortoise_update_interval: int = 1,  # 乌龟每步都更新
    ):
        self.base_scheduler = base_scheduler
        self.guidance_update_interval = guidance_update_interval
        self.tortoise_update_interval = tortoise_update_interval

        # 缓存上一次的引导项
        self.cached_guidance_delta = None
        self.last_guidance_step = -1

        # 统计NFE
        self.total_nfe = 0
        self.saved_nfe = 0

    @classmethod
    def from_scheduler(
        cls,
        scheduler: DDPMScheduler,
        guidance_update_interval: int = 3,
    ):
        """从现有调度器创建THG包装器"""
        return cls(
            base_scheduler=scheduler,
            guidance_update_interval=guidance_update_interval,
        )

    def __getattr__(self, name):
        """代理到基础调度器的属性和方法"""
        return getattr(self.base_scheduler, name)

    def should_update_guidance(self, step_index: int) -> bool:
        """判断是否应该更新引导项（兔子分支）"""
        # 第一步必须更新
        if step_index == 0:
            return True
        # 按间隔更新
        if step_index % self.guidance_update_interval == 0:
            return True
        return False

    def compute_noise_pred_with_thg(
        self,
        noise_pred_cond: torch.Tensor,
        noise_pred_uncond: Optional[torch.Tensor],
        guidance_scale: float,
        step_index: int,
    ) -> Tuple[torch.Tensor, bool]:
        """
        使用THG计算最终噪声预测

        返回：
            (noise_pred, updated_guidance) - 噪声预测和是否更新了引导项的标志
        """
        # 无CFG时直接返回条件预测
        if noise_pred_uncond is None or guidance_scale == 1.0:
            self.total_nfe += 1
            return noise_pred_cond, False

        # 判断是否需要更新引导项
        should_update = self.should_update_guidance(step_index)

        if should_update:
            # 乌龟+兔子：计算新的引导项
            guidance_delta = noise_pred_cond - noise_pred_uncond
            self.cached_guidance_delta = guidance_delta
            self.last_guidance_step = step_index
            self.total_nfe += 2  # cond + uncond
            updated = True
        else:
            # 只用乌龟：复用缓存的引导项
            guidance_delta = self.cached_guidance_delta
            self.total_nfe += 1  # 只有cond
            self.saved_nfe += 1
            updated = False

        # 最终噪声预测
        noise_pred = noise_pred_cond + guidance_scale * guidance_delta

        return noise_pred, updated

    def get_statistics(self) -> dict:
        """获取NFE统计信息"""
        return {
            "total_nfe": self.total_nfe,
            "saved_nfe": self.saved_nfe,
            "efficiency": f"{self.saved_nfe / max(self.total_nfe, 1) * 100:.1f}%",
            "guidance_interval": self.guidance_update_interval,
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.total_nfe = 0
        self.saved_nfe = 0
        self.cached_guidance_delta = None
        self.last_guidance_step = -1


def enable_thg_for_pipeline(
    pipeline,
    guidance_update_interval: int = 3,
    wrap_scheduler: bool = True,
) -> TortoiseHareGuidanceScheduler:
    """
    为Pipeline启用THG优化

    参数：
        pipeline: Diffusion Pipeline对象
        guidance_update_interval: 引导更新间隔（推荐2-4）
        wrap_scheduler: 是否包装调度器（建议True）

    返回：
        THG调度器对象

    使用示例：
        >>> from up2you.schedulers.scheduling_thg import enable_thg_for_pipeline
        >>> thg_scheduler = enable_thg_for_pipeline(rgb_pipe, guidance_update_interval=3)
        >>> # 推理完成后查看统计
        >>> print(thg_scheduler.get_statistics())
    """
    thg_scheduler = TortoiseHareGuidanceScheduler.from_scheduler(
        pipeline.scheduler,
        guidance_update_interval=guidance_update_interval,
    )

    if wrap_scheduler:
        # 直接替换pipeline的调度器
        # 注意：THG调度器会代理所有基础调度器的方法
        pipeline.scheduler = thg_scheduler

    return thg_scheduler
