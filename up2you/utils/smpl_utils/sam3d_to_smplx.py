"""
SAM-3D-Body 到 SMPLX 的转换工具
将 sam-3d-body 的 MHR 模型输出转换为 SMPLX 兼容的参数和渲染
"""

import torch
import numpy as np
from kiui.mesh import Mesh
from .render import Renderer
from .camera import Camera
from .mesh import normalize_vertices


class Sam3DToSMPLXAdapter:
    """
    将 sam-3d-body 输出的 MHR mesh 和参数转换为 SMPLX 兼容格式
    """

    def __init__(self, device="cuda", background_color="white"):
        self.device = device
        self.camera = Camera(device=device)
        self.renderer = Renderer(device=device)
        self.bg_color = self.renderer.get_bg_color(background_color).to(self.device)

    def convert_sam3d_output(self, sam3d_output):
        """
        将 sam-3d-body 的输出转换为可用于渲染的格式

        Args:
            sam3d_output: dict, 包含 sam-3d-body 的输出
                - pred_vertices: (N, 3) 预测的顶点坐标
                - shape_params: (K,) 形状参数
                - body_pose_params: (M,) 身体姿态参数
                - global_rot: (3,) 全局旋转
                等其他参数

        Returns:
            dict: 包含转换后的参数
                - vertices: torch.Tensor (N, 3)
                - faces: torch.Tensor (F, 3)
                - shape_params: torch.Tensor (K,)
                - pose_params: torch.Tensor (M,)
        """
        # 提取顶点和面
        vertices = torch.from_numpy(sam3d_output['pred_vertices']).float().to(self.device)

        # 标准化顶点坐标 (与 AposeRenderer 保持一致)
        vertices_normalized = self.normalize_vertices(vertices.cpu().numpy())
        vertices_normalized = torch.from_numpy(vertices_normalized).float().to(self.device)

        # 提取参数
        shape_params = torch.from_numpy(sam3d_output['shape_params']).float().to(self.device)
        body_pose_params = torch.from_numpy(sam3d_output['body_pose_params']).float().to(self.device)

        return {
            'vertices': vertices_normalized,
            'shape_params': shape_params,
            'body_pose_params': body_pose_params,
            'pred_vertices_raw': vertices,  # 保留原始顶点用于其他用途
        }

    def normalize_vertices(self, vertices, bound=0.9):
        """
        标准化顶点坐标到 [-bound, bound] 范围
        与 AposeRenderer 中的 normalize_vertices 保持一致
        """
        return normalize_vertices(vertices, bound=bound)

    def render_multiview(
        self,
        vertices,
        faces,
        vertex_colors=None,
        height=768,
        width=768,
        num_views=12,
        normal_type="camera",
        return_rgba=False
    ):
        """
        从给定的 mesh 渲染多视角图像

        Args:
            vertices: torch.Tensor (N, 3) 顶点坐标
            faces: torch.Tensor (F, 3) 面索引
            vertex_colors: Optional[torch.Tensor] (N, 4) 顶点颜色
            height: int, 渲染高度
            width: int, 渲染宽度
            num_views: int, 视角数量
            normal_type: str, "camera" 或 "world"
            return_rgba: bool, 是否返回 RGBA

        Returns:
            tuple: (normals, semantics) 渲染的法线图和语义图
        """
        # 构建 mesh
        if vertex_colors is None:
            vertex_colors = torch.ones((vertices.shape[0], 4), device=self.device)
            vertex_colors[:, :3] = 0.8  # 灰白色

        mesh = Mesh(
            v=vertices,
            f=faces,
            vc=vertex_colors,
            device=self.device
        )
        mesh.auto_normal()

        # 生成多视角相机参数
        mvps, rots = self._build_views(num_views)

        # 渲染
        smpl_pkg = self.renderer(
            mesh,
            mvp=mvps,
            h=height,
            w=width,
            shading_mode='albedo',
            bg_color=self.bg_color,
        )

        # 处理法线
        if normal_type == "world":
            smpl_normal = self._camera_to_world_normal(
                smpl_pkg['normal'], smpl_pkg['alpha'], rots, self.bg_color
            )
        elif normal_type == "camera":
            smpl_normal = smpl_pkg['normal']
        else:
            raise ValueError(f"Invalid normal type: {normal_type}")

        smpl_semantic = smpl_pkg['image']

        # 返回格式
        if return_rgba:
            smpl_normal_rgba = torch.cat([smpl_normal, smpl_pkg['alpha']], dim=-1)
            smpl_semantic_rgba = torch.cat([smpl_semantic, smpl_pkg['alpha']], dim=-1)
            return smpl_normal_rgba.permute(0, 3, 1, 2), smpl_semantic_rgba.permute(0, 3, 1, 2)
        else:
            return smpl_normal.permute(0, 3, 1, 2), smpl_semantic.permute(0, 3, 1, 2)

    def _build_views(self, num_views):
        """
        构建均匀分布的环形视角
        """
        if num_views < 1:
            raise ValueError(f"Invalid number of views: {num_views}")

        yaws = [float(i) * (360.0 / num_views) for i in range(num_views)]
        yaws = [(-yaw) % 360.0 for yaw in yaws]
        mvps, rots, _, _ = self.camera.get_orthogonal_camera(yaws)
        return mvps, rots

    def _camera_to_world_normal(self, normals_camera, masks, rot, bg_color):
        """
        将相机空间法线转换到世界空间
        """
        import torch.nn.functional as F

        normals_world = normals_camera * masks * 2 - 1
        normals_world = F.normalize(normals_world, dim=-1)
        normals_world = normals_world * masks - (1 - masks)

        rot_transpose = rot.transpose(1, 2)
        normals_world = torch.bmm(
            normals_world.reshape(len(normals_camera), -1, 3),
            rot_transpose
        ).reshape(*normals_camera.shape)

        normals_world = (normals_world + 1) / 2
        normals_world = normals_world * masks + (1 - masks) * bg_color

        return normals_world
