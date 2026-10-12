#!/usr/bin/env python3
# Copyright (c) 2025
# MambaVision PaddlePaddle Implementation

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
import numpy as np
from typing import Optional, List, Tuple, Union
import math


# ============== Paddle版本的Selective Scan实现 ==============

class SelectiveScanPaddle(nn.Layer):
    """Paddle版本的Selective Scan模块"""

    @staticmethod
    def selective_scan_fn(
            u: Tensor,
            delta: Tensor,
            A: Tensor,
            B: Tensor,
            C: Tensor,
            D: Optional[Tensor] = None,
            z: Optional[Tensor] = None,
            delta_bias: Optional[Tensor] = None,
            delta_softplus: bool = True,
            return_last_state: bool = False,
    ):
        """
        u: (B, L, d_inner//2)
        delta: (B, L, d_inner//2)
        A: (d_inner//2, d_state)
        B: (B, d_state, L)
        C: (B, d_state, L)
        D: (d_inner//2,)
        z: (B, L, d_inner//2)

        返回: (B, L, d_inner//2)
        """
        dtype = u.dtype
        batch, seqlen, dim = u.shape
        dstate = A.shape[1]

        # 对delta进行softplus激活
        if delta_softplus:
            if delta_bias is not None:
                delta = F.softplus(delta + delta_bias)
            else:
                delta = F.softplus(delta)

        # 计算离散化的A和B
        # A_bar = exp(ΔA)
        deltaA = paddle.einsum('bld,dn->bldn', delta, A)  # (B, L, dim, dstate)
        deltaA = paddle.exp(deltaA)

        # B_bar = ΔB
        deltaB = paddle.einsum('bld,bnl->bldn', delta, B)  # (B, L, dim, dstate)

        # 初始化状态
        x = paddle.zeros((batch, dim, dstate), dtype=dtype)
        ys = []

        # 序列扫描
        for i in range(seqlen):
            # x = A_bar * x + B_bar * u
            u_i = u[:, i:i + 1, :]  # (B, 1, dim)
            # 扩展u_i以匹配deltaB的形状
            u_i_expanded = u_i.unsqueeze(-1)  # (B, 1, dim, 1)
            deltaB_i = deltaB[:, i].unsqueeze(1)  # (B, 1, dim, dstate)

            # B_bar * u
            deltaB_u = deltaB_i * u_i_expanded  # (B, 1, dim, dstate)
            deltaB_u = deltaB_u.squeeze(1)  # (B, dim, dstate)

            # 状态更新
            x = deltaA[:, i] * x + deltaB_u

            # y = C * x
            # 注意: C的形状是(B, d_state, L)，这里需要取第i个位置的C
            C_i = C[:, :, i:i + 1]  # (B, d_state, 1)
            C_i = C_i.transpose([0, 2, 1])  # (B, 1, d_state)

            # 扩展x以匹配C_i
            x_expanded = x.unsqueeze(1)  # (B, 1, dim, dstate)
            C_i_expanded = C_i.unsqueeze(2)  # (B, 1, 1, d_state)

            # 计算y_i = sum(x * C_i, axis=-1)
            y_i = paddle.sum(x_expanded * C_i_expanded, axis=-1)  # (B, 1, dim)
            y_i = y_i.squeeze(1)  # (B, dim)
            ys.append(y_i.unsqueeze(1))

        y = paddle.concat(ys, axis=1)  # (B, L, dim)

        # 添加残差连接
        if D is not None:
            y = y + u * D.unsqueeze(0).unsqueeze(0)

        # 添加门控
        if z is not None:
            y = y * F.silu(z)

        if return_last_state:
            return y, x
        return y


# ============== 模型核心组件 ==============

class LayerNorm2d(nn.Layer):
    """2D LayerNorm适配Paddle"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = self.create_parameter(
            shape=normalized_shape,
            default_initializer=nn.initializer.Constant(1.0)
        )
        self.bias = self.create_parameter(
            shape=normalized_shape,
            default_initializer=nn.initializer.Constant(0.0)
        )

    def forward(self, x: Tensor) -> Tensor:
        # 从NCHW转换为NHWC进行LayerNorm
        x = x.transpose([0, 2, 3, 1])
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 转换回NCHW
        x = x.transpose([0, 3, 1, 2])
        return x


class Downsample(nn.Layer):
    """下采样模块"""

    def __init__(self, dim, keep_dim=False):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim

        self.reduction = nn.Sequential(
            nn.Conv2D(dim, dim_out, 3, stride=2, padding=1, bias_attr=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.reduction(x)


class PatchEmbed(nn.Layer):
    """Patch Embedding模块"""

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2D(in_chans, in_dim, 3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(in_dim, epsilon=1e-4),
            nn.ReLU(),
            nn.Conv2D(in_dim, dim, 3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(dim, epsilon=1e-4),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Layer):
    """卷积块"""

    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2D(dim, dim, kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2D(dim, epsilon=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2D(dim, dim, kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2D(dim, epsilon=1e-5)

        self.layer_scale = layer_scale
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = self.create_parameter(
                shape=[dim],
                default_initializer=nn.initializer.Constant(layer_scale)
            )
            self.layer_scale = True
        else:
            self.layer_scale = False

        # DropPath在Paddle中需要自定义
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.layer_scale:
            x = x * self.gamma.reshape([1, -1, 1, 1])

        x = input + self.drop_path(x)
        return x


# ============== DropPath实现 ==============

class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = paddle.to_tensor(
            np.random.binomial(1, keep_prob, shape),
            dtype=x.dtype
        )

        if keep_prob > 0.:
            random_tensor = random_tensor / keep_prob

        return x * random_tensor


# ============== Mamba核心模块 ==============

class MambaVisionMixer(nn.Layer):
    """Mamba Vision Mixer模块"""

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dt_rank: Union[str, int] = "auto",
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            conv_bias: bool = True,
            bias: bool = False,
            layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias_attr=bias)

        # 卷积层
        conv_bias_attr = conv_bias
        self.conv1d_x = nn.Conv1D(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner // 2,
            bias_attr=conv_bias_attr
        )
        self.conv1d_z = nn.Conv1D(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner // 2,
            bias_attr=conv_bias_attr
        )

        # x_proj: 从x映射到(dt_rank + 2*d_state)
        self.x_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias_attr=False
        )

        # dt_proj
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias_attr=True)

        # 参数初始化
        self._init_dt_proj(dt_init, dt_scale, dt_min, dt_max, dt_init_floor)

        # A参数（状态矩阵的对数）
        A = paddle.arange(1, self.d_state + 1, dtype=paddle.float32).reshape([1, -1])
        A = paddle.tile(A, [self.d_inner // 2, 1])
        A_log = paddle.log(A)
        self.A_log = self.create_parameter(
            shape=A_log.shape,
            default_initializer=nn.initializer.Assign(A_log)
        )
        self.A_log.stop_gradient = True  # 不参与权重衰减

        # D参数
        self.D = self.create_parameter(
            shape=[self.d_inner // 2],
            default_initializer=nn.initializer.Constant(1.0)
        )
        self.D.stop_gradient = True  # 不参与权重衰减

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias_attr=bias)

    def _init_dt_proj(self, dt_init: str, dt_scale: float, dt_min: float, dt_max: float, dt_init_floor: float):
        """初始化dt_proj"""
        dt_init_std = self.dt_rank ** -0.5 * dt_scale

        if dt_init == "constant":
            nn.initializer.Constant(dt_init_std)(self.dt_proj.weight)
        elif dt_init == "random":
            nn.initializer.Uniform(-dt_init_std, dt_init_std)(self.dt_proj.weight)
        else:
            raise ValueError(f"Unknown dt_init: {dt_init}")

        # 初始化dt bias
        dt = paddle.exp(
            paddle.uniform([self.d_inner // 2]) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = paddle.clip(dt, min=dt_init_floor)
        inv_dt = dt + paddle.log(-paddle.expm1(-dt))

        self.dt_proj.bias.set_value(inv_dt)
        self.dt_proj.bias.stop_gradient = False

    def forward(self, hidden_states: Tensor) -> Tensor:
        """前向传播"""
        batch, seqlen, _ = hidden_states.shape

        # 输入投影
        xz = self.in_proj(hidden_states)  # (B, L, d_inner)

        # 分离x和z
        xz = xz.transpose([0, 2, 1])  # (B, d_inner, L)
        x, z = paddle.chunk(xz, 2, axis=1)  # 各(B, d_inner//2, L)

        # 卷积处理
        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))

        # 计算A
        A = -paddle.exp(self.A_log)

        # 计算x_proj: (B, L, dt_rank + 2*d_state)
        x_proj = self.x_proj(x.transpose([0, 2, 1]))  # (B, L, dt_rank + 2*d_state)

        # 分离dt, B, C
        dt, B, C = paddle.split(
            x_proj,
            [self.dt_rank, self.d_state, self.d_state],
            axis=-1
        )

        # 计算dt
        dt = self.dt_proj(dt)  # (B, L, d_inner//2)
        dt = dt.transpose([0, 2, 1])  # (B, d_inner//2, L)

        # 调整B, C形状
        B = B.transpose([0, 2, 1])  # (B, d_state, L)
        C = C.transpose([0, 2, 1])  # (B, d_state, L)

        # 转置x用于selective scan
        x_t = x.transpose([0, 2, 1])  # (B, L, d_inner//2)
        dt_t = dt.transpose([0, 2, 1])  # (B, L, d_inner//2)
        z_t = z.transpose([0, 2, 1])  # (B, L, d_inner//2)

        # Selective Scan
        y = SelectiveScanPaddle.selective_scan_fn(
            u=x_t,  # (B, L, d_inner//2)
            delta=dt_t,  # (B, L, d_inner//2)
            A=A,  # (d_inner//2, d_state)
            B=B,  # (B, d_state, L)
            C=C,  # (B, d_state, L)
            D=self.D,  # (d_inner//2,)
            z=z_t,  # (B, L, d_inner//2)
            delta_bias=self.dt_proj.bias,
            delta_softplus=True
        )

        # 合并y和z
        y = paddle.concat([y, z_t], axis=-1)  # (B, L, d_inner)

        # 输出投影
        out = self.out_proj(y)
        return out


# ============== Attention模块 ==============

class Attention(nn.Layer):
    """注意力模块"""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        # 计算QKV
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, self.head_dim])
        qkv = qkv.transpose([2, 0, 3, 1, 4])  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 规范化
        q, k = self.q_norm(q), self.k_norm(k)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # 输出
        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# ============== MLP模块 ==============

class Mlp(nn.Layer):
    """MLP模块"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============== 基础块模块 ==============

def window_partition(x: Tensor, window_size: int) -> Tensor:
    """将特征图划分为窗口"""
    B, C, H, W = x.shape
    x = x.reshape([B, C, H // window_size, window_size, W // window_size, window_size])
    x = x.transpose([0, 2, 4, 3, 5, 1])
    windows = x.reshape([-1, window_size * window_size, C])
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    """将窗口还原为特征图"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 5, 1, 3, 2, 4])
    x = x.reshape([B, -1, H, W])
    return x


class Block(nn.Layer):
    """Transformer/Mamba块"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            counter: int,
            transformer_blocks: List[int],
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_scale: bool = False,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer=nn.GELU,
            layer_scale=None,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        # 选择使用Attention还是Mamba
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

        # Layer Scale
        use_layer_scale = layer_scale is not None and isinstance(layer_scale, (int, float))
        if use_layer_scale:
            self.gamma_1 = self.create_parameter(
                shape=[dim],
                default_initializer=nn.initializer.Constant(layer_scale)
            )
            self.gamma_2 = self.create_parameter(
                shape=[dim],
                default_initializer=nn.initializer.Constant(layer_scale)
            )
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0

    def forward(self, x: Tensor) -> Tensor:
        # 第一个归一化 + mixer + 残差
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        # 第二个归一化 + MLP + 残差
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ============== 层级模块 ==============

class MambaVisionLayer(nn.Layer):
    """MambaVision层级"""

    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: int,
            conv: bool = False,
            downsample: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_scale: bool = False,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[float, List[float]] = 0.,
            layer_scale: Optional[float] = None,
            layer_scale_conv: Optional[float] = None,
            transformer_blocks: List[int] = [],
    ):
        super().__init__()

        self.conv = conv
        self.window_size = window_size

        if conv:
            # 卷积块
            if isinstance(drop_path, list):
                drop_path_list = drop_path
            else:
                drop_path_list = [drop_path] * depth

            self.blocks = nn.LayerList([
                ConvBlock(
                    dim=dim,
                    drop_path=drop_path_list[i],
                    layer_scale=layer_scale_conv,
                    kernel_size=3
                ) for i in range(depth)
            ])
            self.transformer_block = False
        else:
            # Transformer/Mamba块
            if isinstance(drop_path, list):
                drop_path_list = drop_path
            else:
                drop_path_list = [drop_path] * depth

            self.blocks = nn.LayerList([
                Block(
                    dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_list[i],
                    layer_scale=layer_scale
                ) for i in range(depth)
            ])
            self.transformer_block = True

        # 下采样
        self.downsample = None if not downsample else Downsample(dim=dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        _, _, H, W = x.shape

        if self.transformer_block:
            # 窗口划分
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size

            if pad_r > 0 or pad_b > 0:
                x_padded = F.pad(x, [0, pad_r, 0, pad_b], data_format='NCHW')
                _, _, Hp, Wp = x_padded.shape
            else:
                x_padded = x
                Hp, Wp = H, W

            # 转换为序列
            x_seq = window_partition(x_padded, self.window_size)
        else:
            x_seq = x

        # 通过所有块
        for blk in self.blocks:
            x_seq = blk(x_seq)

        if self.transformer_block:
            # 窗口还原
            x_seq = window_reverse(x_seq, self.window_size, Hp, Wp)

            if pad_r > 0 or pad_b > 0:
                x_seq = x_seq[:, :, :H, :W]

        # 下采样
        if self.downsample is not None:
            output = self.downsample(x_seq)
        else:
            output = x_seq

        return output, x_seq


# ============== 主模型 ==============
@manager.MODELS.add_component
class MambaVision(nn.Layer):
    """MambaVision主模型"""

    def __init__(
            self,
            dim: int = 128,
            in_dim: int = 64,
            depths: Tuple[int, ...] = (3, 3, 10, 5),
            window_size: Tuple[int, ...] = (8, 8, 14, 7),
            mlp_ratio: float = 4.0,
            num_heads: Tuple[int, ...] = (2, 4, 8, 16),
            drop_path_rate: float = 0.2,
            in_chans: int = 3,
            num_classes: int = 1000,
            qkv_bias: bool = True,
            qk_scale: bool = False,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            layer_scale: Optional[float] = None,
            layer_scale_conv: Optional[float] = None,
            **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        num_features = int(dim * 2 ** (len(depths) - 1))

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            in_dim=in_dim,
            dim=dim
        )

        # 随机深度衰减
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]

        # 构建层级
        self.levels = nn.LayerList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False

            level = MambaVisionLayer(
                dim=int(dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < 3),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[i] % 2 != 0
                else list(range(depths[i] // 2, depths[i])),
            )
            self.levels.append(level)

        # 头部
        self.norm = nn.BatchNorm2D(num_features, epsilon=1e-5)
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            # Paddle的trunc_normal初始化
            trunc_normal = nn.initializer.TruncatedNormal(std=0.02)
            trunc_normal(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.initializer.Constant(0)(m.bias)
            nn.initializer.Constant(1.0)(m.weight)
        elif isinstance(m, LayerNorm2d):
            nn.initializer.Constant(0)(m.bias)
            nn.initializer.Constant(1.0)(m.weight)
        elif isinstance(m, nn.BatchNorm2D):
            nn.initializer.Constant(1.0)(m.weight)
            nn.initializer.Constant(0)(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """提取特征"""
        x = self.patch_embed(x)

        for level in self.levels:
            x, _ = level(x)

        x = self.norm(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_pretrained(self, checkpoint_path: str, strict: bool = True):
        """加载预训练权重"""
        try:
            # 加载PyTorch权重
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 移除'module.'前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # 转换权重到Paddle格式
            paddle_state_dict = {}
            for key, value in state_dict.items():
                # 跳过不需要的键
                if 'num_batches_tracked' in key:
                    continue

                # 转换键名
                new_key = key

                # 转换权重格式
                if value.dim() == 4:
                    # Conv2D权重: PyTorch [out_c, in_c, h, w] -> Paddle [h, w, in_c, out_c]
                    value = value.numpy().transpose(2, 3, 1, 0)
                elif value.dim() == 2:
                    # Linear权重: 转置
                    value = value.numpy().transpose(1, 0)
                else:
                    value = value.numpy()

                paddle_state_dict[new_key] = value

            # 加载权重
            self.set_state_dict(paddle_state_dict)
            print(f"Successfully loaded weights from {checkpoint_path}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise


# ============== 模型工厂函数 ==============

def create_mambavision_tiny(**kwargs):
    """创建MambaVision-Tiny模型"""
    model = MambaVision(
        dim=96,
        in_dim=64,
        depths=(2, 2, 8, 2),
        window_size=(8, 8, 14, 7),
        num_heads=(2, 4, 8, 16),
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        **kwargs
    )
    return model


def create_mambavision_small(**kwargs):
    """创建MambaVision-Small模型"""
    model = MambaVision(
        dim=96,
        in_dim=64,
        depths=(2, 2, 18, 2),
        window_size=(8, 8, 14, 7),
        num_heads=(2, 4, 8, 16),
        mlp_ratio=4.0,
        drop_path_rate=0.3,
        **kwargs
    )
    return model


def create_mambavision_base(**kwargs):
    """创建MambaVision-Base模型"""
    model = MambaVision(
        dim=128,
        in_dim=64,
        depths=(2, 2, 18, 2),
        window_size=(8, 8, 14, 7),
        num_heads=(4, 8, 16, 32),
        mlp_ratio=4.0,
        drop_path_rate=0.5,
        **kwargs
    )
    return model


# ============== 分割和检测任务适配 ==============

class MambaVisionForSegmentation(nn.Layer):
    """用于分割任务的MambaVision模型"""

    def __init__(
            self,
            dim: int = 128,
            in_dim: int = 64,
            depths: Tuple[int, ...] = (3, 3, 10, 5),
            window_size: Tuple[int, ...] = (8, 8, 14, 7),
            mlp_ratio: float = 4.0,
            num_heads: Tuple[int, ...] = (2, 4, 8, 16),
            drop_path_rate: float = 0.2,
            in_chans: int = 3,
            out_indices: Tuple[int, ...] = (0, 1, 2, 3),
            pretrained: Optional[str] = None,
            **kwargs
    ):
        super().__init__()

        self.out_indices = out_indices
        self.num_features = [int(dim * 2 ** i) for i in range(4)]

        # 主模型
        self.backbone = MambaVision(
            dim=dim,
            in_dim=in_dim,
            depths=depths,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            in_chans=in_chans,
            num_classes=0,  # 不要分类头
            **kwargs
        )

        # 移除分类头部
        del self.backbone.head

        # 为每个输出阶段添加归一化层
        self.norms = nn.LayerList()
        for idx in out_indices:
            norm = LayerNorm2d(self.num_features[idx])
            self.norms.append(norm)

    def forward(self, x: Tensor) -> List[Tensor]:
        """前向传播，返回多尺度特征"""
        x = self.backbone.patch_embed(x)

        features = []
        for i, level in enumerate(self.backbone.levels):
            x, x_out = level(x)
            if i in self.out_indices:
                idx = self.out_indices.index(i)
                feat = self.norms[idx](x_out)
                features.append(feat)

        return features

    def load_pretrained(self, checkpoint_path: str):
        """加载预训练权重"""
        self.backbone.load_pretrained(checkpoint_path)




# ============== 使用示例 ==============

if __name__ == "__main__":
    # 设置paddle随机种子
    paddle.seed(42)

    print("=" * 60)
    print("MambaVision PaddlePaddle Implementation Test")
    print("=" * 60)

    # 测试Tiny模型
    print("\n1. Testing MambaVision-Tiny for classification...")
    model_tiny = create_mambavision_tiny(num_classes=1000)

    # 计算参数量
    total_params = sum(p.numel() for p in model_tiny.parameters() if not p.stop_gradient)
    total_params_m = float(total_params.numpy()) / 1e6

    print(f"Model: MambaVision-Tiny")
    print(f"Total parameters: {total_params_m:.2f}M")

    # 测试前向传播
    x = paddle.randn([1, 3, 224, 224])
    model_tiny.eval()

    with paddle.no_grad():
        out = model_tiny(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Classification test passed!")

    # 测试分割模型
    print("\n2. Testing MambaVision for segmentation...")
    model_seg = MambaVisionForSegmentation(
        dim=96,
        in_dim=64,
        depths=(2, 2, 8, 2),
        window_size=(8, 8, 14, 7),
        num_heads=(2, 4, 8, 16),
        out_indices=(0, 1, 2, 3)
    )

    with paddle.no_grad():
        features = model_seg(x)

    print(f"Number of feature levels: {len(features)}")
    for i, feat in enumerate(features):
        print(f"  Level {i}: shape = {feat.shape}")
    print("✓ Segmentation test passed!")

    # 测试不同分辨率的输入
    print("\n3. Testing with different input resolutions...")
    resolutions = [(224, 224), (256, 256), (512, 512)]

    for h, w in resolutions:
        x_test = paddle.randn([1, 3, h, w])
        with paddle.no_grad():
            out_test = model_tiny(x_test)
        print(f"  Input: {h}x{w}, Output: {out_test.shape} ✓")

    print("\n" + "=" * 60)
    print("All tests passed! MambaVision Paddle implementation is working.")
    print("=" * 60)