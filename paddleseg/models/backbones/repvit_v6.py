# paddleseg/models/backbones/repvit_seg.py
# 完整 PaddleSeg backbone 组件，语义分割专用 - 优化版本
import paddle
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, AdaptiveAvgPool2D, Linear

from paddleseg.cvlibs import manager
from paddleseg.utils import utils


# ---------------- 工具 ----------------
def _make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v if new_v >= 0.9 * v else new_v + divisor


def build_act(act):
    # 优化：统一使用ReLU6提升推理速度和量化友好性
    return {"relu": nn.ReLU6()}[act]


# ---------------- 结构重参数化 DW ----------------
class RepDW(nn.Layer):
    """训练多分支，推理融合为单 3x3 - 修复版本"""

    def __init__(self, ch, stride, kernel_size=3):
        super().__init__()
        self.ch = ch
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 训练分支 - 修复：1x1分支也使用深度可分离卷积
        self.rbr_identity = nn.BatchNorm2D(ch) if stride == 1 else None
        self.rbr_1x1 = self._conv_bn(1, ch, ch, stride, groups=ch)  # 添加groups=ch
        self.rbr_3x3 = self._conv_bn(3, ch, ch, stride, groups=ch)

        self.rbr_reparam = None  # 推理融合后使用

    def _conv_bn(self, k, in_c, out_c, stride=1, groups=1):
        return nn.Sequential(
            Conv2D(in_c, out_c, k, stride, k // 2, groups=groups, bias_attr=False),
            BatchNorm(out_c),
        )

    def forward(self, x):
        # 优化：推理时直接使用融合后的卷积
        if self.rbr_reparam is not None:
            return self.rbr_reparam(x)
        out = 0
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        out += self.rbr_1x1(x)
        out += self.rbr_3x3(x)
        return out

    def switch_to_deploy(self):
        if self.rbr_reparam is not None:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2D(
            self.ch, self.ch, self.kernel_size,
            stride=self.stride, padding=self.padding, groups=self.ch, bias_attr=True
        )
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        # 删除训练分支释放内存
        del self.rbr_3x3, self.rbr_1x1, self.rbr_identity

    def _get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_conv_bn(self.rbr_3x3)
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1)
        kid, bid = self._fuse_conv_bn(self.rbr_identity) if self.rbr_identity else (0, 0)
        k1_padded = paddle.zeros_like(k3)
        k1_padded[:, :, 1:2, 1:2] = k1
        return k3 + k1_padded + kid, b3 + b1 + bid

    def _fuse_conv_bn(self, branch):
        if isinstance(branch, nn.Sequential):
            conv, bn = branch[0], branch[1]
        else:
            conv, bn = None, branch
        if conv is None:
            # 修复：为深度可分离卷积创建正确的identity权重
            # 深度可分离卷积的identity是每个通道的1x1核为1，形状为[ch, 1, 1, 1]
            k = paddle.ones([self.ch, 1, 1, 1], dtype='float32')
        else:
            k = conv.weight
        var = bn._variance
        w = bn.weight
        b = bn.bias
        eps = bn._epsilon
        std = (var + eps).sqrt()
        return k * (w / std).reshape([-1, 1, 1, 1]), b - bn._mean * w / std


# ---------------- 优化的SE模块 ----------------
class SEModule(nn.Layer):
    def __init__(self, ch, reduction=8):  # 优化：增加reduction减少计算
        super().__init__()
        self.pool = AdaptiveAvgPool2D(1)
        mid = max(8, ch // reduction)
        self.fc = nn.Sequential(
            Conv2D(ch, mid, 1), nn.ReLU6(),  # 优化：使用ReLU6替代复杂激活
            Conv2D(mid, ch, 1), nn.Sigmoid()  # 优化：使用Sigmoid替代Hardsigmoid
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


# ---------------- 优化的RepViT Block ----------------
class RepViTBlock(nn.Layer):
    def __init__(self, in_c, out_c, stride, use_se, act="relu"):  # 优化：默认使用relu6
        super().__init__()
        self.stride = stride
        hidden = int(out_c * 1.5)  # 优化：减少expansion ratio从2到1.5

        # 扩展层
        self.expand = nn.Sequential(
            Conv2D(in_c, hidden, 1, bias_attr=False),
            BatchNorm(hidden),
        )

        self.token_mixer = RepDW(hidden, stride)
        self.se = SEModule(hidden) if use_se else None
        self.channel_mixer = nn.Sequential(
            Conv2D(hidden, out_c, 1, bias_attr=False),
            BatchNorm(out_c),
        )
        self.shortcut = (
            nn.Sequential() if stride == 1 and in_c == out_c else None
        )
        self.act = build_act(act)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.act(x)
        x = self.token_mixer(x)
        if self.se:
            x = self.se(x)
        x = self.channel_mixer(x)
        if self.shortcut:
            x = x + identity
        return x


# ---------------- 优化的下采样 ----------------
class DownSample(nn.Layer):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = nn.Sequential(
            Conv2D(in_c, in_c, 3, stride=2, padding=1, groups=in_c, bias_attr=False),
            BatchNorm(in_c)
        )
        self.pw = nn.Sequential(
            Conv2D(in_c, out_c, 1, bias_attr=False),
            BatchNorm(out_c),
            nn.ReLU6(),
        )
        # 优化：简化FFN结构
        self.ffn = nn.Sequential(
            Conv2D(out_c, out_c, 1, bias_attr=False),  # 优化：移除expansion
            BatchNorm(out_c),
            nn.ReLU6(),
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x + self.ffn(x)


# ---------------- 优化的RepViT Backbone ----------------
@manager.BACKBONES.add_component
class RepViTBackboneV6(nn.Layer):
    def __init__(self, arch="m1.0", pretrained=None):
        super().__init__()
        cfg = {
            "m0.75": dict(stem=[20, 40], stage=[40, 80, 160, 320]),  # 新增轻量配置
            "m0.9": dict(stem=[24, 48], stage=[48, 96, 192, 384]),
            "m1.0": dict(stem=[24, 48], stage=[56, 112, 224, 448]),
            "m1.1": dict(stem=[32, 64], stage=[64, 128, 256, 512]),
            "m1.5": dict(stem=[40, 80], stage=[80, 160, 320, 640]),
            "m2.3": dict(stem=[48, 96], stage=[96, 192, 384, 768]),
        }[arch]
        stem_ch, stage_ch = cfg["stem"], cfg["stage"]
        stage_num = [2, 2, 14, 2]

        # stem - 优化：使用ReLU6
        self.stem = nn.Sequential(
            Conv2D(3, stem_ch[0], 3, stride=2, padding=1, bias_attr=False),
            BatchNorm(stem_ch[0]),
            nn.ReLU6(),
            Conv2D(stem_ch[0], stem_ch[1], 3, stride=2, padding=1, bias_attr=False),
            BatchNorm(stem_ch[1]),
            nn.ReLU6(),
        )

        # stages
        self.stages = nn.LayerList()
        in_c = stem_ch[1]
        for i, (ch, num) in enumerate(zip(stage_ch, stage_num)):
            if i != 0:
                self.stages.append(DownSample(in_c, ch))
                in_c = ch
            for j in range(num):
                stride = 2 if j == 0 and i != 0 else 1
                # 优化：减少SE模块使用频率，只在关键位置使用
                use_se = (i >= 2 and j % 4 == 0)  # 只在后两个stage的每4个block使用SE
                self.stages.append(RepViTBlock(in_c, ch, stride, use_se, act="relu"))
                in_c = ch

        # 动态计算stage输出索引
        self.stage_out_idx = []
        idx = 0
        for i, num in enumerate(stage_num):
            if i != 0:
                idx += 1
            idx += num
            self.stage_out_idx.append(idx - 1)

        self.feat_channels = stage_ch
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    nn.initializer.KaimingNormal()(m.weight)
                    if m.bias is not None:
                        nn.initializer.Constant(0)(m.bias)
                elif isinstance(m, nn.BatchNorm2D):
                    nn.initializer.Constant(1)(m.weight)
                    nn.initializer.Constant(0)(m.bias)

    def switch_to_deploy(self):
        """融合所有 RepDW - 推理时必须调用"""
        for m in self.sublayers():
            if isinstance(m, RepDW):
                m.switch_to_deploy()

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.stage_out_idx:
                outs.append(x)
        return outs