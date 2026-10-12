# paddleseg/models/backbones/repvit_seg.py
# 完整 PaddleSeg backbone 组件，语义分割专用
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
    return {"hardswish": nn.Hardswish(), "relu": nn.ReLU()}[act]


# ---------------- 结构重参数化 DW ----------------
class RepDW(nn.Layer):
    """训练多分支，推理融合为单 3x3"""

    def __init__(self, ch, stride, kernel_size=3):
        super().__init__()
        self.ch = ch
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 训练分支
        self.rbr_identity = nn.BatchNorm2D(ch) if stride == 1 else None
        self.rbr_1x1 = self._conv_bn(1, ch, ch, stride)
        self.rbr_3x3 = self._conv_bn(3, ch, ch, stride, groups=ch)

        self.rbr_reparam = None  # 推理融合后使用

    def _conv_bn(self, k, in_c, out_c, stride=1, groups=1):
        return nn.Sequential(
            Conv2D(in_c, out_c, k, stride, k // 2, groups=groups, bias_attr=False),
            BatchNorm(out_c),
        )

    def forward(self, x):
        if self.rbr_reparam is not None:
            return self.rbr_reparam(x)
        out = 0
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        out += self.rbr_1x1(x)
        out += self.rbr_3x3(x)
        return out

    # 以下融合代码与之前相同，不再赘述
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
            k = paddle.eye(self.ch, self.ch).reshape([self.ch, self.ch, 1, 1])
        else:
            k = conv.weight
        var = bn._variance
        w = bn.weight
        b = bn.bias
        eps = bn._epsilon
        std = (var + eps).sqrt()
        return k * (w / std).reshape([-1, 1, 1, 1]), b - bn._mean * w / std


# ---------------- SE ----------------
class SEModule(nn.Layer):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.pool = AdaptiveAvgPool2D(1)
        mid = max(8, ch // reduction)
        self.fc = nn.Sequential(
            Conv2D(ch, mid, 1), nn.ReLU(), Conv2D(mid, ch, 1), nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


# ---------------- RepViT Block ----------------
class RepViTBlock(nn.Layer):
    def __init__(self, in_c, out_c, stride, use_se, act="hardswish"):
        super().__init__()
        self.stride = stride
        hidden = out_c * 2  # expansion=2

        # 添加扩展层
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
        x = self.act(x)
        x = self.expand(x)  # 扩展到hidden通道数
        x = self.token_mixer(x)
        if self.se:
            x = self.se(x)
        x = self.channel_mixer(x)
        if self.shortcut:
            x = x + identity
        return x


# ---------------- 下采样 ----------------
class DownSample(nn.Layer):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = Conv2D(in_c, in_c, 3, stride=2, padding=1, groups=in_c, bias_attr=False)
        self.pw = nn.Sequential(
            Conv2D(in_c, out_c, 1, bias_attr=False),
            BatchNorm(out_c),
            nn.Hardswish(),
        )
        self.ffn = nn.Sequential(
            Conv2D(out_c, out_c * 2, 1),
            nn.Hardswish(),
            Conv2D(out_c * 2, out_c, 1),
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x + self.ffn(x)


# ---------------- RepViT Backbone ----------------
@manager.BACKBONES.add_component
class RepViTBackbone(nn.Layer):
    """
    RepViT backbone for semantic segmentation.
    Returns 4-level features [1/4, 1/8, 1/16, 1/32]
    """

    def __init__(self, arch="m1.0", pretrained=None):
        super().__init__()
        cfg = {
            "m0.9": dict(stem=[24, 48], stage=[48, 96, 192, 384]),
            "m1.0": dict(stem=[24, 48], stage=[56, 112, 224, 448]),
            "m1.1": dict(stem=[32, 64], stage=[64, 128, 256, 512]),
            "m1.5": dict(stem=[40, 80], stage=[80, 160, 320, 640]),
            "m2.3": dict(stem=[48, 96], stage=[96, 192, 384, 768]),
        }[arch]
        stem_ch, stage_ch = cfg["stem"], cfg["stage"]
        # stage block nums 2:2:14:2
        stage_num = [2, 2, 14, 2]

        # stem
        self.stem = nn.Sequential(
            Conv2D(3, stem_ch[0], 3, stride=2, padding=1, bias_attr=False),
            BatchNorm(stem_ch[0]),
            nn.Hardswish(),
            Conv2D(stem_ch[0], stem_ch[1], 3, stride=2, padding=1, bias_attr=False),
            BatchNorm(stem_ch[1]),
            nn.Hardswish(),
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
                use_se = (j % 2 == 0)  # 交叉 SE
                self.stages.append(RepViTBlock(in_c, ch, stride, use_se))
                in_c = ch

        self.feat_channels = stage_ch  # [56,112,224,448] for m1.0

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            # 官方推荐初始化
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    nn.initializer.KaimingNormal()(m.weight)
                    if m.bias is not None:
                        nn.initializer.Constant(0)(m.bias)
                elif isinstance(m, nn.BatchNorm2D):
                    nn.initializer.Constant(1)(m.weight)
                    nn.initializer.Constant(0)(m.bias)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        # 手工索引 4 个阶段输出位置
        stage_out_idx = [0, 3, 17, 31]  # 2+2+14+2=30，对应下采样后
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in stage_out_idx:
                outs.append(x)
        return outs

    def switch_to_deploy(self):
        """融合所有 RepDW"""
        for m in self.sublayers():
            if isinstance(m, RepDW):
                m.switch_to_deploy()