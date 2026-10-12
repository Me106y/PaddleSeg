import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager

class ASPPOnlyContextModule(nn.Layer):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 aspp_dilations=(6, 12, 18),
                 align_corners=False):
        super().__init__()
        self.aspp_convs = nn.LayerList([
            layers.SeparableConvBNReLU(
                in_channels, inter_channels, kernel_size=3,
                dilation=rate, bias_attr=False
            ) for rate in aspp_dilations
        ])
        self.fusion_conv = layers.SeparableConvBNReLU(
            in_channels=inter_channels * len(aspp_dilations),
            out_channels=inter_channels,
            kernel_size=3,
            bias_attr=False
        )
        self.conv_out = layers.SeparableConvBNReLU(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=3,
            bias_attr=False
        )
        self.align_corners = align_corners

    def forward(self, input):
        aspp_features = [aspp_conv(input) for aspp_conv in self.aspp_convs]
        fused_features = paddle.concat(aspp_features, axis=1)
        fused_features = self.fusion_conv(fused_features)
        out = self.conv_out(fused_features)
        return out

class MobileSegHead(nn.Layer):
    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode, use_last_fuse, use_enhanced_cm=True):
        super().__init__()
        self.cm = ASPPOnlyContextModule(
            backbone_out_chs[-1], cm_out_ch, cm_out_ch, aspp_dilations=[6, 12, 18]
        )
        assert hasattr(layers, arm_type), "Not support arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)
        self.arm_list = nn.LayerList()
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)
        self.use_last_fuse = use_last_fuse
        if self.use_last_fuse:
            self.fuse_convs = nn.LayerList()
            for i in range(1, len(arm_out_chs)):
                conv = layers.SeparableConvBNReLU(arm_out_chs[i], arm_out_chs[0], kernel_size=3, bias_attr=False)
                self.fuse_convs.append(conv)
            self.last_conv = layers.SeparableConvBNReLU(len(arm_out_chs) * arm_out_chs[0], arm_out_chs[0], kernel_size=3, bias_attr=False)
    def forward(self, in_feat_list):
        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []
        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)
        if self.use_last_fuse:
            x_list = [out_feat_list[0]]
            size = out_feat_list[0].shape[2:]
            for i, (x, conv) in enumerate(zip(out_feat_list[1:], self.fuse_convs)):
                x = conv(x)
                x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
                x_list.append(x)
            x = paddle.concat(x_list, axis=1)
            x = self.last_conv(x)
            out_feat_list[0] = x
        return out_feat_list

class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = layers.SeparableConvBNReLU(in_chan, mid_chan, kernel_size=3, bias_attr=False)
        self.conv_out = nn.Conv2D(mid_chan, n_classes, kernel_size=1, bias_attr=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

@manager.MODELS.add_component
class MobileSegV4(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[1, 2, 3],
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=64,
                 arm_type='UAFMMobile',
                 arm_out_chs=[32, 48, 64],
                 seg_head_inter_chs=[32, 32, 32],
                 resize_mode='bilinear',
                 use_last_fuse=True,
                 use_enhanced_cm=True,
                 pretrained=None):
        super().__init__()
        assert hasattr(backbone, 'feat_channels'), "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), f"The length of input backbone_indices ({len(backbone_indices)}) should not be greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), f"The max value ({max(backbone_indices)}) of backbone_indices should be less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone
        assert len(backbone_indices) >= 1, "The lenght of backbone_indices should not be lesser than 1"
        self.backbone_indices = backbone_indices
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of arm_out_chs and backbone_indices should be equal"
        self.ppseg_head = MobileSegHead(backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch, arm_type, resize_mode, use_last_fuse, use_enhanced_cm)
        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.LayerList()
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))
        self.pretrained = pretrained
        self.init_weight()
    def forward(self, x):
        x_hw = x.shape[2:]
        feats_backbone = self.backbone(x)
        assert len(feats_backbone) >= len(self.backbone_indices), f"The nums of backbone feats ({len(feats_backbone)}) should be greater or equal than the nums of backbone_indices ({len(self.backbone_indices)})"
        feats_selected = [feats_backbone[i] for i in self.backbone_indices]
        feats_head = self.ppseg_head(feats_selected)
        if self.training:
            logit_list = []
            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)
            logit_list = [F.interpolate(x, x_hw, mode='bilinear', align_corners=False) for x in logit_list]
        else:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]
        return logit_list
    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)