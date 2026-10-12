import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager
from paddleseg.models import layers


class OffsetLearning(nn.Layer):
    """
    修正后的 Offset Learning 模块
    """

    def __init__(self, num_classes, embed_dims, init_std=0.02):
        super(OffsetLearning, self).__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims

        # 类别表征参数 (Class Representation)
        self.cls_repr = self.create_parameter(
            shape=[1, num_classes, embed_dims],
            default_initializer=nn.initializer.TruncatedNormal(std=init_std)
        )

        # 使用 Paddle 推荐的 weight_attr 方式初始化
        w_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=init_std))

        self.mask_norm = nn.LayerNorm(num_classes)
        self.cls_offset_proj = nn.Linear(embed_dims, embed_dims, weight_attr=w_attr, bias_attr=False)
        self.feat_offset_proj = nn.Linear(embed_dims, embed_dims, weight_attr=w_attr, bias_attr=False)

    def forward(self, x):
        b, c, h, w = x.shape
        # x: [B, C, H, W] -> img_feat: [B, HW, C]
        img_feat = x.flatten(2).transpose([0, 2, 1])
        cls_repr = self.cls_repr.expand([b, -1, -1])  # [B, K, C]

        # 计算耦合注意力 (Coupled Attention)
        # 注意：Paddle 中 matmul 处理 [B, K, C] @ [B, C, HW] 没问题
        coupled_attn = paddle.matmul(cls_repr, img_feat.transpose([0, 2, 1]))

        # Class Offset Learning
        cls_attn = F.softmax(coupled_attn, axis=-1)
        cls_offset = self.cls_offset_proj(paddle.matmul(cls_attn, img_feat))
        aligned_cls_repr = cls_repr + cls_offset

        # Feature Offset Learning
        pos_attn = F.softmax(coupled_attn, axis=1)
        feat_offset = self.feat_offset_proj(paddle.matmul(pos_attn.transpose([0, 2, 1]), cls_repr))
        aligned_img_feat = img_feat + feat_offset

        # 生成 Mask
        masks = paddle.matmul(aligned_img_feat, aligned_cls_repr.transpose([0, 2, 1]))
        masks = self.mask_norm(masks)

        # 还原维度 [B, K, H, W]
        masks = masks.transpose([0, 2, 1]).reshape([b, self.num_classes, h, w])
        return masks


class OffSegHead(nn.Layer):
    def __init__(self, num_classes, in_channels, new_channels, channels=256):
        super().__init__()
        self.pre_convs = nn.LayerList()
        for in_c, out_c in zip(in_channels, new_channels):
            self.pre_convs.append(layers.ConvBNReLU(in_c, out_c, 1))

        self.align_conv = layers.ConvBNReLU(sum(new_channels), channels, 1)
        self.offset_learning = OffsetLearning(num_classes, channels)

    def forward(self, inputs):
        refined_feats = []
        for i, x in enumerate(inputs):
            refined_feats.append(self.pre_convs[i](x))

        target_size = refined_feats[0].shape[2:]
        out_feats = [refined_feats[0]]
        for i in range(1, len(refined_feats)):
            # 修正：PaddleSeg 中通常需要对齐像素中心
            out_feats.append(F.interpolate(refined_feats[i], target_size, mode='bilinear', align_corners=False))

        combined = paddle.concat(out_feats, axis=1)
        output = self.align_conv(combined)
        logit = self.offset_learning(output)
        return logit


@manager.MODELS.add_component
class OffSegTiny(nn.Layer):
    def __init__(self, num_classes, backbone, backbone_indices=[0, 1, 2, 3], head_channels=256, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices

        # 动态获取通道，避免硬编码报错
        if hasattr(self.backbone, 'feat_channels'):
            in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        else:
            in_channels = [32, 48, 120, 224]

        new_channels = [32, 32, 64, 128][:len(backbone_indices)]

        self.decode_head = OffSegHead(
            num_classes=num_classes,
            in_channels=in_channels,
            new_channels=new_channels,
            channels=head_channels
        )

        # 定义 3 个分割头以匹配 offseg.yml 中的 coef: [1, 1, 1]
        self.seg_heads = nn.LayerList()
        for _ in range(3):
            self.seg_heads.append(
                nn.Conv2D(num_classes, num_classes, kernel_size=1)
            )

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            layers.load_entire_model(self, self.pretrained)

    def forward(self, x):
        input_shape = x.shape[2:]
        feats = self.backbone(x)
        select_feats = [feats[i] for i in self.backbone_indices]

        # 得到 decode_head 的输出
        logit = self.decode_head(select_feats)

        if self.training:
            # 修正变量名拼写：统一使用 logit_list
            logit_list = []
            for seg_head in self.seg_heads:
                l = seg_head(logit)
                l = F.interpolate(l, input_shape, mode='bilinear', align_corners=False)
                logit_list.append(l)
            return logit_list  # 确保这里返回的是 logit_list 而不是 log_list
        else:
            # 验证/预测时返回主输出
            logit = F.interpolate(logit, input_shape, mode='bilinear', align_corners=False)
            return [logit]