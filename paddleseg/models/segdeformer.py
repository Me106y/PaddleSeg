import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


# --- 1. 内部上下文挖掘 (Internal Context Mining, ICM) ---
class InternalContextMining(nn.Layer):
    """
    单头自注意力机制，用于捕捉图像内部的全局上下文。
    """

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dim, num_heads=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose([0, 2, 1])  # [B, N, C]

        # Self-Attention
        out = self.attn(x_flat, x_flat, x_flat)
        out = self.norm(x_flat + out)

        return out.transpose([0, 2, 1]).reshape([b, c, h, w])


# --- 2. 外部上下文挖掘 (External Context Mining, ECM) ---
class ExternalContextMining(nn.Layer):
    """
    实现论文中的 ECM 逻辑：
    1. Class Token 解耦合 (2C -> C_info + C_class)
    2. 第一阶段：Image-to-Token (生成中间特征并用于辅助分类)
    3. 第二阶段：Token-to-Image (将外部信息写回特征图)
    """

    def __init__(self, dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim

        # 外部可学习 Token [1, num_classes, 2*dim]
        # 论文中提到解耦合，每个 Token 包含 info 和 class 两部分
        self.cls_token = self.create_parameter(
            shape=[1, num_classes, 2 * dim],
            default_initializer=nn.initializer.TruncatedNormal(std=.02))

        # 两个阶段的交叉注意力
        self.cross_attn1 = nn.MultiHeadAttention(dim, num_heads=1)
        self.cross_attn2 = nn.MultiHeadAttention(dim, num_heads=1)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose([0, 2, 1])  # [B, N, C]

        # 1. Token 解耦合
        tokens = self.cls_token.expand([b, -1, -1])  # [B, K, 2C]
        token_info, token_class = paddle.split(tokens, num_or_sections=2, axis=-1)

        # 2. 第一阶段: M_exter (Step 1)
        # Query: x_flat, Key/Value: token_info
        # 论文中此处生成的注意力图用于辅助损失
        x_mid = self.cross_attn1(query=x_flat, key=token_info, value=token_info)
        x_mid = self.norm1(x_flat + x_mid)

        # 辅助预测：利用解耦合出的 token_class 与中间特征计算
        # 模拟论文中的 L_attn 逻辑
        out_cls_mid = paddle.matmul(x_mid, token_class, transpose_y=True)  # [B, N, K]
        out_cls_mid = out_cls_mid.transpose([0, 2, 1]).reshape([b, self.num_classes, h, w])

        # 3. 第二阶段: M_exter (Step 2)
        # 利用第一阶段的输出写回特征
        y_exter = self.cross_attn2(query=x_flat, key=x_mid, value=x_mid)
        y_exter = self.norm2(x_flat + y_exter)

        y_exter = y_exter.transpose([0, 2, 1]).reshape([b, c, h, w])

        return y_exter, out_cls_mid


# --- 3. SegDeformer 主模型 ---
@manager.MODELS.add_component
class SegDeformer(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.pretrained = pretrained

        c1_in, c2_in, c3_in, c4_in = self.backbone.feat_channels

        # MLP/Conv 层对齐通道 (类似 SegFormer)
        self.convs = nn.LayerList([
            layers.ConvBNReLU(c1_in, embedding_dim, 1),
            layers.ConvBNReLU(c2_in, embedding_dim, 1),
            layers.ConvBNReLU(c3_in, embedding_dim, 1),
            layers.ConvBNReLU(c4_in, embedding_dim, 1)
        ])

        # 特征融合
        self.fusion_conv = layers.ConvBNReLU(embedding_dim * 4, embedding_dim, 1)

        # SegDeformer 核心组件
        self.icm = InternalContextMining(embedding_dim)
        self.ecm = ExternalContextMining(embedding_dim, num_classes)

        # 最终分类头
        self.linear_pred = nn.Conv2D(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2D(0.1)

        # 新增辅助预测层, 解决 loss 数量不匹配问题
        self.aux_head = nn.Conv2D(embedding_dim, num_classes, kernel_size=1)

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        # 记录原始输入尺寸，用于最后上采样
        h_in, w_in = x.shape[2:]

        # 1. Backbone 特征提取
        feats = self.backbone(x)

        # 2. 多尺度特征对齐 (统一到最高分辨率特征图的尺寸)
        target_size = feats[0].shape[2:]
        out_feats = []
        for i, f in enumerate(feats):
            f = self.convs[i](f)
            if i > 0:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=self.align_corners)
            out_feats.append(f)

        # 3. 特征聚合 (融合多层级信息)
        fused = self.fusion_conv(paddle.concat(out_feats, axis=1))

        # 4. SegDeformer 核心逻辑
        # 内部上下文挖掘 (ICM)
        y_inter = self.icm(fused)
        # 外部上下文挖掘 (ECM) - 返回增强特征和辅助预测分支
        y_exter, out_cls_mid = self.ecm(fused)

        # 5. 最终特征融合 (根据论文：原特征 + 内部增强 + 外部增强)
        f_aug = fused + y_inter + y_exter

        # 6. 预测头处理
        logit = self.dropout(f_aug)
        logit = self.linear_pred(logit)

        # 7. 辅助预测头 (可选，如果你需要额外的辅助 Loss)
        aux_logit = self.aux_head(fused)

        # 8. 恢复到原始图像尺寸并返回
        if self.training:
            final_out = F.interpolate(logit, size=(h_in, w_in), mode='bilinear', align_corners=self.align_corners)
            out_cls_mid_resized = F.interpolate(out_cls_mid, size=(h_in, w_in), mode='bilinear',
                                                align_corners=self.align_corners)
            aux_out_resized = F.interpolate(aux_logit, size=(h_in, w_in), mode='bilinear',
                                            align_corners=self.align_corners)
            # 返回列表，顺序需与配置文件中的 Loss 对应
            return [final_out, out_cls_mid_resized, aux_out_resized]

        # 预测/评估模式
        final_out = F.interpolate(logit, size=(h_in, w_in), mode='bilinear', align_corners=self.align_corners)
        return [final_out]