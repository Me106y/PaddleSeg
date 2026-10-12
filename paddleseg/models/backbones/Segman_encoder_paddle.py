import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, epsilon=1e-6)
    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])
        x = super().forward(x)
        x = x.transpose([0, 3, 1, 2])
        return x

class Stem(nn.Layer):
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.conv1 = nn.Conv2D(in_chans, embed_dim // 2, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2D(embed_dim // 2)
        self.conv2 = nn.Conv2D(embed_dim // 2, embed_dim // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2D(embed_dim // 2)
        self.conv3 = nn.Conv2D(embed_dim // 2, embed_dim, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2D(embed_dim)
        self.conv4 = nn.Conv2D(embed_dim, embed_dim, 3, padding=1)
        self.bn4 = nn.BatchNorm2D(embed_dim)
    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x

class BasicBlock(nn.Layer):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.norm1 = LayerNorm2d(embed_dim)
        self.conv1 = nn.Conv2D(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.norm2 = LayerNorm2d(embed_dim)
        self.mlp = nn.Sequential(
            nn.Conv2D(embed_dim, ffn_dim, 1),
            nn.GELU(),
            nn.Conv2D(ffn_dim, embed_dim, 1)
        )
    def forward(self, x):
        x = x + self.conv1(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BasicLayer(nn.Layer):
    def __init__(self, embed_dim, depth, ffn_dim):
        super().__init__()
        self.blocks = nn.LayerList([
            BasicBlock(embed_dim, ffn_dim) for _ in range(depth)
        ])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class SegMANEncoderPaddle(nn.Layer):
    def __init__(self,
                 image_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 depths=[2, 2, 6, 2],
                 mlp_ratios=[4, 4, 4, 4],
                 projection=1024):
        super().__init__()
        self.patch_embed = Stem(in_chans=in_chans, embed_dim=embed_dims[0])
        self.layers = nn.LayerList()
        for i in range(len(depths)):
            layer = BasicLayer(
                embed_dim=embed_dims[i],
                depth=depths[i],
                ffn_dim=int(mlp_ratios[i] * embed_dims[i])
            )
            self.layers.append(layer)
            if i < len(depths) - 1:
                self.layers.append(nn.Sequential(
                    nn.Conv2D(embed_dims[i], embed_dims[i+1], 3, stride=2, padding=1),
                    nn.BatchNorm2D(embed_dims[i+1])
                ))
        self.classifier = nn.Sequential(
            nn.Conv2D(embed_dims[-1], projection, 1),
            nn.BatchNorm2D(projection),
            nn.Silu(),
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(projection, num_classes, 1) if num_classes > 0 else nn.Identity()
        )
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        x = x.flatten(1)
        return x