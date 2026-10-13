# configs/paper — 论文表格 ↔ 配置对照

把论文（IEEE TMC《DigiBoard: Occlusion-Free Whiteboard Digitization at the Edge》）两张实验表的每一行，
固定成一个**可直接使用**的配置。权威出处是桌面上的 `消融实验.md`（逐小节的 `yml` + `train/val`
命令 + 实测值），本目录只是把它按「一个表格行一个文件」重新组织。

数据集：`PP-HumanSeg14K`（`data/portrait14k`），`num_classes: 2`，输入 256×144。

---

## TABLE III · 性能对比（6 行）

| 论文行 | 本目录配置 | 实测 mIoU | 出处 / 备注 |
|---|---|---|---|
| DeepLabV3p | `table3_deeplabv3p.yml` | 0.8806 | `消融实验.md` → `Deeplabv3p` |
| BiSeNetV2 | `table3_bisenetv2.yml` | 0.8882 | → `BisnetV2` |
| PP-HumanSegV2-Lite | `table3_pp_humansegv2_lite.yml` | 0.8631 | `workerlog.0` 2026-01-23 |
| PP-LiteSeg | `table3_pp_liteseg.yml` | 0.8294 | → `PP_LiteSeg` |
| EfficientFormerV2-Seg | `table3_efficientformerv2_seg.yml` | 0.8932 | `workerlog.0` run 84 |
| **Ours** | `table3_ours.yml` | **0.9096** | `消融实验.md` → `1.PPM和ASPP拼接` |

## TABLE VI · 分割模型消融（7 行）

| 论文行 | 配置（`configs/quick_start/`） | 实测 mIoU | 出处 |
|---|---|---|---|
| RepViT-Baseline | `repvit_v7.yml` | 0.8848 | `结果.md` → Repvit version7 |
| + GAM | `repvit_v8_20.yml` | 0.8893 | `消融实验.md` → `只有GAM` |
| + GAM + ASPP | `repvit_v8_15.yml` | 0.9044 | → `5.只有ASPP` |
| + GAM + PPM(1236) + FM | `repvit_v8_16.yml` | 0.9023 | → `只有PPM` |
| + GAM + PPM(1236) + ASPP + FM | `repvit_v8_18.yml` | 0.8909 | → `RepViT+GAM+PPM(1236)+ASPP+FM` |
| + GAM + PPM(124) + ASPP + FM | `repvit_v8_9.yml` | 0.8941 | → `3.PPM不进行逐元素相加和ASPP以及自身进行拼接` |
| **+ GAM + PPM(124) + ASPP + MSF** | `repvit_v8_7.yml` | **0.9096** | → `1.PPM和ASPP拼接` |

---

## 已知缺口（复现前必读）

**2026-10-13 更新：** 4 个消融 run 的权重（`repvit_v8_7 / _9 / _15 / _16`）已回迁并归档至仓库根目录
`weights/<run>/`（含 `model.pdparams` + `model.pdstates`），`repvit_v8_16.yml` 亦已找回原件。val 示例：

```bash
python tools/val.py --config configs/quick_start/repvit_v8_16.yml \
  --model_path weights/repvit_v8_16/model.pdparams
```

对比侧 `DeepLabV3p`、`PP-LiteSeg` 的权重仍无。结构核验详情见
`docs/experiment-slimming-plan.html` §6.2。

1. `repvit_v8_16.yml` 已从服务器回迁（原件）；只剩 `bisenet_16k.yml` 无法逐字还原（`BisnetV2_16k`
   小节只有 train 命令、没有 yml 段），需要时按同系列模板重建——那是「重建」而非「还原」。
2. **`MobileSegV9` 已安装**（PPM-only 版，与 `_16` 权重结构精确一致，2026-10-13）。仍缺的类：
   `MobileSegV6`（`repvit_v8_11.yml`）、`MobileSegV7`（`repvit_v8_12.yml`）、骨干
   `RepViTBackboneFinal`（`repvit_final.yml`），以及 **`repvit_v8_9` 的原类实现**
   （其权重含 `input_conv` 与 5 分支融合 `(128, 640)`，仓库现有类均不匹配；源码在 `消融实验.md`
   `3.PPM不进行逐元素相加和ASPP以及自身进行拼接` 小节）。
3. `MobileSegV4 / V5 / V11` 均已注册（v1.3）。注意：`repvit_v8_15.yml` 声明 `type: MobileSegV9`，
   但其权重为 **ASPP-only 结构（= `MobileSegV4` 的 `ASPPOnlyContextModule`）**，直接加载会键不匹配；
   跑该行时把 `type` 改指 `MobileSegV4`，或取回「`5.只有ASPP`」原类补建。
4. **口径冲突已定案（2026-10-13）**：`repvit_v8_7` 权重融合层 `(128, 768)` = 128×6 →
   `MobileSegV2` 实现（PPM 三支与 ASPP 三支全部拼接）；`结果.md` 的「V3」记录不成立
   （V3 逐元素相加后 fusion 应为 512）。
