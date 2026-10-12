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

1. **有 2 个配置无法逐字还原**（两份记录里都没有它们的 yml 内容）：
   `repvit_v8_16.yml`（`消融实验.md` 中该 yml 块为空）与 `bisenet_16k.yml`（`BisnetV2_16k` 小节
   只有 train 命令、没有 yml 段）。前者在 `TABLE VI` 中以 `只有PPM` 的实测值（0.9023 / 11.48 ms）
   留档；两者的超参与同系列模板一致，需要时可按模板重建——但那是「重建」而非「还原」。
2. 部分消融配置引用的**模型类在当前仓库中不存在**，因此这些配置「已还原但暂不可运行」：
   `MobileSegV6`（`repvit_v8_11.yml`）、`MobileSegV7`（`repvit_v8_12.yml`）、`MobileSegV9`
   （`repvit_v8_15.yml`、`repvit_v8_16.yml`），以及骨干 `RepViTBackboneFinal`（`repvit_final.yml`）。
   这些类的源码记录在 `消融实验.md` 各小节的代码块中，需要时可据此补建。
3. `MobileSegV4`（`repvit_v8_9.yml`）与 `MobileSegV5`（`repvit_v8_10.yml`）的**文件存在但未注册**，
   需要在 `paddleseg/models/__init__.py` 里 import 后才能加载。
4. **口径冲突待你裁定**：`repvit_v8_7.yml` 在 `消融实验.md` 中声明 `type: MobileSegV2`，而 `结果.md`
   记录同一次训练用的是 `MobileSegV3`。指标（0.9096 / 10.64 ms）两边一致，但模型类名不一致。
