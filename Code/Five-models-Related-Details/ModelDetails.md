CycleGAN

代码仓库：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

CUT

代码仓库：https://github.com/taesungp/contrastive-unpaired-translation

AttentionGAN

代码仓库：https://github.com/Ha0Tang/AttentionGAN

ConStructs

代码仓库：https://gitlab.com/nct_tso_public/constructs.

FFPE++

代码仓库： https://github.com/DeepMIALab/FFPEPlus.

所有模型均采用以下统一训练设置：

- 框架：PyTorch 1.12.1

- 硬件：NVIDIA Quadro RTX 6000

- Batch size：1

- 初始学习率：0.0002

- 总训练轮数：200

- 学习率衰减策略：100 个 epoch 后采用线性衰减

- 随机种子：实验中使用了多个固定随机种子（包括 0、42、123 、456、789），并报告了多次运行的平均值 ± 标准差，以验证结果稳定性。

- 检查点选择规则：采用验证集损失最低的检查点进行最终评估。

  