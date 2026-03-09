#                  损失曲线图

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# log_files = {
#     'CLOSE': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\constructs\CLOSE_loss_log.txt',
#     'LESION': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\constructs\LESION_loss_log.txt',
#     'FullSample': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\constructs\FullSample_loss_log.txt',
#     'dataset2': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\constructs\NBI_256_loss_log.txt',
# }


log_files = {
    'CLOSE': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\FFPE++\CLOSE_loss_log.txt',
    'LESION': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\FFPE++\LESION_loss_log.txt',
    'FullSample': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\FFPE++\FullSample_loss_log.txt',
    'dataset2': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\FFPE++\NBI_256_loss_log.txt',
}


output_dir = Path(r"E:\wx\dataset_wx\ScienceData\Code\txtfiles/plots")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "G_GAN_loss_per_epoch.png"



def parse_g_gan_log(file_path):
    """
    从 loss_log.txt 提取 epoch 和 G_GAN 值，返回 DataFrame
    """
    epochs = []
    g_gan_values = []

    pattern = r'\(epoch: (\d+), iters: \d+.*?\) G_GAN: ([\d\.]+)'

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                g_gan = float(match.group(2))
                epochs.append(epoch)
                g_gan_values.append(g_gan)

    if not epochs:
        print(f"Warning: No G_GAN found in {file_path}")
        return pd.DataFrame()

    df = pd.DataFrame({'epoch': epochs, 'G_GAN': g_gan_values})


    df_avg = df.groupby('epoch')['G_GAN'].mean().reset_index()
    return df_avg



plt.figure(figsize=(10, 6), dpi=150)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
labels = list(log_files.keys())

all_data = {}

for i, (name, path) in enumerate(log_files.items()):
    df = parse_g_gan_log(path)
    if df.empty:
        continue

    all_data[name] = df
    df = df[df['epoch'] <= 200]

    plt.plot(df['epoch'], df['G_GAN'],
             color=colors[i], linewidth=2,
             label=f'{name} (avg G_GAN)')


plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Average G_GAN Loss', fontsize=14)
plt.title('Generator Adversarial Loss (G_GAN) per Epoch\nAcross Four Datasets', fontsize=16, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()


plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"Figure saved to: {output_path}")

for name, df in all_data.items():
    print(f"{name}: {len(df)} epochs recorded, final avg G_GAN = {df['G_GAN'].iloc[-1]:.4f}")




