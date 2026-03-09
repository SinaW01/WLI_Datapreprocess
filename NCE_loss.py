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
output_path = output_dir / "NCE_loss_per_epoch.png"



def parse_nce_log(file_path):
    
    epochs = []
    nce_values = []

    
    pattern = r'\(epoch: (\d+), iters: \d+.*?\) .*?NCE: ([\d\.]+)'

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                nce = float(match.group(2))
                epochs.append(epoch)
                nce_values.append(nce)

    if not epochs:
        print(f"Warning: No NCE found in {file_path}")
        return pd.DataFrame()

    df = pd.DataFrame({'epoch': epochs, 'NCE': nce_values})

    
    df_avg = df.groupby('epoch')['NCE'].mean().reset_index()
    return df_avg


plt.figure(figsize=(10, 6), dpi=150)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
labels = list(log_files.keys())

all_data = {}

for i, (name, path) in enumerate(log_files.items()):
    df = parse_nce_log(path)
    if df.empty:
        continue

    all_data[name] = df
    df = df[df['epoch'] <= 200]

    plt.plot(df['epoch'], df['NCE'],
             color=colors[i], linewidth=2,
             label=f'{name} (avg NCE)')

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Average NCE Loss', fontsize=14)
plt.title('NCE Contrastive Loss per Epoch\nAcross Four Datasets', fontsize=16, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()


plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"Figure saved to: {output_path}")


for name, df in all_data.items():
    print(f"{name}: {len(df)} epochs recorded, final avg NCE = {df['NCE'].iloc[-1]:.4f}")