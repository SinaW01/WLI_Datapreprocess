import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


log_files = {
    'CLOSE': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\constructs\CLOSE_loss_log.txt',
    'LESION': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\constructs\LESION_loss_log.txt',
    'FullSample': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\constructs\FullSample_loss_log.txt',
    'dataset2': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\constructs\NBI_256_loss_log.txt',
}

# log_files = {
#     'CLOSE': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\FFPE++\CLOSE_loss_log.txt',
#     'LESION': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\FFPE++\LESION_loss_log.txt',
#     'FullSample': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\FFPE++\FullSample_loss_log.txt',
#     'dataset2': r'E:\wx\dataset_wx\ScienceData\Code\txtfiles\FFPE++\NBI_256_loss_log.txt',
# }


output_dir = Path(r"E:\wx\dataset_wx\ScienceData\Code\txtfiles/plots")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "Total_G_loss_per_epoch_up_to_200.png"



def parse_total_g_log(file_path):
   
    epochs = []
    g_values = []

 
    pattern = r'\(epoch: (\d+), iters: \d+.*?\) .*?G: ([\d\.]+)'

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                g_loss = float(match.group(2))
                epochs.append(epoch)
                g_values.append(g_loss)

    if not epochs:
        print(f"Warning: No G loss found in {file_path}")
        return pd.DataFrame()

    df = pd.DataFrame({'epoch': epochs, 'G': g_values})

 
    df_avg = df.groupby('epoch')['G'].mean().reset_index()
    return df_avg



plt.figure(figsize=(10, 6), dpi=150)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  
labels = list(log_files.keys())

all_data = {}

for i, (name, path) in enumerate(log_files.items()):
    df = parse_total_g_log(path)
    if df.empty:
        continue

    all_data[name] = df

  
    df_plot = df[df['epoch'] <= 200]

    if df_plot.empty:
        print(f"Warning: {name} has no data up to epoch 200")
        continue

    plt.plot(df_plot['epoch'], df_plot['G'],
             color=colors[i], linewidth=2,
             label=f'{name} (avg G)')


plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Average Generator Total Loss (G)', fontsize=14)
plt.title('Generator Total Loss (G) per Epoch (up to 250)\nAcross Four Datasets', fontsize=16, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()


plt.savefig(output_path, bbox_inches='tight')
plt.show()

print(f"Figure saved to: {output_path}")


for name, df in all_data.items():
    max_epoch = df['epoch'].max()
    final_g = df['G'].iloc[-1] if not df.empty else np.nan
    print(f"{name}: max epoch = {max_epoch}, final avg G = {final_g:.4f}")