import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

txt_files = {
    'CLOSE': r'.\CLOSE_Gloss_log.txt',
    'LESION': r'.\LESION_Gloss_log.txt',
    'FullSample': r'.\FullSample_Gloss_log.txt',
    'dataset2': r'.\dataset2_Gloss_log.txt',
}

output_dir = Path(r".\plots")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "AttentionGAN_Total_G_loss_per_epoch_up_to_200.png"

def load_epoch_txt(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df

plt.figure(figsize=(10, 6), dpi=150)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

all_data = {}

for i, (name, path) in enumerate(txt_files.items()):
    df = load_epoch_txt(path)

    if df.empty:
        print(f"Warning: {name} empty")
        continue

    all_data[name] = df

    plt.plot(df_plot['epoch'], df_plot['G'],
             color=colors[i], linewidth=2,
             label=f'{name} (avg G)')

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Average Generator Total Loss (G)', fontsize=14)
plt.title('Generator Total Loss (G) per Epoch (up to 200)\nAcross Four Datasets', fontsize=16, pad=15)
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