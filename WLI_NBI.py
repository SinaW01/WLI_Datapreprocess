#    这是测试白光图像和NBI图像在风格上确实存在一定的差异

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random


DIR_trainA = r"E:\wx\dataset_wx\ScienceData\Dataset\dataset2\trainA"      
DIR_testA  = r"E:\wx\dataset_wx\ScienceData\Dataset\dataset2\testA"       
DIR_trainB = r"E:\wx\dataset_wx\ScienceData\Dataset\dataset2\trainB"      


# DIR_trainA = r"E:\wx\dataset_wx\ScienceData\Dataset\dataset1\FullSample\trainA"      
# DIR_testA  = r"E:\wx\dataset_wx\ScienceData\Dataset\dataset1\FullSample\testA"       
# DIR_trainB = r"E:\wx\dataset_wx\ScienceData\Dataset\dataset1\FullSample\trainB"     
   

TARGET_SIZE = (256, 256)            
MAX_SAMPLES = 5000                 
HIST_BINS = 128                     
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_image(p, target_size=None, gray=False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(p), flag)
    if img is None:
        return None
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None and img.shape[:2] != target_size[::-1]:
        img = cv2.resize(img, target_size)
    return img

def get_all_paths(dir_paths):
  
    all_paths = []
    for d in dir_paths:
        paths = list(Path(d).glob("*.[jpJP][pnPN][gG]*"))  
        all_paths.extend(paths)
    return all_paths


def compute_channel_stats(paths, target_size=None):
    means = np.zeros(3)
    stds = np.zeros(3)
    count = 0
    for p in tqdm(paths, desc="Channel stats"):
        img = load_image(p, target_size)
        if img is None:
            continue
        means += img.mean(axis=(0,1))
        stds += img.std(axis=(0,1))
        count += 1
    if count == 0:
        return None, None, 0
    return means / count, stds / count, count


def compute_avg_histogram(paths, bins=HIST_BINS, max_samples=MAX_SAMPLES):
    hist_rgb = np.zeros((3, bins))
    count = 0
    sampled = random.sample(paths, min(max_samples, len(paths)))
    for p in tqdm(sampled, desc="Histogram"):
        img = load_image(p)
        if img is None:
            continue
        for ch in range(3):
            h = cv2.calcHist([img], [ch], None, [bins], [0, 256])[:, 0]
            hist_rgb[ch] += h
        count += 1
    if count == 0:
        return None
    return hist_rgb / count

def plot_histogram_comparison(wl_hist, nbi_hist, save_path="rgb_histogram_comparison.png"):
    if wl_hist is None or nbi_hist is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = ['r', 'g', 'b']
    labels = ['Red', 'Green', 'Blue']
    for i, ax in enumerate(axes):
        ax.plot(wl_hist[i], color=colors[i], label='WLI', alpha=0.8)
        ax.plot(nbi_hist[i], color=colors[i], linestyle='--', label='NBI', alpha=0.8)
        ax.set_title(f"{labels[i]} Channel - Average Histogram")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def compute_mean_image(paths, target_size=TARGET_SIZE, max_samples=MAX_SAMPLES):
    accum = np.zeros((target_size[1], target_size[0], 3), dtype=np.float64)
    count = 0
    sampled = random.sample(paths, min(max_samples, len(paths)))
    for p in tqdm(sampled, desc="Mean image"):
        img = load_image(p, target_size)
        if img is None:
            continue
        accum += img.astype(np.float64)
        count += 1
    if count == 0:
        return None
    return (accum / count).astype(np.uint8)

def plot_mean_images(wl_mean, nbi_mean, save_path="mean_images_comparison.png"):
    if wl_mean is None or nbi_mean is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(wl_mean)
    axes[0].set_title("Average WLI")
    axes[0].axis('off')
    axes[1].imshow(nbi_mean)
    axes[1].set_title("Average NBI")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def compute_laplacian_variance(paths, max_samples=MAX_SAMPLES):
    vars_list = []
    sampled = random.sample(paths, min(max_samples, len(paths)))
    for p in tqdm(sampled, desc="Laplacian variance"):
        gray = load_image(p, gray=True)
        if gray is None:
            continue
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        vars_list.append(lap.var())
    return np.array(vars_list)


def compute_brightness_contrast(paths, max_samples=MAX_SAMPLES):
    means = []
    contrasts = []
    sampled = random.sample(paths, min(max_samples, len(paths)))
    for p in tqdm(sampled, desc="Brightness & Contrast"):
        gray = load_image(p, gray=True)
        if gray is None:
            continue
        m = gray.mean()
        c = gray.std() / (m + 1e-8)
        means.append(m)
        contrasts.append(c)
    return np.array(means), np.array(contrasts)

def plot_boxplots(data_dict, title, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(list(data_dict.values()), labels=list(data_dict.keys()),
                    patch_artist=True, widths=0.5)
    colors = ['#ff9999', '#66b3ff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


if __name__ == "__main__":
   
    wl_paths = get_all_paths([DIR_trainA, DIR_testA])
    nbi_paths = get_all_paths([DIR_trainB])

  
    def count_valid_images(paths):
        count = 0
        for p in tqdm(paths, desc="Counting valid images"):
            if cv2.imread(str(p)) is not None:
                count += 1
        return count

    wl_valid_count = count_valid_images(wl_paths)
    nbi_valid_count = count_valid_images(nbi_paths)

    print(f"White Light (trainA + testA) total files found: {len(wl_paths)}")
    print(f"White Light valid images (readable): {wl_valid_count}")
    print(f"NBI (trainB + testB) total files found: {len(nbi_paths)}")
    print(f"NBI valid images (readable): {nbi_valid_count}\n")

    if wl_valid_count == 0 or nbi_valid_count == 0:
        print("Error: No valid images in one or both domains.")
    else:
      
        wl_mean_rgb, wl_std_rgb, _ = compute_channel_stats(wl_paths, TARGET_SIZE)
        nbi_mean_rgb, nbi_std_rgb, _ = compute_channel_stats(nbi_paths, TARGET_SIZE)

        stats_df = pd.DataFrame({
            'Domain': ['WLI (all)', 'NBI (all)'],
            'R mean ± std': [f"{wl_mean_rgb[0]:.1f} ± {wl_std_rgb[0]:.1f}" if wl_mean_rgb is not None else "N/A",
                             f"{nbi_mean_rgb[0]:.1f} ± {nbi_std_rgb[0]:.1f}" if nbi_mean_rgb is not None else "N/A"],
            'G mean ± std': [f"{wl_mean_rgb[1]:.1f} ± {wl_std_rgb[1]:.1f}" if wl_mean_rgb is not None else "N/A",
                             f"{nbi_mean_rgb[1]:.1f} ± {nbi_std_rgb[1]:.1f}" if nbi_mean_rgb is not None else "N/A"],
            'B mean ± std': [f"{wl_mean_rgb[2]:.1f} ± {wl_std_rgb[2]:.1f}" if wl_mean_rgb is not None else "N/A",
                             f"{nbi_mean_rgb[2]:.1f} ± {nbi_std_rgb[2]:.1f}" if nbi_mean_rgb is not None else "N/A"]
        })
        print("RGB Channel Statistics (full domains):")
        print(stats_df.to_string(index=False))
        print()

    
        wl_hist = compute_avg_histogram(wl_paths)
        nbi_hist = compute_avg_histogram(nbi_paths)
        plot_histogram_comparison(wl_hist, nbi_hist)

  
        wl_mean_img = compute_mean_image(wl_paths)
        nbi_mean_img = compute_mean_image(nbi_paths)
        plot_mean_images(wl_mean_img, nbi_mean_img)

    
        wl_lap = compute_laplacian_variance(wl_paths)
        nbi_lap = compute_laplacian_variance(nbi_paths)
        plot_boxplots(
            {'WLI all': wl_lap, 'NBI all': nbi_lap},
            "Sharpness (Laplacian Variance) - Full Domains",
            "Laplacian Variance",
            "laplacian_variance_boxplot.png"
        )

        wl_bright, wl_contrast = compute_brightness_contrast(wl_paths)
        nbi_bright, nbi_contrast = compute_brightness_contrast(nbi_paths)
        plot_boxplots(
            {'WLI all': wl_bright, 'NBI all': nbi_bright},
            "Brightness (Mean Intensity) - Full Domains",
            "Mean Gray Value",
            "brightness_boxplot.png"
        )
        plot_boxplots(
            {'WLI all': wl_contrast, 'NBI all': nbi_contrast},
            "RMS Contrast - Full Domains",
            "RMS Contrast",
            "contrast_boxplot.png"
        )

        print("\nAll plots saved in current directory.")