                   

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random

DIR_trainA = r".\dataset1\FullSample\trainA"     
DIR_testA  = r".\dataset1\FullSample\testA"       

TARGET_SIZE = (256, 256)           
MAX_SAMPLES = 10000                 
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

def count_valid_images(paths):
    count = 0
    for p in tqdm(paths, desc="Counting valid images"):
        if cv2.imread(str(p)) is not None:
            count += 1
    return count

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
    sampled = random.sample(paths, min(max_samples, len(paths))) if max_samples < len(paths) else paths
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

def plot_histogram_comparison(hist1, hist2, label1, label2, save_path):
    if hist1 is None or hist2 is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = ['r', 'g', 'b']
    labels = ['Red', 'Green', 'Blue']
    for i, ax in enumerate(axes):
        ax.plot(hist1[i], color=colors[i], label=label1, alpha=0.8)
        ax.plot(hist2[i], color=colors[i], linestyle='--', label=label2, alpha=0.8)
        ax.set_title(f"{labels[i]} Channel - Average Histogram")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def compute_mean_image(paths, target_size=TARGET_SIZE, max_samples=MAX_SAMPLES):
    accum = np.zeros((target_size[1], target_size[0], 3), dtype=np.float64)
    count = 0
    sampled = random.sample(paths, min(max_samples, len(paths))) if max_samples < len(paths) else paths
    for p in tqdm(sampled, desc="Mean image"):
        img = load_image(p, target_size)
        if img is None:
            continue
        accum += img.astype(np.float64)
        count += 1
    if count == 0:
        return None
    return (accum / count).astype(np.uint8)

def plot_mean_images_four(wl_train, wl_test, save_path="mean_images_train_test.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if wl_train is not None:
        axes[0,0].imshow(wl_train)
        axes[0,0].set_title("WLI Train Avg")
    if wl_test is not None:
        axes[0,1].imshow(wl_test)
        axes[0,1].set_title("WLI Test Avg")

    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def compute_laplacian_variance(paths, max_samples=MAX_SAMPLES):
    vars_list = []
    sampled = random.sample(paths, min(max_samples, len(paths))) if max_samples < len(paths) else paths
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
    sampled = random.sample(paths, min(max_samples, len(paths))) if max_samples < len(paths) else paths
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
    fig, ax = plt.subplots(figsize=(6, 5))
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
    trainA_paths = list(Path(DIR_trainA).glob("*.[jpJP][pnPN][gG]*"))
    testA_paths  = list(Path(DIR_testA).glob("*.[jpJP][pnPN][gG]*"))

    wl_train_valid = count_valid_images(trainA_paths)
    wl_test_valid  = count_valid_images(testA_paths)
 

    print(f"WLI Train files: {len(trainA_paths)}  valid: {wl_train_valid}")
    print(f"WLI Test  files: {len(testA_paths)}   valid: {wl_test_valid}")

    if min(wl_train_valid, wl_test_valid) == 0:
        print("Error: One or more groups have no valid images.")
    else:
        stats = {}
        for name, paths in [("WL_train", trainA_paths), ("WL_test", testA_paths),]:
            mean_rgb, std_rgb, cnt = compute_channel_stats(paths, TARGET_SIZE)
            stats[name] = {"mean": mean_rgb, "std": std_rgb, "count": cnt}

        print("\nWhite Light domain - Train vs Test RGB Statistics:")
        df_wl = pd.DataFrame({
            "Group": ["WL_train", "WL_test"],
            "R mean ± std": [f"{stats['WL_train']['mean'][0]:.1f} ± {stats['WL_train']['std'][0]:.1f}" if stats['WL_train']['mean'] is not None else "N/A",
                             f"{stats['WL_test']['mean'][0]:.1f} ± {stats['WL_test']['std'][0]:.1f}" if stats['WL_test']['mean'] is not None else "N/A"],
            "G mean ± std": [f"{stats['WL_train']['mean'][1]:.1f} ± {stats['WL_train']['std'][1]:.1f}" if stats['WL_train']['mean'] is not None else "N/A",
                             f"{stats['WL_test']['mean'][1]:.1f} ± {stats['WL_test']['std'][1]:.1f}" if stats['WL_test']['mean'] is not None else "N/A"],
            "B mean ± std": [f"{stats['WL_train']['mean'][2]:.1f} ± {stats['WL_train']['std'][2]:.1f}" if stats['WL_train']['mean'] is not None else "N/A",
                             f"{stats['WL_test']['mean'][2]:.1f} ± {stats['WL_test']['std'][2]:.1f}" if stats['WL_test']['mean'] is not None else "N/A"]
        })
        print(df_wl.to_string(index=False))

        wl_hist_train = compute_avg_histogram(trainA_paths)
        wl_hist_test  = compute_avg_histogram(testA_paths)
        plot_histogram_comparison(wl_hist_train, wl_hist_test, "WLI Train", "WLI Test", "wl_histogram_train_vs_test.png")

        wl_train_mean = compute_mean_image(trainA_paths)
        wl_test_mean  = compute_mean_image(testA_paths)
  
        plot_mean_images_four(wl_train_mean, wl_test_mean)

        wl_lap_train = compute_laplacian_variance(trainA_paths)
        wl_lap_test  = compute_laplacian_variance(testA_paths)
        plot_boxplots({"WLI Train": wl_lap_train, "WLI Test": wl_lap_test},
                      "White Light - Sharpness (Laplacian Variance) Train vs Test",
                      "Laplacian Variance", "wl_laplacian_train_vs_test.png")

      
        wl_bright_train, _ = compute_brightness_contrast(trainA_paths)
        wl_bright_test,  _ = compute_brightness_contrast(testA_paths)
        plot_boxplots({"WLI Train": wl_bright_train, "WLI Test": wl_bright_test},
                      "White Light - Brightness (Mean Intensity) Train vs Test",
                      "Mean Gray Value", "wl_brightness_train_vs_test.png")

 
        _, wl_contrast_train = compute_brightness_contrast(trainA_paths)
        _, wl_contrast_test  = compute_brightness_contrast(testA_paths)
        plot_boxplots({"WLI Train": wl_contrast_train, "WLI Test": wl_contrast_test},
                      "White Light - RMS Contrast Train vs Test",
                      "RMS Contrast", "wl_contrast_train_vs_test.png")

 
        print("\nAll comparison plots (train vs test) saved in current directory.")