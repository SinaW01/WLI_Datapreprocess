import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random
DIR_trainA = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset2\trainA"      # 白光 训练集
DIR_testA  = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset2\testA"       # 白光 测试集
DIR_trainB = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset2\trainB"      # NBI 训练集


# WLI 白光图像
# DIR_trainA = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset1\FullSample\trainA"
# DIR_testA  = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset1\FullSample\testA"
#
# # NBI 图像
# DIR_trainB = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset1\FullSample\trainB"

TARGET_SIZE = (256, 256)
MAX_SAMPLES = 5000
HIST_BINS = 256
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "WLI_NBI_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

def load_image(
        p,
        target_size=None,
        gray=False
):

    flag = (
        cv2.IMREAD_GRAYSCALE
        if gray
        else cv2.IMREAD_COLOR
    )

    img = cv2.imread(
        str(p),
        flag
    )
    if img is None:
        return None
    if not gray:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
    if (
        target_size is not None
        and img.shape[:2] != target_size[::-1]
    ):

        img = cv2.resize(
            img,
            target_size
        )

    return img
def get_all_paths(dir_paths):
    all_paths = []

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff"
    ]

    for d in dir_paths:

        directory = Path(d)

        if not directory.exists():

            print(
                f"\nWARNING: Directory does not exist:\n{d}"
            )

            continue

        for ext in extensions:

            all_paths.extend(
                directory.glob(ext)
            )

    return all_paths

def compute_channel_stats(
        paths,
        target_size=None
):
    means = np.zeros(3)

    stds = np.zeros(3)

    count = 0
    for p in tqdm(
        paths,
        desc="Channel stats"
    ):

        img = load_image(
            p,
            target_size
        )

        if img is None:
            continue

        means += img.mean(
            axis=(0, 1)
        )
        stds += img.std(
            axis=(0, 1)
        )

        count += 1
    if count == 0:

        return (
            None,
            None,
            0
        )
    return (
        means / count,
        stds / count,
        count
    )
def compute_avg_histogram(
        paths,
        bins=HIST_BINS,
        max_samples=MAX_SAMPLES
):

    hist_rgb = np.zeros(
        (3, bins)
    )

    count = 0
    sampled = random.sample(
        paths,
        min(
            max_samples,
            len(paths)
        )
    )
    for p in tqdm(
        sampled,
        desc="Histogram"
    ):

        img = load_image(p)

        if img is None:
            continue
        for ch in range(3):

            h = cv2.calcHist(
                [img],
                [ch],
                None,
                [bins],
                [0, 256]
            )[:, 0]

            hist_rgb[ch] += h

        count += 1
    if count == 0:

        return None
    return (
        hist_rgb / count
    )
def plot_histogram_comparison(
        wl_hist,
        nbi_hist,
        save_path
):
    if (
        wl_hist is None
        or nbi_hist is None
    ):

        print(
            "Histogram data is empty."
        )

        return



    x = np.arange(256)



    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11)
    )

    

    colors = [
        "red",
        "green",
        "blue"
    ]

    channel_names = [
        "Red",
        "Green",
        "Blue"
    ]



    x_ticks = [
        0,
        25,
        50,
        75,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        255
    ]

    for i, ax in enumerate(axes):


        ax.plot(
            x,
            wl_hist[i],
            color=colors[i],
            label="WLI",
            alpha=0.85,
            linewidth=1.5
        )

        ax.plot(
            x,
            nbi_hist[i],
            color=colors[i],
            linestyle="--",
            label="NBI",
            alpha=0.85,
            linewidth=1.5
        )



        ax.set_title(
            f"{channel_names[i]} Channel - Average Histogram",
            fontsize=13
        )



        ax.set_xlim(
            0,
            255
        )

        ax.set_xticks(
            x_ticks
        )


        ax.tick_params(
            axis="x",
            labelsize=8
        )

   
        tick_labels = (
            ax.get_xticklabels()
        )


        tick_labels[-2].set_horizontalalignment(
            "right"
        )

     

        tick_labels[-1].set_horizontalalignment(
            "left"
        )

        ax.set_xlabel(
            "Pixel Intensity (0–255)",
            fontsize=11
        )

    

        ax.set_ylabel(
            "Average Pixel Frequency",
            fontsize=11
        )



        ax.grid(
            True,
            alpha=0.3
        )

  

        ax.legend(
            loc="best"
        )

    

    plt.tight_layout(
        h_pad=1.5
    )


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nHistogram saved to:\n{save_path}"
    )




def compute_mean_image(
        paths,
        target_size=TARGET_SIZE,
        max_samples=MAX_SAMPLES
):
    

    accum = np.zeros(
        (
            target_size[1],
            target_size[0],
            3
        ),
        dtype=np.float64
    )

    count = 0

    sampled = random.sample(
        paths,
        min(
            max_samples,
            len(paths)
        )
    )

    for p in tqdm(
        sampled,
        desc="Mean image"
    ):

        img = load_image(
            p,
            target_size
        )

        if img is None:
            continue

        accum += img.astype(
            np.float64
        )

        count += 1

    if count == 0:

        return None

    return (
        accum / count
    ).astype(
        np.uint8
    )




def plot_mean_images(
        wl_mean,
        nbi_mean,
        save_path
):

    if (
        wl_mean is None
        or nbi_mean is None
    ):

        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )


    axes[0].imshow(
        wl_mean
    )

    axes[0].set_title(
        "Average WLI"
    )

    axes[0].axis(
        "off"
    )



    axes[1].imshow(
        nbi_mean
    )

    axes[1].set_title(
        "Average NBI"
    )

    axes[1].axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()




def compute_laplacian_variance(
        paths,
        max_samples=MAX_SAMPLES
):

    vars_list = []

    sampled = random.sample(
        paths,
        min(
            max_samples,
            len(paths)
        )
    )

    for p in tqdm(
        sampled,
        desc="Laplacian variance"
    ):

        gray = load_image(
            p,
            gray=True
        )

        if gray is None:
            continue

        lap = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        vars_list.append(
            lap.var()
        )

    return np.array(
        vars_list
    )



def compute_brightness_contrast(
        paths,
        max_samples=MAX_SAMPLES
):
    """
    计算：

    Brightness:
        gray.mean()

    Normalized contrast:
        gray.std() / gray.mean()
    """

    means = []

    contrasts = []

    sampled = random.sample(
        paths,
        min(
            max_samples,
            len(paths)
        )
    )

    for p in tqdm(
        sampled,
        desc="Brightness & Contrast"
    ):

        gray = load_image(
            p,
            gray=True
        )

        if gray is None:
            continue



        m = gray.mean()

        c = (
            gray.std()
            /
            (m + 1e-8)
        )

        means.append(m)

        contrasts.append(c)

    return (
        np.array(means),
        np.array(contrasts)
    )




def plot_boxplots(
        data_dict,
        title,
        ylabel,
        save_path
):

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bp = ax.boxplot(
        list(data_dict.values()),
        labels=list(
            data_dict.keys()
        ),
        patch_artist=True,
        widths=0.5
    )


    colors = [
        "#ff9999",
        "#66b3ff"
    ]

    for patch, color in zip(
        bp["boxes"],
        colors
    ):

        patch.set_facecolor(
            color
        )

    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        title
    )



    ax.grid(
        True,
        alpha=0.3,
        axis="y"
    )

    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()




def count_valid_images(
        paths
):

    count = 0

    for p in tqdm(
        paths,
        desc="Counting valid images"
    ):

        if cv2.imread(
            str(p)
        ) is not None:

            count += 1

    return count



if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "WLI vs NBI Image Domain Analysis"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )


   
    wl_paths = get_all_paths(
        [
            DIR_trainA,
            DIR_testA
        ]
    )


    nbi_paths = get_all_paths(
        [
            DIR_trainB
        ]
    )




    wl_valid_count = count_valid_images(
        wl_paths
    )

    nbi_valid_count = count_valid_images(
        nbi_paths
    )


    print(
        "\n" + "=" * 70
    )

    print(
        f"WLI total files found: "
        f"{len(wl_paths)}"
    )

    print(
        f"WLI valid images: "
        f"{wl_valid_count}"
    )

    print(
        f"NBI total files found: "
        f"{len(nbi_paths)}"
    )

    print(
        f"NBI valid images: "
        f"{nbi_valid_count}"
    )

    print(
        "=" * 70
    )



    if (
        wl_valid_count == 0
        or nbi_valid_count == 0
    ):

        print(
            "\nERROR: "
            "No valid images in one or both domains."
        )

        raise SystemExit

    print(
        "\nCalculating RGB channel statistics..."
    )

    wl_mean_rgb, wl_std_rgb, _ = (
        compute_channel_stats(
            wl_paths,
            TARGET_SIZE
        )
    )

    nbi_mean_rgb, nbi_std_rgb, _ = (
        compute_channel_stats(
            nbi_paths,
            TARGET_SIZE
        )
    )


    stats_df = pd.DataFrame({

        "Domain": [
            "WLI (all)",
            "NBI (all)"
        ],

        "R mean ± std": [

            f"{wl_mean_rgb[0]:.1f} ± "
            f"{wl_std_rgb[0]:.1f}",

            f"{nbi_mean_rgb[0]:.1f} ± "
            f"{nbi_std_rgb[0]:.1f}"
        ],

        "G mean ± std": [

            f"{wl_mean_rgb[1]:.1f} ± "
            f"{wl_std_rgb[1]:.1f}",

            f"{nbi_mean_rgb[1]:.1f} ± "
            f"{nbi_std_rgb[1]:.1f}"
        ],

        "B mean ± std": [

            f"{wl_mean_rgb[2]:.1f} ± "
            f"{wl_std_rgb[2]:.1f}",

            f"{nbi_mean_rgb[2]:.1f} ± "
            f"{nbi_std_rgb[2]:.1f}"
        ]

    })


    print(
        "\nRGB Channel Statistics:"
    )

    print(
        stats_df.to_string(
            index=False
        )
    )



    print(
        "\nCalculating RGB histograms..."
    )

    wl_hist = compute_avg_histogram(
        wl_paths
    )

    nbi_hist = compute_avg_histogram(
        nbi_paths
    )


  

    histogram_path = (
        OUTPUT_DIR
        /
        "rgb_histogram_comparison.png"
    )


    plot_histogram_comparison(
        wl_hist,
        nbi_hist,
        histogram_path
    )

    print(
        "\nCalculating mean images..."
    )

    wl_mean_img = compute_mean_image(
        wl_paths
    )

    nbi_mean_img = compute_mean_image(
        nbi_paths
    )


    mean_image_path = (
        OUTPUT_DIR
        /
        "mean_images_comparison.png"
    )


    plot_mean_images(
        wl_mean_img,
        nbi_mean_img,
        mean_image_path
    )


   
    print(
        "\nCalculating Laplacian variance..."
    )

    wl_lap = compute_laplacian_variance(
        wl_paths
    )

    nbi_lap = compute_laplacian_variance(
        nbi_paths
    )


    laplacian_path = (
        OUTPUT_DIR
        /
        "laplacian_variance_boxplot.png"
    )


    plot_boxplots(

        {
            "WLI all": wl_lap,
            "NBI all": nbi_lap
        },

        "Sharpness (Laplacian Variance) - Full Domains",

        "Laplacian Variance",

        laplacian_path
    )


 
    print(
        "\nCalculating brightness and contrast..."
    )

    wl_bright, wl_contrast = (
        compute_brightness_contrast(
            wl_paths
        )
    )

    nbi_bright, nbi_contrast = (
        compute_brightness_contrast(
            nbi_paths
        )
    )


   
    brightness_path = (
        OUTPUT_DIR
        /
        "brightness_boxplot.png"
    )


    plot_boxplots(

        {
            "WLI all": wl_bright,
            "NBI all": nbi_bright
        },

        "Brightness (Mean Intensity) - Full Domains",

        "Mean Gray Value",

        brightness_path
    )


 
    contrast_path = (
        OUTPUT_DIR
        /
        "contrast_boxplot.png"
    )


    plot_boxplots(

        {
            "WLI all": wl_contrast,
            "NBI all": nbi_contrast
        },

        "RMS Contrast - Full Domains",

        "RMS Contrast",

        contrast_path
    )



    print(
        "\n" + "=" * 70
    )

    print(
        "Analysis completed."
    )

    print(
        "=" * 70
    )

    print(
        f"\nAll output files are saved in:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nGenerated files:"
    )

    print(
        "1. rgb_histogram_comparison.png"
    )

    print(
        "2. mean_images_comparison.png"
    )

    print(
        "3. laplacian_variance_boxplot.png"
    )

    print(
        "4. brightness_boxplot.png"
    )

    print(
        "5. contrast_boxplot.png"
    )

    print(
        "\n" + "=" * 70
    )