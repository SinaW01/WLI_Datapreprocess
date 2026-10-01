import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random


# ============================================================
# ===================== 配置部分 =============================
# ============================================================

# ------------------------------------------------------------
# 输入数据路径
# ------------------------------------------------------------

DIR_trainA = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset2\trainA"      # 白光 训练集
DIR_testA  = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset2\testA"       # 白光 测试集
DIR_trainB = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset2\trainB"      # NBI 训练集


# WLI 白光图像
# DIR_trainA = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset1\FullSample\trainA"
# DIR_testA  = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset1\FullSample\testA"
#
# # NBI 图像
# DIR_trainB = r"E:\wx\dataset_wx\Science_data\ScienceData\Dataset\dataset1\FullSample\trainB"


# ------------------------------------------------------------
# 图像及统计参数
# ------------------------------------------------------------

# Mean Image 等部分统计统一调整到 256 × 256
TARGET_SIZE = (256, 256)

# 每个 domain 最多随机采样 5000 张
MAX_SAMPLES = 5000

# ============================================================
# Histogram 使用 256 个 bin
#
# 0   -> 像素强度 0
# 1   -> 像素强度 1
# ...
# 254 -> 像素强度 254
# 255 -> 像素强度 255
# ============================================================

HIST_BINS = 256

# 随机种子
RANDOM_SEED = 42


# ============================================================
# 设置随机种子
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 输出目录
#
# 会自动创建在当前 Python 文件所在目录下
#
# WLI_NBI.py
#     |
#     └── WLI_NBI_results
# ============================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "WLI_NBI_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. 图像读取函数
# ============================================================

def load_image(
        p,
        target_size=None,
        gray=False
):
    """
    读取图像。

    参数：
        p           : 图像路径
        target_size : 是否 resize，例如 (256, 256)
        gray        : 是否读取为灰度图

    返回：
        RGB 图像或灰度图像
    """

    # --------------------------------------------------------
    # OpenCV 读取模式
    # --------------------------------------------------------

    flag = (
        cv2.IMREAD_GRAYSCALE
        if gray
        else cv2.IMREAD_COLOR
    )

    # --------------------------------------------------------
    # 读取图像
    # --------------------------------------------------------

    img = cv2.imread(
        str(p),
        flag
    )

    # --------------------------------------------------------
    # 无法读取
    # --------------------------------------------------------

    if img is None:
        return None

    # --------------------------------------------------------
    # OpenCV 默认 BGR
    # 转换为 RGB
    # --------------------------------------------------------

    if not gray:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

    # --------------------------------------------------------
    # resize
    # --------------------------------------------------------

    if (
        target_size is not None
        and img.shape[:2] != target_size[::-1]
    ):

        img = cv2.resize(
            img,
            target_size
        )

    return img


# ============================================================
# 2. 获取所有图像路径
# ============================================================

def get_all_paths(dir_paths):
    """
    从多个目录中获取所有支持的图像文件。

    支持：
        jpg
        jpeg
        png
        bmp
        tif
        tiff
    """

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

        # ----------------------------------------------------
        # 检查目录是否存在
        # ----------------------------------------------------

        if not directory.exists():

            print(
                f"\nWARNING: Directory does not exist:\n{d}"
            )

            continue

        # ----------------------------------------------------
        # 搜索不同图片格式
        # ----------------------------------------------------

        for ext in extensions:

            all_paths.extend(
                directory.glob(ext)
            )

    return all_paths


# ============================================================
# 3. RGB Channel Statistics
# ============================================================

def compute_channel_stats(
        paths,
        target_size=None
):
    """
    计算每张图像的 RGB mean 和 std，
    最后对所有图像求平均。
    """

    means = np.zeros(3)

    stds = np.zeros(3)

    count = 0

    # --------------------------------------------------------
    # 遍历所有图像
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # RGB mean
        # ----------------------------------------------------

        means += img.mean(
            axis=(0, 1)
        )

        # ----------------------------------------------------
        # RGB std
        # ----------------------------------------------------

        stds += img.std(
            axis=(0, 1)
        )

        count += 1

    # --------------------------------------------------------
    # 没有有效图像
    # --------------------------------------------------------

    if count == 0:

        return (
            None,
            None,
            0
        )

    # --------------------------------------------------------
    # 对图像数量求平均
    # --------------------------------------------------------

    return (
        means / count,
        stds / count,
        count
    )


# ============================================================
# 4. RGB Average Histogram
# ============================================================

def compute_avg_histogram(
        paths,
        bins=HIST_BINS,
        max_samples=MAX_SAMPLES
):
    """
    计算 RGB 三个通道的平均 Histogram。

    bins = 256
    intensity range = 0~255

    返回：
        hist_rgb[0] -> Red
        hist_rgb[1] -> Green
        hist_rgb[2] -> Blue
    """

    # --------------------------------------------------------
    # 初始化 Histogram
    #
    # shape:
    #     3 × 256
    # --------------------------------------------------------

    hist_rgb = np.zeros(
        (3, bins)
    )

    count = 0

    # --------------------------------------------------------
    # 随机采样
    # --------------------------------------------------------

    sampled = random.sample(
        paths,
        min(
            max_samples,
            len(paths)
        )
    )

    # --------------------------------------------------------
    # 遍历图像
    # --------------------------------------------------------

    for p in tqdm(
        sampled,
        desc="Histogram"
    ):

        img = load_image(p)

        if img is None:
            continue

        # ----------------------------------------------------
        # R / G / B
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # 没有有效图像
    # --------------------------------------------------------

    if count == 0:

        return None

    # --------------------------------------------------------
    # 平均 Histogram
    #
    # Y轴：
    # Average Pixel Frequency
    # --------------------------------------------------------

    return (
        hist_rgb / count
    )


# ============================================================
# 5. RGB Histogram 绘图
# ============================================================

def plot_histogram_comparison(
        wl_hist,
        nbi_hist,
        save_path
):
    """
    绘制 WLI 与 NBI 的 RGB Histogram。

    三个子图：
        Red
        Green
        Blue

    X轴：
        Pixel Intensity (0–255)

    Y轴：
        Average Pixel Frequency
    """

    # --------------------------------------------------------
    # 数据检查
    # --------------------------------------------------------

    if (
        wl_hist is None
        or nbi_hist is None
    ):

        print(
            "Histogram data is empty."
        )

        return

    # ========================================================
    # X轴
    #
    # 256 bins 对应 0~255
    # ========================================================

    x = np.arange(256)

    # ========================================================
    # 创建三个独立子图
    #
    # 注意：
    # 不使用 sharex=True
    #
    # 这样三个子图都可以显示自己的 X 轴
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11)
    )

    # ========================================================
    # RGB 颜色
    # ========================================================

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

    # ========================================================
    # X轴刻度
    #
    # 同时保留：
    # 250
    # 255
    #
    # 后面通过字体和对齐方式解决重叠
    # ========================================================

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

    # ========================================================
    # 绘制三个 Channel
    # ========================================================

    for i, ax in enumerate(axes):

        # ----------------------------------------------------
        # WLI
        # ----------------------------------------------------

        ax.plot(
            x,
            wl_hist[i],
            color=colors[i],
            label="WLI",
            alpha=0.85,
            linewidth=1.5
        )

        # ----------------------------------------------------
        # NBI
        # ----------------------------------------------------

        ax.plot(
            x,
            nbi_hist[i],
            color=colors[i],
            linestyle="--",
            label="NBI",
            alpha=0.85,
            linewidth=1.5
        )

        # ====================================================
        # 子图标题
        # ====================================================

        ax.set_title(
            f"{channel_names[i]} Channel - Average Histogram",
            fontsize=13
        )

        # ====================================================
        # X轴范围
        # ====================================================

        ax.set_xlim(
            0,
            255
        )

        # ====================================================
        # X轴刻度
        # ====================================================

        ax.set_xticks(
            x_ticks
        )

        # ====================================================
        # X轴刻度字号
        #
        # 缩小字号，让 250 和 255 可以同时显示
        # ====================================================

        ax.tick_params(
            axis="x",
            labelsize=8
        )

        # ====================================================
        # 获取 X轴标签
        # ====================================================

        tick_labels = (
            ax.get_xticklabels()
        )

        # ----------------------------------------------------
        # 250
        #
        # 右对齐
        # 向左展开
        # ----------------------------------------------------

        tick_labels[-2].set_horizontalalignment(
            "right"
        )

        # ----------------------------------------------------
        # 255
        #
        # 左对齐
        # 向右展开
        # ----------------------------------------------------

        tick_labels[-1].set_horizontalalignment(
            "left"
        )

        # ====================================================
        # X轴标题
        #
        # 每一个子图都有
        # ====================================================

        ax.set_xlabel(
            "Pixel Intensity (0–255)",
            fontsize=11
        )

        # ====================================================
        # Y轴标题
        #
        # 每一个子图都有
        # ====================================================

        ax.set_ylabel(
            "Average Pixel Frequency",
            fontsize=11
        )

        # ====================================================
        # 网格
        # ====================================================

        ax.grid(
            True,
            alpha=0.3
        )

        # ====================================================
        # 图例
        # ====================================================

        ax.legend(
            loc="best"
        )

    # ========================================================
    # 调整子图之间的距离
    # ========================================================

    plt.tight_layout(
        h_pad=1.5
    )

    # ========================================================
    # 保存 Histogram
    # ========================================================

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nHistogram saved to:\n{save_path}"
    )


# ============================================================
# 6. Mean Image
# ============================================================

def compute_mean_image(
        paths,
        target_size=TARGET_SIZE,
        max_samples=MAX_SAMPLES
):
    """
    计算平均图像。
    """

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


# ============================================================
# 7. Mean Image 绘图
# ============================================================

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

    # --------------------------------------------------------
    # WLI
    # --------------------------------------------------------

    axes[0].imshow(
        wl_mean
    )

    axes[0].set_title(
        "Average WLI"
    )

    axes[0].axis(
        "off"
    )

    # --------------------------------------------------------
    # NBI
    # --------------------------------------------------------

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


# ============================================================
# 8. Laplacian Variance
# ============================================================

def compute_laplacian_variance(
        paths,
        max_samples=MAX_SAMPLES
):
    """
    使用 Laplacian variance 衡量图像锐度。
    """

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


# ============================================================
# 9. Brightness & Contrast
# ============================================================

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

        # ----------------------------------------------------
        # Brightness
        # ----------------------------------------------------

        m = gray.mean()

        # ----------------------------------------------------
        # Normalized contrast
        # ----------------------------------------------------

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


# ============================================================
# 10. Boxplot
# ============================================================

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

    # --------------------------------------------------------
    # 箱体颜色
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Y轴
    # --------------------------------------------------------

    ax.set_ylabel(
        ylabel
    )

    # --------------------------------------------------------
    # 标题
    # --------------------------------------------------------

    ax.set_title(
        title
    )

    # --------------------------------------------------------
    # 网格
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.3,
        axis="y"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. 统计有效图像数量
# ============================================================

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


# ============================================================
# ======================= 主程序 =============================
# ============================================================

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


    # ========================================================
    # 1. 获取 WLI 图像
    #
    # WLI = trainA + testA
    # ========================================================

    wl_paths = get_all_paths(
        [
            DIR_trainA,
            DIR_testA
        ]
    )


    # ========================================================
    # 2. 获取 NBI 图像
    #
    # NBI = trainB
    # ========================================================

    nbi_paths = get_all_paths(
        [
            DIR_trainB
        ]
    )


    # ========================================================
    # 3. 统计有效图片
    # ========================================================

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


    # ========================================================
    # 如果没有有效图像，停止
    # ========================================================

    if (
        wl_valid_count == 0
        or nbi_valid_count == 0
    ):

        print(
            "\nERROR: "
            "No valid images in one or both domains."
        )

        raise SystemExit


    # ========================================================
    # 4. RGB Channel Statistics
    # ========================================================

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


    # ========================================================
    # 生成 RGB Statistics 表
    # ========================================================

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


    # ========================================================
    # 5. RGB Histogram
    # ========================================================

    print(
        "\nCalculating RGB histograms..."
    )

    wl_hist = compute_avg_histogram(
        wl_paths
    )

    nbi_hist = compute_avg_histogram(
        nbi_paths
    )


    # --------------------------------------------------------
    # Histogram 输出路径
    # --------------------------------------------------------

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


    # ========================================================
    # 6. Mean Image
    # ========================================================

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


    # ========================================================
    # 7. Laplacian Variance
    # ========================================================

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


    # ========================================================
    # 8. Brightness & Contrast
    # ========================================================

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


    # ========================================================
    # 8.1 Brightness
    # ========================================================

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


    # ========================================================
    # 8.2 Contrast
    # ========================================================

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


    # ========================================================
    # 9. 最终输出信息
    # ========================================================

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