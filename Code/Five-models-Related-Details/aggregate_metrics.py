import glob
import importlib.util
import math
import os
import pathlib
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

# Make local code importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, '1_model'))

try:
    from function_norm.inception import InceptionV3
except ImportError as e:
    raise ImportError(f"Cannot import InceptionV3 from 1_model/function_norm/inception.py: {e}")


PIQE_MODULE_PATH = os.path.join(ROOT_DIR, '2_model', 'piqe', 'piqe.py')
NIQE_MODULE_PATH = os.path.join(ROOT_DIR, '2_model', 'niqe', 'niqe.py')

spec = importlib.util.spec_from_file_location('piqe_module', PIQE_MODULE_PATH)
piqe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(piqe_module)

spec = importlib.util.spec_from_file_location('niqe_module', NIQE_MODULE_PATH)
niqe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(niqe_module)

IMG_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

# User configuration: fill these paths directly in the code.
REF_DIR = r".\AttentionGAN-master\datasets\FullSample\trainB"
FAKE_FOLDERS = [
    r".\AttentionGAN-master\results\FullSample_wx\test\fakeB",
    r".\AttentionGAN-master\results\FullSample_wx\test_seed_42\images",
    r".\AttentionGAN-master\results\FullSample_wx\test_seed_123\images",
    r".\AttentionGAN-master\results\FullSample_wx\test_seed_456\images",
    r".\AttentionGAN-master\results\FullSample_wx\test_seed_789\images",
]
USE_GPU = False
KID_SUBSET_SIZE = 100
KID_IMAGE_SIZE = 299
FID_BATCH_SIZE = 64
OUTPUT_CSV = None


def is_image_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMG_EXT


def list_image_files(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, '**', f'*{ext}'), recursive=True))
    return sorted(files)


def list_fake_folders(fake_root, fake_subfolder=None):
    if not os.path.isdir(fake_root):
        raise ValueError(f'fake_root not found: {fake_root}')

    child_dirs = [os.path.join(fake_root, d) for d in sorted(os.listdir(fake_root))]
    child_dirs = [d for d in child_dirs if os.path.isdir(d)]

    if not child_dirs:
        return [fake_root]

    resolved = []
    for child in child_dirs:
        if fake_subfolder:
            target = os.path.join(child, fake_subfolder)
            if os.path.isdir(target) and list_image_files(target):
                resolved.append(target)
                continue
        if list_image_files(child):
            resolved.append(child)
            continue
        subdirs = [os.path.join(child, sd) for sd in sorted(os.listdir(child)) if os.path.isdir(os.path.join(child, sd))]
        if len(subdirs) == 1 and list_image_files(subdirs[0]):
            resolved.append(subdirs[0])
            continue
        named = os.path.join(child, 'fakeB')
        if os.path.isdir(named) and list_image_files(named):
            resolved.append(named)
            continue
    if resolved:
        return resolved
    return [fake_root]


def ensure_images_exist(folder):
    files = list_image_files(folder)
    if not files:
        raise ValueError(f'No image files found in folder: {folder}')
    return files


def prepare_image_tensor(path, image_size):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'Cannot read image: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(torch.uint8)


def compute_kid(real_dir, fake_dir, image_size=299, subset_size=100, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    kid_metric = KernelInceptionDistance(subset_size=subset_size, normalize=False).to(device)

    real_files = ensure_images_exist(real_dir)
    fake_files = ensure_images_exist(fake_dir)

    for path in tqdm(real_files, desc='KID real images', unit='img'):
        try:
            img = prepare_image_tensor(path, image_size)
        except ValueError:
            continue
        kid_metric.update(img.to(device), real=True)

    for path in tqdm(fake_files, desc='KID fake images', unit='img'):
        try:
            img = prepare_image_tensor(path, image_size)
        except ValueError:
            continue
        kid_metric.update(img.to(device), real=False)

    kid_mean, kid_std = kid_metric.compute()
    return float(kid_mean.cpu()), float(kid_std.cpu())


def compute_fid(real_dir, fake_dir, batch_size=64, use_gpu=False, dims=2048):
    real_dir = pathlib.Path(real_dir)
    fake_dir = pathlib.Path(fake_dir)
    if not real_dir.exists() or not fake_dir.exists():
        raise ValueError('real_dir or fake_dir does not exist')

    def list_files(path):
        return sorted(
            p for p in path.rglob('*')
            if is_image_file(p)
        )

    real_files = list_files(real_dir)
    fake_files = list_files(fake_dir)
    if not real_files:
        raise ValueError(f'No images found under real_dir: {real_dir}')
    if not fake_files:
        raise ValueError(f'No images found under fake_dir: {fake_dir}')

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if use_gpu:
        model.cuda()

    def get_activations(images, model, batch_size=64, dims=2048, cuda=False):
        model.eval()
        d0 = images.shape[0]
        if batch_size > d0:
            batch_size = d0
        n_batches = d0 // batch_size
        n_used_imgs = n_batches * batch_size
        pred_arr = np.empty((n_used_imgs, dims))
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            if cuda:
                batch = batch.cuda()
            pred = model(batch)[0]
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)
        return pred_arr

    def calculate_activation_statistics(images, model, batch_size=64, dims=2048, cuda=False):
        act = get_activations(images, model, batch_size, dims, cuda)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def compute_statistics(path, model, batch_size, dims, cuda):
        files = list_files(path)
        images = []
        for fn in files:
            img = cv2.imread(str(fn)).astype(np.float32)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            img = cv2.resize(img, (256, 256))
            images.append(img)
        images = np.array(images)
        images = images.transpose((0, 3, 1, 2))
        images /= 255.0
        return calculate_activation_statistics(images, model, batch_size, dims, cuda)

    m1, s1 = compute_statistics(real_dir, model, batch_size, dims, use_gpu)
    m2, s2 = compute_statistics(fake_dir, model, batch_size, dims, use_gpu)
    return calculate_frechet_distance(m1, s1, m2, s2)


def compute_image_quality(folder, metric_fn, metric_name):
    files = ensure_images_exist(folder)
    scores = []
    for path in tqdm(files, desc=f'{metric_name} images', unit='img'):
        image = cv2.imread(path)
        if image is None:
            continue
        try:
            score = float(metric_fn(image))
        except Exception as exc:
            raise RuntimeError(f'Error computing {metric_name} for {path}: {exc}')
        scores.append(score)
    if not scores:
        raise ValueError(f'No valid {metric_name} scores computed for folder: {folder}')
    return float(np.mean(scores)), float(np.std(scores, ddof=0))


def aggregate_folder_values(values):
    if not values:
        return (math.nan, math.nan)
    return float(np.mean(values)), float(np.std(values, ddof=0))


def main():
    ref_dir = REF_DIR
    fake_folders = FAKE_FOLDERS
    use_gpu = USE_GPU and torch.cuda.is_available()
    if USE_GPU and not torch.cuda.is_available():
        print('Warning: USE_GPU is True but no CUDA device is available. Falling back to CPU.')

    if not fake_folders:
        raise ValueError('FAKE_FOLDERS must contain at least one fake folder path')

    if len(fake_folders) > 1:
        print(f'Found {len(fake_folders)} fake folders for evaluation:')
        for folder in fake_folders:
            print(' -', folder)
    else:
        print(f'Using fake folder: {fake_folders[0]}')

    ref_files = ensure_images_exist(ref_dir)
    print(f'Reference images: {len(ref_files)}')

    fid_values = []
    kid_values = []
    piqe_means = []
    niqe_means = []

    records = []
    device = torch.device('cuda' if use_gpu else 'cpu')

    for fake_dir in fake_folders:
        print('\nEvaluating folder:', fake_dir)
        fake_files = ensure_images_exist(fake_dir)
        print(f'Fake images: {len(fake_files)}')

        fid_value = compute_fid(ref_dir, fake_dir, batch_size=FID_BATCH_SIZE, use_gpu=use_gpu)
        kid_value, _ = compute_kid(ref_dir, fake_dir, image_size=KID_IMAGE_SIZE, subset_size=KID_SUBSET_SIZE, device=device)
        piqe_mean, _ = compute_image_quality(fake_dir, piqe_module.piqe, 'PIQE')
        niqe_mean, _ = compute_image_quality(fake_dir, niqe_module.niqe, 'NIQE')

        print(f'  FID: {fid_value:.6f}')
        print(f'  KID: {kid_value:.6f}')
        print(f'  PIQE mean: {piqe_mean:.6f}')
        print(f'  NIQE mean: {niqe_mean:.6f}')

        fid_values.append(fid_value)
        kid_values.append(kid_value)
        piqe_means.append(piqe_mean)
        niqe_means.append(niqe_mean)

        records.append({
            'folder': fake_dir,
            'fid': fid_value,
            'kid_value': kid_value,
            'piqe_mean': piqe_mean,
            'niqe_mean': niqe_mean,
        })

    fid_avg, fid_std = aggregate_folder_values(fid_values)
    kid_value_avg, kid_value_std = aggregate_folder_values(kid_values)
    piqe_mean_avg, piqe_mean_std = aggregate_folder_values(piqe_means)
    niqe_mean_avg, niqe_mean_std = aggregate_folder_values(niqe_means)

    print(f'Folders evaluated: {len(fake_folders)}')
    print(f'FID mean: {fid_avg:.6f}   std: {fid_std:.6f}')
    print(f'KID mean: {kid_value_avg:.6f}   std: {kid_value_std:.6f}')
    print(f'PIQE mean-of-means: {piqe_mean_avg:.6f}   std: {piqe_mean_std:.6f}')
    print(f'NIQE mean-of-means: {niqe_mean_avg:.6f}   std: {niqe_mean_std:.6f}')

    if OUTPUT_CSV:
        import csv
        write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
        summary_row = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ref_dir': ref_dir,
            'fake_folders': ';'.join(fake_folders),
            'fid_mean': fid_avg,
            'fid_std': fid_std,
            'kid_mean': kid_value_avg,
            'kid_std': kid_value_std,
            'piqe_mean_avg': piqe_mean_avg,
            'piqe_mean_std': piqe_mean_std,
            'niqe_mean_avg': niqe_mean_avg,
            'niqe_mean_std': niqe_mean_std,
        }
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'time', 'ref_dir', 'fake_folders', 'fid_mean', 'fid_std',
                'kid_mean', 'kid_std',
                'piqe_mean_avg', 'piqe_mean_std', 'niqe_mean_avg', 'niqe_mean_std'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(summary_row)
        print(f'Run summary appended to {OUTPUT_CSV}')

    summary_path = os.path.join(ROOT_DIR, 'aggregate_metrics_summary.txt')
    write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    with open(summary_path, 'a', encoding='utf-8') as f:
        if not write_header:
            f.write('\n')
        f.write(f'=== Run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        f.write('Folders evaluated: ' + str(len(fake_folders)) + '\n')
        f.write(f'FID mean: {fid_avg:.6f}   std: {fid_std:.6f}\n')
        f.write(f'KID mean: {kid_value_avg:.6f}   std: {kid_value_std:.6f}\n')
        f.write(f'PIQE mean-of-means: {piqe_mean_avg:.6f}   std: {piqe_mean_std:.6f}\n')
        f.write(f'NIQE mean-of-means: {niqe_mean_avg:.6f}   std: {niqe_mean_std:.6f}\n')

    print(f'Summary appended to {summary_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
