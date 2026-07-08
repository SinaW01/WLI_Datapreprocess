import os
import sys
import time
import torch
import numpy as np
import random
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package not found. The "--use_wandb" option will cause an error.')


def set_seed(seed):
    """Set all random seeds"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_image_with_original_name(image_dir, image_tensor, original_image_path):
    original_name = os.path.basename(original_image_path)
    save_path = os.path.join(image_dir, original_name)

    im = util.tensor2im(image_tensor)
    util.save_image(im, save_path, aspect_ratio=1.0)


def run_single_seed_test(opt, seed, seed_index):
    set_seed(seed)

    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = False
    opt.no_flip = False
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    web_dir = os.path.join(
        opt.results_dir,
        opt.name,
        '{}_{}_seed_{}'.format(opt.phase, opt.epoch, seed)
    )

    images_dir = os.path.join(web_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    webpage = html.HTML(
        web_dir,
        f'Experiment = {opt.name}, Seed = {seed}'
    )

    timing_data = []
    total_time = 0

    if opt.eval:
        model.eval()
    else:
        model.train()

    for i, data in enumerate(dataset):

        if i >= opt.num_test:
            break

        start_time = time.time()

        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        end_time = time.time()

        generation_time = end_time - start_time
        total_time += generation_time

        if i % 5 == 0:
            print(
                f'  Processing image ({i+1:04d})... '
                f'{img_path[0]} '
                f'(Time: {generation_time:.2f}s)'
            )

        images_dir = os.path.join(web_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        saved_any = False

        for key, image_tensor in visuals.items():
            if 'fake' in key.lower():
                save_image_with_original_name(
                    images_dir,
                    image_tensor,
                    img_path[0]
                )
                saved_any = True
                break

        if not saved_any and len(visuals) > 0:
            first_key = list(visuals.keys())[0]

            save_image_with_original_name(
                images_dir,
                visuals[first_key],
                img_path[0]
            )


    avg_time = total_time / len(timing_data) if timing_data else 0


    try:
        import pandas as pd

        excel_path = os.path.join(
            web_dir,
            'generation_times.xlsx'
        )

        df = pd.DataFrame(timing_data)

        df.loc[len(df)] = {
            'Image': 'Total',
            'Generation Time (seconds)': total_time
        }

        df.loc[len(df)] = {
            'Image': 'Average',
            'Generation Time (seconds)': avg_time
        }

        df.to_excel(excel_path, index=False)

        print(
            f'\n✓ Time statistics saved to: {excel_path}'
        )

    except ImportError:
        print(
            'Warning: pandas is not installed, '
            'skipping Excel time statistics export.'
        )


    webpage.save()


    print(
        f'✓ Seed {seed} inference completed, '
        f'{i+1} images processed'
    )

    print(
        f'  Total time: {total_time:.2f}s, '
        f'Average: {avg_time:.4f}s/image\n'
    )


    return web_dir



def main():

    opt = TestOptions().parse()


    if opt.seed_list:

        seeds = opt.seed_list
        num_seeds = len(seeds)

    else:

        num_seeds = opt.num_seeds

        seeds = [42, 123, 456, 789][:num_seeds]


    result_dirs = []


    for idx, seed in enumerate(seeds, 1):

        web_dir = run_single_seed_test(
            opt,
            seed,
            idx
        )

        result_dirs.append(
            {
                'seed': seed,
                'dir': web_dir
            }
        )


    print("\nGenerated result folders:")

    for i, result in enumerate(result_dirs, 1):

        print(
            f"  {i}. Seed {result['seed']:4d}: "
            f"{result['dir']}"
        )


    summary_path = os.path.join(
        opt.results_dir,
        opt.name,
        f'{opt.phase}_{opt.epoch}_multi_seed_summary.txt'
    )


    with open(summary_path, 'w', encoding='utf-8') as f:

        f.write("Multi-Random Seed Inference Test Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model Name: {opt.name}\n")
        f.write(f"Number of Runs: {len(seeds)}\n")
        f.write(f"Random Seeds: {seeds}\n")
        f.write(f"Number of Test Images: {opt.num_test}\n\n")


        f.write("Generated Result Folders:\n")

        for i, result in enumerate(result_dirs, 1):

            f.write(
                f"  {i}. Seed {result['seed']:4d}: "
                f"{result['dir']}\n"
            )


        f.write("\nDescription:\n")

        f.write(
            "- All seeds share the same image filenames "
            "(1.png, 2.png, etc.)\n"
        )

        f.write(
            "- However, the generated contents may differ "
            "due to randomness\n"
        )

        f.write(
            "- Each seed affects:\n"
        )

        f.write(
            "  * Data shuffling order\n"
        )

        f.write(
            "  * Data augmentation strategies\n"
        )

        f.write(
            "  * Random dropout behavior\n"
        )

        f.write(
            "- Metrics can be calculated and compared "
            "among different seeds\n"
        )


    print(
        f"\nSummary file saved to: {summary_path}"
    )



if __name__ == '__main__':
    main()