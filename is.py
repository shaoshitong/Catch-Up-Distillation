# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Inception Score (IS)."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
from torch_utils import distributed as dist
from training import dataset

#----------------------------------------------------------------------------

def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(no_output_bias=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    dist.print0(f'Calculating statistics for {len(dataset_obj)} images...')
    gen_probs  = []
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        gen_probs.append(features)

    gen_probs = torch.cat(gen_probs, dim=0).cpu().numpy()
    return gen_probs
#----------------------------------------------------------------------------

def calculate_is_from_inception_stats(gen_probs):
    kl = gen_probs*(np.log(gen_probs) - np.log(np.mean(gen_probs,axis=0,keepdims=True)))
    kl = np.mean(np.sum(kl, axis=1))
    score = np.exp(kl).item()
    return float(score)

#----------------------------------------------------------------------------

@click.group()
def main():
#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)

def calc(image_path, ref_path, num_expected, seed, batch):
    """Calculate IS for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    print(ref.keys())
    gen_probs = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    dist.print0('Calculating IS...')
    if dist.get_rank() == 0:
        _is = calculate_is_from_inception_stats(gen_probs)
        print(f'{_is:g}')
        # Save to image_path as a txt file
        with open(os.path.join(image_path, 'is.txt'), 'w') as f:
            f.write(str(_is))
    torch.distributed.barrier()


@main.command(name='calc_multiple')
@click.option('--images', 'image_paths', help='Path to the images', metavar='PATH|ZIP',              type=click.Path(exists=True), multiple=True, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)

def calc_multiple(image_paths, ref_path, num_expected, seed, batch):
    """Calculate IS for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    os.environ['MASTER_PORT'] = f"{29500+ int(os.environ['CUDA_VISIBLE_DEVICES'])+1}"
    print(os.environ['MASTER_PORT'])
    dist.init()
    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    iss = []
    for image_path in image_paths:
        gen_probs = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
        dist.print0(f'Calculating IS for images in {image_path}...')
        if dist.get_rank() == 0:
            _is = calculate_is_from_inception_stats(gen_probs)
            print(f'IS for images in {image_path}: {_is:g}')
            iss.append(round(_is,2))
            # Save to image_path as a txt file
            with open(os.path.join(image_path, 'is_multiplt.txt'), 'w') as f:
                f.write(str(_is)+"\n")
        torch.distributed.barrier()
    print(_is)

#----------------------------------------------------------------------------

@main.command()
@click.option('--data', 'dataset_path', help='Path to the dataset', metavar='PATH|ZIP', type=str, required=True)
@click.option('--dest', 'dest_path',    help='Destination .npz file', metavar='NPZ',    type=str, required=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=64, show_default=True)

def ref(dataset_path, dest_path, batch):
    """Calculate dataset reference statistics needed by 'calc'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch)
    dist.print0(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------