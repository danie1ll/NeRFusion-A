from opt import get_opts

from models.nerfusion import NeRFusion2

from utils import load_ckpt
from datasets.ray_utils import get_rays

from models.rendering import render

from datasets import dataset_dict


def load_test_dataset(hparams, device='cuda:0'):
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir, 'downsample': hparams.downsample}

    test_dataset = dataset(split='train', **kwargs)

    directions = test_dataset.directions.to(device)
    poses = test_dataset.poses.to(device)

    return directions, poses


def forward(model, directions, poses):
    rays_o, rays_d = get_rays(directions, poses)

    kwargs = {'test_time': True, 'random_bg': hparams.random_bg}

    if hparams.scale > 0.5:
        kwargs['exp_step_factor'] = 1 / 256

    return render(model, rays_o, rays_d, **kwargs)


if __name__ == '__main__':
    hparams = get_opts()

    if not hparams.ckpt_path:
        raise ValueError('You need to provide a @ckpt_path for validation!')

    model = NeRFusion2(hparams.scale)
    load_ckpt(model, hparams.ckpt_path)
    model.eval()

    directions, poses = load_test_dataset(hparams)

    results = forward(model, directions, poses)
