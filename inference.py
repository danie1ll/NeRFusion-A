from opt import get_opts

from models.nerfusion import NeRFusion2

from utils import load_ckpt

if __name__ == '__main__':
    hparams = get_opts()

    if not hparams.ckpt_path:
        raise ValueError('You need to provide a @ckpt_path for validation!')

    model = NeRFusion2(hparams.scale)
    load_ckpt(model, hparams.ckpt_path)

    print('Model loaded')