import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='scannet',
                        choices=['scannetpp', 'nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv', 'scannet', 'google_scanned'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    parser.add_argument('--depth_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is TBD for real scene and TBD for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--num_frames_train', type=int, default=800,
                        help='amount of frames randomly samples from each scene sequence for training')
    parser.add_argument('--num_frames_test', type=int, default=80,
                        help='amount of frames randomly samples from each scene sequence for testing')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # scripts
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='''pretrained checkpoint to load (including optimizers, etc).
                        Use if want to continue training''')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='''pretrained checkpoint to load (excluding optimizers, etc). 
                        Use if want to run inference only''')

    parser.add_argument('--skip_depth_loading', action='store_true', default=False, help='weather to attempt using depth for training')
      # logging parameters
    parser.add_argument('--save_video', action='store_true', default=False,
                        help='''create video from test images and save it. 
                        Makes no sense for random non-sequential images from ScanNet
                        ''')
    
    # WANDB
    parser.add_argument('--debug', action='store_true', default=False,
                        help='if debug is enabled, wandb is not used for logging (default: False)')
    parser.add_argument('--use_sweep', action='store_true', default=False,
                        help='use wandb sweep agents for hyperparameter tuning (default: False)')

    # GRU fusion
    parser.add_argument('--use_gru_fusion', action='store_true', default=False,
                    help='use GRU-fusion to get global feature representations for Scannet (default: False)')
    return parser.parse_args()
