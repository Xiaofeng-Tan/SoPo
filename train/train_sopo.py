# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import torch

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)
    #exit()
    
    print("creating model and diffusion...")
    model_opt, diffusion = create_model_and_diffusion(args, data)
    model_opt.to(dist_util.dev())
    model_opt.rot2xyz.smpl_model.eval()
    args.model_path = 'save/humanml_enc_512_50steps/model000750000.pt'
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model_opt, state_dict)
    
    print("creating reference model and diffusion...")
    ref_model, diffusion = create_model_and_diffusion(args, data)
    ref_model.to(dist_util.dev())
    ref_model.rot2xyz.smpl_model.eval()
    args.model_path = 'save/humanml_enc_512_50steps/model000750000.pt'
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(ref_model, state_dict)
    #print(model)
    print('Total params: %.2fM' % (sum(p.numel() for p in model_opt.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model_opt, ref_model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
