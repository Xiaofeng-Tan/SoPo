from omegaconf import DictConfig
import logging
import hydra
import src_tmr.prepare  # noqa
import torch
import numpy as np
from src_tmr.config import read_config
from src_tmr.load import load_model_from_cfg
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from src_tmr.data.collate import collate_x_dict
from src_tmr.model.tmr import get_score_matrix

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="text_motion_sim")
def text_motion_sim(cfg: DictConfig) -> None:
    device = cfg.device
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt_name
    npy_path = cfg.npy
    text = cfg.text

    cfg = read_config(run_dir)

    seed_everything(cfg.seed)

    logger.info("Loading the text model")
    text_model = instantiate(cfg.data.text_to_token_emb, device=device)

    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)

    normalizer = instantiate(cfg.data.motion_loader.normalizer)

    motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
    motion = normalizer(motion)
    motion = motion.to(device)

    motion_x_dict = {"x": motion, "length": len(motion)}

    with torch.inference_mode():
        # motion -> latent
        motion_x_dict = collate_x_dict([motion_x_dict])
        lat_m = model.encode(motion_x_dict, sample_mean=True)[0]

        # text -> latent
        text_x_dict = collate_x_dict(text_model([text]))
        lat_t = model.encode(text_x_dict, sample_mean=True)[0]

        score = get_score_matrix(lat_t, lat_m).cpu()

    score_str = f"{score:.3}"
    logger.info(
        f"The similariy score s (0 <= s <= 1) between the text and the motion is: {score_str}"
    )

def t_m_sim(text, motion, tm_model, text_model, normalizer):
    print(motion.shape)
    motion = normalizer(motion)
    motion = motion.to(device)
    motion_x_dict = {"x": motion, "length": len(motion)}
    with torch.inference_mode():
        # motion -> latent
        motion_x_dict = collate_x_dict([motion_x_dict])
        lat_m = tm_model.encode(motion_x_dict, sample_mean=True)[0]

        # text -> latent
        text_x_dict = collate_x_dict(text_model([text]))
        lat_t = tm_model.encode(text_x_dict, sample_mean=True)[0]

        score = get_score_matrix(lat_t, lat_m)
    #print("???")
    score_str = f"{score:.3}"
    logger.info(
        f"The similariy score s (0 <= s <= 1) between the text and the motion is: {score_str}"
    )
    return score

if __name__ == "__main__":
    #text_motion_sim()
    text = 'A guy is getting ready to do a backflip, but he fails and lands on the floor.'
    device = 'cuda'
    run_dir = './outputs/tmr_humanml3d_guoh3dfeats'
    npy_path = './datasets/motions/humanml3d/new_joint_vecs/000001.npy'
    ckpt_name = 'last'
    cfg = read_config(run_dir)
    seed_everything(cfg.seed)
    
    logger.info("Loading the text model")
    text_model = instantiate(cfg.data.text_to_token_emb, device=device)
    logger.info("Loading the model")
    model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)
    normalizer = instantiate(cfg.data.motion_loader.normalizer)


    motion = torch.from_numpy(np.load(npy_path)).to(torch.float)
    #print(motion.shape)
    #exit()
    print(t_m_sim(text, motion, model, text_model, normalizer))
    print(t_m_sim(text, motion, model, text_model, normalizer))
    print(t_m_sim(text, motion, model, text_model, normalizer))
    print(t_m_sim(text, motion, model, text_model, normalizer))