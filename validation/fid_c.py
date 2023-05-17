import torch
import numpy as np
import jax.numpy as jnp

from torchmetrics.image.fid import FrechetInceptionDistance
def get_fid_model(cfg):
    #DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"
    model = FrechetInceptionDistance(feature=cfg.train_and_test.test.fid_features)
    if DEVICE == "cuda:0":
        model.inception.cuda()

    def compute_fid(generated_imgs, real_images):
        datashape = np.array(jnp.array(cfg.dataset.shape)+jnp.array([0,cfg.dataset.padding*2,cfg.dataset.padding*2,0]))
        datashape[0] = len(generated_imgs)
        datashape[-1] = -1

        if cfg.dataset.name == "mnist":
            generated_imgs = np.stack((generated_imgs,) * 3, axis=-1).reshape(datashape).astype(np.uint8).transpose(0, -1, 1, 2)
            real_images = np.stack((real_images,) * 3, axis=-1).reshape(datashape).astype(np.uint8).transpose(0, -1, 1, 2)
        generated_imgs = torch.from_numpy(generated_imgs)
        real_images = torch.from_numpy(real_images)
        model.update(real_images, real=True) 
        model.update(generated_imgs, real=False) 
        fid = model.compute()
        model.reset()
        return fid
    return compute_fid

    




